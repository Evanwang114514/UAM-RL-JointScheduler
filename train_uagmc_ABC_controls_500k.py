# -*- coding: utf-8 -*-
"""
Focused A/B/C single-factor controls, 500k per cell, 5.0M total PPO steps
========================================================================

Budget design (default)
-----------------------
A. Future granularity (1.0M)
   E3: A_SETCOUNT vs A_SETETA
   Same event population, same DeepSets architecture, same parameters.
   Only individual ETA values are removed/present.

B. Action effect-time alignment (2.0M)
   E4/E5: B_SHARED vs B_CAND
   Same fine events, same projected-state dimension, same network.
   Only projection/event temporal reference changes:
       shared passenger-level horizon
       vs candidate-specific access/effect horizon.

C. Bottleneck-aware self effect (2.0M)
   E5/E6: C_PRESSURE vs C_DW
   Same candidate-centered M5 pipeline and same 3*K CF dimension.
   Only counterfactual feature changes:
       original queue/supply pressure
       vs committed-resource workload delta.

Total:
    10 cells * 500,000 = 5,000,000 PPO timesteps.

Important
---------
- T2 by default, aligned with the completed 28.8M matrix.
- Same PPO settings, reward, train seed, 16x1280 rollout, batch size, etc.
- Checkpoint every 50k and immediately evaluate every checkpoint.
- Default eval seeds: 123,124,125, same as the previous matrix.
- No unrevealed future passenger demand is used.
- Automatically searches the latest completed 6x6 T2 run and extracts the
  corresponding M0 500k and M0 300-500k references for each stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

import train_uagmc_6x6_800k as old


ROOT = Path(__file__).resolve().parent

DEFAULT_TIMESTEPS = 500_000
DEFAULT_TOPOLOGY = "T2"
DEFAULT_TRAIN_SEED = 1
DEFAULT_EVAL_SEEDS = (123, 124, 125)

METHOD_NAMES = {
    "A_SETCOUNT": "M2-SetCount",
    "A_SETETA": "M3-SetETA",
    "B_SHARED": "M3P-shared",
    "B_CAND": "M4-candidate",
    "C_PRESSURE": "M5-pressure",
    "C_DW": "M5-bottleneck-dW",
}

# 10 cells * 500k = 5.0M.
CELL_SPECS = (
    ("A", "E3", "A_SETCOUNT"),
    ("A", "E3", "A_SETETA"),

    ("B", "E4", "B_SHARED"),
    ("B", "E4", "B_CAND"),
    ("B", "E5", "B_SHARED"),
    ("B", "E5", "B_CAND"),

    ("C", "E5", "C_PRESSURE"),
    ("C", "E5", "C_DW"),
    ("C", "E6", "C_PRESSURE"),
    ("C", "E6", "C_DW"),
)

PAIR_SPECS = (
    ("A", "E3", "A_SETCOUNT", "A_SETETA"),
    ("B", "E4", "B_SHARED", "B_CAND"),
    ("B", "E5", "B_SHARED", "B_CAND"),
    ("C", "E5", "C_PRESSURE", "C_DW"),
    ("C", "E6", "C_PRESSURE", "C_DW"),
)

A_METHODS = {"A_SETCOUNT", "A_SETETA"}
B_METHODS = {"B_SHARED", "B_CAND"}
C_METHODS = {"C_PRESSURE", "C_DW"}
ALL_METHODS = tuple(METHOD_NAMES)

# Keep references to the original extractor before monkey-patching old globals.
OriginalUnifiedExtractor = old.UnifiedUAMExtractor


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    old.write_csv(path, list(rows))


def read_csv(path: Path) -> List[Dict[str, Any]]:
    return old.read_csv(path)


def fnum(x: Any, default: float = float("nan")) -> float:
    return old.fnum(x, default)


def mean_finite(xs: Sequence[Any]) -> float:
    arr = np.asarray([fnum(x) for x in xs], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


# =============================================================================
# Exact controlled layouts
# =============================================================================

def control_layout(
    *,
    stage: str,
    method: str,
    topology: str,
    max_events: int,
) -> Dict[str, Any]:
    method = str(method).upper()
    if method not in METHOD_NAMES:
        raise ValueError(f"unsupported control method: {method}")

    cands = old.candidates_for(topology)
    n = len(cands)
    single_dim = 4 + 8 * n
    base_dim = old.NUM_FRAMES * single_dim

    # All six controls use the same M1 passenger-specific branch.
    # A: fine-set only
    # B: fine-set + projected state
    # C: fine-set + projected state + 3D/candidate self-effect
    dims = {
        "base": base_dim,
        "current": old.current_feature_dim(stage, n),
        "passenger": 4,
        "coarse": 0,
        "fine": 4 * int(max_events) * n,
        "project": (10 * n) if method in (B_METHODS | C_METHODS) else 0,
        "cf": (3 * n) if method in C_METHODS else 0,
    }

    slices: Dict[str, Tuple[int, int]] = {}
    pos = 0
    for name in ("base", "current", "passenger", "coarse", "fine", "project", "cf"):
        width = int(dims[name])
        slices[name] = (pos, pos + width)
        pos += width

    return {
        "stage": str(stage).upper(),
        "method": method,
        "topology": str(topology).upper(),
        "candidates": cands,
        "n_candidates": n,
        "num_frames": old.NUM_FRAMES,
        "single_dim": single_dim,
        "base_dim": base_dim,
        "max_events": int(max_events),
        "dims": dims,
        "slices": slices,
        "total_dim": pos,
    }


def pack_event_values(
    etas: Sequence[float],
    *,
    max_events: int,
    horizon: float,
    center: float,
    keep_eta: bool,
) -> Tuple[List[float], List[float]]:
    xs = list(etas)[: int(max_events)]
    if keep_eta:
        values = [
            float((float(x) - float(center)) / max(float(horizon), 1e-6))
            for x in xs
        ]
    else:
        # A_SETCOUNT: same event slots/masks/types, ETA information removed.
        values = [0.0 for _ in xs]

    mask = [1.0] * len(values)
    while len(values) < int(max_events):
        values.append(0.0)
        mask.append(0.0)
    return values, mask


# =============================================================================
# C: bottleneck-aware committed workload delta
# =============================================================================

def bottleneck_cf_features(
    scenario: Any,
    *,
    stage: str,
    vid: int,
    horizon: float,
    project: Sequence[float],
    charger_capacity: int,
    future_horizon: float,
) -> List[float]:
    """
    3D/candidate replacement for current M5 pressure:
        [1, normalized W_after, normalized DeltaW]

    W is a committed-resource workload proxy at the candidate-specific effect
    time. It uses only currently known supply release events:
      ready aircraft,
      turnaround releases,
      charging releases,
      committed empty inbound aircraft,
      and E5/E6 pad timing already represented by the environment.

    No unrevealed future requests and no future uncommitted reposition actions.
    """
    stage = str(stage).upper()
    h = max(0.0, float(horizon))

    # project[1] = current queue + committed passenger arrivals by effect time.
    demand_before = max(0.0, float(project[1]))
    demand_after = demand_before + 1.0

    cap = max(
        1,
        int(old.rulemod.aircraft_capacity(scenario, stage)),
    )
    releases = old.rulemod.known_supply_release_etas(
        scenario,
        int(vid),
        stage,
        int(charger_capacity),
    )

    n_by_h = sum(1 for eta in releases if float(eta) <= h + 1e-9)
    served_capacity = float(cap * n_by_h)

    residual_before = max(0.0, demand_before - served_capacity)
    residual_after = max(0.0, demand_after - served_capacity)

    # Resource cycle penalty after the focal passenger reaches the candidate.
    future_release = [float(x) for x in releases if float(x) > h + 1e-9]
    supply_gap = max(0.0, min(future_release) - h) if future_release else 0.0

    # project slots:
    # 3 charging_future, 6 avg_charge, 7 min_charge,
    # 8 avg_inbound, 9 pad residual at effect time.
    charging_future = max(0.0, float(project[3]))
    avg_charge = max(0.0, float(project[6]))
    min_charge = max(0.0, float(project[7]))
    avg_inbound = max(0.0, float(project[8]))
    pad_delay = max(0.0, float(project[9]))

    resource_cycle = 1.0 + supply_gap + pad_delay
    if stage == "E6":
        # Finite chargers matter only in E6. Keep the proxy smooth and bounded.
        resource_cycle += min_charge
        resource_cycle += 0.25 * avg_charge
        resource_cycle += 0.25 * charging_future
    else:
        # In E5, committed inbound timing still contributes to serviceability.
        resource_cycle += 0.10 * avg_inbound

    w_before = residual_before * resource_cycle / float(cap)
    w_after = residual_after * resource_cycle / float(cap)
    delta_w = w_after - w_before

    # Normalize to roughly the same numerical scale as other residual features.
    scale = max(1.0, float(future_horizon))
    return [
        1.0,
        float(w_after / scale),
        float(delta_w / scale),
    ]


# =============================================================================
# Controlled observation wrapper
# =============================================================================

class ControlObservationWrapper(old.gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        stage: str,
        method: str,
        topology: str,
        future_horizon: float,
        max_events: int,
        charger_capacity: int,
    ):
        super().__init__(env)
        self.stage = str(stage).upper()
        self.method = str(method).upper()
        self.topology = str(topology).upper()
        self.candidates = old.candidates_for(self.topology)
        self.future_horizon = float(future_horizon)
        self.max_events = int(max_events)
        self.charger_capacity = int(charger_capacity)

        self.layout = control_layout(
            stage=self.stage,
            method=self.method,
            topology=self.topology,
            max_events=self.max_events,
        )

        got = int(np.prod(env.observation_space.shape))
        expected = int(self.layout["base_dim"])
        if got != expected:
            raise RuntimeError(
                f"base observation mismatch {got} != {expected} "
                f"for {self.stage}/{self.method}"
            )

        self.observation_space = spaces.Box(
            low=-1e6,
            high=1e6,
            shape=(int(self.layout["total_dim"]),),
            dtype=np.float32,
        )

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        scenario = old.mx.find_scenario(self.env)
        person = old._focal_person(self.env, scenario)

        extras: List[float] = []

        # Same environment-aligned current variables for every control.
        extras.extend(
            old._current_aligned_vector(
                scenario,
                stage=self.stage,
                candidates=self.candidates,
                charger_capacity=self.charger_capacity,
            )
        )

        # Same M1 passenger branch for every control.
        extras.extend(old._focal_od(person))

        access_by_vid = {
            int(vid): old._access_time(scenario, person, int(vid))
            for vid in self.candidates
        }
        shared_horizon = (
            float(np.mean(list(access_by_vid.values())))
            if access_by_vid
            else 0.0
        )

        # Fine event population is identical within every paired comparison.
        for vid in self.candidates:
            aircraft, passengers = old._event_lists(
                scenario,
                int(vid),
                horizon=self.future_horizon,
            )

            if self.method == "A_SETCOUNT":
                center = 0.0
                keep_eta = False
            elif self.method == "A_SETETA":
                center = 0.0
                keep_eta = True
            elif self.method == "B_SHARED":
                center = shared_horizon
                keep_eta = True
            else:
                # B_CAND and both C methods.
                center = access_by_vid[int(vid)]
                keep_eta = True

            a_eta, a_mask = pack_event_values(
                aircraft,
                max_events=self.max_events,
                horizon=self.future_horizon,
                center=center,
                keep_eta=keep_eta,
            )
            p_eta, p_mask = pack_event_values(
                passengers,
                max_events=self.max_events,
                horizon=self.future_horizon,
                center=center,
                keep_eta=keep_eta,
            )
            extras.extend(a_eta)
            extras.extend(a_mask)
            extras.extend(p_eta)
            extras.extend(p_mask)

        projects: List[Tuple[int, float, List[float]]] = []
        if self.method in (B_METHODS | C_METHODS):
            for vid in self.candidates:
                if self.method == "B_SHARED":
                    h = shared_horizon
                else:
                    h = access_by_vid[int(vid)]

                proj = old._project_candidate(
                    scenario,
                    stage=self.stage,
                    vid=int(vid),
                    horizon=h,
                    charger_capacity=self.charger_capacity,
                )
                projects.append((int(vid), float(h), proj))
                extras.extend(proj)

        if self.method == "C_PRESSURE":
            # Exact current-M5 counterfactual feature.
            for _, _, proj in projects:
                extras.extend(old._counterfactual_features(proj))

        if self.method == "C_DW":
            # Same dimension, only replace the self-effect representation.
            for vid, h, proj in projects:
                extras.extend(
                    bottleneck_cf_features(
                        scenario,
                        stage=self.stage,
                        vid=vid,
                        horizon=h,
                        project=proj,
                        charger_capacity=self.charger_capacity,
                        future_horizon=self.future_horizon,
                    )
                )

        base_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        out = np.concatenate(
            [base_obs, np.asarray(extras, dtype=np.float32)],
            axis=0,
        )
        expected = int(self.layout["total_dim"])
        if out.shape != (expected,):
            raise RuntimeError(
                f"observation shape {out.shape} != {(expected,)} "
                f"for {self.stage}/{self.method}"
            )
        return out.astype(np.float32, copy=False)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return self._transform(obs), info
        return self._transform(out)

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return self._transform(obs), reward, terminated, truncated, info
        if len(out) == 4:
            obs, reward, done, info = out
            return self._transform(obs), reward, done, info
        raise RuntimeError(f"unexpected step tuple length={len(out)}")


# =============================================================================
# Extractor
# =============================================================================

class ControlExtractor(OriginalUnifiedExtractor):
    """
    B/C preserve the existing M3/M4/M5 sum-pooling architecture.
    A uses the SAME modified DeepSets encoder on both arms:
        mean(phi(events)) + explicit [aircraft_count, passenger_count].
    Therefore A changes only ETA content, not architecture/cardinality encoding.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        layout: Optional[Dict[str, Any]] = None,
    ):
        if layout is None:
            raise ValueError("layout required")
        super().__init__(
            observation_space,
            features_dim=features_dim,
            layout=layout,
        )
        self.control_method = str(layout["method"]).upper()

        if self.control_method in A_METHODS:
            # Original rho is 32->32. A needs identical 34->32 on both arms.
            self.event_rho = nn.Sequential(
                nn.Linear(34, 32),
                nn.ReLU(),
            )

    def _encode_sets(self, obs: torch.Tensor) -> torch.Tensor:
        if self.control_method not in A_METHODS:
            # Preserve original sum-pooling for B/C exactly.
            return super()._encode_sets(obs)

        fine = self._slice(obs, "fine")
        K = self.max_events
        block = 4 * K
        reps = []

        for c in range(self.n_candidates):
            x = fine[:, c * block:(c + 1) * block]
            a_eta = x[:, 0:K]
            a_mask = x[:, K:2 * K]
            p_eta = x[:, 2 * K:3 * K]
            p_mask = x[:, 3 * K:4 * K]

            eta = torch.cat([a_eta, p_eta], dim=1)
            mask = torch.cat([a_mask, p_mask], dim=1)
            event_type = torch.cat(
                [torch.zeros_like(a_eta), torch.ones_like(p_eta)],
                dim=1,
            )

            elems = torch.stack([eta, event_type], dim=-1)
            emb = self.event_phi(elems) * mask.unsqueeze(-1)

            count_total = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled_mean = emb.sum(dim=1) / count_total

            # Explicit cardinality so mean pooling does not erase counts.
            count_a = a_mask.sum(dim=1, keepdim=True) / float(max(1, K))
            count_p = p_mask.sum(dim=1, keepdim=True) / float(max(1, K))
            rho_in = torch.cat([pooled_mean, count_a, count_p], dim=1)
            reps.append(self.event_rho(rho_in))

        return torch.cat(reps, dim=1)


# =============================================================================
# Reference extraction: compare every new cell with its stage-matched plain M0
# =============================================================================

def _latest_previous_6x6(topology: str) -> Optional[Path]:
    patt = f"uagmc_6x6_{str(topology).upper()}_800k_seed1_*"
    dirs = [
        p for p in (ROOT / "serial_runs").glob(patt)
        if p.is_dir()
    ]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def load_m0_references(topology: str) -> Dict[str, Dict[str, float]]:
    root = _latest_previous_6x6(topology)
    refs: Dict[str, Dict[str, float]] = {}
    if root is None:
        return refs

    for stage in ("E3", "E4", "E5", "E6"):
        curve_path = root / f"{stage}__M0" / "analysis" / "checkpoint_curve.csv"
        rows = read_csv(curve_path)
        if not rows:
            continue

        row500 = next(
            (r for r in rows if int(float(r.get("train_step", -1))) == 500_000),
            None,
        )
        late_rows = [
            r for r in rows
            if 300_000 <= int(float(r.get("train_step", -1))) <= 500_000
        ]

        refs[stage] = {
            "source_root": str(root),
            "M0_500k_Jsys": (
                fnum(row500.get("system_person_minutes_per_passenger_mean"))
                if row500 else float("nan")
            ),
            "M0_500k_ATT": (
                fnum(row500.get("ATT_mean"))
                if row500 else float("nan")
            ),
            "M0_500k_completion": (
                fnum(row500.get("completion_rate_mean"))
                if row500 else float("nan")
            ),
            "M0_300_500k_Jsys_mean": mean_finite(
                [r.get("system_person_minutes_per_passenger_mean") for r in late_rows]
            ),
            "M0_300_500k_ATT_mean": mean_finite(
                [r.get("ATT_mean") for r in late_rows]
            ),
        }
    return refs


def m0_reference_rows(refs: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    out = []
    for stage, r in sorted(refs.items()):
        out.append({"stage": stage, **r})
    return out


# =============================================================================
# Pairwise summaries
# =============================================================================

def augment_with_m0(
    row: Dict[str, Any],
    refs: Dict[str, Dict[str, float]],
    group: str,
) -> Dict[str, Any]:
    out = dict(row)
    out["group"] = group
    ref = refs.get(str(row.get("stage")), {})
    m0_final = fnum(ref.get("M0_500k_Jsys"))
    m0_late = fnum(ref.get("M0_300_500k_Jsys_mean"))
    final_j = fnum(row.get("final_Jsys_per_passenger"))
    late_j = fnum(row.get("late_Jsys_mean"))

    out["M0_500k_Jsys_ref"] = m0_final
    out["final_delta_vs_M0_abs"] = final_j - m0_final
    out["final_delta_vs_M0_pct"] = (
        100.0 * (final_j - m0_final) / m0_final
        if math.isfinite(final_j) and math.isfinite(m0_final) and abs(m0_final) > 1e-12
        else float("nan")
    )

    out["M0_300_500k_Jsys_ref"] = m0_late
    out["late_delta_vs_M0_abs"] = late_j - m0_late
    out["late_delta_vs_M0_pct"] = (
        100.0 * (late_j - m0_late) / m0_late
        if math.isfinite(late_j) and math.isfinite(m0_late) and abs(m0_late) > 1e-12
        else float("nan")
    )
    return out


def build_pair_rows(
    results: Sequence[Dict[str, Any]],
    refs: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    by_key = {
        (str(r.get("stage")), str(r.get("method"))): r
        for r in results
    }
    rows = []
    for group, stage, control, treatment in PAIR_SPECS:
        c = by_key.get((stage, control))
        t = by_key.get((stage, treatment))
        if c is None or t is None:
            continue

        cj = fnum(c.get("late_Jsys_mean"))
        tj = fnum(t.get("late_Jsys_mean"))
        cf = fnum(c.get("final_Jsys_per_passenger"))
        tf = fnum(t.get("final_Jsys_per_passenger"))
        m0 = fnum(refs.get(stage, {}).get("M0_300_500k_Jsys_mean"))

        rows.append({
            "group": group,
            "stage": stage,
            "control": control,
            "treatment": treatment,
            "control_late_Jsys": cj,
            "treatment_late_Jsys": tj,
            "treatment_minus_control_late_abs": tj - cj,
            "treatment_minus_control_late_pct": (
                100.0 * (tj - cj) / cj
                if math.isfinite(cj) and abs(cj) > 1e-12 else float("nan")
            ),
            "control_final_Jsys": cf,
            "treatment_final_Jsys": tf,
            "treatment_minus_control_final_abs": tf - cf,
            "M0_300_500k_Jsys_ref": m0,
            "control_late_delta_vs_M0": cj - m0,
            "treatment_late_delta_vs_M0": tj - m0,
        })
    return rows


# =============================================================================
# Monkey-patch experiment layer only
# =============================================================================

def install_control_layer() -> None:
    old.make_layout = control_layout
    old.MethodObservationWrapper = ControlObservationWrapper
    old.UnifiedUAMExtractor = ControlExtractor
    old.METHOD_NAMES.update(METHOD_NAMES)


# =============================================================================
# Runner
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Focused A/B/C single-factor controls, default total 5.0M PPO steps"
    )
    p.add_argument("--groups", default="A,B,C")
    p.add_argument("--topology", choices=["T2", "T3"], default=DEFAULT_TOPOLOGY)
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    p.add_argument(
        "--eval-seeds",
        default=",".join(str(x) for x in DEFAULT_EVAL_SEEDS),
    )
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--fleet-size", type=int, default=old.FLEET_SIZE)
    p.add_argument("--max-time", type=int, default=old.MAX_TIME)
    p.add_argument("--future-horizon", type=float, default=old.FUTURE_HORIZON_MIN)
    p.add_argument("--max-events", type=int, default=old.MAX_EVENTS_PER_TYPE)
    p.add_argument("--pad-separation", type=float, default=old.PAD_SEPARATION_MIN)
    p.add_argument("--charger-capacity", type=int, default=old.CHARGER_CAPACITY)
    p.add_argument("--output-root", default=None)
    p.add_argument("--resume-root", default=None)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    install_control_layer()

    groups = [
        x.strip().upper()
        for x in str(args.groups).split(",")
        if x.strip()
    ]
    bad = [x for x in groups if x not in ("A", "B", "C")]
    if bad:
        raise ValueError(f"unknown groups: {bad}")

    requested_steps = int(args.timesteps)
    if requested_steps <= 0 or requested_steps % old.CHECKPOINT_INTERVAL != 0:
        raise ValueError(
            f"--timesteps must be positive multiple of {old.CHECKPOINT_INTERVAL}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    eval_seeds = old.parse_ints(args.eval_seeds)
    cells = [x for x in CELL_SPECS if x[0] in groups]
    total_steps = len(cells) * requested_steps

    if args.resume_root:
        root = Path(args.resume_root).expanduser()
        if not root.is_absolute():
            root = (ROOT / root).resolve()
        else:
            root = root.resolve()
    elif args.output_root:
        root = Path(args.output_root).expanduser()
        if not root.is_absolute():
            root = (ROOT / root).resolve()
        else:
            root = root.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = (
            ROOT
            / "serial_runs"
            / f"uagmc_ABC_controls_{args.topology}_{requested_steps//1000}k_seed{args.seed}_{stamp}"
        ).resolve()

    root.mkdir(parents=True, exist_ok=True)

    m0_refs = load_m0_references(args.topology)
    write_csv(root / "m0_reference.csv", m0_reference_rows(m0_refs))

    manifest = {
        "experiment": "UAGMC_FOCUSED_ABC_SINGLE_FACTOR_CONTROLS",
        "created": datetime.now().isoformat(timespec="seconds"),
        "groups": groups,
        "cells": [
            {"group": g, "stage": s, "method": m, "name": METHOD_NAMES[m]}
            for g, s, m in cells
        ],
        "topology": args.topology,
        "timesteps_per_cell": requested_steps,
        "n_cells": len(cells),
        "total_requested_timesteps": total_steps,
        "budget_design_default": "10 cells x 500k = 5.0M",
        "train_seed": int(args.seed),
        "eval_seeds": eval_seeds,
        "checkpoint_interval": old.CHECKPOINT_INTERVAL,
        "future_horizon_min": float(args.future_horizon),
        "max_events": int(args.max_events),
        "controls": {
            "A": (
                "same event population + same DeepSets(mean+explicit counts); "
                "only individual ETA content differs"
            ),
            "B": (
                "same ETA sets + same projection + same original M4 network; "
                "only shared vs candidate-specific temporal reference differs"
            ),
            "C": (
                "same original candidate-centered M5 pipeline and same 3D CF branch; "
                "only pressure vs committed-resource delta-W differs"
            ),
        },
        "M0_reference_source": (
            next(iter(m0_refs.values())).get("source_root")
            if m0_refs else None
        ),
    }
    old.write_json(root / "experiment_manifest.json", manifest)

    print("=" * 132)
    print("FOCUSED A/B/C SINGLE-FACTOR CONTROLS")
    print(
        f"groups={groups} | cells={len(cells)} | "
        f"{requested_steps:,}/cell | TOTAL={total_steps:,} PPO steps"
    )
    print(f"topology={args.topology} | train_seed={args.seed} | eval={eval_seeds}")
    print("A: SetCount vs SetETA")
    print("B: Shared horizon vs Candidate-specific effect time")
    print("C: Current M5 pressure vs Bottleneck-aware DeltaW")
    print(f"output={root}")
    print("=" * 132)

    if not m0_refs:
        print("[WARN] previous 6x6 M0 curves not found; M0 delta columns will be NaN.")
    else:
        for stage in ("E3", "E4", "E5", "E6"):
            if stage in m0_refs:
                r = m0_refs[stage]
                print(
                    f"[M0 ref] {stage}: "
                    f"500k J={fnum(r.get('M0_500k_Jsys')):.3f}, "
                    f"300-500k mean={fnum(r.get('M0_300_500k_Jsys_mean')):.3f}"
                )

    # We deliberately do not spend time rerunning rules here. The experiment's
    # primary comparisons are paired single-factor controls + stage-matched M0.
    rule_refs: Dict[str, Dict[str, float]] = defaultdict(dict)

    results: List[Dict[str, Any]] = []

    # Recover completed cells on resume.
    for group, stage, method in cells:
        sp = root / old.cell_id(stage, method) / "analysis" / "summary.json"
        if sp.exists():
            try:
                summary = json.loads(sp.read_text(encoding="utf-8"))
                row = old.summary_row(summary)
                results.append(augment_with_m0(row, m0_refs, group))
            except Exception:
                pass

    # Deduplicate recovered rows.
    dedup = {}
    for r in results:
        dedup[(r.get("stage"), r.get("method"))] = r
    results = list(dedup.values())
    write_csv(root / "control_results.csv", results)
    write_csv(root / "pairwise_comparisons.csv", build_pair_rows(results, m0_refs))

    for idx, (group, stage, method) in enumerate(cells, 1):
        print(
            f"\n[{idx:02d}/{len(cells):02d}] GROUP {group} | "
            f"{stage}/{method} | {METHOD_NAMES[method]}",
            flush=True,
        )
        try:
            summary = old.run_cell(
                stage=stage,
                method=method,
                topology=args.topology,
                requested_steps=requested_steps,
                seed=int(args.seed),
                device=args.device,
                root=root,
                eval_seeds=eval_seeds,
                rule_refs=rule_refs,
                fleet_size=int(args.fleet_size),
                future_horizon=float(args.future_horizon),
                max_events=int(args.max_events),
                pad_separation=float(args.pad_separation),
                charger_capacity=int(args.charger_capacity),
                max_time=int(args.max_time),
            )

            row = augment_with_m0(
                old.summary_row(summary),
                m0_refs,
                group,
            )

            results = [
                r for r in results
                if not (
                    str(r.get("stage")) == stage
                    and str(r.get("method")) == method
                )
            ]
            results.append(row)

            # Preserve designed execution order.
            order = {
                (s, m): i
                for i, (_, s, m) in enumerate(cells)
            }
            results.sort(
                key=lambda r: order.get(
                    (str(r.get("stage")), str(r.get("method"))),
                    999,
                )
            )

            write_csv(root / "control_results.csv", results)
            write_csv(
                root / "pairwise_comparisons.csv",
                build_pair_rows(results, m0_refs),
            )

            print(
                f"[M0 COMPARE] final delta={fnum(row.get('final_delta_vs_M0_pct')):+.2f}% | "
                f"late delta={fnum(row.get('late_delta_vs_M0_pct')):+.2f}%",
                flush=True,
            )

        except Exception:
            if not args.continue_on_error:
                raise
            print(
                f"[ERROR but continue] group={group} {stage}/{method}; "
                f"see cell_error.json",
                flush=True,
            )

    pair_rows = build_pair_rows(results, m0_refs)
    write_csv(root / "control_results.csv", results)
    write_csv(root / "pairwise_comparisons.csv", pair_rows)

    summary_lines = [
        "FOCUSED A/B/C CONTROL SUMMARY",
        "=" * 118,
        f"total PPO budget = {total_steps:,}",
        "",
        "group | stage | control -> treatment | late J control -> treatment | treatment-control | M0 late ref",
    ]
    for r in pair_rows:
        summary_lines.append(
            f"{r['group']} | {r['stage']} | "
            f"{r['control']} -> {r['treatment']} | "
            f"{fnum(r['control_late_Jsys']):.3f} -> "
            f"{fnum(r['treatment_late_Jsys']):.3f} | "
            f"{fnum(r['treatment_minus_control_late_abs']):+.3f} | "
            f"{fnum(r['M0_300_500k_Jsys_ref']):.3f}"
        )
    (root / "summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 132)
    print("ABC CONTROLS FINISHED")
    print(f"Total PPO timesteps: {total_steps:,}")
    print(f"Main results : {root / 'control_results.csv'}")
    print(f"Paired table : {root / 'pairwise_comparisons.csv'}")
    print(f"M0 refs      : {root / 'm0_reference.csv'}")
    print("Each cell    : <E>__<method>/analysis/checkpoint_curve.csv")
    print("=" * 132)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
