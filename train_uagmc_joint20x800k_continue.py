# -*- coding: utf-8 -*-
"""
FORMAL 45 x 800k UAGMC matrix
==============================

Rows = 5 passenger-side scheduling environments + 4 E2-based joint-control rows.
Columns = Residual + 2x2 future-representation factorial.

Rows
----
E0 : source/legacy replenishment + exactly one passenger per service flight
E1 : E0 physics replaced by fixed conserved fleet + responsive LQ reposition
E2 : E1 + 3 min turnaround
E4 : E2 + finite parallel charging (no finite TLOF)
E5 : E4 + finite shared TLOF/pad calendar; default separation = 0.25 min

J0 : E2 physics, no LQ, joint PPO, Separate + Parallel
J1 : E2 physics, no LQ, joint PPO, Unified-semantic + Parallel
J2 : E2 physics, no LQ, joint PPO, Separate + Sequential-factorized
J3 : E2 physics, no LQ, joint PPO, Unified-semantic + Sequential-factorized

Joint action is deliberately tiny. T2 passenger action has 3 labels and aircraft
has {Hold,V0,V1}; joint rows use a single Discrete(9) pair action.  The PPO policy
head factorizes those 9 logits according to J0/J1/J2/J3, so the environment never
receives a per-aircraft action vector.  At most one eligible empty hub aircraft is
repositioned at a passenger decision epoch.

Columns
-------
M0 RESIDUAL
    Current/source UAGMC history + fixed current-resource superset + focal passenger
    dedicated branch. Future slots are present but zeroed, so M0..M4 share the same
    observation dimension and extractor parameter count.

M1 SHARED
    M0 + committed-event DeepSets + candidate projected state at a common passenger-
    conditioned future horizon mean_k T_access(p,k).

M2 SHARED_UQ
    M1 + symmetric +/- ETA-envelope sensitivity around the shared horizon.
    It is a deterministic local uncertainty/sensitivity feature, not a probabilistic
    forecast of unrevealed demand.

M3 SHARED_AC
    M1 + candidate-own-effect-time residual correction:
        [delta_tau, delta_queue, delta_serviceable_supply]
    Shared future state remains the anchor; own-time state never replaces it.

M4 SHARED_UQ_AC
    M1 + both ETA-envelope sensitivity and action-center residual correction.

The four future methods therefore form a strict 2x2:
                         AC off       AC on
      UQ off             M1           M3
      UQ on              M2           M4
with M0 as the no-explicit-future reference.

Controls
--------
- T2 only: V0,V1 -> V2. This keeps passenger and aircraft labels aligned as
  {null,V0,V1} = {Ground,V0,V1} and {Hold,V0,V1} in joint rows.
- 800,000 PPO timesteps/cell by default; 45 cells = 36.0M requested timesteps.
- 16 SubprocVecEnv x 1280, batch=2048, same PPO hyperparameters as the validated
  6x6 runner.
- passive checkpoint every 50k; immediate evaluation of every checkpoint on
  eval seeds 123/124/125 after each cell.
- fixed observation dimension across E0/E1/E2/E4/E5/J0/J1/J2/J3.
- no unrevealed future passenger request is used.

This script intentionally reuses the validated project physics/evaluation helpers
rather than editing source environment files. Put it beside:
    train_uagmc_6x6_800k.py
    train_uagmc_E0_E2_E6_effect_time_700k.py
    train_uagmc_E3_E6_obs_topology_matrix_800k_FORMAL.py
    train_uagmc_E3_E4_E5_serial_1m.py

Default:
    python train_uagmc_45x800k_formal.py

Smoke test:
    python train_uagmc_45x800k_formal.py --rows E2,J0 --methods M0,M1 --timesteps 50000

Resume:
    python train_uagmc_45x800k_formal.py --resume-root "serial_runs\\uagmc_45x800k_T2_seed1_YYYYMMDD_HHMMSS"
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import gc
import json
import math
import random
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# Validated project-level helpers.
import train_uagmc_6x6_800k as old
import train_uagmc_E0_E2_E6_effect_time_700k as core
import train_uagmc_E3_E6_obs_topology_matrix_800k_FORMAL as mx
import train_uagmc_E3_E4_E5_serial_1m as base


ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "train_data" / "passengers_300.csv"

ROWS = ("E0", "E1", "E2", "E4", "E5", "J0", "J1", "J2", "J3")
SINGLE_ROWS = {"E0", "E1", "E2", "E4", "E5"}
JOINT_ROWS = {"J0", "J1", "J2", "J3"}
METHODS = ("M0", "M1", "M2", "M3", "M4")
METHOD_NAMES = {
    "M0": "RESIDUAL",
    "M1": "SHARED",
    "M2": "SHARED_UQ",
    "M3": "SHARED_AC",
    "M4": "SHARED_UQ_AC",
}
ROW_NAMES = {
    "E0": "SINGLE_PAX_LEGACY",
    "E1": "FIXED_FLEET_LQ",
    "E2": "TURNAROUND_LQ",
    "E4": "TURNAROUND_CHARGE_LQ",
    "E5": "TURNAROUND_CHARGE_TLOF_LQ",
    "J0": "JOINT_SEPARATE_PARALLEL",
    "J1": "JOINT_UNIFIED_PARALLEL",
    "J2": "JOINT_SEPARATE_SEQUENTIAL",
    "J3": "JOINT_UNIFIED_SEQUENTIAL",
}

DEFAULT_TIMESTEPS = 800_000
CHECKPOINT_INTERVAL = 50_000
DEFAULT_TRAIN_SEED = 1
DEFAULT_EVAL_SEEDS = (123, 124, 125)

FLEET_SIZE = 40
MAX_TIME = 600
TURNAROUND_DELAY_MIN = 3.0
PAD_SEPARATION_MIN = 0.25
CHARGER_CAPACITY = 2
FUTURE_HORIZON_MIN = 30.0
MAX_EVENTS_PER_TYPE = 8
ETA_UQ_DELTA_MIN = 2.0

N_ENVS = 16
N_STEPS = 1280
GLOBAL_ROLLOUT = N_ENVS * N_STEPS
BATCH_SIZE = 2048
N_EPOCHS = int(getattr(base, "N_EPOCHS", 10))

DESTINATION = 2
CANDIDATES = (0, 1)
NUM_FRAMES = 6

# Per location fixed resource superset:
# [ready_idle, inbound_empty, turnaround_busy, min_turn_eta,
#  pad_next_free, active_charging, charger_wait, min_charge_eta]
RESOURCE_PER_LOC = 8
RESOURCE_DIM = RESOURCE_PER_LOC * (len(CANDIDATES) + 1)  # + hub V2
FINE_DIM = 4 * MAX_EVENTS_PER_TYPE * len(CANDIDATES)
PROJECT_DIM = 10 * len(CANDIDATES)
UQ_DIM = 2 * len(CANDIDATES)
AC_DIM = 3 * len(CANDIDATES)


# =============================================================================
# Generic helpers
# =============================================================================

def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(np.asarray(x).reshape(-1)[0])
        return y if math.isfinite(y) else default
    except Exception:
        return default


def parse_names(text: str, allowed: Sequence[str]) -> List[str]:
    vals = [x.strip().upper() for x in str(text).split(",") if x.strip()]
    bad = [x for x in vals if x not in allowed]
    if bad:
        raise ValueError(f"Unsupported values={bad}; allowed={list(allowed)}")
    if not vals:
        raise ValueError("Empty selection")
    return vals


def parse_ints(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty integer list")
    return vals


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(mx.jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = {}
            for k in fields:
                v = row.get(k, "")
                if isinstance(v, (dict, list, tuple, np.ndarray)):
                    v = json.dumps(mx.jsonable(v), ensure_ascii=False)
                out[k] = v
            w.writerow(out)


def _blank_stage_stats() -> Dict[str, int]:
    return {
        "single_pax_service_departures": 0,
        "turnaround_starts": 0,
        "turnaround_releases": 0,
        "pad_departure_reservations": 0,
        "pad_landing_reservations": 0,
        "service_pad_blocks": 0,
        "reposition_pad_blocks": 0,
        "e6_charger_active_aircraft_steps": 0,
        "e6_charger_wait_aircraft_steps": 0,
        "e6_max_charger_queue": 0,
    }


def _attach_stage_metadata(scenario: Any, row: str) -> None:
    if row == "E0":
        scenario._e345_stage = "E3_SINGLE_PAX"
    elif row == "E1":
        scenario._e345_stage = "E3_SINGLE_PAX"
    elif row in ("E2", "E4", "J0", "J1", "J2", "J3"):
        scenario._e345_stage = "E4_TURNAROUND"
    elif row == "E5":
        scenario._e345_stage = "E5_PAD"
    else:
        raise ValueError(row)

    if not isinstance(getattr(scenario, "_e345_stats", None), dict):
        scenario._e345_stats = _blank_stage_stats()
    else:
        for k, v in _blank_stage_stats().items():
            scenario._e345_stats.setdefault(k, v)

    if not isinstance(getattr(scenario, "_e345_pad_calendar", None), dict):
        scenario._e345_pad_calendar = {"0": [], "1": [], "2": []}

    scenario.vertiports._e345_scenario = scenario
    scenario._formal45_row = row


# =============================================================================
# Physical row installation
# =============================================================================

def _noop_dispatch_fixed_returns(self) -> None:
    """Joint rows: PPO owns empty-aircraft reposition, so LQ is disabled."""
    return None


def configure_row_process(
    row: str,
    *,
    pad_separation: float,
    charger_capacity: int,
) -> Dict[str, Any]:
    row = str(row).upper()
    core.restore_process_patches()

    # T2 is frozen for this matrix.
    if row == "E0":
        mx.install_topology_patch("T2")
        mx.configure_base_globals(topology="T2", pad_separation=float(pad_separation))
        base.restore_source_methods()
        base.CANDIDATES = list(CANDIDATES)
        base.TO_VERTIPORT = DESTINATION
        base.RETURN_HUB = DESTINATION
        base.MAX_TIME = MAX_TIME
        # Important: only change source service batching to exactly one passenger.
        base.VertiportBuilder.update_objects_state = base._single_pax_update_objects_state
        return {
            "row": row,
            "fleet_mode": "legacy_replenish",
            "single_passenger_service": True,
            "reposition": "source_legacy",
            "turnaround_min": 0.0,
            "finite_charging": False,
            "finite_tlof": False,
        }

    if row == "E1":
        patch = mx.install_physics(
            stage="E3",
            topology="T2",
            pad_separation=float(pad_separation),
            charger_capacity=int(charger_capacity),
        )
        return {"row": row, "mapped_stage": "E3", **dict(patch)}

    if row in ("E2", "J0", "J1", "J2", "J3"):
        patch = mx.install_physics(
            stage="E4",
            topology="T2",
            pad_separation=float(pad_separation),
            charger_capacity=int(charger_capacity),
        )
        if row in JOINT_ROWS:
            base.ConservedFleetScenario._dispatch_fixed_returns = _noop_dispatch_fixed_returns
        return {
            "row": row,
            "mapped_stage": "E4",
            "joint_reposition": row in JOINT_ROWS,
            **dict(patch),
        }

    if row == "E4":
        # Turnaround physics, then finite charging only. No finite pad calendar.
        patch = mx.install_physics(
            stage="E4",
            topology="T2",
            pad_separation=float(pad_separation),
            charger_capacity=int(charger_capacity),
        )
        mx._E6_CHARGER_CAPACITY = int(charger_capacity)
        mx.VertiportBuilder.charge_evtols_at_vertiport = mx._charge_with_finite_capacity
        out = dict(patch)
        out.update(
            {
                "row": row,
                "mapped_stage": "E4+finite_charge_without_pad",
                "finite_charger_capacity": int(charger_capacity),
                "finite_tlof": False,
            }
        )
        return out

    if row == "E5":
        # Existing validated E6 physics is exactly turnaround + finite pad + finite charge.
        patch = mx.install_physics(
            stage="E6",
            topology="T2",
            pad_separation=float(pad_separation),
            charger_capacity=int(charger_capacity),
        )
        out = dict(patch)
        out.update(
            {
                "row": row,
                "mapped_stage": "E6",
                "pad_separation_min": float(pad_separation),
            }
        )
        return out

    raise ValueError(row)


def projection_stage(row: str) -> str:
    """Map the new row ladder onto existing committed-state projection semantics."""
    row = str(row).upper()
    if row in ("E0", "E1"):
        return "E3"  # no turnaround / no finite charger
    if row in ("E2", "J0", "J1", "J2", "J3"):
        return "E4"  # turnaround only
    if row in ("E4", "E5"):
        return "E6"  # turnaround + finite charging; E4 simply has no pad reservations
    raise ValueError(row)


def row_uses_turnaround(row: str) -> bool:
    return row in {"E2", "E4", "E5", "J0", "J1", "J2", "J3"}


def row_uses_finite_charge(row: str) -> bool:
    return row in {"E4", "E5"}


def row_uses_pad(row: str) -> bool:
    return row == "E5"


# =============================================================================
# Fixed observation layout / feature extraction
# =============================================================================

def make_layout(
    *,
    stage: str,
    method: str,
    topology: str,
    max_events: int,
) -> Dict[str, Any]:
    row = str(stage).upper()
    method = str(method).upper()
    if row not in ROWS:
        raise ValueError(row)
    if method not in METHODS:
        raise ValueError(method)
    if str(topology).upper() != "T2":
        raise ValueError("FORMAL45 is intentionally T2-only")

    n = len(CANDIDATES)
    single_dim = 4 + 8 * n
    base_dim = NUM_FRAMES * single_dim
    dims = {
        "base": base_dim,
        "current": RESOURCE_PER_LOC * (n + 1),
        "passenger": 4,
        "fine": 4 * int(max_events) * n,
        "project": 10 * n,
        "uq": 2 * n,
        "ac": 3 * n,
    }
    slices: Dict[str, Tuple[int, int]] = {}
    pos = 0
    for name in ("base", "current", "passenger", "fine", "project", "uq", "ac"):
        width = int(dims[name])
        slices[name] = (pos, pos + width)
        pos += width
    return {
        "row": row,
        "stage": row,
        "method": method,
        "topology": "T2",
        "candidates": list(CANDIDATES),
        "n_candidates": n,
        "num_frames": NUM_FRAMES,
        "single_dim": single_dim,
        "base_dim": base_dim,
        "max_events": int(max_events),
        "dims": dims,
        "slices": slices,
        "total_dim": pos,
        "equal_dimension_across_methods": True,
    }


def _current_superset_vector(
    scenario: Any,
    *,
    row: str,
    charger_capacity: int,
) -> List[float]:
    """Fixed 8 slots per V0/V1/V2; inactive constraints are zero-filled."""
    out: List[float] = []
    use_turn = row_uses_turnaround(row)
    use_charge = row_uses_finite_charge(row)
    use_pad = row_uses_pad(row)

    for vid in list(CANDIDATES) + [DESTINATION]:
        try:
            r = mx._resource_features_for_vid(
                scenario,
                int(vid),
                e6=bool(use_charge),
                charger_capacity=int(charger_capacity),
            )
            ready_idle = float(r[0])
            inbound_empty = float(r[1])
            turnaround_busy = float(r[2]) if use_turn else 0.0
            min_turn_eta = float(r[3]) if use_turn else 0.0
            pad_next_free = float(r[4]) if use_pad else 0.0
            charging = float(r[5])
            active_charging = float(r[6]) if use_charge else 0.0
            min_charge_eta = float(r[7]) if use_charge else 0.0
            charger_wait = max(0.0, charging - active_charging) if use_charge else 0.0
        except Exception:
            # Legacy E0 may not expose all conserved-fleet bookkeeping; retain fixed
            # dimension and fail soft to zeros rather than changing network shape.
            ready_idle = inbound_empty = 0.0
            turnaround_busy = min_turn_eta = 0.0
            pad_next_free = 0.0
            active_charging = charger_wait = min_charge_eta = 0.0

        out.extend(
            [
                ready_idle,
                inbound_empty,
                turnaround_busy,
                min_turn_eta,
                pad_next_free,
                active_charging,
                charger_wait,
                min_charge_eta,
            ]
        )
    if len(out) != RESOURCE_DIM:
        raise RuntimeError(f"resource superset dim {len(out)} != {RESOURCE_DIM}")
    return out


def _safe_project_candidate(
    scenario: Any,
    *,
    row: str,
    vid: int,
    horizon: float,
    charger_capacity: int,
) -> List[float]:
    try:
        return list(
            old._project_candidate(
                scenario,
                stage=projection_stage(row),
                vid=int(vid),
                horizon=max(0.0, float(horizon)),
                charger_capacity=int(charger_capacity),
            )
        )
    except Exception:
        # Conservative legacy fallback: projected queue from already committed
        # access events; no hidden demand is fabricated.
        vp = scenario.vertiports.vertiport_list[str(int(vid))]
        waiting_now = float(len(list(getattr(vp, "person_list", []) or [])))
        try:
            arrived, still = core._committed_access_counts(
                scenario,
                int(vid),
                max(0.0, float(horizon)),
            )
        except Exception:
            arrived, still = 0, 0
        return [
            max(0.0, float(horizon)),
            waiting_now + float(arrived),
            float(still),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]


def _pack_event_set(
    etas: Sequence[float],
    *,
    max_events: int,
    horizon: float,
    center: float,
) -> Tuple[List[float], List[float]]:
    xs = list(etas)[: int(max_events)]
    scale = max(1e-6, float(horizon))
    values = [float((float(x) - float(center)) / scale) for x in xs]
    mask = [1.0] * len(values)
    while len(values) < int(max_events):
        values.append(0.0)
        mask.append(0.0)
    return values, mask


class FormalObservationWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        row: str,
        method: str,
        future_horizon: float,
        max_events: int,
        charger_capacity: int,
        uq_delta: float,
    ):
        super().__init__(env)
        self.row = str(row).upper()
        self.method = str(method).upper()
        self.future_horizon = float(future_horizon)
        self.max_events = int(max_events)
        self.charger_capacity = int(charger_capacity)
        self.uq_delta = float(uq_delta)
        self.layout = make_layout(
            stage=self.row,
            method=self.method,
            topology="T2",
            max_events=self.max_events,
        )
        got = int(np.prod(env.observation_space.shape))
        expected = int(self.layout["base_dim"])
        if got != expected:
            raise RuntimeError(
                f"base observation mismatch row={self.row}: got={got}, expected={expected}"
            )
        self.observation_space = spaces.Box(
            low=-1e6,
            high=1e6,
            shape=(int(self.layout["total_dim"]),),
            dtype=np.float32,
        )

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        scenario = mx.find_scenario(self.env)
        _attach_stage_metadata(scenario, self.row)
        person = old._focal_person(self.env, scenario)

        extras: List[float] = []
        extras.extend(
            _current_superset_vector(
                scenario,
                row=self.row,
                charger_capacity=self.charger_capacity,
            )
        )
        extras.extend(old._focal_od(person))

        access_by_vid = {
            int(v): old._access_time(scenario, person, int(v))
            for v in CANDIDATES
        }
        shared_h = (
            float(np.mean(list(access_by_vid.values())))
            if access_by_vid
            else 0.0
        )

        future_on = self.method in ("M1", "M2", "M3", "M4")
        uq_on = self.method in ("M2", "M4")
        ac_on = self.method in ("M3", "M4")

        # Same DeepSets slots/architecture for all five methods.
        if future_on:
            for vid in CANDIDATES:
                aircraft, passengers = old._event_lists(
                    scenario,
                    int(vid),
                    horizon=self.future_horizon,
                )
                a_eta, a_mask = _pack_event_set(
                    aircraft,
                    max_events=self.max_events,
                    horizon=self.future_horizon,
                    center=shared_h,
                )
                p_eta, p_mask = _pack_event_set(
                    passengers,
                    max_events=self.max_events,
                    horizon=self.future_horizon,
                    center=shared_h,
                )
                extras.extend(a_eta)
                extras.extend(a_mask)
                extras.extend(p_eta)
                extras.extend(p_mask)
        else:
            extras.extend([0.0] * FINE_DIM)

        shared_projects: Dict[int, List[float]] = {}
        if future_on:
            for vid in CANDIDATES:
                proj = _safe_project_candidate(
                    scenario,
                    row=self.row,
                    vid=int(vid),
                    horizon=shared_h,
                    charger_capacity=self.charger_capacity,
                )
                shared_projects[int(vid)] = proj
                extras.extend(proj)
        else:
            extras.extend([0.0] * PROJECT_DIM)

        # Shared-UQ: symmetric local ETA/effect-time envelope around the shared
        # anchor.  Two low-dimensional sensitivities per candidate only.
        if uq_on:
            d = max(1e-6, float(self.uq_delta))
            h_minus = max(0.0, shared_h - d)
            h_plus = shared_h + d
            for vid in CANDIDATES:
                minus = _safe_project_candidate(
                    scenario,
                    row=self.row,
                    vid=int(vid),
                    horizon=h_minus,
                    charger_capacity=self.charger_capacity,
                )
                plus = _safe_project_candidate(
                    scenario,
                    row=self.row,
                    vid=int(vid),
                    horizon=h_plus,
                    charger_capacity=self.charger_capacity,
                )
                # queue sensitivity and serviceable-supply sensitivity per minute
                uq_q = abs(float(plus[1]) - float(minus[1])) / (2.0 * d)
                uq_s = abs(float(plus[5]) - float(minus[5])) / (2.0 * d)
                extras.extend([uq_q, uq_s])
        else:
            extras.extend([0.0] * UQ_DIM)

        # Shared-AC: keep shared future anchor and add only a compact own-effect
        # residual. Never replace the common-horizon state.
        if ac_on:
            for vid in CANDIDATES:
                own_h = float(access_by_vid[int(vid)])
                own = _safe_project_candidate(
                    scenario,
                    row=self.row,
                    vid=int(vid),
                    horizon=own_h,
                    charger_capacity=self.charger_capacity,
                )
                sh = shared_projects[int(vid)]
                delta_tau = (own_h - shared_h) / max(self.future_horizon, 1e-6)
                delta_q = float(own[1]) - float(sh[1])
                delta_s = float(own[5]) - float(sh[5])
                extras.extend([delta_tau, delta_q, delta_s])
        else:
            extras.extend([0.0] * AC_DIM)

        base_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        out = np.concatenate([base_obs, np.asarray(extras, dtype=np.float32)])
        expected = int(self.layout["total_dim"])
        if out.shape != (expected,):
            raise RuntimeError(
                f"formal observation shape={out.shape}, expected={(expected,)} "
                f"row={self.row} method={self.method}"
            )
        return out.astype(np.float32, copy=False)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            _attach_stage_metadata(mx.find_scenario(self.env), self.row)
            return self._transform(obs), info
        _attach_stage_metadata(mx.find_scenario(self.env), self.row)
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
# Unified extractor: identical dimension/parameter count for M0..M4
# =============================================================================

class Formal45Extractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        layout: Optional[Dict[str, Any]] = None,
    ):
        if layout is None:
            raise ValueError("layout required")
        super().__init__(observation_space, features_dim)
        self.layout = dict(layout)
        self.slices = dict(layout["slices"])
        self.dims = dict(layout["dims"])
        self.num_frames = int(layout["num_frames"])
        self.single_dim = int(layout["single_dim"])
        self.n_candidates = int(layout["n_candidates"])
        self.max_events = int(layout["max_events"])

        self.frame_encoder = nn.Sequential(
            nn.Linear(self.single_dim, 128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, 128, num_layers=1, batch_first=True)

        self.passenger_branch = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        numeric_dim = (
            int(self.dims["current"])
            + int(self.dims["project"])
            + int(self.dims["uq"])
            + int(self.dims["ac"])
        )
        self.numeric_branch = nn.Sequential(
            nn.Linear(numeric_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.event_phi = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        # bias=False on the final projection prevents a non-informative M0 zero
        # event set from becoming a trainable constant feature.
        self.event_rho = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            nn.ReLU(),
        )

        concat_dim = 128 + 32 + 64 + 32 * self.n_candidates
        self.output_proj = nn.Sequential(
            nn.Linear(concat_dim, int(features_dim)),
            nn.ReLU(),
        )

    def _slice(self, obs: torch.Tensor, name: str) -> torch.Tensor:
        lo, hi = self.slices[name]
        return obs[:, int(lo):int(hi)]

    def _encode_sets(self, obs: torch.Tensor) -> torch.Tensor:
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
            pooled = emb.sum(dim=1)
            reps.append(self.event_rho(pooled))
        return torch.cat(reps, dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations.float()
        base_obs = self._slice(obs, "base")
        B = base_obs.shape[0]
        frames = base_obs.reshape(B, self.num_frames, self.single_dim)
        x = self.frame_encoder(frames)
        x, _ = self.lstm(x)
        temporal = x[:, -1, :]

        passenger = self.passenger_branch(self._slice(obs, "passenger"))
        numeric = torch.cat(
            [
                self._slice(obs, "current"),
                self._slice(obs, "project"),
                self._slice(obs, "uq"),
                self._slice(obs, "ac"),
            ],
            dim=1,
        )
        numeric = self.numeric_branch(numeric)
        events = self._encode_sets(obs)
        return self.output_proj(torch.cat([temporal, passenger, numeric, events], dim=1))


# =============================================================================
# Joint scheduling: small Discrete(9) pair action + four policy factorizations
# =============================================================================

def _eligible_hub_aircraft(scenario: Any) -> List[Any]:
    local = list(
        scenario.vertiports.evtols_at_vertiport.get(str(DESTINATION), [])
    )
    eligible = [
        e
        for e in local
        if (
            base.state_name(e) == "IDLE"
            and not base.passenger_ids(e)
            and not base.is_turnaround_busy(e)
        )
    ]
    eligible.sort(key=lambda e: str(getattr(e, "id", "")))
    return eligible


def _dispatch_joint_aircraft(scenario: Any, aircraft_action: int) -> bool:
    """0 Hold, 1 -> V0, 2 -> V1. At most one deterministic eligible aircraft."""
    ai = int(aircraft_action)
    if ai == 0:
        return False
    if ai not in (1, 2):
        raise ValueError(ai)
    target = CANDIDATES[ai - 1]
    eligible = _eligible_hub_aircraft(scenario)
    if not eligible:
        return False
    return bool(
        base._start_empty_reposition_checked(
            scenario=scenario,
            evtol=eligible[0],
            origin=str(DESTINATION),
            destination=str(target),
        )
    )


class JointActionWrapper(gym.Wrapper):
    """Decode Discrete(9) -> passenger action x aircraft {Hold,V0,V1}."""
    def __init__(self, env: gym.Env, *, row: str):
        super().__init__(env)
        self.row = str(row).upper()
        if self.row not in JOINT_ROWS:
            raise ValueError(self.row)
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError(f"expected Discrete passenger action, got {env.action_space}")

        # Legacy UAGMC may still advertise a wider Discrete passenger space.
        # FORMAL T2 freezes the passenger semantics to the first three labels:
        # 0=Ground, 1=V0, 2=V1.  PPO therefore only sees these three labels.
        underlying_n = int(env.action_space.n)
        if underlying_n < 3:
            raise RuntimeError(
                f"T2 joint control requires passenger actions 0/1/2 "
                f"={{Ground,V0,V1}}, but underlying action_space={env.action_space}"
            )
        self.underlying_passenger_n = underlying_n
        self.passenger_n = 3
        self.aircraft_n = 3
        self.action_space = spaces.Discrete(self.passenger_n * self.aircraft_n)
        self.observation_space = env.observation_space

    @staticmethod
    def decode(action: Any) -> Tuple[int, int]:
        idx = int(np.asarray(action).reshape(-1)[0])
        if idx < 0 or idx >= 9:
            raise ValueError(idx)
        return idx // 3, idx % 3

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        _attach_stage_metadata(mx.find_scenario(self.env), self.row)
        return out

    def step(self, action):
        passenger_action, aircraft_action = self.decode(action)
        scenario = mx.find_scenario(self.env)
        _attach_stage_metadata(scenario, self.row)

        # Both commitments belong to the same control epoch. The policy may be
        # parallel or sequentially factorized, but physical dispatch is immediate
        # and does not introduce an extra simulation/PPO step.
        aircraft_dispatched = _dispatch_joint_aircraft(scenario, aircraft_action)
        out = self.env.step(passenger_action)

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info)
            info.update(
                {
                    "joint_pair_action": int(passenger_action * 3 + aircraft_action),
                    "passenger_action": passenger_action,
                    "aircraft_action": aircraft_action,
                    "aircraft_dispatched": bool(aircraft_dispatched),
                }
            )
            return obs, reward, terminated, truncated, info
        if len(out) == 4:
            obs, reward, done, info = out
            info = dict(info)
            info.update(
                {
                    "joint_pair_action": int(passenger_action * 3 + aircraft_action),
                    "passenger_action": passenger_action,
                    "aircraft_action": aircraft_action,
                    "aircraft_dispatched": bool(aircraft_dispatched),
                }
            )
            return obs, reward, done, info
        raise RuntimeError(f"unexpected step tuple length={len(out)}")


class JointPairHead(nn.Module):
    """
    Output 9 pair logits.  J0/J1 are parallel factorizations; J2/J3 are
    passenger->aircraft sequential factorizations.  No combinatorial fleet action.
    """
    def __init__(self, latent_dim: int, mode: str):
        super().__init__()
        self.mode = str(mode).upper()
        if self.mode == "J0":
            self.passenger = nn.Linear(latent_dim, 3)
            self.aircraft = nn.Linear(latent_dim, 3)
        elif self.mode == "J2":
            self.passenger = nn.Linear(latent_dim, 3)
            self.aircraft_cond = nn.Linear(latent_dim, 9)
        elif self.mode in ("J1", "J3"):
            # One scorer shared by demand(+1) and supply(-1) roles.
            # Inputs: latent + role_sign + action_onehot(3) + passenger_commit(3).
            self.shared = nn.Sequential(
                nn.Linear(latent_dim + 1 + 3 + 3, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        else:
            raise ValueError(mode)
        self.apply(self._init_small)

    @staticmethod
    def _init_small(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _shared_scores(
        self,
        latent: torch.Tensor,
        *,
        role_sign: float,
        commit: Optional[int],
    ) -> torch.Tensor:
        B = latent.shape[0]
        rows = []
        for a in range(3):
            role = torch.full(
                (B, 1), float(role_sign), dtype=latent.dtype, device=latent.device
            )
            action_oh = torch.zeros(B, 3, dtype=latent.dtype, device=latent.device)
            action_oh[:, a] = 1.0
            commit_oh = torch.zeros(B, 3, dtype=latent.dtype, device=latent.device)
            if commit is not None:
                commit_oh[:, int(commit)] = 1.0
            x = torch.cat([latent, role, action_oh, commit_oh], dim=1)
            rows.append(self.shared(x))
        return torch.cat(rows, dim=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if self.mode == "J0":
            p = self.passenger(latent)              # [B,3]
            a = self.aircraft(latent)               # [B,3]
            return (p.unsqueeze(2) + a.unsqueeze(1)).reshape(-1, 9)

        if self.mode == "J2":
            p = self.passenger(latent)
            a = self.aircraft_cond(latent).reshape(-1, 3, 3)
            return (p.unsqueeze(2) + a).reshape(-1, 9)

        # Unified role semantics. Passenger role is +1 demand; aircraft role is
        # -1 supply.  J3 additionally conditions the aircraft score on the chosen
        # passenger label through the pair logit construction.
        p = self._shared_scores(latent, role_sign=+1.0, commit=None)
        if self.mode == "J1":
            a = self._shared_scores(latent, role_sign=-1.0, commit=None)
            return (p.unsqueeze(2) + a.unsqueeze(1)).reshape(-1, 9)

        pair_rows = []
        for p_idx in range(3):
            a_given_p = self._shared_scores(
                latent, role_sign=-1.0, commit=p_idx
            )
            pair_rows.append(p[:, p_idx:p_idx + 1] + a_given_p)
        return torch.stack(pair_rows, dim=1).reshape(-1, 9)


class JointPairPolicy(ActorCriticPolicy):
    """SB3 policy with a tiny custom 9-logit factorized joint action head."""
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        joint_mode: str = "J0",
        **kwargs,
    ):
        self.joint_mode = str(joint_mode).upper()
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        if not isinstance(action_space, spaces.Discrete) or int(action_space.n) != 9:
            raise RuntimeError(f"JointPairPolicy requires Discrete(9), got {action_space}")
        self.action_net = JointPairHead(
            int(self.mlp_extractor.latent_dim_pi),
            self.joint_mode,
        )
        # super() already built an optimizer for the old action_net; rebuild it so
        # the factorized head parameters are trained.
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )


# =============================================================================
# Environment / model construction
# =============================================================================

def make_formal_env_factory(
    *,
    row: str,
    method: str,
    fleet_size: int,
    env_index: int,
    run_dir: Path,
    future_horizon: float,
    max_events: int,
    pad_separation: float,
    charger_capacity: int,
    uq_delta: float,
    max_time: int,
):
    def _init():
        configure_row_process(
            row,
            pad_separation=float(pad_separation),
            charger_capacity=int(charger_capacity),
        )
        fleet_mode = "legacy_replenish" if row == "E0" else "conserved_closed_loop"
        env = core.make_env(
            max_time=int(max_time),
            log_dir=run_dir / "monitor",
            env_index=int(env_index),
            person_spawn_file=str(TRAIN_FILE),
            candidate_from_vertiports=list(CANDIDATES),
            to_vertiport=DESTINATION,
            enable_logger=False,
            fleet_mode=fleet_mode,
            fleet_size=(None if row == "E0" else int(fleet_size)),
            fleet_assertions=(row != "E0"),
        )()
        scenario = mx.find_scenario(env)
        _attach_stage_metadata(scenario, row)

        # Terminal snapshot wrapper from the validated equal-dimension runner.
        env = core.ExperimentWrapper(
            env,
            stage=("E0" if row == "E0" else "E4"),
            topology="T2",
            fleet_size=int(fleet_size),
        )
        if row in JOINT_ROWS:
            env = JointActionWrapper(env, row=row)
        env = FormalObservationWrapper(
            env,
            row=row,
            method=method,
            future_horizon=future_horizon,
            max_events=max_events,
            charger_capacity=charger_capacity,
            uq_delta=uq_delta,
        )
        return env
    return _init


def build_train_env(
    *,
    stage: str,
    method: str,
    topology: str,
    fleet_size: int,
    seed: int,
    run_dir: Path,
    future_horizon: float,
    max_events: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> VecNormalize:
    row = str(stage).upper()
    if str(topology).upper() != "T2":
        raise ValueError("FORMAL45 is T2-only")
    factories = [
        make_formal_env_factory(
            row=row,
            method=method,
            fleet_size=fleet_size,
            env_index=i,
            run_dir=run_dir,
            future_horizon=future_horizon,
            max_events=max_events,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
            uq_delta=ETA_UQ_DELTA_MIN,
            max_time=max_time,
        )
        for i in range(N_ENVS)
    ]
    raw = SubprocVecEnv(factories, start_method="spawn")
    raw.seed(int(seed))
    return VecNormalize(
        raw,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=float(base.GAMMA),
    )


def build_eval_raw_env(
    *,
    stage: str,
    method: str,
    topology: str,
    fleet_size: int,
    run_dir: Path,
    future_horizon: float,
    max_events: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> DummyVecEnv:
    if str(topology).upper() != "T2":
        raise ValueError("FORMAL45 is T2-only")
    return DummyVecEnv(
        [
            make_formal_env_factory(
                row=str(stage).upper(),
                method=method,
                fleet_size=fleet_size,
                env_index=9999,
                run_dir=run_dir,
                future_horizon=future_horizon,
                max_events=max_events,
                pad_separation=pad_separation,
                charger_capacity=charger_capacity,
                uq_delta=ETA_UQ_DELTA_MIN,
                max_time=max_time,
            )
        ]
    )


def build_model(
    *,
    env: VecNormalize,
    stage: str,
    method: str,
    topology: str,
    max_events: int,
    seed: int,
    run_dir: Path,
    device: str,
) -> PPO:
    row = str(stage).upper()
    layout = make_layout(
        stage=row,
        method=method,
        topology=topology,
        max_events=max_events,
    )
    policy_kwargs: Dict[str, Any] = dict(
        features_extractor_class=Formal45Extractor,
        features_extractor_kwargs=dict(features_dim=128, layout=layout),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    policy: Any = "MlpPolicy"
    if row in JOINT_ROWS:
        policy = JointPairPolicy
        policy_kwargs["joint_mode"] = row

    return PPO(
        policy=policy,
        env=env,
        learning_rate=base.linear_schedule(base.INITIAL_LR),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=float(base.GAMMA),
        gae_lambda=float(base.GAE_LAMBDA),
        clip_range=float(base.CLIP_RANGE),
        ent_coef=float(base.ENT_COEF),
        vf_coef=float(base.VF_COEF),
        max_grad_norm=float(base.MAX_GRAD_NORM),
        policy_kwargs=policy_kwargs,
        seed=int(seed),
        verbose=1,
        device=device,
        tensorboard_log=str(run_dir / "tb"),
    )


# =============================================================================
# Evaluation extension: derive passenger / aircraft shares for joint rows
# =============================================================================

_ORIGINAL_OLD_EVALUATE = old.evaluate_checkpoint


def evaluate_checkpoint_with_joint_shares(**kwargs) -> Dict[str, Any]:
    row = str(kwargs.get("stage", "")).upper()
    result = _ORIGINAL_OLD_EVALUATE(**kwargs)
    if row not in JOINT_ROWS:
        return result

    pair_shares = [fnum(result.get(f"action_{i}_share"), 0.0) for i in range(9)]
    for p in range(3):
        result[f"passenger_action_{p}_share"] = float(
            sum(pair_shares[p * 3 + a] for a in range(3))
        )
    for a in range(3):
        result[f"aircraft_action_{a}_share"] = float(
            sum(pair_shares[p * 3 + a] for p in range(3))
        )
    return result


# =============================================================================
# Matrix bookkeeping / contrasts
# =============================================================================

def _read_summary(root: Path, row: str, method: str) -> Optional[Dict[str, Any]]:
    p = root / f"{row}__{method}" / "analysis" / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def refresh_master_tables(root: Path) -> None:
    rows: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in ROWS:
        for method in METHODS:
            s = _read_summary(root, row, method)
            if s is None:
                continue
            final = s.get("final", {}) or {}
            best = s.get("best_checkpoint", {}) or {}
            rec = {
                "row": row,
                "row_name": ROW_NAMES[row],
                "method": method,
                "method_name": METHOD_NAMES[method],
                "final_step": final.get("train_step"),
                "final_Jsys": final.get("system_person_minutes_per_passenger_mean"),
                "final_ATT": final.get("ATT_mean"),
                "final_completion": final.get("completion_rate_mean"),
                "best_step": best.get("train_step"),
                "best_Jsys": best.get("system_person_minutes_per_passenger_mean"),
                "late_Jsys_mean": s.get("late_Jsys_mean"),
                "late_Jsys_std": s.get("late_Jsys_std"),
                "late_ATT_mean": s.get("late_ATT_mean"),
                "late_completion_mean": s.get("late_completion_mean"),
            }
            rows.append(rec)
            by_key[(row, method)] = rec
    write_csv(root / "formal45_master.csv", rows)

    # Representation contrasts inside each system row.
    contrasts: List[Dict[str, Any]] = []
    pairs = [
        ("FUTURE", "M0", "M1"),
        ("UQ_on_Shared", "M1", "M2"),
        ("AC_on_Shared", "M1", "M3"),
        ("AC_on_SharedUQ", "M2", "M4"),
        ("UQ_on_SharedAC", "M3", "M4"),
    ]
    for row in ROWS:
        for name, control, treatment in pairs:
            c = by_key.get((row, control))
            t = by_key.get((row, treatment))
            if c is None or t is None:
                continue
            cj = fnum(c.get("late_Jsys_mean"))
            tj = fnum(t.get("late_Jsys_mean"))
            contrasts.append(
                {
                    "type": "representation",
                    "row": row,
                    "contrast": name,
                    "control": control,
                    "treatment": treatment,
                    "control_late_Jsys": cj,
                    "treatment_late_Jsys": tj,
                    "improvement_control_minus_treatment": cj - tj,
                    "improvement_pct": (
                        100.0 * (cj - tj) / cj
                        if math.isfinite(cj) and abs(cj) > 1e-12 else float("nan")
                    ),
                }
            )

    # Joint 2x2 contrasts for each representation column.
    for method in METHODS:
        vals = {
            j: by_key.get((j, method))
            for j in ("J0", "J1", "J2", "J3")
        }
        if all(v is not None for v in vals.values()):
            j = {k: fnum(v.get("late_Jsys_mean")) for k, v in vals.items()}
            contrasts.extend(
                [
                    {
                        "type": "joint",
                        "method": method,
                        "contrast": "semantic_unification_parallel",
                        "control": "J0",
                        "treatment": "J1",
                        "improvement_control_minus_treatment": j["J0"] - j["J1"],
                    },
                    {
                        "type": "joint",
                        "method": method,
                        "contrast": "sequential_coupling_separate",
                        "control": "J0",
                        "treatment": "J2",
                        "improvement_control_minus_treatment": j["J0"] - j["J2"],
                    },
                    {
                        "type": "joint",
                        "method": method,
                        "contrast": "semantic_unification_sequential",
                        "control": "J2",
                        "treatment": "J3",
                        "improvement_control_minus_treatment": j["J2"] - j["J3"],
                    },
                    {
                        "type": "joint",
                        "method": method,
                        "contrast": "sequential_coupling_unified",
                        "control": "J1",
                        "treatment": "J3",
                        "improvement_control_minus_treatment": j["J1"] - j["J3"],
                    },
                    {
                        "type": "joint",
                        "method": method,
                        "contrast": "2x2_interaction",
                        "value": (j["J2"] - j["J3"]) - (j["J0"] - j["J1"]),
                    },
                ]
            )
    write_csv(root / "formal45_contrasts.csv", contrasts)


# =============================================================================
# Main experiment
# =============================================================================

def install_old_runner_hooks() -> None:
    """Reuse the validated 6x6 train/checkpoint/eval engine with new semantics."""
    old.METHOD_NAMES = dict(METHOD_NAMES)
    old.METHODS = tuple(METHODS)
    old.STAGES = tuple(ROWS)
    old.CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL
    old.N_ENVS = N_ENVS
    old.N_STEPS = N_STEPS
    old.GLOBAL_ROLLOUT = GLOBAL_ROLLOUT
    old.BATCH_SIZE = BATCH_SIZE
    old.N_EPOCHS = N_EPOCHS
    old.NUM_FRAMES = NUM_FRAMES
    old.make_layout = make_layout
    old.build_train_env = build_train_env
    old.build_eval_raw_env = build_eval_raw_env
    old.build_model = build_model
    old.evaluate_checkpoint = evaluate_checkpoint_with_joint_shares


def run_matrix(args: argparse.Namespace) -> Path:
    rows = parse_names(args.rows, ROWS)
    methods = parse_names(args.methods, METHODS)
    eval_seeds = parse_ints(args.eval_seeds)
    requested_steps = int(args.timesteps)
    if requested_steps <= 0 or requested_steps % CHECKPOINT_INTERVAL != 0:
        raise ValueError(
            f"timesteps must be positive and divisible by {CHECKPOINT_INTERVAL}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.resume_root:
        root = Path(args.resume_root).expanduser()
        if not root.is_absolute():
            root = (ROOT / root).resolve()
        else:
            root = root.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = (
            ROOT
            / "serial_runs"
            / f"uagmc_joint20x{requested_steps//1000}k_T2_seed{args.seed}_{stamp}"
        ).resolve()
    root.mkdir(parents=True, exist_ok=True)

    install_old_runner_hooks()

    manifest = {
        "experiment": "UAGMC_JOINT20_CONTINUATION" if requested_steps == 800_000 else "UAGMC_JOINT_CONTINUATION_SUBSET",
        "created": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "methods": methods,
        "all_formal_rows": list(ROWS),
        "all_formal_methods": list(METHODS),
        "cells_requested": len(rows) * len(methods),
        "timesteps_per_cell": requested_steps,
        "requested_total_timesteps": len(rows) * len(methods) * requested_steps,
        "formal_full_matrix_timesteps": 45 * 800_000,
        "train_seed": int(args.seed),
        "eval_seeds": eval_seeds,
        "topology": "T2",
        "candidates": list(CANDIDATES),
        "destination": DESTINATION,
        "fleet_size_non_E0": FLEET_SIZE,
        "turnaround_delay_min": TURNAROUND_DELAY_MIN,
        "finite_charger_capacity_E4_E5": int(args.charger_capacity),
        "pad_separation_min_E5": float(args.pad_separation),
        "future_horizon_min": FUTURE_HORIZON_MIN,
        "max_events_per_type": MAX_EVENTS_PER_TYPE,
        "eta_uq_delta_min": float(args.uq_delta),
        "observation": {
            "fixed_dimension_across_all_rows_methods": True,
            "resource_superset_dim": RESOURCE_DIM,
            "future_event_dim": FINE_DIM,
            "project_dim": PROJECT_DIM,
            "uq_dim": UQ_DIM,
            "ac_dim": AC_DIM,
            "inactive_slots_zero_filled": True,
        },
        "joint": {
            "physics": "new E2: single-pax + fixed fleet + turnaround",
            "aircraft_actions": ["Hold", "V0", "V1"],
            "joint_action_space": "Discrete(9) pair; no per-aircraft action vector",
            "max_reposition_per_decision": 1,
            "eligible_aircraft_selection": "deterministic lowest ID at hub",
            "J0": "Separate + Parallel",
            "J1": "Unified role-conditioned scorer + Parallel",
            "J2": "Separate + Sequential-factorized",
            "J3": "Unified role-conditioned scorer + Sequential-factorized",
        },
        "methods_factorial": {
            "M0": "Residual, no explicit committed future",
            "M1": "Shared",
            "M2": "Shared + ETA-envelope UQ",
            "M3": "Shared + Action-Center residual",
            "M4": "Shared + ETA-envelope UQ + Action-Center residual",
        },
        "compute": {
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "global_rollout": GLOBAL_ROLLOUT,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "device": device,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
        },
        "unrevealed_future_requests_used": False,
    }
    write_json(root / "experiment_manifest.json", manifest)

    status_rows: List[Dict[str, Any]] = []
    total = len(rows) * len(methods)
    idx = 0
    for row in rows:
        for method in methods:
            idx += 1
            cid = f"{row}__{method}"
            spec = {
                "cell_id": cid,
                "row": row,
                "row_name": ROW_NAMES[row],
                "method": method,
                "method_name": METHOD_NAMES[method],
                "requested_timesteps": requested_steps,
                "physics": configure_row_process(
                    row,
                    pad_separation=float(args.pad_separation),
                    charger_capacity=int(args.charger_capacity),
                ),
            }
            # Restore before subprocess construction; every worker installs its own
            # row patch again in make_formal_env_factory.
            core.restore_process_patches()
            write_json(root / cid / "formal_cell_spec.json", spec)

            print("\n" + "#" * 132)
            print(
                f"FORMAL45 CELL {idx}/{total}: {cid} | {ROW_NAMES[row]} | "
                f"{METHOD_NAMES[method]} | {requested_steps:,} steps"
            )
            print("#" * 132, flush=True)
            started = time.time()
            try:
                summary = old.run_cell(
                    stage=row,
                    method=method,
                    topology="T2",
                    requested_steps=requested_steps,
                    seed=int(args.seed),
                    device=device,
                    root=root,
                    eval_seeds=eval_seeds,
                    rule_refs={},
                    fleet_size=FLEET_SIZE,
                    future_horizon=FUTURE_HORIZON_MIN,
                    max_events=MAX_EVENTS_PER_TYPE,
                    pad_separation=float(args.pad_separation),
                    charger_capacity=int(args.charger_capacity),
                    max_time=MAX_TIME,
                )
                status_rows.append(
                    {
                        "cell_id": cid,
                        "row": row,
                        "method": method,
                        "status": "DONE",
                        "elapsed_seconds": time.time() - started,
                        "late_Jsys_mean": summary.get("late_Jsys_mean"),
                        "late_Jsys_std": summary.get("late_Jsys_std"),
                    }
                )
            except Exception as exc:
                status_rows.append(
                    {
                        "cell_id": cid,
                        "row": row,
                        "method": method,
                        "status": "ERROR",
                        "elapsed_seconds": time.time() - started,
                        "error": repr(exc),
                    }
                )
                write_json(
                    root / cid / "formal45_outer_error.json",
                    {
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                write_csv(root / "formal45_status.csv", status_rows)
                refresh_master_tables(root)
                if not args.continue_on_error:
                    raise
            finally:
                core.restore_process_patches()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            write_csv(root / "formal45_status.csv", status_rows)
            refresh_master_tables(root)

    print("\nFORMAL45 DONE")
    print(f"Root: {root}")
    print(f"Master: {root / 'formal45_master.csv'}")
    print(f"Contrasts: {root / 'formal45_contrasts.csv'}")
    return root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="J0,J1,J2,J3")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    p.add_argument("--eval-seeds", default=",".join(str(x) for x in DEFAULT_EVAL_SEEDS))
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--pad-separation", type=float, default=PAD_SEPARATION_MIN)
    p.add_argument("--charger-capacity", type=int, default=CHARGER_CAPACITY)
    p.add_argument("--uq-delta", type=float, default=ETA_UQ_DELTA_MIN)
    p.add_argument("--resume-root", default=None)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Runtime globals used by wrappers. Keep CLI overrides deterministic inside
    # spawned workers because argparse is re-imported only in __main__.
    global ETA_UQ_DELTA_MIN
    ETA_UQ_DELTA_MIN = float(args.uq_delta)
    run_matrix(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
