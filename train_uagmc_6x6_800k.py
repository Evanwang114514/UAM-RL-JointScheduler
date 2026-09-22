# -*- coding: utf-8 -*-
"""
Controlled UAGMC 6x6 matrix, 800k per cell
===========================================

Physical environments (rows)
----------------------------
E0 : source UAGMC legacy automatic replenishment
E2 : fixed conserved fleet + responsive Longest-Queue reposition,
     source service batch/capacity unchanged
E3 : E2 + exactly one passenger per service flight
E4 : E3 + 3 min turnaround
E5 : E4 + finite shared TLOF/pad calendar
E6 : E5 + finite charging capacity

Methods (columns)
-----------------
M0 EA_UAGMC
    Environment-aligned current-state baseline.  It keeps the source UAGMC
    stacked observation and only appends CURRENT variables needed by newly
    introduced physical constraints.  It does not expose committed future
    arrivals.

M1 PASSENGER_RESIDUAL
    M0 + a dedicated residual branch for the focal passenger OD.  Information
    content is unchanged with respect to source UAGMC (OD already exists in
    the source observation); the network path is changed so passenger identity
    is not easily drowned by the system state.

M2 COARSE_FUTURE
    M1 + committed future information at low granularity.  For every candidate,
    only counts of inbound aircraft and already-committed passengers inside a
    fixed future horizon are exposed.

M3 FINE_FUTURE_DEEPSET
    M1 + the SAME committed future events as M2, but with individual ETAs.
    A permutation-invariant DeepSets branch encodes variable-cardinality event
    sets.  This distinguishes equal-count / different-ETA states.

M4 ACTION_CENTER
    M3 information budget, re-centered on each candidate's own focal-passenger
    access/effect time.  Known/committed physical state is projected to that
    candidate-specific time.  No unrevealed passenger request is used.

M5 ACTION_CENTER_CF
    M4 + deterministic counterfactual residuals caused by assigning the focal
    passenger to each candidate (self-demand / pressure delta).

Controlled design
-----------------
- Main matrix default topology: T2 (V0,V1 -> V2)
- Same passenger trace, PPO hyperparameters, reward, train seed and compute
  profile across all 36 cells
- 16 SubprocVecEnv workers x 1280 steps; batch=2048; PPO update on CUDA
- 50k passive checkpoints; NO evaluation during training
- Immediately after each cell finishes, replay EVERY saved 50k checkpoint on
  eval seeds 123/124/125, save ATT/AWT/completion/Jsys/N/action statistics
- SPF/STTF/QTTI2 no-learning references are run once per stage/topology and
  stored with the matrix so checkpoint gaps to rules are explicit
- Resume-safe: completed cells are skipped when --resume-root is supplied

Typical usage
-------------
python train_uagmc_6x6_800k.py

Smoke test:
python train_uagmc_6x6_800k.py --stages E3 --methods M0,M1 --timesteps 50000

Resume:
python train_uagmc_6x6_800k.py --resume-root "serial_runs\\uagmc_6x6_T2_800k_seed1_YYYYMMDD_HHMMSS"

The script is intentionally self-contained at the experiment layer and imports
the already validated stage physics from the project instead of editing source
environment files.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import train_uagmc_E0_E2_E6_effect_time_700k as core
import train_uagmc_E3_E6_obs_topology_matrix_800k_FORMAL as mx
import train_uagmc_E3_E4_E5_serial_1m as base
import run_uagmc_E0_E2_E6_rule_baselines as rulemod


ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "train_data" / "passengers_300.csv"

STAGES = ("E0", "E2", "E3", "E4", "E5", "E6")
METHODS = ("M0", "M1", "M2", "M3", "M4", "M5")
METHOD_NAMES = {
    "M0": "EA_UAGMC",
    "M1": "PASSENGER_RESIDUAL",
    "M2": "COARSE_FUTURE",
    "M3": "FINE_FUTURE_DEEPSET",
    "M4": "ACTION_CENTER",
    "M5": "ACTION_CENTER_CF",
}

DEFAULT_TOPOLOGY = "T2"
DEFAULT_TIMESTEPS = 800_000
CHECKPOINT_INTERVAL = 50_000
TRAIN_SEED = 1
DEFAULT_EVAL_SEEDS = (123, 124, 125)

FLEET_SIZE = 40
MAX_TIME = 10000
PAD_SEPARATION_MIN = 0.5
CHARGER_CAPACITY = 2
FUTURE_HORIZON_MIN = 30.0
MAX_EVENTS_PER_TYPE = 8

N_ENVS = 16
N_STEPS = 1280
GLOBAL_ROLLOUT = N_ENVS * N_STEPS
BATCH_SIZE = 2048
N_EPOCHS = int(getattr(base, "N_EPOCHS", 10))

DESTINATION = 2
NUM_FRAMES = 6

STAGE_RANK = {"E0": 0, "E2": 1, "E3": 2, "E4": 3, "E5": 4, "E6": 5}


# =============================================================================
# Small utilities
# =============================================================================

def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def fmean(values: Iterable[Any]) -> float:
    arr = np.asarray([fnum(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def fstd(values: Iterable[Any]) -> float:
    arr = np.asarray([fnum(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.std(ddof=0)) if len(arr) else float("nan")


def parse_list(text: str, allowed: Sequence[str]) -> List[str]:
    xs = [x.strip().upper() for x in str(text).split(",") if x.strip()]
    bad = [x for x in xs if x not in allowed]
    if bad:
        raise ValueError(f"unsupported values {bad}; allowed={list(allowed)}")
    return xs


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            cooked = {}
            for k in fields:
                v = row.get(k, "")
                if isinstance(v, (dict, list, tuple)):
                    v = json.dumps(v, ensure_ascii=False)
                cooked[k] = v
            w.writerow(cooked)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def candidates_for(topology: str) -> List[int]:
    return list(core.candidates_for(str(topology).upper()))


def stage_ge(stage: str, target: str) -> bool:
    return STAGE_RANK[str(stage).upper()] >= STAGE_RANK[str(target).upper()]


# =============================================================================
# Observation layout
# =============================================================================

def current_feature_dim(stage: str, n_candidates: int) -> int:
    """Environment-aligned CURRENT-only augmentation dimensionality."""
    stage = str(stage).upper()
    per_candidate = 0
    hub = 0
    if stage_ge(stage, "E2"):
        per_candidate += 1  # ready idle
        hub += 1
    if stage_ge(stage, "E3"):
        per_candidate += 1  # effective service capacity
    if stage_ge(stage, "E4"):
        per_candidate += 2  # turnaround busy, min release ETA
        hub += 2
    if stage_ge(stage, "E5"):
        per_candidate += 1  # TLOF/pad next-free ETA
        hub += 1
    if stage_ge(stage, "E6"):
        per_candidate += 3  # active chargers, charger wait, min charge ETA
        hub += 3
    return n_candidates * per_candidate + hub


def make_layout(
    *,
    stage: str,
    method: str,
    topology: str,
    max_events: int,
) -> Dict[str, Any]:
    cands = candidates_for(topology)
    n = len(cands)
    single_dim = 4 + 8 * n
    base_dim = NUM_FRAMES * single_dim

    dims = {
        "base": base_dim,
        "current": current_feature_dim(stage, n),
        "passenger": 4 if method in ("M1", "M2", "M3", "M4", "M5") else 0,
        "coarse": 2 * n if method == "M2" else 0,
        "fine": (4 * max_events * n) if method in ("M3", "M4", "M5") else 0,
        "project": 10 * n if method in ("M4", "M5") else 0,
        "cf": 3 * n if method == "M5" else 0,
    }

    slices: Dict[str, Tuple[int, int]] = {}
    pos = 0
    for name in ("base", "current", "passenger", "coarse", "fine", "project", "cf"):
        width = int(dims[name])
        slices[name] = (pos, pos + width)
        pos += width

    return {
        "stage": stage,
        "method": method,
        "topology": topology,
        "candidates": cands,
        "n_candidates": n,
        "num_frames": NUM_FRAMES,
        "single_dim": single_dim,
        "base_dim": base_dim,
        "max_events": int(max_events),
        "dims": dims,
        "slices": slices,
        "total_dim": pos,
    }


# =============================================================================
# State/event extraction
# =============================================================================

def _uam_state(env: Any) -> Dict[str, Any]:
    uam = mx.find_uam_wrapper(env)
    return getattr(uam, "state", {}) or {}


def _focal_person(env: Any, scenario: Any) -> Optional[Any]:
    state = _uam_state(env)
    waiting = list(state.get("waiting_decisions", []) or [])
    if not waiting:
        return None
    return core._lookup_person(scenario, waiting[0])


def _focal_od(person: Optional[Any]) -> List[float]:
    if person is None:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(person.origin_position[0]),
        float(person.origin_position[1]),
        float(person.destination_position[0]),
        float(person.destination_position[1]),
    ]


def _access_time(scenario: Any, person: Optional[Any], vid: int) -> float:
    if person is None:
        return 0.0
    return float(core._access_time(scenario, person, int(vid)))


def _event_lists(
    scenario: Any,
    vid: int,
    *,
    horizon: float,
) -> Tuple[List[float], List[float]]:
    """
    Return the SAME legal committed event population used by M2/M3/M4/M5:
      aircraft_etas: currently committed aircraft physically flying to vid
      passenger_etas: already-committed passengers enroute to vid
    Only events with eta <= horizon are included.
    """
    h = float(horizon)
    aircraft_etas: List[float] = []
    passenger_etas: List[float] = []

    for e in list(getattr(scenario, "_all_evtols", {}).values()):
        if mx.state_name(e) != "FLYING":
            continue
        if mx.target_vid(e) != int(vid):
            continue
        eta = fnum(
            getattr(e, "remaining_time", getattr(e, "remaining_flight_time", 0.0)),
            0.0,
        )
        eta = max(0.0, eta)
        if eta <= h + 1e-9:
            aircraft_etas.append(float(eta))

    persons_obj = getattr(scenario, "persons", None)
    persons = getattr(persons_obj, "persons", {}) if persons_obj is not None else {}
    for p in persons.values():
        if str(getattr(p, "state", "")).lower() != "enroute":
            continue
        if str(getattr(p, "sub_state", "")).lower() != "to_vertiport":
            continue
        try:
            pvid = int(getattr(p, "origin_vertiport_id"))
        except Exception:
            continue
        if pvid != int(vid):
            continue
        eta = max(0.0, fnum(getattr(p, "current_timer", 0.0), 0.0))
        if eta <= h + 1e-9:
            passenger_etas.append(float(eta))

    aircraft_etas.sort()
    passenger_etas.sort()
    return aircraft_etas, passenger_etas


def _pack_eta_set(
    etas: Sequence[float],
    *,
    max_events: int,
    horizon: float,
    center: float = 0.0,
) -> Tuple[List[float], List[float]]:
    xs = list(etas)[: int(max_events)]
    values = [
        float((float(x) - float(center)) / max(float(horizon), 1e-6))
        for x in xs
    ]
    mask = [1.0] * len(values)
    while len(values) < int(max_events):
        values.append(0.0)
        mask.append(0.0)
    return values, mask


def _current_aligned_vector(
    scenario: Any,
    *,
    stage: str,
    candidates: Sequence[int],
    charger_capacity: int,
) -> List[float]:
    """
    CURRENT variables only.  Inbound-future count is deliberately excluded;
    it first appears in M2.
    """
    stage = str(stage).upper()
    out: List[float] = []

    for vid in candidates:
        r = mx._resource_features_for_vid(
            scenario,
            int(vid),
            e6=(stage == "E6"),
            charger_capacity=int(charger_capacity),
        )
        ready_idle = float(r[0])
        turnaround_busy = float(r[2])
        min_turn_eta = float(r[3])
        pad_next_free = float(r[4])
        charging = float(r[5])
        active_charging = float(r[6])
        min_charge_eta = float(r[7])
        charger_wait = max(0.0, charging - active_charging)

        if stage_ge(stage, "E2"):
            out.append(ready_idle)
        if stage_ge(stage, "E3"):
            # E3 onward has one passenger per service flight, so the effective
            # current service capacity is one per ready aircraft.
            out.append(ready_idle)
        if stage_ge(stage, "E4"):
            out.extend([turnaround_busy, min_turn_eta])
        if stage_ge(stage, "E5"):
            out.append(pad_next_free)
        if stage_ge(stage, "E6"):
            out.extend([active_charging, charger_wait, min_charge_eta])

    # Destination/hub resources affect the closed-loop aircraft cycle.
    hub = mx._resource_features_for_vid(
        scenario,
        DESTINATION,
        e6=(stage == "E6"),
        charger_capacity=int(charger_capacity),
    )
    hub_ready = float(hub[0])
    hub_turn = float(hub[2])
    hub_turn_eta = float(hub[3])
    hub_pad = float(hub[4])
    hub_charging = float(hub[5])
    hub_active = float(hub[6])
    hub_charge_eta = float(hub[7])
    hub_wait = max(0.0, hub_charging - hub_active)

    if stage_ge(stage, "E2"):
        out.append(hub_ready)
    if stage_ge(stage, "E4"):
        out.extend([hub_turn, hub_turn_eta])
    if stage_ge(stage, "E5"):
        out.append(hub_pad)
    if stage_ge(stage, "E6"):
        out.extend([hub_active, hub_wait, hub_charge_eta])

    return out


def _project_candidate(
    scenario: Any,
    *,
    stage: str,
    vid: int,
    horizon: float,
    charger_capacity: int,
) -> List[float]:
    """
    Candidate-specific state at focal passenger effect/access time.
    The focal passenger itself is NOT injected here; M5 adds that self-effect.
    """
    h = max(0.0, float(horizon))
    vp = scenario.vertiports.vertiport_list[str(int(vid))]
    waiting_now = len(list(getattr(vp, "person_list", []) or []))
    arrived, still_incoming = core._committed_access_counts(scenario, int(vid), h)

    (
        charging_future,
        total_evtols_future,
        total_capacity_future,
        avg_charge_future,
        min_charge_future,
        avg_inbound_future,
    ) = core._project_aircraft_features(
        scenario,
        int(vid),
        h,
        str(stage).upper(),
        int(charger_capacity),
    )

    pad_now = mx._pad_next_free_eta(scenario, int(vid))
    pad_at_effect = max(0.0, float(pad_now) - h)
    projected_waiting = float(waiting_now + arrived)
    serviceable_future = max(
        0.0,
        float(total_evtols_future) - float(charging_future),
    )

    return [
        h,
        projected_waiting,
        float(still_incoming),
        float(charging_future),
        float(total_evtols_future),
        float(serviceable_future),
        float(avg_charge_future),
        float(min_charge_future),
        float(avg_inbound_future),
        float(pad_at_effect),
    ]


def _counterfactual_features(project: Sequence[float]) -> List[float]:
    waiting = float(project[1])
    supply = float(project[5])
    pressure_before = waiting / (1.0 + supply)
    pressure_after = (waiting + 1.0) / (1.0 + supply)
    return [
        1.0,  # deterministic focal self-demand
        pressure_after,
        pressure_after - pressure_before,
    ]


# =============================================================================
# Observation wrapper
# =============================================================================

class MethodObservationWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
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
        self.candidates = candidates_for(self.topology)
        self.future_horizon = float(future_horizon)
        self.max_events = int(max_events)
        self.charger_capacity = int(charger_capacity)
        self.layout = make_layout(
            stage=self.stage,
            method=self.method,
            topology=self.topology,
            max_events=self.max_events,
        )

        got = int(np.prod(env.observation_space.shape))
        expected = int(self.layout["base_dim"])
        if got != expected:
            raise RuntimeError(
                f"base observation mismatch for {self.stage}/{self.method}/{self.topology}: "
                f"got={got}, expected={expected}"
            )

        self.observation_space = spaces.Box(
            low=-1e6,
            high=1e6,
            shape=(int(self.layout["total_dim"]),),
            dtype=np.float32,
        )

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        scenario = mx.find_scenario(self.env)
        person = _focal_person(self.env, scenario)
        extras: List[float] = []

        # M0 is already environment-aligned through this current-state vector.
        extras.extend(
            _current_aligned_vector(
                scenario,
                stage=self.stage,
                candidates=self.candidates,
                charger_capacity=self.charger_capacity,
            )
        )

        if self.method in ("M1", "M2", "M3", "M4", "M5"):
            extras.extend(_focal_od(person))

        if self.method == "M2":
            for vid in self.candidates:
                a, p = _event_lists(
                    scenario,
                    int(vid),
                    horizon=self.future_horizon,
                )
                extras.extend([float(len(a)), float(len(p))])

        if self.method in ("M3", "M4", "M5"):
            for vid in self.candidates:
                a, p = _event_lists(
                    scenario,
                    int(vid),
                    horizon=self.future_horizon,
                )
                center = (
                    _access_time(scenario, person, int(vid))
                    if self.method in ("M4", "M5")
                    else 0.0
                )
                a_eta, a_mask = _pack_eta_set(
                    a,
                    max_events=self.max_events,
                    horizon=self.future_horizon,
                    center=center,
                )
                p_eta, p_mask = _pack_eta_set(
                    p,
                    max_events=self.max_events,
                    horizon=self.future_horizon,
                    center=center,
                )
                extras.extend(a_eta)
                extras.extend(a_mask)
                extras.extend(p_eta)
                extras.extend(p_mask)

        projects: List[List[float]] = []
        if self.method in ("M4", "M5"):
            for vid in self.candidates:
                access = _access_time(scenario, person, int(vid))
                proj = _project_candidate(
                    scenario,
                    stage=self.stage,
                    vid=int(vid),
                    horizon=access,
                    charger_capacity=self.charger_capacity,
                )
                projects.append(proj)
                extras.extend(proj)

        if self.method == "M5":
            for proj in projects:
                extras.extend(_counterfactual_features(proj))

        base_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        out = np.concatenate(
            [base_obs, np.asarray(extras, dtype=np.float32)],
            axis=0,
        )

        expected = int(self.layout["total_dim"])
        if out.shape != (expected,):
            raise RuntimeError(
                f"observation shape mismatch {out.shape}, expected {(expected,)} "
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
        if not isinstance(out, tuple):
            raise RuntimeError(type(out))
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return self._transform(obs), reward, terminated, truncated, info
        if len(out) == 4:
            obs, reward, done, info = out
            return self._transform(obs), reward, done, info
        raise RuntimeError(f"unexpected step tuple length={len(out)}")


# =============================================================================
# Unified temporal + residual + DeepSets feature extractor
# =============================================================================

class UnifiedUAMExtractor(BaseFeaturesExtractor):
    """
    Same backbone across M0-M5:
      source 6-frame observation -> frame MLP -> LSTM
    Optional branches are activated only when their corresponding information
    exists.  M1 gives focal OD its own residual path.  M3-M5 use a true
    permutation-invariant DeepSets branch for the individual event ETAs.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        layout: Optional[Dict[str, Any]] = None,
    ):
        if layout is None:
            raise ValueError("layout is required")
        super().__init__(observation_space, features_dim)
        self.layout = dict(layout)
        self.features_dim_out = int(features_dim)

        single_dim = int(layout["single_dim"])
        self.num_frames = int(layout["num_frames"])
        self.n_candidates = int(layout["n_candidates"])
        self.max_events = int(layout["max_events"])
        self.slices = dict(layout["slices"])
        self.dims = dict(layout["dims"])

        self.frame_encoder = nn.Sequential(
            nn.Linear(single_dim, 128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        self.passenger_branch: Optional[nn.Module]
        if int(self.dims["passenger"]) > 0:
            self.passenger_branch = nn.Sequential(
                nn.Linear(int(self.dims["passenger"]), 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            )
        else:
            self.passenger_branch = None

        numeric_dim = sum(
            int(self.dims[k])
            for k in ("current", "coarse", "project", "cf")
        )
        self.numeric_names = [
            k for k in ("current", "coarse", "project", "cf")
            if int(self.dims[k]) > 0
        ]
        if numeric_dim > 0:
            self.numeric_branch: Optional[nn.Module] = nn.Sequential(
                nn.Linear(numeric_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
        else:
            self.numeric_branch = None

        if int(self.dims["fine"]) > 0:
            # DeepSets: rho(sum_i phi([relative_eta, event_type])).
            self.event_phi: Optional[nn.Module] = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            )
            self.event_rho: Optional[nn.Module] = nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(),
            )
            set_out = 32 * self.n_candidates
        else:
            self.event_phi = None
            self.event_rho = None
            set_out = 0

        concat_dim = 128
        if self.passenger_branch is not None:
            concat_dim += 32
        if self.numeric_branch is not None:
            concat_dim += 64
        concat_dim += set_out

        self.output_proj = nn.Sequential(
            nn.Linear(concat_dim, int(features_dim)),
            nn.ReLU(),
        )

    def _slice(self, obs: torch.Tensor, name: str) -> torch.Tensor:
        lo, hi = self.slices[name]
        return obs[:, int(lo):int(hi)]

    def _encode_sets(self, obs: torch.Tensor) -> torch.Tensor:
        fine = self._slice(obs, "fine")
        B = fine.shape[0]
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
                [
                    torch.zeros_like(a_eta),
                    torch.ones_like(p_eta),
                ],
                dim=1,
            )
            elems = torch.stack([eta, event_type], dim=-1)
            emb = self.event_phi(elems)
            emb = emb * mask.unsqueeze(-1)
            pooled = emb.sum(dim=1)
            reps.append(self.event_rho(pooled))

        return torch.cat(reps, dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations.float()

        base_obs = self._slice(obs, "base")
        B = base_obs.shape[0]
        base_obs = base_obs.reshape(
            B,
            self.num_frames,
            int(self.layout["single_dim"]),
        )
        x = self.frame_encoder(base_obs)
        x, _ = self.lstm(x)
        pieces = [x[:, -1, :]]

        if self.passenger_branch is not None:
            pieces.append(
                self.passenger_branch(
                    self._slice(obs, "passenger")
                )
            )

        if self.numeric_branch is not None:
            numeric = torch.cat(
                [self._slice(obs, name) for name in self.numeric_names],
                dim=1,
            )
            pieces.append(self.numeric_branch(numeric))

        if self.event_phi is not None:
            pieces.append(self._encode_sets(obs))

        return self.output_proj(torch.cat(pieces, dim=1))


# =============================================================================
# Environment/model construction
# =============================================================================

def make_method_env_factory(
    *,
    stage: str,
    method: str,
    topology: str,
    fleet_size: int,
    env_index: int,
    run_dir: Path,
    future_horizon: float,
    max_events: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
):
    def _init():
        env = core.make_experiment_env_factory(
            stage=stage,
            topology=topology,
            encoder_mode="uagmc",
            fleet_size=int(fleet_size),
            env_index=int(env_index),
            run_dir=run_dir,
            pad_separation=float(pad_separation),
            charger_capacity=int(charger_capacity),
            max_time=int(max_time),
        )()
        env = MethodObservationWrapper(
            env,
            stage=stage,
            method=method,
            topology=topology,
            future_horizon=future_horizon,
            max_events=max_events,
            charger_capacity=charger_capacity,
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
    factories = [
        make_method_env_factory(
            stage=stage,
            method=method,
            topology=topology,
            fleet_size=fleet_size,
            env_index=i,
            run_dir=run_dir,
            future_horizon=future_horizon,
            max_events=max_events,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
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
    return DummyVecEnv(
        [
            make_method_env_factory(
                stage=stage,
                method=method,
                topology=topology,
                fleet_size=fleet_size,
                env_index=9999,
                run_dir=run_dir,
                future_horizon=future_horizon,
                max_events=max_events,
                pad_separation=pad_separation,
                charger_capacity=charger_capacity,
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
    layout = make_layout(
        stage=stage,
        method=method,
        topology=topology,
        max_events=max_events,
    )
    policy_kwargs = dict(
        features_extractor_class=UnifiedUAMExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            layout=layout,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    return PPO(
        policy="MlpPolicy",
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
# Rule baselines
# =============================================================================

def run_rule_baselines(
    *,
    root: Path,
    stages: Sequence[str],
    topology: str,
    seeds: Sequence[int],
    fleet_size: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> Dict[str, Dict[str, float]]:
    outdir = root / "rule_baselines"
    agg_path = outdir / "aggregate_results.csv"

    if agg_path.exists():
        agg_rows = read_csv(agg_path)
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        for stage in stages:
            for method in ("SPF", "STTF", "QTTI2"):
                for seed in seeds:
                    run_dir = outdir / f"{stage}__{method}__seed{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        f"[rule] {stage}/{method}/seed={seed}",
                        flush=True,
                    )
                    row = rulemod.run_one(
                        stage=stage,
                        topology=topology,
                        method=method,
                        seed=int(seed),
                        fleet_size=int(fleet_size),
                        pad_separation=float(pad_separation),
                        charger_capacity=int(charger_capacity),
                        max_time=int(max_time),
                        run_dir=run_dir,
                    )
                    rows.append(row)
                    write_csv(outdir / "raw_results.csv", rows)
        agg_rows = rulemod.aggregate(rows)
        write_csv(agg_path, agg_rows)

    refs: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in agg_rows:
        stage = str(row.get("stage", "")).upper()
        method = str(row.get("method", "")).upper()
        if stage and method:
            refs[stage][method] = fnum(
                row.get("system_person_minutes_per_passenger_mean")
            )

    summary_lines = [
        "RULE REFERENCES USED BY 6x6 MATRIX",
        "stage | SPF Jsys/N | STTF Jsys/N | QTTI2 Jsys/N | best",
    ]
    for stage in stages:
        vals = refs.get(stage, {})
        finite = {
            k: v for k, v in vals.items()
            if math.isfinite(v)
        }
        best_name = min(finite, key=finite.get) if finite else "NA"
        summary_lines.append(
            f"{stage:>2} | "
            f"{vals.get('SPF', float('nan')):9.3f} | "
            f"{vals.get('STTF', float('nan')):10.3f} | "
            f"{vals.get('QTTI2', float('nan')):11.3f} | {best_name}"
        )
    (outdir / "summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )
    core.restore_process_patches()
    return refs


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_checkpoint(
    *,
    stage: str,
    method: str,
    topology: str,
    fleet_size: int,
    model_path: Path,
    vec_path: Path,
    train_step: int,
    eval_seed: int,
    run_dir: Path,
    future_horizon: float,
    max_events: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> Dict[str, Any]:
    seed_all(eval_seed)
    raw = build_eval_raw_env(
        stage=stage,
        method=method,
        topology=topology,
        fleet_size=fleet_size,
        run_dir=run_dir / "_eval_monitor",
        future_horizon=future_horizon,
        max_events=max_events,
        pad_separation=pad_separation,
        charger_capacity=charger_capacity,
        max_time=max_time,
    )
    env = VecNormalize.load(str(vec_path), raw)
    env.training = False
    env.norm_reward = False

    try:
        model = PPO.load(str(model_path), env=env, device="cpu")
        model.policy.set_training_mode(False)
        try:
            env.seed(int(eval_seed))
        except Exception:
            pass

        obs = env.reset()
        done = np.asarray([False])
        action_counts: Counter = Counter()
        prob_rows: List[np.ndarray] = []
        reward_sum = 0.0
        episode_steps = 0
        system_person_minutes = 0.0
        terminal_snapshot = None

        while not bool(done[0]):
            scenario = mx.find_scenario(env)
            system_person_minutes += mx.active_system_count(scenario)

            probs = mx.policy_probs(model, obs)
            action, _ = model.predict(obs, deterministic=True)
            ai = int(np.asarray(action).reshape(-1)[0])
            action_counts[ai] += 1
            prob_rows.append(probs.copy())

            obs, reward, done, infos = env.step(action)
            reward_sum += fnum(np.asarray(reward).reshape(-1)[0], 0.0)
            episode_steps += 1

            if infos and isinstance(infos[0], dict):
                terminal_snapshot = infos[0].get(
                    "terminal_snapshot",
                    terminal_snapshot,
                )

            if episode_steps > int(max_time) + 100:
                raise RuntimeError("evaluation exceeded max-time guard")

        if terminal_snapshot is None:
            raise RuntimeError("terminal snapshot missing before VecEnv autoreset")

        metrics = mx.metrics_from_terminal_snapshot(
            snapshot=terminal_snapshot,
            action_counts=action_counts,
            prob_rows=prob_rows,
            system_person_minutes=system_person_minutes,
            reward_sum=reward_sum,
            episode_steps=episode_steps,
        )
        metrics.update(
            {
                "stage": stage,
                "method": method,
                "method_name": METHOD_NAMES[method],
                "topology": topology,
                "train_step": int(train_step),
                "eval_seed": int(eval_seed),
                "model_path": str(model_path),
                "vec_path": str(vec_path),
            }
        )
        return metrics
    finally:
        try:
            env.close()
        except Exception:
            pass
        core.restore_process_patches()
        gc.collect()


def analyze_cell(
    *,
    stage: str,
    method: str,
    topology: str,
    fleet_size: int,
    run_dir: Path,
    requested_steps: int,
    eval_seeds: Sequence[int],
    rule_refs: Dict[str, Dict[str, float]],
    future_horizon: float,
    max_events: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> Dict[str, Any]:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    steps = list(range(CHECKPOINT_INTERVAL, int(requested_steps) + 1, CHECKPOINT_INTERVAL))
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for step in steps:
        model = run_dir / "checkpoints" / f"uam_ppo_{step}_steps.zip"
        vec = run_dir / "checkpoints" / f"uam_ppo_vecnormalize_{step}_steps.pkl"
        if not model.exists() or not vec.exists():
            continue

        for eval_seed in eval_seeds:
            try:
                row = evaluate_checkpoint(
                    stage=stage,
                    method=method,
                    topology=topology,
                    fleet_size=fleet_size,
                    model_path=model,
                    vec_path=vec,
                    train_step=step,
                    eval_seed=int(eval_seed),
                    run_dir=run_dir,
                    future_horizon=future_horizon,
                    max_events=max_events,
                    pad_separation=pad_separation,
                    charger_capacity=charger_capacity,
                    max_time=max_time,
                )
                rows.append(row)
                print(
                    f"    [eval] {stage}/{method} {step//1000:>3}k "
                    f"seed={eval_seed}: ATT={row['ATT']:.3f} "
                    f"AWT={row['AWT']:.3f} "
                    f"finish={row['N_finished']}/{row['N']} "
                    f"Jsys/N={row['system_person_minutes_per_passenger']:.3f}",
                    flush=True,
                )
            except Exception as exc:
                errors.append(
                    {
                        "train_step": step,
                        "eval_seed": eval_seed,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

    write_csv(analysis_dir / "checkpoint_eval_raw.csv", rows)
    write_csv(analysis_dir / "errors.csv", errors)
    if not rows:
        raise RuntimeError(f"no checkpoint evaluation succeeded for {stage}/{method}")

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["train_step"])].append(row)

    refs = dict(rule_refs.get(stage, {}))
    finite_refs = {k: v for k, v in refs.items() if math.isfinite(v)}
    best_rule_name = min(finite_refs, key=finite_refs.get) if finite_refs else ""
    best_rule = finite_refs.get(best_rule_name, float("nan"))

    curve: List[Dict[str, Any]] = []
    metric_keys = (
        "ATT",
        "AWT",
        "AGT_access",
        "AFT",
        "completion_rate",
        "backlog",
        "system_person_minutes_per_passenger",
        "travel_p90",
        "policy_entropy_normalized",
        "mean_policy_max_prob",
    )

    for step in sorted(grouped):
        group = grouped[step]
        agg: Dict[str, Any] = {
            "stage": stage,
            "method": method,
            "method_name": METHOD_NAMES[method],
            "topology": topology,
            "train_step": step,
            "n_eval_seeds": len(group),
        }
        for key in metric_keys:
            agg[f"{key}_mean"] = fmean(r.get(key) for r in group)
            agg[f"{key}_std"] = fstd(r.get(key) for r in group)

        action_keys = sorted(
            {
                k
                for r in group
                for k in r.keys()
                if k.startswith("action_") and k.endswith("_share")
            }
        )
        for key in action_keys:
            agg[f"{key}_mean"] = fmean(r.get(key) for r in group)

        jsys = fnum(agg.get("system_person_minutes_per_passenger_mean"))
        for name in ("SPF", "STTF", "QTTI2"):
            ref = refs.get(name, float("nan"))
            agg[f"{name}_Jsys_per_passenger"] = ref
            agg[f"gap_vs_{name}_pct"] = (
                100.0 * (jsys - ref) / ref
                if math.isfinite(jsys) and math.isfinite(ref) and abs(ref) > 1e-12
                else float("nan")
            )
        agg["best_rule_name"] = best_rule_name
        agg["best_rule_Jsys_per_passenger"] = best_rule
        agg["gap_vs_best_rule_pct"] = (
            100.0 * (jsys - best_rule) / best_rule
            if math.isfinite(jsys) and math.isfinite(best_rule) and abs(best_rule) > 1e-12
            else float("nan")
        )
        curve.append(agg)

    write_csv(analysis_dir / "checkpoint_curve.csv", curve)

    final = next(
        (r for r in curve if int(r["train_step"]) == int(requested_steps)),
        curve[-1],
    )
    # Best is chosen on censor-safe Jsys/N, not completed-passenger ATT.
    best = min(
        curve,
        key=lambda r: fnum(
            r.get("system_person_minutes_per_passenger_mean"),
            float("inf"),
        ),
    )
    late_lo = max(CHECKPOINT_INTERVAL, int(requested_steps) - 200_000)
    late = [r for r in curve if int(r["train_step"]) >= late_lo]

    summary = {
        "stage": stage,
        "method": method,
        "method_name": METHOD_NAMES[method],
        "topology": topology,
        "formal_step": int(requested_steps),
        "final": final,
        "best_checkpoint": best,
        "late_window_lo": late_lo,
        "late_steps": [int(r["train_step"]) for r in late],
        "late_Jsys_mean": fmean(
            r.get("system_person_minutes_per_passenger_mean") for r in late
        ),
        "late_Jsys_std": fstd(
            r.get("system_person_minutes_per_passenger_mean") for r in late
        ),
        "late_ATT_mean": fmean(r.get("ATT_mean") for r in late),
        "late_ATT_std": fstd(r.get("ATT_mean") for r in late),
        "late_completion_mean": fmean(r.get("completion_rate_mean") for r in late),
        "rule_refs": refs,
        "best_rule_name": best_rule_name,
        "best_rule_Jsys_per_passenger": best_rule,
    }
    write_json(analysis_dir / "summary.json", summary)

    lines = [
        f"{stage}/{method} {METHOD_NAMES[method]}",
        "=" * 100,
        f"final step          : {int(final['train_step']):,}",
        f"final ATT           : {fnum(final.get('ATT_mean')):.4f}",
        f"final completion    : {100*fnum(final.get('completion_rate_mean')):.2f}%",
        f"final Jsys/N        : {fnum(final.get('system_person_minutes_per_passenger_mean')):.4f}",
        f"best checkpoint     : {int(best['train_step']):,}",
        f"best Jsys/N         : {fnum(best.get('system_person_minutes_per_passenger_mean')):.4f}",
        f"best rule           : {best_rule_name} = {best_rule:.4f}",
        f"final gap best rule : {fnum(final.get('gap_vs_best_rule_pct')):.2f}%",
        f"late Jsys mean±std  : {summary['late_Jsys_mean']:.4f} ± {summary['late_Jsys_std']:.4f}",
    ]
    (analysis_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return summary


# =============================================================================
# One training cell
# =============================================================================

def cell_id(stage: str, method: str) -> str:
    return f"{stage}__{method}"


def run_cell(
    *,
    stage: str,
    method: str,
    topology: str,
    requested_steps: int,
    seed: int,
    device: str,
    root: Path,
    eval_seeds: Sequence[int],
    rule_refs: Dict[str, Dict[str, float]],
    fleet_size: int,
    future_horizon: float,
    max_events: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> Dict[str, Any]:
    cid = cell_id(stage, method)
    run_dir = root / cid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    done_path = run_dir / "cell_done.json"
    summary_path = run_dir / "analysis" / "summary.json"
    if done_path.exists() and summary_path.exists():
        print(f"[SKIP completed] {cid}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    layout = make_layout(
        stage=stage,
        method=method,
        topology=topology,
        max_events=max_events,
    )
    manifest = {
        "experiment": "UAGMC_CONTROLLED_6x6_800K",
        "cell_id": cid,
        "stage": stage,
        "method": method,
        "method_name": METHOD_NAMES[method],
        "topology": topology,
        "train_seed": int(seed),
        "requested_timesteps": int(requested_steps),
        "formal_checkpoint": int(requested_steps),
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "fleet_size_E2_E6": int(fleet_size),
        "E0_fleet_mode": "legacy_replenish",
        "future_horizon_min": float(future_horizon),
        "max_events_per_type_per_candidate": int(max_events),
        "pad_separation_min": float(pad_separation),
        "charger_capacity_E6": int(charger_capacity),
        "passenger_trace": str(TRAIN_FILE),
        "layout": layout,
        "compute": {
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "global_rollout": GLOBAL_ROLLOUT,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "device": device,
        },
        "control": {
            "reward": "unchanged UAGMC",
            "PPO": "same across all cells",
            "history_frames": NUM_FRAMES,
            "future_information_first_introduced_at": "M2",
            "M2_M3_same_event_population": True,
            "M4_candidate_specific_effect_time": True,
            "unrevealed_future_requests_used": False,
        },
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(run_dir / "run_manifest.json", manifest)

    env = None
    model = None
    started = time.time()
    try:
        env = build_train_env(
            stage=stage,
            method=method,
            topology=topology,
            fleet_size=fleet_size,
            seed=seed,
            run_dir=run_dir,
            future_horizon=future_horizon,
            max_events=max_events,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
            max_time=max_time,
        )

        model = build_model(
            env=env,
            stage=stage,
            method=method,
            topology=topology,
            max_events=max_events,
            seed=seed,
            run_dir=run_dir,
            device=device,
        )

        if CHECKPOINT_INTERVAL % N_ENVS != 0:
            raise ValueError("checkpoint interval must be divisible by n_envs")

        checkpoint_cb = CheckpointCallback(
            save_freq=CHECKPOINT_INTERVAL // N_ENVS,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="uam_ppo",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=0,
        )
        scalar_cb = base.PassiveTrainingScalarCallback(
            run_dir=run_dir,
            interval=CHECKPOINT_INTERVAL,
            requested_steps=int(requested_steps),
        )

        print("\n" + "=" * 132)
        print(
            f"START {cid} | {METHOD_NAMES[method]} | {topology} | "
            f"{requested_steps:,} steps"
        )
        print(
            f"16x1280={GLOBAL_ROLLOUT:,}/rollout | batch={BATCH_SIZE} | "
            f"device={model.device}"
        )
        print("TRAINING ONLY: checkpoints are passive; evaluation starts after this cell.")
        print("=" * 132, flush=True)

        model.learn(
            total_timesteps=int(requested_steps),
            callback=[checkpoint_cb, scalar_cb],
            progress_bar=False,
            reset_num_timesteps=True,
        )

        elapsed = time.time() - started
        formal_model = run_dir / "checkpoints" / f"uam_ppo_{requested_steps}_steps.zip"
        formal_vec = run_dir / "checkpoints" / f"uam_ppo_vecnormalize_{requested_steps}_steps.pkl"
        if not formal_model.exists() or not formal_vec.exists():
            raise FileNotFoundError(
                f"formal checkpoint pair missing: {formal_model}, {formal_vec}"
            )

        train_end = {
            "status": "TRAINED",
            "cell_id": cid,
            "actual_model_num_timesteps": int(model.num_timesteps),
            "requested_timesteps": int(requested_steps),
            "elapsed_seconds": elapsed,
            "fps": float(model.num_timesteps) / max(elapsed, 1e-9),
            "formal_model": str(formal_model),
            "formal_vecnormalize": str(formal_vec),
            "cuda_peak_memory_mb": (
                torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                if torch.cuda.is_available()
                else 0.0
            ),
            "finished": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(run_dir / "train_end.json", train_end)

        try:
            env.close()
        except Exception:
            pass
        env = None
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(
            f"DONE TRAIN {cid} | {elapsed/60:.2f} min | "
            f"IMMEDIATE ATT/CHECKPOINT ANALYSIS",
            flush=True,
        )

        summary = analyze_cell(
            stage=stage,
            method=method,
            topology=topology,
            fleet_size=fleet_size,
            run_dir=run_dir,
            requested_steps=requested_steps,
            eval_seeds=eval_seeds,
            rule_refs=rule_refs,
            future_horizon=future_horizon,
            max_events=max_events,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
            max_time=max_time,
        )
        write_json(
            done_path,
            {
                "status": "DONE",
                "finished": datetime.now().isoformat(timespec="seconds"),
                "summary": summary,
            },
        )
        return summary

    except Exception as exc:
        write_json(
            run_dir / "cell_error.json",
            {
                "cell_id": cid,
                "status": "ERROR",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "finished": datetime.now().isoformat(timespec="seconds"),
            },
        )
        raise
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        model = None
        core.restore_process_patches()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# =============================================================================
# Matrix summary
# =============================================================================

def summary_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    final = summary.get("final", {}) or {}
    best = summary.get("best_checkpoint", {}) or {}
    return {
        "stage": summary.get("stage"),
        "method": summary.get("method"),
        "method_name": summary.get("method_name"),
        "topology": summary.get("topology"),
        "final_step": final.get("train_step"),
        "final_ATT": final.get("ATT_mean"),
        "final_AWT": final.get("AWT_mean"),
        "final_completion": final.get("completion_rate_mean"),
        "final_backlog": final.get("backlog_mean"),
        "final_Jsys_per_passenger": final.get("system_person_minutes_per_passenger_mean"),
        "final_gap_vs_STTF_pct": final.get("gap_vs_STTF_pct"),
        "final_gap_vs_QTTI2_pct": final.get("gap_vs_QTTI2_pct"),
        "final_gap_vs_best_rule_pct": final.get("gap_vs_best_rule_pct"),
        "best_step": best.get("train_step"),
        "best_ATT": best.get("ATT_mean"),
        "best_completion": best.get("completion_rate_mean"),
        "best_Jsys_per_passenger": best.get("system_person_minutes_per_passenger_mean"),
        "best_gap_vs_best_rule_pct": best.get("gap_vs_best_rule_pct"),
        "late_Jsys_mean": summary.get("late_Jsys_mean"),
        "late_Jsys_std": summary.get("late_Jsys_std"),
        "late_ATT_mean": summary.get("late_ATT_mean"),
        "late_ATT_std": summary.get("late_ATT_std"),
        "late_completion_mean": summary.get("late_completion_mean"),
        "best_rule_name": summary.get("best_rule_name"),
        "best_rule_Jsys_per_passenger": summary.get("best_rule_Jsys_per_passenger"),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Controlled E0/E2/E3/E4/E5/E6 x M0..M5 UAGMC matrix, 800k/cell"
    )
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--topology", choices=["T2", "T3"], default=DEFAULT_TOPOLOGY)
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p.add_argument("--seed", type=int, default=TRAIN_SEED)
    p.add_argument(
        "--eval-seeds",
        default=",".join(str(x) for x in DEFAULT_EVAL_SEEDS),
    )
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--fleet-size", type=int, default=FLEET_SIZE)
    p.add_argument("--max-time", type=int, default=MAX_TIME)
    p.add_argument("--future-horizon", type=float, default=FUTURE_HORIZON_MIN)
    p.add_argument("--max-events", type=int, default=MAX_EVENTS_PER_TYPE)
    p.add_argument("--pad-separation", type=float, default=PAD_SEPARATION_MIN)
    p.add_argument("--charger-capacity", type=int, default=CHARGER_CAPACITY)
    p.add_argument("--output-root", default=None)
    p.add_argument("--resume-root", default=None)
    p.add_argument("--skip-rule-baselines", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stages = parse_list(args.stages, STAGES)
    methods = parse_list(args.methods, METHODS)
    eval_seeds = parse_ints(args.eval_seeds)
    requested_steps = int(args.timesteps)

    if requested_steps <= 0 or requested_steps % CHECKPOINT_INTERVAL != 0:
        raise ValueError(
            f"--timesteps must be a positive multiple of {CHECKPOINT_INTERVAL}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

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
            / f"uagmc_6x6_{args.topology}_{requested_steps//1000}k_seed{args.seed}_{stamp}"
        ).resolve()

    root.mkdir(parents=True, exist_ok=True)

    matrix_cells = [(s, m) for s in stages for m in methods]
    manifest = {
        "experiment": "UAGMC_CONTROLLED_6x6_800K",
        "created": datetime.now().isoformat(timespec="seconds"),
        "stages": stages,
        "methods": {m: METHOD_NAMES[m] for m in methods},
        "topology": args.topology,
        "timesteps_per_cell": requested_steps,
        "n_cells": len(matrix_cells),
        "total_requested_timesteps": requested_steps * len(matrix_cells),
        "train_seed": int(args.seed),
        "eval_seeds": eval_seeds,
        "passenger_trace": str(TRAIN_FILE),
        "future_horizon_min": float(args.future_horizon),
        "max_events_per_type_per_candidate": int(args.max_events),
        "fleet_size_E2_E6": int(args.fleet_size),
        "pad_separation_min": float(args.pad_separation),
        "charger_capacity_E6": int(args.charger_capacity),
        "compute": {
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "global_rollout": GLOBAL_ROLLOUT,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "device": args.device,
        },
        "evaluation": (
            "After EACH cell finishes, replay every 50k checkpoint; "
            "save ATT/AWT/completion/backlog/Jsys/action stats and rule gaps."
        ),
    }
    write_json(root / "experiment_manifest.json", manifest)

    print("=" * 132)
    print("CONTROLLED UAGMC 6x6 MATRIX")
    print(
        f"Rows={stages} | Cols={methods} | topology={args.topology} | "
        f"{requested_steps:,}/cell | total={requested_steps*len(matrix_cells):,}"
    )
    print(
        f"Compute={N_ENVS}x{N_STEPS} rollout={GLOBAL_ROLLOUT:,}, "
        f"batch={BATCH_SIZE}, device={args.device}"
    )
    print(f"Output={root}")
    print("=" * 132)

    if args.skip_rule_baselines:
        rule_refs: Dict[str, Dict[str, float]] = defaultdict(dict)
    else:
        rule_refs = run_rule_baselines(
            root=root,
            stages=stages,
            topology=args.topology,
            seeds=eval_seeds,
            fleet_size=int(args.fleet_size),
            pad_separation=float(args.pad_separation),
            charger_capacity=int(args.charger_capacity),
            max_time=int(args.max_time),
        )

    results: List[Dict[str, Any]] = []
    # Recover already completed cells on resume.
    for s, m in matrix_cells:
        sp = root / cell_id(s, m) / "analysis" / "summary.json"
        if sp.exists():
            try:
                results.append(summary_row(json.loads(sp.read_text(encoding="utf-8"))))
            except Exception:
                pass
    write_csv(root / "matrix_results.csv", results)

    for idx, (stage, method) in enumerate(matrix_cells, 1):
        print(
            f"\n[{idx:02d}/{len(matrix_cells):02d}] {stage}/{method} "
            f"{METHOD_NAMES[method]}",
            flush=True,
        )
        try:
            summary = run_cell(
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

            # Replace this cell row if it was recovered earlier.
            new_row = summary_row(summary)
            results = [
                r for r in results
                if not (
                    str(r.get("stage")) == stage
                    and str(r.get("method")) == method
                )
            ]
            results.append(new_row)
            results.sort(
                key=lambda r: (
                    STAGES.index(str(r["stage"])),
                    METHODS.index(str(r["method"])),
                )
            )
            write_csv(root / "matrix_results.csv", results)

            final = summary.get("final", {}) or {}
            print(
                f"[SAVED] {stage}/{method}: "
                f"ATT={fnum(final.get('ATT_mean')):.3f}, "
                f"completion={100*fnum(final.get('completion_rate_mean')):.2f}%, "
                f"Jsys/N={fnum(final.get('system_person_minutes_per_passenger_mean')):.3f}, "
                f"gap-best-rule={fnum(final.get('gap_vs_best_rule_pct')):.2f}%",
                flush=True,
            )
        except Exception:
            if not args.continue_on_error:
                raise
            print(
                f"[ERROR but continue] {stage}/{method}; see cell_error.json",
                flush=True,
            )

    write_csv(root / "matrix_results.csv", results)
    print("\n" + "=" * 132)
    print("6x6 MATRIX FINISHED")
    print(f"Main table : {root / 'matrix_results.csv'}")
    print(f"Rule refs  : {root / 'rule_baselines' / 'aggregate_results.csv'}")
    print("Each cell  : <E>__<M>/analysis/checkpoint_curve.csv + summary.json")
    print("=" * 132)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
