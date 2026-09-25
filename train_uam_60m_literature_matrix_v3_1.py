# -*- coding: utf-8 -*-
r"""
UAM 57.6M LITERATURE-DISCOVERY MATRIX V3
========================================

This is the formal successor to train_uam_7x12_600k_v2.py.
It DOES NOT overwrite old source files. It imports the validated physical/P0
plumbing and defines a new experiment layer.

Budget (default 600k / cell)
----------------------------
Phase A  Passenger temporal representation discovery:
    6 new representations x 3 physical environments = 18 cells = 10.8M

Phase B  World-model discovery on the Top-3 Phase-A representations (S3 only):
    3 representations x 4 WM variants               = 12 cells =  7.2M
    2 counterfactual WM wildcard cells              =  2 cells =  1.2M

Phase C  Joint coordination discovery (S3 physics):
    8 joint architectures x 8 selected temporal pipelines = 64 cells = 38.4M

Formal automatic budget = 96 cells = 57.6M.
The remaining 2.4M under a 60M cap is intentionally NOT auto-consumed.

Key V3 execution changes
------------------------
1) Joint HOLD is REMOVED.
   Joint action = passenger {V0,V1} x aircraft target {V0,V1} = Discrete(4).
   Every joint decision chooses an aircraft return target. If no eligible empty
   hub aircraft exists at that instant, dispatch is recorded as infeasible;
   there is no explicit HOLD action for PPO to collapse into.

2) Simulator hard guard defaults to 2500 minutes (old V2 used 10000).

3) Single-side cells are evaluated immediately after each cell finishes.

4) Joint cells are NOT evaluated during training. The script trains the whole
   8x8 joint matrix first, then evaluates all successfully trained joint cells
   in one deferred evaluation pass.

5) Errors NEVER stop the default formal run. Training/evaluation errors are
   written to CSV/JSON and the runner moves to the next cell. Use --fail-fast
   only for debugging.

6) P0 completion semantics remain mandatory:
   natural completion only when all generated passengers finish;
   reward = -N_active * delta_t; incomplete eval episodes have ATT/AWT=NaN.

Literature-inspired representation cards
-----------------------------------------
R_SRAC   : Shared-reference Action Center
           shared P(tau_bar) + [P(tau_k)-P(tau_bar)]
R_GAMMA  : Gamma/Multi-horizon representation with 0/5/15/30/60-min basis
R_CQM    : Counterfactual-quotient style centered own-effect representation
R_EVENTQ : candidate-time query over committed aircraft/passenger event sets
R_TDM    : horizon-conditioned decision-relevant effect features
R_FIRST  : first-service/supply-ready representation

World-model cards
-----------------
W_TAU    : direct tau-conditioned future-resource residual model
W_VAML   : W_TAU + value-aware future-state loss using the current critic
W_PETS   : tau-conditioned bootstrap ensemble; policy receives mean+std
W_VE     : value-equivalent model; predicts critic value at effect time directly
W_COCO   : wildcard, W_TAU + counterfactual action-separation regularizer

IMPORTANT: W_COCO is explicitly a CoCo-inspired action-sensitivity regularizer,
not a verbatim reimplementation of the visual-world-model paper.

Joint architecture cards (all optimized by PPO; value-decomposition names mean
architecture transplants, not claims of exact original algorithms)
-------------------------------------------------------------------
J0 INDEPENDENT_PARALLEL    : passenger utility + aircraft utility
J1 ACTION_BRANCHING        : shared context -> passenger/aircraft branches
J2 DIRECT_PAIRWISE         : direct 4-way pair scorer
J3 SEQUENTIAL_DUAL_AGENT   : passenger first; aircraft conditioned on passenger
J4 PAIR_VALUE_SEMANTIC     : shared scorer over semantic (p,a) pair codes
J5 QMIX_STYLE              : state-conditioned positive monotonic utility mixer
J6 QTRAN_STYLE             : additive utilities + unconstrained joint residual
J7 EFFECT_PLANNING_PAIR    : semantic pair scorer with pair-effect virtual features

Run full formal matrix
----------------------
conda activate uam5070
cd /d "E:\Study Files\github\UAM-predict\UAGMC-main"
python train_uam_60m_literature_matrix_v3_1.py --device cuda --profile P3

Resume
------
python train_uam_60m_literature_matrix_v3_1.py --resume-root "serial_runs\<run>" --device cuda --profile P3

Deferred joint evaluation only
------------------------------
python train_uam_60m_literature_matrix_v3_1.py --resume-root "serial_runs\<run>" --joint-eval-only
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import time
import traceback
import types
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# Reuse validated V2/P0/physics plumbing.
import train_uam_7x12_600k_v2 as legacy

old = legacy.old
core = legacy.core
mx = legacy.mx
base = legacy.base
rulebase = legacy.rulebase


# =============================================================================
# Formal constants
# =============================================================================

ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "train_data" / "passengers_300.csv"

DEFAULT_TIMESTEPS = 600_000
CHECKPOINT_INTERVAL = 50_000
TRAIN_SEED = 1
DEFAULT_EVAL_SEEDS = (123, 124, 125)

TOPOLOGY = "T2"
CANDIDATES = (0, 1)
DESTINATION = 2
FLEET_SIZE = 40
DEMAND_HORIZON = 300
DEFAULT_HARD_GUARD = 2_500

TURNAROUND_MIN = 1.0
CHARGE_RATE_SCALE = 1.25
CHARGER_CAPACITY = 5
PAD_SEPARATION_MIN = 0.25

NUM_FRAMES = legacy.NUM_FRAMES
SINGLE_FRAME_DIM = legacy.SINGLE_FRAME_DIM
BASE_OBS_DIM = legacy.BASE_OBS_DIM
RESOURCE_DIM = legacy.RESOURCE_DIM
FOCAL_DIM = legacy.FOCAL_DIM

GAMMA = 1.0
GAE_LAMBDA = legacy.GAE_LAMBDA
CLIP_RANGE = legacy.CLIP_RANGE
ENT_COEF = legacy.ENT_COEF
VF_COEF = legacy.VF_COEF
MAX_GRAD_NORM = legacy.MAX_GRAD_NORM
INITIAL_LR = legacy.INITIAL_LR
N_EPOCHS = legacy.N_EPOCHS

# Literature representation dimensions.
N_CANDIDATES = len(CANDIDATES)
PROJECT_PER_CAND = 10
SHARED_DIM = N_CANDIDATES * PROJECT_PER_CAND           # 20
SRAC_DIM = N_CANDIDATES * PROJECT_PER_CAND             # 20
GAMMA_HORIZONS = (0.0, 5.0, 15.0, 30.0, 60.0)
GAMMA_DIM = N_CANDIDATES * len(GAMMA_HORIZONS) * PROJECT_PER_CAND  # 100
CQM_DIM = N_CANDIDATES * PROJECT_PER_CAND              # 20
MAX_EVENTS_PER_TYPE = 8
EVENT_HORIZON = 60.0
EVENT_DIM = N_CANDIDATES * 4 * MAX_EVENTS_PER_TYPE    # eta/mask x aircraft/passenger
TDM_FEATURES_PER_CAND = 5
TDM_DIM = N_CANDIDATES * TDM_FEATURES_PER_CAND        # 10
FIRST_FEATURES_PER_CAND = 5
FIRST_DIM = N_CANDIDATES * FIRST_FEATURES_PER_CAND    # 10
TAU_DIM = N_CANDIDATES                                 # 2 raw minutes
PAIR_FEATURES_PER_ACTION = 4
JOINT_PAIR_N = 4
PAIR_VIRTUAL_DIM = JOINT_PAIR_N * PAIR_FEATURES_PER_ACTION  # 16

# World model.
PETS_ENSEMBLE = 5
WM_AUX_LR = 1e-3
WM_AUX_EPOCHS = 1
WM_AUX_BATCH = 2048
WM_MAX_SAMPLES = 12_000
WM_HORIZON_CAP = 60
VAML_LAMBDA = 0.5
COCO_LAMBDA = 0.10
COCO_MARGIN = 0.05

PHASE_A_REPS = (
    "R_SRAC",
    "R_GAMMA",
    "R_CQM",
    "R_EVENTQ",
    "R_TDM",
    "R_FIRST",
)
BASE_REPS = ("CURRENT", "SHARED")
WM_VARIANTS = ("W_TAU", "W_VAML", "W_PETS", "W_VE")
WILDCARD_WM = "W_COCO"

JOINT_ARCHS = tuple(f"J{i}" for i in range(8))
JOINT_ARCH_NAMES = {
    "J0": "INDEPENDENT_PARALLEL",
    "J1": "ACTION_BRANCHING",
    "J2": "DIRECT_PAIRWISE",
    "J3": "SEQUENTIAL_DUAL_AGENT",
    "J4": "PAIR_VALUE_SEMANTIC",
    "J5": "QMIX_STYLE",
    "J6": "QTRAN_STYLE",
    "J7": "EFFECT_PLANNING_PAIR",
}


# =============================================================================
# Generic helpers
# =============================================================================

def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(np.asarray(x).reshape(-1)[0])
        return y if math.isfinite(y) else default
    except Exception:
        return default


def fmean(xs: Iterable[Any]) -> float:
    a = np.asarray([fnum(x) for x in xs], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def fstd(xs: Iterable[Any]) -> float:
    a = np.asarray([fnum(x) for x in xs], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=0)) if len(a) else float("nan")


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
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
                if isinstance(v, (dict, list, tuple, np.ndarray)):
                    v = json.dumps(v, ensure_ascii=False, default=str)
                cooked[k] = v
            w.writerow(cooked)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def safe_name(x: str) -> str:
    return re_sub_nonword(str(x).upper())


def re_sub_nonword(x: str) -> str:
    import re
    return re.sub(r"[^A-Z0-9_\-]+", "_", x)


def temporal_id(rep: str, wm: Optional[str] = None) -> str:
    rep = str(rep).upper()
    return rep if not wm else f"{rep}__{str(wm).upper()}"


def parse_temporal(method_id: str) -> Tuple[str, Optional[str]]:
    x = str(method_id).upper()
    if "__" in x:
        rep, wm = x.split("__", 1)
        return rep, wm
    return x, None


def method_description(method_id: str) -> Dict[str, Any]:
    rep, wm = parse_temporal(method_id)
    return {"method_id": method_id, "representation": rep, "world_model": wm}


# =============================================================================
# Environment definitions
# =============================================================================

@dataclass(frozen=True)
class EnvSpec:
    key: str
    physical_stage: str
    pad_separation: float
    charger_capacity: int
    joint: bool
    joint_arch: str
    sequential_commit: bool

    @property
    def n_actions(self) -> int:
        return 4 if self.joint else 2


ENV_SPECS: Dict[str, EnvSpec] = {
    "S1": EnvSpec("S1", "E4", 0.0, CHARGER_CAPACITY, False, "SINGLE", False),
    "S2": EnvSpec("S2", "E6", 0.0, CHARGER_CAPACITY, False, "SINGLE", False),
    "S3": EnvSpec("S3", "E6", PAD_SEPARATION_MIN, CHARGER_CAPACITY, False, "SINGLE", False),
}
for _j in JOINT_ARCHS:
    ENV_SPECS[_j] = EnvSpec(
        _j,
        "E6",
        PAD_SEPARATION_MIN,
        CHARGER_CAPACITY,
        True,
        JOINT_ARCH_NAMES[_j],
        _j == "J3",
    )


@dataclass(frozen=True)
class SpeedProfile:
    name: str
    n_envs: int
    n_steps: int
    batch_size: int

    @property
    def rollout(self) -> int:
        return self.n_envs * self.n_steps


PROFILES = {
    "P0": SpeedProfile("P0_5x4096_b2048", 5, 4096, 2048),
    "P1": SpeedProfile("P1_8x2560_b2048", 8, 2560, 2048),
    "P2": SpeedProfile("P2_10x2048_b2048", 10, 2048, 2048),
    "P3": SpeedProfile("P3_16x1280_b2048", 16, 1280, 2048),
}


def configure_worker_physics(max_time: int) -> None:
    # Reuse V2 absolute physics configuration but replace its hard guard.
    legacy.HARD_GUARD = int(max_time)
    legacy.configure_worker_physics()
    for mod in (legacy, base, core, mx, old):
        if hasattr(mod, "MAX_TIME"):
            setattr(mod, "MAX_TIME", int(max_time))


# =============================================================================
# Joint action: NO HOLD, Discrete(4) = 2 passenger x 2 aircraft targets
# =============================================================================

def _noop_reposition(self) -> None:
    return None


class NoHoldJointControlWrapper(gym.Wrapper):
    # pair = passenger_action * 2 + aircraft_target_index
    # passenger_action    : 0=V0, 1=V1
    # aircraft_target_idx : 0=V0, 1=V1
    # There is no HOLD action.

    def __init__(self, env: gym.Env, env_key: str):
        super().__init__(env)
        self.env_key = str(env_key).upper()
        self.env_spec = ENV_SPECS[self.env_key]
        if not self.env_spec.joint:
            raise ValueError(self.env_key)
        self.action_space = spaces.Discrete(4)
        self._disable_auto_reposition()

    def _scenario(self):
        return mx.find_scenario(self.env)

    def _uam(self):
        return mx.find_uam_wrapper(self.env)

    def _disable_auto_reposition(self) -> None:
        sc = self._scenario()
        sc._dispatch_fixed_returns = types.MethodType(_noop_reposition, sc)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._disable_auto_reposition()
        return out

    def _focal_pid(self) -> Optional[str]:
        uam = self._uam()
        waiting = list((getattr(uam, "state", {}) or {}).get("waiting_decisions", []) or [])
        return str(waiting[0]) if waiting else None

    def _commit_passenger_without_time_advance(self, passenger_action: int) -> bool:
        uam = self._uam()
        sc = self._scenario()
        pid = self._focal_pid()
        if pid is None:
            return False
        decision = uam.decoder.decode(int(passenger_action), pid)
        if pid not in sc.waiting_decisions:
            return False
        sc.apply_decision(pid, decision)
        return True

    def _dispatch_aircraft(self, aircraft_target_idx: int) -> Tuple[bool, int, int]:
        ai = int(aircraft_target_idx)
        if ai not in (0, 1):
            raise ValueError(ai)
        target = int(CANDIDATES[ai])
        sc = self._scenario()
        hub = str(DESTINATION)

        local = list(sc.vertiports.evtols_at_vertiport.get(hub, []) or [])
        available = [
            e for e in local
            if (
                mx.state_name(e) == "IDLE"
                and not mx.passenger_ids(e)
                and not base.is_turnaround_busy(e)
            )
        ]
        available.sort(key=lambda e: str(getattr(e, "id", "")))
        eligible_count = len(available)
        if not available:
            return False, target, eligible_count

        ok = bool(
            base._start_empty_reposition_checked(
                scenario=sc,
                evtol=available[0],
                origin=hub,
                destination=str(target),
            )
        )
        return ok, target, eligible_count

    def step(self, action):
        idx = int(np.asarray(action).reshape(-1)[0])
        if idx < 0 or idx >= 4:
            raise ValueError(idx)
        passenger_action = idx // 2
        aircraft_action = idx % 2

        committed = False
        if self.env_spec.sequential_commit:
            committed = self._commit_passenger_without_time_advance(passenger_action)

        dispatched, target, eligible_count = self._dispatch_aircraft(aircraft_action)
        out = self.env.step(passenger_action)

        info_extra = {
            "joint_action": idx,
            "passenger_action": int(passenger_action),
            "aircraft_action": int(aircraft_action),
            "aircraft_target": int(target),
            "aircraft_dispatched": bool(dispatched),
            "aircraft_eligible_hub_count": int(eligible_count),
            "passenger_precommitted": bool(committed),
            "joint_hold_removed": True,
        }

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info)
            info.update(info_extra)
            return obs, reward, terminated, truncated, info
        obs, reward, done, info = out
        info = dict(info)
        info.update(info_extra)
        return obs, reward, done, info


# =============================================================================
# Temporal feature construction
# =============================================================================

def _focal_person(env: Any, scenario: Any) -> Optional[Any]:
    return old._focal_person(env, scenario)


def _access_time(scenario: Any, person: Optional[Any], vid: int) -> float:
    if person is None:
        return 0.0
    try:
        return max(0.0, float(old._access_time(scenario, person, int(vid))))
    except Exception:
        return 0.0


def _focal_od(person: Optional[Any]) -> List[float]:
    return list(legacy._focal_od(person))


def _resource_vector(scenario: Any, spec: EnvSpec) -> List[float]:
    # Same validated current-resource superset as V2.
    proxy = legacy.EnvSpec(
        spec.key if spec.key in legacy.ENV_SPECS else "S3",
        spec.physical_stage,
        spec.pad_separation,
        spec.charger_capacity,
        spec.joint,
        "SINGLE" if not spec.joint else "PAIRWISE_PARALLEL",
        spec.sequential_commit,
    )
    return list(legacy._resource_vector(scenario, proxy))


def _project(scenario: Any, spec: EnvSpec, vid: int, horizon: float) -> List[float]:
    try:
        return [
            float(x) for x in old._project_candidate(
                scenario,
                stage=spec.physical_stage,
                vid=int(vid),
                horizon=max(0.0, float(horizon)),
                charger_capacity=int(spec.charger_capacity),
            )
        ]
    except Exception:
        # Conservative no-hidden-demand fallback.
        vp = scenario.vertiports.vertiport_list[str(int(vid))]
        waiting_now = float(len(list(getattr(vp, "person_list", []) or [])))
        try:
            arrived, still = core._committed_access_counts(
                scenario, int(vid), max(0.0, float(horizon))
            )
        except Exception:
            arrived, still = 0, 0
        return [
            max(0.0, float(horizon)),
            waiting_now + float(arrived),
            float(still),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]


def _shared_srac_gamma_cqm(
    scenario: Any,
    spec: EnvSpec,
    person: Optional[Any],
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    tau = [_access_time(scenario, person, v) for v in CANDIDATES]
    shared_h = float(np.mean(tau)) if tau else 0.0

    shared: List[float] = []
    srac: List[float] = []
    gamma: List[float] = []
    own_all: List[List[float]] = []

    for i, vid in enumerate(CANDIDATES):
        ps = _project(scenario, spec, vid, shared_h)
        po = _project(scenario, spec, vid, tau[i])
        shared.extend(ps)
        srac.extend([float(b - a) for a, b in zip(ps, po)])
        own_all.append(po)
        for h in GAMMA_HORIZONS:
            gamma.extend(_project(scenario, spec, vid, h))

    mean_own = np.mean(np.asarray(own_all, dtype=float), axis=0)
    cqm: List[float] = []
    for po in own_all:
        cqm.extend((np.asarray(po, dtype=float) - mean_own).tolist())

    return tau, shared, srac, gamma, cqm


def _pack_events(
    scenario: Any,
    tau: Sequence[float],
) -> List[float]:
    out: List[float] = []
    for i, vid in enumerate(CANDIDATES):
        try:
            aircraft, passengers = old._event_lists(
                scenario, int(vid), horizon=float(EVENT_HORIZON)
            )
        except Exception:
            aircraft, passengers = [], []

        def pack(xs: Sequence[float]) -> Tuple[List[float], List[float]]:
            vals = [float(x) / EVENT_HORIZON for x in list(xs)[:MAX_EVENTS_PER_TYPE]]
            mask = [1.0] * len(vals)
            while len(vals) < MAX_EVENTS_PER_TYPE:
                vals.append(0.0)
                mask.append(0.0)
            return vals, mask

        a_eta, a_mask = pack(aircraft)
        p_eta, p_mask = pack(passengers)
        out.extend(a_eta)
        out.extend(a_mask)
        out.extend(p_eta)
        out.extend(p_mask)
    if len(out) != EVENT_DIM:
        raise RuntimeError(f"event dim {len(out)} != {EVENT_DIM}")
    return out


def _rule_stage(spec: EnvSpec) -> str:
    return "E4" if spec.physical_stage == "E4" else "E6"


def _first_supply_ready(scenario: Any, spec: EnvSpec, vid: int) -> float:
    try:
        releases = rulebase.known_supply_release_etas(
            scenario,
            int(vid),
            _rule_stage(spec),
            int(spec.charger_capacity),
        )
    except Exception:
        releases = []
    return float(min(releases)) if releases else EVENT_HORIZON


def _tdm_and_first(
    scenario: Any,
    spec: EnvSpec,
    tau: Sequence[float],
) -> Tuple[List[float], List[float]]:
    tdm: List[float] = []
    first: List[float] = []
    for i, vid in enumerate(CANDIDATES):
        own = _project(scenario, spec, vid, tau[i])
        queue = float(own[1]) + 1.0
        committed_incoming = float(own[2]) if len(own) > 2 else 0.0
        supply = float(own[5]) if len(own) > 5 else 0.0
        pressure = queue / (1.0 + supply)
        tdm.extend([
            float(tau[i]),
            queue,
            committed_incoming,
            supply,
            pressure,
        ])

        fs = _first_supply_ready(scenario, spec, vid)
        gap = fs - float(tau[i])
        p0 = _project(scenario, spec, vid, 0.0)
        waiting_now = float(p0[1])
        ready_now = float(p0[5]) if len(p0) > 5 else 0.0
        first.extend([
            float(tau[i]),
            float(fs),
            float(gap),
            waiting_now,
            ready_now,
        ])
    return tdm, first


def _eligible_hub_count(scenario: Any) -> int:
    local = list(scenario.vertiports.evtols_at_vertiport.get(str(DESTINATION), []) or [])
    return sum(
        1 for e in local
        if (
            mx.state_name(e) == "IDLE"
            and not mx.passenger_ids(e)
            and not base.is_turnaround_busy(e)
        )
    )


def _pair_virtual_features(
    scenario: Any,
    spec: EnvSpec,
    tau: Sequence[float],
) -> List[float]:
    # Four pair effects for the no-HOLD joint action space.
    out: List[float] = []
    if not spec.joint:
        return [0.0] * PAIR_VIRTUAL_DIM

    eligible = _eligible_hub_count(scenario) > 0
    for pair in range(4):
        p = pair // 2
        a = pair % 2
        vid = int(CANDIDATES[p])
        aircraft_target = int(CANDIDATES[a])
        own = _project(scenario, spec, vid, tau[p])
        demand_after = float(own[1]) + 1.0
        supply_after = float(own[5]) if len(own) > 5 else 0.0
        if eligible and aircraft_target == vid:
            supply_after += 1.0
        pressure = demand_after / (1.0 + supply_after)
        out.extend([
            float(tau[p]),
            demand_after,
            supply_after,
            pressure,
        ])
    if len(out) != PAIR_VIRTUAL_DIM:
        raise RuntimeError(len(out))
    return out


# =============================================================================
# Fixed observation layout for every method
# =============================================================================

def make_layout() -> Dict[str, Any]:
    dims = {
        "base": BASE_OBS_DIM,
        "resource": RESOURCE_DIM,
        "focal": FOCAL_DIM,
        "tau": TAU_DIM,
        "shared": SHARED_DIM,
        "srac": SRAC_DIM,
        "gamma": GAMMA_DIM,
        "cqm": CQM_DIM,
        "event": EVENT_DIM,
        "tdm": TDM_DIM,
        "first": FIRST_DIM,
        "pair_virtual": PAIR_VIRTUAL_DIM,
    }
    slices = {}
    pos = 0
    for name in dims:
        slices[name] = (pos, pos + int(dims[name]))
        pos += int(dims[name])
    return {
        "dims": dims,
        "slices": slices,
        "total_dim": pos,
        "num_frames": NUM_FRAMES,
        "single_frame_dim": SINGLE_FRAME_DIM,
    }


GLOBAL_LAYOUT = make_layout()




class TauPreservingVecNormalize(VecNormalize):
    """Normalize the full Box observation except the two raw access-time values.

    The tau-conditioned WM needs an exact integer/minute horizon when it builds
    t -> t+tau auxiliary targets from the rollout buffer. Leaving tau in raw
    simulator minutes avoids approximate inversion of a running normalization
    statistic that changes during the same rollout.
    """
    preserve_tau_raw = True

    def normalize_obs(self, obs):
        raw = np.array(obs, copy=True)
        norm = super().normalize_obs(obs)
        lo, hi = GLOBAL_LAYOUT["slices"]["tau"]
        try:
            norm[..., int(lo):int(hi)] = raw[..., int(lo):int(hi)]
        except Exception:
            pass
        return norm

class LiteratureObservationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, env_key: str, method_id: str):
        super().__init__(env)
        self.env_key = str(env_key).upper()
        self.method_id = str(method_id).upper()
        self.env_spec = ENV_SPECS[self.env_key]
        self.layout = GLOBAL_LAYOUT

        got = int(np.prod(env.observation_space.shape))
        if got != BASE_OBS_DIM:
            raise RuntimeError(f"source observation dim {got} != {BASE_OBS_DIM}")
        self.observation_space = spaces.Box(
            low=-1e9,
            high=1e9,
            shape=(int(self.layout["total_dim"]),),
            dtype=np.float32,
        )

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        scenario = mx.find_scenario(self.env)
        person = _focal_person(self.env, scenario)
        resource = _resource_vector(scenario, self.env_spec)
        focal = _focal_od(person)
        tau, shared, srac, gamma, cqm = _shared_srac_gamma_cqm(
            scenario, self.env_spec, person
        )
        event = _pack_events(scenario, tau)
        tdm, first = _tdm_and_first(scenario, self.env_spec, tau)
        pair_virtual = _pair_virtual_features(scenario, self.env_spec, tau)

        out = np.concatenate([
            np.asarray(obs, dtype=np.float32).reshape(-1),
            np.asarray(resource, dtype=np.float32),
            np.asarray(focal, dtype=np.float32),
            np.asarray(tau, dtype=np.float32),
            np.asarray(shared, dtype=np.float32),
            np.asarray(srac, dtype=np.float32),
            np.asarray(gamma, dtype=np.float32),
            np.asarray(cqm, dtype=np.float32),
            np.asarray(event, dtype=np.float32),
            np.asarray(tdm, dtype=np.float32),
            np.asarray(first, dtype=np.float32),
            np.asarray(pair_virtual, dtype=np.float32),
        ])
        if out.shape != (int(self.layout["total_dim"]),):
            raise RuntimeError(f"observation shape={out.shape} expected={(self.layout['total_dim'],)}")
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
            info = dict(info)
            if bool(terminated) or bool(truncated):
                info.setdefault("terminal_snapshot", mx.snapshot_episode(mx.find_scenario(self.env)))
            return self._transform(obs), reward, terminated, truncated, info
        obs, reward, done, info = out
        info = dict(info)
        if bool(done):
            info.setdefault("terminal_snapshot", mx.snapshot_episode(mx.find_scenario(self.env)))
        return self._transform(obs), reward, done, info


# =============================================================================
# Unified feature extractor
# =============================================================================

class LiteratureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        layout: Optional[Dict[str, Any]] = None,
        method_id: str = "CURRENT",
        env_key: str = "S1",
    ):
        if layout is None:
            raise ValueError("layout required")
        super().__init__(observation_space, features_dim)
        self.layout = dict(layout)
        self.slices = dict(layout["slices"])
        self.method_id = str(method_id).upper()
        self.rep, self.wm_variant = parse_temporal(self.method_id)
        self.env_key = str(env_key).upper()
        self.env_spec = ENV_SPECS[self.env_key]
        self.n_actions = self.env_spec.n_actions

        # Common UAGMC temporal/base representation.
        self.frame_encoder = nn.Sequential(nn.Linear(SINGLE_FRAME_DIM, 128), nn.ReLU())
        self.lstm = nn.LSTM(128, 128, num_layers=1, batch_first=True)
        self.resource_branch = nn.Sequential(
            nn.Linear(RESOURCE_DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.focal_branch = nn.Sequential(
            nn.Linear(FOCAL_DIM, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )

        # Every representation branch exists for every method; inactive branch
        # outputs are zeros. This keeps representation-network parameter capacity
        # matched across the Phase-A cards.
        self.shared_branch = nn.Sequential(nn.Linear(SHARED_DIM, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.srac_branch = nn.Sequential(nn.Linear(SRAC_DIM, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.gamma_branch = nn.Sequential(nn.Linear(GAMMA_DIM + TAU_DIM, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.cqm_branch = nn.Sequential(nn.Linear(CQM_DIM, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.tdm_branch = nn.Sequential(nn.Linear(TDM_DIM, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.first_branch = nn.Sequential(nn.Linear(FIRST_DIM, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.pair_virtual_branch = nn.Sequential(
            nn.Linear(PAIR_VIRTUAL_DIM, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )

        # Event-query encoder: candidate tau is the query; committed event set is key/value.
        self.event_phi = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.event_query = nn.Sequential(nn.Linear(1 + N_CANDIDATES, 32), nn.ReLU(), nn.Linear(32, 32))
        self.event_out = nn.Sequential(nn.Linear(32 * N_CANDIDATES, 64), nn.ReLU())

        # World-model modules are instantiated for all methods so policy capacity
        # is not accidentally changed merely by switching the WM flag.
        wm_in = RESOURCE_DIM + self.n_actions + 1
        self.tau_world_model = nn.Sequential(
            nn.Linear(wm_in, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, RESOURCE_DIM)
        )
        self.pets_world_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(wm_in, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, RESOURCE_DIM)
            )
            for _ in range(PETS_ENSEMBLE)
        ])
        self.value_equiv_model = nn.Sequential(
            nn.Linear(wm_in, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.wm_det_project = nn.Sequential(
            nn.Linear(self.n_actions * RESOURCE_DIM, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )
        self.wm_pets_project = nn.Sequential(
            nn.Linear(self.n_actions * RESOURCE_DIM * 2, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )
        self.wm_ve_project = nn.Sequential(
            nn.Linear(self.n_actions, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )

        concat_dim = 128 + 64 + 32 + 32 + 32 + 64 + 32 + 64 + 32 + 32 + 64 + 32
        self.output_proj = nn.Sequential(nn.Linear(concat_dim, int(features_dim)), nn.ReLU())

    def _slice(self, obs: torch.Tensor, name: str) -> torch.Tensor:
        lo, hi = self.slices[name]
        return obs[:, int(lo):int(hi)]

    def _zeros(self, obs: torch.Tensor, width: int) -> torch.Tensor:
        return torch.zeros((obs.shape[0], int(width)), dtype=obs.dtype, device=obs.device)

    def _selected_tau(self, obs: torch.Tensor, action: int) -> torch.Tensor:
        tau = self._slice(obs, "tau")
        p = int(action) if not self.env_spec.joint else int(action) // 2
        p = min(max(p, 0), N_CANDIDATES - 1)
        return tau[:, p:p + 1]

    def _wm_input(self, resource: torch.Tensor, obs: torch.Tensor, action: int) -> torch.Tensor:
        onehot = F.one_hot(
            torch.full((resource.shape[0],), int(action), device=resource.device, dtype=torch.long),
            num_classes=self.n_actions,
        ).float()
        tau = self._selected_tau(obs, action)
        return torch.cat([resource, onehot, tau], dim=1)

    def predict_all_tau_resources(self, obs: torch.Tensor) -> torch.Tensor:
        resource = self._slice(obs, "resource")
        preds = []
        for a in range(self.n_actions):
            delta = self.tau_world_model(self._wm_input(resource, obs, a))
            preds.append(resource + delta)
        return torch.stack(preds, dim=1)  # B,A,R

    def predict_all_pets_resources(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        resource = self._slice(obs, "resource")
        all_actions = []
        for a in range(self.n_actions):
            inp = self._wm_input(resource, obs, a)
            members = torch.stack([resource + m(inp) for m in self.pets_world_models], dim=1)  # B,M,R
            all_actions.append(members)
        stack = torch.stack(all_actions, dim=1)  # B,A,M,R
        return stack.mean(dim=2), stack.std(dim=2, unbiased=False), stack

    def predict_all_values(self, obs: torch.Tensor) -> torch.Tensor:
        resource = self._slice(obs, "resource")
        vals = []
        for a in range(self.n_actions):
            vals.append(self.value_equiv_model(self._wm_input(resource, obs, a)))
        return torch.cat(vals, dim=1)

    def predict_chosen_tau_resource(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        allp = self.predict_all_tau_resources(obs)
        a = actions.long().flatten().clamp(0, self.n_actions - 1)
        return allp[torch.arange(obs.shape[0], device=obs.device), a]

    def predict_chosen_pets_members(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        _, _, stack = self.predict_all_pets_resources(obs)
        a = actions.long().flatten().clamp(0, self.n_actions - 1)
        return stack[torch.arange(obs.shape[0], device=obs.device), a]  # B,M,R

    def predict_chosen_value(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        vals = self.predict_all_values(obs)
        a = actions.long().flatten().clamp(0, self.n_actions - 1)
        return vals[torch.arange(obs.shape[0], device=obs.device), a]

    def world_model_parameters(self):
        if self.wm_variant == "W_PETS":
            return list(self.pets_world_models.parameters())
        if self.wm_variant == "W_VE":
            return list(self.value_equiv_model.parameters())
        return list(self.tau_world_model.parameters())

    def _event_query_latent(self, obs: torch.Tensor) -> torch.Tensor:
        fine = self._slice(obs, "event")
        tau = self._slice(obs, "tau")
        K = MAX_EVENTS_PER_TYPE
        block = 4 * K
        reps = []
        for c in range(N_CANDIDATES):
            x = fine[:, c * block:(c + 1) * block]
            a_eta = x[:, 0:K]
            a_mask = x[:, K:2 * K]
            p_eta = x[:, 2 * K:3 * K]
            p_mask = x[:, 3 * K:4 * K]
            eta = torch.cat([a_eta, p_eta], dim=1)
            mask = torch.cat([a_mask, p_mask], dim=1)
            typ = torch.cat([torch.zeros_like(a_eta), torch.ones_like(p_eta)], dim=1)
            elems = torch.stack([eta, typ], dim=-1)
            emb = self.event_phi(elems)

            cand = F.one_hot(
                torch.full((obs.shape[0],), c, device=obs.device, dtype=torch.long),
                num_classes=N_CANDIDATES,
            ).float()
            q = self.event_query(torch.cat([tau[:, c:c + 1], cand], dim=1))
            score = (emb * q.unsqueeze(1)).sum(dim=-1) / math.sqrt(emb.shape[-1])
            score = score.masked_fill(mask <= 0.0, -1e9)
            weights = torch.softmax(score, dim=1) * mask
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            weights = weights / denom
            pooled = (emb * weights.unsqueeze(-1)).sum(dim=1)
            any_event = (mask.sum(dim=1, keepdim=True) > 0).float()
            reps.append(pooled * any_event)
        return self.event_out(torch.cat(reps, dim=1))

    def _forward_impl(self, observations: torch.Tensor, include_wm: bool) -> torch.Tensor:
        obs = observations.float()
        base_obs = self._slice(obs, "base")
        B = base_obs.shape[0]
        frames = base_obs.reshape(B, NUM_FRAMES, SINGLE_FRAME_DIM)
        x = self.frame_encoder(frames)
        x, _ = self.lstm(x)
        base_latent = x[:, -1, :]
        resource_latent = self.resource_branch(self._slice(obs, "resource"))
        focal_latent = self.focal_branch(self._slice(obs, "focal"))

        shared_latent = self._zeros(obs, 32)
        srac_latent = self._zeros(obs, 32)
        gamma_latent = self._zeros(obs, 64)
        cqm_latent = self._zeros(obs, 32)
        event_latent = self._zeros(obs, 64)
        tdm_latent = self._zeros(obs, 32)
        first_latent = self._zeros(obs, 32)

        if self.rep in ("SHARED", "R_SRAC"):
            shared_latent = self.shared_branch(self._slice(obs, "shared"))
        if self.rep == "R_SRAC":
            srac_latent = self.srac_branch(self._slice(obs, "srac"))
        elif self.rep == "R_GAMMA":
            gamma_latent = self.gamma_branch(torch.cat([self._slice(obs, "gamma"), self._slice(obs, "tau")], dim=1))
        elif self.rep == "R_CQM":
            cqm_latent = self.cqm_branch(self._slice(obs, "cqm"))
        elif self.rep == "R_EVENTQ":
            event_latent = self._event_query_latent(obs)
        elif self.rep == "R_TDM":
            tdm_latent = self.tdm_branch(self._slice(obs, "tdm"))
        elif self.rep == "R_FIRST":
            first_latent = self.first_branch(self._slice(obs, "first"))

        wm_latent = self._zeros(obs, 64)
        if include_wm and self.wm_variant:
            if self.wm_variant == "W_PETS":
                mean, std, _ = self.predict_all_pets_resources(obs)
                wm_latent = self.wm_pets_project(torch.cat([mean.detach().flatten(1), std.detach().flatten(1)], dim=1))
            elif self.wm_variant == "W_VE":
                vals = self.predict_all_values(obs)
                wm_latent = self.wm_ve_project(vals.detach())
            else:
                pred = self.predict_all_tau_resources(obs)
                wm_latent = self.wm_det_project(pred.detach().flatten(1))

        pair_latent = self._zeros(obs, 32)
        if self.env_key == "J7":
            pair_latent = self.pair_virtual_branch(self._slice(obs, "pair_virtual"))

        pieces = [
            base_latent,
            resource_latent,
            focal_latent,
            shared_latent,
            srac_latent,
            gamma_latent,
            cqm_latent,
            event_latent,
            tdm_latent,
            first_latent,
            wm_latent,
            pair_latent,
        ]
        return self.output_proj(torch.cat(pieces, dim=1))

    def forward_without_wm(self, observations: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(observations, include_wm=False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(observations, include_wm=True)


# =============================================================================
# Tau-conditioned auxiliary world model PPO
# =============================================================================

class LiteratureWorldModelPPO(PPO):
    def _extractor(self) -> Optional[LiteratureExtractor]:
        ext = getattr(self.policy, "features_extractor", None)
        return ext if isinstance(ext, LiteratureExtractor) else None

    def _vecnorm(self):
        try:
            return self.get_vec_normalize_env()
        except Exception:
            return None

    def _raw_tau_from_normalized(self, obs_np: np.ndarray, ext: LiteratureExtractor) -> np.ndarray:
        lo, hi = ext.slices["tau"]
        tau_norm = np.asarray(obs_np[..., int(lo):int(hi)], dtype=float)
        vn = self._vecnorm()
        if bool(getattr(vn, "preserve_tau_raw", False)):
            return tau_norm
        rms = getattr(vn, "obs_rms", None) if vn is not None else None
        if rms is None:
            return tau_norm
        mean = np.asarray(rms.mean[int(lo):int(hi)], dtype=float)
        var = np.asarray(rms.var[int(lo):int(hi)], dtype=float)
        eps = float(getattr(vn, "epsilon", 1e-8))
        return tau_norm * np.sqrt(var + eps) + mean

    def _future_pairs(self, ext: LiteratureExtractor):
        obs_np = np.asarray(self.rollout_buffer.observations)
        act_np = np.asarray(self.rollout_buffer.actions)
        starts_np = np.asarray(self.rollout_buffer.episode_starts)
        if obs_np.ndim < 3 or obs_np.shape[0] < 2:
            return None

        n_steps, n_envs = obs_np.shape[:2]
        tau_raw = self._raw_tau_from_normalized(obs_np, ext)
        acts = act_np.reshape(n_steps, n_envs, -1)[..., 0].astype(int)

        idxs: List[Tuple[int, int, int]] = []
        for t in range(n_steps - 1):
            for e in range(n_envs):
                a = int(acts[t, e])
                p = a if not ext.env_spec.joint else a // 2
                p = min(max(p, 0), N_CANDIDATES - 1)
                h = fnum(tau_raw[t, e, p], 1.0)
                off = int(round(max(1.0, min(float(WM_HORIZON_CAP), h))))
                tgt = t + off
                if tgt >= n_steps:
                    continue
                if np.any(starts_np[t + 1:tgt + 1, e] > 0.5):
                    continue
                idxs.append((t, e, tgt))

        if not idxs:
            return None
        if len(idxs) > WM_MAX_SAMPLES:
            pick = np.random.choice(len(idxs), size=WM_MAX_SAMPLES, replace=False)
            idxs = [idxs[int(i)] for i in pick]

        cur = np.stack([obs_np[t, e] for t, e, _ in idxs], axis=0)
        nxt = np.stack([obs_np[tgt, e] for t, e, tgt in idxs], axis=0)
        act = np.asarray([acts[t, e] for t, e, _ in idxs], dtype=np.int64)

        cur_t = torch.as_tensor(cur, device=self.device, dtype=torch.float32)
        nxt_t = torch.as_tensor(nxt, device=self.device, dtype=torch.float32)
        act_t = torch.as_tensor(act, device=self.device, dtype=torch.long)
        return cur_t, act_t, nxt_t

    def _value_without_wm(self, obs: torch.Tensor, ext: LiteratureExtractor) -> torch.Tensor:
        features = ext.forward_without_wm(obs)
        latent_vf = self.policy.mlp_extractor.forward_critic(features)
        return self.policy.value_net(latent_vf).flatten()

    def _loss_batch(
        self,
        ext: LiteratureExtractor,
        cur: torch.Tensor,
        act: torch.Tensor,
        nxt: torch.Tensor,
    ) -> torch.Tensor:
        rlo, rhi = ext.slices["resource"]
        target_resource = nxt[:, int(rlo):int(rhi)].detach()
        variant = ext.wm_variant

        if variant == "W_VE":
            with torch.no_grad():
                target_v = self._value_without_wm(nxt, ext)
            pred_v = ext.predict_chosen_value(cur, act)
            return F.mse_loss(pred_v, target_v)

        if variant == "W_PETS":
            members = ext.predict_chosen_pets_members(cur, act)  # B,M,R
            losses = []
            for m in range(members.shape[1]):
                pred = members[:, m, :]
                mask = (torch.rand(pred.shape[0], device=pred.device) < 0.8).float()
                mse = ((pred - target_resource) ** 2).mean(dim=1)
                losses.append((mse * mask).sum() / mask.sum().clamp_min(1.0))
            return torch.stack(losses).mean()

        pred = ext.predict_chosen_tau_resource(cur, act)
        state_loss = F.mse_loss(pred, target_resource)

        if variant == "W_VAML":
            pseudo = nxt.clone()
            pseudo[:, int(rlo):int(rhi)] = pred
            with torch.no_grad():
                target_v = self._value_without_wm(nxt, ext)
            pred_v = self._value_without_wm(pseudo, ext)
            value_loss = F.mse_loss(pred_v, target_v)
            return state_loss + VAML_LAMBDA * value_loss

        if variant == "W_COCO":
            allp = ext.predict_all_tau_resources(cur)  # B,A,R
            # Counterfactual action-separation regularizer. This intentionally
            # prevents action-conditioned predictions from collapsing to an
            # action-agnostic state-inertia model.
            diffs = []
            for i in range(ext.n_actions):
                for j in range(i + 1, ext.n_actions):
                    d = ((allp[:, i] - allp[:, j]) ** 2).mean(dim=1)
                    diffs.append(F.relu(COCO_MARGIN - d))
            coco = torch.stack(diffs, dim=1).mean() if diffs else state_loss * 0.0
            return state_loss + COCO_LAMBDA * coco

        # W_TAU
        return state_loss

    def _train_world_model(self) -> Optional[float]:
        ext = self._extractor()
        if ext is None or not ext.wm_variant:
            return None
        pairs = self._future_pairs(ext)
        if pairs is None:
            return None
        cur, act, nxt = pairs

        if not hasattr(self, "_wm_optimizer"):
            self._wm_optimizer = torch.optim.Adam(ext.world_model_parameters(), lr=WM_AUX_LR)

        losses = []
        n = cur.shape[0]
        for _ in range(WM_AUX_EPOCHS):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, WM_AUX_BATCH):
                idx = order[start:start + WM_AUX_BATCH]
                loss = self._loss_batch(ext, cur[idx], act[idx], nxt[idx])
                self._wm_optimizer.zero_grad(set_to_none=True)
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ext.world_model_parameters(), 5.0)
                self._wm_optimizer.step()
                # Auxiliary loss must not leave gradients on PPO-only params.
                self.policy.optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else None

    def train(self) -> None:
        wm_loss = self._train_world_model()
        super().train()
        if wm_loss is not None:
            self.logger.record("train/literature_world_model_aux_loss", wm_loss)


# =============================================================================
# Eight Joint architecture cards, all over Discrete(4)
# =============================================================================

class LiteratureJointPolicy(ActorCriticPolicy):
    def __init__(self, *args, joint_arch: str = "J0", **kwargs):
        self.joint_arch = str(joint_arch).upper()
        lr_schedule = kwargs.get("lr_schedule", None)
        if lr_schedule is None and len(args) >= 3:
            lr_schedule = args[2]
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_space, spaces.Discrete) or int(self.action_space.n) != 4:
            raise RuntimeError(f"LiteratureJointPolicy requires Discrete(4), got {self.action_space}")

        d = int(self.mlp_extractor.latent_dim_pi)
        self.action_net = nn.Identity()

        if self.joint_arch == "J0":
            self.p_head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 2))
            self.a_head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 2))
        elif self.joint_arch == "J1":
            self.shared_context = nn.Sequential(nn.Linear(d, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())
            self.p_head = nn.Linear(64, 2)
            self.a_head = nn.Linear(64, 2)
        elif self.joint_arch == "J2":
            self.pair_head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 4))
        elif self.joint_arch == "J3":
            self.p_head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 2))
            self.cond_a = nn.Sequential(nn.Linear(d + 2, 48), nn.ReLU(), nn.Linear(48, 2))
        elif self.joint_arch == "J4":
            self.semantic_pair = nn.Sequential(nn.Linear(d + 4, 64), nn.ReLU(), nn.Linear(64, 1))
        elif self.joint_arch == "J5":
            self.p_util = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 2))
            self.a_util = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 2))
            self.mix_w = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 3))
        elif self.joint_arch == "J6":
            self.p_util = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 2))
            self.a_util = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 2))
            self.joint_residual = nn.Sequential(nn.Linear(d + 4, 48), nn.ReLU(), nn.Linear(48, 1))
        elif self.joint_arch == "J7":
            self.effect_pair = nn.Sequential(nn.Linear(d + 4, 64), nn.ReLU(), nn.Linear(64, 1))
        else:
            raise ValueError(self.joint_arch)

        if lr_schedule is None:
            raise RuntimeError("Cannot recover learning-rate schedule")
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _semantic(self, latent: torch.Tensor, p: int, a: int) -> torch.Tensor:
        B = latent.shape[0]
        p_oh = F.one_hot(
            torch.full((B,), p, device=latent.device, dtype=torch.long), num_classes=2
        ).float()
        a_oh = F.one_hot(
            torch.full((B,), a, device=latent.device, dtype=torch.long), num_classes=2
        ).float()
        return torch.cat([latent, p_oh, a_oh], dim=1)

    def _joint_logits(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.shape[0]
        j = self.joint_arch
        if j == "J0":
            p = self.p_head(latent)
            a = self.a_head(latent)
            return (p.unsqueeze(2) + a.unsqueeze(1)).reshape(B, 4)
        if j == "J1":
            h = self.shared_context(latent)
            p = self.p_head(h)
            a = self.a_head(h)
            return (p.unsqueeze(2) + a.unsqueeze(1)).reshape(B, 4)
        if j == "J2":
            return self.pair_head(latent)
        if j == "J3":
            p_logits = self.p_head(latent)
            p_logp = F.log_softmax(p_logits, dim=1)
            rows = []
            for p in range(2):
                p_oh = F.one_hot(
                    torch.full((B,), p, device=latent.device, dtype=torch.long), num_classes=2
                ).float()
                a_logp = F.log_softmax(self.cond_a(torch.cat([latent, p_oh], dim=1)), dim=1)
                rows.append(p_logp[:, p:p + 1] + a_logp)
            return torch.stack(rows, dim=1).reshape(B, 4)
        if j == "J4":
            vals = []
            for p in range(2):
                for a in range(2):
                    vals.append(self.semantic_pair(self._semantic(latent, p, a)))
            return torch.cat(vals, dim=1)
        if j == "J5":
            # QMIX-style monotonic state-conditioned mixing of two utilities.
            pu = self.p_util(latent)
            au = self.a_util(latent)
            mix = self.mix_w(latent)
            wp = F.softplus(mix[:, 0:1]) + 1e-4
            wa = F.softplus(mix[:, 1:2]) + 1e-4
            b = mix[:, 2:3]
            scores = []
            for p in range(2):
                for a in range(2):
                    scores.append(wp * pu[:, p:p + 1] + wa * au[:, a:a + 1] + b)
            return torch.cat(scores, dim=1)
        if j == "J6":
            # QTRAN-style architectural analogue: decomposed utilities plus an
            # unconstrained pair residual. PPO remains the optimizer.
            pu = self.p_util(latent)
            au = self.a_util(latent)
            scores = []
            for p in range(2):
                for a in range(2):
                    residual = self.joint_residual(self._semantic(latent, p, a))
                    scores.append(pu[:, p:p + 1] + au[:, a:a + 1] + residual)
            return torch.cat(scores, dim=1)
        # J7: pair-effect virtual features are encoded only for J7 by the extractor.
        vals = []
        for p in range(2):
            for a in range(2):
                vals.append(self.effect_pair(self._semantic(latent, p, a)))
        return torch.cat(vals, dim=1)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        logits = self._joint_logits(latent_pi)
        return self.action_dist.proba_distribution(action_logits=logits)


# =============================================================================
# Environment / model construction
# =============================================================================

def make_env_factory(
    *,
    env_key: str,
    method_id: str,
    env_index: int,
    run_dir: Path,
    max_time: int,
):
    spec = ENV_SPECS[str(env_key).upper()]

    def _init():
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        configure_worker_physics(max_time)
        env = core.make_experiment_env_factory(
            stage=spec.physical_stage,
            topology=TOPOLOGY,
            encoder_mode="uagmc",
            fleet_size=FLEET_SIZE,
            env_index=int(env_index),
            run_dir=run_dir,
            pad_separation=float(spec.pad_separation),
            charger_capacity=int(spec.charger_capacity),
            max_time=int(max_time),
        )()
        if spec.joint:
            env = NoHoldJointControlWrapper(env, spec.key)
        env = LiteratureObservationWrapper(env, env_key=spec.key, method_id=method_id)
        return env

    return _init


def build_vec_env(
    *,
    env_key: str,
    method_id: str,
    profile: SpeedProfile,
    seed: int,
    run_dir: Path,
    max_time: int,
) -> VecNormalize:
    factories = [
        make_env_factory(
            env_key=env_key,
            method_id=method_id,
            env_index=i,
            run_dir=run_dir,
            max_time=max_time,
        )
        for i in range(profile.n_envs)
    ]
    raw = SubprocVecEnv(factories, start_method="spawn")
    raw.seed(int(seed))
    return TauPreservingVecNormalize(
        raw,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )


def build_eval_env(
    *,
    env_key: str,
    method_id: str,
    run_dir: Path,
    max_time: int,
) -> DummyVecEnv:
    return DummyVecEnv([
        make_env_factory(
            env_key=env_key,
            method_id=method_id,
            env_index=9999,
            run_dir=run_dir,
            max_time=max_time,
        )
    ])


def build_model(
    *,
    env: VecNormalize,
    env_key: str,
    method_id: str,
    profile: SpeedProfile,
    seed: int,
    run_dir: Path,
    device: str,
):
    spec = ENV_SPECS[env_key]
    _, wm = parse_temporal(method_id)
    policy_kwargs = dict(
        features_extractor_class=LiteratureExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            layout=GLOBAL_LAYOUT,
            method_id=method_id,
            env_key=env_key,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    policy: Any = "MlpPolicy"
    if spec.joint:
        policy = LiteratureJointPolicy
        policy_kwargs["joint_arch"] = env_key
    algo_cls = LiteratureWorldModelPPO if wm else PPO
    return algo_cls(
        policy=policy,
        env=env,
        learning_rate=base.linear_schedule(INITIAL_LR),
        n_steps=int(profile.n_steps),
        batch_size=int(profile.batch_size),
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        seed=int(seed),
        verbose=1,
        device=device,
        tensorboard_log=str(run_dir / "tb"),
    )


def count_params(model: Any) -> int:
    return int(sum(p.numel() for p in model.policy.parameters() if p.requires_grad))


# =============================================================================
# Checkpointing / throughput
# =============================================================================

class ThroughputCallback(BaseCallback):
    def __init__(self, run_dir: Path, interval: int = CHECKPOINT_INTERVAL):
        super().__init__(verbose=0)
        self.run_dir = Path(run_dir)
        self.interval = int(interval)
        self.next_mark = int(interval)
        self.started = 0.0
        self.rows: List[Dict[str, Any]] = []

    def _on_training_start(self) -> None:
        self.started = time.perf_counter()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_mark:
            elapsed = time.perf_counter() - self.started
            self.rows.append({
                "timesteps": int(self.num_timesteps),
                "elapsed_sec": float(elapsed),
                "sps": float(self.num_timesteps / max(elapsed, 1e-9)),
            })
            write_csv(self.run_dir / "training_throughput.csv", self.rows)
            while self.next_mark <= self.num_timesteps:
                self.next_mark += self.interval
        return True


def build_callbacks(run_dir: Path, profile: SpeedProfile) -> List[BaseCallback]:
    if CHECKPOINT_INTERVAL % profile.n_envs != 0:
        raise ValueError("checkpoint interval must divide by n_envs")
    return [
        CheckpointCallback(
            save_freq=CHECKPOINT_INTERVAL // profile.n_envs,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="uam_ppo",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=0,
        ),
        ThroughputCallback(run_dir),
    ]


# =============================================================================
# Evaluation
# =============================================================================

def policy_probs(model: PPO, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor).distribution
        if getattr(dist, "probs", None) is not None:
            p = dist.probs
        else:
            p = torch.softmax(dist.logits, dim=-1)
        return np.asarray(p.detach().cpu().numpy(), dtype=float).reshape(-1)


def evaluate_checkpoint(
    *,
    env_key: str,
    method_id: str,
    model_path: Path,
    vec_path: Path,
    train_step: int,
    eval_seed: int,
    run_dir: Path,
    max_time: int,
) -> Dict[str, Any]:
    seed_all(eval_seed)
    raw = build_eval_env(
        env_key=env_key,
        method_id=method_id,
        run_dir=run_dir / "_eval_monitor",
        max_time=max_time,
    )
    env = TauPreservingVecNormalize.load(str(vec_path), raw)
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
        system_person_minutes = 0.0
        episode_steps = 0
        terminal_snapshot = None

        dispatch_attempts = 0
        dispatch_success = 0
        dispatch_infeasible = 0
        passenger_counts = Counter()
        aircraft_counts = Counter()
        pair_consistency = 0

        while not bool(done[0]):
            scenario = mx.find_scenario(env)
            system_person_minutes += legacy.active_person_count(scenario)

            probs = policy_probs(model, obs)
            action, _ = model.predict(obs, deterministic=True)
            ai = int(np.asarray(action).reshape(-1)[0])
            action_counts[ai] += 1
            prob_rows.append(probs.copy())

            if ENV_SPECS[env_key].joint:
                p = ai // 2
                a = ai % 2
                passenger_counts[p] += 1
                aircraft_counts[a] += 1
                pair_consistency += int(p == a)
                dispatch_attempts += 1

            obs, reward, done, infos = env.step(action)
            reward_sum += fnum(np.asarray(reward).reshape(-1)[0], 0.0)
            episode_steps += 1

            info = infos[0] if infos else {}
            if isinstance(info, dict):
                if "terminal_snapshot" in info:
                    terminal_snapshot = info["terminal_snapshot"]
                if ENV_SPECS[env_key].joint:
                    if bool(info.get("aircraft_dispatched", False)):
                        dispatch_success += 1
                    elif int(info.get("aircraft_eligible_hub_count", 0)) <= 0:
                        dispatch_infeasible += 1

            if episode_steps > int(max_time) + 5:
                raise RuntimeError("evaluation exceeded hard guard")

        if terminal_snapshot is None:
            terminal_snapshot = mx.snapshot_episode(mx.find_scenario(env))

        metrics = mx.metrics_from_terminal_snapshot(
            snapshot=terminal_snapshot,
            action_counts=action_counts,
            prob_rows=prob_rows,
            system_person_minutes=system_person_minutes,
            reward_sum=reward_sum,
            episode_steps=episode_steps,
        )

        completion = fnum(metrics.get("completion_rate"), 0.0)
        valid = completion >= 0.999999
        metrics.update({
            "env_key": env_key,
            "joint_arch_name": JOINT_ARCH_NAMES.get(env_key, "SINGLE"),
            "method_id": method_id,
            **method_description(method_id),
            "train_step": int(train_step),
            "eval_seed": int(eval_seed),
            "valid_full_completion": bool(valid),
            "hard_guard": int(max_time),
            "dispatch_attempts": dispatch_attempts,
            "dispatch_success_rate": dispatch_success / max(1, dispatch_attempts),
            "dispatch_infeasible_rate": dispatch_infeasible / max(1, dispatch_attempts),
            "passenger_V0_share": passenger_counts[0] / max(1, sum(passenger_counts.values())),
            "passenger_V1_share": passenger_counts[1] / max(1, sum(passenger_counts.values())),
            "aircraft_V0_share": aircraft_counts[0] / max(1, sum(aircraft_counts.values())),
            "aircraft_V1_share": aircraft_counts[1] / max(1, sum(aircraft_counts.values())),
            "pair_consistency": pair_consistency / max(1, dispatch_attempts),
        })
        if not valid:
            metrics["ATT"] = float("nan")
            metrics["AWT"] = float("nan")
        return metrics
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            core.restore_process_patches()
        except Exception:
            pass
        gc.collect()


def checkpoint_paths(run_dir: Path, step: int) -> Tuple[Path, Path]:
    return (
        run_dir / "checkpoints" / f"uam_ppo_{step}_steps.zip",
        run_dir / "checkpoints" / f"uam_ppo_vecnormalize_{step}_steps.pkl",
    )


def analyze_cell(
    *,
    env_key: str,
    method_id: str,
    run_dir: Path,
    requested_steps: int,
    eval_seeds: Sequence[int],
    max_time: int,
) -> Dict[str, Any]:
    adir = run_dir / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for step in range(CHECKPOINT_INTERVAL, requested_steps + 1, CHECKPOINT_INTERVAL):
        model_path, vec_path = checkpoint_paths(run_dir, step)
        if not model_path.exists() or not vec_path.exists():
            continue
        for seed in eval_seeds:
            try:
                row = evaluate_checkpoint(
                    env_key=env_key,
                    method_id=method_id,
                    model_path=model_path,
                    vec_path=vec_path,
                    train_step=step,
                    eval_seed=int(seed),
                    run_dir=run_dir,
                    max_time=max_time,
                )
                rows.append(row)
                print(
                    f"  [eval] {env_key}/{method_id} {step//1000:>3}k seed={seed} "
                    f"ATT={fnum(row.get('ATT')):.3f} "
                    f"finish={int(row.get('N_finished',0))}/{int(row.get('N',0))}",
                    flush=True,
                )
            except Exception as exc:
                errors.append({
                    "train_step": step,
                    "eval_seed": seed,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
    write_csv(adir / "checkpoint_eval_raw.csv", rows)
    write_csv(adir / "errors.csv", errors)

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[int(r["train_step"])].append(r)

    curve: List[Dict[str, Any]] = []
    metric_keys = (
        "ATT", "AWT", "AGT_access", "AFT", "completion_rate",
        "travel_p90", "travel_p95", "system_person_minutes_per_passenger",
        "dispatch_success_rate", "dispatch_infeasible_rate", "pair_consistency",
        "passenger_V0_share", "passenger_V1_share", "aircraft_V0_share", "aircraft_V1_share",
    )
    for step in sorted(grouped):
        rs = grouped[step]
        valid = [r for r in rs if bool(r.get("valid_full_completion"))]
        row = {
            "env_key": env_key,
            "method_id": method_id,
            "train_step": step,
            "n_eval_seeds": len(rs),
            "n_valid": len(valid),
            "all_full_completion": len(valid) == len(rs),
        }
        for key in metric_keys:
            src = valid if key in ("ATT", "AWT") else rs
            row[key + "_mean"] = fmean(r.get(key) for r in src)
            row[key + "_std"] = fstd(r.get(key) for r in src)
        curve.append(row)
    write_csv(adir / "checkpoint_curve.csv", curve)

    valid_curve = [
        r for r in curve
        if bool(r.get("all_full_completion")) and math.isfinite(fnum(r.get("ATT_mean")))
    ]
    max_completion = max([fnum(r.get("completion_rate_mean"), 0.0) for r in curve] or [0.0])

    if valid_curve:
        best = min(valid_curve, key=lambda r: fnum(r.get("ATT_mean")))
        final_candidates = [r for r in valid_curve if int(r["train_step"]) == requested_steps]
        final = final_candidates[0] if final_candidates else valid_curve[-1]
        late = [r for r in valid_curve if int(r["train_step"]) >= max(400_000, requested_steps - 200_000)]
        status = "VALID"
        best_step = int(best["train_step"])
        best_att = fnum(best.get("ATT_mean"))
        final_step = int(final["train_step"])
        final_att = fnum(final.get("ATT_mean"))
        late_mean = fmean(r.get("ATT_mean") for r in late)
        late_std = fstd(r.get("ATT_mean") for r in late)
    else:
        status = "NO_VALID_FULL_COMPLETION"
        best_step = -1
        best_att = float("nan")
        final_step = requested_steps
        final_att = float("nan")
        late_mean = float("nan")
        late_std = float("nan")

    summary = {
        "analysis_status": status,
        "env_key": env_key,
        "joint_arch_name": JOINT_ARCH_NAMES.get(env_key, "SINGLE"),
        "method_id": method_id,
        **method_description(method_id),
        "best_step": best_step,
        "best_ATT": best_att,
        "final_step": final_step,
        "final_ATT": final_att,
        "late_ATT_mean": late_mean,
        "late_ATT_std_across_checkpoints": late_std,
        "max_completion_rate": max_completion,
        "n_eval_rows": len(rows),
        "n_eval_errors": len(errors),
    }
    write_json(adir / "cell_summary.json", summary)
    return summary


# =============================================================================
# Training cell lifecycle
# =============================================================================

def cell_id(env_key: str, method_id: str) -> str:
    return f"{env_key}__{safe_name(method_id)}"


def training_complete(run_dir: Path, requested_steps: int) -> bool:
    p = run_dir / "run_end.json"
    model = run_dir / "final_rl_model.zip"
    vec = run_dir / "final_vec_normalize.pkl"
    if not p.exists() or not model.exists() or not vec.exists():
        return False
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return int(obj.get("requested_timesteps", -1)) == int(requested_steps) and str(obj.get("status", "")).upper() in {
            "TRAINED", "TRAINED_PENDING_EVAL", "SUCCESS", "EVAL_FAILED", "SUCCESS_NO_VALID",
        }
    except Exception:
        return False


def analysis_complete(run_dir: Path) -> bool:
    return (run_dir / "analysis" / "cell_summary.json").exists()


def train_cell_only(
    *,
    root: Path,
    env_key: str,
    method_id: str,
    requested_steps: int,
    train_seed: int,
    profile: SpeedProfile,
    device: str,
    max_time: int,
    defer_eval: bool,
) -> Dict[str, Any]:
    cid = cell_id(env_key, method_id)
    run_dir = root / cid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    seed_all(train_seed)
    env = None
    model = None
    started = time.perf_counter()
    try:
        env = build_vec_env(
            env_key=env_key,
            method_id=method_id,
            profile=profile,
            seed=train_seed,
            run_dir=run_dir,
            max_time=max_time,
        )
        model = build_model(
            env=env,
            env_key=env_key,
            method_id=method_id,
            profile=profile,
            seed=train_seed,
            run_dir=run_dir,
            device=device,
        )

        manifest = {
            "cell_id": cid,
            "env": asdict(ENV_SPECS[env_key]),
            **method_description(method_id),
            "requested_timesteps": int(requested_steps),
            "train_seed": int(train_seed),
            "profile": asdict(profile),
            "device": str(model.device),
            "policy_params": count_params(model),
            "hard_guard": int(max_time),
            "joint_hold_removed": bool(ENV_SPECS[env_key].joint),
            "joint_action_space": "Discrete(4)=2 passenger x 2 aircraft targets" if ENV_SPECS[env_key].joint else "Discrete(2)",
            "evaluation_deferred": bool(defer_eval),
            "wm": {
                "variant": parse_temporal(method_id)[1],
                "aux_lr": WM_AUX_LR,
                "aux_epochs": WM_AUX_EPOCHS,
                "horizon_cap_min": WM_HORIZON_CAP,
                "pets_ensemble": PETS_ENSEMBLE,
            },
        }
        write_json(run_dir / "run_manifest.json", manifest)

        print("\n" + "=" * 120)
        print(
            f"START {cid} | {requested_steps:,} | {profile.name} | {device} | "
            f"params={count_params(model):,} | defer_eval={defer_eval}",
            flush=True,
        )
        print("=" * 120, flush=True)

        model.learn(
            total_timesteps=int(requested_steps),
            callback=build_callbacks(run_dir, profile),
            progress_bar=False,
            reset_num_timesteps=True,
        )
        elapsed = time.perf_counter() - started
        model.save(run_dir / "final_rl_model")
        env.save(run_dir / "final_vec_normalize.pkl")
        run_end = {
            "status": "TRAINED_PENDING_EVAL" if defer_eval else "TRAINED",
            "requested_timesteps": int(requested_steps),
            "actual_timesteps": int(model.num_timesteps),
            "elapsed_sec": elapsed,
            "sps": int(model.num_timesteps) / max(elapsed, 1e-9),
        }
        write_json(run_dir / "run_end.json", run_end)
        return run_end
    except Exception:
        write_json(run_dir / "run_end.json", {
            "status": "TRAIN_FAILED",
            "requested_timesteps": int(requested_steps),
            "error": traceback.format_exc(),
        })
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        model = None
        env = None
        try:
            core.restore_process_patches()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def update_run_end_after_analysis(run_dir: Path, summary: Dict[str, Any]) -> None:
    p = run_dir / "run_end.json"
    obj = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    obj["analysis"] = summary
    obj["status"] = "SUCCESS" if summary.get("analysis_status") == "VALID" else "SUCCESS_NO_VALID"
    write_json(p, obj)


def mark_eval_failed(run_dir: Path, exc: BaseException) -> None:
    p = run_dir / "run_end.json"
    obj = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    obj["status"] = "EVAL_FAILED"
    obj["eval_error"] = repr(exc)
    obj["eval_traceback"] = traceback.format_exc()
    write_json(p, obj)


# =============================================================================
# Selection logic for staged discovery
# =============================================================================

def read_summary(root: Path, env_key: str, method_id: str) -> Optional[Dict[str, Any]]:
    p = root / cell_id(env_key, method_id) / "analysis" / "cell_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def method_score(summary: Optional[Dict[str, Any]]) -> float:
    if not summary:
        return 1e15
    att = fnum(summary.get("best_ATT"))
    if math.isfinite(att):
        return att
    comp = fnum(summary.get("max_completion_rate"), 0.0)
    return 1e9 + (1.0 - comp) * 1e6


def select_top_phase_a(root: Path, n: int = 3) -> List[str]:
    ranked = sorted(PHASE_A_REPS, key=lambda m: method_score(read_summary(root, "S3", m)))
    return list(ranked[:n])


def best_method(root: Path, candidates: Sequence[str]) -> Optional[str]:
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda m: method_score(read_summary(root, "S3", m)))
    return ranked[0] if ranked else None


def choose_joint_temporal_columns(
    root: Path,
    top_reps: Sequence[str],
    phase_b_methods: Sequence[str],
    wildcard_methods: Sequence[str],
) -> List[Tuple[str, str]]:
    # T0/T1 are simple anchors. T2-T4 are Phase-A top3. T5-T7 are category winners.
    cols: List[Tuple[str, str]] = [("T0", "CURRENT"), ("T1", "SHARED")]
    for i, rep in enumerate(list(top_reps)[:3], start=2):
        cols.append((f"T{i}", rep))

    tau_candidates = [m for m in phase_b_methods if parse_temporal(m)[1] == "W_TAU"]
    value_candidates = [m for m in phase_b_methods if parse_temporal(m)[1] in ("W_VAML", "W_VE")]
    uncertainty_candidates = [m for m in phase_b_methods if parse_temporal(m)[1] == "W_PETS"] + list(wildcard_methods)

    picks = [
        ("T5", best_method(root, tau_candidates)),
        ("T6", best_method(root, value_candidates)),
        ("T7", best_method(root, uncertainty_candidates)),
    ]
    used = {m for _, m in cols}
    fallback_pool = list(phase_b_methods) + list(wildcard_methods) + list(PHASE_A_REPS)
    for label, pick in picks:
        if pick is None or pick in used:
            for x in sorted(fallback_pool, key=lambda m: method_score(read_summary(root, "S3", m))):
                if x not in used:
                    pick = x
                    break
        if pick is None:
            pick = "CURRENT"
        cols.append((label, pick))
        used.add(pick)
    return cols[:8]


# =============================================================================
# Status/master bookkeeping
# =============================================================================

def append_status(root: Path, row: Dict[str, Any]) -> None:
    p = root / "serial_status.csv"
    rows = read_csv(p)
    rows.append(row)
    write_csv(p, rows)


def build_master(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for p in root.glob("*/analysis/cell_summary.json"):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    rows = sorted(rows, key=lambda r: (str(r.get("env_key")), str(r.get("method_id"))))
    write_csv(root / "matrix_master.csv", rows)
    return rows


def safe_train_and_optional_eval(
    *,
    root: Path,
    env_key: str,
    method_id: str,
    requested_steps: int,
    train_seed: int,
    eval_seeds: Sequence[int],
    profile: SpeedProfile,
    device: str,
    max_time: int,
    defer_eval: bool,
    fail_fast: bool,
) -> None:
    cid = cell_id(env_key, method_id)
    run_dir = root / cid

    try:
        if training_complete(run_dir, requested_steps):
            print(f"[SKIP TRAINED] {cid}", flush=True)
        else:
            train_cell_only(
                root=root,
                env_key=env_key,
                method_id=method_id,
                requested_steps=requested_steps,
                train_seed=train_seed,
                profile=profile,
                device=device,
                max_time=max_time,
                defer_eval=defer_eval,
            )
            append_status(root, {"cell_id": cid, "stage": "TRAIN", "status": "TRAINED"})
    except Exception as exc:
        append_status(root, {
            "cell_id": cid,
            "stage": "TRAIN",
            "status": "FAILED",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        print(f"[TRAIN ERROR -> SKIP] {cid}: {exc!r}", flush=True)
        if fail_fast:
            raise
        return

    if defer_eval:
        return

    try:
        if analysis_complete(run_dir):
            print(f"[SKIP ANALYZED] {cid}", flush=True)
            return
        summary = analyze_cell(
            env_key=env_key,
            method_id=method_id,
            run_dir=run_dir,
            requested_steps=requested_steps,
            eval_seeds=eval_seeds,
            max_time=max_time,
        )
        update_run_end_after_analysis(run_dir, summary)
        append_status(root, {"cell_id": cid, "stage": "EVAL", "status": summary["analysis_status"], **summary})
    except Exception as exc:
        mark_eval_failed(run_dir, exc)
        append_status(root, {
            "cell_id": cid,
            "stage": "EVAL",
            "status": "FAILED",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        print(f"[EVAL ERROR -> SKIP] {cid}: {exc!r}", flush=True)
        if fail_fast:
            raise
    finally:
        build_master(root)


def deferred_joint_evaluation(
    *,
    root: Path,
    joint_plan: Sequence[Dict[str, str]],
    requested_steps: int,
    eval_seeds: Sequence[int],
    max_time: int,
    fail_fast: bool,
) -> None:
    print("\n" + "#" * 120)
    print("DEFERRED JOINT EVALUATION STARTS ONLY AFTER JOINT TRAINING MATRIX FINISHED")
    print("#" * 120, flush=True)

    for i, rec in enumerate(joint_plan, start=1):
        env_key = rec["env_key"]
        method_id = rec["method_id"]
        cid = cell_id(env_key, method_id)
        run_dir = root / cid
        print(f"[JOINT EVAL {i:02d}/{len(joint_plan):02d}] {cid}", flush=True)
        if not training_complete(run_dir, requested_steps):
            append_status(root, {"cell_id": cid, "stage": "JOINT_EVAL", "status": "SKIP_NOT_TRAINED"})
            continue
        if analysis_complete(run_dir):
            print(f"  [SKIP ANALYZED] {cid}", flush=True)
            continue
        try:
            summary = analyze_cell(
                env_key=env_key,
                method_id=method_id,
                run_dir=run_dir,
                requested_steps=requested_steps,
                eval_seeds=eval_seeds,
                max_time=max_time,
            )
            update_run_end_after_analysis(run_dir, summary)
            append_status(root, {"cell_id": cid, "stage": "JOINT_EVAL", "status": summary["analysis_status"], **summary})
        except Exception as exc:
            mark_eval_failed(run_dir, exc)
            append_status(root, {
                "cell_id": cid,
                "stage": "JOINT_EVAL",
                "status": "FAILED",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"  [JOINT EVAL ERROR -> SKIP] {cid}: {exc!r}", flush=True)
            if fail_fast:
                raise
        finally:
            build_master(root)


# =============================================================================
# Formal phase runner
# =============================================================================

def run_full_matrix(args: argparse.Namespace, root: Path, profile: SpeedProfile, device: str) -> None:
    requested_steps = int(args.timesteps)
    eval_seeds = parse_ints(args.eval_seeds)
    train_seed = int(args.train_seed)
    max_time = int(args.max_time)
    fail_fast = bool(args.fail_fast)

    # --------------------------- Phase A ---------------------------
    print("\n" + "=" * 120)
    print("PHASE A | 6 REPRESENTATIONS x 3 SINGLE-SIDE ENVIRONMENTS = 18 CELLS")
    print("=" * 120, flush=True)
    for rep in PHASE_A_REPS:
        for env_key in ("S1", "S2", "S3"):
            safe_train_and_optional_eval(
                root=root,
                env_key=env_key,
                method_id=rep,
                requested_steps=requested_steps,
                train_seed=train_seed,
                eval_seeds=eval_seeds,
                profile=profile,
                device=device,
                max_time=max_time,
                defer_eval=False,
                fail_fast=fail_fast,
            )

    top_reps = select_top_phase_a(root, 3)
    write_json(root / "phase_a_selection.json", {
        "top3": top_reps,
        "criterion": "S3 best full-completion ATT; incomplete-only methods ranked after valid methods",
    })
    print(f"PHASE A TOP3 = {top_reps}", flush=True)

    # --------------------------- Phase B ---------------------------
    print("\n" + "=" * 120)
    print("PHASE B | TOP3 REPRESENTATIONS x 4 WORLD MODELS @ S3 = 12 CELLS")
    print("=" * 120, flush=True)
    phase_b_methods: List[str] = []
    for rep in top_reps:
        for wm in WM_VARIANTS:
            mid = temporal_id(rep, wm)
            phase_b_methods.append(mid)
            safe_train_and_optional_eval(
                root=root,
                env_key="S3",
                method_id=mid,
                requested_steps=requested_steps,
                train_seed=train_seed,
                eval_seeds=eval_seeds,
                profile=profile,
                device=device,
                max_time=max_time,
                defer_eval=False,
                fail_fast=fail_fast,
            )

    # Two COCO-style wildcard cells on top-2 Phase-A representations.
    wildcard_methods: List[str] = []
    print("\n" + "=" * 120)
    print("PHASE B-WILDCARD | TOP2 x W_COCO @ S3 = 2 CELLS")
    print("=" * 120, flush=True)
    for rep in top_reps[:2]:
        mid = temporal_id(rep, WILDCARD_WM)
        wildcard_methods.append(mid)
        safe_train_and_optional_eval(
            root=root,
            env_key="S3",
            method_id=mid,
            requested_steps=requested_steps,
            train_seed=train_seed,
            eval_seeds=eval_seeds,
            profile=profile,
            device=device,
            max_time=max_time,
            defer_eval=False,
            fail_fast=fail_fast,
        )

    temporal_columns = choose_joint_temporal_columns(
        root, top_reps, phase_b_methods, wildcard_methods
    )
    write_json(root / "joint_temporal_selection.json", {
        "columns": [{"column": c, "method_id": m, **method_description(m)} for c, m in temporal_columns]
    })
    print("JOINT TEMPORAL COLUMNS:", temporal_columns, flush=True)

    # --------------------------- Phase C training ---------------------------
    joint_plan: List[Dict[str, str]] = []
    for arch in JOINT_ARCHS:
        for col, mid in temporal_columns:
            joint_plan.append({"env_key": arch, "joint_arch": JOINT_ARCH_NAMES[arch], "column": col, "method_id": mid})
    write_json(root / "joint_plan.json", joint_plan)

    print("\n" + "=" * 120)
    print("PHASE C TRAINING | 8 JOINT ARCHITECTURES x 8 TEMPORAL PIPELINES = 64 CELLS")
    print("IMPORTANT: NO JOINT CHECKPOINT EVALUATION IS RUN DURING THIS TRAINING LOOP")
    print("=" * 120, flush=True)
    for i, rec in enumerate(joint_plan, start=1):
        print(f"\n[JOINT TRAIN {i:02d}/{len(joint_plan):02d}] {rec['env_key']} x {rec['column']}={rec['method_id']}", flush=True)
        safe_train_and_optional_eval(
            root=root,
            env_key=rec["env_key"],
            method_id=rec["method_id"],
            requested_steps=requested_steps,
            train_seed=train_seed,
            eval_seeds=eval_seeds,
            profile=profile,
            device=device,
            max_time=max_time,
            defer_eval=True,
            fail_fast=fail_fast,
        )

    write_json(root / "JOINT_TRAINING_MATRIX_COMPLETE.json", {
        "status": "JOINT_TRAINING_LOOP_FINISHED",
        "n_planned_joint_cells": len(joint_plan),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    # --------------------------- Phase C deferred evaluation ---------------------------
    if not args.skip_joint_eval:
        deferred_joint_evaluation(
            root=root,
            joint_plan=joint_plan,
            requested_steps=requested_steps,
            eval_seeds=eval_seeds,
            max_time=max_time,
            fail_fast=fail_fast,
        )

    build_master(root)
    write_json(root / "RUN_COMPLETE.json", {
        "status": "COMPLETE",
        "formal_cells": 96,
        "timesteps_per_cell": requested_steps,
        "formal_requested_timesteps": 96 * requested_steps,
        "hard_guard": max_time,
        "joint_hold_removed": True,
        "joint_evaluation_deferred_until_after_training": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


# =============================================================================
# CLI / resume
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="57.6M staged literature-discovery UAM matrix")
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    p.add_argument("--eval-seeds", default=",".join(str(x) for x in DEFAULT_EVAL_SEEDS))
    p.add_argument("--max-time", type=int, default=DEFAULT_HARD_GUARD)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--profile", choices=list(PROFILES.keys()), default="P3")
    p.add_argument("--output-root", default=None)
    p.add_argument("--resume-root", default=None)
    p.add_argument("--joint-eval-only", action="store_true")
    p.add_argument("--skip-joint-eval", action="store_true")
    p.add_argument("--fail-fast", action="store_true", help="Debug only. Default formal behavior is continue-on-error.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    legacy.assert_p0_patch()

    if int(args.timesteps) <= 0 or int(args.timesteps) % CHECKPOINT_INTERVAL != 0:
        raise ValueError(f"timesteps must be positive and divisible by {CHECKPOINT_INTERVAL}")
    if int(args.max_time) <= DEMAND_HORIZON:
        raise ValueError("max-time must exceed the 300-min demand horizon")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    profile = PROFILES[args.profile]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_root:
        root = Path(args.resume_root).expanduser().resolve()
    elif args.output_root:
        root = Path(args.output_root).expanduser().resolve()
    else:
        root = (ROOT / "serial_runs" / f"uam_lit57p6m_nohold_seed{args.train_seed}_{stamp}").resolve()
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "UAM_LITERATURE_DISCOVERY_57P6M_NOHOLD_V3",
        "created": datetime.now().isoformat(timespec="seconds"),
        "timesteps_per_cell": int(args.timesteps),
        "formal_cells": 96,
        "formal_budget": 96 * int(args.timesteps),
        "phase_a_cells": 18,
        "phase_b_cells": 14,
        "phase_c_joint_cells": 64,
        "remaining_under_60m_at_600k": 2_400_000 if int(args.timesteps) == 600_000 else None,
        "train_seed": int(args.train_seed),
        "eval_seeds": parse_ints(args.eval_seeds),
        "device": device,
        "profile": asdict(profile),
        "physics": {
            "S1": "E4 turnaround-only ladder stage",
            "S2": "E6 charging with pad separation 0",
            "S3_and_joint": "E6 charging + pad separation 0.25",
            "fleet_size": FLEET_SIZE,
            "hard_guard_min": int(args.max_time),
            "demand_horizon_min": DEMAND_HORIZON,
        },
        "joint": {
            "hold_action_removed": True,
            "passenger_actions": ["V0", "V1"],
            "aircraft_actions": ["V0", "V1"],
            "action_space": "Discrete(4)",
            "evaluation_policy": "deferred until the complete 64-cell joint training loop finishes",
            "architectures": JOINT_ARCH_NAMES,
        },
        "single_eval_policy": "evaluate every checkpoint immediately after each single-side cell",
        "error_policy": "continue after every training/evaluation error unless --fail-fast",
        "phase_a_representations": list(PHASE_A_REPS),
        "phase_b_world_models": list(WM_VARIANTS),
        "wildcard_world_model": WILDCARD_WM,
        "reward": "-N_active * 1 minute",
        "gamma": GAMMA,
        "completion_rule": "full completion required for formal ATT/AWT",
    }
    write_json(root / "experiment_manifest.json", manifest)

    print("=" * 120)
    print("UAM 57.6M LITERATURE-DISCOVERY MATRIX V3")
    print(f"root={root}")
    print(f"timesteps/cell={args.timesteps:,} | formal budget={96*int(args.timesteps):,}")
    print(f"device={device} | profile={profile.name} | max_time={args.max_time}")
    print("JOINT HOLD REMOVED | joint action space = Discrete(4)")
    print("DEFAULT ERROR POLICY = SKIP FAILED CELL AND CONTINUE")
    print("=" * 120, flush=True)

    if args.joint_eval_only:
        p = root / "joint_plan.json"
        if not p.exists():
            raise FileNotFoundError(f"joint_plan.json not found: {p}")
        joint_plan = json.loads(p.read_text(encoding="utf-8"))
        deferred_joint_evaluation(
            root=root,
            joint_plan=joint_plan,
            requested_steps=int(args.timesteps),
            eval_seeds=parse_ints(args.eval_seeds),
            max_time=int(args.max_time),
            fail_fast=bool(args.fail_fast),
        )
        build_master(root)
        return 0

    run_full_matrix(args, root, profile, device)
    print("\nDONE")
    print(f"Master: {root / 'matrix_master.csv'}")
    print(f"Status: {root / 'serial_status.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
