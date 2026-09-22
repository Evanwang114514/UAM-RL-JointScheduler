# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import gc
import inspect
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

# Existing validated project plumbing.
import train_uagmc_6x6_800k as old
import train_uagmc_E0_E2_E6_effect_time_700k as core
import train_uagmc_E3_E6_obs_topology_matrix_800k_FORMAL as mx
import train_uagmc_E3_E4_E5_serial_1m as base
import run_uagmc_E0_E2_E6_rule_baselines as rulebase

from at_obj.scenario import Scenario
from at_obj.vertiport.vertiport_spec import VERTIPORT_CHARGE_RATE
import at_obj.evtol.evtol_builder as evtol_builder
from utilss.uam_rl_wrapper import UAMRLWrapper


# =============================================================================
# Fixed formal experiment definition
# =============================================================================

ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "train_data" / "passengers_300.csv"

ENVIRONMENTS = ("S1", "S2", "S3", "J0", "J1", "J2", "J3")
METHODS = tuple(f"M{i}" for i in range(12))

METHOD_NAMES = {
    "M0": "CURRENT",
    "M1": "SHARED",
    "M2": "AC",
    "M3": "SHARED_AC",
    "M4": "WM",
    "M5": "SHARED_WM",
    "M6": "AC_WM",
    "M7": "SHARED_AC_WM",
    "M8": "TSDM",
    "M9": "SHARED_TSDM",
    "M10": "SHARED_AC_TSDM",
    "M11": "FULL_SHARED_AC_WM_TSDM",
}

METHOD_FLAGS = {
    "M0":  (False, False, False, False),
    "M1":  (True,  False, False, False),
    "M2":  (False, True,  False, False),
    "M3":  (True,  True,  False, False),
    "M4":  (False, False, True,  False),
    "M5":  (True,  False, True,  False),
    "M6":  (False, True,  True,  False),
    "M7":  (True,  True,  True,  False),
    "M8":  (False, False, False, True),
    "M9":  (True,  False, False, True),
    "M10": (True,  True,  False, True),
    "M11": (True,  True,  True,  True),
}

DEFAULT_TIMESTEPS = 600_000
CHECKPOINT_INTERVAL = 50_000
TRAIN_SEED = 1
DEFAULT_EVAL_SEEDS = (123, 124, 125)

TOPOLOGY = "T2"
CANDIDATES = (0, 1)
DESTINATION = 2
FLEET_SIZE = 40
DEMAND_HORIZON = 300
HARD_GUARD = 10_000

# One common physical parameterization across the 3-stage ladder.
TURNAROUND_MIN = 1.0
CHARGE_RATE_SCALE = 1.25
CHARGER_CAPACITY = 5
PAD_SEPARATION_MIN = 0.25
BASE_CHARGE_RATES = {"0": 10.64, "1": 5.32, "2": 5.32}

NUM_FRAMES = 6
SINGLE_FRAME_DIM = 4 + 8 * len(CANDIDATES)
BASE_OBS_DIM = NUM_FRAMES * SINGLE_FRAME_DIM
RESOURCE_DIM = 8 * len(CANDIDATES) + 4
FOCAL_DIM = 4
SHARED_DIM = 10 * len(CANDIDATES)
AC_DIM = 10 * len(CANDIDATES)
SEQ_DIM = 24  # 6 joint pairs x [access, demand_after, supply_after, pressure_after]
TSDM_FEATURES_PER_ACTION = 6

GAMMA = 1.0
GAE_LAMBDA = float(base.GAE_LAMBDA)
CLIP_RANGE = float(base.CLIP_RANGE)
ENT_COEF = float(base.ENT_COEF)
VF_COEF = float(base.VF_COEF)
MAX_GRAD_NORM = float(base.MAX_GRAD_NORM)
INITIAL_LR = float(base.INITIAL_LR)
N_EPOCHS = int(base.N_EPOCHS)

WM_ENSEMBLE = 3
WM_AUX_LR = 1e-3
WM_AUX_EPOCHS = 2
WM_AUX_BATCH = 2048


@dataclass(frozen=True)
class EnvSpec:
    key: str
    physical_stage: str
    pad_separation: float
    charger_capacity: int
    joint: bool
    joint_mode: str
    sequential_commit: bool

    @property
    def n_actions(self) -> int:
        return 6 if self.joint else 2


ENV_SPECS = {
    # Passenger-only: aircraft reposition remains the same LQ rule.
    "S1": EnvSpec("S1", "E4", 0.0, CHARGER_CAPACITY, False, "SINGLE", False),
    # E6 with zero pad separation = turnaround + finite charging, no effective TLOF constraint.
    "S2": EnvSpec("S2", "E6", 0.0, CHARGER_CAPACITY, False, "SINGLE", False),
    "S3": EnvSpec("S3", "E6", PAD_SEPARATION_MIN, CHARGER_CAPACITY, False, "SINGLE", False),

    # Same S3 physics; only the passenger-aircraft coupling structure changes.
    "J0": EnvSpec("J0", "E6", PAD_SEPARATION_MIN, CHARGER_CAPACITY, True, "SEPARATE_PARALLEL", False),
    "J1": EnvSpec("J1", "E6", PAD_SEPARATION_MIN, CHARGER_CAPACITY, True, "PAIRWISE_PARALLEL", False),
    "J2": EnvSpec("J2", "E6", PAD_SEPARATION_MIN, CHARGER_CAPACITY, True, "SEPARATE_SEQUENTIAL", True),
    "J3": EnvSpec("J3", "E6", PAD_SEPARATION_MIN, CHARGER_CAPACITY, True, "PAIRWISE_SEQUENTIAL", True),
}


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
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


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


def parse_names(text: str, allowed: Sequence[str]) -> List[str]:
    xs = [x.strip().upper() for x in str(text).split(",") if x.strip()]
    bad = [x for x in xs if x not in allowed]
    if bad:
        raise ValueError(f"unsupported={bad}; allowed={list(allowed)}")
    return xs


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def active_person_count(scenario: Any) -> int:
    persons_obj = getattr(scenario, "persons", None)
    persons = getattr(persons_obj, "persons", {}) if persons_obj is not None else {}
    return sum(
        1
        for p in persons.values()
        if str(getattr(p, "state", "")).lower() != "finished"
    )


def assert_p0_patch() -> None:
    step_src = inspect.getsource(Scenario.step)
    reset_src = inspect.getsource(Scenario.reset)
    wrap_src = inspect.getsource(UAMRLWrapper.reset)

    missing = []
    if "all_finished" not in step_src or "hard_guard_hit" not in step_src:
        missing.append("completion-based termination")
    if "active_count_start" not in step_src:
        missing.append("reward=-N_active")
    if "person_travel_records = {}" not in reset_src:
        missing.append("reset travel-record clearing")
    if "random.seed(int(seed))" not in reset_src:
        missing.append("Scenario Python RNG seed")
    if "self.env.reset(seed=seed)" not in wrap_src:
        missing.append("UAMRLWrapper seed propagation")

    if missing:
        raise RuntimeError(
            "P0 source fix is not active: "
            + ", ".join(missing)
            + ". Run: python apply_p0_completion_fix_v2.py"
        )


def configure_worker_physics() -> None:
    # Restore absolute values every worker/cell; never compound scaling.
    for vid, base_rate in BASE_CHARGE_RATES.items():
        value = float(base_rate) * CHARGE_RATE_SCALE
        VERTIPORT_CHARGE_RATE[str(vid)] = value
        evtol_builder.VERTIPORT_CHARGE_RATE[str(vid)] = value

    # The stage installers read these globals.
    base.TURNAROUND_DELAY_MIN = TURNAROUND_MIN
    core.TURNAROUND_DELAY_MIN = TURNAROUND_MIN
    if hasattr(mx, "TURNAROUND_DELAY_MIN"):
        mx.TURNAROUND_DELAY_MIN = TURNAROUND_MIN

    for mod in (base, core, mx, old):
        if hasattr(mod, "MAX_TIME"):
            setattr(mod, "MAX_TIME", HARD_GUARD)


# =============================================================================
# Joint aircraft control
# =============================================================================

def _noop_reposition(self) -> None:
    return None


class JointControlWrapper(gym.Wrapper):
    # Joint action index:
    #   pair = passenger_action * 3 + aircraft_action
    #   passenger_action: 0=V0, 1=V1
    #   aircraft_action : 0=HOLD, 1=V0, 2=V1
    #
    # J2/J3 perform the selected passenger commitment before aircraft dispatch,
    # without advancing the simulator clock. The subsequent wrapped step sees
    # the old passenger action again, but Scenario ignores it because that pid
    # has already been removed from waiting_decisions.

    def __init__(self, env: gym.Env, env_key: str):
        super().__init__(env)
        self.env_key = str(env_key).upper()
        self.env_spec = ENV_SPECS[self.env_key]
        if not self.env_spec.joint:
            raise ValueError(self.env_key)

        self.action_space = spaces.Discrete(6)
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

    def _dispatch_aircraft(self, aircraft_action: int) -> Tuple[bool, Optional[int]]:
        a = int(aircraft_action)
        if a == 0:
            return False, None

        target = int(CANDIDATES[a - 1])
        sc = self._scenario()
        hub = str(DESTINATION)

        local = list(sc.vertiports.evtols_at_vertiport.get(hub, []) or [])
        available = [
            e
            for e in local
            if (
                mx.state_name(e) == "IDLE"
                and not mx.passenger_ids(e)
                and not base.is_turnaround_busy(e)
            )
        ]
        available.sort(key=lambda e: str(getattr(e, "id", "")))

        if not available:
            return False, target

        evtol = available[0]
        ok = bool(
            base._start_empty_reposition_checked(
                scenario=sc,
                evtol=evtol,
                origin=hub,
                destination=str(target),
            )
        )
        return ok, target

    def step(self, action):
        ai = int(np.asarray(action).reshape(-1)[0])
        passenger_action = ai // 3
        aircraft_action = ai % 3

        committed = False
        dispatched = False
        target = None

        if self.env_spec.sequential_commit:
            # Passenger first: real commitment, same simulator time.
            committed = self._commit_passenger_without_time_advance(passenger_action)
            dispatched, target = self._dispatch_aircraft(aircraft_action)
        else:
            # Parallel baseline: aircraft acts on pre-commitment physical state.
            dispatched, target = self._dispatch_aircraft(aircraft_action)

        out = self.env.step(passenger_action)

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info)
            info.update({
                "joint_action": ai,
                "passenger_action": passenger_action,
                "aircraft_action": aircraft_action,
                "aircraft_target": target,
                "aircraft_dispatched": bool(dispatched),
                "passenger_precommitted": bool(committed),
            })
            return obs, reward, terminated, truncated, info

        obs, reward, done, info = out
        info = dict(info)
        info.update({
            "joint_action": ai,
            "passenger_action": passenger_action,
            "aircraft_action": aircraft_action,
            "aircraft_target": target,
            "aircraft_dispatched": bool(dispatched),
            "passenger_precommitted": bool(committed),
        })
        return obs, reward, done, info


# =============================================================================
# Temporal representation features
# =============================================================================

def _focal_person(env: Any, scenario: Any) -> Optional[Any]:
    return old._focal_person(env, scenario)


def _access_time(scenario: Any, person: Optional[Any], vid: int) -> float:
    return old._access_time(scenario, person, int(vid))


def _focal_od(person: Optional[Any]) -> List[float]:
    if person is None:
        return [0.0] * 4
    return [
        float(person.origin_position[0]),
        float(person.origin_position[1]),
        float(person.destination_position[0]),
        float(person.destination_position[1]),
    ]


def _resource_vector(scenario: Any, spec: EnvSpec) -> List[float]:
    out: List[float] = []
    is_e6 = spec.physical_stage == "E6"

    for vid in CANDIDATES:
        out.extend(
            float(x)
            for x in mx._resource_features_for_vid(
                scenario,
                int(vid),
                e6=is_e6,
                charger_capacity=int(spec.charger_capacity),
            )
        )

    hub = mx._resource_features_for_vid(
        scenario,
        DESTINATION,
        e6=is_e6,
        charger_capacity=int(spec.charger_capacity),
    )
    # Same common hub summary used in the validated augmented observation.
    out.extend([float(hub[0]), float(hub[2]), float(hub[4]), float(hub[5])])

    if len(out) != RESOURCE_DIM:
        raise RuntimeError(f"resource dim {len(out)} != {RESOURCE_DIM}")
    return out


def _project(
    scenario: Any,
    spec: EnvSpec,
    vid: int,
    horizon: float,
) -> List[float]:
    return [
        float(x)
        for x in old._project_candidate(
            scenario,
            stage=spec.physical_stage,
            vid=int(vid),
            horizon=float(horizon),
            charger_capacity=int(spec.charger_capacity),
        )
    ]


def _shared_and_ac(
    scenario: Any,
    spec: EnvSpec,
    person: Optional[Any],
) -> Tuple[List[float], List[float]]:
    access = {
        int(v): max(0.0, _access_time(scenario, person, int(v)))
        for v in CANDIDATES
    }
    shared_h = fmean(access.values()) if person is not None else 0.0

    shared_features: List[float] = []
    ac_features: List[float] = []

    for vid in CANDIDATES:
        p0 = _project(scenario, spec, int(vid), 0.0)
        ps = _project(scenario, spec, int(vid), shared_h)
        po = _project(scenario, spec, int(vid), access[int(vid)])

        shared_features.extend(ps)

        # AC is independently defined as candidate own-effect-time displacement
        # from the current projection. It therefore remains meaningful without
        # Shared and forms a clean binary factor in M0-M3.
        ac_features.extend(
            float(b - a)
            for a, b in zip(p0, po)
        )

    return shared_features, ac_features


def _rule_stage(spec: EnvSpec) -> str:
    return "E4" if spec.physical_stage == "E4" else "E6"


def _hub_added_supply_ready(
    scenario: Any,
    spec: EnvSpec,
    target_vid: int,
) -> float:
    hub = str(DESTINATION)
    local = list(scenario.vertiports.evtols_at_vertiport.get(hub, []) or [])
    candidates = [
        e
        for e in local
        if (
            mx.state_name(e) == "IDLE"
            and not mx.passenger_ids(e)
            and not base.is_turnaround_busy(e)
        )
    ]
    candidates.sort(key=lambda e: str(getattr(e, "id", "")))
    if not candidates:
        return float("inf")

    e = candidates[0]
    distance = float(
        scenario._distance_between_vertiports(hub, str(int(target_vid)))
    )
    speed = max(1e-6, float(e.spec.max_speed))
    flight = distance / speed * 60.0

    energy = distance * float(e.spec.energy_consumption_kwh_per_km)
    battery_after = float(e.battery_kwh) - energy
    charge_need = max(0.0, float(e.spec.battery_capacity_kwh) - battery_after)
    rate = max(1e-6, float(e.spec.charge_rate_kwh_per_min))
    charge = charge_need / rate

    pad = 0.0
    if spec.pad_separation > 0:
        pad = max(
            float(mx._pad_next_free_eta(scenario, DESTINATION)),
            float(mx._pad_next_free_eta(scenario, int(target_vid))),
        )

    return max(0.0, pad) + flight + TURNAROUND_MIN + charge


def _tsdm_block(
    scenario: Any,
    spec: EnvSpec,
    person: Optional[Any],
    passenger_action: int,
    aircraft_action: int,
) -> List[float]:
    if person is None:
        return [0.0] * TSDM_FEATURES_PER_ACTION

    vid = int(CANDIDATES[int(passenger_action)])
    demand_ready = max(0.0, _access_time(scenario, person, vid))

    try:
        releases = rulebase.known_supply_release_etas(
            scenario,
            vid,
            _rule_stage(spec),
            int(spec.charger_capacity),
        )
    except Exception:
        releases = []

    supply_ready = min(releases) if releases else 60.0

    if spec.joint and int(aircraft_action) in (1, 2):
        target = int(CANDIDATES[int(aircraft_action) - 1])
        if target == vid:
            supply_ready = min(
                float(supply_ready),
                _hub_added_supply_ready(scenario, spec, target),
            )

    gap = float(supply_ready) - float(demand_ready)
    late = max(0.0, gap)
    early = max(0.0, -gap)

    proj = _project(scenario, spec, vid, demand_ready)
    waiting_after = float(proj[1]) + 1.0
    serviceable = float(proj[5])
    if spec.joint and int(aircraft_action) in (1, 2):
        target = int(CANDIDATES[int(aircraft_action) - 1])
        if target == vid:
            serviceable += 1.0

    balance = serviceable - waiting_after

    return [
        float(demand_ready),
        float(supply_ready),
        float(gap),
        float(late),
        float(early),
        float(balance),
    ]


def _all_tsdm(
    scenario: Any,
    spec: EnvSpec,
    person: Optional[Any],
) -> List[float]:
    out: List[float] = []
    if spec.joint:
        for pair in range(6):
            p = pair // 3
            a = pair % 3
            out.extend(_tsdm_block(scenario, spec, person, p, a))
    else:
        for p in range(2):
            out.extend(_tsdm_block(scenario, spec, person, p, 0))
    return out


def _sequential_virtual_features(
    scenario: Any,
    spec: EnvSpec,
    person: Optional[Any],
) -> List[float]:
    if not spec.joint or not spec.sequential_commit or person is None:
        return [0.0] * SEQ_DIM

    out: List[float] = []
    for pair in range(6):
        p = pair // 3
        a = pair % 3
        vid = int(CANDIDATES[p])
        h = max(0.0, _access_time(scenario, person, vid))
        proj = _project(scenario, spec, vid, h)

        demand_after = float(proj[1]) + 1.0
        supply_after = float(proj[5])
        if a in (1, 2) and int(CANDIDATES[a - 1]) == vid:
            supply_after += 1.0
        pressure_after = demand_after / (1.0 + supply_after)

        out.extend([h, demand_after, supply_after, pressure_after])

    if len(out) != SEQ_DIM:
        raise RuntimeError(len(out))
    return out


def make_layout(spec: EnvSpec) -> Dict[str, Any]:
    tsdm_dim = spec.n_actions * TSDM_FEATURES_PER_ACTION
    dims = {
        "base": BASE_OBS_DIM,
        "resource": RESOURCE_DIM,
        "focal": FOCAL_DIM,
        "shared": SHARED_DIM,
        "ac": AC_DIM,
        "tsdm": tsdm_dim,
        "sequential": SEQ_DIM,
    }

    slices = {}
    pos = 0
    for name in ("base", "resource", "focal", "shared", "ac", "tsdm", "sequential"):
        width = int(dims[name])
        slices[name] = (pos, pos + width)
        pos += width

    return {
        "env_key": spec.key,
        "n_actions": spec.n_actions,
        "dims": dims,
        "slices": slices,
        "total_dim": pos,
        "base_dim": BASE_OBS_DIM,
        "single_frame_dim": SINGLE_FRAME_DIM,
        "num_frames": NUM_FRAMES,
    }


class FactorObservationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, env_key: str, method: str):
        super().__init__(env)
        self.env_key = str(env_key).upper()
        self.method = str(method).upper()
        self.env_spec = ENV_SPECS[self.env_key]
        self.layout = make_layout(self.env_spec)

        got = int(np.prod(env.observation_space.shape))
        if got != BASE_OBS_DIM:
            raise RuntimeError(
                f"source observation dim {got} != expected {BASE_OBS_DIM}"
            )

        self.observation_space = spaces.Box(
            low=-1e9,
            high=1e9,
            shape=(int(self.layout["total_dim"]),),
            dtype=np.float32,
        )

        self._last_tsdm: List[float] = [0.0] * int(
            self.layout["dims"]["tsdm"]
        )

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        scenario = mx.find_scenario(self.env)
        person = _focal_person(self.env, scenario)

        resource = _resource_vector(scenario, self.env_spec)
        focal = _focal_od(person)
        shared, ac = _shared_and_ac(scenario, self.env_spec, person)
        tsdm = _all_tsdm(scenario, self.env_spec, person)
        seq = _sequential_virtual_features(scenario, self.env_spec, person)

        self._last_tsdm = list(tsdm)

        out = np.concatenate(
            [
                np.asarray(obs, dtype=np.float32).reshape(-1),
                np.asarray(resource, dtype=np.float32),
                np.asarray(focal, dtype=np.float32),
                np.asarray(shared, dtype=np.float32),
                np.asarray(ac, dtype=np.float32),
                np.asarray(tsdm, dtype=np.float32),
                np.asarray(seq, dtype=np.float32),
            ],
            axis=0,
        )

        if out.shape != (int(self.layout["total_dim"]),):
            raise RuntimeError(
                f"obs shape {out.shape} != {(self.layout['total_dim'],)}"
            )
        return out.astype(np.float32, copy=False)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out
            return self._transform(obs), info
        return self._transform(out)

    def step(self, action):
        ai = int(np.asarray(action).reshape(-1)[0])
        block = self._last_tsdm[
            ai * TSDM_FEATURES_PER_ACTION:
            (ai + 1) * TSDM_FEATURES_PER_ACTION
        ]
        selected_gap = float(block[2]) if len(block) >= 3 else 0.0

        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info)
            info.update({
                "selected_supply_demand_gap": selected_gap,
                "selected_late_supply": max(0.0, selected_gap),
                "selected_early_supply": max(0.0, -selected_gap),
            })
            return self._transform(obs), reward, terminated, truncated, info

        obs, reward, done, info = out
        info = dict(info)
        info.update({
            "selected_supply_demand_gap": selected_gap,
            "selected_late_supply": max(0.0, selected_gap),
            "selected_early_supply": max(0.0, -selected_gap),
        })
        return self._transform(obs), reward, done, info


# =============================================================================
# Fixed-capacity factorial feature extractor + auxiliary residual world model
# =============================================================================

class FactorialExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        layout: Optional[Dict[str, Any]] = None,
        method: str = "M0",
        env_key: str = "S1",
    ):
        if layout is None:
            raise ValueError("layout required")

        super().__init__(observation_space, features_dim)
        self.layout = dict(layout)
        self.method = str(method).upper()
        self.env_key = str(env_key).upper()
        self.flags = METHOD_FLAGS[self.method]
        self.use_shared, self.use_ac, self.use_wm, self.use_tsdm = self.flags
        self.use_sequential = self.env_key in ("J2", "J3")

        self.slices = dict(layout["slices"])
        self.n_actions = int(layout["n_actions"])

        self.frame_encoder = nn.Sequential(
            nn.Linear(SINGLE_FRAME_DIM, 128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        self.resource_branch = nn.Sequential(
            nn.Linear(RESOURCE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.focal_branch = nn.Sequential(
            nn.Linear(FOCAL_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.shared_branch = nn.Sequential(
            nn.Linear(SHARED_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.ac_branch = nn.Sequential(
            nn.Linear(AC_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.tsdm_branch = nn.Sequential(
            nn.Linear(int(layout["dims"]["tsdm"]), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.sequential_branch = nn.Sequential(
            nn.Linear(SEQ_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # World model ensemble predicts next normalized resource vector:
        #   r_{t+1} = r_t + delta_theta(r_t, action)
        self.world_models = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(RESOURCE_DIM + self.n_actions, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, RESOURCE_DIM),
                )
                for _ in range(WM_ENSEMBLE)
            ]
        )
        self.wm_project = nn.Sequential(
            nn.Linear(self.n_actions * RESOURCE_DIM * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        concat_dim = 128 + 64 + 32 + 32 + 32 + 64 + 32 + 32
        self.output_proj = nn.Sequential(
            nn.Linear(concat_dim, int(features_dim)),
            nn.ReLU(),
        )

    def _slice(self, obs: torch.Tensor, name: str) -> torch.Tensor:
        lo, hi = self.slices[name]
        return obs[:, int(lo):int(hi)]

    def _zeros(self, obs: torch.Tensor, width: int) -> torch.Tensor:
        return torch.zeros(
            (obs.shape[0], int(width)),
            dtype=obs.dtype,
            device=obs.device,
        )

    def _world_predictions(
        self,
        resource: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        per_action_mean = []
        per_action_std = []

        for action in range(self.n_actions):
            onehot = F.one_hot(
                torch.full(
                    (resource.shape[0],),
                    action,
                    device=resource.device,
                    dtype=torch.long,
                ),
                num_classes=self.n_actions,
            ).float()
            inp = torch.cat([resource, onehot], dim=1)

            preds = torch.stack(
                [resource + model(inp) for model in self.world_models],
                dim=0,
            )
            per_action_mean.append(preds.mean(dim=0))
            per_action_std.append(preds.std(dim=0, unbiased=False))

        mean = torch.cat(per_action_mean, dim=1)
        std = torch.cat(per_action_std, dim=1)
        return mean, std

    def world_model_parameters(self):
        return self.world_models.parameters()

    def world_model_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        resource = self._slice(obs, "resource")
        target = self._slice(next_obs, "resource").detach()
        a = actions.long().flatten()
        onehot = F.one_hot(a, num_classes=self.n_actions).float()
        inp = torch.cat([resource, onehot], dim=1)

        losses = []
        for model in self.world_models:
            pred = resource + model(inp)

            # Lightweight bootstrap mask keeps ensemble members different.
            mask = (torch.rand(pred.shape[0], device=pred.device) < 0.8).float()
            mse = ((pred - target) ** 2).mean(dim=1)
            loss = (mse * mask).sum() / mask.sum().clamp_min(1.0)
            losses.append(loss)

        return torch.stack(losses).mean()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations.float()

        base_obs = self._slice(obs, "base")
        B = base_obs.shape[0]
        x = base_obs.reshape(B, NUM_FRAMES, SINGLE_FRAME_DIM)
        x = self.frame_encoder(x)
        x, _ = self.lstm(x)
        base_latent = x[:, -1, :]

        resource = self._slice(obs, "resource")
        resource_latent = self.resource_branch(resource)
        focal_latent = self.focal_branch(self._slice(obs, "focal"))

        if self.use_shared:
            shared_latent = self.shared_branch(self._slice(obs, "shared"))
        else:
            shared_latent = self._zeros(obs, 32)

        if self.use_ac:
            ac_latent = self.ac_branch(self._slice(obs, "ac"))
        else:
            ac_latent = self._zeros(obs, 32)

        if self.use_wm:
            mean, std = self._world_predictions(resource)
            wm_latent = self.wm_project(torch.cat([mean, std], dim=1))
        else:
            wm_latent = self._zeros(obs, 64)

        if self.use_tsdm:
            tsdm_latent = self.tsdm_branch(self._slice(obs, "tsdm"))
        else:
            tsdm_latent = self._zeros(obs, 32)

        if self.use_sequential:
            seq_latent = self.sequential_branch(self._slice(obs, "sequential"))
        else:
            seq_latent = self._zeros(obs, 32)

        pieces = [
            base_latent,
            resource_latent,
            focal_latent,
            shared_latent,
            ac_latent,
            wm_latent,
            tsdm_latent,
            seq_latent,
        ]
        return self.output_proj(torch.cat(pieces, dim=1))


class WorldModelPPO(PPO):
    # PPO remains the optimizer. This subclass only adds a supervised
    # one-step residual-dynamics update after each PPO update.

    def _train_world_model(self) -> Optional[float]:
        ext = getattr(self.policy, "features_extractor", None)
        if not isinstance(ext, FactorialExtractor) or not ext.use_wm:
            return None

        # IMPORTANT:
        # PPO flattens RolloutBuffer when super().train() calls get().
        # The auxiliary dynamics update must consume the rollout BEFORE that.
        obs_np = np.asarray(self.rollout_buffer.observations)
        act_np = np.asarray(self.rollout_buffer.actions)
        starts_np = np.asarray(self.rollout_buffer.episode_starts)

        # Fresh rollout layout should be:
        # observations   [n_steps, n_envs, obs_dim]
        # actions        [n_steps, n_envs, action_dim] or [n_steps, n_envs]
        # episode_starts [n_steps, n_envs]
        if obs_np.ndim < 3:
            raise RuntimeError(
                "WorldModelPPO received an already-flattened rollout buffer: "
                f"observations.shape={obs_np.shape}. "
                "WM update must run before super().train()."
            )

        if obs_np.shape[0] < 2:
            return None

        n_steps, n_envs = obs_np.shape[:2]

        if starts_np.shape[0] != n_steps or starts_np.shape[1] != n_envs:
            raise RuntimeError(
                "RolloutBuffer shape mismatch: "
                f"obs={obs_np.shape}, starts={starts_np.shape}"
            )

        # Same-environment t -> t+1 transitions only.
        cur = obs_np[:-1].reshape(
            (n_steps - 1) * n_envs, *obs_np.shape[2:]
        )
        nxt = obs_np[1:].reshape(
            (n_steps - 1) * n_envs, *obs_np.shape[2:]
        )

        if act_np.ndim == 2:
            act = act_np[:-1].reshape((n_steps - 1) * n_envs, 1)
        else:
            act = act_np[:-1].reshape(
                (n_steps - 1) * n_envs, *act_np.shape[2:]
            )

        # If episode_starts[t+1] == 1, reset happened between t and t+1.
        valid = starts_np[1:].reshape(-1) < 0.5

        if cur.shape[0] != valid.shape[0]:
            raise RuntimeError(
                "WM transition/mask mismatch: "
                f"cur={cur.shape}, valid={valid.shape}"
            )

        if not np.any(valid):
            return None

        cur = torch.as_tensor(
            cur[valid], device=self.device, dtype=torch.float32
        )
        nxt = torch.as_tensor(
            nxt[valid], device=self.device, dtype=torch.float32
        )
        act = torch.as_tensor(
            act[valid], device=self.device
        )

        if not hasattr(self, "_wm_optimizer"):
            self._wm_optimizer = torch.optim.Adam(
                list(ext.world_model_parameters()),
                lr=WM_AUX_LR,
            )

        losses = []
        n = cur.shape[0]
        for _ in range(WM_AUX_EPOCHS):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, WM_AUX_BATCH):
                idx = order[start:start + WM_AUX_BATCH]
                loss = ext.world_model_loss(cur[idx], act[idx], nxt[idx])

                self._wm_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(ext.world_model_parameters()),
                    5.0,
                )
                self._wm_optimizer.step()
                losses.append(float(loss.detach().cpu()))

        return float(np.mean(losses)) if losses else None

    def train(self) -> None:
        # WM first, while rollout buffer still preserves temporal/env axes.
        wm_loss = self._train_world_model()

        # PPO second; this call may flatten the rollout buffer internally.
        super().train()

        if wm_loss is not None:
            self.logger.record("train/world_model_aux_loss", wm_loss)


# =============================================================================
# Structured joint action heads
# =============================================================================

class StructuredJointPolicy(ActorCriticPolicy):
    # All four variants use Discrete(6) externally.
    #
    # J0: independent passenger and aircraft logits -> additive joint logits.
    # J1: direct pairwise 6-action scorer.
    # J2: passenger logits + aircraft logits conditioned on passenger action.
    # J3: passenger logits + shared semantic pair scorer.
    #
    # The small hidden widths are chosen so head parameter counts remain in the
    # same order of magnitude. Exact counts are written to every run manifest.

    def __init__(self, *args, joint_mode: str = "J0", **kwargs):
        self.joint_mode = str(joint_mode).upper()

        lr_schedule = kwargs.get("lr_schedule", None)
        if lr_schedule is None and len(args) >= 3:
            lr_schedule = args[2]

        super().__init__(*args, **kwargs)

        d = int(self.mlp_extractor.latent_dim_pi)

        # Remove SB3's default categorical action layer.
        self.action_net = nn.Identity()

        if self.joint_mode == "J0":
            self.passenger_head = nn.Sequential(
                nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 2)
            )
            self.aircraft_head = nn.Sequential(
                nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 3)
            )

        elif self.joint_mode == "J1":
            self.pair_head = nn.Sequential(
                nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 6)
            )

        elif self.joint_mode == "J2":
            self.passenger_head = nn.Sequential(
                nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 2)
            )
            self.conditional_aircraft = nn.Sequential(
                nn.Linear(d + 2, 32), nn.ReLU(), nn.Linear(32, 3)
            )

        elif self.joint_mode == "J3":
            self.passenger_head = nn.Sequential(
                nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 2)
            )
            self.shared_pair_scorer = nn.Sequential(
                nn.Linear(d + 2 + 3, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        else:
            raise ValueError(self.joint_mode)

        if lr_schedule is None:
            raise RuntimeError("Cannot recover SB3 learning-rate schedule.")

        # Rebuild optimizer so it contains the replacement action heads.
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _joint_logits(self, latent_pi: torch.Tensor) -> torch.Tensor:
        B = latent_pi.shape[0]

        if self.joint_mode == "J0":
            p = self.passenger_head(latent_pi)         # Bx2
            a = self.aircraft_head(latent_pi)          # Bx3
            return (p.unsqueeze(2) + a.unsqueeze(1)).reshape(B, 6)

        if self.joint_mode == "J1":
            return self.pair_head(latent_pi)

        if self.joint_mode == "J2":
            p_logits = self.passenger_head(latent_pi)
            p_logp = F.log_softmax(p_logits, dim=1)
            rows = []
            for p in range(2):
                onehot = F.one_hot(
                    torch.full(
                        (B,), p, device=latent_pi.device, dtype=torch.long
                    ),
                    num_classes=2,
                ).float()
                a_logits = self.conditional_aircraft(
                    torch.cat([latent_pi, onehot], dim=1)
                )
                a_logp = F.log_softmax(a_logits, dim=1)
                rows.append(p_logp[:, p:p + 1] + a_logp)
            return torch.stack(rows, dim=1).reshape(B, 6)

        # J3: shared scorer over all passenger-aircraft semantic pairs.
        p_logits = self.passenger_head(latent_pi)
        p_logp = F.log_softmax(p_logits, dim=1)

        pair_scores = []
        for p in range(2):
            row = []
            p_oh = F.one_hot(
                torch.full((B,), p, device=latent_pi.device, dtype=torch.long),
                num_classes=2,
            ).float()
            for a in range(3):
                a_oh = F.one_hot(
                    torch.full((B,), a, device=latent_pi.device, dtype=torch.long),
                    num_classes=3,
                ).float()
                score = self.shared_pair_scorer(
                    torch.cat([latent_pi, p_oh, a_oh], dim=1)
                )
                row.append(score)
            pair_scores.append(torch.cat(row, dim=1))

        scores = torch.stack(pair_scores, dim=1)  # Bx2x3
        a_logp = F.log_softmax(scores, dim=2)
        return (p_logp.unsqueeze(2) + a_logp).reshape(B, 6)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        logits = self._joint_logits(latent_pi)
        return self.action_dist.proba_distribution(action_logits=logits)


# =============================================================================
# Environment/model construction
# =============================================================================

def make_env_factory(
    *,
    env_key: str,
    method: str,
    env_index: int,
    run_dir: Path,
):
    spec = ENV_SPECS[str(env_key).upper()]
    method = str(method).upper()

    def _init():
        configure_worker_physics()

        env = core.make_experiment_env_factory(
            stage=spec.physical_stage,
            topology=TOPOLOGY,
            encoder_mode="uagmc",
            fleet_size=FLEET_SIZE,
            env_index=int(env_index),
            run_dir=run_dir,
            pad_separation=float(spec.pad_separation),
            charger_capacity=int(spec.charger_capacity),
            max_time=HARD_GUARD,
        )()

        if spec.joint:
            env = JointControlWrapper(env, spec.key)

        env = FactorObservationWrapper(
            env,
            env_key=spec.key,
            method=method,
        )
        return env

    return _init


def build_vec_env(
    *,
    env_key: str,
    method: str,
    profile: SpeedProfile,
    seed: int,
    run_dir: Path,
) -> VecNormalize:
    factories = [
        make_env_factory(
            env_key=env_key,
            method=method,
            env_index=i,
            run_dir=run_dir,
        )
        for i in range(profile.n_envs)
    ]
    raw = SubprocVecEnv(factories, start_method="spawn")
    raw.seed(int(seed))
    return VecNormalize(
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
    method: str,
    run_dir: Path,
) -> DummyVecEnv:
    return DummyVecEnv(
        [
            make_env_factory(
                env_key=env_key,
                method=method,
                env_index=9999,
                run_dir=run_dir,
            )
        ]
    )


def build_model(
    *,
    env: VecNormalize,
    env_key: str,
    method: str,
    profile: SpeedProfile,
    seed: int,
    run_dir: Path,
    device: str,
):
    spec = ENV_SPECS[env_key]
    layout = make_layout(spec)
    use_wm = METHOD_FLAGS[method][2]

    policy_kwargs = dict(
        features_extractor_class=FactorialExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            layout=layout,
            method=method,
            env_key=env_key,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    policy: Any = "MlpPolicy"
    if spec.joint:
        policy = StructuredJointPolicy
        policy_kwargs["joint_mode"] = env_key

    algo_cls = WorldModelPPO if use_wm else PPO

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
# Training audit callback
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


def build_callbacks(
    run_dir: Path,
    profile: SpeedProfile,
) -> List[BaseCallback]:
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
    method: str,
    model_path: Path,
    vec_path: Path,
    train_step: int,
    eval_seed: int,
    run_dir: Path,
) -> Dict[str, Any]:
    seed_all(eval_seed)

    raw = build_eval_env(
        env_key=env_key,
        method=method,
        run_dir=run_dir / "_eval_monitor",
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
        system_person_minutes = 0.0
        episode_steps = 0
        terminal_snapshot = None

        gap_abs = []
        gap_late = []
        gap_early = []

        while not bool(done[0]):
            scenario = mx.find_scenario(env)
            system_person_minutes += active_person_count(scenario)

            probs = policy_probs(model, obs)
            action, _ = model.predict(obs, deterministic=True)
            ai = int(np.asarray(action).reshape(-1)[0])
            action_counts[ai] += 1
            prob_rows.append(probs.copy())

            obs, reward, done, infos = env.step(action)
            reward_sum += fnum(np.asarray(reward).reshape(-1)[0], 0.0)
            episode_steps += 1

            info = infos[0] if infos else {}
            if isinstance(info, dict):
                if "terminal_snapshot" in info:
                    terminal_snapshot = info["terminal_snapshot"]
                g = fnum(info.get("selected_supply_demand_gap"))
                if math.isfinite(g):
                    gap_abs.append(abs(g))
                    gap_late.append(max(0.0, g))
                    gap_early.append(max(0.0, -g))

            if episode_steps > HARD_GUARD + 5:
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
            "method": method,
            "method_name": METHOD_NAMES[method],
            "train_step": int(train_step),
            "eval_seed": int(eval_seed),
            "valid_full_completion": bool(valid),
            "mean_abs_supply_demand_gap": fmean(gap_abs),
            "mean_late_supply": fmean(gap_late),
            "mean_early_supply": fmean(gap_early),
            "late_supply_rate": (
                float(np.mean(np.asarray(gap_late) > 1e-9))
                if gap_late else float("nan")
            ),
        })

        # Incomplete/hard-guard episodes are invalid formal samples.
        if not valid:
            metrics["ATT"] = float("nan")
            metrics["AWT"] = float("nan")

        return metrics

    finally:
        try:
            env.close()
        except Exception:
            pass
        core.restore_process_patches()
        gc.collect()


def checkpoint_paths(run_dir: Path, step: int) -> Tuple[Path, Path]:
    model = run_dir / "checkpoints" / f"uam_ppo_{step}_steps.zip"
    vec = run_dir / "checkpoints" / f"uam_ppo_vecnormalize_{step}_steps.pkl"
    return model, vec


def analyze_cell(
    *,
    env_key: str,
    method: str,
    run_dir: Path,
    requested_steps: int,
    eval_seeds: Sequence[int],
) -> Dict[str, Any]:
    adir = run_dir / "analysis"
    adir.mkdir(parents=True, exist_ok=True)

    steps = list(
        range(CHECKPOINT_INTERVAL, requested_steps + 1, CHECKPOINT_INTERVAL)
    )
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for step in steps:
        model_path, vec_path = checkpoint_paths(run_dir, step)
        if not model_path.exists() or not vec_path.exists():
            continue

        for seed in eval_seeds:
            try:
                row = evaluate_checkpoint(
                    env_key=env_key,
                    method=method,
                    model_path=model_path,
                    vec_path=vec_path,
                    train_step=step,
                    eval_seed=int(seed),
                    run_dir=run_dir,
                )
                rows.append(row)
                print(
                    f"  [eval] {env_key}/{method} {step//1000:>3}k "
                    f"seed={seed} ATT={fnum(row.get('ATT')):.3f} "
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

    if not rows:
        raise RuntimeError(f"no checkpoint evaluations succeeded for {env_key}/{method}")

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[int(r["train_step"])].append(r)

    curve = []
    metrics = (
        "ATT", "AWT", "AGT_access", "AFT",
        "completion_rate", "travel_p90", "travel_p95",
        "system_person_minutes_per_passenger",
        "mean_abs_supply_demand_gap",
        "mean_late_supply", "mean_early_supply",
        "late_supply_rate",
    )

    for step in sorted(grouped):
        rs = grouped[step]
        valid = [r for r in rs if bool(r.get("valid_full_completion"))]
        row = {
            "env_key": env_key,
            "method": method,
            "train_step": step,
            "n_eval_seeds": len(rs),
            "n_valid": len(valid),
            "all_full_completion": len(valid) == len(rs),
        }
        for key in metrics:
            src = valid if key in ("ATT", "AWT") else rs
            row[key + "_mean"] = fmean(r.get(key) for r in src)
            row[key + "_std"] = fstd(r.get(key) for r in src)

        action_keys = sorted({
            k for r in rs for k in r.keys()
            if k.startswith("action_") and k.endswith("_share")
        })
        for k in action_keys:
            row[k + "_mean"] = fmean(r.get(k) for r in rs)

        curve.append(row)

    write_csv(adir / "checkpoint_curve.csv", curve)

    valid_curve = [
        r for r in curve
        if bool(r["all_full_completion"])
        and math.isfinite(fnum(r.get("ATT_mean")))
    ]
    if not valid_curve:
        raise RuntimeError(
            f"all evaluated checkpoints invalid/incomplete for {env_key}/{method}"
        )

    best = min(valid_curve, key=lambda r: fnum(r["ATT_mean"]))
    final_candidates = [r for r in valid_curve if int(r["train_step"]) == requested_steps]
    final = final_candidates[0] if final_candidates else valid_curve[-1]

    late = [r for r in valid_curve if int(r["train_step"]) >= max(400_000, requested_steps - 200_000)]

    summary = {
        "env_key": env_key,
        "method": method,
        "method_name": METHOD_NAMES[method],
        "best_step": int(best["train_step"]),
        "best_ATT": fnum(best.get("ATT_mean")),
        "final_step": int(final["train_step"]),
        "final_ATT": fnum(final.get("ATT_mean")),
        "late_ATT_mean": fmean(r.get("ATT_mean") for r in late),
        "late_ATT_std_across_checkpoints": fstd(r.get("ATT_mean") for r in late),
        "final_completion": fnum(final.get("completion_rate_mean")),
        "final_gap_abs": fnum(final.get("mean_abs_supply_demand_gap_mean")),
        "final_late_supply": fnum(final.get("mean_late_supply_mean")),
        "final_late_supply_rate": fnum(final.get("late_supply_rate_mean")),
    }
    write_json(adir / "cell_summary.json", summary)
    return summary


# =============================================================================
# CPU/GPU benchmark
# =============================================================================

def benchmark_compute(
    *,
    root: Path,
    profile_names: Sequence[str],
    devices: Sequence[str],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[str, str]]]:
    rows = []
    bench_root = root / "compute_benchmark"
    bench_root.mkdir(parents=True, exist_ok=True)

    for pname in profile_names:
        profile = PROFILES[pname]
        for device in devices:
            if device == "cuda" and not torch.cuda.is_available():
                continue

            run_dir = bench_root / f"{pname}_{device}"
            run_dir.mkdir(parents=True, exist_ok=True)
            env = None
            model = None

            try:
                seed_all(seed)
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                env = build_vec_env(
                    env_key="J3",
                    method="M11",
                    profile=profile,
                    seed=seed,
                    run_dir=run_dir,
                )
                model = build_model(
                    env=env,
                    env_key="J3",
                    method="M11",
                    profile=profile,
                    seed=seed,
                    run_dir=run_dir,
                    device=device,
                )

                started = time.perf_counter()
                before_steps = int(model.num_timesteps)
                model.learn(
                    total_timesteps=profile.rollout,
                    progress_bar=False,
                    reset_num_timesteps=True,
                )
                elapsed = time.perf_counter() - started
                actual = int(model.num_timesteps) - before_steps

                peak_mb = (
                    torch.cuda.max_memory_allocated() / (1024 ** 2)
                    if device == "cuda"
                    else 0.0
                )

                row = {
                    "profile": pname,
                    "profile_name": profile.name,
                    "device": device,
                    "n_envs": profile.n_envs,
                    "n_steps": profile.n_steps,
                    "batch_size": profile.batch_size,
                    "actual_steps": actual,
                    "elapsed_sec": elapsed,
                    "sps": actual / max(elapsed, 1e-9),
                    "peak_gpu_memory_mb": peak_mb,
                    "policy_params": count_params(model),
                }
                rows.append(row)
                write_csv(bench_root / "benchmark.csv", rows)

                print(
                    f"[bench] {pname}/{device}: "
                    f"{row['sps']:.1f} steps/s | {elapsed:.1f}s | "
                    f"GPU peak={peak_mb:.0f}MB",
                    flush=True,
                )

            except Exception as exc:
                rows.append({
                    "profile": pname,
                    "device": device,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "sps": float("nan"),
                })
                write_csv(bench_root / "benchmark.csv", rows)

            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                model = None
                env = None
                core.restore_process_patches()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    good = [
        r for r in rows
        if math.isfinite(fnum(r.get("sps")))
    ]
    selected = None
    if good:
        winner = max(good, key=lambda r: fnum(r["sps"]))
        selected = (str(winner["profile"]), str(winner["device"]))
        write_json(
            bench_root / "selected.json",
            {
                "profile": selected[0],
                "device": selected[1],
                "sps": fnum(winner["sps"]),
                "representative_cell": "J3/M11",
            },
        )

    return rows, selected


# =============================================================================
# Cell training / matrix runner
# =============================================================================

def cell_id(env_key: str, method: str) -> str:
    return f"{env_key}__{method}"


def cell_complete(run_dir: Path, requested_steps: int) -> bool:
    p = run_dir / "run_end.json"
    q = run_dir / "analysis" / "cell_summary.json"
    if not p.exists() or not q.exists():
        return False
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return (
            str(obj.get("status", "")).upper() == "SUCCESS"
            and int(obj.get("requested_timesteps", -1)) == int(requested_steps)
        )
    except Exception:
        return False


def train_cell(
    *,
    root: Path,
    env_key: str,
    method: str,
    requested_steps: int,
    train_seed: int,
    eval_seeds: Sequence[int],
    profile: SpeedProfile,
    device: str,
) -> Dict[str, Any]:
    cid = cell_id(env_key, method)
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
            method=method,
            profile=profile,
            seed=train_seed,
            run_dir=run_dir,
        )
        model = build_model(
            env=env,
            env_key=env_key,
            method=method,
            profile=profile,
            seed=train_seed,
            run_dir=run_dir,
            device=device,
        )

        spec = ENV_SPECS[env_key]
        manifest = {
            "cell_id": cid,
            "env": asdict(spec),
            "method": method,
            "method_name": METHOD_NAMES[method],
            "method_flags_shared_ac_wm_tsdm": METHOD_FLAGS[method],
            "requested_timesteps": requested_steps,
            "train_seed": train_seed,
            "eval_seeds": list(eval_seeds),
            "profile": asdict(profile),
            "device": str(model.device),
            "policy_params": count_params(model),
            "physics": {
                "fleet_size": FLEET_SIZE,
                "topology": TOPOLOGY,
                "demand_horizon": DEMAND_HORIZON,
                "hard_guard": HARD_GUARD,
                "turnaround_min": TURNAROUND_MIN,
                "charge_rate_scale": CHARGE_RATE_SCALE,
                "charger_capacity": spec.charger_capacity,
                "pad_separation_min": spec.pad_separation,
            },
            "ppo": {
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "clip_range": CLIP_RANGE,
                "ent_coef": ENT_COEF,
                "vf_coef": VF_COEF,
                "n_epochs": N_EPOCHS,
                "initial_lr": INITIAL_LR,
            },
            "world_model": {
                "ensemble": WM_ENSEMBLE,
                "aux_lr": WM_AUX_LR,
                "aux_epochs": WM_AUX_EPOCHS,
                "aux_target": "next normalized current-resource vector",
                "used": bool(METHOD_FLAGS[method][2]),
            },
        }
        write_json(run_dir / "run_manifest.json", manifest)

        print("\n" + "=" * 120)
        print(
            f"START {cid} | {requested_steps:,} | "
            f"{profile.name} | {device} | params={count_params(model):,}"
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
            "status": "TRAINED",
            "requested_timesteps": requested_steps,
            "actual_timesteps": int(model.num_timesteps),
            "elapsed_sec": elapsed,
            "sps": int(model.num_timesteps) / max(elapsed, 1e-9),
        }
        write_json(run_dir / "run_end.json", run_end)

        try:
            env.close()
        except Exception:
            pass
        env = None
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        summary = analyze_cell(
            env_key=env_key,
            method=method,
            run_dir=run_dir,
            requested_steps=requested_steps,
            eval_seeds=eval_seeds,
        )

        run_end["status"] = "SUCCESS"
        run_end["analysis"] = summary
        write_json(run_dir / "run_end.json", run_end)
        return run_end

    except Exception:
        write_json(
            run_dir / "run_end.json",
            {
                "status": "FAILED",
                "requested_timesteps": requested_steps,
                "error": traceback.format_exc(),
            },
        )
        raise

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        core.restore_process_patches()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def build_master(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for env_key in ENVIRONMENTS:
        for method in METHODS:
            p = root / cell_id(env_key, method) / "analysis" / "cell_summary.json"
            if not p.exists():
                continue
            obj = json.loads(p.read_text(encoding="utf-8"))
            rows.append(obj)
    write_csv(root / "matrix_master.csv", rows)
    return rows


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="7 environments x 12 methods x 600k UAM factorial matrix"
    )
    p.add_argument("--envs", default=",".join(ENVIRONMENTS))
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    p.add_argument("--eval-seeds", default=",".join(str(x) for x in DEFAULT_EVAL_SEEDS))

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--profile", choices=["auto", *PROFILES.keys()], default="auto")

    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument(
        "--benchmark-profiles",
        default="P1,P3",
        help="Default compares 8-env and 16-env profiles on both CPU/GPU.",
    )

    p.add_argument("--output-root", default=None)
    p.add_argument("--resume-root", default=None)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    assert_p0_patch()

    envs = parse_names(args.envs, ENVIRONMENTS)
    methods = parse_names(args.methods, METHODS)
    eval_seeds = parse_ints(args.eval_seeds)

    requested_steps = int(args.timesteps)
    if args.smoke:
        envs = ["S1", "S3", "J3"]
        methods = ["M0", "M3", "M11"]
        requested_steps = 50_000
        eval_seeds = [123]

    if requested_steps % CHECKPOINT_INTERVAL != 0:
        raise ValueError(
            f"timesteps must be divisible by {CHECKPOINT_INTERVAL}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_root:
        root = Path(args.resume_root).expanduser().resolve()
    elif args.output_root:
        root = Path(args.output_root).expanduser().resolve()
    else:
        root = (
            ROOT
            / "serial_runs"
            / f"uam_7x12_600k_seed{args.train_seed}_{stamp}"
        ).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Optional CPU/GPU throughput benchmark.
    # ------------------------------------------------------------------
    selected = None
    if args.benchmark or args.benchmark_only:
        profile_names = parse_names(
            args.benchmark_profiles,
            tuple(PROFILES.keys()),
        )
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        _, selected = benchmark_compute(
            root=root,
            profile_names=profile_names,
            devices=devices,
            seed=int(args.train_seed),
        )

        if args.benchmark_only:
            print(f"Benchmark done. Selected={selected}")
            return 0

    # Device/profile selection.
    if args.device == "auto":
        if selected is not None:
            device = selected[1]
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if args.profile == "auto":
        if selected is not None:
            profile = PROFILES[selected[0]]
        else:
            profile = PROFILES["P3"]
    else:
        profile = PROFILES[args.profile]

    experiment_manifest = {
        "experiment": "UAM_7ENV_12METHOD_600K",
        "created": datetime.now().isoformat(timespec="seconds"),
        "envs": envs,
        "methods": methods,
        "timesteps_per_cell": requested_steps,
        "n_cells": len(envs) * len(methods),
        "formal_budget": len(envs) * len(methods) * requested_steps,
        "train_seed": int(args.train_seed),
        "eval_seeds": eval_seeds,
        "device": device,
        "profile": asdict(profile),
        "physics": {
            "fleet_size": FLEET_SIZE,
            "turnaround_min": TURNAROUND_MIN,
            "charge_rate_scale": CHARGE_RATE_SCALE,
            "charger_capacity": CHARGER_CAPACITY,
            "pad_separation_min": PAD_SEPARATION_MIN,
            "demand_horizon": DEMAND_HORIZON,
            "hard_guard": HARD_GUARD,
        },
        "reward": "-N_active * 1 minute",
        "gamma": GAMMA,
        "completion_rule": "normal termination only after all generated passengers finish",
        "matrix_note": (
            "S1/S2/S3 isolate physical complexity under passenger-only RL + LQ aircraft; "
            "J0-J3 freeze S3 physics and change only passenger-aircraft joint structure."
        ),
    }
    write_json(root / "experiment_manifest.json", experiment_manifest)

    status_rows: List[Dict[str, Any]] = []
    total = len(envs) * len(methods)
    idx = 0

    print("=" * 120)
    print("UAM 7x12 FACTORIAL MATRIX")
    print(f"cells={total} | budget={total*requested_steps:,} timesteps")
    print(f"profile={profile.name} | device={device}")
    print(f"output={root}")
    print("=" * 120, flush=True)

    for env_key in envs:
        for method in methods:
            idx += 1
            cid = cell_id(env_key, method)
            run_dir = root / cid

            if cell_complete(run_dir, requested_steps):
                print(f"[{idx:02d}/{total:02d}] SKIP {cid}", flush=True)
                try:
                    summary = json.loads(
                        (run_dir / "analysis" / "cell_summary.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    status_rows.append({
                        "cell_id": cid,
                        "status": "SKIPPED_COMPLETE",
                        **summary,
                    })
                except Exception:
                    pass
                write_csv(root / "serial_status.csv", status_rows)
                build_master(root)
                continue

            print(f"[{idx:02d}/{total:02d}] RUN {cid}", flush=True)

            try:
                result = train_cell(
                    root=root,
                    env_key=env_key,
                    method=method,
                    requested_steps=requested_steps,
                    train_seed=int(args.train_seed),
                    eval_seeds=eval_seeds,
                    profile=profile,
                    device=device,
                )
                summary = result["analysis"]
                status_rows.append({
                    "cell_id": cid,
                    "status": "SUCCESS",
                    **summary,
                })
            except Exception as exc:
                status_rows.append({
                    "cell_id": cid,
                    "status": "FAILED",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
                write_csv(root / "serial_status.csv", status_rows)
                build_master(root)
                if not args.continue_on_error:
                    raise

            write_csv(root / "serial_status.csv", status_rows)
            build_master(root)

    print("\nDONE")
    print(f"Master: {root / 'matrix_master.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
