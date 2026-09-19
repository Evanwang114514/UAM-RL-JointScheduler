# -*- coding: utf-8 -*-
"""
UAGMC equal-dimension candidate-specific effect-time experiment (700k)
=====================================================================

Purpose
-------
Run ONE new method across the physical ladder:
    E0, E2, E3, E4, E5, E6

Default topology:
    T3 = V0 / V1 / V3 -> V2

The control is deliberately strict:
- UAGMC STIN / TemporalLSTMExtractor unchanged
- PPO unchanged
- MCSE/history length unchanged
- action space unchanged
- observation DIMENSION unchanged
- same 8 per-vertiport slots as the source ObservationEncoder

Only the temporal semantics of each candidate block changes.
For the current focal passenger p and candidate k, the 8 variables for k are
constructed at that passenger's own access/effect horizon T_access(p,k), using
ONLY information already committed at decision time:
- already waiting passengers
- passengers already travelling to that vertiport and their remaining access ETA
- current/committed aircraft state and timers
- turnaround release timer
- charging completion timer
- pad calendar timer (used by the physical environment; no hidden future demand)

A minimal focal-action projection is also applied: candidate k's projected
waiting count includes the current focal passenger (+1) at its arrival horizon.
No unrevealed future passenger request is read.

The source UAGMC single-frame layout is preserved exactly:
    [focal OD (4)] + K * [8 vertiport variables]
and the source 6-frame stack is preserved exactly. Historical frames keep the
aligned representation that was valid for the focal passenger at that frame.

Stage definitions
-----------------
E0 : public/source UAGMC legacy automatic aircraft replenishment.
E2 : fixed conserved fleet + original UAGMC service batch/capacity +
     responsive Longest-Queue empty-aircraft reposition.
E3 : E2 + exactly one passenger per service flight.
E4 : E3 + 3 min turnaround.
E5 : E4 + shared finite pad calendar, separation=0.5 min.
E6 : E5 + finite charging capacity, 2 chargers / vertiport.

Compute profile (frozen from 2026-09-19 benchmark)
--------------------------------------------------
Training:
    16 SubprocVecEnv CPU simulators x 1280 steps = 20,480 rollout
    batch_size = 2,048
    n_epochs = 10
    PPO update on CUDA
Evaluation:
    one DummyVecEnv on CPU

The selected profile is the previous measured winner:
    P3_16x1280_b2048

Default budget:
    700,000 timesteps / stage

Run
---
    python train_uagmc_E0_E2_E6_effect_time_700k.py

Useful subset:
    python train_uagmc_E0_E2_E6_effect_time_700k.py --stages E4,E5 --timesteps 700000

Smoke test:
    python train_uagmc_E0_E2_E6_effect_time_700k.py --stages E4 --timesteps 50000 --analysis-steps 50000

This file must be placed beside:
    train_uagmc_E3_E6_obs_topology_matrix_800k_FORMAL.py
    train_uagmc_E3_E4_E5_serial_1m.py
    train_uagmc_reposition_lq_vs_vertisync_800k.py
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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import train_uagmc_E3_E6_obs_topology_matrix_800k_FORMAL as mx
import train_uagmc_reposition_lq_vs_vertisync_800k as rep
from utilss.make_env_fleet import make_env


ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "train_data" / "passengers_300.csv"

STAGES = ("E0", "E2", "E3", "E4", "E5", "E6")
DEFAULT_TOPOLOGY = "T3"
DESTINATION = 2
FLEET_SIZE = 40
MAX_TIME = 600
PAD_SEPARATION_MIN = 0.5
CHARGER_CAPACITY = 2
TURNAROUND_DELAY_MIN = 3.0

DEFAULT_TIMESTEPS = 700_000
DEFAULT_SEED = 1
CHECKPOINT_INTERVAL = 50_000
DEFAULT_ANALYSIS_STEPS = (200_000, 400_000, 500_000, 550_000, 600_000, 650_000, 700_000)
DEFAULT_EVAL_SEEDS = (123, 124, 125)

# Frozen winner from the user's 2026-09-19 E6/O10/T3 benchmark.
PROFILE = mx.SpeedProfile("P3_16x1280_b2048", 16, 1280, 2048)


# =============================================================================
# Generic helpers
# =============================================================================

def parse_list(text: str, allowed: Sequence[str]) -> List[str]:
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


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return default


def fmean(values: Iterable[Any]) -> float:
    arr = np.asarray([fnum(x) for x in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def fstd(values: Iterable[Any]) -> float:
    arr = np.asarray([fnum(x) for x in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    if len(arr) == 1:
        return 0.0
    return float(arr.std(ddof=1))


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def candidates_for(topology: str) -> List[int]:
    return mx.candidates_for(str(topology).upper())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(mx.jsonable(obj), ensure_ascii=False, indent=2),
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
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = {}
            for k, v in row.items():
                if isinstance(v, (dict, list, tuple, np.ndarray)):
                    out[k] = json.dumps(mx.jsonable(v), ensure_ascii=False)
                else:
                    out[k] = v
            w.writerow(out)


# =============================================================================
# Stage setup: E0 / E2 bridge + validated E3-E6 patches
# =============================================================================

def restore_process_patches() -> None:
    """Restore source fixed-fleet methods and source topology constructors."""
    try:
        mx.base.restore_source_methods()
    except Exception:
        pass
    try:
        mx.restore_topology_patch()
    except Exception:
        pass


def configure_stage_process(
    stage: str,
    topology: str,
    pad_separation: float,
    charger_capacity: int,
) -> Dict[str, Any]:
    stage = str(stage).upper()
    topology = str(topology).upper()

    restore_process_patches()

    if stage in ("E3", "E4", "E5", "E6"):
        patch = mx.install_physics(
            stage=stage,
            topology=topology,
            pad_separation=float(pad_separation),
            charger_capacity=int(charger_capacity),
        )
        return dict(patch)

    # E0/E2 still need the exact T3 map injection when topology=T3.
    mx.install_topology_patch(topology)
    cands = candidates_for(topology)

    # Keep shared globals coherent for utilities that read them.
    mx.base.CANDIDATES = list(cands)
    mx.base.TO_VERTIPORT = DESTINATION
    mx.base.RETURN_HUB = DESTINATION
    mx.base.MAX_TIME = MAX_TIME

    if stage == "E0":
        return {
            "stage": "E0_SOURCE_LEGACY_REPLENISH",
            "fleet_mode": "legacy_replenish",
            "aircraft_reposition": "source automatic replenishment",
            "service_batch": "source UAGMC unchanged",
        }

    if stage == "E2":
        # Preserve source service batch/capacity. Change only fixed-fleet empty
        # return dispatch to the already-validated responsive Longest Queue rule.
        rep.CANDIDATES = list(cands)
        rep.RETURN_HUB = DESTINATION
        rep.TO_VERTIPORT = DESTINATION
        rep.MAX_TIME = MAX_TIME
        patch = rep.install_reposition_patch("longest_queue")
        return {
            "stage": "E2_FIXED_LQ_SOURCE_BATCH",
            "fleet_mode": "conserved_closed_loop",
            "fleet_size": FLEET_SIZE,
            "service_batch": "source UAGMC unchanged",
            "reposition_patch": patch,
        }

    raise ValueError(stage)


# =============================================================================
# Equal-dimension effect-time encoder
# =============================================================================

def _lookup_person(scenario: Any, pid: Any) -> Optional[Any]:
    persons_obj = getattr(scenario, "persons", None)
    persons = getattr(persons_obj, "persons", {}) if persons_obj is not None else {}
    if pid in persons:
        return persons[pid]
    for key, person in persons.items():
        if str(key) == str(pid):
            return person
    return None


def _focal_pid(state: Dict[str, Any]) -> Optional[Any]:
    waiting = list((state or {}).get("waiting_decisions", []) or [])
    return waiting[0] if waiting else None


def _access_time(scenario: Any, person: Any, vid: int) -> float:
    vp = scenario.vertiports.vertiport_list[str(int(vid))]
    return float(
        scenario.vehicles.estimate_travel_time(
            origin=person.origin_position,
            destination=vp.vertiport_position,
        )
    )


def _charge_time(evtol: Any) -> float:
    spec = getattr(evtol, "spec", None)
    rate = float(getattr(spec, "charge_rate_kwh_per_min", 0.0) or 0.0)
    cap = float(getattr(spec, "battery_capacity_kwh", 0.0) or 0.0)
    bat = float(getattr(evtol, "battery_kwh", 0.0) or 0.0)
    if rate <= 1e-12:
        return 0.0
    return max(0.0, cap - bat) / rate


def _committed_access_counts(
    scenario: Any,
    vid: int,
    horizon: float,
) -> Tuple[int, int]:
    """Return (arrived_by_horizon, still_incoming_after_horizon).

    Only passengers already committed to this departure vertiport are used.
    """
    arrived = 0
    incoming = 0
    persons_obj = getattr(scenario, "persons", None)
    persons = getattr(persons_obj, "persons", {}) if persons_obj is not None else {}

    for person in persons.values():
        if str(getattr(person, "state", "")).lower() != "enroute":
            continue
        if str(getattr(person, "sub_state", "")).lower() != "to_vertiport":
            continue
        try:
            pvid = int(getattr(person, "origin_vertiport_id"))
        except Exception:
            continue
        if pvid != int(vid):
            continue
        eta = max(0.0, fnum(getattr(person, "current_timer", 0.0), 0.0))
        if eta <= float(horizon) + 1e-9:
            arrived += 1
        else:
            incoming += 1
    return arrived, incoming


def _charging_completion_schedule(
    scenario: Any,
    vid: int,
    stage: str,
    charger_capacity: int,
) -> Dict[str, float]:
    """Causal completion ETA for aircraft already charging locally.

    E6 uses a simple deterministic finite-server schedule for the currently
    known charging queue. No future charging arrivals are inserted.
    """
    local = []
    for e in list(getattr(scenario, "_all_evtols", {}).values()):
        if mx.current_vid(e) != int(vid):
            continue
        if mx.state_name(e) != "CHARGING":
            continue
        if mx.base.is_turnaround_busy(e):
            continue
        local.append(e)

    if not local:
        return {}

    local.sort(
        key=lambda e: (
            float(getattr(e, "_e6_charge_queue_enter_time", 0.0) or 0.0),
            str(getattr(e, "id", "")),
        )
    )

    if str(stage).upper() != "E6":
        return {str(getattr(e, "id", "")): _charge_time(e) for e in local}

    import heapq
    cap = max(1, int(charger_capacity))
    servers = [0.0 for _ in range(cap)]
    heapq.heapify(servers)
    out: Dict[str, float] = {}
    for e in local:
        free = heapq.heappop(servers)
        finish = free + _charge_time(e)
        heapq.heappush(servers, finish)
        out[str(getattr(e, "id", ""))] = float(finish)
    return out


def _project_aircraft_features(
    scenario: Any,
    vid: int,
    horizon: float,
    stage: str,
    charger_capacity: int,
) -> Tuple[float, float, float, float, float, float]:
    """Project the source aircraft-related slots to candidate effect time.

    Returns:
        charging_count_at_horizon,
        total_local_aircraft_at_horizon,
        total_capacity_at_horizon,
        avg_residual_charge_time,
        min_residual_charge_time,
        avg_residual_inbound_flight_time

    The projection follows only currently committed timers. It does not predict
    future requests or future dispatch decisions.
    """
    h = max(0.0, float(horizon))
    stage = str(stage).upper()
    now = float(getattr(scenario, "time", 0.0))
    local_charge_schedule = _charging_completion_schedule(
        scenario, int(vid), stage, int(charger_capacity)
    )

    charging_residual: List[float] = []
    inbound_residual: List[float] = []
    total_local = 0
    total_capacity = 0.0

    evtols = list(getattr(scenario, "_all_evtols", {}).values())

    for e in evtols:
        st = mx.state_name(e)
        cur = mx.current_vid(e)
        tar = mx.target_vid(e)
        cap = float(getattr(getattr(e, "spec", None), "capacity", 0.0) or 0.0)

        # Aircraft currently flying toward this candidate.
        if st == "FLYING" and tar == int(vid):
            eta = max(0.0, fnum(getattr(e, "remaining_time", 0.0), 0.0))
            if eta > h + 1e-9:
                inbound_residual.append(eta - h)
                continue

            # It has physically arrived by the focal horizon.
            total_local += 1
            total_capacity += cap
            after_arrival = h - eta
            turn = TURNAROUND_DELAY_MIN if stage in ("E4", "E5", "E6") else 0.0
            if after_arrival + 1e-9 < turn:
                # Hidden behind turnaround; not counted as charging yet.
                continue

            charge_left = max(0.0, _charge_time(e) - max(0.0, after_arrival - turn))
            if charge_left > 1e-9:
                charging_residual.append(charge_left)
            continue

        # Flying elsewhere is not local at the candidate horizon under the
        # no-uncommitted-action projection.
        if st == "FLYING":
            continue

        if cur != int(vid):
            continue

        # Aircraft is physically local now and remains local unless a future
        # uncommitted action dispatches it; such actions are deliberately not
        # predicted.
        total_local += 1
        total_capacity += cap

        if mx.base.is_turnaround_busy(e):
            release_abs = getattr(e, "_e345_turnaround_release_time", None)
            release_eta = (
                max(0.0, float(release_abs) - now)
                if release_abs is not None
                else 0.0
            )
            if release_eta > h + 1e-9:
                continue
            post = str(getattr(getattr(e, "_e345_post_turnaround_state", None), "name", getattr(e, "_e345_post_turnaround_state", ""))).upper()
            if post == "CHARGING":
                residual = max(0.0, _charge_time(e) - max(0.0, h - release_eta))
                if residual > 1e-9:
                    charging_residual.append(residual)
            continue

        if st == "CHARGING":
            eid = str(getattr(e, "id", ""))
            completion_eta = local_charge_schedule.get(eid, _charge_time(e))
            residual = max(0.0, float(completion_eta) - h)
            if residual > 1e-9:
                charging_residual.append(residual)
            continue

    avg_charge = float(np.mean(charging_residual)) if charging_residual else 0.0
    min_charge = float(np.min(charging_residual)) if charging_residual else 0.0
    avg_flight = float(np.mean(inbound_residual)) if inbound_residual else 0.0

    return (
        float(len(charging_residual)),
        float(total_local),
        float(total_capacity),
        avg_charge,
        min_charge,
        avg_flight,
    )


class EffectTimeEncoder:
    """Drop-in encoder with EXACTLY the source UAGMC observation dimension."""

    def __init__(
        self,
        original_encoder: Any,
        *,
        stage: str,
        candidates: Sequence[int],
        destination: int,
        charger_capacity: int,
        focal_projection: bool = True,
    ):
        self.original_encoder = original_encoder
        self.stage = str(stage).upper()
        self.candidates = [int(x) for x in candidates]
        self.destination = int(destination)
        self.charger_capacity = int(charger_capacity)
        self.focal_projection = bool(focal_projection)

        # Critical control: keep the exact source dimension.
        self.person_dim = int(getattr(original_encoder, "person_dim", 4))
        self.vertiport_dim = int(getattr(original_encoder, "vertiport_dim", 8))
        self.num_vertiports = int(getattr(original_encoder, "num_vertiports", len(self.candidates)))
        self.obs_dim = int(getattr(original_encoder, "obs_dim"))

        expected = 4 + len(self.candidates) * 8
        if self.obs_dim != expected:
            raise RuntimeError(
                f"Equal-dimension guard failed: source obs_dim={self.obs_dim}, "
                f"expected 4+{len(self.candidates)}*8={expected}"
            )

    def encode(self, env: Any, state: Dict[str, Any]) -> np.ndarray:
        pid = _focal_pid(state)
        person = _lookup_person(env, pid) if pid is not None else None

        obs: List[float] = []
        if person is None:
            obs.extend([0.0, 0.0, 0.0, 0.0])
        else:
            obs.extend([float(x) for x in person.origin_position])
            obs.extend([float(x) for x in person.destination_position])

        # Follow the source vertiport dictionary order, excluding destination.
        encoded_candidates: List[int] = []
        for vid_raw, vp in env.vertiports.vertiport_list.items():
            vid = int(vid_raw)
            if vid == self.destination:
                continue
            encoded_candidates.append(vid)

            horizon = _access_time(env, person, vid) if person is not None else 0.0

            waiting_now = len(list(getattr(vp, "person_list", []) or []))
            committed_arrived, committed_still_incoming = _committed_access_counts(
                env, vid, horizon
            )

            waiting_future = float(waiting_now + committed_arrived)
            if person is not None and self.focal_projection:
                # Simple action-centered demand projection: if this candidate is
                # chosen, the focal passenger joins this candidate at its own
                # effect time.
                waiting_future += 1.0

            incoming_future = float(committed_still_incoming)

            (
                charging_future,
                total_evtols_future,
                total_capacity_future,
                avg_charge_future,
                min_charge_future,
                avg_flight_future,
            ) = _project_aircraft_features(
                env,
                vid,
                horizon,
                self.stage,
                self.charger_capacity,
            )

            obs.extend(
                [
                    waiting_future,
                    incoming_future,
                    charging_future,
                    total_evtols_future,
                    total_capacity_future,
                    avg_charge_future,
                    min_charge_future,
                    avg_flight_future,
                ]
            )

        if encoded_candidates != self.candidates:
            raise RuntimeError(
                f"Candidate/order mismatch: encoder saw {encoded_candidates}, "
                f"expected {self.candidates}"
            )

        arr = np.asarray(obs, dtype=np.float32)
        if arr.shape != (self.obs_dim,):
            raise RuntimeError(
                f"Effect-time encoder shape={arr.shape}, expected={(self.obs_dim,)}"
            )
        return arr


# =============================================================================
# Outer wrapper: T3 fixed-fleet normalization + terminal snapshot
# =============================================================================

class ExperimentWrapper(gym.Wrapper):
    def __init__(self, env: Any, *, stage: str, topology: str, fleet_size: int):
        super().__init__(env)
        self.stage = str(stage).upper()
        self.topology = str(topology).upper()
        self.fleet_size = int(fleet_size)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._t3_allocation = None

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        if self.topology == "T3" and self.stage != "E0":
            scenario = mx.find_scenario(self.env)
            self._t3_allocation = mx.normalize_t3_initial_allocation(
                scenario, self.fleet_size
            )
            # Refresh with whichever encoder is installed (source or effect-time).
            obs = mx.refresh_uam_observation_after_t3_redistribution(self.env)
        else:
            self._t3_allocation = None

        return obs, info

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info)
            if bool(terminated) or bool(truncated):
                info["terminal_snapshot"] = mx.snapshot_episode(mx.find_scenario(self.env))
            return obs, reward, terminated, truncated, info
        if len(out) == 4:
            obs, reward, done, info = out
            info = dict(info)
            if bool(done):
                info["terminal_snapshot"] = mx.snapshot_episode(mx.find_scenario(self.env))
            return obs, reward, done, info
        raise RuntimeError(f"Unexpected env.step tuple length={len(out)}")


# =============================================================================
# Environment construction
# =============================================================================

def make_experiment_env_factory(
    *,
    stage: str,
    topology: str,
    encoder_mode: str,
    fleet_size: int,
    env_index: int,
    run_dir: Path,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
):
    def _init():
        configure_stage_process(
            stage=stage,
            topology=topology,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
        )
        cands = candidates_for(topology)

        fleet_mode = "legacy_replenish" if stage == "E0" else "conserved_closed_loop"
        env = make_env(
            max_time=int(max_time),
            log_dir=run_dir / "monitor",
            env_index=int(env_index),
            person_spawn_file=str(TRAIN_FILE),
            candidate_from_vertiports=list(cands),
            to_vertiport=DESTINATION,
            enable_logger=False,
            fleet_mode=fleet_mode,
            fleet_size=(None if stage == "E0" else int(fleet_size)),
            fleet_assertions=(stage != "E0"),
        )()

        if encoder_mode == "effect_time":
            uam = mx.find_uam_wrapper(env)
            uam.encoder = EffectTimeEncoder(
                uam.encoder,
                stage=stage,
                candidates=cands,
                destination=DESTINATION,
                charger_capacity=charger_capacity,
                focal_projection=True,
            )
            # obs_dim is unchanged, so UAMRLWrapper.observation_space stays valid.
        elif encoder_mode != "uagmc":
            raise ValueError(encoder_mode)

        env = ExperimentWrapper(
            env,
            stage=stage,
            topology=topology,
            fleet_size=fleet_size,
        )
        return env

    return _init


def build_train_env(
    *,
    stage: str,
    topology: str,
    encoder_mode: str,
    seed: int,
    run_dir: Path,
    fleet_size: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> VecNormalize:
    mx.apply_speed_profile(PROFILE)
    factories = [
        make_experiment_env_factory(
            stage=stage,
            topology=topology,
            encoder_mode=encoder_mode,
            fleet_size=fleet_size,
            env_index=i,
            run_dir=run_dir,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
            max_time=max_time,
        )
        for i in range(PROFILE.n_envs)
    ]
    raw = SubprocVecEnv(factories, start_method="spawn")
    raw.seed(int(seed))
    return VecNormalize(
        raw,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=mx.base.GAMMA,
    )


def build_eval_raw_env(
    *,
    stage: str,
    topology: str,
    encoder_mode: str,
    run_dir: Path,
    fleet_size: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
):
    return DummyVecEnv(
        [
            make_experiment_env_factory(
                stage=stage,
                topology=topology,
                encoder_mode=encoder_mode,
                fleet_size=fleet_size,
                env_index=9999,
                run_dir=run_dir,
                pad_separation=pad_separation,
                charger_capacity=charger_capacity,
                max_time=max_time,
            )
        ]
    )


# =============================================================================
# Evaluation / analysis
# =============================================================================

def evaluate_checkpoint(
    *,
    stage: str,
    topology: str,
    encoder_mode: str,
    model_path: Path,
    vec_path: Path,
    train_step: int,
    eval_seed: int,
    run_dir: Path,
    fleet_size: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> Dict[str, Any]:
    seed_all(eval_seed)
    raw = build_eval_raw_env(
        stage=stage,
        topology=topology,
        encoder_mode=encoder_mode,
        run_dir=run_dir / "_eval_monitor",
        fleet_size=fleet_size,
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
        action_counts = Counter()
        prob_rows: List[np.ndarray] = []
        reward_sum = 0.0
        system_person_minutes = 0.0
        episode_steps = 0
        terminal_snapshot = None

        while not bool(done[0]):
            scenario = mx.find_scenario(env)
            system_person_minutes += mx.active_system_count(scenario)

            p = mx.policy_probs(model, obs)
            action, _ = model.predict(obs, deterministic=True)
            ai = int(np.asarray(action).reshape(-1)[0])
            action_counts[ai] += 1
            prob_rows.append(p.copy())

            obs, reward, done, infos = env.step(action)
            reward_sum += fnum(np.asarray(reward).reshape(-1)[0], 0.0)
            episode_steps += 1

            if infos and isinstance(infos[0], dict) and "terminal_snapshot" in infos[0]:
                terminal_snapshot = infos[0]["terminal_snapshot"]

            if episode_steps > max_time + 100:
                raise RuntimeError("Evaluation exceeded max-time guard")

        if terminal_snapshot is None:
            raise RuntimeError("terminal_snapshot missing before VecEnv autoreset")

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
                "topology": topology,
                "encoder_mode": encoder_mode,
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
        restore_process_patches()
        gc.collect()


def analyze_stage(
    *,
    stage: str,
    topology: str,
    encoder_mode: str,
    run_dir: Path,
    requested_steps: int,
    analysis_steps: Sequence[int],
    eval_seeds: Sequence[int],
    fleet_size: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> Dict[str, Any]:
    adir = run_dir / "analysis"
    adir.mkdir(parents=True, exist_ok=True)

    steps = sorted({int(x) for x in analysis_steps if int(x) <= requested_steps} | {int(requested_steps)})
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for step in steps:
        model_path = run_dir / "checkpoints" / f"uam_ppo_{step}_steps.zip"
        vec_path = run_dir / "checkpoints" / f"uam_ppo_vecnormalize_{step}_steps.pkl"
        if not model_path.exists() or not vec_path.exists():
            continue
        for eval_seed in eval_seeds:
            try:
                row = evaluate_checkpoint(
                    stage=stage,
                    topology=topology,
                    encoder_mode=encoder_mode,
                    model_path=model_path,
                    vec_path=vec_path,
                    train_step=step,
                    eval_seed=int(eval_seed),
                    run_dir=run_dir,
                    fleet_size=fleet_size,
                    pad_separation=pad_separation,
                    charger_capacity=charger_capacity,
                    max_time=max_time,
                )
                rows.append(row)
                print(
                    f"    [eval] {stage} {step:,} seed={eval_seed} | "
                    f"ATT={row['ATT']:.3f} | finish={row['N_finished']}/{row['N']} | "
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

    write_csv(adir / "checkpoint_eval_raw.csv", rows)
    write_csv(adir / "errors.csv", errors)
    if not rows:
        raise RuntimeError(f"No evaluation rows for {stage}")

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[int(r["train_step"])].append(r)

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

    curve: List[Dict[str, Any]] = []
    for step in sorted(grouped):
        group = grouped[step]
        agg: Dict[str, Any] = {"train_step": step, "n_eval_seeds": len(group)}
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
        curve.append(agg)

    write_csv(adir / "checkpoint_curve.csv", curve)
    final = next((r for r in curve if int(r["train_step"]) == requested_steps), None)
    if final is None:
        raise RuntimeError(f"Formal {requested_steps} checkpoint was not evaluated")

    late_lo = max(0, requested_steps - 200_000)
    late = [r for r in curve if late_lo <= int(r["train_step"]) <= requested_steps]
    late_js = [fnum(r.get("system_person_minutes_per_passenger_mean")) for r in late]
    late_att = [fnum(r.get("ATT_mean")) for r in late]

    summary = {
        "stage": stage,
        "topology": topology,
        "encoder_mode": encoder_mode,
        "formal_step": requested_steps,
        "final": final,
        "late_window_lo": late_lo,
        "late_window_steps": [int(r["train_step"]) for r in late],
        "late_ATT_mean": fmean(late_att),
        "late_ATT_checkpoint_std": fstd(late_att),
        "late_Jsys_mean": fmean(late_js),
        "late_Jsys_checkpoint_std": fstd(late_js),
        "late_Jsys_best": min([x for x in late_js if np.isfinite(x)], default=float("nan")),
        "late_Jsys_final_regression": (
            fnum(final.get("system_person_minutes_per_passenger_mean"))
            - min([x for x in late_js if np.isfinite(x)], default=float("nan"))
        ),
        "evaluation_errors": len(errors),
    }
    write_json(adir / "summary.json", summary)
    return summary


# =============================================================================
# One-stage training
# =============================================================================

def train_stage(
    *,
    stage: str,
    topology: str,
    encoder_mode: str,
    seed: int,
    requested_steps: int,
    root: Path,
    device: str,
    analysis_steps: Sequence[int],
    eval_seeds: Sequence[int],
    fleet_size: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> Dict[str, Any]:
    run_dir = root / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    seed_all(seed)
    mx.apply_speed_profile(PROFILE)

    env = None
    model = None
    started = time.time()
    try:
        env = build_train_env(
            stage=stage,
            topology=topology,
            encoder_mode=encoder_mode,
            seed=seed,
            run_dir=run_dir,
            fleet_size=fleet_size,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
            max_time=max_time,
        )
        model = mx.build_model(
            env=env,
            profile=PROFILE,
            seed=seed,
            run_dir=run_dir,
            device=device,
        )

        run_manifest = {
            "stage": stage,
            "topology": topology,
            "candidates": candidates_for(topology),
            "destination": DESTINATION,
            "encoder_mode": encoder_mode,
            "equal_dimension_to_source_uagmc": True,
            "effect_time_rule": (
                "each candidate block uses that focal passenger's own access horizon; "
                "committed passenger/aircraft timers only; focal passenger +1 projection"
                if encoder_mode == "effect_time"
                else "source UAGMC ObservationEncoder unchanged"
            ),
            "train_seed": seed,
            "requested_timesteps": requested_steps,
            "fleet_size_E2_E6": fleet_size,
            "E0_fleet_mode": "legacy_replenish",
            "pad_separation_min": pad_separation,
            "charger_capacity_E6": charger_capacity,
            "compute_profile": {
                "name": PROFILE.name,
                "n_envs": PROFILE.n_envs,
                "n_steps": PROFILE.n_steps,
                "global_rollout": PROFILE.n_envs * PROFILE.n_steps,
                "batch_size": PROFILE.batch_size,
                "n_epochs": mx.base.N_EPOCHS,
                "training_device": device,
                "evaluation_device": "cpu",
            },
        }
        write_json(run_dir / "run_manifest.json", run_manifest)

        callbacks = mx.build_callbacks(
            run_dir=run_dir,
            profile=PROFILE,
            requested_steps=requested_steps,
        )

        print("\n" + "=" * 132)
        print(
            f"START {stage} | {encoder_mode} | {topology} | {requested_steps:,} steps | "
            f"16x1280 batch=2048 | device={model.device}"
        )
        print("=" * 132, flush=True)

        model.learn(
            total_timesteps=int(requested_steps),
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=True,
        )

        elapsed = time.time() - started
        actual_steps = int(model.num_timesteps)
        model.save(run_dir / "final_rl_model")
        env.save(run_dir / "final_vec_normalize.pkl")

        formal_model = run_dir / "checkpoints" / f"uam_ppo_{requested_steps}_steps.zip"
        formal_vec = run_dir / "checkpoints" / f"uam_ppo_vecnormalize_{requested_steps}_steps.pkl"
        if not formal_model.exists() or not formal_vec.exists():
            raise FileNotFoundError(
                f"Exact {requested_steps} checkpoint pair missing: {formal_model}, {formal_vec}"
            )

        result = {
            "status": "SUCCESS",
            "stage": stage,
            "encoder_mode": encoder_mode,
            "requested_timesteps": requested_steps,
            "actual_rollout_final_timesteps": actual_steps,
            "elapsed_seconds": elapsed,
            "training_fps": actual_steps / max(elapsed, 1e-9),
            "formal_model": str(formal_model),
            "formal_vec": str(formal_vec),
            "finished": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(run_dir / "run_end.json", result)

        try:
            env.close()
        except Exception:
            pass
        env = None
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        summary = analyze_stage(
            stage=stage,
            topology=topology,
            encoder_mode=encoder_mode,
            run_dir=run_dir,
            requested_steps=requested_steps,
            analysis_steps=analysis_steps,
            eval_seeds=eval_seeds,
            fleet_size=fleet_size,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
            max_time=max_time,
        )
        result["analysis"] = summary
        return result
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        restore_process_patches()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# =============================================================================
# Suite runner (also imported by the E0/E2 source-UAGMC supplement file)
# =============================================================================

def run_experiment(
    *,
    stages: Sequence[str],
    topology: str,
    encoder_mode: str,
    timesteps: int,
    seed: int,
    device: str,
    output_root: Optional[str],
    analysis_steps: Sequence[int],
    eval_seeds: Sequence[int],
    continue_on_error: bool = False,
) -> Path:
    stages = [str(s).upper() for s in stages]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise ValueError(f"Unsupported stages={bad}")
    topology = str(topology).upper()
    if topology not in ("T2", "T3"):
        raise ValueError(topology)
    if encoder_mode not in ("effect_time", "uagmc"):
        raise ValueError(encoder_mode)
    if timesteps % CHECKPOINT_INTERVAL != 0:
        raise ValueError(f"timesteps must be divisible by {CHECKPOINT_INTERVAL}")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_root:
        root = Path(output_root).expanduser()
        if not root.is_absolute():
            root = (ROOT / root).resolve()
        else:
            root = root.resolve()
    else:
        tag = "effect_time" if encoder_mode == "effect_time" else "source_uagmc"
        root = (
            ROOT
            / "serial_runs"
            / f"uagmc_{tag}_{'_'.join(stages)}_{topology}_{timesteps//1000}k_seed{seed}_{stamp}"
        ).resolve()
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "UAGMC_EQUAL_DIMENSION_EFFECT_TIME" if encoder_mode == "effect_time" else "UAGMC_SOURCE_OBSERVATION_SUPPLEMENT",
        "created": datetime.now().isoformat(timespec="seconds"),
        "stages": stages,
        "topology": topology,
        "encoder_mode": encoder_mode,
        "timesteps_per_stage": timesteps,
        "train_seed": seed,
        "fleet_size_E2_E6": FLEET_SIZE,
        "max_time": MAX_TIME,
        "passenger_trace": str(TRAIN_FILE),
        "pad_separation_min_E5_E6": PAD_SEPARATION_MIN,
        "charger_capacity_E6": CHARGER_CAPACITY,
        "profile": {
            "name": PROFILE.name,
            "n_envs": 16,
            "n_steps": 1280,
            "global_rollout": 20480,
            "batch_size": 2048,
            "training_device": device,
            "evaluation_device": "cpu",
        },
        "equal_observation_dimension": True,
        "note": (
            "Effect-time mode changes observation semantics only, not dimension/network/PPO."
            if encoder_mode == "effect_time"
            else "Source UAGMC observation/history unchanged."
        ),
    }
    write_json(root / "experiment_manifest.json", manifest)

    print("=" * 132)
    print(f"Experiment        : {manifest['experiment']}")
    print(f"Stages            : {stages}")
    print(f"Topology          : {topology}")
    print(f"Encoder           : {encoder_mode}")
    print(f"Steps/stage       : {timesteps:,}")
    print("Compute           : 16 CPU envs x 1280; batch=2048; CUDA PPO; CPU eval")
    print(f"Output            : {root}")
    print("=" * 132)

    status_rows: List[Dict[str, Any]] = []
    for stage in stages:
        try:
            result = train_stage(
                stage=stage,
                topology=topology,
                encoder_mode=encoder_mode,
                seed=seed,
                requested_steps=timesteps,
                root=root,
                device=device,
                analysis_steps=analysis_steps,
                eval_seeds=eval_seeds,
                fleet_size=FLEET_SIZE,
                pad_separation=PAD_SEPARATION_MIN,
                charger_capacity=CHARGER_CAPACITY,
                max_time=MAX_TIME,
            )
            final = result["analysis"]["final"]
            status_rows.append(
                {
                    "stage": stage,
                    "status": "SUCCESS",
                    "ATT_final": final.get("ATT_mean"),
                    "completion_final": final.get("completion_rate_mean"),
                    "Jsys_final": final.get("system_person_minutes_per_passenger_mean"),
                    "late_Jsys_std": result["analysis"].get("late_Jsys_checkpoint_std"),
                    "late_Jsys_final_regression": result["analysis"].get("late_Jsys_final_regression"),
                }
            )
        except Exception as exc:
            status_rows.append(
                {
                    "stage": stage,
                    "status": "FAILED",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            write_csv(root / "serial_status.csv", status_rows)
            if not continue_on_error:
                raise
        write_csv(root / "serial_status.csv", status_rows)

    print("\nDONE")
    print(f"Results: {root / 'serial_status.csv'}")
    return root


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--topology", default=DEFAULT_TOPOLOGY, choices=["T2", "T3"])
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--output-root", default=None)
    p.add_argument(
        "--analysis-steps",
        default=",".join(str(x) for x in DEFAULT_ANALYSIS_STEPS),
    )
    p.add_argument(
        "--eval-seeds",
        default=",".join(str(x) for x in DEFAULT_EVAL_SEEDS),
    )
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_experiment(
        stages=parse_list(args.stages, STAGES),
        topology=args.topology,
        encoder_mode="effect_time",
        timesteps=int(args.timesteps),
        seed=int(args.seed),
        device=args.device,
        output_root=args.output_root,
        analysis_steps=parse_ints(args.analysis_steps),
        eval_seeds=parse_ints(args.eval_seeds),
        continue_on_error=bool(args.continue_on_error),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
