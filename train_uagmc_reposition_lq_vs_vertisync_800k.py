# -*- coding: utf-8 -*-
"""
UAGMC N=16 aircraft-reposition comparison
=========================================

Two passenger-PPO training runs, with ONLY the empty-aircraft reposition rule changed:

A) longest_queue
   - When one or more EMPTY + IDLE aircraft are at the arrival hub V2,
     send them to the departure vertiport with the largest current passenger queue.
   - For multiple aircraft in the same dispatch event, subtract one aircraft-load
     from that queue virtually before assigning the next aircraft. This avoids
     sending every simultaneous empty aircraft to the same queue.

B) vertisync_simple
   - A deliberately SIMPLE VertiSync-style rebalancing approximation.
   - At a synchronization/rebalancing epoch, snapshot current queues Q0,Q1.
   - Convert the frozen backlog into required aircraft-loads ceil(Q_i / capacity).
   - Count currently TAKEOFF-READY (IDLE) aircraft already at each origin plus
     empty aircraft already flying toward that origin.
   - Freeze the resulting rebalancing deficits as one cycle plan.
   - Empty IDLE aircraft at V2 are dispatched according to that frozen plan.
   - New requests arriving after the snapshot do NOT alter the active plan;
     they are considered at the next cycle.

IMPORTANT: vertisync_simple is NOT the paper's full MILP/slot scheduler.
It preserves only the paper's core high-level logic relevant to this experiment:
    * centralized aircraft location + queue information;
    * cycle/snapshot operation;
    * synchronous service/rebalancing planning;
    * rebalancing toward origins that need aircraft;
    * no use of future demand rates.

It DOES NOT reproduce:
    * slot-level airspace occupancy;
    * takeoff/landing separation constraints of VertiSync;
    * the exact MILP objective;
    * exact "only pre-cycle requests are serviced" gating of the service engine.
The existing UAGMC service/boarding/charging dynamics remain untouched.

Charging / SoC
--------------
Both methods preserve the existing UAGMC charging/SoC logic.
Only aircraft whose state is IDLE are eligible for empty reposition.
CHARGING aircraft are never force-dispatched by these policies.
The existing ConservedFleetScenario._start_empty_reposition(...) is used,
so the existing empty-flight time/energy logic remains active.

Frozen environment
------------------
fleet_mode               = conserved_closed_loop
fleet_size               = 16
initial allocation       = V0:12, V1:4
passenger candidates     = [0,1]
service destination      = V2
passenger trace          = train_data/passengers_300.csv
UAGMC batching/capacity  = unchanged
reward                    = unchanged
observation/history       = unchanged
PPO architecture          = unchanged

Fast training configuration
---------------------------
16 SubprocVecEnv CPU simulators
1280 steps / env
global rollout = 16 * 1280 = 20,480
batch_size = 512
CUDA PPO update
NO mid-training evaluation
checkpoints every 50k

Default experiment
------------------
method 1: longest_queue      800,000 timesteps, train seed=1
method 2: vertisync_simple   800,000 timesteps, train seed=1
total requested budget       1,600,000 timesteps

The same train seed is intentionally used for the two methods so this is a
paired development comparison. Final paper claims should still be repeated
with multiple training seeds.

Usage
-----
Place this file at UAGMC-main root:

    train_uagmc_reposition_lq_vs_vertisync_800k.py

Then:

    python train_uagmc_reposition_lq_vs_vertisync_800k.py

Optional:
    python train_uagmc_reposition_lq_vs_vertisync_800k.py --seed 1
    python train_uagmc_reposition_lq_vs_vertisync_800k.py --methods longest_queue
    python train_uagmc_reposition_lq_vs_vertisync_800k.py --methods vertisync_simple

Outputs
-------
serial_runs/
  uagmc_reposition_LQ_vs_VertiSync_N16_seed1_800k_<timestamp>/
    experiment_manifest.json
    serial_status.csv
    longest_queue/
      checkpoints/
      final_rl_model.zip
      final_vec_normalize.pkl
      training_milestones.csv
      run_manifest.json
      run_end.json
    vertisync_simple/
      ...

Scientific caution
------------------
This script monkey-patches ONLY ConservedFleetScenario._dispatch_fixed_returns
inside each environment process. If the local fixed-fleet implementation no
longer exposes _dispatch_fixed_returns or _start_empty_reposition, the script
fails fast instead of silently changing semantics.
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import math
import random
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from utilss.make_env_fleet import make_env
from utilss.sb3_utils import linear_schedule


ROOT = Path(__file__).resolve().parent

# =============================================================================
# Frozen experiment constants
# =============================================================================

FLEET_SIZE = 16
CANDIDATES = [0, 1]
RETURN_HUB = 2
TO_VERTIPORT = 2

TRAIN_FILE = ROOT / "train_data" / "passengers_300.csv"
MAX_TIME = 500

N_ENVS = 16
N_STEPS = 1280
GLOBAL_ROLLOUT = N_ENVS * N_STEPS  # 20,480

BATCH_SIZE = 512
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.25
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
INITIAL_LR = 5e-4

DEFAULT_SEED = 1
DEFAULT_TIMESTEPS = 800_000
DEFAULT_CHECKPOINT_INTERVAL = 50_000

VALID_METHODS = ("longest_queue", "vertisync_simple")


# =============================================================================
# Generic utilities
# =============================================================================

def jsonable(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    try:
        json.dumps(x)
        return x
    except Exception:
        return repr(x)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    serial = {}
    for k, v in row.items():
        if isinstance(v, (dict, list, tuple, np.ndarray)):
            serial[k] = json.dumps(jsonable(v), ensure_ascii=False)
        else:
            serial[k] = v

    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(serial.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(serial)
        f.flush()


def parse_methods(text: str) -> List[str]:
    vals = [x.strip().lower() for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("method list cannot be empty")

    bad = [x for x in vals if x not in VALID_METHODS]
    if bad:
        raise ValueError(
            f"Unknown method(s): {bad}; valid={VALID_METHODS}"
        )
    return vals


def state_name(evtol: Any) -> str:
    st = getattr(evtol, "state", None)
    if hasattr(st, "name"):
        return str(st.name).upper()
    return str(st).upper()


def current_vertiport(evtol: Any) -> Optional[int]:
    for attr in (
        "current_vertiport_id",
        "vertiport_id",
        "current_location",
    ):
        if hasattr(evtol, attr):
            value = getattr(evtol, attr)
            try:
                return int(value)
            except Exception:
                pass
    return None


def target_vertiport(evtol: Any) -> Optional[int]:
    for attr in (
        "target_vertiport_id",
        "destination_vertiport_id",
        "destination_id",
        "target_vertiport",
        "target_id",
    ):
        if hasattr(evtol, attr):
            value = getattr(evtol, attr)
            try:
                return int(value)
            except Exception:
                pass
    return None


def passenger_ids(evtol: Any) -> List[Any]:
    for attr in (
        "passenger_ids",
        "passengers",
        "person_ids",
    ):
        if hasattr(evtol, attr):
            value = getattr(evtol, attr)
            if value is None:
                return []
            try:
                return list(value)
            except Exception:
                return []
    return []


def aircraft_id(evtol: Any) -> Any:
    for attr in ("id", "evtol_id", "aircraft_id"):
        if hasattr(evtol, attr):
            return getattr(evtol, attr)
    return None


def sorted_evtols(scenario: Any) -> List[Tuple[Any, Any]]:
    data = getattr(scenario, "_all_evtols", {}) or {}
    items = list(data.items())
    items.sort(key=lambda kv: str(kv[0]))
    return items


def queue_length(scenario: Any, vid: int) -> int:
    vp = scenario.vertiports.vertiport_list[str(int(vid))]
    return len(list(getattr(vp, "person_list", []) or []))


def aircraft_capacity(scenario: Any) -> int:
    for _, evtol in sorted_evtols(scenario):
        for attr in ("capacity", "passenger_capacity"):
            if hasattr(evtol, attr):
                try:
                    c = int(getattr(evtol, attr))
                    if c > 0:
                        return c
                except Exception:
                    pass
    # Public UAGMC source uses 4 seats.
    return 4


def idle_empty_at_hub(scenario: Any) -> List[Tuple[Any, Any]]:
    out = []
    hub = int(
        getattr(
            scenario,
            "fleet_return_vertiport",
            getattr(scenario, "_fleet_return_vertiport", RETURN_HUB),
        )
    )

    for eid, evtol in sorted_evtols(scenario):
        if state_name(evtol) != "IDLE":
            continue
        if passenger_ids(evtol):
            continue
        if current_vertiport(evtol) != hub:
            continue
        out.append((eid, evtol))

    return out


def ready_supply_at_origin(scenario: Any, vid: int) -> int:
    """
    Count aircraft that can plausibly serve the frozen cycle backlog:
      - IDLE empty aircraft already at origin;
      - empty FLYING aircraft already repositioning toward origin.

    CHARGING aircraft are NOT counted as immediately takeoff-ready.
    This keeps UAGMC charging as a physical availability constraint.
    """
    total = 0

    for _, evtol in sorted_evtols(scenario):
        st = state_name(evtol)

        if st == "IDLE":
            if (
                current_vertiport(evtol) == int(vid)
                and not passenger_ids(evtol)
            ):
                total += 1

        elif st == "FLYING":
            if (
                target_vertiport(evtol) == int(vid)
                and not passenger_ids(evtol)
            ):
                total += 1

    return total


# =============================================================================
# Robust call into existing fixed-fleet empty-reposition physics
# =============================================================================

def start_empty_reposition_existing(
    scenario: Any,
    evtol: Any,
    eid: Any,
    target_vid: int,
) -> None:
    """
    Call scenario._start_empty_reposition using the local implementation's
    signature. We deliberately reuse the existing fixed-fleet flight physics
    rather than reimplementing energy/time bookkeeping here.
    """
    if not hasattr(scenario, "_start_empty_reposition"):
        raise RuntimeError(
            "ConservedFleetScenario has no _start_empty_reposition; "
            "cannot preserve the existing empty-flight physics."
        )

    method = scenario._start_empty_reposition
    origin_vid = current_vertiport(evtol)

    if origin_vid is None:
        raise RuntimeError(
            f"Cannot determine current vertiport for aircraft {eid}"
        )

    # First try parameter-name-based dispatch.
    try:
        sig = inspect.signature(method)
        kwargs: Dict[str, Any] = {}
        unresolved = []

        for p in sig.parameters.values():
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            name = p.name.lower()

            if (
                ("evtol" in name or "aircraft" in name)
                and "id" in name
            ):
                kwargs[p.name] = eid
            elif "evtol" in name or "aircraft" in name:
                kwargs[p.name] = evtol
            elif (
                "origin" in name
                or "source" in name
                or name.startswith("from")
            ):
                kwargs[p.name] = int(origin_vid)
            elif (
                "destination" in name
                or "target" in name
                or name.startswith("to_")
                or name in ("to", "dest", "dest_id")
            ):
                kwargs[p.name] = int(target_vid)
            else:
                if p.default is inspect._empty:
                    unresolved.append(p.name)

        if not unresolved:
            method(**kwargs)
            return
    except TypeError:
        pass

    # Fallbacks for plausible local signatures.
    candidates = [
        (evtol, int(origin_vid), int(target_vid)),
        (eid, int(origin_vid), int(target_vid)),
        (evtol, int(target_vid)),
        (eid, int(target_vid)),
    ]

    last_error = None
    for args in candidates:
        try:
            method(*args)
            return
        except TypeError as exc:
            last_error = exc

    raise RuntimeError(
        "Could not call _start_empty_reposition with the local signature. "
        f"signature={inspect.signature(method)}; last_error={last_error!r}"
    )


# =============================================================================
# Reposition policy A: longest queue
# =============================================================================

def dispatch_longest_queue(scenario: Any) -> None:
    available = idle_empty_at_hub(scenario)
    if not available:
        return

    cap = aircraft_capacity(scenario)

    # Current queues only. New future passengers are not used.
    virtual_q = {
        vid: queue_length(scenario, vid)
        for vid in CANDIDATES
    }

    for eid, evtol in available:
        best_vid = max(
            CANDIDATES,
            key=lambda v: (
                virtual_q[v],
                -ready_supply_at_origin(scenario, v),
                -v,
            ),
        )

        # If no currently waiting passenger exists anywhere, park at V2.
        # Do not use hidden future demand to pre-position.
        if virtual_q[best_vid] <= 0:
            break

        start_empty_reposition_existing(
            scenario=scenario,
            evtol=evtol,
            eid=eid,
            target_vid=best_vid,
        )

        # One aircraft can serve one UAGMC batch/load. Use a virtual subtraction
        # before assigning the next simultaneous empty aircraft.
        virtual_q[best_vid] = max(
            0,
            virtual_q[best_vid] - cap,
        )


# =============================================================================
# Reposition policy B: simplified VertiSync-style cycle plan
# =============================================================================

def _sync_reset_state(scenario: Any) -> None:
    scenario._vs_simple_cycle_id = 0
    scenario._vs_simple_plan = {vid: 0 for vid in CANDIDATES}
    scenario._vs_simple_snapshot_q = {vid: 0 for vid in CANDIDATES}
    scenario._vs_simple_last_cycle_time = None


def _sync_plan_empty(scenario: Any) -> bool:
    plan = getattr(scenario, "_vs_simple_plan", None)
    if not isinstance(plan, dict):
        return True
    return sum(int(plan.get(v, 0)) for v in CANDIDATES) <= 0


def _build_sync_cycle_plan(scenario: Any) -> None:
    if not hasattr(scenario, "_vs_simple_cycle_id"):
        _sync_reset_state(scenario)

    cap = aircraft_capacity(scenario)

    snapshot_q = {
        vid: queue_length(scenario, vid)
        for vid in CANDIDATES
    }

    # Frozen service requirement in aircraft loads.
    required_loads = {
        vid: int(math.ceil(snapshot_q[vid] / max(1, cap)))
        for vid in CANDIDATES
    }

    # Centralized aircraft-location information.
    ready_supply = {
        vid: ready_supply_at_origin(scenario, vid)
        for vid in CANDIDATES
    }

    # Rebalancing deficit for this frozen snapshot.
    plan = {
        vid: max(
            0,
            required_loads[vid] - ready_supply[vid],
        )
        for vid in CANDIDATES
    }

    scenario._vs_simple_cycle_id = int(
        getattr(scenario, "_vs_simple_cycle_id", 0)
    ) + 1
    scenario._vs_simple_plan = plan
    scenario._vs_simple_snapshot_q = snapshot_q
    scenario._vs_simple_last_cycle_time = getattr(
        scenario, "time", None
    )


def dispatch_vertisync_simple(scenario: Any) -> None:
    available = idle_empty_at_hub(scenario)
    if not available:
        return

    if not hasattr(scenario, "_vs_simple_plan"):
        _sync_reset_state(scenario)

    # The previous frozen cycle plan has been staged -> start a new cycle from
    # the CURRENT queues. Requests that arrived while a prior plan was active
    # did not alter that prior plan.
    if _sync_plan_empty(scenario):
        _build_sync_cycle_plan(scenario)

    plan = scenario._vs_simple_plan

    # No service/rebalancing need in this cycle: keep the aircraft parked at V2.
    if sum(int(plan.get(v, 0)) for v in CANDIDATES) <= 0:
        return

    for eid, evtol in available:
        positive = [
            v for v in CANDIDATES
            if int(plan.get(v, 0)) > 0
        ]
        if not positive:
            break

        # Synchronous frozen-plan priority:
        # largest remaining aircraft deficit first; then larger frozen queue.
        target = max(
            positive,
            key=lambda v: (
                int(plan.get(v, 0)),
                int(
                    getattr(
                        scenario,
                        "_vs_simple_snapshot_q",
                        {},
                    ).get(v, 0)
                ),
                -v,
            ),
        )

        start_empty_reposition_existing(
            scenario=scenario,
            evtol=evtol,
            eid=eid,
            target_vid=target,
        )

        plan[target] = int(plan.get(target, 0)) - 1


# =============================================================================
# Monkey-patch installer
# =============================================================================

def install_reposition_patch(method_name: str) -> Dict[str, Any]:
    """
    Patch ONLY ConservedFleetScenario._dispatch_fixed_returns in the current
    process. Under SubprocVecEnv each simulator process receives its own patch.
    """
    from at_obj.scenario_fixed_fleet import ConservedFleetScenario

    if not hasattr(ConservedFleetScenario, "_dispatch_fixed_returns"):
        raise RuntimeError(
            "Local ConservedFleetScenario has no _dispatch_fixed_returns. "
            "Refusing to silently change another part of the environment."
        )

    if not hasattr(ConservedFleetScenario, "_start_empty_reposition"):
        raise RuntimeError(
            "Local ConservedFleetScenario has no _start_empty_reposition."
        )

    if method_name == "longest_queue":
        new_method = dispatch_longest_queue
    elif method_name == "vertisync_simple":
        new_method = dispatch_vertisync_simple
    else:
        raise ValueError(method_name)

    # Keep original once per process for audit/debugging.
    if not hasattr(
        ConservedFleetScenario,
        "_original_dispatch_fixed_returns_for_comparison",
    ):
        ConservedFleetScenario._original_dispatch_fixed_returns_for_comparison = (
            ConservedFleetScenario._dispatch_fixed_returns
        )

    ConservedFleetScenario._dispatch_fixed_returns = new_method

    return {
        "class": (
            f"{ConservedFleetScenario.__module__}."
            f"{ConservedFleetScenario.__name__}"
        ),
        "patched_method": "_dispatch_fixed_returns",
        "policy": method_name,
        "empty_flight_physics": "_start_empty_reposition (existing implementation)",
    }


# =============================================================================
# Environment factories
# =============================================================================

def make_policy_env_factory(
    *,
    method_name: str,
    env_index: int,
    run_dir: Path,
):
    """
    Pickle-safe closure for Windows SubprocVecEnv.
    """
    def _init():
        install_reposition_patch(method_name)

        return make_env(
            max_time=MAX_TIME,
            log_dir=run_dir / "monitor",
            env_index=env_index,
            person_spawn_file=str(TRAIN_FILE),
            candidate_from_vertiports=CANDIDATES,
            to_vertiport=TO_VERTIPORT,
            enable_logger=False,
            fleet_mode="conserved_closed_loop",
            fleet_size=FLEET_SIZE,
            fleet_assertions=True,
        )()

    return _init


def find_scenario(obj: Any):
    if hasattr(obj, "venv"):
        obj = obj.venv
    if hasattr(obj, "envs") and obj.envs:
        obj = obj.envs[0]

    seen = set()

    for _ in range(60):
        if id(obj) in seen:
            break
        seen.add(id(obj))

        if (
            hasattr(obj, "_all_evtols")
            and hasattr(obj, "vertiports")
            and hasattr(obj, "get_fixed_fleet_diagnostics")
        ):
            return obj

        for attr in ("scenario", "env", "unwrapped"):
            if hasattr(obj, attr):
                nxt = getattr(obj, attr)
                if nxt is not None and nxt is not obj:
                    obj = nxt
                    break
        else:
            break

    raise RuntimeError(
        "Could not locate ConservedFleetScenario below wrapper stack."
    )


def preflight(method_name: str, seed: int, run_dir: Path) -> Dict[str, Any]:
    """
    Fast structural preflight. No training and no full evaluation episode.
    """
    raw = DummyVecEnv(
        [
            make_policy_env_factory(
                method_name=method_name,
                env_index=999,
                run_dir=run_dir / "_preflight",
            )
        ]
    )

    try:
        raw.seed(seed)
        raw.reset()

        scenario = find_scenario(raw)
        diag = scenario.get_fixed_fleet_diagnostics()

        if int(diag.get("fleet_size", -1)) != FLEET_SIZE:
            raise AssertionError(
                f"Expected N={FLEET_SIZE}, got {diag}"
            )

        alloc = diag.get("initial_allocation", {})
        if (
            int(alloc.get("0", -1)) != 12
            or int(alloc.get("1", -1)) != 4
        ):
            raise AssertionError(
                f"Expected initial V0/V1=12/4, got {alloc}"
            )

        scenario._assert_fixed_fleet()

        patch = install_reposition_patch(method_name)

        return {
            "method": method_name,
            "fleet_diagnostics": diag,
            "patch": patch,
            "queue_v0_at_reset": queue_length(scenario, 0),
            "queue_v1_at_reset": queue_length(scenario, 1),
            "aircraft_capacity_detected": aircraft_capacity(scenario),
        }

    finally:
        raw.close()


def build_train_env(
    *,
    method_name: str,
    seed: int,
    run_dir: Path,
) -> VecNormalize:
    factories = [
        make_policy_env_factory(
            method_name=method_name,
            env_index=i,
            run_dir=run_dir,
        )
        for i in range(N_ENVS)
    ]

    raw = SubprocVecEnv(
        factories,
        start_method="spawn",
    )
    raw.seed(seed)

    env = VecNormalize(
        raw,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )

    return env


# =============================================================================
# PPO
# =============================================================================

class PassiveTrainingScalarCallback(BaseCallback):
    """
    Passive logger only. No validation / no replay.
    """

    def __init__(
        self,
        run_dir: Path,
        interval: int,
        requested_steps: int,
    ):
        super().__init__(verbose=0)
        self.path = run_dir / "training_milestones.csv"
        self.interval = int(interval)
        self.requested_steps = int(requested_steps)
        self.next_mark = int(interval)

    def _on_step(self) -> bool:
        while (
            self.num_timesteps >= self.next_mark
            and self.next_mark <= self.requested_steps
        ):
            values = getattr(
                self.model.logger,
                "name_to_value",
                {},
            ) or {}

            append_csv(
                self.path,
                {
                    "milestone": self.next_mark,
                    "model_num_timesteps": int(self.num_timesteps),
                    "approx_kl": values.get(
                        "train/approx_kl", np.nan
                    ),
                    "clip_fraction": values.get(
                        "train/clip_fraction", np.nan
                    ),
                    "entropy_loss": values.get(
                        "train/entropy_loss", np.nan
                    ),
                    "explained_variance": values.get(
                        "train/explained_variance", np.nan
                    ),
                    "policy_gradient_loss": values.get(
                        "train/policy_gradient_loss", np.nan
                    ),
                    "value_loss": values.get(
                        "train/value_loss", np.nan
                    ),
                    "learning_rate": values.get(
                        "train/learning_rate", np.nan
                    ),
                },
            )

            print(
                f"[{self.next_mark:,}] passive checkpoint marker "
                f"(NO evaluation)",
                flush=True,
            )

            self.next_mark += self.interval

        return True


def build_model(
    *,
    env: VecNormalize,
    seed: int,
    run_dir: Path,
    device: str,
) -> PPO:
    from utilss.encoding import TemporalLSTMExtractor

    policy_kwargs = dict(
        net_arch=dict(
            features_extractor_class=TemporalLSTMExtractor,
            pi=[256, 256],
            vf=[256, 256],
        )
    )

    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=linear_schedule(INITIAL_LR),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
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


# =============================================================================
# One method training run
# =============================================================================

def train_method(
    *,
    method_name: str,
    seed: int,
    requested_steps: int,
    checkpoint_interval: int,
    root: Path,
    device: str,
) -> Dict[str, Any]:
    run_dir = root / method_name
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    env = None
    model = None
    start = time.time()

    expected_rollout_end = (
        math.ceil(requested_steps / GLOBAL_ROLLOUT)
        * GLOBAL_ROLLOUT
    )

    manifest = {
        "experiment": "UAGMC_N16_REPOSITION_COMPARISON",
        "method": method_name,
        "train_seed": seed,
        "requested_timesteps": requested_steps,
        "formal_checkpoint": requested_steps,
        "expected_rollout_aligned_final": expected_rollout_end,
        "fleet_size": FLEET_SIZE,
        "initial_allocation": {"0": 12, "1": 4},
        "reposition_origin": RETURN_HUB,
        "candidate_departure_vertiports": CANDIDATES,
        "service_destination": TO_VERTIPORT,
        "charging_soc_semantics": "unchanged UAGMC / fixed-fleet implementation",
        "passenger_trace": str(TRAIN_FILE),
        "execution": {
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "global_rollout": GLOBAL_ROLLOUT,
            "batch_size": BATCH_SIZE,
            "device": device,
            "no_mid_training_evaluation": True,
        },
        "method_definition": {
            "longest_queue": (
                "Dynamic current-queue rule; simultaneous empty aircraft are "
                "assigned greedily using virtual queue subtraction by one "
                "aircraft load."
            ),
            "vertisync_simple": (
                "Cycle/snapshot approximation: freeze current queues, convert "
                "to required aircraft loads, subtract current ready/inbound "
                "origin supply, and execute the frozen rebalancing deficit plan."
            ),
        }[method_name],
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    write_json(run_dir / "run_manifest.json", manifest)

    try:
        pf = preflight(
            method_name=method_name,
            seed=seed,
            run_dir=run_dir,
        )
        write_json(
            run_dir / "preflight.json",
            pf,
        )

        env = build_train_env(
            method_name=method_name,
            seed=seed,
            run_dir=run_dir,
        )

        model = build_model(
            env=env,
            seed=seed,
            run_dir=run_dir,
            device=device,
        )

        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_interval % N_ENVS != 0:
            raise ValueError(
                f"checkpoint interval {checkpoint_interval} must "
                f"be divisible by n_envs={N_ENVS}"
            )

        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_interval // N_ENVS,
            save_path=str(ckpt_dir),
            name_prefix="uam_ppo",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=0,
        )

        scalar_cb = PassiveTrainingScalarCallback(
            run_dir=run_dir,
            interval=checkpoint_interval,
            requested_steps=requested_steps,
        )

        print("\n" + "=" * 128, flush=True)
        print(
            f"START {method_name.upper()} | "
            f"seed={seed} | requested={requested_steps:,}",
            flush=True,
        )
        print(
            f"N=16 | V0/V1=12/4 | "
            f"{N_ENVS} envs x {N_STEPS} = {GLOBAL_ROLLOUT:,}/rollout | "
            f"batch={BATCH_SIZE} | device={model.device}",
            flush=True,
        )
        print(
            "TRAINING ONLY: no validation, no checkpoint replay",
            flush=True,
        )
        print("=" * 128, flush=True)

        model.learn(
            total_timesteps=requested_steps,
            callback=[checkpoint_cb, scalar_cb],
            progress_bar=False,
            reset_num_timesteps=True,
        )

        actual_steps = int(model.num_timesteps)

        model.save(run_dir / "final_rl_model")
        env.save(run_dir / "final_vec_normalize.pkl")

        formal_model = (
            ckpt_dir
            / f"uam_ppo_{requested_steps}_steps.zip"
        )
        formal_vec = (
            ckpt_dir
            / f"uam_ppo_vecnormalize_{requested_steps}_steps.pkl"
        )

        if not formal_model.exists():
            raise FileNotFoundError(
                f"Missing exact formal checkpoint: {formal_model}"
            )
        if not formal_vec.exists():
            raise FileNotFoundError(
                f"Missing exact formal VecNormalize: {formal_vec}"
            )

        elapsed = time.time() - start

        result = {
            "status": "SUCCESS",
            "method": method_name,
            "train_seed": seed,
            "requested_timesteps": requested_steps,
            "actual_final_timesteps": actual_steps,
            "formal_model": str(formal_model),
            "formal_vecnormalize": str(formal_vec),
            "elapsed_seconds": elapsed,
            "requested_effective_fps": (
                requested_steps / max(elapsed, 1e-9)
            ),
            "cuda_peak_memory_mb": (
                torch.cuda.max_memory_allocated()
                / (1024.0 * 1024.0)
                if torch.cuda.is_available() and device == "cuda"
                else 0.0
            ),
            "finished": datetime.now().isoformat(timespec="seconds"),
        }

        write_json(run_dir / "run_end.json", result)

        print(
            f"DONE {method_name} | nominal={requested_steps:,} | "
            f"rollout-final={actual_steps:,} | "
            f"time={elapsed/60:.2f} min | "
            f"effective={result['requested_effective_fps']:.1f} FPS",
            flush=True,
        )

        return result

    except Exception as exc:
        error = {
            "status": "ERROR",
            "method": method_name,
            "train_seed": seed,
            "requested_timesteps": requested_steps,
            "elapsed_seconds": time.time() - start,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "finished": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(run_dir / "run_end.json", error)
        raise

    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass

        del model
        del env
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "UAGMC N=16: Longest-Queue vs simplified VertiSync "
            "empty-reposition comparison, 800k each"
        )
    )

    p.add_argument(
        "--methods",
        default="longest_queue,vertisync_simple",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
    )
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
    )
    p.add_argument(
        "--output-root",
        default=None,
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    return p.parse_args()


def resolve_device(x: str) -> str:
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if x == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False"
        )

    return x


def main():
    args = parse_args()

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)

    methods = parse_methods(args.methods)
    seed = int(args.seed)
    timesteps = int(args.timesteps)
    checkpoint_interval = int(args.checkpoint_interval)
    device = resolve_device(args.device)

    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    if timesteps % checkpoint_interval != 0:
        raise ValueError(
            "timesteps must be divisible by checkpoint interval"
        )
    if checkpoint_interval % N_ENVS != 0:
        raise ValueError(
            f"checkpoint interval must be divisible by {N_ENVS}"
        )

    if args.output_root:
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
            / (
                f"uagmc_reposition_LQ_vs_VertiSync_"
                f"N16_seed{seed}_800k_{stamp}"
            )
        ).resolve()

    root.mkdir(parents=True, exist_ok=True)

    experiment_manifest = {
        "experiment": "UAGMC_N16_REPOSITION_LQ_VS_VERTISYNC_SIMPLE",
        "methods": methods,
        "train_seed": seed,
        "timesteps_per_method": timesteps,
        "requested_total_budget": timesteps * len(methods),
        "fleet_size": FLEET_SIZE,
        "initial_allocation": {"0": 12, "1": 4},
        "passenger_trace": str(TRAIN_FILE),
        "execution": {
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "global_rollout": GLOBAL_ROLLOUT,
            "batch_size": BATCH_SIZE,
            "device": device,
            "no_mid_training_evaluation": True,
        },
        "comparison_control": (
            "Passenger PPO/reward/observation/batching/charging fixed; "
            "only empty-aircraft dispatch from V2 is changed."
        ),
        "vertisync_scope": (
            "Simplified cycle/snapshot rebalancing approximation, NOT full "
            "VertiSync MILP/slot/separation scheduler."
        ),
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    write_json(
        root / "experiment_manifest.json",
        experiment_manifest,
    )

    print("=" * 128)
    print("UAGMC N=16 AIRCRAFT REPOSITION COMPARISON")
    print("=" * 128)
    print(f"Methods           : {methods}")
    print(f"Train seed        : {seed}")
    print(f"Timesteps/method  : {timesteps:,}")
    print(f"Total requested   : {timesteps * len(methods):,}")
    print(f"Fleet             : 16 (initial V0/V1=12/4)")
    print(f"Fast execution    : 16 envs x 1280, batch=512")
    print(f"Device            : {device}")
    print("Mid-training eval : NONE")
    print(f"Output            : {root}")
    print("=" * 128)

    status_path = root / "serial_status.csv"
    successes = 0
    failures = 0

    for idx, method_name in enumerate(methods, start=1):
        print(
            f"\n### METHOD {idx}/{len(methods)}: {method_name} ###",
            flush=True,
        )

        try:
            result = train_method(
                method_name=method_name,
                seed=seed,
                requested_steps=timesteps,
                checkpoint_interval=checkpoint_interval,
                root=root,
                device=device,
            )

            append_csv(
                status_path,
                {
                    "method": method_name,
                    "status": "SUCCESS",
                    "train_seed": seed,
                    "requested_timesteps": timesteps,
                    "actual_final_timesteps": result[
                        "actual_final_timesteps"
                    ],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "requested_effective_fps": result[
                        "requested_effective_fps"
                    ],
                },
            )
            successes += 1

        except Exception as exc:
            append_csv(
                status_path,
                {
                    "method": method_name,
                    "status": "ERROR",
                    "train_seed": seed,
                    "requested_timesteps": timesteps,
                    "actual_final_timesteps": "",
                    "elapsed_seconds": "",
                    "requested_effective_fps": "",
                    "error": repr(exc),
                },
            )
            failures += 1

            print(traceback.format_exc(), flush=True)

            if not args.continue_on_error:
                raise

    write_json(
        root / "experiment_end.json",
        {
            "status": (
                "SUCCESS"
                if failures == 0
                else "PARTIAL_ERROR"
            ),
            "successes": successes,
            "failures": failures,
            "methods": methods,
            "train_seed": seed,
            "timesteps_per_method": timesteps,
            "finished": datetime.now().isoformat(timespec="seconds"),
        },
    )

    print("\n" + "=" * 128)
    print("REPOSITION COMPARISON TRAINING FINISHED")
    print("=" * 128)
    print(f"Success : {successes}/{len(methods)}")
    print(f"Root    : {root}")
    print(
        "Evaluate checkpoints only AFTER both methods finish.",
        flush=True,
    )
    print("=" * 128)

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
