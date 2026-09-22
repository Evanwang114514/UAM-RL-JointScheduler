# -*- coding: utf-8 -*-
"""
UAGMC controlled complexity ladder: E3 -> E4 -> E5
===================================================

Serial experiment, one file, 1,000,000 PPO timesteps per stage.

Base retained from the current validated branch
-----------------------------------------------
- UAGMC original 2-departure + 1-arrival task
- candidate departure vertiports: V0 / V1
- common service destination: V2
- fixed conserved fleet
- Longest-Queue empty-aircraft reposition
- original passenger trace: train_data/passengers_300.csv
- original reward
- original observation/history
- original PPO/policy architecture and hyperparameters
- original charging / battery semantics
- 1 min UAGMC simulation step
- same train seed for all stages
- 16 SubprocVecEnv, n_steps=1280, batch_size=512
- CUDA PPO update
- checkpoints every 50k
- NO mid-training evaluation

Only the physical complexity is peeled in:

E3_SINGLE_PAX
    Base + single-passenger service:
        4-seat/full-batch service -> exactly 1 passenger per service flight.
    Longest-Queue remains conceptually identical, but one "aircraft load"
    is now one passenger because the service batch size is one.

E4_TURNAROUND
    E3 + fixed turnaround delay:
        every service/reposition arrival undergoes TURNAROUND_DELAY_MIN minutes
        of ground recovery before charging/IDLE availability.
    Turnaround capacity is deliberately unlimited in E4.
    Turnaround state is NOT added to the PPO observation.

E5_PAD
    E4 + finite shared TLOF/pad:
        pad capacity = 1 at V0/V1/V2
        takeoff and landing use the same per-vertiport reservation calendar
        PAD_SEPARATION_MIN separation.
        A flight departs only when both its origin takeoff slot and its
        predicted destination landing slot can be reserved.
        No airborne holding is introduced.
    Pad reservation state is NOT added to the PPO observation.

Scientific control rule
-----------------------
Within E3/E4/E5, the selected fleet size is frozen and identical.

By default --fleet-size auto performs a NO-TRAINING E3 physical-feasibility
scan BEFORE any PPO training and picks the smallest tested fleet that can
complete the fixed 300-passenger trace with a simple static passenger-routing
policy at the requested completion threshold. This calibration is outside the
E3->E5 ladder and is recorded in the manifest.

If you want the strict "keep N=16 even if single-pax becomes capacity-starved"
stress test instead, run:
    --fleet-size 16

The auto calibration exists because changing 4 passengers/flight -> 1 passenger/
flight can reduce physical throughput enough that an N=16 failure would measure
fleet scarcity rather than UAGMC learning.

Implementation notes
--------------------
This file does NOT overwrite:
    at_obj/scenario.py
    at_obj/scenario_fixed_fleet.py
    at_obj/vertiport/vertiport_builder.py

It uses process-local monkey patches. Each SubprocVecEnv process installs the
requested stage before constructing its environment.

The stage patches are fail-fast and preserve the original UAGMC mechanics
outside the explicitly listed changes.

Usage
-----
Place this file in UAGMC-main:

    train_uagmc_E3_E4_E5_serial_1m.py

Default (recommended: auto fleet calibration once, then freeze):
    python train_uagmc_E3_E4_E5_serial_1m.py

Strict N=16 stress version:
    python train_uagmc_E3_E4_E5_serial_1m.py --fleet-size 16

Run only selected stages:
    python train_uagmc_E3_E4_E5_serial_1m.py --stages E3,E4,E5

Outputs
-------
serial_runs/
  uagmc_E3_E4_E5_LQ_1m_seed1_<timestamp>/
    experiment_manifest.json
    fleet_calibration.csv
    fleet_calibration_summary.json
    serial_status.csv
    E3_SINGLE_PAX/
       checkpoints/
       final_rl_model.zip
       final_vec_normalize.pkl
       preflight.json
       run_manifest.json
       run_end.json
       training_milestones.csv
    E4_TURNAROUND/
       ...
    E5_PAD/
       ...
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

from at_obj.scenario_fixed_fleet import ConservedFleetScenario
from at_obj.vertiport.vertiport_builder import VertiportBuilder
from at_obj.evtol.evtol import eVTOL
from at_obj.evtol.vehicle_state import VehicleState


ROOT = Path(__file__).resolve().parent

# =============================================================================
# Frozen experiment configuration
# =============================================================================

CANDIDATES = [0, 1]
TO_VERTIPORT = 2
RETURN_HUB = 2
TRAIN_FILE = ROOT / "train_data" / "passengers_300.csv"

# Keep the same training episode horizon as the previous fast LQ/Sync trainer.
MAX_TIME = 10000
DEFAULT_SEED = 1
DEFAULT_TIMESTEPS = 1_000_000
DEFAULT_CHECKPOINT_INTERVAL = 50_000

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

SERVICE_BATCH_SIZE = 1
TURNAROUND_DELAY_MIN = 3

PAD_CAPACITY = {
    "0": 1,
    "1": 1,
    "2": 1,
}
PAD_SEPARATION_MIN = 1.0

STAGE_ALIASES = {
    "E3": "E3_SINGLE_PAX",
    "E3_SINGLE_PAX": "E3_SINGLE_PAX",
    "E4": "E4_TURNAROUND",
    "E4_TURNAROUND": "E4_TURNAROUND",
    "E5": "E5_PAD",
    "E5_PAD": "E5_PAD",
}
STAGE_ORDER = (
    "E3_SINGLE_PAX",
    "E4_TURNAROUND",
    "E5_PAD",
)

# Physical-feasibility calibration defaults.
AUTO_FLEET_CANDIDATES = [24, 32, 40, 48, 56, 64, 72, 80, 88, 96]
AUTO_STATIC_RATIOS = [0.40, 0.50, 0.60, 0.70]
AUTO_PHASES = [0, 1]
AUTO_COMPLETION_TARGET = 0.98

# =============================================================================
# Capture source methods BEFORE installing any complexity patch
# =============================================================================

_ORIG_VP_UPDATE = VertiportBuilder.update_objects_state
_ORIG_VP_CHARGE = VertiportBuilder.charge_evtols_at_vertiport
_ORIG_EVTOL_STEP = eVTOL.step
_ORIG_SCENARIO_RESET = ConservedFleetScenario.reset
_ORIG_SCENARIO_STEP = ConservedFleetScenario.step
_ORIG_REGISTER_ARRIVED = ConservedFleetScenario._register_arrived_evtol
_ORIG_DISPATCH_RETURNS = ConservedFleetScenario._dispatch_fixed_returns


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
    if isinstance(x, (list, tuple, set)):
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
        w = csv.DictWriter(f, fieldnames=list(serial.keys()))
        if not exists:
            w.writeheader()
        w.writerow(serial)
        f.flush()


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

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
                    out[k] = json.dumps(jsonable(v), ensure_ascii=False)
                else:
                    out[k] = v
            w.writerow(out)


def fnum(x: Any, default=float("nan")) -> float:
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return default


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("integer list cannot be empty")
    return vals


def parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("float list cannot be empty")
    return vals


def parse_stages(text: str) -> List[str]:
    raw = [x.strip().upper() for x in str(text).split(",") if x.strip()]
    if not raw:
        raise ValueError("stage list cannot be empty")

    out = []
    for x in raw:
        if x not in STAGE_ALIASES:
            raise ValueError(
                f"Unknown stage {x}. Valid aliases={sorted(STAGE_ALIASES)}"
            )
        canonical = STAGE_ALIASES[x]
        if canonical not in out:
            out.append(canonical)

    out.sort(key=STAGE_ORDER.index)
    return out


def state_name(evtol: Any) -> str:
    st = getattr(evtol, "state", None)
    if hasattr(st, "name"):
        return str(st.name).upper()
    return str(st).upper()


def passenger_ids(evtol: Any) -> List[Any]:
    value = getattr(evtol, "passenger_ids", [])
    try:
        return list(value or [])
    except Exception:
        return []


def current_vertiport(evtol: Any) -> Optional[int]:
    try:
        return int(getattr(evtol, "current_vertiport_id"))
    except Exception:
        return None


def target_vertiport(evtol: Any) -> Optional[int]:
    try:
        v = getattr(evtol, "target_vertiport_id")
        return None if v is None else int(v)
    except Exception:
        return None


def sorted_evtols(scenario: Any) -> List[Tuple[Any, Any]]:
    data = getattr(scenario, "_all_evtols", {}) or {}
    items = list(data.items())
    items.sort(key=lambda kv: str(kv[0]))
    return items


def queue_length(scenario: Any, vid: int) -> int:
    vp = scenario.vertiports.vertiport_list[str(int(vid))]
    return len(list(getattr(vp, "person_list", []) or []))


def is_turnaround_busy(evtol: Any) -> bool:
    return bool(getattr(evtol, "_e345_turnaround_busy", False))


# =============================================================================
# Pad calendar (E5 only)
# =============================================================================

def _pad_calendar(scenario: Any) -> Dict[str, List[float]]:
    cal = getattr(scenario, "_e345_pad_calendar", None)
    if not isinstance(cal, dict):
        cal = {str(v): [] for v in (0, 1, 2)}
        scenario._e345_pad_calendar = cal
    return cal


def _cleanup_pad_calendar(scenario: Any) -> None:
    cal = _pad_calendar(scenario)
    now = float(getattr(scenario, "time", 0.0))
    cutoff = now - PAD_SEPARATION_MIN - 1e-9
    for vid in list(cal):
        cal[vid] = [
            float(t)
            for t in cal.get(vid, [])
            if float(t) >= cutoff
        ]


def _slot_free(
    scenario: Any,
    vid: str,
    slot_time: float,
) -> bool:
    cal = _pad_calendar(scenario)
    vid = str(vid)
    existing = sorted(float(t) for t in cal.get(vid, []))
    capacity = int(PAD_CAPACITY.get(vid, 1))

    conflicts = sum(
        1
        for t in existing
        if abs(float(slot_time) - t) < PAD_SEPARATION_MIN - 1e-12
    )
    return conflicts < capacity


def _reserve_pad_pair(
    scenario: Any,
    origin: str,
    destination: str,
    departure_time: float,
    flight_time_min: float,
) -> Optional[List[Tuple[str, float]]]:
    """
    Shared TLOF reservation:
      - origin departure slot
      - predicted destination landing slot

    If either is unavailable, reserve nothing and return None.
    """
    origin = str(origin)
    destination = str(destination)
    t0 = float(departure_time)
    t1 = float(departure_time) + float(flight_time_min)

    if not _slot_free(scenario, origin, t0):
        return None
    if not _slot_free(scenario, destination, t1):
        return None

    cal = _pad_calendar(scenario)
    cal.setdefault(origin, []).append(t0)
    cal.setdefault(destination, []).append(t1)

    scenario._e345_stats["pad_departure_reservations"] += 1
    scenario._e345_stats["pad_landing_reservations"] += 1

    return [(origin, t0), (destination, t1)]


def _rollback_pad_tokens(
    scenario: Any,
    tokens: Optional[List[Tuple[str, float]]],
) -> None:
    if not tokens:
        return
    cal = _pad_calendar(scenario)
    for vid, t in tokens:
        arr = cal.get(str(vid), [])
        for i, value in enumerate(arr):
            if abs(float(value) - float(t)) <= 1e-12:
                arr.pop(i)
                break


# =============================================================================
# Turnaround lifecycle (E4/E5)
# =============================================================================

def _release_due_turnaround(scenario: Any) -> None:
    now = float(getattr(scenario, "time", 0.0))

    for _, evtol in sorted_evtols(scenario):
        if not is_turnaround_busy(evtol):
            continue

        release_time = float(
            getattr(evtol, "_e345_turnaround_release_time", math.inf)
        )

        if now + 1e-12 < release_time:
            continue

        post_state = getattr(
            evtol,
            "_e345_post_turnaround_state",
            VehicleState.CHARGING,
        )

        evtol._e345_turnaround_busy = False
        evtol._e345_turnaround_release_time = None
        evtol.state = post_state

        scenario._e345_stats["turnaround_releases"] += 1


def _register_arrived_with_turnaround(
    self: ConservedFleetScenario,
    evtol: Any,
):
    # Reuse fixed-fleet registration first.
    _ORIG_REGISTER_ARRIVED(self, evtol)

    stage = str(getattr(self, "_e345_stage", "E3_SINGLE_PAX"))
    if stage not in ("E4_TURNAROUND", "E5_PAD"):
        return

    # Preserve what fixed-fleet logic wanted after landing (IDLE vs CHARGING),
    # but hide the aircraft behind WAITING for the turnaround duration.
    post_state = evtol.state

    evtol._e345_post_turnaround_state = post_state
    evtol._e345_turnaround_busy = True
    evtol._e345_turnaround_release_time = (
        float(getattr(self, "time", 0.0))
        + float(TURNAROUND_DELAY_MIN)
    )

    # WAITING already exists in UAGMC VehicleState and is not a new observed
    # turnaround feature. Our service/reposition logic explicitly excludes
    # _e345_turnaround_busy aircraft even though eVTOL.is_available() normally
    # treats WAITING as available.
    evtol.state = VehicleState.WAITING

    self._e345_stats["turnaround_starts"] += 1


def _evtol_step_with_turnaround(self: eVTOL, delta_time_min: float):
    if is_turnaround_busy(self):
        # No flight/charging/IDLE transition during turnaround.
        self.just_arrived = False
        self.just_departed = False
        return
    return _ORIG_EVTOL_STEP(self, delta_time_min)


def _charge_with_turnaround(
    self: VertiportBuilder,
    time_step_min: float = 1.0,
):
    """
    Same UAGMC charging update, except turnaround-busy aircraft do not charge.
    """
    for _vid, evtol_list in self.evtols_at_vertiport.items():
        for e in evtol_list:
            if is_turnaround_busy(e):
                continue

            if state_name(e) == "CHARGING":
                e.battery_kwh += (
                    e.spec.charge_rate_kwh_per_min
                    * float(time_step_min)
                )
                e.battery_kwh = min(
                    e.battery_kwh,
                    e.spec.battery_capacity_kwh,
                )
                if e.battery_kwh >= e.spec.battery_capacity_kwh:
                    e.state = VehicleState.IDLE


# =============================================================================
# E3 single-passenger service dispatch (+ E4/E5 availability/resource gates)
# =============================================================================

def _single_pax_update_objects_state(
    self: VertiportBuilder,
    time: int,
):
    """
    Derived from public UAGMC VertiportBuilder.update_objects_state.

    Explicit changes:
      1) dispatch threshold = 1 waiting passenger
      2) exactly 1 passenger bound to a service flight
      3) turnaround-busy aircraft are unavailable
      4) E5 additionally requires origin+destination pad reservations

    Charging and local eVTOL stepping remain in the same order/semantics.
    """
    self.time = time
    scenario = getattr(self, "_e345_scenario", None)

    if scenario is None:
        raise RuntimeError(
            "E3/E4/E5 patch cannot locate owning scenario from VertiportBuilder"
        )

    stage = str(getattr(scenario, "_e345_stage", "E3_SINGLE_PAX"))

    for vertiport_id, vertiport in self.vertiport_list.items():
        # Keep UAGMC availability semantics, then remove turnaround-busy units.
        available_evtols = [
            e
            for e in self.evtols_at_vertiport[str(vertiport_id)]
            if e.is_available() and not is_turnaround_busy(e)
        ]

        for e in available_evtols:
            if int(getattr(vertiport, "wait_person", 0)) < SERVICE_BATCH_SIZE:
                continue

            dest_vertiport_id = str(TO_VERTIPORT)

            distance_km = self._calculate_distance(
                vertiport.vertiport_position,
                self.vertiport_list[dest_vertiport_id].vertiport_position,
            )

            required_energy = (
                distance_km * e.spec.energy_consumption_kwh_per_km
            )

            # Preserve public-source service energy test.
            if e.battery_kwh < required_energy:
                continue

            pad_tokens = None
            if stage == "E5_PAD":
                flight_time_min = (
                    distance_km / e.spec.max_speed * 60.0
                )
                pad_tokens = _reserve_pad_pair(
                    scenario=scenario,
                    origin=str(vertiport_id),
                    destination=dest_vertiport_id,
                    departure_time=float(time),
                    flight_time_min=flight_time_min,
                )
                if pad_tokens is None:
                    scenario._e345_stats["service_pad_blocks"] += 1
                    continue

            passenger_ids_now = list(
                vertiport.person_list[:SERVICE_BATCH_SIZE]
            )

            if not passenger_ids_now:
                _rollback_pad_tokens(scenario, pad_tokens)
                continue

            try:
                # Queue bookkeeping: exactly one passenger.
                vertiport.person_list = vertiport.person_list[
                    SERVICE_BATCH_SIZE:
                ]
                vertiport.wait_person -= SERVICE_BATCH_SIZE
                vertiport.leave_person += SERVICE_BATCH_SIZE

                e.start_flight(
                    dest_vertiport_id=dest_vertiport_id,
                    distance_km=distance_km,
                    passenger_ids=passenger_ids_now,
                )

                self.evtols_at_vertiport[str(vertiport_id)].remove(e)

                scenario._e345_stats["single_pax_service_departures"] += 1

            except Exception:
                # Restore queue if flight creation fails.
                vertiport.person_list = (
                    passenger_ids_now + list(vertiport.person_list)
                )
                vertiport.wait_person += SERVICE_BATCH_SIZE
                vertiport.leave_person -= SERVICE_BATCH_SIZE
                _rollback_pad_tokens(scenario, pad_tokens)
                raise

            # Preserve public UAGMC: at most one service departure from a
            # vertiport during one simulation step.
            break

    # Preserve source order: locally registered aircraft step, then charging.
    for evtol_list in self.evtols_at_vertiport.values():
        for e in evtol_list:
            e.step(1.0)

    self.charge_evtols_at_vertiport(time_step_min=1.0)


# =============================================================================
# Longest Queue empty reposition, adjusted only for actual E3 service load
# =============================================================================

def _ready_supply_at_origin(
    scenario: ConservedFleetScenario,
    vid: int,
) -> int:
    total = 0

    for _, evtol in sorted_evtols(scenario):
        if is_turnaround_busy(evtol):
            continue

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


def _start_empty_reposition_checked(
    scenario: ConservedFleetScenario,
    evtol: Any,
    origin: str,
    destination: str,
) -> bool:
    """
    Reuse fixed-fleet empty-flight physics.
    E5 only adds pad reservation before the same physical reposition call.
    """
    stage = str(getattr(scenario, "_e345_stage", "E3_SINGLE_PAX"))

    origin = str(origin)
    destination = str(destination)

    distance = scenario._distance_between_vertiports(
        origin,
        destination,
    )
    energy = distance * evtol.spec.energy_consumption_kwh_per_km

    # Same energy gate as ConservedFleetScenario._start_empty_reposition.
    if evtol.battery_kwh < energy:
        evtol.state = VehicleState.CHARGING
        return False

    pad_tokens = None

    if stage == "E5_PAD":
        flight_time_min = (
            distance / evtol.spec.max_speed * 60.0
        )
        pad_tokens = _reserve_pad_pair(
            scenario=scenario,
            origin=origin,
            destination=destination,
            departure_time=float(getattr(scenario, "time", 0.0)),
            flight_time_min=flight_time_min,
        )

        if pad_tokens is None:
            scenario._e345_stats["reposition_pad_blocks"] += 1
            return False

    try:
        ok = bool(
            scenario._start_empty_reposition(
                evtol,
                origin,
                destination,
            )
        )
    except Exception:
        _rollback_pad_tokens(scenario, pad_tokens)
        raise

    if not ok:
        _rollback_pad_tokens(scenario, pad_tokens)

    return ok


def _dispatch_longest_queue_single_load(
    self: ConservedFleetScenario,
):
    """
    Same responsive LQ logic as the preceding LQ experiment:
      - current queue only
      - longest queue first
      - tie-break by less ready supply, then station id

    The virtual queue decrement changes from aircraft seat capacity to ONE
    because E3 explicitly changes a physical aircraft service load to one
    passenger. This is not a different heuristic; it is the same "one
    aircraft-load" subtraction under the new service definition.
    """
    arrival_vid = str(
        getattr(self, "fixed_fleet_return_vertiport", RETURN_HUB)
    )

    local = list(
        self.vertiports.evtols_at_vertiport.get(arrival_vid, [])
    )

    available = [
        e
        for e in local
        if (
            state_name(e) == "IDLE"
            and not passenger_ids(e)
            and not is_turnaround_busy(e)
        )
    ]

    if not available:
        return

    available.sort(key=lambda x: str(getattr(x, "id", "")))

    virtual_q = {
        vid: queue_length(self, vid)
        for vid in CANDIDATES
    }

    for evtol in available:
        target = max(
            CANDIDATES,
            key=lambda v: (
                virtual_q[v],
                -_ready_supply_at_origin(self, v),
                -v,
            ),
        )

        if virtual_q[target] <= 0:
            break

        ok = _start_empty_reposition_checked(
            scenario=self,
            evtol=evtol,
            origin=arrival_vid,
            destination=str(target),
        )

        if not ok:
            # If V2 pad/energy blocks the first selected aircraft, the same
            # physical constraint will generally block another simultaneous
            # departure as well. Leave it for the next simulation step.
            break

        virtual_q[target] = max(
            0,
            virtual_q[target] - SERVICE_BATCH_SIZE,
        )


# =============================================================================
# Scenario reset / step wrappers
# =============================================================================

def _scenario_reset_stage(
    self: ConservedFleetScenario,
    seed=None,
):
    # Stage is assigned by install_stage_to_env after construction; preserve
    # class-level requested value across reset.
    stage = str(
        getattr(
            self,
            "_e345_stage",
            getattr(
                ConservedFleetScenario,
                "_e345_requested_stage",
                "E3_SINGLE_PAX",
            ),
        )
    )

    self._e345_stage = stage
    self._e345_pad_calendar = {
        "0": [],
        "1": [],
        "2": [],
    }
    self._e345_stats = {
        "single_pax_service_departures": 0,
        "turnaround_starts": 0,
        "turnaround_releases": 0,
        "pad_departure_reservations": 0,
        "pad_landing_reservations": 0,
        "service_pad_blocks": 0,
        "reposition_pad_blocks": 0,
    }

    result = _ORIG_SCENARIO_RESET(self, seed=seed)

    self.vertiports._e345_scenario = self

    return result


def _scenario_step_stage(
    self: ConservedFleetScenario,
    action=None,
):
    self.vertiports._e345_scenario = self

    if str(getattr(self, "_e345_stage", "")) in (
        "E4_TURNAROUND",
        "E5_PAD",
    ):
        _release_due_turnaround(self)

    if str(getattr(self, "_e345_stage", "")) == "E5_PAD":
        _cleanup_pad_calendar(self)

    return _ORIG_SCENARIO_STEP(self, action=action)


# =============================================================================
# Patch installation
# =============================================================================

def restore_source_methods() -> None:
    VertiportBuilder.update_objects_state = _ORIG_VP_UPDATE
    VertiportBuilder.charge_evtols_at_vertiport = _ORIG_VP_CHARGE
    eVTOL.step = _ORIG_EVTOL_STEP
    ConservedFleetScenario.reset = _ORIG_SCENARIO_RESET
    ConservedFleetScenario.step = _ORIG_SCENARIO_STEP
    ConservedFleetScenario._register_arrived_evtol = _ORIG_REGISTER_ARRIVED
    ConservedFleetScenario._dispatch_fixed_returns = _ORIG_DISPATCH_RETURNS


def install_stage_patch(stage: str) -> Dict[str, Any]:
    if stage not in STAGE_ORDER:
        raise ValueError(stage)

    restore_source_methods()

    # Used by reset before per-instance attribute can exist.
    ConservedFleetScenario._e345_requested_stage = stage

    VertiportBuilder.update_objects_state = _single_pax_update_objects_state
    VertiportBuilder.charge_evtols_at_vertiport = _charge_with_turnaround
    eVTOL.step = _evtol_step_with_turnaround

    ConservedFleetScenario.reset = _scenario_reset_stage
    ConservedFleetScenario.step = _scenario_step_stage
    ConservedFleetScenario._dispatch_fixed_returns = (
        _dispatch_longest_queue_single_load
    )

    if stage in ("E4_TURNAROUND", "E5_PAD"):
        ConservedFleetScenario._register_arrived_evtol = (
            _register_arrived_with_turnaround
        )

    return {
        "stage": stage,
        "service_batch_size": SERVICE_BATCH_SIZE,
        "aircraft_reposition": "responsive_longest_queue",
        "turnaround_delay_min": (
            TURNAROUND_DELAY_MIN
            if stage in ("E4_TURNAROUND", "E5_PAD")
            else 0
        ),
        "turnaround_capacity": (
            "unlimited"
            if stage in ("E4_TURNAROUND", "E5_PAD")
            else "not_applicable"
        ),
        "pad_enabled": stage == "E5_PAD",
        "pad_capacity": (
            dict(PAD_CAPACITY)
            if stage == "E5_PAD"
            else None
        ),
        "pad_separation_min": (
            PAD_SEPARATION_MIN
            if stage == "E5_PAD"
            else None
        ),
        "observation_augmented": False,
        "reward_changed": False,
    }


# =============================================================================
# Environment construction
# =============================================================================

def make_stage_env_factory(
    *,
    stage: str,
    fleet_size: int,
    env_index: int,
    run_dir: Path,
    max_time: int = MAX_TIME,
):
    def _init():
        install_stage_patch(stage)

        env = make_env(
            max_time=max_time,
            log_dir=run_dir / "monitor",
            env_index=env_index,
            person_spawn_file=str(TRAIN_FILE),
            candidate_from_vertiports=CANDIDATES,
            to_vertiport=TO_VERTIPORT,
            enable_logger=False,
            fleet_mode="conserved_closed_loop",
            fleet_size=int(fleet_size),
            fleet_assertions=True,
        )()

        # Find the scenario once so the requested stage is explicit per object.
        sc = find_scenario(env)
        sc._e345_stage = stage
        sc.vertiports._e345_scenario = sc

        return env

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
        f"Could not locate ConservedFleetScenario; stopped at {type(obj)}"
    )


def find_wrapper(obj: Any):
    seen = set()

    for _ in range(50):
        if id(obj) in seen:
            break
        seen.add(id(obj))

        if all(
            hasattr(obj, k)
            for k in ("state", "encoder", "decoder", "env")
        ):
            return obj

        if hasattr(obj, "env"):
            nxt = getattr(obj, "env")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue

        if hasattr(obj, "unwrapped"):
            nxt = getattr(obj, "unwrapped")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue

        break

    raise RuntimeError(
        f"Could not locate UAMRLWrapper; stopped at {type(obj)}"
    )


def unwrap_step(out):
    if not isinstance(out, tuple):
        raise RuntimeError(
            f"Unexpected env.step return type: {type(out)}"
        )

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return (
            obs,
            reward,
            bool(terminated),
            bool(truncated),
            info,
        )

    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, bool(done), False, info

    raise RuntimeError(
        f"Unexpected env.step tuple length: {len(out)}"
    )


# =============================================================================
# Structural preflight
# =============================================================================

def run_preflight(
    *,
    stage: str,
    fleet_size: int,
    seed: int,
    run_dir: Path,
) -> Dict[str, Any]:
    raw = DummyVecEnv(
        [
            make_stage_env_factory(
                stage=stage,
                fleet_size=fleet_size,
                env_index=999,
                run_dir=run_dir / "_preflight",
            )
        ]
    )

    try:
        raw.seed(seed)
        raw.reset()

        scenario = find_scenario(raw)

        if int(scenario.fixed_fleet_size) != int(fleet_size):
            raise AssertionError(
                f"fleet mismatch: {scenario.fixed_fleet_size} != {fleet_size}"
            )

        scenario._assert_fixed_fleet()

        # A few no-learning steps catch lifecycle / monkey-patch failures.
        for _ in range(25):
            action = np.array([0], dtype=np.int64)
            _obs, _reward, done, infos = raw.step(action)
            scenario._assert_fixed_fleet()
            if bool(done[0]):
                break

        diag = scenario.get_fixed_fleet_diagnostics()

        return {
            "stage": stage,
            "fleet_size": fleet_size,
            "patch": install_stage_patch(stage),
            "fixed_fleet_diagnostics": diag,
            "stage_stats": dict(
                getattr(scenario, "_e345_stats", {})
            ),
        }

    finally:
        raw.close()
        restore_source_methods()


# =============================================================================
# No-training E3 fleet calibration
# =============================================================================

class StaticRatioPolicy:
    def __init__(self, p_v0: float, phase: int):
        self.p_v0 = float(p_v0)
        self.acc = (
            int(phase) * 0.3819660112501051
        ) % 1.0
        self.counts = Counter()

    def choose(self, wrapper: Any) -> int:
        state = getattr(wrapper, "state", None)
        waiting = (
            state.get("waiting_decisions", [])
            if isinstance(state, dict)
            else []
        )

        if not waiting:
            return 0

        if self.p_v0 <= 0:
            action = 1
        elif self.p_v0 >= 1:
            action = 0
        else:
            self.acc += self.p_v0
            if self.acc >= 1.0:
                action = 0
                self.acc -= 1.0
            else:
                action = 1

        self.counts[action] += 1
        return action


def _completion_and_awt(scenario: Any) -> Tuple[float, float, int]:
    pobj = getattr(scenario, "persons", None)
    persons = getattr(pobj, "persons", {}) if pobj is not None else {}
    persons = persons or {}

    finished = {
        str(x)
        for x in (getattr(scenario, "finished_ids", []) or [])
    }

    waits = []
    n_finished = 0

    for pid_raw, p in persons.items():
        pid = str(pid_raw)
        is_finished = (
            pid in finished
            or str(getattr(p, "state", "")).lower() == "finished"
        )

        if not is_finished:
            continue

        n_finished += 1

        stats = getattr(p, "time_stats", {}) or {}
        w = fnum(stats.get("wait_uam", np.nan))
        if np.isfinite(w):
            waits.append(w)

    n = len(persons)
    completion = (
        n_finished / n
        if n > 0
        else float("nan")
    )
    awt = (
        float(np.mean(waits))
        if waits
        else float("nan")
    )

    return completion, awt, n_finished


def run_static_calibration_episode(
    *,
    fleet_size: int,
    p_v0: float,
    phase: int,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    install_stage_patch("E3_SINGLE_PAX")

    env = make_stage_env_factory(
        stage="E3_SINGLE_PAX",
        fleet_size=fleet_size,
        env_index=(
            fleet_size * 10000
            + int(round(p_v0 * 100)) * 10
            + phase
        ),
        run_dir=out_dir,
        max_time=MAX_TIME,
    )()

    try:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

        wrapper = find_wrapper(env)
        scenario = find_scenario(env)
        policy = StaticRatioPolicy(p_v0, phase)

        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            action = policy.choose(wrapper)
            _obs, _reward, terminated, truncated, _info = unwrap_step(
                env.step(action)
            )

            scenario._assert_fixed_fleet()

            steps += 1
            if steps > MAX_TIME + 100:
                raise RuntimeError(
                    f"calibration episode exceeded guard: {steps}"
                )

        completion, awt, n_finished = _completion_and_awt(
            scenario
        )

        return {
            "fleet_size": int(fleet_size),
            "p_v0": float(p_v0),
            "phase": int(phase),
            "seed": int(seed),
            "completion_rate": completion,
            "n_finished": n_finished,
            "AWT": awt,
            "env_steps": steps,
            "stage": "E3_SINGLE_PAX",
            "policy": "static_ratio_no_training",
        }

    finally:
        try:
            env.close()
        except Exception:
            pass
        restore_source_methods()


def calibrate_fleet_size(
    *,
    candidates: Sequence[int],
    ratios: Sequence[float],
    phases: Sequence[int],
    seed: int,
    output_root: Path,
    completion_target: float,
) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 128)
    print("NO-TRAINING E3 FLEET CALIBRATION")
    print("=" * 128)
    print(f"Candidates        : {list(candidates)}")
    print(f"Static P(V0)      : {list(ratios)}")
    print(f"Phases            : {list(phases)}")
    print(f"Completion target : {completion_target:.4f}")
    print("Selection rule    : smallest fleet with one static ratio meeting")
    print("                    completion target across ALL tested phases")
    print("=" * 128)

    selected = None
    selected_ratio = None

    for fleet_size in candidates:
        ratio_phase_rows: Dict[float, List[Dict[str, Any]]] = {
            float(p): []
            for p in ratios
        }

        for p in ratios:
            for phase in phases:
                row = run_static_calibration_episode(
                    fleet_size=int(fleet_size),
                    p_v0=float(p),
                    phase=int(phase),
                    seed=int(seed),
                    out_dir=output_root / "_fleet_calibration_monitor",
                )
                rows.append(row)
                ratio_phase_rows[float(p)].append(row)

                print(
                    f"N={fleet_size:>3} | P(V0)={p:.2f} | phase={phase} | "
                    f"finish={100*row['completion_rate']:.2f}% | "
                    f"AWT={row['AWT']:.3f}",
                    flush=True,
                )

        feasible_ratios = []

        for p, group in ratio_phase_rows.items():
            if group and all(
                float(r["completion_rate"])
                >= float(completion_target)
                for r in group
            ):
                feasible_ratios.append(p)

        if feasible_ratios:
            # Prefer the feasible ratio closest to 0.5; ratio is only for
            # physical calibration and is not used by PPO training.
            selected_ratio = min(
                feasible_ratios,
                key=lambda x: (abs(x - 0.5), x),
            )
            selected = int(fleet_size)
            break

    if selected is None:
        raise RuntimeError(
            "Auto fleet calibration found no feasible fleet. "
            "Expand --fleet-candidates or lower the explicitly chosen "
            "--completion-target after inspecting fleet_calibration.csv."
        )

    summary = {
        "selected_fleet_size": selected,
        "selected_static_ratio_for_calibration_only": selected_ratio,
        "completion_target": completion_target,
        "fleet_candidates_tested": list(candidates),
        "static_ratios_tested": list(ratios),
        "phases_tested": list(phases),
        "selection_rule": (
            "smallest fleet for which at least one deterministic static "
            "passenger-routing ratio reaches the completion target in every "
            "tested phase; PPO performance is not used"
        ),
        "note": (
            "The selected fleet is frozen unchanged for E3, E4, and E5."
        ),
    }

    print("-" * 128)
    print(
        f"SELECTED FLEET = {selected} "
        f"(calibration-only static P(V0)={selected_ratio:.2f})"
    )
    print("=" * 128)

    return selected, rows, summary


# =============================================================================
# PPO environment/model
# =============================================================================

def build_train_env(
    *,
    stage: str,
    fleet_size: int,
    seed: int,
    run_dir: Path,
) -> VecNormalize:
    factories = [
        make_stage_env_factory(
            stage=stage,
            fleet_size=fleet_size,
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

    return VecNormalize(
        raw,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )


def build_model(
    *,
    env: VecNormalize,
    seed: int,
    run_dir: Path,
    device: str,
) -> PPO:
    from utilss.encoding import TemporalLSTMExtractor

    # Keep the exact configuration used in the preceding fast experiments.
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


class PassiveTrainingScalarCallback(BaseCallback):
    """
    Passive only:
      - records PPO logger scalars
      - performs NO evaluation / replay / env reset
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
            values = (
                getattr(self.model.logger, "name_to_value", {})
                or {}
            )

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
                f"[{self.next_mark:,}] passive milestone "
                f"(NO evaluation)",
                flush=True,
            )

            self.next_mark += self.interval

        return True


# =============================================================================
# One stage training
# =============================================================================

def train_stage(
    *,
    stage: str,
    fleet_size: int,
    seed: int,
    requested_steps: int,
    checkpoint_interval: int,
    root: Path,
    device: str,
) -> Dict[str, Any]:
    run_dir = root / stage
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    env = None
    model = None
    started = time.time()

    expected_rollout_end = (
        math.ceil(requested_steps / GLOBAL_ROLLOUT)
        * GLOBAL_ROLLOUT
    )

    patch = install_stage_patch(stage)

    manifest = {
        "experiment": "UAGMC_E3_E4_E5_CONTROLLED_COMPLEXITY",
        "stage": stage,
        "train_seed": seed,
        "requested_timesteps": requested_steps,
        "expected_rollout_aligned_final": expected_rollout_end,
        "formal_checkpoint": requested_steps,
        "fleet_size": fleet_size,
        "candidate_departure_vertiports": CANDIDATES,
        "service_destination": TO_VERTIPORT,
        "passenger_trace": str(TRAIN_FILE),
        "max_time": MAX_TIME,
        "physical_patch": patch,
        "frozen_across_E3_E4_E5": {
            "aircraft_reposition": "Longest Queue",
            "reward": "unchanged UAGMC",
            "observation": "unchanged UAGMC",
            "history": "unchanged UAGMC",
            "charging_battery": "unchanged UAGMC",
            "map": "original UAGMC 2+1",
            "passenger_trace": str(TRAIN_FILE),
            "PPO": {
                "n_envs": N_ENVS,
                "n_steps": N_STEPS,
                "global_rollout": GLOBAL_ROLLOUT,
                "batch_size": BATCH_SIZE,
                "n_epochs": N_EPOCHS,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "clip_range": CLIP_RANGE,
                "ent_coef": ENT_COEF,
                "vf_coef": VF_COEF,
                "max_grad_norm": MAX_GRAD_NORM,
                "initial_lr": INITIAL_LR,
                "lr_schedule": "linear",
            },
        },
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    write_json(
        run_dir / "run_manifest.json",
        manifest,
    )

    try:
        pf = run_preflight(
            stage=stage,
            fleet_size=fleet_size,
            seed=seed,
            run_dir=run_dir,
        )
        write_json(
            run_dir / "preflight.json",
            pf,
        )

        env = build_train_env(
            stage=stage,
            fleet_size=fleet_size,
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

        print("\n" + "=" * 128)
        print(
            f"START {stage} | seed={seed} | "
            f"fleet={fleet_size} | requested={requested_steps:,}"
        )
        print("=" * 128)
        print(
            f"16 envs x 1280 = {GLOBAL_ROLLOUT:,}/rollout | "
            f"batch=512 | device={model.device}"
        )
        print(
            "Passenger PPO / reward / observation / map / trace frozen."
        )
        print(
            "Training only: NO mid-training validation or checkpoint replay."
        )
        print("=" * 128)

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

        elapsed = time.time() - started

        result = {
            "status": "SUCCESS",
            "stage": stage,
            "fleet_size": fleet_size,
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

        write_json(
            run_dir / "run_end.json",
            result,
        )

        print(
            f"DONE {stage} | nominal={requested_steps:,} | "
            f"rollout-final={actual_steps:,} | "
            f"time={elapsed/60:.2f} min | "
            f"effective={result['requested_effective_fps']:.1f} FPS",
            flush=True,
        )

        return result

    except Exception as exc:
        error = {
            "status": "ERROR",
            "stage": stage,
            "fleet_size": fleet_size,
            "train_seed": seed,
            "requested_timesteps": requested_steps,
            "elapsed_seconds": time.time() - started,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "finished": datetime.now().isoformat(timespec="seconds"),
        }

        write_json(
            run_dir / "run_end.json",
            error,
        )

        raise

    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass

        model = None
        env = None

        restore_source_methods()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Serial E3/E4/E5 UAGMC complexity ladder, 1M PPO steps each"
        )
    )

    p.add_argument(
        "--stages",
        default="E3,E4,E5",
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
        "--fleet-size",
        default="auto",
        help=(
            "'auto' (recommended) or integer. Auto calibrates once on E3 "
            "with NO PPO training, then freezes that fleet for E3/E4/E5."
        ),
    )
    p.add_argument(
        "--fleet-candidates",
        default="24,32,40,48,56,64,72,80,88,96",
    )
    p.add_argument(
        "--static-ratios",
        default="0.40,0.50,0.60,0.70",
    )
    p.add_argument(
        "--calibration-phases",
        default="0,1",
    )
    p.add_argument(
        "--completion-target",
        type=float,
        default=AUTO_COMPLETION_TARGET,
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

    stages = parse_stages(args.stages)
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

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_root:
        root = Path(args.output_root).expanduser()
        if not root.is_absolute():
            root = (ROOT / root).resolve()
        else:
            root = root.resolve()
    else:
        root = (
            ROOT
            / "serial_runs"
            / f"uagmc_E3_E4_E5_LQ_1m_seed{seed}_{stamp}"
        ).resolve()

    root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # One-time fleet selection BEFORE the ladder
    # -------------------------------------------------------------------------
    calibration_rows: List[Dict[str, Any]] = []

    if str(args.fleet_size).strip().lower() == "auto":
        fleet_candidates = parse_int_list(args.fleet_candidates)
        static_ratios = parse_float_list(args.static_ratios)
        phases = parse_int_list(args.calibration_phases)

        if any(n <= 0 for n in fleet_candidates):
            raise ValueError("fleet candidates must be positive")

        # Preserve exact 3:1 initial allocation when possible.
        if any(n % 4 != 0 for n in fleet_candidates):
            raise ValueError(
                "For controlled comparison, auto fleet candidates must be "
                "multiples of 4 so the original V0:V1=3:1 allocation is exact."
            )

        fleet_size, calibration_rows, calibration_summary = (
            calibrate_fleet_size(
                candidates=fleet_candidates,
                ratios=static_ratios,
                phases=phases,
                seed=seed,
                output_root=root,
                completion_target=float(args.completion_target),
            )
        )

        write_csv(
            root / "fleet_calibration.csv",
            calibration_rows,
        )
        write_json(
            root / "fleet_calibration_summary.json",
            calibration_summary,
        )

        fleet_selection_mode = "no_training_E3_auto_calibration"

    else:
        fleet_size = int(args.fleet_size)
        if fleet_size <= 0:
            raise ValueError("fleet size must be positive")

        calibration_summary = {
            "selected_fleet_size": fleet_size,
            "selection_rule": "user_explicit",
            "note": (
                "No auto capacity normalization was performed."
            ),
        }

        write_json(
            root / "fleet_calibration_summary.json",
            calibration_summary,
        )

        fleet_selection_mode = "user_explicit"

    # -------------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------------
    experiment_manifest = {
        "experiment": "UAGMC_CONTROLLED_COMPLEXITY_E3_E4_E5",
        "stages": stages,
        "train_seed": seed,
        "timesteps_per_stage": timesteps,
        "requested_total_training_budget": (
            timesteps * len(stages)
        ),
        "fleet_size_frozen_for_all_stages": fleet_size,
        "fleet_selection_mode": fleet_selection_mode,
        "passenger_trace": str(TRAIN_FILE),
        "candidate_departures": CANDIDATES,
        "destination": TO_VERTIPORT,
        "stage_definition": {
            "E3_SINGLE_PAX": {
                "single_passenger_service": True,
                "turnaround": False,
                "finite_pad": False,
            },
            "E4_TURNAROUND": {
                "single_passenger_service": True,
                "turnaround": {
                    "delay_min": TURNAROUND_DELAY_MIN,
                    "capacity": "unlimited",
                },
                "finite_pad": False,
            },
            "E5_PAD": {
                "single_passenger_service": True,
                "turnaround": {
                    "delay_min": TURNAROUND_DELAY_MIN,
                    "capacity": "unlimited",
                },
                "finite_pad": {
                    "capacity": PAD_CAPACITY,
                    "separation_min": PAD_SEPARATION_MIN,
                    "shared_takeoff_landing": True,
                    "reservation": True,
                    "airborne_holding": False,
                },
            },
        },
        "frozen_controls": {
            "map": "original UAGMC 2+1",
            "aircraft_reposition": "Longest Queue",
            "reward": "original UAGMC unchanged",
            "observation": "original UAGMC unchanged",
            "history": "original UAGMC unchanged",
            "charging_battery": "original UAGMC unchanged",
            "max_time": MAX_TIME,
            "PPO": {
                "n_envs": N_ENVS,
                "n_steps": N_STEPS,
                "global_rollout": GLOBAL_ROLLOUT,
                "batch_size": BATCH_SIZE,
                "n_epochs": N_EPOCHS,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "clip_range": CLIP_RANGE,
                "ent_coef": ENT_COEF,
                "vf_coef": VF_COEF,
                "initial_lr": INITIAL_LR,
                "lr_schedule": "linear",
                "device": device,
            },
            "no_mid_training_evaluation": True,
        },
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    write_json(
        root / "experiment_manifest.json",
        experiment_manifest,
    )

    print("\n" + "=" * 128)
    print("UAGMC CONTROLLED COMPLEXITY LADDER | E3 -> E4 -> E5")
    print("=" * 128)
    print(f"Stages             : {stages}")
    print(f"Train seed         : {seed}")
    print(f"Fleet size (frozen): {fleet_size}")
    print(f"Fleet selection    : {fleet_selection_mode}")
    print(f"Steps / stage      : {timesteps:,}")
    print(
        f"Total requested    : {timesteps * len(stages):,}"
    )
    print(
        f"Execution          : 16 envs x 1280, batch=512, {device}"
    )
    print("Mid-training eval  : NONE")
    print(f"Output             : {root}")
    print("=" * 128)

    status_path = root / "serial_status.csv"
    successes = 0
    failures = 0

    for idx, stage in enumerate(stages, start=1):
        print(
            f"\n### STAGE {idx}/{len(stages)}: {stage} ###",
            flush=True,
        )

        try:
            result = train_stage(
                stage=stage,
                fleet_size=fleet_size,
                seed=seed,
                requested_steps=timesteps,
                checkpoint_interval=checkpoint_interval,
                root=root,
                device=device,
            )

            append_csv(
                status_path,
                {
                    "stage": stage,
                    "status": "SUCCESS",
                    "fleet_size": fleet_size,
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
                    "stage": stage,
                    "status": "ERROR",
                    "fleet_size": fleet_size,
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
            "stages": stages,
            "fleet_size": fleet_size,
            "train_seed": seed,
            "timesteps_per_stage": timesteps,
            "finished": datetime.now().isoformat(timespec="seconds"),
        },
    )

    print("\n" + "=" * 128)
    print("E3/E4/E5 SERIAL TRAINING FINISHED")
    print("=" * 128)
    print(f"Success : {successes}/{len(stages)}")
    print(f"Root    : {root}")
    print(
        "Evaluate checkpoints only AFTER all requested stages finish."
    )
    print("=" * 128)

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
