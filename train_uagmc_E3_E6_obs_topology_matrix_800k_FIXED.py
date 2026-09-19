# -*- coding: utf-8 -*-
r"""
UAGMC E3-E6 x Observation x Topology matrix, 800k per cell
==========================================================

Purpose
-------
Run the controlled 32-cell matrix:

    4 physical stages  : E3, E4, E5, E6
    4 observation modes: O00, O10, O01, O11
    2 topology sizes    : T2, T3
    800,000 PPO timesteps per cell

Total formal budget:
    4 * 4 * 2 * 800,000 = 25,600,000 timesteps

The script is intentionally built on top of the already validated:
    train_uagmc_E3_E4_E5_serial_1m.py

It does NOT overwrite source environment files.

Observation factors
-------------------
O00 = LEGACY + focal passenger observation
O10 = AUGMENTED resource observation + focal passenger observation
O01 = LEGACY with focal passenger OD removed
O11 = AUGMENTED resource observation with focal passenger OD removed

"Passenger removed" means ONLY the current focal passenger's first 4 OD
coordinates are zeroed in every UAGMC history frame. Aggregate passenger
system state (waiting/incoming counts) is preserved.

Augmented current-resource vector, per departure candidate:
    ready idle empty aircraft
    inbound empty aircraft
    turnaround-busy aircraft
    minimum turnaround release ETA
    pad next-free ETA
    charging aircraft
    active charging aircraft under E6 charger capacity
    minimum charge completion ETA

Plus four common destination/hub resource features:
    ready idle empty aircraft
    turnaround-busy aircraft
    pad next-free ETA
    charging aircraft

Topology factors
----------------
T2:
    candidates V0, V1 -> destination V2
    original UAGMC 2-origin topology

T3:
    candidates V0, V1, V3 -> destination V2
    a third departure vertiport V3 is injected into the original map at [0, 10]
    while the same passenger trace / same 300 requests / same destination are
    retained. Aggregate demand is therefore unchanged.

    With fleet N=40, the reset allocation is deterministically normalized to:
        V0: 20, V1: 10, V3: 10
    by moving 10 aircraft from the original V0 pool to V3 after reset.
    This keeps total fleet size unchanged and prevents an empty new origin.

Physical stages
---------------
E3:
    single passenger per service flight

E4:
    E3 + 3 min turnaround

E5:
    E4 + shared takeoff/landing pad calendar

E6:
    E5 + finite parallel charger capacity

IMPORTANT:
The previous exploratory E5 with 1.0 min pad separation was strongly
capacity-starved. This matrix defaults to 0.5 min separation so the new
factorial experiment is not automatically dominated by the known E5
throughput cliff. The selected value is written to the manifest and can be
overridden by --pad-separation.

E6 charger capacity defaults to 2 aircraft per vertiport and can be changed
by --charger-capacity.

GPU speed selection
-------------------
Before the 25.6M matrix, the script benchmarks several CUDA PPO profiles on
the HEAVIEST representative environment (E6/O10/T3) and selects the profile
with the highest measured actual timesteps / wall-clock second.

Profiles:
    P0: 16 envs x 1280, batch 512
    P1: 20 envs x 1024, batch 1024
    P2: 20 envs x 1024, batch 2048
    P3: 20 envs x 1280, batch 2560
    P4: 25 envs x 1024, batch 1600

All n_env values divide 50,000 exactly so formal 50k checkpoints remain exact.

The benchmark also samples nvidia-smi GPU utilization when available.
The selected profile is frozen for ALL 32 cells.

Per-cell immediate analysis
---------------------------
After every 800k training cell, the script immediately evaluates:
    200k, 400k, 600k, 650k, 700k, 750k, 800k

with deterministic evaluation seeds 123,124,125.

It saves:
    ATT / AWT / access / flight
    completion / backlog
    unfinished-aware system passenger-minutes
    action shares
    normalized policy entropy
    late-window (600k..800k) mean / std
    final 800k metrics

The master matrix CSV and factor-effect CSV are refreshed after EVERY cell.
Therefore a long overnight run can be stopped without losing previous results.

Main dependencies
-----------------
This file must sit beside:
    train_uagmc_E3_E4_E5_serial_1m.py

in UAGMC-main.

Default run
-----------
    python train_uagmc_E3_E6_obs_topology_matrix_800k.py

Resume a prior matrix folder
----------------------------
    python train_uagmc_E3_E6_obs_topology_matrix_800k.py ^
      --output-root "serial_runs\YOUR_MATRIX_FOLDER" ^
      --resume ^
      --skip-benchmark

Useful subset smoke test
------------------------
    python train_uagmc_E3_E6_obs_topology_matrix_800k.py ^
      --stages E3 ^
      --obs-modes O00 ^
      --topologies T2 ^
      --timesteps 50000 ^
      --benchmark-steps 10000

T3 compatibility fix
--------------------
Public UAGMC defines per-vertiport eVTOL charge rates only for V0/V1/V2.
For T3, this runner adds V3 before any eVTOL is spawned and clones V1's
charge rate onto V3. This prevents the fixed-fleet reset from failing with
KeyError('3') while keeping the topology comparison controlled.

Scientific status
-----------------
This file has been syntax-checked by ChatGPT but cannot be runtime-tested
against the user's local Windows UAGMC repository here. It is deliberately
fail-fast: structural mismatches stop before silently changing experiment
semantics.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

try:
    import train_uagmc_E3_E4_E5_serial_1m as base
except Exception as exc:
    raise RuntimeError(
        "Cannot import train_uagmc_E3_E4_E5_serial_1m.py. "
        "Put this matrix runner beside the validated E3/E4/E5 training file."
    ) from exc

from at_obj.map.map import Map
from at_obj.vertiport.vertiport_builder import VertiportBuilder
from at_obj.scenario_fixed_fleet import ConservedFleetScenario
from at_obj.evtol.evtol import eVTOL
from at_obj.evtol.vehicle_state import VehicleState
import at_obj.vertiport.vertiport_spec as vertiport_spec_module
import at_obj.evtol.evtol_builder as evtol_builder_module


ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "train_data" / "passengers_300.csv"

FORMAL_TIMESTEPS = 800_000
CHECKPOINT_INTERVAL = 50_000
TRAIN_SEED = 1
FLEET_SIZE = 40
MAX_TIME = 600

TURNAROUND_DELAY_MIN = 3.0
DEFAULT_PAD_SEPARATION_MIN = 0.5
DEFAULT_CHARGER_CAPACITY = 2

DESTINATION = 2
T2_CANDIDATES = [0, 1]
T3_CANDIDATES = [0, 1, 3]
T3_POSITION = [0, 10]

STAGES = ("E3", "E4", "E5", "E6")
OBS_MODES = ("O00", "O10", "O01", "O11")
TOPOLOGIES = ("T2", "T3")

ANALYSIS_STEPS_DEFAULT = (
    200_000,
    400_000,
    600_000,
    650_000,
    700_000,
    750_000,
    800_000,
)

OBS_DESCRIPTIONS = {
    "O00": "legacy observation + focal passenger OD",
    "O10": "augmented current resource observation + focal passenger OD",
    "O01": "legacy observation with focal passenger OD removed",
    "O11": "augmented current resource observation with focal passenger OD removed",
}


@dataclass(frozen=True)
class SpeedProfile:
    name: str
    n_envs: int
    n_steps: int
    batch_size: int


SPEED_PROFILES = (
    SpeedProfile("P0_16x1280_b512", 16, 1280, 512),
    SpeedProfile("P1_20x1024_b1024", 20, 1024, 1024),
    SpeedProfile("P2_20x1024_b2048", 20, 1024, 2048),
    SpeedProfile("P3_20x1280_b2560", 20, 1280, 2560),
    SpeedProfile("P4_25x1024_b1600", 25, 1024, 1600),
)


# =============================================================================
# Original constructor capture for process-local topology patches
# =============================================================================

_ORIG_MAP_INIT = Map.__init__
_ORIG_VP_INIT = VertiportBuilder.__init__

# Public UAGMC defines VERTIPORT_CHARGE_RATE only for V0/V1/V2.
# T3 injects V3, so spawning a fixed-fleet eVTOL at V3 would otherwise raise:
#     KeyError: '3'
# For the topology-size experiment, V3 intentionally clones V1's charging
# infrastructure so T2->T3 does not introduce a new charging-rate confound.
_ORIG_V3_CHARGE_RATE_PRESENT = (
    "3" in vertiport_spec_module.VERTIPORT_CHARGE_RATE
)
_ORIG_V3_CHARGE_RATE = (
    vertiport_spec_module.VERTIPORT_CHARGE_RATE.get("3")
)


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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fields: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for row in rows:
            out = {}
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    out[key] = json.dumps(jsonable(value), ensure_ascii=False)
                else:
                    out[key] = value
            w.writerow(out)


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple, np.ndarray)):
            serial[key] = json.dumps(jsonable(value), ensure_ascii=False)
        else:
            serial[key] = value

    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(serial.keys()))
        if not exists:
            w.writeheader()
        w.writerow(serial)
        f.flush()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


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


def parse_list(text: str, allowed: Sequence[str]) -> List[str]:
    values = [x.strip().upper() for x in str(text).split(",") if x.strip()]
    aliases = {
        "E3_SINGLE_PAX": "E3",
        "E4_TURNAROUND": "E4",
        "E5_PAD": "E5",
        "E6_CHARGING": "E6",
    }
    values = [aliases.get(x, x) for x in values]
    bad = [x for x in values if x not in allowed]
    if bad:
        raise ValueError(f"Unsupported values {bad}; allowed={allowed}")
    if not values:
        raise ValueError("Empty selection")
    return values


def parse_ints(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty integer list")
    return vals


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_fast_cuda() -> Dict[str, Any]:
    info = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }

    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gb": props.total_memory / (1024 ** 3),
                "gpu_capability": list(torch.cuda.get_device_capability(0)),
                "tf32_matmul": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
            }
        )

    return info


# =============================================================================
# Wrapper-tree helpers
# =============================================================================

def find_scenario(obj: Any) -> ConservedFleetScenario:
    if hasattr(obj, "venv"):
        obj = obj.venv
    if hasattr(obj, "envs") and obj.envs:
        obj = obj.envs[0]

    seen = set()
    for _ in range(80):
        if id(obj) in seen:
            break
        seen.add(id(obj))

        if (
            hasattr(obj, "_all_evtols")
            and hasattr(obj, "vertiports")
            and hasattr(obj, "person_travel_records")
        ):
            return obj

        if hasattr(obj, "scenario"):
            nxt = getattr(obj, "scenario")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue

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

    raise RuntimeError(f"Cannot locate ConservedFleetScenario below {type(obj)}")


def find_uam_wrapper(obj: Any):
    if hasattr(obj, "venv"):
        obj = obj.venv
    if hasattr(obj, "envs") and obj.envs:
        obj = obj.envs[0]

    seen = set()
    for _ in range(80):
        if id(obj) in seen:
            break
        seen.add(id(obj))

        if all(
            hasattr(obj, name)
            for name in ("encoder", "obs_buffer", "state", "num_frames")
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

    raise RuntimeError(f"Cannot locate UAMRLWrapper below {type(obj)}")


def state_name(evtol: Any) -> str:
    state = getattr(evtol, "state", None)
    return str(getattr(state, "name", state)).upper()


def passenger_ids(evtol: Any) -> List[Any]:
    return list(getattr(evtol, "passenger_ids", []) or [])


def current_vid(evtol: Any) -> Optional[int]:
    raw = getattr(evtol, "current_vertiport_id", None)
    try:
        return int(raw) if raw is not None else None
    except Exception:
        return None


def target_vid(evtol: Any) -> Optional[int]:
    raw = getattr(evtol, "target_vertiport_id", None)
    try:
        return int(raw) if raw is not None else None
    except Exception:
        return None


# =============================================================================
# T3 topology injection
# =============================================================================

def _restore_v3_charge_rate() -> None:
    rates = vertiport_spec_module.VERTIPORT_CHARGE_RATE

    if _ORIG_V3_CHARGE_RATE_PRESENT:
        rates["3"] = _ORIG_V3_CHARGE_RATE
    else:
        rates.pop("3", None)

    # evtol_builder imported this dictionary at module import time. Rebind as a
    # defensive measure in case a local source revision replaced the object.
    evtol_builder_module.VERTIPORT_CHARGE_RATE = rates


def _install_t3_infrastructure_constants() -> Dict[str, Any]:
    rates = vertiport_spec_module.VERTIPORT_CHARGE_RATE

    if "1" not in rates:
        raise RuntimeError(
            "Cannot construct T3 fairly: V1 is missing from "
            "VERTIPORT_CHARGE_RATE."
        )

    # Clone V1: V1 and the new V3 both receive 25% of the N=40 initial fleet.
    # This preserves the topology-size factor without adding charging-rate
    # heterogeneity as an uncontrolled factor.
    rates["3"] = float(rates["1"])
    evtol_builder_module.VERTIPORT_CHARGE_RATE = rates

    if "3" not in evtol_builder_module.VERTIPORT_CHARGE_RATE:
        raise RuntimeError(
            "T3 charge-rate injection failed before eVTOL spawning."
        )

    return {
        "V3_charge_rate_source": "clone_V1",
        "V3_charge_rate": float(rates["3"]),
    }


def restore_topology_patch() -> None:
    Map.__init__ = _ORIG_MAP_INIT
    VertiportBuilder.__init__ = _ORIG_VP_INIT
    _restore_v3_charge_rate()


def install_topology_patch(topology: str) -> None:
    restore_topology_patch()

    if topology == "T2":
        return

    _install_t3_infrastructure_constants()

    if topology != "T3":
        raise ValueError(topology)

    def _map_init_t3(self, *args, **kwargs):
        _ORIG_MAP_INIT(self, *args, **kwargs)
        self.vertiport_station[3] = list(T3_POSITION)
        self.vertiport_station_num = max(
            int(getattr(self, "vertiport_station_num", 0)),
            4,
        )

    def _vp_init_t3(self, *args, **kwargs):
        _ORIG_VP_INIT(self, *args, **kwargs)
        if not hasattr(self, "vertiport_evtol_capacity"):
            self.vertiport_evtol_capacity = {}
        self.vertiport_evtol_capacity["3"] = max(
            1,
            int(self.vertiport_evtol_capacity.get("3", 1)),
        )

    Map.__init__ = _map_init_t3
    VertiportBuilder.__init__ = _vp_init_t3


def candidates_for(topology: str) -> List[int]:
    if topology == "T2":
        return list(T2_CANDIDATES)
    if topology == "T3":
        return list(T3_CANDIDATES)
    raise ValueError(topology)


def _move_evtol_between_vertiports(
    scenario: ConservedFleetScenario,
    evtol: Any,
    source: str,
    dest: str,
) -> None:
    src_list = scenario.vertiports.evtols_at_vertiport[source]
    dst_list = scenario.vertiports.evtols_at_vertiport[dest]

    if evtol not in src_list:
        raise RuntimeError(
            f"Cannot move {getattr(evtol, 'id', '?')}: not present at V{source}"
        )

    src_list.remove(evtol)
    dst_list.append(evtol)

    evtol.current_vertiport_id = str(dest)

    if hasattr(evtol, "target_vertiport_id"):
        evtol.target_vertiport_id = None

    fixed_home = getattr(scenario, "_fixed_home", None)
    if isinstance(fixed_home, dict):
        fixed_home[getattr(evtol, "id")] = str(dest)


def normalize_t3_initial_allocation(
    scenario: ConservedFleetScenario,
    fleet_size: int,
) -> Dict[str, int]:
    if "3" not in scenario.vertiports.vertiport_list:
        raise RuntimeError("T3 map injection failed: V3 is absent")

    if "3" not in scenario.vertiports.evtols_at_vertiport:
        scenario.vertiports.evtols_at_vertiport["3"] = []

    desired = {
        "0": int(fleet_size // 2),
        "1": int(fleet_size // 4),
        "3": int(fleet_size - (fleet_size // 2) - (fleet_size // 4)),
    }

    current = {
        vid: len(scenario.vertiports.evtols_at_vertiport.get(vid, []))
        for vid in ("0", "1", "3")
    }

    need3 = max(0, desired["3"] - current["3"])

    if need3 > 0:
        donors = []
        for source in ("0", "1"):
            surplus = max(0, current[source] - desired[source])
            if surplus <= 0:
                continue

            local = [
                e
                for e in scenario.vertiports.evtols_at_vertiport[source]
                if state_name(e) != "FLYING"
            ]
            local.sort(key=lambda e: str(getattr(e, "id", "")))

            for e in local[:surplus]:
                donors.append((source, e))

        if len(donors) < need3:
            raise RuntimeError(
                f"T3 allocation normalization lacks donor aircraft: "
                f"need={need3}, donors={len(donors)}, current={current}, desired={desired}"
            )

        for source, evtol in donors[:need3]:
            _move_evtol_between_vertiports(
                scenario=scenario,
                evtol=evtol,
                source=source,
                dest="3",
            )

    final = {
        vid: len(scenario.vertiports.evtols_at_vertiport.get(vid, []))
        for vid in ("0", "1", "3")
    }

    if sum(final.values()) != int(fleet_size):
        # Some aircraft could theoretically sit at V2 at reset; fail instead of
        # silently hiding an allocation mismatch.
        all_local = sum(
            len(v)
            for v in scenario.vertiports.evtols_at_vertiport.values()
        )
        if all_local != int(fleet_size):
            raise RuntimeError(
                f"Unexpected fixed-fleet allocation after T3 normalization: "
                f"origins={final}, all_local={all_local}, N={fleet_size}"
            )

    if final != desired:
        raise RuntimeError(
            f"T3 initial allocation mismatch: final={final}, desired={desired}"
        )

    if hasattr(scenario, "_fixed_initial_allocation"):
        try:
            scenario._fixed_initial_allocation = dict(final)
        except Exception:
            pass

    return final


# =============================================================================
# E6 finite charging resource
# =============================================================================

_E6_CHARGER_CAPACITY = DEFAULT_CHARGER_CAPACITY


def _charge_with_finite_capacity(
    self: VertiportBuilder,
    time_step_min: float = 1.0,
):
    scenario = getattr(self, "_e345_scenario", None)
    now = float(getattr(scenario, "time", 0.0)) if scenario is not None else 0.0

    for vid, evtol_list in self.evtols_at_vertiport.items():
        charging = []

        for e in evtol_list:
            if base.is_turnaround_busy(e):
                continue
            if state_name(e) != "CHARGING":
                continue

            if not hasattr(e, "_e6_charge_queue_enter_time"):
                e._e6_charge_queue_enter_time = now

            charging.append(e)

        charging.sort(
            key=lambda e: (
                float(getattr(e, "_e6_charge_queue_enter_time", now)),
                str(getattr(e, "id", "")),
            )
        )

        cap = int(_E6_CHARGER_CAPACITY)
        active = charging[:cap]
        waiting = charging[cap:]

        if scenario is not None:
            stats = getattr(scenario, "_e345_stats", None)
            if isinstance(stats, dict):
                stats.setdefault("e6_charger_active_aircraft_steps", 0)
                stats.setdefault("e6_charger_wait_aircraft_steps", 0)
                stats.setdefault("e6_max_charger_queue", 0)

                stats["e6_charger_active_aircraft_steps"] += len(active)
                stats["e6_charger_wait_aircraft_steps"] += len(waiting)
                stats["e6_max_charger_queue"] = max(
                    int(stats["e6_max_charger_queue"]),
                    len(charging),
                )

        for e in active:
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
                if hasattr(e, "_e6_charge_queue_enter_time"):
                    delattr(e, "_e6_charge_queue_enter_time")


# =============================================================================
# Physical stage installation
# =============================================================================

def configure_base_globals(
    *,
    topology: str,
    pad_separation: float,
) -> None:
    cands = candidates_for(topology)

    base.CANDIDATES = list(cands)
    base.TO_VERTIPORT = DESTINATION
    base.RETURN_HUB = DESTINATION
    base.MAX_TIME = MAX_TIME
    base.TURNAROUND_DELAY_MIN = TURNAROUND_DELAY_MIN

    pad_capacity = {
        str(v): 1
        for v in set(cands + [DESTINATION])
    }
    base.PAD_CAPACITY = pad_capacity
    base.PAD_SEPARATION_MIN = float(pad_separation)


def install_physics(
    *,
    stage: str,
    topology: str,
    pad_separation: float,
    charger_capacity: int,
) -> Dict[str, Any]:
    global _E6_CHARGER_CAPACITY

    install_topology_patch(topology)
    configure_base_globals(
        topology=topology,
        pad_separation=pad_separation,
    )

    if stage == "E3":
        patch = base.install_stage_patch("E3_SINGLE_PAX")
    elif stage == "E4":
        patch = base.install_stage_patch("E4_TURNAROUND")
    elif stage == "E5":
        patch = base.install_stage_patch("E5_PAD")
    elif stage == "E6":
        patch = base.install_stage_patch("E5_PAD")
        _E6_CHARGER_CAPACITY = int(charger_capacity)
        VertiportBuilder.charge_evtols_at_vertiport = (
            _charge_with_finite_capacity
        )
        patch = dict(patch)
        patch.update(
            {
                "stage": "E6_CHARGING_RESOURCE",
                "finite_charger_capacity": int(charger_capacity),
            }
        )
    else:
        raise ValueError(stage)

    return patch


# =============================================================================
# Observation augmentation/removal wrapper
# =============================================================================

def _pad_next_free_eta(
    scenario: ConservedFleetScenario,
    vid: int,
) -> float:
    cal = getattr(scenario, "_e345_pad_calendar", {}) or {}
    times = list(cal.get(str(vid), []) or [])

    if not times:
        return 0.0

    now = float(getattr(scenario, "time", 0.0))
    sep = float(getattr(base, "PAD_SEPARATION_MIN", 0.0))
    return max(
        0.0,
        max(float(x) for x in times) + sep - now,
    )


def _resource_features_for_vid(
    scenario: ConservedFleetScenario,
    vid: int,
    *,
    e6: bool,
    charger_capacity: int,
) -> List[float]:
    ready_idle = 0
    inbound_empty = 0
    turnaround_busy = 0
    turn_etas = []
    charging = []
    charge_etas = []

    now = float(getattr(scenario, "time", 0.0))

    for evtol in list(getattr(scenario, "_all_evtols", {}).values()):
        cur = current_vid(evtol)
        tar = target_vid(evtol)
        st = state_name(evtol)

        if (
            cur == int(vid)
            and st == "IDLE"
            and not passenger_ids(evtol)
            and not base.is_turnaround_busy(evtol)
        ):
            ready_idle += 1

        if (
            st == "FLYING"
            and tar == int(vid)
            and not passenger_ids(evtol)
        ):
            inbound_empty += 1

        if cur == int(vid) and base.is_turnaround_busy(evtol):
            turnaround_busy += 1
            release = getattr(
                evtol,
                "_e345_turnaround_release_time",
                None,
            )
            if release is not None:
                turn_etas.append(
                    max(0.0, float(release) - now)
                )

        if cur == int(vid) and st == "CHARGING":
            charging.append(evtol)
            rate = float(
                getattr(
                    getattr(evtol, "spec", None),
                    "charge_rate_kwh_per_min",
                    0.0,
                )
                or 0.0
            )
            cap = float(
                getattr(
                    getattr(evtol, "spec", None),
                    "battery_capacity_kwh",
                    0.0,
                )
                or 0.0
            )
            bat = float(getattr(evtol, "battery_kwh", 0.0) or 0.0)
            if rate > 1e-12:
                charge_etas.append(
                    max(0.0, cap - bat) / rate
                )

    active_charging = (
        min(len(charging), int(charger_capacity))
        if e6
        else len(charging)
    )

    return [
        float(ready_idle),
        float(inbound_empty),
        float(turnaround_busy),
        float(min(turn_etas) if turn_etas else 0.0),
        float(_pad_next_free_eta(scenario, vid)),
        float(len(charging)),
        float(active_charging),
        float(min(charge_etas) if charge_etas else 0.0),
    ]


def build_augmented_resource_vector(
    scenario: ConservedFleetScenario,
    *,
    candidates: Sequence[int],
    stage: str,
    charger_capacity: int,
) -> np.ndarray:
    values: List[float] = []

    for vid in candidates:
        values.extend(
            _resource_features_for_vid(
                scenario,
                int(vid),
                e6=(stage == "E6"),
                charger_capacity=charger_capacity,
            )
        )

    hub = _resource_features_for_vid(
        scenario,
        DESTINATION,
        e6=(stage == "E6"),
        charger_capacity=charger_capacity,
    )

    # Common destination/hub: ready, turnaround, pad, charging.
    values.extend(
        [
            hub[0],
            hub[2],
            hub[4],
            hub[5],
        ]
    )

    return np.asarray(values, dtype=np.float32)


def remove_focal_passenger_from_stacked_obs(
    obs: np.ndarray,
    uam_wrapper: Any,
) -> np.ndarray:
    out = np.asarray(obs, dtype=np.float32).copy().reshape(-1)

    num_frames = int(getattr(uam_wrapper, "num_frames", 6))
    single_dim = int(
        getattr(
            uam_wrapper,
            "single_obs_dim",
            len(out) // max(num_frames, 1),
        )
    )

    expected = num_frames * single_dim
    if expected != len(out):
        raise RuntimeError(
            f"Cannot identify UAGMC frame layout for passenger removal: "
            f"len={len(out)}, frames={num_frames}, single={single_dim}"
        )

    if single_dim < 4:
        raise RuntimeError(
            f"single observation dim {single_dim} < passenger_dim=4"
        )

    for frame in range(num_frames):
        start = frame * single_dim
        out[start:start + 4] = 0.0

    return out


def refresh_uam_observation_after_t3_redistribution(env: Any) -> np.ndarray:
    uam = find_uam_wrapper(env)
    scenario = find_scenario(env)

    uam.state = scenario.get_state()
    current = uam.encoder.encode(
        scenario,
        uam.state,
    )

    uam.obs_buffer.clear()
    for _ in range(int(uam.num_frames)):
        uam.obs_buffer.append(
            np.asarray(current, dtype=np.float32).copy()
        )

    return uam._get_stacked_obs()


def snapshot_episode(scenario: ConservedFleetScenario) -> Dict[str, Any]:
    persons_obj = getattr(scenario, "persons", None)
    persons = (
        getattr(persons_obj, "persons", {})
        if persons_obj is not None
        else {}
    ) or {}

    records = getattr(
        scenario,
        "person_travel_records",
        {},
    ) or {}

    finished_raw = set(
        getattr(scenario, "finished_ids", []) or []
    )
    finished_str = {str(x) for x in finished_raw}

    rows = []

    for pid_raw, person in persons.items():
        pid = str(pid_raw)
        recs = (
            records.get(pid_raw)
            or records.get(pid)
            or []
        )
        rec = recs[-1] if recs else {}

        start = rec.get("start_time")
        end = rec.get("end_time")

        travel = float("nan")
        if start is not None and end is not None:
            try:
                travel = float(end) - float(start)
            except Exception:
                pass

        stats = getattr(person, "time_stats", {}) or {}

        rows.append(
            {
                "pid": pid,
                "finished": bool(
                    pid_raw in finished_raw
                    or pid in finished_str
                    or end is not None
                    or str(getattr(person, "state", "")).lower()
                    == "finished"
                ),
                "travel_time": travel,
                "access": fnum(stats.get("to_vertiport", np.nan)),
                "wait": fnum(stats.get("wait_uam", np.nan)),
                "fly": fnum(stats.get("fly", np.nan)),
            }
        )

    queues = {}
    for vid, vp in scenario.vertiports.vertiport_list.items():
        queues[str(vid)] = len(
            list(getattr(vp, "person_list", []) or [])
        )

    diag = {}
    if hasattr(scenario, "get_fixed_fleet_diagnostics"):
        try:
            diag = scenario.get_fixed_fleet_diagnostics() or {}
        except Exception:
            pass

    return {
        "rows": rows,
        "n_persons": len(persons),
        "queues": queues,
        "fleet_diag": diag,
        "stage_stats": dict(
            getattr(scenario, "_e345_stats", {}) or {}
        ),
    }


class MatrixObservationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        stage: str,
        obs_mode: str,
        topology: str,
        fleet_size: int,
        charger_capacity: int,
    ):
        super().__init__(env)

        self.matrix_stage = str(stage)
        self.obs_mode = str(obs_mode)
        self.topology = str(topology)
        self.matrix_fleet_size = int(fleet_size)
        self.charger_capacity = int(charger_capacity)
        self.candidates = candidates_for(topology)

        uam = find_uam_wrapper(env)
        base_dim = int(np.prod(env.observation_space.shape))

        self.remove_passenger = obs_mode in ("O01", "O11")
        self.augment_resources = obs_mode in ("O10", "O11")

        extra_dim = (
            len(self.candidates) * 8 + 4
            if self.augment_resources
            else 0
        )

        self._base_dim = base_dim
        self._extra_dim = extra_dim
        self._uam_num_frames = int(getattr(uam, "num_frames", 6))

        self.observation_space = spaces.Box(
            low=0.0,
            high=1e6,
            shape=(base_dim + extra_dim,),
            dtype=np.float32,
        )

        self._t3_allocation = None

    def _ensure_scenario_extras(self):
        scenario = find_scenario(self.env)

        if not hasattr(scenario, "_e345_pad_calendar"):
            scenario._e345_pad_calendar = {}

        for vid in set(self.candidates + [DESTINATION]):
            scenario._e345_pad_calendar.setdefault(str(vid), [])

        stats = getattr(scenario, "_e345_stats", None)
        if isinstance(stats, dict):
            stats.setdefault("e6_charger_active_aircraft_steps", 0)
            stats.setdefault("e6_charger_wait_aircraft_steps", 0)
            stats.setdefault("e6_max_charger_queue", 0)

        return scenario

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        scenario = self._ensure_scenario_extras()
        uam = find_uam_wrapper(self.env)

        x = np.asarray(obs, dtype=np.float32).reshape(-1)

        if len(x) != self._base_dim:
            raise RuntimeError(
                f"Base observation shape changed: "
                f"got {len(x)}, expected {self._base_dim}"
            )

        if self.remove_passenger:
            x = remove_focal_passenger_from_stacked_obs(
                x,
                uam,
            )

        if self.augment_resources:
            extra = build_augmented_resource_vector(
                scenario,
                candidates=self.candidates,
                stage=self.matrix_stage,
                charger_capacity=self.charger_capacity,
            )
            x = np.concatenate([x, extra], axis=0)

        return x.astype(np.float32, copy=False)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs = out
            info = {}

        scenario = self._ensure_scenario_extras()

        if self.topology == "T3":
            self._t3_allocation = normalize_t3_initial_allocation(
                scenario,
                self.matrix_fleet_size,
            )
            obs = refresh_uam_observation_after_t3_redistribution(
                self.env
            )
        else:
            self._t3_allocation = None

        return self._transform(obs), info

    def step(self, action):
        out = self.env.step(action)

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info)

            if bool(terminated) or bool(truncated):
                info["terminal_snapshot"] = snapshot_episode(
                    find_scenario(self.env)
                )

            return (
                self._transform(obs),
                reward,
                terminated,
                truncated,
                info,
            )

        if len(out) == 4:
            obs, reward, done, info = out
            info = dict(info)

            if bool(done):
                info["terminal_snapshot"] = snapshot_episode(
                    find_scenario(self.env)
                )

            return self._transform(obs), reward, done, info

        raise RuntimeError(
            f"Unexpected env.step tuple length: {len(out)}"
        )


# =============================================================================
# Environment factories
# =============================================================================

def make_matrix_env_factory(
    *,
    stage: str,
    obs_mode: str,
    topology: str,
    fleet_size: int,
    env_index: int,
    run_dir: Path,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
):
    def _init():
        patch = install_physics(
            stage=stage,
            topology=topology,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
        )

        cands = candidates_for(topology)

        env = base.make_env(
            max_time=int(max_time),
            log_dir=run_dir / "monitor",
            env_index=int(env_index),
            person_spawn_file=str(TRAIN_FILE),
            candidate_from_vertiports=list(cands),
            to_vertiport=DESTINATION,
            enable_logger=False,
            fleet_mode="conserved_closed_loop",
            fleet_size=int(fleet_size),
            fleet_assertions=True,
        )()

        env = MatrixObservationWrapper(
            env,
            stage=stage,
            obs_mode=obs_mode,
            topology=topology,
            fleet_size=fleet_size,
            charger_capacity=charger_capacity,
        )

        env._matrix_patch_manifest = patch
        return env

    return _init


def build_vec_env(
    *,
    stage: str,
    obs_mode: str,
    topology: str,
    fleet_size: int,
    profile: SpeedProfile,
    seed: int,
    run_dir: Path,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> VecNormalize:
    factories = [
        make_matrix_env_factory(
            stage=stage,
            obs_mode=obs_mode,
            topology=topology,
            fleet_size=fleet_size,
            env_index=i,
            run_dir=run_dir,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
            max_time=max_time,
        )
        for i in range(profile.n_envs)
    ]

    raw = SubprocVecEnv(
        factories,
        start_method="spawn",
    )
    raw.seed(int(seed))

    return VecNormalize(
        raw,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=base.GAMMA,
    )


def build_eval_raw_env(
    *,
    stage: str,
    obs_mode: str,
    topology: str,
    fleet_size: int,
    run_dir: Path,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
):
    return DummyVecEnv(
        [
            make_matrix_env_factory(
                stage=stage,
                obs_mode=obs_mode,
                topology=topology,
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
# GPU utilization sampler
# =============================================================================

class GPUSampler:
    def __init__(self, interval_sec: float = 0.5):
        self.interval_sec = float(interval_sec)
        self.samples: List[Tuple[float, float]] = []
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def _sample_once(self) -> Optional[Tuple[float, float]]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            ).strip().splitlines()[0]

            parts = [x.strip() for x in out.split(",")]
            return float(parts[0]), float(parts[1])
        except Exception:
            return None

    def _run(self):
        while not self.stop_event.is_set():
            sample = self._sample_once()
            if sample is not None:
                self.samples.append(sample)
            self.stop_event.wait(self.interval_sec)

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)

        if not self.samples:
            return {
                "gpu_util_mean_pct": float("nan"),
                "gpu_util_max_pct": float("nan"),
                "gpu_mem_mean_mb": float("nan"),
                "gpu_mem_max_mb": float("nan"),
            }

        arr = np.asarray(self.samples, dtype=float)
        return {
            "gpu_util_mean_pct": float(arr[:, 0].mean()),
            "gpu_util_max_pct": float(arr[:, 0].max()),
            "gpu_mem_mean_mb": float(arr[:, 1].mean()),
            "gpu_mem_max_mb": float(arr[:, 1].max()),
        }


# =============================================================================
# Dynamic PPO config
# =============================================================================

def apply_speed_profile(profile: SpeedProfile) -> None:
    if CHECKPOINT_INTERVAL % profile.n_envs != 0:
        raise ValueError(
            f"{profile.name}: n_envs={profile.n_envs} does not divide "
            f"{CHECKPOINT_INTERVAL}"
        )

    global_rollout = profile.n_envs * profile.n_steps

    if global_rollout % profile.batch_size != 0:
        raise ValueError(
            f"{profile.name}: rollout={global_rollout} not divisible by "
            f"batch={profile.batch_size}"
        )

    base.N_ENVS = int(profile.n_envs)
    base.N_STEPS = int(profile.n_steps)
    base.GLOBAL_ROLLOUT = int(global_rollout)
    base.BATCH_SIZE = int(profile.batch_size)
    base.MAX_TIME = MAX_TIME


def build_model(
    *,
    env: VecNormalize,
    profile: SpeedProfile,
    seed: int,
    run_dir: Path,
    device: str,
) -> PPO:
    apply_speed_profile(profile)

    return base.build_model(
        env=env,
        seed=int(seed),
        run_dir=run_dir,
        device=device,
    )


# =============================================================================
# Speed benchmark
# =============================================================================

def benchmark_one_profile(
    *,
    profile: SpeedProfile,
    benchmark_steps: int,
    root: Path,
    pad_separation: float,
    charger_capacity: int,
    device: str,
) -> Dict[str, Any]:
    run_dir = root / "_speed_benchmark" / profile.name
    run_dir.mkdir(parents=True, exist_ok=True)

    apply_speed_profile(profile)
    seed_all(991)

    env = None
    model = None
    sampler = GPUSampler()
    started = time.time()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        env = build_vec_env(
            stage="E6",
            obs_mode="O10",
            topology="T3",
            fleet_size=FLEET_SIZE,
            profile=profile,
            seed=991,
            run_dir=run_dir,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
            max_time=MAX_TIME,
        )

        model = build_model(
            env=env,
            profile=profile,
            seed=991,
            run_dir=run_dir,
            device=device,
        )

        sampler.start()

        model.learn(
            total_timesteps=int(benchmark_steps),
            progress_bar=False,
            reset_num_timesteps=True,
        )

        gpu = sampler.stop()
        elapsed = time.time() - started
        actual = int(model.num_timesteps)

        row = {
            "status": "SUCCESS",
            "profile": profile.name,
            "n_envs": profile.n_envs,
            "n_steps": profile.n_steps,
            "global_rollout": profile.n_envs * profile.n_steps,
            "batch_size": profile.batch_size,
            "requested_steps": int(benchmark_steps),
            "actual_steps": actual,
            "elapsed_seconds": elapsed,
            "actual_fps": actual / max(elapsed, 1e-9),
            "requested_fps": int(benchmark_steps) / max(elapsed, 1e-9),
            "cuda_peak_memory_mb": (
                torch.cuda.max_memory_allocated()
                / (1024.0 * 1024.0)
                if torch.cuda.is_available()
                else 0.0
            ),
            **gpu,
        }

        write_json(
            run_dir / "benchmark_result.json",
            row,
        )
        return row

    except Exception as exc:
        try:
            gpu = sampler.stop()
        except Exception:
            gpu = {}

        row = {
            "status": "ERROR",
            "profile": profile.name,
            "n_envs": profile.n_envs,
            "n_steps": profile.n_steps,
            "batch_size": profile.batch_size,
            "elapsed_seconds": time.time() - started,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            **gpu,
        }

        write_json(
            run_dir / "benchmark_result.json",
            row,
        )
        return row

    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass

        try:
            base.restore_source_methods()
        except Exception:
            pass

        restore_topology_patch()

        model = None
        env = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def select_fastest_profile(
    *,
    root: Path,
    benchmark_steps: int,
    pad_separation: float,
    charger_capacity: int,
    device: str,
) -> SpeedProfile:
    rows = []

    print("\n" + "=" * 132)
    print("GPU/CPU THROUGHPUT BENCHMARK | representative cell = E6 / O10 / T3")
    print("=" * 132)

    for profile in SPEED_PROFILES:
        print(
            f"[benchmark] {profile.name}: "
            f"{profile.n_envs} env x {profile.n_steps}, batch={profile.batch_size}",
            flush=True,
        )

        row = benchmark_one_profile(
            profile=profile,
            benchmark_steps=benchmark_steps,
            root=root,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
            device=device,
        )
        rows.append(row)

        if row["status"] == "SUCCESS":
            print(
                f"    fps={row['actual_fps']:.1f} | "
                f"GPU mean={fnum(row.get('gpu_util_mean_pct')):.1f}% | "
                f"GPU max={fnum(row.get('gpu_util_max_pct')):.1f}% | "
                f"peak torch={fnum(row.get('cuda_peak_memory_mb')):.0f} MB"
            )
        else:
            print(f"    ERROR: {row.get('error')}")

    write_csv(
        root / "speed_benchmark.csv",
        rows,
    )

    success = [
        r
        for r in rows
        if r.get("status") == "SUCCESS"
        and np.isfinite(fnum(r.get("actual_fps")))
    ]

    if not success:
        raise RuntimeError(
            "All speed benchmark profiles failed. "
            "Inspect _speed_benchmark/*/benchmark_result.json."
        )

    best = max(
        success,
        key=lambda r: (
            fnum(r["actual_fps"], -math.inf),
            fnum(r.get("gpu_util_mean_pct"), -math.inf),
        ),
    )

    selected = next(
        p
        for p in SPEED_PROFILES
        if p.name == best["profile"]
    )

    write_json(
        root / "selected_speed_profile.json",
        {
            "selection_rule": "highest actual timesteps / wall-clock second",
            "representative_cell": "E6/O10/T3",
            "selected": asdict(selected),
            "benchmark_row": best,
        },
    )

    print("=" * 132)
    print(
        f"SELECTED FASTEST PROFILE: {selected.name} | "
        f"fps={best['actual_fps']:.1f}"
    )
    print("=" * 132)

    return selected


def load_selected_profile(root: Path) -> SpeedProfile:
    path = root / "selected_speed_profile.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No selected speed profile exists: {path}"
        )

    obj = json.loads(path.read_text(encoding="utf-8"))
    p = obj["selected"]

    return SpeedProfile(
        name=str(p["name"]),
        n_envs=int(p["n_envs"]),
        n_steps=int(p["n_steps"]),
        batch_size=int(p["batch_size"]),
    )


# =============================================================================
# Training scalar callback alias
# =============================================================================

def build_callbacks(
    *,
    run_dir: Path,
    profile: SpeedProfile,
    requested_steps: int,
):
    if CHECKPOINT_INTERVAL % profile.n_envs != 0:
        raise ValueError(
            "Checkpoint interval must be divisible by selected n_envs"
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_INTERVAL // profile.n_envs,
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

    return [checkpoint_cb, scalar_cb]


# =============================================================================
# Evaluation
# =============================================================================

def policy_probs(model: PPO, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor).distribution

        if getattr(dist, "probs", None) is not None:
            p = dist.probs
        elif getattr(dist, "logits", None) is not None:
            p = torch.softmax(dist.logits, dim=-1)
        else:
            raise RuntimeError("Policy distribution exposes neither probs nor logits")

        return np.asarray(
            p.detach().cpu().numpy(),
            dtype=float,
        ).reshape(-1, int(model.action_space.n))[0]


def active_system_count(scenario: ConservedFleetScenario) -> int:
    records = getattr(scenario, "person_travel_records", {}) or {}
    total = 0

    for recs in records.values():
        if not recs:
            continue
        rec = recs[-1]
        if rec.get("start_time") is not None and rec.get("end_time") is None:
            total += 1

    return total


def metrics_from_terminal_snapshot(
    *,
    snapshot: Dict[str, Any],
    action_counts: Counter,
    prob_rows: List[np.ndarray],
    system_person_minutes: float,
    reward_sum: float,
    episode_steps: int,
) -> Dict[str, Any]:
    rows = snapshot.get("rows", []) or []

    completed = [
        r
        for r in rows
        if r.get("finished")
        and np.isfinite(fnum(r.get("travel_time")))
    ]

    travel = np.asarray(
        [fnum(r["travel_time"]) for r in completed],
        dtype=float,
    )
    travel = travel[np.isfinite(travel)]

    def component(key: str) -> float:
        arr = np.asarray(
            [fnum(r.get(key)) for r in completed],
            dtype=float,
        )
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if len(arr) else float("nan")

    n = int(snapshot.get("n_persons", len(rows)))
    nf = len(completed)
    total_actions = max(1, sum(action_counts.values()))

    result = {
        "ATT": float(travel.mean()) if len(travel) else float("nan"),
        "AWT": component("wait"),
        "AGT_access": component("access"),
        "AFT": component("fly"),
        "travel_p50": float(np.percentile(travel, 50)) if len(travel) else float("nan"),
        "travel_p90": float(np.percentile(travel, 90)) if len(travel) else float("nan"),
        "travel_p95": float(np.percentile(travel, 95)) if len(travel) else float("nan"),
        "N": n,
        "N_finished": nf,
        "completion_rate": nf / n if n else float("nan"),
        "backlog": n - nf,
        "system_person_minutes": float(system_person_minutes),
        "system_person_minutes_per_passenger": (
            float(system_person_minutes) / n
            if n
            else float("nan")
        ),
        "episode_reward": float(reward_sum),
        "episode_steps": int(episode_steps),
    }

    n_actions = max(
        list(action_counts.keys()) + [0]
    ) + 1

    if prob_rows:
        prob_arr = np.vstack(prob_rows)
        n_actions = prob_arr.shape[1]

        entropy = -np.sum(
            prob_arr * np.log(
                np.clip(prob_arr, 1e-12, 1.0)
            ),
            axis=1,
        )

        result["policy_entropy_normalized"] = float(
            entropy.mean() / math.log(n_actions)
        )
        result["mean_policy_max_prob"] = float(
            np.max(prob_arr, axis=1).mean()
        )
    else:
        result["policy_entropy_normalized"] = float("nan")
        result["mean_policy_max_prob"] = float("nan")

    for action in range(n_actions):
        result[f"action_{action}_share"] = (
            action_counts.get(action, 0)
            / total_actions
        )

        if prob_rows:
            result[f"mean_policy_prob_{action}"] = float(
                np.vstack(prob_rows)[:, action].mean()
            )

    stats = snapshot.get("stage_stats", {}) or {}

    for key in (
        "single_pax_service_departures",
        "turnaround_starts",
        "turnaround_releases",
        "pad_departure_reservations",
        "pad_landing_reservations",
        "service_pad_blocks",
        "reposition_pad_blocks",
        "e6_charger_active_aircraft_steps",
        "e6_charger_wait_aircraft_steps",
        "e6_max_charger_queue",
    ):
        result[key] = stats.get(key, 0)

    result["final_queues"] = snapshot.get("queues", {})
    result["fleet_diag"] = snapshot.get("fleet_diag", {})

    return result


def evaluate_checkpoint(
    *,
    stage: str,
    obs_mode: str,
    topology: str,
    fleet_size: int,
    model_path: Path,
    vec_path: Path,
    train_step: int,
    eval_seed: int,
    run_dir: Path,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> Dict[str, Any]:
    seed_all(eval_seed)

    raw = build_eval_raw_env(
        stage=stage,
        obs_mode=obs_mode,
        topology=topology,
        fleet_size=fleet_size,
        run_dir=run_dir / "_eval_monitor",
        pad_separation=pad_separation,
        charger_capacity=charger_capacity,
        max_time=max_time,
    )

    env = VecNormalize.load(
        str(vec_path),
        raw,
    )
    env.training = False
    env.norm_reward = False

    try:
        model = PPO.load(
            str(model_path),
            env=env,
            device="cpu",
        )
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
        episode_steps = 0
        system_person_minutes = 0.0
        terminal_snapshot = None

        while not bool(done[0]):
            scenario = find_scenario(env)

            system_person_minutes += active_system_count(
                scenario
            )

            p = policy_probs(
                model,
                obs,
            )

            action, _ = model.predict(
                obs,
                deterministic=True,
            )
            ai = int(np.asarray(action).reshape(-1)[0])

            action_counts[ai] += 1
            prob_rows.append(p.copy())

            obs, reward, done, infos = env.step(action)

            reward_sum += fnum(
                np.asarray(reward).reshape(-1)[0],
                0.0,
            )
            episode_steps += 1

            if (
                infos
                and isinstance(infos[0], dict)
                and "terminal_snapshot" in infos[0]
            ):
                terminal_snapshot = infos[0][
                    "terminal_snapshot"
                ]

            if episode_steps > max_time + 100:
                raise RuntimeError(
                    "Evaluation episode exceeded max-time guard"
                )

        if terminal_snapshot is None:
            raise RuntimeError(
                "Terminal snapshot was not captured before VecEnv autoreset"
            )

        metrics = metrics_from_terminal_snapshot(
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
                "obs_mode": obs_mode,
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

        try:
            base.restore_source_methods()
        except Exception:
            pass

        restore_topology_patch()

        gc.collect()


def analyze_trained_cell(
    *,
    stage: str,
    obs_mode: str,
    topology: str,
    fleet_size: int,
    run_dir: Path,
    requested_steps: int,
    analysis_steps: Sequence[int],
    eval_seeds: Sequence[int],
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> Dict[str, Any]:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []

    valid_steps = sorted(
        set(
            int(s)
            for s in analysis_steps
            if int(s) <= int(requested_steps)
        )
    )

    if int(requested_steps) not in valid_steps:
        valid_steps.append(int(requested_steps))

    for step in valid_steps:
        model = (
            run_dir
            / "checkpoints"
            / f"uam_ppo_{step}_steps.zip"
        )
        vec = (
            run_dir
            / "checkpoints"
            / f"uam_ppo_vecnormalize_{step}_steps.pkl"
        )

        if not model.exists() or not vec.exists():
            continue

        for eval_seed in eval_seeds:
            try:
                row = evaluate_checkpoint(
                    stage=stage,
                    obs_mode=obs_mode,
                    topology=topology,
                    fleet_size=fleet_size,
                    model_path=model,
                    vec_path=vec,
                    train_step=step,
                    eval_seed=int(eval_seed),
                    run_dir=run_dir,
                    pad_separation=pad_separation,
                    charger_capacity=charger_capacity,
                    max_time=max_time,
                )
                rows.append(row)

                print(
                    f"    [eval] {step:,} seed={eval_seed} | "
                    f"ATT={row['ATT']:.3f} | "
                    f"AWT={row['AWT']:.3f} | "
                    f"finish={row['N_finished']}/{row['N']} | "
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

    write_csv(
        analysis_dir / "checkpoint_eval_raw.csv",
        rows,
    )
    write_csv(
        analysis_dir / "errors.csv",
        errors,
    )

    if not rows:
        raise RuntimeError(
            f"No post-training evaluation results for {stage}/{obs_mode}/{topology}"
        )

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["train_step"])].append(row)

    curve = []

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
        agg = {
            "train_step": step,
            "n_eval_seeds": len(group),
        }

        for key in metric_keys:
            agg[f"{key}_mean"] = fmean(
                r.get(key)
                for r in group
            )
            agg[f"{key}_std"] = fstd(
                r.get(key)
                for r in group
            )

        # Action-share columns are dynamic (2 or 3 actions).
        action_keys = sorted(
            {
                key
                for r in group
                for key in r.keys()
                if key.startswith("action_")
                and key.endswith("_share")
            }
        )

        for key in action_keys:
            agg[f"{key}_mean"] = fmean(
                r.get(key)
                for r in group
            )

        curve.append(agg)

    write_csv(
        analysis_dir / "checkpoint_curve.csv",
        curve,
    )

    final_candidates = [
        r
        for r in curve
        if int(r["train_step"]) == int(requested_steps)
    ]

    if not final_candidates:
        raise RuntimeError(
            f"Formal {requested_steps} checkpoint was not evaluated"
        )

    final = final_candidates[0]

    late = [
        r
        for r in curve
        if 600_000 <= int(r["train_step"]) <= int(requested_steps)
    ]

    summary = {
        "stage": stage,
        "obs_mode": obs_mode,
        "obs_description": OBS_DESCRIPTIONS[obs_mode],
        "topology": topology,
        "fleet_size": fleet_size,
        "formal_step": requested_steps,
        "final": final,
        "late_window_steps": [
            int(r["train_step"])
            for r in late
        ],
        "late_ATT_mean": fmean(
            r.get("ATT_mean")
            for r in late
        ),
        "late_ATT_checkpoint_std": fstd(
            r.get("ATT_mean")
            for r in late
        ),
        "late_AWT_mean": fmean(
            r.get("AWT_mean")
            for r in late
        ),
        "late_completion_mean": fmean(
            r.get("completion_rate_mean")
            for r in late
        ),
        "late_Jsys_per_passenger_mean": fmean(
            r.get("system_person_minutes_per_passenger_mean")
            for r in late
        ),
        "evaluation_errors": len(errors),
    }

    write_json(
        analysis_dir / "summary.json",
        summary,
    )

    lines = [
        f"{stage} / {obs_mode} / {topology}",
        "=" * 84,
        f"Final ATT       : {fnum(final.get('ATT_mean')):.3f}",
        f"Final AWT       : {fnum(final.get('AWT_mean')):.3f}",
        f"Final completion: {100*fnum(final.get('completion_rate_mean')):.2f}%",
        f"Final Jsys/N    : {fnum(final.get('system_person_minutes_per_passenger_mean')):.3f}",
        f"Final entropy   : {fnum(final.get('policy_entropy_normalized_mean')):.3f}",
        f"Late ATT mean   : {fnum(summary.get('late_ATT_mean')):.3f}",
        f"Late ATT ckpt SD: {fnum(summary.get('late_ATT_checkpoint_std')):.3f}",
    ]

    (analysis_dir / "summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return summary


# =============================================================================
# Matrix-level immediate factor analysis
# =============================================================================

def rebuild_factor_effects(master_rows: Sequence[Dict[str, Any]], root: Path) -> None:
    lookup = {}
    for row in master_rows:
        key = (
            row.get("stage"),
            row.get("topology"),
            row.get("obs_mode"),
        )
        lookup[key] = row

    rows = []

    for stage in STAGES:
        for topology in TOPOLOGIES:
            def g(obs: str, metric: str) -> float:
                r = lookup.get((stage, topology, obs))
                if not r:
                    return float("nan")
                return fnum(r.get(metric))

            for metric in (
                "final_ATT",
                "final_AWT",
                "final_completion",
                "final_Jsys_per_passenger",
                "late_ATT_mean",
            ):
                o00 = g("O00", metric)
                o10 = g("O10", metric)
                o01 = g("O01", metric)
                o11 = g("O11", metric)

                rows.append(
                    {
                        "stage": stage,
                        "topology": topology,
                        "metric": metric,
                        "O00": o00,
                        "O10": o10,
                        "O01": o01,
                        "O11": o11,
                        "aug_effect_with_passenger_O10_minus_O00": (
                            o10 - o00
                            if np.isfinite(o10) and np.isfinite(o00)
                            else np.nan
                        ),
                        "passenger_removal_effect_legacy_O01_minus_O00": (
                            o01 - o00
                            if np.isfinite(o01) and np.isfinite(o00)
                            else np.nan
                        ),
                        "aug_effect_without_passenger_O11_minus_O01": (
                            o11 - o01
                            if np.isfinite(o11) and np.isfinite(o01)
                            else np.nan
                        ),
                        "resource_x_passenger_interaction": (
                            (o11 - o01) - (o10 - o00)
                            if all(
                                np.isfinite(x)
                                for x in (o00, o10, o01, o11)
                            )
                            else np.nan
                        ),
                    }
                )

    write_csv(
        root / "factor_effects.csv",
        rows,
    )


def rebuild_latest_summary(master_rows: Sequence[Dict[str, Any]], root: Path) -> None:
    completed = len(master_rows)
    total = len(STAGES) * len(OBS_MODES) * len(TOPOLOGIES)

    feasible = [
        r
        for r in master_rows
        if fnum(r.get("final_completion")) >= 0.98
    ]

    best = (
        min(
            feasible,
            key=lambda r: fnum(r.get("final_ATT"), math.inf),
        )
        if feasible
        else None
    )

    lines = [
        "UAGMC E3-E6 observation/topology matrix",
        "=" * 100,
        f"Completed cells: {completed}/{total}",
        "",
    ]

    if best is not None:
        lines += [
            "Best currently completed >=98% completion cell:",
            (
                f"  {best['stage']} / {best['obs_mode']} / {best['topology']} | "
                f"ATT={fnum(best['final_ATT']):.3f} | "
                f"AWT={fnum(best['final_AWT']):.3f} | "
                f"completion={100*fnum(best['final_completion']):.2f}%"
            ),
            "",
        ]

    lines.append("Completed rows:")
    for row in master_rows:
        lines.append(
            f"  {row['stage']}/{row['obs_mode']}/{row['topology']} | "
            f"ATT={fnum(row['final_ATT']):.3f} | "
            f"AWT={fnum(row['final_AWT']):.3f} | "
            f"completion={100*fnum(row['final_completion']):.2f}% | "
            f"lateATT={fnum(row['late_ATT_mean']):.3f} | "
            f"fps={fnum(row['training_fps']):.1f}"
        )

    (root / "latest_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# Per-cell training
# =============================================================================

def cell_id(stage: str, obs_mode: str, topology: str) -> str:
    return f"{stage}__{obs_mode}__{topology}"


def completed_cell(run_dir: Path, requested_steps: int) -> bool:
    end = run_dir / "run_end.json"
    summary = run_dir / "analysis" / "summary.json"

    if not end.exists() or not summary.exists():
        return False

    obj = json.loads(end.read_text(encoding="utf-8"))
    return (
        str(obj.get("status", "")).upper() == "SUCCESS"
        and int(obj.get("requested_timesteps", -1)) == int(requested_steps)
    )


def train_cell(
    *,
    stage: str,
    obs_mode: str,
    topology: str,
    profile: SpeedProfile,
    root: Path,
    requested_steps: int,
    train_seed: int,
    fleet_size: int,
    pad_separation: float,
    charger_capacity: int,
    device: str,
    analysis_steps: Sequence[int],
    eval_seeds: Sequence[int],
    max_time: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cid = cell_id(stage, obs_mode, topology)
    run_dir = root / cid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    seed_all(train_seed)
    apply_speed_profile(profile)

    env = None
    model = None
    sampler = GPUSampler()

    started = time.time()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        env = build_vec_env(
            stage=stage,
            obs_mode=obs_mode,
            topology=topology,
            fleet_size=fleet_size,
            profile=profile,
            seed=train_seed,
            run_dir=run_dir,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
            max_time=max_time,
        )

        model = build_model(
            env=env,
            profile=profile,
            seed=train_seed,
            run_dir=run_dir,
            device=device,
        )

        run_manifest = {
            "cell_id": cid,
            "stage": stage,
            "obs_mode": obs_mode,
            "obs_description": OBS_DESCRIPTIONS[obs_mode],
            "topology": topology,
            "candidates": candidates_for(topology),
            "destination": DESTINATION,
            "t3_position": T3_POSITION if topology == "T3" else None,
            "t3_target_initial_allocation_N40": (
                {"0": 20, "1": 10, "3": 10}
                if topology == "T3" and fleet_size == 40
                else None
            ),
            "fleet_size": fleet_size,
            "train_seed": train_seed,
            "requested_timesteps": requested_steps,
            "max_time": max_time,
            "passenger_trace": str(TRAIN_FILE),
            "pad_separation_min": pad_separation,
            "charger_capacity": (
                charger_capacity if stage == "E6" else None
            ),
            "speed_profile": asdict(profile),
            "device": str(model.device),
            "ppo": {
                "n_epochs": base.N_EPOCHS,
                "gamma": base.GAMMA,
                "gae_lambda": base.GAE_LAMBDA,
                "clip_range": base.CLIP_RANGE,
                "ent_coef": base.ENT_COEF,
                "vf_coef": base.VF_COEF,
                "max_grad_norm": base.MAX_GRAD_NORM,
                "initial_lr": base.INITIAL_LR,
            },
        }

        write_json(
            run_dir / "run_manifest.json",
            run_manifest,
        )

        callbacks = build_callbacks(
            run_dir=run_dir,
            profile=profile,
            requested_steps=requested_steps,
        )

        print("\n" + "=" * 132)
        print(
            f"START CELL {cid} | "
            f"{requested_steps:,} steps | "
            f"{profile.name} | device={model.device}"
        )
        print("=" * 132)

        sampler.start()

        model.learn(
            total_timesteps=int(requested_steps),
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=True,
        )

        gpu_stats = sampler.stop()
        elapsed = time.time() - started
        actual_steps = int(model.num_timesteps)

        model.save(run_dir / "final_rl_model")
        env.save(run_dir / "final_vec_normalize.pkl")

        formal_model = (
            run_dir
            / "checkpoints"
            / f"uam_ppo_{requested_steps}_steps.zip"
        )
        formal_vec = (
            run_dir
            / "checkpoints"
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

        train_result = {
            "status": "SUCCESS",
            "cell_id": cid,
            "stage": stage,
            "obs_mode": obs_mode,
            "topology": topology,
            "requested_timesteps": requested_steps,
            "actual_rollout_final_timesteps": actual_steps,
            "elapsed_seconds": elapsed,
            "training_fps": actual_steps / max(elapsed, 1e-9),
            "formal_model": str(formal_model),
            "formal_vec": str(formal_vec),
            "speed_profile": asdict(profile),
            "cuda_peak_memory_mb": (
                torch.cuda.max_memory_allocated()
                / (1024.0 * 1024.0)
                if torch.cuda.is_available()
                else 0.0
            ),
            **gpu_stats,
            "finished": datetime.now().isoformat(timespec="seconds"),
        }

        write_json(
            run_dir / "run_end.json",
            train_result,
        )

        # Close training env and release CUDA before CPU post-analysis.
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
            f"DONE TRAIN {cid} | "
            f"time={elapsed/60:.2f} min | "
            f"fps={train_result['training_fps']:.1f} | "
            f"GPUmean={fnum(train_result.get('gpu_util_mean_pct')):.1f}%",
            flush=True,
        )

        print(
            f"IMMEDIATE ANALYSIS {cid}",
            flush=True,
        )

        analysis = analyze_trained_cell(
            stage=stage,
            obs_mode=obs_mode,
            topology=topology,
            fleet_size=fleet_size,
            run_dir=run_dir,
            requested_steps=requested_steps,
            analysis_steps=analysis_steps,
            eval_seeds=eval_seeds,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
            max_time=max_time,
        )

        return train_result, analysis

    except Exception as exc:
        try:
            gpu_stats = sampler.stop()
        except Exception:
            gpu_stats = {}

        error = {
            "status": "ERROR",
            "cell_id": cid,
            "stage": stage,
            "obs_mode": obs_mode,
            "topology": topology,
            "requested_timesteps": requested_steps,
            "elapsed_seconds": time.time() - started,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            **gpu_stats,
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

        try:
            base.restore_source_methods()
        except Exception:
            pass

        restore_topology_patch()

        model = None
        env = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def master_row_from(
    train_result: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    final = analysis["final"]

    row = {
        "cell_id": train_result["cell_id"],
        "stage": train_result["stage"],
        "obs_mode": train_result["obs_mode"],
        "topology": train_result["topology"],
        "final_ATT": final.get("ATT_mean"),
        "final_AWT": final.get("AWT_mean"),
        "final_access": final.get("AGT_access_mean"),
        "final_flight": final.get("AFT_mean"),
        "final_completion": final.get("completion_rate_mean"),
        "final_backlog": final.get("backlog_mean"),
        "final_Jsys_per_passenger": final.get(
            "system_person_minutes_per_passenger_mean"
        ),
        "final_p90": final.get("travel_p90_mean"),
        "final_entropy": final.get(
            "policy_entropy_normalized_mean"
        ),
        "late_ATT_mean": analysis.get("late_ATT_mean"),
        "late_ATT_checkpoint_std": analysis.get(
            "late_ATT_checkpoint_std"
        ),
        "late_AWT_mean": analysis.get("late_AWT_mean"),
        "late_completion_mean": analysis.get(
            "late_completion_mean"
        ),
        "late_Jsys_per_passenger_mean": analysis.get(
            "late_Jsys_per_passenger_mean"
        ),
        "training_fps": train_result.get("training_fps"),
        "gpu_util_mean_pct": train_result.get(
            "gpu_util_mean_pct"
        ),
        "gpu_util_max_pct": train_result.get(
            "gpu_util_max_pct"
        ),
        "cuda_peak_memory_mb": train_result.get(
            "cuda_peak_memory_mb"
        ),
        "elapsed_seconds": train_result.get(
            "elapsed_seconds"
        ),
        "formal_model": train_result.get(
            "formal_model"
        ),
        "formal_vec": train_result.get(
            "formal_vec"
        ),
    }

    for key, value in final.items():
        if key.startswith("action_") and key.endswith("_share_mean"):
            row[key] = value

    return row


# =============================================================================
# Bundle current results
# =============================================================================

def build_progress_zip(root: Path) -> Path:
    zpath = root / "UPLOAD_PROGRESS_MATRIX_RESULTS.zip"

    with zipfile.ZipFile(
        zpath,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        top_files = (
            "experiment_manifest.json",
            "speed_benchmark.csv",
            "selected_speed_profile.json",
            "matrix_results.csv",
            "factor_effects.csv",
            "latest_summary.txt",
            "progress.json",
            "errors.csv",
        )

        for name in top_files:
            path = root / name
            if path.exists():
                z.write(path, path.relative_to(root).as_posix())

        for summary in root.glob(
            "E*__O*__T*/analysis/summary.json"
        ):
            z.write(
                summary,
                summary.relative_to(root).as_posix(),
            )

        for summary in root.glob(
            "E*__O*__T*/analysis/summary.txt"
        ):
            z.write(
                summary,
                summary.relative_to(root).as_posix(),
            )

        for curve in root.glob(
            "E*__O*__T*/analysis/checkpoint_curve.csv"
        ):
            z.write(
                curve,
                curve.relative_to(root).as_posix(),
            )

    return zpath


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "UAGMC E3-E6 x O00/O10/O01/O11 x T2/T3 matrix, "
            "800k per cell with speed auto-selection and immediate analysis."
        )
    )

    p.add_argument(
        "--stages",
        default="E3,E4,E5,E6",
    )
    p.add_argument(
        "--obs-modes",
        default="O00,O10,O01,O11",
    )
    p.add_argument(
        "--topologies",
        default="T2,T3",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=FORMAL_TIMESTEPS,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=TRAIN_SEED,
    )
    p.add_argument(
        "--fleet-size",
        type=int,
        default=FLEET_SIZE,
    )
    p.add_argument(
        "--max-time",
        type=int,
        default=MAX_TIME,
    )
    p.add_argument(
        "--pad-separation",
        type=float,
        default=DEFAULT_PAD_SEPARATION_MIN,
    )
    p.add_argument(
        "--charger-capacity",
        type=int,
        default=DEFAULT_CHARGER_CAPACITY,
    )
    p.add_argument(
        "--benchmark-steps",
        type=int,
        default=50_000,
    )
    p.add_argument(
        "--skip-benchmark",
        action="store_true",
    )
    p.add_argument(
        "--rebenchmark",
        action="store_true",
    )
    p.add_argument(
        "--analysis-steps",
        default=",".join(
            str(x)
            for x in ANALYSIS_STEPS_DEFAULT
        ),
    )
    p.add_argument(
        "--eval-seeds",
        default="123,124,125",
    )
    p.add_argument(
        "--device",
        choices=["cuda", "auto", "cpu"],
        default="cuda",
    )
    p.add_argument(
        "--output-root",
        default=None,
    )
    p.add_argument(
        "--resume",
        action="store_true",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    return p.parse_args()


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False"
        )

    return value


def main() -> int:
    args = parse_args()

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)

    stages = parse_list(args.stages, STAGES)
    obs_modes = parse_list(args.obs_modes, OBS_MODES)
    topologies = parse_list(args.topologies, TOPOLOGIES)

    requested_steps = int(args.timesteps)
    train_seed = int(args.seed)
    fleet_size = int(args.fleet_size)
    max_time = int(args.max_time)
    pad_separation = float(args.pad_separation)
    charger_capacity = int(args.charger_capacity)
    benchmark_steps = int(args.benchmark_steps)
    device = resolve_device(args.device)

    analysis_steps = parse_ints(args.analysis_steps)
    eval_seeds = parse_ints(args.eval_seeds)

    if requested_steps <= 0:
        raise ValueError("timesteps must be positive")

    if requested_steps % CHECKPOINT_INTERVAL != 0:
        raise ValueError(
            f"--timesteps must be divisible by {CHECKPOINT_INTERVAL}"
        )

    if fleet_size <= 0:
        raise ValueError("fleet size must be positive")

    if pad_separation < 0:
        raise ValueError("pad separation must be non-negative")

    if charger_capacity <= 0:
        raise ValueError("charger capacity must be positive")

    cuda_info = enable_fast_cuda()

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
            / (
                f"uagmc_E3_E6_obs_topology_matrix_"
                f"{requested_steps//1000}k_seed{train_seed}_{stamp}"
            )
        ).resolve()

    root.mkdir(parents=True, exist_ok=True)

    matrix_cells = [
        (stage, obs, topology)
        for stage in stages
        for topology in topologies
        for obs in obs_modes
    ]

    manifest = {
        "experiment": "UAGMC_E3_E6_OBSERVATION_TOPOLOGY_MATRIX",
        "created": datetime.now().isoformat(timespec="seconds"),
        "stages": stages,
        "obs_modes": {
            k: OBS_DESCRIPTIONS[k]
            for k in obs_modes
        },
        "topologies": topologies,
        "matrix_cells": [
            cell_id(*cell)
            for cell in matrix_cells
        ],
        "n_cells": len(matrix_cells),
        "timesteps_per_cell": requested_steps,
        "formal_matrix_budget": len(matrix_cells) * requested_steps,
        "train_seed": train_seed,
        "fleet_size": fleet_size,
        "max_time_train_and_eval": max_time,
        "passenger_trace": str(TRAIN_FILE),
        "same_aggregate_demand_T2_T3": True,
        "destination": DESTINATION,
        "T2_candidates": T2_CANDIDATES,
        "T3_candidates": T3_CANDIDATES,
        "T3_position": T3_POSITION,
        "T3_charge_rate_design": (
            "V3 clones V1 charge rate to isolate topology-size effect"
        ),
        "T3_charge_rate_kwh_per_min": float(
            vertiport_spec_module.VERTIPORT_CHARGE_RATE.get("1", float("nan"))
        ),
        "T3_initial_allocation_N40": (
            {"0": 20, "1": 10, "3": 10}
            if fleet_size == 40
            else "50%/25%/remaining"
        ),
        "turnaround_delay_min": TURNAROUND_DELAY_MIN,
        "pad_separation_min": pad_separation,
        "charger_capacity_E6": charger_capacity,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "analysis_steps": analysis_steps,
        "eval_seeds": eval_seeds,
        "device_requested": device,
        "cuda_runtime": cuda_info,
        "speed_selection": (
            "benchmark representative E6/O10/T3 and choose highest actual FPS"
        ),
        "ppo_controls": {
            "n_epochs": base.N_EPOCHS,
            "gamma": base.GAMMA,
            "gae_lambda": base.GAE_LAMBDA,
            "clip_range": base.CLIP_RANGE,
            "ent_coef": base.ENT_COEF,
            "vf_coef": base.VF_COEF,
            "max_grad_norm": base.MAX_GRAD_NORM,
            "initial_lr": base.INITIAL_LR,
            "lr_schedule": "linear",
        },
    }

    write_json(
        root / "experiment_manifest.json",
        manifest,
    )

    print("=" * 132)
    print("UAGMC E3-E6 x OBSERVATION x TOPOLOGY MATRIX")
    print("=" * 132)
    print(f"Output root       : {root}")
    print(f"Cells             : {len(matrix_cells)}")
    print(f"Steps/cell        : {requested_steps:,}")
    print(f"Formal budget     : {len(matrix_cells)*requested_steps:,}")
    print(f"Fleet             : N={fleet_size}")
    print(f"Pad separation    : {pad_separation:.3f} min")
    print(f"E6 chargers       : {charger_capacity} / vertiport")
    print(f"Train/eval horizon: {max_time} min")
    print(f"Device            : {device}")
    if torch.cuda.is_available():
        print(f"GPU               : {cuda_info.get('gpu_name')}")
    print("=" * 132)

    if args.dry_run:
        for i, (stage, obs, topology) in enumerate(matrix_cells, 1):
            print(
                f"{i:02d}. {cell_id(stage, obs, topology)}"
            )
        return 0

    # -------------------------------------------------------------------------
    # Benchmark / load selected speed profile
    # -------------------------------------------------------------------------
    selected_path = root / "selected_speed_profile.json"

    if (
        selected_path.exists()
        and not args.rebenchmark
    ):
        profile = load_selected_profile(root)
        print(
            f"Reuse selected speed profile: {profile.name}"
        )
    elif args.skip_benchmark:
        profile = SPEED_PROFILES[0]
        write_json(
            selected_path,
            {
                "selection_rule": "benchmark skipped by user",
                "selected": asdict(profile),
            },
        )
        print(
            f"Benchmark skipped; use baseline profile {profile.name}"
        )
    else:
        profile = select_fastest_profile(
            root=root,
            benchmark_steps=benchmark_steps,
            pad_separation=pad_separation,
            charger_capacity=charger_capacity,
            device=device,
        )

    apply_speed_profile(profile)

    # -------------------------------------------------------------------------
    # Resume existing master results
    # -------------------------------------------------------------------------
    master_path = root / "matrix_results.csv"
    existing_rows_raw = read_csv(master_path)

    master_rows: List[Dict[str, Any]] = []
    completed_ids = set()

    for row in existing_rows_raw:
        master_rows.append(dict(row))
        completed_ids.add(str(row.get("cell_id")))

    errors_path = root / "errors.csv"

    total = len(matrix_cells)

    for index, (stage, obs_mode, topology) in enumerate(matrix_cells, 1):
        cid = cell_id(stage, obs_mode, topology)
        run_dir = root / cid

        if (
            args.resume
            and cid in completed_ids
            and completed_cell(run_dir, requested_steps)
        ):
            print(
                f"[{index:02d}/{total:02d}] SKIP completed {cid}",
                flush=True,
            )
            continue

        print(
            f"\n[{index:02d}/{total:02d}] RUN {cid}",
            flush=True,
        )

        try:
            train_result, analysis = train_cell(
                stage=stage,
                obs_mode=obs_mode,
                topology=topology,
                profile=profile,
                root=root,
                requested_steps=requested_steps,
                train_seed=train_seed,
                fleet_size=fleet_size,
                pad_separation=pad_separation,
                charger_capacity=charger_capacity,
                device=device,
                analysis_steps=analysis_steps,
                eval_seeds=eval_seeds,
                max_time=max_time,
            )

            row = master_row_from(
                train_result,
                analysis,
            )

            # Replace same cell on rerun instead of duplicating.
            master_rows = [
                r
                for r in master_rows
                if str(r.get("cell_id")) != cid
            ]
            master_rows.append(row)

            master_rows.sort(
                key=lambda r: (
                    STAGES.index(str(r["stage"])),
                    TOPOLOGIES.index(str(r["topology"])),
                    OBS_MODES.index(str(r["obs_mode"])),
                )
            )

            write_csv(
                master_path,
                master_rows,
            )

            rebuild_factor_effects(
                master_rows,
                root,
            )
            rebuild_latest_summary(
                master_rows,
                root,
            )

            progress = {
                "completed_cells": len(master_rows),
                "total_cells": total,
                "last_completed": cid,
                "selected_speed_profile": asdict(profile),
                "updated": datetime.now().isoformat(timespec="seconds"),
            }

            write_json(
                root / "progress.json",
                progress,
            )

            bundle = build_progress_zip(root)

            print(
                f"SAVED ANALYSIS {cid} | "
                f"matrix={len(master_rows)}/{total} | "
                f"bundle={bundle.name}",
                flush=True,
            )

        except Exception as exc:
            err = {
                "cell_id": cid,
                "stage": stage,
                "obs_mode": obs_mode,
                "topology": topology,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "time": datetime.now().isoformat(timespec="seconds"),
            }

            append_csv(
                errors_path,
                err,
            )

            print(
                f"ERROR {cid}: {repr(exc)}",
                flush=True,
            )

            if not args.continue_on_error:
                raise

    # -------------------------------------------------------------------------
    # Final save
    # -------------------------------------------------------------------------
    rebuild_factor_effects(
        master_rows,
        root,
    )
    rebuild_latest_summary(
        master_rows,
        root,
    )
    final_bundle = build_progress_zip(root)

    write_json(
        root / "progress.json",
        {
            "completed_cells": len(master_rows),
            "total_cells": total,
            "selected_speed_profile": asdict(profile),
            "finished": datetime.now().isoformat(timespec="seconds"),
        },
    )

    print("\n" + "=" * 132)
    print("MATRIX RUN COMPLETE / CURRENT PROGRESS SAVED")
    print("=" * 132)
    print(f"Completed: {len(master_rows)}/{total}")
    print(f"Master   : {master_path}")
    print(f"Factors  : {root / 'factor_effects.csv'}")
    print(f"Summary  : {root / 'latest_summary.txt'}")
    print(f"Upload   : {final_bundle}")
    print("=" * 132)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
