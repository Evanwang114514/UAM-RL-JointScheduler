# -*- coding: utf-8 -*-
"""
Diagnose the FORMAL45 checkpoint with the most unfinished passengers.

What this script does
---------------------
1) Scans the passenger-only FORMAL45 cells:
      E0/E1/E2/E4/E5 x M0/M1/M2/M3/M4
   and automatically selects the evaluated checkpoint with the LOWEST
   N_finished (unless --cell/--step overrides it).

2) Replays that exact trained PPO checkpoint with the exact FORMAL45
   environment/observation semantics, but gives the simulator a larger
   safety horizon so that it does NOT auto-reset at t=600.

3) Captures the REAL simulator state immediately after processing t=600:
   - every unfinished passenger
   - passenger state/sub_state/timer/chosen vertiport
   - vertiport queue membership + queue position
   - onboard-aircraft membership
   - per-passenger time_stats and travel record
   - aircraft states / batteries / turnaround / charge queue
   - queue/resource summary

4) Continues the SAME deterministic policy after t=600, with no new demand
   after the original demand horizon, until:
      a) all passengers finish, or
      b) --safety-cap is reached.
   This reveals whether the t=600 "unfinished" passengers are merely
   right-censored backlog or truly stuck/deadlocked.

Outputs
-------
<run_root>/diagnostics/worst_unfinished_<cell>_<step>_<timestamp>/
    selection.json
    summary.json
    unfinished_at_600.csv
    unfinished_at_600_with_final.csv
    all_passengers_final.csv
    queues_at_600.csv
    aircraft_at_600.csv
    resource_at_600.json
    timeline.csv
    DIAGNOSTIC_RESULTS.zip

Typical use
-----------
python diagnose_formal45_worst_unfinished.py

Force a particular checkpoint:
python diagnose_formal45_worst_unfinished.py --cell E0__M0 --step 800000

Do not drain after t=600:
python diagnose_formal45_worst_unfinished.py --no-drain
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parent

# Exact formal implementation used for the completed 45-cell run.
import train_uagmc_45x800k_formal_JOINTFIX as formal


PASSENGER_ROWS = ("E0", "E1", "E2", "E4", "E5")
METHODS = ("M0", "M1", "M2", "M3", "M4")

DEFAULT_RUN_ROOT = (
    ROOT
    / "serial_runs"
    / "uagmc_45x800k_T2_seed1_20260921_005821"
)

DEFAULT_HORIZON = 600
DEFAULT_SAFETY_CAP = 2000
DEFAULT_EVAL_SEED = 123


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def jint(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def enum_name(x: Any) -> str:
    try:
        return str(x.name)
    except Exception:
        return str(x)


def pid_of(x: Any) -> str:
    if isinstance(x, (str, int, np.integer)):
        return str(x)
    for attr in ("person_id", "id", "pid"):
        if hasattr(x, attr):
            try:
                return str(getattr(x, attr))
            except Exception:
                pass
    return repr(x)


def jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if hasattr(x, "name"):
        try:
            return str(x.name)
        except Exception:
            pass
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(obj), ensure_ascii=False, indent=2),
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
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = {}
            for k in fields:
                v = row.get(k, "")
                if isinstance(v, (dict, list, tuple, set, np.ndarray)):
                    v = json.dumps(jsonable(v), ensure_ascii=False)
                out[k] = v
            w.writerow(out)


def zip_dir(folder: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.resolve() != zip_path.resolve():
                zf.write(p, p.relative_to(folder))


# -----------------------------------------------------------------------------
# Select the checkpoint with the largest unfinished backlog
# -----------------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def scan_worst_checkpoint(
    run_root: Path,
    eval_seed: int,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []

    for stage in PASSENGER_ROWS:
        for method in METHODS:
            cell = f"{stage}__{method}"
            p = run_root / cell / "analysis" / "checkpoint_eval_raw.csv"
            for row in read_csv_rows(p):
                if "eval_seed" in row and str(row.get("eval_seed", "")) != str(eval_seed):
                    continue

                n = jint(row.get("N"), 300)
                nf = jint(row.get("N_finished"), n)
                step = jint(row.get("train_step"), -1)
                if step <= 0:
                    continue

                j = fnum(row.get("system_person_minutes_per_passenger"), -1.0)
                att = fnum(row.get("ATT"), float("nan"))

                candidates.append(
                    {
                        "cell": cell,
                        "stage": stage,
                        "method": method,
                        "train_step": step,
                        "N": n,
                        "N_finished": nf,
                        "backlog": n - nf,
                        "completion_rate": (nf / n if n else float("nan")),
                        "Jsys_per_passenger": j,
                        "ATT": att,
                        "source_csv": str(p),
                    }
                )

    if not candidates:
        # Conservative fallback if analysis CSVs were removed from the bundle.
        return {
            "cell": "E0__M0",
            "stage": "E0",
            "method": "M0",
            "train_step": 800000,
            "N": None,
            "N_finished": None,
            "backlog": None,
            "completion_rate": None,
            "Jsys_per_passenger": None,
            "ATT": None,
            "selection_fallback": True,
        }

    # Primary key: fewest finished.
    # Tie-break: highest Jsys, then later checkpoint.
    candidates.sort(
        key=lambda r: (
            int(r["N_finished"]),
            -fnum(r["Jsys_per_passenger"], -1.0),
            -int(r["train_step"]),
            str(r["cell"]),
        )
    )
    out = dict(candidates[0])
    out["selection_fallback"] = False
    out["num_evaluated_candidates_scanned"] = len(candidates)
    return out


def parse_cell(cell: str) -> Tuple[str, str]:
    s = str(cell).upper().strip()
    if "__" not in s:
        raise ValueError("--cell must look like E0__M0")
    stage, method = s.split("__", 1)
    if stage not in PASSENGER_ROWS:
        raise ValueError(f"passenger-only stage required; got {stage}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method}")
    return stage, method


# -----------------------------------------------------------------------------
# Real simulator state extraction
# -----------------------------------------------------------------------------

def get_person_dict(scenario: Any) -> Dict[str, Any]:
    obj = getattr(scenario, "persons", None)
    raw = getattr(obj, "persons", {}) if obj is not None else {}
    return {str(k): v for k, v in (raw or {}).items()}


def get_records(scenario: Any) -> Dict[str, List[Dict[str, Any]]]:
    raw = getattr(scenario, "person_travel_records", {}) or {}
    return {str(k): list(v or []) for k, v in raw.items()}


def finished_set(scenario: Any) -> set[str]:
    return {str(x) for x in (getattr(scenario, "finished_ids", []) or [])}


def all_evtols(scenario: Any) -> List[Any]:
    raw = getattr(scenario, "_all_evtols", None)
    if isinstance(raw, dict) and raw:
        return list(raw.values())

    seen = {}
    vp_builder = getattr(scenario, "vertiports", None)
    at_vp = getattr(vp_builder, "evtols_at_vertiport", {}) if vp_builder is not None else {}
    for xs in (at_vp or {}).values():
        for e in xs or []:
            seen[str(getattr(e, "id", id(e)))] = e

    evtol_builder = getattr(scenario, "evtols", None)
    for attr in ("evtols", "_all_evtols"):
        d = getattr(evtol_builder, attr, None) if evtol_builder is not None else None
        if isinstance(d, dict):
            for e in d.values():
                seen[str(getattr(e, "id", id(e)))] = e
    return list(seen.values())


def queue_membership(scenario: Any) -> Dict[str, Tuple[str, int]]:
    out: Dict[str, Tuple[str, int]] = {}
    vp_builder = getattr(scenario, "vertiports", None)
    vp_list = getattr(vp_builder, "vertiport_list", {}) if vp_builder is not None else {}
    for vid, vp in (vp_list or {}).items():
        items = list(getattr(vp, "person_list", []) or [])
        for i, item in enumerate(items):
            out[pid_of(item)] = (str(vid), i)
    return out


def onboard_membership(scenario: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for e in all_evtols(scenario):
        eid = str(getattr(e, "id", ""))
        for p in list(getattr(e, "passenger_ids", []) or []):
            out[pid_of(p)] = eid
    return out


def aircraft_state_rows(
    scenario: Any,
    charger_capacity: int,
) -> List[Dict[str, Any]]:
    evs = all_evtols(scenario)

    # Derive finite-charger position exactly as the E4/E5 patch does:
    # sort CHARGING aircraft at each station by queue-enter time; first cap active.
    charging_by_vid: Dict[str, List[Any]] = defaultdict(list)
    for e in evs:
        st = enum_name(getattr(e, "state", "UNKNOWN"))
        if st == "CHARGING":
            vid = str(getattr(e, "current_vertiport_id", ""))
            charging_by_vid[vid].append(e)

    charger_role: Dict[str, str] = {}
    charger_rank: Dict[str, int] = {}
    for vid, xs in charging_by_vid.items():
        xs.sort(
            key=lambda e: (
                fnum(getattr(e, "_e6_charge_queue_enter_time", math.inf), math.inf),
                str(getattr(e, "id", "")),
            )
        )
        for i, e in enumerate(xs):
            eid = str(getattr(e, "id", ""))
            charger_rank[eid] = i
            charger_role[eid] = "ACTIVE" if i < int(charger_capacity) else "WAITING"

    rows = []
    for e in sorted(evs, key=lambda x: str(getattr(x, "id", ""))):
        eid = str(getattr(e, "id", ""))
        spec = getattr(e, "spec", None)
        rows.append(
            {
                "aircraft_id": eid,
                "state": enum_name(getattr(e, "state", "UNKNOWN")),
                "current_vertiport_id": getattr(e, "current_vertiport_id", None),
                "target_vertiport_id": getattr(e, "target_vertiport_id", None),
                "passenger_ids": [pid_of(x) for x in (getattr(e, "passenger_ids", []) or [])],
                "remaining_flight_time": getattr(e, "remaining_flight_time", None),
                "battery_kwh": getattr(e, "battery_kwh", None),
                "battery_capacity_kwh": getattr(spec, "battery_capacity_kwh", None),
                "charge_rate_kwh_per_min": getattr(spec, "charge_rate_kwh_per_min", None),
                "turnaround_busy": bool(getattr(e, "_e345_turnaround_busy", False)),
                "turnaround_release_time": getattr(e, "_e345_turnaround_release_time", None),
                "post_turnaround_state": enum_name(
                    getattr(e, "_e345_post_turnaround_state", "")
                ),
                "charge_queue_enter_time": getattr(e, "_e6_charge_queue_enter_time", None),
                "finite_charger_role": charger_role.get(eid, ""),
                "finite_charger_rank": charger_rank.get(eid, ""),
            }
        )
    return rows


def queue_rows(scenario: Any) -> List[Dict[str, Any]]:
    rows = []
    vp_builder = getattr(scenario, "vertiports", None)
    vp_list = getattr(vp_builder, "vertiport_list", {}) if vp_builder is not None else {}
    for vid, vp in sorted((vp_list or {}).items(), key=lambda kv: str(kv[0])):
        people = [pid_of(x) for x in (getattr(vp, "person_list", []) or [])]
        rows.append(
            {
                "vertiport_id": str(vid),
                "queue_len": len(people),
                "wait_person_field": getattr(vp, "wait_person", None),
                "leave_person_field": getattr(vp, "leave_person", None),
                "passenger_ids": people,
            }
        )
    return rows


def person_row(
    scenario: Any,
    pid: str,
    person: Any,
    qmap: Dict[str, Tuple[str, int]],
    onboard: Dict[str, str],
) -> Dict[str, Any]:
    records = get_records(scenario)
    recs = records.get(str(pid), [])
    rec = recs[-1] if recs else {}
    stats = getattr(person, "time_stats", {}) or {}
    q = qmap.get(str(pid))

    start = rec.get("start_time")
    end = rec.get("end_time")
    dtd = None
    if start is not None and end is not None:
        try:
            dtd = float(end) - float(start)
        except Exception:
            pass

    # Try several likely request/spawn-time names without assuming one implementation.
    spawn_time = None
    for attr in ("spawn_time", "request_time", "generation_time", "arrival_time"):
        if hasattr(person, attr):
            spawn_time = getattr(person, attr)
            break

    return {
        "pid": str(pid),
        "finished": (
            str(pid) in finished_set(scenario)
            or end is not None
            or str(getattr(person, "state", "")).lower() == "finished"
        ),
        "state": getattr(person, "state", None),
        "sub_state": getattr(person, "sub_state", None),
        "method": getattr(person, "method", None),
        "spawn_time_attr": spawn_time,
        "decision_start_time": start,
        "end_time": end,
        "DTD_if_finished": dtd,
        "chosen_from_vertiport": rec.get("from"),
        "chosen_to_vertiport": rec.get("to"),
        "origin_vertiport_id_attr": getattr(person, "origin_vertiport_id", None),
        "current_timer": getattr(person, "current_timer", None),
        "t_drive_pickup": getattr(person, "t_drive_pickup", None),
        "in_vertiport_queue": q is not None,
        "queue_vertiport_id": (q[0] if q else ""),
        "queue_position_zero_based": (q[1] if q else ""),
        "onboard_aircraft_id": onboard.get(str(pid), ""),
        "time_to_vertiport": stats.get("to_vertiport"),
        "time_wait_uam": stats.get("wait_uam"),
        "time_fly": stats.get("fly"),
        "time_stats_json": dict(stats),
        "last_travel_record_json": dict(rec),
        "all_travel_records_json": list(recs),
    }


def snapshot_real_state(
    scenario: Any,
    *,
    charger_capacity: int,
    label: str,
) -> Dict[str, Any]:
    persons = get_person_dict(scenario)
    qmap = queue_membership(scenario)
    onboard = onboard_membership(scenario)
    fset = finished_set(scenario)

    passenger_rows = [
        person_row(scenario, pid, person, qmap, onboard)
        for pid, person in sorted(persons.items(), key=lambda kv: kv[0])
    ]
    unfinished = [r for r in passenger_rows if not bool(r["finished"])]

    arows = aircraft_state_rows(scenario, charger_capacity=charger_capacity)
    qrows = queue_rows(scenario)

    state_counts = Counter(str(r["state"]) for r in arows)
    queue_total = sum(int(r["queue_len"]) for r in qrows)

    pad_calendar = getattr(scenario, "_e345_pad_calendar", {}) or {}
    stage_stats = getattr(scenario, "_e345_stats", {}) or {}

    try:
        current_superset = formal._current_superset_vector(
            scenario,
            row=str(getattr(scenario, "_formal45_row", "")),
            charger_capacity=int(charger_capacity),
        )
    except Exception as exc:
        current_superset = {"error": repr(exc)}

    return {
        "label": label,
        "scenario_time": getattr(scenario, "time", None),
        "N": len(persons),
        "N_finished": len([r for r in passenger_rows if r["finished"]]),
        "N_unfinished": len(unfinished),
        "finished_ids_count": len(fset),
        "queue_total": queue_total,
        "waiting_decisions": [pid_of(x) for x in (getattr(scenario, "waiting_decisions", []) or [])],
        "queue_rows": qrows,
        "aircraft_state_counts": dict(state_counts),
        "aircraft_rows": arows,
        "unfinished_rows": unfinished,
        "all_passenger_rows": passenger_rows,
        "pad_calendar": pad_calendar,
        "stage_stats": stage_stats,
        "current_superset_vector": current_superset,
    }


def timeline_row(
    scenario: Any,
    *,
    processed_time: int,
    charger_capacity: int,
) -> Dict[str, Any]:
    persons = get_person_dict(scenario)
    fset = finished_set(scenario)
    qrows = queue_rows(scenario)
    arows = aircraft_state_rows(scenario, charger_capacity)
    states = Counter(str(r["state"]) for r in arows)

    row: Dict[str, Any] = {
        "processed_time": int(processed_time),
        "scenario_time_after_step": getattr(scenario, "time", None),
        "N": len(persons),
        "N_finished": len(fset),
        "backlog": len(persons) - len(fset),
        "queue_total": sum(int(x["queue_len"]) for x in qrows),
        "waiting_decisions": len(getattr(scenario, "waiting_decisions", []) or []),
    }
    for q in qrows:
        row[f"queue_V{q['vertiport_id']}"] = q["queue_len"]
    for k, v in states.items():
        row[f"aircraft_{k}"] = int(v)

    charging_active = sum(1 for x in arows if x["finite_charger_role"] == "ACTIVE")
    charging_wait = sum(1 for x in arows if x["finite_charger_role"] == "WAITING")
    turn_busy = sum(1 for x in arows if x["turnaround_busy"])
    row["charging_active"] = charging_active
    row["charging_waiting"] = charging_wait
    row["turnaround_busy"] = turn_busy
    return row


# -----------------------------------------------------------------------------
# Main replay
# -----------------------------------------------------------------------------

def resolve_checkpoint(
    run_root: Path,
    *,
    cell_override: Optional[str],
    step_override: Optional[int],
    eval_seed: int,
) -> Dict[str, Any]:
    if cell_override:
        stage, method = parse_cell(cell_override)
        step = int(step_override or 800000)
        selection = {
            "cell": f"{stage}__{method}",
            "stage": stage,
            "method": method,
            "train_step": step,
            "manual_override": True,
        }
    else:
        selection = scan_worst_checkpoint(run_root, eval_seed)
        if step_override:
            selection["train_step"] = int(step_override)
        selection["manual_override"] = False

    cell = selection["cell"]
    step = int(selection["train_step"])
    cell_dir = run_root / cell
    model = cell_dir / "checkpoints" / f"uam_ppo_{step}_steps.zip"
    vec = cell_dir / "checkpoints" / f"uam_ppo_vecnormalize_{step}_steps.pkl"

    if not model.exists():
        raise FileNotFoundError(model)
    if not vec.exists():
        raise FileNotFoundError(vec)

    selection["cell_dir"] = str(cell_dir)
    selection["model_path"] = str(model)
    selection["vecnormalize_path"] = str(vec)
    return selection


def run(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root).expanduser()
    if not run_root.is_absolute():
        run_root = (ROOT / run_root).resolve()
    else:
        run_root = run_root.resolve()

    if not run_root.exists():
        raise FileNotFoundError(run_root)

    selection = resolve_checkpoint(
        run_root,
        cell_override=args.cell,
        step_override=args.step,
        eval_seed=int(args.eval_seed),
    )
    stage = str(selection["stage"])
    method = str(selection["method"])
    step = int(selection["train_step"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        run_root
        / "diagnostics"
        / f"worst_unfinished_{selection['cell']}_{step}_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "selection.json", selection)

    print("=" * 110)
    print("FORMAL45 WORST-UNFINISHED DIAGNOSTIC")
    print("=" * 110)
    print(f"run root : {run_root}")
    print(f"selected : {selection['cell']} @ {step:,}")
    if selection.get("N_finished") is not None:
        print(
            f"analysis  : finish={selection.get('N_finished')}/{selection.get('N')} "
            f"| backlog={selection.get('backlog')} "
            f"| completion={selection.get('completion_rate')}"
        )
    print(f"model    : {selection['model_path']}")
    print(f"vecnorm  : {selection['vecnormalize_path']}")
    print(f"horizon  : t={args.horizon}")
    print(f"drain    : {not args.no_drain} | safety cap={args.safety_cap}")
    print("=" * 110, flush=True)

    # Important: environment max_time is deliberately ABOVE the diagnostic safety
    # cap. We stop manually, so DummyVecEnv cannot auto-reset and erase the real
    # terminal state we want to inspect.
    env_max_time = int(args.safety_cap) + 10

    factory = formal.make_formal_env_factory(
        row=stage,
        method=method,
        fleet_size=int(args.fleet_size),
        env_index=9999,
        run_dir=out_dir / "_monitor",
        future_horizon=float(args.future_horizon),
        max_events=int(args.max_events),
        pad_separation=float(args.pad_separation),
        charger_capacity=int(args.charger_capacity),
        uq_delta=float(formal.ETA_UQ_DELTA_MIN),
        max_time=env_max_time,
    )

    raw = DummyVecEnv([factory])
    env = VecNormalize.load(selection["vecnormalize_path"], raw)
    env.training = False
    env.norm_reward = False

    model = PPO.load(
        selection["model_path"],
        env=env,
        device="cpu",
    )
    model.policy.set_training_mode(False)

    try:
        try:
            env.seed(int(args.eval_seed))
        except Exception:
            pass
        obs = env.reset()

        timeline: List[Dict[str, Any]] = []
        snap600: Optional[Dict[str, Any]] = None
        stop_snapshot: Optional[Dict[str, Any]] = None
        unfinished_ids_at_600: List[str] = []

        step_counter = 0
        drain_success = False
        drain_finish_processed_time: Optional[int] = None

        while True:
            scenario = formal.mx.find_scenario(env)
            before_time = int(getattr(scenario, "time", step_counter))

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            step_counter += 1

            # We intentionally configured max_time above safety_cap, therefore
            # done should remain false during our diagnostic window.
            scenario = formal.mx.find_scenario(env)
            after_time = int(getattr(scenario, "time", before_time + 1))
            processed_time = after_time - 1

            if (
                processed_time >= int(args.timeline_start)
                and (
                    processed_time == int(args.horizon)
                    or processed_time % int(args.timeline_interval) == 0
                )
            ):
                timeline.append(
                    timeline_row(
                        scenario,
                        processed_time=processed_time,
                        charger_capacity=int(args.charger_capacity),
                    )
                )

            if snap600 is None and processed_time >= int(args.horizon):
                snap600 = snapshot_real_state(
                    scenario,
                    charger_capacity=int(args.charger_capacity),
                    label=f"after_processing_t{args.horizon}",
                )
                unfinished_ids_at_600 = [
                    str(r["pid"]) for r in snap600["unfinished_rows"]
                ]

                print(
                    f"[t={args.horizon}] "
                    f"finished={snap600['N_finished']}/{snap600['N']} | "
                    f"unfinished={snap600['N_unfinished']} | "
                    f"queue_total={snap600['queue_total']}",
                    flush=True,
                )

                write_csv(out_dir / "unfinished_at_600.csv", snap600["unfinished_rows"])
                write_csv(out_dir / "queues_at_600.csv", snap600["queue_rows"])
                write_csv(out_dir / "aircraft_at_600.csv", snap600["aircraft_rows"])
                write_json(
                    out_dir / "resource_at_600.json",
                    {
                        k: v
                        for k, v in snap600.items()
                        if k not in ("unfinished_rows", "all_passenger_rows", "aircraft_rows", "queue_rows")
                    },
                )

                if args.no_drain:
                    stop_snapshot = snap600
                    break

            # Drain-to-empty test.
            persons = get_person_dict(scenario)
            nf = len(finished_set(scenario))
            if snap600 is not None and persons and nf >= len(persons):
                drain_success = True
                drain_finish_processed_time = processed_time
                stop_snapshot = snapshot_real_state(
                    scenario,
                    charger_capacity=int(args.charger_capacity),
                    label="drain_all_finished",
                )
                print(
                    f"[DRAIN COMPLETE] all {nf}/{len(persons)} finished "
                    f"after processing t={processed_time}",
                    flush=True,
                )
                break

            if processed_time >= int(args.safety_cap):
                stop_snapshot = snapshot_real_state(
                    scenario,
                    charger_capacity=int(args.charger_capacity),
                    label=f"safety_cap_{args.safety_cap}",
                )
                print(
                    f"[SAFETY CAP] finished={stop_snapshot['N_finished']}/"
                    f"{stop_snapshot['N']} at t={processed_time}",
                    flush=True,
                )
                break

            if bool(np.asarray(done).reshape(-1)[0]):
                # This should not occur because env_max_time > safety_cap.
                raise RuntimeError(
                    "Environment terminated before manual safety cap; "
                    "detailed state may have been auto-reset."
                )

        if snap600 is None:
            raise RuntimeError("Failed to capture the t=600 state")
        if stop_snapshot is None:
            raise RuntimeError("Failed to capture final diagnostic state")

        # Merge t=600 unfinished passengers with their eventual drain result.
        final_by_pid = {
            str(r["pid"]): r
            for r in stop_snapshot["all_passenger_rows"]
        }
        merged = []
        for r600 in snap600["unfinished_rows"]:
            pid = str(r600["pid"])
            rf = final_by_pid.get(pid, {})
            end = rf.get("end_time")
            remaining_after_600 = None
            if end is not None:
                try:
                    remaining_after_600 = float(end) - float(args.horizon)
                except Exception:
                    pass

            merged.append(
                {
                    **{f"at600_{k}": v for k, v in r600.items()},
                    "final_finished": rf.get("finished"),
                    "final_state": rf.get("state"),
                    "final_sub_state": rf.get("sub_state"),
                    "final_end_time": end,
                    "final_DTD": rf.get("DTD_if_finished"),
                    "remaining_time_after_600": remaining_after_600,
                    "final_queue_vertiport_id": rf.get("queue_vertiport_id"),
                    "final_onboard_aircraft_id": rf.get("onboard_aircraft_id"),
                }
            )

        write_csv(out_dir / "unfinished_at_600_with_final.csv", merged)
        write_csv(out_dir / "all_passengers_final.csv", stop_snapshot["all_passenger_rows"])
        write_csv(out_dir / "timeline.csv", timeline)

        finished_dtd = [
            fnum(r.get("DTD_if_finished"))
            for r in stop_snapshot["all_passenger_rows"]
            if bool(r.get("finished")) and math.isfinite(fnum(r.get("DTD_if_finished")))
        ]
        all300_att = (
            float(np.mean(finished_dtd))
            if drain_success and len(finished_dtd) == int(stop_snapshot["N"])
            else None
        )

        location_counts = Counter()
        for r in snap600["unfinished_rows"]:
            if r.get("in_vertiport_queue"):
                location_counts[f"queue_V{r.get('queue_vertiport_id')}"] += 1
            elif r.get("onboard_aircraft_id"):
                location_counts["onboard_aircraft"] += 1
            else:
                sub = str(r.get("sub_state") or "")
                state = str(r.get("state") or "")
                location_counts[f"other:{state}/{sub}"] += 1

        summary = {
            "selected_checkpoint": selection,
            "eval_seed": int(args.eval_seed),
            "original_formal_horizon": int(args.horizon),
            "replay_environment_internal_max_time": env_max_time,
            "safety_cap": int(args.safety_cap),
            "t600": {
                "N": snap600["N"],
                "N_finished": snap600["N_finished"],
                "N_unfinished": snap600["N_unfinished"],
                "completion_rate": (
                    snap600["N_finished"] / snap600["N"]
                    if snap600["N"] else None
                ),
                "queue_total": snap600["queue_total"],
                "unfinished_location_counts": dict(location_counts),
                "aircraft_state_counts": snap600["aircraft_state_counts"],
                "stage_stats": snap600["stage_stats"],
            },
            "drain": {
                "enabled": not args.no_drain,
                "success_all_finished": drain_success,
                "all_finished_processed_time": drain_finish_processed_time,
                "final_N_finished": stop_snapshot["N_finished"],
                "final_N_unfinished": stop_snapshot["N_unfinished"],
                "all_passenger_ATT_if_fully_drained": all300_att,
                "unfinished_from_t600_count": len(unfinished_ids_at_600),
                "unfinished_from_t600_eventually_finished": sum(
                    1 for r in merged if bool(r.get("final_finished"))
                ),
            },
            "interpretation_hint": (
                "If most/all t=600 unfinished passengers later finish during drain, "
                "the original completion deficit is right-censoring/backlog at the "
                "fixed horizon rather than passenger loss. If many remain unfinished "
                "at the safety cap, inspect their states/queues/aircraft resources for "
                "a true deadlock or capacity defect."
            ),
        }
        write_json(out_dir / "summary.json", summary)

        zip_path = out_dir / "DIAGNOSTIC_RESULTS.zip"
        zip_dir(out_dir, zip_path)

        print("-" * 110)
        print(f"RESULT DIR : {out_dir}")
        print(f"UPLOAD ZIP : {zip_path}")
        print(
            f"t={args.horizon}: {snap600['N_finished']}/{snap600['N']} finished | "
            f"{snap600['N_unfinished']} unfinished"
        )
        if drain_success:
            print(
                f"DRAIN: all finished by t={drain_finish_processed_time} | "
                f"all-passenger ATT={all300_att:.3f}"
            )
        else:
            print(
                f"DRAIN: NOT fully drained by t={args.safety_cap} | "
                f"remaining={stop_snapshot['N_unfinished']}"
            )
        print("=" * 110)

        return out_dir

    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            formal.core.restore_process_patches()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT),
        help="Completed FORMAL45 run root.",
    )
    p.add_argument(
        "--cell",
        default=None,
        help="Optional override, e.g. E0__M0. Otherwise auto-select worst checkpoint.",
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        help="Optional checkpoint step override.",
    )
    p.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)

    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--safety-cap", type=int, default=DEFAULT_SAFETY_CAP)
    p.add_argument("--no-drain", action="store_true")

    # Must match the formal run.
    p.add_argument("--fleet-size", type=int, default=40)
    p.add_argument("--future-horizon", type=float, default=30.0)
    p.add_argument("--max-events", type=int, default=8)
    p.add_argument("--pad-separation", type=float, default=0.25)
    p.add_argument("--charger-capacity", type=int, default=2)

    p.add_argument("--timeline-start", type=int, default=300)
    p.add_argument("--timeline-interval", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
