# -*- coding: utf-8 -*-
"""
E0/E2/E3/E4/E5/E6 Stage-Aware Effect-Time Diagnostic (NO TRAINING)
==================================================================

Purpose
-------
Run the SAME no-training mechanism diagnostic on every formal physics stage:

    E0, E2, E3, E4, E5, E6

using the stage-matched M0 (plain environment-aligned UAGMC) checkpoint at the
same training step (default 500k), same passenger trace, same topology, and one
fixed evaluation seed (default 123).

This script is a mechanism diagnostic, NOT a training script and NOT an online
oracle.  It measures:

1) same-passenger candidate access/effect-time heterogeneity;
2) decision-time -> candidate-specific own-effect-time state drift;
3) shared future horizon -> candidate-specific own-effect-time mismatch;
4) decision-time-known committed passenger-event boundary crossings;
5) decision-time-known committed aircraft/supply-release boundary crossings;
6) stage-aware resource drift:
      E0 : common queue/supply state
      E2 : + conserved-fleet supply semantics (common resource channels)
      E3 : + one-passenger-per-flight serviceability proxy
      E4 : + turnaround busy / release ETA
      E5 : + TLOF/pad next-free ETA
      E6 : + finite-charger active/wait/release ETA
7) offline realized-trajectory ranking flips (diagnostic only);
8) passenger ground-access timer audit using:
      state == enroute AND sub_state == to_vertiport
      remaining ETA = current_timer + 1

Scientific guardrail
--------------------
Future snapshots are taken from the REALIZED rollout and therefore can contain
effects of later policy actions and later demand.  They are OFFLINE diagnostic
evidence that temporal mismatch exists.  They are NEVER fed into the policy and
must not be described as an online future-state predictor.

Committed-event crossing metrics use only information already known at the
decision time.

Default compute
---------------
CPU-only inference, six episodes, no PPO learning.  Safe to run while the GPU
A/B/C training suite is running.

Expected project location
-------------------------
Place beside:
    train_uagmc_6x6_800k.py

Typical use
-----------
python diagnose_uagmc_E0_E6_stageaware.py

One-stage smoke:
python diagnose_uagmc_E0_E6_stageaware.py --stages E4

Explicit previous run:
python diagnose_uagmc_E0_E6_stageaware.py ^
  --run-root "serial_runs\\uagmc_6x6_T2_800k_seed1_YYYYMMDD_HHMMSS"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import train_uagmc_6x6_800k as exp


ROOT = Path(__file__).resolve().parent
DEFAULT_STAGES = ("E0", "E2", "E3", "E4", "E5", "E6")
DEFAULT_TOPOLOGY = "T2"
DEFAULT_STEP = 500_000
DEFAULT_EVAL_SEED = 123

# Common channels exist / are interpretable across every formal stage.
COMMON_KEYS = (
    "waiting",
    "access_incoming",
    "ready_idle",
    "inbound_empty",
    "charging",
    "total_local_aircraft",
    "total_local_capacity",
)

# Stage-specific cumulative channels.  These are intentionally cumulative:
# E5 contains E4 turnaround + E5 pad, E6 contains E5 + charging bottleneck.
STAGE_EXTRA_KEYS = {
    "E0": (),
    "E2": (),
    "E3": ("serviceable_now",),
    "E4": (
        "serviceable_now",
        "turnaround_busy",
        "min_turnaround_eta",
    ),
    "E5": (
        "serviceable_now",
        "turnaround_busy",
        "min_turnaround_eta",
        "pad_next_free_eta",
    ),
    "E6": (
        "serviceable_now",
        "turnaround_busy",
        "min_turnaround_eta",
        "pad_next_free_eta",
        "active_charging",
        "charger_wait",
        "min_charge_eta",
    ),
}

DETAIL_EXTRA_KEYS = (
    "min_access_eta",
    "avg_access_eta",
)


# =============================================================================
# Small helpers
# =============================================================================

def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(np.asarray(x).reshape(-1)[0])
        return y if math.isfinite(y) else default
    except Exception:
        return default


def finite(values: Iterable[Any]) -> List[float]:
    out = []
    for x in values:
        y = fnum(x)
        if math.isfinite(y):
            out.append(y)
    return out


def mean(values: Iterable[Any]) -> float:
    xs = finite(values)
    return float(np.mean(xs)) if xs else float("nan")


def median(values: Iterable[Any]) -> float:
    xs = finite(values)
    return float(np.median(xs)) if xs else float("nan")


def percentile(values: Iterable[Any], q: float) -> float:
    xs = finite(values)
    return float(np.percentile(xs, q)) if xs else float("nan")


def std(values: Iterable[Any]) -> float:
    xs = finite(values)
    return float(np.std(xs, ddof=0)) if xs else float("nan")


def parse_list(text: str, allowed: Sequence[str]) -> List[str]:
    xs = [x.strip().upper() for x in str(text).split(",") if x.strip()]
    bad = [x for x in xs if x not in allowed]
    if bad:
        raise ValueError(f"unsupported values={bad}; allowed={list(allowed)}")
    return xs


def jsonable(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    return x


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
    for r in rows:
        for k in r:
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
                    v = json.dumps(jsonable(v), ensure_ascii=False)
                cooked[k] = v
            w.writerow(cooked)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalized_drift(
    now: Dict[str, float],
    future: Dict[str, float],
    keys: Sequence[str],
) -> float:
    vals = []
    for k in keys:
        a = fnum(now.get(k, 0.0), 0.0)
        b = fnum(future.get(k, 0.0), 0.0)
        vals.append(abs(b - a) / (1.0 + abs(a)))
    return float(np.mean(vals)) if vals else 0.0


def any_change(
    a: Dict[str, float],
    b: Dict[str, float],
    keys: Sequence[str],
    tol: float = 1e-12,
) -> bool:
    return any(
        abs(fnum(b.get(k, 0.0), 0.0) - fnum(a.get(k, 0.0), 0.0)) > tol
        for k in keys
    )


def argmin_stable(values: Dict[int, float]) -> Optional[int]:
    if not values:
        return None
    return min(values.items(), key=lambda kv: (float(kv[1]), int(kv[0])))[0]


def build_zip(root: Path, filename: str) -> Path:
    zpath = root / filename
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p == zpath:
                continue
            if p.suffix.lower() in {".csv", ".json", ".txt"}:
                z.write(p, p.relative_to(root).as_posix())
    return zpath


# =============================================================================
# Locate completed 28.8M run / M0 checkpoint
# =============================================================================

def latest_6x6_run(topology: str) -> Path:
    pattern = f"uagmc_6x6_{str(topology).upper()}_800k_seed1_*"
    candidates = [
        p for p in (ROOT / "serial_runs").glob(pattern)
        if p.is_dir() and (p / "matrix_results.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find completed previous run matching serial_runs/{pattern}. "
            "Use --run-root explicitly."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def resolve_run_root(value: Optional[str], topology: str) -> Path:
    if not value:
        return latest_6x6_run(topology)
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    else:
        p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def checkpoint_paths(run_root: Path, stage: str, step: int) -> Tuple[Path, Path]:
    cdir = run_root / f"{stage}__M0" / "checkpoints"
    model = cdir / f"uam_ppo_{int(step)}_steps.zip"
    vec = cdir / f"uam_ppo_vecnormalize_{int(step)}_steps.pkl"
    if not model.exists():
        raise FileNotFoundError(model)
    if not vec.exists():
        raise FileNotFoundError(vec)
    return model, vec


def old_m0_performance(run_root: Path, stage: str, step: int) -> Dict[str, Any]:
    curve = read_csv(
        run_root / f"{stage}__M0" / "analysis" / "checkpoint_curve.csv"
    )
    row = next(
        (
            r for r in curve
            if int(round(fnum(r.get("train_step"), -1))) == int(step)
        ),
        None,
    )
    if row is None:
        return {}
    return {
        "M0_Jsys_per_passenger": fnum(
            row.get("system_person_minutes_per_passenger_mean")
        ),
        "M0_ATT": fnum(row.get("ATT_mean")),
        "M0_AWT": fnum(row.get("AWT_mean")),
        "M0_completion": fnum(row.get("completion_rate_mean")),
        "M0_action_0_share": fnum(row.get("action_0_share_mean")),
        "M0_action_1_share": fnum(row.get("action_1_share_mean")),
    }


# =============================================================================
# Correct committed passenger semantics / timer audit
# =============================================================================

def person_dict(scenario: Any) -> Dict[str, Any]:
    obj = getattr(scenario, "persons", None)
    raw = getattr(obj, "persons", {}) if obj is not None else {}
    return {str(k): v for k, v in raw.items()}


def is_access_committed(person: Any) -> bool:
    return (
        str(getattr(person, "state", "")).lower() == "enroute"
        and str(getattr(person, "sub_state", "")).lower() == "to_vertiport"
    )


def remaining_access_eta(person: Any) -> Optional[float]:
    if not is_access_committed(person):
        return None
    try:
        timer = float(getattr(person, "current_timer"))
    except Exception:
        return None
    if not math.isfinite(timer):
        return None
    # Correct discrete-time interpretation established by the old unified audit.
    return max(0.0, timer + 1.0)


def committed_passenger_events(
    scenario: Any,
    candidates: Sequence[int],
    exclude_pid: Optional[str],
) -> Dict[int, List[Tuple[str, float]]]:
    out = {int(v): [] for v in candidates}
    cand = set(out)
    excluded = str(exclude_pid) if exclude_pid is not None else None

    for pid, p in person_dict(scenario).items():
        if excluded is not None and str(pid) == excluded:
            continue
        if not is_access_committed(p):
            continue
        try:
            vid = int(getattr(p, "origin_vertiport_id"))
        except Exception:
            continue
        if vid not in cand:
            continue
        eta = remaining_access_eta(p)
        if eta is not None:
            out[vid].append((str(pid), float(eta)))

    for vid in out:
        out[vid].sort(key=lambda x: (x[1], x[0]))
    return out


class AccessTimerAudit:
    def __init__(self):
        self.prev: Dict[str, Tuple[int, float, int]] = {}
        self.rows: List[Dict[str, Any]] = []

    def observe(self, scenario: Any, sim_time: int) -> None:
        current: Dict[str, Tuple[int, float, int]] = {}
        for pid, p in person_dict(scenario).items():
            if not is_access_committed(p):
                continue
            try:
                timer = float(getattr(p, "current_timer"))
                vid = int(getattr(p, "origin_vertiport_id"))
            except Exception:
                continue
            if not math.isfinite(timer):
                continue
            current[pid] = (int(sim_time), timer, vid)

            if pid in self.prev:
                t0, timer0, vid0 = self.prev[pid]
                dt = int(sim_time) - int(t0)
                if dt > 0 and vid0 == vid:
                    slope = (timer - timer0) / float(dt)
                    self.rows.append({
                        "pid": pid,
                        "candidate_vertiport": vid,
                        "time0": t0,
                        "time1": sim_time,
                        "dt": dt,
                        "timer0": timer0,
                        "timer1": timer,
                        "timer_delta": timer - timer0,
                        "timer_slope_per_step": slope,
                        "is_near_minus_1": int(abs(slope + 1.0) <= 1e-9),
                    })
        self.prev = current

    def summary(self) -> Dict[str, Any]:
        slopes = finite(r["timer_slope_per_step"] for r in self.rows)
        good = sum(int(r["is_near_minus_1"]) for r in self.rows)
        return {
            "n_transitions": len(self.rows),
            "mean_slope_per_step": mean(slopes),
            "median_slope_per_step": median(slopes),
            "fraction_exact_minus_1": (
                float(good) / len(self.rows)
                if self.rows else float("nan")
            ),
        }


# =============================================================================
# Stage-aware state extraction
# =============================================================================

def local_aircraft_totals(scenario: Any, vid: int) -> Tuple[float, float]:
    n = 0.0
    cap = 0.0
    for e in list(getattr(scenario, "_all_evtols", {}).values()):
        if exp.mx.current_vid(e) != int(vid):
            continue
        if exp.mx.state_name(e) == "FLYING":
            continue
        n += 1.0
        cap += fnum(getattr(getattr(e, "spec", None), "capacity", 0.0), 0.0)
    return n, cap


def extract_stage_state(
    scenario: Any,
    candidates: Sequence[int],
    *,
    stage: str,
    charger_capacity: int,
) -> Dict[int, Dict[str, float]]:
    stage = str(stage).upper()
    persons = person_dict(scenario)
    out: Dict[int, Dict[str, float]] = {}

    for vid in candidates:
        vid = int(vid)
        vp = scenario.vertiports.vertiport_list[str(vid)]
        waiting = float(len(list(getattr(vp, "person_list", []) or [])))

        access_etas = []
        for p in persons.values():
            if not is_access_committed(p):
                continue
            try:
                pvid = int(getattr(p, "origin_vertiport_id"))
            except Exception:
                continue
            if pvid != vid:
                continue
            eta = remaining_access_eta(p)
            if eta is not None:
                access_etas.append(float(eta))

        r = exp.mx._resource_features_for_vid(
            scenario,
            vid,
            e6=(stage == "E6"),
            charger_capacity=int(charger_capacity),
        )
        ready_idle = float(r[0])
        inbound_empty = float(r[1])
        turnaround_busy = float(r[2])
        min_turnaround_eta = float(r[3])
        pad_next_free_eta = float(r[4])
        charging = float(r[5])
        active_charging = float(r[6])
        min_charge_eta = float(r[7])
        charger_wait = max(0.0, charging - active_charging)

        total_local, total_capacity = local_aircraft_totals(scenario, vid)

        # Current immediately serviceable aircraft.  E3+ is one passenger per
        # service flight; ready_idle is therefore a direct current serviceability
        # count.  For E0/E2 this is still recorded but not added as an extra key.
        serviceable_now = ready_idle

        out[vid] = {
            "waiting": waiting,
            "access_incoming": float(len(access_etas)),
            "ready_idle": ready_idle,
            "inbound_empty": inbound_empty,
            "charging": charging,
            "total_local_aircraft": total_local,
            "total_local_capacity": total_capacity,
            "serviceable_now": serviceable_now,
            "turnaround_busy": turnaround_busy,
            "min_turnaround_eta": min_turnaround_eta,
            "pad_next_free_eta": pad_next_free_eta,
            "active_charging": active_charging,
            "charger_wait": charger_wait,
            "min_charge_eta": min_charge_eta,
            "min_access_eta": (
                float(min(access_etas)) if access_etas else 0.0
            ),
            "avg_access_eta": (
                float(np.mean(access_etas)) if access_etas else 0.0
            ),
        }

    return out


def stage_keys(stage: str) -> Tuple[str, ...]:
    return tuple(COMMON_KEYS) + tuple(STAGE_EXTRA_KEYS[str(stage).upper()])


def common_pressure(s: Dict[str, float]) -> float:
    burden = fnum(s.get("waiting"), 0.0) + fnum(s.get("access_incoming"), 0.0)
    supply = fnum(s.get("ready_idle"), 0.0)
    return burden - supply


def resource_pressure(s: Dict[str, float], stage: str) -> float:
    """
    Descriptive ranking proxy only.  Never presented as an optimal-cost oracle.
    """
    stage = str(stage).upper()
    x = common_pressure(s)

    if stage in ("E4", "E5", "E6"):
        x += fnum(s.get("turnaround_busy"), 0.0)
        x += 0.10 * fnum(s.get("min_turnaround_eta"), 0.0)

    if stage in ("E5", "E6"):
        x += 0.10 * fnum(s.get("pad_next_free_eta"), 0.0)

    if stage == "E6":
        x += fnum(s.get("charger_wait"), 0.0)
        x += 0.10 * fnum(s.get("min_charge_eta"), 0.0)

    return float(x)


# =============================================================================
# Committed supply events known at decision time
# =============================================================================

def committed_supply_release_etas(
    scenario: Any,
    vid: int,
    *,
    stage: str,
    charger_capacity: int,
) -> List[float]:
    """
    Reuse the same transparent online-legal event extractor as the analytical
    rule baseline.  It includes current ready supply and committed release ETAs
    from turnaround / charging / already-committed empty inbound aircraft.
    """
    return [
        float(x)
        for x in exp.rulemod.known_supply_release_etas(
            scenario,
            int(vid),
            str(stage).upper(),
            int(charger_capacity),
        )
    ]


# =============================================================================
# One-stage diagnostic rollout
# =============================================================================

def run_stage(
    *,
    run_root: Path,
    output_root: Path,
    stage: str,
    topology: str,
    step: int,
    eval_seed: int,
    device: str,
    fleet_size: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
) -> Dict[str, Any]:
    stage = str(stage).upper()
    model_path, vec_path = checkpoint_paths(run_root, stage, step)
    outdir = output_root / stage
    outdir.mkdir(parents=True, exist_ok=True)

    exp.seed_all(eval_seed)

    raw = exp.build_eval_raw_env(
        stage=stage,
        method="M0",
        topology=topology,
        fleet_size=int(fleet_size),
        run_dir=outdir / "_monitor",
        future_horizon=float(exp.FUTURE_HORIZON_MIN),
        max_events=int(exp.MAX_EVENTS_PER_TYPE),
        pad_separation=float(pad_separation),
        charger_capacity=int(charger_capacity),
        max_time=int(max_time),
    )
    env = VecNormalize.load(str(vec_path), raw)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env, device=device)
    model.policy.set_training_mode(False)

    try:
        try:
            env.seed(int(eval_seed))
        except Exception:
            pass

        obs = env.reset()
        done = np.asarray([False])

        candidates = exp.candidates_for(topology)
        snapshots: Dict[int, Dict[int, Dict[str, float]]] = {}
        decisions: List[Dict[str, Any]] = []
        timer_audit = AccessTimerAudit()
        action_counts: Counter = Counter()

        steps = 0
        while not bool(done[0]):
            scenario = exp.mx.find_scenario(env)
            sim_time = int(getattr(scenario, "time", steps))

            if sim_time not in snapshots:
                snapshots[sim_time] = extract_stage_state(
                    scenario,
                    candidates,
                    stage=stage,
                    charger_capacity=charger_capacity,
                )

            timer_audit.observe(scenario, sim_time)

            uam_state = exp._uam_state(env)
            waiting = list(uam_state.get("waiting_decisions", []) or [])
            focal_pid = str(waiting[0]) if waiting else None
            focal_person = (
                exp.core._lookup_person(scenario, waiting[0])
                if waiting else None
            )

            probs = exp.mx.policy_probs(model, obs)
            action, _ = model.predict(obs, deterministic=True)
            ai = int(np.asarray(action).reshape(-1)[0])
            action_counts[ai] += 1

            if focal_pid is not None and focal_person is not None:
                access = {
                    int(vid): float(
                        exp._access_time(scenario, focal_person, int(vid))
                    )
                    for vid in candidates
                }
                passenger_events = committed_passenger_events(
                    scenario,
                    candidates,
                    exclude_pid=focal_pid,
                )
                supply_events = {
                    int(vid): committed_supply_release_etas(
                        scenario,
                        int(vid),
                        stage=stage,
                        charger_capacity=charger_capacity,
                    )
                    for vid in candidates
                }

                decisions.append({
                    "decision_id": len(decisions),
                    "sim_time": sim_time,
                    "pid": focal_pid,
                    "chosen_action_index": ai,
                    "chosen_vertiport": int(candidates[ai]),
                    "policy_probs": [float(x) for x in np.asarray(probs).reshape(-1)],
                    "access_times": access,
                    "now_state": extract_stage_state(
                        scenario,
                        candidates,
                        stage=stage,
                        charger_capacity=charger_capacity,
                    ),
                    "committed_passenger_events": passenger_events,
                    "committed_supply_release_etas": supply_events,
                })

            obs, _, done, _ = env.step(action)
            steps += 1
            if steps > int(max_time) + 100:
                raise RuntimeError(
                    f"{stage}: rollout exceeded max-time guard"
                )

        available_times = sorted(snapshots)

        def snapshot_time(target: float) -> Optional[int]:
            t = int(math.ceil(float(target)))
            for x in available_times:
                if x >= t:
                    return x
            return None

        candidate_rows: List[Dict[str, Any]] = []
        decision_rows: List[Dict[str, Any]] = []

        common_keys = tuple(COMMON_KEYS)
        skeys = stage_keys(stage)

        for d in decisions:
            access_vals = list(d["access_times"].values())
            if not access_vals:
                continue
            h_min = float(min(access_vals))
            h_max = float(max(access_vals))
            h_mean = float(np.mean(access_vals))
            h_spread = h_max - h_min

            shared_t = snapshot_time(float(d["sim_time"]) + h_mean)

            now_wait: Dict[int, float] = {}
            own_wait: Dict[int, float] = {}
            shared_wait: Dict[int, float] = {}

            now_pressure: Dict[int, float] = {}
            own_pressure: Dict[int, float] = {}
            shared_pressure: Dict[int, float] = {}

            now_resource: Dict[int, float] = {}
            own_resource: Dict[int, float] = {}
            shared_resource: Dict[int, float] = {}

            per_common_now_own = []
            per_stage_now_own = []
            per_common_shared_own = []
            per_stage_shared_own = []

            passenger_boundary_diff = False
            supply_boundary_diff = False
            passenger_cross_positive = False
            supply_cross_positive = False
            complete = True

            for action_index, vid0 in enumerate(candidates):
                vid = int(vid0)
                h = float(d["access_times"][vid])
                own_t = snapshot_time(float(d["sim_time"]) + h)
                now_state = d["now_state"][vid]

                row: Dict[str, Any] = {
                    "stage": stage,
                    "decision_id": int(d["decision_id"]),
                    "sim_time": int(d["sim_time"]),
                    "pid": d["pid"],
                    "candidate_action_index": action_index,
                    "candidate_vertiport": vid,
                    "checkpoint_choice": int(
                        vid == int(d["chosen_vertiport"])
                    ),
                    "policy_prob": (
                        d["policy_probs"][action_index]
                        if action_index < len(d["policy_probs"])
                        else float("nan")
                    ),
                    "access_time": h,
                    "decision_access_min": h_min,
                    "decision_access_max": h_max,
                    "decision_access_mean": h_mean,
                    "decision_access_spread": h_spread,
                    "own_effect_time": float(d["sim_time"]) + h,
                    "own_snapshot_time": own_t if own_t is not None else "",
                    "shared_effect_time": float(d["sim_time"]) + h_mean,
                    "shared_snapshot_time": shared_t if shared_t is not None else "",
                }

                for key in tuple(skeys) + DETAIL_EXTRA_KEYS:
                    row[f"now_{key}"] = fnum(now_state.get(key, 0.0), 0.0)

                p_events = list(
                    d["committed_passenger_events"].get(vid, [])
                )
                s_events = list(
                    d["committed_supply_release_etas"].get(vid, [])
                )

                p_own = sum(1 for _, eta in p_events if float(eta) <= h + 1e-12)
                p_shared = sum(
                    1 for _, eta in p_events if float(eta) <= h_mean + 1e-12
                )
                s_own = sum(1 for eta in s_events if float(eta) <= h + 1e-12)
                s_shared = sum(
                    1 for eta in s_events if float(eta) <= h_mean + 1e-12
                )

                row.update({
                    "committed_passenger_events_now": len(p_events),
                    "passenger_cross_before_own": p_own,
                    "passenger_cross_before_shared": p_shared,
                    "passenger_own_minus_shared_cross": p_own - p_shared,
                    "committed_supply_events_now": len(s_events),
                    "supply_cross_before_own": s_own,
                    "supply_cross_before_shared": s_shared,
                    "supply_own_minus_shared_cross": s_own - s_shared,
                })

                passenger_cross_positive |= p_own > 0
                supply_cross_positive |= s_own > 0
                passenger_boundary_diff |= p_own != p_shared
                supply_boundary_diff |= s_own != s_shared

                if own_t is None or shared_t is None:
                    row["future_available"] = 0
                    complete = False
                    candidate_rows.append(row)
                    continue

                own_state = snapshots[own_t][vid]
                shared_state = snapshots[shared_t][vid]
                row["future_available"] = 1

                for key in tuple(skeys) + DETAIL_EXTRA_KEYS:
                    row[f"own_{key}"] = fnum(own_state.get(key, 0.0), 0.0)
                    row[f"shared_{key}"] = fnum(shared_state.get(key, 0.0), 0.0)
                    row[f"delta_now_to_own_{key}"] = (
                        fnum(own_state.get(key, 0.0), 0.0)
                        - fnum(now_state.get(key, 0.0), 0.0)
                    )
                    row[f"delta_shared_to_own_{key}"] = (
                        fnum(own_state.get(key, 0.0), 0.0)
                        - fnum(shared_state.get(key, 0.0), 0.0)
                    )

                c_no = normalized_drift(now_state, own_state, common_keys)
                st_no = normalized_drift(now_state, own_state, skeys)
                c_so = normalized_drift(shared_state, own_state, common_keys)
                st_so = normalized_drift(shared_state, own_state, skeys)

                row.update({
                    "common_drift_now_to_own": c_no,
                    "stage_drift_now_to_own": st_no,
                    "common_drift_shared_to_own": c_so,
                    "stage_drift_shared_to_own": st_so,
                    "common_any_change_now_to_own": int(
                        any_change(now_state, own_state, common_keys)
                    ),
                    "stage_any_change_now_to_own": int(
                        any_change(now_state, own_state, skeys)
                    ),
                    "common_any_change_shared_to_own": int(
                        any_change(shared_state, own_state, common_keys)
                    ),
                    "stage_any_change_shared_to_own": int(
                        any_change(shared_state, own_state, skeys)
                    ),
                })

                per_common_now_own.append(c_no)
                per_stage_now_own.append(st_no)
                per_common_shared_own.append(c_so)
                per_stage_shared_own.append(st_so)

                now_wait[vid] = fnum(now_state["waiting"], 0.0)
                own_wait[vid] = fnum(own_state["waiting"], 0.0)
                shared_wait[vid] = fnum(shared_state["waiting"], 0.0)

                now_pressure[vid] = common_pressure(now_state)
                own_pressure[vid] = common_pressure(own_state)
                shared_pressure[vid] = common_pressure(shared_state)

                now_resource[vid] = resource_pressure(now_state, stage)
                own_resource[vid] = resource_pressure(own_state, stage)
                shared_resource[vid] = resource_pressure(shared_state, stage)

                candidate_rows.append(row)

            drow: Dict[str, Any] = {
                "stage": stage,
                "decision_id": int(d["decision_id"]),
                "sim_time": int(d["sim_time"]),
                "pid": d["pid"],
                "chosen_vertiport": int(d["chosen_vertiport"]),
                "access_delay_min": h_min,
                "access_delay_max": h_max,
                "access_delay_mean": h_mean,
                "access_delay_spread": h_spread,
                "candidate_count": len(candidates),
                "complete_oracle": int(complete),
                "mean_common_drift_now_to_own": mean(per_common_now_own),
                "mean_stage_drift_now_to_own": mean(per_stage_now_own),
                "mean_common_drift_shared_to_own": mean(per_common_shared_own),
                "mean_stage_drift_shared_to_own": mean(per_stage_shared_own),
                "passenger_positive_crossing": int(passenger_cross_positive),
                "passenger_shared_boundary_diff": int(passenger_boundary_diff),
                "supply_positive_crossing": int(supply_cross_positive),
                "supply_shared_boundary_diff": int(supply_boundary_diff),
            }

            if complete and len(own_wait) == len(candidates):
                drow.update({
                    "now_best_waiting": argmin_stable(now_wait),
                    "own_best_waiting": argmin_stable(own_wait),
                    "shared_best_waiting": argmin_stable(shared_wait),
                    "flip_now_to_own_waiting": int(
                        argmin_stable(now_wait) != argmin_stable(own_wait)
                    ),
                    "flip_shared_to_own_waiting": int(
                        argmin_stable(shared_wait) != argmin_stable(own_wait)
                    ),
                    "now_best_common_pressure": argmin_stable(now_pressure),
                    "own_best_common_pressure": argmin_stable(own_pressure),
                    "shared_best_common_pressure": argmin_stable(shared_pressure),
                    "flip_now_to_own_common_pressure": int(
                        argmin_stable(now_pressure) != argmin_stable(own_pressure)
                    ),
                    "flip_shared_to_own_common_pressure": int(
                        argmin_stable(shared_pressure) != argmin_stable(own_pressure)
                    ),
                    "now_best_resource_pressure": argmin_stable(now_resource),
                    "own_best_resource_pressure": argmin_stable(own_resource),
                    "shared_best_resource_pressure": argmin_stable(shared_resource),
                    "flip_now_to_own_resource_pressure": int(
                        argmin_stable(now_resource) != argmin_stable(own_resource)
                    ),
                    "flip_shared_to_own_resource_pressure": int(
                        argmin_stable(shared_resource) != argmin_stable(own_resource)
                    ),
                })

            decision_rows.append(drow)

        complete_candidates = [
            r for r in candidate_rows
            if int(r.get("future_available", 0)) == 1
        ]
        complete_decisions = [
            r for r in decision_rows
            if int(r.get("complete_oracle", 0)) == 1
        ]

        # Variable-level drift table: especially important for E4/E5/E6.
        variable_rows = []
        for key in tuple(skeys) + DETAIL_EXTRA_KEYS:
            variable_rows.append({
                "stage": stage,
                "variable": key,
                "is_common_key": int(key in common_keys),
                "is_stage_key": int(key in skeys),
                "mean_abs_delta_now_to_own": mean(
                    abs(fnum(r.get(f"delta_now_to_own_{key}"), 0.0))
                    for r in complete_candidates
                ),
                "mean_abs_delta_shared_to_own": mean(
                    abs(fnum(r.get(f"delta_shared_to_own_{key}"), 0.0))
                    for r in complete_candidates
                ),
                "change_rate_now_to_own": mean(
                    int(abs(fnum(r.get(f"delta_now_to_own_{key}"), 0.0)) > 1e-12)
                    for r in complete_candidates
                ),
                "change_rate_shared_to_own": mean(
                    int(abs(fnum(r.get(f"delta_shared_to_own_{key}"), 0.0)) > 1e-12)
                    for r in complete_candidates
                ),
            })

        timer_summary = timer_audit.summary()
        perf = old_m0_performance(run_root, stage, step)

        summary = {
            "stage": stage,
            "topology": topology,
            "method": "M0",
            "train_step": int(step),
            "eval_seed": int(eval_seed),
            "model": str(model_path),
            "vecnormalize": str(vec_path),
            "recorded_decisions": len(decisions),
            "complete_decisions": len(complete_decisions),
            "complete_candidate_rows": len(complete_candidates),
            "M0_performance_from_28_8M": perf,
            "same_passenger_candidate_delay": {
                "mean_min_access": mean(
                    r["access_delay_min"] for r in decision_rows
                ),
                "mean_max_access": mean(
                    r["access_delay_max"] for r in decision_rows
                ),
                "mean_spread": mean(
                    r["access_delay_spread"] for r in decision_rows
                ),
                "median_spread": median(
                    r["access_delay_spread"] for r in decision_rows
                ),
                "p90_spread": percentile(
                    (r["access_delay_spread"] for r in decision_rows),
                    90,
                ),
                "fraction_spread_ge_2": mean(
                    int(fnum(r["access_delay_spread"]) >= 2.0)
                    for r in decision_rows
                ),
                "fraction_spread_ge_5": mean(
                    int(fnum(r["access_delay_spread"]) >= 5.0)
                    for r in decision_rows
                ),
            },
            "decision_time_to_own_effect_time": {
                "mean_common_drift": mean(
                    r["common_drift_now_to_own"]
                    for r in complete_candidates
                ),
                "mean_stage_drift": mean(
                    r["stage_drift_now_to_own"]
                    for r in complete_candidates
                ),
                "common_any_change_rate": mean(
                    r["common_any_change_now_to_own"]
                    for r in complete_candidates
                ),
                "stage_any_change_rate": mean(
                    r["stage_any_change_now_to_own"]
                    for r in complete_candidates
                ),
            },
            "shared_horizon_to_candidate_effect_time": {
                "shared_reference": "mean candidate access time for same passenger",
                "mean_common_drift": mean(
                    r["common_drift_shared_to_own"]
                    for r in complete_candidates
                ),
                "mean_stage_drift": mean(
                    r["stage_drift_shared_to_own"]
                    for r in complete_candidates
                ),
                "common_any_change_rate": mean(
                    r["common_any_change_shared_to_own"]
                    for r in complete_candidates
                ),
                "stage_any_change_rate": mean(
                    r["stage_any_change_shared_to_own"]
                    for r in complete_candidates
                ),
            },
            "known_committed_boundary_crossing": {
                "passenger_positive_crossing_rate": mean(
                    r["passenger_positive_crossing"]
                    for r in decision_rows
                ),
                "passenger_shared_boundary_diff_rate": mean(
                    r["passenger_shared_boundary_diff"]
                    for r in decision_rows
                ),
                "supply_positive_crossing_rate": mean(
                    r["supply_positive_crossing"]
                    for r in decision_rows
                ),
                "supply_shared_boundary_diff_rate": mean(
                    r["supply_shared_boundary_diff"]
                    for r in decision_rows
                ),
                "guardrail": (
                    "Only decision-time-known passenger access commitments and "
                    "committed aircraft/supply releases are counted."
                ),
            },
            "offline_realized_rank_flip": {
                "now_to_own_waiting": mean(
                    r.get("flip_now_to_own_waiting")
                    for r in complete_decisions
                ),
                "shared_to_own_waiting": mean(
                    r.get("flip_shared_to_own_waiting")
                    for r in complete_decisions
                ),
                "now_to_own_common_pressure": mean(
                    r.get("flip_now_to_own_common_pressure")
                    for r in complete_decisions
                ),
                "shared_to_own_common_pressure": mean(
                    r.get("flip_shared_to_own_common_pressure")
                    for r in complete_decisions
                ),
                "now_to_own_resource_pressure": mean(
                    r.get("flip_now_to_own_resource_pressure")
                    for r in complete_decisions
                ),
                "shared_to_own_resource_pressure": mean(
                    r.get("flip_shared_to_own_resource_pressure")
                    for r in complete_decisions
                ),
                "guardrail": (
                    "Realized future state is offline oracle evidence only; "
                    "rank flips do not identify counterfactual ATT-optimal actions."
                ),
            },
            "timer_semantics_audit": timer_summary,
            "stage_keys": list(skeys),
            "common_keys": list(common_keys),
            "scientific_guardrails": [
                "NO TRAINING occurs in this script.",
                "Future realized snapshots are OFFLINE diagnostic only.",
                "No realized future snapshot is used by the policy.",
                "Committed boundary metrics use only decision-time-known events.",
                "This diagnostic establishes mismatch/mechanism evidence, not ATT gain.",
            ],
        }

        write_csv(outdir / "candidate_detail.csv", candidate_rows)
        write_csv(outdir / "decision_summary.csv", decision_rows)
        write_csv(outdir / "variable_drift_summary.csv", variable_rows)
        write_csv(outdir / "timer_semantics_audit.csv", timer_audit.rows)
        write_json(outdir / "summary.json", summary)

        # Compact human-readable stage report.
        lines = [
            f"{stage} M0@{step:,} STAGE-AWARE EFFECT-TIME DIAGNOSTIC",
            "=" * 110,
            f"decisions                    : {len(decisions)}",
            f"complete decisions           : {len(complete_decisions)}",
            f"access spread mean           : {summary['same_passenger_candidate_delay']['mean_spread']:.4f}",
            f"now->own common drift        : {summary['decision_time_to_own_effect_time']['mean_common_drift']:.4f}",
            f"now->own stage drift         : {summary['decision_time_to_own_effect_time']['mean_stage_drift']:.4f}",
            f"shared->own common drift     : {summary['shared_horizon_to_candidate_effect_time']['mean_common_drift']:.4f}",
            f"shared->own stage drift      : {summary['shared_horizon_to_candidate_effect_time']['mean_stage_drift']:.4f}",
            f"passenger boundary diff rate : {summary['known_committed_boundary_crossing']['passenger_shared_boundary_diff_rate']:.4f}",
            f"supply boundary diff rate    : {summary['known_committed_boundary_crossing']['supply_shared_boundary_diff_rate']:.4f}",
            f"shared->own resource rankflip: {summary['offline_realized_rank_flip']['shared_to_own_resource_pressure']:.4f}",
            f"timer exact -1 fraction      : {timer_summary['fraction_exact_minus_1']:.4f}",
            "",
            "OFFLINE ORACLE GUARDRAIL: realized future snapshots are diagnostic only.",
        ]
        (outdir / "summary.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        # One flat row for cross-stage comparison.
        flat = {
            "stage": stage,
            "train_step": step,
            "eval_seed": eval_seed,
            "recorded_decisions": len(decisions),
            "complete_decisions": len(complete_decisions),
            "M0_Jsys_per_passenger": perf.get("M0_Jsys_per_passenger"),
            "M0_ATT": perf.get("M0_ATT"),
            "M0_completion": perf.get("M0_completion"),
            "access_spread_mean": summary["same_passenger_candidate_delay"]["mean_spread"],
            "access_spread_p90": summary["same_passenger_candidate_delay"]["p90_spread"],
            "now_to_own_common_drift": summary["decision_time_to_own_effect_time"]["mean_common_drift"],
            "now_to_own_stage_drift": summary["decision_time_to_own_effect_time"]["mean_stage_drift"],
            "now_to_own_common_change_rate": summary["decision_time_to_own_effect_time"]["common_any_change_rate"],
            "now_to_own_stage_change_rate": summary["decision_time_to_own_effect_time"]["stage_any_change_rate"],
            "shared_to_own_common_drift": summary["shared_horizon_to_candidate_effect_time"]["mean_common_drift"],
            "shared_to_own_stage_drift": summary["shared_horizon_to_candidate_effect_time"]["mean_stage_drift"],
            "shared_to_own_common_change_rate": summary["shared_horizon_to_candidate_effect_time"]["common_any_change_rate"],
            "shared_to_own_stage_change_rate": summary["shared_horizon_to_candidate_effect_time"]["stage_any_change_rate"],
            "passenger_positive_crossing_rate": summary["known_committed_boundary_crossing"]["passenger_positive_crossing_rate"],
            "passenger_shared_boundary_diff_rate": summary["known_committed_boundary_crossing"]["passenger_shared_boundary_diff_rate"],
            "supply_positive_crossing_rate": summary["known_committed_boundary_crossing"]["supply_positive_crossing_rate"],
            "supply_shared_boundary_diff_rate": summary["known_committed_boundary_crossing"]["supply_shared_boundary_diff_rate"],
            "now_to_own_waiting_rankflip": summary["offline_realized_rank_flip"]["now_to_own_waiting"],
            "shared_to_own_waiting_rankflip": summary["offline_realized_rank_flip"]["shared_to_own_waiting"],
            "now_to_own_common_pressure_rankflip": summary["offline_realized_rank_flip"]["now_to_own_common_pressure"],
            "shared_to_own_common_pressure_rankflip": summary["offline_realized_rank_flip"]["shared_to_own_common_pressure"],
            "now_to_own_resource_pressure_rankflip": summary["offline_realized_rank_flip"]["now_to_own_resource_pressure"],
            "shared_to_own_resource_pressure_rankflip": summary["offline_realized_rank_flip"]["shared_to_own_resource_pressure"],
            "timer_exact_minus1_fraction": timer_summary["fraction_exact_minus_1"],
        }
        return {
            "flat": flat,
            "summary": summary,
            "variable_rows": variable_rows,
        }

    finally:
        try:
            env.close()
        except Exception:
            pass
        exp.core.restore_process_patches()


# =============================================================================
# Cross-stage runner
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "NO-TRAINING stage-aware effect-time diagnostic for "
            "E0/E2/E3/E4/E5/E6 using stage-matched M0 checkpoints."
        )
    )
    p.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    p.add_argument("--topology", choices=["T2", "T3"], default=DEFAULT_TOPOLOGY)
    p.add_argument("--step", type=int, default=DEFAULT_STEP)
    p.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--run-root", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--fleet-size", type=int, default=exp.FLEET_SIZE)
    p.add_argument("--pad-separation", type=float, default=exp.PAD_SEPARATION_MIN)
    p.add_argument("--charger-capacity", type=int, default=exp.CHARGER_CAPACITY)
    p.add_argument("--max-time", type=int, default=exp.MAX_TIME)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stages = parse_list(args.stages, DEFAULT_STAGES)
    run_root = resolve_run_root(args.run_root, args.topology)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Default CPU is deliberate: this diagnostic can run beside GPU training.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_root:
        output_root = Path(args.output_root).expanduser()
        if not output_root.is_absolute():
            output_root = (ROOT / output_root).resolve()
        else:
            output_root = output_root.resolve()
    else:
        output_root = (
            ROOT
            / "diagnostics"
            / (
                f"uagmc_E0_E6_stageaware_{args.topology}_"
                f"M0_{args.step//1000}k_seed{args.eval_seed}_{stamp}"
            )
        ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "UAGMC_E0_E6_STAGE_AWARE_EFFECT_TIME_DIAGNOSTIC",
        "no_training": True,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_28_8M_run": str(run_root),
        "stages": stages,
        "topology": args.topology,
        "method": "M0",
        "checkpoint_step": int(args.step),
        "eval_seed": int(args.eval_seed),
        "device": device,
        "fleet_size_E2_E6": int(args.fleet_size),
        "pad_separation_min": float(args.pad_separation),
        "charger_capacity_E6": int(args.charger_capacity),
        "max_time": int(args.max_time),
        "timer_semantics": "committed passenger ETA = current_timer + 1",
        "future_snapshot_semantics": "OFFLINE realized-trajectory diagnostic only",
        "common_keys": list(COMMON_KEYS),
        "stage_extra_keys": STAGE_EXTRA_KEYS,
    }
    write_json(output_root / "manifest.json", manifest)

    print("=" * 132)
    print("UAGMC E0/E2/E3/E4/E5/E6 STAGE-AWARE EFFECT-TIME DIAGNOSTIC | NO TRAINING")
    print(f"Source run : {run_root}")
    print(f"Stages     : {stages}")
    print(f"Model      : stage-matched M0 @ {args.step:,}")
    print(f"Topology   : {args.topology}")
    print(f"Eval seed  : {args.eval_seed}")
    print(f"Device     : {device} (CPU recommended while GPU training is running)")
    print(f"Output     : {output_root}")
    print("=" * 132)

    stage_rows: List[Dict[str, Any]] = []
    all_variable_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for i, stage in enumerate(stages, 1):
        print(f"\n[{i:02d}/{len(stages):02d}] {stage} / M0@{args.step//1000}k", flush=True)
        try:
            result = run_stage(
                run_root=run_root,
                output_root=output_root,
                stage=stage,
                topology=args.topology,
                step=int(args.step),
                eval_seed=int(args.eval_seed),
                device=device,
                fleet_size=int(args.fleet_size),
                pad_separation=float(args.pad_separation),
                charger_capacity=int(args.charger_capacity),
                max_time=int(args.max_time),
            )
            row = result["flat"]
            stage_rows.append(row)
            all_variable_rows.extend(result["variable_rows"])

            write_csv(output_root / "stage_summary.csv", stage_rows)
            write_csv(
                output_root / "all_variable_drift_summary.csv",
                all_variable_rows,
            )
            print(
                f"  spread={fnum(row['access_spread_mean']):.3f} | "
                f"now->own(stage)={fnum(row['now_to_own_stage_drift']):.3f} | "
                f"shared->own(stage)={fnum(row['shared_to_own_stage_drift']):.3f} | "
                f"P-boundary={fnum(row['passenger_shared_boundary_diff_rate']):.3f} | "
                f"S-boundary={fnum(row['supply_shared_boundary_diff_rate']):.3f} | "
                f"resource-rankflip={fnum(row['shared_to_own_resource_pressure_rankflip']):.3f}",
                flush=True,
            )
        except Exception as exc:
            errors.append({
                "stage": stage,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            write_csv(output_root / "errors.csv", errors)
            print(f"  [ERROR] {stage}: {repr(exc)}", flush=True)
            if not args.continue_on_error:
                raise
        finally:
            exp.core.restore_process_patches()

    write_csv(output_root / "stage_summary.csv", stage_rows)
    write_csv(
        output_root / "all_variable_drift_summary.csv",
        all_variable_rows,
    )
    write_csv(output_root / "errors.csv", errors)

    # Compact cross-stage text report.
    lines = [
        "UAGMC E0-E6 STAGE-AWARE EFFECT-TIME DIAGNOSTIC | NO TRAINING",
        "=" * 132,
        (
            "stage | M0 Jsys | access spread | now->own stage drift | "
            "shared->own stage drift | P-boundary | S-boundary | shared resource rankflip"
        ),
    ]
    for r in stage_rows:
        lines.append(
            f"{r['stage']:>2} | "
            f"{fnum(r.get('M0_Jsys_per_passenger')):8.3f} | "
            f"{fnum(r.get('access_spread_mean')):8.3f} | "
            f"{fnum(r.get('now_to_own_stage_drift')):8.4f} | "
            f"{fnum(r.get('shared_to_own_stage_drift')):8.4f} | "
            f"{fnum(r.get('passenger_shared_boundary_diff_rate')):7.3f} | "
            f"{fnum(r.get('supply_shared_boundary_diff_rate')):7.3f} | "
            f"{fnum(r.get('shared_to_own_resource_pressure_rankflip')):7.3f}"
        )
    lines.extend([
        "",
        "Guardrail: realized future snapshots are OFFLINE diagnostics only.",
        "No training occurs in this script.",
    ])
    (output_root / "summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    bundle = build_zip(
        output_root,
        "UPLOAD_THIS_uagmc_E0_E6_stageaware_diagnostic.zip",
    )

    print("\n" + "=" * 132)
    print("STAGE-AWARE DIAGNOSTIC FINISHED")
    print(f"Stage table : {output_root / 'stage_summary.csv'}")
    print(f"Variables   : {output_root / 'all_variable_drift_summary.csv'}")
    print(f"Errors      : {output_root / 'errors.csv'}")
    print("UPLOAD THIS FILE TO CHATGPT:")
    print(bundle)
    print("=" * 132)

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
