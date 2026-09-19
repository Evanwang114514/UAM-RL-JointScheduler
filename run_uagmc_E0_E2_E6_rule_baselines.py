# -*- coding: utf-8 -*-
"""
E0/E2/E3/E4/E5/E6 no-learning rule baselines (T3 by default)
================================================================

Purpose
-------
Give every physics level an absolute, no-learning reference under the SAME
passenger trace and the SAME environment implementation used by the formal
experiments.

Rules
-----
SPF
    Source-consistent shortest physical/access rule: choose the candidate with
    minimum focal-passenger ground access time.

STTF
    Source-consistent total-time heuristic: access time + a CURRENT snapshot
    queue-delay proxy.  It deliberately uses only decision-time information.

QTTI2
    Redesigned QTTI for the closed-loop environments.  The old public-source
    QTTI uses hard-coded queue/service-rate constants tied to the original two
    vertiports.  QTTI2 instead uses only ONLINE-LEGAL committed information:
      * passengers already travelling to each candidate and their remaining ETA;
      * aircraft already ready / charging / turnaround / empty-inbound and their
        causal release ETA;
      * focal passenger's candidate-specific access horizon.
    It estimates residual queue at the focal passenger's arrival and converts
    that residual workload into a queue-time index.  No unrevealed future
    passenger request is used.

Important
---------
This is a RULE BASELINE suite, not the proposed trainable Effect-Time method.
QTTI2 is intentionally simple and transparent so it can serve as a strong
analytical/action-centered baseline.

Compute
-------
No PPO training is performed, so this script is CPU-only.  This is deliberate:
GPU is reserved for the two PPO training files.  Evaluation seeds default to
123/124/125.

Place this file beside:
    train_uagmc_E0_E2_E6_effect_time_700k.py
in UAGMC-main.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import train_uagmc_E0_E2_E6_effect_time_700k as core


ROOT = Path(__file__).resolve().parent
DEFAULT_STAGES = ("E0", "E2", "E3", "E4", "E5", "E6")
DEFAULT_METHODS = ("SPF", "STTF", "QTTI2")
DEFAULT_TOPOLOGY = "T3"
DEFAULT_SEEDS = (123, 124, 125)
SERVICE_RATE_WINDOW_MIN = 10.0


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return default


def mean(xs: Iterable[Any]) -> float:
    arr = np.asarray([fnum(x) for x in xs], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def std(xs: Iterable[Any]) -> float:
    arr = np.asarray([fnum(x) for x in xs], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0 if len(arr) == 1 else float("nan")
    return float(arr.std(ddof=1))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = {}
            for k, v in row.items():
                if isinstance(v, (dict, list, tuple, np.ndarray)):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    out[k] = json.dumps(v, ensure_ascii=False)
                else:
                    out[k] = v
            w.writerow(out)


def parse_csv_list(text: str) -> List[str]:
    return [x.strip().upper() for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def find_uam(env: Any):
    return core.mx.find_uam_wrapper(env)


def find_scenario(env: Any):
    return core.mx.find_scenario(env)


def focal_person(env: Any) -> Optional[Any]:
    uam = find_uam(env)
    waiting = list((getattr(uam, "state", {}) or {}).get("waiting_decisions", []) or [])
    if not waiting:
        return None
    return core._lookup_person(find_scenario(env), waiting[0])


def queue_len(scenario: Any, vid: int) -> int:
    vp = scenario.vertiports.vertiport_list[str(int(vid))]
    return len(list(getattr(vp, "person_list", []) or []))


def aircraft_capacity(scenario: Any, stage: str) -> int:
    if str(stage).upper() in ("E3", "E4", "E5", "E6"):
        return 1
    for e in list(getattr(scenario, "_all_evtols", {}).values()):
        try:
            c = int(getattr(getattr(e, "spec", None), "capacity", 0) or 0)
            if c > 0:
                return c
        except Exception:
            pass
    return 4


def access_time(scenario: Any, person: Any, vid: int) -> float:
    return float(core._access_time(scenario, person, int(vid)))


def committed_passenger_arrivals(scenario: Any, vid: int, horizon: float) -> int:
    """Already-committed passengers whose ground-access ETA is <= horizon."""
    persons_obj = getattr(scenario, "persons", None)
    persons = getattr(persons_obj, "persons", {}) if persons_obj is not None else {}
    n = 0
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
        if eta <= float(horizon) + 1e-9:
            n += 1
    return n


def _pad_release_extra(scenario: Any, vid: int, at_time: float) -> float:
    """Known calendar delay relative to a projected event time."""
    cal = getattr(scenario, "_e345_pad_calendar", {}) or {}
    times = list(cal.get(str(int(vid)), []) or [])
    if not times:
        return 0.0
    sep = float(getattr(core.mx.base, "PAD_SEPARATION_MIN", 0.0) or 0.0)
    now = float(getattr(scenario, "time", 0.0))
    next_free_abs = max(float(x) for x in times) + sep
    return max(0.0, next_free_abs - (now + float(at_time)))


def _charging_schedule_for_vid(
    scenario: Any,
    vid: int,
    stage: str,
    charger_capacity: int,
) -> Dict[str, float]:
    """Return causal completion ETA for charging aircraft currently at vid."""
    charging = []
    for e in list(getattr(scenario, "_all_evtols", {}).values()):
        if core.mx.current_vid(e) != int(vid):
            continue
        if core.mx.state_name(e) != "CHARGING":
            continue
        charging.append(e)

    if not charging:
        return {}

    charging.sort(
        key=lambda e: (
            fnum(getattr(e, "_e6_charge_queue_enter_time", 0.0), 0.0),
            str(getattr(e, "id", "")),
        )
    )

    durations = [(e, max(0.0, core._charge_time(e))) for e in charging]
    if str(stage).upper() != "E6":
        return {str(getattr(e, "id", i)): d for i, (e, d) in enumerate(durations)}

    cap = max(1, int(charger_capacity))
    lanes = [0.0 for _ in range(cap)]
    result: Dict[str, float] = {}
    for i, (e, dur) in enumerate(durations):
        lane = min(range(cap), key=lambda j: lanes[j])
        finish = lanes[lane] + dur
        lanes[lane] = finish
        result[str(getattr(e, "id", i))] = finish
    return result


def known_supply_release_etas(
    scenario: Any,
    vid: int,
    stage: str,
    charger_capacity: int,
) -> List[float]:
    """Known/committed supply releases at a departure vertiport.

    This is deliberately conservative: it never assumes a future empty
    reposition that has not yet been committed by the simulator.
    """
    stage = str(stage).upper()
    now = float(getattr(scenario, "time", 0.0))
    charge_schedule = _charging_schedule_for_vid(scenario, vid, stage, charger_capacity)
    etas: List[float] = []

    for e in list(getattr(scenario, "_all_evtols", {}).values()):
        st = core.mx.state_name(e)
        cur = core.mx.current_vid(e)
        tar = core.mx.target_vid(e)
        empty = len(core.mx.passenger_ids(e)) == 0
        eid = str(getattr(e, "id", ""))

        # Ready, local empty supply.
        if cur == int(vid) and st == "IDLE" and empty and not core.mx.base.is_turnaround_busy(e):
            etas.append(0.0)
            continue

        # Turnaround is a known committed release.
        if cur == int(vid) and core.mx.base.is_turnaround_busy(e):
            rel_abs = getattr(e, "_e345_turnaround_release_time", None)
            rel = max(0.0, fnum(rel_abs, now) - now) if rel_abs is not None else 0.0
            # Fixed turnaround releases to the state selected by the environment.
            post = getattr(e, "_e345_post_turnaround_state", None)
            post_name = str(getattr(post, "name", post)).upper()
            if post_name == "CHARGING":
                rel += max(0.0, core._charge_time(e))
            etas.append(rel)
            continue

        # Local charging completion is committed.
        if cur == int(vid) and st == "CHARGING" and empty:
            etas.append(max(0.0, charge_schedule.get(eid, core._charge_time(e))))
            continue

        # Empty aircraft already flying to the candidate is committed supply.
        if st == "FLYING" and tar == int(vid) and empty:
            eta = max(0.0, fnum(getattr(e, "remaining_time", 0.0), 0.0))
            if stage in ("E4", "E5", "E6"):
                eta += float(core.TURNAROUND_DELAY_MIN)
            # After arrival the source aircraft enters charging; charge duration
            # can be estimated from its current post-flight battery state.
            eta += max(0.0, core._charge_time(e))
            if stage in ("E5", "E6"):
                eta += _pad_release_extra(scenario, vid, eta)
            etas.append(eta)

    etas.sort()
    return etas


def snapshot_wait_proxy(
    scenario: Any,
    vid: int,
    stage: str,
    charger_capacity: int,
) -> float:
    """Simple decision-time queue delay used by STTF."""
    q = queue_len(scenario, vid)
    cap = aircraft_capacity(scenario, stage)
    releases = known_supply_release_etas(scenario, vid, stage, charger_capacity)
    ready = sum(1 for x in releases if x <= 1e-9)
    if q <= 0:
        return 0.0
    if ready > 0:
        # One simulator minute per current batch as a transparent queue proxy.
        return float(math.ceil(q / max(1, cap * ready)))
    if releases:
        return float(releases[0]) + float(math.ceil(q / max(1, cap)))
    # No known supply release: finite large penalty, not infinity, so all
    # candidates remain comparable.
    return 60.0 + float(math.ceil(q / max(1, cap)))


def qtti2_wait_proxy(
    scenario: Any,
    vid: int,
    stage: str,
    horizon: float,
    charger_capacity: int,
) -> Tuple[float, Dict[str, float]]:
    """Arrival-aligned QTTI-v2 queue-time index.

    Residual workload at focal arrival:
        current queue
      + already-committed access arrivals by horizon
      - known supply releases by horizon * service batch.

    Then estimate a simple service rate from known releases in a fixed online
    window.  This is a heuristic baseline, not an oracle simulator.
    """
    h = max(0.0, float(horizon))
    cap = aircraft_capacity(scenario, stage)
    q_now = queue_len(scenario, vid)
    committed_in = committed_passenger_arrivals(scenario, vid, h)
    releases = known_supply_release_etas(scenario, vid, stage, charger_capacity)
    n_by_h = sum(1 for eta in releases if eta <= h + 1e-9)
    residual = max(0.0, float(q_now + committed_in - cap * n_by_h))

    # Known supply opportunities in [h, h+W], plus already ready supply.
    w = float(SERVICE_RATE_WINDOW_MIN)
    n_window = sum(1 for eta in releases if h < eta <= h + w + 1e-9)
    ready_at_h = max(0, n_by_h)
    # Use at least one batch per window as a conservative finite denominator.
    service_rate = max(
        float(cap) / w,
        float(cap * max(1, n_window + min(1, ready_at_h))) / w,
    )
    wait = residual / service_rate

    # If no supply is known by arrival, include time to first known future release.
    future = [eta for eta in releases if eta > h + 1e-9]
    if residual > 0 and n_by_h == 0:
        wait += (min(future) - h) if future else 60.0

    return float(wait), {
        "q_now": float(q_now),
        "committed_in_by_h": float(committed_in),
        "known_supply_by_h": float(n_by_h),
        "batch_capacity": float(cap),
        "residual_queue_at_arrival": float(residual),
        "service_rate_proxy": float(service_rate),
    }


def choose_action(
    env: Any,
    stage: str,
    topology: str,
    method: str,
    charger_capacity: int,
) -> Tuple[int, Dict[str, Any]]:
    scenario = find_scenario(env)
    person = focal_person(env)
    cands = core.candidates_for(topology)
    if person is None:
        return 0, {"idle_decision": True}

    method = str(method).upper()
    scores: Dict[int, float] = {}
    details: Dict[int, Dict[str, float]] = {}

    for vid in cands:
        h = access_time(scenario, person, vid)
        if method == "SPF":
            score = h
            detail = {"access": h}
        elif method == "STTF":
            wait = snapshot_wait_proxy(scenario, vid, stage, charger_capacity)
            score = h + wait
            detail = {"access": h, "snapshot_wait": wait}
        elif method == "QTTI2":
            wait, extra = qtti2_wait_proxy(
                scenario, vid, stage, h, charger_capacity
            )
            score = h + wait
            detail = {"access": h, "qtti2_wait": wait, **extra}
        else:
            raise ValueError(method)
        scores[int(vid)] = float(score)
        details[int(vid)] = detail

    chosen_vid = min(cands, key=lambda v: (scores[int(v)], int(v)))
    action = cands.index(chosen_vid)
    return int(action), {
        "chosen_vid": int(chosen_vid),
        "scores": scores,
        "details": details,
    }


def run_one(
    *,
    stage: str,
    topology: str,
    method: str,
    seed: int,
    fleet_size: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
    run_dir: Path,
) -> Dict[str, Any]:
    factory = core.make_experiment_env_factory(
        stage=stage,
        topology=topology,
        encoder_mode="uagmc",  # rules ignore obs; keep source wrapper semantics
        fleet_size=fleet_size,
        env_index=int(seed),
        run_dir=run_dir,
        pad_separation=pad_separation,
        charger_capacity=charger_capacity,
        max_time=max_time,
    )
    env = factory()

    try:
        try:
            out = env.reset(seed=int(seed))
        except TypeError:
            out = env.reset()

        done = False
        steps = 0
        reward_sum = 0.0
        action_counts = Counter()
        system_person_minutes = 0.0
        decision_rows: List[Dict[str, Any]] = []
        terminal_snapshot = None

        while not done:
            scenario = find_scenario(env)
            system_person_minutes += core.mx.active_system_count(scenario)

            action, detail = choose_action(
                env, stage, topology, method, charger_capacity
            )
            action_counts[int(action)] += 1

            if not detail.get("idle_decision", False):
                decision_rows.append({
                    "sim_time": int(getattr(scenario, "time", steps)),
                    "action": int(action),
                    "method": method,
                    "detail": detail,
                })

            out = env.step(action)
            if len(out) == 5:
                _, reward, terminated, truncated, info = out
                done = bool(terminated) or bool(truncated)
            else:
                _, reward, done, info = out
                done = bool(done)
            reward_sum += fnum(reward, 0.0)
            steps += 1

            if isinstance(info, dict) and "terminal_snapshot" in info:
                terminal_snapshot = info["terminal_snapshot"]

            if steps > int(max_time) + 100:
                raise RuntimeError("baseline episode exceeded max-time guard")

        if terminal_snapshot is None:
            terminal_snapshot = core.mx.snapshot_episode(find_scenario(env))

        metrics = core.mx.metrics_from_terminal_snapshot(
            snapshot=terminal_snapshot,
            action_counts=action_counts,
            prob_rows=[],
            system_person_minutes=system_person_minutes,
            reward_sum=reward_sum,
            episode_steps=steps,
        )
        metrics.update({
            "stage": stage,
            "topology": topology,
            "method": method,
            "eval_seed": int(seed),
        })

        # Raw decision audit is especially useful for redesigned QTTI2.
        if method == "QTTI2":
            flat = []
            for row in decision_rows:
                d = row["detail"]
                base = {
                    "sim_time": row["sim_time"],
                    "action": row["action"],
                    "chosen_vid": d.get("chosen_vid"),
                }
                for vid, score in (d.get("scores") or {}).items():
                    base[f"score_v{vid}"] = score
                flat.append(base)
            write_csv(run_dir / "qtti2_decisions.csv", flat)

        return metrics
    finally:
        try:
            env.close()
        except Exception:
            pass
        core.restore_process_patches()


def aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (str(r["stage"]), str(r["topology"]), str(r["method"]))
        groups.setdefault(key, []).append(r)

    metrics = [
        "ATT", "AWT", "AGT_access", "AFT", "completion_rate", "backlog",
        "system_person_minutes_per_passenger", "travel_p90",
    ]
    out = []
    for (stage, topology, method), rs in sorted(groups.items()):
        row: Dict[str, Any] = {
            "stage": stage,
            "topology": topology,
            "method": method,
            "n_eval_seeds": len(rs),
        }
        for m in metrics:
            row[m + "_mean"] = mean(r.get(m) for r in rs)
            row[m + "_std"] = std(r.get(m) for r in rs)
        action_keys = sorted({k for r in rs for k in r if k.startswith("action_") and k.endswith("_share")})
        for k in action_keys:
            row[k + "_mean"] = mean(r.get(k) for r in rs)
        out.append(row)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="E0/E2/E3/E4/E5/E6 no-learning SPF/STTF/QTTI2 baselines")
    p.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--topology", choices=["T2", "T3"], default=DEFAULT_TOPOLOGY)
    p.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    p.add_argument("--fleet-size", type=int, default=core.FLEET_SIZE)
    p.add_argument("--pad-separation", type=float, default=core.PAD_SEPARATION_MIN)
    p.add_argument("--charger-capacity", type=int, default=core.CHARGER_CAPACITY)
    p.add_argument("--max-time", type=int, default=core.MAX_TIME)
    p.add_argument("--output-root", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stages = parse_csv_list(args.stages)
    methods = parse_csv_list(args.methods)
    seeds = parse_ints(args.seeds)

    bad_s = [x for x in stages if x not in DEFAULT_STAGES]
    bad_m = [x for x in methods if x not in DEFAULT_METHODS]
    if bad_s:
        raise ValueError(f"unsupported stages: {bad_s}")
    if bad_m:
        raise ValueError(f"unsupported methods: {bad_m}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (ROOT / "serial_runs" / f"uagmc_rule_baselines_E0_E2_E6_{args.topology}_{stamp}").resolve()
    )
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "UAGMC_E0_E2_E6_RULE_BASELINES",
        "created": datetime.now().isoformat(timespec="seconds"),
        "stages": stages,
        "methods": methods,
        "topology": args.topology,
        "seeds": seeds,
        "fleet_size_E2_E6": args.fleet_size,
        "E0_fleet_semantics": "source legacy automatic replenishment",
        "pad_separation_min": args.pad_separation,
        "charger_capacity_E6": args.charger_capacity,
        "max_time": args.max_time,
        "passenger_trace": str(core.TRAIN_FILE),
        "SPF": "min focal passenger ground-access time",
        "STTF": "access + decision-time snapshot queue/supply wait proxy",
        "QTTI2": (
            "access + arrival-aligned residual-queue/service-rate index using only "
            "already committed passenger/aircraft events; no unrevealed demand"
        ),
        "compute": "CPU-only no-learning baseline evaluation",
    }
    core.write_json(root / "manifest.json", manifest)

    rows: List[Dict[str, Any]] = []
    total = len(stages) * len(methods) * len(seeds)
    idx = 0
    print("=" * 118)
    print("UAGMC E0/E2/E3/E4/E5/E6 RULE BASELINES")
    print(f"Topology={args.topology} | runs={total} | CPU-only")
    print("=" * 118)

    for stage in stages:
        for method in methods:
            for seed in seeds:
                idx += 1
                run_dir = root / f"{stage}__{method}__seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                print(f"[{idx:02d}/{total:02d}] {stage} / {method} / seed={seed}", flush=True)
                row = run_one(
                    stage=stage,
                    topology=args.topology,
                    method=method,
                    seed=seed,
                    fleet_size=args.fleet_size,
                    pad_separation=args.pad_separation,
                    charger_capacity=args.charger_capacity,
                    max_time=args.max_time,
                    run_dir=run_dir,
                )
                rows.append(row)
                write_csv(root / "raw_results.csv", rows)
                write_csv(root / "aggregate_results.csv", aggregate(rows))
                print(
                    f"  ATT={fnum(row.get('ATT')):.3f} | AWT={fnum(row.get('AWT')):.3f} | "
                    f"finish={int(row.get('N_finished', 0))}/{int(row.get('N', 0))} | "
                    f"Jsys/N={fnum(row.get('system_person_minutes_per_passenger')):.3f}",
                    flush=True,
                )

    agg = aggregate(rows)
    write_csv(root / "raw_results.csv", rows)
    write_csv(root / "aggregate_results.csv", agg)

    summary_lines = [
        "UAGMC RULE BASELINE SUMMARY",
        "=" * 100,
        "stage | method | ATT | AWT | completion | Jsys/N",
    ]
    for r in agg:
        summary_lines.append(
            f"{r['stage']:>2} | {r['method']:<5} | "
            f"{fnum(r.get('ATT_mean')):8.3f} | {fnum(r.get('AWT_mean')):8.3f} | "
            f"{100*fnum(r.get('completion_rate_mean')):7.2f}% | "
            f"{fnum(r.get('system_person_minutes_per_passenger_mean')):8.3f}"
        )
    (root / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nDONE")
    print(f"Results: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

