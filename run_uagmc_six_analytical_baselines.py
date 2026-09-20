# -*- coding: utf-8 -*-
"""
Six online-legal analytical/rule baselines for E0/E2/E3/E4/E5/E6.

Methods
-------
SPF
    Shortest passenger access time.
STTF
    Existing decision-time access + snapshot wait proxy.
QTTI2
    Existing arrival-aligned residual queue/service-rate heuristic.
CSM
    Committed-Supply Matching.  Discretely matches the FIFO passenger batches
    known at decision time to exact committed aircraft-release events.  It
    predicts focal departure time, including the currently-known origin TLOF
    calendar in E5/E6.
ECTF
    Earliest Completion-Time Forecast.  CSM + candidate-specific eVTOL flight
    time + currently-known destination TLOF availability.  It is a transparent
    door-to-door completion-time forecast for the focal passenger.
MPTC
    Marginal Person-Time Cost.  ECTF plus the change in future waiting time of
    all passengers already queued or already committed/enroute to the chosen
    candidate.  This approximates the system objective rather than only the
    focal passenger's own completion time.

Causality / legality
--------------------
The three new methods use only state already known at the decision time:
current queues, already-committed passenger access ETAs, ready/charging/
turnaround aircraft, already-committed empty inbound aircraft, and current TLOF
calendars.  They DO NOT read unrevealed future passenger requests and DO NOT
simulate uncommitted future reposition decisions.

When an exact future service event cannot be known causally (e.g. an incomplete
E0/E2 full-capacity batch or demand beyond all currently committed supply), a
fixed transparent penalty is used instead of peeking at future requests/actions.

This file extends the validated environment/evaluation plumbing in:
    run_uagmc_E0_E2_E6_rule_baselines.py
and should be placed beside it in UAGMC-main.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import run_uagmc_E0_E2_E6_rule_baselines as old
import train_uagmc_E0_E2_E6_effect_time_700k as core


ROOT = Path(__file__).resolve().parent
DEFAULT_STAGES = ("E0", "E2", "E3", "E4", "E5", "E6")
DEFAULT_METHODS = ("SPF", "STTF", "QTTI2", "CSM", "ECTF", "MPTC")
DEFAULT_TOPOLOGY = "T2"  # aligned with the completed 6x6 main experiment
DEFAULT_SEEDS = (123, 124, 125)
DEFAULT_UNKNOWN_EVENT_PENALTY_MIN = 60.0

# Convenience aliases for the user's shorthand.
METHOD_ALIASES = {
    "STTI": "STTF",
    "DTTI": "QTTI2",
    "QTTI": "QTTI2",
}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def fnum(x: Any, default: float = float("nan")) -> float:
    return old.fnum(x, default)


def mean(xs: Iterable[Any]) -> float:
    return old.mean(xs)


def std(xs: Iterable[Any]) -> float:
    return old.std(xs)


def normalize_method(name: str) -> str:
    x = str(name).strip().upper()
    return METHOD_ALIASES.get(x, x)


def parse_methods(text: str) -> List[str]:
    return [normalize_method(x) for x in str(text).split(",") if x.strip()]


def stage_has_pad(stage: str) -> bool:
    return str(stage).upper() in ("E5", "E6")


def pad_separation() -> float:
    return float(getattr(core.mx.base, "PAD_SEPARATION_MIN", core.PAD_SEPARATION_MIN) or 0.0)


# -----------------------------------------------------------------------------
# Online-legal committed event extraction
# -----------------------------------------------------------------------------

def committed_passenger_eta_list(scenario: Any, vid: int) -> List[float]:
    """All already-committed passenger access ETAs to candidate vid.

    No unrevealed future requests are inspected.
    """
    out: List[float] = []
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
        out.append(float(eta))
    out.sort()
    return out


def service_flight_time(scenario: Any, origin_vid: int, destination_vid: int) -> float:
    """Deterministic eVTOL flight time from the simulator geometry/spec.

    The fixed-fleet environment uses Euclidean vertiport distance and
    distance / max_speed * 60.  We reproduce exactly that calculation here.
    """
    try:
        if hasattr(scenario, "_distance_between_vertiports"):
            distance = float(
                scenario._distance_between_vertiports(str(int(origin_vid)), str(int(destination_vid)))
            )
        else:
            p1 = scenario.vertiports.vertiport_list[str(int(origin_vid))].vertiport_position
            p2 = scenario.vertiports.vertiport_list[str(int(destination_vid))].vertiport_position
            distance = math.sqrt(
                (float(p2[0]) - float(p1[0])) ** 2
                + (float(p2[1]) - float(p1[1])) ** 2
            )

        speed = None
        for e in list(getattr(scenario, "_all_evtols", {}).values()):
            v = fnum(getattr(getattr(e, "spec", None), "max_speed", np.nan))
            if np.isfinite(v) and v > 1e-9:
                speed = float(v)
                break
        if speed is None:
            # Fail-safe only.  A normal UAGMC scenario always has an eVTOL spec.
            return 0.0
        return float(distance / speed * 60.0)
    except Exception:
        return 0.0


def current_pad_next_free_eta(scenario: Any, vid: int, stage: str) -> float:
    if not stage_has_pad(stage):
        return 0.0
    return max(0.0, fnum(core.mx._pad_next_free_eta(scenario, int(vid)), 0.0))


# -----------------------------------------------------------------------------
# Discrete committed-event service schedule
# -----------------------------------------------------------------------------

def _passenger_population(
    scenario: Any,
    vid: int,
    *,
    focal_horizon: Optional[float],
) -> List[Tuple[str, float, bool]]:
    """Known FIFO passenger population, relative to now.

    Current queue members are already present at eta=0.  Already committed
    enroute passengers keep their known access ETA.  The optional focal
    passenger is inserted after already-committed passengers at an equal ETA.
    """
    items: List[Tuple[str, float, bool, int]] = []
    q = old.queue_len(scenario, int(vid))
    for i in range(int(q)):
        items.append((f"Q{i}", 0.0, False, 0))
    for i, eta in enumerate(committed_passenger_eta_list(scenario, int(vid))):
        items.append((f"C{i}", float(eta), False, 0))
    if focal_horizon is not None:
        items.append(("FOCAL", max(0.0, float(focal_horizon)), True, 1))

    items.sort(key=lambda x: (x[1], x[3], x[0]))
    return [(pid, eta, is_focal) for pid, eta, is_focal, _ in items]


def _virtual_schedule(
    scenario: Any,
    *,
    stage: str,
    vid: int,
    destination: int,
    focal_horizon: Optional[float],
    charger_capacity: int,
    unknown_penalty: float,
    include_destination: bool,
) -> Dict[str, Any]:
    """Project FIFO service using only committed passenger/supply events.

    Each known aircraft release is a single service opportunity.  For E0/E2,
    a flight requires a full source-capacity batch; an incomplete last batch has
    no causal known departure time and therefore receives the fixed unknown
    event penalty.  E3-E6 use capacity=1.
    """
    cap = max(1, int(old.aircraft_capacity(scenario, stage)))
    people = _passenger_population(
        scenario,
        int(vid),
        focal_horizon=focal_horizon,
    )
    releases = list(
        old.known_supply_release_etas(
            scenario,
            int(vid),
            stage,
            int(charger_capacity),
        )
    )
    releases = sorted(max(0.0, float(x)) for x in releases)

    origin_pad_next = current_pad_next_free_eta(scenario, int(vid), stage)
    dest_pad_next = current_pad_next_free_eta(scenario, int(destination), stage)
    sep = pad_separation() if stage_has_pad(stage) else 0.0
    flight = service_flight_time(scenario, int(vid), int(destination))

    service_time: Dict[str, float] = {}
    completion_time: Dict[str, float] = {}
    fallback_batches = 0
    incomplete_batch = False

    batches = [people[i : i + cap] for i in range(0, len(people), cap)]
    for bi, batch in enumerate(batches):
        if not batch:
            continue
        batch_ready = max(float(x[1]) for x in batch)
        is_full = len(batch) == cap

        if bi < len(releases):
            supply_ready = float(releases[bi])
        else:
            # We refuse to peek at future LQ/replenishment actions.  Use a
            # transparent finite penalty after the last known supply event.
            fallback_batches += 1
            last = float(releases[-1]) if releases else 0.0
            supply_ready = max(batch_ready, last) + float(unknown_penalty) * fallback_batches

        if not is_full and cap > 1:
            # Source E0/E2 only departs with a full eVTOL batch.  The missing
            # future passenger is unrevealed, so its arrival time is unknown.
            incomplete_batch = True
            supply_ready = max(supply_ready, batch_ready + float(unknown_penalty))

        depart = max(batch_ready, supply_ready)

        if stage_has_pad(stage):
            depart = max(depart, origin_pad_next)

        arrive = depart + float(flight)

        if include_destination and stage_has_pad(stage):
            # The environment reserves origin takeoff + destination landing.
            # If the destination's currently-known next slot is later than the
            # physical arrival, delay the departure so the landing is feasible.
            if dest_pad_next > arrive:
                shift = dest_pad_next - arrive
                depart += shift
                arrive += shift
            origin_pad_next = depart + sep
            dest_pad_next = arrive + sep
        elif stage_has_pad(stage):
            origin_pad_next = depart + sep

        for pid, _, _ in batch:
            service_time[pid] = float(depart)
            completion_time[pid] = float(arrive)

    total_future_wait = 0.0
    for pid, eta, is_focal in people:
        if is_focal:
            continue
        st = service_time.get(pid)
        if st is not None:
            total_future_wait += max(0.0, float(st) - float(eta))

    return {
        "capacity": cap,
        "people": people,
        "known_supply_releases": releases,
        "service_time": service_time,
        "completion_time": completion_time,
        "focal_departure": service_time.get("FOCAL", float("inf")),
        "focal_completion": completion_time.get("FOCAL", float("inf")),
        "total_existing_future_wait": float(total_future_wait),
        "flight_time": float(flight),
        "fallback_batches": int(fallback_batches),
        "incomplete_batch": bool(incomplete_batch),
        "origin_pad_next_now": current_pad_next_free_eta(scenario, int(vid), stage),
        "destination_pad_next_now": current_pad_next_free_eta(scenario, int(destination), stage),
    }


def committed_supply_matching_score(
    scenario: Any,
    *,
    stage: str,
    vid: int,
    focal_horizon: float,
    charger_capacity: int,
    unknown_penalty: float,
) -> Tuple[float, Dict[str, Any]]:
    sched = _virtual_schedule(
        scenario,
        stage=stage,
        vid=int(vid),
        destination=int(core.DESTINATION),
        focal_horizon=float(focal_horizon),
        charger_capacity=int(charger_capacity),
        unknown_penalty=float(unknown_penalty),
        include_destination=False,
    )
    score = float(sched["focal_departure"])
    return score, {
        "access": float(focal_horizon),
        "predicted_departure": score,
        "predicted_wait_after_access": max(0.0, score - float(focal_horizon)),
        "known_supply_count": len(sched["known_supply_releases"]),
        "fallback_batches": sched["fallback_batches"],
        "incomplete_batch": sched["incomplete_batch"],
    }


def earliest_completion_score(
    scenario: Any,
    *,
    stage: str,
    vid: int,
    focal_horizon: float,
    charger_capacity: int,
    unknown_penalty: float,
) -> Tuple[float, Dict[str, Any]]:
    sched = _virtual_schedule(
        scenario,
        stage=stage,
        vid=int(vid),
        destination=int(core.DESTINATION),
        focal_horizon=float(focal_horizon),
        charger_capacity=int(charger_capacity),
        unknown_penalty=float(unknown_penalty),
        include_destination=True,
    )
    score = float(sched["focal_completion"])
    return score, {
        "access": float(focal_horizon),
        "predicted_departure": sched["focal_departure"],
        "flight_time": sched["flight_time"],
        "predicted_completion": score,
        "origin_pad_next_now": sched["origin_pad_next_now"],
        "destination_pad_next_now": sched["destination_pad_next_now"],
        "known_supply_count": len(sched["known_supply_releases"]),
        "fallback_batches": sched["fallback_batches"],
        "incomplete_batch": sched["incomplete_batch"],
    }


def marginal_person_time_score(
    scenario: Any,
    *,
    stage: str,
    vid: int,
    focal_horizon: float,
    charger_capacity: int,
    unknown_penalty: float,
) -> Tuple[float, Dict[str, Any]]:
    before = _virtual_schedule(
        scenario,
        stage=stage,
        vid=int(vid),
        destination=int(core.DESTINATION),
        focal_horizon=None,
        charger_capacity=int(charger_capacity),
        unknown_penalty=float(unknown_penalty),
        include_destination=True,
    )
    after = _virtual_schedule(
        scenario,
        stage=stage,
        vid=int(vid),
        destination=int(core.DESTINATION),
        focal_horizon=float(focal_horizon),
        charger_capacity=int(charger_capacity),
        unknown_penalty=float(unknown_penalty),
        include_destination=True,
    )

    focal_completion = float(after["focal_completion"])
    externality = (
        float(after["total_existing_future_wait"])
        - float(before["total_existing_future_wait"])
    )
    score = focal_completion + externality

    return float(score), {
        "access": float(focal_horizon),
        "focal_completion": focal_completion,
        "existing_wait_before": before["total_existing_future_wait"],
        "existing_wait_after": after["total_existing_future_wait"],
        "marginal_existing_wait": float(externality),
        "marginal_person_time_score": float(score),
        "flight_time": after["flight_time"],
        "known_supply_count": len(after["known_supply_releases"]),
        "fallback_batches": after["fallback_batches"],
        "incomplete_batch": after["incomplete_batch"],
    }


# -----------------------------------------------------------------------------
# Policy selection
# -----------------------------------------------------------------------------

def choose_action(
    env: Any,
    stage: str,
    topology: str,
    method: str,
    charger_capacity: int,
    unknown_penalty: float,
) -> Tuple[int, Dict[str, Any]]:
    method = normalize_method(method)

    if method in ("SPF", "STTF", "QTTI2"):
        return old.choose_action(
            env,
            stage,
            topology,
            method,
            charger_capacity,
        )

    scenario = old.find_scenario(env)
    person = old.focal_person(env)
    cands = core.candidates_for(topology)
    if person is None:
        return 0, {"idle_decision": True}

    scores: Dict[int, float] = {}
    details: Dict[int, Dict[str, Any]] = {}

    for vid in cands:
        h = old.access_time(scenario, person, int(vid))
        if method == "CSM":
            score, detail = committed_supply_matching_score(
                scenario,
                stage=stage,
                vid=int(vid),
                focal_horizon=h,
                charger_capacity=charger_capacity,
                unknown_penalty=unknown_penalty,
            )
        elif method == "ECTF":
            score, detail = earliest_completion_score(
                scenario,
                stage=stage,
                vid=int(vid),
                focal_horizon=h,
                charger_capacity=charger_capacity,
                unknown_penalty=unknown_penalty,
            )
        elif method == "MPTC":
            score, detail = marginal_person_time_score(
                scenario,
                stage=stage,
                vid=int(vid),
                focal_horizon=h,
                charger_capacity=charger_capacity,
                unknown_penalty=unknown_penalty,
            )
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


# -----------------------------------------------------------------------------
# Episode evaluation
# -----------------------------------------------------------------------------

def _flatten_decision_row(row: Dict[str, Any]) -> Dict[str, Any]:
    d = row.get("detail", {}) or {}
    out: Dict[str, Any] = {
        "sim_time": row.get("sim_time"),
        "action": row.get("action"),
        "method": row.get("method"),
        "chosen_vid": d.get("chosen_vid"),
    }
    for vid, score in (d.get("scores") or {}).items():
        out[f"score_v{vid}"] = score
    # Keep full per-candidate diagnostics as JSON for auditability.
    out["details_json"] = json.dumps(d.get("details") or {}, ensure_ascii=False)
    return out


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
    unknown_penalty: float,
    run_dir: Path,
) -> Dict[str, Any]:
    method = normalize_method(method)
    factory = core.make_experiment_env_factory(
        stage=stage,
        topology=topology,
        encoder_mode="uagmc",
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
            env.reset(seed=int(seed))
        except TypeError:
            env.reset()

        done = False
        steps = 0
        reward_sum = 0.0
        action_counts = Counter()
        system_person_minutes = 0.0
        decision_rows: List[Dict[str, Any]] = []
        terminal_snapshot = None

        while not done:
            scenario = old.find_scenario(env)
            system_person_minutes += core.mx.active_system_count(scenario)

            action, detail = choose_action(
                env,
                stage,
                topology,
                method,
                charger_capacity,
                unknown_penalty,
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
            terminal_snapshot = core.mx.snapshot_episode(old.find_scenario(env))

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
            "unknown_event_penalty_min": float(unknown_penalty),
        })

        old.write_csv(
            run_dir / f"{method.lower()}_decisions.csv",
            [_flatten_decision_row(r) for r in decision_rows],
        )
        return metrics
    finally:
        try:
            env.close()
        except Exception:
            pass
        core.restore_process_patches()


def aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return old.aggregate(rows)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Six online-legal analytical UAGMC baselines over E0/E2/E3/E4/E5/E6"
    )
    p.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--topology", choices=["T2", "T3"], default=DEFAULT_TOPOLOGY)
    p.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    p.add_argument("--fleet-size", type=int, default=core.FLEET_SIZE)
    p.add_argument("--pad-separation", type=float, default=core.PAD_SEPARATION_MIN)
    p.add_argument("--charger-capacity", type=int, default=core.CHARGER_CAPACITY)
    p.add_argument("--max-time", type=int, default=core.MAX_TIME)
    p.add_argument(
        "--unknown-event-penalty",
        type=float,
        default=DEFAULT_UNKNOWN_EVENT_PENALTY_MIN,
        help="finite penalty when causal future supply/full-batch time is unknown",
    )
    p.add_argument("--output-root", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stages = old.parse_csv_list(args.stages)
    methods = parse_methods(args.methods)
    seeds = old.parse_ints(args.seeds)

    bad_s = [x for x in stages if x not in DEFAULT_STAGES]
    bad_m = [x for x in methods if x not in DEFAULT_METHODS]
    if bad_s:
        raise ValueError(f"unsupported stages: {bad_s}")
    if bad_m:
        raise ValueError(f"unsupported methods: {bad_m}")
    if args.unknown_event_penalty <= 0:
        raise ValueError("--unknown-event-penalty must be > 0")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (ROOT / "serial_runs" / f"uagmc_six_analytical_baselines_{args.topology}_{stamp}").resolve()
    )
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "UAGMC_SIX_ANALYTICAL_BASELINES",
        "created": datetime.now().isoformat(timespec="seconds"),
        "stages": stages,
        "methods": methods,
        "method_aliases": METHOD_ALIASES,
        "topology": args.topology,
        "seeds": seeds,
        "fleet_size_E2_E6": args.fleet_size,
        "E0_fleet_semantics": "source legacy automatic replenishment",
        "pad_separation_min": args.pad_separation,
        "charger_capacity_E6": args.charger_capacity,
        "max_time": args.max_time,
        "unknown_event_penalty_min": args.unknown_event_penalty,
        "passenger_trace": str(core.TRAIN_FILE),
        "causality": "no unrevealed future passenger requests; no uncommitted future reposition actions",
        "method_descriptions": {
            "SPF": "minimum focal passenger ground-access time",
            "STTF": "access + decision-time snapshot queue/supply wait proxy",
            "QTTI2": "access + arrival-aligned residual queue/service-rate index",
            "CSM": "discrete FIFO committed-passenger / committed-supply matching to predicted departure",
            "ECTF": "CSM + service-flight time + current origin/destination TLOF feasibility",
            "MPTC": "ECTF focal completion + marginal future waiting imposed on already known passengers",
        },
        "compute": "CPU-only no-learning baseline evaluation",
    }
    core.write_json(root / "manifest.json", manifest)

    rows: List[Dict[str, Any]] = []
    total = len(stages) * len(methods) * len(seeds)
    idx = 0

    print("=" * 126)
    print("UAGMC SIX ANALYTICAL BASELINES")
    print(f"Topology={args.topology} | methods={methods} | runs={total} | CPU-only")
    print("=" * 126)

    for stage in stages:
        for method in methods:
            for seed in seeds:
                idx += 1
                run_dir = root / f"{stage}__{method}__seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                print(f"[{idx:03d}/{total:03d}] {stage} / {method} / seed={seed}", flush=True)

                row = run_one(
                    stage=stage,
                    topology=args.topology,
                    method=method,
                    seed=seed,
                    fleet_size=args.fleet_size,
                    pad_separation=args.pad_separation,
                    charger_capacity=args.charger_capacity,
                    max_time=args.max_time,
                    unknown_penalty=args.unknown_event_penalty,
                    run_dir=run_dir,
                )
                rows.append(row)
                old.write_csv(root / "raw_results.csv", rows)
                old.write_csv(root / "aggregate_results.csv", aggregate(rows))

                print(
                    f"  ATT={fnum(row.get('ATT')):.3f} | "
                    f"AWT={fnum(row.get('AWT')):.3f} | "
                    f"finish={int(row.get('N_finished', 0))}/{int(row.get('N', 0))} | "
                    f"Jsys/N={fnum(row.get('system_person_minutes_per_passenger')):.3f}",
                    flush=True,
                )

    agg = aggregate(rows)
    old.write_csv(root / "raw_results.csv", rows)
    old.write_csv(root / "aggregate_results.csv", agg)

    # Stage-wise ranking is descriptive only; lower Jsys/N is the configured
    # system objective proxy and is safer than completed-only ATT under censoring.
    summary_lines = [
        "UAGMC SIX ANALYTICAL BASELINE SUMMARY",
        "=" * 112,
        "stage | method | ATT | AWT | completion | Jsys/N",
    ]
    for stage in stages:
        stage_rows = [r for r in agg if r["stage"] == stage]
        stage_rows.sort(key=lambda r: fnum(r.get("system_person_minutes_per_passenger_mean"), float("inf")))
        for r in stage_rows:
            summary_lines.append(
                f"{r['stage']:>2} | {r['method']:<5} | "
                f"{fnum(r.get('ATT_mean')):8.3f} | {fnum(r.get('AWT_mean')):8.3f} | "
                f"{100*fnum(r.get('completion_rate_mean')):7.2f}% | "
                f"{fnum(r.get('system_person_minutes_per_passenger_mean')):8.3f}"
            )
        summary_lines.append("-")

    (root / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nDONE")
    print(f"Results: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
