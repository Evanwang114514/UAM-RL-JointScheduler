# -*- coding: utf-8 -*-
"""
UAGMC fixed-fleet FAIRNESS scanner (NO TRAINING)
================================================

Goal
----
Select a fixed-fleet size for a fair comparison between:

E0: legacy automatic aircraft replenishment
E1: fixed conserved fleet + simple return-to-home reposition

The fleet size is calibrated ONLY from no-training physical feasibility.
It is never selected from PPO performance.

Main rule
---------
Use fleet sizes that preserve the original V0:V1 supply ratio exactly
(default multiples of 4 -> 3:1). For each fleet size, scan deterministic
STATIC passenger routing ratios P(V0). The recommended main fleet size is
THE SMALLEST fleet for which the supply-matched static routing ratio is
robustly feasible across all tested phase offsets.

This prevents two confounds:
- too-small fleet: physical infeasibility can be mistaken for PPO failure;
- too-large fleet: excess aircraft can make routing nearly trivial.

Default scan
------------
fleet sizes: 8,12,16,20,24
P(V0):       0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
phases:      0,1,2,3
passengers:  train_data/passengers_300.csv
max_time:    600

Outputs
-------
fixed_fleet_fairness_scan/
    raw_runs.csv
    ratio_aggregate.csv
    fleet_summary.csv
    recommendation.json
    recommendation.txt
    manifest.json
    completion_heatmap.png
    awt_heatmap.png
    backlog_heatmap.png
    matched_completion.png
    matched_backlog.png
    matched_awt.png
    errors.csv

Run
---
python run_uagmc_fixed_fleet_fairness_scan.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utilss.make_env_fleet import make_env

ROOT = Path(__file__).resolve().parent


def parse_int_list(s: str) -> List[int]:
    x = [int(v.strip()) for v in str(s).split(",") if v.strip()]
    if not x:
        raise ValueError("empty integer list")
    return x


def parse_float_list(s: str) -> List[float]:
    x = [float(v.strip()) for v in str(s).split(",") if v.strip()]
    if not x:
        raise ValueError("empty float list")
    return x


def fnum(x: Any, default=float("nan")) -> float:
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return default


def fmean(xs: Iterable[Any]) -> float:
    a = np.asarray([fnum(x) for x in xs], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def fstd(xs: Iterable[Any]) -> float:
    a = np.asarray([fnum(x) for x in xs], dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    return 0.0 if len(a) == 1 else float(a.std(ddof=1))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            o = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                o[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
            w.writerow(o)


def find_wrapper(env: Any):
    obj, seen = env, set()
    for _ in range(50):
        if id(obj) in seen:
            break
        seen.add(id(obj))
        if all(hasattr(obj, k) for k in ("state", "encoder", "decoder", "env")):
            return obj
        if hasattr(obj, "env") and getattr(obj, "env") is not obj:
            obj = obj.env
            continue
        if hasattr(obj, "unwrapped") and obj.unwrapped is not obj:
            obj = obj.unwrapped
            continue
        break
    raise RuntimeError(f"Cannot locate UAMRLWrapper; stopped at {type(obj)}")


def find_scenario(env: Any):
    obj, seen = env, set()
    for _ in range(60):
        if id(obj) in seen:
            break
        seen.add(id(obj))
        if all(hasattr(obj, k) for k in ("person_travel_records", "persons", "finished_ids", "vertiports", "_all_evtols")):
            return obj
        if hasattr(obj, "scenario") and getattr(obj, "scenario") is not obj:
            obj = obj.scenario
            continue
        if hasattr(obj, "env") and getattr(obj, "env") is not obj:
            obj = obj.env
            continue
        if hasattr(obj, "unwrapped") and obj.unwrapped is not obj:
            obj = obj.unwrapped
            continue
        break
    raise RuntimeError(f"Cannot locate Scenario; stopped at {type(obj)}")


def unwrap_step(out):
    if not isinstance(out, tuple):
        raise RuntimeError(f"unexpected env.step return type: {type(out)}")
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated), bool(truncated), info
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, bool(done), False, info
    raise RuntimeError(f"unexpected env.step tuple length: {len(out)}")


def get_waiting_pid(wrapper: Any) -> Optional[str]:
    s = getattr(wrapper, "state", None)
    if not isinstance(s, dict):
        return None
    waiting = s.get("waiting_decisions", [])
    return str(waiting[0]) if waiting else None


class StaticRatioPolicy:
    """Deterministic static routing; reads no OD, queues, aircraft state, or time."""

    def __init__(self, p_v0: float, phase: int):
        if not (0.0 <= p_v0 <= 1.0):
            raise ValueError("p_v0 must be in [0,1]")
        self.p_v0 = float(p_v0)
        self.acc = (int(phase) * 0.3819660112501051) % 1.0
        self.n = 0
        self.counts = Counter()

    def choose(self, wrapper: Any) -> int:
        if get_waiting_pid(wrapper) is None:
            return 0
        if self.p_v0 <= 0.0:
            a = 1
        elif self.p_v0 >= 1.0:
            a = 0
        else:
            self.acc += self.p_v0
            if self.acc >= 1.0:
                a = 0
                self.acc -= 1.0
            else:
                a = 1
        self.n += 1
        self.counts[a] += 1
        return a


def evtol_state_name(e: Any) -> str:
    try:
        return str(e.state.name).upper()
    except Exception:
        return str(getattr(e, "state", "UNKNOWN")).upper()


def person_rows(scenario: Any) -> List[Dict[str, Any]]:
    pobj = getattr(scenario, "persons", None)
    persons = getattr(pobj, "persons", {}) if pobj else {}
    records = getattr(scenario, "person_travel_records", {}) or {}
    finished_ids = {str(x) for x in (getattr(scenario, "finished_ids", []) or [])}
    out = []

    for pid_raw, p in persons.items():
        pid = str(pid_raw)
        recs = records.get(pid_raw) or records.get(pid) or []
        rec = recs[-1] if recs else {}
        st, et = rec.get("start_time"), rec.get("end_time")
        tt = float("nan")
        if st is not None and et is not None:
            try:
                tt = float(et) - float(st)
            except Exception:
                pass
        stats = getattr(p, "time_stats", {}) or {}
        finished = bool(pid in finished_ids or et is not None or str(getattr(p, "state", "")).lower() == "finished")
        out.append({
            "finished": finished,
            "travel": tt,
            "access": fnum(stats.get("to_vertiport", np.nan)),
            "wait": fnum(stats.get("wait_uam", np.nan)),
            "fly": fnum(stats.get("fly", np.nan)),
        })
    return out


def queue_counts(scenario: Any, candidates: Sequence[int]) -> Dict[int, int]:
    return {
        int(v): len(list(getattr(scenario.vertiports.vertiport_list[str(int(v))], "person_list", [])))
        for v in candidates
    }


def compute_metrics(scenario: Any, policy: StaticRatioPolicy, candidates: Sequence[int]) -> Dict[str, Any]:
    rows = person_rows(scenario)
    comp = [r for r in rows if r["finished"] and np.isfinite(r["travel"])]
    travel = np.asarray([r["travel"] for r in comp], dtype=float)

    def mean_key(k):
        a = np.asarray([fnum(r[k]) for r in comp], dtype=float)
        a = a[np.isfinite(a)]
        return float(a.mean()) if len(a) else float("nan")

    n = len(rows)
    nf = len(comp)
    q = queue_counts(scenario, candidates)
    diag = scenario.get_fixed_fleet_diagnostics()
    denom = max(1, policy.n)

    out = {
        "N_passengers": n,
        "N_finished": nf,
        "completion_rate": nf / n if n else float("nan"),
        "final_backlog": n - nf,
        "final_queue_total": sum(q.values()),
        "ATT": float(travel.mean()) if len(travel) else float("nan"),
        "ATT_p90": float(np.percentile(travel, 90)) if len(travel) else float("nan"),
        "AWT": mean_key("wait"),
        "AGT_access": mean_key("access"),
        "AFT": mean_key("fly"),
        "policy_decisions": policy.n,
        "realized_v0_share": policy.counts.get(0, 0) / denom,
        "realized_v1_share": policy.counts.get(1, 0) / denom,
        "service_flights_arrived": int(diag.get("service_arrivals", 0)),
        "reposition_departures": int(diag.get("reposition_departures", 0)),
        "reposition_arrivals": int(diag.get("reposition_arrivals", 0)),
        "initial_allocation": diag.get("initial_allocation", {}),
    }
    for v in candidates:
        out[f"final_queue_v{v}"] = q[int(v)]
    return out


def run_episode(
    fleet_size: int,
    p_v0: float,
    phase: int,
    seed: int,
    passenger_file: Path,
    max_time: int,
    candidates: Sequence[int],
    to_vertiport: int,
    monitor_dir: Path,
) -> Dict[str, Any]:
    run_id = fleet_size * 100000 + int(round(p_v0 * 1000)) * 10 + int(phase)
    env = make_env(
        max_time=max_time,
        log_dir=monitor_dir,
        env_index=run_id,
        person_spawn_file=str(passenger_file),
        candidate_from_vertiports=list(candidates),
        to_vertiport=to_vertiport,
        enable_logger=False,
        fleet_mode="conserved_closed_loop",
        fleet_size=fleet_size,
        fleet_assertions=True,
    )()

    try:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

        wrapper = find_wrapper(env)
        scenario = find_scenario(env)
        alloc = scenario.get_fixed_fleet_allocation()
        n0 = int(alloc.get(str(candidates[0]), 0))
        n1 = int(alloc.get(str(candidates[1]), 0))
        matched = n0 / (n0 + n1)
        policy = StaticRatioPolicy(p_v0, phase)

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            a = policy.choose(wrapper)
            _, _, terminated, truncated, _ = unwrap_step(env.step(a))
            scenario._assert_fixed_fleet()
            steps += 1
            if steps > max_time + 100:
                raise RuntimeError(f"episode exceeded expected horizon: {steps}")

        m = compute_metrics(scenario, policy, candidates)
        m.update({
            "fleet_size": fleet_size,
            "p_v0_target": p_v0,
            "p_v1_target": 1.0 - p_v0,
            "phase": phase,
            "seed": seed,
            "initial_v0_aircraft": n0,
            "initial_v1_aircraft": n1,
            "supply_matched_p_v0": matched,
            "distance_to_supply_ratio": abs(p_v0 - matched),
            "env_steps": steps,
        })
        return m
    finally:
        try:
            env.close()
        except Exception:
            pass


def aggregate(raw: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for r in raw:
        groups[(int(r["fleet_size"]), round(float(r["p_v0_target"]), 8))].append(r)

    metrics = (
        "completion_rate", "final_backlog", "final_queue_total",
        "ATT", "ATT_p90", "AWT", "AGT_access", "AFT",
        "service_flights_arrived", "reposition_departures", "reposition_arrivals",
        "realized_v0_share",
    )
    out = []

    for (n, p), g in sorted(groups.items()):
        row = {
            "fleet_size": n,
            "p_v0_target": p,
            "p_v1_target": 1.0 - p,
            "n_runs": len(g),
            "initial_v0_aircraft": int(g[0]["initial_v0_aircraft"]),
            "initial_v1_aircraft": int(g[0]["initial_v1_aircraft"]),
            "supply_matched_p_v0": float(g[0]["supply_matched_p_v0"]),
            "distance_to_supply_ratio": abs(p - float(g[0]["supply_matched_p_v0"])),
        }
        for k in metrics:
            vals = np.asarray([fnum(x.get(k, np.nan)) for x in g], dtype=float)
            good = vals[np.isfinite(vals)]
            row[f"{k}_mean"] = float(good.mean()) if len(good) else float("nan")
            row[f"{k}_std"] = 0.0 if len(good) == 1 else (float(good.std(ddof=1)) if len(good) else float("nan"))
            row[f"{k}_min"] = float(good.min()) if len(good) else float("nan")
            row[f"{k}_max"] = float(good.max()) if len(good) else float("nan")
        out.append(row)
    return out


def build_fleet_summary(
    ratio_rows: Sequence[Dict[str, Any]],
    target_completion: float,
    max_backlog_fraction: float,
    n_passengers: int,
) -> List[Dict[str, Any]]:
    by = defaultdict(list)
    for r in ratio_rows:
        by[int(r["fleet_size"])].append(r)

    max_backlog_allowed = max(1, int(math.ceil(max_backlog_fraction * n_passengers)))
    out = []

    for n in sorted(by):
        rows = by[n]
        supply_p = float(rows[0]["supply_matched_p_v0"])
        matched = min(rows, key=lambda r: (abs(float(r["p_v0_target"]) - supply_p), float(r["p_v0_target"])))

        c_worst = fnum(matched["completion_rate_min"])
        b_worst = fnum(matched["final_backlog_max"])
        feasible = bool(
            np.isfinite(c_worst)
            and c_worst >= target_completion
            and np.isfinite(b_worst)
            and b_worst <= max_backlog_allowed
        )

        # Diagnostic only: best tested static ratio, NOT used to choose main N.
        best = sorted(
            rows,
            key=lambda r: (
                -fnum(r["completion_rate_min"], -1),
                fnum(r["final_backlog_max"], float("inf")),
                fnum(r["AWT_mean"], float("inf")),
            ),
        )[0]

        out.append({
            "fleet_size": n,
            "initial_v0_aircraft": int(matched["initial_v0_aircraft"]),
            "initial_v1_aircraft": int(matched["initial_v1_aircraft"]),
            "supply_matched_p_v0": supply_p,
            "tested_matched_p_v0": float(matched["p_v0_target"]),
            "matched_completion_mean": fnum(matched["completion_rate_mean"]),
            "matched_completion_worst_phase": c_worst,
            "matched_backlog_mean": fnum(matched["final_backlog_mean"]),
            "matched_backlog_worst_phase": b_worst,
            "matched_AWT_mean": fnum(matched["AWT_mean"]),
            "matched_ATT_mean": fnum(matched["ATT_mean"]),
            "matched_service_flights_mean": fnum(matched["service_flights_arrived_mean"]),
            "matched_feasible": feasible,
            "best_static_p_v0": float(best["p_v0_target"]),
            "best_static_completion_worst_phase": fnum(best["completion_rate_min"]),
            "best_static_backlog_worst_phase": fnum(best["final_backlog_max"]),
            "best_static_AWT_mean": fnum(best["AWT_mean"]),
            "target_completion": target_completion,
            "max_backlog_allowed": max_backlog_allowed,
        })
    return out


def choose_recommendation(summary: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    feasible = [r for r in summary if bool(r["matched_feasible"])]
    if not feasible:
        mx = max(int(r["fleet_size"]) for r in summary)
        return {
            "status": "NO_FEASIBLE_FLEET_IN_SCAN",
            "recommended_fleet_size": None,
            "reason": "No tested fleet is robustly feasible under supply-matched static routing. Extend scan upward before PPO training.",
            "next_suggested_fleet_sizes": [mx + 4, mx + 8],
        }

    main = min(feasible, key=lambda r: int(r["fleet_size"]))
    idx = list(summary).index(main)
    lower = summary[idx - 1] if idx > 0 else None
    upper = summary[idx + 1] if idx + 1 < len(summary) else None

    def short(r):
        if r is None:
            return None
        return {
            "fleet_size": int(r["fleet_size"]),
            "matched_feasible": bool(r["matched_feasible"]),
            "matched_completion_worst_phase": float(r["matched_completion_worst_phase"]),
            "matched_backlog_worst_phase": float(r["matched_backlog_worst_phase"]),
            "matched_AWT_mean": float(r["matched_AWT_mean"]),
        }

    return {
        "status": "RECOMMENDED",
        "recommended_fleet_size": int(main["fleet_size"]),
        "selection_rule": "smallest fleet whose supply-matched static routing is robustly feasible across all tested phase offsets",
        "recommended_initial_allocation": {
            "V0": int(main["initial_v0_aircraft"]),
            "V1": int(main["initial_v1_aircraft"]),
        },
        "recommended_supply_matched_p_v0": float(main["supply_matched_p_v0"]),
        "matched_completion_worst_phase": float(main["matched_completion_worst_phase"]),
        "matched_backlog_worst_phase": float(main["matched_backlog_worst_phase"]),
        "matched_AWT_mean": float(main["matched_AWT_mean"]),
        "lower_neighbor": short(lower),
        "upper_neighbor": short(upper),
        "fairness_note": "Fleet size was selected without using PPO performance or forcing fixed-fleet ATT to match legacy-replenishment ATT.",
    }


def plot_heatmap(rows, field, title, label, path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped] {e}")
        return
    fleets = sorted({int(r["fleet_size"]) for r in rows})
    ratios = sorted({float(r["p_v0_target"]) for r in rows})
    z = np.full((len(fleets), len(ratios)), np.nan)
    fi = {v:i for i,v in enumerate(fleets)}
    ri = {v:i for i,v in enumerate(ratios)}
    for r in rows:
        z[fi[int(r["fleet_size"])], ri[float(r["p_v0_target"])]] = fnum(r[field])
    fig = plt.figure(figsize=(10,5.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(z, aspect="auto", origin="lower")
    ax.set_xticks(range(len(ratios))); ax.set_xticklabels([f"{x:.2f}" for x in ratios], rotation=45)
    ax.set_yticks(range(len(fleets))); ax.set_yticklabels(fleets)
    ax.set_xlabel("Static P(V0)"); ax.set_ylabel("Fixed fleet size"); ax.set_title(title)
    cb = fig.colorbar(im, ax=ax); cb.set_label(label)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_curve(summary, key, ylabel, title, path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped] {e}")
        return
    x = [int(r["fleet_size"]) for r in summary]
    y = [fnum(r[key]) for r in summary]
    fig = plt.figure(figsize=(7.5,4.5)); ax = fig.add_subplot(111)
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Fixed fleet size"); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def write_recommendation(path: Path, summary, rec):
    lines = [
        "="*118,
        "UAGMC FIXED-FLEET FAIRNESS CALIBRATION",
        "="*118,
        "",
        "Rule: select fleet size from NO-TRAINING physical feasibility only.",
        "Do NOT select N from PPO performance and do NOT force E1 ATT to equal E0 ATT.",
        "",
        "Supply-matched static routing:",
        "-"*118,
    ]
    for r in summary:
        lines.append(
            f"N={int(r['fleet_size']):>3} | alloc=({int(r['initial_v0_aircraft'])},{int(r['initial_v1_aircraft'])}) | "
            f"P(V0)={float(r['tested_matched_p_v0']):.2f} | "
            f"completion worst={100*float(r['matched_completion_worst_phase']):6.2f}% | "
            f"backlog worst={float(r['matched_backlog_worst_phase']):5.1f} | "
            f"AWT={float(r['matched_AWT_mean']):8.3f} | feasible={bool(r['matched_feasible'])}"
        )
    lines += [
        "", "Recommendation:", "-"*118,
        json.dumps(rec, indent=2, ensure_ascii=False),
        "",
        "For the later E0 vs E1 PPO comparison keep identical:",
        "passenger trace, topology, batching, reward, observation, PPO/network/hyperparameters, seeds and training budget.",
        "",
        "Evidence of algorithm degradation should come from slower convergence, larger seed variance, lower completion despite feasibility,",
        "worse gap to E1's own feasible static reference, or persistent harmful queue/action concentration -- not from higher absolute E1 ATT alone.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fleet-sizes", default="8,12,16,20,24")
    p.add_argument("--ratios", default="0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00")
    p.add_argument("--phases", default="0,1,2,3")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--passenger-file", default="train_data/passengers_300.csv")
    p.add_argument("--max-time", type=int, default=600)
    p.add_argument("--candidates", default="0,1")
    p.add_argument("--to-vertiport", type=int, default=2)
    p.add_argument("--target-completion", type=float, default=0.986667)
    p.add_argument("--max-backlog-fraction", type=float, default=0.02)
    p.add_argument("--output-dir", default="fixed_fleet_fairness_scan")
    return p.parse_args()


def main():
    args = parse_args()
    fleets = parse_int_list(args.fleet_sizes)
    ratios = parse_float_list(args.ratios)
    phases = parse_int_list(args.phases)
    candidates = parse_int_list(args.candidates)

    bad = [n for n in fleets if n % 4 != 0]
    if bad:
        raise ValueError(f"Use multiples of 4 to preserve exact 3:1 V0:V1 allocation. Invalid: {bad}")
    if any(r < 0 or r > 1 for r in ratios):
        raise ValueError("ratios must be in [0,1]")

    passenger_file = Path(args.passenger_file).expanduser()
    if not passenger_file.is_absolute():
        passenger_file = (ROOT / passenger_file).resolve()
    if not passenger_file.exists():
        raise FileNotFoundError(passenger_file)

    outdir = Path(args.output_dir).expanduser()
    if not outdir.is_absolute():
        outdir = (ROOT / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    monitor_dir = outdir / "_monitor"; monitor_dir.mkdir(exist_ok=True)

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "no-training physical calibration for fair E0 legacy-replenishment vs E1 conserved-fleet PPO comparison",
        "fleet_mode": "conserved_closed_loop",
        "fleet_sizes": fleets,
        "ratios_p_v0": ratios,
        "phases": phases,
        "seed": args.seed,
        "passenger_file": str(passenger_file),
        "max_time": args.max_time,
        "candidate_from_vertiports": candidates,
        "to_vertiport": args.to_vertiport,
        "target_completion": args.target_completion,
        "max_backlog_fraction": args.max_backlog_fraction,
        "selection_rule": "smallest fleet whose supply-matched static routing is feasible for every tested phase",
        "n_runs": len(fleets)*len(ratios)*len(phases),
    }
    (outdir/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("="*128)
    print("UAGMC FIXED-FLEET FAIRNESS SCAN | NO TRAINING")
    print("="*128)
    print(f"Fleets            : {fleets}")
    print(f"Static P(V0)      : {ratios}")
    print(f"Phases            : {phases}")
    print(f"Target completion : {args.target_completion:.6f}")
    print(f"Total episodes    : {manifest['n_runs']}")
    print(f"Output            : {outdir}")
    print("="*128)

    raw, errors = [], []
    job, total = 0, manifest["n_runs"]
    for n in fleets:
        for p0 in ratios:
            for phase in phases:
                job += 1
                print(f"[{job:>3}/{total}] N={n:<3} P(V0)={p0:.2f} phase={phase}", flush=True)
                try:
                    r = run_episode(n, p0, phase, args.seed, passenger_file, args.max_time, candidates, args.to_vertiport, monitor_dir)
                    raw.append(r)
                    print(f"    completion={100*r['completion_rate']:.2f}% | backlog={r['final_backlog']} | ATT={r['ATT']:.3f} | AWT={r['AWT']:.3f}", flush=True)
                except Exception as e:
                    errors.append({"fleet_size":n,"p_v0_target":p0,"phase":phase,"error":repr(e)})
                    print(f"    ERROR: {repr(e)}", flush=True)

    write_csv(outdir/"raw_runs.csv", raw)
    write_csv(outdir/"errors.csv", errors)
    agg = aggregate(raw)
    write_csv(outdir/"ratio_aggregate.csv", agg)

    n_passengers = int(raw[0]["N_passengers"]) if raw else 300
    summary = build_fleet_summary(agg, args.target_completion, args.max_backlog_fraction, n_passengers)
    write_csv(outdir/"fleet_summary.csv", summary)
    rec = choose_recommendation(summary)
    (outdir/"recommendation.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    write_recommendation(outdir/"recommendation.txt", summary, rec)

    plot_heatmap(agg, "completion_rate_min", "Worst-phase completion", "Completion rate", outdir/"completion_heatmap.png")
    plot_heatmap(agg, "AWT_mean", "Mean AWT", "AWT (min)", outdir/"awt_heatmap.png")
    plot_heatmap(agg, "final_backlog_max", "Worst-phase final backlog", "Unfinished passengers", outdir/"backlog_heatmap.png")
    plot_curve(summary, "matched_completion_worst_phase", "Worst-phase completion", "Supply-matched static routing", outdir/"matched_completion.png")
    plot_curve(summary, "matched_backlog_worst_phase", "Worst-phase final backlog", "Supply-matched static routing", outdir/"matched_backlog.png")
    plot_curve(summary, "matched_AWT_mean", "Mean AWT (min)", "Supply-matched static routing", outdir/"matched_awt.png")

    print("\n" + "="*128)
    print("FAIRNESS SCAN COMPLETE")
    print("="*128)
    for r in summary:
        print(
            f"N={int(r['fleet_size']):>3} | alloc=({int(r['initial_v0_aircraft'])},{int(r['initial_v1_aircraft'])}) | "
            f"matched P(V0)={float(r['tested_matched_p_v0']):.2f} | "
            f"worst completion={100*float(r['matched_completion_worst_phase']):6.2f}% | "
            f"worst backlog={float(r['matched_backlog_worst_phase']):5.1f} | "
            f"AWT={float(r['matched_AWT_mean']):8.3f} | feasible={bool(r['matched_feasible'])}"
        )
    print("-"*128)
    print(json.dumps(rec, indent=2, ensure_ascii=False))
    print("-"*128)
    print(f"Fleet summary  : {outdir/'fleet_summary.csv'}")
    print(f"Recommendation : {outdir/'recommendation.txt'}")
    print(f"Errors         : {outdir/'errors.csv'}")

    if errors:
        raise SystemExit(f"{len(errors)} run(s) failed; inspect errors.csv")
    print("ALL FAIRNESS-SCAN RUNS PASSED")


if __name__ == "__main__":
    main()
