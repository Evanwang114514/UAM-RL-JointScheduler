# -*- coding: utf-8 -*-
"""
FORMAL25 exact analytical/rule baselines for the 5 passenger-only environments
used by train_uagmc_45x800k_formal_JOINTFIX.py.

Rows (exact FORMAL45 physics)
-----------------------------
E0 : legacy replenishment + single-passenger service
E1 : fixed conserved fleet + LQ reposition + single-passenger service
E2 : E1 + 3-min turnaround
E4 : E2 + finite charging (2 chargers/vertiport), NO finite TLOF calendar
E5 : E4 + finite TLOF/pad calendar, pad separation = 0.25 min

Baselines
---------
SPF, STTF, QTTI2, CSM, ECTF, MPTC

Design
------
* Reuses the EXACT physical row installer from the completed FORMAL45 runner.
* T2 only: passenger action set is {V0, V1}; destination is V2.
* No PPO / no learning; CPU rollout only.
* Default evaluation seeds: 123,124,125.
* Saves raw_results.csv, aggregate_results.csv, best_per_stage.csv,
  summary.txt, manifest.json, and per-cell decision traces.
* Resume-safe: an existing <stage>__<method>__seed<seed>/result.json is reused.

Important
---------
The scoring implementation is inherited from the previously validated
run_uagmc_6env_6baseline_36runs.py.  Only the PHYSICAL ENVIRONMENT is replaced
by the exact FORMAL25 ladder.  A small compatibility patch makes E0 single-pax
for heuristic capacity calculations and makes E4 finite-charge-aware while
keeping TLOF disabled.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Existing project modules.  Keep this file beside them in UAGMC-main.
import run_uagmc_6env_6baseline_36runs as legacy
import train_uagmc_45x800k_formal_JOINTFIX as formal


ROOT = Path(__file__).resolve().parent
STAGES = ("E0", "E1", "E2", "E4", "E5")
METHODS = ("SPF", "STTF", "QTTI2", "CSM", "ECTF", "MPTC")
DEFAULT_SEEDS = (123, 124, 125)

# How the old analytical score functions should interpret each new physical row.
# E4 maps to old E6 scoring semantics to expose finite-charging release times;
# pad use is independently overridden below, so E4 does NOT gain fake TLOF.
SCORE_STAGE = {
    "E0": "E0",  # legacy replenishment; capacity is patched to exactly 1
    "E1": "E3",  # single-pax fixed fleet
    "E2": "E4",  # + turnaround
    "E4": "E6",  # + finite charge, but pad explicitly OFF
    "E5": "E6",  # + finite charge + pad
}

ROW_DESCRIPTION = {
    "E0": "legacy replenishment + single-pax service",
    "E1": "fixed conserved fleet + LQ reposition + single-pax service",
    "E2": "E1 + 3-min turnaround",
    "E4": "E2 + finite charging (2/vertiport), no finite TLOF",
    "E5": "E4 + finite TLOF, pad separation 0.25 min",
}

METHOD_DESCRIPTION = {
    "SPF": "shortest focal passenger ground-access time",
    "STTF": "access + decision-time snapshot queue/supply wait proxy",
    "QTTI2": "arrival-aligned residual queue/service-rate index using committed events",
    "CSM": "FIFO committed-passenger / committed-supply matching",
    "ECTF": "CSM + service-flight time + currently-known TLOF feasibility",
    "MPTC": "ECTF + marginal future waiting imposed on already-known passengers",
}


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(np.asarray(x).reshape(-1)[0])
        return y if math.isfinite(y) else default
    except Exception:
        return default


def parse_names(text: str, allowed: Sequence[str]) -> List[str]:
    vals = [x.strip().upper() for x in str(text).split(",") if x.strip()]
    bad = [x for x in vals if x not in allowed]
    if bad:
        raise ValueError(f"unsupported values={bad}; allowed={list(allowed)}")
    if not vals:
        raise ValueError("empty selection")
    return vals


def parse_ints(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty seed selection")
    return vals


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(formal.mx.jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                val = row.get(key, "")
                if isinstance(val, (dict, list, tuple, np.ndarray)):
                    val = json.dumps(formal.mx.jsonable(val), ensure_ascii=False)
                out[key] = val
            w.writerow(out)


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        formal.seed_all(int(seed))
    except Exception:
        pass


@contextlib.contextmanager
def heuristic_semantics(row: str):
    """Temporarily align OLD scoring helpers to the NEW FORMAL25 row semantics.

    This changes scoring interpretation only.  Physical dynamics always come
    from formal.configure_row_process().
    """
    row = str(row).upper()
    rules = legacy.old

    saved: List[Tuple[Any, str, Any]] = []

    def patch(obj: Any, name: str, value: Any) -> None:
        if hasattr(obj, name):
            saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

    # E0 in the formal ladder is legacy fleet mechanics but EXACTLY one passenger
    # per service flight.  Old E0 scoring historically assumed source capacity.
    if hasattr(rules, "aircraft_capacity"):
        orig_capacity = getattr(rules, "aircraft_capacity")

        def capacity(scenario: Any, stage: str) -> int:
            if row == "E0":
                return 1
            return int(orig_capacity(scenario, stage))

        patch(rules, "aircraft_capacity", capacity)

    # CSM/ECTF/MPTC use these globals in the 6-baseline runner.
    patch(legacy, "stage_has_pad", lambda _stage: bool(formal.row_uses_pad(row)))
    patch(legacy, "pad_separation", lambda: float(formal.PAD_SEPARATION_MIN))

    # If the older SPF/STTF/QTTI2 helper exposes a pad predicate, make it match
    # the exact row too.  This is intentionally best-effort for compatibility.
    for name in ("stage_has_pad", "uses_pad", "has_pad"):
        if hasattr(rules, name):
            patch(rules, name, lambda _stage, _row=row: bool(formal.row_uses_pad(_row)))

    try:
        yield
    finally:
        for obj, name, original in reversed(saved):
            setattr(obj, name, original)


def make_exact_env(
    *,
    row: str,
    seed: int,
    run_dir: Path,
    fleet_size: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
):
    """Build the same passenger-only physical environment used in FORMAL45."""
    row = str(row).upper()
    if row not in STAGES:
        raise ValueError(row)

    formal.configure_row_process(
        row,
        pad_separation=float(pad_separation),
        charger_capacity=int(charger_capacity),
    )

    fleet_mode = "legacy_replenish" if row == "E0" else "conserved_closed_loop"
    env = formal.core.make_env(
        max_time=int(max_time),
        log_dir=run_dir / "monitor",
        env_index=int(seed),
        person_spawn_file=str(formal.TRAIN_FILE),
        candidate_from_vertiports=list(formal.CANDIDATES),
        to_vertiport=int(formal.DESTINATION),
        enable_logger=False,
        fleet_mode=fleet_mode,
        fleet_size=(None if row == "E0" else int(fleet_size)),
        fleet_assertions=(row != "E0"),
    )()

    scenario = formal.mx.find_scenario(env)
    formal._attach_stage_metadata(scenario, row)

    # Same terminal-snapshot wrapper used by FORMAL45.  The wrapper's stage label
    # is bookkeeping only; actual physics is already installed above.
    env = formal.core.ExperimentWrapper(
        env,
        stage=("E0" if row == "E0" else "E4"),
        topology="T2",
        fleet_size=int(fleet_size),
    )
    return env


def choose_action(
    env: Any,
    *,
    row: str,
    method: str,
    charger_capacity: int,
    unknown_penalty: float,
) -> Tuple[int, Dict[str, Any]]:
    """Use the validated six-baseline scorer on the exact formal row."""
    score_stage = SCORE_STAGE[str(row).upper()]
    with heuristic_semantics(row):
        return legacy.choose_action(
            env,
            score_stage,
            "T2",
            method,
            int(charger_capacity),
            float(unknown_penalty),
        )


def flatten_decision_row(row: Dict[str, Any]) -> Dict[str, Any]:
    d = row.get("detail", {}) or {}
    out: Dict[str, Any] = {
        "sim_time": row.get("sim_time"),
        "action": row.get("action"),
        "method": row.get("method"),
        "chosen_vid": d.get("chosen_vid"),
    }
    for vid, score in (d.get("scores") or {}).items():
        out[f"score_v{vid}"] = score
    out["details_json"] = json.dumps(d.get("details") or {}, ensure_ascii=False)
    return out


def run_one(
    *,
    row: str,
    method: str,
    seed: int,
    fleet_size: int,
    pad_separation: float,
    charger_capacity: int,
    max_time: int,
    unknown_penalty: float,
    run_dir: Path,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    result_file = run_dir / "result.json"
    if result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))

    seed_all(seed)
    env = make_exact_env(
        row=row,
        seed=seed,
        run_dir=run_dir,
        fleet_size=fleet_size,
        pad_separation=pad_separation,
        charger_capacity=charger_capacity,
        max_time=max_time,
    )

    try:
        try:
            env.reset(seed=int(seed))
        except TypeError:
            env.reset()

        done = False
        steps = 0
        reward_sum = 0.0
        action_counts: Counter = Counter()
        system_person_minutes = 0.0
        terminal_snapshot = None
        decision_rows: List[Dict[str, Any]] = []

        while not done:
            scenario = legacy.old.find_scenario(env)
            system_person_minutes += formal.core.mx.active_system_count(scenario)

            action, detail = choose_action(
                env,
                row=row,
                method=method,
                charger_capacity=charger_capacity,
                unknown_penalty=unknown_penalty,
            )
            action = int(action)
            action_counts[action] += 1

            if not detail.get("idle_decision", False):
                decision_rows.append(
                    {
                        "sim_time": int(getattr(scenario, "time", steps)),
                        "action": action,
                        "method": method,
                        "detail": detail,
                    }
                )

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
                raise RuntimeError(
                    f"episode exceeded guard: row={row}, method={method}, seed={seed}"
                )

        if terminal_snapshot is None:
            terminal_snapshot = formal.core.mx.snapshot_episode(
                legacy.old.find_scenario(env)
            )

        metrics = formal.core.mx.metrics_from_terminal_snapshot(
            snapshot=terminal_snapshot,
            action_counts=action_counts,
            prob_rows=[],
            system_person_minutes=system_person_minutes,
            reward_sum=reward_sum,
            episode_steps=steps,
        )
        metrics.update(
            {
                "stage": row,
                "row_name": formal.ROW_NAMES.get(row, row),
                "physical_description": ROW_DESCRIPTION[row],
                "topology": "T2",
                "method": method,
                "eval_seed": int(seed),
                "score_stage_compat": SCORE_STAGE[row],
                "unknown_event_penalty_min": float(unknown_penalty),
                "pad_separation_min": float(pad_separation),
                "charger_capacity": int(charger_capacity),
                "fleet_size": (None if row == "E0" else int(fleet_size)),
            }
        )

        write_csv(
            run_dir / f"{method.lower()}_decisions.csv",
            [flatten_decision_row(r) for r in decision_rows],
        )
        write_json(result_file, metrics)
        return metrics

    finally:
        try:
            env.close()
        except Exception:
            pass
        formal.core.restore_process_patches()


def aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = [
        "ATT",
        "AWT",
        "AGT_access",
        "AFT",
        "completion_rate",
        "backlog",
        "system_person_minutes_per_passenger",
        "travel_p90",
    ]
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((str(r["stage"]), str(r["method"])), []).append(r)

    out: List[Dict[str, Any]] = []
    for (stage, method), rs in groups.items():
        row: Dict[str, Any] = {
            "stage": stage,
            "row_name": formal.ROW_NAMES.get(stage, stage),
            "method": method,
            "n_eval_seeds": len(rs),
        }
        for metric in metrics:
            vals = [fnum(r.get(metric)) for r in rs]
            vals = [v for v in vals if math.isfinite(v)]
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        for a in range(2):
            vals = [fnum(r.get(f"action_{a}_share")) for r in rs]
            vals = [v for v in vals if math.isfinite(v)]
            row[f"action_{a}_share_mean"] = float(np.mean(vals)) if vals else float("nan")
        out.append(row)

    order_s = {s: i for i, s in enumerate(STAGES)}
    order_m = {m: i for i, m in enumerate(METHODS)}
    out.sort(key=lambda r: (order_s[r["stage"]], order_m[r["method"]]))
    return out


def strongest_rows(agg: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Save both best-ATT and best-censor-safe-Jsys baselines per environment."""
    out: List[Dict[str, Any]] = []
    for stage in STAGES:
        rs = [r for r in agg if r.get("stage") == stage]
        if not rs:
            continue
        best_att = min(rs, key=lambda r: fnum(r.get("ATT_mean"), float("inf")))
        best_j = min(
            rs,
            key=lambda r: fnum(
                r.get("system_person_minutes_per_passenger_mean"), float("inf")
            ),
        )
        out.append(
            {
                "stage": stage,
                "row_name": formal.ROW_NAMES.get(stage, stage),
                "best_ATT_method": best_att["method"],
                "best_ATT": best_att.get("ATT_mean"),
                "best_ATT_completion": best_att.get("completion_rate_mean"),
                "best_ATT_JsysN": best_att.get(
                    "system_person_minutes_per_passenger_mean"
                ),
                "best_Jsys_method": best_j["method"],
                "best_JsysN": best_j.get(
                    "system_person_minutes_per_passenger_mean"
                ),
                "best_Jsys_ATT": best_j.get("ATT_mean"),
                "best_Jsys_completion": best_j.get("completion_rate_mean"),
                "recommended_comparison_metric": (
                    "ATT"
                    if min(fnum(r.get("completion_rate_mean"), 0.0) for r in rs)
                    >= 0.98
                    else "Jsys/N (ATT is right-censored)"
                ),
            }
        )
    return out


def write_summary(root: Path, agg: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "FORMAL25 EXACT ANALYTICAL BASELINE SUMMARY",
        "=" * 118,
        "stage | method | ATT(mean±sd) | completion | Jsys/N(mean±sd)",
    ]
    for stage in STAGES:
        for r in [x for x in agg if x.get("stage") == stage]:
            lines.append(
                f"{stage:>2} | {str(r['method']):<5} | "
                f"{fnum(r.get('ATT_mean')):8.3f} ± {fnum(r.get('ATT_std'),0):6.3f} | "
                f"{100*fnum(r.get('completion_rate_mean'),0):7.2f}% | "
                f"{fnum(r.get('system_person_minutes_per_passenger_mean')):8.3f} ± "
                f"{fnum(r.get('system_person_minutes_per_passenger_std'),0):6.3f}"
            )
        lines.append("-")

    lines.append("")
    lines.append("STRONGEST PER ENVIRONMENT")
    lines.append("=" * 118)
    for r in strongest_rows(agg):
        lines.append(
            f"{r['stage']}: best ATT={r['best_ATT_method']} {fnum(r['best_ATT']):.3f}; "
            f"best Jsys/N={r['best_Jsys_method']} {fnum(r['best_JsysN']):.3f}; "
            f"primary={r['recommended_comparison_metric']}"
        )
    (root / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exact FORMAL25 E0/E1/E2/E4/E5 x 6 analytical baselines"
    )
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    p.add_argument("--fleet-size", type=int, default=formal.FLEET_SIZE)
    p.add_argument("--pad-separation", type=float, default=formal.PAD_SEPARATION_MIN)
    p.add_argument("--charger-capacity", type=int, default=formal.CHARGER_CAPACITY)
    p.add_argument("--max-time", type=int, default=formal.MAX_TIME)
    p.add_argument(
        "--unknown-event-penalty",
        type=float,
        default=float(legacy.DEFAULT_UNKNOWN_EVENT_PENALTY_MIN),
    )
    p.add_argument("--output-root", default=None)
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record error and continue remaining cells",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stages = parse_names(args.stages, STAGES)
    methods = parse_names(args.methods, METHODS)
    seeds = parse_ints(args.seeds)

    if float(args.pad_separation) != float(formal.PAD_SEPARATION_MIN):
        print(
            f"[WARN] pad separation override={args.pad_separation}; "
            f"FORMAL45 used {formal.PAD_SEPARATION_MIN}"
        )
    if int(args.charger_capacity) != int(formal.CHARGER_CAPACITY):
        print(
            f"[WARN] charger capacity override={args.charger_capacity}; "
            f"FORMAL45 used {formal.CHARGER_CAPACITY}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (
            ROOT
            / "serial_runs"
            / f"formal25_6analytic_T2_{stamp}"
        ).resolve()
    )
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "FORMAL25_EXACT_5ENV_6_ANALYTICAL_BASELINES",
        "created": datetime.now().isoformat(timespec="seconds"),
        "stages": stages,
        "methods": methods,
        "seeds": seeds,
        "topology": "T2",
        "candidate_actions": list(formal.CANDIDATES),
        "destination": int(formal.DESTINATION),
        "fleet_size": int(args.fleet_size),
        "pad_separation_min": float(args.pad_separation),
        "charger_capacity": int(args.charger_capacity),
        "max_time": int(args.max_time),
        "passenger_trace": str(formal.TRAIN_FILE),
        "unknown_event_penalty_min": float(args.unknown_event_penalty),
        "row_description": ROW_DESCRIPTION,
        "method_description": METHOD_DESCRIPTION,
        "score_stage_compatibility_map": SCORE_STAGE,
        "physics_source": "train_uagmc_45x800k_formal_JOINTFIX.configure_row_process",
        "scoring_source": "run_uagmc_6env_6baseline_36runs",
        "causality": (
            "no unrevealed future passenger requests; no uncommitted future "
            "reposition actions in committed-future heuristics"
        ),
        "selection_note": (
            "best_per_stage.csv reports both lowest ATT and lowest censor-safe "
            "Jsys/N; use Jsys/N when completion is materially below 98%"
        ),
    }
    write_json(root / "manifest.json", manifest)

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    total = len(stages) * len(methods) * len(seeds)
    idx = 0

    print("=" * 132)
    print("FORMAL25 EXACT ANALYTICAL BASELINES")
    print(f"stages={stages}")
    print(f"methods={methods}")
    print(f"seeds={seeds} | total episodes={total} | CPU-only")
    print(
        f"fleet={args.fleet_size} | charger={args.charger_capacity} | "
        f"pad_sep={args.pad_separation} | max_time={args.max_time}"
    )
    print("=" * 132)

    for stage in stages:
        for method in methods:
            for seed in seeds:
                idx += 1
                run_dir = root / f"{stage}__{method}__seed{seed}"
                print(
                    f"[{idx:03d}/{total:03d}] {stage}/{method}/seed={seed}",
                    flush=True,
                )
                try:
                    row = run_one(
                        row=stage,
                        method=method,
                        seed=seed,
                        fleet_size=int(args.fleet_size),
                        pad_separation=float(args.pad_separation),
                        charger_capacity=int(args.charger_capacity),
                        max_time=int(args.max_time),
                        unknown_penalty=float(args.unknown_event_penalty),
                        run_dir=run_dir,
                    )
                    rows.append(row)
                    print(
                        f"  ATT={fnum(row.get('ATT')):.3f} | "
                        f"AWT={fnum(row.get('AWT')):.3f} | "
                        f"finish={int(row.get('N_finished',0))}/{int(row.get('N',0))} | "
                        f"Jsys/N={fnum(row.get('system_person_minutes_per_passenger')):.3f}",
                        flush=True,
                    )
                except Exception as exc:
                    err = {
                        "stage": stage,
                        "method": method,
                        "seed": int(seed),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    errors.append(err)
                    print(f"  [ERROR] {repr(exc)}", flush=True)
                    write_csv(root / "errors.csv", errors)
                    if not args.continue_on_error:
                        raise

                agg = aggregate(rows)
                write_csv(root / "raw_results.csv", rows)
                write_csv(root / "aggregate_results.csv", agg)
                write_csv(root / "best_per_stage.csv", strongest_rows(agg))
                write_summary(root, agg)

    agg = aggregate(rows)
    write_csv(root / "raw_results.csv", rows)
    write_csv(root / "aggregate_results.csv", agg)
    write_csv(root / "best_per_stage.csv", strongest_rows(agg))
    write_csv(root / "errors.csv", errors)
    write_summary(root, agg)

    write_json(
        root / "progress.json",
        {
            "finished": True,
            "episodes_requested": total,
            "episodes_ok": len(rows),
            "episodes_failed": len(errors),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    # Small upload bundle: exclude monitor files and decision traces to keep it tiny.
    try:
        import zipfile

        zip_path = root / "FORMAL25_ANALYTICAL_BASELINES_RESULTS.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name in (
                "manifest.json",
                "raw_results.csv",
                "aggregate_results.csv",
                "best_per_stage.csv",
                "errors.csv",
                "summary.txt",
                "progress.json",
            ):
                p = root / name
                if p.exists():
                    zf.write(p, arcname=name)
        print(f"UPLOAD ZIP: {zip_path}")
    except Exception as exc:
        print(f"[WARN] result ZIP creation failed: {exc!r}")

    print("\nDONE")
    print(f"Results: {root}")
    print(f"OK={len(rows)} / requested={total} | errors={len(errors)}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
