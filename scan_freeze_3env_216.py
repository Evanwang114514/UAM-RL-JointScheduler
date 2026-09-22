# -*- coding: utf-8 -*-
"""
Fast 3-environment physical-parameter calibration for Formal45-style UAM.

Retained environments:
E2 = turnaround
E4 = turnaround + finite charging
E5 = turnaround + finite charging + finite shared TLOF

Default grid:
E2: 4 turnaround x 3 charge-rate scales = 12 configs
E4: 4 turnaround x 2 charge-rate scales x 3 charger capacities = 24 configs
E5: 3 turnaround x 2 charge-rate scales x 3 charger capacities x 2 pad separations = 36 configs
72 configs x 3 analytical baselines (STTF,QTTI2,ECTF) = 216 simulations.

Run:
    python scan_freeze_3env_216.py --workers 12

Smoke:
    python scan_freeze_3env_216.py --workers 3 --smoke
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
HARD_GUARD = 10_000
DEFAULT_SEED = 123
FLEET_SIZE = 40
METHODS = ("STTF", "QTTI2", "ECTF")
UNKNOWN_PENALTY = 60.0
BASE_CHARGE_RATES = {"0": 10.64, "1": 5.32, "2": 5.32}
TARGET_ATT = {"E2": 68.0, "E4": 80.0, "E5": 92.0}


@dataclass(frozen=True)
class Config:
    row: str
    turnaround_min: float
    charge_rate_scale: float
    charger_capacity: int
    pad_separation_min: float

    @property
    def config_id(self) -> str:
        return (
            f"{self.row}_ta{self.turnaround_min:g}"
            f"_cr{self.charge_rate_scale:g}"
            f"_cc{self.charger_capacity}"
            f"_pad{self.pad_separation_min:g}"
        )


def fnum(x: Any, default=float("nan")) -> float:
    try:
        y = float(np.asarray(x).reshape(-1)[0])
        return y if math.isfinite(y) else default
    except Exception:
        return default


def mean(xs: Iterable[Any]) -> float:
    a = np.asarray([fnum(x) for x in xs], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def median(xs: Iterable[Any]) -> float:
    a = np.asarray([fnum(x) for x in xs], dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


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
            out = {}
            for k, v in r.items():
                out[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
            w.writerow(out)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_formal_module():
    for name in ("train_uagmc_45x800k_formal_JOINTFIX", "train_uagmc_45x800k_formal"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            pass
    raise ModuleNotFoundError(
        "Need train_uagmc_45x800k_formal_JOINTFIX.py or train_uagmc_45x800k_formal.py in repo root."
    )


def load_baseline_module():
    try:
        return importlib.import_module("run_uagmc_6env_6baseline_36runs")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Need run_uagmc_6env_6baseline_36runs.py in repo root."
        ) from e


def build_grid(smoke: bool = False) -> List[Config]:
    if smoke:
        return [
            Config("E2", 2.0, 1.0, 99, 0.0),
            Config("E4", 2.0, 1.25, 3, 0.0),
            Config("E5", 2.0, 1.25, 4, 0.25),
        ]

    out = []
    for ta, cr in product((1.0, 2.0, 3.0, 4.0), (0.80, 1.00, 1.25)):
        out.append(Config("E2", ta, cr, 99, 0.0))
    for ta, cr, cc in product((1.0, 2.0, 3.0, 4.0), (1.00, 1.25), (2, 3, 4)):
        out.append(Config("E4", ta, cr, cc, 0.0))
    for ta, cr, cc, pad in product((1.0, 2.0, 3.0), (1.00, 1.25), (3, 4, 5), (0.25, 0.50)):
        out.append(Config("E5", ta, cr, cc, pad))
    assert len(out) == 72
    return out


def proxy_stage(row: str) -> str:
    if row == "E2":
        return "E4"
    if row in ("E4", "E5"):
        return "E6"
    raise ValueError(row)


def set_physical_globals(formal, baseline, cfg: Config) -> None:
    import at_obj.vertiport.vertiport_spec as vp_spec
    import at_obj.evtol.evtol_builder as evtol_builder

    for vid, base_rate in BASE_CHARGE_RATES.items():
        value = float(base_rate) * float(cfg.charge_rate_scale)
        vp_spec.VERTIPORT_CHARGE_RATE[str(vid)] = value
        evtol_builder.VERTIPORT_CHARGE_RATE[str(vid)] = value

    formal.mx.TURNAROUND_DELAY_MIN = float(cfg.turnaround_min)
    formal.base.TURNAROUND_DELAY_MIN = float(cfg.turnaround_min)
    formal.core.TURNAROUND_DELAY_MIN = float(cfg.turnaround_min)
    baseline.core.TURNAROUND_DELAY_MIN = float(cfg.turnaround_min)

    for mod in (formal, formal.mx, formal.base, formal.core):
        if hasattr(mod, "MAX_TIME"):
            setattr(mod, "MAX_TIME", HARD_GUARD)


def run_one(job: Dict[str, Any]) -> Dict[str, Any]:
    cfg = Config(**job["config"])
    method = str(job["method"]).upper()
    seed = int(job["seed"])
    run_dir = Path(job["run_dir"])

    formal = load_formal_module()
    baseline = load_baseline_module()
    set_physical_globals(formal, baseline, cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    env = None
    try:
        factory = formal.make_formal_env_factory(
            row=cfg.row,
            method="M0",
            fleet_size=FLEET_SIZE,
            env_index=seed,
            run_dir=run_dir,
            future_horizon=float(getattr(formal, "FUTURE_HORIZON_MIN", 30.0)),
            max_events=int(getattr(formal, "MAX_EVENTS_PER_TYPE", 8)),
            pad_separation=float(cfg.pad_separation_min),
            charger_capacity=int(cfg.charger_capacity),
            uq_delta=float(getattr(formal, "ETA_UQ_DELTA_MIN", 2.0)),
            max_time=HARD_GUARD,
        )
        env = factory()
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

        done = False
        steps = 0
        system_person_minutes = 0.0
        reward_sum = 0.0
        action_counts = {}
        terminal_snapshot = None

        while not done:
            scenario = formal.mx.find_scenario(env)
            system_person_minutes += formal.mx.active_system_count(scenario)

            action, _ = baseline.choose_action(
                env,
                proxy_stage(cfg.row),
                "T2",
                method,
                int(cfg.charger_capacity),
                float(UNKNOWN_PENALTY),
            )
            ai = int(action)
            action_counts[ai] = action_counts.get(ai, 0) + 1

            out = env.step(ai)
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

            if steps > HARD_GUARD + 5:
                raise RuntimeError("Exceeded hard guard without termination.")

        scenario = formal.mx.find_scenario(env)
        if terminal_snapshot is None:
            terminal_snapshot = formal.mx.snapshot_episode(scenario)

        from collections import Counter
        metrics = formal.mx.metrics_from_terminal_snapshot(
            snapshot=terminal_snapshot,
            action_counts=Counter(action_counts),
            prob_rows=[],
            system_person_minutes=system_person_minutes,
            reward_sum=reward_sum,
            episode_steps=steps,
        )

        return {
            **asdict(cfg),
            "config_id": cfg.config_id,
            "method": method,
            "seed": seed,
            "ATT": fnum(metrics.get("ATT")),
            "AWT": fnum(metrics.get("AWT")),
            "AGT_access": fnum(metrics.get("AGT_access")),
            "AFT": fnum(metrics.get("AFT")),
            "travel_p90": fnum(metrics.get("travel_p90")),
            "N": int(fnum(metrics.get("N"), 0)),
            "N_finished": int(fnum(metrics.get("N_finished"), 0)),
            "completion_rate": fnum(metrics.get("completion_rate"), 0.0),
            "backlog": int(fnum(metrics.get("backlog"), 0)),
            "Jsys_per_passenger": fnum(metrics.get("system_person_minutes_per_passenger")),
            "episode_steps": int(fnum(metrics.get("episode_steps"), steps)),
            "final_time": int(getattr(scenario, "time", steps)),
            "charger_active_steps": int(fnum(metrics.get("e6_charger_active_aircraft_steps"), 0)),
            "charger_wait_steps": int(fnum(metrics.get("e6_charger_wait_aircraft_steps"), 0)),
            "max_charger_queue": int(fnum(metrics.get("e6_max_charger_queue"), 0)),
            "service_pad_blocks": int(fnum(metrics.get("service_pad_blocks"), 0)),
            "reposition_pad_blocks": int(fnum(metrics.get("reposition_pad_blocks"), 0)),
            "turnaround_starts": int(fnum(metrics.get("turnaround_starts"), 0)),
            "hard_guard_hit": bool(
                fnum(metrics.get("completion_rate"), 0.0) < 0.999999
                and int(getattr(scenario, "time", steps)) >= HARD_GUARD
            ),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            formal.core.restore_process_patches()
        except Exception:
            pass


def summarize(raw: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = {}
    for r in raw:
        groups.setdefault(str(r["config_id"]), []).append(r)

    out = []
    for cid, rs in groups.items():
        r0 = rs[0]
        atts = [fnum(r.get("ATT")) for r in rs]
        comps = [fnum(r.get("completion_rate"), 0.0) for r in rs]
        js = [fnum(r.get("Jsys_per_passenger")) for r in rs]
        row = {
            "config_id": cid,
            "row": r0["row"],
            "turnaround_min": fnum(r0["turnaround_min"]),
            "charge_rate_scale": fnum(r0["charge_rate_scale"]),
            "charger_capacity": int(fnum(r0["charger_capacity"])),
            "pad_separation_min": fnum(r0["pad_separation_min"]),
            "n_methods": len(rs),
            "all_complete": all(x >= 0.999999 for x in comps),
            "min_completion": min(comps) if comps else 0.0,
            "ATT_best": min(atts) if atts else float("nan"),
            "ATT_median": median(atts),
            "ATT_mean": mean(atts),
            "ATT_worst": max(atts) if atts else float("nan"),
            "Jsys_median": median(js),
            "final_time_max": max(int(fnum(r["final_time"], 0)) for r in rs),
            "charger_wait_mean": mean(r.get("charger_wait_steps") for r in rs),
            "max_charger_queue_max": max(int(fnum(r.get("max_charger_queue"), 0)) for r in rs),
            "pad_blocks_mean": mean(
                int(fnum(r.get("service_pad_blocks"), 0))
                + int(fnum(r.get("reposition_pad_blocks"), 0))
                for r in rs
            ),
        }
        for r in rs:
            m = str(r["method"])
            row[f"ATT_{m}"] = fnum(r.get("ATT"))
            row[f"completion_{m}"] = fnum(r.get("completion_rate"))
        out.append(row)
    return out


def activity_penalty(r: Dict[str, Any]) -> float:
    row = str(r["row"])
    p = 0.0
    if row == "E4" and fnum(r.get("charger_wait_mean"), 0.0) <= 0:
        p += 15.0
    if row == "E5":
        if fnum(r.get("charger_wait_mean"), 0.0) <= 0:
            p += 12.0
        if fnum(r.get("pad_blocks_mean"), 0.0) <= 0:
            p += 12.0
    return p


def individual_score(r: Dict[str, Any]) -> float:
    row = str(r["row"])
    att = fnum(r.get("ATT_median"), 1e9)
    score = abs(att - TARGET_ATT[row])
    if not bool(r.get("all_complete")):
        score += 10000.0 * (1.0 - fnum(r.get("min_completion"), 0.0))
    if att > 100.0:
        score += 8.0 * (att - 100.0)
    return score + activity_penalty(r)


def choose_freeze(summary: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_row = {row: [r for r in summary if str(r["row"]) == row] for row in ("E2", "E4", "E5")}
    best_trip, best_score = None, float("inf")

    for a in by_row["E2"]:
        for b in by_row["E4"]:
            for c in by_row["E5"]:
                trip = [a, b, c]
                score = sum(individual_score(x) for x in trip)
                x2, x4, x5 = [fnum(x.get("ATT_median"), 1e9) for x in trip]

                for gap in (x4 - x2, x5 - x4):
                    if gap < 3.0:
                        score += 40.0 * (3.0 - gap)
                    elif gap > 25.0:
                        score += 2.0 * (gap - 25.0)

                if max(x2, x4, x5) > 100.0:
                    score += 50.0 * (max(x2, x4, x5) - 100.0)

                if score < best_score:
                    best_score = score
                    best_trip = trip

    if best_trip is None:
        raise RuntimeError("No E2/E4/E5 triple could be selected.")
    return {str(r["row"]): dict(r) for r in best_trip}


def frozen_py_text(selected: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "# Auto-generated by scan_freeze_3env_216.py",
        "FROZEN_ENV_PARAMS = {",
    ]
    for row in ("E2", "E4", "E5"):
        r = selected[row]
        lines.extend([
            f'    "{row}": {{',
            f'        "turnaround_min": {fnum(r["turnaround_min"]):g},',
            f'        "charge_rate_scale": {fnum(r["charge_rate_scale"]):g},',
            f'        "charger_capacity": {int(fnum(r["charger_capacity"]))},',
            f'        "pad_separation_min": {fnum(r["pad_separation_min"]):g},',
            f'        "fleet_size": {FLEET_SIZE},',
            f'        "hard_guard": {HARD_GUARD},',
            f'        "calibration_ATT_median": {fnum(r["ATT_median"]):.6f},',
            "    },",
        ])
    lines.extend(["}", ""])
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--output-root", default=None)
    p.add_argument("--resume-root", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    methods = tuple(x.strip().upper() for x in args.methods.split(",") if x.strip())
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise ValueError(f"Unsupported methods={bad}; use {METHODS}")

    grid = build_grid(bool(args.smoke))
    expected = len(grid) * len(methods)

    if args.resume_root:
        root = Path(args.resume_root).expanduser().resolve()
    elif args.output_root:
        root = Path(args.output_root).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = (ROOT / "serial_runs" / f"env3_grid_{expected}runs_{stamp}").resolve()
    root.mkdir(parents=True, exist_ok=True)

    raw_path = root / "raw_results.csv"
    err_path = root / "errors.csv"
    old_rows = read_csv(raw_path)
    done_keys = {
        (r.get("config_id"), str(r.get("method", "")).upper(), int(float(r.get("seed", 0))))
        for r in old_rows if r.get("config_id")
    }

    raw = [dict(r) for r in old_rows]
    errors = [dict(r) for r in read_csv(err_path)]
    jobs = []
    for cfg in grid:
        for method in methods:
            key = (cfg.config_id, method, int(args.seed))
            if key in done_keys:
                continue
            jobs.append({
                "config": asdict(cfg),
                "method": method,
                "seed": int(args.seed),
                "run_dir": str(root / "runs" / cfg.config_id / method),
            })

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "retained_rows": ["E2", "E4", "E5"],
        "grid_configs": len(grid),
        "methods": list(methods),
        "total_simulations": expected,
        "remaining_simulations": len(jobs),
        "workers": int(args.workers),
        "fleet_size": FLEET_SIZE,
        "topology": "T2",
        "seed": int(args.seed),
        "hard_guard": HARD_GUARD,
        "base_charge_rates_kwh_per_min": BASE_CHARGE_RATES,
        "target_ATT_centers": TARGET_ATT,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 118)
    print("3-ENV PHYSICAL CALIBRATION")
    print(f"configs={len(grid)} | total simulations={expected} | remaining={len(jobs)}")
    print(f"methods={methods} | workers={args.workers}")
    print(f"output={root}")
    print("=" * 118, flush=True)

    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futs = {ex.submit(run_one, j): j for j in jobs}
            finished = 0
            for fut in as_completed(futs):
                j = futs[fut]
                finished += 1
                try:
                    row = fut.result()
                    raw.append(row)
                    write_csv(raw_path, raw)
                    print(
                        f"[{finished:03d}/{len(jobs):03d}] {row['config_id']} {row['method']} | "
                        f"ATT={fnum(row['ATT']):.2f} finish={row['N_finished']}/{row['N']} t={row['final_time']}",
                        flush=True,
                    )
                except Exception as exc:
                    cfg = Config(**j["config"])
                    er = {
                        "config_id": cfg.config_id,
                        "method": j["method"],
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    errors.append(er)
                    write_csv(err_path, errors)
                    print(f"[ERROR] {er['config_id']} {er['method']}: {er['error']}", flush=True)

    raw = read_csv(raw_path)
    summary = summarize(raw)
    for r in summary:
        r["selection_score_individual"] = individual_score(r)
    summary.sort(key=lambda r: (str(r["row"]), fnum(r["selection_score_individual"])))
    write_csv(root / "config_summary.csv", summary)

    if all(any(str(r["row"]) == row for r in summary) for row in ("E2", "E4", "E5")):
        selected = choose_freeze(summary)
        (root / "recommended_freeze.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (root / "frozen_3env_params.py").write_text(frozen_py_text(selected), encoding="utf-8")

        lines = ["TOP CANDIDATES", "=" * 110]
        for row in ("E2", "E4", "E5"):
            lines.append(f"\n[{row}]")
            candidates = [r for r in summary if str(r["row"]) == row][:10]
            for r in candidates:
                lines.append(
                    f"{r['config_id']:<34} | med={fnum(r['ATT_median']):7.2f} "
                    f"best={fnum(r['ATT_best']):7.2f} worst={fnum(r['ATT_worst']):7.2f} | "
                    f"complete={r['all_complete']} | charge_wait={fnum(r['charger_wait_mean']):8.1f} | "
                    f"pad_blocks={fnum(r['pad_blocks_mean']):7.1f}"
                )
            s = selected[row]
            lines.append(f"SELECTED -> {s['config_id']} | ATT median={fnum(s['ATT_median']):.2f}")
        (root / "top_candidates.txt").write_text("\n".join(lines), encoding="utf-8")

        print("\nRECOMMENDED FREEZE")
        for row in ("E2", "E4", "E5"):
            r = selected[row]
            print(
                f"{row}: turnaround={r['turnaround_min']} | charge_scale={r['charge_rate_scale']} | "
                f"charger_cap={r['charger_capacity']} | pad={r['pad_separation_min']} | "
                f"ATT_median={fnum(r['ATT_median']):.2f}"
            )

    print(f"\nDONE: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
