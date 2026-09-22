# -*- coding: utf-8 -*-
"""
Drain one incomplete UAGMC checkpoint to 100% completion.

Default behavior:
1) Search serial_runs/uagmc_6x6_*/**/analysis/checkpoint_eval_raw.csv
2) Pick the row with the LOWEST completion_rate (< 100%)
3) Reload that exact PPO + VecNormalize checkpoint
4) Rebuild the SAME stage/method environment, but extend max_time far beyond 600
5) Keep the deterministic policy unchanged, stop new demand at the original time,
   and continue the SAME episode until every passenger finishes or hard_guard
6) Save full-completion ATT/AWT/access/flight/Jsys plus queue/aircraft traces

This script DOES NOT modify source environment files.
Put it in the repository root beside train_uagmc_6x6_800k.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import train_uagmc_6x6_800k as old


ROOT = Path(__file__).resolve().parent


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_checkpoint_path(
    value: str,
    cell_dir: Path,
    fallback_name: str,
) -> Path:
    if value:
        p = Path(value)
        if p.exists():
            return p.resolve()
        # Old CSV may contain an absolute path from the same machine but moved root.
        candidate = cell_dir / "checkpoints" / p.name
        if candidate.exists():
            return candidate.resolve()

    candidate = cell_dir / "checkpoints" / fallback_name
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot resolve checkpoint: csv={value!r}, fallback={candidate}"
    )


def discover_worst_incomplete(
    serial_root: Path,
    prefer_stage: Optional[str] = None,
    prefer_method: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Scan standard 6x6 runs only, then select the lowest-completion raw eval row.
    """
    candidates = []

    patterns = [
        "uagmc_6x6_*/**/analysis/checkpoint_eval_raw.csv",
        "**/uagmc_6x6_*/**/analysis/checkpoint_eval_raw.csv",
    ]

    files = set()
    for patt in patterns:
        files.update(serial_root.glob(patt))

    for csv_path in sorted(files):
        # Expected: <run_root>/<E>__<M>/analysis/checkpoint_eval_raw.csv
        cell_dir = csv_path.parent.parent
        cell_name = cell_dir.name

        if "__" not in cell_name:
            continue

        stage_from_dir, method_from_dir = cell_name.split("__", 1)

        if prefer_stage and stage_from_dir.upper() != prefer_stage.upper():
            continue
        if prefer_method and method_from_dir.upper() != prefer_method.upper():
            continue

        rows = read_csv(csv_path)
        for row in rows:
            completion = fnum(row.get("completion_rate"))
            if not math.isfinite(completion) or completion >= 0.999999:
                continue

            stage = str(row.get("stage") or stage_from_dir).upper()
            method = str(row.get("method") or method_from_dir).upper()
            topology = str(row.get("topology") or "T2").upper()
            step = int(fnum(row.get("train_step"), 0))
            eval_seed = int(fnum(row.get("eval_seed"), 123))

            if stage not in old.STAGES:
                continue
            if method not in old.METHODS:
                continue
            if step <= 0:
                continue

            try:
                model_path = resolve_checkpoint_path(
                    str(row.get("model_path") or ""),
                    cell_dir,
                    f"uam_ppo_{step}_steps.zip",
                )
                vec_path = resolve_checkpoint_path(
                    str(row.get("vec_path") or ""),
                    cell_dir,
                    f"uam_ppo_vecnormalize_{step}_steps.pkl",
                )
            except FileNotFoundError:
                continue

            candidates.append(
                {
                    "completion_rate": completion,
                    "stage": stage,
                    "method": method,
                    "topology": topology,
                    "train_step": step,
                    "eval_seed": eval_seed,
                    "model_path": model_path,
                    "vec_path": vec_path,
                    "cell_dir": cell_dir.resolve(),
                    "source_csv": csv_path.resolve(),
                    "original_ATT": fnum(row.get("ATT")),
                    "original_AWT": fnum(row.get("AWT")),
                    "original_Jsys_per_passenger": fnum(
                        row.get("system_person_minutes_per_passenger")
                    ),
                    "original_N": int(fnum(row.get("N"), 0)),
                    "original_N_finished": int(fnum(row.get("N_finished"), 0)),
                }
            )

    if not candidates:
        raise FileNotFoundError(
            "No incomplete 6x6 checkpoint was found under "
            f"{serial_root}. Use --model/--vec/--stage/--method for manual mode."
        )

    candidates.sort(
        key=lambda x: (
            x["completion_rate"],
            -x["train_step"],
            str(x["source_csv"]),
        )
    )
    return candidates[0]


def aircraft_state_snapshot(scenario: Any) -> Dict[str, Any]:
    global_state = Counter()
    per_vp = defaultdict(Counter)
    target_counts = Counter()

    all_evtols = getattr(scenario, "_all_evtols", {}) or {}
    for evtol in all_evtols.values():
        state = getattr(evtol, "state", None)
        state_name = getattr(state, "name", str(state))
        state_name = str(state_name)

        cur = getattr(evtol, "current_vertiport_id", None)
        cur = "None" if cur is None else str(cur)

        tar = getattr(evtol, "target_vertiport_id", None)
        tar = "None" if tar is None else str(tar)

        global_state[state_name] += 1
        per_vp[cur][state_name] += 1
        target_counts[tar] += 1

    out: Dict[str, Any] = {
        "aircraft_total": len(all_evtols),
    }

    for state_name, count in sorted(global_state.items()):
        out[f"aircraft_state_{state_name}"] = count

    for vid, ctr in sorted(per_vp.items()):
        out[f"aircraft_at_V{vid}_total"] = sum(ctr.values())
        for state_name, count in sorted(ctr.items()):
            out[f"aircraft_at_V{vid}_{state_name}"] = count

    for vid, count in sorted(target_counts.items()):
        out[f"aircraft_target_V{vid}"] = count

    return out


def completion_counts(scenario: Any) -> tuple[int, int]:
    snap = old.mx.snapshot_episode(scenario)
    rows = snap.get("rows", []) or []
    n = int(snap.get("n_persons", len(rows)))
    nf = sum(
        1
        for r in rows
        if bool(r.get("finished"))
        and math.isfinite(fnum(r.get("travel_time")))
    )
    return n, nf


def max_end_time(scenario: Any) -> float:
    vals = []
    records = getattr(scenario, "person_travel_records", {}) or {}
    for recs in records.values():
        if not recs:
            continue
        end = recs[-1].get("end_time")
        if end is not None:
            v = fnum(end)
            if math.isfinite(v):
                vals.append(v)
    return max(vals) if vals else float("nan")


def make_trace_row(scenario: Any) -> Dict[str, Any]:
    snap = old.mx.snapshot_episode(scenario)
    rows = snap.get("rows", []) or []
    n = int(snap.get("n_persons", len(rows)))
    nf = sum(
        1
        for r in rows
        if bool(r.get("finished"))
        and math.isfinite(fnum(r.get("travel_time")))
    )

    row: Dict[str, Any] = {
        "time": int(getattr(scenario, "time", -1)),
        "N": n,
        "N_finished": nf,
        "N_unfinished": n - nf,
        "completion_rate": (nf / n if n else float("nan")),
        "active_system_count": int(old.mx.active_system_count(scenario)),
    }

    for vid, q in sorted((snap.get("queues", {}) or {}).items()):
        row[f"queue_V{vid}"] = int(q)

    stage_stats = snap.get("stage_stats", {}) or {}
    for k in (
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
        row[k] = stage_stats.get(k, 0)

    row.update(aircraft_state_snapshot(scenario))
    return row


def read_cell_config(cell_dir: Path) -> Dict[str, Any]:
    manifest = load_json(cell_dir / "run_manifest.json")
    return {
        "fleet_size": int(
            manifest.get("fleet_size_E2_E6", getattr(old, "FLEET_SIZE", 40))
        ),
        "future_horizon": float(
            manifest.get("future_horizon_min", getattr(old, "FUTURE_HORIZON_MIN", 30.0))
        ),
        "max_events": int(
            manifest.get(
                "max_events_per_type_per_candidate",
                getattr(old, "MAX_EVENTS_PER_TYPE", 8),
            )
        ),
        "pad_separation": float(
            manifest.get("pad_separation_min", getattr(old, "PAD_SEPARATION_MIN", 0.5))
        ),
        "charger_capacity": int(
            manifest.get("charger_capacity_E6", getattr(old, "CHARGER_CAPACITY", 2))
        ),
    }


def run_drain(
    *,
    stage: str,
    method: str,
    topology: str,
    model_path: Path,
    vec_path: Path,
    cell_dir: Path,
    eval_seed: int,
    hard_guard: int,
    log_interval: int,
    output_dir: Path,
) -> Dict[str, Any]:
    cfg = read_cell_config(cell_dir)

    seed_all(eval_seed)

    # Key idea:
    # Rebuild the same experiment environment but give it a much larger max_time.
    # Passenger generation still stops at Scenario.passenger_generation_end_time
    # (default 300 in current source), so after that this is a pure drain phase.
    raw = old.build_eval_raw_env(
        stage=stage,
        method=method,
        topology=topology,
        fleet_size=cfg["fleet_size"],
        run_dir=output_dir / "_eval_monitor",
        future_horizon=cfg["future_horizon"],
        max_events=cfg["max_events"],
        pad_separation=cfg["pad_separation"],
        charger_capacity=cfg["charger_capacity"],
        max_time=int(hard_guard + 200),
    )

    env = VecNormalize.load(str(vec_path), raw)
    env.training = False
    env.norm_reward = False

    action_counts = Counter()
    prob_rows: List[np.ndarray] = []
    trace_rows: List[Dict[str, Any]] = []
    reward_sum = 0.0
    system_person_minutes = 0.0
    episode_steps = 0
    completed = False
    reached_guard = False

    try:
        model = PPO.load(str(model_path), env=env, device="cpu")
        model.policy.set_training_mode(False)

        try:
            env.seed(int(eval_seed))
        except Exception:
            pass

        obs = env.reset()

        # Record t=0 state.
        scenario = old.mx.find_scenario(env)
        trace_rows.append(make_trace_row(scenario))

        while True:
            scenario = old.mx.find_scenario(env)
            now = int(getattr(scenario, "time", episode_steps))
            n, nf = completion_counts(scenario)

            generation_end = int(
                getattr(scenario, "passenger_generation_end_time", 300)
            )

            # All loaded passengers must finish, and demand generation must be over.
            if now > generation_end and n > 0 and nf == n:
                completed = True
                break

            if now >= hard_guard:
                reached_guard = True
                break

            system_person_minutes += old.mx.active_system_count(scenario)

            probs = old.mx.policy_probs(model, obs)
            action, _ = model.predict(obs, deterministic=True)
            ai = int(np.asarray(action).reshape(-1)[0])

            action_counts[ai] += 1
            prob_rows.append(np.asarray(probs, dtype=float).copy())

            obs, reward, done, infos = env.step(action)
            reward_sum += fnum(np.asarray(reward).reshape(-1)[0], 0.0)
            episode_steps += 1

            scenario = old.mx.find_scenario(env)
            now_after = int(getattr(scenario, "time", episode_steps))

            if (
                now_after == 600
                or now_after % max(1, int(log_interval)) == 0
            ):
                row = make_trace_row(scenario)
                trace_rows.append(row)
                print(
                    f"[t={row['time']:>5}] "
                    f"finished={row['N_finished']}/{row['N']} "
                    f"unfinished={row['N_unfinished']} "
                    f"active={row['active_system_count']} "
                    + " ".join(
                        f"{k}={v}"
                        for k, v in row.items()
                        if str(k).startswith("queue_V")
                    ),
                    flush=True,
                )

            # We intentionally should never hit VecEnv done before our own guard.
            if bool(np.asarray(done).reshape(-1)[0]):
                raise RuntimeError(
                    "Environment ended before manual completion/guard. "
                    "Increase hard_guard or inspect max_time propagation."
                )

        scenario = old.mx.find_scenario(env)
        final_snapshot = old.mx.snapshot_episode(scenario)

        # Always append exact final state.
        final_trace = make_trace_row(scenario)
        if not trace_rows or trace_rows[-1].get("time") != final_trace.get("time"):
            trace_rows.append(final_trace)

        metrics = old.mx.metrics_from_terminal_snapshot(
            snapshot=final_snapshot,
            action_counts=action_counts,
            prob_rows=prob_rows,
            system_person_minutes=system_person_minutes,
            reward_sum=reward_sum,
            episode_steps=episode_steps,
        )

        metrics.update(
            {
                "stage": stage,
                "method": method,
                "topology": topology,
                "eval_seed": int(eval_seed),
                "model_path": str(model_path),
                "vec_path": str(vec_path),
                "completed_all_passengers": bool(completed),
                "reached_hard_guard": bool(reached_guard),
                "hard_guard": int(hard_guard),
                "final_scenario_time": int(getattr(scenario, "time", -1)),
                "last_passenger_end_time": max_end_time(scenario),
                "demand_generation_end_time": int(
                    getattr(scenario, "passenger_generation_end_time", 300)
                ),
            }
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "drain_trace.csv", trace_rows)
        (output_dir / "final_metrics.json").write_text(
            json.dumps(jsonable(metrics), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return metrics

    finally:
        try:
            env.close()
        except Exception:
            pass
        old.core.restore_process_patches()


def parse_args():
    p = argparse.ArgumentParser(
        description="Drain one incomplete 6x6 UAGMC checkpoint to completion."
    )
    p.add_argument(
        "--serial-root",
        default="serial_runs",
        help="Where existing training runs live.",
    )
    p.add_argument("--stage", default=None, help="Optional auto-scan filter, e.g. E0")
    p.add_argument("--method", default=None, help="Optional auto-scan filter, e.g. M1")

    # Manual mode. If --model is given, all required metadata must be provided.
    p.add_argument("--model", default=None)
    p.add_argument("--vec", default=None)
    p.add_argument("--cell-dir", default=None)
    p.add_argument("--topology", default=None, choices=["T2", "T3"])
    p.add_argument("--eval-seed", type=int, default=None)

    p.add_argument("--hard-guard", type=int, default=10000)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--output-root", default="diagnostics")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    serial_root = Path(args.serial_root)
    if not serial_root.is_absolute():
        serial_root = (ROOT / serial_root).resolve()

    manual = args.model is not None

    if manual:
        if not all([args.vec, args.stage, args.method, args.topology, args.cell_dir]):
            raise ValueError(
                "Manual mode requires --model --vec --stage --method "
                "--topology --cell-dir"
            )

        model_path = Path(args.model)
        vec_path = Path(args.vec)
        cell_dir = Path(args.cell_dir)

        if not model_path.is_absolute():
            model_path = (ROOT / model_path).resolve()
        if not vec_path.is_absolute():
            vec_path = (ROOT / vec_path).resolve()
        if not cell_dir.is_absolute():
            cell_dir = (ROOT / cell_dir).resolve()

        chosen = {
            "completion_rate": float("nan"),
            "stage": args.stage.upper(),
            "method": args.method.upper(),
            "topology": args.topology.upper(),
            "train_step": -1,
            "eval_seed": int(args.eval_seed if args.eval_seed is not None else 123),
            "model_path": model_path,
            "vec_path": vec_path,
            "cell_dir": cell_dir,
            "source_csv": None,
            "original_ATT": float("nan"),
            "original_AWT": float("nan"),
            "original_Jsys_per_passenger": float("nan"),
            "original_N": 0,
            "original_N_finished": 0,
        }
    else:
        chosen = discover_worst_incomplete(
            serial_root=serial_root,
            prefer_stage=args.stage,
            prefer_method=args.method,
        )
        if args.eval_seed is not None:
            chosen["eval_seed"] = int(args.eval_seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()

    name = (
        f"drain_{chosen['stage']}_{chosen['method']}_"
        f"{chosen['train_step'] if chosen['train_step'] >= 0 else 'manual'}_"
        f"seed{chosen['eval_seed']}_{stamp}"
    )
    output_dir = output_root / name

    print("=" * 110)
    print("DRAIN-TO-COMPLETION DIAGNOSTIC")
    print("=" * 110)
    print(f"stage/method    : {chosen['stage']}/{chosen['method']}")
    print(f"topology        : {chosen['topology']}")
    print(f"train step      : {chosen['train_step']}")
    print(f"eval seed       : {chosen['eval_seed']}")
    print(f"model           : {chosen['model_path']}")
    print(f"vecnormalize    : {chosen['vec_path']}")
    print(f"original finish : {chosen['original_N_finished']}/{chosen['original_N']}")
    print(f"original comp   : {chosen['completion_rate']}")
    print(f"original ATT    : {chosen['original_ATT']}")
    print(f"original Jsys/N : {chosen['original_Jsys_per_passenger']}")
    print(f"hard guard      : {args.hard_guard}")
    print(f"output          : {output_dir}")
    print("=" * 110)

    metrics = run_drain(
        stage=chosen["stage"],
        method=chosen["method"],
        topology=chosen["topology"],
        model_path=Path(chosen["model_path"]),
        vec_path=Path(chosen["vec_path"]),
        cell_dir=Path(chosen["cell_dir"]),
        eval_seed=int(chosen["eval_seed"]),
        hard_guard=int(args.hard_guard),
        log_interval=int(args.log_interval),
        output_dir=output_dir,
    )

    summary = {
        "selected_checkpoint": chosen,
        "full_drain_metrics": metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("FINAL DRAIN RESULT")
    print("=" * 110)
    print(
        f"completion      : {metrics.get('N_finished')}/{metrics.get('N')} "
        f"({100.0*fnum(metrics.get('completion_rate'), 0.0):.2f}%)"
    )
    print(f"ATT             : {fnum(metrics.get('ATT')):.4f}")
    print(f"AWT             : {fnum(metrics.get('AWT')):.4f}")
    print(f"AGT/access      : {fnum(metrics.get('AGT_access')):.4f}")
    print(f"AFT             : {fnum(metrics.get('AFT')):.4f}")
    print(
        f"Jsys/N          : "
        f"{fnum(metrics.get('system_person_minutes_per_passenger')):.4f}"
    )
    print(f"final sim time  : {metrics.get('final_scenario_time')}")
    print(f"last pax end    : {metrics.get('last_passenger_end_time')}")
    print(f"hit hard guard  : {metrics.get('reached_hard_guard')}")
    print(f"trace           : {output_dir / 'drain_trace.csv'}")
    print(f"metrics         : {output_dir / 'final_metrics.json'}")
    print("=" * 110)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
