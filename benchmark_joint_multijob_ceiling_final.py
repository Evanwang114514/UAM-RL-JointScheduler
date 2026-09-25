# -*- coding: utf-8 -*-
"""
Final UAM P40 multi-job concurrency ceiling benchmark.

Each PPO is scientifically unchanged:
- J1 + R_TDM
- 40 env x 512 n_steps
- batch 4096
- rollout 20,480
- FULL info
- CUDA
- no ATT eval / checkpoint / model save

Base sweep:
B3 = 3 concurrent P40 jobs
B4 = 4 concurrent P40 jobs
B5 = 5 concurrent P40 jobs

B6 = 6 concurrent P40 jobs, run automatically only if:
- B5 aggregate wall SPS improves over B4 by at least 5%, and
- B5 CPU p95 is below 95%.

Default: 5 PPO updates per worker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
import benchmark_joint_multijob_gpu_1p8m as base

ROOT = Path(__file__).resolve().parent
DEFAULT_UPDATES = 5
MIN_GAIN = 0.05
CPU_P95_LIMIT = 95.0

BASE_CANDIDATES: List[Dict[str, Any]] = [
    {"id": "B3", "jobs": 3, "start_delays": [0.0, 0.0, 0.0], "description": "3x P40 synchronized fresh control"},
    {"id": "B4", "jobs": 4, "start_delays": [0.0, 0.0, 0.0, 0.0], "description": "4x P40 synchronized"},
    {"id": "B5", "jobs": 5, "start_delays": [0.0, 0.0, 0.0, 0.0, 0.0], "description": "5x P40 synchronized"},
]
B6 = {"id": "B6", "jobs": 6, "start_delays": [0.0] * 6, "description": "6x P40 synchronized conditional probe"}


def ff(x, default=float("nan")):
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def gain(new, old):
    a = ff(new.get("aggregate_wall_sps"))
    b = ff(old.get("aggregate_wall_sps"))
    if not (math.isfinite(a) and math.isfinite(b) and b > 0):
        return float("nan")
    return a / b - 1.0


def parse_args():
    p = argparse.ArgumentParser(description="Final P40 concurrency ceiling benchmark")
    p.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    p.add_argument("--force-b6", action="store_true")
    p.add_argument("--cooldown", type=float, default=5.0)
    p.add_argument("--output-root", default="")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable.")
    if args.updates < 2:
        raise SystemExit("--updates must be >= 2")

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    if args.output_root:
        root = Path(args.output_root).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = ROOT / "serial_runs" / f"multijob_ceiling_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    config = {
        "purpose": "final production concurrency ceiling sweep",
        "per_job_profile": {
            "env_key": base.ENV_KEY,
            "method_id": base.METHOD_ID,
            "n_envs": base.N_ENVS,
            "n_steps": base.N_STEPS,
            "batch_size": base.BATCH_SIZE,
            "rollout": base.ROLLOUT,
        },
        "updates_per_worker": args.updates,
        "b6_trigger": {"min_gain_B5_vs_B4": MIN_GAIN, "cpu_p95_limit": CPU_P95_LIMIT},
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    base.write_json(root / "benchmark_config.json", config)

    print("\nFINAL UAM P40 MULTI-JOB CONCURRENCY CEILING SWEEP")
    print(f"GPU         : {config['gpu_name']}")
    print(f"Per PPO     : {base.N_ENVS} env x {base.N_STEPS} steps")
    print(f"Batch       : {base.BATCH_SIZE}")
    print(f"Rollout/PPO : {base.ROLLOUT:,}")
    print(f"Updates/job : {args.updates}")
    print("Base sweep  : B3, B4, B5")
    print("B6          : conditional")
    print(f"Output      : {root}", flush=True)

    stage_summaries = []
    hardware_rows = []
    worker_rows = []

    for i, c in enumerate(BASE_CANDIDATES):
        s = base.run_stage(
            candidate=c,
            root=root,
            updates=args.updates,
            global_hardware_rows=hardware_rows,
            global_worker_rows=worker_rows,
        )
        stage_summaries.append(s)
        base.write_csv(root / "hardware_samples.csv", hardware_rows)
        base.write_csv(root / "workers_summary.csv", worker_rows)
        if i + 1 < len(BASE_CANDIDATES) and args.cooldown > 0:
            time.sleep(args.cooldown)

    by_id = {x["candidate"]: x for x in stage_summaries}
    b4, b5 = by_id["B4"], by_id["B5"]
    g = gain(b5, b4)
    b5_cpu_p95 = ff(b5.get("p95_cpu_percent"), 999.0)

    run_b6 = bool(args.force_b6) or (
        math.isfinite(g) and g >= MIN_GAIN and b5_cpu_p95 < CPU_P95_LIMIT
    )

    reason = (
        f"B5 vs B4 gain={g*100:.2f}% ; B5 CPU p95={b5_cpu_p95:.1f}% ; "
        f"thresholds=>={MIN_GAIN*100:.0f}% and <{CPU_P95_LIMIT:.0f}%"
    )
    print(f"\nB6 DECISION: {'RUN' if run_b6 else 'SKIP'} | {reason}", flush=True)

    if run_b6:
        if args.cooldown > 0:
            time.sleep(args.cooldown)
        s = base.run_stage(
            candidate=B6,
            root=root,
            updates=args.updates,
            global_hardware_rows=hardware_rows,
            global_worker_rows=worker_rows,
        )
        stage_summaries.append(s)
        base.write_csv(root / "hardware_samples.csv", hardware_rows)
        base.write_csv(root / "workers_summary.csv", worker_rows)

    ordered = sorted(stage_summaries, key=lambda x: int(x["jobs_requested"]))
    prev = None
    for row in ordered:
        row["marginal_gain_vs_previous"] = float("nan") if prev is None else gain(row, prev)
        prev = row

    ranked = sorted(
        stage_summaries,
        key=lambda x: ff(x.get("aggregate_wall_sps"), -1.0),
        reverse=True,
    )
    for i, row in enumerate(ranked, 1):
        row["rank"] = i
    base.write_csv(root / "ceiling_ranking.csv", ranked)

    # Production recommendation: walk upward in concurrency while marginal gain
    # remains >=5% and CPU p95 stays below 95%.
    safe = [x for x in ordered if str(x.get("status", "")).upper() == "OK" and ff(x.get("p95_cpu_percent"), 999) < CPU_P95_LIMIT]
    if safe:
        chosen = safe[0]
        reasoning = [f"start at {chosen['candidate']}"]
        for cur in safe[1:]:
            gg = gain(cur, chosen)
            if math.isfinite(gg) and gg >= MIN_GAIN:
                chosen = cur
                reasoning.append(f"advance to {cur['candidate']} (+{gg*100:.1f}% aggregate SPS)")
            else:
                reasoning.append(f"stop before {cur['candidate']} ({gg*100:.1f}% gain <5%)")
                break
        recommendation = {
            "status": "RECOMMENDED",
            "candidate": chosen["candidate"],
            "jobs": int(chosen["jobs_requested"]),
            "aggregate_wall_sps": ff(chosen.get("aggregate_wall_sps")),
            "per_worker_median_steady_sps": ff(chosen.get("per_worker_median_steady_sps")),
            "mean_gpu_util": ff(chosen.get("mean_gpu_util")),
            "p95_gpu_util": ff(chosen.get("p95_gpu_util")),
            "mean_cpu_percent": ff(chosen.get("mean_cpu_percent")),
            "p95_cpu_percent": ff(chosen.get("p95_cpu_percent")),
            "max_gpu_mem_mb": ff(chosen.get("max_gpu_mem_mb")),
            "rule": "increase jobs only while marginal aggregate SPS gain >=5% and CPU p95 <95%",
            "reasoning": reasoning,
            "b6_triggered": run_b6,
            "b6_trigger_reason": reason,
        }
    else:
        recommendation = {
            "status": "NO_SAFE_CANDIDATE",
            "b6_triggered": run_b6,
            "b6_trigger_reason": reason,
        }

    base.write_json(root / "production_recommendation.json", recommendation)

    print("\n" + "#" * 120)
    print("FINAL CEILING RANKING")
    print("#" * 120)
    for i, row in enumerate(ranked, 1):
        mg = ff(row.get("marginal_gain_vs_previous"))
        mg_text = "n/a" if not math.isfinite(mg) else f"{mg*100:+.1f}%"
        print(
            f"{i:02d}. {row['candidate']} | jobs={row['jobs_requested']} | "
            f"agg SPS={ff(row.get('aggregate_wall_sps')):.1f} | "
            f"marginal={mg_text} | "
            f"worker SPS={ff(row.get('per_worker_median_steady_sps')):.1f} | "
            f"GPU avg/p95={ff(row.get('mean_gpu_util')):.1f}/{ff(row.get('p95_gpu_util')):.1f}% | "
            f"CPU avg/p95={ff(row.get('mean_cpu_percent')):.1f}/{ff(row.get('p95_cpu_percent')):.1f}% | "
            f"VRAM max={ff(row.get('max_gpu_mem_mb')):.0f} MB",
            flush=True,
        )

    print("\nPRODUCTION RECOMMENDATION")
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    print(f"\nRanking        : {root / 'ceiling_ranking.csv'}")
    print(f"Recommendation : {root / 'production_recommendation.json'}")
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
