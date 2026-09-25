# -*- coding: utf-8 -*-
"""
UAM Joint PPO multi-job GPU utilization benchmark (~1.8M transitions total).

Goal
----
Keep EACH PPO scientifically unchanged at:
    40 env x 512 n_steps = 20,480 transitions/update
    batch_size = 4096
and exploit the RTX 4090D by running multiple INDEPENDENT PPO cells concurrently.

This avoids the learning-risk found when a single PPO was changed to
128/160 environments with very short per-env rollout horizons.

Default engineering sweep
-------------------------
A1  1 x P40, no stagger
A2  2 x P40, synchronized start
A3  2 x P40, 3 s learning-start stagger
A4  3 x P40, synchronized start
A5  3 x P40, 2 s learning-start stagger

Every worker:
- J1 + R_TDM
- FULL training info
- CUDA network updates
- 40 env x 512 n_steps
- batch 4096
- default 8 PPO updates = 163,840 transitions
- NO ATT evaluation
- NO checkpoint saving
- NO model saving

Default total:
(1 + 2 + 2 + 3 + 3) * 8 * 20,480 = 1,802,240 transitions.

Primary metric
--------------
aggregate_wall_sps = total transitions completed by all workers /
                     concurrent stage wall-clock time

Also reports:
- per-worker steady median SPS (updates 2..N)
- aggregate steady SPS = sum(per-worker medians)
- speedup vs A1
- parallel efficiency
- GPU utilization / memory / power / temperature
- machine CPU utilization / RAM
- rollout time vs PPO train time

Run
---
CUDA_VISIBLE_DEVICES=0 python benchmark_joint_multijob_gpu_1p8m.py

Subset / longer run
-------------------
CUDA_VISIBLE_DEVICES=0 python benchmark_joint_multijob_gpu_1p8m.py --candidates A1,A2,A4
CUDA_VISIBLE_DEVICES=0 python benchmark_joint_multijob_gpu_1p8m.py --updates 12

Outputs
-------
serial_runs/multijob_gpu_<timestamp>/
    benchmark_config.json
    multijob_ranking.csv
    workers_summary.csv
    hardware_samples.csv
    A1/
      stage_summary.json
      worker_00/
        worker_updates.csv
        worker_summary.json
        stdout.log
      ...
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Avoid BLAS/OpenMP thread-pool explosion across 40*N simulator workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

import train_uam_60m_literature_matrix_v3_1 as lit


ROOT = Path(__file__).resolve().parent

ENV_KEY = "J1"
METHOD_ID = "R_TDM"

N_ENVS = 40
N_STEPS = 512
BATCH_SIZE = 4096
ROLLOUT = N_ENVS * N_STEPS
MAX_TIME = 2500
DEFAULT_UPDATES = 8

assert ROLLOUT == 20_480

# start_delays are applied AFTER each worker has built its env/model, immediately
# before model.learn(). They intentionally de-phase PPO GPU update bursts without
# altering any PPO/environment setting.
CANDIDATES: List[Dict[str, Any]] = [
    {
        "id": "A1",
        "jobs": 1,
        "start_delays": [0.0],
        "description": "1x P40 baseline",
    },
    {
        "id": "A2",
        "jobs": 2,
        "start_delays": [0.0, 0.0],
        "description": "2x P40 synchronized",
    },
    {
        "id": "A3",
        "jobs": 2,
        "start_delays": [0.0, 3.0],
        "description": "2x P40 staggered by 3s",
    },
    {
        "id": "A4",
        "jobs": 3,
        "start_delays": [0.0, 0.0, 0.0],
        "description": "3x P40 synchronized",
    },
    {
        "id": "A5",
        "jobs": 3,
        "start_delays": [0.0, 2.0, 4.0],
        "description": "3x P40 staggered by 2s",
    },
]


# =============================================================================
# Generic helpers
# =============================================================================

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
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
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def finite_values(xs: Sequence[Any]) -> List[float]:
    out = []
    for x in xs:
        try:
            y = float(x)
            if math.isfinite(y):
                out.append(y)
        except Exception:
            pass
    return out


def safe_mean(xs: Sequence[Any]) -> float:
    vals = finite_values(xs)
    return float(np.mean(vals)) if vals else float("nan")


def safe_median(xs: Sequence[Any]) -> float:
    vals = finite_values(xs)
    return float(statistics.median(vals)) if vals else float("nan")


def safe_max(xs: Sequence[Any]) -> float:
    vals = finite_values(xs)
    return float(max(vals)) if vals else float("nan")


def safe_percentile(xs: Sequence[Any], q: float) -> float:
    vals = finite_values(xs)
    return float(np.percentile(vals, q)) if vals else float("nan")


def safe_cv(xs: Sequence[Any]) -> float:
    vals = np.asarray(finite_values(xs), dtype=float)
    if len(vals) < 2 or abs(float(vals.mean())) < 1e-12:
        return float("nan")
    return float(vals.std(ddof=0) / vals.mean())


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# Worker-side timing
# =============================================================================

class WorkerUpdateTimer:
    """Measure rollout/train/update wall time without touching PPO math."""

    def __init__(self, model: Any, out_csv: Path, worker_id: str):
        self.model = model
        self.out_csv = Path(out_csv)
        self.worker_id = str(worker_id)

        self.update_idx = 0
        self.prev_timesteps = int(model.num_timesteps)
        self.update_started = 0.0
        self.rollout_sec = 0.0
        self.rows: List[Dict[str, Any]] = []

        self._orig_collect = model.collect_rollouts
        self._orig_train = model.train
        self._patch()

    def _patch(self) -> None:
        def timed_collect(*args, **kwargs):
            self.update_idx += 1
            self.update_started = time.perf_counter()
            t0 = time.perf_counter()
            out = self._orig_collect(*args, **kwargs)
            self.rollout_sec = time.perf_counter() - t0
            return out

        def timed_train(*args, **kwargs):
            t0 = time.perf_counter()
            out = self._orig_train(*args, **kwargs)

            # Synchronize only for timing correctness. This happens after PPO train
            # and does not change the optimizer/math.
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            train_sec = time.perf_counter() - t0
            total_sec = time.perf_counter() - self.update_started

            now_steps = int(self.model.num_timesteps)
            delta_steps = now_steps - self.prev_timesteps
            self.prev_timesteps = now_steps

            row = {
                "worker_id": self.worker_id,
                "update": self.update_idx,
                "timesteps": now_steps,
                "delta_steps": delta_steps,
                "rollout_sec": float(self.rollout_sec),
                "ppo_train_sec": float(train_sec),
                "total_update_sec": float(total_sec),
                "rollout_sps": (
                    float(delta_steps / self.rollout_sec)
                    if self.rollout_sec > 0
                    else float("nan")
                ),
                "total_sps": (
                    float(delta_steps / total_sec)
                    if total_sec > 0
                    else float("nan")
                ),
                "train_fraction": (
                    float(train_sec / total_sec)
                    if total_sec > 0
                    else float("nan")
                ),
            }
            self.rows.append(row)
            write_csv(self.out_csv, self.rows)

            print(
                f"[{self.worker_id}] update={self.update_idx:02d} "
                f"rollout={self.rollout_sec:.2f}s "
                f"train={train_sec:.2f}s "
                f"total={total_sec:.2f}s "
                f"SPS={row['total_sps']:.1f}",
                flush=True,
            )
            return out

        self.model.collect_rollouts = timed_collect
        self.model.train = timed_train


def assert_cuda_model(model: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    dev = torch.device(model.device)
    if dev.type != "cuda":
        raise RuntimeError(f"Expected CUDA model, got {dev}")
    for name, p in model.policy.named_parameters():
        if p.device.type != "cuda":
            raise RuntimeError(f"Policy param {name} is on {p.device}")


def worker_main(args: argparse.Namespace) -> int:
    """
    One independent PPO training process.
    It still uses exactly P40 = 40x512 internally.
    """

    run_dir = Path(args.worker_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    worker_id = str(args.worker_id)
    seed = int(args.seed)
    updates = int(args.updates)
    start_delay = float(args.start_delay)

    profile = lit.SpeedProfile(
        name=f"MULTIJOB_P40_{worker_id}",
        n_envs=N_ENVS,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
    )
    if profile.rollout != ROLLOUT:
        raise RuntimeError(profile.rollout)

    env = None
    model = None
    timer = None

    summary_path = run_dir / "worker_summary.json"
    started_wall = time.time()

    try:
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        torch.cuda.empty_cache()
        lit.seed_all(seed)

        env = lit.build_vec_env(
            env_key=ENV_KEY,
            method_id=METHOD_ID,
            profile=profile,
            seed=seed,
            run_dir=run_dir,
            max_time=MAX_TIME,
        )

        model = lit.build_model(
            env=env,
            env_key=ENV_KEY,
            method_id=METHOD_ID,
            profile=profile,
            seed=seed,
            run_dir=run_dir,
            device="cuda",
        )
        model.verbose = 0
        assert_cuda_model(model)

        timer = WorkerUpdateTimer(
            model=model,
            out_csv=run_dir / "worker_updates.csv",
            worker_id=worker_id,
        )

        print(
            f"READY {worker_id} | seed={seed} | "
            f"P40={N_ENVS}x{N_STEPS} | batch={BATCH_SIZE} | "
            f"delay={start_delay:.1f}s | params={lit.count_params(model):,}",
            flush=True,
        )

        if start_delay > 0:
            time.sleep(start_delay)

        learn_start_wall = time.time()
        learn_start_perf = time.perf_counter()

        model.learn(
            total_timesteps=int(ROLLOUT * updates),
            callback=None,
            reset_num_timesteps=True,
            progress_bar=False,
        )

        torch.cuda.synchronize()

        learn_end_perf = time.perf_counter()
        learn_end_wall = time.time()
        learn_elapsed = learn_end_perf - learn_start_perf

        rows = list(timer.rows)
        steady = rows[1:] if len(rows) >= 2 else rows

        summary = {
            "status": "OK",
            "worker_id": worker_id,
            "seed": seed,
            "start_delay_sec": start_delay,
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "rollout": ROLLOUT,
            "updates_requested": updates,
            "updates_completed": len(rows),
            "actual_timesteps": int(model.num_timesteps),
            "process_started_wall": started_wall,
            "learn_start_wall": learn_start_wall,
            "learn_end_wall": learn_end_wall,
            "learn_elapsed_sec": float(learn_elapsed),
            "learn_sps": float(model.num_timesteps / max(learn_elapsed, 1e-9)),
            "steady_median_sps": safe_median(
                [r.get("total_sps") for r in steady]
            ),
            "steady_mean_sps": safe_mean(
                [r.get("total_sps") for r in steady]
            ),
            "steady_cv_sps": safe_cv(
                [r.get("total_sps") for r in steady]
            ),
            "median_rollout_sec": safe_median(
                [r.get("rollout_sec") for r in steady]
            ),
            "median_ppo_train_sec": safe_median(
                [r.get("ppo_train_sec") for r in steady]
            ),
            "median_train_fraction": safe_median(
                [r.get("train_fraction") for r in steady]
            ),
        }
        write_json(summary_path, summary)

        print(
            f"DONE {worker_id} | "
            f"learn_sps={summary['learn_sps']:.1f} | "
            f"steady_median={summary['steady_median_sps']:.1f}",
            flush=True,
        )
        return 0

    except Exception as exc:
        err = traceback.format_exc()
        (run_dir / "error.txt").write_text(err, encoding="utf-8")
        write_json(
            summary_path,
            {
                "status": "FAILED",
                "worker_id": worker_id,
                "seed": seed,
                "start_delay_sec": start_delay,
                "error": repr(exc),
                "traceback": err,
                "process_started_wall": started_wall,
            },
        )
        print(f"FAILED {worker_id}: {exc!r}", flush=True)
        return 2

    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass

        model = None
        env = None
        timer = None

        try:
            lit.core.restore_process_patches()
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Parent-side hardware sampling
# =============================================================================

class HardwareSampler:
    """
    Stage-level hardware sampler.

    We intentionally sample once per second from the parent only, rather than
    launching nvidia-smi from every PPO worker.
    """

    def __init__(
        self,
        candidate_id: str,
        out_csv: Path,
        interval_sec: float = 1.0,
        gpu_index: int = 0,
    ):
        self.candidate_id = str(candidate_id)
        self.out_csv = Path(out_csv)
        self.interval_sec = float(interval_sec)
        self.gpu_index = int(gpu_index)
        self.rows: List[Dict[str, Any]] = []

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        try:
            import psutil  # type: ignore
            self.psutil = psutil
        except Exception:
            self.psutil = None

    def _gpu_snapshot(self) -> Dict[str, float]:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            ).strip().splitlines()[0]
            vals = [x.strip() for x in out.split(",")]
            return {
                "gpu_util": float(vals[0]),
                "gpu_mem_used_mb": float(vals[1]),
                "gpu_mem_total_mb": float(vals[2]),
                "gpu_power_w": float(vals[3]),
                "gpu_temp_c": float(vals[4]),
            }
        except Exception:
            return {
                "gpu_util": float("nan"),
                "gpu_mem_used_mb": float("nan"),
                "gpu_mem_total_mb": float("nan"),
                "gpu_power_w": float("nan"),
                "gpu_temp_c": float("nan"),
            }

    def _system_snapshot(self) -> Dict[str, float]:
        if self.psutil is None:
            return {
                "cpu_percent": float("nan"),
                "ram_used_gb": float("nan"),
                "ram_percent": float("nan"),
            }
        try:
            vm = self.psutil.virtual_memory()
            return {
                "cpu_percent": float(self.psutil.cpu_percent(interval=None)),
                "ram_used_gb": float((vm.total - vm.available) / (1024 ** 3)),
                "ram_percent": float(vm.percent),
            }
        except Exception:
            return {
                "cpu_percent": float("nan"),
                "ram_used_gb": float("nan"),
                "ram_percent": float("nan"),
            }

    def _loop(self) -> None:
        # Prime psutil CPU measurement.
        if self.psutil is not None:
            try:
                self.psutil.cpu_percent(interval=None)
            except Exception:
                pass

        while not self._stop.is_set():
            row = {
                "candidate": self.candidate_id,
                "wall_time": time.time(),
                "iso_time": datetime.now().isoformat(timespec="seconds"),
                **self._gpu_snapshot(),
                **self._system_snapshot(),
            }
            self.rows.append(row)
            write_csv(self.out_csv, self.rows)
            self._stop.wait(self.interval_sec)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None


# =============================================================================
# Parent-side candidate orchestration
# =============================================================================

def run_stage(
    *,
    candidate: Dict[str, Any],
    root: Path,
    updates: int,
    global_hardware_rows: List[Dict[str, Any]],
    global_worker_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cid = str(candidate["id"])
    jobs = int(candidate["jobs"])
    delays = list(candidate["start_delays"])

    if len(delays) != jobs:
        raise RuntimeError(f"{cid}: jobs={jobs} but delays={delays}")

    stage_dir = root / cid
    stage_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print(
        f"{cid} | {candidate['description']} | jobs={jobs} | "
        f"each=P40(40x512,b4096) | updates/job={updates} | "
        f"candidate transitions={jobs * updates * ROLLOUT:,}"
    )
    print("=" * 120, flush=True)

    sampler = HardwareSampler(
        candidate_id=cid,
        out_csv=stage_dir / "hardware_samples.csv",
        interval_sec=1.0,
        gpu_index=0,
    )

    children: List[subprocess.Popen] = []
    logs = []
    stage_start = time.perf_counter()

    try:
        sampler.start()

        # Launch all worker processes immediately. Staggering, if requested,
        # happens inside worker after environment/model construction.
        for j in range(jobs):
            worker_id = f"{cid}_W{j:02d}"
            worker_dir = stage_dir / f"worker_{j:02d}"
            worker_dir.mkdir(parents=True, exist_ok=True)

            # Different seeds avoid perfectly identical simulator trajectories.
            seed = 1000 + j

            log_path = worker_dir / "stdout.log"
            log_f = log_path.open("w", encoding="utf-8", buffering=1)
            logs.append(log_f)

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-id", worker_id,
                "--worker-dir", str(worker_dir),
                "--seed", str(seed),
                "--updates", str(updates),
                "--start-delay", str(float(delays[j])),
            ]

            p = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
                env=os.environ.copy(),
            )
            children.append(p)

            print(
                f"LAUNCHED {worker_id} | pid={p.pid} | "
                f"seed={seed} | start_delay={delays[j]:.1f}s",
                flush=True,
            )

        last_status = 0.0
        while True:
            alive = [p for p in children if p.poll() is None]
            if not alive:
                break

            now = time.perf_counter()
            if now - last_status >= 10.0:
                gpu = sampler.rows[-1] if sampler.rows else {}
                print(
                    f"[{cid}] alive={len(alive)}/{jobs} | "
                    f"GPU={gpu.get('gpu_util', float('nan')):.0f}% | "
                    f"VRAM={gpu.get('gpu_mem_used_mb', float('nan')):.0f} MB | "
                    f"CPU={gpu.get('cpu_percent', float('nan')):.1f}%",
                    flush=True,
                )
                last_status = now

            time.sleep(1.0)

        stage_wall_sec = time.perf_counter() - stage_start
        sampler.stop()

        worker_summaries: List[Dict[str, Any]] = []
        for j, p in enumerate(children):
            worker_dir = stage_dir / f"worker_{j:02d}"
            summary_path = worker_dir / "worker_summary.json"
            if summary_path.exists():
                row = read_json(summary_path)
            else:
                row = {
                    "status": "FAILED",
                    "worker_id": f"{cid}_W{j:02d}",
                    "error": f"missing worker_summary.json; returncode={p.returncode}",
                }

            row["candidate"] = cid
            row["candidate_jobs"] = jobs
            row["returncode"] = p.returncode
            worker_summaries.append(row)
            global_worker_rows.append(row)

        write_csv(stage_dir / "workers_summary.csv", worker_summaries)

        # Merge stage hardware rows into global hardware CSV.
        for row in sampler.rows:
            global_hardware_rows.append(dict(row))

        ok_workers = [
            w for w in worker_summaries
            if str(w.get("status", "")).upper() == "OK"
        ]

        total_steps = int(
            sum(int(w.get("actual_timesteps", 0)) for w in ok_workers)
        )

        learn_starts = finite_values(
            [w.get("learn_start_wall") for w in ok_workers]
        )
        learn_ends = finite_values(
            [w.get("learn_end_wall") for w in ok_workers]
        )
        if learn_starts and learn_ends:
            concurrent_train_window = max(learn_ends) - min(learn_starts)
        else:
            concurrent_train_window = float("nan")

        aggregate_wall_sps = (
            float(total_steps / stage_wall_sec)
            if stage_wall_sec > 0
            else float("nan")
        )
        aggregate_train_window_sps = (
            float(total_steps / concurrent_train_window)
            if math.isfinite(concurrent_train_window)
            and concurrent_train_window > 0
            else float("nan")
        )

        # Sum of each worker's update-2..N median throughput. This is a useful
        # steady-state companion metric, though wall SPS is the primary metric.
        aggregate_steady_sps = float(
            sum(
                float(w.get("steady_median_sps", 0.0))
                for w in ok_workers
                if math.isfinite(float(w.get("steady_median_sps", float("nan"))))
            )
        )

        h = sampler.rows
        summary = {
            "candidate": cid,
            "description": candidate["description"],
            "jobs_requested": jobs,
            "jobs_ok": len(ok_workers),
            "status": "OK" if len(ok_workers) == jobs else "PARTIAL_OR_FAILED",
            "n_envs_per_job": N_ENVS,
            "n_steps_per_job": N_STEPS,
            "batch_size_per_job": BATCH_SIZE,
            "rollout_per_job": ROLLOUT,
            "updates_per_job": updates,
            "requested_total_steps": jobs * updates * ROLLOUT,
            "actual_total_steps": total_steps,
            "stage_wall_sec": float(stage_wall_sec),
            "concurrent_train_window_sec": float(concurrent_train_window),
            "aggregate_wall_sps": aggregate_wall_sps,
            "aggregate_train_window_sps": aggregate_train_window_sps,
            "aggregate_steady_sps": aggregate_steady_sps,
            "per_worker_median_steady_sps": safe_median(
                [w.get("steady_median_sps") for w in ok_workers]
            ),
            "per_worker_min_steady_sps": (
                min(
                    finite_values(
                        [w.get("steady_median_sps") for w in ok_workers]
                    )
                )
                if finite_values(
                    [w.get("steady_median_sps") for w in ok_workers]
                )
                else float("nan")
            ),
            "median_rollout_sec_per_worker": safe_median(
                [w.get("median_rollout_sec") for w in ok_workers]
            ),
            "median_ppo_train_sec_per_worker": safe_median(
                [w.get("median_ppo_train_sec") for w in ok_workers]
            ),
            "mean_gpu_util": safe_mean([r.get("gpu_util") for r in h]),
            "p95_gpu_util": safe_percentile(
                [r.get("gpu_util") for r in h], 95
            ),
            "max_gpu_util": safe_max([r.get("gpu_util") for r in h]),
            "mean_gpu_mem_mb": safe_mean(
                [r.get("gpu_mem_used_mb") for r in h]
            ),
            "max_gpu_mem_mb": safe_max(
                [r.get("gpu_mem_used_mb") for r in h]
            ),
            "mean_gpu_power_w": safe_mean(
                [r.get("gpu_power_w") for r in h]
            ),
            "max_gpu_temp_c": safe_max(
                [r.get("gpu_temp_c") for r in h]
            ),
            "mean_cpu_percent": safe_mean(
                [r.get("cpu_percent") for r in h]
            ),
            "p95_cpu_percent": safe_percentile(
                [r.get("cpu_percent") for r in h], 95
            ),
            "max_ram_used_gb": safe_max(
                [r.get("ram_used_gb") for r in h]
            ),
            "start_delays_sec": delays,
        }

        write_json(stage_dir / "stage_summary.json", summary)

        print(
            f"\n{cid} DONE | "
            f"aggregate wall SPS={aggregate_wall_sps:.1f} | "
            f"train-window SPS={aggregate_train_window_sps:.1f} | "
            f"GPU avg={summary['mean_gpu_util']:.1f}% | "
            f"GPU p95={summary['p95_gpu_util']:.1f}% | "
            f"VRAM max={summary['max_gpu_mem_mb']:.0f} MB | "
            f"CPU avg={summary['mean_cpu_percent']:.1f}%",
            flush=True,
        )

        return summary

    except KeyboardInterrupt:
        print(f"\n[{cid}] interrupted; terminating OWN worker processes...", flush=True)
        for p in children:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        for p in children:
            try:
                p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        raise

    finally:
        try:
            sampler.stop()
        except Exception:
            pass
        for f in logs:
            try:
                f.close()
            except Exception:
                pass


# =============================================================================
# Parent CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UAM Joint P40 multi-job GPU utilization benchmark"
    )

    p.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--worker-id", default="", help=argparse.SUPPRESS)
    p.add_argument("--worker-dir", default="", help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--start-delay", type=float, default=0.0, help=argparse.SUPPRESS)

    p.add_argument(
        "--updates",
        type=int,
        default=DEFAULT_UPDATES,
        help="PPO updates per worker; default 8 = 163,840 transitions/worker.",
    )
    p.add_argument(
        "--candidates",
        default="ALL",
        help="Comma-separated A1..A5, default ALL.",
    )
    p.add_argument(
        "--output-root",
        default="",
        help="Optional output directory.",
    )
    p.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Seconds to wait between candidates; default 5.",
    )
    return p.parse_args()


def parent_main(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; refusing CPU-network benchmark.")

    if int(args.updates) < 2:
        raise SystemExit("--updates must be >=2 so update 1 can be warm-up.")

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    if str(args.candidates).strip().upper() == "ALL":
        selected = list(CANDIDATES)
    else:
        wanted = {
            x.strip().upper()
            for x in str(args.candidates).split(",")
            if x.strip()
        }
        selected = [
            c for c in CANDIDATES
            if str(c["id"]).upper() in wanted
        ]
        missing = wanted - {
            str(c["id"]).upper() for c in selected
        }
        if missing:
            raise SystemExit(f"Unknown candidates: {sorted(missing)}")

    if args.output_root:
        root = Path(args.output_root).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = ROOT / "serial_runs" / f"multijob_gpu_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    config = {
        "purpose": (
            "Use multiple independent P40 PPO cells to improve aggregate "
            "GPU/machine utilization WITHOUT shortening any PPO rollout horizon."
        ),
        "env_key": ENV_KEY,
        "method_id": METHOD_ID,
        "per_job_profile": {
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "rollout": ROLLOUT,
        },
        "updates_per_worker": int(args.updates),
        "timesteps_per_worker": int(args.updates) * ROLLOUT,
        "selected_candidates": selected,
        "total_requested_transitions": int(
            sum(int(c["jobs"]) for c in selected)
            * int(args.updates)
            * ROLLOUT
        ),
        "ranking_metric": "aggregate_wall_sps",
        "secondary_metric": "aggregate_train_window_sps",
        "device": "cuda-only",
        "full_info": True,
        "att_eval": False,
        "checkpoint_save": False,
        "model_save": False,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(root / "benchmark_config.json", config)

    print("\nUAM JOINT MULTI-JOB GPU UTILIZATION BENCHMARK")
    print(f"GPU          : {config['gpu_name']}")
    print(f"Torch/CUDA   : {config['torch_version']} / {config['cuda_version']}")
    print(f"Per PPO      : P40 = {N_ENVS} env x {N_STEPS} steps")
    print(f"Batch        : {BATCH_SIZE}")
    print(f"Rollout/PPO  : {ROLLOUT:,}")
    print(f"Updates/job  : {args.updates}")
    print(f"Candidates   : {[c['id'] for c in selected]}")
    print(f"Total budget : {config['total_requested_transitions']:,}")
    print(f"Output       : {root}")
    print("ATT EVAL     : DISABLED")
    print("CHECKPOINT   : DISABLED")
    print("MODEL SAVE   : DISABLED")
    print("Every PPO    : <=40 environments", flush=True)

    stage_summaries: List[Dict[str, Any]] = []
    global_hardware_rows: List[Dict[str, Any]] = []
    global_worker_rows: List[Dict[str, Any]] = []

    for idx, candidate in enumerate(selected):
        stage_summaries.append(
            run_stage(
                candidate=candidate,
                root=root,
                updates=int(args.updates),
                global_hardware_rows=global_hardware_rows,
                global_worker_rows=global_worker_rows,
            )
        )

        write_csv(root / "hardware_samples.csv", global_hardware_rows)
        write_csv(root / "workers_summary.csv", global_worker_rows)

        if idx + 1 < len(selected) and float(args.cooldown) > 0:
            print(
                f"Cooldown {float(args.cooldown):.1f}s before next candidate...",
                flush=True,
            )
            time.sleep(float(args.cooldown))

    ok = [
        s for s in stage_summaries
        if str(s.get("status")) == "OK"
    ]

    baseline = next(
        (s for s in ok if s.get("candidate") == "A1"),
        None,
    )
    baseline_sps = (
        float(baseline["aggregate_wall_sps"])
        if baseline is not None
        else float("nan")
    )

    for s in stage_summaries:
        sps = float(s.get("aggregate_wall_sps", float("nan")))
        jobs = int(s.get("jobs_requested", 1))
        if (
            math.isfinite(sps)
            and math.isfinite(baseline_sps)
            and baseline_sps > 0
        ):
            s["speedup_vs_A1"] = sps / baseline_sps
            s["parallel_efficiency_vs_A1"] = (
                sps / (baseline_sps * jobs)
            )
        else:
            s["speedup_vs_A1"] = float("nan")
            s["parallel_efficiency_vs_A1"] = float("nan")

    ranked_ok = sorted(
        [s for s in stage_summaries if str(s.get("status")) == "OK"],
        key=lambda x: float(x.get("aggregate_wall_sps", -1.0)),
        reverse=True,
    )
    failed = [
        s for s in stage_summaries
        if str(s.get("status")) != "OK"
    ]
    ranked = ranked_ok + failed

    for i, s in enumerate(ranked, start=1):
        s["rank"] = i if str(s.get("status")) == "OK" else ""

    write_csv(root / "multijob_ranking.csv", ranked)

    print("\n" + "#" * 120)
    print("FINAL MULTI-JOB RANKING")
    print("#" * 120)

    for i, s in enumerate(ranked_ok, start=1):
        print(
            f"{i:02d}. {s['candidate']} | jobs={s['jobs_requested']} | "
            f"agg wall SPS={float(s['aggregate_wall_sps']):.1f} | "
            f"train-window SPS={float(s['aggregate_train_window_sps']):.1f} | "
            f"speedup={float(s.get('speedup_vs_A1', float('nan'))):.3f}x | "
            f"eff={float(s.get('parallel_efficiency_vs_A1', float('nan'))):.3f} | "
            f"GPU avg={float(s['mean_gpu_util']):.1f}% | "
            f"GPU p95={float(s['p95_gpu_util']):.1f}% | "
            f"VRAM max={float(s['max_gpu_mem_mb']):.0f}MB | "
            f"CPU avg={float(s['mean_cpu_percent']):.1f}%",
            flush=True,
        )

    if ranked_ok:
        winner = ranked_ok[0]
        print(
            f"\nWINNER = {winner['candidate']} | "
            f"{winner['description']} | "
            f"aggregate wall SPS={winner['aggregate_wall_sps']:.1f}",
            flush=True,
        )

    print(f"\nRanking   : {root / 'multijob_ranking.csv'}")
    print(f"Workers   : {root / 'workers_summary.csv'}")
    print(f"Hardware  : {root / 'hardware_samples.csv'}")
    print("Done.", flush=True)
    return 0


def main() -> int:
    args = parse_args()

    if args.worker:
        if not args.worker_dir or not args.worker_id:
            raise SystemExit("--worker requires --worker-dir and --worker-id")
        return worker_main(args)

    return parent_main(args)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    raise SystemExit(main())
