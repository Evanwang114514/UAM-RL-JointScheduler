# -*- coding: utf-8 -*-
"""
Pure engineering benchmark for UAM Joint PPO throughput.

Purpose
-------
Measure how well different SubprocVecEnv parallelism / PPO batch-size settings
feed the GPU. This is NOT a scientific ATT experiment.

Fixed experiment
----------------
- Joint environment: J1 (ACTION_BRANCHING)
- Temporal representation: R_TDM
- Network updates: CUDA ONLY
- Rollout size per PPO update: 20,480 transitions for every candidate
- Default: 8 PPO updates = 163,840 transitions per candidate
- No checkpoint saving
- No ATT evaluation
- No model saving

Outputs
-------
serial_runs/speed_bench_joint_gpu_<timestamp>/
    speed_updates.csv   # one row per PPO update
    speed_ranking.csv   # candidates ranked by steady-state median SPS (updates 2..N)
    benchmark_config.json
    <candidate>/error.txt  # only when a candidate fails

Run
---
conda activate uam5070
cd /d "E:\\Study Files\\github\\UAM-predict\\UAGMC-main"
python benchmark_joint_gpu_speed.py

Optional
--------
python benchmark_joint_gpu_speed.py --candidates C1,C2,C3,C4
python benchmark_joint_gpu_speed.py --updates 8
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import subprocess
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Keep CPU threads for environment / orchestration, not neural-network training.
# The PPO network itself is asserted to live on CUDA.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

import train_uam_60m_literature_matrix_v3_1 as lit


ROOT = Path(__file__).resolve().parent
ROLL_OUT_PER_UPDATE = 20_480
DEFAULT_UPDATES = 8
ENV_KEY = "J1"
METHOD_ID = "R_TDM"
TRAIN_SEED = 1
MAX_TIME = 2_500

# All candidates use CUDA. Only env parallelism and PPO minibatch size change.
CANDIDATES: List[Dict[str, Any]] = [
    {"id": "C1", "n_envs": 8,  "n_steps": 2560, "batch_size": 2048},
    {"id": "C2", "n_envs": 16, "n_steps": 1280, "batch_size": 2048},
    {"id": "C3", "n_envs": 20, "n_steps": 1024, "batch_size": 2048},
    {"id": "C4", "n_envs": 32, "n_steps": 640,  "batch_size": 2048},
    {"id": "C5", "n_envs": 16, "n_steps": 1280, "batch_size": 4096},
    {"id": "C6", "n_envs": 20, "n_steps": 1024, "batch_size": 4096},
    {"id": "C7", "n_envs": 32, "n_steps": 640,  "batch_size": 4096},
    {"id": "C8", "n_envs": 40, "n_steps": 512,  "batch_size": 4096},
]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
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
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def safe_mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def safe_max(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(max(vals)) if vals else float("nan")


def safe_median(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(statistics.median(vals)) if vals else float("nan")


def safe_cv(xs: Sequence[float]) -> float:
    vals = np.asarray(
        [float(x) for x in xs if x is not None and math.isfinite(float(x))],
        dtype=float,
    )
    if len(vals) < 2 or abs(float(vals.mean())) < 1e-12:
        return float("nan")
    return float(vals.std(ddof=0) / vals.mean())


class GPUSampler:
    """
    Low-overhead GPU sampler.

    Preferred path: NVML Python bindings, sampled in-process every 0.25 s.
    Fallback path: one nvidia-smi snapshot at the end of each update to avoid
    repeatedly spawning nvidia-smi and distorting the benchmark.
    """

    def __init__(self, device_index: int = 0, interval_sec: float = 0.25):
        self.device_index = int(device_index)
        self.interval_sec = float(interval_sec)
        self.samples: List[Dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = None
        self._handle = None
        self.mode = "nvidia-smi-snapshot"

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.mode = "pynvml-continuous"
        except Exception:
            self._nvml = None
            self._handle = None

    def _nvml_sample(self) -> Optional[Dict[str, float]]:
        if self._nvml is None or self._handle is None:
            return None
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            power = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            temp = self._nvml.nvmlDeviceGetTemperature(
                self._handle, self._nvml.NVML_TEMPERATURE_GPU
            )
            return {
                "gpu_util": float(util.gpu),
                "gpu_mem_mb": float(mem.used / 1024.0 / 1024.0),
                "gpu_power_w": float(power),
                "gpu_temp_c": float(temp),
            }
        except Exception:
            return None

    def _smi_snapshot(self) -> Optional[Dict[str, float]]:
        cmd = [
            "nvidia-smi",
            f"--id={self.device_index}",
            "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, text=True, timeout=3
            ).strip().splitlines()[0]
            vals = [x.strip() for x in out.split(",")]
            return {
                "gpu_util": float(vals[0]),
                "gpu_mem_mb": float(vals[1]),
                "gpu_power_w": float(vals[2]),
                "gpu_temp_c": float(vals[3]),
            }
        except Exception:
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            s = self._nvml_sample()
            if s is not None:
                self.samples.append(s)
            self._stop.wait(self.interval_sec)

    def start_update(self) -> None:
        self.samples = []
        self._stop.clear()
        if self._nvml is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def finish_update(self) -> Dict[str, float]:
        if self._nvml is not None:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            self._thread = None
        else:
            s = self._smi_snapshot()
            if s is not None:
                self.samples.append(s)

        util = [x["gpu_util"] for x in self.samples]
        mem = [x["gpu_mem_mb"] for x in self.samples]
        power = [x["gpu_power_w"] for x in self.samples]
        temp = [x["gpu_temp_c"] for x in self.samples]
        return {
            "gpu_sample_mode": self.mode,
            "gpu_samples": len(self.samples),
            "gpu_util_avg": safe_mean(util),
            "gpu_util_max": safe_max(util),
            "gpu_mem_avg_mb": safe_mean(mem),
            "gpu_mem_max_mb": safe_max(mem),
            "gpu_power_avg_w": safe_mean(power),
            "gpu_power_max_w": safe_max(power),
            "gpu_temp_avg_c": safe_mean(temp),
            "gpu_temp_max_c": safe_max(temp),
        }

    def close(self) -> None:
        try:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


class PerUpdateTimer:
    """
    Instruments one SB3 model without changing PPO math.

    collect_rollouts() -> rollout wall time
    train()            -> PPO optimization wall time
    """

    def __init__(
        self,
        model: Any,
        candidate: Dict[str, Any],
        root: Path,
        sampler: GPUSampler,
        all_rows: List[Dict[str, Any]],
    ):
        self.model = model
        self.candidate = dict(candidate)
        self.root = Path(root)
        self.sampler = sampler
        self.all_rows = all_rows
        self.local_rows: List[Dict[str, Any]] = []

        self.update_idx = 0
        self.prev_timesteps = int(model.num_timesteps)
        self.update_start = 0.0
        self.rollout_sec = 0.0

        self._orig_collect = model.collect_rollouts
        self._orig_train = model.train
        self._patch()

    def _patch(self) -> None:
        def timed_collect(*args, **kwargs):
            self.update_idx += 1
            self.update_start = time.perf_counter()
            self.sampler.start_update()

            t0 = time.perf_counter()
            out = self._orig_collect(*args, **kwargs)
            self.rollout_sec = time.perf_counter() - t0
            return out

        def timed_train(*args, **kwargs):
            t0 = time.perf_counter()
            out = self._orig_train(*args, **kwargs)
            train_sec = time.perf_counter() - t0
            total_sec = time.perf_counter() - self.update_start

            now_steps = int(self.model.num_timesteps)
            delta_steps = now_steps - self.prev_timesteps
            self.prev_timesteps = now_steps

            gpu = self.sampler.finish_update()

            rollout_sps = (
                float(delta_steps / self.rollout_sec)
                if self.rollout_sec > 0 else float("nan")
            )
            train_equiv_sps = (
                float(delta_steps / train_sec)
                if train_sec > 0 else float("nan")
            )
            total_sps = (
                float(delta_steps / total_sec)
                if total_sec > 0 else float("nan")
            )

            row: Dict[str, Any] = {
                "candidate": self.candidate["id"],
                "update": int(self.update_idx),
                "device": "cuda",
                "env_key": ENV_KEY,
                "method_id": METHOD_ID,
                "n_envs": int(self.candidate["n_envs"]),
                "n_steps": int(self.candidate["n_steps"]),
                "batch_size": int(self.candidate["batch_size"]),
                "rollout_per_update": int(delta_steps),
                "timesteps": now_steps,
                "rollout_sec": float(self.rollout_sec),
                "ppo_train_sec": float(train_sec),
                "total_update_sec": float(total_sec),
                "rollout_sps": rollout_sps,
                "train_equiv_sps": train_equiv_sps,
                "total_sps": total_sps,
                "train_time_fraction": (
                    float(train_sec / total_sec)
                    if total_sec > 0 else float("nan")
                ),
                **gpu,
            }

            self.local_rows.append(row)
            self.all_rows.append(row)
            write_csv(self.root / "speed_updates.csv", self.all_rows)

            print(
                f"[{row['candidate']}] update={row['update']:02d} "
                f"steps={row['timesteps']:>7d} | "
                f"rollout={row['rollout_sec']:.2f}s | "
                f"train={row['ppo_train_sec']:.2f}s | "
                f"TOTAL={row['total_update_sec']:.2f}s | "
                f"SPS={row['total_sps']:.1f} | "
                f"GPU={row['gpu_util_avg']:.1f}% "
                f"(max {row['gpu_util_max']:.1f}%)",
                flush=True,
            )
            return out

        self.model.collect_rollouts = timed_collect
        self.model.train = timed_train


def summarize_candidate(
    candidate: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    status: str = "OK",
    error: str = "",
) -> Dict[str, Any]:
    rows = list(rows)
    # Update 1 is warm-up. Rank on updates 2..N whenever possible.
    steady = rows[1:] if len(rows) >= 2 else rows

    total_sps = [float(r["total_sps"]) for r in steady]
    rollout_sps = [float(r["rollout_sps"]) for r in steady]
    rollout_sec = [float(r["rollout_sec"]) for r in steady]
    train_sec = [float(r["ppo_train_sec"]) for r in steady]
    total_sec = [float(r["total_update_sec"]) for r in steady]
    train_frac = [float(r["train_time_fraction"]) for r in steady]
    gpu_avg = [float(r["gpu_util_avg"]) for r in steady]
    gpu_max = [float(r["gpu_util_max"]) for r in steady]
    gpu_mem = [float(r["gpu_mem_max_mb"]) for r in steady]
    gpu_power = [float(r["gpu_power_avg_w"]) for r in steady]
    gpu_temp = [float(r["gpu_temp_max_c"]) for r in steady]

    return {
        "candidate": candidate["id"],
        "status": status,
        "error": error,
        "device": "cuda",
        "n_envs": candidate["n_envs"],
        "n_steps": candidate["n_steps"],
        "batch_size": candidate["batch_size"],
        "rollout_per_update": candidate["n_envs"] * candidate["n_steps"],
        "updates_completed": len(rows),
        "steady_updates_used": len(steady),
        "median_total_sps": safe_median(total_sps),
        "mean_total_sps": safe_mean(total_sps),
        "cv_total_sps": safe_cv(total_sps),
        "median_rollout_sps": safe_median(rollout_sps),
        "median_rollout_sec": safe_median(rollout_sec),
        "median_ppo_train_sec": safe_median(train_sec),
        "median_total_update_sec": safe_median(total_sec),
        "median_train_time_fraction": safe_median(train_frac),
        "mean_gpu_util": safe_mean(gpu_avg),
        "max_gpu_util_seen": safe_max(gpu_max),
        "max_gpu_mem_mb": safe_max(gpu_mem),
        "mean_gpu_power_w": safe_mean(gpu_power),
        "max_gpu_temp_c": safe_max(gpu_temp),
    }


def assert_cuda_model(model: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark intentionally has no CPU-training mode.")

    dev = torch.device(model.device)
    if dev.type != "cuda":
        raise RuntimeError(f"Model device is {dev}, expected CUDA.")

    bad = []
    for name, p in model.policy.named_parameters():
        if p.device.type != "cuda":
            bad.append((name, str(p.device)))
            if len(bad) >= 5:
                break
    if bad:
        raise RuntimeError(f"Some policy parameters are not on CUDA: {bad}")


def run_candidate(
    candidate: Dict[str, Any],
    root: Path,
    updates: int,
    max_time: int,
    all_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cid = str(candidate["id"])
    run_dir = root / cid
    run_dir.mkdir(parents=True, exist_ok=True)

    profile = lit.SpeedProfile(
        name=f"BENCH_{cid}_{candidate['n_envs']}x{candidate['n_steps']}_b{candidate['batch_size']}",
        n_envs=int(candidate["n_envs"]),
        n_steps=int(candidate["n_steps"]),
        batch_size=int(candidate["batch_size"]),
    )

    if profile.rollout != ROLL_OUT_PER_UPDATE:
        raise RuntimeError(
            f"{cid}: rollout={profile.rollout}, expected {ROLL_OUT_PER_UPDATE}"
        )
    if profile.rollout % profile.batch_size != 0:
        raise RuntimeError(
            f"{cid}: batch_size={profile.batch_size} does not divide rollout={profile.rollout}"
        )

    env = None
    model = None
    sampler = None
    local_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 110)
    print(
        f"{cid} | CUDA ONLY | n_envs={profile.n_envs} | "
        f"n_steps={profile.n_steps} | batch={profile.batch_size} | "
        f"updates={updates} | total_steps={profile.rollout * updates}"
    )
    print("=" * 110, flush=True)

    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        lit.seed_all(TRAIN_SEED)

        env = lit.build_vec_env(
            env_key=ENV_KEY,
            method_id=METHOD_ID,
            profile=profile,
            seed=TRAIN_SEED,
            run_dir=run_dir,
            max_time=int(max_time),
        )

        model = lit.build_model(
            env=env,
            env_key=ENV_KEY,
            method_id=METHOD_ID,
            profile=profile,
            seed=TRAIN_SEED,
            run_dir=run_dir,
            device="cuda",
        )
        model.verbose = 0

        assert_cuda_model(model)
        print(
            f"CUDA ASSERTION OK | model.device={model.device} | "
            f"policy_params={lit.count_params(model):,}",
            flush=True,
        )

        cuda_index = int(torch.cuda.current_device())
        sampler = GPUSampler(device_index=cuda_index)
        print(f"GPU sampling mode = {sampler.mode}", flush=True)

        timer = PerUpdateTimer(
            model=model,
            candidate=candidate,
            root=root,
            sampler=sampler,
            all_rows=all_rows,
        )
        local_rows = timer.local_rows

        total_timesteps = int(profile.rollout * updates)

        # No callbacks, no checkpoints, no evaluation, no model saving.
        model.learn(
            total_timesteps=total_timesteps,
            callback=None,
            reset_num_timesteps=True,
            progress_bar=False,
        )

        torch.cuda.synchronize()

        return summarize_candidate(candidate, local_rows, status="OK")

    except Exception as exc:
        err = traceback.format_exc()
        (run_dir / "error.txt").write_text(err, encoding="utf-8")
        print(f"[{cid} FAILED] {exc!r}", flush=True)
        return summarize_candidate(
            candidate,
            local_rows,
            status="FAILED",
            error=repr(exc),
        )

    finally:
        if sampler is not None:
            sampler.close()
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        try:
            del env
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPU-only Joint PPO throughput benchmark"
    )
    p.add_argument(
        "--updates",
        type=int,
        default=DEFAULT_UPDATES,
        help="PPO updates per candidate; default 8 = 163,840 transitions.",
    )
    p.add_argument(
        "--candidates",
        default="ALL",
        help="Comma-separated candidate IDs, e.g. C1,C2,C7; default ALL.",
    )
    p.add_argument("--max-time", type=int, default=MAX_TIME)
    p.add_argument(
        "--output-root",
        default="",
        help="Optional existing/new output directory. Default creates serial_runs/speed_bench_joint_gpu_<timestamp>.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing to run CPU network benchmark.")

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
        selected = [c for c in CANDIDATES if c["id"].upper() in wanted]
        missing = wanted - {c["id"].upper() for c in selected}
        if missing:
            raise SystemExit(f"Unknown candidates: {sorted(missing)}")

    if int(args.updates) < 2:
        raise SystemExit("--updates must be >= 2 so update 1 can be treated as warm-up.")

    if args.output_root:
        root = Path(args.output_root).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = ROOT / "serial_runs" / f"speed_bench_joint_gpu_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    config = {
        "purpose": "pure engineering speed benchmark; no ATT evaluation",
        "env_key": ENV_KEY,
        "method_id": METHOD_ID,
        "device": "cuda-only",
        "network_update_device": "cuda-only",
        "rollout_per_update": ROLL_OUT_PER_UPDATE,
        "updates_per_candidate": int(args.updates),
        "timesteps_per_candidate": int(ROLL_OUT_PER_UPDATE * int(args.updates)),
        "ranking_metric": "median total_sps over updates 2..N",
        "train_seed": TRAIN_SEED,
        "max_time": int(args.max_time),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "selected_candidates": selected,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (root / "benchmark_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nGPU-ONLY UAM JOINT SPEED BENCHMARK")
    print(f"GPU       : {config['gpu_name']}")
    print(f"Torch     : {config['torch_version']} | CUDA {config['cuda_version']}")
    print(f"Env       : {ENV_KEY}")
    print(f"Method    : {METHOD_ID}")
    print(f"Per update: {ROLL_OUT_PER_UPDATE:,} transitions")
    print(f"Updates   : {args.updates}")
    print(f"Output    : {root}")
    print("ATT EVAL  : DISABLED")
    print("CHECKPOINT: DISABLED")
    print("MODEL SAVE: DISABLED")
    print("NETWORK   : CUDA ONLY", flush=True)

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for candidate in selected:
        summary = run_candidate(
            candidate=candidate,
            root=root,
            updates=int(args.updates),
            max_time=int(args.max_time),
            all_rows=all_rows,
        )
        summaries.append(summary)

        ok = [x for x in summaries if x["status"] == "OK"]
        bad = [x for x in summaries if x["status"] != "OK"]
        ranked = sorted(
            ok,
            key=lambda x: (
                -float(x["median_total_sps"])
                if math.isfinite(float(x["median_total_sps"]))
                else float("inf")
            ),
        ) + bad
        for i, row in enumerate(ranked, start=1):
            row["rank"] = i if row["status"] == "OK" else ""
        write_csv(root / "speed_ranking.csv", ranked)

        if summary["status"] == "OK":
            print(
                f"SUMMARY {candidate['id']} | "
                f"steady median SPS={summary['median_total_sps']:.1f} | "
                f"mean={summary['mean_total_sps']:.1f} | "
                f"CV={summary['cv_total_sps']:.4f} | "
                f"GPU avg={summary['mean_gpu_util']:.1f}% | "
                f"train fraction={summary['median_train_time_fraction']:.3f}",
                flush=True,
            )

    ok = [x for x in summaries if x["status"] == "OK"]
    if ok:
        ranked_ok = sorted(ok, key=lambda x: float(x["median_total_sps"]), reverse=True)
        base = next((x for x in ranked_ok if x["candidate"] == "C2"), None)
        base_sps = float(base["median_total_sps"]) if base else float("nan")

        for row in summaries:
            if (
                row["status"] == "OK"
                and math.isfinite(base_sps)
                and base_sps > 0
            ):
                row["speedup_vs_C2_current_P3"] = (
                    float(row["median_total_sps"]) / base_sps
                )
            else:
                row["speedup_vs_C2_current_P3"] = float("nan")

        ranked = sorted(
            [x for x in summaries if x["status"] == "OK"],
            key=lambda x: float(x["median_total_sps"]),
            reverse=True,
        ) + [x for x in summaries if x["status"] != "OK"]

        for i, row in enumerate(ranked, start=1):
            row["rank"] = i if row["status"] == "OK" else ""
        write_csv(root / "speed_ranking.csv", ranked)

        winner = ranked_ok[0]
        print("\n" + "#" * 110)
        print("FINAL SPEED RANKING")
        print("#" * 110)
        for i, row in enumerate(ranked_ok, start=1):
            speedup = row.get("speedup_vs_C2_current_P3", float("nan"))
            speedup_text = f"{speedup:.3f}x" if math.isfinite(float(speedup)) else "n/a"
            print(
                f"{i:02d}. {row['candidate']} | "
                f"envs={row['n_envs']:>2} batch={row['batch_size']:>4} | "
                f"median SPS={row['median_total_sps']:.1f} | "
                f"GPU={row['mean_gpu_util']:.1f}% | "
                f"vs C2={speedup_text}"
            )
        print(
            f"\nWINNER = {winner['candidate']} | "
            f"median SPS={winner['median_total_sps']:.1f}",
            flush=True,
        )

    print(f"\nDetailed updates : {root / 'speed_updates.csv'}")
    print(f"Final ranking    : {root / 'speed_ranking.csv'}")
    print("Done.", flush=True)


if __name__ == "__main__":
    # Required for Windows spawn-based SubprocVecEnv.
    import multiprocessing as mp

    mp.freeze_support()
    main()
