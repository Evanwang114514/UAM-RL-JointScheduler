# -*- coding: utf-8 -*-
"""
Speed benchmark for train_uagmc_45x800k_formal.py
==================================================

Purpose
-------
Benchmark the validated 45x800k runner with several training-equivalent
parallelism profiles while keeping:
- global rollout = 20,480 transitions
- PPO batch size = 2,048
- PPO epochs / gamma / GAE / clip / reward / architecture unchanged

Default representative workloads:
    E2:M0   light / turnaround + residual
    E5:M4   heaviest single-side physics + heaviest representation
    J3:M4   joint PPO + heaviest representation

Default profiles:
    P5   =  5 envs x 4096 steps, batch 2048
    P8   =  8 envs x 2560 steps, batch 2048
    P10  = 10 envs x 2048 steps, batch 2048
    P16  = 16 envs x 1280 steps, batch 2048  (current formal profile)

For every workload/profile it measures:
1) environment-only throughput (random actions; one 20,480-transition block)
2) full PPO throughput after one warm-up rollout
3) GPU utilization / memory / power / temperature when nvidia-smi is available
4) host CPU / RAM load when psutil is available

It writes:
    speed_benchmark_raw.csv
    speed_benchmark_summary.csv
    speed_benchmark_profile_ranking.csv
    speed_benchmark_manifest.json
    speed_benchmark_summary.txt
    FORMAL45_SPEED_BENCH_<timestamp>.zip

IMPORTANT
---------
Do NOT run this at the same time as the formal 45x800k training if you want
meaningful speed numbers. A second Python/CUDA process will contend for the
same CPU/GPU and make both the benchmark and the formal training slower.

Typical:
    python benchmark_uagmc_45x_speed.py

Fast one-workload check:
    python benchmark_uagmc_45x_speed.py --workloads E5:M4 --timed-rollouts 1

Repeat for more reliable ranking:
    python benchmark_uagmc_45x_speed.py --repeats 2

The script must sit beside:
    train_uagmc_45x800k_formal.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import gc
import json
import math
import shutil
import statistics
import subprocess
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import train_uagmc_45x800k_formal as formal


ROOT = Path(__file__).resolve().parent
GLOBAL_ROLLOUT = 20_480
DEFAULT_BATCH = 2_048


@dataclass(frozen=True)
class Profile:
    name: str
    n_envs: int
    n_steps: int
    batch_size: int = DEFAULT_BATCH

    @property
    def rollout(self) -> int:
        return int(self.n_envs) * int(self.n_steps)


PROFILES: Dict[str, Profile] = {
    "P5": Profile("P5_5x4096_b2048", 5, 4096),
    "P8": Profile("P8_8x2560_b2048", 8, 2560),
    "P10": Profile("P10_10x2048_b2048", 10, 2048),
    "P16": Profile("P16_16x1280_b2048", 16, 1280),
}

DEFAULT_WORKLOADS = ("E2:M0", "E5:M4", "J3:M4")
DEFAULT_PROFILES = ("P5", "P8", "P10", "P16")


def _finite(x: Any) -> Optional[float]:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def _mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if _finite(x) is not None]
    return float(statistics.fmean(vals)) if vals else float("nan")


def _percentile(xs: Sequence[float], q: float) -> float:
    vals = np.asarray([float(x) for x in xs if _finite(x) is not None], dtype=float)
    if not len(vals):
        return float("nan")
    return float(np.percentile(vals, q))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_workloads(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for raw in str(text).split(","):
        raw = raw.strip().upper()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"Workload must be ROW:METHOD, got {raw!r}")
        row, method = [x.strip().upper() for x in raw.split(":", 1)]
        if row not in formal.ROWS:
            raise ValueError(f"Unknown row={row}; allowed={formal.ROWS}")
        if method not in formal.METHODS:
            raise ValueError(f"Unknown method={method}; allowed={formal.METHODS}")
        out.append((row, method))
    if not out:
        raise ValueError("No workloads selected")
    return out


def parse_profiles(text: str) -> List[Profile]:
    names = [x.strip().upper() for x in str(text).split(",") if x.strip()]
    bad = [x for x in names if x not in PROFILES]
    if bad:
        raise ValueError(f"Unknown profiles={bad}; allowed={list(PROFILES)}")
    out = [PROFILES[x] for x in names]
    if not out:
        raise ValueError("No profiles selected")
    for p in out:
        if p.rollout != GLOBAL_ROLLOUT:
            raise ValueError(f"{p.name}: rollout={p.rollout} != {GLOBAL_ROLLOUT}")
        if GLOBAL_ROLLOUT % p.batch_size != 0:
            raise ValueError(f"{p.name}: batch_size must divide global rollout")
    return out


class ResourceSampler:
    """Best-effort CPU/RAM/GPU sampler; benchmark still works without psutil/nvidia-smi."""

    def __init__(self, interval: float = 0.5):
        self.interval = max(0.2, float(interval))
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.samples: List[Dict[str, float]] = []
        try:
            import psutil  # type: ignore
            self.psutil = psutil
        except Exception:
            self.psutil = None
        self.nvidia_smi = shutil.which("nvidia-smi")

    def _sample_gpu(self) -> Dict[str, float]:
        if not self.nvidia_smi:
            return {}
        cmd = [
            self.nvidia_smi,
            "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            ).strip().splitlines()
            if not out:
                return {}
            parts = [x.strip() for x in out[0].split(",")]
            if len(parts) < 4:
                return {}
            return {
                "gpu_util_pct": float(parts[0]),
                "gpu_mem_mb": float(parts[1]),
                "gpu_power_w": float(parts[2]),
                "gpu_temp_c": float(parts[3]),
            }
        except Exception:
            return {}

    def _sample_host(self) -> Dict[str, float]:
        if self.psutil is None:
            return {}
        try:
            return {
                "cpu_util_pct": float(self.psutil.cpu_percent(interval=None)),
                "ram_util_pct": float(self.psutil.virtual_memory().percent),
            }
        except Exception:
            return {}

    def _loop(self) -> None:
        if self.psutil is not None:
            try:
                self.psutil.cpu_percent(interval=None)
            except Exception:
                pass
        while not self.stop_event.is_set():
            row: Dict[str, float] = {"sample_time": time.time()}
            row.update(self._sample_host())
            row.update(self._sample_gpu())
            self.samples.append(row)
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.samples = []
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        self.thread = None
        out: Dict[str, float] = {"resource_samples": float(len(self.samples))}
        keys = (
            "cpu_util_pct",
            "ram_util_pct",
            "gpu_util_pct",
            "gpu_mem_mb",
            "gpu_power_w",
            "gpu_temp_c",
        )
        for key in keys:
            vals = [s[key] for s in self.samples if key in s and math.isfinite(s[key])]
            out[f"{key}_mean"] = _mean(vals)
            out[f"{key}_p95"] = _percentile(vals, 95)
            out[f"{key}_max"] = max(vals) if vals else float("nan")
        return out


def patch_profile(profile: Profile) -> Dict[str, int]:
    old = {
        "N_ENVS": int(formal.N_ENVS),
        "N_STEPS": int(formal.N_STEPS),
        "GLOBAL_ROLLOUT": int(formal.GLOBAL_ROLLOUT),
        "BATCH_SIZE": int(formal.BATCH_SIZE),
    }
    formal.N_ENVS = int(profile.n_envs)
    formal.N_STEPS = int(profile.n_steps)
    formal.GLOBAL_ROLLOUT = int(profile.rollout)
    formal.BATCH_SIZE = int(profile.batch_size)
    return old


def restore_profile(old: Dict[str, int]) -> None:
    formal.N_ENVS = int(old["N_ENVS"])
    formal.N_STEPS = int(old["N_STEPS"])
    formal.GLOBAL_ROLLOUT = int(old["GLOBAL_ROLLOUT"])
    formal.BATCH_SIZE = int(old["BATCH_SIZE"])


def build_env(*, row: str, method: str, seed: int, run_dir: Path):
    return formal.build_train_env(
        stage=row,
        method=method,
        topology="T2",
        fleet_size=formal.FLEET_SIZE,
        seed=int(seed),
        run_dir=run_dir,
        future_horizon=formal.FUTURE_HORIZON_MIN,
        max_events=formal.MAX_EVENTS_PER_TYPE,
        pad_separation=formal.PAD_SEPARATION_MIN,
        charger_capacity=formal.CHARGER_CAPACITY,
        max_time=formal.MAX_TIME,
    )


def env_only_benchmark(env, *, profile: Profile, total_transitions: int) -> Tuple[float, float, int]:
    total_transitions = max(int(total_transitions), int(profile.n_envs))
    vector_steps = max(1, math.ceil(total_transitions / profile.n_envs))
    env.reset()
    for _ in range(min(8, vector_steps)):
        actions = np.asarray([env.action_space.sample() for _ in range(profile.n_envs)], dtype=np.int64)
        env.step(actions)
    t0 = time.perf_counter()
    for _ in range(vector_steps):
        actions = np.asarray([env.action_space.sample() for _ in range(profile.n_envs)], dtype=np.int64)
        env.step(actions)
    elapsed = time.perf_counter() - t0
    actual = vector_steps * profile.n_envs
    return actual / max(elapsed, 1e-9), elapsed, actual


def benchmark_one(
    *,
    row: str,
    method: str,
    profile: Profile,
    repeat: int,
    seed: int,
    device: str,
    root: Path,
    warmup_rollouts: int,
    timed_rollouts: int,
    env_only_transitions: int,
    sample_interval: float,
) -> Dict[str, Any]:
    run_dir = root / f"{row}__{method}" / profile.name / f"repeat_{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    previous = patch_profile(profile)
    env = None
    model = None
    sampler = ResourceSampler(interval=sample_interval)

    try:
        formal.seed_all(seed + repeat)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        env = build_env(row=row, method=method, seed=seed + repeat, run_dir=run_dir / "env_only")
        env_fps, env_elapsed, env_actual = env_only_benchmark(
            env, profile=profile, total_transitions=env_only_transitions
        )
        try:
            env.close()
        except Exception:
            pass
        env = None
        gc.collect()

        env = build_env(row=row, method=method, seed=seed + repeat, run_dir=run_dir / "ppo")
        model = formal.build_model(
            env=env,
            stage=row,
            method=method,
            topology="T2",
            max_events=formal.MAX_EVENTS_PER_TYPE,
            seed=seed + repeat,
            run_dir=run_dir / "ppo",
            device=device,
        )

        warmup_steps = max(0, int(warmup_rollouts)) * profile.rollout
        timed_steps = max(1, int(timed_rollouts)) * profile.rollout

        if warmup_steps > 0:
            model.learn(total_timesteps=warmup_steps, progress_bar=False, reset_num_timesteps=True)

        before_steps = int(model.num_timesteps)
        sampler.start()
        t0 = time.perf_counter()
        model.learn(total_timesteps=timed_steps, progress_bar=False, reset_num_timesteps=False)
        train_elapsed = time.perf_counter() - t0
        resources = sampler.stop()
        after_steps = int(model.num_timesteps)
        actual_train_steps = after_steps - before_steps
        train_fps = actual_train_steps / max(train_elapsed, 1e-9)

        return {
            "status": "OK",
            "row": row,
            "row_name": formal.ROW_NAMES[row],
            "method": method,
            "method_name": formal.METHOD_NAMES[method],
            "profile_name": profile.name,
            "repeat": repeat,
            "seed": seed + repeat,
            "device": str(model.device),
            "n_envs": profile.n_envs,
            "n_steps": profile.n_steps,
            "global_rollout": profile.rollout,
            "batch_size": profile.batch_size,
            "n_epochs": int(formal.N_EPOCHS),
            "env_only_transitions": env_actual,
            "env_only_elapsed_sec": env_elapsed,
            "env_only_fps": env_fps,
            "warmup_rollouts": int(warmup_rollouts),
            "timed_rollouts": int(timed_rollouts),
            "timed_requested_steps": timed_steps,
            "timed_actual_steps": actual_train_steps,
            "train_elapsed_sec": train_elapsed,
            "train_fps": train_fps,
            "sec_per_rollout": train_elapsed / max(1, int(timed_rollouts)),
            "estimated_hours_per_800k_cell": 800_000.0 / max(train_fps, 1e-9) / 3600.0,
            "cuda_peak_memory_mb": (
                torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                if torch.cuda.is_available()
                else 0.0
            ),
            **resources,
        }

    except Exception as exc:
        try:
            resources = sampler.stop()
        except Exception:
            resources = {}
        return {
            "status": "ERROR",
            "row": row,
            "method": method,
            "profile_name": profile.name,
            "repeat": repeat,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            **resources,
        }
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        model = None
        env = None
        restore_profile(previous)
        try:
            formal.core.restore_process_patches()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def aggregate(raw: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in raw:
        if row.get("status") != "OK":
            continue
        key = (str(row["row"]), str(row["method"]), str(row["profile_name"]))
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for (r, m, p), rows in sorted(groups.items()):
        first = rows[0]
        summary: Dict[str, Any] = {
            "row": r,
            "method": m,
            "profile_name": p,
            "n_envs": first["n_envs"],
            "n_steps": first["n_steps"],
            "batch_size": first["batch_size"],
            "repeats_ok": len(rows),
        }
        for key in (
            "env_only_fps",
            "train_fps",
            "sec_per_rollout",
            "estimated_hours_per_800k_cell",
            "cuda_peak_memory_mb",
            "cpu_util_pct_mean",
            "ram_util_pct_mean",
            "gpu_util_pct_mean",
            "gpu_mem_mb_mean",
            "gpu_power_w_mean",
            "gpu_temp_c_mean",
        ):
            vals = [float(x[key]) for x in rows if key in x and _finite(x[key]) is not None]
            summary[f"{key}_mean"] = _mean(vals)
            summary[f"{key}_std"] = float(np.std(vals, ddof=0)) if vals else float("nan")
        out.append(summary)

    baseline: Dict[Tuple[str, str], float] = {}
    for row in out:
        if str(row["profile_name"]).startswith("P16_"):
            baseline[(str(row["row"]), str(row["method"]))] = float(row["train_fps_mean"])

    for row in out:
        b = baseline.get((str(row["row"]), str(row["method"])))
        row["speedup_vs_P16"] = (
            float(row["train_fps_mean"]) / b if b and b > 0 else float("nan")
        )
    return out


def overall_profile_ranking(summary: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_profile: Dict[str, List[Dict[str, Any]]] = {}
    for row in summary:
        by_profile.setdefault(str(row["profile_name"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for profile, rows in by_profile.items():
        fps = [float(r["train_fps_mean"]) for r in rows if _finite(r.get("train_fps_mean"))]
        speedups = [
            float(r["speedup_vs_P16"])
            for r in rows
            if _finite(r.get("speedup_vs_P16")) and float(r["speedup_vs_P16"]) > 0
        ]
        geo = (
            float(math.exp(statistics.fmean(math.log(x) for x in speedups)))
            if speedups else float("nan")
        )
        out.append(
            {
                "profile_name": profile,
                "workloads_completed": len(rows),
                "mean_train_fps": _mean(fps),
                "geomean_speedup_vs_P16": geo,
                "mean_estimated_hours_per_800k_cell": _mean(
                    [
                        float(r["estimated_hours_per_800k_cell_mean"])
                        for r in rows
                        if _finite(r.get("estimated_hours_per_800k_cell_mean"))
                    ]
                ),
                "mean_gpu_util_pct": _mean(
                    [
                        float(r["gpu_util_pct_mean_mean"])
                        for r in rows
                        if _finite(r.get("gpu_util_pct_mean_mean"))
                    ]
                ),
            }
        )

    out.sort(
        key=lambda x: (
            -float(x["geomean_speedup_vs_P16"])
            if _finite(x["geomean_speedup_vs_P16"])
            else float("inf")
        )
    )
    return out


def make_zip(root: Path, paths: Sequence[Path]) -> Path:
    zip_path = root / f"FORMAL45_SPEED_BENCH_{root.name.split('_')[-1]}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p.exists() and p.is_file():
                zf.write(p, arcname=p.name)
    return zip_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS))
    p.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup-rollouts", type=int, default=1)
    p.add_argument("--timed-rollouts", type=int, default=2)
    p.add_argument("--env-only-transitions", type=int, default=GLOBAL_ROLLOUT)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--sample-interval", type=float, default=0.5)
    p.add_argument("--output-root", default=None)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workloads = parse_workloads(args.workloads)
    profiles = parse_profiles(args.profiles)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (ROOT / "serial_runs" / f"formal45_speedbench_{stamp}").resolve()
    )
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "formal_runner": "train_uagmc_45x800k_formal.py",
        "workloads": [{"row": r, "method": m} for r, m in workloads],
        "profiles": [dict(asdict(p), rollout=p.rollout) for p in profiles],
        "repeats": int(args.repeats),
        "warmup_rollouts": int(args.warmup_rollouts),
        "timed_rollouts": int(args.timed_rollouts),
        "env_only_transitions": int(args.env_only_transitions),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "controls": {
            "global_rollout_fixed": GLOBAL_ROLLOUT,
            "batch_size_fixed": DEFAULT_BATCH,
            "ppo_epochs_unchanged": int(formal.N_EPOCHS),
            "representation_and_physics_unchanged": True,
            "warning": "Do not run concurrently with formal training for valid measurements.",
        },
    }
    manifest_path = root / "speed_benchmark_manifest.json"
    _write_json(manifest_path, manifest)

    raw_rows: List[Dict[str, Any]] = []
    total = len(workloads) * len(profiles) * max(1, int(args.repeats))
    index = 0

    print("=" * 120)
    print("FORMAL45 SPEED BENCHMARK")
    print(f"Workloads : {workloads}")
    print(f"Profiles  : {[p.name for p in profiles]}")
    print(f"Repeats   : {args.repeats}")
    print(f"Device    : {device}")
    print(f"Output    : {root}")
    print("IMPORTANT : benchmark and formal training should NOT run simultaneously.")
    print("=" * 120, flush=True)

    for repeat in range(max(1, int(args.repeats))):
        profile_order = profiles if repeat % 2 == 0 else list(reversed(profiles))
        for row, method in workloads:
            for profile in profile_order:
                index += 1
                print(f"\n[{index}/{total}] {row}/{method} | {profile.name} | repeat={repeat}", flush=True)
                result = benchmark_one(
                    row=row,
                    method=method,
                    profile=profile,
                    repeat=repeat,
                    seed=int(args.seed),
                    device=device,
                    root=root,
                    warmup_rollouts=int(args.warmup_rollouts),
                    timed_rollouts=int(args.timed_rollouts),
                    env_only_transitions=int(args.env_only_transitions),
                    sample_interval=float(args.sample_interval),
                )
                raw_rows.append(result)
                _write_csv(root / "speed_benchmark_raw.csv", raw_rows)

                if result.get("status") == "OK":
                    print(
                        "  env-only="
                        f"{float(result['env_only_fps']):.1f} fps | "
                        "PPO="
                        f"{float(result['train_fps']):.1f} fps | "
                        "800k≈"
                        f"{float(result['estimated_hours_per_800k_cell']):.2f} h | "
                        "GPU util="
                        f"{float(result.get('gpu_util_pct_mean', float('nan'))):.1f}%",
                        flush=True,
                    )
                else:
                    print(f"  ERROR: {result.get('error')}", flush=True)
                    if not args.continue_on_error:
                        _write_json(root / "FAILED.json", result)
                        raise RuntimeError(result.get("error"))

    summary = aggregate(raw_rows)
    ranking = overall_profile_ranking(summary)

    summary_path = root / "speed_benchmark_summary.csv"
    ranking_path = root / "speed_benchmark_profile_ranking.csv"
    _write_csv(summary_path, summary)
    _write_csv(ranking_path, ranking)

    lines = [
        "FORMAL45 SPEED BENCHMARK SUMMARY",
        "=" * 110,
        "",
        "Overall profile ranking (geometric mean speedup vs current P16):",
    ]
    for i, row in enumerate(ranking, start=1):
        lines.append(
            f"{i:>2}. {row['profile_name']:<24} "
            f"geo-speedup={float(row['geomean_speedup_vs_P16']):.3f}x | "
            f"mean FPS={float(row['mean_train_fps']):.1f} | "
            f"800k/cell≈{float(row['mean_estimated_hours_per_800k_cell']):.2f} h | "
            f"GPU util≈{float(row['mean_gpu_util_pct']):.1f}%"
        )

    lines += ["", "Per workload/profile:"]
    for row in summary:
        lines.append(
            f"{row['row']}/{row['method']} | {row['profile_name']:<24} "
            f"env={float(row['env_only_fps_mean']):.1f} fps | "
            f"train={float(row['train_fps_mean']):.1f} fps | "
            f"speedup={float(row['speedup_vs_P16']):.3f}x | "
            f"GPU={float(row.get('gpu_util_pct_mean_mean', float('nan'))):.1f}%"
        )

    summary_txt = root / "speed_benchmark_summary.txt"
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    zip_path = make_zip(
        root,
        [
            manifest_path,
            root / "speed_benchmark_raw.csv",
            summary_path,
            ranking_path,
            summary_txt,
        ],
    )

    print("\n" + "=" * 120)
    print("BENCHMARK DONE")
    print(f"Summary : {summary_txt}")
    print(f"ZIP     : {zip_path}")
    print("Send the ZIP to ChatGPT for profile selection / speed diagnosis.")
    print("=" * 120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
