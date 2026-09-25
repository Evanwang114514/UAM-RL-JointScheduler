# -*- coding: utf-8 -*-
"""
UAM Joint PPO server scaling benchmark V2 (~1.0M transitions total).

Purpose
-------
Second engineering-only speed sweep after the first 10-way benchmark found:
- 40 env x 512, batch 4096: ~2.37k SPS
- 64 env x 320, batch 4096: ~2.74k FULL / ~2.82k SLIM

This sweep asks one narrow question:
"Does pushing environment parallelism beyond 64 workers buy more throughput,
and does training-only info slimming help at high worker counts?"

Scientific/training settings held fixed
---------------------------------------
- Joint environment: J1 (ACTION_BRANCHING)
- Temporal representation: R_TDM
- PPO implementation/model/reward/obs/action unchanged
- CUDA network updates only
- Global rollout = 20,480 transitions/update
- PPO batch size = 4,096
- Seed = 1
- No ATT evaluation
- No checkpoint save
- No model save

Budget
------
10 candidates x 5 updates x 20,480 = 1,024,000 transitions.
Update 1 is warm-up; ranking uses updates 2..5.

Candidates
----------
V01 FULL  40 env x 512 steps   (old baseline control)
V02 SLIM  40 env x 512
V03 FULL  64 env x 320         (old fast control)
V04 SLIM  64 env x 320         (old winner control)
V05 FULL  80 env x 256
V06 SLIM  80 env x 256
V07 FULL 128 env x 160
V08 SLIM 128 env x 160
V09 FULL 160 env x 128
V10 SLIM 160 env x 128

SLIM only strips optional per-step diagnostic info before SubprocVecEnv IPC.
It does NOT change observation, action, reward, done, physics, or policy.

Run
---
python benchmark_joint_server_scaling_1m_v2.py

Subset
------
python benchmark_joint_server_scaling_1m_v2.py --candidates V05,V06,V07,V08

Outputs
-------
serial_runs/server_scaling_1m_v2_<timestamp>/
    scaling_updates.csv
    scaling_ranking.csv
    benchmark_config.json
    <candidate>/error.txt
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Avoid each worker spawning its own BLAS/OpenMP thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gymnasium as gym
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

import benchmark_joint_gpu_speed as bench
import train_uam_60m_literature_matrix_v3_1 as lit


ROOT = Path(__file__).resolve().parent
ROLL_OUT_PER_UPDATE = 20_480
DEFAULT_UPDATES = 5
ENV_KEY = "J1"
METHOD_ID = "R_TDM"
TRAIN_SEED = 1
MAX_TIME = 2_500
BATCH_SIZE = 4_096

CANDIDATES: List[Dict[str, Any]] = [
    {"id": "V01", "n_envs":  40, "n_steps": 512, "batch_size": BATCH_SIZE, "info_mode": "FULL"},
    {"id": "V02", "n_envs":  40, "n_steps": 512, "batch_size": BATCH_SIZE, "info_mode": "SLIM"},
    {"id": "V03", "n_envs":  64, "n_steps": 320, "batch_size": BATCH_SIZE, "info_mode": "FULL"},
    {"id": "V04", "n_envs":  64, "n_steps": 320, "batch_size": BATCH_SIZE, "info_mode": "SLIM"},
    {"id": "V05", "n_envs":  80, "n_steps": 256, "batch_size": BATCH_SIZE, "info_mode": "FULL"},
    {"id": "V06", "n_envs":  80, "n_steps": 256, "batch_size": BATCH_SIZE, "info_mode": "SLIM"},
    {"id": "V07", "n_envs": 128, "n_steps": 160, "batch_size": BATCH_SIZE, "info_mode": "FULL"},
    {"id": "V08", "n_envs": 128, "n_steps": 160, "batch_size": BATCH_SIZE, "info_mode": "SLIM"},
    {"id": "V09", "n_envs": 160, "n_steps": 128, "batch_size": BATCH_SIZE, "info_mode": "FULL"},
    {"id": "V10", "n_envs": 160, "n_steps": 128, "batch_size": BATCH_SIZE, "info_mode": "SLIM"},
]

BASELINE_ID = "V01"


class SlimTrainingInfoWrapper(gym.Wrapper):
    """Drop optional diagnostic info during training-only IPC."""

    @staticmethod
    def _slim(info: Any) -> Dict[str, Any]:
        if not isinstance(info, dict):
            return {}
        out: Dict[str, Any] = {}
        # Preserve Monitor-style episode statistics when present.
        if "episode" in info:
            out["episode"] = info["episode"]
        return out

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return obs, reward, terminated, truncated, self._slim(info)
        obs, reward, done, info = out
        return obs, reward, done, self._slim(info)


def make_variant_factory(
    *,
    env_index: int,
    run_dir: Path,
    max_time: int,
    info_mode: str,
):
    base_factory = lit.make_env_factory(
        env_key=ENV_KEY,
        method_id=METHOD_ID,
        env_index=int(env_index),
        run_dir=run_dir,
        max_time=int(max_time),
    )

    def _init():
        env = base_factory()
        if str(info_mode).upper() == "SLIM":
            env = SlimTrainingInfoWrapper(env)
        return env

    return _init


def build_vec_env_variant(
    *,
    candidate: Dict[str, Any],
    run_dir: Path,
    max_time: int,
):
    factories = [
        make_variant_factory(
            env_index=i,
            run_dir=run_dir,
            max_time=max_time,
            info_mode=str(candidate["info_mode"]),
        )
        for i in range(int(candidate["n_envs"]))
    ]

    raw = SubprocVecEnv(factories, start_method="spawn")
    raw.seed(TRAIN_SEED)

    return lit.TauPreservingVecNormalize(
        raw,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=lit.GAMMA,
    )


def run_candidate(
    *,
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
        name=(
            f"SCALING_{cid}_{candidate['n_envs']}x{candidate['n_steps']}"
            f"_b{candidate['batch_size']}_{candidate['info_mode']}"
        ),
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
            f"{cid}: batch_size={profile.batch_size} does not divide "
            f"rollout={profile.rollout}"
        )

    env = None
    model = None
    sampler = None
    local_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 120)
    print(
        f"{cid} | {candidate['info_mode']} | CUDA | "
        f"envs={profile.n_envs} x steps={profile.n_steps} | "
        f"batch={profile.batch_size} | updates={updates} | "
        f"candidate_steps={profile.rollout * updates:,}"
    )
    print("=" * 120, flush=True)

    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        lit.seed_all(TRAIN_SEED)

        env = build_vec_env_variant(
            candidate=candidate,
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

        bench.assert_cuda_model(model)
        print(
            f"CUDA ASSERTION OK | model.device={model.device} | "
            f"policy_params={lit.count_params(model):,}",
            flush=True,
        )

        sampler = bench.GPUSampler(device_index=int(torch.cuda.current_device()))
        print(f"GPU sampling mode = {sampler.mode}", flush=True)

        timer = bench.PerUpdateTimer(
            model=model,
            candidate=candidate,
            root=root,
            sampler=sampler,
            all_rows=all_rows,
        )
        local_rows = timer.local_rows

        model.learn(
            total_timesteps=int(profile.rollout * updates),
            callback=None,
            reset_num_timesteps=True,
            progress_bar=False,
        )
        torch.cuda.synchronize()

        summary = bench.summarize_candidate(candidate, local_rows, status="OK")
        summary["info_mode"] = candidate["info_mode"]
        return summary

    except Exception as exc:
        err = traceback.format_exc()
        (run_dir / "error.txt").write_text(err, encoding="utf-8")
        print(f"[{cid} FAILED] {exc!r}", flush=True)
        summary = bench.summarize_candidate(
            candidate,
            local_rows,
            status="FAILED",
            error=repr(exc),
        )
        summary["info_mode"] = candidate["info_mode"]
        return summary

    finally:
        if sampler is not None:
            sampler.close()
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        model = None
        env = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UAM Joint server scaling benchmark V2 (~1M total)"
    )
    p.add_argument(
        "--updates",
        type=int,
        default=DEFAULT_UPDATES,
        help=(
            "Updates/candidate. Default 5: "
            "10*5*20,480 = 1,024,000 total transitions."
        ),
    )
    p.add_argument(
        "--candidates",
        default="ALL",
        help="Comma-separated IDs, e.g. V05,V06,V07,V08; default ALL.",
    )
    p.add_argument("--max-time", type=int, default=MAX_TIME)
    p.add_argument(
        "--output-root",
        default="",
        help="Optional output dir; default serial_runs/server_scaling_1m_v2_<timestamp>.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; refusing CPU network benchmark.")

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    if int(args.updates) < 2:
        raise SystemExit("--updates must be >= 2 so update 1 is warm-up.")

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

    if args.output_root:
        root = Path(args.output_root).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = ROOT / "serial_runs" / f"server_scaling_1m_v2_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    config = {
        "purpose": "engineering parallelism scaling benchmark; no ATT eval",
        "env_key": ENV_KEY,
        "method_id": METHOD_ID,
        "device": "cuda-only",
        "network_update_device": "cuda-only",
        "vec_backend": "SB3 SubprocVecEnv(spawn)",
        "rollout_per_update": ROLL_OUT_PER_UPDATE,
        "batch_size": BATCH_SIZE,
        "updates_per_candidate": int(args.updates),
        "timesteps_per_candidate": int(
            ROLL_OUT_PER_UPDATE * int(args.updates)
        ),
        "candidate_count": len(selected),
        "total_requested_timesteps": int(
            ROLL_OUT_PER_UPDATE * int(args.updates) * len(selected)
        ),
        "ranking_metric": "median total_sps over updates 2..N",
        "baseline_candidate": BASELINE_ID,
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

    print("\nUAM JOINT SERVER SCALING BENCHMARK V2")
    print(f"GPU         : {config['gpu_name']}")
    print(f"Torch       : {config['torch_version']} | CUDA {config['cuda_version']}")
    print(f"Env/Method  : {ENV_KEY} / {METHOD_ID}")
    print(f"Per update  : {ROLL_OUT_PER_UPDATE:,}")
    print(f"Batch       : {BATCH_SIZE:,}")
    print(f"Updates     : {args.updates}")
    print(f"Candidates  : {len(selected)}")
    print(f"Total steps : {config['total_requested_timesteps']:,}")
    print(f"Output      : {root}")
    print("ATT EVAL    : DISABLED")
    print("CHECKPOINT  : DISABLED")
    print("MODEL SAVE  : DISABLED")
    print("NETWORK     : CUDA ONLY", flush=True)

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for candidate in selected:
        summaries.append(
            run_candidate(
                candidate=candidate,
                root=root,
                updates=int(args.updates),
                max_time=int(args.max_time),
                all_rows=all_rows,
            )
        )
        bench.write_csv(root / "scaling_updates.csv", all_rows)

    ok = [x for x in summaries if x.get("status") == "OK"]
    baseline = next(
        (x for x in ok if x.get("candidate") == BASELINE_ID),
        None,
    )
    baseline_sps = (
        float(baseline["median_total_sps"])
        if baseline is not None
        and math.isfinite(float(baseline["median_total_sps"]))
        else float("nan")
    )

    for row in summaries:
        sps = float(row.get("median_total_sps", float("nan")))
        row["speedup_vs_V01_full40"] = (
            sps / baseline_sps
            if math.isfinite(sps)
            and math.isfinite(baseline_sps)
            and baseline_sps > 0
            else float("nan")
        )

    ranked_ok = sorted(
        ok,
        key=lambda x: float(x["median_total_sps"]),
        reverse=True,
    )
    failed = [x for x in summaries if x.get("status") != "OK"]
    ranked = ranked_ok + failed

    for i, row in enumerate(ranked, start=1):
        row["rank"] = i if row.get("status") == "OK" else ""

    bench.write_csv(root / "scaling_ranking.csv", ranked)

    print("\n" + "#" * 120)
    print("FINAL SCALING V2 RANKING")
    print("#" * 120)

    for i, row in enumerate(ranked_ok, start=1):
        speedup = float(row.get("speedup_vs_V01_full40", float("nan")))
        speedup_text = (
            f"{speedup:.3f}x" if math.isfinite(speedup) else "n/a"
        )
        print(
            f"{i:02d}. {row['candidate']} | "
            f"{row['info_mode']:<4} | "
            f"envs={int(row['n_envs']):>3} | "
            f"steps={int(row['n_steps']):>3} | "
            f"median SPS={float(row['median_total_sps']):.1f} | "
            f"rollout={float(row['median_rollout_sec']):.2f}s | "
            f"train={float(row['median_ppo_train_sec']):.2f}s | "
            f"GPU={float(row['mean_gpu_util']):.1f}% | "
            f"CV={float(row['cv_total_sps']):.4f} | "
            f"vs V01={speedup_text}",
            flush=True,
        )

    if ranked_ok:
        winner = ranked_ok[0]
        print(
            f"\nWINNER = {winner['candidate']} | "
            f"{winner['info_mode']} | "
            f"envs={winner['n_envs']} | "
            f"n_steps={winner['n_steps']} | "
            f"median SPS={winner['median_total_sps']:.1f}",
            flush=True,
        )

    if failed:
        print("\nFAILED:")
        for row in failed:
            print(f"- {row['candidate']}: {row.get('error', '')}")

    print(f"\nDetailed updates : {root / 'scaling_updates.csv'}")
    print(f"Final ranking    : {root / 'scaling_ranking.csv'}")
    print("Done.", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
