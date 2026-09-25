# -*- coding: utf-8 -*-
"""
10-way engineering throughput benchmark for the current UAM Joint PPO stack.

Goal
----
Pick one faster SERVER production profile without changing the scientific PPO
objective, reward, observation, action space, model architecture, or global
rollout size.

Fixed scientific/training settings
----------------------------------
- Environment: J1 (ACTION_BRANCHING)
- Temporal representation: R_TDM
- Device: CUDA only
- Global rollout: 20,480 transitions/update
- PPO batch size: 4,096
- Default: 5 updates/candidate
- 10 candidates -> 10 * 5 * 20,480 = 1,024,000 total transitions
- No ATT evaluation
- No checkpoint saving
- No model saving
- Update 1 is warm-up; ranking uses updates 2..N

What changes
------------
Only engineering execution settings:
1) number of parallel SubprocVecEnv workers;
2) whether training-only info dictionaries are slimmed before IPC.

"SLIM" does NOT change obs/action/reward/done/physics. It only drops nonessential
per-step diagnostic info during training. SB3's SubprocVecEnv still injects its
own terminal_observation / TimeLimit.truncated on episode end.

Candidates
----------
T01 FULL 16 env x 1280
T02 FULL 20 env x 1024
T03 FULL 32 env x  640
T04 FULL 40 env x  512   <- current C8 baseline
T05 FULL 64 env x  320
T06 SLIM 16 env x 1280
T07 SLIM 20 env x 1024
T08 SLIM 32 env x  640
T09 SLIM 40 env x  512
T10 SLIM 64 env x  320

All candidates:
    batch_size = 4096
    rollout    = 20480

Run
---
python benchmark_joint_server_10way.py

Subset
------
python benchmark_joint_server_10way.py --candidates T04,T09,T10

Outputs
-------
serial_runs/server10_joint_speed_<timestamp>/
    server10_updates.csv
    server10_ranking.csv
    benchmark_config.json
    <candidate>/error.txt
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
    {"id": "T01", "n_envs": 16, "n_steps": 1280, "batch_size": BATCH_SIZE, "info_mode": "FULL"},
    {"id": "T02", "n_envs": 20, "n_steps": 1024, "batch_size": BATCH_SIZE, "info_mode": "FULL"},
    {"id": "T03", "n_envs": 32, "n_steps":  640, "batch_size": BATCH_SIZE, "info_mode": "FULL"},
    {"id": "T04", "n_envs": 40, "n_steps":  512, "batch_size": BATCH_SIZE, "info_mode": "FULL"},
    {"id": "T05", "n_envs": 64, "n_steps":  320, "batch_size": BATCH_SIZE, "info_mode": "FULL"},
    {"id": "T06", "n_envs": 16, "n_steps": 1280, "batch_size": BATCH_SIZE, "info_mode": "SLIM"},
    {"id": "T07", "n_envs": 20, "n_steps": 1024, "batch_size": BATCH_SIZE, "info_mode": "SLIM"},
    {"id": "T08", "n_envs": 32, "n_steps":  640, "batch_size": BATCH_SIZE, "info_mode": "SLIM"},
    {"id": "T09", "n_envs": 40, "n_steps":  512, "batch_size": BATCH_SIZE, "info_mode": "SLIM"},
    {"id": "T10", "n_envs": 64, "n_steps":  320, "batch_size": BATCH_SIZE, "info_mode": "SLIM"},
]

BASELINE_ID = "T04"


class SlimTrainingInfoWrapper(gym.Wrapper):
    """
    Reduce Python IPC payload during training.

    Only the optional Monitor-style "episode" summary is kept if present.
    Observation, reward, termination/truncation and environment state are
    untouched. SubprocVecEnv adds terminal_observation and
    TimeLimit.truncated after this wrapper returns.
    """

    @staticmethod
    def _slim(info: Any) -> Dict[str, Any]:
        if not isinstance(info, dict):
            return {}
        out: Dict[str, Any] = {}
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
    env_key: str,
    method_id: str,
    env_index: int,
    run_dir: Path,
    max_time: int,
    info_mode: str,
):
    base_factory = lit.make_env_factory(
        env_key=env_key,
        method_id=method_id,
        env_index=env_index,
        run_dir=run_dir,
        max_time=max_time,
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
            env_key=ENV_KEY,
            method_id=METHOD_ID,
            env_index=i,
            run_dir=run_dir,
            max_time=max_time,
            info_mode=str(candidate["info_mode"]),
        )
        for i in range(int(candidate["n_envs"]))
    ]

    # Keep the same backend/start method as the current validated benchmark.
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
        name=f"SERVER10_{cid}_{candidate['n_envs']}x{candidate['n_steps']}_b{candidate['batch_size']}_{candidate['info_mode']}",
        n_envs=int(candidate["n_envs"]),
        n_steps=int(candidate["n_steps"]),
        batch_size=int(candidate["batch_size"]),
    )

    if profile.rollout != ROLL_OUT_PER_UPDATE:
        raise RuntimeError(f"{cid}: rollout={profile.rollout}, expected {ROLL_OUT_PER_UPDATE}")
    if profile.rollout % profile.batch_size != 0:
        raise RuntimeError(f"{cid}: batch_size={profile.batch_size} does not divide rollout={profile.rollout}")

    env = None
    model = None
    sampler = None
    local_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 118)
    print(
        f"{cid} | {candidate['info_mode']} INFO | CUDA ONLY | "
        f"n_envs={profile.n_envs} | n_steps={profile.n_steps} | "
        f"batch={profile.batch_size} | updates={updates} | "
        f"total_steps={profile.rollout * updates}"
    )
    print("=" * 118, flush=True)

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

        cuda_index = int(torch.cuda.current_device())
        sampler = bench.GPUSampler(device_index=cuda_index)
        print(f"GPU sampling mode = {sampler.mode}", flush=True)

        timer = bench.PerUpdateTimer(
            model=model,
            candidate=candidate,
            root=root,
            sampler=sampler,
            all_rows=all_rows,
        )
        local_rows = timer.local_rows

        total_timesteps = int(profile.rollout * updates)
        model.learn(
            total_timesteps=total_timesteps,
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
    p = argparse.ArgumentParser(description="10-way server Joint PPO throughput benchmark")
    p.add_argument(
        "--updates",
        type=int,
        default=DEFAULT_UPDATES,
        help="Updates per candidate. Default 5 => 102,400 steps/candidate => 1,024,000 total for all 10.",
    )
    p.add_argument(
        "--candidates",
        default="ALL",
        help="Comma-separated candidate IDs, e.g. T04,T09,T10; default ALL.",
    )
    p.add_argument("--max-time", type=int, default=MAX_TIME)
    p.add_argument(
        "--output-root",
        default="",
        help="Optional output directory. Default: serial_runs/server10_joint_speed_<timestamp>.",
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

    if int(args.updates) < 2:
        raise SystemExit("--updates must be >=2 so update 1 can be warm-up.")

    if str(args.candidates).strip().upper() == "ALL":
        selected = list(CANDIDATES)
    else:
        wanted = {x.strip().upper() for x in str(args.candidates).split(",") if x.strip()}
        selected = [c for c in CANDIDATES if c["id"].upper() in wanted]
        missing = wanted - {c["id"].upper() for c in selected}
        if missing:
            raise SystemExit(f"Unknown candidates: {sorted(missing)}")

    if args.output_root:
        root = Path(args.output_root).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = ROOT / "serial_runs" / f"server10_joint_speed_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    config = {
        "purpose": "engineering speed benchmark; no ATT evaluation",
        "env_key": ENV_KEY,
        "method_id": METHOD_ID,
        "device": "cuda-only",
        "network_update_device": "cuda-only",
        "vec_backend": "SB3 SubprocVecEnv(spawn)",
        "rollout_per_update": ROLL_OUT_PER_UPDATE,
        "batch_size": BATCH_SIZE,
        "updates_per_candidate": int(args.updates),
        "timesteps_per_candidate": int(ROLL_OUT_PER_UPDATE * int(args.updates)),
        "selected_candidate_count": len(selected),
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

    print("\n10-WAY UAM JOINT SERVER SPEED BENCHMARK")
    print(f"GPU        : {config['gpu_name']}")
    print(f"Torch      : {config['torch_version']} | CUDA {config['cuda_version']}")
    print(f"Env/Method : {ENV_KEY} / {METHOD_ID}")
    print(f"Per update : {ROLL_OUT_PER_UPDATE:,} transitions")
    print(f"Batch      : {BATCH_SIZE:,}")
    print(f"Updates    : {args.updates}")
    print(f"Candidates : {len(selected)}")
    print(f"Total steps: {config['total_requested_timesteps']:,}")
    print(f"Output     : {root}")
    print("ATT EVAL   : DISABLED")
    print("CHECKPOINT : DISABLED")
    print("MODEL SAVE : DISABLED")
    print("NETWORK    : CUDA ONLY", flush=True)

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

        # Keep the main update CSV name specific to this benchmark.
        bench.write_csv(root / "server10_updates.csv", all_rows)

    ok = [x for x in summaries if x.get("status") == "OK"]
    baseline = next((x for x in ok if x.get("candidate") == BASELINE_ID), None)
    baseline_sps = (
        float(baseline["median_total_sps"])
        if baseline is not None and math.isfinite(float(baseline["median_total_sps"]))
        else float("nan")
    )

    for row in summaries:
        sps = float(row.get("median_total_sps", float("nan")))
        row["speedup_vs_T04_full40"] = (
            sps / baseline_sps
            if math.isfinite(sps) and math.isfinite(baseline_sps) and baseline_sps > 0
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

    bench.write_csv(root / "server10_ranking.csv", ranked)

    print("\n" + "#" * 118)
    print("FINAL SERVER10 RANKING")
    print("#" * 118)

    for i, row in enumerate(ranked_ok, start=1):
        speedup = float(row.get("speedup_vs_T04_full40", float("nan")))
        speedup_text = f"{speedup:.3f}x" if math.isfinite(speedup) else "n/a"
        print(
            f"{i:02d}. {row['candidate']} | "
            f"{row['info_mode']:<4} | "
            f"envs={int(row['n_envs']):>2} | "
            f"median SPS={float(row['median_total_sps']):.1f} | "
            f"rollout={float(row['median_rollout_sec']):.2f}s | "
            f"train={float(row['median_ppo_train_sec']):.2f}s | "
            f"GPU={float(row['mean_gpu_util']):.1f}% | "
            f"CV={float(row['cv_total_sps']):.4f} | "
            f"vs T04={speedup_text}",
            flush=True,
        )

    if ranked_ok:
        winner = ranked_ok[0]
        print(
            f"\nWINNER = {winner['candidate']} | "
            f"{winner['info_mode']} | envs={winner['n_envs']} | "
            f"median SPS={winner['median_total_sps']:.1f}",
            flush=True,
        )

    if failed:
        print("\nFAILED:")
        for row in failed:
            print(f"- {row['candidate']}: {row.get('error', '')}")

    print(f"\nDetailed updates : {root / 'server10_updates.csv'}")
    print(f"Final ranking    : {root / 'server10_ranking.csv'}")
    print("Done.", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
