# -*- coding: utf-8 -*-
"""
UAGMC E1 fixed-fleet FAST formal training
=========================================

Goal
----
Train the SAME UAGMC PPO logic after changing only aircraft supply dynamics to:

    conserved_closed_loop
    fleet_size = 16
    initial allocation = V0:12, V1:4

Formal matrix:
    train seeds = 0,1,2
    requested budget = 1,000,000 timesteps / seed
    device = CUDA
    NO evaluation / rollout replay during training

Speed configuration
-------------------
Use the previously benchmarked regular fast configuration:

    16 parallel CPU environments
    1280 steps per environment
    batch_size = 512
    GPU PPO update

The key fairness property is preserved:

    old global rollout = 5 * 4096 = 20,480
    new global rollout = 16 * 1280 = 20,480

Therefore:
    - same global rollout size
    - same batch_size=512
    - same 40 minibatches / epoch
    - same 10 PPO epochs
    - same PPO hyperparameters
    - only vectorization / per-env rollout partition changes

Important strict-fairness note
------------------------------
Changing 5x4096 -> 16x1280 can still change trajectory-fragment structure.
For a publication-grade E0-vs-E1 comparison, run the E0 control with THIS SAME
trainer configuration too. This script supports:

    --env-mode fixed   (default; E1, N=16)
    --env-mode legacy  (matched fast E0 control)

No mid-training evaluation
--------------------------
This script DOES NOT:
    - load checkpoints for evaluation
    - run deterministic validation episodes
    - run effect-time diagnostics
    - replay passenger traces during training

It only:
    - trains continuously
    - saves passive model + VecNormalize checkpoints every 50k
    - records lightweight PPO logger scalars

Checkpoint saving pauses only for serialization; it does not run any rollout.

1M and SB3 rollout alignment
------------------------------
Global rollout size is 20,480, so SB3 normally finishes the current rollout:
    requested 1,000,000 -> rollout-aligned final 1,003,520

Because 50,000 is divisible by 16, an exact 1,000,000-step checkpoint is saved
during that final rollout. Use that checkpoint for the nominal 1M comparison.

Files
-----
Put this file in UAGMC-main root and run:

    python train_uagmc_fixedfleet16_fast_3seed_1m.py

Matched E0 control later:

    python train_uagmc_fixedfleet16_fast_3seed_1m.py --env-mode legacy

Outputs:
    serial_runs/
      uagmc_E1_fixed16_fast16env_3seed_1M_<timestamp>/
        experiment_manifest.json
        serial_status.csv
        seed_0/
        seed_1/
        seed_2/
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from utilss.make_env_fleet import make_env
from utilss.sb3_utils import linear_schedule


ROOT = Path(__file__).resolve().parent

# =============================================================================
# Frozen scientific configuration
# =============================================================================

FLEET_SIZE = 16
CANDIDATES = [0, 1]
TO_VERTIPORT = 2

TRAIN_FILE = ROOT / "train_data" / "passengers_300.csv"
MAX_TIME = 500

# Fast configuration: same global rollout as original 5*4096.
N_ENVS = 16
N_STEPS = 1280
GLOBAL_ROLLOUT = N_ENVS * N_STEPS  # 20,480

BATCH_SIZE = 512
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.25
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
INITIAL_LR = 5e-4

DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_TIMESTEPS = 1_000_000
DEFAULT_CHECKPOINT_INTERVAL = 50_000


# =============================================================================
# Utilities
# =============================================================================

def jsonable(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    serial = {}
    for k, v in row.items():
        if isinstance(v, (dict, list, tuple, np.ndarray)):
            serial[k] = json.dumps(jsonable(v), ensure_ascii=False)
        else:
            serial[k] = v

    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(serial.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(serial)
        f.flush()


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty seed list")
    return vals


def find_scenario(obj: Any):
    if hasattr(obj, "venv"):
        obj = obj.venv

    if hasattr(obj, "envs") and obj.envs:
        obj = obj.envs[0]

    seen = set()
    for _ in range(60):
        if id(obj) in seen:
            break
        seen.add(id(obj))

        if (
            hasattr(obj, "_all_evtols")
            and hasattr(obj, "vertiports")
            and hasattr(obj, "get_fixed_fleet_diagnostics")
        ):
            return obj

        for attr in ("scenario", "env", "unwrapped"):
            if hasattr(obj, attr):
                nxt = getattr(obj, attr)
                if nxt is not None and nxt is not obj:
                    obj = nxt
                    break
        else:
            break

    raise RuntimeError(
        "Cannot locate ConservedFleetScenario; stopped at "
        f"{type(obj).__module__}.{type(obj).__name__}"
    )


# =============================================================================
# No-evaluation passive scalar callback
# =============================================================================

class PassiveTrainingScalarCallback(BaseCallback):
    """
    Records PPO logger scalars when total timesteps cross each 50k milestone.

    Does NOT:
      - predict actions
      - reset environments
      - run eval episodes
      - replay checkpoints
    """

    def __init__(
        self,
        run_dir: Path,
        interval: int,
        requested_steps: int,
    ):
        super().__init__(verbose=0)
        self.path = Path(run_dir) / "training_milestones.csv"
        self.interval = int(interval)
        self.requested_steps = int(requested_steps)
        self.next_mark = int(interval)

    def _on_step(self) -> bool:
        while (
            self.num_timesteps >= self.next_mark
            and self.next_mark <= self.requested_steps
        ):
            values = getattr(self.model.logger, "name_to_value", {}) or {}

            append_csv(
                self.path,
                {
                    "milestone": self.next_mark,
                    "model_num_timesteps": int(self.num_timesteps),
                    "approx_kl": values.get("train/approx_kl", np.nan),
                    "clip_fraction": values.get("train/clip_fraction", np.nan),
                    "entropy_loss": values.get("train/entropy_loss", np.nan),
                    "explained_variance": values.get(
                        "train/explained_variance", np.nan
                    ),
                    "policy_gradient_loss": values.get(
                        "train/policy_gradient_loss", np.nan
                    ),
                    "value_loss": values.get("train/value_loss", np.nan),
                    "learning_rate": values.get("train/learning_rate", np.nan),
                },
            )

            print(
                f"[passive milestone] {self.next_mark:,} "
                f"(no evaluation)",
                flush=True,
            )
            self.next_mark += self.interval

        return True


# =============================================================================
# Environment construction
# =============================================================================

def env_factory(
    *,
    env_index: int,
    run_dir: Path,
    env_mode: str,
):
    """
    Return a pickle-safe factory from utilss.make_env_fleet.make_env.
    """
    kwargs = dict(
        max_time=MAX_TIME,
        log_dir=run_dir / "monitor",
        env_index=env_index,
        person_spawn_file=str(TRAIN_FILE),
        candidate_from_vertiports=CANDIDATES,
        to_vertiport=TO_VERTIPORT,
        enable_logger=False,
    )

    if env_mode == "fixed":
        kwargs.update(
            fleet_mode="conserved_closed_loop",
            fleet_size=FLEET_SIZE,
            fleet_assertions=True,
        )
    elif env_mode == "legacy":
        kwargs.update(
            fleet_mode="legacy_replenish",
            fleet_size=None,
            fleet_assertions=False,
        )
    else:
        raise ValueError(env_mode)

    return make_env(**kwargs)


def run_preflight_fixed(seed: int, run_dir: Path) -> Dict[str, Any]:
    """
    Separate one-env preflight. It never touches the training VecNormalize.
    """
    raw = DummyVecEnv(
        [
            env_factory(
                env_index=999,
                run_dir=run_dir / "_preflight",
                env_mode="fixed",
            )
        ]
    )

    try:
        raw.seed(seed)
        raw.reset()

        scenario = find_scenario(raw)
        diag = scenario.get_fixed_fleet_diagnostics()

        fleet_size = int(diag.get("fleet_size", -1))
        allocation = diag.get("initial_allocation", {})

        if fleet_size != 16:
            raise AssertionError(
                f"Expected fixed fleet 16, got {fleet_size}"
            )

        if (
            int(allocation.get("0", -1)) != 12
            or int(allocation.get("1", -1)) != 4
        ):
            raise AssertionError(
                f"Expected V0/V1 allocation 12/4, got {allocation}"
            )

        scenario._assert_fixed_fleet()
        return diag

    finally:
        raw.close()


def build_train_env(
    *,
    seed: int,
    run_dir: Path,
    env_mode: str,
) -> VecNormalize:
    factories = [
        env_factory(
            env_index=i,
            run_dir=run_dir,
            env_mode=env_mode,
        )
        for i in range(N_ENVS)
    ]

    # Windows-safe parallel CPU simulation.
    raw = SubprocVecEnv(
        factories,
        start_method="spawn",
    )
    raw.seed(seed)

    env = VecNormalize(
        raw,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )

    return env


# =============================================================================
# PPO construction
# =============================================================================

def build_model(
    *,
    env: VecNormalize,
    seed: int,
    run_dir: Path,
    device: str,
) -> PPO:
    """
    Preserve the same runtime PPO/policy configuration as the E0 reproduction.
    Do not "fix" the public-source feature-extractor wiring here.
    """
    from utilss.encoding import TemporalLSTMExtractor

    policy_kwargs = dict(
        net_arch=dict(
            features_extractor_class=TemporalLSTMExtractor,
            pi=[256, 256],
            vf=[256, 256],
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=linear_schedule(INITIAL_LR),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        seed=int(seed),
        verbose=1,
        device=device,
        tensorboard_log=str(run_dir / "tb"),
    )

    return model


# =============================================================================
# One seed
# =============================================================================

def train_one_seed(
    *,
    seed: int,
    requested_steps: int,
    checkpoint_interval: int,
    root: Path,
    device: str,
    env_mode: str,
) -> Dict[str, Any]:
    run_dir = root / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    env = None
    model = None
    start = time.time()

    expected_final = (
        math.ceil(requested_steps / GLOBAL_ROLLOUT)
        * GLOBAL_ROLLOUT
    )

    run_manifest = {
        "seed": seed,
        "env_mode": env_mode,
        "requested_timesteps": requested_steps,
        "expected_rollout_aligned_final": expected_final,
        "formal_nominal_checkpoint": requested_steps,
        "device": device,
        "execution": {
            "vec_env": "SubprocVecEnv",
            "start_method": "spawn",
            "n_envs": N_ENVS,
            "n_steps_per_env": N_STEPS,
            "global_rollout": GLOBAL_ROLLOUT,
            "batch_size": BATCH_SIZE,
            "minibatches_per_epoch": GLOBAL_ROLLOUT // BATCH_SIZE,
        },
        "environment": {
            "passenger_file": str(TRAIN_FILE),
            "max_time": MAX_TIME,
            "candidate_from_vertiports": CANDIDATES,
            "to_vertiport": TO_VERTIPORT,
            "fleet_mode": (
                "conserved_closed_loop"
                if env_mode == "fixed"
                else "legacy_replenish"
            ),
            "fleet_size": 16 if env_mode == "fixed" else None,
            "expected_allocation": (
                {"0": 12, "1": 4}
                if env_mode == "fixed"
                else None
            ),
        },
        "ppo": {
            "learning_rate": "linear_schedule(5e-4)",
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip_range": CLIP_RANGE,
            "ent_coef": ENT_COEF,
            "vf_coef": VF_COEF,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "no_mid_training_evaluation": True,
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    write_json(run_dir / "run_manifest.json", run_manifest)

    try:
        if env_mode == "fixed":
            preflight = run_preflight_fixed(seed, run_dir)
            write_json(
                run_dir / "preflight_fixed_fleet.json",
                preflight,
            )

        env = build_train_env(
            seed=seed,
            run_dir=run_dir,
            env_mode=env_mode,
        )

        model = build_model(
            env=env,
            seed=seed,
            run_dir=run_dir,
            device=device,
        )

        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Callback frequency is counted in vector-env step calls.
        if checkpoint_interval % N_ENVS != 0:
            raise ValueError(
                f"{checkpoint_interval} must be divisible by "
                f"n_envs={N_ENVS} for exact total-step checkpoints."
            )

        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_interval // N_ENVS,
            save_path=str(checkpoint_dir),
            name_prefix="uam_ppo",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=0,
        )

        scalar_callback = PassiveTrainingScalarCallback(
            run_dir=run_dir,
            interval=checkpoint_interval,
            requested_steps=requested_steps,
        )

        print("=" * 124, flush=True)
        print(
            f"START | mode={env_mode.upper()} | seed={seed} | "
            f"requested={requested_steps:,}",
            flush=True,
        )
        print(
            f"FAST PATH: {N_ENVS} CPU env processes x {N_STEPS} "
            f"steps = {GLOBAL_ROLLOUT:,} global rollout | "
            f"batch={BATCH_SIZE} | GPU={model.device}",
            flush=True,
        )
        print(
            "NO MID-TRAINING EVALUATION / NO CHECKPOINT REPLAY",
            flush=True,
        )
        print("=" * 124, flush=True)

        model.learn(
            total_timesteps=requested_steps,
            callback=[checkpoint_callback, scalar_callback],
            progress_bar=False,
            reset_num_timesteps=True,
        )

        actual_steps = int(model.num_timesteps)

        # Rollout-aligned final alias.
        model.save(run_dir / "final_rl_model")
        env.save(run_dir / "final_vec_normalize.pkl")

        formal_model = (
            checkpoint_dir
            / f"uam_ppo_{requested_steps}_steps.zip"
        )
        formal_vec = (
            checkpoint_dir
            / f"uam_ppo_vecnormalize_{requested_steps}_steps.pkl"
        )

        if not formal_model.exists():
            raise FileNotFoundError(
                f"Missing nominal {requested_steps:,} checkpoint: "
                f"{formal_model}"
            )

        if not formal_vec.exists():
            raise FileNotFoundError(
                f"Missing nominal {requested_steps:,} VecNormalize: "
                f"{formal_vec}"
            )

        elapsed = time.time() - start

        result = {
            "status": "SUCCESS",
            "seed": seed,
            "env_mode": env_mode,
            "requested_timesteps": requested_steps,
            "actual_final_timesteps": actual_steps,
            "formal_checkpoint_timesteps": requested_steps,
            "formal_model": str(formal_model),
            "formal_vecnormalize": str(formal_vec),
            "elapsed_seconds": elapsed,
            "requested_effective_fps": (
                requested_steps / max(elapsed, 1e-9)
            ),
            "actual_effective_fps": (
                actual_steps / max(elapsed, 1e-9)
            ),
            "cuda_peak_memory_mb": (
                torch.cuda.max_memory_allocated()
                / (1024.0 * 1024.0)
                if device == "cuda" and torch.cuda.is_available()
                else 0.0
            ),
            "finished": datetime.now().isoformat(timespec="seconds"),
        }

        write_json(run_dir / "run_end.json", result)

        print(
            f"DONE seed={seed} | nominal={requested_steps:,} | "
            f"rollout-final={actual_steps:,} | "
            f"time={elapsed/60:.2f} min | "
            f"effective={result['requested_effective_fps']:.1f} FPS",
            flush=True,
        )

        return result

    except Exception as exc:
        elapsed = time.time() - start

        error = {
            "status": "ERROR",
            "seed": seed,
            "env_mode": env_mode,
            "requested_timesteps": requested_steps,
            "elapsed_seconds": elapsed,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "finished": datetime.now().isoformat(timespec="seconds"),
        }

        write_json(run_dir / "run_end.json", error)
        raise

    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass

        del model
        del env
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fast matched UAGMC trainer: 16 CPU envs + GPU PPO, "
            "3 seeds x nominal 1M, no mid-training eval"
        )
    )
    p.add_argument(
        "--env-mode",
        choices=["fixed", "legacy"],
        default="fixed",
        help=(
            "fixed = E1 conserved N=16; "
            "legacy = matched fast E0 replenishment control"
        ),
    )
    p.add_argument(
        "--seeds",
        default="0,1,2",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
    )
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
    )
    p.add_argument(
        "--output-root",
        default=None,
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    return p.parse_args()


def resolve_device(x: str) -> str:
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if x == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False"
        )

    return x


def main():
    args = parse_args()

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)

    seeds = parse_int_list(args.seeds)
    requested_steps = int(args.timesteps)
    checkpoint_interval = int(args.checkpoint_interval)
    device = resolve_device(args.device)

    if requested_steps <= 0:
        raise ValueError("timesteps must be positive")

    if checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")

    if requested_steps % checkpoint_interval != 0:
        raise ValueError(
            "timesteps must be divisible by checkpoint interval"
        )

    if checkpoint_interval % N_ENVS != 0:
        raise ValueError(
            f"checkpoint interval must be divisible by {N_ENVS}"
        )

    if args.output_root:
        root = Path(args.output_root).expanduser()
        if not root.is_absolute():
            root = (ROOT / root).resolve()
        else:
            root = root.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = (
            "E1_fixed16"
            if args.env_mode == "fixed"
            else "E0_legacy"
        )
        root = (
            ROOT
            / "serial_runs"
            / f"uagmc_{label}_fast16env_3seed_1M_{stamp}"
        ).resolve()

    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": (
            "UAGMC_E1_FIXED16_FAST"
            if args.env_mode == "fixed"
            else "UAGMC_E0_LEGACY_FAST_MATCHED"
        ),
        "env_mode": args.env_mode,
        "seeds": seeds,
        "requested_steps_per_seed": requested_steps,
        "requested_total_budget": requested_steps * len(seeds),
        "device": device,
        "execution": {
            "vec_env": "SubprocVecEnv",
            "start_method": "spawn",
            "n_envs": N_ENVS,
            "n_steps_per_env": N_STEPS,
            "global_rollout": GLOBAL_ROLLOUT,
            "batch_size": BATCH_SIZE,
        },
        "fairness": {
            "original_global_rollout_5x4096": 5 * 4096,
            "fast_global_rollout_16x1280": GLOBAL_ROLLOUT,
            "same_global_rollout": (5 * 4096 == GLOBAL_ROLLOUT),
            "no_mid_training_evaluation": True,
            "strict_comparison_note": (
                "Use this same fast configuration for both E0 and E1 "
                "because 16x1280 changes trajectory-fragment partition "
                "relative to 5x4096."
            ),
        },
        "checkpoint_interval": checkpoint_interval,
        "passenger_file": str(TRAIN_FILE),
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    write_json(root / "experiment_manifest.json", manifest)

    print("=" * 124)
    print("UAGMC FAST FORMAL TRAINING")
    print("=" * 124)
    print(f"Mode              : {args.env_mode}")
    print(f"Seeds             : {seeds}")
    print(f"Nominal / seed    : {requested_steps:,}")
    print(f"Requested total   : {requested_steps * len(seeds):,}")
    print(
        f"Execution         : {N_ENVS} Subproc CPU envs x "
        f"{N_STEPS} = {GLOBAL_ROLLOUT:,}/rollout"
    )
    print(f"Batch             : {BATCH_SIZE}")
    print(f"Device            : {device}")
    print("Mid-train eval    : NONE")
    print(f"Output            : {root}")
    print("=" * 124)

    status_path = root / "serial_status.csv"

    success = 0
    failed = 0

    for idx, seed in enumerate(seeds, start=1):
        print(
            f"\n### seed {seed} ({idx}/{len(seeds)}) ###",
            flush=True,
        )

        try:
            result = train_one_seed(
                seed=seed,
                requested_steps=requested_steps,
                checkpoint_interval=checkpoint_interval,
                root=root,
                device=device,
                env_mode=args.env_mode,
            )

            append_csv(
                status_path,
                {
                    "seed": seed,
                    "status": "SUCCESS",
                    "env_mode": args.env_mode,
                    "requested_timesteps": requested_steps,
                    "actual_final_timesteps": result[
                        "actual_final_timesteps"
                    ],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "requested_effective_fps": result[
                        "requested_effective_fps"
                    ],
                },
            )
            success += 1

        except Exception as exc:
            append_csv(
                status_path,
                {
                    "seed": seed,
                    "status": "ERROR",
                    "env_mode": args.env_mode,
                    "requested_timesteps": requested_steps,
                    "actual_final_timesteps": "",
                    "elapsed_seconds": "",
                    "requested_effective_fps": "",
                    "error": repr(exc),
                },
            )

            failed += 1
            print(traceback.format_exc(), flush=True)

            if not args.continue_on_error:
                raise

    write_json(
        root / "experiment_end.json",
        {
            "status": (
                "SUCCESS" if failed == 0 else "PARTIAL_ERROR"
            ),
            "successes": success,
            "failures": failed,
            "env_mode": args.env_mode,
            "seeds": seeds,
            "requested_steps_per_seed": requested_steps,
            "finished": datetime.now().isoformat(timespec="seconds"),
        },
    )

    print("\n" + "=" * 124)
    print("TRAINING MATRIX FINISHED")
    print("=" * 124)
    print(f"Success : {success}/{len(seeds)}")
    print(f"Root    : {root}")
    print(
        "No evaluation was run during training. "
        "Evaluate checkpoints only AFTER all seeds finish."
    )
    print("=" * 124)

    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
