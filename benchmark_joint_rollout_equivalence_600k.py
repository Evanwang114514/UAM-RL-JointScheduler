# -*- coding: utf-8 -*-
"""
PPO rollout-length / parallel-env learning-equivalence experiment.

Question
--------
With the SAME PPO, global rollout size, batch size, reward, observation,
action space, environment physics, architecture, seed, and total training
budget, does increasing n_envs (and therefore shortening n_steps per env)
change learning quality?

Fixed experiment
----------------
Environment     : J1 (ACTION_BRANCHING)
Representation  : R_TDM
Device          : CUDA
Train seed      : 1
Eval seeds      : 123,124,125
Batch size      : 4096
Global rollout  : 20,480 transitions/update
Training budget : 600,000 nominal steps/profile
Checkpoint      : every nominal 50k, saved at first vector step >= mark
Evaluation      : every saved checkpoint x 3 eval seeds
Info mode       : FULL (no slimming)

Profiles
--------
P40   :  40 env x 512 steps = 20,480
P128  : 128 env x 160 steps = 20,480
P160  : 160 env x 128 steps = 20,480

Outputs
-------
serial_runs/rollout_equivalence_<timestamp>/
    experiment_manifest.json
    profile_summary.csv
    profile_checkpoint_curves.csv
    profile_checkpoint_diagnostics.csv
    P40/J1__R_TDM/...
    P128/J1__R_TDM/...
    P160/J1__R_TDM/...

Each profile directory contains:
    checkpoints/uam_ppo_<nominal>_steps.zip
    checkpoints/uam_ppo_vecnormalize_<nominal>_steps.pkl
    training_throughput.csv
    run_manifest.json
    run_end.json
    analysis/checkpoint_eval_raw.csv
    analysis/checkpoint_curve.csv
    analysis/cell_summary.json

Run
---
python benchmark_joint_rollout_equivalence_600k.py

Subset
------
python benchmark_joint_rollout_equivalence_600k.py --profiles P40,P160

Resume / evaluate missing work
------------------------------
python benchmark_joint_rollout_equivalence_600k.py --resume-root "serial_runs/rollout_equivalence_<timestamp>"

Notes
-----
- This is NOT a speed-only benchmark. It is a learning-quality control.
- n_envs changes the number of independent trajectory chunks per PPO update.
- n_steps changes the temporal length of each chunk.
- global rollout stays fixed at 20,480, so each PPO update uses the same number
  of transitions and the same batch size.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

import train_uam_60m_literature_matrix_v3_1 as v3


ROOT = Path(__file__).resolve().parent

ENV_KEY = "J1"
METHOD_ID = "R_TDM"
TRAIN_SEED = 1
EVAL_SEEDS = (123, 124, 125)

DEFAULT_TIMESTEPS = 600_000
CHECKPOINT_INTERVAL = 50_000
MAX_TIME = 2_500
BATCH_SIZE = 4_096
GLOBAL_ROLLOUT = 20_480

PROFILES: Dict[str, v3.SpeedProfile] = {
    "P40": v3.SpeedProfile(
        "P40_40x512_b4096",
        n_envs=40,
        n_steps=512,
        batch_size=BATCH_SIZE,
    ),
    "P128": v3.SpeedProfile(
        "P128_128x160_b4096",
        n_envs=128,
        n_steps=160,
        batch_size=BATCH_SIZE,
    ),
    "P160": v3.SpeedProfile(
        "P160_160x128_b4096",
        n_envs=160,
        n_steps=128,
        batch_size=BATCH_SIZE,
    ),
}

for _name, _profile in PROFILES.items():
    assert _profile.rollout == GLOBAL_ROLLOUT, (_name, _profile.rollout)
    assert GLOBAL_ROLLOUT % _profile.batch_size == 0


def truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes"}


class NominalCheckpointAndDiagnostics(BaseCallback):
    """
    Save checkpoints at nominal 50k labels even when n_envs does not divide
    50,000. The actual save occurs at the first vector step >= nominal mark.

    Also records the most recently available SB3 train/* diagnostics. These
    values correspond to the latest completed PPO update visible to the logger.
    """

    def __init__(
        self,
        run_dir: Path,
        profile_name: str,
        interval: int = CHECKPOINT_INTERVAL,
    ):
        super().__init__(verbose=0)
        self.run_dir = Path(run_dir)
        self.profile_name = str(profile_name)
        self.interval = int(interval)
        self.next_mark = int(interval)
        self.started = 0.0
        self.rows: List[Dict[str, Any]] = []

    def _on_training_start(self) -> None:
        self.started = time.perf_counter()
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _logger_value(self, key: str) -> float:
        try:
            value = self.model.logger.name_to_value.get(key, float("nan"))
            return float(value)
        except Exception:
            return float("nan")

    def _save_mark(self, mark: int) -> None:
        ck = self.run_dir / "checkpoints"

        # v3.analyze_cell expects these exact nominal names.
        self.model.save(str(ck / f"uam_ppo_{mark}_steps.zip"))
        vn = self.model.get_vec_normalize_env()
        if vn is None:
            raise RuntimeError("VecNormalize environment not found")
        vn.save(str(ck / f"uam_ppo_vecnormalize_{mark}_steps.pkl"))

        elapsed = time.perf_counter() - self.started
        row = {
            "profile": self.profile_name,
            "nominal_timesteps": int(mark),
            "actual_timesteps": int(self.num_timesteps),
            "checkpoint_overshoot": int(self.num_timesteps - mark),
            "elapsed_sec": float(elapsed),
            "sps": float(self.num_timesteps / max(elapsed, 1e-9)),
            "train_loss": self._logger_value("train/loss"),
            "policy_gradient_loss": self._logger_value("train/policy_gradient_loss"),
            "value_loss": self._logger_value("train/value_loss"),
            "entropy_loss": self._logger_value("train/entropy_loss"),
            "approx_kl": self._logger_value("train/approx_kl"),
            "clip_fraction": self._logger_value("train/clip_fraction"),
            "explained_variance": self._logger_value("train/explained_variance"),
        }
        self.rows.append(row)
        v3.write_csv(self.run_dir / "training_throughput.csv", self.rows)

        print(
            f"[{self.profile_name} CKPT] nominal={mark:,} "
            f"actual={self.num_timesteps:,} "
            f"overshoot={self.num_timesteps-mark} "
            f"SPS={row['sps']:.1f}",
            flush=True,
        )

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_mark:
            self._save_mark(self.next_mark)
            self.next_mark += self.interval
        return True


def profile_run_dir(root: Path, profile_id: str) -> Path:
    return root / profile_id / v3.cell_id(ENV_KEY, METHOD_ID)


def training_complete(run_dir: Path, requested_steps: int) -> bool:
    p = run_dir / "run_end.json"
    final_model = run_dir / "final_rl_model.zip"
    final_vec = run_dir / "final_vec_normalize.pkl"
    if not p.exists() or not final_model.exists() or not final_vec.exists():
        return False
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return (
            str(obj.get("status", "")).upper() in {"TRAINED", "SUCCESS"}
            and int(obj.get("requested_timesteps", -1)) == int(requested_steps)
        )
    except Exception:
        return False


def analysis_complete(run_dir: Path) -> bool:
    return (run_dir / "analysis" / "cell_summary.json").exists()


def train_profile(
    *,
    root: Path,
    profile_id: str,
    profile: v3.SpeedProfile,
    requested_steps: int,
    device: str,
    max_time: int,
) -> Dict[str, Any]:
    run_dir = profile_run_dir(root, profile_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if training_complete(run_dir, requested_steps):
        print(f"[SKIP TRAINED] {profile_id}", flush=True)
        return json.loads((run_dir / "run_end.json").read_text(encoding="utf-8"))

    v3.seed_all(TRAIN_SEED)
    env = None
    model = None
    callback = None
    started = time.perf_counter()

    try:
        env = v3.build_vec_env(
            env_key=ENV_KEY,
            method_id=METHOD_ID,
            profile=profile,
            seed=TRAIN_SEED,
            run_dir=run_dir,
            max_time=max_time,
        )

        model = v3.build_model(
            env=env,
            env_key=ENV_KEY,
            method_id=METHOD_ID,
            profile=profile,
            seed=TRAIN_SEED,
            run_dir=run_dir,
            device=device,
        )

        callback = NominalCheckpointAndDiagnostics(
            run_dir=run_dir,
            profile_name=profile_id,
            interval=CHECKPOINT_INTERVAL,
        )

        manifest = {
            "profile_id": profile_id,
            "profile": asdict(profile),
            "env_key": ENV_KEY,
            "method_id": METHOD_ID,
            "train_seed": TRAIN_SEED,
            "requested_timesteps": int(requested_steps),
            "global_rollout": int(profile.rollout),
            "batch_size": int(profile.batch_size),
            "device": str(model.device),
            "policy_params": v3.count_params(model),
            "hard_guard": int(max_time),
            "checkpoint_interval_nominal": CHECKPOINT_INTERVAL,
            "full_info": True,
            "scientific_question": (
                "learning equivalence under fixed global rollout while changing "
                "n_envs and per-env rollout horizon n_steps"
            ),
        }
        v3.write_json(run_dir / "run_manifest.json", manifest)

        print("\n" + "=" * 120)
        print(
            f"START {profile_id} | "
            f"envs={profile.n_envs} x n_steps={profile.n_steps} | "
            f"rollout={profile.rollout} | batch={profile.batch_size} | "
            f"requested={requested_steps:,} | device={model.device}",
            flush=True,
        )
        print("=" * 120, flush=True)

        model.learn(
            total_timesteps=int(requested_steps),
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=True,
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - started

        model.save(run_dir / "final_rl_model")
        env.save(run_dir / "final_vec_normalize.pkl")

        end = {
            "status": "TRAINED",
            "profile_id": profile_id,
            "requested_timesteps": int(requested_steps),
            "actual_timesteps": int(model.num_timesteps),
            "elapsed_sec": float(elapsed),
            "sps": float(model.num_timesteps / max(elapsed, 1e-9)),
            "n_envs": int(profile.n_envs),
            "n_steps": int(profile.n_steps),
            "global_rollout": int(profile.rollout),
            "batch_size": int(profile.batch_size),
        }
        v3.write_json(run_dir / "run_end.json", end)
        return end

    except Exception:
        err = traceback.format_exc()
        v3.write_json(
            run_dir / "run_end.json",
            {
                "status": "TRAIN_FAILED",
                "profile_id": profile_id,
                "requested_timesteps": int(requested_steps),
                "error": err,
            },
        )
        raise

    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        model = None
        env = None
        callback = None
        try:
            v3.core.restore_process_patches()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def evaluate_profile(
    *,
    root: Path,
    profile_id: str,
    requested_steps: int,
    eval_seeds: Sequence[int],
    max_time: int,
) -> Dict[str, Any]:
    run_dir = profile_run_dir(root, profile_id)

    if analysis_complete(run_dir):
        print(f"[SKIP ANALYZED] {profile_id}", flush=True)
        return json.loads(
            (run_dir / "analysis" / "cell_summary.json").read_text(encoding="utf-8")
        )

    print(
        f"\n[EVAL START] {profile_id} | "
        f"checkpoints=50k..{requested_steps//1000}k | seeds={list(eval_seeds)}",
        flush=True,
    )

    return v3.analyze_cell(
        env_key=ENV_KEY,
        method_id=METHOD_ID,
        run_dir=run_dir,
        requested_steps=int(requested_steps),
        eval_seeds=tuple(int(x) for x in eval_seeds),
        max_time=int(max_time),
    )


def combine_results(
    *,
    root: Path,
    selected_profiles: Sequence[str],
    requested_steps: int,
) -> None:
    combined_curve: List[Dict[str, Any]] = []
    combined_diag: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for pid in selected_profiles:
        profile = PROFILES[pid]
        run_dir = profile_run_dir(root, pid)

        curve = v3.read_csv(run_dir / "analysis" / "checkpoint_curve.csv")
        for row in curve:
            row = dict(row)
            row["profile_id"] = pid
            row["n_envs"] = profile.n_envs
            row["n_steps"] = profile.n_steps
            row["global_rollout"] = profile.rollout
            combined_curve.append(row)

        diag = v3.read_csv(run_dir / "training_throughput.csv")
        for row in diag:
            row = dict(row)
            row["profile_id"] = pid
            row["n_envs"] = profile.n_envs
            row["n_steps"] = profile.n_steps
            combined_diag.append(row)

        end_path = run_dir / "run_end.json"
        end = (
            json.loads(end_path.read_text(encoding="utf-8"))
            if end_path.exists()
            else {}
        )

        valid_curve = []
        for row in curve:
            att = v3.fnum(row.get("ATT_mean"))
            if truthy(row.get("all_full_completion")) and math.isfinite(att):
                valid_curve.append(row)

        valid_curve.sort(key=lambda x: int(float(x.get("train_step", 0))))

        best_att = float("nan")
        best_step = -1
        final_att = float("nan")
        final_completion = float("nan")
        late3_att = float("nan")
        collapse = float("nan")
        mean_dispatch_success = float("nan")
        mean_dispatch_infeasible = float("nan")

        if valid_curve:
            best = min(valid_curve, key=lambda x: v3.fnum(x.get("ATT_mean")))
            best_att = v3.fnum(best.get("ATT_mean"))
            best_step = int(float(best.get("train_step", -1)))
            final = valid_curve[-1]
            final_att = v3.fnum(final.get("ATT_mean"))
            final_completion = v3.fnum(final.get("completion_rate_mean"))
            late3_att = v3.fmean(
                x.get("ATT_mean") for x in valid_curve[-3:]
            )
            collapse = final_att - best_att
            mean_dispatch_success = v3.fmean(
                x.get("dispatch_success_rate_mean") for x in valid_curve
            )
            mean_dispatch_infeasible = v3.fmean(
                x.get("dispatch_infeasible_rate_mean") for x in valid_curve
            )
        elif curve:
            final = sorted(
                curve,
                key=lambda x: int(float(x.get("train_step", 0))),
            )[-1]
            final_completion = v3.fnum(final.get("completion_rate_mean"))

        summary_rows.append(
            {
                "profile_id": pid,
                "n_envs": profile.n_envs,
                "n_steps": profile.n_steps,
                "global_rollout": profile.rollout,
                "batch_size": profile.batch_size,
                "requested_timesteps": requested_steps,
                "actual_timesteps": end.get("actual_timesteps", ""),
                "train_elapsed_sec": end.get("elapsed_sec", ""),
                "train_sps": end.get("sps", ""),
                "n_curve_points": len(curve),
                "n_valid_full_completion_points": len(valid_curve),
                "best_ATT": best_att,
                "best_step": best_step,
                "final_ATT": final_att,
                "late3_ATT": late3_att,
                "collapse_final_minus_best": collapse,
                "final_completion_rate": final_completion,
                "mean_dispatch_success_rate": mean_dispatch_success,
                "mean_dispatch_infeasible_rate": mean_dispatch_infeasible,
            }
        )

    v3.write_csv(root / "profile_checkpoint_curves.csv", combined_curve)
    v3.write_csv(root / "profile_checkpoint_diagnostics.csv", combined_diag)
    v3.write_csv(root / "profile_summary.csv", summary_rows)

    print("\n" + "#" * 120)
    print("PROFILE SUMMARY")
    print("#" * 120)
    for row in summary_rows:
        print(
            f"{row['profile_id']:>4} | "
            f"envs={row['n_envs']:>3} steps={row['n_steps']:>3} | "
            f"SPS={v3.fnum(row.get('train_sps')):>7.1f} | "
            f"bestATT={v3.fnum(row.get('best_ATT')):>8.3f} | "
            f"finalATT={v3.fnum(row.get('final_ATT')):>8.3f} | "
            f"late3={v3.fnum(row.get('late3_ATT')):>8.3f} | "
            f"completion={v3.fnum(row.get('final_completion_rate')):>6.3f} | "
            f"collapse={v3.fnum(row.get('collapse_final_minus_best')):>8.3f}",
            flush=True,
        )

    print(f"\nCombined summary : {root / 'profile_summary.csv'}")
    print(f"Combined curves  : {root / 'profile_checkpoint_curves.csv'}")
    print(f"Train diagnostics: {root / 'profile_checkpoint_diagnostics.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPO parallel-env / rollout-horizon learning-equivalence experiment"
    )
    p.add_argument(
        "--profiles",
        default="P40,P128,P160",
        help="Comma-separated subset of P40,P128,P160.",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Nominal training steps per profile; default 600000.",
    )
    p.add_argument(
        "--eval-seeds",
        default="123,124,125",
        help="Comma-separated eval seeds.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-time", type=int, default=MAX_TIME)
    p.add_argument(
        "--resume-root",
        default="",
        help="Existing experiment root to resume; otherwise creates a new one.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if str(args.device).lower().startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    selected = [
        x.strip().upper()
        for x in str(args.profiles).split(",")
        if x.strip()
    ]
    bad = [x for x in selected if x not in PROFILES]
    if bad:
        raise SystemExit(f"Unknown profiles {bad}; allowed={list(PROFILES)}")

    eval_seeds = tuple(
        int(x.strip())
        for x in str(args.eval_seeds).split(",")
        if x.strip()
    )

    requested_steps = int(args.timesteps)
    if requested_steps < CHECKPOINT_INTERVAL:
        raise SystemExit(
            f"--timesteps must be >= {CHECKPOINT_INTERVAL}"
        )

    if args.resume_root:
        root = Path(args.resume_root).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = ROOT / "serial_runs" / f"rollout_equivalence_{stamp}"

    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "parallel_env_rollout_horizon_learning_equivalence",
        "env_key": ENV_KEY,
        "method_id": METHOD_ID,
        "train_seed": TRAIN_SEED,
        "eval_seeds": list(eval_seeds),
        "requested_timesteps_per_profile": requested_steps,
        "checkpoint_interval_nominal": CHECKPOINT_INTERVAL,
        "batch_size": BATCH_SIZE,
        "global_rollout": GLOBAL_ROLLOUT,
        "device": args.device,
        "max_time": int(args.max_time),
        "profiles": {
            k: asdict(PROFILES[k])
            for k in selected
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "controlled_variables": [
            "same J1 environment",
            "same R_TDM representation",
            "same PPO implementation",
            "same train seed",
            "same global rollout=20480",
            "same batch=4096",
            "same reward/obs/action/physics",
            "same FULL info behavior",
        ],
        "changed_variables": [
            "n_envs",
            "n_steps per environment",
        ],
    }
    v3.write_json(root / "experiment_manifest.json", manifest)

    print("\nPPO ROLLOUT-HORIZON LEARNING-EQUIVALENCE EXPERIMENT")
    print(f"Root       : {root}")
    print(f"GPU        : {manifest['gpu_name']}")
    print(f"Profiles   : {selected}")
    print(f"Per profile: {requested_steps:,} nominal training steps")
    print(f"Eval seeds : {eval_seeds}")
    print(f"Rollout    : {GLOBAL_ROLLOUT:,} transitions/update")
    print(f"Batch      : {BATCH_SIZE:,}")
    print("FULL INFO  : YES", flush=True)

    for pid in selected:
        profile = PROFILES[pid]
        try:
            train_profile(
                root=root,
                profile_id=pid,
                profile=profile,
                requested_steps=requested_steps,
                device=str(args.device),
                max_time=int(args.max_time),
            )
        except Exception as exc:
            print(
                f"[TRAIN FAILED] {pid}: {exc!r}\n{traceback.format_exc()}",
                flush=True,
            )
            continue

        try:
            evaluate_profile(
                root=root,
                profile_id=pid,
                requested_steps=requested_steps,
                eval_seeds=eval_seeds,
                max_time=int(args.max_time),
            )
        except Exception as exc:
            print(
                f"[EVAL FAILED] {pid}: {exc!r}\n{traceback.format_exc()}",
                flush=True,
            )

    combine_results(
        root=root,
        selected_profiles=selected,
        requested_steps=requested_steps,
    )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
