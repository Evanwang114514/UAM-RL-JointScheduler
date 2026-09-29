# -*- coding: utf-8 -*-
"""
PRE-5B 80M AUTO FUNNEL
======================

One-shot pre-500M experiment driver for the UAM project.

Design philosophy
-----------------
The ~80M budget is NOT another method zoo. It removes lower-level confounders
before the later ~500M campaign, where the budget should mainly be used for:
  (1) adding/upgrading literature-inspired methods;
  (2) deepening the strongest existing method families.

The phases are strictly inherited:
  P1  PPO production profile selection         10.8M
  P2  S3 demand-load operating-point selection 13.0M
  P3  Joint architecture selection             10.8M
  P4  matched Single/Joint reversal analysis    9.6M
  P5  selected methods: long-run + fresh seeds 27.0M
  P6  final two-method confirmation              8.8M
                                                ------
  TOTAL                                         80.0M

Each phase writes a selection JSON used automatically by the next phase.

Wall-clock design
-----------------
* Production runs use 40 envs.
* A single 10-env profile is retained only as a diagnostic in P1.
* Intermediate phases use sparse checkpoint evaluation.
* All underlying 50k checkpoints are still saved by the validated project code,
  so full formal evaluation can be performed offline after this 80M funnel.
* No method-specific PPO tuning is performed.

Expected rough wall time
------------------------
The code only prints an ETA estimate. With the historical/current rough rates:
  P40 fast  ~2300 SPS
  P40 dense ~1800 SPS
  P10 ref    ~630 SPS
the pure training portion should normally fit comfortably inside ~20 h.
Actual speed depends on server load and the selected P40 profile.

Required repository files
-------------------------
train_uam_nextgen_100m_matrix.py
train_uam_60m_literature_matrix_v3_1.py
train_uam_60m_jointfirst_v5_minimal_reposition.py
and their already validated dependencies.

Run
---
CUDA_VISIBLE_DEVICES=0 python train_uam_pre5b_80m_autofunnel.py

Background
----------
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python train_uam_pre5b_80m_autofunnel.py' \
  > /home/SY/wyw/logs/pre5b_80m.log 2>&1 &

Resume
------
CUDA_VISIBLE_DEVICES=0 python train_uam_pre5b_80m_autofunnel.py \
  --resume-root serial_runs/uam_pre5b80m_YYYYMMDD_HHMMSS

Plan only
---------
python train_uam_pre5b_80m_autofunnel.py --plan-only
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import train_uam_nextgen_100m_matrix as ng

v3 = ng.v3
v4 = ng.v4
v5 = ng.v5

ROOT = Path(__file__).resolve().parent
BASE_TRACE = ROOT / "train_data" / "passengers_300.csv"
MAX_TIME = 2500

# Fast sparse evaluation during the 80M funnel.
FAST_EVAL_SEEDS = (123,)
LONG_EVAL_SEEDS = (123, 124)

# =============================================================================
# EXACT BUDGET
# =============================================================================

P1_BUDGET = 10_800_000
P2_BUDGET = 13_000_000
P3_BUDGET = 10_800_000
P4_BUDGET = 9_600_000
P5_BUDGET = 27_000_000
P6_BUDGET = 8_800_000

TOTAL_BUDGET = (
    P1_BUDGET + P2_BUDGET + P3_BUDGET
    + P4_BUDGET + P5_BUDGET + P6_BUDGET
)
assert TOTAL_BUDGET == 80_000_000

# =============================================================================
# PROFILES
# =============================================================================

# Historical production configuration.
P40_FAST = v3.SpeedProfile(
    "P40_FAST_40x512_b4096",
    40, 512, 4096,
)

# Same rollout, much denser PPO updates.
P40_DENSE = v3.SpeedProfile(
    "P40_DENSE_40x512_b1024",
    40, 512, 1024,
)

# Diagnostic reference only. It can NEVER become the production profile.
P10_REF = v3.SpeedProfile(
    "P10_REF_10x2048_b1024",
    10, 2048, 1024,
)

PROFILES: Dict[str, Any] = {
    P40_FAST.name: P40_FAST,
    P40_DENSE.name: P40_DENSE,
    P10_REF.name: P10_REF,
}

P40_CANDIDATES = (
    P40_FAST.name,
    P40_DENSE.name,
)

# Used only to print ETA and as a very low-priority tie breaker.
EXPECTED_SPS = {
    P40_FAST.name: 2300.0,
    P40_DENSE.name: 1800.0,
    P10_REF.name: 630.0,
}

# =============================================================================
# PHASE DEFINITIONS
# =============================================================================

# P1 = 3 methods x 3 profiles x 3 seeds x 400k = 10.8M
P1_METHODS = (
    "UAGMC_SOURCE",
    "A_ANCHOR_QUOTIENT",
    "W_DREAMER_BALANCED",
)
P1_SEEDS = (1, 2, 3)
P1_STEPS = 400_000

# P2:
# 5 loads x [
#   UAGMC_SOURCE: 3 x 400k = 1.2M
#   CURRENT:      2 x 300k = 0.6M
#   A_ANCHOR:     2 x 400k = 0.8M
# ] = 13.0M
LOADS = (0.80, 0.85, 0.90, 0.95, 1.00)
P2_UAGMC_SEEDS = (11, 12, 13)
P2_CURRENT_SEEDS = (21, 22)
P2_ANCHOR_SEEDS = (31, 32)

# P3 = 3 methods x 3 Joint architectures x 3 seeds x 400k = 10.8M
# All three use the SAME V5 clean minimal Joint physical wrapper.
P3_METHODS = (
    "CURRENT",
    "S_TDM_EVENT_FUSION",
    "A_QPLEX_DUPLEX",
)
P3_JOINT_ENVS = ("J0", "J4", "J7")
P3_SEEDS = (41, 42, 43)
P3_STEPS = 400_000

# P4 = 6 methods x 2 regimes x 2 seeds x 400k = 9.6M
P4_METHODS = (
    "CURRENT",
    "S_TDM_EVENT_FUSION",
    "S_GATV2_EVENT",
    "A_ANCHOR_QUOTIENT",
    "W_DREAMER_BALANCED",
    "SA_TDMFUSION_ICM",
)
P4_SEEDS = (51, 52)
P4_STEPS = 400_000

# P5 = 6 selected method-regime pairs x 3 seeds x 1.5M = 27M
P5_SEEDS = (61, 62, 63)
P5_STEPS = 1_500_000

# P6 = 2 selected pairs x 4 seeds x 1.1M = 8.8M
P6_SEEDS = (71, 72, 73, 74)
P6_STEPS = 1_100_000

# =============================================================================
# GENERIC HELPERS
# =============================================================================

def canonical(x: str) -> str:
    return str(x).strip().upper()


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def fmean(xs: Iterable[Any], default: float = float("nan")) -> float:
    vals = [fnum(x) for x in xs]
    vals = [x for x in vals if math.isfinite(x)]
    return float(np.mean(vals)) if vals else default


def fstd(xs: Iterable[Any], default: float = float("nan")) -> float:
    vals = [fnum(x) for x in xs]
    vals = [x for x in vals if math.isfinite(x)]
    return float(np.std(vals, ddof=0)) if vals else default


def fmedian(xs: Iterable[Any], default: float = float("nan")) -> float:
    vals = [fnum(x) for x in xs]
    vals = [x for x in vals if math.isfinite(x)]
    return float(np.median(vals)) if vals else default


def safe_ratio(a: float, b: float, default: float = 999.0) -> float:
    if not finite(a) or not finite(b) or abs(float(b)) < 1e-9:
        return default
    return float(a) / float(b)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            cooked = {}
            for k in fields:
                v = row.get(k, "")
                if isinstance(v, (dict, list, tuple, np.ndarray)):
                    v = json.dumps(v, ensure_ascii=False, default=str)
                cooked[k] = v
            w.writerow(cooked)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def phase_banner(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120, flush=True)


def safe_slug(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


# =============================================================================
# DETERMINISTIC NESTED LOAD TRACES
# =============================================================================

def make_nested_load_traces(out_dir: Path) -> Dict[str, str]:
    """
    Build deterministic nested subsets of passengers_300.csv:
        D80 subset D85 subset D90 subset D95 subset D100

    No method result is used to construct the traces.
    """
    if not BASE_TRACE.exists():
        raise FileNotFoundError(BASE_TRACE)

    out_dir.mkdir(parents=True, exist_ok=True)

    with BASE_TRACE.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"empty passenger trace: {BASE_TRACE}")

    fields = list(rows[0].keys())

    scored: List[Tuple[float, Dict[str, str]]] = []
    for i, row in enumerate(rows):
        payload = (
            f"PRE5B80|{i}|"
            + "|".join(str(row.get(k, "")) for k in fields)
        )
        h = hashlib.sha256(payload.encode("utf-8")).digest()
        u = int.from_bytes(h[:8], "big") / float(2**64 - 1)
        scored.append((u, row))

    mapping: Dict[str, str] = {}
    manifest = []

    for rho in LOADS:
        key = f"{int(round(rho * 100)):03d}"
        path = out_dir / f"passengers_load_{key}.csv"

        if rho >= 0.999999:
            kept = [row for _, row in scored]
        else:
            kept = [row for u, row in scored if u < rho]

        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(kept)

        mapping[key] = str(path.resolve())

        manifest.append({
            "load": float(rho),
            "key": key,
            "n_rows": len(kept),
            "base_rows": len(rows),
            "fraction": len(kept) / max(1, len(rows)),
            "path": str(path.resolve()),
        })

    write_csv(out_dir / "load_trace_manifest.csv", manifest)
    write_json(out_dir / "load_trace_manifest.json", {
        "base_trace": str(BASE_TRACE.resolve()),
        "construction": (
            "deterministic nested SHA256 row thinning; "
            "100% keeps the full base trace"
        ),
        "loads": manifest,
    })

    return mapping


def set_trace(path: str | Path) -> None:
    """
    Set training/evaluation trace in the main process.
    Spawned workers set the same value again inside master_make_env_factory().
    """
    p = Path(path).resolve()
    os.environ["UAM_TRAIN_FILE"] = str(p)

    for mod in (
        v3,
        getattr(v3, "legacy", None),
        getattr(v3, "core", None),
    ):
        if mod is not None and hasattr(mod, "TRAIN_FILE"):
            setattr(mod, "TRAIN_FILE", p)


# =============================================================================
# MASTER ENVIRONMENT FACTORY
# =============================================================================

def master_make_env_factory(
    *,
    env_key: str,
    method_id: str,
    env_index: int,
    run_dir: Path,
    max_time: int,
):
    """
    Same validated S3/J physics as NextGen+V5, but with an explicit
    passenger-trace override that survives SubprocVecEnv spawn.
    """
    spec = v3.ENV_SPECS[str(env_key).upper()]

    def _init():
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        trace_path = Path(
            os.environ.get("UAM_TRAIN_FILE", str(BASE_TRACE))
        ).resolve()

        for mod in (
            v3,
            getattr(v3, "legacy", None),
            getattr(v3, "core", None),
        ):
            if mod is not None and hasattr(mod, "TRAIN_FILE"):
                setattr(mod, "TRAIN_FILE", trace_path)

        v3.configure_worker_physics(int(max_time))

        env = v3.core.make_experiment_env_factory(
            stage=spec.physical_stage,
            topology=v3.TOPOLOGY,
            encoder_mode="uagmc",
            fleet_size=v3.FLEET_SIZE,
            env_index=int(env_index),
            run_dir=run_dir,
            pad_separation=float(spec.pad_separation),
            charger_capacity=int(spec.charger_capacity),
            max_time=int(max_time),
        )()

        # All Joint envs use the same V5 clean minimal physical semantics.
        if spec.joint:
            env = v5.CleanMinimalJointWrapper(env, spec.key)

        env = v3.LiteratureObservationWrapper(
            env,
            env_key=spec.key,
            method_id=method_id,
        )
        return env

    return _init


# =============================================================================
# UAGMC_SOURCE COMMON-PROFILE BASELINE
# =============================================================================

class SourceUAGMCExtractor(BaseFeaturesExtractor):
    """
    Reads ONLY the original 6-frame base slice from the current observation.
    It intentionally ignores resource/tau/event/future branches.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        layout: Optional[Dict[str, Any]] = None,
        method_id: str = "UAGMC_SOURCE",
        env_key: str = "S3",
    ):
        if layout is None:
            raise ValueError("layout required")

        super().__init__(observation_space, features_dim)

        self.slices = dict(layout["slices"])

        self.frame_encoder = nn.Linear(
            v3.SINGLE_FRAME_DIM,
            int(features_dim),
        )
        self.lstm = nn.LSTM(
            input_size=int(features_dim),
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.output_proj = nn.Linear(
            128,
            int(features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        lo, hi = self.slices["base"]
        base = observations.float()[:, int(lo):int(hi)]

        frames = base.reshape(
            base.shape[0],
            v3.NUM_FRAMES,
            v3.SINGLE_FRAME_DIM,
        )

        x = self.frame_encoder(frames)
        x, _ = self.lstm(x)
        return self.output_proj(x[:, -1, :])


def build_uagmc_source_model(
    *,
    env,
    env_key: str,
    method_id: str,
    profile,
    seed: int,
    run_dir: Path,
    device: str,
):
    spec = v3.ENV_SPECS[str(env_key).upper()]

    policy_kwargs: Dict[str, Any] = dict(
        features_extractor_class=SourceUAGMCExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            layout=v3.GLOBAL_LAYOUT,
            method_id="UAGMC_SOURCE",
            env_key=env_key,
        ),
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256],
        ),
    )

    policy: Any = "MlpPolicy"

    if spec.joint:
        policy = v4.DiscoveryJointPolicy
        policy_kwargs["joint_arch"] = str(env_key).upper()

    return PPO(
        policy=policy,
        env=env,
        learning_rate=v3.base.linear_schedule(v3.INITIAL_LR),
        n_steps=int(profile.n_steps),
        batch_size=int(profile.batch_size),
        n_epochs=int(v3.N_EPOCHS),
        gamma=float(v3.GAMMA),
        gae_lambda=float(v3.GAE_LAMBDA),
        clip_range=float(v3.CLIP_RANGE),
        ent_coef=float(v3.ENT_COEF),
        vf_coef=float(v3.VF_COEF),
        max_grad_norm=float(v3.MAX_GRAD_NORM),
        policy_kwargs=policy_kwargs,
        seed=int(seed),
        verbose=1,
        device=device,
        tensorboard_log=str(run_dir / "tb"),
    )


_ORIGINAL_NEXTGEN_BUILD_MODEL = ng.nextgen_build_model


def master_build_model(
    *,
    env,
    env_key: str,
    method_id: str,
    profile,
    seed: int,
    run_dir: Path,
    device: str,
):
    if canonical(method_id) == "UAGMC_SOURCE":
        return build_uagmc_source_model(
            env=env,
            env_key=env_key,
            method_id=method_id,
            profile=profile,
            seed=seed,
            run_dir=run_dir,
            device=device,
        )

    return _ORIGINAL_NEXTGEN_BUILD_MODEL(
        env=env,
        env_key=env_key,
        method_id=method_id,
        profile=profile,
        seed=seed,
        run_dir=run_dir,
        device=device,
    )


def install_master_hooks() -> None:
    ng.install_nextgen_hooks()

    # Override only the two places needed by this funnel:
    # 1) explicit trace inheritance;
    # 2) UAGMC_SOURCE common-profile baseline.
    v3.make_env_factory = master_make_env_factory
    v3.build_model = master_build_model


# =============================================================================
# SPARSE EVALUATION
# =============================================================================

def checkpoint_step_from_name(path: Path) -> Optional[int]:
    m = re.search(r"uam_ppo_(\d+)_steps\.zip$", path.name)
    return int(m.group(1)) if m else None


def available_checkpoint_steps(run_dir: Path) -> List[int]:
    out = []
    for p in (run_dir / "checkpoints").glob("uam_ppo_*_steps.zip"):
        s = checkpoint_step_from_name(p)
        if s is not None:
            _, vec = v3.checkpoint_paths(run_dir, s)
            if vec.exists():
                out.append(int(s))
    return sorted(set(out))


def nearest_checkpoint(
    steps: Sequence[int],
    target: int,
) -> Optional[int]:
    if not steps:
        return None
    return min(
        steps,
        key=lambda s: (
            abs(int(s) - int(target)),
            -int(s),
        ),
    )


def select_sparse_steps(
    run_dir: Path,
    requested_steps: int,
    long_run: bool,
) -> List[int]:
    avail = available_checkpoint_steps(run_dir)

    if not avail:
        return []

    if long_run:
        targets = [
            int(requested_steps * 0.20),
            int(requested_steps * 0.40),
            int(requested_steps * 0.60),
            int(requested_steps * 0.80),
            int(requested_steps),
        ]
    else:
        targets = [
            int(requested_steps * 0.50),
            int(requested_steps * 0.75),
            int(requested_steps),
        ]

    chosen = []
    for t in targets:
        s = nearest_checkpoint(avail, t)
        if s is not None:
            chosen.append(int(s))

    return sorted(set(chosen))


def sparse_eval(
    *,
    run_dir: Path,
    env_key: str,
    method: str,
    requested_steps: int,
    eval_seeds: Sequence[int],
    long_run: bool,
) -> Dict[str, Any]:
    adir = run_dir / "analysis"
    adir.mkdir(parents=True, exist_ok=True)

    summary_path = adir / "pre5b_sparse_summary.json"
    raw_path = adir / "pre5b_sparse_eval.csv"
    curve_path = adir / "pre5b_sparse_curve.csv"

    # Resume-friendly.
    if summary_path.exists():
        return read_json(summary_path, {})

    steps = select_sparse_steps(
        run_dir,
        requested_steps=requested_steps,
        long_run=long_run,
    )

    rows: List[Dict[str, Any]] = []

    for step in steps:
        model_path, vec_path = v3.checkpoint_paths(
            run_dir,
            step,
        )

        for seed in eval_seeds:
            try:
                row = v3.evaluate_checkpoint(
                    env_key=env_key,
                    method_id=method,
                    model_path=model_path,
                    vec_path=vec_path,
                    train_step=int(step),
                    eval_seed=int(seed),
                    run_dir=run_dir,
                    max_time=MAX_TIME,
                )
                rows.append(row)

                print(
                    f"  [SPARSE EVAL] {env_key}/{method} "
                    f"{step//1000}k seed={seed} "
                    f"ATT={fnum(row.get('ATT')):.3f} "
                    f"completion={fnum(row.get('completion_rate'), 0.0):.3f}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"  [SPARSE EVAL ERROR] "
                    f"{env_key}/{method} step={step} "
                    f"seed={seed}: {exc!r}",
                    flush=True,
                )

    write_csv(raw_path, rows)

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[int(r["train_step"])].append(r)

    curve: List[Dict[str, Any]] = []

    for step in sorted(grouped):
        rs = grouped[step]

        valid = [
            r for r in rs
            if bool(r.get("valid_full_completion"))
            and finite(r.get("ATT"))
        ]

        curve.append({
            "train_step": int(step),
            "n_eval": len(rs),
            "n_valid": len(valid),
            "all_full_completion": (
                len(valid) == len(rs)
                and len(rs) == len(eval_seeds)
            ),
            "ATT_mean": fmean(
                r.get("ATT") for r in valid
            ),
            "ATT_std": fstd(
                r.get("ATT") for r in valid
            ),
            "AWT_mean": fmean(
                r.get("AWT") for r in valid
            ),
            "completion_mean": fmean(
                r.get("completion_rate") for r in rs
            ),
            "passenger_V0_share": fmean(
                r.get("passenger_V0_share") for r in rs
            ),
            "aircraft_V0_share": fmean(
                r.get("aircraft_V0_share") for r in rs
            ),
            "pair_consistency": fmean(
                r.get("pair_consistency") for r in rs
            ),
        })

    write_csv(curve_path, curve)

    valid_curve = [
        r for r in curve
        if bool(r["all_full_completion"])
        and finite(r["ATT_mean"])
    ]

    if not valid_curve:
        summary = {
            "status": "NO_VALID",
            "env_key": env_key,
            "method": method,
            "requested_steps": int(requested_steps),
            "n_points": 0,
            "best": float("nan"),
            "best_step": -1,
            "late": float("nan"),
            "final": float("nan"),
            "collapse": float("nan"),
            "completion_mean": fmean(
                r.get("completion_mean") for r in curve
            ),
        }
    else:
        vals = np.asarray(
            [float(r["ATT_mean"]) for r in valid_curve],
            dtype=float,
        )
        best_i = int(np.argmin(vals))

        summary = {
            "status": "VALID",
            "env_key": env_key,
            "method": method,
            "requested_steps": int(requested_steps),
            "n_points": len(valid_curve),
            "best": float(vals[best_i]),
            "best_step": int(
                valid_curve[best_i]["train_step"]
            ),
            "late": float(np.mean(vals[-2:])),
            "final": float(vals[-1]),
            "collapse": float(
                vals[-1] - vals[best_i]
            ),
            "completion_mean": fmean(
                r.get("completion_mean")
                for r in valid_curve
            ),
        }

    write_json(summary_path, summary)
    return summary


# =============================================================================
# ONE TRAINING CELL
# =============================================================================

def run_cell(
    *,
    root: Path,
    phase: str,
    tag: str,
    env_key: str,
    method: str,
    seed: int,
    timesteps: int,
    profile,
    trace_path: str | Path,
    device: str,
    eval_seeds: Sequence[int],
    long_run: bool,
    fail_fast: bool,
) -> Dict[str, Any]:
    set_trace(trace_path)

    cell_root = (
        root
        / phase
        / safe_slug(tag)
        / f"seed{int(seed)}"
    )
    cell_root.mkdir(parents=True, exist_ok=True)

    run_dir = cell_root / v3.cell_id(
        env_key,
        canonical(method),
    )

    meta_path = run_dir / "analysis" / "pre5b_cell_meta.json"

    phase_banner(
        f"{phase} | {tag} | {env_key} | "
        f"{canonical(method)} | seed={seed} | "
        f"steps={timesteps:,} | profile={profile.name}"
    )

    t0 = time.perf_counter()

    try:
        v3.safe_train_and_optional_eval(
            root=cell_root,
            env_key=env_key,
            method_id=canonical(method),
            requested_steps=int(timesteps),
            train_seed=int(seed),
            eval_seeds=eval_seeds,
            profile=profile,
            device=device,
            max_time=MAX_TIME,
            defer_eval=True,
            fail_fast=bool(fail_fast),
        )

        train_wall_sec = time.perf_counter() - t0

        sm = sparse_eval(
            run_dir=run_dir,
            env_key=env_key,
            method=canonical(method),
            requested_steps=int(timesteps),
            eval_seeds=eval_seeds,
            long_run=bool(long_run),
        )

        out = {
            "phase": phase,
            "tag": tag,
            "env_key": env_key,
            "method": canonical(method),
            "seed": int(seed),
            "timesteps": int(timesteps),
            "profile": profile.name,
            "n_envs": int(profile.n_envs),
            "n_steps": int(profile.n_steps),
            "batch_size": int(profile.batch_size),
            "rollout": int(profile.rollout),
            "trace_path": str(Path(trace_path).resolve()),
            "train_wall_sec_this_call": float(train_wall_sec),
            "observed_sps_this_call": (
                float(timesteps) / train_wall_sec
                if train_wall_sec > 0
                else float("nan")
            ),
            **sm,
        }

        write_json(meta_path, out)
        return out

    except Exception as exc:
        out = {
            "phase": phase,
            "tag": tag,
            "env_key": env_key,
            "method": canonical(method),
            "seed": int(seed),
            "timesteps": int(timesteps),
            "profile": profile.name,
            "status": "FAILED",
            "error": repr(exc),
        }

        write_json(meta_path, out)

        print(
            f"[CELL FAILED] {out}",
            flush=True,
        )

        if fail_fast:
            raise

        return out

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_cells(
    rows: Sequence[Dict[str, Any]],
    group_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        key = tuple(r.get(k) for k in group_keys)
        groups[key].append(r)

    out: List[Dict[str, Any]] = []

    for key, rs in groups.items():
        valid = [
            r for r in rs
            if str(r.get("status")) == "VALID"
            and finite(r.get("final"))
        ]

        finals = [fnum(r.get("final")) for r in valid]
        lates = [fnum(r.get("late")) for r in valid]
        bests = [fnum(r.get("best")) for r in valid]
        collapses = [fnum(r.get("collapse")) for r in valid]

        rec = {
            k: v
            for k, v in zip(group_keys, key)
        }

        final_mean = fmean(finals)
        final_std = fstd(finals)

        rec.update({
            "n_planned": len(rs),
            "n_valid": len(valid),
            "best_mean": fmean(bests),
            "late_mean": fmean(lates),
            "final_mean": final_mean,
            "final_std": final_std,
            "final_cv": (
                final_std / final_mean
                if finite(final_std)
                and finite(final_mean)
                and abs(final_mean) > 1e-9
                else float("inf")
            ),
            "worst_final": (
                max(finals)
                if finals
                else float("inf")
            ),
            "collapse_mean": fmean(collapses),
            "collapse_ratio": (
                fmean(collapses) / final_mean
                if finite(fmean(collapses))
                and finite(final_mean)
                and abs(final_mean) > 1e-9
                else float("inf")
            ),
            "completion_mean": fmean(
                r.get("completion_mean")
                for r in valid
            ),
            "observed_sps_median": fmedian(
                r.get("observed_sps_this_call")
                for r in rs
            ),
        })

        out.append(rec)

    return out


def robust_key(r: Dict[str, Any]) -> Tuple[float, ...]:
    return (
        fnum(r.get("late_mean"), float("inf")),
        fnum(r.get("worst_final"), float("inf")),
        fnum(r.get("final_mean"), float("inf")),
        fnum(r.get("final_std"), float("inf")),
    )


# =============================================================================
# P1 | GLOBAL PPO PRODUCTION PROFILE
# =============================================================================

def run_p1(
    root: Path,
    device: str,
    fail_fast: bool,
) -> Dict[str, Any]:
    phase = "P1_PROFILE"
    phase_root = root / phase
    selection_path = phase_root / "PROFILE_SELECTED.json"

    old = read_json(selection_path)
    if old:
        print(f"[P1 RESUME] selected={old['selected_profile']}")
        return old

    rows: List[Dict[str, Any]] = []

    for profile in (
        P40_FAST,
        P40_DENSE,
        P10_REF,
    ):
        for method in P1_METHODS:
            for seed in P1_SEEDS:
                rec = run_cell(
                    root=root,
                    phase=phase,
                    tag=profile.name,
                    env_key="S3",
                    method=method,
                    seed=seed,
                    timesteps=P1_STEPS,
                    profile=profile,
                    trace_path=BASE_TRACE,
                    device=device,
                    eval_seeds=FAST_EVAL_SEEDS,
                    long_run=False,
                    fail_fast=fail_fast,
                )
                rows.append(rec)

                write_csv(
                    phase_root / "cells.csv",
                    rows,
                )

    agg = aggregate_cells(
        rows,
        group_keys=("profile", "method"),
    )
    write_csv(
        phase_root / "profile_method_summary.csv",
        agg,
    )

    # Production selection is intentionally restricted to P40 profiles.
    method_min_final: Dict[str, float] = {}
    for method in P1_METHODS:
        vals = [
            fnum(r.get("final_mean"))
            for r in agg
            if r.get("profile") in P40_CANDIDATES
            and r.get("method") == method
            and finite(r.get("final_mean"))
        ]
        method_min_final[method] = min(vals) if vals else float("inf")

    rankings = []

    for pname in P40_CANDIDATES:
        rs = [
            r for r in agg
            if r.get("profile") == pname
        ]

        if not rs:
            continue

        cvs = []
        rets = []
        perf_gaps = []

        for r in rs:
            cvs.append(
                fnum(r.get("final_cv"), 9.0)
            )
            rets.append(
                max(
                    0.0,
                    fnum(r.get("collapse_ratio"), 9.0),
                )
            )

            base = method_min_final.get(
                r["method"],
                float("inf"),
            )
            perf_gaps.append(
                max(
                    0.0,
                    safe_ratio(
                        fnum(r.get("final_mean")),
                        base,
                        default=9.0,
                    ) - 1.0,
                )
            )

        stability = (
            fmedian(cvs, 9.0)
            + fmedian(rets, 9.0)
        )
        performance = fmedian(
            perf_gaps,
            9.0,
        )

        score = (
            stability
            + 0.25 * performance
        )

        rankings.append({
            "profile": pname,
            "stability_component": stability,
            "performance_component": performance,
            "score": score,
            "expected_sps": EXPECTED_SPS[pname],
        })

    rankings.sort(
        key=lambda r: (
            fnum(r["score"], float("inf")),
            -fnum(r["expected_sps"], 0.0),
        )
    )

    if not rankings:
        # Always keep the funnel moving.
        selected_name = P40_FAST.name
    else:
        selected_name = rankings[0]["profile"]

    result = {
        "selected_profile": selected_name,
        "profile": asdict(PROFILES[selected_name]),
        "selection_rule": (
            "P40 only; median cross-seed CV + normalized retention "
            "+ 0.25 * normalized final-performance gap; speed tie-breaker"
        ),
        "diagnostic_profile_not_selectable": P10_REF.name,
        "ranking": rankings,
    }

    write_csv(
        phase_root / "profile_ranking.csv",
        rankings,
    )
    write_json(selection_path, result)

    return result


# =============================================================================
# P2 | S3 LOAD OPERATING POINT
# =============================================================================

def run_p2(
    root: Path,
    profile,
    trace_map: Dict[str, str],
    device: str,
    fail_fast: bool,
) -> Dict[str, Any]:
    phase = "P2_LOAD"
    phase_root = root / phase
    selection_path = phase_root / "ENV_SELECTED.json"

    old = read_json(selection_path)
    if old:
        print(f"[P2 RESUME] selected_load={old['selected_load']}")
        return old

    rows: List[Dict[str, Any]] = []

    for rho in LOADS:
        key = f"{int(round(rho * 100)):03d}"
        trace = trace_map[key]
        tag = f"LOAD_{key}"

        for seed in P2_UAGMC_SEEDS:
            rows.append(
                run_cell(
                    root=root,
                    phase=phase,
                    tag=tag,
                    env_key="S3",
                    method="UAGMC_SOURCE",
                    seed=seed,
                    timesteps=400_000,
                    profile=profile,
                    trace_path=trace,
                    device=device,
                    eval_seeds=FAST_EVAL_SEEDS,
                    long_run=False,
                    fail_fast=fail_fast,
                )
            )

        for seed in P2_CURRENT_SEEDS:
            rows.append(
                run_cell(
                    root=root,
                    phase=phase,
                    tag=tag,
                    env_key="S3",
                    method="CURRENT",
                    seed=seed,
                    timesteps=300_000,
                    profile=profile,
                    trace_path=trace,
                    device=device,
                    eval_seeds=FAST_EVAL_SEEDS,
                    long_run=False,
                    fail_fast=fail_fast,
                )
            )

        for seed in P2_ANCHOR_SEEDS:
            rows.append(
                run_cell(
                    root=root,
                    phase=phase,
                    tag=tag,
                    env_key="S3",
                    method="A_ANCHOR_QUOTIENT",
                    seed=seed,
                    timesteps=400_000,
                    profile=profile,
                    trace_path=trace,
                    device=device,
                    eval_seeds=FAST_EVAL_SEEDS,
                    long_run=False,
                    fail_fast=fail_fast,
                )
            )

        write_csv(
            phase_root / "cells.csv",
            rows,
        )

    # Add load field into aggregation.
    for r in rows:
        tag = str(r.get("tag", ""))
        m = re.search(r"LOAD_(\d+)", tag)
        if m:
            r["load"] = int(m.group(1)) / 100.0

    agg = aggregate_cells(
        rows,
        group_keys=("load", "method"),
    )
    write_csv(
        phase_root / "load_method_summary.csv",
        agg,
    )

    # Environment selection uses UAGMC_SOURCE difficulty/stability only.
    # CURRENT and A_ANCHOR are diagnostics and do not decide the operating point.
    #
    # criticality ~= seed-CV + normalized within-run collapse.
    # We choose the load closest to a moderate target criticality (0.20),
    # with a small preference for the higher-load environment.
    target_criticality = 0.20

    rankings = []

    for rho in LOADS:
        candidates = [
            r for r in agg
            if abs(fnum(r.get("load")) - rho) < 1e-9
            and r.get("method") == "UAGMC_SOURCE"
        ]

        if not candidates:
            rankings.append({
                "load": rho,
                "score": 999.0,
                "reason": "missing UAGMC_SOURCE summary",
            })
            continue

        r = candidates[0]

        cv = max(
            0.0,
            fnum(r.get("final_cv"), 9.0),
        )
        retention = max(
            0.0,
            fnum(r.get("collapse_ratio"), 9.0),
        )

        criticality = cv + retention

        invalid_penalty = (
            0.0
            if int(r.get("n_valid", 0))
            == int(r.get("n_planned", 0))
            else 10.0
        )

        score = (
            invalid_penalty
            + abs(criticality - target_criticality)
            - 0.02 * float(rho)
        )

        rankings.append({
            "load": rho,
            "uagm_source_final": r.get("final_mean"),
            "uagm_source_cv": cv,
            "uagm_source_collapse_ratio": retention,
            "criticality": criticality,
            "target_criticality": target_criticality,
            "score": score,
        })

    rankings.sort(
        key=lambda r: (
            fnum(r.get("score"), 999.0),
            -fnum(r.get("load"), 0.0),
        )
    )

    selected_load = (
        float(rankings[0]["load"])
        if rankings
        else 1.0
    )
    selected_key = f"{int(round(selected_load * 100)):03d}"

    result = {
        "selected_load": selected_load,
        "selected_trace": trace_map[selected_key],
        "selection_rule": (
            "UAGMC_SOURCE-only automatic operating-point selection: "
            "criticality = seed-CV + normalized collapse; choose closest "
            "to target 0.20; full-valid preferred; higher load tie-breaker. "
            "CURRENT/A_ANCHOR are diagnostic only."
        ),
        "ranking": rankings,
    }

    write_csv(
        phase_root / "load_ranking.csv",
        rankings,
    )
    write_json(selection_path, result)

    return result


# =============================================================================
# P3 | JOINT ARCHITECTURE UNDER FIXED V5 PHYSICS
# =============================================================================

def run_p3(
    root: Path,
    profile,
    trace_path: str,
    device: str,
    fail_fast: bool,
) -> Dict[str, Any]:
    phase = "P3_JOINT"
    phase_root = root / phase
    selection_path = phase_root / "JOINT_SELECTED.json"

    old = read_json(selection_path)
    if old:
        print(f"[P3 RESUME] selected_joint={old['selected_joint_env']}")
        return old

    rows: List[Dict[str, Any]] = []

    for env_key in P3_JOINT_ENVS:
        for method in P3_METHODS:
            for seed in P3_SEEDS:
                rows.append(
                    run_cell(
                        root=root,
                        phase=phase,
                        tag=f"JOINT_{env_key}",
                        env_key=env_key,
                        method=method,
                        seed=seed,
                        timesteps=P3_STEPS,
                        profile=profile,
                        trace_path=trace_path,
                        device=device,
                        eval_seeds=FAST_EVAL_SEEDS,
                        long_run=False,
                        fail_fast=fail_fast,
                    )
                )

                write_csv(
                    phase_root / "cells.csv",
                    rows,
                )

    agg = aggregate_cells(
        rows,
        group_keys=("env_key", "method"),
    )
    write_csv(
        phase_root / "joint_method_summary.csv",
        agg,
    )

    # Normalize ATT within each method so one method scale cannot dominate.
    min_final_by_method: Dict[str, float] = {}

    for m in P3_METHODS:
        vals = [
            fnum(r.get("final_mean"))
            for r in agg
            if r.get("method") == m
            and finite(r.get("final_mean"))
        ]
        min_final_by_method[m] = (
            min(vals)
            if vals
            else float("inf")
        )

    rankings = []

    for env_key in P3_JOINT_ENVS:
        rs = [
            r for r in agg
            if r.get("env_key") == env_key
        ]

        perf = []
        cvs = []
        rets = []

        for r in rs:
            base = min_final_by_method.get(
                r["method"],
                float("inf"),
            )

            perf.append(
                max(
                    0.0,
                    safe_ratio(
                        fnum(r.get("final_mean")),
                        base,
                        default=9.0,
                    ) - 1.0,
                )
            )

            cvs.append(
                max(
                    0.0,
                    fnum(r.get("final_cv"), 9.0),
                )
            )

            rets.append(
                max(
                    0.0,
                    fnum(r.get("collapse_ratio"), 9.0),
                )
            )

        score = (
            fmedian(perf, 9.0)
            + fmedian(cvs, 9.0)
            + fmedian(rets, 9.0)
        )

        rankings.append({
            "env_key": env_key,
            "normalized_performance_gap": fmedian(perf, 9.0),
            "seed_cv": fmedian(cvs, 9.0),
            "collapse_ratio": fmedian(rets, 9.0),
            "score": score,
        })

    rankings.sort(
        key=lambda r: fnum(
            r.get("score"),
            999.0,
        )
    )

    selected_joint = (
        str(rankings[0]["env_key"])
        if rankings
        else "J0"
    )

    result = {
        "selected_joint_env": selected_joint,
        "v5_physics_frozen": True,
        "selection_rule": (
            "median normalized ATT gap + cross-seed CV + "
            "normalized collapse across CURRENT/TDM/QPLEX"
        ),
        "ranking": rankings,
    }

    write_csv(
        phase_root / "joint_ranking.csv",
        rankings,
    )
    write_json(selection_path, result)

    return result


# =============================================================================
# P4 | MATCHED SINGLE / JOINT REVERSAL
# =============================================================================

def run_p4(
    root: Path,
    profile,
    trace_path: str,
    joint_env: str,
    device: str,
    fail_fast: bool,
) -> Dict[str, Any]:
    phase = "P4_SJ"
    phase_root = root / phase
    selection_path = phase_root / "SJ_CLASSIFICATION.json"

    old = read_json(selection_path)
    if old:
        print("[P4 RESUME] classification already exists")
        return old

    rows: List[Dict[str, Any]] = []

    for method in P4_METHODS:
        for env_key in ("S3", joint_env):
            regime = (
                "SINGLE"
                if env_key == "S3"
                else "JOINT"
            )

            for seed in P4_SEEDS:
                rows.append(
                    run_cell(
                        root=root,
                        phase=phase,
                        tag=regime,
                        env_key=env_key,
                        method=method,
                        seed=seed,
                        timesteps=P4_STEPS,
                        profile=profile,
                        trace_path=trace_path,
                        device=device,
                        eval_seeds=FAST_EVAL_SEEDS,
                        long_run=False,
                        fail_fast=fail_fast,
                    )
                )

                write_csv(
                    phase_root / "cells.csv",
                    rows,
                )

    agg = aggregate_cells(
        rows,
        group_keys=("env_key", "method"),
    )
    write_csv(
        phase_root / "sj_summary.csv",
        agg,
    )

    by_method: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for r in agg:
        reg = (
            "S"
            if r["env_key"] == "S3"
            else "J"
        )
        by_method[r["method"]][reg] = r

    classes = []

    for method in P4_METHODS:
        s = by_method.get(method, {}).get("S", {})
        j = by_method.get(method, {}).get("J", {})

        sf = fnum(
            s.get("final_mean"),
            float("inf"),
        )
        jf = fnum(
            j.get("final_mean"),
            float("inf"),
        )

        delta = jf - sf

        ref = min(sf, jf)

        rel_delta = (
            delta / ref
            if finite(ref)
            and ref > 0
            and finite(delta)
            else float("nan")
        )

        max_ret = max(
            fnum(s.get("collapse_ratio"), 0.0),
            fnum(j.get("collapse_ratio"), 0.0),
        )

        max_cv = max(
            fnum(s.get("final_cv"), 0.0),
            fnum(j.get("final_cv"), 0.0),
        )

        if not finite(sf) or not finite(jf):
            cls = "INCOMPLETE"
        elif max_ret > 0.20 or max_cv > 0.20:
            if abs(rel_delta) >= 0.10:
                cls = "UNSTABLE_REVERSAL"
            else:
                cls = "UNSTABLE"
        elif rel_delta <= -0.10:
            cls = "J_STRONG"
        elif rel_delta >= 0.10:
            cls = "S_STRONG"
        else:
            cls = "CROSS_REGIME"

        classes.append({
            "method": method,
            "single_final": sf,
            "joint_final": jf,
            "delta_J_minus_S": delta,
            "relative_delta": rel_delta,
            "single_cv": s.get("final_cv"),
            "joint_cv": j.get("final_cv"),
            "single_collapse_ratio": s.get("collapse_ratio"),
            "joint_collapse_ratio": j.get("collapse_ratio"),
            "single_best": s.get("best_mean"),
            "joint_best": j.get("best_mean"),
            "classification": cls,
        })

    write_csv(
        phase_root / "SJ_CLASSIFICATION.csv",
        classes,
    )

    result = {
        "joint_env": joint_env,
        "rows": classes,
        "classification_rule": (
            ">=10% relative S/J difference => specialist; "
            "otherwise cross-regime; >20% CV or collapse ratio "
            "adds unstable label"
        ),
    }

    write_json(selection_path, result)
    return result


# =============================================================================
# P5 METHOD-REGIME SLOT SELECTION
# =============================================================================

def choose_p5_pairs(
    p4: Dict[str, Any],
    joint_env: str,
) -> List[Dict[str, str]]:
    rows = list(p4["rows"])

    # Lookup.
    by_method = {
        r["method"]: r
        for r in rows
    }

    chosen: List[Dict[str, str]] = []
    used = set()

    def add(method: str, env_key: str, role: str) -> None:
        m = canonical(method)
        if m in used:
            return
        chosen.append({
            "method": m,
            "env_key": env_key,
            "role": role,
        })
        used.add(m)

    # Slot 1: mandatory source-style baseline.
    add(
        "UAGMC_SOURCE",
        "S3",
        "mandatory_baseline",
    )

    # Slot 2: strongest Single method.
    single_sorted = sorted(
        rows,
        key=lambda r: (
            fnum(r.get("single_final"), float("inf")),
            fnum(r.get("single_cv"), float("inf")),
            fnum(r.get("single_collapse_ratio"), float("inf")),
        ),
    )
    if single_sorted:
        add(
            single_sorted[0]["method"],
            "S3",
            "best_single",
        )

    # Slot 3: strongest Joint method.
    joint_sorted = sorted(
        rows,
        key=lambda r: (
            fnum(r.get("joint_final"), float("inf")),
            fnum(r.get("joint_cv"), float("inf")),
            fnum(r.get("joint_collapse_ratio"), float("inf")),
        ),
    )
    if joint_sorted:
        add(
            joint_sorted[0]["method"],
            joint_env,
            "best_joint",
        )

    # Slot 4: best cross-regime balance.
    cross_sorted = sorted(
        rows,
        key=lambda r: (
            max(
                fnum(r.get("single_final"), float("inf")),
                fnum(r.get("joint_final"), float("inf")),
            ),
            abs(
                fnum(r.get("relative_delta"), float("inf"))
            ),
        ),
    )
    for r in cross_sorted:
        if canonical(r["method"]) not in used:
            env = (
                "S3"
                if fnum(r["single_final"], float("inf"))
                <= fnum(r["joint_final"], float("inf"))
                else joint_env
            )
            add(
                r["method"],
                env,
                "best_cross_regime",
            )
            break

    # Slot 5: strongest positive Joint reversal.
    reversal_sorted = sorted(
        rows,
        key=lambda r: fnum(
            r.get("delta_J_minus_S"),
            float("inf"),
        ),
    )
    for r in reversal_sorted:
        if canonical(r["method"]) not in used:
            add(
                r["method"],
                joint_env,
                "strongest_joint_reversal",
            )
            break

    # Slot 6: strongest reachability but controversial retention.
    controversial = sorted(
        rows,
        key=lambda r: (
            min(
                fnum(r.get("single_best"), float("inf")),
                fnum(r.get("joint_best"), float("inf")),
            ),
            -max(
                fnum(r.get("single_collapse_ratio"), 0.0),
                fnum(r.get("joint_collapse_ratio"), 0.0),
            ),
        ),
    )
    for r in controversial:
        if canonical(r["method"]) not in used:
            env = (
                "S3"
                if fnum(r.get("single_best"), float("inf"))
                <= fnum(r.get("joint_best"), float("inf"))
                else joint_env
            )
            add(
                r["method"],
                env,
                "reachability_contested",
            )
            break

    # Deterministic fill if slot overlap occurred.
    overall = sorted(
        rows,
        key=lambda r: min(
            fnum(r.get("single_final"), float("inf")),
            fnum(r.get("joint_final"), float("inf")),
        ),
    )

    for r in overall:
        if len(chosen) >= 6:
            break
        if canonical(r["method"]) in used:
            continue

        env = (
            "S3"
            if fnum(r.get("single_final"), float("inf"))
            <= fnum(r.get("joint_final"), float("inf"))
            else joint_env
        )

        add(
            r["method"],
            env,
            "deterministic_fill",
        )

    if len(chosen) != 6:
        raise RuntimeError(
            f"P5 needs 6 unique pairs, got {chosen}"
        )

    return chosen


# =============================================================================
# P5 | LONG RUN + FRESH SEEDS
# =============================================================================

def run_p5(
    root: Path,
    profile,
    trace_path: str,
    p4: Dict[str, Any],
    joint_env: str,
    device: str,
    fail_fast: bool,
) -> Dict[str, Any]:
    phase = "P5_LONGRUN"
    phase_root = root / phase
    selection_path = phase_root / "LONGRUN_SELECTION.json"

    old = read_json(selection_path)
    if old:
        print("[P5 RESUME] long-run selection already exists")
        return old

    pairs = choose_p5_pairs(
        p4,
        joint_env,
    )

    write_json(
        phase_root / "P5_pairs.json",
        {
            "pairs": pairs,
            "selection_source": "P4 matched S/J classification",
        },
    )

    rows: List[Dict[str, Any]] = []

    for rec in pairs:
        method = rec["method"]
        env_key = rec["env_key"]
        role = rec["role"]

        for seed in P5_SEEDS:
            x = run_cell(
                root=root,
                phase=phase,
                tag=f"{role}__{env_key}",
                env_key=env_key,
                method=method,
                seed=seed,
                timesteps=P5_STEPS,
                profile=profile,
                trace_path=trace_path,
                device=device,
                eval_seeds=LONG_EVAL_SEEDS,
                long_run=True,
                fail_fast=fail_fast,
            )
            x["role"] = role
            rows.append(x)

            write_csv(
                phase_root / "cells.csv",
                rows,
            )

    agg = aggregate_cells(
        rows,
        group_keys=("env_key", "method", "role"),
    )

    agg.sort(key=robust_key)

    write_csv(
        phase_root / "longrun_ranking.csv",
        agg,
    )

    # P6 excludes UAGMC_SOURCE from the two "method-development" finalists.
    candidates = [
        r for r in agg
        if r["method"] != "UAGMC_SOURCE"
        and finite(r.get("final_mean"))
    ]

    if not candidates:
        candidates = [
            r for r in agg
            if finite(r.get("final_mean"))
        ]

    stable = sorted(
        candidates,
        key=robust_key,
    )[0]

    # High-potential finalist:
    # prefer very strong reachability, but require a different method.
    hp_candidates = [
        r for r in candidates
        if r["method"] != stable["method"]
    ]

    if hp_candidates:
        high_potential = sorted(
            hp_candidates,
            key=lambda r: (
                fnum(r.get("best_mean"), float("inf")),
                fnum(r.get("final_mean"), float("inf")),
            ),
        )[0]
    else:
        high_potential = stable

    final_two = [
        {
            "method": stable["method"],
            "env_key": stable["env_key"],
            "role": "best_stable",
        },
        {
            "method": high_potential["method"],
            "env_key": high_potential["env_key"],
            "role": "best_high_potential",
        },
    ]

    result = {
        "pairs": pairs,
        "ranking": agg,
        "final_two": final_two,
        "selection_rule": (
            "best_stable = Late -> worst Final -> Final mean -> Final std; "
            "high_potential = lowest long-run Best among remaining methods"
        ),
    }

    write_json(selection_path, result)
    return result


# =============================================================================
# P6 | FINAL TWO-METHOD CONFIRMATION
# =============================================================================

def run_p6(
    root: Path,
    profile,
    trace_path: str,
    p5: Dict[str, Any],
    device: str,
    fail_fast: bool,
) -> Dict[str, Any]:
    phase = "P6_CONFIRM"
    phase_root = root / phase
    final_path = phase_root / "PRE5B_FINAL.json"

    old = read_json(final_path)
    if old:
        print("[P6 RESUME] final result already exists")
        return old

    pairs = list(p5["final_two"])
    if len(pairs) != 2:
        raise RuntimeError(
            f"P6 requires exactly two pairs, got {pairs}"
        )

    rows: List[Dict[str, Any]] = []

    for rec in pairs:
        for seed in P6_SEEDS:
            x = run_cell(
                root=root,
                phase=phase,
                tag=rec["role"],
                env_key=rec["env_key"],
                method=rec["method"],
                seed=seed,
                timesteps=P6_STEPS,
                profile=profile,
                trace_path=trace_path,
                device=device,
                eval_seeds=LONG_EVAL_SEEDS,
                long_run=True,
                fail_fast=fail_fast,
            )
            x["role"] = rec["role"]
            rows.append(x)

            write_csv(
                phase_root / "cells.csv",
                rows,
            )

    agg = aggregate_cells(
        rows,
        group_keys=("env_key", "method", "role"),
    )
    agg.sort(key=robust_key)

    write_csv(
        phase_root / "confirmation_ranking.csv",
        agg,
    )

    result = {
        "confirmation_ranking": agg,
        "deep_dive_methods_for_500M": [
            {
                "method": r["method"],
                "env_key": r["env_key"],
                "role": r["role"],
            }
            for r in agg
        ],
    }

    write_json(final_path, result)
    return result


# =============================================================================
# PLAN / ETA
# =============================================================================

def build_budget_table() -> List[Dict[str, Any]]:
    return [
        {
            "phase": "P1_PROFILE",
            "budget": P1_BUDGET,
            "purpose": "P40 global PPO profile + P10 diagnostic",
        },
        {
            "phase": "P2_LOAD",
            "budget": P2_BUDGET,
            "purpose": "S3 nested demand-load operating point",
        },
        {
            "phase": "P3_JOINT",
            "budget": P3_BUDGET,
            "purpose": "Joint architecture under fixed V5 physics",
        },
        {
            "phase": "P4_SJ",
            "budget": P4_BUDGET,
            "purpose": "matched Single/Joint reversal analysis",
        },
        {
            "phase": "P5_LONGRUN",
            "budget": P5_BUDGET,
            "purpose": "selected long-run + fresh-seed stability",
        },
        {
            "phase": "P6_CONFIRM",
            "budget": P6_BUDGET,
            "purpose": "final two-method confirmation",
        },
    ]


def print_eta() -> None:
    """
    Conservative startup estimate before P1 knows which P40 wins.
    """
    # P1:
    # each profile = 3 methods * 3 seeds * 400k = 3.6M
    p1_sec = (
        3_600_000 / EXPECTED_SPS[P40_FAST.name]
        + 3_600_000 / EXPECTED_SPS[P40_DENSE.name]
        + 3_600_000 / EXPECTED_SPS[P10_REF.name]
    )

    remaining = TOTAL_BUDGET - P1_BUDGET

    # Show both P40 scenarios.
    fast_h = (
        p1_sec
        + remaining / EXPECTED_SPS[P40_FAST.name]
    ) / 3600.0

    dense_h = (
        p1_sec
        + remaining / EXPECTED_SPS[P40_DENSE.name]
    ) / 3600.0

    print(
        f"[ETA, pure training only] "
        f"if P40_FAST selected ~{fast_h:.2f} h; "
        f"if P40_DENSE selected ~{dense_h:.2f} h. "
        f"Sparse eval/startup overhead is additional.",
        flush=True,
    )


# =============================================================================
# MANIFEST
# =============================================================================

def write_master_manifest(root: Path) -> None:
    write_json(
        root / "PRE5B80_MASTER_MANIFEST.json",
        {
            "experiment": "PRE5B_80M_AUTO_FUNNEL",
            "created": datetime.now().isoformat(timespec="seconds"),
            "total_requested_transitions": TOTAL_BUDGET,
            "budget_table": build_budget_table(),
            "frozen_physics": {
                "single": "S3",
                "joint_physics": "V5 clean minimal target-priority",
                "topology": v3.TOPOLOGY,
                "fleet_size": v3.FLEET_SIZE,
                "turnaround_min": v3.TURNAROUND_MIN,
                "charge_rate_scale": v3.CHARGE_RATE_SCALE,
                "charger_capacity": v3.CHARGER_CAPACITY,
                "pad_separation_min": v3.PAD_SEPARATION_MIN,
                "hard_guard": MAX_TIME,
            },
            "profile_candidates": {
                k: asdict(v)
                for k, v in PROFILES.items()
            },
            "P10_is_diagnostic_only": True,
            "no_method_specific_PPO_tuning": True,
            "sparse_eval": {
                "P1_to_P4": list(FAST_EVAL_SEEDS),
                "P5_to_P6": list(LONG_EVAL_SEEDS),
                "note": (
                    "All normal 50k checkpoints remain saved; "
                    "full formal evaluation can be done offline."
                ),
            },
        },
    )


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Pre-500M 80M automatic funnel"
    )

    ap.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
    )

    ap.add_argument(
        "--resume-root",
        default="",
    )

    ap.add_argument(
        "--output-root",
        default="",
    )

    ap.add_argument(
        "--plan-only",
        action="store_true",
    )

    ap.add_argument(
        "--fail-fast",
        action="store_true",
    )

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False"
        )

    # Validate method registry before any expensive training.
    ng.install_nextgen_hooks()

    required = [
        "CURRENT",
        "A_ANCHOR_QUOTIENT",
        "W_DREAMER_BALANCED",
        "S_TDM_EVENT_FUSION",
        "A_QPLEX_DUPLEX",
        "S_GATV2_EVENT",
        "SA_TDMFUSION_ICM",
    ]

    # Call descriptions as an early registry sanity check.
    for m in required:
        ng.method_description(m)

    # Reinstall our master hooks after the registry sanity check.
    install_master_hooks()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.resume_root:
        root = Path(args.resume_root).expanduser().resolve()
    elif args.output_root:
        root = Path(args.output_root).expanduser().resolve()
    else:
        root = (
            ROOT
            / "serial_runs"
            / f"uam_pre5b80m_{stamp}"
        ).resolve()

    root.mkdir(parents=True, exist_ok=True)

    write_master_manifest(root)

    trace_map = make_nested_load_traces(
        root / "trace_bank"
    )

    print("=" * 120)
    print("PRE-5B 80M AUTO FUNNEL")
    print(f"root      : {root}")
    print(f"device    : {args.device}")
    if torch.cuda.is_available():
        print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"budget    : {TOTAL_BUDGET:,} requested PPO transitions")
    print("=" * 120)

    print_eta()

    write_csv(
        root / "budget_plan.csv",
        build_budget_table(),
    )

    if args.plan_only:
        print("[PLAN ONLY] no training started.")
        return 0

    # -----------------------------------------------------------------
    # P1 | profile
    # -----------------------------------------------------------------
    p1 = run_p1(
        root,
        device=args.device,
        fail_fast=args.fail_fast,
    )

    selected_profile_name = p1["selected_profile"]
    profile = PROFILES[selected_profile_name]

    print(
        f"\n[INHERIT] P1 -> selected profile = "
        f"{selected_profile_name}",
        flush=True,
    )

    # -----------------------------------------------------------------
    # P2 | load
    # -----------------------------------------------------------------
    p2 = run_p2(
        root,
        profile=profile,
        trace_map=trace_map,
        device=args.device,
        fail_fast=args.fail_fast,
    )

    selected_load = float(p2["selected_load"])
    selected_trace = str(p2["selected_trace"])

    print(
        f"\n[INHERIT] P2 -> selected S3 load = "
        f"{selected_load:.2f}",
        flush=True,
    )

    # All subsequent phases inherit this exact trace.
    set_trace(selected_trace)

    # -----------------------------------------------------------------
    # P3 | Joint
    # -----------------------------------------------------------------
    p3 = run_p3(
        root,
        profile=profile,
        trace_path=selected_trace,
        device=args.device,
        fail_fast=args.fail_fast,
    )

    joint_env = str(p3["selected_joint_env"])

    print(
        f"\n[INHERIT] P3 -> selected Joint env = "
        f"{joint_env}",
        flush=True,
    )

    # -----------------------------------------------------------------
    # P4 | matched S/J
    # -----------------------------------------------------------------
    p4 = run_p4(
        root,
        profile=profile,
        trace_path=selected_trace,
        joint_env=joint_env,
        device=args.device,
        fail_fast=args.fail_fast,
    )

    print(
        "\n[INHERIT] P4 -> method/regime classes frozen "
        "for long-run slot selection",
        flush=True,
    )

    # -----------------------------------------------------------------
    # P5 | long-run
    # -----------------------------------------------------------------
    p5 = run_p5(
        root,
        profile=profile,
        trace_path=selected_trace,
        p4=p4,
        joint_env=joint_env,
        device=args.device,
        fail_fast=args.fail_fast,
    )

    print(
        f"\n[INHERIT] P5 -> final two = "
        f"{p5['final_two']}",
        flush=True,
    )

    # -----------------------------------------------------------------
    # P6 | confirmation
    # -----------------------------------------------------------------
    p6 = run_p6(
        root,
        profile=profile,
        trace_path=selected_trace,
        p5=p5,
        device=args.device,
        fail_fast=args.fail_fast,
    )

    # -----------------------------------------------------------------
    # Final frozen config for the later 500M campaign.
    # -----------------------------------------------------------------
    final = {
        "experiment_root": str(root),
        "total_requested_transitions": TOTAL_BUDGET,
        "selected_profile": p1,
        "selected_environment": p2,
        "selected_joint": p3,
        "sj_classification": p4,
        "longrun_selection": p5,
        "confirmation": p6,
        "PRE5B_FREEZE": {
            "profile": selected_profile_name,
            "profile_spec": asdict(profile),
            "main_env": "S3",
            "main_load": selected_load,
            "main_trace": selected_trace,
            "joint_env": joint_env,
            "joint_physics": "V5 clean minimal target-priority",
            "deep_dive_methods_for_500M": p6[
                "deep_dive_methods_for_500M"
            ],
        },
    }

    write_json(
        root / "PRE5B_FREEZE_MANIFEST.json",
        final,
    )

    print("\n" + "#" * 120)
    print("PRE-5B 80M COMPLETE")
    print("#" * 120)
    print(
        json.dumps(
            final["PRE5B_FREEZE"],
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"\nFinal manifest: "
        f"{root / 'PRE5B_FREEZE_MANIFEST.json'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
