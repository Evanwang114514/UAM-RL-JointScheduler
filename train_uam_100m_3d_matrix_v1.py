# -*- coding: utf-8 -*-
"""
UAM 100M 3-D matrix | corrected/frozen contract
================================================

Scientific design
-----------------
10 Methods x 4 Control structures x 3 Environments x 800k = 96M transitions.
Reserve budget: 5 manual rescue cells x 800k = 4M. Grand nominal budget = 100M.

Environment axis (exact current S1/S2/S3 physics)
    S1: fixed fleet=40 + turnaround=1 min
    S2: S1 + finite charging (capacity=5, scale=1.25)
    S3: S2 + pad/TLOF separation=0.25 min

Control axis
    LQ          : passenger learned, aircraft responsive longest-queue heuristic
    J4          : validated V5-clean semantic pair scorer (project incumbent)
    HAPPO_STYLE : heterogeneous passenger/aircraft factorized actor card,
                  passenger-first conditional factorization, centralized critic.
                  This is a literature-inspired control card, NOT a verbatim HARL implementation.
    BPTA_STYLE  : passenger-first autoregressive actor with differentiable
                  aircraft-response feedback into passenger logits.
                  This is a BPTA/BPPO-inspired control card, NOT a verbatim reproduction.

Method axis
    UAGMC_SOURCE        strict source-style representation baseline
    CURRENT_BASE        current-resource + focal-OD common base
    S_SHARED_HORIZON    restored ABC B_SHARED/M3P-shared temporal control
    S_TDM_EVENT_FUSION  project S
    S_GATV2_EVENT       literature-inspired S
    A_ACTION_CENTER     restored ABC B_CAND/M4-candidate action-centered control
    A_ICM_AC            literature-inspired AC
    SA_TDMFUSION_ICM    project S+AC
    SA_TGAT_FACMAC      literature-inspired S+AC
    SA_GATV2_QPLEX      literature-inspired S+AC

Important contracts
-------------------
* Requires the Stage-I UAM_FINAL_FREEZE_V2.json + .sha256.
* Requires the Stage12 corrected reward source markers already present.
* Uses ONE train seed by default: 601.
* Uses the frozen train trace and frozen held-out validation bank.
* PPO profile/hyperparameters are checked against the freeze.
* Run length is 800k, but LR is deliberately evaluated on the original
  1.536M horizon so that the 0..800k schedule matches Stage12.
* Checkpoints every 50k. Main metric = validation-bank LWATT over final 20%
  (normally 650k/700k/750k/800k).
* No historical result is silently imported. Old experiments are method-selection
  evidence only. Any reuse must be explicitly audited outside this script.

Typical usage
-------------
Plan only:
    python train_uam_100m_3d_matrix_v1.py --plan-only \
      --freeze serial_runs/uam_stage12_v2_.../UAM_FINAL_FREEZE_V2.json

Full nominal 96M matrix:
    CUDA_VISIBLE_DEVICES=0 python train_uam_100m_3d_matrix_v1.py \
      --freeze serial_runs/uam_stage12_v2_.../UAM_FINAL_FREEZE_V2.json \
      --device cuda

Resume same root:
    CUDA_VISIBLE_DEVICES=0 python train_uam_100m_3d_matrix_v1.py \
      --freeze serial_runs/uam_stage12_v2_.../UAM_FINAL_FREEZE_V2.json \
      --resume-root serial_runs/uam_100m_3d_v1_YYYYMMDD_HHMMSS \
      --device cuda

UAGMC surface first (12 cells = 9.6M):
    python train_uam_100m_3d_matrix_v1.py --phase uagmc --freeze ... --device cuda

One method / smoke:
    python train_uam_100m_3d_matrix_v1.py --methods S_SHARED_HORIZON \
      --envs S1 --controls LQ --timesteps 50000 --freeze ... --device cuda

Manual rescue example (kept outside the 96M main matrix):
    python train_uam_100m_3d_matrix_v1.py --methods S_SHARED_HORIZON \
      --envs S1 --controls LQ --seed 602 --timesteps 800000 \
      --output-root serial_runs/uam_100m_rescue_seed602 --freeze ... --device cuda
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
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO

# -----------------------------------------------------------------------------
# Validated project stack. This file must live in the UAGMC-main project root.
# -----------------------------------------------------------------------------
import train_uam_stage12_freeze_baselines_v2 as stage12
import train_uam_pre5b_80m_autofunnel as pre80
import train_uam_pre500m_50m_freeze_supplement as pre50
import train_uam_nextgen_100m_matrix as ng
import train_uagmc_ABC_controls_500k as abc

v3 = pre80.v3
v4 = pre80.v4
v5 = pre80.v5

ROOT = Path(__file__).resolve().parent
DEFAULT_TIMESTEPS = 800_000
LR_HORIZON_STEPS = 1_536_000
CHECKPOINT_INTERVAL = 50_000
DEFAULT_SEED = 601
DEFAULT_EVAL_SEEDS = (123, 124)
HARD_GUARD = 2_500

ENV_STAGES = ("S1", "S2", "S3")
CONTROLS = ("LQ", "J4", "HAPPO_STYLE", "BPTA_STYLE")

METHODS = (
    "UAGMC_SOURCE",
    "CURRENT_BASE",
    "S_SHARED_HORIZON",
    "S_TDM_EVENT_FUSION",
    "S_GATV2_EVENT",
    "A_ACTION_CENTER",
    "A_ICM_AC",
    "SA_TDMFUSION_ICM",
    "SA_TGAT_FACMAC",
    "SA_GATV2_QPLEX",
)

METHOD_INTERNAL = {
    "UAGMC_SOURCE": "UAGMC_SOURCE",
    "CURRENT_BASE": "CURRENT",
    "S_SHARED_HORIZON": "B_SHARED",   # ABC M3P-shared
    "S_TDM_EVENT_FUSION": "S_TDM_EVENT_FUSION",
    "S_GATV2_EVENT": "S_GATV2_EVENT",
    "A_ACTION_CENTER": "B_CAND",      # ABC M4-candidate
    "A_ICM_AC": "A_ICM_AC",
    "SA_TDMFUSION_ICM": "SA_TDMFUSION_ICM",
    "SA_TGAT_FACMAC": "SA_TGAT_FACMAC",
    "SA_GATV2_QPLEX": "SA_GATV2_QPLEX",
}

SPECIAL_ABC = {"S_SHARED_HORIZON", "A_ACTION_CENTER"}
PUBLIC_FAMILY = {
    "UAGMC_SOURCE": "BASELINE",
    "CURRENT_BASE": "BASE",
    "S_SHARED_HORIZON": "S",
    "S_TDM_EVENT_FUSION": "S",
    "S_GATV2_EVENT": "S",
    "A_ACTION_CENTER": "AC",
    "A_ICM_AC": "AC",
    "SA_TDMFUSION_ICM": "S+AC",
    "SA_TGAT_FACMAC": "S+AC",
    "SA_GATV2_QPLEX": "S+AC",
}

# Active run length is used only to map SB3 progress_remaining to the fixed
# 1.536M LR horizon. Main experiment uses 800k; smoke tests may override it.
_ACTIVE_TIMESTEPS = DEFAULT_TIMESTEPS
_FREEZE: Dict[str, Any] = {}


def canonical(x: Any) -> str:
    return str(x).strip().upper()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except Exception:
        return "UNAVAILABLE"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not Path(path).exists():
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            cooked = {}
            for k in fields:
                v = r.get(k, "")
                if isinstance(v, (dict, list, tuple)):
                    v = json.dumps(v, ensure_ascii=False, default=str)
                cooked[k] = v
            w.writerow(cooked)


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def mean(xs: Iterable[Any], default: float = float("nan")) -> float:
    vals = [fnum(x) for x in xs]
    vals = [x for x in vals if math.isfinite(x)]
    return float(np.mean(vals)) if vals else default


def sample_sd(xs: Iterable[Any], default: float = float("nan")) -> float:
    vals = [fnum(x) for x in xs]
    vals = [x for x in vals if math.isfinite(x)]
    if len(vals) < 2:
        return 0.0 if len(vals) == 1 else default
    return float(np.std(vals, ddof=1))


def safe_slug(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def parse_csv_choice(text: str, allowed: Sequence[str]) -> List[str]:
    if not str(text).strip():
        return list(allowed)
    vals = [canonical(x) for x in str(text).split(",") if x.strip()]
    bad = [x for x in vals if x not in allowed]
    if bad:
        raise ValueError(f"unsupported values {bad}; allowed={list(allowed)}")
    return vals


# =============================================================================
# Freeze / runtime contract
# =============================================================================

def verify_freeze_compatible(path: Path) -> Dict[str, Any]:
    """Verify exact freeze file/source/trace locks without requiring HEAD unchanged.

    Adding this new driver and committing it must not invalidate the old freeze;
    what matters is that frozen source files and traces have their recorded hashes.
    """
    path = Path(path).expanduser().resolve()
    sha_path = path.with_suffix(".sha256")
    if not path.is_file() or not sha_path.is_file():
        raise FileNotFoundError("need UAM_FINAL_FREEZE_V2.json and matching .sha256")
    expected = sha_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256(path)
    if expected != actual:
        raise RuntimeError("freeze SHA256 mismatch")
    fr = read_json(path)
    if fr.get("schema") != "UAM_FINAL_FREEZE_V2":
        raise RuntimeError(f"wrong freeze schema: {fr.get('schema')}")

    for rec in fr.get("reward_contract", {}).get("source_records", []):
        p = Path(rec["path"])
        if not p.is_file() or sha256(p) != rec.get("after"):
            raise RuntimeError(f"reward source lock mismatch: {p}")
    for bank_key in ("train_trace_locks", "heldout_trace_locks"):
        for rec in fr.get("demand_protocol", {}).get(bank_key, []):
            p = Path(rec["path"])
            if not p.is_file() or sha256(p) != rec.get("sha256"):
                raise RuntimeError(f"trace lock mismatch: {p}")

    ppo = fr.get("frozen_ppo", {})
    required = {
        "n_envs": 10,
        "n_steps": 2048,
        "global_rollout": 20480,
        "batch_size": 4096,
        "n_epochs": 10,
    }
    for k, v in required.items():
        if int(ppo.get(k, -1)) != int(v):
            raise RuntimeError(f"frozen PPO {k}={ppo.get(k)} != expected {v}")
    if abs(float(ppo.get("gamma", -1)) - 1.0) > 1e-12:
        raise RuntimeError("frozen gamma must be 1.0")

    env = fr.get("frozen_environment", {})
    if env.get("topology") != "T2" or int(env.get("fleet_size", -1)) != 40:
        raise RuntimeError("freeze must be T2/fleet40")
    if abs(float(env.get("selected_load", -1.0)) - 1.0) > 1e-12:
        raise RuntimeError("this matrix is defined on frozen load=1.0")
    if int(env.get("hard_guard_min", -1)) != HARD_GUARD:
        raise RuntimeError("freeze hard guard must be 2500 min")

    train_bank = list(fr.get("demand_protocol", {}).get("train_trace_bank", []))
    heldout = list(fr.get("demand_protocol", {}).get("heldout_trace_bank", []))
    if len(train_bank) != 1:
        raise RuntimeError("100M v1 requires the selected FIXED_TRACE train bank of size 1")
    if not heldout:
        raise RuntimeError("held-out validation bank is empty")

    # The Stage12 reward patch must already be present; never patch silently here.
    stage12.ensure_reward_contract(ROOT, auto_patch=False)

    fr["_sha256"] = actual
    fr["_path"] = str(path)
    return fr


def assert_runtime_matches_freeze(fr: Dict[str, Any]) -> None:
    p = fr["frozen_ppo"]
    checks = [
        ("gamma", float(v3.GAMMA), float(p["gamma"])),
        ("gae_lambda", float(v3.GAE_LAMBDA), float(p["gae_lambda"])),
        ("clip_range", float(v3.CLIP_RANGE), float(p["clip_range"])),
        ("initial_lr", float(v3.INITIAL_LR), float(p["initial_lr"])),
        ("ent_coef", float(v3.ENT_COEF), float(p["ent_coef"])),
        ("vf_coef", float(v3.VF_COEF), float(p["vf_coef"])),
        ("max_grad_norm", float(v3.MAX_GRAD_NORM), float(p["max_grad_norm"])),
    ]
    bad = [(k, a, b) for k, a, b in checks if abs(a - b) > 1e-10]
    if bad:
        raise RuntimeError(f"runtime/freeze PPO mismatch: {bad}")


# =============================================================================
# Registry and synthetic Environment x Control specs
# =============================================================================

def synthetic_key(stage: str, control: str) -> str:
    return f"{canonical(stage)}__{canonical(control)}"


def split_key(key: str) -> Tuple[str, str]:
    a, b = canonical(key).split("__", 1)
    return a, b


def install_registry_and_specs() -> None:
    # Install all NextGen primitive/combination cards, then add the 70M combo.
    ng.install_nextgen_hooks()
    ng.install_registry()
    v4.COMBO_REGISTRY["SA_GATV2_QPLEX"] = (
        "S_GATV2_EVENT", "A_QPLEX_DUPLEX",
    )

    # Synthetic keys make the environment and control axes independent.
    for stage in ENV_STAGES:
        phys = v3.ENV_SPECS[stage]
        for control in CONTROLS:
            key = synthetic_key(stage, control)
            joint = control != "LQ"
            v3.ENV_SPECS[key] = v3.EnvSpec(
                key=key,
                physical_stage=phys.physical_stage,
                pad_separation=float(phys.pad_separation),
                charger_capacity=int(phys.charger_capacity),
                joint=joint,
                joint_arch=control,
                sequential_commit=False,
            )
            v3.JOINT_ARCH_NAMES[key] = "SINGLE_LQ" if not joint else control


# =============================================================================
# Literature-inspired control cards
# =============================================================================

class HAPPOStyleJointPolicy(v4.DiscoveryJointPolicy):
    """HAPPO-inspired heterogeneous factorized actor card over Discrete(4).

    This intentionally keeps the frozen PPO optimizer/critic. It is NOT a
    verbatim HARL/HAPPO reproduction. The controlled transplant is:
      * separate passenger and aircraft actor trunks;
      * passenger-first conditional aircraft policy;
      * centralized value function from the full representation.
    """

    def __init__(self, *args, **kwargs):
        lr_schedule = kwargs.get("lr_schedule", None)
        if lr_schedule is None and len(args) >= 3:
            lr_schedule = args[2]
        super().__init__(*args, joint_arch="J0", **kwargs)
        d = int(self.mlp_extractor.latent_dim_pi)
        self.hap_p_trunk = nn.Sequential(nn.Linear(d, 96), nn.Tanh(), nn.Linear(96, 64), nn.Tanh())
        self.hap_a_trunk = nn.Sequential(nn.Linear(d, 96), nn.Tanh(), nn.Linear(96, 64), nn.Tanh())
        self.hap_p = nn.Linear(64, 2)
        self.hap_a = nn.Sequential(nn.Linear(64 + 2, 64), nn.Tanh(), nn.Linear(64, 2))
        if lr_schedule is None:
            raise RuntimeError("cannot recover lr schedule")
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1.0), **self.optimizer_kwargs
        )

    def _joint_logits(self, latent: torch.Tensor) -> torch.Tensor:
        b = latent.shape[0]
        hp = self.hap_p_trunk(latent)
        ha = self.hap_a_trunk(latent)
        p_logp = F.log_softmax(self.hap_p(hp), dim=-1)
        rows = []
        for p in range(2):
            p_oh = F.one_hot(
                torch.full((b,), p, dtype=torch.long, device=latent.device), num_classes=2
            ).float()
            a_logp = F.log_softmax(self.hap_a(torch.cat([ha, p_oh], dim=-1)), dim=-1)
            rows.append(p_logp[:, p:p + 1] + a_logp)
        return torch.stack(rows, dim=1).reshape(b, 4)


class BPTAStyleJointPolicy(v4.DiscoveryJointPolicy):
    """BPTA/BPPO-inspired backward inter-agent feedback actor card.

    For each passenger action, an aircraft response distribution is computed.
    A differentiable feedback network maps that response back into the passenger
    logits before the final autoregressive joint distribution is formed.
    This preserves the frozen PPO optimizer and is intentionally labelled STYLE.
    """

    def __init__(self, *args, **kwargs):
        lr_schedule = kwargs.get("lr_schedule", None)
        if lr_schedule is None and len(args) >= 3:
            lr_schedule = args[2]
        super().__init__(*args, joint_arch="J0", **kwargs)
        d = int(self.mlp_extractor.latent_dim_pi)
        self.bpta_p = nn.Sequential(nn.Linear(d, 96), nn.ReLU(), nn.Linear(96, 2))
        self.bpta_a = nn.Sequential(nn.Linear(d + 2, 96), nn.ReLU(), nn.Linear(96, 2))
        self.bpta_feedback = nn.Sequential(
            nn.Linear(d + 2 + 2, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.bpta_beta = 0.35
        if lr_schedule is None:
            raise RuntimeError("cannot recover lr schedule")
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1.0), **self.optimizer_kwargs
        )

    def _joint_logits(self, latent: torch.Tensor) -> torch.Tensor:
        b = latent.shape[0]
        p_base = self.bpta_p(latent)
        a_logits_rows = []
        feedback = []
        for p in range(2):
            p_oh = F.one_hot(
                torch.full((b,), p, dtype=torch.long, device=latent.device), num_classes=2
            ).float()
            a_logits = self.bpta_a(torch.cat([latent, p_oh], dim=-1))
            a_probs = F.softmax(a_logits, dim=-1)
            fb = self.bpta_feedback(torch.cat([latent, p_oh, a_probs], dim=-1))
            a_logits_rows.append(a_logits)
            feedback.append(fb)
        feedback_t = torch.cat(feedback, dim=1)
        p_logits = p_base + self.bpta_beta * feedback_t
        p_logp = F.log_softmax(p_logits, dim=-1)
        rows = []
        for p in range(2):
            a_logp = F.log_softmax(a_logits_rows[p], dim=-1)
            rows.append(p_logp[:, p:p + 1] + a_logp)
        return torch.stack(rows, dim=1).reshape(b, 4)


# =============================================================================
# Observation factory: exact physical stage x independent control structure
# =============================================================================

def _set_worker_trace() -> None:
    trace = Path(os.environ["UAM_TRAIN_FILE"]).resolve()
    for mod in (v3, getattr(v3, "legacy", None), getattr(v3, "core", None)):
        if mod is not None and hasattr(mod, "TRAIN_FILE"):
            setattr(mod, "TRAIN_FILE", trace)


def matrix_make_env_factory(
    *, env_key: str, method_id: str, env_index: int, run_dir: Path, max_time: int,
):
    spec = v3.ENV_SPECS[canonical(env_key)]
    stage, control = split_key(env_key)
    public_method = canonical(method_id)

    def _init():
        # spawn 子进程会重新导入模块，不会继承主进程动态注册的环境和方法。
        # 必须先恢复完整注册表，再创建联合控制或观测包装器；不改变实验参数。
        install_registry_and_specs()
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        _set_worker_trace()
        v3.configure_worker_physics(int(max_time))
        raw = v3.core.make_experiment_env_factory(
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

        if control != "LQ":
            raw = v5.CleanMinimalJointWrapper(raw, env_key)

        if public_method in SPECIAL_ABC:
            abc_method = METHOD_INTERNAL[public_method]
            return abc.ControlObservationWrapper(
                raw,
                stage=spec.physical_stage,
                method=abc_method,
                topology=v3.TOPOLOGY,
                future_horizon=float(abc.old.FUTURE_HORIZON_MIN),
                max_events=int(abc.old.MAX_EVENTS_PER_TYPE),
                charger_capacity=int(spec.charger_capacity),
            )

        # All other methods use the validated common current/focal/event layout.
        return v3.LiteratureObservationWrapper(
            raw, env_key=env_key, method_id=METHOD_INTERNAL[public_method]
        )

    return _init


# =============================================================================
# Model construction with fixed 1.536M LR horizon
# =============================================================================

def fixed_horizon_lr(initial_lr: float):
    run_steps = int(_ACTIVE_TIMESTEPS)
    horizon = int(LR_HORIZON_STEPS)

    def _schedule(progress_remaining: float) -> float:
        # SB3 progress_remaining is defined relative to model.learn(total_timesteps=run_steps).
        elapsed = float(run_steps) * (1.0 - float(progress_remaining))
        return float(initial_lr) * max(0.0, 1.0 - elapsed / float(horizon))

    return _schedule


def method_description(public_method: str) -> Dict[str, Any]:
    m = canonical(public_method)
    if m == "UAGMC_SOURCE":
        return {
            "method_id": m,
            "family": "BASELINE",
            "components": ["UAGMC_SOURCE"],
            "literature_or_role": "strict source-style temporal representation; no separate focal branch",
            "style_only_not_verbatim": False,
        }
    if m == "CURRENT_BASE":
        d = ng.method_description("CURRENT")
        d.update({"method_id": m, "family": "BASE", "components": ["CURRENT"]})
        return d
    if m == "S_SHARED_HORIZON":
        return {
            "method_id": m,
            "family": "S",
            "components": ["ABC_B_SHARED", "M3P_SHARED"],
            "literature_or_role": "restored 5M ABC shared/common temporal-reference control",
            "scientific_question": "cross-action temporal comparability via one passenger-level shared horizon",
            "historical_origin": "train_uagmc_ABC_controls_500k.py / B_SHARED",
            "style_only_not_verbatim": False,
        }
    if m == "A_ACTION_CENTER":
        return {
            "method_id": m,
            "family": "AC",
            "components": ["ABC_B_CAND", "M4_CANDIDATE"],
            "literature_or_role": "restored candidate/action-centered effect-time control",
            "scientific_question": "candidate-specific effect-time projection",
            "historical_origin": "train_uagmc_ABC_controls_500k.py / B_CAND",
            "style_only_not_verbatim": False,
        }
    internal = METHOD_INTERNAL[m]
    d = ng.method_description(internal)
    d = dict(d)
    d["method_id"] = m
    d["family"] = PUBLIC_FAMILY[m]
    d["internal_method_id"] = internal
    return d


def _abc_layout(env_key: str, public_method: str) -> Dict[str, Any]:
    spec = v3.ENV_SPECS[env_key]
    return abc.control_layout(
        stage=spec.physical_stage,
        method=METHOD_INTERNAL[public_method],
        topology=v3.TOPOLOGY,
        max_events=int(abc.old.MAX_EVENTS_PER_TYPE),
    )


def matrix_build_model(
    *, env, env_key: str, method_id: str, profile, seed: int,
    run_dir: Path, device: str,
):
    public = canonical(method_id)
    _, control = split_key(env_key)
    internal = METHOD_INTERNAL[public]

    if public == "UAGMC_SOURCE":
        extractor_cls = pre80.SourceUAGMCExtractor
        extractor_kwargs = dict(
            features_dim=128,
            layout=v3.GLOBAL_LAYOUT,
            method_id="UAGMC_SOURCE",
            env_key=env_key,
        )
        algo_cls = PPO
    elif public in SPECIAL_ABC:
        extractor_cls = abc.ControlExtractor
        extractor_kwargs = dict(features_dim=128, layout=_abc_layout(env_key, public))
        algo_cls = PPO
    else:
        extractor_cls = ng.NextGenExtractor
        extractor_kwargs = dict(
            features_dim=128,
            layout=v3.GLOBAL_LAYOUT,
            method_id=internal,
            env_key=env_key,
        )
        m = canonical(internal)
        if v4.custom_method_needs_aux(m):
            algo_cls = v4.DiscoveryAuxPPO
        elif m in v4.NATIVE_WM_METHODS:
            algo_cls = v3.LiteratureWorldModelPPO
        else:
            algo_cls = PPO

    policy_kwargs: Dict[str, Any] = dict(
        features_extractor_class=extractor_cls,
        features_extractor_kwargs=extractor_kwargs,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    policy: Any = "MlpPolicy"
    if control == "J4":
        policy = v4.DiscoveryJointPolicy
        policy_kwargs["joint_arch"] = "J4"
    elif control == "HAPPO_STYLE":
        policy = HAPPOStyleJointPolicy
    elif control == "BPTA_STYLE":
        policy = BPTAStyleJointPolicy

    fr_ppo = _FREEZE["frozen_ppo"]
    return algo_cls(
        policy=policy,
        env=env,
        learning_rate=fixed_horizon_lr(float(fr_ppo["initial_lr"])),
        n_steps=int(profile.n_steps),
        batch_size=int(profile.batch_size),
        n_epochs=int(fr_ppo["n_epochs"]),
        gamma=float(fr_ppo["gamma"]),
        gae_lambda=float(fr_ppo["gae_lambda"]),
        clip_range=float(fr_ppo["clip_range"]),
        ent_coef=float(fr_ppo["ent_coef"]),
        vf_coef=float(fr_ppo["vf_coef"]),
        max_grad_norm=float(fr_ppo["max_grad_norm"]),
        policy_kwargs=policy_kwargs,
        seed=int(seed),
        verbose=1,
        device=device,
        tensorboard_log=str(run_dir / "tb"),
    )


def install_matrix_hooks() -> None:
    v3.CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL
    v3.make_env_factory = matrix_make_env_factory
    v3.build_model = matrix_build_model
    v3.build_callbacks = v4.build_callbacks
    v3.method_description = method_description


# =============================================================================
# Dense frozen validation evaluation
# =============================================================================

def checkpoint_pairs(run_dir: Path) -> List[Tuple[int, Path, Path]]:
    out: List[Tuple[int, Path, Path]] = []
    for model in (run_dir / "checkpoints").glob("uam_ppo_*_steps.zip"):
        m = re.search(r"uam_ppo_(\d+)_steps\.zip$", model.name)
        if not m:
            continue
        step = int(m.group(1))
        vec = run_dir / "checkpoints" / f"uam_ppo_vecnormalize_{step}_steps.pkl"
        if vec.is_file():
            out.append((step, model, vec))
    end = read_json(run_dir / "run_end.json", {})
    final_model = run_dir / "final_rl_model.zip"
    final_vec = run_dir / "final_vec_normalize.pkl"
    final_step = int(end.get("actual_timesteps", 0) or end.get("requested_timesteps", 0) or 0)
    if final_step > 0 and final_model.is_file() and final_vec.is_file():
        out.append((final_step, final_model, final_vec))
    by_step = {int(s): (int(s), m, v) for s, m, v in out}
    return [by_step[k] for k in sorted(by_step)]


def eval_cached(
    *, run_dir: Path, env_key: str, method: str, train_seed: int,
    step: int, model: Path, vec: Path, trace: str, eval_seed: int,
) -> Dict[str, Any]:
    trace_path = Path(trace).resolve()
    cache_dir = run_dir / "analysis" / "matrix_eval_cache" / f"step_{step}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{trace_path.stem}_{sha256(trace_path)[:8]}__seed{eval_seed}.json"
    old = read_json(cache, None)
    if isinstance(old, dict):
        return old
    pre50.set_trace_everywhere(trace_path)
    row = v3.evaluate_checkpoint(
        env_key=env_key,
        method_id=method,
        model_path=model,
        vec_path=vec,
        train_step=step,
        eval_seed=int(eval_seed),
        run_dir=cache_dir / "monitor",
        max_time=HARD_GUARD,
    )
    row = dict(row)
    row.update({
        "trace_path": str(trace_path), "eval_seed": int(eval_seed),
        "train_seed": int(train_seed),
    })
    write_json(cache, row)
    return row


def rolling3_best(dense: Sequence[Dict[str, Any]]) -> Tuple[float, int]:
    vals = [(int(r["step"]), fnum(r["ATT"])) for r in dense if r.get("valid") and math.isfinite(fnum(r.get("ATT")))]
    if len(vals) < 3:
        if not vals:
            return float("nan"), -1
        j = int(np.argmin([v for _, v in vals]))
        return vals[j][1], vals[j][0]
    best_val, best_end = float("inf"), -1
    for i in range(2, len(vals)):
        x = float(np.mean([vals[i - 2][1], vals[i - 1][1], vals[i][1]]))
        if x < best_val:
            best_val, best_end = x, vals[i][0]
    return best_val, best_end


def evaluate_cell(
    *, run_dir: Path, env_key: str, method: str, train_seed: int,
    heldout: Sequence[str], eval_seeds: Sequence[int],
) -> Dict[str, Any]:
    out = run_dir / "analysis" / "MATRIX_MODEL_SUMMARY.json"
    old = read_json(out, None)
    if isinstance(old, dict) and old.get("status") == "VALID":
        return old

    pairs = checkpoint_pairs(run_dir)
    pairs = [(s, m, v) for s, m, v in pairs if s <= int(_ACTIVE_TIMESTEPS)]
    if not pairs:
        raise RuntimeError(f"no checkpoint pairs in {run_dir}")
    canonical_trace = str(Path(heldout[0]).resolve())

    dense: List[Dict[str, Any]] = []
    for step, model, vec in pairs:
        r = eval_cached(
            run_dir=run_dir, env_key=env_key, method=method,
            train_seed=train_seed, step=step, model=model, vec=vec,
            trace=canonical_trace, eval_seed=123,
        )
        dense.append({
            "step": step,
            "ATT": fnum(r.get("ATT")),
            "AWT": fnum(r.get("AWT")),
            "completion": fnum(r.get("completion_rate")),
            "valid": bool(r.get("valid_full_completion")) and math.isfinite(fnum(r.get("ATT"))),
        })
    valid_dense = [r for r in dense if r["valid"]]
    if not valid_dense:
        raise RuntimeError(f"no valid dense ATT points: {env_key}/{method}")
    max_step = max(int(r["step"]) for r in valid_dense)
    late_start = 0.80 * float(max_step)
    late_pairs = [(s, m, v) for s, m, v in pairs if s >= late_start]

    late_bank: List[Dict[str, Any]] = []
    for step, model, vec in late_pairs:
        episodes = []
        for trace in heldout:
            for es in eval_seeds:
                episodes.append(eval_cached(
                    run_dir=run_dir, env_key=env_key, method=method,
                    train_seed=train_seed, step=step, model=model, vec=vec,
                    trace=trace, eval_seed=int(es),
                ))
        valid = [r for r in episodes if bool(r.get("valid_full_completion")) and math.isfinite(fnum(r.get("ATT")))]
        if len(valid) != len(episodes):
            raise RuntimeError(f"incomplete heldout episode: {env_key}/{method}/step{step}")
        late_bank.append({
            "step": step,
            "n_eval": len(episodes),
            "n_valid": len(valid),
            "ATT": mean(r.get("ATT") for r in valid),
            "AWT": mean(r.get("AWT") for r in valid),
        })

    dense_vals = [float(r["ATT"]) for r in valid_dense]
    best_single = min(dense_vals)
    best_single_step = int(valid_dense[int(np.argmin(dense_vals))]["step"])
    best_r3, best_r3_end = rolling3_best(valid_dense)
    dense_late = [r for r in valid_dense if int(r["step"]) >= late_start]
    late_vals = [float(r["ATT"]) for r in dense_late]
    dense_late_mean = mean(late_vals)
    retention_r3 = (
        (dense_late_mean - best_r3) / best_r3
        if math.isfinite(best_r3) and best_r3 > 0 else float("nan")
    )
    xs = [float(r["step"]) / 100_000.0 for r in dense_late]
    ys = late_vals
    slope = float("nan")
    if len(xs) >= 2:
        xb, yb = mean(xs), mean(ys)
        den = sum((x - xb) ** 2 for x in xs)
        if den > 0:
            slope = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / den

    lwatt = mean(r["ATT"] for r in late_bank)
    final_bank = float(late_bank[-1]["ATT"])
    summary = {
        "status": "VALID",
        "env_key": env_key,
        "environment_stage": split_key(env_key)[0],
        "control": split_key(env_key)[1],
        "method": method,
        "family": PUBLIC_FAMILY[method],
        "train_seed": int(train_seed),
        "requested_timesteps": int(_ACTIVE_TIMESTEPS),
        "lr_horizon_steps": int(LR_HORIZON_STEPS),
        "LWATT_validation": lwatt,
        "Final_validation": final_bank,
        "late_window_start_fraction": 0.80,
        "late_window_steps": [int(r["step"]) for r in late_bank],
        "dense_best_single": best_single,
        "dense_best_single_step": best_single_step,
        "best_rolling3_ATT": best_r3,
        "best_rolling3_end_step": best_r3_end,
        "rolling3_to_late_retention": retention_r3,
        "dense_late_mean": dense_late_mean,
        "dense_late_sd": sample_sd(late_vals),
        "dense_late_range": max(late_vals) - min(late_vals) if late_vals else float("nan"),
        "dense_late_slope_min_per_100k": slope,
        "validation_traces": len(heldout),
        "validation_eval_seeds": list(eval_seeds),
        "primary_rank_metric": "LWATT_validation",
        "selection_priority": "absolute ATT + within-run late stability; no cross-train-seed ranking",
    }
    write_csv(run_dir / "analysis" / "MATRIX_DENSE_CURVE.csv", dense)
    write_csv(run_dir / "analysis" / "MATRIX_LATE_BANK.csv", late_bank)
    write_json(out, summary)

    end = read_json(run_dir / "run_end.json", {})
    end["matrix_analysis"] = summary
    end["status"] = "COMPLETE"
    write_json(run_dir / "run_end.json", end)
    return summary


# =============================================================================
# Design / training / aggregation
# =============================================================================

def cell_root(root: Path, seed: int) -> Path:
    p = root / "RL" / f"seed{int(seed)}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_plan(methods: Sequence[str], controls: Sequence[str], envs: Sequence[str]) -> List[Tuple[str, str]]:
    # Method-major is deliberate: UAGMC surface completes first by default.
    plan: List[Tuple[str, str]] = []
    for method in methods:
        for stage in envs:
            for control in controls:
                plan.append((synthetic_key(stage, control), method))
    return plan


def run_manifest(root: Path, fr: Dict[str, Any], profile, plan: Sequence[Tuple[str, str]], seed: int) -> None:
    manifest = {
        "schema": "UAM_100M_3D_MATRIX_V1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "driver": Path(__file__).name,
        "driver_sha256": sha256(Path(__file__)),
        "git_commit_current": current_commit(),
        "freeze_file": fr["_path"],
        "freeze_sha256": fr["_sha256"],
        "freeze_git_commit": fr.get("git_commit"),
        "train_seed": int(seed),
        "timesteps_per_cell": int(_ACTIVE_TIMESTEPS),
        "lr_horizon_steps": int(LR_HORIZON_STEPS),
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "profile": asdict(profile),
        "environment_axis": list(ENV_STAGES),
        "control_axis": list(CONTROLS),
        "method_axis": [method_description(m) for m in METHODS],
        "selected_plan": [{"env_key": e, "method": m} for e, m in plan],
        "nominal_full_matrix_cells": len(METHODS) * len(CONTROLS) * len(ENV_STAGES),
        "nominal_full_matrix_steps": len(METHODS) * len(CONTROLS) * len(ENV_STAGES) * DEFAULT_TIMESTEPS,
        "reserved_rescue_steps": 5 * DEFAULT_TIMESTEPS,
        "literature_control_boundary": {
            "HAPPO_STYLE": "heterogeneous/sequential actor card; PPO optimizer retained; not verbatim HAPPO",
            "BPTA_STYLE": "differentiable backward-response actor card; PPO optimizer retained; not verbatim BPPO",
        },
        "historical_data_policy": "historical runs select candidates only; no historical ATT is inserted into this locked matrix",
    }
    path = root / "MATRIX_MANIFEST.json"
    old = read_json(path, None)
    if old is not None:
        # Resume is allowed only if the immutable scientific contract matches.
        immutable = [
            "freeze_sha256", "train_seed", "timesteps_per_cell", "lr_horizon_steps",
            "checkpoint_interval", "environment_axis", "control_axis", "method_axis",
        ]
        mismatch = [k for k in immutable if old.get(k) != manifest.get(k)]
        if mismatch:
            raise RuntimeError(f"resume manifest mismatch: {mismatch}")
    else:
        write_json(path, manifest)


def train_one(root: Path, env_key: str, method: str, seed: int, profile, device: str) -> Path:
    rr = cell_root(root, seed)
    run_dir = rr / v3.cell_id(env_key, method)
    end = read_json(run_dir / "run_end.json", {})
    if end.get("status") == "COMPLETE" and (run_dir / "analysis" / "MATRIX_MODEL_SUMMARY.json").exists():
        print(f"[CACHE COMPLETE] {env_key}/{method}", flush=True)
        return run_dir

    train_trace = Path(_FREEZE["demand_protocol"]["train_trace_bank"][0]).resolve()
    pre50.set_trace_everywhere(train_trace)
    try:
        if not v3.training_complete(run_dir, int(_ACTIVE_TIMESTEPS)):
            v3.train_cell_only(
                root=rr,
                env_key=env_key,
                method_id=method,
                requested_steps=int(_ACTIVE_TIMESTEPS),
                train_seed=int(seed),
                profile=profile,
                device=device,
                max_time=HARD_GUARD,
                defer_eval=True,
            )
        heldout = list(_FREEZE["demand_protocol"]["heldout_trace_bank"])
        evaluate_cell(
            run_dir=run_dir,
            env_key=env_key,
            method=method,
            train_seed=seed,
            heldout=heldout,
            eval_seeds=DEFAULT_EVAL_SEEDS,
        )
    except Exception:
        write_json(run_dir / "MATRIX_FAILURE.json", {
            "env_key": env_key,
            "method": method,
            "seed": seed,
            "error": traceback.format_exc(),
        })
        raise
    finally:
        try:
            v3.core.restore_process_patches()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return run_dir


def aggregate(root: Path, plan: Sequence[Tuple[str, str]], seed: int) -> None:
    rows: List[Dict[str, Any]] = []
    rr = cell_root(root, seed)
    for env_key, method in plan:
        p = rr / v3.cell_id(env_key, method) / "analysis" / "MATRIX_MODEL_SUMMARY.json"
        s = read_json(p, None)
        if isinstance(s, dict) and s.get("status") == "VALID":
            rows.append(dict(s))

    # Same env/control UAGMC and CURRENT_BASE references.
    ref_u = {(r["environment_stage"], r["control"]): r for r in rows if r["method"] == "UAGMC_SOURCE"}
    ref_b = {(r["environment_stage"], r["control"]): r for r in rows if r["method"] == "CURRENT_BASE"}
    for r in rows:
        key = (r["environment_stage"], r["control"])
        u = ref_u.get(key)
        b = ref_b.get(key)
        if u:
            uv = fnum(u["LWATT_validation"])
            rv = fnum(r["LWATT_validation"])
            r["UAGMC_LWATT"] = uv
            r["delta_vs_UAGMC"] = rv - uv
            r["improvement_vs_UAGMC_pct"] = 100.0 * (uv - rv) / uv if uv > 0 else float("nan")
        if b:
            bv = fnum(b["LWATT_validation"])
            rv = fnum(r["LWATT_validation"])
            r["BASE_LWATT"] = bv
            r["delta_vs_CURRENT_BASE"] = rv - bv
            r["improvement_vs_CURRENT_BASE_pct"] = 100.0 * (bv - rv) / bv if bv > 0 else float("nan")

    write_csv(root / "MATRIX_RESULTS.csv", rows)
    write_json(root / "MATRIX_RESULTS.json", rows)

    # Compact pivot-like table for quick inspection.
    quick = sorted(rows, key=lambda r: (
        ENV_STAGES.index(r["environment_stage"]),
        CONTROLS.index(r["control"]),
        METHODS.index(r["method"]),
    ))
    write_csv(root / "MATRIX_RESULTS_SORTED.csv", quick)


def print_plan(plan: Sequence[Tuple[str, str]], timesteps: int) -> None:
    print("=" * 108)
    print("UAM 100M 3-D MATRIX V1")
    print("=" * 108)
    print(f"selected cells : {len(plan)}")
    print(f"steps / cell   : {timesteps:,}")
    print(f"selected budget: {len(plan) * timesteps:,}")
    print(f"full main cube : {len(METHODS)*len(CONTROLS)*len(ENV_STAGES)} cells = 96,000,000")
    print(f"rescue reserve : 5 x 800,000 = 4,000,000 (manual only)")
    print(f"LR horizon     : {LR_HORIZON_STEPS:,}")
    print("order          : method-major; UAGMC surface first")
    print("-" * 108)
    for i, (e, m) in enumerate(plan, 1):
        print(f"{i:03d}  {m:<24}  {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--freeze", type=str, required=True, help="UAM_FINAL_FREEZE_V2.json")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p.add_argument("--envs", type=str, default="", help="comma list: S1,S2,S3")
    p.add_argument("--controls", type=str, default="", help="comma list: LQ,J4,HAPPO_STYLE,BPTA_STYLE")
    p.add_argument("--methods", type=str, default="", help="comma list of method IDs")
    p.add_argument("--phase", choices=("all", "uagmc", "base", "methods"), default="all")
    p.add_argument("--resume-root", type=str, default="")
    p.add_argument("--output-root", type=str, default="")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--max-cells", type=int, default=0, help="debug only; 0=no limit")
    return p.parse_args()


def main() -> int:
    global _ACTIVE_TIMESTEPS, _FREEZE
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available() and not args.plan_only:
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if int(args.timesteps) <= 0 or int(args.timesteps) > LR_HORIZON_STEPS:
        raise ValueError(f"timesteps must be 1..{LR_HORIZON_STEPS}")
    _ACTIVE_TIMESTEPS = int(args.timesteps)

    _FREEZE = verify_freeze_compatible(Path(args.freeze))
    install_registry_and_specs()
    assert_runtime_matches_freeze(_FREEZE)
    install_matrix_hooks()

    profile = v3.SpeedProfile(
        str(_FREEZE["frozen_ppo"]["profile_name"]),
        int(_FREEZE["frozen_ppo"]["n_envs"]),
        int(_FREEZE["frozen_ppo"]["n_steps"]),
        int(_FREEZE["frozen_ppo"]["batch_size"]),
    )

    envs = parse_csv_choice(args.envs, ENV_STAGES)
    controls = parse_csv_choice(args.controls, CONTROLS)
    methods = parse_csv_choice(args.methods, METHODS)
    if args.phase == "uagmc":
        methods = ["UAGMC_SOURCE"]
    elif args.phase == "base":
        methods = ["CURRENT_BASE"]
    elif args.phase == "methods" and not args.methods:
        methods = [m for m in METHODS if m not in ("UAGMC_SOURCE", "CURRENT_BASE")]

    plan = build_plan(methods, controls, envs)
    if args.max_cells > 0:
        plan = plan[: int(args.max_cells)]
    print_plan(plan, _ACTIVE_TIMESTEPS)
    if args.plan_only:
        return 0

    if args.resume_root:
        root = Path(args.resume_root).expanduser().resolve()
    elif args.output_root:
        root = Path(args.output_root).expanduser().resolve()
    else:
        root = ROOT / "serial_runs" / f"uam_100m_3d_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root.mkdir(parents=True, exist_ok=True)
    run_manifest(root, _FREEZE, profile, plan, int(args.seed))

    failures: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, (env_key, method) in enumerate(plan, 1):
        print("\n" + "=" * 108, flush=True)
        print(f"[CELL {i:03d}/{len(plan):03d}] {method} @ {env_key} | seed={args.seed} | {_ACTIVE_TIMESTEPS:,}", flush=True)
        print("=" * 108, flush=True)
        try:
            train_one(root, env_key, method, int(args.seed), profile, args.device)
            aggregate(root, plan, int(args.seed))
        except Exception as exc:
            failures.append({
                "env_key": env_key, "method": method, "seed": int(args.seed),
                "error": repr(exc), "traceback": traceback.format_exc(),
            })
            write_json(root / "FAILURES.json", failures)
            print(f"[FAILED] {method} @ {env_key}: {exc!r}", flush=True)
            if args.fail_fast:
                raise

    aggregate(root, plan, int(args.seed))
    write_json(root / "RUN_COMPLETE.json", {
        "status": "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
        "cells_requested": len(plan),
        "failures": len(failures),
        "elapsed_hours": (time.time() - t0) / 3600.0,
        "nominal_transitions_requested": len(plan) * int(_ACTIVE_TIMESTEPS),
        "root": str(root),
    })
    print(f"\n[DONE] root={root}")
    print(f"[DONE] failures={len(failures)} | results={root/'MATRIX_RESULTS.csv'}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
