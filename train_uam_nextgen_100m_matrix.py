# -*- coding: utf-8 -*-
"""
UAM NEXT-GEN 100.2M MATRIX
==========================

Purpose
-------
One large follow-up matrix after the V4/V5 discovery rounds.
It does NOT change the frozen UAM physics, reward, completion semantics, or V5
clean joint dispatcher.  The goal is to spend the next ~100M transitions on:

  1) stronger S / AC / WM mechanisms,
  2) upgrades inspired by later literature,
  3) controlled two-way / limited three-way interactions,
  4) multi-train-seed robustness rather than one lucky 600k trajectory.

Budget
------
Single S3:
    34 method configs x train seeds (2,3) x 600k = 40.8M
Joint J0 (V5 clean minimal target-priority control):
    33 method configs x train seeds (2,3,4) x 600k = 59.4M
TOTAL:
    167 cells x 600k = 100.2M requested PPO transitions

Why seeds start from 2
----------------------
Seed=1 already exists for many anchors from the previous Single/V5 matrices.
This run spends new compute on seeds 2/3 (Single) and 2/3/4 (Joint).  For old
methods, seed=1 can later be merged with this output; new methods still receive
2 Single seeds and 3 Joint seeds in this matrix.

Frozen environment contract
---------------------------
Single:
    S3 physics, passenger PPO V0/V1, original responsive Longest-Queue aircraft
    repositioning.
Joint:
    J0 policy skeleton + V5 clean minimal reposition wrapper.  PPO changes only
    aircraft target priority when BOTH queues are positive; original simulator
    dispatch timing/capacity/eligibility stay intact.

Production profile
------------------
    P40 = 40 env x 512 steps = 20,480 rollout, batch=4096, CUDA.
    50k nominal checkpoints through 600k.
    Eval seeds = 123,124,125.

Scientific organization
-----------------------
About half of each matrix is standalone S/AC/WM exploration; the other half is
controlled S+AC, S+WM, AC+WM, plus only two three-way tests.

New STYLE/INSPIRED cards (not verbatim reproductions)
-----------------------------------------------------
S_DYGFORMER_PATCH
    DyGFormer-style temporal-history patching over committed event tokens.
S_GRAPHMIXER_TEMP
    GraphMixer-style simple temporal token/channel mixing over multi-horizon
    committed projections.
A_QPLEX_DUPLEX
    QPLEX-inspired shared baseline + centered candidate-advantage latent.
A_FACMAC_MIXER
    FACMAC-inspired factorized passenger/aircraft utility mixing latent.
W_TDMPC2_VALUE
    TD-MPC2-inspired value-consistent known-future card implemented as
    TDM + value-equivalent event residual (uses existing trained auxiliaries).
W_DREAMER_BALANCED
    Dreamer-style robust latent/value card implemented as uncertainty ensemble
    + value-equivalent event residual (uses existing trained auxiliaries).

IMPORTANT
---------
These labels denote controlled mechanism transplants inside the existing PPO
stack.  Do NOT claim verbatim reproduction of DyGFormer, GraphMixer, QPLEX,
FACMAC, TD-MPC2, DreamerV3, etc.

Required files beside this script
---------------------------------
train_uam_60m_literature_matrix_v3_1.py
train_uam_60m_jointfirst_v4_deferred_joint_eval.py
train_uam_60m_jointfirst_v5_minimal_reposition.py
train_uam_s3_20x600k_litupgrade.py

Examples
--------
Plan only:
    python train_uam_nextgen_100m_matrix.py --plan-only

Full run:
    CUDA_VISIBLE_DEVICES=0 python train_uam_nextgen_100m_matrix.py

Single only:
    CUDA_VISIBLE_DEVICES=0 python train_uam_nextgen_100m_matrix.py --phase single

Joint only:
    CUDA_VISIBLE_DEVICES=0 python train_uam_nextgen_100m_matrix.py --phase joint

Smoke one method:
    CUDA_VISIBLE_DEVICES=0 python train_uam_nextgen_100m_matrix.py \
        --phase single --methods S_DYGFORMER_PATCH --timesteps 50000 \
        --single-seeds 2 --eval-seeds 123

Resume:
    CUDA_VISIBLE_DEVICES=0 python train_uam_nextgen_100m_matrix.py \
        --resume-root serial_runs/uam_nextgen100m_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
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
import torch.nn.functional as F
from stable_baselines3 import PPO

import train_uam_60m_literature_matrix_v3_1 as v3
import train_uam_60m_jointfirst_v4_deferred_joint_eval as v4
import train_uam_60m_jointfirst_v5_minimal_reposition as v5
import train_uam_s3_20x600k_litupgrade as up


# =============================================================================
# Constants
# =============================================================================

ROOT = Path(__file__).resolve().parent
DEFAULT_TIMESTEPS = 600_000
CHECKPOINT_INTERVAL = 50_000
DEFAULT_EVAL_SEEDS = (123, 124, 125)
DEFAULT_SINGLE_SEEDS = (2, 3)
DEFAULT_JOINT_SEEDS = (2, 3, 4)
MAX_TIME = 2_500
SINGLE_ENV = "S3"
JOINT_ENV = "J0"

P40 = v3.SpeedProfile(
    "P40_40x512_b4096",
    n_envs=40,
    n_steps=512,
    batch_size=4096,
)
assert P40.rollout == 20_480


# =============================================================================
# Matrix design
# =============================================================================

# 17 standalone + 17 interactions = 34 Single configs.
SINGLE_METHODS: Tuple[str, ...] = (
    # ---- anchors / best surviving S ----
    "CURRENT",
    "R_EVENTQ",
    "S_TDM_EVENT_FUSION",
    "S_GATV2_EVENT",
    "S_TGAT_EVENT",
    "S_ADAPTIVE_EFFECT_HORIZON",
    "S_DYGFORMER_PATCH",
    "S_GRAPHMIXER_TEMP",

    # ---- AC ----
    "A_ICM_AC",
    "A_DENOISED_VALUE",
    "A_ANCHOR_QUOTIENT",
    "A_QPLEX_DUPLEX",
    "A_FACMAC_MIXER",

    # ---- WM ----
    "W_VALUE_CONTROLLABLE",
    "W_UNC_VALUE",
    "W_TDMPC2_VALUE",
    "W_DREAMER_BALANCED",

    # ---- S + AC ----
    "SA_TDMFUSION_ICM",
    "SA_GATV2_ICM",
    "SA_GATV2_DENOISED",
    "SA_DYGFORMER_ICM",
    "SA_GRAPHMIXER_QPLEX",
    "SA_TGAT_FACMAC",

    # ---- S + WM ----
    "SW_GATV2_VALUECTRL",
    "SW_DYGFORMER_TDMPC",
    "SW_GRAPHMIXER_DREAMER",
    "SW_TDMFUSION_VALUECTRL",
    "SW_TGAT_UNCVALUE",

    # ---- AC + WM ----
    "AW_ICM_VALUECTRL",
    "AW_DENOISED_CVAML",
    "AW_QPLEX_DREAMER",
    "AW_FACMAC_VALUECTRL",

    # ---- deliberately limited three-way ----
    "SAW_GATV2_ICM_VALUECTRL",
    "SAW_DYGFORMER_QPLEX_VE",
)
assert len(SINGLE_METHODS) == 34

# 16 standalone + the same 17 interactions = 33 Joint configs.
# Joint stays on J0 to avoid mixing architecture effects back into the method screen.
JOINT_METHODS: Tuple[str, ...] = (
    "CURRENT",
    "R_TDM",
    "R_EVENTQ",
    "S_TDM_EVENT_FUSION",
    "S_GATV2_EVENT",
    "S_TGAT_EVENT",
    "S_DYGFORMER_PATCH",
    "S_GRAPHMIXER_TEMP",
    "A_ICM_AC",
    "A_DENOISED_VALUE",
    "A_QPLEX_DUPLEX",
    "A_FACMAC_MIXER",
    "W_VALUE_CONTROLLABLE",
    "W_TDMPC2_VALUE",
    "W_DREAMER_BALANCED",
    "W_TDM_CVAML",

    "SA_TDMFUSION_ICM",
    "SA_GATV2_ICM",
    "SA_GATV2_DENOISED",
    "SA_DYGFORMER_ICM",
    "SA_GRAPHMIXER_QPLEX",
    "SA_TGAT_FACMAC",

    "SW_GATV2_VALUECTRL",
    "SW_DYGFORMER_TDMPC",
    "SW_GRAPHMIXER_DREAMER",
    "SW_TDMFUSION_VALUECTRL",
    "SW_TGAT_UNCVALUE",

    "AW_ICM_VALUECTRL",
    "AW_DENOISED_CVAML",
    "AW_QPLEX_DREAMER",
    "AW_FACMAC_VALUECTRL",

    "SAW_GATV2_ICM_VALUECTRL",
    "SAW_DYGFORMER_QPLEX_VE",
)
assert len(JOINT_METHODS) == 33


# Registry entries must resolve to <=3 fixed-width mechanism latents because
# V4's capacity-matched fusion intentionally supports at most three components.
NEW_COMBOS: Dict[str, Tuple[str, ...]] = {
    # WM upgrade aliases.
    "W_TDMPC2_VALUE": (
        "R_TDM",
        "W_EVENTQ_VE_EMA",
    ),
    "W_DREAMER_BALANCED": (
        "W_GAMMA_PETS_STEVE",
        "W_EVENTQ_VE_EMA",
    ),

    # S + AC.
    "SA_TDMFUSION_ICM": (
        "S_TDM_EVENT_FUSION",
        "A_ICM_AC",
    ),
    "SA_GATV2_ICM": (
        "S_GATV2_EVENT",
        "A_ICM_AC",
    ),
    "SA_GATV2_DENOISED": (
        "S_GATV2_EVENT",
        "A_ICM_AC",
        "A_COMA_AC",
    ),
    "SA_DYGFORMER_ICM": (
        "S_DYGFORMER_PATCH",
        "A_ICM_AC",
    ),
    "SA_GRAPHMIXER_QPLEX": (
        "S_GRAPHMIXER_TEMP",
        "A_QPLEX_DUPLEX",
    ),
    "SA_TGAT_FACMAC": (
        "S_TGAT_EVENT",
        "A_FACMAC_MIXER",
    ),

    # S + WM.
    "SW_GATV2_VALUECTRL": (
        "S_GATV2_EVENT",
        "W_EVENTQ_VE_EMA",
        "W_EVENTQ_COCO_CF",
    ),
    "SW_DYGFORMER_TDMPC": (
        "S_DYGFORMER_PATCH",
        "R_TDM",
        "W_EVENTQ_VE_EMA",
    ),
    "SW_GRAPHMIXER_DREAMER": (
        "S_GRAPHMIXER_TEMP",
        "W_GAMMA_PETS_STEVE",
        "W_EVENTQ_VE_EMA",
    ),
    "SW_TDMFUSION_VALUECTRL": (
        "S_TDM_EVENT_FUSION",
        "W_EVENTQ_VE_EMA",
        "W_EVENTQ_COCO_CF",
    ),
    "SW_TGAT_UNCVALUE": (
        "S_TGAT_EVENT",
        "W_GAMMA_PETS_STEVE",
        "W_TDM_CVAML",
    ),

    # AC + WM.
    "AW_ICM_VALUECTRL": (
        "A_ICM_AC",
        "W_EVENTQ_VE_EMA",
        "W_EVENTQ_COCO_CF",
    ),
    "AW_DENOISED_CVAML": (
        "A_ICM_AC",
        "A_COMA_AC",
        "W_TDM_CVAML",
    ),
    "AW_QPLEX_DREAMER": (
        "A_QPLEX_DUPLEX",
        "W_GAMMA_PETS_STEVE",
        "W_EVENTQ_VE_EMA",
    ),
    "AW_FACMAC_VALUECTRL": (
        "A_FACMAC_MIXER",
        "W_EVENTQ_VE_EMA",
        "W_EVENTQ_COCO_CF",
    ),

    # Limited three-way interactions.
    "SAW_GATV2_ICM_VALUECTRL": (
        "S_GATV2_EVENT",
        "A_ICM_AC",
        "W_EVENTQ_VE_EMA",
    ),
    "SAW_DYGFORMER_QPLEX_VE": (
        "S_DYGFORMER_PATCH",
        "A_QPLEX_DUPLEX",
        "W_EVENTQ_VE_EMA",
    ),
}


METHOD_META: Dict[str, Dict[str, str]] = {
    "CURRENT": {
        "family": "ANCHOR",
        "role": "current-state anchor",
        "literature": "project anchor",
    },
    "R_TDM": {
        "family": "S",
        "role": "effect-time temporal decision summary",
        "literature": "project TDM anchor",
    },
    "R_EVENTQ": {
        "family": "S",
        "role": "effect-time query over committed events",
        "literature": "project EVENTQ anchor",
    },
    "S_TDM_EVENT_FUSION": {
        "family": "S",
        "role": "strong V5 Joint shared/event fusion control",
        "literature": "project-specific TDM + committed-event fusion",
    },
    "S_GATV2_EVENT": {
        "family": "S",
        "role": "dynamic candidate relation attention",
        "literature": "GATv2-inspired",
    },
    "S_TGAT_EVENT": {
        "family": "S",
        "role": "event-level temporal attention",
        "literature": "TGAT/TGN-inspired",
    },
    "S_ADAPTIVE_EFFECT_HORIZON": {
        "family": "S",
        "role": "effect-centered adaptive multi-horizon aggregation",
        "literature": "adaptive-horizon inspired",
    },
    "S_DYGFORMER_PATCH": {
        "family": "S",
        "role": "patch temporal event history before relation mixing",
        "literature": "DyGFormer-style history patching",
    },
    "S_GRAPHMIXER_TEMP": {
        "family": "S",
        "role": "simple token/channel mixing of temporal projections",
        "literature": "GraphMixer-style temporal mixing",
    },
    "A_ICM_AC": {
        "family": "AC",
        "role": "strong V5 Joint controllability control",
        "literature": "ICM-inspired inverse dynamics",
    },
    "A_DENOISED_VALUE": {
        "family": "AC",
        "role": "controllability plus value relevance",
        "literature": "Denoised-MDP/COMA-inspired",
    },
    "A_ANCHOR_QUOTIENT": {
        "family": "AC",
        "role": "absolute anchor plus relative action quotient",
        "literature": "counterfactual quotient-inspired",
    },
    "A_QPLEX_DUPLEX": {
        "family": "AC",
        "role": "shared baseline plus centered candidate advantage",
        "literature": "QPLEX-inspired duplex advantage decomposition",
    },
    "A_FACMAC_MIXER": {
        "family": "AC",
        "role": "factorized passenger/aircraft utility mixer",
        "literature": "FACMAC-inspired nonlinear factor mixing",
    },
    "W_VALUE_CONTROLLABLE": {
        "family": "WM",
        "role": "value relevance plus controllability residual",
        "literature": "Value Equivalence + CoCo-inspired",
    },
    "W_UNC_VALUE": {
        "family": "WM",
        "role": "uncertainty plus value-aware residual",
        "literature": "PETS/STEVE + calibrated-VAML-inspired",
    },
    "W_TDMPC2_VALUE": {
        "family": "WM",
        "role": "decision-time value-consistent known-future latent",
        "literature": "TD-MPC2-inspired value consistency; controlled transplant",
    },
    "W_DREAMER_BALANCED": {
        "family": "WM",
        "role": "uncertainty/value-balanced latent world model",
        "literature": "Dreamer-style robust latent/value modeling; controlled transplant",
    },
    "W_TDM_CVAML": {
        "family": "WM",
        "role": "value-aware TDM residual comparator",
        "literature": "calibrated-VAML inspired",
    },
}


# =============================================================================
# Helpers
# =============================================================================

def canonical(x: Any) -> str:
    return str(x).strip().upper()


def parse_ints(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty integer list")
    return vals


def parse_methods(text: str, universe: Sequence[str]) -> List[str]:
    if not str(text).strip():
        return list(universe)
    vals = [canonical(x) for x in str(text).split(",") if x.strip()]
    allowed = {canonical(x) for x in universe}
    bad = [x for x in vals if x not in allowed]
    if bad:
        raise ValueError(f"unsupported methods for selected phase: {bad}")
    return vals


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes"}


def install_registry() -> None:
    # Register the prior 20-cell upgrades first.
    up.install_registry()
    for name, comps in NEW_COMBOS.items():
        v4.COMBO_REGISTRY[canonical(name)] = tuple(canonical(c) for c in comps)


def components_for(method_id: str) -> Tuple[str, ...]:
    return tuple(v4.components_for(canonical(method_id)))


def infer_family(method_id: str) -> str:
    m = canonical(method_id)
    if m in METHOD_META:
        return METHOD_META[m]["family"]
    comps = components_for(m)
    families = []
    for c in comps:
        cc = canonical(c)
        if cc.startswith("S_") or cc in {"R_TDM", "R_EVENTQ"}:
            families.append("S")
        elif cc.startswith("A_"):
            families.append("AC")
        elif cc.startswith("W_"):
            families.append("WM")
    uniq = []
    for f in families:
        if f not in uniq:
            uniq.append(f)
    return "+".join(uniq) if uniq else v4.method_family(m)


def method_description(method_id: str) -> Dict[str, Any]:
    m = canonical(method_id)
    meta = METHOD_META.get(m, {})
    comps = list(components_for(m))
    return {
        "method_id": m,
        "family": meta.get("family", infer_family(m)),
        "components": comps,
        "role": meta.get("role", "controlled interaction card"),
        "literature_or_role": meta.get(
            "literature",
            "controlled S/AC/WM interaction built from registered components",
        ),
        "style_inspired_not_verbatim": True,
    }


# =============================================================================
# Next-generation extractor
# =============================================================================

class NextGenExtractor(up.LiteratureV2Extractor):
    """
    Prior V4 + 20-cell extractor, plus four new fixed-width mechanism cards.

    New cards are deliberately representation modules.  PPO still optimizes the
    task objective; old auxiliary modules (ICM/COMA/VE/PETS/CoCo/CVAML) retain
    their validated V4 auxiliary losses whenever they appear in a combo.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        K = v3.N_CANDIDATES
        H = len(v3.GAMMA_HORIZONS)
        P = v3.PROJECT_PER_CAND
        self._ng_K = K
        self._ng_H = H
        self._ng_P = P

        # ------------------------------------------------------------------
        # DyGFormer-style temporal patching.
        # Each candidate has 2*K_event event tokens (aircraft + passenger).
        # We encode each event relative to that candidate's action-effect time,
        # group consecutive sorted events into fixed patches, mix patches, then
        # aggregate both candidates.  This is a mechanism transplant, not the
        # original DyGFormer architecture.
        # ------------------------------------------------------------------
        self.ng_patch_size = 4
        self.ng_event_width = 2 * v3.MAX_EVENTS_PER_TYPE
        if self.ng_event_width % self.ng_patch_size != 0:
            raise RuntimeError("event width must be divisible by patch size")
        self.ng_n_patches = self.ng_event_width // self.ng_patch_size

        self.ng_event_token = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.ng_patch_proj = nn.Sequential(
            nn.Linear(self.ng_patch_size * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.ng_patch_attn = nn.MultiheadAttention(
            64, num_heads=4, batch_first=True
        )
        self.ng_patch_out = nn.Sequential(
            nn.Linear(64 * K, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # GraphMixer-style simple temporal mixer.
        # Fixed tokens = candidate x horizon.  Each token sees one projected
        # state plus signed/absolute distance from focal effect time.
        # ------------------------------------------------------------------
        self.ng_mixer_tokens = K * H
        self.ng_mix_token_enc = nn.Sequential(
            nn.Linear(P + 2, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
        )
        self.ng_token_mlp = nn.Sequential(
            nn.Linear(self.ng_mixer_tokens, self.ng_mixer_tokens * 2),
            nn.GELU(),
            nn.Linear(self.ng_mixer_tokens * 2, self.ng_mixer_tokens),
        )
        self.ng_channel_mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )
        self.ng_mix_out = nn.Sequential(
            nn.Linear(32 * K, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # QPLEX-inspired duplex advantage latent.
        # Preserve an absolute shared baseline while explicitly encoding
        # centered candidate utilities / pairwise advantage contrast.
        # ------------------------------------------------------------------
        self.ng_qplex_candidate = nn.Sequential(
            nn.Linear(P, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.ng_qplex_anchor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.ng_qplex_pair = nn.Sequential(
            nn.Linear(v3.PAIR_VIRTUAL_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.ng_qplex_out = nn.Sequential(
            nn.Linear(32 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # FACMAC-inspired factorized utility mixer.
        # Passenger candidate utilities and aircraft/pair context remain
        # separate, then a state-conditioned positive mixer couples them.
        # ------------------------------------------------------------------
        self.ng_fac_passenger = nn.Sequential(
            nn.Linear(P, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
        )
        self.ng_fac_aircraft = nn.Sequential(
            nn.Linear(v3.PAIR_VIRTUAL_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.ng_fac_gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.ng_fac_out = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def _candidate_event_features(
        self,
        obs: torch.Tensor,
        c: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        elems, mask = self._candidate_event_tokens(obs, c)
        # inherited elems[...,0] = eta, elems[...,1] = type
        eta = elems[..., 0]
        typ = elems[..., 1]
        tau = self._slice(obs, "tau").reshape(
            obs.shape[0], v3.N_CANDIDATES
        )[:, c:c + 1]
        rel = (eta - tau) / 60.0
        feat = torch.stack(
            [
                eta / 60.0,
                typ,
                rel,
                rel.abs(),
                torch.sin(math.pi * rel),
                torch.cos(math.pi * rel),
            ],
            dim=-1,
        )
        return feat, mask

    def _dygformer_patch_latent(self, obs: torch.Tensor) -> torch.Tensor:
        reps: List[torch.Tensor] = []
        for c in range(v3.N_CANDIDATES):
            feat, mask = self._candidate_event_features(obs, c)
            B, E, _ = feat.shape
            z = self.ng_event_token(feat)
            z = z * mask.unsqueeze(-1)

            z = z.reshape(
                B,
                self.ng_n_patches,
                self.ng_patch_size * 32,
            )
            p = self.ng_patch_proj(z)

            # Patch is valid if it contains at least one real event.
            pmask = mask.reshape(
                B, self.ng_n_patches, self.ng_patch_size
            ).sum(dim=-1) > 0
            key_padding = ~pmask
            all_masked = key_padding.all(dim=1)
            if bool(all_masked.any()):
                key_padding = key_padding.clone()
                key_padding[all_masked, 0] = False

            y, _ = self.ng_patch_attn(
                p, p, p,
                key_padding_mask=key_padding,
                need_weights=False,
            )
            valid = pmask.float()
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (y * valid.unsqueeze(-1)).sum(dim=1) / denom
            reps.append(pooled)

        return self.ng_patch_out(torch.cat(reps, dim=1))

    def _graphmixer_temp_latent(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        K = v3.N_CANDIDATES
        H = len(v3.GAMMA_HORIZONS)
        P = v3.PROJECT_PER_CAND

        g = self._slice(obs, "gamma").reshape(B, K, H, P)
        tau = self._slice(obs, "tau").reshape(B, K)
        horizons = torch.as_tensor(
            list(v3.GAMMA_HORIZONS),
            device=obs.device,
            dtype=obs.dtype,
        ).view(1, 1, H)
        diff = (horizons - tau.unsqueeze(-1)) / 60.0
        feat = torch.cat(
            [g, diff.unsqueeze(-1), diff.abs().unsqueeze(-1)],
            dim=-1,
        )
        z = self.ng_mix_token_enc(feat).reshape(
            B, self.ng_mixer_tokens, 32
        )

        # token mixing
        t = z.transpose(1, 2)  # B,C,T
        t = t + self.ng_token_mlp(t)
        z = t.transpose(1, 2)

        # channel mixing
        z = z + self.ng_channel_mlp(z)
        z = z.reshape(B, K, H, 32).mean(dim=2)
        return self.ng_mix_out(z.flatten(1))

    def _own_projected_candidates(self, obs: torch.Tensor) -> torch.Tensor:
        shared = self._slice(obs, "shared").reshape(
            obs.shape[0],
            v3.N_CANDIDATES,
            v3.PROJECT_PER_CAND,
        )
        srac = self._slice(obs, "srac").reshape(
            obs.shape[0],
            v3.N_CANDIDATES,
            v3.PROJECT_PER_CAND,
        )
        return shared + srac

    def _qplex_duplex_latent(self, obs: torch.Tensor) -> torch.Tensor:
        own = self._own_projected_candidates(obs)
        cand = torch.stack(
            [self.ng_qplex_candidate(own[:, c]) for c in range(v3.N_CANDIDATES)],
            dim=1,
        )
        mean = cand.mean(dim=1)
        centered = cand - mean.unsqueeze(1)
        contrast = centered[:, 0] - centered[:, 1]
        anchor = self.ng_qplex_anchor(self.base_only(obs))
        pair = self.ng_qplex_pair(self._slice(obs, "pair_virtual"))
        x = torch.cat([anchor, mean, contrast, pair], dim=1)
        return self.ng_qplex_out(x)

    def _facmac_mixer_latent(self, obs: torch.Tensor) -> torch.Tensor:
        own = self._own_projected_candidates(obs)
        p0 = self.ng_fac_passenger(own[:, 0])
        p1 = self.ng_fac_passenger(own[:, 1])
        air = self.ng_fac_aircraft(self._slice(obs, "pair_virtual"))
        core = self.base_only(obs)
        # Positive state-conditioned utility weights, FACMAC/QMIX-style spirit.
        w = F.softplus(self.ng_fac_gate(core)) + 1e-3
        mixed = torch.cat(
            [
                w[:, 0:1] * p0,
                w[:, 1:2] * p1,
                w[:, 2:3] * air,
            ],
            dim=1,
        )
        return self.ng_fac_out(mixed)

    def component_latent(
        self,
        obs: torch.Tensor,
        component: str,
    ) -> torch.Tensor:
        c = canonical(component)
        if c == "S_DYGFORMER_PATCH":
            return self._dygformer_patch_latent(obs)
        if c == "S_GRAPHMIXER_TEMP":
            return self._graphmixer_temp_latent(obs)
        if c == "A_QPLEX_DUPLEX":
            return self._qplex_duplex_latent(obs)
        if c == "A_FACMAC_MIXER":
            return self._facmac_mixer_latent(obs)
        return super().component_latent(obs, c)


# =============================================================================
# Model construction / hooks
# =============================================================================

def nextgen_build_model(
    *,
    env,
    env_key: str,
    method_id: str,
    profile: v3.SpeedProfile,
    seed: int,
    run_dir: Path,
    device: str,
):
    spec = v3.ENV_SPECS[env_key]
    policy_kwargs: Dict[str, Any] = dict(
        features_extractor_class=NextGenExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            layout=v3.GLOBAL_LAYOUT,
            method_id=method_id,
            env_key=env_key,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    policy: Any = "MlpPolicy"
    if spec.joint:
        policy = v4.DiscoveryJointPolicy
        policy_kwargs["joint_arch"] = env_key

    m = canonical(method_id)
    if v4.custom_method_needs_aux(m):
        algo_cls = v4.DiscoveryAuxPPO
    elif m in v4.NATIVE_WM_METHODS:
        algo_cls = v3.LiteratureWorldModelPPO
    else:
        algo_cls = PPO

    return algo_cls(
        policy=policy,
        env=env,
        learning_rate=v3.base.linear_schedule(v3.INITIAL_LR),
        n_steps=int(profile.n_steps),
        batch_size=int(profile.batch_size),
        n_epochs=v3.N_EPOCHS,
        gamma=v3.GAMMA,
        gae_lambda=v3.GAE_LAMBDA,
        clip_range=v3.CLIP_RANGE,
        ent_coef=v3.ENT_COEF,
        vf_coef=v3.VF_COEF,
        max_grad_norm=v3.MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        seed=int(seed),
        verbose=1,
        device=device,
        tensorboard_log=str(run_dir / "tb"),
    )


def install_nextgen_hooks() -> None:
    # V5 first: installs clean Joint environment + V4 policies/method semantics.
    v5.install_v5_hooks()

    # Register prior 20-cell cards and all new combinations.
    install_registry()

    # Replace only model/extractor description after V5 has patched env factory.
    v3.build_model = nextgen_build_model
    v3.build_callbacks = v4.build_callbacks
    v3.method_description = method_description

    # Freeze J0 naming / P40 production profile.
    v4.FAST_PROFILE = P40


# =============================================================================
# Plan / aggregation
# =============================================================================

def make_design_rows(
    single_methods: Sequence[str],
    joint_methods: Sequence[str],
    single_seeds: Sequence[int],
    joint_seeds: Sequence[int],
    timesteps: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 0
    for seed in single_seeds:
        for m in single_methods:
            idx += 1
            rows.append({
                "index": idx,
                "phase": "SINGLE",
                "env_key": SINGLE_ENV,
                "train_seed": int(seed),
                "method_id": m,
                "timesteps": int(timesteps),
                **method_description(m),
            })
    for seed in joint_seeds:
        for m in joint_methods:
            idx += 1
            rows.append({
                "index": idx,
                "phase": "JOINT",
                "env_key": JOINT_ENV,
                "train_seed": int(seed),
                "method_id": m,
                "timesteps": int(timesteps),
                **method_description(m),
            })
    return rows


def read_curve(run_dir: Path) -> List[Dict[str, Any]]:
    return v3.read_csv(run_dir / "analysis" / "checkpoint_curve.csv")


def curve_metrics(run_dir: Path) -> Dict[str, Any]:
    rows = read_curve(run_dir)
    pts: List[Tuple[int, float, float, float]] = []
    for r in rows:
        att = v3.fnum(r.get("ATT_mean"))
        awt = v3.fnum(r.get("AWT_mean"))
        completion = v3.fnum(r.get("completion_mean"))
        full = truthy(r.get("all_full_completion"))
        if full and finite(att):
            pts.append((
                int(float(r.get("train_step", 0))),
                float(att),
                float(awt) if finite(awt) else float("nan"),
                float(completion) if finite(completion) else float("nan"),
            ))
    pts.sort(key=lambda x: x[0])
    if not pts:
        return {
            "n_valid": 0,
            "best": float("nan"),
            "best_step": -1,
            "mean12": float("nan"),
            "late3": float("nan"),
            "final": float("nan"),
            "collapse": float("nan"),
            "final_awt": float("nan"),
        }
    vals = np.asarray([x[1] for x in pts], dtype=float)
    best_i = int(np.argmin(vals))
    return {
        "n_valid": len(pts),
        "best": float(vals[best_i]),
        "best_step": int(pts[best_i][0]),
        "mean12": float(np.mean(vals)),
        "late3": float(np.mean(vals[-3:])),
        "final": float(vals[-1]),
        "collapse": float(vals[-1] - vals[best_i]),
        "final_awt": float(pts[-1][2]),
    }


def build_global_aggregate(
    root: Path,
    design: Sequence[Dict[str, Any]],
) -> None:
    # One row per trained/evaluated cell.
    cell_rows: List[Dict[str, Any]] = []
    for rec in design:
        phase = rec["phase"]
        seed = int(rec["train_seed"])
        env_key = rec["env_key"]
        method = rec["method_id"]
        subroot = root / (
            f"single_seed{seed}" if phase == "SINGLE" else f"joint_seed{seed}"
        )
        run_dir = subroot / v3.cell_id(env_key, method)
        met = curve_metrics(run_dir)
        cell_rows.append({
            "phase": phase,
            "env_key": env_key,
            "train_seed": seed,
            "method_id": method,
            "family": method_description(method)["family"],
            "components": json.dumps(
                method_description(method)["components"], ensure_ascii=False
            ),
            "run_dir": str(run_dir),
            **met,
        })
    v3.write_csv(root / "aggregate_cells.csv", cell_rows)

    # Cross-train-seed aggregation by phase+method.
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in cell_rows:
        groups.setdefault((r["phase"], r["method_id"]), []).append(r)

    agg: List[Dict[str, Any]] = []
    for (phase, method), rs in sorted(groups.items()):
        valid = [r for r in rs if finite(r.get("final"))]
        finals = [float(r["final"]) for r in valid]
        bests = [float(r["best"]) for r in valid]
        lates = [float(r["late3"]) for r in valid]
        means = [float(r["mean12"]) for r in valid]
        collapses = [float(r["collapse"]) for r in valid]
        agg.append({
            "phase": phase,
            "method_id": method,
            "family": method_description(method)["family"],
            "n_train_seeds_planned": len(rs),
            "n_train_seeds_valid": len(valid),
            "train_seeds": ",".join(str(r["train_seed"]) for r in rs),
            "best_mean": float(np.mean(bests)) if bests else float("nan"),
            "best_std": float(np.std(bests)) if bests else float("nan"),
            "mean12_mean": float(np.mean(means)) if means else float("nan"),
            "late3_mean": float(np.mean(lates)) if lates else float("nan"),
            "late3_std": float(np.std(lates)) if lates else float("nan"),
            "final_mean": float(np.mean(finals)) if finals else float("nan"),
            "final_std": float(np.std(finals)) if finals else float("nan"),
            "collapse_mean": float(np.mean(collapses)) if collapses else float("nan"),
        })
    v3.write_csv(root / "aggregate_cross_seed.csv", agg)


# =============================================================================
# Training runners
# =============================================================================

def run_single_seed(
    *,
    root: Path,
    seed: int,
    methods: Sequence[str],
    timesteps: int,
    eval_seeds: Sequence[int],
    device: str,
    fail_fast: bool,
) -> None:
    subroot = root / f"single_seed{seed}"
    subroot.mkdir(parents=True, exist_ok=True)

    v3.write_json(subroot / "seed_manifest.json", {
        "phase": "SINGLE",
        "env_key": SINGLE_ENV,
        "train_seed": int(seed),
        "methods": list(methods),
        "timesteps": int(timesteps),
        "eval_seeds": list(eval_seeds),
        "profile": asdict(P40),
        "aircraft_control": "original responsive Longest-Queue",
    })

    for i, method in enumerate(methods, start=1):
        print(
            f"\n[SINGLE seed={seed} {i:02d}/{len(methods):02d}] "
            f"{SINGLE_ENV}__{method}",
            flush=True,
        )
        v3.safe_train_and_optional_eval(
            root=subroot,
            env_key=SINGLE_ENV,
            method_id=method,
            requested_steps=int(timesteps),
            train_seed=int(seed),
            eval_seeds=eval_seeds,
            profile=P40,
            device=device,
            max_time=MAX_TIME,
            defer_eval=False,
            fail_fast=bool(fail_fast),
        )
        v3.build_master(subroot)


def run_joint_seed(
    *,
    root: Path,
    seed: int,
    methods: Sequence[str],
    timesteps: int,
    eval_seeds: Sequence[int],
    device: str,
    fail_fast: bool,
    skip_eval: bool,
) -> None:
    subroot = root / f"joint_seed{seed}"
    subroot.mkdir(parents=True, exist_ok=True)

    plan = [
        {
            "env_key": JOINT_ENV,
            "joint_arch": v4.JOINT_ARCH_NAMES_V4[JOINT_ENV],
            "method_id": method,
        }
        for method in methods
    ]
    v3.write_json(subroot / "joint_plan.json", plan)
    v3.write_json(subroot / "seed_manifest.json", {
        "phase": "JOINT",
        "env_key": JOINT_ENV,
        "joint_arch": v4.JOINT_ARCH_NAMES_V4[JOINT_ENV],
        "train_seed": int(seed),
        "methods": list(methods),
        "timesteps": int(timesteps),
        "eval_seeds": list(eval_seeds),
        "profile": asdict(P40),
        "wrapper": "V5 clean minimal target-priority reposition",
        "deferred_eval": True,
    })

    # Train all Joint cells first so evaluation does not repeatedly interrupt GPU.
    for i, method in enumerate(methods, start=1):
        print(
            f"\n[JOINT TRAIN seed={seed} {i:02d}/{len(methods):02d}] "
            f"{JOINT_ENV}__{method}",
            flush=True,
        )
        v3.safe_train_and_optional_eval(
            root=subroot,
            env_key=JOINT_ENV,
            method_id=method,
            requested_steps=int(timesteps),
            train_seed=int(seed),
            eval_seeds=eval_seeds,
            profile=P40,
            device=device,
            max_time=MAX_TIME,
            defer_eval=True,
            fail_fast=bool(fail_fast),
        )

    if skip_eval:
        print(f"[JOINT seed={seed}] --skip-joint-eval set; evaluation deferred.")
        return

    # Reuse validated V3 formal evaluator for every 50k checkpoint x eval seed.
    v3.deferred_joint_evaluation(
        root=subroot,
        joint_plan=plan,
        requested_steps=int(timesteps),
        eval_seeds=eval_seeds,
        max_time=MAX_TIME,
        fail_fast=bool(fail_fast),
    )
    v3.build_master(subroot)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="UAM next-generation 100.2M Single+Joint literature matrix"
    )
    ap.add_argument(
        "--phase",
        choices=("all", "single", "joint"),
        default="all",
    )
    ap.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    ap.add_argument("--eval-seeds", default="123,124,125")
    ap.add_argument("--single-seeds", default="2,3")
    ap.add_argument("--joint-seeds", default="2,3,4")
    ap.add_argument(
        "--methods",
        default="",
        help=(
            "optional comma-separated filter; must belong to the selected phase. "
            "For --phase all, the filter is intersected with each phase."
        ),
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    ap.add_argument("--resume-root", default="")
    ap.add_argument("--output-root", default="")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--skip-joint-eval", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if int(args.timesteps) <= 0:
        raise ValueError("timesteps must be positive")
    if int(args.timesteps) % CHECKPOINT_INTERVAL != 0:
        raise ValueError(
            f"timesteps must be divisible by {CHECKPOINT_INTERVAL}"
        )

    install_nextgen_hooks()

    eval_seeds = parse_ints(args.eval_seeds)
    single_seeds = parse_ints(args.single_seeds)
    joint_seeds = parse_ints(args.joint_seeds)

    if args.phase == "single":
        single_methods = parse_methods(args.methods, SINGLE_METHODS)
        joint_methods: List[str] = []
    elif args.phase == "joint":
        single_methods = []
        joint_methods = parse_methods(args.methods, JOINT_METHODS)
    else:
        if str(args.methods).strip():
            requested = {
                canonical(x)
                for x in str(args.methods).split(",")
                if x.strip()
            }
            known = set(SINGLE_METHODS) | set(JOINT_METHODS)
            bad = sorted(requested - known)
            if bad:
                raise ValueError(f"unknown methods={bad}")
            single_methods = [m for m in SINGLE_METHODS if m in requested]
            joint_methods = [m for m in JOINT_METHODS if m in requested]
        else:
            single_methods = list(SINGLE_METHODS)
            joint_methods = list(JOINT_METHODS)

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_root:
        root = Path(args.resume_root).expanduser().resolve()
    elif args.output_root:
        root = Path(args.output_root).expanduser().resolve()
    else:
        root = (
            ROOT / "serial_runs" / f"uam_nextgen100m_{stamp}"
        ).resolve()
    root.mkdir(parents=True, exist_ok=True)

    design = make_design_rows(
        single_methods,
        joint_methods,
        single_seeds if single_methods else [],
        joint_seeds if joint_methods else [],
        int(args.timesteps),
    )

    n_single = len(single_methods) * len(single_seeds) if single_methods else 0
    n_joint = len(joint_methods) * len(joint_seeds) if joint_methods else 0
    n_total = n_single + n_joint
    single_budget = n_single * int(args.timesteps)
    joint_budget = n_joint * int(args.timesteps)
    total_budget = single_budget + joint_budget

    manifest = {
        "experiment": "UAM_NEXTGEN_100P2M_SINGLE40P8_JOINT59P4",
        "created": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "phase": args.phase,
        "requested_steps_per_cell": int(args.timesteps),
        "single": {
            "env_key": SINGLE_ENV,
            "methods": list(single_methods),
            "n_method_configs": len(single_methods),
            "train_seeds": list(single_seeds) if single_methods else [],
            "n_cells": n_single,
            "budget": single_budget,
            "aircraft": "original responsive Longest-Queue",
        },
        "joint": {
            "env_key": JOINT_ENV,
            "joint_arch": v4.JOINT_ARCH_NAMES_V4[JOINT_ENV],
            "methods": list(joint_methods),
            "n_method_configs": len(joint_methods),
            "train_seeds": list(joint_seeds) if joint_methods else [],
            "n_cells": n_joint,
            "budget": joint_budget,
            "wrapper": "V5 clean minimal target-priority reposition",
            "evaluation": "deferred per train seed until all Joint training cells finish",
        },
        "total_cells": n_total,
        "total_budget": total_budget,
        "eval_seeds": eval_seeds,
        "profile": asdict(P40),
        "max_time": MAX_TIME,
        "physics": asdict(v3.ENV_SPECS[SINGLE_ENV]),
        "scientific_scope": {
            "standalone": "roughly half S / AC / WM mechanism exploration",
            "interactions": "roughly half S+AC / S+WM / AC+WM plus only two triples",
            "joint_architecture_frozen": "J0 to isolate method effects",
            "new_cards_are_style_inspired": True,
        },
        "new_literature_cards": {
            "S_DYGFORMER_PATCH": "DyGFormer-style temporal patching",
            "S_GRAPHMIXER_TEMP": "GraphMixer-style temporal mixing",
            "A_QPLEX_DUPLEX": "QPLEX-style duplex advantage representation",
            "A_FACMAC_MIXER": "FACMAC-style factorized utility mixing",
            "W_TDMPC2_VALUE": "TD-MPC2-style value consistency using TDM + VE residual",
            "W_DREAMER_BALANCED": "Dreamer-style uncertainty/value-balanced residual",
        },
        "caveat": (
            "Literature labels are controlled STYLE/INSPIRED mechanism transplants, "
            "not verbatim reproductions."
        ),
    }
    v3.write_json(root / "experiment_manifest.json", manifest)
    v3.write_csv(root / "matrix_design.csv", design)

    print("=" * 128)
    print("UAM NEXT-GEN 100.2M MATRIX")
    print(f"root        : {root}")
    print(f"device      : {device}")
    print(f"profile     : {P40.name} | rollout={P40.rollout:,} | batch={P40.batch_size}")
    print(
        f"SINGLE      : {len(single_methods)} configs x "
        f"{len(single_seeds) if single_methods else 0} seeds = "
        f"{n_single} cells = {single_budget/1e6:.1f}M"
    )
    print(
        f"JOINT       : {len(joint_methods)} configs x "
        f"{len(joint_seeds) if joint_methods else 0} seeds = "
        f"{n_joint} cells = {joint_budget/1e6:.1f}M"
    )
    print(f"TOTAL       : {n_total} cells = {total_budget/1e6:.1f}M")
    print(f"eval seeds  : {eval_seeds}")
    print("=" * 128, flush=True)

    if args.plan_only:
        print("\nSINGLE METHODS")
        for i, m in enumerate(single_methods, 1):
            print(f"  S{i:02d} {m:<30s} {method_description(m)['family']}")
        print("\nJOINT METHODS")
        for i, m in enumerate(joint_methods, 1):
            print(f"  J{i:02d} {m:<30s} {method_description(m)['family']}")
        print(f"\nDesign CSV: {root / 'matrix_design.csv'}")
        return 0

    # ------------------------------- Single -------------------------------
    if single_methods:
        for seed in single_seeds:
            try:
                run_single_seed(
                    root=root,
                    seed=int(seed),
                    methods=single_methods,
                    timesteps=int(args.timesteps),
                    eval_seeds=eval_seeds,
                    device=device,
                    fail_fast=bool(args.fail_fast),
                )
            finally:
                build_global_aggregate(root, design)

    # -------------------------------- Joint -------------------------------
    if joint_methods:
        for seed in joint_seeds:
            try:
                run_joint_seed(
                    root=root,
                    seed=int(seed),
                    methods=joint_methods,
                    timesteps=int(args.timesteps),
                    eval_seeds=eval_seeds,
                    device=device,
                    fail_fast=bool(args.fail_fast),
                    skip_eval=bool(args.skip_joint_eval),
                )
            finally:
                build_global_aggregate(root, design)

    build_global_aggregate(root, design)
    v3.write_json(root / "RUN_COMPLETE.json", {
        "status": "COMPLETE",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_cells": n_total,
        "total_requested_timesteps": total_budget,
        "single_budget": single_budget,
        "joint_budget": joint_budget,
        "aggregate_cells": str(root / "aggregate_cells.csv"),
        "aggregate_cross_seed": str(root / "aggregate_cross_seed.csv"),
    })

    print("\n" + "#" * 128)
    print("DONE")
    print(f"root                 : {root}")
    print(f"aggregate cells      : {root / 'aggregate_cells.csv'}")
    print(f"aggregate cross-seed : {root / 'aggregate_cross_seed.csv'}")
    print("#" * 128, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())