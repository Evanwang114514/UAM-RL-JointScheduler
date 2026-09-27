# -*- coding: utf-8 -*-
"""
20 x 600k SINGLE-SIDE literature-upgrade matrix on fixed S3 physics
==================================================================

Goal
----
After the first literature screen, do NOT expand another random zoo.
This file tests whether the best surviving S / AC / WM ideas can be lifted by
specific later literature fixes, and whether two-way combinations are genuinely
complementary.

ALL 20 cells are passenger-only SINGLE models on the SAME S3 physics:
    T2, N=40, single-passenger service,
    turnaround=1 min,
    finite charging scale=1.25, charger capacity=5,
    pad/TLOF separation=0.25 min,
    automatic responsive Longest-Queue aircraft reposition.

PPO / reward / physics / observation superset are frozen.
600k per cell, train seed=1, eval seeds=123/124/125.
P40 = 40 env x 512 = 20,480 rollout, batch=4096.
Total requested RL budget = 20 * 600k = 12.0M.

Scientific layout
-----------------
A. Existing anchors (8)
  CURRENT
  R_EVENTQ
  S_EVENTGRAPH_GAT
  S_HKSL_MULTI
  A_COMA_AC
  A_ICM_AC
  W_EVENTQ_COCO_CF
  W_TDM_CVAML

B. New S-v2 representation fixes (3)
  S_GATV2_EVENT
      GATv2-inspired dynamic candidate attention over committed-event/TDM nodes.
      Fixes the static-attention limitation of vanilla GAT-style scoring.
  S_TGAT_EVENT
      TGAT/TGN-inspired event-level temporal attention.
      Unlike the old EventGraph card, events are NOT mean-pooled before relation
      modeling; timestamp/type/candidate/effect-time-relative features interact.
  S_ADAPTIVE_EFFECT_HORIZON
      effect-time-centered adaptive horizon weighting over the existing projected
      multi-horizon states; replaces a fixed "take last GRU state" summary.

C. New AC-v2 fixes (2)
  A_ANCHOR_QUOTIENT
      Shared absolute anchor + CQM-style action quotient residual.
      Explicitly fixes the pure-delta loss of absolute congestion context.
  A_DENOISED_VALUE
      ICM controllability + COMA return/value relevance together.
      Denoised-MDP-inspired control: controllable alone is not enough.

D. New WM-v2 fixes (2)
  W_VALUE_CONTROLLABLE
      Value Equivalence + CoCo controllability (two trained auxiliaries).
      Tests value relevance + action distinguishability together.
  W_UNC_VALUE
      PETS/STEVE uncertainty + C-VAML value awareness.
      Tests uncertainty-aware horizon information + value-aware residual.

E. Two-way controlled combinations (5)
  SA_GATV2_COMA
  SA_TGAT_DENOISED
  SW_TGAT_VALUECTRL
  SW_GATV2_UNCVALUE
  AW_ANCHORQ_VALUECTRL

Important wording
-----------------
The new cards are LITERATURE-INSPIRED, not verbatim reproductions of GATv2,
TGAT/TGN, Denoised MDP, MACURA, or calibrated VAML.  The purpose is a controlled
mechanism screen inside the existing UAM PPO stack.  If one wins, reproduce the
winning paper mechanism more faithfully in the next phase.

Required existing files beside this script
------------------------------------------
train_uam_60m_literature_matrix_v3_1.py
train_uam_60m_jointfirst_v4_deferred_joint_eval.py
(and their validated project dependencies)

Default
-------
python train_uam_s3_20x600k_litupgrade.py

Smoke
-----
python train_uam_s3_20x600k_litupgrade.py --methods S_GATV2_EVENT --timesteps 50000

Resume
------
python train_uam_s3_20x600k_litupgrade.py \
  --resume-root serial_runs/uam_s3_litupgrade20_seed1_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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

ROOT = Path(__file__).resolve().parent
ENV_KEY = "S3"
DEFAULT_TIMESTEPS = 600_000
TRAIN_SEED = 1
DEFAULT_EVAL_SEEDS = (123, 124, 125)
MAX_TIME = 2_500

# Frozen production-safe geometry from the dedicated scaling study.
P40 = v3.SpeedProfile(
    "P40_40x512_b4096",
    n_envs=40,
    n_steps=512,
    batch_size=4096,
)
assert P40.rollout == 20_480

METHODS = (
    # Existing anchors
    "CURRENT",
    "R_EVENTQ",
    "S_EVENTGRAPH_GAT",
    "S_HKSL_MULTI",
    "A_COMA_AC",
    "A_ICM_AC",
    "W_EVENTQ_COCO_CF",
    "W_TDM_CVAML",

    # S-v2
    "S_GATV2_EVENT",
    "S_TGAT_EVENT",
    "S_ADAPTIVE_EFFECT_HORIZON",

    # AC-v2
    "A_ANCHOR_QUOTIENT",
    "A_DENOISED_VALUE",

    # WM-v2
    "W_VALUE_CONTROLLABLE",
    "W_UNC_VALUE",

    # Pairwise combinations
    "SA_GATV2_COMA",
    "SA_TGAT_DENOISED",
    "SW_TGAT_VALUECTRL",
    "SW_GATV2_UNCVALUE",
    "AW_ANCHORQ_VALUECTRL",
)
assert len(METHODS) == 20

# Register combinations using ONLY 1-3 components because the validated V4
# fixed-capacity fusion supports at most three 64-d mechanism latents.
COMBOS: Dict[str, Tuple[str, ...]] = {
    # AC-v2: controllability + return/value relevance
    "A_DENOISED_VALUE": ("A_ICM_AC", "A_COMA_AC"),

    # WM-v2
    "W_VALUE_CONTROLLABLE": ("W_EVENTQ_VE_EMA", "W_EVENTQ_COCO_CF"),
    "W_UNC_VALUE": ("W_GAMMA_PETS_STEVE", "W_TDM_CVAML"),

    # S + AC
    "SA_GATV2_COMA": ("S_GATV2_EVENT", "A_COMA_AC"),
    "SA_TGAT_DENOISED": ("S_TGAT_EVENT", "A_ICM_AC", "A_COMA_AC"),

    # S + WM
    "SW_TGAT_VALUECTRL": ("S_TGAT_EVENT", "W_EVENTQ_VE_EMA", "W_EVENTQ_COCO_CF"),
    "SW_GATV2_UNCVALUE": ("S_GATV2_EVENT", "W_GAMMA_PETS_STEVE", "W_TDM_CVAML"),

    # AC + WM
    "AW_ANCHORQ_VALUECTRL": ("A_ANCHOR_QUOTIENT", "W_EVENTQ_VE_EMA", "W_EVENTQ_COCO_CF"),
}

DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "CURRENT": {
        "family": "ANCHOR",
        "question": "source/current-state UAGMC-style reference",
        "literature": "UAGMC/current-state control",
    },
    "R_EVENTQ": {
        "family": "S",
        "question": "is effect-time querying of committed events already sufficient?",
        "literature": "project EVENTQ anchor",
    },
    "S_EVENTGRAPH_GAT": {
        "family": "S",
        "question": "does candidate relation modeling stabilize EVENTQ information?",
        "literature": "GAT-inspired event/resource graph",
    },
    "S_HKSL_MULTI": {
        "family": "S",
        "question": "does multi-horizon context help beyond one effect time?",
        "literature": "HKSL/multi-horizon inspired",
    },
    "A_COMA_AC": {
        "family": "AC",
        "question": "does relative action-value/advantage representation help?",
        "literature": "COMA-inspired",
    },
    "A_ICM_AC": {
        "family": "AC",
        "question": "does controllability-focused latent help?",
        "literature": "ICM-inspired inverse dynamics",
    },
    "W_EVENTQ_COCO_CF": {
        "family": "WM",
        "question": "does action-distinguishable residual future stabilize EVENTQ?",
        "literature": "CoCo/counterfactual controllability inspired",
    },
    "W_TDM_CVAML": {
        "family": "WM",
        "question": "does value-aware residual modeling improve TDM?",
        "literature": "VAML/calibrated-VAML inspired",
    },
    "S_GATV2_EVENT": {
        "family": "S-v2",
        "question": "does dynamic GATv2-style candidate attention improve old EventGraph?",
        "literature": "GATv2-inspired dynamic attention",
    },
    "S_TGAT_EVENT": {
        "family": "S-v2",
        "question": "does preserving event-level time structure beat pre-pooled EventGraph?",
        "literature": "TGAT/TGN-inspired temporal event attention",
    },
    "S_ADAPTIVE_EFFECT_HORIZON": {
        "family": "S-v2",
        "question": "can passenger-specific effect-centered horizon weighting beat fixed multi-horizon summary?",
        "literature": "adaptive-horizon / uncertainty-trust inspired",
    },
    "A_ANCHOR_QUOTIENT": {
        "family": "AC-v2",
        "question": "does restoring absolute shared context fix pure quotient/delta aliasing?",
        "literature": "CQM-inspired quotient + shared anchor",
    },
    "A_DENOISED_VALUE": {
        "family": "AC-v2",
        "question": "is controllable AND value-relevant better than controllability alone?",
        "literature": "Denoised-MDP-inspired ICM + COMA fusion",
    },
    "W_VALUE_CONTROLLABLE": {
        "family": "WM-v2",
        "question": "does value relevance + action distinguishability form a better residual model?",
        "literature": "Value Equivalence + CoCo-inspired",
    },
    "W_UNC_VALUE": {
        "family": "WM-v2",
        "question": "does uncertainty-aware horizon information complement value-aware modeling?",
        "literature": "PETS/STEVE + C-VAML-inspired",
    },
    "SA_GATV2_COMA": {
        "family": "S+AC",
        "question": "does action advantage add value on top of dynamic event graph context?",
        "literature": "GATv2 + COMA/MAAC-style decomposition",
    },
    "SA_TGAT_DENOISED": {
        "family": "S+AC",
        "question": "does event-level temporal context + controllable/value-relevant residual outperform either alone?",
        "literature": "TGAT/TGN + Denoised-MDP/COMA inspired",
    },
    "SW_TGAT_VALUECTRL": {
        "family": "S+WM",
        "question": "does structured temporal event context + value/controllability residual WM help?",
        "literature": "temporal graph + structured/value-aware world-model inspired",
    },
    "SW_GATV2_UNCVALUE": {
        "family": "S+WM",
        "question": "does dynamic event graph + uncertainty/value-aware WM improve robustness?",
        "literature": "GATv2 + PETS/STEVE + VAML inspired",
    },
    "AW_ANCHORQ_VALUECTRL": {
        "family": "AC+WM",
        "question": "does action quotient with absolute anchor + value/controllability WM form a better action-effect model?",
        "literature": "CQM + Value Equivalence + controllability inspired",
    },
}


def canonical(x: str) -> str:
    return str(x).upper().strip()


def parse_names(text: str) -> List[str]:
    vals = [canonical(x) for x in str(text).split(",") if x.strip()]
    bad = [x for x in vals if x not in METHODS]
    if bad:
        raise ValueError(f"unsupported methods={bad}")
    return vals


def parse_ints(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty seed list")
    return vals


# -----------------------------------------------------------------------------
# New literature-inspired representation modules
# -----------------------------------------------------------------------------

class LiteratureV2Extractor(v4.DiscoveryExtractor):
    """V4 extractor + four controlled representation fixes.

    All existing V4 components are inherited exactly.
    New components are representation-only; old AC/WM auxiliaries are reused
    unchanged when they appear inside the registered combinations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        K = v3.N_CANDIDATES
        P = v3.PROJECT_PER_CAND
        H = len(v3.GAMMA_HORIZONS)

        # GATv2-style dynamic candidate attention.
        self.v2_graph_node = nn.Sequential(
            nn.Linear(32 + v3.TDM_FEATURES_PER_CAND, 64),
            nn.ReLU(),
        )
        self.v2_gat_heads = 4
        self.v2_gat_head_dim = 16
        self.v2_gat_score = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, self.v2_gat_heads),
        )
        self.v2_gat_value = nn.Linear(64, 64)
        self.v2_gat_out = nn.Sequential(
            nn.Linear(64 * K, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
        )

        # TGAT/TGN-style global event-level temporal attention.
        # token = [type, cand0, cand1, rel_tau, sin(rel_tau), cos(rel_tau), eta_scaled]
        self.tgat_event_elem = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.tgat_attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.tgat_out = nn.Sequential(
            nn.Linear(64 * K + v3.TDM_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Effect-centered adaptive horizon attention.
        self.ah_enc = nn.Sequential(
            nn.Linear(P, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
        )
        self.ah_score = nn.Sequential(
            nn.Linear(34, 32),  # encoded state + signed/absolute h-tau
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.ah_out = nn.Sequential(
            nn.Linear(32 * K, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
        )

        # Shared absolute anchor + quotient residual.
        self.aq_anchor = nn.Sequential(
            nn.Linear(P * K, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.aq_delta = nn.Sequential(
            nn.Linear(2 * P, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

    def _event_pool32(self, obs: torch.Tensor) -> torch.Tensor:
        reps = []
        for c in range(v3.N_CANDIDATES):
            elems, mask = self._candidate_event_tokens(obs, c)
            emb = self.set_event_elem(elems)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / denom
            reps.append(pooled)
        return torch.stack(reps, dim=1)  # B,K,32

    def _gatv2_event_latent(self, obs: torch.Tensor) -> torch.Tensor:
        ev = self._event_pool32(obs)
        tdm = self._slice(obs, "tdm").reshape(
            obs.shape[0], v3.N_CANDIDATES, v3.TDM_FEATURES_PER_CAND
        )
        nodes = self.v2_graph_node(torch.cat([ev, tdm], dim=-1))  # B,K,64
        B, K, _ = nodes.shape

        ni = nodes.unsqueeze(2).expand(B, K, K, 64)
        nj = nodes.unsqueeze(1).expand(B, K, K, 64)
        score = self.v2_gat_score(torch.cat([ni, nj], dim=-1))  # B,K,K,H
        attn = torch.softmax(score, dim=2)

        val = self.v2_gat_value(nodes).reshape(
            B, K, self.v2_gat_heads, self.v2_gat_head_dim
        )
        valj = val.unsqueeze(1).expand(
            B, K, K, self.v2_gat_heads, self.v2_gat_head_dim
        )
        out = (attn.unsqueeze(-1) * valj).sum(dim=2).reshape(B, K, 64)
        out = out + nodes
        return self.v2_gat_out(out.flatten(1))

    def _tgat_event_latent(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        tau = self._slice(obs, "tau").reshape(B, v3.N_CANDIDATES)
        token_blocks = []
        mask_blocks = []
        block_sizes = []

        for c in range(v3.N_CANDIDATES):
            elems, mask = self._candidate_event_tokens(obs, c)
            eta = elems[..., 0]
            typ = elems[..., 1]
            tc = tau[:, c:c + 1]
            rel = (eta - tc) / 60.0
            cand0 = torch.full_like(eta, 1.0 if c == 0 else 0.0)
            cand1 = torch.full_like(eta, 1.0 if c == 1 else 0.0)
            feat = torch.stack([
                typ,
                cand0,
                cand1,
                rel,
                torch.sin(math.pi * rel),
                torch.cos(math.pi * rel),
                eta / 60.0,
            ], dim=-1)
            token_blocks.append(feat)
            mask_blocks.append(mask)
            block_sizes.append(feat.shape[1])

        feats = torch.cat(token_blocks, dim=1)
        valid = torch.cat(mask_blocks, dim=1)
        key_padding = valid <= 0.0

        all_masked = key_padding.all(dim=1)
        if bool(all_masked.any()):
            key_padding = key_padding.clone()
            key_padding[all_masked, 0] = False

        emb = self.tgat_event_elem(feats)
        attn, _ = self.tgat_attn(
            emb, emb, emb,
            key_padding_mask=key_padding,
            need_weights=False,
        )

        reps = []
        start = 0
        for c, width in enumerate(block_sizes):
            sl = slice(start, start + width)
            m = valid[:, sl]
            y = attn[:, sl, :]
            denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (y * m.unsqueeze(-1)).sum(dim=1) / denom
            reps.append(pooled)
            start += width

        tdm = self._slice(obs, "tdm")
        return self.tgat_out(torch.cat(reps + [tdm], dim=1))

    def _adaptive_effect_horizon_latent(self, obs: torch.Tensor) -> torch.Tensor:
        g = self._slice(obs, "gamma").reshape(
            obs.shape[0],
            v3.N_CANDIDATES,
            len(v3.GAMMA_HORIZONS),
            v3.PROJECT_PER_CAND,
        )
        tau = self._slice(obs, "tau").reshape(obs.shape[0], v3.N_CANDIDATES)
        horizons = torch.as_tensor(
            list(v3.GAMMA_HORIZONS),
            device=obs.device,
            dtype=obs.dtype,
        ).view(1, -1)

        reps = []
        for c in range(v3.N_CANDIDATES):
            z = self.ah_enc(g[:, c])  # B,H,32
            diff = (horizons - tau[:, c:c + 1]) / 60.0
            feat = torch.cat([
                z,
                diff.unsqueeze(-1),
                diff.abs().unsqueeze(-1),
            ], dim=-1)
            score = self.ah_score(feat).squeeze(-1)
            w = torch.softmax(score, dim=1)
            reps.append((z * w.unsqueeze(-1)).sum(dim=1))
        return self.ah_out(torch.cat(reps, dim=1))

    def _anchor_quotient_latent(self, obs: torch.Tensor) -> torch.Tensor:
        shared = self._slice(obs, "shared").reshape(
            obs.shape[0], v3.N_CANDIDATES, v3.PROJECT_PER_CAND
        )
        srac = self._slice(obs, "srac").reshape(
            obs.shape[0], v3.N_CANDIDATES, v3.PROJECT_PER_CAND
        )
        own = shared + srac
        delta = own[:, 0] - own[:, 1]
        anchor = self.aq_anchor(shared.flatten(1))
        quotient = self.aq_delta(torch.cat([delta, -delta], dim=1))
        return torch.cat([anchor, quotient], dim=1)  # 64

    def component_latent(self, obs: torch.Tensor, component: str) -> torch.Tensor:
        c = canonical(component)
        if c == "S_GATV2_EVENT":
            return self._gatv2_event_latent(obs)
        if c == "S_TGAT_EVENT":
            return self._tgat_event_latent(obs)
        if c == "S_ADAPTIVE_EFFECT_HORIZON":
            return self._adaptive_effect_horizon_latent(obs)
        if c == "A_ANCHOR_QUOTIENT":
            return self._anchor_quotient_latent(obs)
        return super().component_latent(obs, c)


# -----------------------------------------------------------------------------
# Registry / descriptions
# -----------------------------------------------------------------------------

def install_registry() -> None:
    for k, v in COMBOS.items():
        v4.COMBO_REGISTRY[canonical(k)] = tuple(canonical(x) for x in v)


def method_description(method_id: str) -> Dict[str, Any]:
    m = canonical(method_id)
    d = DESCRIPTIONS.get(m, {})
    comps = list(v4.components_for(m))
    return {
        "method_id": m,
        "family": d.get("family", v4.method_family(m)),
        "components": comps,
        "literature_or_role": d.get("literature", "controlled literature-upgrade card"),
        "scientific_question": d.get("question", ""),
        "representation": m,
        "world_model": (
            "auxiliary residual model present"
            if any(
                c in {
                    "W_GAMMA_PETS_STEVE",
                    "W_EVENTQ_VE_EMA",
                    "W_EVENTQ_COCO_CF",
                    "W_TDM_CVAML",
                }
                for c in comps
            )
            else None
        ),
    }


def build_model(
    *,
    env,
    env_key: str,
    method_id: str,
    profile: v3.SpeedProfile,
    seed: int,
    run_dir: Path,
    device: str,
):
    policy_kwargs = dict(
        features_extractor_class=LiteratureV2Extractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            layout=v3.GLOBAL_LAYOUT,
            method_id=method_id,
            env_key=env_key,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    m = canonical(method_id)
    if v4.custom_method_needs_aux(m):
        algo_cls = v4.DiscoveryAuxPPO
    elif m in v4.NATIVE_WM_METHODS:
        algo_cls = v3.LiteratureWorldModelPPO
    else:
        algo_cls = PPO

    return algo_cls(
        policy="MlpPolicy",
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


def install_hooks() -> None:
    install_registry()
    # Reuse V4 nominal checkpoints and auxiliary PPO, but replace the extractor.
    v3.build_model = build_model
    v3.build_callbacks = v4.build_callbacks
    v3.method_description = method_description


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    ap.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    ap.add_argument("--eval-seeds", default="123,124,125")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--resume-root", default="")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    install_hooks()

    methods = parse_names(args.methods)
    eval_seeds = parse_ints(args.eval_seeds)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    if args.resume_root:
        root = Path(args.resume_root).resolve()
    else:
        root = ROOT / "serial_runs" / (
            "uam_s3_litupgrade20_seed"
            f"{args.train_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    root.mkdir(parents=True, exist_ok=True)

    design = []
    for i, m in enumerate(methods, start=1):
        rec = method_description(m)
        rec.update({
            "index": i,
            "env_key": ENV_KEY,
            "timesteps": int(args.timesteps),
        })
        design.append(rec)

    v3.write_json(root / "experiment_manifest.json", {
        "experiment": "UAM_S3_20x600K_LITERATURE_UPGRADE",
        "created": datetime.now().isoformat(timespec="seconds"),
        "env_key": ENV_KEY,
        "single_side_only": True,
        "aircraft_reposition": "automatic responsive Longest-Queue",
        "methods": methods,
        "n_cells": len(methods),
        "requested_steps_per_cell": int(args.timesteps),
        "formal_budget": len(methods) * int(args.timesteps),
        "train_seed": int(args.train_seed),
        "eval_seeds": eval_seeds,
        "profile": asdict(P40),
        "device": device,
        "physics": asdict(v3.ENV_SPECS[ENV_KEY]),
        "scientific_scope": {
            "S_v2": [
                "GATv2-style dynamic attention",
                "TGAT/TGN-style event-level temporal attention",
                "effect-centered adaptive horizon",
            ],
            "AC_v2": [
                "absolute shared anchor + quotient residual",
                "controllable + value-relevant (Denoised-MDP-inspired)",
            ],
            "WM_v2": [
                "value + controllability residual",
                "uncertainty + value-aware residual",
            ],
            "pairwise": ["S+AC", "S+WM", "AC+WM"],
            "no_triple_combo": True,
        },
        "caveat": (
            "All new cards are literature-inspired controlled mechanisms, "
            "not verbatim paper reproductions."
        ),
    })
    v3.write_csv(root / "matrix_design.csv", design)

    print("=" * 128)
    print("S3 20x600k LITERATURE-UPGRADE MATRIX")
    print(f"root={root}")
    print(f"cells={len(methods)}")
    print(f"budget={len(methods)*int(args.timesteps):,} requested PPO timesteps")
    print(f"profile={P40.name} rollout={P40.rollout} batch={P40.batch_size}")
    print("=" * 128, flush=True)

    for i, method in enumerate(methods, start=1):
        print(
            f"\n[{i:02d}/{len(methods):02d}] {ENV_KEY}__{method} | "
            f"{DESCRIPTIONS.get(method, {}).get('family', '')}",
            flush=True,
        )
        v3.safe_train_and_optional_eval(
            root=root,
            env_key=ENV_KEY,
            method_id=method,
            requested_steps=int(args.timesteps),
            train_seed=int(args.train_seed),
            eval_seeds=eval_seeds,
            profile=P40,
            device=device,
            max_time=MAX_TIME,
            defer_eval=False,
            fail_fast=bool(args.fail_fast),
        )
        v3.build_master(root)

    print("\nDONE")
    print(f"Master: {root / 'matrix_master.csv'}")
    print(f"Design: {root / 'matrix_design.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
