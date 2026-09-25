# -*- coding: utf-8 -*-
r"""
UAM 60M JOINT-FIRST DISCOVERY MATRIX V4
========================================

Purpose
-------
Successor experiment layer for ``train_uam_60m_literature_matrix_v3_1.py``.
It reuses the validated physics, completion semantics, reward, no-HOLD joint
wrapper, observation plumbing and evaluation code from V3.1, but changes the
scientific schedule and adds literature-inspired representation/coordination
modules.

Budget (600k/cell)
------------------
J0  Joint architecture screen:
      8 joint architectures x 3 anchors                 = 24 cells = 14.4M
J1  Joint x representation interaction:
      Top-3 joint architectures x 12 methods            = 36 cells = 21.6M
J2  Best joint architecture x interaction/controls:
      1 joint architecture x 7 methods                  =  7 cells =  4.2M
S   Single-side mechanism isolation (RUNS LAST):
      11 selected methods x S1/S2/S3                    = 33 cells = 19.8M

TOTAL                                                    100 cells = 60.0M
Single : Joint = 33 : 67 ~= 1 : 2.03

Fast production profile (selected by benchmark)
------------------------------------------------
    n_envs    = 32
    n_steps   = 640
    batch     = 4096
    rollout   = 20,480 transitions/update
    device    = CUDA

IMPORTANT
---------
* The old V3.1 file is not modified.
* Old Joint cells trained with batch=2048 are PILOT data only.  This V4 starts
  a fresh matrix so all formal cells use one optimization protocol.
* Literature names ending in STYLE/INSPIRED are architectural/mechanistic
  transplants, not claims of verbatim reproduction of the original algorithm.
* Training seed is intentionally single-seed discovery.  Additional training
  seeds should only be run after finalists are selected.
* Checkpoint labels remain nominal 50k/100k/.../600k.  Because n_envs=32 does
  not divide 50,000, the exact save occurs on the first vector step at/after
  each nominal mark (<=31 transitions later) but is written using the nominal
  checkpoint filename expected by the validated V3.1 evaluator.

Default run
-----------
conda activate uam5070
cd /d "E:\\Study Files\\github\\UAM-predict\\UAGMC-main"
python train_uam_60m_jointfirst_v4.py

Resume
------
python train_uam_60m_jointfirst_v4.py --resume-root "serial_runs\\<run>"

Run a later stage only (requires prerequisite selection JSON in resume root)
----------------------------------------------------------------------------
python train_uam_60m_jointfirst_v4.py --resume-root "serial_runs\\<run>" --stage J1
python train_uam_60m_jointfirst_v4.py --resume-root "serial_runs\\<run>" --stage J2
python train_uam_60m_jointfirst_v4.py --resume-root "serial_runs\\<run>" --stage S
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy

import train_uam_60m_literature_matrix_v3_1 as v3

# Preserve V3.1 callables before install_v4_hooks() monkey-patches module globals.
_V3_METHOD_DESCRIPTION = v3.method_description


# =============================================================================
# Formal constants / fast profile
# =============================================================================

ROOT = Path(__file__).resolve().parent
DEFAULT_TIMESTEPS = 600_000
CHECKPOINT_INTERVAL = 50_000
TRAIN_SEED = 1
DEFAULT_EVAL_SEEDS = (123, 124, 125)
DEFAULT_MAX_TIME = 2_500

FAST_PROFILE = v3.SpeedProfile(
    "PFAST_32x640_b4096",
    n_envs=32,
    n_steps=640,
    batch_size=4096,
)
assert FAST_PROFILE.rollout == 20_480

# Keep the scientific physics fixed to V3.1.
JOINT_KEYS = tuple(f"J{i}" for i in range(8))

# New architecture meanings.  J0-J3 retain V3.1 semantics; J4-J7 replace the
# low-information pseudo-QMIX/QTRAN/effect cards with higher-value questions.
JOINT_ARCH_NAMES_V4 = {
    "J0": "INDEPENDENT_PARALLEL",
    "J1": "ACTION_BRANCHING",
    "J2": "DIRECT_PAIRWISE",
    "J3": "SEQUENTIAL_DUAL_AGENT",
    "J4": "MAPPO_STYLE_CTDE_BRANCH",
    "J5": "COMA_STYLE_COUNTERFACTUAL_PAIR",
    "J6": "AIR_VALUEPLAN_STYLE",
    "J7": "PAIR_VALUEPLAN_STYLE",
}

# Phase J0: architecture screen over three clean anchors.
J0_ANCHORS = (
    "CURRENT",
    "R_TDM",
    "R_EVENTQ",
)

# Phase J1: new lower-level mechanisms.  Four per family.
SHARED_METHODS = (
    "S_SETTRANS_EVENT",
    "S_EVENTGRAPH_GAT",
    "S_HKSL_MULTI",
    "S_TDM_EVENT_FUSION",
)
AC_METHODS = (
    "A_DIRECT_CQM",
    "A_COMA_AC",
    "A_ICM_AC",
    "A_DEEPMDP_AC",
)
WM_METHODS = (
    "W_GAMMA_PETS_STEVE",
    "W_EVENTQ_VE_EMA",
    "W_EVENTQ_COCO_CF",
    "W_TDM_CVAML",
)
J1_METHODS = SHARED_METHODS + AC_METHODS + WM_METHODS

# Native V3.1 controls used without changing their scientific definition.
NATIVE_METHODS = {
    "CURRENT",
    "SHARED",
    "R_SRAC",
    "R_GAMMA",
    "R_CQM",
    "R_EVENTQ",
    "R_TDM",
    "R_FIRST",
    "R_GAMMA__W_PETS",
    "R_EVENTQ__W_COCO",
    "R_TDM__W_TAU",
}
NATIVE_WM_METHODS = {
    "R_GAMMA__W_PETS",
    "R_EVENTQ__W_COCO",
    "R_TDM__W_TAU",
}

# Dynamic combo registry.  Persisted plans reconstruct this on resume.
COMBO_REGISTRY: Dict[str, Tuple[str, ...]] = {}

# Auxiliary training.
AUX_LR = 1e-3
AUX_BATCH = 2048
AUX_MAX_SAMPLES = 12_000
AUX_EPOCHS = 1
COCO_CF_LAMBDA = 0.10
CVAML_LAMBDA = 0.5
EMA_TAU = 0.995
SHORT_HORIZON = 3


# =============================================================================
# Naming / descriptions / components
# =============================================================================

def canonical(x: str) -> str:
    return str(x).upper().strip()


def method_family(method_id: str) -> str:
    m = canonical(method_id)
    if m.startswith("S_"):
        return "SHARED"
    if m.startswith("A_"):
        return "AC"
    if m.startswith("W_"):
        return "WM"
    if m.startswith("C_"):
        return "COMBO"
    if m in NATIVE_METHODS:
        return "ANCHOR"
    return "OTHER"


def components_for(method_id: str) -> Tuple[str, ...]:
    m = canonical(method_id)
    if m in COMBO_REGISTRY:
        return COMBO_REGISTRY[m]
    return (m,)


def has_component(method_id: str, component: str) -> bool:
    c = canonical(component)
    return c in {canonical(x) for x in components_for(method_id)}


def method_description(method_id: str) -> Dict[str, Any]:
    m = canonical(method_id)
    comps = list(components_for(m))
    source_map = {
        "S_SETTRANS_EVENT": "Set Transformer (ICML 2019) inspired event self-attention + effect-time query",
        "S_EVENTGRAPH_GAT": "GAT (ICLR 2018) inspired two-candidate event/resource graph attention",
        "S_HKSL_MULTI": "multi-horizon / hierarchical forward representation inspired by HKSL / TMLR multi-horizon work",
        "S_TDM_EVENT_FUSION": "project-specific fusion of TDM-style decision summary + committed event structure",
        "A_DIRECT_CQM": "Counterfactual Quotient Model (2026 preprint) inspired direct action-difference representation",
        "A_COMA_AC": "COMA (AAAI 2018) inspired counterfactual relative action-value representation",
        "A_ICM_AC": "ICM (ICML 2017) inspired inverse-dynamics controllable latent",
        "A_DEEPMDP_AC": "DeepMDP (ICML 2019) inspired reward + latent transition representation",
        "W_GAMMA_PETS_STEVE": "PETS + STEVE inspired uncertainty-weighted multi-horizon future",
        "W_EVENTQ_VE_EMA": "Value Equivalence (NeurIPS 2020) + EMA target inspired value-equivalent event future",
        "W_EVENTQ_COCO_CF": "CoCo (2026 preprint) inspired action-controllability consistency over EVENTQ",
        "W_TDM_CVAML": "VAML / calibrated value-aware modeling inspired TDM residual world model",
    }
    if m in COMBO_REGISTRY:
        src = "selected interaction combo: " + " + ".join(comps)
    elif m in source_map:
        src = source_map[m]
    else:
        base_desc = _V3_METHOD_DESCRIPTION(m)
        src = f"V3.1 native control ({base_desc.get('representation')}, {base_desc.get('world_model')})"
    return {
        "method_id": m,
        "family": method_family(m),
        "components": comps,
        "literature_or_role": src,
        "representation": m if m not in NATIVE_METHODS else _V3_METHOD_DESCRIPTION(m).get("representation"),
        "world_model": _V3_METHOD_DESCRIPTION(m).get("world_model") if m in NATIVE_METHODS else None,
    }


# =============================================================================
# New unified extractor
# =============================================================================

class DiscoveryExtractor(v3.LiteratureExtractor):
    """V3.1 extractor plus fixed-capacity modules for V4 discovery methods.

    Native V3.1 methods call the exact V3.1 forward path.  Custom methods use a
    common core plus up to three 64-d mechanism latents, padded to fixed width,
    so custom-method policy capacity is matched across the discovery cards.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method_id_v4 = canonical(self.method_id)
        self.components_v4 = components_for(self.method_id_v4)

        core_in = 128 + 64 + 32
        self.v4_core_proj = nn.Sequential(
            nn.Linear(core_in, 192), nn.ReLU(), nn.Linear(192, 128), nn.ReLU()
        )
        self.v4_fusion = nn.Sequential(
            nn.Linear(128 + 3 * 64, 192), nn.ReLU(), nn.Linear(192, 128), nn.ReLU()
        )
        # J6/J7 planning cards receive the virtual post-action pair semantics
        # already computed by the validated V3.1 observation wrapper.
        self.pair_context128 = nn.Sequential(
            nn.Linear(v3.PAIR_VIRTUAL_DIM, 128), nn.ReLU(), nn.Linear(128, 128)
        )

        # Shared / set / graph modules.
        self.set_event_elem = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.set_event_attn = nn.MultiheadAttention(32, num_heads=4, batch_first=True)
        self.set_event_out = nn.Sequential(nn.Linear(32 * v3.N_CANDIDATES, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())

        self.graph_node = nn.Sequential(nn.Linear(32 + v3.TDM_FEATURES_PER_CAND, 64), nn.ReLU())
        self.graph_attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.graph_out = nn.Sequential(nn.Linear(64 * v3.N_CANDIDATES, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())

        self.horizon_enc = nn.Sequential(nn.Linear(v3.PROJECT_PER_CAND, 48), nn.ReLU(), nn.Linear(48, 32), nn.ReLU())
        self.horizon_gru = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.horizon_out = nn.Sequential(nn.Linear(64 * v3.N_CANDIDATES, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())

        self.tdm_event_fuse = nn.Sequential(nn.Linear(32 + 64, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())

        # AC modules.
        self.direct_cqm = nn.Sequential(nn.Linear(2 * v3.PROJECT_PER_CAND, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())
        self.coma_q = nn.Sequential(nn.Linear(128, 96), nn.ReLU(), nn.Linear(96, self.n_actions))
        self.coma_out = nn.Sequential(nn.Linear(self.n_actions, 64), nn.ReLU())

        icm_in = v3.RESOURCE_DIM + v3.TDM_DIM
        self.icm_state = nn.Sequential(nn.Linear(icm_in, 96), nn.ReLU(), nn.Linear(96, 32), nn.ReLU())
        self.icm_inverse = nn.Sequential(nn.Linear(64, 96), nn.ReLU(), nn.Linear(96, self.n_actions))
        self.icm_action = nn.Sequential(nn.Linear(32 + self.n_actions, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.icm_out = nn.Sequential(nn.Linear(self.n_actions * 16, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())

        self.deepmdp_state = nn.Sequential(nn.Linear(icm_in, 96), nn.ReLU(), nn.Linear(96, 32), nn.ReLU())
        self.deepmdp_dyn = nn.Sequential(nn.Linear(32 + self.n_actions, 96), nn.ReLU(), nn.Linear(96, 32))
        self.deepmdp_reward = nn.Sequential(nn.Linear(32 + self.n_actions, 64), nn.ReLU(), nn.Linear(64, 1))
        self.deepmdp_out = nn.Sequential(nn.Linear(self.n_actions * 32, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())

        # WM extensions.  V3 tau/PETS/value models already exist in the parent.
        self.steve_out = nn.Sequential(
            nn.Linear(self.n_actions * (v3.RESOURCE_DIM + len(v3.GAMMA_HORIZONS)), 128),
            nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )
        self.ve_ema_out = nn.Sequential(nn.Linear(self.n_actions + 64, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())
        self.coco_action_classifier = nn.Sequential(nn.Linear(v3.RESOURCE_DIM, 64), nn.ReLU(), nn.Linear(64, self.n_actions))
        self.coco_event_out = nn.Sequential(nn.Linear(64 + 64, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())
        self.cvaml_out = nn.Sequential(nn.Linear(32 + 64, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())

    # ---------- common core ----------
    def base_only(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations.float()
        base_obs = self._slice(obs, "base")
        B = base_obs.shape[0]
        frames = base_obs.reshape(B, v3.NUM_FRAMES, v3.SINGLE_FRAME_DIM)
        x = self.frame_encoder(frames)
        x, _ = self.lstm(x)
        base_latent = x[:, -1, :]
        resource_latent = self.resource_branch(self._slice(obs, "resource"))
        focal_latent = self.focal_branch(self._slice(obs, "focal"))
        core = self.v4_core_proj(torch.cat([base_latent, resource_latent, focal_latent], dim=1))
        if self.env_key in {"J6", "J7"}:
            core = core + 0.25 * self.pair_context128(self._slice(obs, "pair_virtual"))
        return core

    # ---------- event token helpers ----------
    def _candidate_event_tokens(self, obs: torch.Tensor, c: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fine = self._slice(obs, "event")
        K = v3.MAX_EVENTS_PER_TYPE
        block = 4 * K
        x = fine[:, c * block:(c + 1) * block]
        a_eta = x[:, 0:K]
        a_mask = x[:, K:2 * K]
        p_eta = x[:, 2 * K:3 * K]
        p_mask = x[:, 3 * K:4 * K]
        eta = torch.cat([a_eta, p_eta], dim=1)
        mask = torch.cat([a_mask, p_mask], dim=1)
        typ = torch.cat([torch.zeros_like(a_eta), torch.ones_like(p_eta)], dim=1)
        elems = torch.stack([eta, typ], dim=-1)
        return elems, mask

    def _deepset_event_latent(self, obs: torch.Tensor) -> torch.Tensor:
        reps = []
        for c in range(v3.N_CANDIDATES):
            elems, mask = self._candidate_event_tokens(obs, c)
            emb = self.set_event_elem(elems)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / denom
            reps.append(pooled)
        return self.set_event_out(torch.cat(reps, dim=1))

    def _settrans_event_latent(self, obs: torch.Tensor) -> torch.Tensor:
        reps = []
        for c in range(v3.N_CANDIDATES):
            elems, mask = self._candidate_event_tokens(obs, c)
            emb = self.set_event_elem(elems)
            key_padding = mask <= 0.0
            # Avoid all-masked rows causing NaNs by temporarily unmasking one zero token.
            all_masked = key_padding.all(dim=1)
            if bool(all_masked.any()):
                key_padding = key_padding.clone()
                key_padding[all_masked, 0] = False
            attn, _ = self.set_event_attn(emb, emb, emb, key_padding_mask=key_padding, need_weights=False)
            valid = (~key_padding).float()
            pooled = (attn * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = pooled * (~all_masked).float().unsqueeze(1)
            reps.append(pooled)
        return self.set_event_out(torch.cat(reps, dim=1))

    def _graph_event_latent(self, obs: torch.Tensor) -> torch.Tensor:
        event_reps = []
        tdm = self._slice(obs, "tdm").reshape(obs.shape[0], v3.N_CANDIDATES, v3.TDM_FEATURES_PER_CAND)
        for c in range(v3.N_CANDIDATES):
            elems, mask = self._candidate_event_tokens(obs, c)
            emb = self.set_event_elem(elems)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / denom
            event_reps.append(pooled)
        ev = torch.stack(event_reps, dim=1)
        nodes = self.graph_node(torch.cat([ev, tdm], dim=-1))
        out, _ = self.graph_attn(nodes, nodes, nodes, need_weights=False)
        return self.graph_out(out.flatten(1))

    def _hksl_multi_latent(self, obs: torch.Tensor) -> torch.Tensor:
        g = self._slice(obs, "gamma")
        B = g.shape[0]
        g = g.reshape(B, v3.N_CANDIDATES, len(v3.GAMMA_HORIZONS), v3.PROJECT_PER_CAND)
        reps = []
        for c in range(v3.N_CANDIDATES):
            h = self.horizon_enc(g[:, c])
            y, _ = self.horizon_gru(h)
            reps.append(y[:, -1, :])
        return self.horizon_out(torch.cat(reps, dim=1))

    # ---------- AC helpers ----------
    def _direct_cqm_latent(self, obs: torch.Tensor) -> torch.Tensor:
        # V3 computes shared=P(shared_tau) and srac=P(own_tau)-P(shared_tau).
        # Therefore shared+srac reconstructs candidate-specific own projection.
        shared = self._slice(obs, "shared").reshape(obs.shape[0], v3.N_CANDIDATES, v3.PROJECT_PER_CAND)
        srac = self._slice(obs, "srac").reshape(obs.shape[0], v3.N_CANDIDATES, v3.PROJECT_PER_CAND)
        own = shared + srac
        delta = own[:, 0] - own[:, 1]
        return self.direct_cqm(torch.cat([delta, -delta], dim=1))

    def coma_values(self, obs: torch.Tensor, detach_core: bool = False) -> torch.Tensor:
        core = self.base_only(obs)
        if detach_core:
            core = core.detach()
        return self.coma_q(core)

    def _coma_latent(self, obs: torch.Tensor) -> torch.Tensor:
        q = self.coma_values(obs, detach_core=False)
        w = torch.softmax(q, dim=1)
        baseline = (w * q).sum(dim=1, keepdim=True)
        adv = q - baseline
        return self.coma_out(adv)

    def icm_z(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self._slice(obs, "resource"), self._slice(obs, "tdm")], dim=1)
        return self.icm_state(x)

    def _icm_latent(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.icm_z(obs)
        per_action = []
        for a in range(self.n_actions):
            oh = F.one_hot(torch.full((obs.shape[0],), a, device=obs.device, dtype=torch.long), self.n_actions).float()
            per_action.append(self.icm_action(torch.cat([z, oh], dim=1)))
        return self.icm_out(torch.cat(per_action, dim=1))

    def deepmdp_z(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self._slice(obs, "resource"), self._slice(obs, "tdm")], dim=1)
        return self.deepmdp_state(x)

    def deepmdp_next(self, z: torch.Tensor, action: int | torch.Tensor) -> torch.Tensor:
        if isinstance(action, int):
            a = torch.full((z.shape[0],), action, device=z.device, dtype=torch.long)
        else:
            a = action.long().flatten().clamp(0, self.n_actions - 1)
        oh = F.one_hot(a, self.n_actions).float()
        return z + self.deepmdp_dyn(torch.cat([z, oh], dim=1))

    def deepmdp_reward_pred(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a = action.long().flatten().clamp(0, self.n_actions - 1)
        oh = F.one_hot(a, self.n_actions).float()
        return self.deepmdp_reward(torch.cat([z, oh], dim=1)).flatten()

    def _deepmdp_latent(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.deepmdp_z(obs)
        diffs = [self.deepmdp_next(z, a) - z for a in range(self.n_actions)]
        return self.deepmdp_out(torch.cat(diffs, dim=1))

    # ---------- WM helpers ----------
    def _wm_input_horizon(self, resource: torch.Tensor, action: int, horizon: float) -> torch.Tensor:
        oh = F.one_hot(
            torch.full((resource.shape[0],), int(action), device=resource.device, dtype=torch.long),
            self.n_actions,
        ).float()
        h = torch.full((resource.shape[0], 1), float(horizon), device=resource.device, dtype=resource.dtype)
        return torch.cat([resource, oh, h], dim=1)

    def pets_horizon_stack(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        resource = self._slice(obs, "resource")
        means, stds = [], []
        for a in range(self.n_actions):
            am, ast = [], []
            for h in v3.GAMMA_HORIZONS:
                inp = self._wm_input_horizon(resource, a, h)
                members = torch.stack([resource + m(inp) for m in self.pets_world_models], dim=1)
                am.append(members.mean(dim=1))
                ast.append(members.std(dim=1, unbiased=False))
            means.append(torch.stack(am, dim=1))
            stds.append(torch.stack(ast, dim=1))
        return torch.stack(means, dim=1), torch.stack(stds, dim=1)  # B,A,H,R

    def _steve_latent(self, obs: torch.Tensor) -> torch.Tensor:
        mean, std = self.pets_horizon_stack(obs)
        unc = std.mean(dim=-1).clamp_min(1e-4)  # B,A,H
        weights = (1.0 / (unc ** 2)).softmax(dim=-1)
        mixed = (mean * weights.unsqueeze(-1)).sum(dim=2)  # B,A,R
        return self.steve_out(torch.cat([mixed.flatten(1), weights.flatten(1)], dim=1))

    def _ve_ema_latent(self, obs: torch.Tensor) -> torch.Tensor:
        event = self._event_query_latent(obs)
        vals = self.predict_all_values(obs)
        return self.ve_ema_out(torch.cat([event, vals], dim=1))

    def _coco_cf_latent(self, obs: torch.Tensor) -> torch.Tensor:
        event = self._event_query_latent(obs)
        pred = self.predict_all_tau_resources(obs)
        wm = self.wm_det_project(pred.flatten(1))
        return self.coco_event_out(torch.cat([event, wm], dim=1))

    def _cvaml_latent(self, obs: torch.Tensor) -> torch.Tensor:
        tdm = self.tdm_branch(self._slice(obs, "tdm"))
        pred = self.predict_all_tau_resources(obs)
        wm = self.wm_det_project(pred.flatten(1))
        return self.cvaml_out(torch.cat([tdm, wm], dim=1))

    def component_latent(self, obs: torch.Tensor, component: str) -> torch.Tensor:
        c = canonical(component)
        if c == "S_SETTRANS_EVENT":
            return self._settrans_event_latent(obs)
        if c == "S_EVENTGRAPH_GAT":
            return self._graph_event_latent(obs)
        if c == "S_HKSL_MULTI":
            return self._hksl_multi_latent(obs)
        if c == "S_TDM_EVENT_FUSION":
            tdm = self.tdm_branch(self._slice(obs, "tdm"))
            ev = self._event_query_latent(obs)
            return self.tdm_event_fuse(torch.cat([tdm, ev], dim=1))
        if c == "A_DIRECT_CQM":
            return self._direct_cqm_latent(obs)
        if c == "A_COMA_AC":
            return self._coma_latent(obs)
        if c == "A_ICM_AC":
            return self._icm_latent(obs)
        if c == "A_DEEPMDP_AC":
            return self._deepmdp_latent(obs)
        if c == "W_GAMMA_PETS_STEVE":
            return self._steve_latent(obs)
        if c == "W_EVENTQ_VE_EMA":
            return self._ve_ema_latent(obs)
        if c == "W_EVENTQ_COCO_CF":
            return self._coco_cf_latent(obs)
        if c == "W_TDM_CVAML":
            return self._cvaml_latent(obs)
        # Native components used only if selected into a combo/control.
        if c == "R_TDM":
            x = self.tdm_branch(self._slice(obs, "tdm"))
            return F.pad(x, (0, 32))
        if c == "R_EVENTQ":
            return self._event_query_latent(obs)
        if c == "R_CQM":
            x = self.cqm_branch(self._slice(obs, "cqm"))
            return F.pad(x, (0, 32))
        if c == "SHARED":
            x = self.shared_branch(self._slice(obs, "shared"))
            return F.pad(x, (0, 32))
        if c == "R_SRAC":
            s = self.shared_branch(self._slice(obs, "shared"))
            a = self.srac_branch(self._slice(obs, "srac"))
            return torch.cat([s, a], dim=1)
        if c == "R_TDM__W_TAU":
            tdm = self.tdm_branch(self._slice(obs, "tdm"))
            pred = self.predict_all_tau_resources(obs)
            wm = self.wm_det_project(pred.flatten(1))
            return self.cvaml_out(torch.cat([tdm, wm], dim=1))
        raise KeyError(f"Unknown V4 component: {c}")

    def custom_forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations.float()
        core = self.base_only(obs)
        latents = [self.component_latent(obs, c) for c in self.components_v4]
        if len(latents) > 3:
            raise RuntimeError(f"At most 3 components supported, got {self.components_v4}")
        while len(latents) < 3:
            latents.append(torch.zeros((obs.shape[0], 64), device=obs.device, dtype=obs.dtype))
        return self.v4_fusion(torch.cat([core] + latents, dim=1))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.method_id_v4 in NATIVE_METHODS:
            out = super().forward(observations)
            # V3.1 already injects pair_virtual for J7; J6 is the new
            # aircraft-value-planning card and needs the same virtual semantics.
            if self.env_key == "J6":
                out = out + 0.25 * self.pair_context128(self._slice(observations.float(), "pair_virtual"))
            return out
        return self.custom_forward(observations)

    # Aux parameter groups.
    def aux_parameters_for(self, component: str) -> List[nn.Parameter]:
        c = canonical(component)
        modules: List[nn.Module] = []
        if c == "A_COMA_AC":
            modules = [self.coma_q]
        elif c == "A_ICM_AC":
            modules = [self.icm_state, self.icm_inverse]
        elif c == "A_DEEPMDP_AC":
            modules = [self.deepmdp_state, self.deepmdp_dyn, self.deepmdp_reward]
        elif c == "W_GAMMA_PETS_STEVE":
            modules = list(self.pets_world_models)
        elif c == "W_EVENTQ_VE_EMA":
            modules = [self.value_equiv_model]
        elif c == "W_EVENTQ_COCO_CF":
            modules = [self.tau_world_model, self.coco_action_classifier]
        elif c == "W_TDM_CVAML":
            modules = [self.tau_world_model]
        params: List[nn.Parameter] = []
        for m in modules:
            params.extend(list(m.parameters()))
        return params


# =============================================================================
# New joint policy cards
# =============================================================================

class DiscoveryJointPolicy(ActorCriticPolicy):
    """Eight Joint cards over Discrete(4).

    J4/J5/J6/J7 are explicitly STYLE/INSPIRED cards.  They retain PPO as the
    optimizer so the experiment isolates coordination structure rather than
    swapping the entire RL algorithm.
    """

    def __init__(self, *args, joint_arch: str = "J0", **kwargs):
        self.joint_arch = canonical(joint_arch)
        lr_schedule = kwargs.get("lr_schedule", None)
        if lr_schedule is None and len(args) >= 3:
            lr_schedule = args[2]
        super().__init__(*args, **kwargs)

        d = int(self.mlp_extractor.latent_dim_pi)
        self.action_net = nn.Identity()

        self.p_head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 2))
        self.a_head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 2))
        self.shared_context = nn.Sequential(nn.Linear(d, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())
        self.pair_head = nn.Sequential(nn.Linear(d, 96), nn.ReLU(), nn.Linear(96, 4))
        self.cond_a = nn.Sequential(nn.Linear(d + 2, 64), nn.ReLU(), nn.Linear(64, 2))

        # J4: MAPPO-style CTDE actor card.  In this single-controller simulator,
        # the critic is already centralized on the full state; the experimental
        # change is a factorized actor with cross-gating between branches.
        self.mappo_shared = nn.Sequential(nn.Linear(d, 96), nn.Tanh(), nn.Linear(96, 64), nn.Tanh())
        self.mappo_gate = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
        self.mappo_p = nn.Linear(64, 2)
        self.mappo_a = nn.Linear(64, 2)

        # J5: COMA-style counterfactual pair scoring.
        self.coma_q = nn.Sequential(nn.Linear(d, 96), nn.ReLU(), nn.Linear(96, 4))

        # J6/J7: planning-style decompositions.  They learn values from virtual
        # post-action semantics already embedded by the observation extractor;
        # they do not claim exact KDD dispatcher reproduction.
        self.air_passenger = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 2))
        self.air_pair_value = nn.Sequential(nn.Linear(d, 96), nn.ReLU(), nn.Linear(96, 4))
        self.pair_immediate = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 4))
        self.pair_future = nn.Sequential(nn.Linear(d, 96), nn.ReLU(), nn.Linear(96, 4))

        if lr_schedule is None:
            raise RuntimeError("Cannot recover learning-rate schedule")
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _joint_logits(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.shape[0]
        j = self.joint_arch
        if j == "J0":
            p = self.p_head(latent)
            a = self.a_head(latent)
            return (p.unsqueeze(2) + a.unsqueeze(1)).reshape(B, 4)
        if j == "J1":
            h = self.shared_context(latent)
            p = self.mappo_p(h)
            a = self.mappo_a(h)
            return (p.unsqueeze(2) + a.unsqueeze(1)).reshape(B, 4)
        if j == "J2":
            return self.pair_head(latent)
        if j == "J3":
            p_logits = self.p_head(latent)
            p_logp = F.log_softmax(p_logits, dim=1)
            rows = []
            for p in range(2):
                p_oh = F.one_hot(torch.full((B,), p, device=latent.device, dtype=torch.long), 2).float()
                a_logp = F.log_softmax(self.cond_a(torch.cat([latent, p_oh], dim=1)), dim=1)
                rows.append(p_logp[:, p:p + 1] + a_logp)
            return torch.stack(rows, dim=1).reshape(B, 4)
        if j == "J4":
            h = self.mappo_shared(latent)
            p = self.mappo_p(h)
            a = self.mappo_a(h)
            gate = self.mappo_gate(h).reshape(B, 2, 2)
            return (p.unsqueeze(2) + a.unsqueeze(1) + 0.25 * gate).reshape(B, 4)
        if j == "J5":
            q = self.coma_q(latent).reshape(B, 2, 2)
            # Counterfactual baselines along each branch; lower common-mode
            # components are cancelled before producing pair logits.
            wa = torch.softmax(q, dim=2)
            wp = torch.softmax(q, dim=1)
            ba = (wa * q).sum(dim=2, keepdim=True)
            bp = (wp * q).sum(dim=1, keepdim=True)
            adv = q - 0.5 * (ba + bp)
            return adv.reshape(B, 4)
        if j == "J6":
            p = self.air_passenger(latent)
            pair_v = self.air_pair_value(latent).reshape(B, 2, 2)
            # Passenger policy + aircraft candidate future-value ranking.
            return (p.unsqueeze(2) + pair_v).reshape(B, 4)
        if j == "J7":
            immediate = self.pair_immediate(latent)
            future = self.pair_future(latent)
            return immediate + future
        raise ValueError(j)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        return self.action_dist.proba_distribution(action_logits=self._joint_logits(latent_pi))


# =============================================================================
# Auxiliary PPO for AC / WM discovery cards
# =============================================================================

class DiscoveryAuxPPO(PPO):
    def _ext(self) -> Optional[DiscoveryExtractor]:
        x = getattr(self.policy, "features_extractor", None)
        return x if isinstance(x, DiscoveryExtractor) else None

    def _components(self) -> Tuple[str, ...]:
        ext = self._ext()
        return ext.components_v4 if ext is not None else tuple()

    def _optimizer_for(self, comp: str, ext: DiscoveryExtractor):
        if not hasattr(self, "_aux_optimizers"):
            self._aux_optimizers = {}
        c = canonical(comp)
        if c not in self._aux_optimizers:
            params = ext.aux_parameters_for(c)
            if not params:
                return None
            self._aux_optimizers[c] = torch.optim.Adam(params, lr=AUX_LR)
        return self._aux_optimizers[c]

    def _pair_batch(self, ext: DiscoveryExtractor):
        obs = np.asarray(self.rollout_buffer.observations)
        act = np.asarray(self.rollout_buffer.actions).reshape(obs.shape[0], obs.shape[1], -1)[..., 0].astype(np.int64)
        starts = np.asarray(self.rollout_buffer.episode_starts)
        rewards = np.asarray(self.rollout_buffer.rewards)
        returns = np.asarray(self.rollout_buffer.returns)
        idx = []
        for t in range(obs.shape[0] - 1):
            for e in range(obs.shape[1]):
                if starts[t + 1, e] > 0.5:
                    continue
                idx.append((t, e))
        if not idx:
            return None
        if len(idx) > AUX_MAX_SAMPLES:
            pick = np.random.choice(len(idx), AUX_MAX_SAMPLES, replace=False)
            idx = [idx[int(i)] for i in pick]
        cur = torch.as_tensor(np.stack([obs[t, e] for t, e in idx]), device=self.device, dtype=torch.float32)
        nxt = torch.as_tensor(np.stack([obs[t + 1, e] for t, e in idx]), device=self.device, dtype=torch.float32)
        a = torch.as_tensor([act[t, e] for t, e in idx], device=self.device, dtype=torch.long)
        r = torch.as_tensor([rewards[t, e] for t, e in idx], device=self.device, dtype=torch.float32)
        ret = torch.as_tensor([returns[t, e] for t, e in idx], device=self.device, dtype=torch.float32)
        return cur, a, r, ret, nxt

    def _future_pairs(self, ext: DiscoveryExtractor):
        # Same committed tau target semantics as V3.1 world-model discovery.
        obs_np = np.asarray(self.rollout_buffer.observations)
        act_np = np.asarray(self.rollout_buffer.actions)
        starts_np = np.asarray(self.rollout_buffer.episode_starts)
        if obs_np.ndim < 3 or obs_np.shape[0] < 2:
            return None
        n_steps, n_envs = obs_np.shape[:2]
        lo, hi = ext.slices["tau"]
        tau_raw = np.asarray(obs_np[..., int(lo):int(hi)], dtype=float)
        acts = act_np.reshape(n_steps, n_envs, -1)[..., 0].astype(int)
        idxs: List[Tuple[int, int, int]] = []
        for t in range(n_steps - 1):
            for e in range(n_envs):
                a = int(acts[t, e])
                p = a if not ext.env_spec.joint else a // 2
                p = min(max(p, 0), v3.N_CANDIDATES - 1)
                h = v3.fnum(tau_raw[t, e, p], 1.0)
                off = int(round(max(1.0, min(float(v3.WM_HORIZON_CAP), h))))
                tgt = t + off
                if tgt >= n_steps:
                    continue
                if np.any(starts_np[t + 1:tgt + 1, e] > 0.5):
                    continue
                idxs.append((t, e, tgt))
        if not idxs:
            return None
        if len(idxs) > AUX_MAX_SAMPLES:
            pick = np.random.choice(len(idxs), AUX_MAX_SAMPLES, replace=False)
            idxs = [idxs[int(i)] for i in pick]
        cur = torch.as_tensor(np.stack([obs_np[t, e] for t, e, _ in idxs]), device=self.device, dtype=torch.float32)
        nxt = torch.as_tensor(np.stack([obs_np[tgt, e] for t, e, tgt in idxs]), device=self.device, dtype=torch.float32)
        act = torch.as_tensor([acts[t, e] for t, e, _ in idxs], device=self.device, dtype=torch.long)
        return cur, act, nxt

    def _base_value(self, obs: torch.Tensor, policy=None) -> torch.Tensor:
        pol = self.policy if policy is None else policy
        ext = pol.features_extractor
        assert isinstance(ext, DiscoveryExtractor)
        feat = ext.base_only(obs)
        latent = pol.mlp_extractor.forward_critic(feat)
        return pol.value_net(latent).flatten()

    def _target_policy(self):
        if not hasattr(self, "_ema_policy"):
            self._ema_policy = copy.deepcopy(self.policy).to(self.device)
            self._ema_policy.set_training_mode(False)
            for p in self._ema_policy.parameters():
                p.requires_grad_(False)
        return self._ema_policy

    @torch.no_grad()
    def _update_ema(self):
        if not hasattr(self, "_ema_policy"):
            return
        src = dict(self.policy.named_parameters())
        for name, p_t in self._ema_policy.named_parameters():
            p_t.mul_(EMA_TAU).add_(src[name], alpha=1.0 - EMA_TAU)
        src_b = dict(self.policy.named_buffers())
        for name, b_t in self._ema_policy.named_buffers():
            if name in src_b and b_t.dtype.is_floating_point:
                b_t.mul_(EMA_TAU).add_(src_b[name], alpha=1.0 - EMA_TAU)
            elif name in src_b:
                b_t.copy_(src_b[name])

    def _run_batches(self, n: int):
        for _ in range(AUX_EPOCHS):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, AUX_BATCH):
                yield order[start:start + AUX_BATCH]

    def _train_component(self, comp: str, ext: DiscoveryExtractor) -> Optional[float]:
        c = canonical(comp)
        opt = self._optimizer_for(c, ext)
        if opt is None:
            return None
        losses: List[float] = []

        if c in {"A_COMA_AC", "A_ICM_AC", "A_DEEPMDP_AC"}:
            batch = self._pair_batch(ext)
            if batch is None:
                return None
            cur, act, reward, returns, nxt = batch
            n = cur.shape[0]
            for ix in self._run_batches(n):
                if c == "A_COMA_AC":
                    q = ext.coma_values(cur[ix], detach_core=True)
                    chosen = q.gather(1, act[ix].view(-1, 1)).flatten()
                    loss = F.smooth_l1_loss(chosen, returns[ix].detach())
                elif c == "A_ICM_AC":
                    z0 = ext.icm_z(cur[ix])
                    z1 = ext.icm_z(nxt[ix])
                    logits = ext.icm_inverse(torch.cat([z0, z1], dim=1))
                    loss = F.cross_entropy(logits, act[ix])
                else:
                    z0 = ext.deepmdp_z(cur[ix])
                    z1 = ext.deepmdp_z(nxt[ix]).detach()
                    pred = ext.deepmdp_next(z0, act[ix])
                    rp = ext.deepmdp_reward_pred(z0, act[ix])
                    loss = F.mse_loss(pred, z1) + 0.25 * F.mse_loss(rp, reward[ix])
                opt.zero_grad(set_to_none=True)
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ext.aux_parameters_for(c), 5.0)
                opt.step()
                self.policy.optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else None

        if c in {"W_GAMMA_PETS_STEVE", "W_EVENTQ_VE_EMA", "W_EVENTQ_COCO_CF", "W_TDM_CVAML"}:
            pairs = self._future_pairs(ext)
            if pairs is None:
                return None
            cur, act, nxt = pairs
            rlo, rhi = ext.slices["resource"]
            target_resource = nxt[:, int(rlo):int(rhi)].detach()
            n = cur.shape[0]
            for ix in self._run_batches(n):
                cc, aa, nnxt = cur[ix], act[ix], nxt[ix]
                tr = target_resource[ix]
                if c == "W_GAMMA_PETS_STEVE":
                    members = ext.predict_chosen_pets_members(cc, aa)
                    mloss = []
                    for m in range(members.shape[1]):
                        pred = members[:, m, :]
                        mask = (torch.rand(pred.shape[0], device=pred.device) < 0.8).float()
                        mse = ((pred - tr) ** 2).mean(dim=1)
                        mloss.append((mse * mask).sum() / mask.sum().clamp_min(1.0))
                    loss = torch.stack(mloss).mean()
                elif c == "W_EVENTQ_VE_EMA":
                    target_policy = self._target_policy()
                    with torch.no_grad():
                        target_v = self._base_value(nnxt, target_policy)
                    pred_v = ext.predict_chosen_value(cc, aa)
                    loss = F.mse_loss(pred_v, target_v)
                elif c == "W_EVENTQ_COCO_CF":
                    pred = ext.predict_chosen_tau_resource(cc, aa)
                    state_loss = F.mse_loss(pred, tr)
                    allp = ext.predict_all_tau_resources(cc)
                    resource_now = ext._slice(cc, "resource").unsqueeze(1)
                    deltas = allp - resource_now
                    logits = ext.coco_action_classifier(deltas.reshape(-1, v3.RESOURCE_DIM))
                    labels = torch.arange(ext.n_actions, device=cc.device).view(1, -1).expand(cc.shape[0], -1).reshape(-1)
                    controllability = F.cross_entropy(logits, labels)
                    loss = state_loss + COCO_CF_LAMBDA * controllability
                else:
                    # C-VAML-inspired normalized value-aware loss.  This is not
                    # a verbatim implementation of the 2025 calibration paper.
                    self.policy.set_training_mode(True)
                    pred = ext.predict_chosen_tau_resource(cc, aa)
                    state_loss = F.mse_loss(pred, tr)
                    pseudo = nnxt.clone()
                    pseudo[:, int(rlo):int(rhi)] = pred
                    with torch.no_grad():
                        target_v = self._base_value(nnxt)
                    pred_v = self._base_value(pseudo)
                    scale = target_v.var(unbiased=False).detach().clamp_min(1.0)
                    value_loss = F.mse_loss(pred_v, target_v) / scale
                    loss = state_loss + CVAML_LAMBDA * value_loss
                opt.zero_grad(set_to_none=True)
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ext.aux_parameters_for(c), 5.0)
                opt.step()
                self.policy.optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else None
        return None

    def train(self) -> None:
        ext = self._ext()
        aux_logs = {}
        if ext is not None:
            for comp in ext.components_v4:
                val = self._train_component(comp, ext)
                if val is not None:
                    aux_logs[canonical(comp)] = val
        super().train()
        self._update_ema()
        for comp, val in aux_logs.items():
            self.logger.record(f"train/v4_aux_{comp.lower()}", val)


# =============================================================================
# Fast checkpoint callback: exact nominal 50k names with n_envs=32
# =============================================================================

class NominalCheckpointCallback(BaseCallback):
    def __init__(self, run_dir: Path, interval: int = CHECKPOINT_INTERVAL):
        super().__init__(verbose=0)
        self.run_dir = Path(run_dir)
        self.interval = int(interval)
        self.next_mark = int(interval)
        self.started = 0.0
        self.rows: List[Dict[str, Any]] = []

    def _on_training_start(self) -> None:
        self.started = time.perf_counter()
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _save_mark(self, mark: int) -> None:
        ck = self.run_dir / "checkpoints"
        self.model.save(str(ck / f"uam_ppo_{mark}_steps.zip"))
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            vn.save(str(ck / f"uam_ppo_vecnormalize_{mark}_steps.pkl"))
        elapsed = time.perf_counter() - self.started
        self.rows.append({
            "nominal_timesteps": int(mark),
            "actual_timesteps": int(self.num_timesteps),
            "checkpoint_overshoot": int(self.num_timesteps - mark),
            "elapsed_sec": float(elapsed),
            "sps": float(self.num_timesteps / max(elapsed, 1e-9)),
        })
        v3.write_csv(self.run_dir / "training_throughput.csv", self.rows)
        print(
            f"[CKPT] nominal={mark:,} actual={self.num_timesteps:,} "
            f"overshoot={self.num_timesteps-mark} SPS={self.rows[-1]['sps']:.1f}",
            flush=True,
        )

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_mark:
            self._save_mark(self.next_mark)
            self.next_mark += self.interval
        return True


def build_callbacks(run_dir: Path, profile: v3.SpeedProfile) -> List[BaseCallback]:
    # Works for n_envs=32 even though 50k is not divisible by 32.
    return [NominalCheckpointCallback(run_dir, CHECKPOINT_INTERVAL)]


# =============================================================================
# Build model / monkey-patch validated V3 lifecycle
# =============================================================================

def custom_method_needs_aux(method_id: str) -> bool:
    aux = {
        "A_COMA_AC", "A_ICM_AC", "A_DEEPMDP_AC",
        "W_GAMMA_PETS_STEVE", "W_EVENTQ_VE_EMA", "W_EVENTQ_COCO_CF", "W_TDM_CVAML",
    }
    return any(canonical(c) in aux for c in components_for(method_id))


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
    spec = v3.ENV_SPECS[env_key]
    policy_kwargs = dict(
        features_extractor_class=DiscoveryExtractor,
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
        policy = DiscoveryJointPolicy
        policy_kwargs["joint_arch"] = env_key

    m = canonical(method_id)
    if custom_method_needs_aux(m):
        algo_cls = DiscoveryAuxPPO
    elif m in NATIVE_WM_METHODS:
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


def install_v4_hooks() -> None:
    # V3 lifecycle resolves these module globals at runtime, so replacing them
    # lets us reuse all validated environment/train/eval/completion bookkeeping.
    v3.build_model = build_model
    v3.build_callbacks = build_callbacks
    v3.method_description = method_description
    v3.JOINT_ARCH_NAMES.clear()
    v3.JOINT_ARCH_NAMES.update(JOINT_ARCH_NAMES_V4)


# =============================================================================
# Robust curve statistics / staged selection
# =============================================================================

def _truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes"}


def curve_stats(root: Path, env_key: str, method_id: str) -> Dict[str, float]:
    run = root / v3.cell_id(env_key, method_id)

    # Formal analysis has priority whenever it exists.
    p = run / "analysis" / "checkpoint_curve.csv"
    rows = v3.read_csv(p)
    valid = []
    for r in rows:
        att = v3.fnum(r.get("ATT_mean"))
        if _truthy(r.get("all_full_completion")) and math.isfinite(att):
            valid.append((int(float(r["train_step"])), att))
    if valid:
        valid.sort()
        vals = [a for _, a in valid]
        best = min(vals)
        final = vals[-1]
        late3 = float(np.mean(vals[-3:]))
        mean12 = float(np.mean(vals))
        return {
            "best": float(best),
            "mean12": mean12,
            "late3": late3,
            "final": float(final),
            "collapse": float(final - best),
            "n_valid": len(valid),
        }

    # Joint formal evaluation is intentionally deferred until the full 60M
    # training matrix is finished.  Adaptive J0/J1/J2 routing therefore uses
    # a tiny NON-FORMAL selection probe: final 600k model only, eval seeds only.
    # It is stored outside analysis/ and is never reported as the paper result.
    probe = run / "selection_probe.json"
    if probe.exists():
        try:
            obj = json.loads(probe.read_text(encoding="utf-8"))
            att = v3.fnum(obj.get("ATT_mean"))
            if bool(obj.get("all_full_completion")) and math.isfinite(att):
                return {
                    "best": float(att),
                    "mean12": float(att),
                    "late3": float(att),
                    "final": float(att),
                    "collapse": 0.0,
                    "n_valid": int(obj.get("n_valid", 1)),
                }
        except Exception:
            pass

    return {
        "best": 1e12, "mean12": 1e12, "late3": 1e12,
        "final": 1e12, "collapse": 1e12, "n_valid": 0,
    }


def aggregate_method(root: Path, env_keys: Sequence[str], method_id: str) -> Dict[str, float]:
    stats = [curve_stats(root, e, method_id) for e in env_keys]
    return {
        "best": float(np.median([s["best"] for s in stats])),
        "mean12": float(np.median([s["mean12"] for s in stats])),
        "late3": float(np.median([s["late3"] for s in stats])),
        "collapse": float(np.median([s["collapse"] for s in stats])),
        "min_valid": int(min(s["n_valid"] for s in stats)),
    }


def aggregate_arch(root: Path, arch: str, methods: Sequence[str]) -> Dict[str, float]:
    stats = [curve_stats(root, arch, m) for m in methods]
    return {
        "best": float(np.median([s["best"] for s in stats])),
        "mean12": float(np.median([s["mean12"] for s in stats])),
        "late3": float(np.median([s["late3"] for s in stats])),
        "collapse": float(np.median([s["collapse"] for s in stats])),
        "min_valid": int(min(s["n_valid"] for s in stats)),
    }


def dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    keys = ("mean12", "late3", "best", "collapse")
    return all(a[k] <= b[k] for k in keys) and any(a[k] < b[k] for k in keys)


def pareto_rank(items: Sequence[Tuple[str, Dict[str, float]]]) -> List[Tuple[str, Dict[str, float]]]:
    items = list(items)
    front = []
    rest = []
    for i, (name, stat) in enumerate(items):
        if stat.get("min_valid", 0) <= 0:
            rest.append((name, stat))
            continue
        dom = any(dominates(other, stat) for j, (_, other) in enumerate(items) if i != j and other.get("min_valid", 0) > 0)
        (rest if dom else front).append((name, stat))
    key = lambda x: (x[1]["mean12"], x[1]["late3"], x[1]["best"], x[1]["collapse"])
    return sorted(front, key=key) + sorted(rest, key=key)


def select_top_arches_j0(root: Path, n: int = 3) -> List[str]:
    ranked = pareto_rank([(j, aggregate_arch(root, j, J0_ANCHORS)) for j in JOINT_KEYS])
    out = [j for j, _ in ranked[:n]]
    v3.write_json(root / "selection_j0_top3.json", {
        "criterion": "Pareto over median mean12/late3/best/collapse across CURRENT,TDM,EVENTQ; completion required",
        "ranking": [{"arch": j, **s} for j, s in ranked],
        "top3": out,
    })
    return out


def rank_family(root: Path, top_arches: Sequence[str], methods: Sequence[str], family: str) -> List[str]:
    ranked = pareto_rank([(m, aggregate_method(root, top_arches, m)) for m in methods])
    out = [m for m, _ in ranked]
    v3.write_json(root / f"selection_j1_{family.lower()}.json", {
        "criterion": "Pareto over median stats across selected Joint Top3",
        "ranking": [{"method": m, **s} for m, s in ranked],
        "ordered": out,
    })
    return out


def select_best_joint_after_j1(root: Path, top_arches: Sequence[str]) -> str:
    methods = list(J0_ANCHORS) + list(J1_METHODS)
    ranked = pareto_rank([(j, aggregate_arch(root, j, methods)) for j in top_arches])
    best = ranked[0][0]
    v3.write_json(root / "selection_j1_best_joint.json", {
        "criterion": "robust aggregate over all 15 anchor+mechanism methods",
        "ranking": [{"arch": j, **s} for j, s in ranked],
        "best_joint": best,
    })
    return best


# =============================================================================
# Plans / resume reconstruction
# =============================================================================

def register_j2_combos(shared_best: str, ac_best: str, wm_best: str) -> Dict[str, Tuple[str, ...]]:
    combos = {
        "C_SA": (shared_best, ac_best),
        "C_SW": (shared_best, wm_best),
        "C_AW": (ac_best, wm_best),
        "C_SAW": (shared_best, ac_best, wm_best),
    }
    COMBO_REGISTRY.update({canonical(k): tuple(canonical(x) for x in v) for k, v in combos.items()})
    return COMBO_REGISTRY


def load_selection_state(root: Path) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    p = root / "phase_j1_plan.json"
    if p.exists():
        state["j1"] = json.loads(p.read_text(encoding="utf-8"))
    p = root / "phase_j2_plan.json"
    if p.exists():
        obj = json.loads(p.read_text(encoding="utf-8"))
        state["j2"] = obj
        for k, vals in obj.get("combo_registry", {}).items():
            COMBO_REGISTRY[canonical(k)] = tuple(canonical(x) for x in vals)
    p = root / "phase_single_plan.json"
    if p.exists():
        obj = json.loads(p.read_text(encoding="utf-8"))
        state["single"] = obj
        for k, vals in obj.get("combo_registry", {}).items():
            COMBO_REGISTRY[canonical(k)] = tuple(canonical(x) for x in vals)
    return state


# =============================================================================
# Execution helpers
# =============================================================================

def run_cell(
    root: Path,
    env_key: str,
    method_id: str,
    args: argparse.Namespace,
) -> None:
    # Requested evaluation schedule:
    #   * ALL Joint cells: train only now; formal checkpoint evaluation deferred.
    #   * Single-side S1/S2/S3: evaluate immediately after each cell.
    is_joint = str(env_key).upper() in JOINT_KEYS
    v3.safe_train_and_optional_eval(
        root=root,
        env_key=env_key,
        method_id=method_id,
        requested_steps=int(args.timesteps),
        train_seed=int(args.train_seed),
        eval_seeds=v3.parse_ints(args.eval_seeds),
        profile=FAST_PROFILE,
        device="cuda",
        max_time=int(args.max_time),
        defer_eval=bool(is_joint),
        fail_fast=bool(args.fail_fast),
    )


def selection_probe_cell(
    root: Path,
    env_key: str,
    method_id: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Tiny routing-only probe for adaptive Joint phases.

    This is deliberately NOT a formal experiment evaluation:
      - final model only (600k in formal runs),
      - the configured eval seeds,
      - stored as selection_probe.json outside analysis/,
      - never inserted into matrix_master.csv or paper checkpoint curves.

    Full 50k..600k Joint evaluation is deferred until all 100 training cells
    have completed.
    """
    cid = v3.cell_id(env_key, method_id)
    run_dir = root / cid
    probe_path = run_dir / "selection_probe.json"

    if probe_path.exists():
        try:
            obj = json.loads(probe_path.read_text(encoding="utf-8"))
            if int(obj.get("train_step", -1)) == int(args.timesteps):
                return obj
        except Exception:
            pass

    if not v3.training_complete(run_dir, int(args.timesteps)):
        obj = {
            "cell_id": cid,
            "env_key": env_key,
            "method_id": method_id,
            "train_step": int(args.timesteps),
            "status": "SKIP_NOT_TRAINED",
            "ATT_mean": float("nan"),
            "all_full_completion": False,
            "n_valid": 0,
        }
        v3.write_json(probe_path, obj)
        return obj

    model_path = run_dir / "final_rl_model.zip"
    vec_path = run_dir / "final_vec_normalize.pkl"
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    eval_seeds = v3.parse_ints(args.eval_seeds)

    print(f"[SELECTION PROBE] {cid} | final model only | seeds={eval_seeds}", flush=True)
    for seed in eval_seeds:
        try:
            row = v3.evaluate_checkpoint(
                env_key=env_key,
                method_id=method_id,
                model_path=model_path,
                vec_path=vec_path,
                train_step=int(args.timesteps),
                eval_seed=int(seed),
                run_dir=run_dir,
                max_time=int(args.max_time),
            )
            rows.append(row)
        except Exception as exc:
            errors.append({
                "eval_seed": int(seed),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            if bool(args.fail_fast):
                raise

    valid = [
        r for r in rows
        if bool(r.get("valid_full_completion"))
        and math.isfinite(v3.fnum(r.get("ATT")))
    ]
    all_full = len(rows) == len(eval_seeds) and len(valid) == len(eval_seeds)
    att_mean = v3.fmean(r.get("ATT") for r in valid) if all_full else float("nan")
    att_std = v3.fstd(r.get("ATT") for r in valid) if all_full else float("nan")

    obj = {
        "cell_id": cid,
        "env_key": env_key,
        "method_id": method_id,
        "train_step": int(args.timesteps),
        "status": "VALID" if all_full else "NO_VALID_FULL_COMPLETION",
        "ATT_mean": att_mean,
        "ATT_std": att_std,
        "all_full_completion": bool(all_full),
        "n_valid": len(valid),
        "n_eval_seeds": len(rows),
        "n_errors": len(errors),
        "purpose": "adaptive-routing-only; NOT formal evaluation",
    }
    v3.write_json(probe_path, obj)
    if errors:
        v3.write_json(run_dir / "selection_probe_errors.json", errors)
    return obj


def selection_probe_many(
    root: Path,
    cells: Sequence[Tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    for env_key, method_id in cells:
        selection_probe_cell(root, env_key, method_id, args)


def collect_joint_plan(root: Path) -> List[Dict[str, str]]:
    """Reconstruct the exact 67 adaptive Joint cells for final formal evaluation."""
    plan: List[Dict[str, str]] = []

    # J0: fixed 24
    for j in JOINT_KEYS:
        for m in J0_ANCHORS:
            plan.append({"env_key": j, "method_id": m})

    # J1: adaptive 36
    p1 = root / "phase_j1_plan.json"
    if not p1.exists():
        raise FileNotFoundError(f"Missing Joint phase plan: {p1}")
    j1 = json.loads(p1.read_text(encoding="utf-8"))
    for rec in j1.get("cells", []):
        plan.append({"env_key": rec["env_key"], "method_id": rec["method_id"]})

    # J2: adaptive 7
    p2 = root / "phase_j2_plan.json"
    if not p2.exists():
        raise FileNotFoundError(f"Missing Joint phase plan: {p2}")
    j2 = json.loads(p2.read_text(encoding="utf-8"))
    bj = j2["best_joint"]
    for m in j2.get("methods", []):
        plan.append({"env_key": bj, "method_id": m})

    # Preserve order while guarding against an accidental duplicate.
    uniq: List[Dict[str, str]] = []
    seen = set()
    for rec in plan:
        key = (canonical(rec["env_key"]), canonical(rec["method_id"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append({"env_key": key[0], "method_id": key[1]})

    if len(uniq) != 67:
        raise RuntimeError(f"Expected 67 unique Joint cells, reconstructed {len(uniq)}")
    v3.write_json(root / "final_joint_eval_plan.json", {
        "cells": uniq,
        "count": len(uniq),
        "schedule": "formal evaluation only after all 100 training cells finish",
    })
    return uniq



def run_phase_j0(root: Path, args: argparse.Namespace) -> List[str]:
    print("\n" + "=" * 120)
    print("PHASE J0 | 8 JOINT ARCHITECTURES x 3 CLEAN ANCHORS = 24 CELLS = 14.4M")
    print("=" * 120, flush=True)
    cells = [(j, m) for j in JOINT_KEYS for m in J0_ANCHORS]
    for j, m in cells:
        run_cell(root, j, m, args)

    # Routing-only probe; full Joint checkpoint curves remain deferred.
    selection_probe_many(root, cells, args)
    top3 = select_top_arches_j0(root, 3)
    print(f"J0 TOP3 = {top3}", flush=True)
    return top3


def run_phase_j1(root: Path, args: argparse.Namespace, top3: Sequence[str]) -> Dict[str, Any]:
    print("\n" + "=" * 120)
    print("PHASE J1 | TOP3 JOINT x (4 SHARED + 4 AC + 4 WM) = 36 CELLS = 21.6M")
    print("=" * 120, flush=True)
    plan = [{"env_key": j, "method_id": m, "family": method_family(m)} for j in top3 for m in J1_METHODS]
    v3.write_json(root / "phase_j1_plan.json", {"top3_joint": list(top3), "cells": plan})
    for rec in plan:
        run_cell(root, rec["env_key"], rec["method_id"], args)

    selection_probe_many(
        root,
        [(rec["env_key"], rec["method_id"]) for rec in plan],
        args,
    )
    shared_rank = rank_family(root, top3, SHARED_METHODS, "SHARED")
    ac_rank = rank_family(root, top3, AC_METHODS, "AC")
    wm_rank = rank_family(root, top3, WM_METHODS, "WM")
    best_joint = select_best_joint_after_j1(root, top3)
    out = {
        "top3_joint": list(top3),
        "best_joint": best_joint,
        "shared_rank": shared_rank,
        "ac_rank": ac_rank,
        "wm_rank": wm_rank,
    }
    v3.write_json(root / "phase_j1_selection.json", out)
    return out


def run_phase_j2(root: Path, args: argparse.Namespace, sel: Dict[str, Any]) -> Dict[str, Any]:
    best_joint = sel["best_joint"]
    s1 = sel["shared_rank"][0]
    a1 = sel["ac_rank"][0]
    w1 = sel["wm_rank"][0]
    register_j2_combos(s1, a1, w1)

    methods = ["C_SA", "C_SW", "C_AW", "C_SAW", "SHARED", "R_SRAC", "R_TDM__W_TAU"]
    plan = {
        "best_joint": best_joint,
        "shared_best": s1,
        "ac_best": a1,
        "wm_best": w1,
        "methods": methods,
        "combo_registry": {k: list(v) for k, v in COMBO_REGISTRY.items()},
    }
    v3.write_json(root / "phase_j2_plan.json", plan)

    print("\n" + "=" * 120)
    print("PHASE J2 | BEST JOINT x 7 INTERACTION / NEGATIVE-CONTROL CELLS = 4.2M")
    print(f"BEST JOINT={best_joint} | S*={s1} | A*={a1} | W*={w1}")
    print("=" * 120, flush=True)
    for m in methods:
        run_cell(root, best_joint, m, args)

    selection_probe_many(root, [(best_joint, m) for m in methods], args)
    pair = ["C_SA", "C_SW", "C_AW"]
    ranked_pair = pareto_rank([(m, aggregate_method(root, [best_joint], m)) for m in pair])
    best_pair = ranked_pair[0][0]
    out = {**plan, "best_pair_combo": best_pair, "full_combo": "C_SAW"}
    v3.write_json(root / "phase_j2_selection.json", out)
    return out


def run_phase_single(root: Path, args: argparse.Namespace, j1sel: Dict[str, Any], j2sel: Dict[str, Any]) -> None:
    # 3 anchors + Top2 Shared + Top2 AC + Top2 WM + best pair + full combo = 11.
    methods = [
        "CURRENT", "R_TDM", "R_EVENTQ",
        *j1sel["shared_rank"][:2],
        *j1sel["ac_rank"][:2],
        *j1sel["wm_rank"][:2],
        j2sel["best_pair_combo"],
        j2sel["full_combo"],
    ]
    # preserve order / remove accidental duplicates
    methods = list(dict.fromkeys(canonical(m) for m in methods))
    if len(methods) != 11:
        raise RuntimeError(f"Single plan expected 11 unique methods, got {len(methods)}: {methods}")

    plan = {
        "methods": methods,
        "envs": ["S1", "S2", "S3"],
        "combo_registry": {k: list(v) for k, v in COMBO_REGISTRY.items()},
        "cells": 33,
    }
    v3.write_json(root / "phase_single_plan.json", plan)

    print("\n" + "=" * 120)
    print("PHASE S (RUNS LAST) | 11 FINALISTS x S1/S2/S3 = 33 CELLS = 19.8M")
    print("=" * 120, flush=True)
    for m in methods:
        for e in ("S1", "S2", "S3"):
            run_cell(root, e, m, args)


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="60M Joint-first UAM discovery matrix V4")
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    p.add_argument("--eval-seeds", default=",".join(str(x) for x in DEFAULT_EVAL_SEEDS))
    p.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME)
    p.add_argument("--resume-root", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--stage", choices=["ALL", "J0", "J1", "J2", "S"], default="ALL")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--plan-only", action="store_true", help="Write/print static J0/J1 method catalog without training.")
    p.add_argument("--smoke-cell", default=None, help="Run exactly one cell, e.g. J0__CURRENT, then exit. Use --timesteps 50000 for a short smoke test.")
    return p.parse_args()


def write_manifest(root: Path, args: argparse.Namespace) -> None:
    manifest = {
        "experiment": "UAM_60M_JOINT_FIRST_DISCOVERY_V4",
        "created": datetime.now().isoformat(timespec="seconds"),
        "formal_cells": 100,
        "timesteps_per_cell": int(args.timesteps),
        "formal_budget": 100 * int(args.timesteps),
        "joint_cells": 67,
        "single_cells": 33,
        "single_to_joint_ratio": "33:67 ~= 1:2.03",
        "phase_J0": {"cells": 24, "budget": 24 * int(args.timesteps), "anchors": list(J0_ANCHORS)},
        "phase_J1": {"cells": 36, "budget": 36 * int(args.timesteps), "shared": list(SHARED_METHODS), "ac": list(AC_METHODS), "wm": list(WM_METHODS)},
        "phase_J2": {"cells": 7, "budget": 7 * int(args.timesteps)},
        "phase_S": {"cells": 33, "budget": 33 * int(args.timesteps), "runs_last": True},
        "joint_architectures": JOINT_ARCH_NAMES_V4,
        "train_seed": int(args.train_seed),
        "eval_seeds": v3.parse_ints(args.eval_seeds),
        "device": "cuda",
        "profile": asdict(FAST_PROFILE),
        "checkpoint_interval_nominal": CHECKPOINT_INTERVAL,
        "physics": {
            "topology": v3.TOPOLOGY,
            "fleet_size": v3.FLEET_SIZE,
            "demand_horizon_min": v3.DEMAND_HORIZON,
            "hard_guard_min": int(args.max_time),
            "charger_capacity": v3.CHARGER_CAPACITY,
            "pad_separation_joint_S3": v3.PAD_SEPARATION_MIN,
            "joint_hold_removed": True,
        },
        "selection": "Adaptive routing uses final-model selection_probe only; formal Joint 12-checkpoint evaluation is deferred until all 60M training cells finish.",
        "evaluation_schedule": "Joint: formal eval after all 100 training cells; Single: immediate per-cell eval",
        "literature_fidelity_note": "STYLE/INSPIRED labels are not verbatim reproductions unless explicitly stated.",
    }
    v3.write_json(root / "experiment_manifest.json", manifest)


def require_file(root: Path, name: str) -> Dict[str, Any]:
    p = root / name
    if not p.exists():
        raise FileNotFoundError(f"Required prerequisite selection file missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    v3.legacy.assert_p0_patch()
    install_v4_hooks()

    if not torch.cuda.is_available():
        raise RuntimeError("V4 production matrix requires CUDA; torch.cuda.is_available() is False")
    if int(args.timesteps) <= 0 or int(args.timesteps) % CHECKPOINT_INTERVAL != 0:
        raise ValueError(f"timesteps must be positive and divisible by {CHECKPOINT_INTERVAL}")
    if int(args.max_time) <= v3.DEMAND_HORIZON:
        raise ValueError("max-time must exceed demand horizon")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_root:
        root = Path(args.resume_root).expanduser().resolve()
    elif args.output_root:
        root = Path(args.output_root).expanduser().resolve()
    else:
        root = (ROOT / "serial_runs" / f"uam_jointfirst60m_v4_seed{args.train_seed}_{stamp}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    load_selection_state(root)
    write_manifest(root, args)

    print("=" * 120)
    print("UAM 60M JOINT-FIRST DISCOVERY MATRIX V4")
    print(f"root={root}")
    print(f"device=cuda | profile={FAST_PROFILE.name} | rollout={FAST_PROFILE.rollout:,}")
    print(f"timesteps/cell={args.timesteps:,} | total formal budget={100*int(args.timesteps):,}")
    print("ORDER = J0(train+probe) -> J1(train+probe) -> J2(train+probe) -> SINGLE(train+immediate eval) -> JOINT FORMAL EVAL")
    print("JOINT:SINGLE = 67:33 | SINGLE:JOINT ~= 1:2.03")
    print("=" * 120, flush=True)

    if args.plan_only:
        print("J0 anchors:", J0_ANCHORS)
        print("J1 Shared:", SHARED_METHODS)
        print("J1 AC:", AC_METHODS)
        print("J1 WM:", WM_METHODS)
        return 0

    if args.smoke_cell:
        cell = canonical(args.smoke_cell)
        if "__" not in cell:
            raise ValueError("--smoke-cell must look like J0__CURRENT")
        env_key, method_id = cell.split("__", 1)
        if env_key not in v3.ENV_SPECS:
            raise ValueError(f"Unknown smoke env_key: {env_key}")
        print(f"SMOKE CELL ONLY: {env_key}__{method_id}", flush=True)
        run_cell(root, env_key, method_id, args)
        print("SMOKE CELL COMPLETE", flush=True)
        return 0

    stage = args.stage.upper()

    # ALL intentionally always respects the causal order.
    if stage in {"ALL", "J0"}:
        top3 = run_phase_j0(root, args)
        if stage == "J0":
            return 0
    else:
        top3 = require_file(root, "selection_j0_top3.json")["top3"]

    if stage in {"ALL", "J1"}:
        j1sel = run_phase_j1(root, args, top3)
        if stage == "J1":
            return 0
    else:
        j1sel = require_file(root, "phase_j1_selection.json")

    if stage in {"ALL", "J2"}:
        j2sel = run_phase_j2(root, args, j1sel)
        if stage == "J2":
            return 0
    else:
        j2sel = require_file(root, "phase_j2_selection.json")
        # reconstruct combos before single-only resume
        for k, vals in require_file(root, "phase_j2_plan.json").get("combo_registry", {}).items():
            COMBO_REGISTRY[canonical(k)] = tuple(canonical(x) for x in vals)

    if stage in {"ALL", "S"}:
        run_phase_single(root, args, j1sel, j2sel)

        # User-requested schedule: formal Joint evaluation happens only now,
        # after all 100 training cells (67 Joint + 33 Single) have finished.
        joint_plan = collect_joint_plan(root)
        print("\n" + "#" * 120)
        print("ALL 60M TRAINING CELLS FINISHED -> STARTING ONE UNIFIED FORMAL JOINT EVALUATION")
        print("#" * 120, flush=True)
        v3.deferred_joint_evaluation(
            root=root,
            joint_plan=joint_plan,
            requested_steps=int(args.timesteps),
            eval_seeds=v3.parse_ints(args.eval_seeds),
            max_time=int(args.max_time),
            fail_fast=bool(args.fail_fast),
        )

    v3.build_master(root)
    v3.write_json(root / "RUN_COMPLETE.json", {
        "status": "COMPLETE",
        "formal_cells": 100,
        "joint_cells": 67,
        "single_cells": 33,
        "timesteps_per_cell": int(args.timesteps),
        "formal_requested_timesteps": 100 * int(args.timesteps),
        "joint_formal_eval_deferred_until_after_60m_training": True,
        "single_eval_immediate_per_cell": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    print("\nDONE")
    print(f"Master: {root / 'matrix_master.csv'}")
    print(f"Status: {root / 'serial_status.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
