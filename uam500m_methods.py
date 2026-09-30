# -*- coding: utf-8 -*-
"""500M 阶段的受控方法变体；不修改仿真物理、奖励或未来信息边界。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import train_uam_pre5b_80m_autofunnel as pre80

v3 = pre80.v3
v4 = pre80.v4
ng = pre80.ng


# 这里的别名只改变辅助训练，主表示仍与原对照完全相同。
VARIANT_COMPONENTS = {
    "S_EMA": ("S_TDM_EVENT_FUSION",),
    "S_MASKED_EVENT_EMA": ("S_TDM_EVENT_FUSION",),
    "AC_RELEVANT_ICM": ("A_ICM_AC",),
    "SA_ALTERNATE": ("S_TDM_EVENT_FUSION", "A_ICM_AC"),
    "SA_EMA_RELEVANT": ("S_TDM_EVENT_FUSION", "A_ICM_AC"),
    "SA_MASKED_RELEVANT": ("S_TDM_EVENT_FUSION", "A_ICM_AC"),
    "J4C_S": ("S_TDM_EVENT_FUSION",),
    "J4C_AC": ("A_ICM_AC",),
    "J4C_SA": ("S_TDM_EVENT_FUSION", "A_ICM_AC"),
}

EMA_METHODS = {"S_EMA", "S_MASKED_EVENT_EMA", "SA_EMA_RELEVANT", "SA_MASKED_RELEVANT"}
MASKED_METHODS = {"S_MASKED_EVENT_EMA", "SA_MASKED_RELEVANT"}
RELEVANT_METHODS = {"AC_RELEVANT_ICM", "SA_EMA_RELEVANT", "SA_MASKED_RELEVANT"}
AUX_METHODS = EMA_METHODS | RELEVANT_METHODS | {"SA_ALTERNATE"}


def method_name(model: PPO) -> str:
    ext = getattr(model.policy, "features_extractor", None)
    return str(getattr(ext, "method_id_v4", getattr(ext, "method_id", ""))).upper()


class RelevanceRecorder(BaseCallback):
    """记录执行后的物理生效标签，仅作为辅助损失监督，不作为在线观测。"""

    def _on_rollout_start(self) -> None:
        self.model._uam500m_relevance = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        self.model._uam500m_relevance.append([
            bool(info.get("v5_rl_choice_active", False))
            for info in infos
        ])
        return True


class ConditionalJ4Policy(v4.DiscoveryJointPolicy):
    """J4 的条件式 actor 对照；critic 与 PPO 更新仍沿用原 J4。"""

    def _joint_logits(self, latent: torch.Tensor) -> torch.Tensor:
        h = self.mappo_shared(latent)
        p_logp = F.log_softmax(self.mappo_p(h), dim=-1)
        gate = self.mappo_gate(h).reshape(-1, 2, 2)
        rows = []
        for passenger in range(2):
            one_hot = F.one_hot(
                torch.full((latent.shape[0],), passenger, dtype=torch.long, device=latent.device),
                num_classes=2,
            ).float()
            aircraft_logits = self.cond_a(torch.cat([latent, one_hot], dim=-1))
            aircraft_logp = F.log_softmax(aircraft_logits + 0.25 * gate[:, passenger], dim=-1)
            rows.append(p_logp[:, passenger:passenger + 1] + aircraft_logp)
        return torch.stack(rows, dim=1).reshape(-1, 4)


class StabilizedAuxPPO(v4.DiscoveryAuxPPO):
    """在原 PPO/ICM 上做可归因的辅助训练变体。"""

    def _relevance_icm(self, ext: Any) -> float | None:
        obs = np.asarray(self.rollout_buffer.observations)
        acts = np.asarray(self.rollout_buffer.actions).reshape(obs.shape[0], obs.shape[1], -1)[..., 0]
        starts = np.asarray(self.rollout_buffer.episode_starts)
        flags = np.asarray(getattr(self, "_uam500m_relevance", []), dtype=bool)
        if flags.shape != acts.shape:
            raise RuntimeError(f"V5 生效标签与 rollout 未对齐：{flags.shape} != {acts.shape}")
        pairs = [(t, e) for t in range(obs.shape[0] - 1)
                 for e in range(obs.shape[1]) if starts[t + 1, e] <= 0.5]
        if not pairs:
            return None
        if len(pairs) > v4.AUX_MAX_SAMPLES:
            chosen = np.random.choice(len(pairs), v4.AUX_MAX_SAMPLES, replace=False)
            pairs = [pairs[int(i)] for i in chosen]
        cur = torch.as_tensor(np.stack([obs[t, e] for t, e in pairs]), device=self.device, dtype=torch.float32)
        nxt = torch.as_tensor(np.stack([obs[t + 1, e] for t, e in pairs]), device=self.device, dtype=torch.float32)
        act = torch.as_tensor([acts[t, e] for t, e in pairs], device=self.device, dtype=torch.long)
        active = torch.as_tensor([flags[t, e] for t, e in pairs], device=self.device, dtype=torch.bool)
        opt = self._optimizer_for("A_ICM_AC", ext)
        if opt is None:
            return None
        losses = []
        for ix in self._run_batches(len(pairs)):
            z0, z1 = ext.icm_z(cur[ix]), ext.icm_z(nxt[ix])
            logits = ext.icm_inverse(torch.cat([z0, z1], dim=-1))
            if logits.shape[1] == 4:
                full = F.cross_entropy(logits, act[ix], reduction="none")
                passenger_logp = torch.logsumexp(F.log_softmax(logits, dim=-1).reshape(-1, 2, 2), dim=-1)
                passenger = F.nll_loss(passenger_logp, act[ix] // 2, reduction="none")
                loss = torch.where(active[ix], full, passenger).mean()
            else:
                loss = F.cross_entropy(logits, act[ix])
            opt.zero_grad(set_to_none=True)
            self.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ext.aux_parameters_for("A_ICM_AC"), 5.0)
            opt.step()
            self.policy.optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        self.logger.record("train/uam500m_aircraft_relevance_rate", float(active.float().mean().cpu()))
        return float(np.mean(losses))

    def _train_component(self, comp: str, ext: Any) -> float | None:
        name = method_name(self)
        if comp.upper() == "A_ICM_AC" and name in RELEVANT_METHODS:
            return self._relevance_icm(ext)
        if comp.upper() == "A_ICM_AC" and name == "SA_ALTERNATE":
            # 交替 rollout 更新辅助头，检验辅助梯度持续干扰的假设。
            self._uam500m_aux_round = getattr(self, "_uam500m_aux_round", 0) + 1
            if self._uam500m_aux_round % 2 == 0:
                return None
        return super()._train_component(comp, ext)

    def _masked_observation(self, obs: torch.Tensor, ext: Any) -> torch.Tensor:
        masked = obs.clone()
        lo, hi = ext.slices["event"]
        event = masked[:, int(lo):int(hi)].reshape(
            -1, v3.N_CANDIDATES, 4, v3.MAX_EVENTS_PER_TYPE
        )
        drop = torch.rand(
            (event.shape[0], v3.N_CANDIDATES, 2, v3.MAX_EVENTS_PER_TYPE),
            device=event.device,
        ) < 0.15
        event[:, :, 0].masked_fill_(drop[:, :, 0], 0.0)
        event[:, :, 1].masked_fill_(drop[:, :, 0], 0.0)
        event[:, :, 2].masked_fill_(drop[:, :, 1], 0.0)
        event[:, :, 3].masked_fill_(drop[:, :, 1], 0.0)
        return masked

    def _temporal_ema_aux(self) -> None:
        name = method_name(self)
        ext = self._ext()
        if ext is None:
            raise RuntimeError("EMA 辅助训练要求 DiscoveryExtractor 子类")
        teacher = self._target_policy()
        teacher_ext = teacher.features_extractor
        obs_np = np.asarray(self.rollout_buffer.observations).reshape(-1, self.observation_space.shape[0])
        count = min(2048, len(obs_np))
        ix = np.random.choice(len(obs_np), count, replace=False)
        obs = torch.as_tensor(obs_np[ix], device=self.device, dtype=torch.float32)
        student_obs = self._masked_observation(obs, ext) if name in MASKED_METHODS else obs
        if not hasattr(self, "_uam500m_ema_optimizer"):
            modules = [ext.tdm_branch, ext.event_phi, ext.event_query, ext.event_out, ext.tdm_event_fuse]
            params = [p for module in modules for p in module.parameters()]
            self._uam500m_ema_optimizer = torch.optim.Adam(params, lr=1e-5)
            self._uam500m_ema_params = params
        with torch.no_grad():
            target = teacher_ext.component_latent(obs, "S_TDM_EVENT_FUSION")
        student = ext.component_latent(student_obs, "S_TDM_EVENT_FUSION")
        loss = 0.05 * F.mse_loss(student, target)
        if name in MASKED_METHODS:
            # 方差下限仅防止辅助潜变量塌缩，不改变真实 ATT 奖励。
            std = torch.sqrt(student.var(dim=0, unbiased=False) + 1e-4)
            loss = loss + 0.001 * F.relu(0.1 - std).mean()
        self._uam500m_ema_optimizer.zero_grad(set_to_none=True)
        self.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._uam500m_ema_params, 1.0)
        self._uam500m_ema_optimizer.step()
        self.policy.optimizer.zero_grad(set_to_none=True)
        self.logger.record("train/uam500m_temporal_ema_loss", float(loss.detach().cpu()))

    def train(self) -> None:
        if method_name(self) in EMA_METHODS:
            self._temporal_ema_aux()
        super().train()


def install_hooks() -> None:
    """复用现有 V5、trace、PPO 接口，仅注册新的方法别名。"""
    pre80.install_master_hooks()
    for name, components in VARIANT_COMPONENTS.items():
        v4.COMBO_REGISTRY[name] = components
    original_builder = v3.build_model

    def build_model(*, env, env_key, method_id, profile, seed, run_dir, device):
        name = str(method_id).upper()
        if name not in VARIANT_COMPONENTS:
            return original_builder(env=env, env_key=env_key, method_id=name,
                                    profile=profile, seed=seed, run_dir=run_dir, device=device)
        spec = v3.ENV_SPECS[str(env_key).upper()]
        policy: Any = "MlpPolicy"
        policy_kwargs: dict[str, Any] = dict(
            features_extractor_class=ng.NextGenExtractor,
            features_extractor_kwargs=dict(features_dim=128, layout=v3.GLOBAL_LAYOUT,
                                           method_id=name, env_key=env_key),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )
        if spec.joint:
            policy = ConditionalJ4Policy if name.startswith("J4C_") else v4.DiscoveryJointPolicy
            policy_kwargs["joint_arch"] = str(env_key).upper()
        algorithm = StabilizedAuxPPO if name in AUX_METHODS else (
            v4.DiscoveryAuxPPO if v4.custom_method_needs_aux(name) else PPO
        )
        return algorithm(
            policy=policy, env=env,
            learning_rate=v3.base.linear_schedule(v3.INITIAL_LR),
            n_steps=int(profile.n_steps), batch_size=int(profile.batch_size),
            n_epochs=int(v3.N_EPOCHS), gamma=float(v3.GAMMA),
            gae_lambda=float(v3.GAE_LAMBDA), clip_range=float(v3.CLIP_RANGE),
            ent_coef=float(v3.ENT_COEF), vf_coef=float(v3.VF_COEF),
            max_grad_norm=float(v3.MAX_GRAD_NORM), policy_kwargs=policy_kwargs,
            seed=int(seed), verbose=1, device=device,
            tensorboard_log=str(run_dir / "tb"),
        )

    v3.build_model = build_model
