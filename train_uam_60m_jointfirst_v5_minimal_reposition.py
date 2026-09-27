# -*- coding: utf-8 -*-
"""
UAM 60M JOINT-FIRST V5 — CLEAN MINIMAL REPOSITION
=================================================

This file is a thin successor of:
    train_uam_60m_jointfirst_v4_deferred_joint_eval.py

It intentionally keeps the V4 literature-oriented 60M matrix, selection logic,
representations, actor/critic/world-model variants, PPO hyper-parameters and
formal evaluation pipeline, but fixes the Joint aircraft-control confound.

Single:
    passenger = PPO chooses V0/V1
    aircraft  = original Longest-Queue reposition

Joint V5:
    passenger = PPO chooses V0/V1
    aircraft  = PPO adds ONE V0/V1 reposition-target decision

Nothing else in aircraft dispatch is supposed to change:
    - dispatcher still runs at the original Scenario lifecycle position
    - same eligible-aircraft test
    - same multi-aircraft loop per simulator step
    - same pad / energy / charging / turnaround checks
    - same reward and S3 physics
    - no separate HOLD action
    - no manual aircraft dispatch before env.step()

Meaningful-choice gate:
    0 positive queues -> dispatch nobody
    1 positive queue  -> automatically use that queue (same as LQ)
    2 positive queues -> PPO target decides priority

If the PPO-selected target is depleted while additional eligible aircraft are
still available, the only remaining positive queue is served automatically.
Therefore the model changes target priority/allocation, not dispatch capacity.

Literature rationale (STYLE/INSPIRED, not verbatim reproductions):
    - Tavakoli et al.: action branching for multi-dimensional discrete control
    - Lowe et al.: coordinated actor-critic / centralized information
    - Foerster et al.: COMA counterfactual credit assignment
    - Lee et al.: Set Transformer for set-structured inputs
    - Huang & Ontanon: state-dependent action relevance / masking motivation

Default formal budget:
    100 cells x 600k = 60M requested transitions
    same V4 schedule: 24 J0 + 36 J1 + 7 J2 + 33 Single

Per-PPO production profile:
    40 env x 512 steps = 20,480 rollout
    batch = 4,096
    CUDA

Important:
    This file does NOT modify the old V3/V4 files.
    It cannot mathematically guarantee "never collapse"; it removes the known
    wrapper confounds and prevents the aircraft branch from physically choosing
    an empty destination when there is only one meaningful queue.
"""

from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import train_uam_60m_literature_matrix_v3_1 as v3
import train_uam_60m_jointfirst_v4_deferred_joint_eval as v4


ROOT = Path(__file__).resolve().parent
CANDIDATES = tuple(int(x) for x in v3.CANDIDATES)
DESTINATION = int(v3.DESTINATION)

if CANDIDATES != (0, 1):
    raise RuntimeError(
        f"V5 expects frozen T2 candidates (0,1), got {CANDIDATES}"
    )

# Production profile fixed by the speed/scaling experiment.
P40 = v3.SpeedProfile(
    "P40_40x512_b4096",
    n_envs=40,
    n_steps=512,
    batch_size=4096,
)
assert P40.rollout == 20_480


# =============================================================================
# CLEAN aircraft target-only dispatcher
# =============================================================================

def _clean_dispatch_fixed_returns(self) -> None:
    """
    Same Single LQ dispatch mechanism, except PPO supplies destination priority
    ONLY when both V0 and V1 currently have positive queues.
    """
    hub = str(
        getattr(self, "fixed_fleet_return_vertiport", DESTINATION)
    )

    local = list(
        self.vertiports.evtols_at_vertiport.get(hub, []) or []
    )

    available = [
        e for e in local
        if (
            v3.mx.state_name(e) == "IDLE"
            and not v3.mx.passenger_ids(e)
            and not v3.base.is_turnaround_busy(e)
        )
    ]
    available.sort(key=lambda e: str(getattr(e, "id", "")))

    requested_target = int(
        getattr(self, "_v5_aircraft_target", CANDIDATES[0])
    )
    if requested_target not in CANDIDATES:
        raise RuntimeError(f"invalid V5 target: {requested_target}")

    virtual_q = {
        int(v): int(v3.base.queue_length(self, int(v)))
        for v in CANDIDATES
    }

    stats = {
        "eligible_hub_count": len(available),
        "attempts": 0,
        "successes": 0,
        "blocked": 0,
        "rl_choice_active": False,
        "forced_single_target": False,
        "zero_total_queue": False,
        "requested_target": requested_target,
        "first_effective_target": -1,
    }

    if not available:
        self._v5_last_dispatch = stats
        return

    service_batch = int(
        getattr(v3.base, "SERVICE_BATCH_SIZE", 1)
    )

    for evtol in available:
        positive = [
            int(v)
            for v in CANDIDATES
            if virtual_q[int(v)] > 0
        ]

        if not positive:
            stats["zero_total_queue"] = True
            break

        if len(positive) == 1:
            # No genuine decision exists: preserve Single-like behavior.
            target = int(positive[0])
            stats["forced_single_target"] = True
        else:
            # Only here does the extra aircraft PPO decision affect physics.
            target = requested_target
            stats["rl_choice_active"] = True

        if stats["first_effective_target"] < 0:
            stats["first_effective_target"] = target

        stats["attempts"] += 1

        ok = bool(
            v3.base._start_empty_reposition_checked(
                scenario=self,
                evtol=evtol,
                origin=hub,
                destination=str(target),
            )
        )

        if not ok:
            # Preserve the original Single dispatcher stop condition.
            stats["blocked"] += 1
            break

        stats["successes"] += 1
        virtual_q[target] = max(
            0,
            virtual_q[target] - service_batch,
        )

    self._v5_last_dispatch = stats


class CleanMinimalJointWrapper(gym.Wrapper):
    """
    Discrete(4):
        pair = passenger_action * 2 + aircraft_action

        passenger_action: 0=V0, 1=V1
        aircraft_action : 0=V0, 1=V1

    The wrapper does NOT start an aircraft reposition.
    It only records the model's target, then calls the original env.step().
    """

    def __init__(self, env: gym.Env, env_key: str):
        super().__init__(env)
        self.env_key = str(env_key).upper()
        self.action_space = spaces.Discrete(4)
        self._install()

    def _scenario(self):
        return v3.mx.find_scenario(self.env)

    def _install(self) -> None:
        sc = self._scenario()
        sc._v5_aircraft_target = int(CANDIDATES[0])
        sc._v5_last_dispatch = {}
        sc._dispatch_fixed_returns = types.MethodType(
            _clean_dispatch_fixed_returns,
            sc,
        )

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._install()
        return out

    def step(self, action):
        idx = int(np.asarray(action).reshape(-1)[0])
        if idx < 0 or idx >= 4:
            raise ValueError(idx)

        passenger_action = idx // 2
        aircraft_action = idx % 2
        target = int(CANDIDATES[aircraft_action])

        sc = self._scenario()
        sc._v5_aircraft_target = target
        sc._v5_last_dispatch = {}

        # Critical V5 behavior:
        # aircraft is NOT dispatched here.
        out = self.env.step(passenger_action)

        stats = dict(
            getattr(sc, "_v5_last_dispatch", {}) or {}
        )

        info_extra = {
            "joint_action": idx,
            "passenger_action": int(passenger_action),
            "aircraft_action": int(aircraft_action),
            "aircraft_target": int(target),

            # compatibility with existing evaluator
            "aircraft_dispatched": bool(
                int(stats.get("successes", 0)) > 0
            ),
            "aircraft_eligible_hub_count": int(
                stats.get("eligible_hub_count", 0)
            ),

            # V5 diagnostics
            "v5_dispatch_attempts": int(
                stats.get("attempts", 0)
            ),
            "v5_dispatch_successes": int(
                stats.get("successes", 0)
            ),
            "v5_dispatch_blocked": int(
                stats.get("blocked", 0)
            ),
            "v5_rl_choice_active": bool(
                stats.get("rl_choice_active", False)
            ),
            "v5_forced_single_target": bool(
                stats.get("forced_single_target", False)
            ),
            "v5_zero_total_queue": bool(
                stats.get("zero_total_queue", False)
            ),
            "v5_first_effective_target": int(
                stats.get("first_effective_target", -1)
            ),
            "minimal_target_only_v5": True,
            "joint_hold_removed": False,
        }

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info)
            info.update(info_extra)
            return obs, reward, terminated, truncated, info

        obs, reward, done, info = out
        info = dict(info)
        info.update(info_extra)
        return obs, reward, done, info


# =============================================================================
# Environment factory: identical to V3.1 except Joint wrapper
# =============================================================================

def make_env_factory_v5(
    *,
    env_key: str,
    method_id: str,
    env_index: int,
    run_dir: Path,
    max_time: int,
):
    spec = v3.ENV_SPECS[str(env_key).upper()]

    def _init():
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

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

        if spec.joint:
            env = CleanMinimalJointWrapper(
                env,
                spec.key,
            )

        env = v3.LiteratureObservationWrapper(
            env,
            env_key=spec.key,
            method_id=method_id,
        )
        return env

    return _init


# =============================================================================
# Patch V4 scientific driver, not its old source file
# =============================================================================

_ORIGINAL_INSTALL_V4_HOOKS = v4.install_v4_hooks
_ORIGINAL_WRITE_MANIFEST = v4.write_manifest


def install_v5_hooks() -> None:
    # First install all V4 literature-method/model hooks.
    _ORIGINAL_INSTALL_V4_HOOKS()

    # Then replace ONLY the environment factory used by train + eval.
    v3.make_env_factory = make_env_factory_v5
    v3.NoHoldJointControlWrapper = CleanMinimalJointWrapper

    # Freeze per-PPO profile at P40.
    v4.FAST_PROFILE = P40


def write_manifest_v5(
    root: Path,
    args,
) -> None:
    # Let V4 write all its method/schedule metadata first.
    _ORIGINAL_WRITE_MANIFEST(root, args)

    p = root / "experiment_manifest.json"
    obj = json.loads(p.read_text(encoding="utf-8"))

    obj["experiment"] = (
        "UAM_60M_JOINT_FIRST_V5_CLEAN_MINIMAL_REPOSITION"
    )
    obj["created_v5"] = datetime.now().isoformat(
        timespec="seconds"
    )
    obj["profile"] = asdict(P40)

    obj["physics"]["joint_hold_removed"] = False
    obj["physics"]["manual_aircraft_dispatch_before_env_step"] = False
    obj["physics"]["one_aircraft_max_per_step"] = False

    obj["single_vs_joint_contract"] = {
        "single_passenger": "PPO V0/V1",
        "single_aircraft": "original Longest-Queue reposition",
        "joint_passenger": "PPO V0/V1",
        "joint_extra_aircraft_decision": (
            "V0/V1 reposition target priority only"
        ),
        "reward_changed": False,
        "physics_changed": False,
        "dispatch_lifecycle_changed": False,
        "eligible_aircraft_rule_changed": False,
        "dispatch_capacity_changed": False,
        "separate_hold_action": False,
    }

    obj["meaningful_choice_gate"] = {
        "0_positive_queues": "no reposition",
        "1_positive_queue": (
            "automatic only-positive target; aircraft PPO ignored"
        ),
        "2_positive_queues": (
            "aircraft PPO target decides priority"
        ),
        "purpose": (
            "reduce meaningless aircraft decisions and preserve "
            "Single-like circulation outside genuine choice states"
        ),
        "fidelity_note": (
            "physical relevance gating; not exact MaskablePPO"
        ),
    }

    obj["literature_basis_v5"] = {
        "action_branching": (
            "Tavakoli et al., Action Branching Architectures"
        ),
        "centralized_coordination": (
            "Lowe et al., Multi-Agent Actor-Critic"
        ),
        "counterfactual_credit": (
            "Foerster et al., COMA"
        ),
        "set_representation": (
            "Lee et al., Set Transformer"
        ),
        "action_relevance": (
            "Huang & Ontanon, Invalid Action Masking"
        ),
        "note": (
            "STYLE/INSPIRED labels are not verbatim reproductions."
        ),
    }

    p.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# Replace V4 entry hooks before calling its main().
v4.install_v4_hooks = install_v5_hooks
v4.write_manifest = write_manifest_v5
v4.FAST_PROFILE = P40


def inject_default_v5_output_root() -> None:
    """
    V4 otherwise names a fresh root with 'v4'.  Keep resume/output-root exactly
    as supplied by the user, but give new runs an unmistakable V5 folder.
    """
    argv = sys.argv[1:]
    has_root = (
        "--resume-root" in argv
        or "--output-root" in argv
    )
    if has_root:
        return

    # Do not interfere with plan-only if user wants no fixed root?  A plan-only
    # root is harmless and keeps behavior deterministic.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (
        ROOT
        / "serial_runs"
        / f"uam_jointfirst60m_v5_clean_seed1_{stamp}"
    )
    sys.argv.extend([
        "--output-root",
        str(out),
    ])


def main() -> int:
    inject_default_v5_output_root()

    print("=" * 120)
    print(
        "UAM 60M V5 | CLEAN MINIMAL JOINT REPOSITION"
    )
    print(
        f"P40 = {P40.n_envs} env x {P40.n_steps} "
        f"| rollout={P40.rollout:,} | batch={P40.batch_size}"
    )
    print(
        "Joint delta = passenger PPO + ONE aircraft V0/V1 target-priority choice"
    )
    print(
        "Old pre-step manual dispatch / one-aircraft cap / forced no-HOLD = REMOVED"
    )
    print(
        "Meaningful gate = aircraft PPO affects physics only when BOTH queues are positive"
    )
    print("=" * 120, flush=True)

    return v4.main()


if __name__ == "__main__":
    raise SystemExit(main())
