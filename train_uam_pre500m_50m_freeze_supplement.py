# -*- coding: utf-8 -*-
"""
PRE-500M 50M FREEZE SUPPLEMENT
===============================

Deep supplement to train_uam_pre5b_80m_autofunnel.py.

This is NOT a new method zoo.  It consumes ~50M requested PPO transitions to
freeze the remaining experiment-level confounders before the later ~500M method
campaign:

  F0  env-count speed sanity (10/20/40 env; never below 10)      0.5M
  F1  causal env-count comparison at fixed global rollout       10.8M
  F2  batch/update-density confirmation                          3.6M
  F3  analytical-headroom + load re-freeze                      5.4M
  F4  fresh UAGMC reference distribution                         1.2M
  F5  empirical demand-realization robustness / protocol         3.6M
  F6  J4 physical action-relevance audit                          8.1M
  F7  fully-frozen matched Single/Joint confirmation              9.6M
  F8  omitted Joint long-run confirmation (QPLEX + GATv2)         7.2M
                                                                 -----
  TOTAL                                                          50.0M

Important scientific boundaries
-------------------------------
1) F1 isolates n_envs while keeping global rollout fixed at 20,480 and batch
   fixed at 4,096:
       10 x 2048, 20 x 1024, 40 x 512.
   Thus it directly revisits the env-count question that was previously reduced
   mainly for wall-clock reasons.

2) F3 chooses the operating load using ONLY the source-style UAGMC baseline,
   six online-legal analytical/rule baselines, and stability/retention.  New
   methods do not choose the benchmark load.  CURRENT is a post-selection sanity
   diagnostic only.

3) F5 creates EMPIRICAL-RESAMPLED demand traces from the original passenger
   trace: same load/count and time support, but independently resampled O/D
   tuples and independently selected active time points.  This is a robustness
   bank; it must NOT be described as an exact reproduction of the original
   demand generator.

4) F6 audits the V5/J4 meaningful-choice gate by reading the actual simulator
   diagnostics (v5_rl_choice_active, forced_single_target, zero_total_queue,
   eligible hub count) during held-out evaluation.  It DOES NOT pretend to
   implement policy-gradient masking.  The final manifest explicitly separates
   physical action relevance from PPO credit semantics.

5) F7 is the paper-facing matched S/J confirmation after profile, load, demand
   protocol and J4 physical semantics have all been frozen.  Single and Joint
   runs with the same train seed always use the same demand trace.

Required files beside this script
---------------------------------
train_uam_pre5b_80m_autofunnel.py
train_uam_nextgen_100m_matrix.py
train_uam_60m_literature_matrix_v3_1.py
train_uam_60m_jointfirst_v5_minimal_reposition.py
run_uagmc_6env_6baseline_36runs.py
and their validated dependencies.

The completed 80M parent is required by default.  The usual parent is:
    serial_runs/uam_pre5b80m_MAIN

Run
---
CUDA_VISIBLE_DEVICES=0 python train_uam_pre500m_50m_freeze_supplement.py \
  --parent80-root serial_runs/uam_pre5b80m_MAIN

Resume
------
CUDA_VISIBLE_DEVICES=0 python train_uam_pre500m_50m_freeze_supplement.py \
  --parent80-root serial_runs/uam_pre5b80m_MAIN \
  --resume-root serial_runs/uam_pre500m_freeze50m_MAIN

Plan only
---------
python train_uam_pre500m_50m_freeze_supplement.py \
  --parent80-root serial_runs/uam_pre5b80m_MAIN --plan-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from stable_baselines3 import PPO

import train_uam_pre5b_80m_autofunnel as pre80
import run_uagmc_6env_6baseline_36runs as rulebase


v3 = pre80.v3
ng = pre80.ng
ROOT = Path(__file__).resolve().parent
MAX_TIME = int(pre80.MAX_TIME)

# =============================================================================
# EXACT GPU BUDGET
# =============================================================================

F0_BUDGET = 500_000
F1_BUDGET = 10_800_000
F2_BUDGET = 3_600_000
F3_BUDGET = 5_400_000
F4_BUDGET = 1_200_000
F5_BUDGET = 3_600_000
F6_BUDGET = 8_100_000
F7_BUDGET = 9_600_000
F8_BUDGET = 7_200_000
TOTAL_BUDGET = sum((
    F0_BUDGET, F1_BUDGET, F2_BUDGET, F3_BUDGET, F4_BUDGET,
    F5_BUDGET, F6_BUDGET, F7_BUDGET, F8_BUDGET,
))
assert TOTAL_BUDGET == 50_000_000

FAST_EVAL_SEEDS = (123,)
LONG_EVAL_SEEDS = (123, 124)

# =============================================================================
# PROFILES / METHODS
# =============================================================================

# Fixed global rollout = 20,480; only n_envs / per-env horizon changes.
P10 = v3.SpeedProfile("F1_P10_10x2048_b4096", 10, 2048, 4096)
P20 = v3.SpeedProfile("F1_P20_20x1024_b4096", 20, 1024, 4096)
P40 = v3.SpeedProfile("F1_P40_40x512_b4096", 40, 512, 4096)
ENV_PROFILES = (P10, P20, P40)

F1_METHOD_SPECS = (
    ("UAGMC_SOURCE", "S3"),
    ("S_TDM_EVENT_FUSION", "J4"),
    ("A_ANCHOR_QUOTIENT", "J4"),
)
F1_SEEDS = (101, 102, 103)
F1_STEPS = 400_000

LOADS = (0.80, 0.85, 0.90, 0.95, 1.00)
ANALYTICAL_METHODS = ("SPF", "STTF", "QTTI2", "CSM", "ECTF", "MPTC")
ANALYTICAL_SEEDS = (123, 124, 125)

# =============================================================================
# GENERIC HELPERS
# =============================================================================


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(np.asarray(x).reshape(-1)[0])
        return y if math.isfinite(y) else default
    except Exception:
        return default


def fmean(xs: Iterable[Any], default: float = float("nan")) -> float:
    vals = [fnum(x) for x in xs]
    vals = [x for x in vals if finite(x)]
    return float(np.mean(vals)) if vals else default


def fstd(xs: Iterable[Any], default: float = float("nan")) -> float:
    vals = [fnum(x) for x in xs]
    vals = [x for x in vals if finite(x)]
    return float(np.std(vals, ddof=0)) if vals else default


def fmedian(xs: Iterable[Any], default: float = float("nan")) -> float:
    vals = [fnum(x) for x in xs]
    vals = [x for x in vals if finite(x)]
    return float(np.median(vals)) if vals else default


def write_json(path: Path, obj: Any) -> None:
    pre80.write_json(Path(path), obj)


def read_json(path: Path, default: Any = None) -> Any:
    return pre80.read_json(Path(path), default)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    pre80.write_csv(Path(path), rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    return pre80.read_csv(Path(path))


def banner(text: str) -> None:
    print("\n" + "=" * 124)
    print(text)
    print("=" * 124, flush=True)


def profile_dict(profile) -> Dict[str, Any]:
    return {
        "name": str(profile.name),
        "n_envs": int(profile.n_envs),
        "n_steps": int(profile.n_steps),
        "batch_size": int(profile.batch_size),
        "rollout": int(profile.rollout),
    }


def profile_from_dict(d: Dict[str, Any]):
    return v3.SpeedProfile(
        str(d["name"]),
        int(d["n_envs"]),
        int(d["n_steps"]),
        int(d["batch_size"]),
    )


def load_key(rho: float) -> str:
    return f"{int(round(float(rho) * 100)):03d}"


def resolved(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def phase_done(path: Path) -> bool:
    return Path(path).exists()


def require_file(path: Path, why: str = "") -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        extra = f" ({why})" if why else ""
        raise FileNotFoundError(f"missing required file: {p}{extra}")
    return p


def cell_run_dir(
    *, root: Path, phase: str, tag: str, env_key: str, method: str, seed: int,
) -> Path:
    cell_root = root / phase / pre80.safe_slug(tag) / f"seed{int(seed)}"
    return cell_root / v3.cell_id(env_key, pre80.canonical(method))


def run_cell_cached(
    *, root: Path, phase: str, tag: str, env_key: str, method: str,
    seed: int, timesteps: int, profile, trace_path: str | Path,
    device: str, eval_seeds: Sequence[int], long_run: bool, fail_fast: bool,
) -> Dict[str, Any]:
    run_dir = cell_run_dir(
        root=root, phase=phase, tag=tag, env_key=env_key,
        method=method, seed=seed,
    )
    meta = run_dir / "analysis" / "pre5b_cell_meta.json"
    old = read_json(meta, None)
    if isinstance(old, dict):
        same = (
            int(old.get("timesteps", -1)) == int(timesteps)
            and str(old.get("profile", "")) == str(profile.name)
            and resolved(old.get("trace_path", trace_path)) == resolved(trace_path)
        )
        avail = pre80.available_checkpoint_steps(run_dir)
        complete = bool(avail) and max(avail) >= int(timesteps)
        if same and complete:
            print(
                f"[CACHE] {phase}/{tag}/{env_key}/{method}/seed{seed} "
                f"@ {max(avail):,}", flush=True,
            )
            return old

    return pre80.run_cell(
        root=root,
        phase=phase,
        tag=tag,
        env_key=env_key,
        method=method,
        seed=int(seed),
        timesteps=int(timesteps),
        profile=profile,
        trace_path=trace_path,
        device=device,
        eval_seeds=tuple(eval_seeds),
        long_run=bool(long_run),
        fail_fast=bool(fail_fast),
    )


def aggregate(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    return pre80.aggregate_cells(rows, group_keys=tuple(keys))


def add_profile_fields(row: Dict[str, Any], profile) -> Dict[str, Any]:
    out = dict(row)
    out["profile"] = profile.name
    out["n_envs"] = int(profile.n_envs)
    out["n_steps"] = int(profile.n_steps)
    out["batch_size"] = int(profile.batch_size)
    return out


def score_profile_summaries(
    agg_rows: Sequence[Dict[str, Any]],
    candidate_names: Sequence[str],
    method_names: Sequence[str],
) -> List[Dict[str, Any]]:
    rows = list(agg_rows)
    method_min: Dict[str, float] = {}
    for m in method_names:
        vals = [
            fnum(r.get("final_mean"), float("inf"))
            for r in rows
            if r.get("method") == m and r.get("profile") in candidate_names
        ]
        vals = [x for x in vals if finite(x) and x > 0]
        method_min[m] = min(vals) if vals else float("inf")

    speed_by_profile: Dict[str, float] = {}
    for p_name in candidate_names:
        speed_by_profile[p_name] = fmedian(
            r.get("observed_sps_median")
            for r in rows if r.get("profile") == p_name
        )
    max_speed = max(
        [x for x in speed_by_profile.values() if finite(x) and x > 0] or [1.0]
    )

    ranking: List[Dict[str, Any]] = []
    for p_name in candidate_names:
        per_method = []
        invalid = 0
        for m in method_names:
            candidates = [
                r for r in rows
                if r.get("profile") == p_name and r.get("method") == m
            ]
            if not candidates:
                invalid += 1
                continue
            r = candidates[0]
            final = fnum(r.get("final_mean"), float("inf"))
            ref = method_min.get(m, float("inf"))
            perf_gap = (
                max(0.0, final / ref - 1.0)
                if finite(final) and finite(ref) and ref > 0 else 9.0
            )
            cv = max(0.0, fnum(r.get("final_cv"), 9.0))
            collapse = max(0.0, fnum(r.get("collapse_ratio"), 9.0))
            full_valid = int(r.get("n_valid", 0)) == int(r.get("n_planned", 0))
            if not full_valid:
                invalid += 1
            per_method.append({
                "method": m,
                "final": final,
                "perf_gap": perf_gap,
                "cv": cv,
                "collapse_ratio": collapse,
                "quality": perf_gap + cv + collapse,
                "full_valid": full_valid,
            })

        quality = fmedian((x["quality"] for x in per_method), 99.0)
        speed = speed_by_profile.get(p_name, float("nan"))
        speed_penalty = (
            max(0.0, max_speed / speed - 1.0)
            if finite(speed) and speed > 0 else 9.0
        )
        # Quality dominates; speed acts only as a small soft penalty.
        score = 10.0 * invalid + quality + 0.03 * speed_penalty
        ranking.append({
            "profile": p_name,
            "quality_score": quality,
            "speed_sps": speed,
            "speed_penalty": speed_penalty,
            "invalid_method_count": invalid,
            "score": score,
            "per_method": per_method,
        })

    ranking.sort(key=lambda r: (fnum(r.get("score"), 999.0), -fnum(r.get("speed_sps"), 0.0)))
    return ranking


def set_trace_everywhere(path: str | Path) -> None:
    p = Path(path).expanduser().resolve()
    pre80.set_trace(p)
    for mod in (
        rulebase,
        getattr(rulebase, "old", None),
        getattr(rulebase, "core", None),
    ):
        if mod is not None and hasattr(mod, "TRAIN_FILE"):
            try:
                setattr(mod, "TRAIN_FILE", p)
            except Exception:
                pass


# =============================================================================
# 80M PARENT / TRACE MAP
# =============================================================================


def discover_parent80(explicit: str = "") -> Path:
    if explicit:
        return require_file(Path(explicit), "--parent80-root")

    direct = ROOT / "serial_runs" / "uam_pre5b80m_MAIN"
    if direct.exists():
        return direct.resolve()

    candidates = sorted(
        (ROOT / "serial_runs").glob("uam_pre5b80m_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("cannot discover an 80M parent run")
    return candidates[0].resolve()


def load_parent_contract(parent: Path, allow_incomplete: bool) -> Dict[str, Any]:
    p1 = require_file(parent / "P1_PROFILE" / "PROFILE_SELECTED.json")
    p2 = require_file(parent / "P2_LOAD" / "ENV_SELECTED.json")
    p3 = require_file(parent / "P3_JOINT" / "JOINT_SELECTED.json")
    p6 = parent / "P6_CONFIRM" / "PRE5B_FINAL.json"
    if not p6.exists() and not allow_incomplete:
        raise RuntimeError(
            f"80M parent is not fully complete: missing {p6}. "
            "Finish the 80M run first or pass --allow-parent-incomplete explicitly."
        )

    return {
        "parent_root": str(parent),
        "p1": read_json(p1, {}),
        "p2": read_json(p2, {}),
        "p3": read_json(p3, {}),
        "p6": read_json(p6, {}) if p6.exists() else {},
        "parent_complete": p6.exists(),
    }


def load_trace_map(parent: Path, local_root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rho in LOADS:
        k = load_key(rho)
        p = parent / "trace_bank" / f"passengers_load_{k}.csv"
        if p.exists():
            out[k] = str(p.resolve())
    if len(out) == len(LOADS):
        return out
    # Deterministic fallback identical to the 80M construction.
    return pre80.make_nested_load_traces(local_root / "trace_bank_nested")


# =============================================================================
# HELD-OUT EVALUATION HELPERS
# =============================================================================


def final_checkpoint(run_dir: Path) -> Tuple[int, Path, Path]:
    steps = pre80.available_checkpoint_steps(run_dir)
    if not steps:
        raise RuntimeError(f"no complete checkpoint: {run_dir}")
    step = int(max(steps))
    model_path, vec_path = v3.checkpoint_paths(run_dir, step)
    return step, model_path, vec_path


def eval_final_on_trace_cached(
    *, run_dir: Path, env_key: str, method: str, trace_path: str | Path,
    eval_seed: int = 123, label: str = "heldout",
) -> Dict[str, Any]:
    trace_path = Path(trace_path).resolve()
    step, model_path, vec_path = final_checkpoint(run_dir)
    cache_dir = run_dir / "analysis" / label
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{trace_path.stem}__seed{int(eval_seed)}.json"
    old = read_json(cache, None)
    if isinstance(old, dict) and int(old.get("train_step", -1)) == step:
        return old

    set_trace_everywhere(trace_path)
    eval_root = cache_dir / f"_monitor__{trace_path.stem}__seed{int(eval_seed)}"
    row = v3.evaluate_checkpoint(
        env_key=env_key,
        method_id=pre80.canonical(method),
        model_path=model_path,
        vec_path=vec_path,
        train_step=step,
        eval_seed=int(eval_seed),
        run_dir=eval_root,
        max_time=MAX_TIME,
    )
    row = dict(row)
    row["trace_path"] = str(trace_path)
    row["trace_name"] = trace_path.name
    write_json(cache, row)
    return row


def policy_probs(model: PPO, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor).distribution
        if getattr(dist, "probs", None) is not None:
            p = dist.probs
        else:
            p = torch.softmax(dist.logits, dim=-1)
        return np.asarray(p.detach().cpu().numpy(), dtype=float).reshape(-1)


def entropy_from_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum()) if len(p) else float("nan")


def relevance_eval_cached(
    *, run_dir: Path, env_key: str, method: str, trace_path: str | Path,
    eval_seed: int = 123,
) -> Dict[str, Any]:
    trace_path = Path(trace_path).resolve()
    step, model_path, vec_path = final_checkpoint(run_dir)
    cache_dir = run_dir / "analysis" / "joint_relevance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{trace_path.stem}__seed{int(eval_seed)}.json"
    old = read_json(cache, None)
    if isinstance(old, dict) and int(old.get("train_step", -1)) == step:
        return old

    set_trace_everywhere(trace_path)
    try:
        v3.seed_all(int(eval_seed))
    except Exception:
        pass

    raw = v3.build_eval_env(
        env_key=env_key,
        method_id=pre80.canonical(method),
        run_dir=cache_dir / f"_monitor__{trace_path.stem}",
        max_time=MAX_TIME,
    )
    env = v3.TauPreservingVecNormalize.load(str(vec_path), raw)
    env.training = False
    env.norm_reward = False

    try:
        model = PPO.load(str(model_path), env=env, device="cpu")
        model.policy.set_training_mode(False)
        try:
            env.seed(int(eval_seed))
        except Exception:
            pass

        obs = env.reset()
        done = np.asarray([False])
        steps = 0
        reward_sum = 0.0
        system_person_minutes = 0.0
        action_counts: Counter = Counter()
        terminal_snapshot = None

        choice_active = 0
        forced_single = 0
        zero_queue = 0
        eligible = 0
        dispatched = 0
        blocked = 0
        attempts = 0
        pair_entropy: List[float] = []
        passenger_entropy: List[float] = []
        aircraft_entropy: List[float] = []
        aircraft_entropy_active: List[float] = []

        while not bool(done[0]):
            scenario = v3.mx.find_scenario(env)
            try:
                system_person_minutes += v3.legacy.active_person_count(scenario)
            except Exception:
                pass

            probs = policy_probs(model, obs)
            if probs.size == 4:
                pair_entropy.append(entropy_from_probs(probs))
                pp = np.asarray([probs[0] + probs[1], probs[2] + probs[3]])
                aa = np.asarray([probs[0] + probs[2], probs[1] + probs[3]])
                passenger_entropy.append(entropy_from_probs(pp))
                aircraft_entropy.append(entropy_from_probs(aa))

            action, _ = model.predict(obs, deterministic=True)
            ai = int(np.asarray(action).reshape(-1)[0])
            action_counts[ai] += 1

            obs, reward, done, infos = env.step(action)
            reward_sum += fnum(np.asarray(reward).reshape(-1)[0], 0.0)
            steps += 1
            info = infos[0] if infos else {}
            if isinstance(info, dict):
                if "terminal_snapshot" in info:
                    terminal_snapshot = info["terminal_snapshot"]
                active = bool(info.get("v5_rl_choice_active", False))
                choice_active += int(active)
                forced_single += int(bool(info.get("v5_forced_single_target", False)))
                zero_queue += int(bool(info.get("v5_zero_total_queue", False)))
                eligible += int(int(info.get("aircraft_eligible_hub_count", 0)) > 0)
                dispatched += int(bool(info.get("aircraft_dispatched", False)) )
                blocked += int(int(info.get("v5_dispatch_blocked", 0)) > 0)
                attempts += int(info.get("v5_dispatch_attempts", 0) or 0)
                if active and probs.size == 4:
                    aa = np.asarray([probs[0] + probs[2], probs[1] + probs[3]])
                    aircraft_entropy_active.append(entropy_from_probs(aa))

            if steps > MAX_TIME + 100:
                raise RuntimeError("relevance evaluation exceeded hard guard")

        if terminal_snapshot is None:
            terminal_snapshot = v3.mx.snapshot_episode(v3.mx.find_scenario(env))

        metrics = v3.mx.metrics_from_terminal_snapshot(
            snapshot=terminal_snapshot,
            action_counts=action_counts,
            prob_rows=[],
            system_person_minutes=system_person_minutes,
            reward_sum=reward_sum,
            episode_steps=steps,
        )

        out = {
            **metrics,
            "train_step": step,
            "env_key": env_key,
            "method": pre80.canonical(method),
            "eval_seed": int(eval_seed),
            "trace_path": str(trace_path),
            "episode_steps": int(steps),
            "choice_active_count": int(choice_active),
            "choice_active_rate_all": choice_active / max(1, steps),
            "choice_active_rate_eligible": choice_active / max(1, eligible),
            "forced_single_rate_all": forced_single / max(1, steps),
            "zero_queue_rate_all": zero_queue / max(1, steps),
            "eligible_rate_all": eligible / max(1, steps),
            "dispatch_rate_all": dispatched / max(1, steps),
            "blocked_rate_all": blocked / max(1, steps),
            "dispatch_attempts_per_step": attempts / max(1, steps),
            "pair_entropy_mean": fmean(pair_entropy),
            "passenger_entropy_mean": fmean(passenger_entropy),
            "aircraft_entropy_mean": fmean(aircraft_entropy),
            "aircraft_entropy_when_active_mean": fmean(aircraft_entropy_active),
            "physical_semantics": "V5_CLEAN_MINIMAL_REPOSITION",
            "credit_semantics": "STANDARD_4WAY_PPO_NO_RELEVANCE_MASK",
        }
        write_json(cache, out)
        return out
    finally:
        try:
            env.close()
        except Exception:
            pass


# =============================================================================
# ANALYTICAL BASELINES FOR CURRENT S3
# =============================================================================


def analytical_one_cached(
    *, out_root: Path, trace_path: str | Path, load: float,
    method: str, seed: int,
) -> Dict[str, Any]:
    tag = f"LOAD_{load_key(load)}"
    run_dir = out_root / "analytical" / tag / f"{method}__seed{int(seed)}"
    result_file = run_dir / "result.json"
    old = read_json(result_file, None)
    if isinstance(old, dict):
        return old

    set_trace_everywhere(trace_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    row = rulebase.run_one(
        stage="E6",
        topology="T2",
        method=method,
        seed=int(seed),
        fleet_size=int(v3.FLEET_SIZE),
        pad_separation=float(v3.PAD_SEPARATION_MIN),
        charger_capacity=int(v3.CHARGER_CAPACITY),
        max_time=MAX_TIME,
        unknown_penalty=float(rulebase.DEFAULT_UNKNOWN_EVENT_PENALTY_MIN),
        run_dir=run_dir,
    )
    row = dict(row)
    row["load"] = float(load)
    row["trace_path"] = resolved(trace_path)
    row["physical_env"] = "S3/E6 exact current physics"
    write_json(result_file, row)
    return row


def aggregate_analytical(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[float, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(float(r["load"]), str(r["method"]))].append(r)
    out: List[Dict[str, Any]] = []
    for (rho, method), rs in groups.items():
        atts = [fnum(r.get("ATT")) for r in rs]
        comps = [fnum(r.get("completion_rate")) for r in rs]
        js = [fnum(r.get("system_person_minutes_per_passenger")) for r in rs]
        out.append({
            "load": rho,
            "method": method,
            "n": len(rs),
            "ATT_mean": fmean(atts),
            "ATT_std": fstd(atts),
            "completion_mean": fmean(comps),
            "JsysN_mean": fmean(js),
        })
    out.sort(key=lambda r: (float(r["load"]), str(r["method"])))
    return out


# =============================================================================
# EMPIRICAL-RESAMPLED TRACE BANK
# =============================================================================


def make_empirical_trace_bank(
    *, selected_trace: str | Path, out_dir: Path,
    n_train: int = 3, n_test: int = 4,
) -> Dict[str, Any]:
    manifest_path = out_dir / "empirical_trace_bank.json"
    old = read_json(manifest_path, None)
    if isinstance(old, dict):
        return old

    base_path = require_file(pre80.BASE_TRACE, "base passenger trace")
    selected_trace = require_file(Path(selected_trace), "selected-load trace")
    out_dir.mkdir(parents=True, exist_ok=True)

    with base_path.open("r", encoding="utf-8-sig", newline="") as f:
        base_rows = list(csv.DictReader(f))
    with selected_trace.open("r", encoding="utf-8-sig", newline="") as f:
        selected_rows = list(csv.DictReader(f))

    if not base_rows or not selected_rows:
        raise RuntimeError("empty passenger trace")
    required = {"id", "time", "origin_x", "origin_y", "dest_x", "dest_y"}
    if not required.issubset(base_rows[0].keys()):
        raise RuntimeError(
            f"unexpected passenger CSV columns: {list(base_rows[0].keys())}"
        )

    n_target = len(selected_rows)
    if n_target > len(base_rows):
        raise RuntimeError("selected trace larger than base trace")

    def build_one(kind: str, index: int, rng_seed: int) -> str:
        rng = np.random.default_rng(int(rng_seed))
        time_idx = np.sort(rng.choice(len(base_rows), size=n_target, replace=False))
        od_idx = rng.integers(0, len(base_rows), size=n_target)
        rows: List[Dict[str, Any]] = []
        for j, (ti, oi) in enumerate(zip(time_idx, od_idx)):
            trow = base_rows[int(ti)]
            orow = base_rows[int(oi)]
            rows.append({
                "id": f"{kind.upper()}{index}_P{j}",
                "time": trow["time"],
                "origin_x": orow["origin_x"],
                "origin_y": orow["origin_y"],
                "dest_x": orow["dest_x"],
                "dest_y": orow["dest_y"],
            })
        p = out_dir / f"passengers_{kind}_{index}.csv"
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["id", "time", "origin_x", "origin_y", "dest_x", "dest_y"],
            )
            w.writeheader()
            w.writerows(rows)
        return str(p.resolve())

    train = [build_one("train", i, 9100 + i) for i in range(n_train)]
    test = [build_one("test", i, 9900 + i) for i in range(n_test)]

    manifest = {
        "construction": (
            "empirical resampling from passengers_300.csv: exact selected-load row count; "
            "active time points sampled without replacement from base time support; O/D tuples "
            "sampled with replacement from empirical joint O/D rows. This is a robustness bank, "
            "NOT an exact reproduction of the original demand generator."
        ),
        "base_trace": str(base_path.resolve()),
        "selected_trace": str(selected_trace.resolve()),
        "selected_row_count": n_target,
        "base_row_count": len(base_rows),
        "train_traces": train,
        "test_traces": test,
    }
    write_json(manifest_path, manifest)
    return manifest


def trace_for_seed(
    *, protocol: str, seed: int, seed_order: Sequence[int],
    selected_trace: str, bank: Dict[str, Any],
) -> str:
    if protocol == "FIXED_TRACE":
        return str(Path(selected_trace).resolve())
    train = list(bank["train_traces"])
    if not train:
        raise RuntimeError("empty train trace bank")
    try:
        idx = list(seed_order).index(int(seed))
    except ValueError:
        idx = abs(int(seed)) % len(train)
    return str(Path(train[idx % len(train)]).resolve())


# =============================================================================
# F0 | SPEED SANITY
# =============================================================================


def run_f0(root: Path, trace_path: str, device: str, fail_fast: bool) -> Dict[str, Any]:
    phase = "F0_SPEED"
    out_path = root / phase / "F0_DONE.json"
    old = read_json(out_path, None)
    if isinstance(old, dict):
        return old

    jobs = [
        (P10, "UAGMC_SOURCE", "S3"),
        (P20, "UAGMC_SOURCE", "S3"),
        (P40, "UAGMC_SOURCE", "S3"),
        (P10, "S_TDM_EVENT_FUSION", "J4"),
        (P40, "S_TDM_EVENT_FUSION", "J4"),
    ]
    rows = []
    for profile, method, env_key in jobs:
        x = run_cell_cached(
            root=root, phase=phase, tag=profile.name, env_key=env_key,
            method=method, seed=91, timesteps=100_000, profile=profile,
            trace_path=trace_path, device=device, eval_seeds=FAST_EVAL_SEEDS,
            long_run=False, fail_fast=fail_fast,
        )
        rows.append(x)
        write_csv(root / phase / "cells.csv", rows)

    result = {
        "budget": F0_BUDGET,
        "purpose": "speed sanity only; no scientific selection",
        "rows": rows,
    }
    write_json(out_path, result)
    return result


# =============================================================================
# F1 | ENV COUNT FREEZE
# =============================================================================


def run_f1(root: Path, trace_path: str, device: str, fail_fast: bool) -> Dict[str, Any]:
    phase = "F1_ENVCOUNT"
    out_path = root / phase / "ENVCOUNT_SELECTED.json"
    old = read_json(out_path, None)
    if isinstance(old, dict):
        return old

    rows: List[Dict[str, Any]] = []
    for profile in ENV_PROFILES:
        for method, env_key in F1_METHOD_SPECS:
            for seed in F1_SEEDS:
                x = run_cell_cached(
                    root=root, phase=phase, tag=profile.name, env_key=env_key,
                    method=method, seed=seed, timesteps=F1_STEPS,
                    profile=profile, trace_path=trace_path, device=device,
                    eval_seeds=FAST_EVAL_SEEDS, long_run=False,
                    fail_fast=fail_fast,
                )
                x = add_profile_fields(x, profile)
                rows.append(x)
                write_csv(root / phase / "cells.csv", rows)

    agg = aggregate(rows, ("profile", "method"))
    write_csv(root / phase / "profile_method_summary.csv", agg)

    ranking = score_profile_summaries(
        agg,
        candidate_names=[p.name for p in ENV_PROFILES],
        method_names=[m for m, _ in F1_METHOD_SPECS],
    )
    by_name = {p.name: p for p in ENV_PROFILES}
    selected = by_name[str(ranking[0]["profile"])]

    result = {
        "budget": F1_BUDGET,
        "selected_profile": profile_dict(selected),
        "selection_rule": (
            "fixed global rollout=20,480 and batch=4,096; median across UAGMC/TDM/Anchor "
            "of normalized final gap + seed-CV + collapse ratio; small 0.03 throughput penalty"
        ),
        "ranking": ranking,
        "causal_isolation": {
            "global_rollout_fixed": 20_480,
            "batch_fixed": 4_096,
            "n_env_candidates": [10, 20, 40],
            "per_env_horizon": [2048, 1024, 512],
        },
    }
    write_json(out_path, result)
    return result


# =============================================================================
# F2 | BATCH / UPDATE DENSITY FREEZE
# =============================================================================


def run_f2(
    root: Path, trace_path: str, f1: Dict[str, Any], device: str, fail_fast: bool,
) -> Dict[str, Any]:
    phase = "F2_BATCH"
    out_path = root / phase / "PPO_PROFILE_FINAL.json"
    old = read_json(out_path, None)
    if isinstance(old, dict):
        return old

    base_profile = profile_from_dict(f1["selected_profile"])
    dense_profile = v3.SpeedProfile(
        f"F2_P{base_profile.n_envs}_x{base_profile.n_steps}_b1024",
        int(base_profile.n_envs), int(base_profile.n_steps), 1024,
    )

    dense_rows: List[Dict[str, Any]] = []
    for method, env_key in F1_METHOD_SPECS:
        for seed in F1_SEEDS:
            x = run_cell_cached(
                root=root, phase=phase, tag=dense_profile.name, env_key=env_key,
                method=method, seed=seed, timesteps=F1_STEPS,
                profile=dense_profile, trace_path=trace_path, device=device,
                eval_seeds=FAST_EVAL_SEEDS, long_run=False,
                fail_fast=fail_fast,
            )
            x = add_profile_fields(x, dense_profile)
            dense_rows.append(x)
            write_csv(root / phase / "dense_cells.csv", dense_rows)

    # Reuse F1 selected-env-count b4096 rows; no redundant GPU budget.
    f1_rows = read_csv(root / "F1_ENVCOUNT" / "cells.csv")
    base_rows = []
    for r in f1_rows:
        if str(r.get("profile")) == str(base_profile.name):
            x = dict(r)
            x["profile"] = base_profile.name
            base_rows.append(x)

    # aggregate_cells accepts numeric strings via fnum.
    combined = base_rows + dense_rows
    agg = aggregate(combined, ("profile", "method"))
    write_csv(root / phase / "batch_method_summary.csv", agg)

    ranking = score_profile_summaries(
        agg,
        candidate_names=[base_profile.name, dense_profile.name],
        method_names=[m for m, _ in F1_METHOD_SPECS],
    )
    by_name = {base_profile.name: base_profile, dense_profile.name: dense_profile}
    selected = by_name[str(ranking[0]["profile"])]

    result = {
        "budget_new_training": F2_BUDGET,
        "selected_profile": profile_dict(selected),
        "baseline_reused_from_F1": profile_dict(base_profile),
        "dense_candidate": profile_dict(dense_profile),
        "selection_rule": (
            "same n_envs/n_steps/global rollout; batch 4096 vs 1024 only; "
            "same quality-first score as F1"
        ),
        "ranking": ranking,
    }
    write_json(out_path, result)
    return result


# =============================================================================
# F3 | ANALYTICAL HEADROOM + LOAD RE-FREEZE
# =============================================================================


def run_f3(
    root: Path, trace_map: Dict[str, str], profile, device: str, fail_fast: bool,
) -> Dict[str, Any]:
    phase = "F3_LOAD"
    out_path = root / phase / "LOAD_FINAL.json"
    old = read_json(out_path, None)
    if isinstance(old, dict):
        return old

    phase_root = root / phase
    analytical_rows: List[Dict[str, Any]] = []
    for rho in LOADS:
        trace = trace_map[load_key(rho)]
        for method in ANALYTICAL_METHODS:
            for seed in ANALYTICAL_SEEDS:
                row = analytical_one_cached(
                    out_root=phase_root,
                    trace_path=trace,
                    load=rho,
                    method=method,
                    seed=seed,
                )
                analytical_rows.append(row)
                write_csv(phase_root / "analytical_raw.csv", analytical_rows)

    analytical_agg = aggregate_analytical(analytical_rows)
    write_csv(phase_root / "analytical_summary.csv", analytical_agg)

    uagmc_rows: List[Dict[str, Any]] = []
    for rho in LOADS:
        trace = trace_map[load_key(rho)]
        for seed in (201, 202, 203):
            x = run_cell_cached(
                root=root, phase=phase, tag=f"UAGMC_LOAD_{load_key(rho)}",
                env_key="S3", method="UAGMC_SOURCE", seed=seed,
                timesteps=300_000, profile=profile, trace_path=trace,
                device=device, eval_seeds=FAST_EVAL_SEEDS,
                long_run=False, fail_fast=fail_fast,
            )
            x["load"] = float(rho)
            uagmc_rows.append(x)
            write_csv(phase_root / "uagmc_cells.csv", uagmc_rows)

    uagg = aggregate(uagmc_rows, ("load", "method"))
    write_csv(phase_root / "uagmc_load_summary.csv", uagg)

    rankings: List[Dict[str, Any]] = []
    for rho in LOADS:
        ars = [r for r in analytical_agg if abs(float(r["load"]) - rho) < 1e-9]
        complete_ars = [
            r for r in ars
            if fnum(r.get("completion_mean"), 0.0) >= 0.98
            and finite(r.get("ATT_mean"))
        ]
        best_a = (
            min(complete_ars, key=lambda r: fnum(r.get("ATT_mean"), float("inf")))
            if complete_ars else None
        )
        urs = [r for r in uagg if abs(fnum(r.get("load")) - rho) < 1e-9]
        ur = urs[0] if urs else {}

        ua = fnum(ur.get("final_mean"), float("inf"))
        aa = fnum(best_a.get("ATT_mean"), float("inf")) if best_a else float("inf")
        headroom = (
            ua / aa - 1.0
            if finite(ua) and finite(aa) and aa > 0 else float("inf")
        )
        criticality = max(0.0, fnum(ur.get("final_cv"), 9.0)) + max(
            0.0, fnum(ur.get("collapse_ratio"), 9.0)
        )
        full_valid = int(ur.get("n_valid", 0)) == int(ur.get("n_planned", -1))
        penalty = 0.0 if (best_a is not None and full_valid) else 10.0
        score = (
            penalty
            + abs(headroom - 0.175)
            + 0.5 * abs(criticality - 0.20)
            - 0.01 * float(rho)
        )
        rankings.append({
            "load": float(rho),
            "analytical_best_method": best_a.get("method") if best_a else None,
            "analytical_best_ATT": aa,
            "analytical_best_completion": (
                best_a.get("completion_mean") if best_a else None
            ),
            "uagmc_final": ua,
            "uagmc_cv": ur.get("final_cv"),
            "uagmc_collapse_ratio": ur.get("collapse_ratio"),
            "headroom_ratio": headroom,
            "criticality": criticality,
            "target_headroom": 0.175,
            "target_criticality": 0.20,
            "score": score,
        })

    rankings.sort(key=lambda r: (fnum(r.get("score"), 999.0), -fnum(r.get("load"), 0.0)))
    selected_load = float(rankings[0]["load"])
    selected_trace = trace_map[load_key(selected_load)]
    write_csv(phase_root / "load_ranking.csv", rankings)

    # CURRENT sanity on top-3 UAGMC+analytical loads.  It does not select load.
    sanity_rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(rankings[:3]):
        rho = float(rec["load"])
        x = run_cell_cached(
            root=root, phase=phase, tag=f"CURRENT_SANITY_{load_key(rho)}",
            env_key="S3", method="CURRENT", seed=204 + i,
            timesteps=300_000, profile=profile,
            trace_path=trace_map[load_key(rho)], device=device,
            eval_seeds=FAST_EVAL_SEEDS, long_run=False,
            fail_fast=fail_fast,
        )
        x["load"] = rho
        sanity_rows.append(x)
        write_csv(phase_root / "current_sanity.csv", sanity_rows)

    result = {
        "budget_gpu": F3_BUDGET,
        "selected_load": selected_load,
        "selected_trace": resolved(selected_trace),
        "selection_rule": (
            "six online-legal analytical baselines + UAGMC_SOURCE only; score targets "
            "~17.5% analytical headroom and ~0.20 UAGMC seed-CV+collapse criticality; "
            "CURRENT is post-selection sanity only"
        ),
        "ranking": rankings,
        "analytical_methods": list(ANALYTICAL_METHODS),
        "current_sanity_does_not_select": True,
    }
    write_json(out_path, result)
    pre80.install_master_hooks()  # defensive reset after CPU baseline patch cleanup
    return result


# =============================================================================
# F4 | FRESH UAGMC REFERENCE
# =============================================================================


def run_f4(
    root: Path, profile, trace_path: str, device: str, fail_fast: bool,
) -> Dict[str, Any]:
    phase = "F4_UAGMC_REFERENCE"
    out_path = root / phase / "UAGMC_REFERENCE.json"
    old = read_json(out_path, None)
    if isinstance(old, dict):
        return old

    rows = []
    for seed in (211, 212, 213):
        x = run_cell_cached(
            root=root, phase=phase, tag="FROZEN_S3", env_key="S3",
            method="UAGMC_SOURCE", seed=seed, timesteps=400_000,
            profile=profile, trace_path=trace_path, device=device,
            eval_seeds=FAST_EVAL_SEEDS, long_run=False,
            fail_fast=fail_fast,
        )
        rows.append(x)
        write_csv(root / phase / "cells.csv", rows)
    agg = aggregate(rows, ("env_key", "method"))
    result = {
        "budget": F4_BUDGET,
        "fresh_seeds": [211, 212, 213],
        "summary": agg[0] if agg else {},
    }
    write_csv(root / phase / "summary.csv", agg)
    write_json(out_path, result)
    return result


# =============================================================================
# F5 | DEMAND REALIZATION PROTOCOL
# =============================================================================


def heldout_protocol_score(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r["protocol"]), str(r["method"]))].append(r)

    summaries = []
    for (protocol, method), rs in groups.items():
        valid = [
            r for r in rs
            if bool(r.get("valid_full_completion")) and finite(r.get("ATT"))
        ]
        atts = [fnum(r.get("ATT")) for r in valid]
        mean = fmean(atts, float("inf"))
        std = fstd(atts, float("inf"))
        summaries.append({
            "protocol": protocol,
            "method": method,
            "n_eval": len(rs),
            "n_valid": len(valid),
            "ATT_mean": mean,
            "ATT_std": std,
            "ATT_cv": std / mean if finite(std) and finite(mean) and mean > 0 else float("inf"),
        })

    method_min: Dict[str, float] = {}
    for m in sorted({str(r["method"]) for r in summaries}):
        vals = [
            fnum(r.get("ATT_mean"), float("inf"))
            for r in summaries if r["method"] == m
        ]
        vals = [x for x in vals if finite(x) and x > 0]
        method_min[m] = min(vals) if vals else float("inf")

    ranking = []
    for protocol in sorted({str(r["protocol"]) for r in summaries}):
        comps = []
        invalid = 0
        for r in [x for x in summaries if x["protocol"] == protocol]:
            ref = method_min.get(str(r["method"]), float("inf"))
            mean = fnum(r.get("ATT_mean"), float("inf"))
            gap = mean / ref - 1.0 if finite(mean) and finite(ref) and ref > 0 else 9.0
            cv = fnum(r.get("ATT_cv"), 9.0)
            valid_rate = int(r.get("n_valid", 0)) / max(1, int(r.get("n_eval", 0)))
            invalid += int(valid_rate < 1.0)
            comps.append(gap + 0.5 * cv)
        ranking.append({
            "protocol": protocol,
            "score": 10.0 * invalid + fmedian(comps, 99.0),
            "invalid_method_count": invalid,
        })
    ranking.sort(key=lambda r: fnum(r.get("score"), 999.0))
    return summaries + [{"_ranking": ranking}]


def run_f5(
    root: Path, profile, selected_trace: str, device: str, fail_fast: bool,
) -> Dict[str, Any]:
    phase = "F5_DEMAND"
    out_path = root / phase / "DEMAND_PROTOCOL_FINAL.json"
    old = read_json(out_path, None)
    if isinstance(old, dict):
        return old

    bank = make_empirical_trace_bank(
        selected_trace=selected_trace,
        out_dir=root / phase / "trace_bank",
    )
    methods = (("UAGMC_SOURCE", "S3"), ("S_TDM_EVENT_FUSION", "J4"))
    seeds = (301, 302, 303)
    train_rows = []
    cell_index: List[Dict[str, Any]] = []

    for protocol in ("FIXED_TRACE", "MULTI_REALIZATION_ACROSS_REPLICATES"):
        for method, env_key in methods:
            for seed in seeds:
                train_trace = trace_for_seed(
                    protocol=protocol,
                    seed=seed,
                    seed_order=seeds,
                    selected_trace=selected_trace,
                    bank=bank,
                )
                tag = f"{protocol}__{env_key}"
                x = run_cell_cached(
                    root=root, phase=phase, tag=tag, env_key=env_key,
                    method=method, seed=seed, timesteps=300_000,
                    profile=profile, trace_path=train_trace, device=device,
                    eval_seeds=FAST_EVAL_SEEDS, long_run=False,
                    fail_fast=fail_fast,
                )
                x["protocol"] = protocol
                x["train_trace"] = train_trace
                train_rows.append(x)
                cell_index.append({
                    "protocol": protocol, "method": method, "env_key": env_key,
                    "seed": seed, "tag": tag, "train_trace": train_trace,
                })
                write_csv(root / phase / "train_cells.csv", train_rows)

    heldout_rows: List[Dict[str, Any]] = []
    for c in cell_index:
        run_dir = cell_run_dir(
            root=root, phase=phase, tag=c["tag"], env_key=c["env_key"],
            method=c["method"], seed=int(c["seed"]),
        )
        for test_trace in bank["test_traces"]:
            row = eval_final_on_trace_cached(
                run_dir=run_dir,
                env_key=c["env_key"],
                method=c["method"],
                trace_path=test_trace,
                eval_seed=123,
                label="demand_heldout",
            )
            row = dict(row)
            row.update({
                "protocol": c["protocol"],
                "method": c["method"],
                "env_key": c["env_key"],
                "train_seed": c["seed"],
                "train_trace": c["train_trace"],
                "test_trace": test_trace,
            })
            heldout_rows.append(row)
            write_csv(root / phase / "heldout_eval.csv", heldout_rows)

    scored = heldout_protocol_score(heldout_rows)
    ranking = [x for x in scored if "_ranking" in x][0]["_ranking"]
    summaries = [x for x in scored if "_ranking" not in x]
    write_csv(root / phase / "heldout_protocol_summary.csv", summaries)
    write_csv(root / phase / "protocol_ranking.csv", ranking)

    selected_protocol = str(ranking[0]["protocol"])
    result = {
        "budget": F5_BUDGET,
        "selected_protocol": selected_protocol,
        "selected_trace": resolved(selected_trace),
        "trace_bank": bank,
        "ranking": ranking,
        "summary": summaries,
        "interpretation_boundary": (
            "empirical-resampled demand robustness; do not label as exact independent "
            "draws from the original unknown generator"
        ),
    }
    write_json(out_path, result)
    return result


# =============================================================================
# F6 | JOINT ACTION-RELEVANCE AUDIT
# =============================================================================


def run_f6(
    root: Path, profile, selected_trace: str, demand: Dict[str, Any],
    device: str, fail_fast: bool,
) -> Dict[str, Any]:
    phase = "F6_JOINT_RELEVANCE"
    out_path = root / phase / "JOINT_RELEVANCE_AUDIT.json"
    old = read_json(out_path, None)
    if isinstance(old, dict):
        return old

    protocol = str(demand["selected_protocol"])
    bank = demand["trace_bank"]
    seeds = (401, 402, 403)
    methods = ("CURRENT", "S_TDM_EVENT_FUSION", "A_QPLEX_DUPLEX")
    train_rows: List[Dict[str, Any]] = []
    cells = []
    for method in methods:
        for seed in seeds:
            train_trace = trace_for_seed(
                protocol=protocol, seed=seed, seed_order=seeds,
                selected_trace=selected_trace, bank=bank,
            )
            tag = f"J4_AUDIT__{method}"
            x = run_cell_cached(
                root=root, phase=phase, tag=tag, env_key="J4",
                method=method, seed=seed, timesteps=900_000,
                profile=profile, trace_path=train_trace, device=device,
                eval_seeds=LONG_EVAL_SEEDS, long_run=True,
                fail_fast=fail_fast,
            )
            x["train_trace"] = train_trace
            train_rows.append(x)
            cells.append({
                "method": method, "seed": seed, "tag": tag,
                "train_trace": train_trace,
            })
            write_csv(root / phase / "train_cells.csv", train_rows)

    audit_rows: List[Dict[str, Any]] = []
    audit_test_traces = list(bank["test_traces"])[:3]
    for c in cells:
        run_dir = cell_run_dir(
            root=root, phase=phase, tag=c["tag"], env_key="J4",
            method=c["method"], seed=int(c["seed"]),
        )
        for test_trace in audit_test_traces:
            row = relevance_eval_cached(
                run_dir=run_dir, env_key="J4", method=c["method"],
                trace_path=test_trace, eval_seed=123,
            )
            row = dict(row)
            row.update({
                "train_seed": c["seed"],
                "train_trace": c["train_trace"],
                "test_trace": test_trace,
            })
            audit_rows.append(row)
            write_csv(root / phase / "relevance_eval.csv", audit_rows)

    summaries = []
    for method in methods:
        rs = [r for r in audit_rows if r.get("method") == method]
        atts = [fnum(r.get("ATT")) for r in rs if finite(r.get("ATT"))]
        summaries.append({
            "method": method,
            "n_eval": len(rs),
            "ATT_mean": fmean(atts),
            "ATT_cv": (
                fstd(atts) / fmean(atts)
                if finite(fmean(atts)) and fmean(atts) > 0 else float("inf")
            ),
            "choice_active_rate_all": fmean(r.get("choice_active_rate_all") for r in rs),
            "choice_active_rate_eligible": fmean(r.get("choice_active_rate_eligible") for r in rs),
            "eligible_rate_all": fmean(r.get("eligible_rate_all") for r in rs),
            "forced_single_rate_all": fmean(r.get("forced_single_rate_all") for r in rs),
            "zero_queue_rate_all": fmean(r.get("zero_queue_rate_all") for r in rs),
            "aircraft_entropy_mean": fmean(r.get("aircraft_entropy_mean") for r in rs),
            "aircraft_entropy_when_active_mean": fmean(
                r.get("aircraft_entropy_when_active_mean") for r in rs
            ),
        })
    write_csv(root / phase / "relevance_summary.csv", summaries)

    overall_active = fmean((r.get("choice_active_rate_all") for r in audit_rows), 0.0)
    if overall_active >= 0.25:
        status = "PHYSICALLY_MATERIAL"
    elif overall_active >= 0.10:
        status = "SPARSE_BUT_PRESENT"
    else:
        status = "LOW_RELEVANCE_WARNING"

    result = {
        "budget": F6_BUDGET,
        "frozen_joint_env": "J4",
        "frozen_physical_semantics": "V5_CLEAN_MINIMAL_REPOSITION",
        "credit_semantics": "STANDARD_4WAY_PPO_NO_RELEVANCE_MASK",
        "overall_choice_active_rate_all": overall_active,
        "audit_status": status,
        "summary": summaries,
        "scientific_boundary": (
            "This phase measures when the aircraft branch changes physics. It does not "
            "modify PPO log-prob/entropy credit on physically irrelevant aircraft choices."
        ),
    }
    write_json(out_path, result)
    return result


# =============================================================================
# F7 | FULLY FROZEN MATCHED SINGLE / JOINT
# =============================================================================


def classify_sj(rows: Sequence[Dict[str, Any]], methods: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r["method"]), str(r["regime"]))].append(r)

    out = []
    for method in methods:
        stats = {}
        for regime in ("SINGLE", "JOINT"):
            rs = groups.get((method, regime), [])
            valid = [
                r for r in rs
                if bool(r.get("valid_full_completion")) and finite(r.get("ATT"))
            ]
            vals = [fnum(r.get("ATT")) for r in valid]
            mean = fmean(vals, float("inf"))
            std = fstd(vals, float("inf"))
            stats[regime] = {
                "mean": mean,
                "std": std,
                "cv": std / mean if finite(std) and finite(mean) and mean > 0 else float("inf"),
                "n_valid": len(valid),
                "n_eval": len(rs),
            }
        sf = stats["SINGLE"]["mean"]
        jf = stats["JOINT"]["mean"]
        delta = jf - sf
        ref = min(sf, jf)
        rel = delta / ref if finite(delta) and finite(ref) and ref > 0 else float("nan")
        max_cv = max(stats["SINGLE"]["cv"], stats["JOINT"]["cv"])
        incomplete = (
            stats["SINGLE"]["n_valid"] != stats["SINGLE"]["n_eval"]
            or stats["JOINT"]["n_valid"] != stats["JOINT"]["n_eval"]
        )
        if incomplete:
            cls = "INCOMPLETE"
        elif max_cv > 0.20:
            cls = "UNSTABLE_REVERSAL" if abs(rel) >= 0.10 else "UNSTABLE"
        elif rel <= -0.10:
            cls = "J_STRONG"
        elif rel >= 0.10:
            cls = "S_STRONG"
        else:
            cls = "CROSS_REGIME"
        out.append({
            "method": method,
            "single_heldout_ATT": sf,
            "joint_heldout_ATT": jf,
            "delta_J_minus_S": delta,
            "relative_delta": rel,
            "single_cv": stats["SINGLE"]["cv"],
            "joint_cv": stats["JOINT"]["cv"],
            "single_n_valid": stats["SINGLE"]["n_valid"],
            "joint_n_valid": stats["JOINT"]["n_valid"],
            "classification": cls,
        })
    return out


def run_f7(
    root: Path, profile, selected_trace: str, demand: Dict[str, Any],
    device: str, fail_fast: bool,
) -> Dict[str, Any]:
    phase = "F7_FORMAL_SJ"
    out_path = root / phase / "FORMAL_SJ_FINAL.json"
    old = read_json(out_path, None)
    if isinstance(old, dict):
        return old

    methods = (
        "S_TDM_EVENT_FUSION",
        "S_GATV2_EVENT",
        "A_ANCHOR_QUOTIENT",
        "SA_TDMFUSION_ICM",
    )
    seeds = (501, 502, 503)
    protocol = str(demand["selected_protocol"])
    bank = demand["trace_bank"]
    cells = []
    train_rows = []

    for method in methods:
        for seed in seeds:
            train_trace = trace_for_seed(
                protocol=protocol, seed=seed, seed_order=seeds,
                selected_trace=selected_trace, bank=bank,
            )
            for env_key, regime in (("S3", "SINGLE"), ("J4", "JOINT")):
                tag = f"{regime}__{method}"
                x = run_cell_cached(
                    root=root, phase=phase, tag=tag, env_key=env_key,
                    method=method, seed=seed, timesteps=400_000,
                    profile=profile, trace_path=train_trace, device=device,
                    eval_seeds=FAST_EVAL_SEEDS, long_run=False,
                    fail_fast=fail_fast,
                )
                x.update({"regime": regime, "train_trace": train_trace})
                train_rows.append(x)
                cells.append({
                    "method": method, "seed": seed, "env_key": env_key,
                    "regime": regime, "tag": tag, "train_trace": train_trace,
                })
                write_csv(root / phase / "train_cells.csv", train_rows)

    heldout_rows: List[Dict[str, Any]] = []
    test_traces = list(bank["test_traces"])[:3]
    for c in cells:
        run_dir = cell_run_dir(
            root=root, phase=phase, tag=c["tag"], env_key=c["env_key"],
            method=c["method"], seed=int(c["seed"]),
        )
        for test_trace in test_traces:
            row = eval_final_on_trace_cached(
                run_dir=run_dir, env_key=c["env_key"], method=c["method"],
                trace_path=test_trace, eval_seed=123, label="formal_sj_heldout",
            )
            row = dict(row)
            row.update({
                "method": c["method"],
                "train_seed": c["seed"],
                "regime": c["regime"],
                "train_trace": c["train_trace"],
                "test_trace": test_trace,
            })
            heldout_rows.append(row)
            write_csv(root / phase / "heldout_eval.csv", heldout_rows)

    classes = classify_sj(heldout_rows, methods)
    write_csv(root / phase / "FORMAL_SJ_CLASSIFICATION.csv", classes)
    result = {
        "budget": F7_BUDGET,
        "profile": profile_dict(profile),
        "demand_protocol": protocol,
        "joint_env": "J4",
        "classification_basis": "fresh 3 train seeds x 3 held-out empirical-resampled traces",
        "rows": classes,
    }
    write_json(out_path, result)
    return result


# =============================================================================
# F8 | QPLEX + GATV2 LONG-RUN CONFIRMATION
# =============================================================================


def run_f8(
    root: Path, profile, selected_trace: str, demand: Dict[str, Any],
    device: str, fail_fast: bool,
) -> Dict[str, Any]:
    phase = "F8_OMITTED_LONGRUN"
    out_path = root / phase / "LONGRUN_FINAL.json"
    old = read_json(out_path, None)
    if isinstance(old, dict):
        return old

    methods = ("A_QPLEX_DUPLEX", "S_GATV2_EVENT")
    seeds = (601, 602, 603)
    protocol = str(demand["selected_protocol"])
    bank = demand["trace_bank"]
    train_rows = []
    cells = []

    for method in methods:
        for seed in seeds:
            train_trace = trace_for_seed(
                protocol=protocol, seed=seed, seed_order=seeds,
                selected_trace=selected_trace, bank=bank,
            )
            tag = f"J4_LONGRUN__{method}"
            x = run_cell_cached(
                root=root, phase=phase, tag=tag, env_key="J4",
                method=method, seed=seed, timesteps=1_200_000,
                profile=profile, trace_path=train_trace, device=device,
                eval_seeds=LONG_EVAL_SEEDS, long_run=True,
                fail_fast=fail_fast,
            )
            x["train_trace"] = train_trace
            train_rows.append(x)
            cells.append({
                "method": method, "seed": seed, "tag": tag,
                "train_trace": train_trace,
            })
            write_csv(root / phase / "train_cells.csv", train_rows)

    heldout_rows: List[Dict[str, Any]] = []
    for c in cells:
        run_dir = cell_run_dir(
            root=root, phase=phase, tag=c["tag"], env_key="J4",
            method=c["method"], seed=int(c["seed"]),
        )
        for test_trace in bank["test_traces"]:
            row = eval_final_on_trace_cached(
                run_dir=run_dir, env_key="J4", method=c["method"],
                trace_path=test_trace, eval_seed=123, label="longrun_heldout",
            )
            row = dict(row)
            row.update({
                "method": c["method"],
                "train_seed": c["seed"],
                "train_trace": c["train_trace"],
                "test_trace": test_trace,
            })
            heldout_rows.append(row)
            write_csv(root / phase / "heldout_eval.csv", heldout_rows)

    train_agg = aggregate(train_rows, ("method",))
    heldout_summary = []
    for method in methods:
        rs = [r for r in heldout_rows if r.get("method") == method]
        valid = [r for r in rs if bool(r.get("valid_full_completion")) and finite(r.get("ATT"))]
        vals = [fnum(r.get("ATT")) for r in valid]
        mean = fmean(vals, float("inf"))
        std = fstd(vals, float("inf"))
        tr = [r for r in train_agg if r.get("method") == method]
        collapse = fnum(tr[0].get("collapse_ratio"), float("inf")) if tr else float("inf")
        heldout_summary.append({
            "method": method,
            "n_eval": len(rs),
            "n_valid": len(valid),
            "heldout_ATT_mean": mean,
            "heldout_ATT_std": std,
            "heldout_ATT_cv": std / mean if finite(std) and finite(mean) and mean > 0 else float("inf"),
            "heldout_worst_ATT": max(vals) if vals else float("inf"),
            "train_collapse_ratio": collapse,
        })
    heldout_summary.sort(key=lambda r: (
        fnum(r.get("heldout_ATT_mean"), float("inf")),
        fnum(r.get("heldout_ATT_cv"), float("inf")),
        fnum(r.get("train_collapse_ratio"), float("inf")),
    ))
    write_csv(root / phase / "longrun_heldout_ranking.csv", heldout_summary)

    result = {
        "budget": F8_BUDGET,
        "ranking": heldout_summary,
        "purpose": (
            "close two evidence gaps left by the 80M funnel: historical Joint QPLEX "
            "candidate and low-CV GATv2 Joint candidate"
        ),
    }
    write_json(out_path, result)
    return result


# =============================================================================
# FINAL FREEZE CONTRACT
# =============================================================================


def write_final_freeze(
    root: Path, parent_contract: Dict[str, Any], f1: Dict[str, Any],
    f2: Dict[str, Any], f3: Dict[str, Any], f4: Dict[str, Any],
    f5: Dict[str, Any], f6: Dict[str, Any], f7: Dict[str, Any],
    f8: Dict[str, Any],
) -> Dict[str, Any]:
    out_path = root / "PRE500M_FINAL_FREEZE.json"

    profile = f2["selected_profile"]
    warnings = []
    if str(f6.get("audit_status")) == "LOW_RELEVANCE_WARNING":
        warnings.append(
            "J4 aircraft branch is physically meaningful on a low fraction of decisions; "
            "standard PPO still assigns 4-way log-prob/entropy credit in irrelevant states."
        )
    warnings.append(
        "F5 traces are empirical-resampled robustness realizations, not exact draws from an "
        "independently verified original demand generator."
    )

    result = {
        "experiment": "PRE500M_50M_FREEZE_SUPPLEMENT",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "requested_gpu_transitions": TOTAL_BUDGET,
        "parent80": parent_contract,
        "frozen_ppo": {
            "profile_name": profile["name"],
            "n_envs": int(profile["n_envs"]),
            "n_steps": int(profile["n_steps"]),
            "global_rollout": int(profile["rollout"]),
            "batch_size": int(profile["batch_size"]),
            "n_epochs": int(v3.N_EPOCHS),
            "gamma": float(v3.GAMMA),
            "gae_lambda": float(v3.GAE_LAMBDA),
            "clip_range": float(v3.CLIP_RANGE),
            "initial_lr": float(v3.INITIAL_LR),
        },
        "frozen_environment": {
            "physics": "S3",
            "topology": str(v3.TOPOLOGY),
            "fleet_size": int(v3.FLEET_SIZE),
            "selected_load": float(f3["selected_load"]),
            "selected_trace": str(f3["selected_trace"]),
            "pad_separation_min": float(v3.PAD_SEPARATION_MIN),
            "charger_capacity": int(v3.CHARGER_CAPACITY),
            "hard_guard_min": MAX_TIME,
        },
        "demand_protocol": {
            "selected_protocol": f5["selected_protocol"],
            "train_trace_bank": f5["trace_bank"]["train_traces"],
            "heldout_trace_bank": f5["trace_bank"]["test_traces"],
            "boundary": f5["interpretation_boundary"],
        },
        "joint_contract": {
            "joint_env": "J4",
            "physical_semantics": f6["frozen_physical_semantics"],
            "credit_semantics": f6["credit_semantics"],
            "choice_active_rate_all": f6["overall_choice_active_rate_all"],
            "audit_status": f6["audit_status"],
        },
        "uagmc_reference": f4,
        "formal_single_joint": f7,
        "omitted_longrun_confirmation": f8,
        "500M_rule": (
            "Do not retune global PPO profile, S3 load, demand protocol or J4 physical "
            "semantics method-by-method. The 500M campaign should spend its budget on "
            "new literature-inspired methods and deeper versions of evidence-backed families."
        ),
        "warnings": warnings,
    }
    write_json(out_path, result)
    return result


# =============================================================================
# MANIFEST / CLI
# =============================================================================


def write_manifest(root: Path, parent: Path) -> None:
    write_json(root / "FREEZE50M_MANIFEST.json", {
        "experiment": "PRE500M_50M_FREEZE_SUPPLEMENT",
        "created": datetime.now().isoformat(timespec="seconds"),
        "parent80_root": str(parent),
        "budget": {
            "F0_SPEED": F0_BUDGET,
            "F1_ENVCOUNT": F1_BUDGET,
            "F2_BATCH": F2_BUDGET,
            "F3_LOAD": F3_BUDGET,
            "F4_UAGMC_REFERENCE": F4_BUDGET,
            "F5_DEMAND": F5_BUDGET,
            "F6_JOINT_RELEVANCE": F6_BUDGET,
            "F7_FORMAL_SJ": F7_BUDGET,
            "F8_OMITTED_LONGRUN": F8_BUDGET,
            "TOTAL": TOTAL_BUDGET,
        },
        "minimum_n_envs": 10,
        "env_count_candidates": [10, 20, 40],
        "global_rollout": 20_480,
        "no_method_specific_ppo_tuning": True,
    })


def print_plan(parent: Path) -> None:
    print("=" * 124)
    print("PRE-500M 50M FREEZE SUPPLEMENT")
    print(f"parent80 : {parent}")
    print(f"GPU budget: {TOTAL_BUDGET:,}")
    print("F0  speed sanity                         0.5M")
    print("F1  env-count 10/20/40                  10.8M")
    print("F2  batch 4096 vs 1024                   3.6M")
    print("F3  analytical headroom + load           5.4M")
    print("F4  fresh UAGMC reference                 1.2M")
    print("F5  demand-realization robustness         3.6M")
    print("F6  J4 relevance audit                    8.1M")
    print("F7  fully-frozen matched S/J              9.6M")
    print("F8  QPLEX + GATv2 long-run                7.2M")
    print("=" * 124)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pre-500M 50M freeze supplement")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--parent80-root", default="")
    ap.add_argument("--output-root", default="")
    ap.add_argument("--resume-root", default="")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--allow-parent-incomplete", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    # Registry sanity before expensive work.
    ng.install_nextgen_hooks()
    for m in (
        "CURRENT", "UAGMC_SOURCE", "S_TDM_EVENT_FUSION", "A_ANCHOR_QUOTIENT",
        "S_GATV2_EVENT", "SA_TDMFUSION_ICM", "A_QPLEX_DUPLEX",
    ):
        if m != "UAGMC_SOURCE":
            ng.method_description(m)
    pre80.install_master_hooks()

    parent = discover_parent80(args.parent80_root)
    parent_contract = load_parent_contract(parent, args.allow_parent_incomplete)
    print_plan(parent)
    if args.plan_only:
        print("[PLAN ONLY] no training started.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_root:
        root = Path(args.resume_root).expanduser().resolve()
    elif args.output_root:
        root = Path(args.output_root).expanduser().resolve()
    else:
        root = (ROOT / "serial_runs" / f"uam_pre500m_freeze50m_{stamp}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_manifest(root, parent)

    trace_map = load_trace_map(parent, root)
    inherited_trace = str(parent_contract["p2"]["selected_trace"])
    require_file(Path(inherited_trace), "80M selected trace")

    print(f"root      : {root}")
    print(f"device    : {args.device}")
    if torch.cuda.is_available():
        print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"80M load  : {parent_contract['p2'].get('selected_load')}")
    print(f"80M Joint : {parent_contract['p3'].get('selected_joint_env')}")
    print(flush=True)

    banner("F0 | SPEED SANITY")
    f0 = run_f0(root, inherited_trace, args.device, args.fail_fast)
    print("[INHERIT] F0 -> speed evidence only; F1 still compares all 10/20/40 env profiles")

    banner("F1 | ENV COUNT FREEZE")
    f1 = run_f1(root, inherited_trace, args.device, args.fail_fast)
    print(f"[INHERIT] F1 -> {f1['selected_profile']}", flush=True)

    banner("F2 | BATCH / UPDATE DENSITY FREEZE")
    f2 = run_f2(root, inherited_trace, f1, args.device, args.fail_fast)
    final_profile = profile_from_dict(f2["selected_profile"])
    print(f"[INHERIT] F2 -> {f2['selected_profile']}", flush=True)

    banner("F3 | ANALYTICAL HEADROOM + LOAD RE-FREEZE")
    f3 = run_f3(root, trace_map, final_profile, args.device, args.fail_fast)
    selected_trace = str(f3["selected_trace"])
    print(
        f"[INHERIT] F3 -> S3 load={f3['selected_load']:.2f} | {selected_trace}",
        flush=True,
    )

    banner("F4 | FRESH UAGMC REFERENCE")
    f4 = run_f4(root, final_profile, selected_trace, args.device, args.fail_fast)
    print("[INHERIT] F4 -> frozen UAGMC reference distribution", flush=True)

    banner("F5 | DEMAND REALIZATION ROBUSTNESS")
    f5 = run_f5(root, final_profile, selected_trace, args.device, args.fail_fast)
    print(f"[INHERIT] F5 -> {f5['selected_protocol']}", flush=True)

    banner("F6 | J4 PHYSICAL ACTION-RELEVANCE AUDIT")
    f6 = run_f6(
        root, final_profile, selected_trace, f5, args.device, args.fail_fast,
    )
    print(
        f"[INHERIT] F6 -> J4 V5 physical semantics frozen | audit={f6['audit_status']} | "
        f"active={f6['overall_choice_active_rate_all']:.3f}",
        flush=True,
    )

    banner("F7 | FULLY FROZEN MATCHED SINGLE / JOINT")
    f7 = run_f7(
        root, final_profile, selected_trace, f5, args.device, args.fail_fast,
    )
    print("[INHERIT] F7 -> formal frozen S/J classifications written", flush=True)

    banner("F8 | OMITTED JOINT LONG-RUN CONFIRMATION")
    f8 = run_f8(
        root, final_profile, selected_trace, f5, args.device, args.fail_fast,
    )

    final = write_final_freeze(
        root, parent_contract, f1, f2, f3, f4, f5, f6, f7, f8,
    )
    print("\n" + "=" * 124)
    print("PRE-500M FREEZE SUPPLEMENT COMPLETE")
    print(f"root  : {root}")
    print(f"final : {root / 'PRE500M_FINAL_FREEZE.json'}")
    print(f"profile: {final['frozen_ppo']['profile_name']}")
    print(f"load   : {final['frozen_environment']['selected_load']}")
    print(f"demand : {final['demand_protocol']['selected_protocol']}")
    print(f"joint  : {final['joint_contract']['joint_env']} / {final['joint_contract']['audit_status']}")
    print("=" * 124, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
