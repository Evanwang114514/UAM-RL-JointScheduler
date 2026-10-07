# -*- coding: utf-8 -*-
"""
UAM Stage-I -> Stage-II automatic freeze/baseline pipeline (V2)
================================================================

Purpose
-------
One command executes the two pre-method stages agreed for the UAM project:

Stage I  (about 14.7456M GPU transitions)
  0. P0 reward-contract source fix + runtime invariant audit.
  1. Freeze PPO batch size: 1024 vs 4096, fixed 10 x 2048 rollout.
  2. Freeze demand load: 0.80/0.85/0.90/0.95/1.00 using
     six CPU analytical baselines + UAGMC_SOURCE.
  3. Freeze demand protocol: FIXED_TRACE vs MULTI_REALIZATION_ACROSS_REPLICATES.
  4. Write UAM_FINAL_FREEZE_V2.json and SHA256 lock.

Stage II (84.48M GPU transitions)
  Exactly 11 matched RL cells, 5 fresh training seeds, 1.536M transitions each:
    SINGLE/S3:
      UAGMC_SOURCE, S_TDM_EVENT_FUSION, A_ICM_AC
    JOINT/J3:
      UAGMC_SOURCE, SHARED, S_TDM_EVENT_FUSION, A_ICM_AC
    JOINT/J4:
      UAGMC_SOURCE, SHARED, S_TDM_EVENT_FUSION, A_ICM_AC

  Stage II MUST consume the Stage-I freeze file created in this same suite.
  There is NO project-default fallback and NO method substitution.

Evaluation contract
-------------------
For every Stage-II trained model:
  1) absolute ATT: validation-bank LWATT over the final 20% checkpoints;
  2) within-run stability: dense 50k canonical-trace curve, best->late
     retention, late SD/slope, fixed-300k->late;
  3) across-seed stability: model-level LWATT Mean/SD/CV/Worst/Range.

CPU analytical baselines
------------------------
SPF, STTF, QTTI2, CSM, ECTF, MPTC are re-run under the *current locked runtime*
and frozen traces. Historical 85/165 numbers are never reused.

Important scientific boundary
-----------------------------
The Stage-II validation bank is still development/validation data, NOT the
untouched final-test bank. Final test remains deferred until Stage III method
selection is complete.

Typical usage
-------------
  # Inspect exact plan only (does not patch sources or train):
  python train_uam_stage12_freeze_baselines_v2.py --plan-only

  # Full automatic Stage I -> Stage II:
  python train_uam_stage12_freeze_baselines_v2.py --device cuda

  # Resume the exact suite after interruption:
  python train_uam_stage12_freeze_baselines_v2.py --device cuda \
      --resume-root serial_runs/uam_stage12_v2_YYYYMMDD_HHMMSS

  # Only finish Stage I, then stop:
  python train_uam_stage12_freeze_baselines_v2.py --stage1-only --device cuda

  # Continue Stage II from an already-complete Stage-I suite:
  python train_uam_stage12_freeze_baselines_v2.py --stage2-only --device cuda \
      --resume-root serial_runs/uam_stage12_v2_YYYYMMDD_HHMMSS

Notes
-----
- The script patches ONLY the newly identified reward-count bug:
  completed passenger IDs must never re-enter N_unfinished after their Person
  state changes from finished to removed. Backups use suffix .bak_stage12_v2.
- If the expected source layout is not recognized, the script stops instead of
  guessing a source edit.
- P0 patch is applied BEFORE importing any project training modules so spawned
  SubprocVecEnv workers observe the same source contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
ROLLOUT = 20_480
STAGE1_STEPS = 15 * ROLLOUT       # 307,200
STAGE2_STEPS = 75 * ROLLOUT       # 1,536,000
SAVE_EVERY = 50_000
HARD_GUARD = 2_500

LOADS = (0.80, 0.85, 0.90, 0.95, 1.00)
ANALYTICAL_METHODS = ("SPF", "STTF", "QTTI2", "CSM", "ECTF", "MPTC")

STAGE1_METHODS = (
    ("UAGMC_SOURCE", "S3"),
    ("S_TDM_EVENT_FUSION", "S3"),
    ("A_ICM_AC", "S3"),
)
BATCH_SEEDS = (101, 102, 103)
LOAD_BASE_SEEDS = (201, 202)
LOAD_CONFIRM_SEED = 203
DEMAND_SEEDS = (301, 302, 303)
ANALYTICAL_SEEDS = (123, 124, 125)
EVAL_SEEDS = (123, 124)
STAGE2_SEEDS = (601, 602, 603, 604, 605)

STAGE2_LANES = {
    "SINGLE_S3": (
        ("UAGMC_SOURCE", "S3"),
        ("S_TDM_EVENT_FUSION", "S3"),
        ("A_ICM_AC", "S3"),
    ),
    "JOINT_J3": (
        ("UAGMC_SOURCE", "J3"),
        ("SHARED", "J3"),
        ("S_TDM_EVENT_FUSION", "J3"),
        ("A_ICM_AC", "J3"),
    ),
    "JOINT_J4": (
        ("UAGMC_SOURCE", "J4"),
        ("SHARED", "J4"),
        ("S_TDM_EVENT_FUSION", "J4"),
        ("A_ICM_AC", "J4"),
    ),
}

EXPECTED_STAGE1_GPU = (
    2 * 3 * 3 * STAGE1_STEPS       # batch
    + (5 * 2 + 2) * STAGE1_STEPS  # load: 2 each + third seed top-2
    + 2 * 3 * 3 * STAGE1_STEPS    # demand
)
EXPECTED_STAGE2_GPU = 11 * 5 * STAGE2_STEPS
EXPECTED_TOTAL_GPU = EXPECTED_STAGE1_GPU + EXPECTED_STAGE2_GPU
assert EXPECTED_STAGE1_GPU == 14_745_600
assert EXPECTED_STAGE2_GPU == 84_480_000
assert EXPECTED_TOTAL_GPU == 99_225_600

REWARD_MARKER = "STAGE12_REWARD_CONTRACT_V2_FINISHED_IDS"
PATCH_SUFFIX = ".bak_stage12_v2"


# -----------------------------------------------------------------------------
# Generic helpers (standard library only: safe before project imports)
# -----------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            cooked = {}
            for k in fields:
                v = row.get(k, "")
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


def finite(x: Any) -> bool:
    return math.isfinite(fnum(x))


def mean(xs: Iterable[Any], default: float = float("nan")) -> float:
    vals = [fnum(x) for x in xs]
    vals = [x for x in vals if math.isfinite(x)]
    return float(sum(vals) / len(vals)) if vals else default


def sample_sd(xs: Iterable[Any], default: float = float("nan")) -> float:
    vals = [fnum(x) for x in xs]
    vals = [x for x in vals if math.isfinite(x)]
    return float(statistics.stdev(vals)) if len(vals) >= 2 else default


def median(xs: Iterable[Any], default: float = float("nan")) -> float:
    vals = [fnum(x) for x in xs]
    vals = [x for x in vals if math.isfinite(x)]
    return float(statistics.median(vals)) if vals else default


def current_git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root,
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except Exception:
        return "UNAVAILABLE"


def banner(msg: str) -> None:
    print("\n" + "=" * 126)
    print(msg)
    print("=" * 126, flush=True)


def safe_slug(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


# -----------------------------------------------------------------------------
# P0 reward source contract
# -----------------------------------------------------------------------------

def _backup_once(path: Path) -> None:
    bak = Path(str(path) + PATCH_SUFFIX)
    if not bak.exists():
        shutil.copy2(path, bak)


def _patch_scenario_reward(path: Path) -> Dict[str, Any]:
    before = sha256(path)
    text = path.read_text(encoding="utf-8")

    if REWARD_MARKER in text:
        if "pid not in _finished_stage12" not in text or "self.finished_ids" not in text:
            raise RuntimeError(f"{path}: reward marker exists but contract body is inconsistent")
        return {"path": str(path), "changed": False, "before": before, "after": before}

    old = '''        # Person-minutes in [t, t+1): every already-generated unfinished passenger counts.\n        active_count_start = sum(\n            1\n            for person in self.persons.persons.values()\n            if str(getattr(person, "state", "")).lower() != "finished"\n        )\n        reward = -float(active_count_start)\n'''
    new = f'''        # {REWARD_MARKER}\n        # Person-minutes in [t,t+1): count every already-generated passenger\n        # that has NOT physically completed.  finished_ids is authoritative:\n        # PersonInfo later changes state finished -> removed, so state!=finished\n        # would incorrectly make completed passengers active again.\n        _finished_stage12 = set(self.finished_ids)\n        active_count_start = sum(\n            1\n            for pid in self.persons.persons.keys()\n            if pid not in _finished_stage12\n        )\n        reward = -float(active_count_start)\n'''
    if old not in text:
        raise RuntimeError(
            f"{path}: current Scenario.step reward block differs from the audited main-branch layout; "
            "refusing to guess a source patch"
        )
    _backup_once(path)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    after = sha256(path)
    return {"path": str(path), "changed": True, "before": before, "after": after}


def _patch_active_person_count(path: Path) -> Dict[str, Any]:
    before = sha256(path)
    text = path.read_text(encoding="utf-8")
    marker = "STAGE12_ACTIVE_COUNT_V2_FINISHED_IDS"
    if marker in text:
        if "pid not in finished" not in text:
            raise RuntimeError(f"{path}: active-count marker exists but body is inconsistent")
        return {"path": str(path), "changed": False, "before": before, "after": before}

    old = '''def active_person_count(scenario: Any) -> int:\n    persons_obj = getattr(scenario, "persons", None)\n    persons = getattr(persons_obj, "persons", {}) if persons_obj is not None else {}\n    return sum(\n        1\n        for p in persons.values()\n        if str(getattr(p, "state", "")).lower() != "finished"\n    )\n'''
    new = f'''def active_person_count(scenario: Any) -> int:\n    # {marker}\n    persons_obj = getattr(scenario, "persons", None)\n    persons = getattr(persons_obj, "persons", {{}}) if persons_obj is not None else {{}}\n    finished = set(getattr(scenario, "finished_ids", []) or [])\n    return sum(1 for pid in persons.keys() if pid not in finished)\n'''
    if old not in text:
        raise RuntimeError(
            f"{path}: active_person_count differs from the audited main-branch layout; "
            "refusing to guess a source patch"
        )
    _backup_once(path)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    after = sha256(path)
    return {"path": str(path), "changed": True, "before": before, "after": after}


def ensure_reward_contract(root: Path, *, auto_patch: bool) -> Dict[str, Any]:
    scenario = root / "at_obj" / "scenario.py"
    legacy = root / "train_uam_7x12_600k_v2.py"
    for p in (scenario, legacy):
        if not p.exists():
            raise FileNotFoundError(p)

    if not auto_patch:
        s = scenario.read_text(encoding="utf-8")
        l = legacy.read_text(encoding="utf-8")
        if REWARD_MARKER not in s or "STAGE12_ACTIVE_COUNT_V2_FINISHED_IDS" not in l:
            raise RuntimeError(
                "--no-auto-patch was requested, but the Stage12 reward contract is not already present"
            )
        records = [
            {"path": str(scenario), "changed": False, "before": sha256(scenario), "after": sha256(scenario)},
            {"path": str(legacy), "changed": False, "before": sha256(legacy), "after": sha256(legacy)},
        ]
    else:
        records = [_patch_scenario_reward(scenario), _patch_active_person_count(legacy)]

    return {
        "contract": "reward=-N_generated_unfinished_at_step_start*1min",
        "authoritative_completion": "scenario.finished_ids",
        "reason": "finished passengers later become state=removed and must never re-enter active count",
        "source_records": records,
        "source_hashes_after": {Path(r["path"]).name: r["after"] for r in records},
        "backup_suffix": PATCH_SUFFIX,
    }


# -----------------------------------------------------------------------------
# Delayed project imports: MUST happen only after reward source patch
# -----------------------------------------------------------------------------

def load_stack() -> SimpleNamespace:
    import torch
    import numpy as np
    import train_uam_pre5b_80m_autofunnel as pre80
    import train_uam_pre500m_50m_freeze_supplement as pre50
    import train_uam_500m_stagewise as core
    import uam500m_methods as methods

    # Installs V5 clean physical wrapper, UAGMC_SOURCE common-profile extractor,
    # and current S/AC method registry/builders.
    methods.install_hooks()
    return SimpleNamespace(
        torch=torch, np=np, pre80=pre80, pre50=pre50,
        core=core, methods=methods, v3=pre80.v3, v4=pre80.v4,
    )


# -----------------------------------------------------------------------------
# P0 runtime invariant audit
# -----------------------------------------------------------------------------

def runtime_reward_audit(stack: SimpleNamespace, suite: Path) -> Dict[str, Any]:
    out = suite / "STAGE1" / "P0_REWARD_AUDIT" / "audit.json"
    old = read_json(out, None)
    if isinstance(old, dict) and old.get("status") == "PASS":
        return old

    pre80, pre50, v3 = stack.pre80, stack.pre50, stack.v3
    trace = Path(pre80.BASE_TRACE).resolve()
    pre50.set_trace_everywhere(trace)
    run_dir = out.parent / "episode"
    run_dir.mkdir(parents=True, exist_ok=True)

    factory = pre50.rulebase.core.make_experiment_env_factory(
        stage="E6", topology="T2", encoder_mode="uagmc",
        fleet_size=int(v3.FLEET_SIZE), env_index=987,
        run_dir=run_dir, pad_separation=float(v3.PAD_SEPARATION_MIN),
        charger_capacity=int(v3.CHARGER_CAPACITY), max_time=HARD_GUARD,
    )
    env = factory()
    reward_sum = 0.0
    steps = 0
    try:
        try:
            env.reset(seed=987)
        except TypeError:
            env.reset()
        done = False
        while not done:
            action, _ = pre50.rulebase.choose_action(
                env, "E6", "T2", "SPF", int(v3.CHARGER_CAPACITY),
                float(pre50.rulebase.DEFAULT_UNKNOWN_EVENT_PENALTY_MIN),
            )
            result = env.step(int(action))
            if len(result) == 5:
                _, reward, terminated, truncated, _ = result
                done = bool(terminated) or bool(truncated)
            else:
                _, reward, done, _ = result
                done = bool(done)
            reward_sum += fnum(reward, 0.0)
            steps += 1
            if steps > HARD_GUARD + 100:
                raise RuntimeError("P0 reward audit exceeded hard guard")

        scenario = pre50.rulebase.old.find_scenario(env)
        persons = dict(getattr(getattr(scenario, "persons", None), "persons", {}) or {})
        finished = set(getattr(scenario, "finished_ids", []) or [])
        travel = []
        missing = []
        for pid, person in persons.items():
            start = getattr(person, "spawn_time", None)
            end = getattr(person, "end_time", None)
            if start is None or end is None or pid not in finished:
                missing.append(pid)
                continue
            travel.append(float(end) - float(start))
        if missing:
            raise RuntimeError(f"P0 reward audit has unfinished/missing travel times: n={len(missing)}")
        travel_sum = float(sum(travel))
        gap = float(abs((-reward_sum) - travel_sum))
        # With this simulator timing the identity should be exact; tolerate tiny FP noise only.
        tolerance = max(1e-8, 1e-10 * max(1.0, travel_sum))
        status = "PASS" if gap <= tolerance else "FAIL"
        audit = {
            "status": status,
            "trace": str(trace),
            "trace_sha256": sha256(trace),
            "method": "SPF",
            "seed": 987,
            "N": len(persons),
            "episode_steps": steps,
            "negative_reward_sum": -reward_sum,
            "sum_finish_minus_spawn": travel_sum,
            "absolute_gap": gap,
            "tolerance": tolerance,
        }
        write_json(out, audit)
        if status != "PASS":
            raise RuntimeError(f"P0 reward invariant FAILED: {audit}")
        return audit
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            v3.core.restore_process_patches()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Stage I helpers
# -----------------------------------------------------------------------------

def profile_dict(profile: Any) -> Dict[str, Any]:
    return {
        "name": str(profile.name), "n_envs": int(profile.n_envs),
        "n_steps": int(profile.n_steps), "batch_size": int(profile.batch_size),
        "rollout": int(profile.n_envs) * int(profile.n_steps),
    }


def stage1_batch(stack: SimpleNamespace, suite: Path, device: str, fail_fast: bool) -> Dict[str, Any]:
    root = suite / "STAGE1"
    phase = "S1_BATCH"
    out = root / phase / "BATCH_FINAL.json"
    old = read_json(out, None)
    if isinstance(old, dict):
        return old

    pre80, pre50, v3 = stack.pre80, stack.pre50, stack.v3
    trace = str(Path(pre80.BASE_TRACE).resolve())
    profiles = [
        v3.SpeedProfile("S1_10x2048_b1024", 10, 2048, 1024),
        v3.SpeedProfile("S1_10x2048_b4096", 10, 2048, 4096),
    ]
    rows: List[Dict[str, Any]] = []
    for profile in profiles:
        for method, env_key in STAGE1_METHODS:
            for seed in BATCH_SEEDS:
                x = pre50.run_cell_cached(
                    root=root, phase=phase, tag=profile.name,
                    env_key=env_key, method=method, seed=seed,
                    timesteps=STAGE1_STEPS, profile=profile, trace_path=trace,
                    device=device, eval_seeds=(123,), long_run=False,
                    fail_fast=fail_fast,
                )
                x = dict(x)
                x.update(profile_dict(profile))
                x["profile"] = profile.name
                rows.append(x)
                write_csv(root / phase / "cells.csv", rows)

    agg = pre50.aggregate(rows, ("profile", "method"))
    ranking = pre50.score_profile_summaries(
        agg, candidate_names=[p.name for p in profiles],
        method_names=[m for m, _ in STAGE1_METHODS],
    )
    by_name = {p.name: p for p in profiles}
    selected = by_name[str(ranking[0]["profile"])]
    result = {
        "selected_profile": profile_dict(selected),
        "ranking": ranking,
        "selection_rule": (
            "fixed n_envs=10,n_steps=2048,rollout=20480; compare batch1024 vs4096 "
            "across UAGMC/S/AC with quality-first normalized ATT + seed stability + retention"
        ),
        "planned_gpu_transitions": 2 * 3 * 3 * STAGE1_STEPS,
    }
    write_csv(root / phase / "summary.csv", agg)
    write_json(out, result)
    return result


def _load_ranking(stack: SimpleNamespace, analytical_agg: Sequence[Dict[str, Any]], uagmc_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pre50 = stack.pre50
    uagg = pre50.aggregate(uagmc_rows, ("load", "method"))
    ranking: List[Dict[str, Any]] = []
    for rho in LOADS:
        ars = [r for r in analytical_agg if abs(fnum(r.get("load")) - rho) < 1e-9]
        ars = [r for r in ars if fnum(r.get("completion_mean"), 0.0) >= 0.98 and finite(r.get("ATT_mean"))]
        best_a = min(ars, key=lambda r: fnum(r.get("ATT_mean"), float("inf"))) if ars else None
        urs = [r for r in uagg if abs(fnum(r.get("load")) - rho) < 1e-9]
        ur = urs[0] if urs else {}
        ua = fnum(ur.get("final_mean"), float("inf"))
        aa = fnum(best_a.get("ATT_mean"), float("inf")) if best_a else float("inf")
        headroom = ua / aa - 1.0 if finite(ua) and finite(aa) and aa > 0 else float("inf")
        criticality = max(0.0, fnum(ur.get("final_cv"), 9.0)) + max(0.0, fnum(ur.get("collapse_ratio"), 9.0))
        valid = int(ur.get("n_valid", 0)) == int(ur.get("n_planned", -1)) and best_a is not None
        score = (0.0 if valid else 10.0) + abs(headroom - 0.175) + 0.5 * abs(criticality - 0.20) - 0.01 * rho
        ranking.append({
            "load": rho,
            "analytical_best_method": best_a.get("method") if best_a else None,
            "analytical_best_ATT": aa,
            "uagmc_final": ua,
            "uagmc_cv": ur.get("final_cv"),
            "uagmc_collapse_ratio": ur.get("collapse_ratio"),
            "headroom_ratio": headroom,
            "criticality": criticality,
            "n_uagmc_models": ur.get("n_valid"),
            "score": score,
        })
    ranking.sort(key=lambda r: (fnum(r["score"], 999.0), -float(r["load"])))
    return ranking


def stage1_load(stack: SimpleNamespace, suite: Path, profile: Any, device: str, fail_fast: bool) -> Dict[str, Any]:
    root = suite / "STAGE1"
    phase = "S1_LOAD"
    out = root / phase / "LOAD_FINAL.json"
    old = read_json(out, None)
    if isinstance(old, dict):
        return old

    pre80, pre50 = stack.pre80, stack.pre50
    phase_root = root / phase
    trace_map = pre80.make_nested_load_traces(phase_root / "load_traces")

    analytical_rows: List[Dict[str, Any]] = []
    for rho in LOADS:
        trace = trace_map[f"{int(round(rho * 100)):03d}"]
        for method in ANALYTICAL_METHODS:
            for seed in ANALYTICAL_SEEDS:
                row = pre50.analytical_one_cached(
                    out_root=phase_root, trace_path=trace, load=rho,
                    method=method, seed=seed,
                )
                analytical_rows.append(row)
                write_csv(phase_root / "analytical_raw.csv", analytical_rows)
    analytical_agg = pre50.aggregate_analytical(analytical_rows)
    write_csv(phase_root / "analytical_summary.csv", analytical_agg)

    uagmc_rows: List[Dict[str, Any]] = []
    for rho in LOADS:
        trace = trace_map[f"{int(round(rho * 100)):03d}"]
        for seed in LOAD_BASE_SEEDS:
            x = pre50.run_cell_cached(
                root=root, phase=phase, tag=f"UAGMC_LOAD_{int(round(rho*100)):03d}",
                env_key="S3", method="UAGMC_SOURCE", seed=seed,
                timesteps=STAGE1_STEPS, profile=profile, trace_path=trace,
                device=device, eval_seeds=(123,), long_run=False, fail_fast=fail_fast,
            )
            x = dict(x); x["load"] = rho
            uagmc_rows.append(x)
            write_csv(phase_root / "uagmc_cells.csv", uagmc_rows)

    prelim = _load_ranking(stack, analytical_agg, uagmc_rows)
    top2 = [float(r["load"]) for r in prelim[:2]]
    for rho in top2:
        trace = trace_map[f"{int(round(rho * 100)):03d}"]
        x = pre50.run_cell_cached(
            root=root, phase=phase, tag=f"UAGMC_LOAD_{int(round(rho*100)):03d}",
            env_key="S3", method="UAGMC_SOURCE", seed=LOAD_CONFIRM_SEED,
            timesteps=STAGE1_STEPS, profile=profile, trace_path=trace,
            device=device, eval_seeds=(123,), long_run=False, fail_fast=fail_fast,
        )
        x = dict(x); x["load"] = rho
        uagmc_rows.append(x)
        write_csv(phase_root / "uagmc_cells.csv", uagmc_rows)

    ranking = _load_ranking(stack, analytical_agg, uagmc_rows)
    selected_load = float(ranking[0]["load"])
    selected_trace = str(Path(trace_map[f"{int(round(selected_load*100)):03d}"]).resolve())
    result = {
        "selected_load": selected_load,
        "selected_trace": selected_trace,
        "preliminary_top2": top2,
        "ranking": ranking,
        "selection_rule": (
            "six current-runtime online-legal analytical baselines + UAGMC_SOURCE; "
            "target about 17.5% analytical headroom and 0.20 seed-CV+retention criticality"
        ),
        "planned_gpu_transitions": (5 * 2 + 2) * STAGE1_STEPS,
        "cpu_analytical_episodes": len(LOADS) * len(ANALYTICAL_METHODS) * len(ANALYTICAL_SEEDS),
    }
    write_csv(phase_root / "load_ranking.csv", ranking)
    write_json(out, result)
    return result


def _demand_protocol_summary(eval_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # First aggregate all validation episodes *within one trained model*.
    model_groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in eval_rows:
        model_groups[(str(r["protocol"]), str(r["method"]), int(r["train_seed"]))].append(r)
    model_rows: List[Dict[str, Any]] = []
    for (protocol, method, train_seed), rs in model_groups.items():
        valid = [r for r in rs if bool(r.get("valid_full_completion")) and finite(r.get("ATT"))]
        model_rows.append({
            "protocol": protocol, "method": method, "train_seed": train_seed,
            "n_eval": len(rs), "n_valid": len(valid),
            "ATT_model_mean": mean((r.get("ATT") for r in valid), float("inf")),
            "ATT_model_worst": max([fnum(r.get("ATT")) for r in valid], default=float("inf")),
        })

    group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in model_rows:
        group[(r["protocol"], r["method"])].append(r)
    summaries: List[Dict[str, Any]] = []
    for (protocol, method), rs in group.items():
        vals = [fnum(r["ATT_model_mean"], float("inf")) for r in rs if r["n_valid"] == r["n_eval"]]
        m = mean(vals, float("inf")); sd = sample_sd(vals, float("inf"))
        summaries.append({
            "protocol": protocol, "method": method,
            "n_trained_models": len(rs), "n_complete_models": len(vals),
            "ATT_mean_across_models": m, "ATT_sd_across_models": sd,
            "ATT_cv_across_models": sd / m if finite(sd) and finite(m) and m > 0 else float("inf"),
            "ATT_worst_model": max(vals) if vals else float("inf"),
            "statistical_unit": "trained_model",
        })
    return model_rows, summaries


def stage1_demand(stack: SimpleNamespace, suite: Path, profile: Any, selected_trace: str, device: str, fail_fast: bool) -> Dict[str, Any]:
    root = suite / "STAGE1"
    phase = "S1_DEMAND"
    phase_root = root / phase
    out = phase_root / "DEMAND_FINAL.json"
    old = read_json(out, None)
    if isinstance(old, dict):
        return old

    pre50 = stack.pre50
    bank = pre50.make_empirical_trace_bank(
        selected_trace=selected_trace, out_dir=phase_root / "trace_bank",
        n_train=3, n_test=4,
    )
    protocols = ("FIXED_TRACE", "MULTI_REALIZATION_ACROSS_REPLICATES")
    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []

    for protocol in protocols:
        for method, env_key in STAGE1_METHODS:
            for seed in DEMAND_SEEDS:
                train_trace = pre50.trace_for_seed(
                    protocol=protocol, seed=seed, seed_order=DEMAND_SEEDS,
                    selected_trace=selected_trace, bank=bank,
                )
                tag = f"{protocol}__{method}"
                x = pre50.run_cell_cached(
                    root=root, phase=phase, tag=tag, env_key=env_key,
                    method=method, seed=seed, timesteps=STAGE1_STEPS,
                    profile=profile, trace_path=train_trace, device=device,
                    eval_seeds=(123,), long_run=False, fail_fast=fail_fast,
                )
                x = dict(x); x.update({"protocol": protocol, "train_trace": train_trace})
                train_rows.append(x)
                write_csv(phase_root / "train_cells.csv", train_rows)

                run_dir = pre50.cell_run_dir(
                    root=root, phase=phase, tag=tag, env_key=env_key,
                    method=method, seed=seed,
                )
                for trace in bank["test_traces"]:
                    for eval_seed in EVAL_SEEDS:
                        row = pre50.eval_final_on_trace_cached(
                            run_dir=run_dir, env_key=env_key, method=method,
                            trace_path=trace, eval_seed=eval_seed,
                            label="validation_bank",
                        )
                        row = dict(row)
                        row.update({"protocol": protocol, "method": method, "train_seed": seed})
                        eval_rows.append(row)
                        write_csv(phase_root / "validation_raw.csv", eval_rows)

    model_rows, summaries = _demand_protocol_summary(eval_rows)
    method_min: Dict[str, float] = {}
    for method in [m for m, _ in STAGE1_METHODS]:
        vals = [fnum(r["ATT_mean_across_models"], float("inf")) for r in summaries if r["method"] == method]
        method_min[method] = min(vals) if vals else float("inf")

    ranking: List[Dict[str, Any]] = []
    for protocol in protocols:
        comps = []
        invalid = 0
        for r in [x for x in summaries if x["protocol"] == protocol]:
            ref = method_min.get(r["method"], float("inf"))
            att = fnum(r["ATT_mean_across_models"], float("inf"))
            gap = att / ref - 1.0 if finite(att) and finite(ref) and ref > 0 else 9.0
            cv = fnum(r["ATT_cv_across_models"], 9.0)
            invalid += int(int(r["n_complete_models"]) != len(DEMAND_SEEDS))
            comps.append(gap + 0.5 * cv)
        ranking.append({
            "protocol": protocol,
            "score": 10.0 * invalid + median(comps, 99.0),
            "invalid_method_count": invalid,
        })
    ranking.sort(key=lambda r: fnum(r["score"], 999.0))
    selected = str(ranking[0]["protocol"])
    train_bank = [str(Path(selected_trace).resolve())] if selected == "FIXED_TRACE" else [str(Path(p).resolve()) for p in bank["train_traces"]]
    result = {
        "selected_protocol": selected,
        "trace_bank": bank,
        "train_trace_bank": train_bank,
        "heldout_trace_bank": [str(Path(p).resolve()) for p in bank["test_traces"]],
        "ranking": ranking,
        "model_level_rows": model_rows,
        "method_summaries": summaries,
        "selection_rule": "aggregate validation traces inside each trained model first, then compare across training seeds",
        "interpretation_boundary": "heldout traces are validation/development only, never final test",
        "planned_gpu_transitions": 2 * 3 * 3 * STAGE1_STEPS,
    }
    write_csv(phase_root / "model_level_validation.csv", model_rows)
    write_csv(phase_root / "protocol_method_summary.csv", summaries)
    write_csv(phase_root / "protocol_ranking.csv", ranking)
    write_json(out, result)
    return result


def trace_locks(paths: Sequence[str]) -> List[Dict[str, str]]:
    out = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        out.append({"path": str(p), "sha256": sha256(p)})
    return out


def write_stage1_freeze(
    stack: SimpleNamespace, suite: Path, patch_info: Dict[str, Any], reward_audit: Dict[str, Any],
    batch: Dict[str, Any], load: Dict[str, Any], demand: Dict[str, Any],
) -> Tuple[Path, Dict[str, Any]]:
    freeze_path = suite / "UAM_FINAL_FREEZE_V2.json"
    sha_path = suite / "UAM_FINAL_FREEZE_V2.sha256"
    if freeze_path.exists() and sha_path.exists():
        data = read_json(freeze_path)
        expected = sha_path.read_text(encoding="utf-8").strip().split()[0]
        actual = sha256(freeze_path)
        if actual != expected:
            raise RuntimeError("existing Stage-I freeze SHA256 mismatch")
        return freeze_path, data

    v3 = stack.v3
    p = batch["selected_profile"]
    train_bank = demand["train_trace_bank"]
    valid_bank = demand["heldout_trace_bank"]
    freeze = {
        "schema": "UAM_FINAL_FREEZE_V2",
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": current_git_commit(ROOT),
        "stage1_planned_gpu_transitions": EXPECTED_STAGE1_GPU,
        "reward_contract": {
            **patch_info,
            "runtime_invariant_audit": reward_audit,
        },
        "frozen_ppo": {
            "profile_name": p["name"],
            "n_envs": int(p["n_envs"]), "n_steps": int(p["n_steps"]),
            "global_rollout": int(p["rollout"]), "batch_size": int(p["batch_size"]),
            "n_epochs": int(v3.N_EPOCHS), "gamma": float(v3.GAMMA),
            "gae_lambda": float(v3.GAE_LAMBDA), "clip_range": float(v3.CLIP_RANGE),
            "initial_lr": float(v3.INITIAL_LR), "ent_coef": float(v3.ENT_COEF),
            "vf_coef": float(v3.VF_COEF), "max_grad_norm": float(v3.MAX_GRAD_NORM),
            "lr_schedule": "linear_from_initial_to_zero_over_each_full_run",
        },
        "frozen_environment": {
            "physics": "S3", "topology": "T2", "fleet_size": int(v3.FLEET_SIZE),
            "selected_load": float(load["selected_load"]),
            "selected_trace": str(Path(load["selected_trace"]).resolve()),
            "selected_trace_sha256": sha256(Path(load["selected_trace"]).resolve()),
            "pad_separation_min": float(v3.PAD_SEPARATION_MIN),
            "charger_capacity": int(v3.CHARGER_CAPACITY),
            "hard_guard_min": HARD_GUARD,
        },
        "demand_protocol": {
            "selected_protocol": demand["selected_protocol"],
            "train_trace_bank": train_bank,
            "train_trace_locks": trace_locks(train_bank),
            "heldout_trace_bank": valid_bank,
            "heldout_trace_locks": trace_locks(valid_bank),
            "boundary": demand["interpretation_boundary"],
        },
        "joint_contract": {
            "physical_semantics": "V5_CLEAN_MINIMAL_REPOSITION",
            "allowed_stage2_joint_carriers": ["J3", "J4"],
            "J3": "P(a_passenger|s) * P(a_aircraft|s,a_passenger)",
            "J4": "semantic pair scorer P(a_passenger,a_aircraft|s)",
            "credit_semantics": "STANDARD_4WAY_PPO_NO_RELEVANCE_MASK",
        },
        "formal_training_protocol": {
            "stage2_steps_per_model": STAGE2_STEPS,
            "stage2_training_seeds": list(STAGE2_SEEDS),
            "checkpoint_target_interval": SAVE_EVERY,
            "dense_curve": "all saved ~50k checkpoints on canonical validation trace",
            "absolute_ATT": "validation-bank LWATT over final 20% checkpoints",
            "statistical_unit": "trained_model",
            "final_test": "DEFERRED_UNTOUCHED_UNTIL_STAGE3_COMPLETE",
        },
        "stage1_evidence": {"batch": batch, "load": load, "demand": demand},
    }
    write_json(freeze_path, freeze)
    digest = sha256(freeze_path)
    sha_path.write_text(f"{digest}  {freeze_path.name}\n", encoding="utf-8")
    return freeze_path, freeze


def verify_freeze(freeze_path: Path) -> Dict[str, Any]:
    sha_path = freeze_path.with_suffix(".sha256")
    if not freeze_path.is_file() or not sha_path.is_file():
        raise FileNotFoundError("Stage II requires Stage-I freeze JSON + SHA256; no fallback is allowed")
    expected = sha_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256(freeze_path)
    if expected != actual:
        raise RuntimeError("Stage-I freeze file SHA256 mismatch")
    freeze = read_json(freeze_path)
    if freeze.get("schema") != "UAM_FINAL_FREEZE_V2":
        raise RuntimeError("wrong freeze schema")
    if freeze.get("git_commit") != current_git_commit(ROOT):
        raise RuntimeError("git commit changed after Stage-I freeze; refusing Stage-II mixed runtime")
    if freeze.get("joint_contract", {}).get("allowed_stage2_joint_carriers") != ["J3", "J4"]:
        raise RuntimeError("Stage-II joint carriers are not exactly J3/J4")
    p = freeze.get("frozen_ppo", {})
    if int(p.get("n_envs", 0)) != 10 or int(p.get("n_steps", 0)) != 2048 or int(p.get("global_rollout", 0)) != ROLLOUT:
        raise RuntimeError("frozen PPO rollout is not 10x2048=20480")
    # Source and trace locks are re-verified before any Stage-II training.
    for rec in freeze["reward_contract"]["source_records"]:
        path = Path(rec["path"])
        if sha256(path) != rec["after"]:
            raise RuntimeError(f"reward-contract source changed after Stage I: {path}")
    for bank_key in ("train_trace_locks", "heldout_trace_locks"):
        for rec in freeze["demand_protocol"][bank_key]:
            path = Path(rec["path"])
            if not path.is_file() or sha256(path) != rec["sha256"]:
                raise RuntimeError(f"trace lock mismatch: {path}")
    freeze["_sha256"] = actual
    return freeze


# -----------------------------------------------------------------------------
# Stage II CPU analytical baseline under locked validation bank
# -----------------------------------------------------------------------------

def stage2_cpu_analytical(stack: SimpleNamespace, suite: Path, freeze: Dict[str, Any]) -> Dict[str, Any]:
    root = suite / "STAGE2" / "CPU_ANALYTICAL"
    out = root / "ANALYTICAL_BASELINE_FREEZE.json"
    old = read_json(out, None)
    if isinstance(old, dict):
        return old

    pre50 = stack.pre50
    rows: List[Dict[str, Any]] = []
    load = float(freeze["frozen_environment"]["selected_load"])
    traces = list(freeze["demand_protocol"]["heldout_trace_bank"])
    for ti, trace in enumerate(traces):
        troot = root / f"trace_{ti:02d}_{sha256(Path(trace))[:10]}"
        for method in ANALYTICAL_METHODS:
            for seed in ANALYTICAL_SEEDS:
                row = pre50.analytical_one_cached(
                    out_root=troot, trace_path=trace, load=load,
                    method=method, seed=seed,
                )
                row = dict(row); row["validation_trace"] = str(Path(trace).resolve())
                rows.append(row)
                write_csv(root / "raw.csv", rows)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r["method"])].append(r)
    summary = []
    for method, rs in groups.items():
        valid = [r for r in rs if fnum(r.get("completion_rate"), 0.0) >= 0.98 and finite(r.get("ATT"))]
        vals = [fnum(r["ATT"]) for r in valid]
        m = mean(vals, float("inf")); sd = sample_sd(vals, float("inf"))
        summary.append({
            "method": method, "n_eval_episodes": len(rs), "n_valid": len(valid),
            "ATT_mean": m, "ATT_sd": sd,
            "ATT_cv": sd / m if finite(sd) and finite(m) and m > 0 else float("inf"),
            "ATT_worst": max(vals) if vals else float("inf"),
            "within_training_run_stability": "N/A_NO_TRAINING",
        })
    summary.sort(key=lambda r: fnum(r["ATT_mean"], float("inf")))
    best = summary[0] if summary else None
    result = {
        "runtime": "current Stage-I locked code/physics/traces; historical analytical numbers not reused",
        "best": best,
        "methods": summary,
        "validation_traces": traces,
        "eval_seeds": list(ANALYTICAL_SEEDS),
        "episodes": len(rows),
        "comparison_boundary": (
            "analytical methods use frozen S3 passenger-decision physics with deterministic aircraft rule; "
            "they are a common non-learning system baseline for Single and Joint RL"
        ),
    }
    write_csv(root / "summary.csv", summary)
    write_json(out, result)
    return result


# -----------------------------------------------------------------------------
# Stage II training + evaluation
# -----------------------------------------------------------------------------

def stage2_plan(stack: SimpleNamespace) -> List[Any]:
    cells = []
    Cell = stack.core.Cell
    for lane, specs in STAGE2_LANES.items():
        for method, env in specs:
            for seed in STAGE2_SEEDS:
                cells.append(Cell(f"S2_{lane}", method, env, seed, STAGE2_STEPS))
    if len(cells) != 55 or len({c.key for c in cells}) != 55:
        raise RuntimeError("Stage-II plan must contain exactly 55 unique cells")
    if sum(c.steps for c in cells) != EXPECTED_STAGE2_GPU:
        raise RuntimeError("Stage-II budget mismatch")
    return cells


def _checkpoint_pairs(stack: SimpleNamespace, run_dir: Path) -> List[Tuple[int, Path, Path]]:
    out = []
    for model in (run_dir / "checkpoints").glob("uam_ppo_*_steps.zip"):
        m = re.search(r"uam_ppo_(\d+)_steps\.zip$", model.name)
        if not m:
            continue
        step = int(m.group(1))
        vec = run_dir / "checkpoints" / f"uam_ppo_vecnormalize_{step}_steps.pkl"
        if vec.is_file():
            out.append((step, model, vec))

    # Include the true end-of-run weights (1.536M in the formal Stage-II plan),
    # not only the last periodic ~50k checkpoint (normally 1.50M).
    end = read_json(run_dir / "run_end.json", {})
    final_model = run_dir / "final_model.zip"
    final_vec = run_dir / "final_vecnormalize.pkl"
    final_step = int(end.get("actual_steps", 0) or 0)
    if end.get("status") == "COMPLETE" and final_step > 0 and final_model.is_file() and final_vec.is_file():
        out.append((final_step, final_model, final_vec))

    # A final checkpoint can coincide with a periodic one; prefer the explicit
    # final pair for that step.
    by_step = {}
    for triple in out:
        by_step[int(triple[0])] = triple
    out = [by_step[k] for k in sorted(by_step)]
    return out


def _eval_checkpoint_cached(
    stack: SimpleNamespace, run_dir: Path, cell: Any, step: int,
    model: Path, vec: Path, trace: str, eval_seed: int,
) -> Dict[str, Any]:
    cache_dir = run_dir / "analysis" / "stage12_eval_cache" / f"step_{step}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    trace_path = Path(trace).resolve()
    cache = cache_dir / f"{trace_path.stem}_{sha256(trace_path)[:8]}__seed{eval_seed}.json"
    old = read_json(cache, None)
    if isinstance(old, dict):
        return old
    stack.pre50.set_trace_everywhere(trace_path)
    row = stack.v3.evaluate_checkpoint(
        env_key=cell.env, method_id=cell.method,
        model_path=model, vec_path=vec, train_step=step,
        eval_seed=int(eval_seed), run_dir=cache_dir / "monitor",
        max_time=HARD_GUARD,
    )
    row = dict(row)
    row.update({"trace_path": str(trace_path), "eval_seed": eval_seed, "train_seed": cell.seed})
    write_json(cache, row)
    return row


def evaluate_stage2_model(stack: SimpleNamespace, suite: Path, cell: Any, freeze: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = stack.core.cell_dir(suite / "STAGE2" / "RL", cell)
    out = run_dir / "analysis" / "STAGE12_MODEL_SUMMARY.json"
    old = read_json(out, None)
    if isinstance(old, dict) and old.get("status") == "VALID":
        return old
    end = read_json(run_dir / "run_end.json", {})
    if end.get("status") != "COMPLETE":
        raise RuntimeError(f"cannot evaluate incomplete Stage-II cell: {cell.key}")

    pairs = _checkpoint_pairs(stack, run_dir)
    if not pairs:
        raise RuntimeError(f"no paired checkpoints: {run_dir}")
    validation = list(freeze["demand_protocol"]["heldout_trace_bank"])
    if not validation:
        raise RuntimeError("empty validation trace bank")
    canonical = validation[0]

    # Dense curve: every saved checkpoint, one canonical validation trace/seed.
    dense = []
    for step, model, vec in pairs:
        r = _eval_checkpoint_cached(stack, run_dir, cell, step, model, vec, canonical, 123)
        dense.append({
            "step": step,
            "ATT": fnum(r.get("ATT")), "AWT": fnum(r.get("AWT")),
            "completion": fnum(r.get("completion_rate")),
            "valid": bool(r.get("valid_full_completion")) and finite(r.get("ATT")),
        })
    valid_dense = [r for r in dense if r["valid"]]
    if not valid_dense:
        raise RuntimeError(f"no full-completion dense ATT points: {cell.key}")

    max_step = max(r["step"] for r in valid_dense)
    threshold = 0.80 * max_step
    late_pairs = [(s, m, v) for s, m, v in pairs if s >= threshold]
    if not late_pairs:
        late_pairs = pairs[-3:]

    # Main absolute ATT: all validation traces / eval RNGs on all final-20% checkpoints.
    late_step_rows = []
    for step, model, vec in late_pairs:
        episode_rows = []
        for trace in validation:
            for eval_seed in EVAL_SEEDS:
                episode_rows.append(_eval_checkpoint_cached(
                    stack, run_dir, cell, step, model, vec, trace, eval_seed,
                ))
        valid = [r for r in episode_rows if bool(r.get("valid_full_completion")) and finite(r.get("ATT"))]
        late_step_rows.append({
            "step": step, "n_eval": len(episode_rows), "n_valid": len(valid),
            "ATT": mean((r.get("ATT") for r in valid), float("inf")),
            "AWT": mean(r.get("AWT") for r in valid),
        })
    if any(r["n_valid"] != r["n_eval"] for r in late_step_rows):
        raise RuntimeError(f"incomplete validation-bank episode in late window: {cell.key}")

    dense_att = [float(r["ATT"]) for r in valid_dense]
    dense_best = min(dense_att)
    dense_best_step = valid_dense[dense_att.index(dense_best)]["step"]
    dense_late = [r for r in valid_dense if r["step"] >= 0.80 * max_step]
    dense_late_vals = [float(r["ATT"]) for r in dense_late]
    dense_late_mean = mean(dense_late_vals)
    retention = (dense_late_mean - dense_best) / dense_best if dense_best > 0 else float("nan")

    xs = [float(r["step"]) / 100_000.0 for r in dense_late]
    ys = dense_late_vals
    if len(xs) >= 2:
        xbar, ybar = mean(xs), mean(ys)
        denom = sum((x - xbar) ** 2 for x in xs)
        slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom if denom > 0 else 0.0
    else:
        slope = float("nan")

    fixed300 = min(valid_dense, key=lambda r: abs(int(r["step"]) - 300_000))
    lwatt = mean(r["ATT"] for r in late_step_rows)
    final_bank = late_step_rows[-1]["ATT"]
    summary = {
        "status": "VALID",
        "stage": cell.stage, "env": cell.env, "method": cell.method,
        "train_seed": cell.seed,
        # Dimension 1: absolute ATT
        "LWATT_validation": lwatt,
        "Final_validation": final_bank,
        "late_window_start_fraction": 0.80,
        "late_window_steps": [int(r["step"]) for r in late_step_rows],
        # Dimension 2: within-run stability
        "dense_curve_reference": "canonical validation trace + eval_seed123",
        "dense_best": dense_best,
        "dense_best_step": int(dense_best_step),
        "dense_late_mean": dense_late_mean,
        "best_to_late_retention": retention,
        "dense_late_sd": sample_sd(dense_late_vals),
        "dense_late_slope_min_per_100k": slope,
        "fixed300_step": int(fixed300["step"]),
        "fixed300_ATT": float(fixed300["ATT"]),
        "fixed300_to_late_delta": dense_late_mean - float(fixed300["ATT"]),
        "best_position_fraction": float(dense_best_step) / float(max_step),
        "catastrophic_retreat_over_20pct": bool(retention > 0.20),
        "n_dense_checkpoints": len(valid_dense),
        # Evaluation scope
        "validation_traces": len(validation),
        "validation_eval_seeds": list(EVAL_SEEDS),
        "statistical_unit": "trained_model",
    }
    write_csv(run_dir / "analysis" / "STAGE12_DENSE_CURVE.csv", dense)
    write_csv(run_dir / "analysis" / "STAGE12_LATE_BANK.csv", late_step_rows)
    write_json(out, summary)
    return summary


def aggregate_stage2(
    suite: Path, plan: Sequence[Any], model_summaries: Sequence[Dict[str, Any]],
    analytical: Dict[str, Any], freeze: Dict[str, Any],
) -> Dict[str, Any]:
    # lane is encoded in cell.stage as S2_<lane>.
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in model_summaries:
        lane = str(r["stage"])[3:] if str(r["stage"]).startswith("S2_") else str(r["stage"])
        groups[(lane, str(r["env"]), str(r["method"]))].append(r)

    rows = []
    analytical_best = fnum((analytical.get("best") or {}).get("ATT_mean"), float("inf"))
    for (lane, env, method), rs in sorted(groups.items()):
        vals = [fnum(r["LWATT_validation"], float("inf")) for r in rs]
        rets = [fnum(r["best_to_late_retention"], float("inf")) for r in rs]
        m = mean(vals, float("inf")); sd = sample_sd(vals, float("inf"))
        rows.append({
            "lane": lane, "env": env, "method": method,
            "n_train_seeds": len(rs),
            # Dimension 1
            "LWATT_mean": m,
            # Dimension 2
            "retention_mean": mean(rets),
            "retention_worst": max(rets) if rets else float("inf"),
            "n_catastrophic_over_20pct": sum(1 for x in rets if x > 0.20),
            "late_slope_mean": mean(r.get("dense_late_slope_min_per_100k") for r in rs),
            # Dimension 3
            "LWATT_sd": sd,
            "LWATT_cv": sd / m if finite(sd) and finite(m) and m > 0 else float("inf"),
            "LWATT_worst": max(vals) if vals else float("inf"),
            "LWATT_range": max(vals) - min(vals) if vals else float("inf"),
            "analytical_best_ATT": analytical_best,
            "delta_vs_analytical": m - analytical_best if finite(analytical_best) else float("nan"),
            "relative_vs_analytical": m / analytical_best - 1.0 if finite(analytical_best) and analytical_best > 0 else float("nan"),
            "statistical_unit": "trained_model",
        })

    # Add same-lane UAGMC deltas.
    refs = {(r["lane"], r["env"]): r for r in rows if r["method"] == "UAGMC_SOURCE"}
    for r in rows:
        ref = refs.get((r["lane"], r["env"]))
        if ref:
            u = fnum(ref["LWATT_mean"], float("inf"))
            r["uagmc_reference_ATT"] = u
            r["delta_vs_uagmc"] = fnum(r["LWATT_mean"]) - u
            r["relative_vs_uagmc"] = fnum(r["LWATT_mean"]) / u - 1.0 if u > 0 else float("nan")
        else:
            r["uagmc_reference_ATT"] = None
            r["delta_vs_uagmc"] = None
            r["relative_vs_uagmc"] = None

    result = {
        "schema": "UAM_BASELINE_FREEZE_V2",
        "created": datetime.now().isoformat(timespec="seconds"),
        "parent_freeze_sha256": freeze["_sha256"],
        "stage2_planned_gpu_transitions": EXPECTED_STAGE2_GPU,
        "stage2_cells": len(plan),
        "stage2_training_seeds": list(STAGE2_SEEDS),
        "methods": rows,
        "analytical": analytical,
        "three_required_dimensions": [
            "absolute_ATT_LWATT_validation",
            "within_run_ATT_stability_retention_slope",
            "across_seed_stability_SD_CV_worst_range",
        ],
        "final_test_used": False,
    }
    write_csv(suite / "STAGE2" / "BASELINE_TABLE.csv", rows)
    write_json(suite / "UAM_BASELINE_FREEZE_V2.json", result)
    digest = sha256(suite / "UAM_BASELINE_FREEZE_V2.json")
    (suite / "UAM_BASELINE_FREEZE_V2.sha256").write_text(
        f"{digest}  UAM_BASELINE_FREEZE_V2.json\n", encoding="utf-8"
    )
    return result


def run_stage2(stack: SimpleNamespace, suite: Path, freeze_path: Path, device: str, fail_fast: bool, skip_cpu_analytical: bool) -> Dict[str, Any]:
    banner("STAGE II | STRICTLY CONSUME STAGE-I FREEZE -> 11 RL BASELINES")
    freeze = verify_freeze(freeze_path)
    stack.methods.install_hooks()  # defensive re-install after CPU baseline helpers
    profile = stack.core.profile_from_freeze(freeze)
    plan = stage2_plan(stack)

    manifest = {
        "freeze_file": str(freeze_path), "freeze_sha256": freeze["_sha256"],
        "profile": profile_dict(profile), "cells": [asdict(c) for c in plan],
        "planned_gpu_transitions": EXPECTED_STAGE2_GPU,
        "no_fallback": True, "no_method_substitution": True,
        "joint_carriers": ["J3", "J4"],
    }
    manifest_path = suite / "STAGE2" / "STAGE2_MANIFEST.json"
    old_manifest = read_json(manifest_path, None)
    if old_manifest is not None and old_manifest != manifest:
        raise RuntimeError("Stage-II manifest differs from existing suite; refusing mixed experiment")
    write_json(manifest_path, manifest)

    analytical = (
        {"best": None, "methods": [], "skipped": True}
        if skip_cpu_analytical else stage2_cpu_analytical(stack, suite, freeze)
    )
    # CPU analytical helpers may restore process hooks; re-install before RL.
    stack.methods.install_hooks()

    rl_root = suite / "STAGE2" / "RL"
    failures = []
    if device == "cuda" and not stack.torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    for idx, cell in enumerate(plan, 1):
        print(f"[STAGE2 TRAIN {idx:02d}/{len(plan):02d}] {cell.key}", flush=True)
        try:
            stack.core.run_cell(
                rl_root, cell, freeze, profile, device,
                retry_failed=True,
            )
        except Exception as exc:
            failures.append({"cell": cell.key, "error": repr(exc), "traceback": traceback.format_exc()})
            write_json(suite / "STAGE2" / "training_failures.json", failures)
            print(f"[FAILED] {cell.key}: {exc!r}", flush=True)
            if fail_fast:
                raise

    incomplete = []
    for cell in plan:
        end = read_json(stack.core.cell_dir(rl_root, cell) / "run_end.json", {})
        if end.get("status") != "COMPLETE":
            incomplete.append(cell.key)
    if incomplete:
        raise RuntimeError(
            f"Stage II has {len(incomplete)} incomplete cells; same --resume-root will continue them. "
            f"No replacement/fallback was used. First few: {incomplete[:5]}"
        )

    model_summaries = []
    for idx, cell in enumerate(plan, 1):
        print(f"[STAGE2 EVAL {idx:02d}/{len(plan):02d}] {cell.key}", flush=True)
        model_summaries.append(evaluate_stage2_model(stack, suite, cell, freeze))
        write_csv(suite / "STAGE2" / "MODEL_SUMMARIES.csv", model_summaries)

    return aggregate_stage2(suite, plan, model_summaries, analytical, freeze)


# -----------------------------------------------------------------------------
# Stage I driver
# -----------------------------------------------------------------------------

def run_stage1(stack: SimpleNamespace, suite: Path, patch_info: Dict[str, Any], device: str, fail_fast: bool) -> Tuple[Path, Dict[str, Any]]:
    banner("STAGE I | FREEZE ENVIRONMENT / PPO / LOAD / DEMAND")
    reward_audit = runtime_reward_audit(stack, suite)
    print(f"[P0 PASS] reward identity gap={reward_audit['absolute_gap']:.3e}", flush=True)

    batch = stage1_batch(stack, suite, device, fail_fast)
    p = batch["selected_profile"]
    profile = stack.v3.SpeedProfile(str(p["name"]), int(p["n_envs"]), int(p["n_steps"]), int(p["batch_size"]))
    print(f"[S1 BATCH] selected={p}", flush=True)

    load = stage1_load(stack, suite, profile, device, fail_fast)
    print(f"[S1 LOAD] selected={load['selected_load']:.2f} | {load['selected_trace']}", flush=True)

    demand = stage1_demand(stack, suite, profile, str(load["selected_trace"]), device, fail_fast)
    print(f"[S1 DEMAND] selected={demand['selected_protocol']}", flush=True)

    freeze_path, freeze = write_stage1_freeze(
        stack, suite, patch_info, reward_audit, batch, load, demand,
    )
    print(f"[STAGE I LOCKED] {freeze_path} | sha256={sha256(freeze_path)}", flush=True)
    if EXPECTED_STAGE1_GPU != 14_745_600:
        raise RuntimeError("internal Stage-I budget invariant failed")
    return freeze_path, freeze


# -----------------------------------------------------------------------------
# Immutable suite contract
# -----------------------------------------------------------------------------

def lock_suite_contract(suite: Path, patch_info: Dict[str, Any]) -> None:
    contract = {
        "schema": "UAM_STAGE12_SUITE_V2",
        "script_sha256": sha256(Path(__file__).resolve()),
        "git_commit": current_git_commit(ROOT),
        "reward_source_hashes": patch_info["source_hashes_after"],
        "plan": plan_dict(),
        "no_stage2_fallback": True,
        "no_method_substitution": True,
    }
    path = suite / "STAGE12_SUITE_CONTRACT.json"
    old = read_json(path, None)
    if old is not None and old != contract:
        raise RuntimeError(
            "existing suite contract differs from this script/runtime; use the original script/config "
            "or start a new --output-root"
        )
    write_json(path, contract)


# -----------------------------------------------------------------------------
# Plan / CLI
# -----------------------------------------------------------------------------

def plan_dict() -> Dict[str, Any]:
    stage2_methods = []
    for lane, specs in STAGE2_LANES.items():
        for method, env in specs:
            stage2_methods.append({"lane": lane, "method": method, "env": env})
    return {
        "rollout": ROLLOUT,
        "stage1": {
            "batch": {"cells": 18, "steps_per_cell": STAGE1_STEPS, "gpu": 18 * STAGE1_STEPS},
            "load": {"cells": 12, "steps_per_cell": STAGE1_STEPS, "gpu": 12 * STAGE1_STEPS,
                     "cpu_analytical_episodes": 90},
            "demand": {"cells": 18, "steps_per_cell": STAGE1_STEPS, "gpu": 18 * STAGE1_STEPS},
            "total_gpu": EXPECTED_STAGE1_GPU,
        },
        "stage2": {
            "methods": stage2_methods,
            "n_methods": 11, "training_seeds": list(STAGE2_SEEDS),
            "steps_per_model": STAGE2_STEPS,
            "cells": 55, "gpu": EXPECTED_STAGE2_GPU,
            "cpu_analytical_validation_episodes": 4 * 6 * 3,
        },
        "grand_total_gpu": EXPECTED_TOTAL_GPU,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--resume-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument("--stage2-only", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--no-auto-patch", action="store_true",
                    help="require reward fix to already exist; never edit source")
    ap.add_argument("--skip-cpu-formal", action="store_true",
                    help="debug only: skip Stage-II formal CPU analytical validation")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage1_only and args.stage2_only:
        raise ValueError("--stage1-only and --stage2-only are mutually exclusive")
    if args.plan_only:
        print(json.dumps(plan_dict(), ensure_ascii=False, indent=2))
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_root:
        suite = args.resume_root.expanduser().resolve()
        if not suite.exists():
            raise FileNotFoundError(suite)
    elif args.output_root:
        suite = args.output_root.expanduser().resolve()
    else:
        suite = (ROOT / "serial_runs" / f"uam_stage12_v2_{stamp}").resolve()
    suite.mkdir(parents=True, exist_ok=True)

    # Critical: patch/verify source BEFORE any project module import.
    patch_info = ensure_reward_contract(ROOT, auto_patch=not args.no_auto_patch)
    write_json(suite / "P0_SOURCE_PATCH.json", patch_info)
    lock_suite_contract(suite, patch_info)

    stack = load_stack()
    if args.device == "cuda" and not stack.torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; script will not silently downgrade to CPU")

    print("=" * 126)
    print("UAM STAGE-I -> STAGE-II AUTOMATIC PIPELINE V2")
    print(f"suite             : {suite}")
    print(f"git               : {current_git_commit(ROOT)}")
    print(f"Stage-I GPU       : {EXPECTED_STAGE1_GPU:,}")
    print(f"Stage-II GPU      : {EXPECTED_STAGE2_GPU:,}")
    print(f"TOTAL GPU         : {EXPECTED_TOTAL_GPU:,}")
    print("Stage-II methods  : 11 = Single(3) + J3(4) + J4(4)")
    print("NO FALLBACK       : Stage II cannot start without Stage-I JSON+SHA lock")
    print("=" * 126, flush=True)

    freeze_path = suite / "UAM_FINAL_FREEZE_V2.json"
    if not args.stage2_only:
        freeze_path, _ = run_stage1(stack, suite, patch_info, args.device, args.fail_fast)
        if args.stage1_only:
            print(f"[DONE] Stage I only. Freeze={freeze_path}")
            return 0
    else:
        if not freeze_path.is_file():
            raise FileNotFoundError(
                f"--stage2-only requires the same suite's Stage-I freeze: {freeze_path}"
            )

    # Automatic hand-off.  This call re-reads and cryptographically verifies the
    # Stage-I file; it never trusts in-memory defaults from Stage I.
    result = run_stage2(
        stack, suite, freeze_path, args.device, args.fail_fast,
        skip_cpu_analytical=args.skip_cpu_formal,
    )
    print("\n" + "=" * 126)
    print("STAGE I + STAGE II COMPLETE")
    print(f"suite          : {suite}")
    print(f"freeze         : {freeze_path}")
    print(f"baseline freeze: {suite / 'UAM_BASELINE_FREEZE_V2.json'}")
    print(f"planned GPU    : {EXPECTED_TOTAL_GPU:,}")
    print(f"baseline rows  : {len(result.get('methods', []))}")
    print("=" * 126, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
