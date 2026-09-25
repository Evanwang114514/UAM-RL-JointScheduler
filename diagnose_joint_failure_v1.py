# -*- coding: utf-8 -*-
r"""
ZERO-TRAINING JOINT FAILURE DIAGNOSTIC
======================================

Purpose
-------
Use already-trained J0/J1/J2/J3 checkpoints from train_uam_7x12_600k_v2.py to
distinguish:

A) bad learned joint policy / passenger-aircraft coordination failure
B) JointControlWrapper / dispatch / physical-lifecycle failure
C) simply very slow but still draining policy

For every selected cell we run the SAME checkpoint under four controllers:

  LEARNED
      original deterministic joint policy.

  FIX_AIRCRAFT
      keep learned passenger choice, replace aircraft branch with a simple
      backlog/supply-pressure rescue controller.

  FIX_PASSENGER
      keep learned aircraft choice, replace passenger branch with a simple
      low-pressure passenger routing controller.

  HEURISTIC_JOINT
      replace both branches with the same transparent pressure heuristics.

If LEARNED stalls but FIX_AIRCRAFT or HEURISTIC_JOINT completes, the physical
environment and Joint wrapper can drain the same demand: failure is mainly
policy/coordination. If even HEURISTIC_JOINT cannot drain and dispatch attempts
fail despite eligible hub aircraft, the Joint implementation/physics is
suspicious.

NO PPO TRAINING IS PERFORMED.

Expected repository file:
  train_uam_7x12_600k_v2.py

Typical command:
  python diagnose_joint_failure_v1.py ^
      --run-root "serial_runs\uam_7x12_600k_seed1_20260922_012649" ^
      --cells "J0__M0,J0__M8,J1__M0,J2__M0,J3__M0" ^
      --checkpoint-mode final ^
      --seeds 123 ^
      --max-time 2500

Auto mode:
  python diagnose_joint_failure_v1.py ^
      --run-root "serial_runs\uam_7x12_600k_seed1_20260922_012649" ^
      --cells auto --max-cells 12 --checkpoint-mode final --seeds 123
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


# ---------------------------------------------------------------------------
# Generic I/O
# ---------------------------------------------------------------------------

def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(np.asarray(x).reshape(-1)[0])
        return y if math.isfinite(y) else default
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {}
            for k in fields:
                v = r.get(k, "")
                if isinstance(v, (dict, list, tuple, np.ndarray)):
                    v = json.dumps(v, ensure_ascii=False, default=str)
                out[k] = v
            w.writerow(out)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Source / checkpoint discovery
# ---------------------------------------------------------------------------

def import_runner(name: str):
    if name.endswith(".py"):
        name = Path(name).stem
    return importlib.import_module(name)


def parse_cell(cell: str) -> Tuple[str, str]:
    m = re.fullmatch(r"(J[0-3])__(M(?:[0-9]|1[01]))", cell.upper())
    if not m:
        raise ValueError(f"Invalid cell name: {cell}")
    return m.group(1), m.group(2)


def list_joint_cells(run_root: Path) -> List[str]:
    out = []
    for p in run_root.iterdir():
        if p.is_dir() and re.fullmatch(r"J[0-3]__M(?:[0-9]|1[01])", p.name.upper()):
            if (p / "checkpoints").exists():
                out.append(p.name.upper())
    return sorted(out)


def curve_badness(cell_dir: Path) -> Tuple[float, float]:
    """Lower max completion => more suspicious. Higher ATT used only as tiebreak."""
    rows = read_csv(cell_dir / "analysis" / "checkpoint_curve.csv")
    comps = [fnum(r.get("completion_rate_mean")) for r in rows]
    comps = [x for x in comps if math.isfinite(x)]
    atts = [fnum(r.get("ATT_mean")) for r in rows]
    atts = [x for x in atts if math.isfinite(x)]
    best_comp = max(comps) if comps else -1.0
    worst_att = max(atts) if atts else float("inf")
    return best_comp, worst_att


def auto_select_cells(run_root: Path, max_cells: int) -> List[str]:
    cells = list_joint_cells(run_root)
    if not cells:
        return []
    # Prioritize cells which never achieved full completion.
    ranked = sorted(
        cells,
        key=lambda c: (
            curve_badness(run_root / c)[0],       # low completion first
            -curve_badness(run_root / c)[1],      # high ATT first
            c,
        )
    )
    selected = ranked[:max_cells]

    # Ensure at least one control from each joint architecture if available.
    for j in ("J0", "J1", "J2", "J3"):
        if any(x.startswith(j + "__") for x in selected):
            continue
        pool = [x for x in ranked if x.startswith(j + "__")]
        if pool:
            if len(selected) >= max_cells:
                selected[-1] = pool[0]
            else:
                selected.append(pool[0])
    return list(dict.fromkeys(selected))[:max_cells]


def available_steps(cell_dir: Path) -> List[int]:
    out = []
    for p in (cell_dir / "checkpoints").glob("uam_ppo_*_steps.zip"):
        m = re.search(r"uam_ppo_(\d+)_steps\.zip$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def choose_step(cell_dir: Path, mode: str, explicit_step: Optional[int]) -> int:
    steps = available_steps(cell_dir)
    if not steps:
        raise FileNotFoundError(f"No checkpoint zips: {cell_dir}")
    if explicit_step is not None:
        if explicit_step not in steps:
            raise ValueError(f"{cell_dir.name}: step={explicit_step} not in {steps}")
        return explicit_step

    mode = mode.lower()
    if mode == "final":
        return max(steps)

    curve = read_csv(cell_dir / "analysis" / "checkpoint_curve.csv")
    by_step = {int(float(r["train_step"])): r for r in curve if r.get("train_step")}

    if mode == "best":
        valid = [
            (s, r) for s, r in by_step.items()
            if s in steps
            and fnum(r.get("completion_rate_mean"), 0.0) >= 0.999999
            and math.isfinite(fnum(r.get("ATT_mean")))
        ]
        if valid:
            return min(valid, key=lambda sr: fnum(sr[1].get("ATT_mean")))[0]
        return max(steps)

    if mode == "worst-completion":
        vals = [
            (s, r) for s, r in by_step.items()
            if s in steps and math.isfinite(fnum(r.get("completion_rate_mean")))
        ]
        if vals:
            return min(vals, key=lambda sr: fnum(sr[1].get("completion_rate_mean")))[0]
        return max(steps)

    raise ValueError(f"Unsupported checkpoint mode: {mode}")


# ---------------------------------------------------------------------------
# Scenario introspection
# ---------------------------------------------------------------------------

def sim_time(scenario: Any, fallback: int) -> float:
    for obj_name in ("env", "simpy_env"):
        obj = getattr(scenario, obj_name, None)
        now = getattr(obj, "now", None)
        if now is not None:
            try:
                return float(now)
            except Exception:
                pass
    for name in ("current_time", "time", "sim_time"):
        x = getattr(scenario, name, None)
        if x is not None and not callable(x):
            try:
                return float(x)
            except Exception:
                pass
    return float(fallback)


def persons_and_finished(scenario: Any) -> Tuple[int, int]:
    persons_obj = getattr(scenario, "persons", None)
    persons = getattr(persons_obj, "persons", {}) if persons_obj is not None else {}
    persons = persons or {}
    finished_raw = set(getattr(scenario, "finished_ids", []) or [])
    finished_str = {str(x) for x in finished_raw}

    n_finished = 0
    for pid_raw, person in persons.items():
        pid = str(pid_raw)
        done = (
            pid_raw in finished_raw
            or pid in finished_str
            or str(getattr(person, "state", "")).lower() == "finished"
        )
        if done:
            n_finished += 1
    return len(persons), n_finished


def queue_lengths(scenario: Any, candidates: Sequence[int]) -> Dict[int, int]:
    out = {}
    for vid in candidates:
        vp = scenario.vertiports.vertiport_list.get(str(vid))
        if vp is None:
            vp = scenario.vertiports.vertiport_list.get(int(vid))
        out[int(vid)] = len(list(getattr(vp, "person_list", []) or [])) if vp else 0
    return out


def local_ready_counts(exp: Any, scenario: Any, vids: Sequence[int]) -> Dict[int, int]:
    out = {}
    for vid in vids:
        local = list(scenario.vertiports.evtols_at_vertiport.get(str(vid), []) or [])
        ready = 0
        for e in local:
            try:
                state = str(exp.mx.state_name(e)).upper()
            except Exception:
                state = str(getattr(e, "state", "")).upper()
            try:
                pax = list(exp.mx.passenger_ids(e))
            except Exception:
                pax = list(getattr(e, "passenger_ids", []) or [])
            try:
                turn = bool(exp.base.is_turnaround_busy(e))
            except Exception:
                turn = False
            if state == "IDLE" and not pax and not turn:
                ready += 1
        out[int(vid)] = ready
    return out


def hub_eligible_count(exp: Any, scenario: Any) -> int:
    return local_ready_counts(exp, scenario, [int(exp.DESTINATION)])[int(exp.DESTINATION)]


def fleet_diag(scenario: Any) -> Dict[str, Any]:
    if hasattr(scenario, "get_fixed_fleet_diagnostics"):
        try:
            d = scenario.get_fixed_fleet_diagnostics() or {}
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {}


def pressure_state(exp: Any, scenario: Any) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, int]]:
    cands = [int(x) for x in exp.CANDIDATES]
    q = queue_lengths(scenario, cands)
    ready = local_ready_counts(exp, scenario, cands)
    p = {k: float(q[k]) / float(1 + ready[k]) for k in cands}
    return p, q, ready


def heuristic_passenger_action(exp: Any, scenario: Any) -> int:
    p, q, ready = pressure_state(exp, scenario)
    cands = [int(x) for x in exp.CANDIDATES]
    # Passenger chooses lower pressure. Stable tie -> first candidate.
    best_idx = min(range(len(cands)), key=lambda i: (p[cands[i]], q[cands[i]], i))
    return int(best_idx)


def heuristic_aircraft_action(exp: Any, scenario: Any) -> int:
    p, q, ready = pressure_state(exp, scenario)
    cands = [int(x) for x in exp.CANDIDATES]
    if hub_eligible_count(exp, scenario) <= 0:
        return 0
    # No backlog anywhere -> HOLD.
    if max(q.values()) <= 0:
        return 0
    target_idx = max(range(len(cands)), key=lambda i: (p[cands[i]], q[cands[i]], -i))
    return int(target_idx + 1)  # 1=V0, 2=V1


# ---------------------------------------------------------------------------
# Model / environment loading
# ---------------------------------------------------------------------------

def checkpoint_paths(cell_dir: Path, step: int) -> Tuple[Path, Path]:
    model = cell_dir / "checkpoints" / f"uam_ppo_{step}_steps.zip"
    vec = cell_dir / "checkpoints" / f"uam_ppo_vecnormalize_{step}_steps.pkl"
    if not model.exists():
        raise FileNotFoundError(model)
    if not vec.exists():
        raise FileNotFoundError(vec)
    return model, vec


def build_loaded(exp: Any, cell_dir: Path, env_key: str, method: str,
                 step: int, seed: int, max_time: int):
    # Override only for this diagnostic process. configure_worker_physics() in
    # the runner propagates the value to the imported physical modules.
    exp.HARD_GUARD = int(max_time)
    exp.configure_worker_physics()

    raw = exp.build_eval_env(
        env_key=env_key,
        method=method,
        run_dir=cell_dir / "_joint_diag_monitor",
    )
    model_path, vec_path = checkpoint_paths(cell_dir, step)
    env = VecNormalize.load(str(vec_path), raw)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env, device="cpu")
    model.policy.set_training_mode(False)

    try:
        env.seed(int(seed))
    except Exception:
        pass

    obs = env.reset()
    return env, model, obs, model_path, vec_path


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

CONTROLLERS = ("LEARNED", "FIX_AIRCRAFT", "FIX_PASSENGER", "HEURISTIC_JOINT")


def learned_action(model: PPO, obs: np.ndarray) -> int:
    a, _ = model.predict(obs, deterministic=True)
    return int(np.asarray(a).reshape(-1)[0])


def policy_probs(model: PPO, obs: np.ndarray) -> np.ndarray:
    try:
        with torch.no_grad():
            x, _ = model.policy.obs_to_tensor(obs)
            dist = model.policy.get_distribution(x).distribution
            if getattr(dist, "probs", None) is not None:
                p = dist.probs
            else:
                p = torch.softmax(dist.logits, dim=-1)
            return np.asarray(p.detach().cpu().numpy(), dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def run_rollout(
    *,
    exp: Any,
    cell_dir: Path,
    env_key: str,
    method: str,
    step: int,
    seed: int,
    max_time: int,
    controller: str,
    sample_every: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    exp.seed_all(seed)
    env = model = None

    action_counts = Counter()
    passenger_counts = Counter()
    aircraft_counts = Counter()
    dispatch_attempts = 0
    dispatch_success = 0
    dispatch_fail_no_eligible = 0
    dispatch_fail_despite_eligible = 0
    pair_consistency = 0
    pair_nonhold = 0

    queue_starve_run = {int(k): 0 for k in exp.CANDIDATES}
    queue_starve_max = {int(k): 0 for k in exp.CANDIDATES}

    timeline: List[Dict[str, Any]] = []
    finished_hist: List[Tuple[float, int, int]] = []

    terminal_snapshot = None
    done_reason = "UNKNOWN"
    reward_sum = 0.0

    try:
        env, model, obs, model_path, vec_path = build_loaded(
            exp, cell_dir, env_key, method, step, seed, max_time
        )

        done = np.asarray([False])
        episode_steps = 0
        initial_diag = {}

        while not bool(done[0]):
            scenario = exp.mx.find_scenario(env)
            t = sim_time(scenario, episode_steps)
            n_persons, n_finished = persons_and_finished(scenario)
            finished_hist.append((t, n_persons, n_finished))

            pstate, q_before, ready_before = pressure_state(exp, scenario)
            eligible_before = hub_eligible_count(exp, scenario)

            # Learned proposal is always available and is used in the two
            # partial-ablation controllers.
            lai = learned_action(model, obs)
            lp = lai // 3
            la = lai % 3

            if controller == "LEARNED":
                pa, aa = lp, la
            elif controller == "FIX_AIRCRAFT":
                pa, aa = lp, heuristic_aircraft_action(exp, scenario)
            elif controller == "FIX_PASSENGER":
                pa, aa = heuristic_passenger_action(exp, scenario), la
            elif controller == "HEURISTIC_JOINT":
                pa = heuristic_passenger_action(exp, scenario)
                aa = heuristic_aircraft_action(exp, scenario)
            else:
                raise ValueError(controller)

            ai = int(pa * 3 + aa)
            action_counts[ai] += 1
            passenger_counts[pa] += 1
            aircraft_counts[aa] += 1
            if aa != 0:
                dispatch_attempts += 1
                pair_nonhold += 1
                if aa == pa + 1:
                    pair_consistency += 1

            probs = policy_probs(model, obs) if controller == "LEARNED" else np.asarray([])

            obs, reward, done, infos = env.step(np.asarray([ai], dtype=np.int64))
            reward_sum += fnum(np.asarray(reward).reshape(-1)[0], 0.0)
            episode_steps += 1

            info = infos[0] if infos else {}
            if not isinstance(info, dict):
                info = {}

            dispatched = bool(info.get("aircraft_dispatched", False))
            if aa != 0:
                if dispatched:
                    dispatch_success += 1
                elif eligible_before <= 0:
                    dispatch_fail_no_eligible += 1
                else:
                    dispatch_fail_despite_eligible += 1

            if "terminal_snapshot" in info:
                terminal_snapshot = info["terminal_snapshot"]

            # Queue-starvation diagnostic: passenger queue exists but there is
            # no immediately ready service aircraft at that origin.
            scenario_after = exp.mx.find_scenario(env)
            q_after = queue_lengths(scenario_after, [int(x) for x in exp.CANDIDATES])
            ready_after = local_ready_counts(exp, scenario_after, [int(x) for x in exp.CANDIDATES])
            for k in q_after:
                if q_after[k] > 0 and ready_after[k] <= 0:
                    queue_starve_run[k] += 1
                    queue_starve_max[k] = max(queue_starve_max[k], queue_starve_run[k])
                else:
                    queue_starve_run[k] = 0

            if episode_steps == 1:
                initial_diag = fleet_diag(scenario_after)

            if episode_steps % sample_every == 0 or bool(done[0]) or episode_steps <= 5:
                nn, nf = persons_and_finished(scenario_after)
                row = {
                    "cell": cell_dir.name,
                    "env_key": env_key,
                    "method": method,
                    "step": step,
                    "seed": seed,
                    "controller": controller,
                    "episode_step": episode_steps,
                    "sim_time": sim_time(scenario_after, episode_steps),
                    "n_generated": nn,
                    "n_finished": nf,
                    "active": max(0, nn - nf),
                    "q_V0": q_after.get(0, 0),
                    "q_V1": q_after.get(1, 0),
                    "ready_V0": ready_after.get(0, 0),
                    "ready_V1": ready_after.get(1, 0),
                    "hub_eligible": hub_eligible_count(exp, scenario_after),
                    "learned_joint_action": lai,
                    "used_joint_action": ai,
                    "passenger_action": pa,
                    "aircraft_action": aa,
                    "dispatch_success_this_step": dispatched,
                    "learned_max_prob": float(np.max(probs)) if probs.size else float("nan"),
                    "fleet_diag": fleet_diag(scenario_after),
                }
                timeline.append(row)

            # Independent diagnostic wall: never depend only on env internals.
            if episode_steps >= int(max_time) + 5:
                done_reason = "DIAGNOSTIC_GUARD"
                break

        scenario = exp.mx.find_scenario(env)
        final_t = sim_time(scenario, episode_steps)
        n_persons, n_finished = persons_and_finished(scenario)
        if terminal_snapshot is not None:
            try:
                rows = terminal_snapshot.get("rows", []) or []
                if rows:
                    n_persons = int(terminal_snapshot.get("n_persons", len(rows)))
                    n_finished = sum(1 for r in rows if bool(r.get("finished")))
            except Exception:
                pass

        completion = n_finished / n_persons if n_persons else float("nan")

        if bool(done[0]):
            if completion >= 0.999999:
                done_reason = "FULL_COMPLETION"
            elif episode_steps >= max_time - 2:
                done_reason = "HARD_GUARD_INCOMPLETE"
            else:
                done_reason = "ENV_TERMINATED_INCOMPLETE"

        # Completion-time summaries use the final generated population as denominator.
        def t_fraction(frac: float) -> float:
            if n_persons <= 0:
                return float("nan")
            threshold = frac * n_persons
            for tt, _, ff in finished_hist:
                if ff >= threshold:
                    return float(tt)
            return float("nan")

        # Progress windows make plateaus visible.
        def finished_at(tcut: float) -> int:
            vals = [(tt, ff) for tt, _, ff in finished_hist if tt <= tcut]
            return vals[-1][1] if vals else 0

        summary = {
            "cell": cell_dir.name,
            "env_key": env_key,
            "method": method,
            "train_step": step,
            "eval_seed": seed,
            "controller": controller,
            "max_time": max_time,
            "done_reason": done_reason,
            "sim_time_final": final_t,
            "episode_steps": episode_steps,
            "n_generated_final": n_persons,
            "n_finished_final": n_finished,
            "completion_rate": completion,
            "backlog_final": n_persons - n_finished,
            "reward_sum": reward_sum,
            "t50": t_fraction(0.50),
            "t90": t_fraction(0.90),
            "t99": t_fraction(0.99),
            "t100": t_fraction(1.00),
            "finished_t300": finished_at(300),
            "finished_t600": finished_at(600),
            "finished_t900": finished_at(900),
            "finished_t1200": finished_at(1200),
            "finished_t1500": finished_at(1500),
            "finished_t2000": finished_at(2000),
            "finished_t2500": finished_at(2500),
            "passenger_V0_share": passenger_counts[0] / max(1, sum(passenger_counts.values())),
            "passenger_V1_share": passenger_counts[1] / max(1, sum(passenger_counts.values())),
            "aircraft_HOLD_share": aircraft_counts[0] / max(1, sum(aircraft_counts.values())),
            "aircraft_V0_share": aircraft_counts[1] / max(1, sum(aircraft_counts.values())),
            "aircraft_V1_share": aircraft_counts[2] / max(1, sum(aircraft_counts.values())),
            "dispatch_attempts": dispatch_attempts,
            "dispatch_success": dispatch_success,
            "dispatch_success_rate": dispatch_success / max(1, dispatch_attempts),
            "dispatch_fail_no_eligible": dispatch_fail_no_eligible,
            "dispatch_fail_despite_eligible": dispatch_fail_despite_eligible,
            "dispatch_fail_despite_eligible_rate": (
                dispatch_fail_despite_eligible / max(1, dispatch_attempts)
            ),
            "pair_consistency_nonhold": pair_consistency / max(1, pair_nonhold),
            "max_supply_starvation_V0": queue_starve_max.get(0, 0),
            "max_supply_starvation_V1": queue_starve_max.get(1, 0),
            "initial_fleet_diag": initial_diag,
            "final_fleet_diag": fleet_diag(scenario),
            "model_path": str(model_path),
            "vec_path": str(vec_path),
        }
        return summary, timeline

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            exp.core.restore_process_patches()
        except Exception:
            pass
        gc.collect()


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def verdict_for_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by = {str(r["controller"]): r for r in rows}
    L = by.get("LEARNED")
    A = by.get("FIX_AIRCRAFT")
    P = by.get("FIX_PASSENGER")
    H = by.get("HEURISTIC_JOINT")
    if not L:
        return {"verdict": "NO_LEARNED_RESULT", "confidence": "LOW"}

    lc = fnum(L.get("completion_rate"), 0.0)
    ac = fnum(A.get("completion_rate"), 0.0) if A else float("nan")
    pc = fnum(P.get("completion_rate"), 0.0) if P else float("nan")
    hc = fnum(H.get("completion_rate"), 0.0) if H else float("nan")

    if lc >= 0.999999:
        return {
            "verdict": "LEARNED_COMPLETES",
            "confidence": "HIGH",
            "explanation": "This checkpoint does not show the unfinished-passenger failure under the diagnostic horizon.",
        }

    # Strong implementation signal: even a transparent controller asks to
    # dispatch while eligible hub aircraft exist, but wrapper rejects often.
    if H and hc < 0.999999 and fnum(H.get("dispatch_fail_despite_eligible_rate"), 0.0) > 0.10:
        return {
            "verdict": "JOINT_DISPATCH_OR_PHYSICS_SUSPECT",
            "confidence": "HIGH",
            "explanation": "Heuristic joint control also fails, with >10% dispatch attempts rejected despite eligible hub aircraft.",
        }

    if A and ac >= 0.999999 and lc < 0.999999:
        return {
            "verdict": "AIRCRAFT_BRANCH_OR_COORDINATION_FAILURE",
            "confidence": "HIGH",
            "explanation": "Same learned passenger choices complete after replacing only the aircraft branch.",
        }

    if P and pc >= 0.999999 and lc < 0.999999:
        return {
            "verdict": "PASSENGER_BRANCH_FAILURE",
            "confidence": "HIGH",
            "explanation": "Same learned aircraft choices complete after replacing only passenger routing.",
        }

    if H and hc >= 0.999999 and lc < 0.999999:
        return {
            "verdict": "LEARNED_JOINT_POLICY_FAILURE",
            "confidence": "HIGH",
            "explanation": "Transparent passenger+aircraft heuristics drain the same Joint environment, while the learned policy does not.",
        }

    if H and hc > lc + 0.25:
        return {
            "verdict": "LIKELY_POLICY_FAILURE_BUT_NOT_FULLY_RESCUED",
            "confidence": "MEDIUM",
            "explanation": "Heuristic joint control materially raises completion but still does not fully drain by the diagnostic horizon.",
        }

    if H and hc < 0.999999:
        return {
            "verdict": "JOINT_ENV_OR_CAPACITY_SUSPECT_INCONCLUSIVE",
            "confidence": "MEDIUM",
            "explanation": "Even the transparent joint heuristic cannot fully drain. Inspect fleet_diag, dispatch failures, and timeline before blaming PPO.",
        }

    return {
        "verdict": "INCONCLUSIVE",
        "confidence": "LOW",
        "explanation": "Controller substitutions do not cleanly isolate the failure.",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-root",
        required=True,
        help="Existing uam_7x12 run root, e.g. serial_runs\\uam_7x12_600k_seed1_...",
    )
    p.add_argument("--runner", default="train_uam_7x12_600k_v2")
    p.add_argument(
        "--cells",
        default="auto",
        help='Comma-separated cells, e.g. "J0__M0,J0__M8,J1__M0"; or auto.',
    )
    p.add_argument("--max-cells", type=int, default=12)
    p.add_argument(
        "--checkpoint-mode",
        choices=["final", "best", "worst-completion"],
        default="final",
    )
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--seeds", default="123")
    p.add_argument("--max-time", type=int, default=2500)
    p.add_argument("--sample-every", type=int, default=25)
    p.add_argument(
        "--controllers",
        default="LEARNED,FIX_AIRCRAFT,FIX_PASSENGER,HEURISTIC_JOINT",
    )
    p.add_argument("--output", default=None)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    if not run_root.exists():
        raise FileNotFoundError(run_root)

    exp = import_runner(args.runner)
    if hasattr(exp, "assert_p0_patch"):
        exp.assert_p0_patch()

    if str(args.cells).strip().lower() == "auto":
        cells = auto_select_cells(run_root, int(args.max_cells))
    else:
        cells = [x.strip().upper() for x in str(args.cells).split(",") if x.strip()]
    if not cells:
        raise RuntimeError("No Joint cells selected/found.")

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    controllers = [x.strip().upper() for x in str(args.controllers).split(",") if x.strip()]
    bad = [x for x in controllers if x not in CONTROLLERS]
    if bad:
        raise ValueError(f"Unknown controllers={bad}; allowed={CONTROLLERS}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = (
        Path(args.output).resolve()
        if args.output
        else run_root / f"joint_failure_diagnostic_{stamp}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_root": str(run_root),
        "runner": args.runner,
        "cells": cells,
        "seeds": seeds,
        "controllers": controllers,
        "checkpoint_mode": args.checkpoint_mode,
        "explicit_step": args.step,
        "diagnostic_max_time": args.max_time,
        "sample_every": args.sample_every,
        "note": "ZERO PPO TRAINING; controller substitution diagnosis.",
        "joint_physics_from_runner": {
            k: str(getattr(exp, "ENV_SPECS", {}).get(k, ""))
            for k in ("J0", "J1", "J2", "J3")
        },
    }
    write_json(out_root / "manifest.json", manifest)

    summaries: List[Dict[str, Any]] = []
    timelines: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    total_jobs = len(cells) * len(seeds) * len(controllers)
    job = 0

    print("=" * 118)
    print("ZERO-TRAINING JOINT FAILURE DIAGNOSTIC")
    print(f"run_root={run_root}")
    print(f"cells={cells}")
    print(f"controllers={controllers} | seeds={seeds} | max_time={args.max_time}")
    print("=" * 118, flush=True)

    for cell in cells:
        env_key, method = parse_cell(cell)
        cell_dir = run_root / cell
        step = choose_step(cell_dir, args.checkpoint_mode, args.step)
        print(f"\nCELL {cell} | step={step:,}", flush=True)

        for seed in seeds:
            for controller in controllers:
                job += 1
                print(f"  [{job:03d}/{total_jobs:03d}] seed={seed} {controller}", flush=True)
                try:
                    summary, timeline = run_rollout(
                        exp=exp,
                        cell_dir=cell_dir,
                        env_key=env_key,
                        method=method,
                        step=step,
                        seed=seed,
                        max_time=int(args.max_time),
                        controller=controller,
                        sample_every=int(args.sample_every),
                    )
                    summaries.append(summary)
                    timelines.extend(timeline)
                    print(
                        f"      finish={summary['n_finished_final']}/{summary['n_generated_final']} "
                        f"({100*fnum(summary['completion_rate'],0):.1f}%) "
                        f"t={summary['sim_time_final']:.0f} "
                        f"hold={100*fnum(summary['aircraft_HOLD_share'],0):.1f}% "
                        f"dispatch_ok={100*fnum(summary['dispatch_success_rate'],0):.1f}%",
                        flush=True,
                    )
                except Exception as exc:
                    err = {
                        "cell": cell,
                        "step": step,
                        "seed": seed,
                        "controller": controller,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    errors.append(err)
                    print("      ERROR:", repr(exc), flush=True)
                    if not args.continue_on_error:
                        write_csv(out_root / "errors.csv", errors)
                        raise

                write_csv(out_root / "rollout_summary.csv", summaries)
                write_csv(out_root / "timeline.csv", timelines)
                write_csv(out_root / "errors.csv", errors)

    # Controller-substitution verdicts per cell/seed.
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in summaries:
        grouped[(str(r["cell"]), int(r["eval_seed"]))].append(r)

    verdict_rows = []
    for (cell, seed), rows in sorted(grouped.items()):
        v = verdict_for_group(rows)
        learned = next((r for r in rows if r["controller"] == "LEARNED"), {})
        fix_a = next((r for r in rows if r["controller"] == "FIX_AIRCRAFT"), {})
        fix_p = next((r for r in rows if r["controller"] == "FIX_PASSENGER"), {})
        heur = next((r for r in rows if r["controller"] == "HEURISTIC_JOINT"), {})
        verdict_rows.append({
            "cell": cell,
            "seed": seed,
            "verdict": v.get("verdict"),
            "confidence": v.get("confidence"),
            "explanation": v.get("explanation"),
            "learned_completion": learned.get("completion_rate"),
            "fix_aircraft_completion": fix_a.get("completion_rate"),
            "fix_passenger_completion": fix_p.get("completion_rate"),
            "heuristic_joint_completion": heur.get("completion_rate"),
            "learned_hold_share": learned.get("aircraft_HOLD_share"),
            "learned_dispatch_success_rate": learned.get("dispatch_success_rate"),
            "learned_dispatch_fail_despite_eligible_rate": learned.get("dispatch_fail_despite_eligible_rate"),
            "learned_starvation_V0": learned.get("max_supply_starvation_V0"),
            "learned_starvation_V1": learned.get("max_supply_starvation_V1"),
        })

    write_csv(out_root / "VERDICT.csv", verdict_rows)

    # Aggregate verdict counts.
    counts = Counter(str(r.get("verdict")) for r in verdict_rows)
    write_json(
        out_root / "RUN_COMPLETE.json",
        {
            "status": "COMPLETE",
            "output": str(out_root),
            "n_rollouts": len(summaries),
            "n_errors": len(errors),
            "verdict_counts": dict(counts),
            "files": [
                "VERDICT.csv",
                "rollout_summary.csv",
                "timeline.csv",
                "errors.csv",
                "manifest.json",
            ],
        },
    )

    print("\n" + "=" * 118)
    print("DONE")
    print(f"Output : {out_root}")
    print(f"Verdict: {out_root / 'VERDICT.csv'}")
    print(f"Summary: {out_root / 'rollout_summary.csv'}")
    print(f"Timeline:{out_root / 'timeline.csv'}")
    print("Verdict counts:", dict(counts))
    print("=" * 118)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
