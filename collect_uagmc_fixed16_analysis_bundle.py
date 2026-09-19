#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UAGMC fixed-fleet N=16 post-training COMPLETE analysis collector
================================================================

What this script does
---------------------
This is a POST-TRAINING evaluator/collector. It does not train or modify models.

For the completed E1 experiment:
    fixed conserved fleet N=16
    train seeds 0,1,2
    checkpoints every 50k
    nominal horizon 1,000,000 steps

it automatically:

1) finds every checkpoint (50k,100k,...,1M) and its own VecNormalize;
2) evaluates every checkpoint on eval seeds 123,124,125;
3) uses the frozen E1 environment:
       fleet_mode="conserved_closed_loop"
       fleet_size=16
       V0/V1 initial allocation = 12/4
       candidates=[0,1], destination=V2
4) records:
       ATT, Access, AWT, AFT, ATT residual
       P50/P90/P95/max travel time
       completion, unfinished backlog
       final queues
       service/reposition flight counts
       fixed-fleet final state/location diagnostics
       deterministic passenger-decision action shares
       mean policy probabilities P(V0), P(V1)
       policy entropy / normalized entropy
       mean confidence and probability margin
       episode reward / horizon
5) aggregates:
       eval-seed mean/std for every train-seed/checkpoint
       train-seed mean/std for every checkpoint
6) merges passive PPO training diagnostics from training_milestones.csv;
7) optionally evaluates the official original UAGMC final model in LEGACY mode;
8) copies the N=16 static fleet calibration files if present;
9) copies the historical E0 learning curve if already present;
10) creates plots + summary.txt;
11) creates ONE small ZIP to upload to ChatGPT.
    It intentionally does NOT include heavy model .zip checkpoint files.

Typical use
-----------
Put this file in:
    E:\\Study Files\\github\\UAM-predict\\UAGMC-main\\

Then simply run:

    python collect_uagmc_fixed16_analysis_bundle.py

It auto-detects the newest:
    serial_runs/uagmc_E1_fixed16_fast16env_3seed_1M_*

Or specify it explicitly:

    python collect_uagmc_fixed16_analysis_bundle.py ^
      --run-root "serial_runs\\uagmc_E1_fixed16_fast16env_3seed_1M_20260918_XXXXXX"

Default evaluation:
    train seeds: all discovered seed_* folders
    checkpoints: every 50k up to 1M
    eval seeds: 123,124,125
    passenger trace: train_data/passengers_300.csv
    evaluation max_time: 600
    deterministic policy
    inference device: cpu

Output
------
<run-root>/complete_analysis/
    analysis_manifest.json
    checkpoint_inventory.csv
    episode_metrics.csv
    curve_by_train_seed.csv
    curve_across_train_seeds.csv
    final_1m_by_seed.csv
    final_1m_across_seeds.csv
    training_diagnostics.csv
    official_legacy_reference.csv
    errors.csv
    summary.txt
    *.png
    source_logs/...
    static_reference/...
    historical_E0/...
    UPLOAD_THIS_analysis_bundle.zip

Upload ONLY:
    UPLOAD_THIS_analysis_bundle.zip

Scientific notes
----------------
- Primary policy evaluation is deterministic, matching prior UAGMC evaluation.
- Action shares/probabilities are measured at detected passenger-decision states.
  If the wrapper's waiting_decisions signal cannot be found, the script falls
  back to all environment steps and explicitly records that fallback.
- ATT is computed over completed passengers from UAGMC travel records.
  Therefore completion/backlog must always be read together with ATT.
- This evaluation uses max_time=600 by default to match the previous E0
  learning-curve evaluation and the fixed-fleet static calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utilss.make_env_fleet import make_env  # noqa: E402


# =============================================================================
# Frozen E1 evaluation configuration
# =============================================================================

FLEET_MODE = "conserved_closed_loop"
FLEET_SIZE = 16
CANDIDATES = [0, 1]
TO_VERTIPORT = 2

DEFAULT_EVAL_SEEDS = [123, 124, 125]
DEFAULT_EVAL_EVERY = 50_000
DEFAULT_MAX_STEP = 1_000_000
DEFAULT_MAX_TIME = 600
DEFAULT_PASSENGER_FILE = ROOT / "train_data" / "passengers_300.csv"


# =============================================================================
# Generic helpers
# =============================================================================

def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return default


def finite_mean(values: Iterable[Any]) -> float:
    arr = np.asarray([as_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def finite_std(values: Iterable[Any]) -> float:
    arr = np.asarray([as_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    if len(arr) == 1:
        return 0.0
    return float(arr.std(ddof=1))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(text: str) -> List[int]:
    out = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("integer list cannot be empty")
    return out


def jsonable(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    try:
        json.dumps(x)
        return x
    except Exception:
        return repr(x)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)

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

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            serial = {}
            for k, v in row.items():
                if isinstance(v, (dict, list, tuple, np.ndarray)):
                    serial[k] = json.dumps(jsonable(v), ensure_ascii=False)
                else:
                    serial[k] = v
            writer.writerow(serial)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# =============================================================================
# Run-root / checkpoint discovery
# =============================================================================

@dataclass(frozen=True)
class ModelSpec:
    source_group: str
    train_seed: Optional[int]
    train_step: Optional[int]
    model_path: Path
    vecnormalize_path: Path
    label: str
    env_mode: str


def auto_find_run_root() -> Path:
    serial = ROOT / "serial_runs"

    patterns = [
        "uagmc_E1_fixed16_fast16env_3seed_1M_*",
        "uagmc_E1_fixed16_fast16env_3seed_1m_*",
        "*E1*fixed16*3seed*1M*",
        "*fixed16*3seed*",
    ]

    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(
            p for p in serial.glob(pattern)
            if p.is_dir()
        )
        if candidates:
            break

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-detect the E1 run root under serial_runs/. "
            "Use --run-root explicitly."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def checkpoint_step(path: Path) -> Optional[int]:
    m = re.search(r"uam_ppo_(\d+)_steps", path.stem)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d+)_steps", path.stem)
    if m:
        return int(m.group(1))

    return None


def discover_checkpoints(
    run_root: Path,
    eval_every: int,
    max_step: int,
    wanted_train_seeds: Optional[Sequence[int]],
) -> List[ModelSpec]:
    wanted = set(wanted_train_seeds) if wanted_train_seeds else None
    specs: List[ModelSpec] = []

    seed_dirs = sorted(
        [
            p for p in run_root.glob("seed_*")
            if p.is_dir() and re.fullmatch(r"seed_\d+", p.name)
        ],
        key=lambda p: int(p.name.split("_")[-1]),
    )

    if not seed_dirs:
        raise FileNotFoundError(
            f"No seed_* folders found under {run_root}"
        )

    for seed_dir in seed_dirs:
        train_seed = int(seed_dir.name.split("_")[-1])

        if wanted is not None and train_seed not in wanted:
            continue

        ckpt_dir = seed_dir / "checkpoints"
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"Missing checkpoint directory: {ckpt_dir}"
            )

        for model_path in sorted(ckpt_dir.glob("uam_ppo_*_steps.zip")):
            step = checkpoint_step(model_path)
            if step is None:
                continue
            if step <= 0 or step > max_step:
                continue
            if step % eval_every != 0:
                continue

            vec_path = (
                ckpt_dir
                / f"uam_ppo_vecnormalize_{step}_steps.pkl"
            )

            if not vec_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint {model_path.name} has no matched "
                    f"VecNormalize: {vec_path}"
                )

            specs.append(
                ModelSpec(
                    source_group="E1_fixed16",
                    train_seed=train_seed,
                    train_step=step,
                    model_path=model_path.resolve(),
                    vecnormalize_path=vec_path.resolve(),
                    label=f"seed{train_seed}_{step}",
                    env_mode="fixed",
                )
            )

    specs.sort(
        key=lambda s: (
            -1 if s.train_seed is None else s.train_seed,
            -1 if s.train_step is None else s.train_step,
        )
    )

    if not specs:
        raise RuntimeError(
            "No checkpoint specs matched the requested filters."
        )

    return specs


def checkpoint_inventory(specs: Sequence[ModelSpec]) -> List[Dict[str, Any]]:
    return [
        {
            "source_group": s.source_group,
            "train_seed": s.train_seed,
            "train_step": s.train_step,
            "model_path": str(s.model_path),
            "model_bytes": s.model_path.stat().st_size,
            "vecnormalize_path": str(s.vecnormalize_path),
            "vecnormalize_bytes": s.vecnormalize_path.stat().st_size,
        }
        for s in specs
    ]


# =============================================================================
# Locate UAGMC scenario / decision signal
# =============================================================================

def find_scenario(env: Any):
    obj = env
    seen = set()

    for _ in range(50):
        if id(obj) in seen:
            break
        seen.add(id(obj))

        if all(
            hasattr(obj, key)
            for key in (
                "person_travel_records",
                "persons",
                "finished_ids",
            )
        ):
            return obj

        if hasattr(obj, "scenario"):
            sc = getattr(obj, "scenario")
            if sc is not None and all(
                hasattr(sc, key)
                for key in (
                    "person_travel_records",
                    "persons",
                    "finished_ids",
                )
            ):
                return sc

        if hasattr(obj, "env"):
            nxt = getattr(obj, "env")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue

        if hasattr(obj, "unwrapped"):
            nxt = getattr(obj, "unwrapped")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue

        break

    raise RuntimeError("Could not locate UAGMC Scenario.")


def waiting_decision_count(env: Any) -> Optional[int]:
    """
    Returns:
      integer >=0 if waiting_decisions signal is found;
      None if wrapper stack does not expose such a signal.
    """
    obj = env
    seen = set()

    for _ in range(50):
        if id(obj) in seen:
            break
        seen.add(id(obj))

        state = getattr(obj, "state", None)
        if isinstance(state, dict) and "waiting_decisions" in state:
            try:
                return len(state.get("waiting_decisions") or [])
            except Exception:
                pass

        if hasattr(obj, "waiting_decisions"):
            try:
                return len(getattr(obj, "waiting_decisions") or [])
            except Exception:
                pass

        if hasattr(obj, "env"):
            nxt = getattr(obj, "env")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue

        if hasattr(obj, "unwrapped"):
            nxt = getattr(obj, "unwrapped")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue

        break

    return None


# =============================================================================
# Terminal capture BEFORE DummyVecEnv auto-reset
# =============================================================================

class TerminalPassengerCapture(gym.Wrapper):
    def _snapshot(self) -> Dict[str, Any]:
        scenario = find_scenario(self.env)

        persons_obj = getattr(scenario, "persons", None)
        persons = getattr(persons_obj, "persons", {}) if persons_obj else {}
        persons = persons or {}

        records = getattr(scenario, "person_travel_records", {}) or {}
        finished_raw = set(
            getattr(scenario, "finished_ids", []) or []
        )
        finished_str = {str(x) for x in finished_raw}

        rows: List[Dict[str, Any]] = []

        for pid_raw, person in persons.items():
            pid = str(pid_raw)

            recs = (
                records.get(pid_raw)
                or records.get(pid)
                or []
            )
            last = recs[-1] if recs else {}

            start_time = last.get("start_time")
            end_time = last.get("end_time")

            travel_time = float("nan")
            if start_time is not None and end_time is not None:
                try:
                    travel_time = float(end_time) - float(start_time)
                except Exception:
                    pass

            stats = getattr(person, "time_stats", {}) or {}

            rows.append(
                {
                    "pid": pid,
                    "finished": bool(
                        pid_raw in finished_raw
                        or pid in finished_str
                        or end_time is not None
                    ),
                    "method": getattr(person, "method", None),
                    "state": str(getattr(person, "state", "")),
                    "sub_state": str(getattr(person, "sub_state", "")),
                    "from_vertiport": last.get("from"),
                    "to_vertiport": last.get("to"),
                    "start_time": start_time,
                    "end_time": end_time,
                    "travel_time": travel_time,
                    "access_time": as_float(
                        stats.get("to_vertiport", np.nan)
                    ),
                    "wait_uam_time": as_float(
                        stats.get("wait_uam", np.nan)
                    ),
                    "fly_time": as_float(
                        stats.get("fly", np.nan)
                    ),
                }
            )

        queue_counts: Dict[str, int] = {}
        try:
            for vid in CANDIDATES:
                vp = scenario.vertiports.vertiport_list[str(vid)]
                queue_counts[str(vid)] = len(
                    list(getattr(vp, "person_list", []))
                )
        except Exception:
            pass

        unfinished_states = Counter()
        unfinished_substates = Counter()

        for row in rows:
            if not row["finished"]:
                unfinished_states[str(row["state"])] += 1
                unfinished_substates[str(row["sub_state"])] += 1

        fixed_diag: Dict[str, Any] = {}
        if hasattr(scenario, "get_fixed_fleet_diagnostics"):
            try:
                fixed_diag = (
                    scenario.get_fixed_fleet_diagnostics()
                    or {}
                )
            except Exception:
                fixed_diag = {}

        return {
            "time": getattr(scenario, "time", None),
            "n_persons": len(persons),
            "n_finished_ids": len(finished_raw),
            "rows": rows,
            "queue_counts": queue_counts,
            "unfinished_state_counts": dict(unfinished_states),
            "unfinished_substate_counts": dict(unfinished_substates),
            "fixed_fleet_diagnostics": fixed_diag,
        }

    def step(self, action):
        result = self.env.step(action)

        if not isinstance(result, tuple):
            raise RuntimeError(
                f"Unexpected env.step return: {type(result)}"
            )

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            if bool(terminated) or bool(truncated):
                info = dict(info)
                info["uagmc_terminal_snapshot"] = self._snapshot()
            return obs, reward, terminated, truncated, info

        if len(result) == 4:
            obs, reward, done, info = result
            if bool(done):
                info = dict(info)
                info["uagmc_terminal_snapshot"] = self._snapshot()
            return obs, reward, done, info

        raise RuntimeError(
            f"Unexpected env.step tuple length: {len(result)}"
        )


# =============================================================================
# Evaluation environment
# =============================================================================

def make_eval_env(
    passenger_file: Path,
    max_time: int,
    vec_path: Path,
    monitor_dir: Path,
    env_mode: str,
):
    def _build():
        kwargs = dict(
            max_time=max_time,
            log_dir=monitor_dir,
            env_index=0,
            candidate_from_vertiports=CANDIDATES,
            to_vertiport=TO_VERTIPORT,
            person_spawn_file=str(passenger_file),
            enable_logger=False,
        )

        if env_mode == "fixed":
            kwargs.update(
                fleet_mode="conserved_closed_loop",
                fleet_size=FLEET_SIZE,
                fleet_assertions=True,
            )
        elif env_mode == "legacy":
            kwargs.update(
                fleet_mode="legacy_replenish",
                fleet_size=None,
                fleet_assertions=False,
            )
        else:
            raise ValueError(env_mode)

        return TerminalPassengerCapture(make_env(**kwargs)())

    raw = DummyVecEnv([_build])

    env = VecNormalize.load(
        str(vec_path),
        raw,
    )
    env.training = False
    env.norm_reward = False

    return env


def raw_single_env(vec_env) -> Any:
    obj = vec_env

    if hasattr(obj, "venv"):
        obj = obj.venv

    if hasattr(obj, "envs") and obj.envs:
        return obj.envs[0]

    return obj


# =============================================================================
# Policy-distribution diagnostics
# =============================================================================

def policy_probs(model: PPO, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor)
        base = getattr(dist, "distribution", None)

        if base is None:
            raise RuntimeError(
                "Cannot access underlying action distribution."
            )

        if hasattr(base, "probs") and base.probs is not None:
            probs = base.probs
        elif hasattr(base, "logits") and base.logits is not None:
            probs = torch.softmax(base.logits, dim=-1)
        else:
            raise RuntimeError(
                "Categorical action distribution exposes neither probs nor logits."
            )

        arr = probs.detach().cpu().numpy()

    arr = np.asarray(arr, dtype=float).reshape(-1, len(CANDIDATES))
    return arr[0]


# =============================================================================
# Episode metrics
# =============================================================================

def metrics_from_snapshot(
    snapshot: Dict[str, Any],
    decision_actions: Counter,
    all_actions: Counter,
    decision_probs: List[np.ndarray],
    total_reward: float,
    episode_steps: int,
    decision_gate_found: bool,
) -> Dict[str, Any]:
    rows = snapshot.get("rows", []) or []

    completed = [
        r
        for r in rows
        if bool(r.get("finished"))
        and np.isfinite(as_float(r.get("travel_time")))
    ]

    travel = np.asarray(
        [as_float(r.get("travel_time")) for r in completed],
        dtype=float,
    )
    travel = travel[np.isfinite(travel)]

    def component(key: str) -> float:
        arr = np.asarray(
            [as_float(r.get(key)) for r in completed],
            dtype=float,
        )
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if len(arr) else float("nan")

    att = float(travel.mean()) if len(travel) else float("nan")
    access = component("access_time")
    awt = component("wait_uam_time")
    aft = component("fly_time")

    residual = float("nan")
    if all(math.isfinite(x) for x in (att, access, awt, aft)):
        residual = att - access - awt - aft

    n = int(snapshot.get("n_persons", len(rows)))
    n_finished = len(completed)
    backlog = n - n_finished

    out: Dict[str, Any] = {
        "ATT": att,
        "AGT_access": access,
        "AWT": awt,
        "AFT": aft,
        "ATT_minus_components": residual,
        "travel_time_median": (
            float(np.median(travel))
            if len(travel)
            else float("nan")
        ),
        "travel_time_p90": (
            float(np.percentile(travel, 90))
            if len(travel)
            else float("nan")
        ),
        "travel_time_p95": (
            float(np.percentile(travel, 95))
            if len(travel)
            else float("nan")
        ),
        "travel_time_max": (
            float(np.max(travel))
            if len(travel)
            else float("nan")
        ),
        "N": n,
        "N_finished": n_finished,
        "completion_rate": (
            n_finished / n if n > 0 else float("nan")
        ),
        "final_backlog": backlog,
        "episode_reward": float(total_reward),
        "episode_steps": int(episode_steps),
        "terminal_scenario_time": snapshot.get("time"),
        "decision_gate_found": bool(decision_gate_found),
        "n_decision_actions": int(sum(decision_actions.values())),
        "n_all_step_actions": int(sum(all_actions.values())),
        "unfinished_state_counts": snapshot.get(
            "unfinished_state_counts", {}
        ),
        "unfinished_substate_counts": snapshot.get(
            "unfinished_substate_counts", {}
        ),
    }

    q = snapshot.get("queue_counts", {}) or {}
    for vid in CANDIDATES:
        out[f"final_queue_v{vid}"] = int(q.get(str(vid), 0))

    # Decision-only deterministic action shares.
    decision_denom = max(1, int(sum(decision_actions.values())))
    all_denom = max(1, int(sum(all_actions.values())))

    for action_idx, vid in enumerate(CANDIDATES):
        dcount = int(decision_actions.get(action_idx, 0))
        acount = int(all_actions.get(action_idx, 0))

        out[f"decision_action_v{vid}_count"] = dcount
        out[f"decision_action_v{vid}_share"] = (
            dcount / decision_denom
        )
        out[f"allstep_action_v{vid}_share"] = (
            acount / all_denom
        )

    # Policy probabilities at passenger-decision states.
    if decision_probs:
        probs = np.vstack(decision_probs)

        eps = 1e-12
        entropy = -np.sum(
            probs * np.log(np.clip(probs, eps, 1.0)),
            axis=1,
        )
        max_prob = np.max(probs, axis=1)

        if probs.shape[1] == 2:
            margin = np.abs(probs[:, 0] - probs[:, 1])
        else:
            sorted_probs = np.sort(probs, axis=1)
            margin = (
                sorted_probs[:, -1] - sorted_probs[:, -2]
            )

        for action_idx, vid in enumerate(CANDIDATES):
            out[f"mean_policy_prob_v{vid}"] = float(
                probs[:, action_idx].mean()
            )

        out["policy_entropy_nats"] = float(entropy.mean())
        out["policy_entropy_normalized"] = float(
            entropy.mean() / math.log(probs.shape[1])
        )
        out["mean_policy_max_prob"] = float(max_prob.mean())
        out["mean_policy_margin"] = float(margin.mean())
    else:
        for vid in CANDIDATES:
            out[f"mean_policy_prob_v{vid}"] = float("nan")
        out["policy_entropy_nats"] = float("nan")
        out["policy_entropy_normalized"] = float("nan")
        out["mean_policy_max_prob"] = float("nan")
        out["mean_policy_margin"] = float("nan")

    # Fixed-fleet diagnostics.
    diag = snapshot.get("fixed_fleet_diagnostics", {}) or {}

    out["fleet_size_diag"] = diag.get("fleet_size", np.nan)
    out["initial_allocation"] = diag.get("initial_allocation", {})
    out["service_arrivals"] = diag.get("service_arrivals", np.nan)
    out["reposition_departures"] = diag.get(
        "reposition_departures", np.nan
    )
    out["reposition_arrivals"] = diag.get(
        "reposition_arrivals", np.nan
    )
    out["final_aircraft_states"] = diag.get("state_counts", {})
    out["final_aircraft_locations"] = diag.get(
        "current_location_counts", {}
    )

    return out


# =============================================================================
# Run one checkpoint episode
# =============================================================================

def run_one_episode(
    spec: ModelSpec,
    passenger_file: Path,
    eval_seed: int,
    max_time: int,
    device: str,
    monitor_dir: Path,
) -> Dict[str, Any]:
    seed_all(eval_seed)

    env = make_eval_env(
        passenger_file=passenger_file,
        max_time=max_time,
        vec_path=spec.vecnormalize_path,
        monitor_dir=monitor_dir,
        env_mode=spec.env_mode,
    )

    try:
        model = PPO.load(
            str(spec.model_path),
            env=env,
            device=device,
        )
        model.policy.set_training_mode(False)

        try:
            env.seed(eval_seed)
        except Exception:
            pass

        obs = env.reset()
        done = np.array([False], dtype=bool)

        base_env = raw_single_env(env)

        decision_actions: Counter = Counter()
        all_actions: Counter = Counter()
        decision_probs: List[np.ndarray] = []

        total_reward = 0.0
        steps = 0
        terminal_snapshot = None

        saw_gate_signal = False

        while not bool(done[0]):
            waiting = waiting_decision_count(base_env)
            if waiting is not None:
                saw_gate_signal = True

            probs = policy_probs(model, obs)

            action, _ = model.predict(
                obs,
                deterministic=True,
            )
            action_int = int(np.asarray(action).reshape(-1)[0])
            all_actions[action_int] += 1

            # Preferred: count only actual passenger-decision states.
            if waiting is not None and waiting > 0:
                decision_actions[action_int] += 1
                decision_probs.append(probs.copy())

            obs, reward, done, infos = env.step(action)

            total_reward += as_float(
                np.asarray(reward).reshape(-1)[0],
                0.0,
            )
            steps += 1

            if (
                infos
                and isinstance(infos[0], dict)
                and "uagmc_terminal_snapshot" in infos[0]
            ):
                terminal_snapshot = infos[0][
                    "uagmc_terminal_snapshot"
                ]

            if steps > max_time + 100:
                raise RuntimeError(
                    f"Episode exceeded expected horizon: "
                    f"{steps} > {max_time}+100"
                )

        if terminal_snapshot is None:
            raise RuntimeError(
                "Terminal passenger snapshot was not captured."
            )

        # Fallback if this source version does not expose waiting_decisions.
        if not saw_gate_signal:
            decision_actions = Counter(all_actions)
            # We did not save all-step probs; rerun is unnecessary for the
            # core operational metrics. Probability diagnostics are marked NaN.
            decision_probs = []

        metrics = metrics_from_snapshot(
            snapshot=terminal_snapshot,
            decision_actions=decision_actions,
            all_actions=all_actions,
            decision_probs=decision_probs,
            total_reward=total_reward,
            episode_steps=steps,
            decision_gate_found=saw_gate_signal,
        )

        metrics.update(
            {
                "source_group": spec.source_group,
                "train_seed": (
                    spec.train_seed
                    if spec.train_seed is not None
                    else -1
                ),
                "train_step": (
                    spec.train_step
                    if spec.train_step is not None
                    else -1
                ),
                "model_label": spec.label,
                "model_path": str(spec.model_path),
                "vecnormalize_path": str(
                    spec.vecnormalize_path
                ),
                "eval_seed": int(eval_seed),
                "passenger_file": str(passenger_file),
                "deterministic": True,
                "environment_mode": spec.env_mode,
                "evaluation_max_time": int(max_time),
            }
        )

        return metrics

    finally:
        env.close()


# =============================================================================
# Aggregation
# =============================================================================

METRIC_COLUMNS = [
    "ATT",
    "AGT_access",
    "AWT",
    "AFT",
    "ATT_minus_components",
    "travel_time_median",
    "travel_time_p90",
    "travel_time_p95",
    "travel_time_max",
    "completion_rate",
    "final_backlog",
    "episode_reward",
    "episode_steps",
    "n_decision_actions",
    "decision_action_v0_share",
    "decision_action_v1_share",
    "allstep_action_v0_share",
    "allstep_action_v1_share",
    "mean_policy_prob_v0",
    "mean_policy_prob_v1",
    "policy_entropy_nats",
    "policy_entropy_normalized",
    "mean_policy_max_prob",
    "mean_policy_margin",
    "final_queue_v0",
    "final_queue_v1",
    "service_arrivals",
    "reposition_departures",
    "reposition_arrivals",
]


def group_rows(
    rows: Sequence[Dict[str, Any]],
    key_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = tuple(row.get(k) for k in key_fields)
        groups[key].append(row)

    out: List[Dict[str, Any]] = []

    for key, group in sorted(
        groups.items(),
        key=lambda kv: tuple(
            -1 if x is None else x for x in kv[0]
        ),
    ):
        summary = dict(zip(key_fields, key))
        summary["n_evals"] = len(group)

        for col in METRIC_COLUMNS:
            vals = [as_float(r.get(col)) for r in group]
            summary[f"{col}_mean"] = finite_mean(vals)
            summary[f"{col}_std"] = finite_std(vals)

        # Useful exact counts / flags.
        summary["decision_gate_found_all"] = all(
            bool(r.get("decision_gate_found"))
            for r in group
        )

        out.append(summary)

    return out


def make_seed_curve(
    episode_rows: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    serial = [
        r for r in episode_rows
        if r.get("source_group") == "E1_fixed16"
    ]

    return group_rows(
        serial,
        key_fields=(
            "source_group",
            "train_seed",
            "train_step",
        ),
    )


def make_across_seed_curve(
    seed_curve: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for row in seed_curve:
        by_step[int(row["train_step"])].append(row)

    out: List[Dict[str, Any]] = []

    metric_mean_cols = sorted(
        {
            k
            for row in seed_curve
            for k in row
            if k.endswith("_mean")
        }
    )

    for step in sorted(by_step):
        group = by_step[step]

        row: Dict[str, Any] = {
            "train_step": step,
            "n_train_seeds": len(group),
        }

        for col in metric_mean_cols:
            vals = [as_float(r.get(col)) for r in group]
            base = col[:-5]
            row[f"{base}_seed_mean"] = finite_mean(vals)
            row[f"{base}_seed_std"] = finite_std(vals)

        out.append(row)

    return out


# =============================================================================
# Training-diagnostic collection
# =============================================================================

def collect_training_diagnostics(
    run_root: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for seed_dir in sorted(
        run_root.glob("seed_*"),
        key=lambda p: int(p.name.split("_")[-1])
        if p.name.split("_")[-1].isdigit()
        else 999999,
    ):
        if not seed_dir.is_dir():
            continue

        m = re.fullmatch(r"seed_(\d+)", seed_dir.name)
        if not m:
            continue

        seed = int(m.group(1))
        path = seed_dir / "training_milestones.csv"

        for r in read_csv(path):
            rows.append(
                {
                    "train_seed": seed,
                    **r,
                }
            )

    return rows


# =============================================================================
# Official legacy reference
# =============================================================================

def official_reference_spec() -> Optional[ModelSpec]:
    model = ROOT / "models" / "final_rl_model.zip"
    vec = ROOT / "models" / "final_vec_normalize.pkl"

    if not model.exists() or not vec.exists():
        return None

    return ModelSpec(
        source_group="official_legacy",
        train_seed=None,
        train_step=None,
        model_path=model.resolve(),
        vecnormalize_path=vec.resolve(),
        label="official_final",
        env_mode="legacy",
    )


# =============================================================================
# Copy auxiliary references into the analysis folder
# =============================================================================

def copy_if_exists(src: Path, dst: Path) -> Optional[Path]:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def collect_auxiliary_files(
    run_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    found: Dict[str, Any] = {}

    # Core run metadata/logs.
    source_logs = output_dir / "source_logs"

    for name in (
        "experiment_manifest.json",
        "experiment_end.json",
        "serial_status.csv",
    ):
        copied = copy_if_exists(
            run_root / name,
            source_logs / name,
        )
        if copied:
            found[name] = str(copied)

    for seed_dir in sorted(run_root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue

        for name in (
            "run_manifest.json",
            "run_end.json",
            "training_milestones.csv",
            "preflight_fixed_fleet.json",
        ):
            copied = copy_if_exists(
                seed_dir / name,
                source_logs / seed_dir.name / name,
            )
            if copied:
                found[f"{seed_dir.name}/{name}"] = str(copied)

    # Static N=16 fleet calibration.
    static_root = ROOT / "fixed_fleet_fairness_scan"
    static_out = output_dir / "static_reference"

    for name in (
        "fleet_summary.csv",
        "ratio_aggregate.csv",
        "recommendation.json",
        "recommendation.txt",
        "manifest.json",
    ):
        copied = copy_if_exists(
            static_root / name,
            static_out / name,
        )
        if copied:
            found[f"static/{name}"] = str(copied)

    # Historical E0 curve (the earlier 5-env reproduction), if present.
    historical_candidates = sorted(
        [
            p
            for p in (ROOT / "serial_runs").glob("uagmc_gpu_10m_*")
            if p.is_dir()
            and (p / "learning_curve_eval").exists()
        ],
        key=lambda p: p.stat().st_mtime,
    )

    if historical_candidates:
        hist = historical_candidates[-1]
        hist_eval = hist / "learning_curve_eval"
        hist_out = output_dir / "historical_E0"

        for name in (
            "curve_across_train_seeds.csv",
            "curve_by_train_seed.csv",
            "official_reference.csv",
            "summary.txt",
            "eval_manifest.json",
        ):
            copied = copy_if_exists(
                hist_eval / name,
                hist_out / name,
            )
            if copied:
                found[f"historical_E0/{name}"] = str(copied)

        found["historical_E0_run_root"] = str(hist)

    return found


# =============================================================================
# Plots
# =============================================================================

def make_errorbar_plot(
    across: Sequence[Dict[str, Any]],
    metric: str,
    ylabel: str,
    out_path: Path,
) -> None:
    if not across:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped] {exc}")
        return

    x = np.asarray(
        [int(r["train_step"]) for r in across],
        dtype=float,
    )
    y = np.asarray(
        [as_float(r.get(f"{metric}_seed_mean")) for r in across],
        dtype=float,
    )
    e = np.asarray(
        [as_float(r.get(f"{metric}_seed_std"), 0.0) for r in across],
        dtype=float,
    )

    fig = plt.figure(figsize=(8.0, 4.8))
    ax = fig.add_subplot(111)
    ax.errorbar(
        x,
        y,
        yerr=e,
        marker="o",
        capsize=3,
    )
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"E1 fixed-fleet N=16: {metric} (mean ± SD across train seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_action_plot(
    across: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    if not across:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped] {exc}")
        return

    x = [int(r["train_step"]) for r in across]
    v0 = [
        as_float(r.get("decision_action_v0_share_seed_mean"))
        for r in across
    ]
    v1 = [
        as_float(r.get("decision_action_v1_share_seed_mean"))
        for r in across
    ]

    fig = plt.figure(figsize=(8.0, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(x, v0, marker="o", label="V0 deterministic share")
    ax.plot(x, v1, marker="o", label="V1 deterministic share")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Passenger-decision action share")
    ax.set_title("E1 fixed-fleet N=16: deterministic routing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_policy_prob_plot(
    across: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    if not across:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped] {exc}")
        return

    x = [int(r["train_step"]) for r in across]
    v0 = [
        as_float(r.get("mean_policy_prob_v0_seed_mean"))
        for r in across
    ]
    v1 = [
        as_float(r.get("mean_policy_prob_v1_seed_mean"))
        for r in across
    ]

    fig = plt.figure(figsize=(8.0, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(x, v0, marker="o", label="Mean P(V0)")
    ax.plot(x, v1, marker="o", label="Mean P(V1)")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean policy probability")
    ax.set_title("E1 fixed-fleet N=16: policy probabilities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# =============================================================================
# Text report
# =============================================================================

def make_summary(
    output_dir: Path,
    run_root: Path,
    across: Sequence[Dict[str, Any]],
    seed_curve: Sequence[Dict[str, Any]],
    official_rows: Sequence[Dict[str, Any]],
    errors: Sequence[Dict[str, Any]],
) -> None:
    lines = [
        "=" * 118,
        "UAGMC E1 FIXED-FLEET N=16 COMPLETE POST-TRAINING ANALYSIS",
        "=" * 118,
        "",
        f"run root: {run_root}",
        "environment: conserved_closed_loop, N=16, V0/V1 initial allocation=12/4",
        "evaluation: deterministic, passengers_300.csv, max_time=600",
        "",
        "Cross-train-seed learning curve",
        "-" * 118,
    ]

    for r in across:
        lines.append(
            f"{int(r['train_step']):>9,d} | "
            f"ATT={as_float(r.get('ATT_seed_mean')):8.3f}"
            f"±{as_float(r.get('ATT_seed_std')):6.3f} | "
            f"AWT={as_float(r.get('AWT_seed_mean')):8.3f} | "
            f"completion={100*as_float(r.get('completion_rate_seed_mean')):6.2f}% | "
            f"backlog={as_float(r.get('final_backlog_seed_mean')):6.2f} | "
            f"V0={100*as_float(r.get('decision_action_v0_share_seed_mean')):6.2f}% | "
            f"Hnorm={as_float(r.get('policy_entropy_normalized_seed_mean')):6.3f}"
        )

    if across:
        final = max(
            across,
            key=lambda r: int(r["train_step"]),
        )

        feasible = [
            r for r in across
            if as_float(r.get("completion_rate_seed_mean")) >= 0.98
        ]
        best = (
            min(feasible, key=lambda r: as_float(r.get("ATT_seed_mean")))
            if feasible
            else min(
                across,
                key=lambda r: (
                    -as_float(r.get("completion_rate_seed_mean"), -1.0),
                    as_float(r.get("ATT_seed_mean"), float("inf")),
                ),
            )
        )

        lines += [
            "",
            "Key checkpoints",
            "-" * 118,
            (
                f"final scanned step={int(final['train_step']):,} | "
                f"ATT={as_float(final.get('ATT_seed_mean')):.4f}±"
                f"{as_float(final.get('ATT_seed_std')):.4f} | "
                f"AWT={as_float(final.get('AWT_seed_mean')):.4f} | "
                f"completion={100*as_float(final.get('completion_rate_seed_mean')):.2f}%"
            ),
            (
                f"best high-completion cross-seed ATT step={int(best['train_step']):,} | "
                f"ATT={as_float(best.get('ATT_seed_mean')):.4f} | "
                f"completion={100*as_float(best.get('completion_rate_seed_mean')):.2f}%"
            ),
        ]

    if official_rows:
        off = group_rows(
            official_rows,
            key_fields=("source_group", "model_label"),
        )[0]

        lines += [
            "",
            "Official legacy-replenishment reference",
            "-" * 118,
            (
                f"ATT={as_float(off.get('ATT_mean')):.4f} | "
                f"AWT={as_float(off.get('AWT_mean')):.4f} | "
                f"Access={as_float(off.get('AGT_access_mean')):.4f} | "
                f"AFT={as_float(off.get('AFT_mean')):.4f} | "
                f"completion={100*as_float(off.get('completion_rate_mean')):.2f}%"
            ),
            (
                "NOTE: this official legacy reference is contextual. "
                "For the strict fast-architecture E0-vs-E1 experiment, "
                "E0 should be trained with the same 16x1280 execution configuration."
            ),
        ]

    lines += [
        "",
        "Interpretation guardrails",
        "-" * 118,
        "1. ATT is over completed passengers; inspect completion/backlog first.",
        "2. Near-single-action routing is not automatically failure; pair it with ATT/AWT/backlog.",
        "3. Policy entropy/probabilities are measured on passenger-decision states when the wrapper exposes waiting_decisions.",
        "4. Checkpoint evaluation is entirely post-training and did not perturb training.",
        "",
        f"errors: {len(errors)}",
    ]

    (output_dir / "summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# ZIP bundle
# =============================================================================

def build_upload_zip(
    output_dir: Path,
) -> Path:
    zip_path = output_dir / "UPLOAD_THIS_analysis_bundle.zip"

    allowed_suffixes = {
        ".csv",
        ".json",
        ".txt",
        ".png",
    }

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            if path == zip_path:
                continue
            if path.suffix.lower() not in allowed_suffixes:
                continue

            rel = path.relative_to(output_dir)
            z.write(path, rel.as_posix())

    return zip_path


# =============================================================================
# CLI / main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate all fixed-fleet N=16 checkpoints and build a compact "
            "analysis bundle for upload."
        )
    )

    p.add_argument(
        "--run-root",
        default=None,
        help="Default: auto-detect newest E1 fixed16 3seed 1M run.",
    )
    p.add_argument(
        "--train-seeds",
        default=None,
        help="Optional subset, e.g. 0,1,2. Default: all discovered.",
    )
    p.add_argument(
        "--eval-seeds",
        default="123,124,125",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=DEFAULT_EVAL_EVERY,
    )
    p.add_argument(
        "--max-step",
        type=int,
        default=DEFAULT_MAX_STEP,
    )
    p.add_argument(
        "--passenger-file",
        default=str(DEFAULT_PASSENGER_FILE),
    )
    p.add_argument(
        "--max-time",
        type=int,
        default=DEFAULT_MAX_TIME,
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
    )
    p.add_argument(
        "--output-dir",
        default=None,
    )
    p.add_argument(
        "--skip-official-reference",
        action="store_true",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
    )

    return p.parse_args()


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False"
        )
    return requested


def main() -> int:
    args = parse_args()

    run_root = (
        Path(args.run_root).expanduser()
        if args.run_root
        else auto_find_run_root()
    )
    if not run_root.is_absolute():
        run_root = (ROOT / run_root).resolve()
    else:
        run_root = run_root.resolve()

    if not run_root.exists():
        raise FileNotFoundError(run_root)

    passenger_file = Path(args.passenger_file).expanduser()
    if not passenger_file.is_absolute():
        passenger_file = (ROOT / passenger_file).resolve()
    else:
        passenger_file = passenger_file.resolve()

    if not passenger_file.exists():
        raise FileNotFoundError(passenger_file)

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else run_root / "complete_analysis"
    )
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = output_dir / "_monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    train_seeds = (
        parse_int_list(args.train_seeds)
        if args.train_seeds
        else None
    )
    eval_seeds = parse_int_list(args.eval_seeds)
    device = choose_device(args.device)

    specs = discover_checkpoints(
        run_root=run_root,
        eval_every=int(args.eval_every),
        max_step=int(args.max_step),
        wanted_train_seeds=train_seeds,
    )

    inventory = checkpoint_inventory(specs)
    write_csv(
        output_dir / "checkpoint_inventory.csv",
        inventory,
    )

    discovered_train_seeds = sorted(
        {
            int(s.train_seed)
            for s in specs
            if s.train_seed is not None
        }
    )
    discovered_steps = sorted(
        {
            int(s.train_step)
            for s in specs
            if s.train_step is not None
        }
    )

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_root": str(run_root),
        "passenger_file": str(passenger_file),
        "train_seeds": discovered_train_seeds,
        "eval_seeds": eval_seeds,
        "checkpoint_steps": discovered_steps,
        "n_checkpoints": len(specs),
        "evaluation_max_time": int(args.max_time),
        "evaluation_policy": "deterministic",
        "inference_device": device,
        "environment": {
            "fleet_mode": FLEET_MODE,
            "fleet_size": FLEET_SIZE,
            "expected_allocation": {"0": 12, "1": 4},
            "candidate_from_vertiports": CANDIDATES,
            "to_vertiport": TO_VERTIPORT,
        },
        "metrics": [
            "ATT",
            "AGT_access",
            "AWT",
            "AFT",
            "completion_rate",
            "final_backlog",
            "travel-time tails",
            "final queues",
            "service/reposition flight counts",
            "decision-only deterministic action shares",
            "mean policy probabilities",
            "policy entropy",
            "policy confidence/margin",
            "training PPO scalars",
        ],
    }
    write_json(
        output_dir / "analysis_manifest.json",
        manifest,
    )

    print("=" * 128)
    print("UAGMC E1 FIXED-FLEET N=16 | COMPLETE POST-TRAINING ANALYSIS")
    print("=" * 128)
    print(f"Run root        : {run_root}")
    print(f"Train seeds     : {discovered_train_seeds}")
    print(f"Checkpoint steps: {discovered_steps}")
    print(f"Checkpoints     : {len(specs)}")
    print(f"Eval seeds      : {eval_seeds}")
    print(f"Passenger trace : {passenger_file}")
    print(f"Eval max_time   : {args.max_time}")
    print(f"Inference       : {device}")
    print(f"Output          : {output_dir}")
    print("=" * 128)

    # Collect original training logs first.
    training_diag = collect_training_diagnostics(run_root)
    write_csv(
        output_dir / "training_diagnostics.csv",
        training_diag,
    )

    auxiliary = collect_auxiliary_files(
        run_root,
        output_dir,
    )
    write_json(
        output_dir / "auxiliary_files.json",
        auxiliary,
    )

    # Main checkpoint evaluation.
    episode_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    total_jobs = len(specs) * len(eval_seeds)
    job = 0

    for spec in specs:
        for eval_seed in eval_seeds:
            job += 1

            print(
                f"[{job:>4}/{total_jobs}] "
                f"train_seed={spec.train_seed} | "
                f"step={spec.train_step:,} | "
                f"eval_seed={eval_seed}",
                flush=True,
            )

            try:
                row = run_one_episode(
                    spec=spec,
                    passenger_file=passenger_file,
                    eval_seed=eval_seed,
                    max_time=int(args.max_time),
                    device=device,
                    monitor_dir=monitor_dir,
                )
                episode_rows.append(row)

                print(
                    "      "
                    f"ATT={row['ATT']:.3f} | "
                    f"AWT={row['AWT']:.3f} | "
                    f"finish={row['N_finished']}/{row['N']} | "
                    f"backlog={row['final_backlog']} | "
                    f"V0={100*row['decision_action_v0_share']:.1f}% | "
                    f"H={row['policy_entropy_normalized']:.3f}",
                    flush=True,
                )

            except Exception as exc:
                err = {
                    "train_seed": spec.train_seed,
                    "train_step": spec.train_step,
                    "eval_seed": eval_seed,
                    "model_path": str(spec.model_path),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(err)

                print(
                    f"      ERROR: {repr(exc)}",
                    flush=True,
                )

                if args.fail_fast:
                    raise

    write_csv(
        output_dir / "episode_metrics.csv",
        episode_rows,
    )
    write_csv(
        output_dir / "errors.csv",
        errors,
    )

    seed_curve = make_seed_curve(episode_rows)
    across_curve = make_across_seed_curve(seed_curve)

    write_csv(
        output_dir / "curve_by_train_seed.csv",
        seed_curve,
    )
    write_csv(
        output_dir / "curve_across_train_seeds.csv",
        across_curve,
    )

    # Final 1M summaries.
    final_step = (
        max(discovered_steps)
        if discovered_steps
        else DEFAULT_MAX_STEP
    )

    final_by_seed = [
        r for r in seed_curve
        if int(r["train_step"]) == final_step
    ]
    final_across = [
        r for r in across_curve
        if int(r["train_step"]) == final_step
    ]

    write_csv(
        output_dir / "final_1m_by_seed.csv",
        final_by_seed,
    )
    write_csv(
        output_dir / "final_1m_across_seeds.csv",
        final_across,
    )

    # Official original model in legacy environment: contextual reference.
    official_rows: List[Dict[str, Any]] = []

    if not args.skip_official_reference:
        off_spec = official_reference_spec()

        if off_spec is not None:
            print("\nOfficial legacy-replenishment reference:")

            for eval_seed in eval_seeds:
                try:
                    row = run_one_episode(
                        spec=off_spec,
                        passenger_file=passenger_file,
                        eval_seed=eval_seed,
                        max_time=int(args.max_time),
                        device=device,
                        monitor_dir=monitor_dir,
                    )
                    official_rows.append(row)

                    print(
                        f"  eval_seed={eval_seed} | "
                        f"ATT={row['ATT']:.3f} | "
                        f"AWT={row['AWT']:.3f} | "
                        f"completion={100*row['completion_rate']:.2f}%"
                    )

                except Exception as exc:
                    errors.append(
                        {
                            "train_seed": -1,
                            "train_step": -1,
                            "eval_seed": eval_seed,
                            "model_path": str(off_spec.model_path),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

    official_summary = (
        group_rows(
            official_rows,
            key_fields=("source_group", "model_label"),
        )
        if official_rows
        else []
    )

    write_csv(
        output_dir / "official_legacy_reference.csv",
        official_summary,
    )
    write_csv(
        output_dir / "errors.csv",
        errors,
    )

    # Plots.
    make_errorbar_plot(
        across_curve,
        "ATT",
        "ATT (min)",
        output_dir / "att_curve.png",
    )
    make_errorbar_plot(
        across_curve,
        "AWT",
        "AWT (min)",
        output_dir / "awt_curve.png",
    )
    make_errorbar_plot(
        across_curve,
        "completion_rate",
        "Completion rate",
        output_dir / "completion_curve.png",
    )
    make_errorbar_plot(
        across_curve,
        "final_backlog",
        "Unfinished passengers",
        output_dir / "backlog_curve.png",
    )
    make_errorbar_plot(
        across_curve,
        "policy_entropy_normalized",
        "Normalized policy entropy",
        output_dir / "entropy_curve.png",
    )
    make_errorbar_plot(
        across_curve,
        "mean_policy_margin",
        "Mean |P(V0)-P(V1)|",
        output_dir / "policy_margin_curve.png",
    )
    make_action_plot(
        across_curve,
        output_dir / "action_share_curve.png",
    )
    make_policy_prob_plot(
        across_curve,
        output_dir / "policy_probability_curve.png",
    )

    make_summary(
        output_dir=output_dir,
        run_root=run_root,
        across=across_curve,
        seed_curve=seed_curve,
        official_rows=official_rows,
        errors=errors,
    )

    zip_path = build_upload_zip(output_dir)

    print("\n" + "=" * 128)
    print("ANALYSIS COLLECTION COMPLETE")
    print("=" * 128)
    print(f"Episode metrics : {output_dir / 'episode_metrics.csv'}")
    print(f"Per-seed curve  : {output_dir / 'curve_by_train_seed.csv'}")
    print(f"Cross-seed curve: {output_dir / 'curve_across_train_seeds.csv'}")
    print(f"Final summary   : {output_dir / 'final_1m_across_seeds.csv'}")
    print(f"Training diag   : {output_dir / 'training_diagnostics.csv'}")
    print(f"Report          : {output_dir / 'summary.txt'}")
    print(f"Errors          : {output_dir / 'errors.csv'}")
    print("-" * 128)
    print("UPLOAD THIS FILE TO CHATGPT:")
    print(zip_path)
    print("=" * 128)

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
