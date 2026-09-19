# -*- coding: utf-8 -*-
"""
UAGMC fixed-fleet PRE-TRAINING no-learning baseline suite
=========================================================

目的
----
在正式 PPO 训练之前，先回答：

    固定守恒机队环境本身是否“可服务”？
    哪个 fleet size 是：
        - 明显供给不足；
        - 临界/有调度压力；
        - 明显供给富余？

这一步不训练任何模型。

默认实验矩阵
------------
fleet_size:
    4, 6, 8, 10

passenger routing:
    all_v0      : 所有乘客选 V0
    all_v1      : 所有乘客选 V1
    balanced    : passenger decision 0/1 确定性交替
    min_access  : 当前 passenger 选择 ground access time 更短的站

eval seeds:
    123, 124, 125

总运行数:
    4 * 4 * 3 = 48

环境
----
只使用：
    fleet_mode="conserved_closed_loop"

其他保持 UAGMC 源码语义：
- passenger file
- V0/V1 -> V2 topology
- original batching / aircraft capacity
- original reward
- original charging / SoC
- original time step
- original passenger completion definition

输出
----
fixed_fleet_pretrain_baselines/
    run_metrics.csv
    aggregate_metrics.csv
    operating_regime.csv
    manifest.json
    summary.txt
    completion_vs_fleet.png
    att_vs_fleet.png
    awt_vs_fleet.png
    backlog_vs_fleet.png

放置位置
--------
E:\\Study Files\\github\\UAM-predict\\UAGMC-main\\
run_uagmc_fixed_fleet_pretrain_baselines.py

依赖
----
已经存在：
    at_obj\\scenario_fixed_fleet.py
    utilss\\make_env_fleet.py

运行
----
python run_uagmc_fixed_fleet_pretrain_baselines.py

可改：
python run_uagmc_fixed_fleet_pretrain_baselines.py ^
  --fleet-sizes 4,6,8,10 ^
  --seeds 123,124,125 ^
  --passenger-file train_data/passengers_300.csv ^
  --max-time 600
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utilss.make_env_fleet import make_env


ROOT = Path(__file__).resolve().parent
DEFAULT_POLICIES = ("all_v0", "all_v1", "balanced", "min_access")


# =============================================================================
# Generic helpers
# =============================================================================

def parse_int_list(text: str) -> List[int]:
    out = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("integer list cannot be empty")
    return out


def parse_str_list(text: str) -> List[str]:
    out = [x.strip().lower() for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("string list cannot be empty")
    return out


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return default


def finite_mean(xs: Iterable[Any]) -> float:
    arr = np.asarray([as_float(x) for x in xs], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def finite_std(xs: Iterable[Any]) -> float:
    arr = np.asarray([as_float(x) for x in xs], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    if len(arr) == 1:
        return 0.0
    return float(arr.std(ddof=1))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fields: List[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            out = {}
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    out[key] = json.dumps(value, ensure_ascii=False)
                else:
                    out[key] = value
            writer.writerow(out)


def normalize_person_id(item: Any) -> str:
    if isinstance(item, (str, int, np.integer)):
        return str(item)

    for attr in ("person_id", "id", "pid"):
        if hasattr(item, attr):
            try:
                return str(getattr(item, attr))
            except Exception:
                pass

    return repr(item)


# =============================================================================
# Wrapper / Scenario discovery
# =============================================================================

def find_wrapper(env: Any):
    """
    Find official UAMRLWrapper:
        has .state .encoder .decoder .env
    """
    obj = env
    seen = set()

    for _ in range(40):
        if id(obj) in seen:
            break
        seen.add(id(obj))

        if (
            hasattr(obj, "state")
            and hasattr(obj, "encoder")
            and hasattr(obj, "decoder")
            and hasattr(obj, "env")
        ):
            return obj

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

    raise RuntimeError(
        "Cannot locate UAMRLWrapper. "
        f"Stopped at {type(obj).__module__}.{type(obj).__name__}"
    )


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
                "vertiports",
                "_all_evtols",
            )
        ):
            return obj

        if hasattr(obj, "scenario"):
            nxt = getattr(obj, "scenario")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue

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

    raise RuntimeError(
        "Cannot locate Scenario. "
        f"Stopped at {type(obj).__module__}.{type(obj).__name__}"
    )


def unwrap_step(step_out):
    if not isinstance(step_out, tuple):
        raise RuntimeError(f"unexpected step return type: {type(step_out)}")

    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        return obs, reward, bool(terminated), bool(truncated), info

    if len(step_out) == 4:
        obs, reward, done, info = step_out
        return obs, reward, bool(done), False, info

    raise RuntimeError(f"unexpected step tuple length: {len(step_out)}")


# =============================================================================
# Passenger policy
# =============================================================================

def get_waiting_pid(wrapper: Any) -> Optional[str]:
    state = getattr(wrapper, "state", None)
    if not isinstance(state, dict):
        return None

    waiting = state.get("waiting_decisions", [])
    if not waiting:
        return None

    return str(waiting[0])


def estimate_access_times(
    scenario: Any,
    pid: str,
    candidate_ids: Sequence[int],
) -> Dict[int, float]:
    """
    Reuse the same ground travel-time estimator used by Scenario.apply_decision.
    """
    persons_obj = getattr(scenario, "persons", None)
    persons = getattr(persons_obj, "persons", {}) if persons_obj else {}

    if pid not in persons:
        # Defensive fallback for int/string key mismatch.
        found = None
        for key, value in persons.items():
            if str(key) == str(pid):
                found = value
                break
        if found is None:
            raise KeyError(f"passenger {pid} not found")
        person = found
    else:
        person = persons[pid]

    result = {}

    for vid in candidate_ids:
        vp = scenario.vertiports.vertiport_list[str(int(vid))]
        t = scenario.vehicles.estimate_travel_time(
            origin=person.origin_position,
            destination=vp.vertiport_position,
        )
        result[int(vid)] = float(t)

    return result


class NoTrainPassengerPolicy:
    def __init__(
        self,
        name: str,
        candidate_ids: Sequence[int],
    ):
        name = str(name).lower()
        if name not in DEFAULT_POLICIES:
            raise ValueError(
                f"unknown policy={name}; valid={DEFAULT_POLICIES}"
            )

        self.name = name
        self.candidate_ids = [int(x) for x in candidate_ids]
        self.decision_count = 0
        self.action_counts = Counter()

    def choose_action(
        self,
        wrapper: Any,
        scenario: Any,
    ) -> int:
        """
        Return action index, not vertiport id.
        """
        pid = get_waiting_pid(wrapper)

        # No passenger decision currently pending.
        # Action is ignored by wrapper; keep deterministic 0.
        if pid is None:
            return 0

        if self.name == "all_v0":
            action = 0

        elif self.name == "all_v1":
            action = 1

        elif self.name == "balanced":
            action = self.decision_count % len(self.candidate_ids)

        elif self.name == "min_access":
            access = estimate_access_times(
                scenario,
                pid,
                self.candidate_ids,
            )
            chosen_vid = min(
                self.candidate_ids,
                key=lambda vid: (access[vid], vid),
            )
            action = self.candidate_ids.index(chosen_vid)

        else:
            raise RuntimeError(self.name)

        self.decision_count += 1
        self.action_counts[action] += 1
        return int(action)


# =============================================================================
# Metrics
# =============================================================================

def evtol_state_name(evtol: Any) -> str:
    try:
        return str(evtol.state.name).upper()
    except Exception:
        return str(getattr(evtol, "state", "UNKNOWN")).upper()


def passenger_snapshot(scenario: Any) -> List[Dict[str, Any]]:
    persons_obj = getattr(scenario, "persons", None)
    persons = getattr(persons_obj, "persons", {}) if persons_obj else {}
    records = getattr(scenario, "person_travel_records", {}) or {}
    finished_ids = {str(x) for x in (getattr(scenario, "finished_ids", []) or [])}

    rows = []

    for pid_raw, person in persons.items():
        pid = str(pid_raw)

        recs = (
            records.get(pid_raw, None)
            or records.get(pid, None)
            or []
        )
        last = recs[-1] if recs else {}

        start = last.get("start_time")
        end = last.get("end_time")

        travel = float("nan")
        if start is not None and end is not None:
            try:
                travel = float(end) - float(start)
            except Exception:
                pass

        stats = getattr(person, "time_stats", {}) or {}

        rows.append(
            {
                "pid": pid,
                "finished": bool(
                    pid in finished_ids
                    or end is not None
                    or str(getattr(person, "state", "")).lower() == "finished"
                ),
                "state": str(getattr(person, "state", "")),
                "sub_state": str(getattr(person, "sub_state", "")),
                "origin_vertiport_id": getattr(
                    person, "origin_vertiport_id", None
                ),
                "travel_time": travel,
                "access_time": as_float(
                    stats.get("to_vertiport", np.nan)
                ),
                "wait_time": as_float(
                    stats.get("wait_uam", np.nan)
                ),
                "fly_time": as_float(
                    stats.get("fly", np.nan)
                ),
            }
        )

    return rows


def current_queue_counts(
    scenario: Any,
    candidate_ids: Sequence[int],
) -> Dict[int, int]:
    result = {}
    for vid in candidate_ids:
        vp = scenario.vertiports.vertiport_list[str(int(vid))]
        result[int(vid)] = len(list(getattr(vp, "person_list", [])))
    return result


def current_enroute_access_counts(
    scenario: Any,
    candidate_ids: Sequence[int],
) -> Dict[int, int]:
    result = {int(v): 0 for v in candidate_ids}

    persons_obj = getattr(scenario, "persons", None)
    persons = getattr(persons_obj, "persons", {}) if persons_obj else {}

    for person in persons.values():
        if str(getattr(person, "state", "")).lower() != "enroute":
            continue

        # Only passenger ground-access phase.
        sub_state = str(getattr(person, "sub_state", "")).lower()
        if sub_state and sub_state != "to_vertiport":
            continue

        try:
            vid = int(getattr(person, "origin_vertiport_id"))
        except Exception:
            continue

        if vid in result:
            result[vid] += 1

    return result


def compute_run_metrics(
    scenario: Any,
    passenger_policy: NoTrainPassengerPolicy,
    candidate_ids: Sequence[int],
    state_tick_counts: Counter,
    location_tick_counts: Counter,
    total_ticks: int,
    total_reward: float,
) -> Dict[str, Any]:
    rows = passenger_snapshot(scenario)
    completed = [
        r
        for r in rows
        if r["finished"] and np.isfinite(r["travel_time"])
    ]

    travel = np.asarray(
        [r["travel_time"] for r in completed],
        dtype=float,
    )

    def avg_component(key: str) -> float:
        vals = np.asarray(
            [as_float(r.get(key, np.nan)) for r in completed],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if len(vals) else float("nan")

    n_total = len(rows)
    n_finished = len(completed)
    queue_counts = current_queue_counts(scenario, candidate_ids)
    access_counts = current_enroute_access_counts(scenario, candidate_ids)

    state_counter = Counter(
        str(r.get("state", "")).lower()
        for r in rows
        if not r["finished"]
    )

    fleet_diag = (
        scenario.get_fixed_fleet_diagnostics()
        if hasattr(scenario, "get_fixed_fleet_diagnostics")
        else {}
    )

    out: Dict[str, Any] = {
        "N": n_total,
        "N_finished": n_finished,
        "completion_rate": (
            n_finished / n_total if n_total > 0 else float("nan")
        ),
        "final_unfinished": n_total - n_finished,
        "final_queue_total": int(sum(queue_counts.values())),
        "final_access_enroute_total": int(sum(access_counts.values())),
        "ATT": float(travel.mean()) if len(travel) else float("nan"),
        "ATT_p50": (
            float(np.percentile(travel, 50))
            if len(travel)
            else float("nan")
        ),
        "ATT_p90": (
            float(np.percentile(travel, 90))
            if len(travel)
            else float("nan")
        ),
        "ATT_p95": (
            float(np.percentile(travel, 95))
            if len(travel)
            else float("nan")
        ),
        "AGT_access": avg_component("access_time"),
        "AWT": avg_component("wait_time"),
        "AFT": avg_component("fly_time"),
        "episode_reward": float(total_reward),
        "policy_decisions": int(passenger_policy.decision_count),
        "service_flights_arrived": int(
            fleet_diag.get("service_arrivals", 0)
        ),
        "reposition_departures": int(
            fleet_diag.get("reposition_departures", 0)
        ),
        "reposition_arrivals": int(
            fleet_diag.get("reposition_arrivals", 0)
        ),
        "fleet_size_check": int(
            fleet_diag.get("fleet_size", -1)
        ),
        "initial_allocation": fleet_diag.get(
            "initial_allocation", {}
        ),
        "final_aircraft_states": fleet_diag.get(
            "state_counts", {}
        ),
        "final_aircraft_locations": fleet_diag.get(
            "current_location_counts", {}
        ),
        "unfinished_state_counts": dict(state_counter),
    }

    for vid in candidate_ids:
        out[f"final_queue_v{vid}"] = int(queue_counts[int(vid)])
        out[f"final_access_enroute_v{vid}"] = int(
            access_counts[int(vid)]
        )

    denom = max(1, passenger_policy.decision_count)
    for action_idx, vid in enumerate(candidate_ids):
        count = int(passenger_policy.action_counts.get(action_idx, 0))
        out[f"action_v{vid}_count"] = count
        out[f"action_v{vid}_share"] = count / denom

    # Time-average aircraft state/location occupancy.
    if total_ticks > 0:
        for state in (
            "IDLE",
            "CHARGING",
            "FLYING",
        ):
            out[f"aircraft_state_{state.lower()}_mean_count"] = (
                state_tick_counts.get(state, 0) / total_ticks
            )

        for vid in ["0", "1", "2"]:
            out[f"aircraft_location_v{vid}_mean_count"] = (
                location_tick_counts.get(vid, 0) / total_ticks
            )

    return out


# =============================================================================
# One episode
# =============================================================================

def sample_aircraft_occupancy(
    scenario: Any,
    state_tick_counts: Counter,
    location_tick_counts: Counter,
):
    for evtol in scenario._all_evtols.values():
        state_tick_counts[evtol_state_name(evtol)] += 1
        location_tick_counts[
            str(getattr(evtol, "current_vertiport_id", "UNKNOWN"))
        ] += 1


def run_episode(
    fleet_size: int,
    policy_name: str,
    seed: int,
    passenger_file: Path,
    max_time: int,
    candidate_ids: Sequence[int],
    to_vertiport: int,
    monitor_dir: Path,
) -> Dict[str, Any]:
    env = make_env(
        max_time=max_time,
        log_dir=monitor_dir,
        env_index=(
            fleet_size * 10000
            + list(DEFAULT_POLICIES).index(policy_name) * 100
            + seed
        ),
        person_spawn_file=str(passenger_file),
        candidate_from_vertiports=list(candidate_ids),
        to_vertiport=to_vertiport,
        enable_logger=False,
        fleet_mode="conserved_closed_loop",
        fleet_size=fleet_size,
        fleet_assertions=True,
    )()

    try:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

        wrapper = find_wrapper(env)
        scenario = find_scenario(env)

        policy = NoTrainPassengerPolicy(
            policy_name,
            candidate_ids,
        )

        state_tick_counts = Counter()
        location_tick_counts = Counter()
        total_ticks = 0
        total_reward = 0.0

        terminated = False
        truncated = False
        step_count = 0

        while not (terminated or truncated):
            # Snapshot BEFORE step.
            sample_aircraft_occupancy(
                scenario,
                state_tick_counts,
                location_tick_counts,
            )
            total_ticks += 1

            action = policy.choose_action(
                wrapper,
                scenario,
            )

            out = env.step(action)
            _, reward, terminated, truncated, _ = unwrap_step(out)

            total_reward += as_float(reward, 0.0)
            step_count += 1

            if step_count > max_time + 100:
                raise RuntimeError(
                    f"episode exceeded expected horizon: "
                    f"steps={step_count}, max_time={max_time}"
                )

        metrics = compute_run_metrics(
            scenario=scenario,
            passenger_policy=policy,
            candidate_ids=candidate_ids,
            state_tick_counts=state_tick_counts,
            location_tick_counts=location_tick_counts,
            total_ticks=total_ticks,
            total_reward=total_reward,
        )

        metrics.update(
            {
                "fleet_size": int(fleet_size),
                "policy": policy_name,
                "seed": int(seed),
                "max_time": int(max_time),
                "passenger_file": str(passenger_file),
                "env_steps": int(step_count),
            }
        )

        return metrics

    finally:
        try:
            env.close()
        except Exception:
            pass


# =============================================================================
# Aggregation + regime labeling
# =============================================================================

AGG_METRICS = (
    "completion_rate",
    "final_unfinished",
    "final_queue_total",
    "ATT",
    "AGT_access",
    "AWT",
    "AFT",
    "ATT_p90",
    "service_flights_arrived",
    "reposition_departures",
    "reposition_arrivals",
    "aircraft_state_idle_mean_count",
    "aircraft_state_charging_mean_count",
    "aircraft_state_flying_mean_count",
    "action_v0_share",
    "action_v1_share",
)


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        groups[(int(row["fleet_size"]), str(row["policy"]))].append(row)

    out = []

    for (fleet_size, policy), group in sorted(groups.items()):
        row: Dict[str, Any] = {
            "fleet_size": fleet_size,
            "policy": policy,
            "n_seeds": len(group),
        }

        for metric in AGG_METRICS:
            vals = [g.get(metric, np.nan) for g in group]
            row[f"{metric}_mean"] = finite_mean(vals)
            row[f"{metric}_std"] = finite_std(vals)

        out.append(row)

    return out


def build_operating_regime(
    aggregate: Sequence[Dict[str, Any]],
    feasible_completion: float,
    low_awt_threshold: float,
) -> List[Dict[str, Any]]:
    """
    This is only a screening label, not a scientific conclusion.

    Per fleet size:
      best completion across the four no-training policies
      best AWT among policies whose completion is feasible

    suggested label:
      UNDERSUPPLIED:
          no policy reaches feasible_completion
      CRITICAL:
          some policy reaches feasible_completion, but best feasible AWT
          is still above low_awt_threshold
      SUPPLY_COMFORTABLE:
          feasible and at least one policy has AWT <= low_awt_threshold
    """
    by_size: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for row in aggregate:
        by_size[int(row["fleet_size"])].append(row)

    out = []

    for size in sorted(by_size):
        group = by_size[size]

        best_completion_row = max(
            group,
            key=lambda r: as_float(
                r.get("completion_rate_mean", float("-inf"))
            ),
        )

        feasible = [
            r
            for r in group
            if as_float(r.get("completion_rate_mean"))
            >= feasible_completion
        ]

        if not feasible:
            label = "UNDERSUPPLIED"
            best_awt = float("nan")
            best_feasible_policy = ""
        else:
            best_awt_row = min(
                feasible,
                key=lambda r: as_float(
                    r.get("AWT_mean", float("inf"))
                ),
            )
            best_awt = as_float(best_awt_row.get("AWT_mean"))
            best_feasible_policy = str(best_awt_row["policy"])

            if best_awt <= low_awt_threshold:
                label = "SUPPLY_COMFORTABLE"
            else:
                label = "CRITICAL"

        out.append(
            {
                "fleet_size": size,
                "best_completion": as_float(
                    best_completion_row.get("completion_rate_mean")
                ),
                "best_completion_policy": str(
                    best_completion_row["policy"]
                ),
                "best_feasible_AWT": best_awt,
                "best_feasible_policy": best_feasible_policy,
                "screening_label": label,
                "feasible_completion_threshold": feasible_completion,
                "low_AWT_threshold": low_awt_threshold,
            }
        )

    return out


# =============================================================================
# Plots
# =============================================================================

def plot_metric(
    aggregate: Sequence[Dict[str, Any]],
    metric: str,
    ylabel: str,
    out_path: Path,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[PLOT] matplotlib unavailable: {exc}")
        return

    policies = list(DEFAULT_POLICIES)

    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)

    for policy in policies:
        rows = [
            r for r in aggregate
            if r["policy"] == policy
        ]
        rows.sort(key=lambda r: int(r["fleet_size"]))

        if not rows:
            continue

        x = [int(r["fleet_size"]) for r in rows]
        y = [as_float(r.get(f"{metric}_mean")) for r in rows]
        e = [as_float(r.get(f"{metric}_std")) for r in rows]

        ax.errorbar(
            x,
            y,
            yerr=e,
            marker="o",
            capsize=3,
            label=policy,
        )

    ax.set_xlabel("Fixed fleet size")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Fixed-fleet no-training baseline: {metric}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# =============================================================================
# Report
# =============================================================================

def write_summary(
    output_dir: Path,
    aggregate: Sequence[Dict[str, Any]],
    regime: Sequence[Dict[str, Any]],
):
    lines = [
        "=" * 112,
        "UAGMC FIXED-FLEET PRE-TRAINING NO-LEARNING BASELINES",
        "=" * 112,
        "",
        "Purpose:",
        "  Determine whether a fixed fleet is physically serviceable before PPO training.",
        "",
        "Per-policy aggregate:",
        "-" * 112,
    ]

    for row in aggregate:
        lines.append(
            f"N={int(row['fleet_size']):>2} | "
            f"{row['policy']:<10} | "
            f"completion={100*as_float(row.get('completion_rate_mean')):6.2f}% "
            f"±{100*as_float(row.get('completion_rate_std')):5.2f} | "
            f"ATT={as_float(row.get('ATT_mean')):7.3f} | "
            f"AWT={as_float(row.get('AWT_mean')):7.3f} | "
            f"Access={as_float(row.get('AGT_access_mean')):7.3f} | "
            f"AFT={as_float(row.get('AFT_mean')):7.3f} | "
            f"backlog={as_float(row.get('final_unfinished_mean')):7.2f} | "
            f"serviceFlights={as_float(row.get('service_flights_arrived_mean')):7.2f}"
        )

    lines += [
        "",
        "Screening operating regime:",
        "-" * 112,
    ]

    for row in regime:
        lines.append(
            f"N={int(row['fleet_size']):>2} | "
            f"{row['screening_label']:<20} | "
            f"best completion={100*as_float(row['best_completion']):6.2f}% "
            f"({row['best_completion_policy']}) | "
            f"best feasible AWT={as_float(row['best_feasible_AWT']):7.3f} "
            f"({row['best_feasible_policy']})"
        )

    lines += [
        "",
        "Interpretation rule:",
        "-" * 112,
        "UNDERSUPPLIED:",
        "  none of the simple no-training policies can complete the requested threshold.",
        "  PPO failure at this N should not be called a learning failure.",
        "",
        "CRITICAL:",
        "  the environment is serviceable, but waiting remains material.",
        "  This is the most useful region for the first fixed-fleet PPO training.",
        "",
        "SUPPLY_COMFORTABLE:",
        "  the simple baseline already achieves high completion and low AWT.",
        "  This can be useful as an easier control / upper-fleet condition.",
        "",
        "Important:",
        "  screening labels are operational diagnostics, not paper conclusions.",
    ]

    (output_dir / "summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "UAGMC fixed-fleet no-training capacity/routing baseline suite"
        )
    )

    p.add_argument(
        "--fleet-sizes",
        default="4,6,8,10",
    )
    p.add_argument(
        "--policies",
        default="all_v0,all_v1,balanced,min_access",
    )
    p.add_argument(
        "--seeds",
        default="123,124,125",
    )
    p.add_argument(
        "--passenger-file",
        default="train_data/passengers_300.csv",
    )
    p.add_argument(
        "--max-time",
        type=int,
        default=600,
    )
    p.add_argument(
        "--candidates",
        default="0,1",
    )
    p.add_argument(
        "--to-vertiport",
        type=int,
        default=2,
    )
    p.add_argument(
        "--output-dir",
        default="fixed_fleet_pretrain_baselines",
    )

    # Screening labels only.
    p.add_argument(
        "--feasible-completion",
        type=float,
        default=0.98,
        help="screening threshold only; default 0.98",
    )
    p.add_argument(
        "--low-awt-threshold",
        type=float,
        default=3.0,
        help="screening threshold only; default 3 min",
    )

    return p.parse_args()


def main():
    args = parse_args()

    fleet_sizes = parse_int_list(args.fleet_sizes)
    seeds = parse_int_list(args.seeds)
    policies = parse_str_list(args.policies)
    candidate_ids = parse_int_list(args.candidates)

    invalid = [
        p for p in policies
        if p not in DEFAULT_POLICIES
    ]
    if invalid:
        raise ValueError(
            f"invalid policies={invalid}; valid={DEFAULT_POLICIES}"
        )

    if candidate_ids != [0, 1]:
        print(
            "[Warning] This suite is designed for the original UAGMC "
            f"two-candidate setting; got candidates={candidate_ids}"
        )

    passenger_file = Path(args.passenger_file).expanduser()
    if not passenger_file.is_absolute():
        passenger_file = (ROOT / passenger_file).resolve()
    else:
        passenger_file = passenger_file.resolve()

    if not passenger_file.exists():
        raise FileNotFoundError(passenger_file)

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = output_dir / "_monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "fixed-fleet pre-training no-learning capacity/routing baselines",
        "fleet_mode": "conserved_closed_loop",
        "fleet_sizes": fleet_sizes,
        "policies": policies,
        "seeds": seeds,
        "passenger_file": str(passenger_file),
        "max_time": int(args.max_time),
        "candidate_from_vertiports": candidate_ids,
        "to_vertiport": int(args.to_vertiport),
        "n_runs": len(fleet_sizes) * len(policies) * len(seeds),
        "feasible_completion_screening_threshold": float(
            args.feasible_completion
        ),
        "low_awt_screening_threshold": float(
            args.low_awt_threshold
        ),
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 128)
    print("UAGMC FIXED-FLEET PRE-TRAINING BASELINE SUITE")
    print("=" * 128)
    print(f"Fleet sizes     : {fleet_sizes}")
    print(f"Policies        : {policies}")
    print(f"Seeds           : {seeds}")
    print(f"Passenger file  : {passenger_file}")
    print(f"Total runs      : {manifest['n_runs']}")
    print(f"Output          : {output_dir}")
    print("=" * 128)

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    total_jobs = manifest["n_runs"]
    job = 0

    for fleet_size in fleet_sizes:
        for policy in policies:
            for seed in seeds:
                job += 1

                print(
                    f"[{job:>3}/{total_jobs}] "
                    f"N={fleet_size:<3} policy={policy:<10} seed={seed}",
                    flush=True,
                )

                try:
                    row = run_episode(
                        fleet_size=fleet_size,
                        policy_name=policy,
                        seed=seed,
                        passenger_file=passenger_file,
                        max_time=int(args.max_time),
                        candidate_ids=candidate_ids,
                        to_vertiport=int(args.to_vertiport),
                        monitor_dir=monitor_dir,
                    )
                    rows.append(row)

                    print(
                        "    "
                        f"completion={100*row['completion_rate']:.2f}% | "
                        f"ATT={row['ATT']:.3f} | "
                        f"AWT={row['AWT']:.3f} | "
                        f"backlog={row['final_unfinished']} | "
                        f"serviceFlights={row['service_flights_arrived']}",
                        flush=True,
                    )

                except Exception as exc:
                    errors.append(
                        {
                            "fleet_size": fleet_size,
                            "policy": policy,
                            "seed": seed,
                            "error": repr(exc),
                        }
                    )
                    print(
                        f"    ERROR: {repr(exc)}",
                        flush=True,
                    )

    write_csv(
        output_dir / "run_metrics.csv",
        rows,
    )
    write_csv(
        output_dir / "errors.csv",
        errors,
    )

    aggregate = aggregate_rows(rows)
    write_csv(
        output_dir / "aggregate_metrics.csv",
        aggregate,
    )

    regime = build_operating_regime(
        aggregate,
        feasible_completion=float(args.feasible_completion),
        low_awt_threshold=float(args.low_awt_threshold),
    )
    write_csv(
        output_dir / "operating_regime.csv",
        regime,
    )

    plot_metric(
        aggregate,
        "completion_rate",
        "Completion rate",
        output_dir / "completion_vs_fleet.png",
    )
    plot_metric(
        aggregate,
        "ATT",
        "ATT (min)",
        output_dir / "att_vs_fleet.png",
    )
    plot_metric(
        aggregate,
        "AWT",
        "AWT (min)",
        output_dir / "awt_vs_fleet.png",
    )
    plot_metric(
        aggregate,
        "final_unfinished",
        "Final unfinished passengers",
        output_dir / "backlog_vs_fleet.png",
    )

    write_summary(
        output_dir,
        aggregate,
        regime,
    )

    print("\n" + "=" * 128)
    print("DONE")
    print("=" * 128)
    print(f"Raw runs       : {output_dir / 'run_metrics.csv'}")
    print(f"Aggregate      : {output_dir / 'aggregate_metrics.csv'}")
    print(f"Operating mode : {output_dir / 'operating_regime.csv'}")
    print(f"Summary        : {output_dir / 'summary.txt'}")
    print(f"Errors         : {output_dir / 'errors.csv'}")

    if errors:
        print(
            f"WARNING: {len(errors)} run(s) failed. "
            "Inspect errors.csv before interpreting results."
        )
        raise SystemExit(2)

    print("ALL PRE-TRAINING BASELINE RUNS PASSED")


if __name__ == "__main__":
    main()
