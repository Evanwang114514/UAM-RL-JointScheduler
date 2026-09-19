# -*- coding: utf-8 -*-
"""
UAGMC Stage-0.8：Passenger current_timer 语义审计（无训练）

目的
----
前几轮 candidate-relative peeling 使用了：
    person.current_timer

并假设它代表：
    “当前 passenger 距离到达 departure vertiport 的剩余 access ETA”。

Stage-0.7 的 same-cohort 结果表明，这个假设非常可疑。
本脚本不训练、不修改 policy、不构造新的 observation，只做一个最小审计：

1. 沿原 checkpoint 真实 rollout，每个 simulation step 记录所有 state="enroute" passenger：
   - current_timer
   - origin_vertiport_id
   - method / state / sub_state
   - time_stats 中可读取的标量
   - 根据 origin_position -> assigned departure vertiport 重新计算的 static access travel time（若接口可用）

2. 对每个 passenger 继续跟踪，直到它第一次离开 enroute：
   - 记录真实 exit time
   - 记录 exit 后是否进入 waiting queue
   - 对此前所有 timer observation 计算 actual_remaining = exit_time - observation_time

3. 比较三种解释：
   A. Remaining-ETA 假设：
        current_timer ≈ actual_remaining
   B. Elapsed-time 假设：
        static_total_access - current_timer ≈ actual_remaining
   C. 其他/混合：
        两者都不吻合

4. 同时检查 timer 的逐步变化方向：
   - 每个 simulation step 大约 -1：更像 remaining countdown
   - 每个 simulation step 大约 +1：更像 elapsed counter
   - 其他：说明 current_timer 可能使用不同单位/更新逻辑

输出
----
diagnostics/uagmc_timer_semantics_audit/<timestamp>/
    summary.json
    timer_observations.csv
    passenger_summary.csv
    timer_step_deltas.csv
    resolved_config.json

注意
----
- 完全不训练。
- 真实 future transition 只用于离线审计，不作为 policy 输入。
- 本脚本首先回答“current_timer 是什么”，不是验证最终方法。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import diagnose_uagmc_candidate_temporal_mismatch as stage0
except Exception as exc:
    raise RuntimeError(
        "无法导入 diagnose_uagmc_candidate_temporal_mismatch.py。\n"
        "请把本文件与 Stage-0 脚本放在 UAGMC 源码根目录。"
    ) from exc


# =============================================================================
# 通用工具
# =============================================================================

def mean_or_nan(values: Sequence[float]) -> float:
    vals = [
        float(x)
        for x in values
        if x is not None and np.isfinite(float(x))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def median_or_nan(values: Sequence[float]) -> float:
    vals = [
        float(x)
        for x in values
        if x is not None and np.isfinite(float(x))
    ]
    return float(np.median(vals)) if vals else float("nan")


def percentile_or_nan(values: Sequence[float], q: float) -> float:
    vals = [
        float(x)
        for x in values
        if x is not None and np.isfinite(float(x))
    ]
    return float(np.percentile(vals, q)) if vals else float("nan")


def safe_float(value: Any) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def jsonable(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    return x


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pearson_or_nan(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if np.isfinite(float(x)) and np.isfinite(float(y))
    ]

    if len(pairs) < 3:
        return float("nan")

    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)

    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def mae_or_nan(xs: Sequence[float], ys: Sequence[float]) -> float:
    vals = [
        abs(float(x) - float(y))
        for x, y in zip(xs, ys)
        if np.isfinite(float(x)) and np.isfinite(float(y))
    ]
    return float(np.mean(vals)) if vals else float("nan")


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
# UAGMC passenger / queue 读取
# =============================================================================

def get_person_dict(scenario: Any) -> Dict[str, Any]:
    persons_obj = getattr(scenario, "persons", None)
    raw = (
        getattr(persons_obj, "persons", {})
        if persons_obj is not None
        else {}
    )

    return {
        str(pid): person
        for pid, person in raw.items()
    }


def get_waiting_ids_all_vertiports(scenario: Any) -> Set[str]:
    waiting: Set[str] = set()

    vertiport_list = getattr(
        getattr(scenario, "vertiports", None),
        "vertiport_list",
        {},
    ) or {}

    for vp in vertiport_list.values():
        for item in list(getattr(vp, "person_list", [])):
            waiting.add(normalize_person_id(item))

    return waiting


def person_scalar_snapshot(person: Any) -> Dict[str, Any]:
    """
    记录一些常见字段，尽量帮助核对 current_timer 的源码语义。
    不强依赖字段必须存在。
    """
    row: Dict[str, Any] = {}

    attrs = (
        "state",
        "sub_state",
        "method",
        "origin_vertiport_id",
        "destination_vertiport_id",
        "current_timer",
        "travel_time",
        "access_time",
        "remaining_time",
        "remaining_travel_time",
        "target_time",
        "start_time",
        "spawn_time",
    )

    for attr in attrs:
        value = getattr(person, attr, None)

        if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
            row[attr] = value

    stats = getattr(person, "time_stats", None)

    if isinstance(stats, dict):
        for key, value in stats.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                row[f"time_stats_{key}"] = value

    return row


def estimate_static_access_time(
    scenario: Any,
    person: Any,
) -> float:
    """
    使用和前几轮相同的 travel-time estimator，
    重新计算该 passenger 到其已分配 departure vertiport 的静态 access time。
    """
    try:
        vid = str(int(getattr(person, "origin_vertiport_id")))
        vertiport = scenario.vertiports.vertiport_list[vid]

        return float(
            scenario.vehicles.estimate_travel_time(
                origin=person.origin_position,
                destination=vertiport.vertiport_position,
            )
        )
    except Exception:
        return float("nan")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="UAGMC Stage-0.8 passenger current_timer semantics audit，无训练"
    )

    parser.add_argument(
        "--project-root",
        default=str(Path.cwd()),
    )

    parser.add_argument(
        "--uagmc-root",
        default=None,
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    parser.add_argument(
        "--vecnorm",
        default=None,
    )

    parser.add_argument(
        "--passengers",
        default=None,
    )

    parser.add_argument(
        "--candidates",
        default=None,
    )

    parser.add_argument(
        "--to-vertiport",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-time",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
    )

    parser.add_argument(
        "--output-dir",
        default="diagnostics/uagmc_timer_semantics_audit",
    )

    return parser.parse_args()


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()

    uagmc_root = stage0.discover_uagmc_root(
        project_root,
        args.uagmc_root,
    )

    model_path = stage0.discover_model(
        uagmc_root,
        project_root,
        args.model,
    )

    vecnorm_path = stage0.discover_vecnorm(
        uagmc_root,
        project_root,
        model_path,
        args.vecnorm,
    )

    passenger_path = stage0.discover_passenger_file(
        uagmc_root,
        project_root,
        args.passengers,
    )

    sys.path.insert(0, str(uagmc_root))

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from utilss.make_env import make_env

    initial_candidates = (
        [
            int(x.strip())
            for x in args.candidates.split(",")
            if x.strip()
        ]
        if args.candidates
        else [0, 1]
    )

    run_stamp = time.strftime("%Y%m%d_%H%M%S")

    output_root = (
        project_root
        / args.output_dir
        / run_stamp
    )

    monitor_dir = output_root / "_monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv(
        [
            make_env(
                max_time=args.max_time,
                log_dir=monitor_dir,
                env_index=0,
                candidate_from_vertiports=initial_candidates,
                to_vertiport=args.to_vertiport,
                person_spawn_file=str(passenger_path),
                enable_logger=False,
            )
        ]
    )

    if vecnorm_path is not None:
        env = VecNormalize.load(
            str(vecnorm_path),
            env,
        )
        env.training = False
        env.norm_reward = False

    device = stage0.resolve_device(args.device)

    model = PPO.load(
        str(model_path),
        env=env,
        device=device,
    )

    wrapper = stage0.unwrap_uagmc_wrapper(env)
    scenario = wrapper.env

    candidate_ids = stage0.get_candidate_ids(
        wrapper,
        args.candidates,
    )

    print("=" * 116)
    print("UAGMC STAGE-0.8 | CURRENT_TIMER SEMANTICS AUDIT | NO TRAINING")
    print("=" * 116)
    print(f"Project root : {project_root}")
    print(f"UAGMC root   : {uagmc_root}")
    print(f"Model        : {model_path}")
    print(f"VecNormalize : {vecnorm_path}")
    print(f"Passengers   : {passenger_path}")
    print(f"Candidates   : {candidate_ids}")
    print(f"Output       : {output_root}")
    print("-" * 116)

    # 每个 pid 保存所有 enroute observation。
    history_by_pid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # 上一个 simulation step 仍处于 enroute 的 passenger。
    prev_enroute_ids: Set[str] = set()

    # 每个 pid 第一次真实离开 enroute 的时刻和状态。
    exit_info: Dict[str, Dict[str, Any]] = {}

    obs = env.reset()
    done = False
    step_count = 0

    while not done:
        sim_time = int(
            getattr(
                scenario,
                "time",
                step_count,
            )
        )

        persons = get_person_dict(scenario)
        waiting_ids = get_waiting_ids_all_vertiports(scenario)

        current_enroute_ids: Set[str] = set()

        # 先检查从上一 step 的 enroute 集合中消失的人。
        for pid in prev_enroute_ids:
            person = persons.get(pid)

            current_state = (
                str(getattr(person, "state", "MISSING"))
                if person is not None
                else "MISSING"
            )

            if (
                person is not None
                and current_state.lower() == "enroute"
            ):
                continue

            if pid not in exit_info:
                exit_info[pid] = {
                    "exit_time": sim_time,
                    "exit_state": current_state,
                    "entered_waiting_queue": int(pid in waiting_ids),
                }

        # 当前所有 enroute passenger 做 snapshot。
        for pid, person in persons.items():
            state = str(getattr(person, "state", "")).lower()

            if state != "enroute":
                continue

            current_enroute_ids.add(pid)

            row: Dict[str, Any] = {
                "pid": pid,
                "observation_time": sim_time,
                "current_timer": safe_float(
                    getattr(person, "current_timer", float("nan"))
                ),
                "static_access_time": estimate_static_access_time(
                    scenario,
                    person,
                ),
            }

            row.update(
                person_scalar_snapshot(person)
            )

            history_by_pid[pid].append(row)

        prev_enroute_ids = current_enroute_ids

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, _, dones, _ = env.step(action)

        done = bool(
            np.asarray(dones).reshape(-1)[0]
        )

        step_count += 1

        if step_count > args.max_time + 1000:
            print("[警告] 超过安全 step 上限，强制结束。")
            break

    terminal_time = int(
        getattr(
            scenario,
            "time",
            step_count,
        )
    )

    # terminal 时再解析一次仍未登记 exit 的 passenger。
    persons = get_person_dict(scenario)
    waiting_ids = get_waiting_ids_all_vertiports(scenario)

    for pid in list(prev_enroute_ids):
        person = persons.get(pid)

        state = (
            str(getattr(person, "state", "MISSING"))
            if person is not None
            else "MISSING"
        )

        if state.lower() != "enroute" and pid not in exit_info:
            exit_info[pid] = {
                "exit_time": terminal_time,
                "exit_state": state,
                "entered_waiting_queue": int(pid in waiting_ids),
            }

    # =========================================================================
    # 解析 observation -> actual remaining
    # =========================================================================

    observation_rows: List[Dict[str, Any]] = []
    passenger_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []

    for pid, hist in sorted(history_by_pid.items()):
        hist = sorted(
            hist,
            key=lambda x: int(x["observation_time"]),
        )

        info = exit_info.get(pid)

        if info is None:
            continue

        exit_time = int(info["exit_time"])

        resolved_for_pid: List[Dict[str, Any]] = []

        for row in hist:
            obs_time = int(row["observation_time"])

            if obs_time >= exit_time:
                continue

            out = dict(row)

            actual_remaining = float(
                exit_time - obs_time
            )

            timer = safe_float(
                row.get("current_timer")
            )

            static_total = safe_float(
                row.get("static_access_time")
            )

            elapsed_model_remaining = (
                static_total - timer
                if np.isfinite(static_total) and np.isfinite(timer)
                else float("nan")
            )

            out["exit_time"] = exit_time
            out["exit_state"] = info["exit_state"]
            out["entered_waiting_queue"] = info["entered_waiting_queue"]

            out["actual_remaining_steps"] = actual_remaining

            # Remaining-ETA 假设的直接误差。
            out["remaining_eta_abs_error"] = (
                abs(timer - actual_remaining)
                if np.isfinite(timer)
                else float("nan")
            )

            # Elapsed-time 假设：
            # static_total_access - current_timer 应接近 actual remaining。
            out["elapsed_model_remaining"] = elapsed_model_remaining

            out["elapsed_model_abs_error"] = (
                abs(elapsed_model_remaining - actual_remaining)
                if np.isfinite(elapsed_model_remaining)
                else float("nan")
            )

            # 如果 timer 是 elapsed counter，
            # timer + remaining 应近似常数（总 access duration）。
            out["timer_plus_actual_remaining"] = (
                timer + actual_remaining
                if np.isfinite(timer)
                else float("nan")
            )

            # 如果 timer 是 remaining countdown，
            # timer - actual_remaining 应接近 0。
            out["timer_minus_actual_remaining"] = (
                timer - actual_remaining
                if np.isfinite(timer)
                else float("nan")
            )

            observation_rows.append(out)
            resolved_for_pid.append(out)

        # 相邻 step 的 timer slope。
        slopes: List[float] = []

        for a, b in zip(hist[:-1], hist[1:]):
            t0 = int(a["observation_time"])
            t1 = int(b["observation_time"])

            dt = t1 - t0

            timer0 = safe_float(a.get("current_timer"))
            timer1 = safe_float(b.get("current_timer"))

            if dt <= 0 or not (
                np.isfinite(timer0)
                and np.isfinite(timer1)
            ):
                continue

            slope = (timer1 - timer0) / float(dt)
            slopes.append(slope)

            delta_rows.append(
                {
                    "pid": pid,
                    "time0": t0,
                    "time1": t1,
                    "dt": dt,
                    "timer0": timer0,
                    "timer1": timer1,
                    "timer_delta": timer1 - timer0,
                    "timer_slope_per_step": slope,
                }
            )

        if resolved_for_pid:
            timers = [
                safe_float(r["current_timer"])
                for r in resolved_for_pid
            ]

            actual_remaining = [
                float(r["actual_remaining_steps"])
                for r in resolved_for_pid
            ]

            elapsed_remaining = [
                safe_float(r["elapsed_model_remaining"])
                for r in resolved_for_pid
            ]

            timer_plus_remaining = [
                safe_float(r["timer_plus_actual_remaining"])
                for r in resolved_for_pid
            ]

            passenger_rows.append(
                {
                    "pid": pid,
                    "n_observations": len(resolved_for_pid),
                    "first_observation_time": int(
                        resolved_for_pid[0]["observation_time"]
                    ),
                    "last_observation_time": int(
                        resolved_for_pid[-1]["observation_time"]
                    ),
                    "exit_time": exit_time,
                    "exit_state": info["exit_state"],
                    "entered_waiting_queue": info["entered_waiting_queue"],
                    "origin_vertiport_id": resolved_for_pid[0].get(
                        "origin_vertiport_id",
                        "",
                    ),
                    "static_access_time": safe_float(
                        resolved_for_pid[0].get("static_access_time")
                    ),
                    "first_current_timer": timers[0],
                    "last_current_timer": timers[-1],
                    "median_timer_slope_per_step": median_or_nan(slopes),
                    "remaining_eta_mae": mae_or_nan(
                        timers,
                        actual_remaining,
                    ),
                    "elapsed_model_mae": mae_or_nan(
                        elapsed_remaining,
                        actual_remaining,
                    ),
                    "timer_plus_remaining_std": float(
                        np.std(
                            [
                                x
                                for x in timer_plus_remaining
                                if np.isfinite(x)
                            ]
                        )
                    ) if any(
                        np.isfinite(x)
                        for x in timer_plus_remaining
                    ) else float("nan"),
                }
            )

    # =========================================================================
    # 汇总语义判定
    # =========================================================================

    timers = [
        safe_float(r["current_timer"])
        for r in observation_rows
    ]

    actual_remaining = [
        safe_float(r["actual_remaining_steps"])
        for r in observation_rows
    ]

    static_access = [
        safe_float(r["static_access_time"])
        for r in observation_rows
    ]

    elapsed_model_remaining = [
        safe_float(r["elapsed_model_remaining"])
        for r in observation_rows
    ]

    slopes = [
        safe_float(r["timer_slope_per_step"])
        for r in delta_rows
    ]

    remaining_mae = mae_or_nan(
        timers,
        actual_remaining,
    )

    elapsed_mae = mae_or_nan(
        elapsed_model_remaining,
        actual_remaining,
    )

    median_slope = median_or_nan(slopes)

    # 简单审计分类，不作为论文结论，只帮助下一步修代码。
    if (
        np.isfinite(remaining_mae)
        and np.isfinite(elapsed_mae)
        and np.isfinite(median_slope)
    ):
        if (
            remaining_mae <= 0.8 * elapsed_mae
            and median_slope < -0.25
        ):
            semantic_guess = "remaining_eta_like"
        elif (
            elapsed_mae <= 0.8 * remaining_mae
            and median_slope > 0.25
        ):
            semantic_guess = "elapsed_time_like"
        else:
            semantic_guess = "ambiguous_or_different_units"
    else:
        semantic_guess = "insufficient_evidence"

    def frac_close(values: Sequence[float], target: float, tol: float = 0.15) -> float:
        valid = [
            float(x)
            for x in values
            if np.isfinite(float(x))
        ]

        if not valid:
            return float("nan")

        return float(
            np.mean(
                [
                    abs(x - target) <= tol
                    for x in valid
                ]
            )
        )

    summary = {
        "stage": "Stage-0.8 current_timer semantics audit no-training",
        "project_root": str(project_root),
        "uagmc_root": str(uagmc_root),
        "model": str(model_path),
        "vecnormalize": (
            str(vecnorm_path)
            if vecnorm_path
            else None
        ),
        "passenger_trace": str(passenger_path),
        "candidate_vertiports": candidate_ids,
        "resolved_passengers": len(passenger_rows),
        "resolved_timer_observations": len(observation_rows),
        "timer_step_delta_rows": len(delta_rows),
        "timer_dynamics": {
            "median_slope_per_step": median_slope,
            "mean_slope_per_step": mean_or_nan(slopes),
            "fraction_slope_near_plus_1": frac_close(slopes, 1.0),
            "fraction_slope_near_minus_1": frac_close(slopes, -1.0),
            "fraction_slope_near_zero": frac_close(slopes, 0.0),
            "p10_slope": percentile_or_nan(slopes, 10),
            "p90_slope": percentile_or_nan(slopes, 90),
        },
        "remaining_eta_hypothesis": {
            "pearson_timer_vs_actual_remaining": pearson_or_nan(
                timers,
                actual_remaining,
            ),
            "mae_current_timer_vs_actual_remaining": remaining_mae,
            "median_abs_error": median_or_nan(
                [
                    abs(t - r)
                    for t, r in zip(timers, actual_remaining)
                    if np.isfinite(t) and np.isfinite(r)
                ]
            ),
        },
        "elapsed_time_hypothesis": {
            "pearson_static_minus_timer_vs_actual_remaining": pearson_or_nan(
                elapsed_model_remaining,
                actual_remaining,
            ),
            "mae_static_minus_timer_vs_actual_remaining": elapsed_mae,
            "median_abs_error": median_or_nan(
                [
                    abs(e - r)
                    for e, r in zip(elapsed_model_remaining, actual_remaining)
                    if np.isfinite(e) and np.isfinite(r)
                ]
            ),
        },
        "auxiliary": {
            "pearson_static_access_vs_actual_total_proxy": pearson_or_nan(
                static_access,
                [
                    (
                        safe_float(r["current_timer"])
                        + safe_float(r["actual_remaining_steps"])
                    )
                    for r in observation_rows
                ],
            ),
            "median_timer_plus_remaining": median_or_nan(
                [
                    safe_float(r["timer_plus_actual_remaining"])
                    for r in observation_rows
                ]
            ),
            "median_timer_minus_remaining": median_or_nan(
                [
                    safe_float(r["timer_minus_actual_remaining"])
                    for r in observation_rows
                ]
            ),
        },
        "semantic_guess": semantic_guess,
        "interpretation": {
            "remaining_eta_like": (
                "current_timer 更像剩余 ETA；Stage-0.7 可能主要是单位或 horizon 对齐问题。"
            ),
            "elapsed_time_like": (
                "current_timer 更像已过去 access 时间；此前把它直接当剩余 ETA 的 peeling 语义错误。"
            ),
            "ambiguous_or_different_units": (
                "current_timer 既不像简单剩余 ETA，也不像 static_total-current_timer；"
                "应继续查看 Person/Scenario 源码中的 timer 更新逻辑或单位。"
            ),
            "insufficient_evidence": (
                "有效样本不足，不能判定 timer 语义。"
            ),
        }[semantic_guess],
        "guardrails": [
            "真实 exit_time 只用于离线 timer 语义审计，不作为在线 policy 输入。",
            "actual_remaining_steps 是 simulation time 差；如果 current_timer 使用不同单位，MAE 会暴露该问题。",
            "static_access_time 通过当前源码 travel-time estimator 重算，只作为 elapsed-time 假设的参照。",
            "若 timer 不是简单 countdown/elapsed，本实验不会强行解释，应回到源码确认字段定义。",
        ],
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_root / "timer_observations.csv",
        observation_rows,
    )

    write_csv(
        output_root / "passenger_summary.csv",
        passenger_rows,
    )

    write_csv(
        output_root / "timer_step_deltas.csv",
        delta_rows,
    )

    with (
        output_root / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            jsonable(summary),
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (
        output_root / "resolved_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            jsonable(
                {
                    "project_root": project_root,
                    "uagmc_root": uagmc_root,
                    "model": model_path,
                    "vecnormalize": vecnorm_path,
                    "passenger_trace": passenger_path,
                    "candidate_vertiports": candidate_ids,
                    "to_vertiport": args.to_vertiport,
                    "device": device,
                }
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    env.close()

    print("\n" + "=" * 116)
    print("STAGE-0.8 SUMMARY")
    print("=" * 116)
    print(
        f"Resolved passengers                  : {len(passenger_rows)}"
    )
    print(
        f"Resolved timer observations          : {len(observation_rows)}"
    )
    print(
        f"Median timer slope / simulation step : {median_slope:.6f}"
    )
    print(
        f"Fraction slope near +1               : "
        f"{summary['timer_dynamics']['fraction_slope_near_plus_1']:.4f}"
    )
    print(
        f"Fraction slope near -1               : "
        f"{summary['timer_dynamics']['fraction_slope_near_minus_1']:.4f}"
    )
    print(
        f"Timer vs actual remaining correlation: "
        f"{summary['remaining_eta_hypothesis']['pearson_timer_vs_actual_remaining']:.4f}"
    )
    print(
        f"Remaining-ETA hypothesis MAE         : {remaining_mae:.6f}"
    )
    print(
        f"Elapsed-time hypothesis MAE          : {elapsed_mae:.6f}"
    )
    print(
        f"Semantic guess                       : {semantic_guess}"
    )
    print("-" * 116)
    print(summary["interpretation"])
    print(f"结果目录：{output_root}")
    print("=" * 116)


if __name__ == "__main__":
    main()
