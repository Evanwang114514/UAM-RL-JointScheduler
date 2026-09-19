# -*- coding: utf-8 -*-
"""
UAGMC Stage-0.7：Committed-Cohort Oracle Test（无训练）

目的
----
Stage-0.6 暴露出一个关键问题：
完整未来 Oracle state 会混入 decision time 之后才 reveal 的新 passenger，
因此不能公平评价“只处理当前 committed workload”的 candidate-relative projection。

本脚本改成严格的 same-cohort comparison：

对每个 passenger decision、每个 candidate vertiport：
1) 在 decision time 冻结该站当时已经存在的 background cohort：
   - 当前 waiting passenger IDs
   - 当前已经 enroute 到该站的 passenger IDs
   不包含当前正在等待 RL 动作的 focal passenger。
2) P0：保持该 cohort 的当前 waiting / incoming 状态。
3) P1：只按 committed access timer 推进 arrival，不做 service。
4) P2：按 committed access timer + causal historical service rate 推进。
5) Oracle-C：沿 checkpoint 的真实 rollout 到 candidate realization horizon，
   只检查“同一批冻结 passenger IDs”此时到底仍 incoming、waiting，还是已经离开 departure workload。
   decision time 之后才 reveal 的新 passenger 完全不计入 Oracle-C。

核心判据
--------
D(P2_C, Oracle_C) < D(P1_C, Oracle_C)
    -> service consumption 确实是 arrival-only peeling 的重要缺项。

D(P2_C, Oracle_C) < D(P0_C, Oracle_C)
    -> candidate-relative arrival+service peeling 对“当前 committed cohort”
       比原始 snapshot 更接近其 realization-time workload state。

重要边界
--------
- 完全不训练。
- Oracle-C 只用于离线诊断，不能作为在线 policy 输入。
- P2 的 service rate 只使用当前 decision 之前已发生的 queue departure history；
  不读取未来 passenger / 未来 action。
- Oracle-C 虽然排除了未来新 passenger 的直接计数污染，但真实 cohort 的实际 service
  仍可能间接受未来系统演化影响。因此本实验验证的是 projection 是否抓住主要 temporal
  mechanism，不等价于“未来 cohort 完全可由当前信息精确决定”。

依赖
----
请将本文件放在 UAGMC 源码根目录，并保留：
    diagnose_uagmc_candidate_temporal_mismatch.py
    diagnose_uagmc_arrival_service_peeling.py

代码注释全部使用中文。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import diagnose_uagmc_candidate_temporal_mismatch as stage0
except Exception as exc:
    raise RuntimeError(
        "无法导入 diagnose_uagmc_candidate_temporal_mismatch.py。\n"
        "请确认 Stage-0 脚本与本文件位于同一 UAGMC 根目录。"
    ) from exc

try:
    import diagnose_uagmc_arrival_service_peeling as stage06
except Exception as exc:
    raise RuntimeError(
        "无法导入 diagnose_uagmc_arrival_service_peeling.py。\n"
        "请确认 Stage-0.6 脚本与本文件位于同一 UAGMC 根目录。"
    ) from exc


# =============================================================================
# 基础工具
# =============================================================================

COHORT_STATES = ("waiting", "incoming", "served")


def mean_or_nan(values: Sequence[float]) -> float:
    vals = [
        float(x)
        for x in values
        if x is not None and np.isfinite(float(x))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def percentile_or_nan(values: Sequence[float], q: float) -> float:
    vals = [
        float(x)
        for x in values
        if x is not None and np.isfinite(float(x))
    ]
    return float(np.percentile(vals, q)) if vals else float("nan")


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


def argmin_stable(values: Dict[int, float]) -> int:
    return min(values.items(), key=lambda item: (item[1], item[0]))[0]


def _rankdata_average(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1

        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j

    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if np.isfinite(float(x)) and np.isfinite(float(y))
    ]

    if len(pairs) < 3:
        return float("nan")

    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)

    rx = _rankdata_average(x)
    ry = _rankdata_average(y)

    if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
        return float("nan")

    return float(np.corrcoef(rx, ry)[0, 1])


def cohort_tv_error(
    pred: Dict[str, float],
    oracle: Dict[str, float],
    cohort_size: int,
) -> float:
    """
    同一 cohort 三状态分布的 total-variation-style error。

    waiting + incoming + served 对同一 cohort 应守恒。
    0.5 * L1 / N 取值理论上约在 [0,1]：
        0 = cohort 状态分布完全一致；
        1 = 完全错位。
    """
    n = max(1.0, float(cohort_size))

    l1 = sum(
        abs(
            float(pred.get(k, 0.0))
            - float(oracle.get(k, 0.0))
        )
        for k in COHORT_STATES
    )

    return float(0.5 * l1 / n)


# =============================================================================
# Passenger identity / cohort 提取
# =============================================================================

def normalize_person_id(item: Any) -> str:
    """兼容 queue 中保存 id 或 person object 的不同实现。"""
    if isinstance(item, (str, int, np.integer)):
        return str(item)

    for attr in ("person_id", "id", "pid"):
        if hasattr(item, attr):
            try:
                return str(getattr(item, attr))
            except Exception:
                pass

    return repr(item)


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


def get_waiting_ids_by_station(
    scenario: Any,
    candidate_ids: Sequence[int],
) -> Dict[int, Set[str]]:
    out: Dict[int, Set[str]] = {}

    for vid in candidate_ids:
        vertiport = scenario.vertiports.vertiport_list[str(int(vid))]
        person_list = list(getattr(vertiport, "person_list", []))

        out[int(vid)] = {
            normalize_person_id(x)
            for x in person_list
        }

    return out


def get_enroute_ids_and_timers(
    scenario: Any,
    candidate_ids: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    """
    当前已经被过去 policy 分配、正在 ground access 的 committed passenger。
    """
    candidate_set = {int(v) for v in candidate_ids}

    out: Dict[int, Dict[str, float]] = {
        int(v): {}
        for v in candidate_ids
    }

    persons = get_person_dict(scenario)

    for pid, person in persons.items():
        if str(getattr(person, "state", "")).lower() != "enroute":
            continue

        try:
            vid = int(getattr(person, "origin_vertiport_id"))
        except Exception:
            continue

        if vid not in candidate_set:
            continue

        try:
            timer = float(getattr(person, "current_timer"))
        except Exception:
            continue

        if not np.isfinite(timer):
            continue

        out[vid][pid] = max(0.0, timer)

    return out


def freeze_background_cohort(
    scenario: Any,
    candidate_ids: Sequence[int],
    focal_pid: Optional[str],
) -> Dict[int, Dict[str, Any]]:
    """
    冻结当前已经属于各 candidate station 的 background committed cohort。

    focal passenger 当前还没做 action，因此不属于任何 candidate 的既有 committed workload。
    为避免 wrapper/source 版本差异导致它偶然出现在某个容器里，这里显式排除。
    """
    waiting = get_waiting_ids_by_station(
        scenario,
        candidate_ids,
    )

    enroute = get_enroute_ids_and_timers(
        scenario,
        candidate_ids,
    )

    focal = str(focal_pid) if focal_pid is not None else None

    out: Dict[int, Dict[str, Any]] = {}

    for vid in candidate_ids:
        vid = int(vid)

        waiting_ids = set(waiting.get(vid, set()))
        enroute_map = dict(enroute.get(vid, {}))

        if focal is not None:
            waiting_ids.discard(focal)
            enroute_map.pop(focal, None)

        # 理论上 waiting / enroute 应互斥；若源码状态短暂重叠，优先以 waiting 为准。
        for pid in list(enroute_map.keys()):
            if pid in waiting_ids:
                enroute_map.pop(pid, None)

        all_ids = set(waiting_ids) | set(enroute_map.keys())

        out[vid] = {
            "waiting_ids": waiting_ids,
            "enroute_timers": enroute_map,
            "cohort_ids": all_ids,
            "cohort_size": len(all_ids),
        }

    return out


# =============================================================================
# P0 / P1 / P2 cohort projection
# =============================================================================

def build_p0_cohort(
    cohort: Dict[str, Any],
) -> Dict[str, float]:
    waiting = float(len(cohort["waiting_ids"]))
    incoming = float(len(cohort["enroute_timers"]))

    return {
        "waiting": waiting,
        "incoming": incoming,
        "served": 0.0,
    }


def build_p1_cohort(
    cohort: Dict[str, Any],
    horizon: float,
) -> Dict[str, float]:
    """
    Arrival-only：
    committed enroute 在 timer 到期后转为 waiting，
    不考虑 service。
    """
    h = max(0.0, float(horizon))

    timers = list(cohort["enroute_timers"].values())

    arrived = sum(
        1
        for t in timers
        if float(t) <= h + 1e-12
    )

    still_incoming = sum(
        1
        for t in timers
        if float(t) > h + 1e-12
    )

    waiting = (
        len(cohort["waiting_ids"])
        + arrived
    )

    return {
        "waiting": float(waiting),
        "incoming": float(still_incoming),
        "served": 0.0,
    }


def build_p2_cohort(
    cohort: Dict[str, Any],
    horizon: float,
    service_rate: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Arrival + Service：
    使用 Stage-0.6 的 causal service proxy 和 fluid queue，
    只推进当前冻结 cohort。
    """
    h = max(0.0, float(horizon))

    arrival_times = list(
        cohort["enroute_timers"].values()
    )

    projected_waiting, service_used = (
        stage06.fluid_queue_with_committed_arrivals(
            initial_waiting=float(
                len(cohort["waiting_ids"])
            ),
            arrival_times=arrival_times,
            horizon=h,
            service_rate=float(service_rate),
        )
    )

    still_incoming = sum(
        1
        for t in arrival_times
        if float(t) > h + 1e-12
    )

    n = float(cohort["cohort_size"])

    # fluid queue 可能产生小数 served，这是有意保留的连续近似。
    projected_served = max(
        0.0,
        n
        - float(projected_waiting)
        - float(still_incoming),
    )

    pred = {
        "waiting": float(projected_waiting),
        "incoming": float(still_incoming),
        "served": float(projected_served),
    }

    meta = {
        "service_rate_estimate": float(service_rate),
        "estimated_service_used": float(service_used),
    }

    return pred, meta


# =============================================================================
# Oracle-C：只检查冻结 cohort 的真实未来状态
# =============================================================================

def observe_cohort_oracle(
    scenario: Any,
    vid: int,
    cohort_ids: Set[str],
) -> Dict[str, Any]:
    """
    在真实 rollout 的当前时刻，只对冻结 cohort IDs 分类。

    分类：
    - waiting：当前仍在该 candidate vertiport.person_list；
    - incoming：当前仍 state=enroute 且目标 departure vertiport 是该 candidate；
    - served：其余 frozen cohort IDs。

    “served”在这里更准确地说是“已经离开该 departure workload”：
    可能已进入载客飞行或完成行程。对 candidate departure workload 来说二者都代表
    不再占用该站的 waiting/incoming workload。
    """
    vid = int(vid)

    waiting_now = get_waiting_ids_by_station(
        scenario,
        [vid],
    )[vid]

    persons = get_person_dict(scenario)

    waiting_ids: Set[str] = set()
    incoming_ids: Set[str] = set()
    served_ids: Set[str] = set()
    missing_ids: Set[str] = set()

    for pid in cohort_ids:
        if pid in waiting_now:
            waiting_ids.add(pid)
            continue

        person = persons.get(str(pid))

        if person is None:
            # 极端情况下对象可能已被源码清理；从 departure workload 角度视为已离开，
            # 但单独记录 missing 便于审计。
            served_ids.add(pid)
            missing_ids.add(pid)
            continue

        state = str(getattr(person, "state", "")).lower()

        try:
            person_vid = int(getattr(person, "origin_vertiport_id"))
        except Exception:
            person_vid = None

        if state == "enroute" and person_vid == vid:
            incoming_ids.add(pid)
        else:
            served_ids.add(pid)

    return {
        "waiting": float(len(waiting_ids)),
        "incoming": float(len(incoming_ids)),
        "served": float(len(served_ids)),
        "waiting_ids": waiting_ids,
        "incoming_ids": incoming_ids,
        "served_ids": served_ids,
        "missing_ids": missing_ids,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "UAGMC Stage-0.7 Committed-Cohort Oracle Test，无训练"
        )
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
        "--service-window",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--service-estimator",
        choices=[
            "history_mean",
            "history_p75",
            "hybrid",
        ],
        default="history_mean",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "diagnostics/"
            "uagmc_committed_cohort_oracle"
        ),
    )

    return parser.parse_args()


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    args = parse_args()

    project_root = (
        Path(args.project_root)
        .expanduser()
        .resolve()
    )

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
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        VecNormalize,
    )
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

    device = stage0.resolve_device(
        args.device
    )

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

    if len(candidate_ids) != int(wrapper.action_space.n):
        raise RuntimeError(
            "candidate 数量与 action_space.n 不一致。"
        )

    service_tracker = stage06.CausalServiceRateTracker(
        candidate_ids=candidate_ids,
        window=args.service_window,
    )

    print("=" * 118)
    print(
        "UAGMC STAGE-0.7 | COMMITTED-COHORT ORACLE TEST | NO TRAINING"
    )
    print("=" * 118)
    print(f"Project root      : {project_root}")
    print(f"UAGMC root        : {uagmc_root}")
    print(f"Model             : {model_path}")
    print(f"VecNormalize      : {vecnorm_path}")
    print(f"Passengers        : {passenger_path}")
    print(f"Candidates        : {candidate_ids}")
    print(f"Service estimator : {args.service_estimator}")
    print(f"Service window    : {args.service_window}")
    print(f"Output            : {output_root}")
    print("-" * 118)

    # 每个 pending item 对应一个 decision-candidate，
    # 到其 realization horizon 时在真实 rollout 上读取 same-cohort Oracle。
    pending: List[Dict[str, Any]] = []
    resolved_rows: List[Dict[str, Any]] = []
    decision_meta: Dict[int, Dict[str, Any]] = {}

    obs = env.reset()
    done = False
    step_count = 0
    decision_counter = 0

    while not done:
        sim_time = int(
            getattr(
                scenario,
                "time",
                step_count,
            )
        )

        # 先解析已经到 horizon 的 pending candidate。
        still_pending: List[Dict[str, Any]] = []

        for item in pending:
            if sim_time < int(item["target_step"]):
                still_pending.append(item)
                continue

            oracle = observe_cohort_oracle(
                scenario=scenario,
                vid=int(item["candidate_vertiport"]),
                cohort_ids=set(item["cohort_ids"]),
            )

            row = dict(item["row_base"])

            for name, state in (
                ("p0", item["p0"]),
                ("p1", item["p1"]),
                ("p2", item["p2"]),
                ("oracle", oracle),
            ):
                for key in COHORT_STATES:
                    row[f"{name}_{key}"] = float(state[key])

            n = int(item["cohort_size"])

            row["error_p0_to_oracle"] = cohort_tv_error(
                item["p0"],
                oracle,
                n,
            )

            row["error_p1_to_oracle"] = cohort_tv_error(
                item["p1"],
                oracle,
                n,
            )

            row["error_p2_to_oracle"] = cohort_tv_error(
                item["p2"],
                oracle,
                n,
            )

            row["gain_p1_vs_p0"] = (
                row["error_p0_to_oracle"]
                - row["error_p1_to_oracle"]
            )

            row["gain_p2_vs_p0"] = (
                row["error_p0_to_oracle"]
                - row["error_p2_to_oracle"]
            )

            row["gain_p2_vs_p1"] = (
                row["error_p1_to_oracle"]
                - row["error_p2_to_oracle"]
            )

            row["p1_improves_over_p0"] = int(
                row["gain_p1_vs_p0"] > 1e-12
            )

            row["p2_improves_over_p0"] = int(
                row["gain_p2_vs_p0"] > 1e-12
            )

            row["p2_improves_over_p1"] = int(
                row["gain_p2_vs_p1"] > 1e-12
            )

            row["oracle_residual_workload"] = (
                float(oracle["waiting"])
                + float(oracle["incoming"])
            )

            row["p0_residual_workload"] = (
                float(item["p0"]["waiting"])
                + float(item["p0"]["incoming"])
            )

            row["p1_residual_workload"] = (
                float(item["p1"]["waiting"])
                + float(item["p1"]["incoming"])
            )

            row["p2_residual_workload"] = (
                float(item["p2"]["waiting"])
                + float(item["p2"]["incoming"])
            )

            row["oracle_missing_ids"] = int(
                len(oracle["missing_ids"])
            )

            resolved_rows.append(row)

        pending = still_pending

        # 当前 queue transition 更新 causal service history。
        current_waiting_ids = stage06.get_waiting_ids(
            scenario,
            candidate_ids,
        )

        service_tracker.observe(
            current_waiting_ids
        )

        focal_pid = stage0.get_waiting_pid(
            wrapper
        )

        probs = stage0.get_policy_probs(
            model,
            obs,
        )

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action_index = int(
            np.asarray(action).reshape(-1)[0]
        )

        if focal_pid is not None:
            decision_id = decision_counter
            decision_counter += 1

            access_times = stage0.estimate_access_times(
                scenario,
                focal_pid,
                candidate_ids,
            )

            frozen = freeze_background_cohort(
                scenario,
                candidate_ids,
                focal_pid,
            )

            if probs is None or len(probs) != len(candidate_ids):
                policy_probs = [float("nan")] * len(candidate_ids)
                policy_margin = float("nan")
            else:
                policy_probs = [float(x) for x in probs]
                ordered = sorted(policy_probs, reverse=True)
                policy_margin = (
                    ordered[0] - ordered[1]
                    if len(ordered) >= 2
                    else 1.0
                )

            decision_meta[decision_id] = {
                "decision_id": decision_id,
                "sim_time": sim_time,
                "pid": str(focal_pid),
                "chosen_vertiport": int(candidate_ids[action_index]),
                "policy_margin": float(policy_margin),
                "access_times": {
                    int(k): float(v)
                    for k, v in access_times.items()
                },
            }

            for candidate_action_index, vid in enumerate(candidate_ids):
                vid = int(vid)
                h = float(access_times[vid])
                cohort = frozen[vid]

                p0 = build_p0_cohort(
                    cohort
                )

                p1 = build_p1_cohort(
                    cohort,
                    h,
                )

                mu, service_meta = service_tracker.estimate(
                    scenario=scenario,
                    vid=vid,
                    mode=args.service_estimator,
                )

                p2, p2_meta = build_p2_cohort(
                    cohort=cohort,
                    horizon=h,
                    service_rate=mu,
                )

                row_base = {
                    "decision_id": decision_id,
                    "sim_time": sim_time,
                    "pid": str(focal_pid),
                    "candidate_action_index": candidate_action_index,
                    "candidate_vertiport": vid,
                    "checkpoint_choice": int(
                        candidate_action_index == action_index
                    ),
                    "policy_prob": float(
                        policy_probs[candidate_action_index]
                    ),
                    "policy_margin": float(policy_margin),
                    "access_time": h,
                    "candidate_realization_time": (
                        float(sim_time) + h
                    ),
                    "target_step": int(
                        math.ceil(
                            float(sim_time) + h
                        )
                    ),
                    "cohort_size": int(
                        cohort["cohort_size"]
                    ),
                    "cohort_waiting_at_decision": int(
                        len(cohort["waiting_ids"])
                    ),
                    "cohort_enroute_at_decision": int(
                        len(cohort["enroute_timers"])
                    ),
                    "service_rate_estimate": float(mu),
                    "service_history_n": int(
                        service_meta["service_history_n"]
                    ),
                    "service_history_mean": float(
                        service_meta["service_history_mean"]
                    ) if np.isfinite(
                        service_meta["service_history_mean"]
                    ) else "",
                    "service_history_p75": float(
                        service_meta["service_history_p75"]
                    ) if np.isfinite(
                        service_meta["service_history_p75"]
                    ) else "",
                    "current_idle_seat_capacity": float(
                        service_meta["current_idle_seat_capacity"]
                    ),
                    "estimated_service_used": float(
                        p2_meta["estimated_service_used"]
                    ),
                }

                pending.append(
                    {
                        "decision_id": decision_id,
                        "candidate_vertiport": vid,
                        "target_step": row_base["target_step"],
                        "cohort_ids": set(cohort["cohort_ids"]),
                        "cohort_size": int(cohort["cohort_size"]),
                        "p0": p0,
                        "p1": p1,
                        "p2": p2,
                        "row_base": row_base,
                    }
                )

        obs, _, dones, _ = env.step(
            action
        )

        done = bool(
            np.asarray(dones).reshape(-1)[0]
        )

        step_count += 1

        if step_count > args.max_time + 1000:
            print(
                "[警告] 超过安全 step 上限，强制结束。"
            )
            break

    # episode 结束后，尽量用 terminal state 解析已经到 horizon 的剩余项；
    # 还没到 target_step 的 item 不做伪造，直接记为 unresolved。
    terminal_time = int(
        getattr(
            scenario,
            "time",
            step_count,
        )
    )

    unresolved_rows: List[Dict[str, Any]] = []

    for item in pending:
        if terminal_time >= int(item["target_step"]):
            oracle = observe_cohort_oracle(
                scenario=scenario,
                vid=int(item["candidate_vertiport"]),
                cohort_ids=set(item["cohort_ids"]),
            )

            row = dict(item["row_base"])

            for name, state in (
                ("p0", item["p0"]),
                ("p1", item["p1"]),
                ("p2", item["p2"]),
                ("oracle", oracle),
            ):
                for key in COHORT_STATES:
                    row[f"{name}_{key}"] = float(state[key])

            n = int(item["cohort_size"])

            row["error_p0_to_oracle"] = cohort_tv_error(
                item["p0"],
                oracle,
                n,
            )
            row["error_p1_to_oracle"] = cohort_tv_error(
                item["p1"],
                oracle,
                n,
            )
            row["error_p2_to_oracle"] = cohort_tv_error(
                item["p2"],
                oracle,
                n,
            )

            row["gain_p1_vs_p0"] = (
                row["error_p0_to_oracle"]
                - row["error_p1_to_oracle"]
            )
            row["gain_p2_vs_p0"] = (
                row["error_p0_to_oracle"]
                - row["error_p2_to_oracle"]
            )
            row["gain_p2_vs_p1"] = (
                row["error_p1_to_oracle"]
                - row["error_p2_to_oracle"]
            )

            row["p1_improves_over_p0"] = int(
                row["gain_p1_vs_p0"] > 1e-12
            )
            row["p2_improves_over_p0"] = int(
                row["gain_p2_vs_p0"] > 1e-12
            )
            row["p2_improves_over_p1"] = int(
                row["gain_p2_vs_p1"] > 1e-12
            )

            row["oracle_residual_workload"] = (
                float(oracle["waiting"])
                + float(oracle["incoming"])
            )
            row["p0_residual_workload"] = (
                float(item["p0"]["waiting"])
                + float(item["p0"]["incoming"])
            )
            row["p1_residual_workload"] = (
                float(item["p1"]["waiting"])
                + float(item["p1"]["incoming"])
            )
            row["p2_residual_workload"] = (
                float(item["p2"]["waiting"])
                + float(item["p2"]["incoming"])
            )
            row["oracle_missing_ids"] = int(
                len(oracle["missing_ids"])
            )

            resolved_rows.append(row)
        else:
            unresolved_rows.append(
                dict(item["row_base"])
            )

    # =============================================================================
    # Decision-level ranking
    # =============================================================================

    by_decision: Dict[int, List[Dict[str, Any]]] = {}

    for row in resolved_rows:
        by_decision.setdefault(
            int(row["decision_id"]),
            [],
        ).append(row)

    decision_rows: List[Dict[str, Any]] = []

    for decision_id, rows in sorted(by_decision.items()):
        if len(rows) != len(candidate_ids):
            continue

        p0_workload = {
            int(r["candidate_vertiport"]): float(
                r["p0_residual_workload"]
            )
            for r in rows
        }

        p1_workload = {
            int(r["candidate_vertiport"]): float(
                r["p1_residual_workload"]
            )
            for r in rows
        }

        p2_workload = {
            int(r["candidate_vertiport"]): float(
                r["p2_residual_workload"]
            )
            for r in rows
        }

        oracle_workload = {
            int(r["candidate_vertiport"]): float(
                r["oracle_residual_workload"]
            )
            for r in rows
        }

        meta = decision_meta[decision_id]
        access_values = list(meta["access_times"].values())

        p0_best = argmin_stable(p0_workload)
        p1_best = argmin_stable(p1_workload)
        p2_best = argmin_stable(p2_workload)
        oracle_best = argmin_stable(oracle_workload)

        decision_rows.append(
            {
                "decision_id": decision_id,
                "sim_time": meta["sim_time"],
                "pid": meta["pid"],
                "chosen_vertiport": meta["chosen_vertiport"],
                "policy_margin": meta["policy_margin"],
                "access_delay_min": min(access_values),
                "access_delay_max": max(access_values),
                "access_delay_spread": (
                    max(access_values)
                    - min(access_values)
                ),
                "mean_error_p0": mean_or_nan(
                    [r["error_p0_to_oracle"] for r in rows]
                ),
                "mean_error_p1": mean_or_nan(
                    [r["error_p1_to_oracle"] for r in rows]
                ),
                "mean_error_p2": mean_or_nan(
                    [r["error_p2_to_oracle"] for r in rows]
                ),
                "p0_best_residual_workload": p0_best,
                "p1_best_residual_workload": p1_best,
                "p2_best_residual_workload": p2_best,
                "oracle_best_residual_workload": oracle_best,
                "p0_matches_oracle": int(p0_best == oracle_best),
                "p1_matches_oracle": int(p1_best == oracle_best),
                "p2_matches_oracle": int(p2_best == oracle_best),
            }
        )

    # =============================================================================
    # Horizon bins
    # =============================================================================

    horizon_bin_rows: List[Dict[str, Any]] = []

    if resolved_rows:
        horizons = np.asarray(
            [
                float(r["access_time"])
                for r in resolved_rows
            ],
            dtype=float,
        )

        edges = np.unique(
            np.quantile(
                horizons,
                [0.0, 0.25, 0.50, 0.75, 1.0],
            )
        )

        for i in range(len(edges) - 1):
            lo = float(edges[i])
            hi = float(edges[i + 1])

            if i == len(edges) - 2:
                rows = [
                    r
                    for r in resolved_rows
                    if lo <= float(r["access_time"]) <= hi
                ]
            else:
                rows = [
                    r
                    for r in resolved_rows
                    if lo <= float(r["access_time"]) < hi
                ]

            if not rows:
                continue

            horizon_bin_rows.append(
                {
                    "horizon_lo": lo,
                    "horizon_hi": hi,
                    "n": len(rows),
                    "mean_cohort_size": mean_or_nan(
                        [r["cohort_size"] for r in rows]
                    ),
                    "mean_error_p0": mean_or_nan(
                        [r["error_p0_to_oracle"] for r in rows]
                    ),
                    "mean_error_p1": mean_or_nan(
                        [r["error_p1_to_oracle"] for r in rows]
                    ),
                    "mean_error_p2": mean_or_nan(
                        [r["error_p2_to_oracle"] for r in rows]
                    ),
                    "p1_improve_over_p0_rate": mean_or_nan(
                        [r["p1_improves_over_p0"] for r in rows]
                    ),
                    "p2_improve_over_p0_rate": mean_or_nan(
                        [r["p2_improves_over_p0"] for r in rows]
                    ),
                    "p2_improve_over_p1_rate": mean_or_nan(
                        [r["p2_improves_over_p1"] for r in rows]
                    ),
                    "mean_service_rate_estimate": mean_or_nan(
                        [r["service_rate_estimate"] for r in rows]
                    ),
                    "mean_estimated_service_used": mean_or_nan(
                        [r["estimated_service_used"] for r in rows]
                    ),
                    "mean_oracle_residual_workload": mean_or_nan(
                        [r["oracle_residual_workload"] for r in rows]
                    ),
                }
            )

    # =============================================================================
    # Summary
    # =============================================================================

    e0 = [
        float(r["error_p0_to_oracle"])
        for r in resolved_rows
    ]
    e1 = [
        float(r["error_p1_to_oracle"])
        for r in resolved_rows
    ]
    e2 = [
        float(r["error_p2_to_oracle"])
        for r in resolved_rows
    ]

    access = [
        float(r["access_time"])
        for r in resolved_rows
    ]

    summary = {
        "stage": (
            "Stage-0.7 committed-cohort oracle no-training diagnostic"
        ),
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
        "service_estimator": args.service_estimator,
        "service_window": args.service_window,
        "recorded_decisions": decision_counter,
        "resolved_candidate_rows": len(resolved_rows),
        "complete_decisions": len(decision_rows),
        "unresolved_candidate_rows": len(unresolved_rows),
        "cohort_definition": (
            "background waiting + already-enroute passengers at decision time; "
            "focal undecided passenger excluded"
        ),
        "error_metric": (
            "0.5 * L1(waiting,incoming,served) / cohort_size"
        ),
        "cohort_size": {
            "mean": mean_or_nan(
                [r["cohort_size"] for r in resolved_rows]
            ),
            "median": percentile_or_nan(
                [r["cohort_size"] for r in resolved_rows],
                50,
            ),
            "p10": percentile_or_nan(
                [r["cohort_size"] for r in resolved_rows],
                10,
            ),
            "p90": percentile_or_nan(
                [r["cohort_size"] for r in resolved_rows],
                90,
            ),
            "zero_cohort_rate": mean_or_nan(
                [
                    int(int(r["cohort_size"]) == 0)
                    for r in resolved_rows
                ]
            ),
        },
        "distance_to_same_cohort_oracle": {
            "mean_p0_snapshot": mean_or_nan(e0),
            "mean_p1_arrival_only": mean_or_nan(e1),
            "mean_p2_arrival_service": mean_or_nan(e2),
            "median_p0_snapshot": percentile_or_nan(e0, 50),
            "median_p1_arrival_only": percentile_or_nan(e1, 50),
            "median_p2_arrival_service": percentile_or_nan(e2, 50),
            "p90_p0_snapshot": percentile_or_nan(e0, 90),
            "p90_p1_arrival_only": percentile_or_nan(e1, 90),
            "p90_p2_arrival_service": percentile_or_nan(e2, 90),
        },
        "improvement": {
            "mean_gain_p1_vs_p0": mean_or_nan(
                [r["gain_p1_vs_p0"] for r in resolved_rows]
            ),
            "mean_gain_p2_vs_p0": mean_or_nan(
                [r["gain_p2_vs_p0"] for r in resolved_rows]
            ),
            "mean_gain_p2_vs_p1": mean_or_nan(
                [r["gain_p2_vs_p1"] for r in resolved_rows]
            ),
            "p1_improves_over_p0_rate": mean_or_nan(
                [r["p1_improves_over_p0"] for r in resolved_rows]
            ),
            "p2_improves_over_p0_rate": mean_or_nan(
                [r["p2_improves_over_p0"] for r in resolved_rows]
            ),
            "p2_improves_over_p1_rate": mean_or_nan(
                [r["p2_improves_over_p1"] for r in resolved_rows]
            ),
        },
        "decision_ranking_match_to_same_cohort_oracle": {
            "p0": mean_or_nan(
                [r["p0_matches_oracle"] for r in decision_rows]
            ),
            "p1": mean_or_nan(
                [r["p1_matches_oracle"] for r in decision_rows]
            ),
            "p2": mean_or_nan(
                [r["p2_matches_oracle"] for r in decision_rows]
            ),
        },
        "horizon_relationship": {
            "rho_access_vs_p0_error": spearman_rho(access, e0),
            "rho_access_vs_p1_error": spearman_rho(access, e1),
            "rho_access_vs_p2_error": spearman_rho(access, e2),
            "rho_access_vs_actual_served_fraction": spearman_rho(
                access,
                [
                    (
                        float(r["oracle_served"])
                        / max(1.0, float(r["cohort_size"]))
                    )
                    for r in resolved_rows
                ],
            ),
        },
        "service_diagnostic": {
            "mean_estimated_service_used": mean_or_nan(
                [r["estimated_service_used"] for r in resolved_rows]
            ),
            "mean_actual_cohort_served": mean_or_nan(
                [r["oracle_served"] for r in resolved_rows]
            ),
            "mean_service_rate_estimate": mean_or_nan(
                [r["service_rate_estimate"] for r in resolved_rows]
            ),
            "mean_abs_service_count_error": mean_or_nan(
                [
                    abs(
                        float(r["estimated_service_used"])
                        - float(r["oracle_served"])
                    )
                    for r in resolved_rows
                ]
            ),
        },
        "oracle_audit": {
            "total_missing_ids": int(
                sum(
                    int(r["oracle_missing_ids"])
                    for r in resolved_rows
                )
            ),
            "rows_with_missing_ids": int(
                sum(
                    int(r["oracle_missing_ids"]) > 0
                    for r in resolved_rows
                )
            ),
        },
        "horizon_bins": horizon_bin_rows,
        "interpretation_guardrails": [
            (
                "Oracle-C 只统计 decision time 冻结的同一 cohort IDs，"
                "未来新 reveal passenger 不直接进入评价。"
            ),
            (
                "focal passenger 是当前待决策 action 本身，"
                "不属于 decision time 已经 committed 到任一 candidate 的 background cohort，"
                "因此本实验显式排除 focal passenger。"
            ),
            (
                "Oracle-C 中 served 表示 passenger 已离开 candidate departure workload；"
                "可能正在载客飞行或已完成行程。"
            ),
            (
                "P2 仍是 causal service proxy，不是 exact committed simulator。"
            ),
            (
                "真实 cohort service 仍可能间接受未来系统演化影响，"
                "所以本实验只检验 projection 是否抓住主要 temporal mechanism。"
            ),
            (
                "若 P2 显著优于 P0/P1，才值得进入 equal-information U1/U2/U3 训练；"
                "否则优先修正 service semantics，而不是增加复杂网络。"
            ),
        ],
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_root / "candidate_detail.csv",
        resolved_rows,
    )

    write_csv(
        output_root / "decision_summary.csv",
        decision_rows,
    )

    write_csv(
        output_root / "access_horizon_bins.csv",
        horizon_bin_rows,
    )

    write_csv(
        output_root / "unresolved_candidates.csv",
        unresolved_rows,
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
                    "service_estimator": args.service_estimator,
                    "service_window": args.service_window,
                    "cohort_definition": (
                        "background waiting + already-enroute at decision time; "
                        "focal passenger excluded"
                    ),
                }
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    env.close()

    print("\n" + "=" * 118)
    print("STAGE-0.7 SUMMARY")
    print("=" * 118)
    print(
        f"Recorded decisions             : {decision_counter}"
    )
    print(
        f"Resolved candidate rows        : {len(resolved_rows)}"
    )
    print(
        f"Complete decisions             : {len(decision_rows)}"
    )
    print(
        f"Mean cohort size               : "
        f"{summary['cohort_size']['mean']:.3f}"
    )
    print(
        f"Mean P0 -> Cohort Oracle error : "
        f"{summary['distance_to_same_cohort_oracle']['mean_p0_snapshot']:.6f}"
    )
    print(
        f"Mean P1 -> Cohort Oracle error : "
        f"{summary['distance_to_same_cohort_oracle']['mean_p1_arrival_only']:.6f}"
    )
    print(
        f"Mean P2 -> Cohort Oracle error : "
        f"{summary['distance_to_same_cohort_oracle']['mean_p2_arrival_service']:.6f}"
    )
    print(
        f"P2 improves over P0           : "
        f"{summary['improvement']['p2_improves_over_p0_rate']:.4f}"
    )
    print(
        f"P2 improves over P1           : "
        f"{summary['improvement']['p2_improves_over_p1_rate']:.4f}"
    )
    print(
        f"Oracle rank match P0/P1/P2    : "
        f"{summary['decision_ranking_match_to_same_cohort_oracle']['p0']:.4f} / "
        f"{summary['decision_ranking_match_to_same_cohort_oracle']['p1']:.4f} / "
        f"{summary['decision_ranking_match_to_same_cohort_oracle']['p2']:.4f}"
    )
    print(
        f"Mean est./actual cohort served: "
        f"{summary['service_diagnostic']['mean_estimated_service_used']:.3f} / "
        f"{summary['service_diagnostic']['mean_actual_cohort_served']:.3f}"
    )
    print("-" * 118)
    print(
        "注意：Oracle-C 只检查 decision time 冻结的同一 passenger cohort；"
        "未来新 reveal passenger 不进入计数。"
    )
    print(f"结果目录：{output_root}")
    print("=" * 118)


if __name__ == "__main__":
    main()
