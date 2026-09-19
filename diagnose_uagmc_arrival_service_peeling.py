# -*- coding: utf-8 -*-
"""
UAGMC Stage-0.6：Arrival-only vs Arrival+Service candidate peeling（无训练）

研究目的
--------
上一轮 Stage-0.5 已经发现：
    只把 committed passenger 从 incoming 搬到 waiting
    会严重高估 candidate realization horizon 上的 queue，
因为 access delay 内系统仍在持续服务 passenger。

本脚本继续完全不训练，比较四个对象：
    P0 Snapshot
    P1 Arrival-only peeling
    P2 Arrival + causal service peeling
    Oracle realized future

核心问题：
    在不读取未来 unrevealed passenger、也不使用未来 policy action 的条件下，
    给 committed arrival 补上“候选 horizon 内持续服务”之后，
    是否能明显缩小到 Oracle future 的 demand-state distance？

重要信息边界
------------
1. Oracle 只用于离线诊断，绝不作为在线输入。
2. P1 只使用当前已经 enroute 的 passenger timer。
3. P2 的 service rate 只使用“当前决策之前已经观察到的 queue departure history”。
4. 对历史不足的 station，使用当前 IDLE eVTOL 的 seat capacity 作为冷启动 proxy；
   如果当前源码中无法可靠读取，则退化到 0，不会偷看未来。
5. 本实验仍不证明新 representation 会降低 ATT，只验证 service consumption
   是否是 Stage-0.5 arrival-only projection 失败的主要缺项。

依赖
----
请把本文件和 Stage-0 脚本放在同一 UAGMC 根目录：
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
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import diagnose_uagmc_candidate_temporal_mismatch as stage0
except Exception as exc:
    raise RuntimeError(
        "无法导入 diagnose_uagmc_candidate_temporal_mismatch.py。\n"
        "请确认本脚本与 Stage-0 脚本都位于 UAGMC 源码根目录。"
    ) from exc


# =============================================================================
# 通用工具
# =============================================================================

DEMAND_KEYS = (
    "waiting",
    "incoming",
    "min_incoming_passenger_time",
    "avg_incoming_passenger_time",
)


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
    if isinstance(x, (list, tuple)):
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


def demand_distance(
    a: Dict[str, float],
    b: Dict[str, float],
) -> float:
    """
    只比较 passenger demand-side state。

    这里故意不把 charging / aircraft state 混进来，
    因为本轮只验证 arrival + service 是否能修复 passenger peeling。
    """
    vals: List[float] = []

    for key in DEMAND_KEYS:
        av = float(a.get(key, 0.0))
        bv = float(b.get(key, 0.0))
        vals.append(abs(bv - av) / (1.0 + abs(av)))

    return float(np.mean(vals)) if vals else 0.0


def burden_proxy(state: Dict[str, float]) -> float:
    return (
        float(state.get("waiting", 0.0))
        + float(state.get("incoming", 0.0))
    )


def _rankdata_average(values: Sequence[float]) -> np.ndarray:
    """不依赖 scipy 的平均秩实现。"""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1

        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def spearman_rho(
    xs: Sequence[float],
    ys: Sequence[float],
) -> float:
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

    if float(np.std(rx)) <= 1e-12:
        return float("nan")
    if float(np.std(ry)) <= 1e-12:
        return float("nan")

    return float(np.corrcoef(rx, ry)[0, 1])


# =============================================================================
# 当前 committed passenger 与队列身份
# =============================================================================

def normalize_person_id(item: Any) -> str:
    """
    尽可能把 vertiport.person_list 中的元素变成稳定 passenger id。
    兼容字符串 id / 数值 id / person object。
    """
    if isinstance(item, (str, int, np.integer)):
        return str(item)

    for attr in ("person_id", "id", "pid"):
        if hasattr(item, attr):
            try:
                return str(getattr(item, attr))
            except Exception:
                pass

    return repr(item)


def get_waiting_ids(
    scenario: Any,
    candidate_ids: Sequence[int],
) -> Dict[int, set]:
    """读取当前真实 queue 中的 passenger identity。"""
    out: Dict[int, set] = {}

    for vid in candidate_ids:
        vertiport = scenario.vertiports.vertiport_list[str(int(vid))]
        person_list = getattr(vertiport, "person_list", [])
        out[int(vid)] = {
            normalize_person_id(x)
            for x in list(person_list)
        }

    return out


def collect_committed_passenger_timers(
    scenario: Any,
    candidate_ids: Sequence[int],
) -> Dict[int, List[float]]:
    """
    只读取当前已经 enroute 到 departure vertiport 的 passenger。

    未 reveal passenger 不会进入这里。
    """
    out: Dict[int, List[float]] = {
        int(v): []
        for v in candidate_ids
    }

    persons_obj = getattr(scenario, "persons", None)
    persons = (
        getattr(persons_obj, "persons", {})
        if persons_obj is not None
        else {}
    )

    for person in persons.values():
        if str(getattr(person, "state", "")).lower() != "enroute":
            continue

        try:
            vid = int(getattr(person, "origin_vertiport_id"))
        except Exception:
            continue

        if vid not in out:
            continue

        try:
            timer = float(getattr(person, "current_timer"))
        except Exception:
            continue

        if np.isfinite(timer):
            out[vid].append(max(0.0, timer))

    return out


# =============================================================================
# 当前 aircraft service-capacity proxy
# =============================================================================

def evtol_state_name(evtol: Any) -> str:
    try:
        return str(evtol.state.name)
    except Exception:
        return str(getattr(evtol, "state", "UNKNOWN"))


def current_idle_seat_capacity(
    scenario: Any,
    vid: int,
) -> float:
    """
    冷启动时使用的当前物理 service-capacity proxy。

    只统计当前已经位于该站且 state=IDLE 的 eVTOL seat capacity。
    不预测未来飞机，不生成 replacement，不读取未来 rollout。
    """
    try:
        evtols = list(
            scenario.vertiports.evtols_at_vertiport.get(
                str(int(vid)),
                [],
            )
        )
    except Exception:
        return 0.0

    total = 0.0

    for ev in evtols:
        if evtol_state_name(ev).upper() != "IDLE":
            continue

        try:
            cap = float(ev.spec.capacity)
        except Exception:
            cap = 0.0

        if np.isfinite(cap) and cap > 0.0:
            total += cap

    return float(total)


# =============================================================================
# 因果历史 service-rate estimator
# =============================================================================

class CausalServiceRateTracker:
    """
    只利用当前时刻以前已经发生的 queue departure history。

    departure 的定义：
        前一时刻真实 waiting queue 中存在，
        当前时刻已经从该 queue 消失的 passenger 数。

    在当前 UAGMC departure queue 中没有显式 cancellation 的前提下，
    这个量可作为“已经发生的服务出队”诊断 proxy。

    为避免需求不足把 service capacity 低估得太严重，
    只在前一时刻 queue 非空时把 departure 样本加入历史。
    """

    def __init__(
        self,
        candidate_ids: Sequence[int],
        window: int = 30,
    ):
        self.candidate_ids = [int(v) for v in candidate_ids]
        self.window = int(window)

        self.history: Dict[int, Deque[float]] = {
            v: deque(maxlen=self.window)
            for v in self.candidate_ids
        }

        self.prev_waiting_ids: Optional[Dict[int, set]] = None

    def observe(
        self,
        current_waiting_ids: Dict[int, set],
    ) -> Dict[int, float]:
        """
        用“上一时刻 -> 当前时刻”已经发生的 transition 更新历史。

        注意：调用顺序必须是每个 simulation step 开头先 observe，
        再用 estimate() 给当前 passenger decision 做 projection。
        这样 estimate 不会看到当前时刻之后的数据。
        """
        departed_now = {
            v: 0.0
            for v in self.candidate_ids
        }

        if self.prev_waiting_ids is not None:
            for v in self.candidate_ids:
                prev_ids = self.prev_waiting_ids.get(v, set())
                now_ids = current_waiting_ids.get(v, set())

                departed = len(prev_ids - now_ids)
                departed_now[v] = float(departed)

                if len(prev_ids) > 0:
                    self.history[v].append(float(departed))

        self.prev_waiting_ids = {
            v: set(current_waiting_ids.get(v, set()))
            for v in self.candidate_ids
        }

        return departed_now

    def estimate(
        self,
        scenario: Any,
        vid: int,
        mode: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        返回当前时刻可用的 causal service-rate estimate，单位 pax/min。

        mode:
            history_mean:
                有历史时只用 backlog 条件下的历史平均 departure rate；
                无历史时用 current idle-seat capacity 冷启动。
            history_p75:
                有历史时用 75% 分位数；
                无历史时用 current idle-seat capacity。
            hybrid:
                历史估计与当前 idle-seat capacity 都有时取 min，
                避免历史吞吐超过当前瞬时物理 proxy；
                若只有一个可用则使用可用者。
        """
        vid = int(vid)
        hist = list(self.history[vid])

        history_mean = (
            float(np.mean(hist))
            if hist
            else float("nan")
        )

        history_p75 = (
            float(np.percentile(hist, 75))
            if hist
            else float("nan")
        )

        idle_capacity = current_idle_seat_capacity(
            scenario,
            vid,
        )

        if mode == "history_mean":
            mu = (
                history_mean
                if np.isfinite(history_mean)
                else idle_capacity
            )
        elif mode == "history_p75":
            mu = (
                history_p75
                if np.isfinite(history_p75)
                else idle_capacity
            )
        elif mode == "hybrid":
            candidates = []

            if np.isfinite(history_mean):
                candidates.append(history_mean)

            if idle_capacity > 0.0:
                candidates.append(idle_capacity)

            mu = min(candidates) if candidates else 0.0
        else:
            raise ValueError(
                f"未知 service estimator mode：{mode}"
            )

        mu = max(0.0, float(mu))

        meta = {
            "service_history_n": len(hist),
            "service_history_mean": history_mean,
            "service_history_p75": history_p75,
            "current_idle_seat_capacity": idle_capacity,
            "service_rate_estimate": mu,
            "service_estimator_mode": mode,
        }

        return mu, meta


# =============================================================================
# P1 / P2 candidate-relative passenger projection
# =============================================================================

def build_arrival_only_projection(
    snapshot: Dict[str, float],
    timers: Sequence[float],
    horizon: float,
) -> Dict[str, float]:
    """
    P1：上一轮 arrival-only peeling。

    只做：
        incoming timer <= horizon  -> waiting
    不消化 queue。
    """
    h = max(0.0, float(horizon))
    timers = [max(0.0, float(t)) for t in timers]

    arrived = [
        t
        for t in timers
        if t <= h + 1e-12
    ]

    remaining = [
        t - h
        for t in timers
        if t > h + 1e-12
    ]

    out = dict(snapshot)

    out["waiting"] = (
        float(snapshot.get("waiting", 0.0))
        + float(len(arrived))
    )

    out["incoming"] = float(len(remaining))

    out["min_incoming_passenger_time"] = (
        float(min(remaining))
        if remaining
        else 0.0
    )

    out["avg_incoming_passenger_time"] = (
        float(np.mean(remaining))
        if remaining
        else 0.0
    )

    return out


def fluid_queue_with_committed_arrivals(
    initial_waiting: float,
    arrival_times: Sequence[float],
    horizon: float,
    service_rate: float,
) -> Tuple[float, float]:
    """
    用一个极简的 deterministic fluid queue 推进 passenger demand。

    当前 queue 以 rate=service_rate 持续被服务；
    committed passenger 在各自 timer 时刻加入 queue；
    不加入未来 unrevealed passenger。

    返回：
        projected_waiting
        estimated_service_used
    """
    h = max(0.0, float(horizon))
    mu = max(0.0, float(service_rate))
    q = max(0.0, float(initial_waiting))

    events = sorted(
        max(0.0, float(t))
        for t in arrival_times
        if float(t) <= h + 1e-12
    )

    last_t = 0.0
    service_used = 0.0

    for event_t in events:
        dt = max(0.0, event_t - last_t)

        potential = mu * dt
        actual = min(q, potential)

        q -= actual
        service_used += actual

        # 一个 committed passenger 完成 access，进入 queue。
        q += 1.0
        last_t = event_t

    # 最后一个 committed arrival 到 candidate horizon 之间继续服务。
    dt = max(0.0, h - last_t)
    potential = mu * dt
    actual = min(q, potential)

    q -= actual
    service_used += actual

    return max(0.0, q), service_used


def build_arrival_service_projection(
    snapshot: Dict[str, float],
    timers: Sequence[float],
    horizon: float,
    service_rate: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    P2：Arrival + causal service peeling。

    核心：
        Q_proj = resolve(
            current queue,
            committed arrival timers,
            causal service-rate estimate,
            candidate horizon
        )
    """
    h = max(0.0, float(horizon))
    timers = [max(0.0, float(t)) for t in timers]

    projected_waiting, service_used = (
        fluid_queue_with_committed_arrivals(
            initial_waiting=float(
                snapshot.get("waiting", 0.0)
            ),
            arrival_times=timers,
            horizon=h,
            service_rate=service_rate,
        )
    )

    remaining = [
        t - h
        for t in timers
        if t > h + 1e-12
    ]

    arrived_count = sum(
        1
        for t in timers
        if t <= h + 1e-12
    )

    out = dict(snapshot)

    out["waiting"] = float(projected_waiting)
    out["incoming"] = float(len(remaining))

    out["min_incoming_passenger_time"] = (
        float(min(remaining))
        if remaining
        else 0.0
    )

    out["avg_incoming_passenger_time"] = (
        float(np.mean(remaining))
        if remaining
        else 0.0
    )

    meta = {
        "committed_arrivals_by_horizon": float(
            arrived_count
        ),
        "estimated_service_used": float(
            service_used
        ),
        "projected_waiting": float(
            projected_waiting
        ),
    }

    return out, meta


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "UAGMC Stage-0.6 Arrival-only vs "
            "Arrival+Service peeling，无训练"
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
        help="历史 service departure 窗口长度，默认 30 个 simulation steps。",
    )

    parser.add_argument(
        "--service-estimator",
        choices=[
            "history_mean",
            "history_p75",
            "hybrid",
        ],
        default="history_mean",
        help=(
            "P2 的 causal service-rate estimator。"
            "默认 history_mean。"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "diagnostics/"
            "uagmc_arrival_service_peeling"
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

    if not project_root.exists():
        raise FileNotFoundError(project_root)

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

    sys.path.insert(
        0,
        str(uagmc_root),
    )

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

    run_stamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    output_root = (
        project_root
        / args.output_dir
        / run_stamp
    )

    monitor_dir = (
        output_root
        / "_monitor"
    )

    monitor_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = DummyVecEnv(
        [
            make_env(
                max_time=args.max_time,
                log_dir=monitor_dir,
                env_index=0,
                candidate_from_vertiports=(
                    initial_candidates
                ),
                to_vertiport=(
                    args.to_vertiport
                ),
                person_spawn_file=str(
                    passenger_path
                ),
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

    wrapper = stage0.unwrap_uagmc_wrapper(
        env
    )

    scenario = wrapper.env

    candidate_ids = stage0.get_candidate_ids(
        wrapper,
        args.candidates,
    )

    if len(candidate_ids) != int(
        wrapper.action_space.n
    ):
        raise RuntimeError(
            "candidate 数量与 action_space.n 不一致。"
        )

    service_tracker = (
        CausalServiceRateTracker(
            candidate_ids=candidate_ids,
            window=args.service_window,
        )
    )

    print("=" * 116)
    print(
        "UAGMC STAGE-0.6 | "
        "SNAPSHOT vs ARRIVAL-ONLY vs "
        "ARRIVAL+SERVICE vs ORACLE | "
        "NO TRAINING"
    )
    print("=" * 116)
    print(f"Project root      : {project_root}")
    print(f"UAGMC root        : {uagmc_root}")
    print(f"Model             : {model_path}")
    print(f"VecNormalize      : {vecnorm_path}")
    print(f"Passengers        : {passenger_path}")
    print(f"Candidates        : {candidate_ids}")
    print(
        f"Service estimator : "
        f"{args.service_estimator}"
    )
    print(
        f"Service window    : "
        f"{args.service_window}"
    )
    print(f"Output            : {output_root}")
    print("-" * 116)

    # -------------------------------------------------------------------------
    # 第一遍：沿 checkpoint 真实 rollout 记录 oracle future，
    # 同时在每个 decision 时刻只用过去 history 构造 P1 / P2。
    # -------------------------------------------------------------------------

    oracle_snapshots: Dict[
        int,
        Dict[int, Dict[str, float]]
    ] = {}

    decisions: List[
        Dict[str, Any]
    ] = []

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

        current_state = (
            stage0.extract_vertiport_state(
                scenario,
                candidate_ids,
            )
        )

        oracle_snapshots[
            sim_time
        ] = current_state

        # 先用当前 queue identity 更新“过去已经发生”的 service history。
        current_waiting_ids = get_waiting_ids(
            scenario,
            candidate_ids,
        )

        departed_last_step = (
            service_tracker.observe(
                current_waiting_ids
            )
        )

        pid = stage0.get_waiting_pid(
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
            np.asarray(action)
            .reshape(-1)[0]
        )

        if pid is not None:
            access_times = (
                stage0.estimate_access_times(
                    scenario,
                    pid,
                    candidate_ids,
                )
            )

            committed_timers = (
                collect_committed_passenger_timers(
                    scenario,
                    candidate_ids,
                )
            )

            if (
                probs is None
                or len(probs)
                != len(candidate_ids)
            ):
                policy_probs = [
                    float("nan")
                ] * len(candidate_ids)

                policy_margin = float(
                    "nan"
                )
            else:
                policy_probs = [
                    float(x)
                    for x in probs
                ]

                ordered = sorted(
                    policy_probs,
                    reverse=True,
                )

                policy_margin = (
                    float(
                        ordered[0]
                        - ordered[1]
                    )
                    if len(ordered) >= 2
                    else 1.0
                )

            p1_by_candidate = {}
            p2_by_candidate = {}
            service_meta_by_candidate = {}

            for vid in candidate_ids:
                vid = int(vid)
                h = float(
                    access_times[vid]
                )

                timers = list(
                    committed_timers.get(
                        vid,
                        [],
                    )
                )

                p1 = (
                    build_arrival_only_projection(
                        snapshot=(
                            current_state[vid]
                        ),
                        timers=timers,
                        horizon=h,
                    )
                )

                mu, service_meta = (
                    service_tracker.estimate(
                        scenario=scenario,
                        vid=vid,
                        mode=(
                            args.service_estimator
                        ),
                    )
                )

                p2, p2_meta = (
                    build_arrival_service_projection(
                        snapshot=(
                            current_state[vid]
                        ),
                        timers=timers,
                        horizon=h,
                        service_rate=mu,
                    )
                )

                service_meta.update(
                    p2_meta
                )

                service_meta[
                    "departed_last_observed_step"
                ] = float(
                    departed_last_step.get(
                        vid,
                        0.0,
                    )
                )

                p1_by_candidate[vid] = p1
                p2_by_candidate[vid] = p2

                service_meta_by_candidate[
                    vid
                ] = service_meta

            decisions.append(
                {
                    "decision_id": (
                        len(decisions)
                    ),
                    "sim_time": sim_time,
                    "pid": pid,
                    "chosen_action_index": (
                        action_index
                    ),
                    "chosen_vertiport": int(
                        candidate_ids[
                            action_index
                        ]
                    ),
                    "access_times": (
                        access_times
                    ),
                    "policy_probs": (
                        policy_probs
                    ),
                    "policy_margin": (
                        policy_margin
                    ),
                    "snapshot": (
                        current_state
                    ),
                    "p1_arrival_only": (
                        p1_by_candidate
                    ),
                    "p2_arrival_service": (
                        p2_by_candidate
                    ),
                    "service_meta": (
                        service_meta_by_candidate
                    ),
                }
            )

        obs, _, dones, _ = env.step(
            action
        )

        done = bool(
            np.asarray(dones)
            .reshape(-1)[0]
        )

        step_count += 1

        if (
            step_count
            > args.max_time + 1000
        ):
            print(
                "[警告] 超过安全 step 上限，"
                "强制结束。"
            )
            break

    final_time = int(
        getattr(
            scenario,
            "time",
            step_count,
        )
    )

    if (
        final_time
        not in oracle_snapshots
    ):
        oracle_snapshots[
            final_time
        ] = (
            stage0.extract_vertiport_state(
                scenario,
                candidate_ids,
            )
        )

    available_times = sorted(
        oracle_snapshots.keys()
    )

    def future_snapshot_time(
        target_time: float,
    ) -> Optional[int]:
        target = int(
            math.ceil(target_time)
        )

        for t in available_times:
            if t >= target:
                return t

        return None

    # -------------------------------------------------------------------------
    # 第二遍：对 P0 / P1 / P2 与 Oracle 做等信息比较。
    # -------------------------------------------------------------------------

    candidate_rows: List[
        Dict[str, Any]
    ] = []

    decision_rows: List[
        Dict[str, Any]
    ] = []

    d_p0_all: List[float] = []
    d_p1_all: List[float] = []
    d_p2_all: List[float] = []

    p1_gain_all: List[float] = []
    p2_gain_all: List[float] = []

    p2_vs_p1_gain_all: List[float] = []

    access_all: List[float] = []
    mu_all: List[float] = []
    service_used_all: List[float] = []

    for decision in decisions:
        p0_waiting: Dict[int, float] = {}
        p1_waiting: Dict[int, float] = {}
        p2_waiting: Dict[int, float] = {}
        oracle_waiting: Dict[int, float] = {}

        p0_burden: Dict[int, float] = {}
        p1_burden: Dict[int, float] = {}
        p2_burden: Dict[int, float] = {}
        oracle_burden: Dict[int, float] = {}

        per_dec_p0: List[float] = []
        per_dec_p1: List[float] = []
        per_dec_p2: List[float] = []

        all_future_available = True

        for action_index, vid in enumerate(
            candidate_ids
        ):
            vid = int(vid)

            h = float(
                decision["access_times"][
                    vid
                ]
            )

            target_time = (
                float(
                    decision["sim_time"]
                )
                + h
            )

            future_t = (
                future_snapshot_time(
                    target_time
                )
            )

            p0 = (
                decision["snapshot"][
                    vid
                ]
            )

            p1 = (
                decision[
                    "p1_arrival_only"
                ][vid]
            )

            p2 = (
                decision[
                    "p2_arrival_service"
                ][vid]
            )

            service_meta = (
                decision[
                    "service_meta"
                ][vid]
            )

            row: Dict[str, Any] = {
                "decision_id": (
                    decision[
                        "decision_id"
                    ]
                ),
                "sim_time": (
                    decision[
                        "sim_time"
                    ]
                ),
                "pid": (
                    decision["pid"]
                ),
                "candidate_action_index": (
                    action_index
                ),
                "candidate_vertiport": (
                    vid
                ),
                "checkpoint_choice": int(
                    vid
                    == decision[
                        "chosen_vertiport"
                    ]
                ),
                "access_time": h,
                "candidate_realization_time": (
                    target_time
                ),
                "future_snapshot_time": (
                    future_t
                    if future_t
                    is not None
                    else ""
                ),
                "policy_prob": (
                    decision[
                        "policy_probs"
                    ][action_index]
                ),
                "policy_margin": (
                    decision[
                        "policy_margin"
                    ]
                ),
            }

            for key, value in (
                service_meta.items()
            ):
                row[key] = value

            for key in DEMAND_KEYS:
                row[
                    f"p0_snapshot_{key}"
                ] = float(
                    p0.get(
                        key,
                        0.0,
                    )
                )

                row[
                    f"p1_arrival_only_{key}"
                ] = float(
                    p1.get(
                        key,
                        0.0,
                    )
                )

                row[
                    f"p2_arrival_service_{key}"
                ] = float(
                    p2.get(
                        key,
                        0.0,
                    )
                )

            p0_waiting[vid] = float(
                p0.get(
                    "waiting",
                    0.0,
                )
            )

            p1_waiting[vid] = float(
                p1.get(
                    "waiting",
                    0.0,
                )
            )

            p2_waiting[vid] = float(
                p2.get(
                    "waiting",
                    0.0,
                )
            )

            p0_burden[vid] = (
                burden_proxy(p0)
            )

            p1_burden[vid] = (
                burden_proxy(p1)
            )

            p2_burden[vid] = (
                burden_proxy(p2)
            )

            if future_t is None:
                all_future_available = (
                    False
                )
                row[
                    "future_available"
                ] = 0
                candidate_rows.append(
                    row
                )
                continue

            oracle = (
                oracle_snapshots[
                    future_t
                ][vid]
            )

            row[
                "future_available"
            ] = 1

            for key in DEMAND_KEYS:
                row[
                    f"oracle_{key}"
                ] = float(
                    oracle.get(
                        key,
                        0.0,
                    )
                )

            d_p0 = demand_distance(
                p0,
                oracle,
            )

            d_p1 = demand_distance(
                p1,
                oracle,
            )

            d_p2 = demand_distance(
                p2,
                oracle,
            )

            gain_p1 = (
                d_p0 - d_p1
            )

            gain_p2 = (
                d_p0 - d_p2
            )

            gain_p2_vs_p1 = (
                d_p1 - d_p2
            )

            row[
                "distance_p0_snapshot_to_oracle"
            ] = d_p0

            row[
                "distance_p1_arrival_only_to_oracle"
            ] = d_p1

            row[
                "distance_p2_arrival_service_to_oracle"
            ] = d_p2

            row[
                "gain_p1_vs_p0"
            ] = gain_p1

            row[
                "gain_p2_vs_p0"
            ] = gain_p2

            row[
                "gain_p2_vs_p1"
            ] = gain_p2_vs_p1

            row[
                "p1_improves_over_snapshot"
            ] = int(
                gain_p1 > 0.0
            )

            row[
                "p2_improves_over_snapshot"
            ] = int(
                gain_p2 > 0.0
            )

            row[
                "p2_improves_over_p1"
            ] = int(
                gain_p2_vs_p1
                > 0.0
            )

            oracle_waiting[vid] = (
                float(
                    oracle.get(
                        "waiting",
                        0.0,
                    )
                )
            )

            oracle_burden[vid] = (
                burden_proxy(
                    oracle
                )
            )

            per_dec_p0.append(
                d_p0
            )

            per_dec_p1.append(
                d_p1
            )

            per_dec_p2.append(
                d_p2
            )

            d_p0_all.append(
                d_p0
            )

            d_p1_all.append(
                d_p1
            )

            d_p2_all.append(
                d_p2
            )

            p1_gain_all.append(
                gain_p1
            )

            p2_gain_all.append(
                gain_p2
            )

            p2_vs_p1_gain_all.append(
                gain_p2_vs_p1
            )

            access_all.append(
                h
            )

            mu_all.append(
                float(
                    service_meta[
                        "service_rate_estimate"
                    ]
                )
            )

            service_used_all.append(
                float(
                    service_meta[
                        "estimated_service_used"
                    ]
                )
            )

            candidate_rows.append(
                row
            )

        if (
            not all_future_available
            or len(
                oracle_waiting
            )
            != len(
                candidate_ids
            )
        ):
            continue

        p0_best_waiting = (
            argmin_stable(
                p0_waiting
            )
        )

        p1_best_waiting = (
            argmin_stable(
                p1_waiting
            )
        )

        p2_best_waiting = (
            argmin_stable(
                p2_waiting
            )
        )

        oracle_best_waiting = (
            argmin_stable(
                oracle_waiting
            )
        )

        p0_best_burden = (
            argmin_stable(
                p0_burden
            )
        )

        p1_best_burden = (
            argmin_stable(
                p1_burden
            )
        )

        p2_best_burden = (
            argmin_stable(
                p2_burden
            )
        )

        oracle_best_burden = (
            argmin_stable(
                oracle_burden
            )
        )

        access_values = [
            float(x)
            for x in decision[
                "access_times"
            ].values()
        ]

        decision_rows.append(
            {
                "decision_id": (
                    decision[
                        "decision_id"
                    ]
                ),
                "sim_time": (
                    decision[
                        "sim_time"
                    ]
                ),
                "pid": (
                    decision["pid"]
                ),
                "chosen_vertiport": (
                    decision[
                        "chosen_vertiport"
                    ]
                ),
                "access_delay_min": (
                    min(
                        access_values
                    )
                ),
                "access_delay_max": (
                    max(
                        access_values
                    )
                ),
                "access_delay_spread": (
                    max(
                        access_values
                    )
                    - min(
                        access_values
                    )
                ),
                "policy_margin": (
                    decision[
                        "policy_margin"
                    ]
                ),
                "mean_distance_p0_to_oracle": (
                    mean_or_nan(
                        per_dec_p0
                    )
                ),
                "mean_distance_p1_to_oracle": (
                    mean_or_nan(
                        per_dec_p1
                    )
                ),
                "mean_distance_p2_to_oracle": (
                    mean_or_nan(
                        per_dec_p2
                    )
                ),
                "p0_best_waiting": (
                    p0_best_waiting
                ),
                "p1_best_waiting": (
                    p1_best_waiting
                ),
                "p2_best_waiting": (
                    p2_best_waiting
                ),
                "oracle_best_waiting": (
                    oracle_best_waiting
                ),
                "p0_matches_oracle_waiting": int(
                    p0_best_waiting
                    == oracle_best_waiting
                ),
                "p1_matches_oracle_waiting": int(
                    p1_best_waiting
                    == oracle_best_waiting
                ),
                "p2_matches_oracle_waiting": int(
                    p2_best_waiting
                    == oracle_best_waiting
                ),
                "p0_best_burden": (
                    p0_best_burden
                ),
                "p1_best_burden": (
                    p1_best_burden
                ),
                "p2_best_burden": (
                    p2_best_burden
                ),
                "oracle_best_burden": (
                    oracle_best_burden
                ),
                "p0_matches_oracle_burden": int(
                    p0_best_burden
                    == oracle_best_burden
                ),
                "p1_matches_oracle_burden": int(
                    p1_best_burden
                    == oracle_best_burden
                ),
                "p2_matches_oracle_burden": int(
                    p2_best_burden
                    == oracle_best_burden
                ),
            }
        )

    # -------------------------------------------------------------------------
    # 按 absolute candidate access horizon 分桶。
    # -------------------------------------------------------------------------

    usable_candidates = [
        row
        for row in candidate_rows
        if int(
            row.get(
                "future_available",
                0,
            )
        ) == 1
        and np.isfinite(
            float(
                row.get(
                    "distance_p0_snapshot_to_oracle",
                    float("nan"),
                )
            )
        )
    ]

    horizon_bin_rows: List[
        Dict[str, Any]
    ] = []

    if usable_candidates:
        horizons = np.asarray(
            [
                float(
                    r["access_time"]
                )
                for r in usable_candidates
            ],
            dtype=float,
        )

        edges = np.unique(
            np.quantile(
                horizons,
                [
                    0.0,
                    0.25,
                    0.50,
                    0.75,
                    1.0,
                ],
            )
        )

        for i in range(
            len(edges) - 1
        ):
            lo = float(
                edges[i]
            )

            hi = float(
                edges[i + 1]
            )

            if (
                i
                == len(edges) - 2
            ):
                rows = [
                    r
                    for r
                    in usable_candidates
                    if lo
                    <= float(
                        r["access_time"]
                    )
                    <= hi
                ]
            else:
                rows = [
                    r
                    for r
                    in usable_candidates
                    if lo
                    <= float(
                        r["access_time"]
                    )
                    < hi
                ]

            if not rows:
                continue

            horizon_bin_rows.append(
                {
                    "horizon_lo": lo,
                    "horizon_hi": hi,
                    "n": len(rows),
                    "mean_p0_to_oracle": (
                        mean_or_nan(
                            [
                                r[
                                    "distance_p0_snapshot_to_oracle"
                                ]
                                for r in rows
                            ]
                        )
                    ),
                    "mean_p1_to_oracle": (
                        mean_or_nan(
                            [
                                r[
                                    "distance_p1_arrival_only_to_oracle"
                                ]
                                for r in rows
                            ]
                        )
                    ),
                    "mean_p2_to_oracle": (
                        mean_or_nan(
                            [
                                r[
                                    "distance_p2_arrival_service_to_oracle"
                                ]
                                for r in rows
                            ]
                        )
                    ),
                    "p1_improve_rate": (
                        mean_or_nan(
                            [
                                r[
                                    "p1_improves_over_snapshot"
                                ]
                                for r in rows
                            ]
                        )
                    ),
                    "p2_improve_rate": (
                        mean_or_nan(
                            [
                                r[
                                    "p2_improves_over_snapshot"
                                ]
                                for r in rows
                            ]
                        )
                    ),
                    "p2_beats_p1_rate": (
                        mean_or_nan(
                            [
                                r[
                                    "p2_improves_over_p1"
                                ]
                                for r in rows
                            ]
                        )
                    ),
                    "mean_service_rate_estimate": (
                        mean_or_nan(
                            [
                                r[
                                    "service_rate_estimate"
                                ]
                                for r in rows
                            ]
                        )
                    ),
                    "mean_estimated_service_used": (
                        mean_or_nan(
                            [
                                r[
                                    "estimated_service_used"
                                ]
                                for r in rows
                            ]
                        )
                    ),
                }
            )

    # -------------------------------------------------------------------------
    # 输出 summary。
    # -------------------------------------------------------------------------

    summary = {
        "stage": (
            "Stage-0.6 arrival-only vs "
            "arrival+service no-training diagnostic"
        ),
        "project_root": str(
            project_root
        ),
        "uagmc_root": str(
            uagmc_root
        ),
        "model": str(
            model_path
        ),
        "vecnormalize": (
            str(
                vecnorm_path
            )
            if vecnorm_path
            else None
        ),
        "passenger_trace": str(
            passenger_path
        ),
        "candidate_vertiports": (
            candidate_ids
        ),
        "service_estimator": (
            args.service_estimator
        ),
        "service_window": (
            args.service_window
        ),
        "recorded_decisions": (
            len(decisions)
        ),
        "analyzable_decisions": (
            len(decision_rows)
        ),
        "candidate_rows": (
            len(candidate_rows)
        ),
        "demand_keys": list(
            DEMAND_KEYS
        ),
        "distance_to_oracle": {
            "mean_p0_snapshot": (
                mean_or_nan(
                    d_p0_all
                )
            ),
            "mean_p1_arrival_only": (
                mean_or_nan(
                    d_p1_all
                )
            ),
            "mean_p2_arrival_service": (
                mean_or_nan(
                    d_p2_all
                )
            ),
            "median_p0_snapshot": (
                percentile_or_nan(
                    d_p0_all,
                    50,
                )
            ),
            "median_p1_arrival_only": (
                percentile_or_nan(
                    d_p1_all,
                    50,
                )
            ),
            "median_p2_arrival_service": (
                percentile_or_nan(
                    d_p2_all,
                    50,
                )
            ),
        },
        "improvement": {
            "mean_gain_p1_vs_p0": (
                mean_or_nan(
                    p1_gain_all
                )
            ),
            "mean_gain_p2_vs_p0": (
                mean_or_nan(
                    p2_gain_all
                )
            ),
            "mean_gain_p2_vs_p1": (
                mean_or_nan(
                    p2_vs_p1_gain_all
                )
            ),
            "p1_improves_over_p0_rate": (
                mean_or_nan(
                    [
                        int(
                            x > 0.0
                        )
                        for x
                        in p1_gain_all
                    ]
                )
            ),
            "p2_improves_over_p0_rate": (
                mean_or_nan(
                    [
                        int(
                            x > 0.0
                        )
                        for x
                        in p2_gain_all
                    ]
                )
            ),
            "p2_improves_over_p1_rate": (
                mean_or_nan(
                    [
                        int(
                            x > 0.0
                        )
                        for x
                        in p2_vs_p1_gain_all
                    ]
                )
            ),
        },
        "ranking_match_to_oracle": {
            "waiting_p0": mean_or_nan(
                [
                    r[
                        "p0_matches_oracle_waiting"
                    ]
                    for r in decision_rows
                ]
            ),
            "waiting_p1": mean_or_nan(
                [
                    r[
                        "p1_matches_oracle_waiting"
                    ]
                    for r in decision_rows
                ]
            ),
            "waiting_p2": mean_or_nan(
                [
                    r[
                        "p2_matches_oracle_waiting"
                    ]
                    for r in decision_rows
                ]
            ),
            "burden_p0": mean_or_nan(
                [
                    r[
                        "p0_matches_oracle_burden"
                    ]
                    for r in decision_rows
                ]
            ),
            "burden_p1": mean_or_nan(
                [
                    r[
                        "p1_matches_oracle_burden"
                    ]
                    for r in decision_rows
                ]
            ),
            "burden_p2": mean_or_nan(
                [
                    r[
                        "p2_matches_oracle_burden"
                    ]
                    for r in decision_rows
                ]
            ),
        },
        "service_proxy": {
            "mean_rate": mean_or_nan(
                mu_all
            ),
            "median_rate": (
                percentile_or_nan(
                    mu_all,
                    50,
                )
            ),
            "p90_rate": (
                percentile_or_nan(
                    mu_all,
                    90,
                )
            ),
            "mean_estimated_service_used": (
                mean_or_nan(
                    service_used_all
                )
            ),
        },
        "horizon_relationship": {
            "rho_access_vs_p0_error": (
                spearman_rho(
                    access_all,
                    d_p0_all,
                )
            ),
            "rho_access_vs_p1_error": (
                spearman_rho(
                    access_all,
                    d_p1_all,
                )
            ),
            "rho_access_vs_p2_error": (
                spearman_rho(
                    access_all,
                    d_p2_all,
                )
            ),
            "rho_access_vs_service_used": (
                spearman_rho(
                    access_all,
                    service_used_all,
                )
            ),
        },
        "horizon_bins": (
            horizon_bin_rows
        ),
        "interpretation_guardrails": [
            (
                "P1/P2 都只使用当前已经 "
                "enroute 的 committed passenger；"
                "未来 unrevealed passenger 不进入 projection。"
            ),
            (
                "P2 service rate 只由当前决策之前"
                "已经发生的 queue-departure history "
                "与当前 IDLE eVTOL capacity proxy 构造。"
            ),
            (
                "P2 是 causal service proxy，"
                "不是 UAGMC 源码的 exact future simulator。"
            ),
            (
                "Oracle 来自 checkpoint 真实后续 rollout，"
                "只用于离线评价 projection error。"
            ),
            (
                "如果 P2 明显优于 P1，只能证明"
                "service consumption 是 arrival-only peeling "
                "的重要缺项；仍不能直接证明最终 RL 方法提高 ATT。"
            ),
            (
                "如果 P2 仍不优于 Snapshot，下一步应检查"
                "service-rate proxy 是否与源码真实 service process 对齐，"
                "而不是立即增加更复杂网络。"
            ),
        ],
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_root
        / "candidate_detail.csv",
        candidate_rows,
    )

    write_csv(
        output_root
        / "decision_summary.csv",
        decision_rows,
    )

    write_csv(
        output_root
        / "access_horizon_bins.csv",
        horizon_bin_rows,
    )

    with (
        output_root
        / "summary.json"
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
        output_root
        / "resolved_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            jsonable(
                {
                    "project_root": (
                        project_root
                    ),
                    "uagmc_root": (
                        uagmc_root
                    ),
                    "model": model_path,
                    "vecnormalize": (
                        vecnorm_path
                    ),
                    "passenger_trace": (
                        passenger_path
                    ),
                    "candidate_vertiports": (
                        candidate_ids
                    ),
                    "to_vertiport": (
                        args.to_vertiport
                    ),
                    "device": device,
                    "service_estimator": (
                        args.service_estimator
                    ),
                    "service_window": (
                        args.service_window
                    ),
                }
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    env.close()

    print(
        "\n"
        + "=" * 116
    )
    print(
        "STAGE-0.6 SUMMARY"
    )
    print(
        "=" * 116
    )
    print(
        f"Recorded decisions           : "
        f"{len(decisions)}"
    )
    print(
        f"Analyzable decisions         : "
        f"{len(decision_rows)}"
    )
    print(
        f"Mean P0 Snapshot -> Oracle   : "
        f"{summary['distance_to_oracle']['mean_p0_snapshot']:.6f}"
    )
    print(
        f"Mean P1 Arrival -> Oracle    : "
        f"{summary['distance_to_oracle']['mean_p1_arrival_only']:.6f}"
    )
    print(
        f"Mean P2 Arr+Svc -> Oracle    : "
        f"{summary['distance_to_oracle']['mean_p2_arrival_service']:.6f}"
    )
    print(
        f"P1 improves over P0         : "
        f"{summary['improvement']['p1_improves_over_p0_rate']:.4f}"
    )
    print(
        f"P2 improves over P0         : "
        f"{summary['improvement']['p2_improves_over_p0_rate']:.4f}"
    )
    print(
        f"P2 improves over P1         : "
        f"{summary['improvement']['p2_improves_over_p1_rate']:.4f}"
    )
    print(
        f"Oracle waiting match P0/P1/P2: "
        f"{summary['ranking_match_to_oracle']['waiting_p0']:.4f} / "
        f"{summary['ranking_match_to_oracle']['waiting_p1']:.4f} / "
        f"{summary['ranking_match_to_oracle']['waiting_p2']:.4f}"
    )
    print(
        f"Mean service-rate estimate  : "
        f"{summary['service_proxy']['mean_rate']:.4f} pax/min"
    )
    print(
        f"rho(access, P0 error)       : "
        f"{summary['horizon_relationship']['rho_access_vs_p0_error']:.4f}"
    )
    print(
        f"rho(access, P2 error)       : "
        f"{summary['horizon_relationship']['rho_access_vs_p2_error']:.4f}"
    )
    print(
        "-" * 116
    )
    print(
        "注意：P2 是 causal service proxy，"
        "不是 exact future simulator；"
        "Oracle 只用于离线诊断。"
    )
    print(
        f"结果目录：{output_root}"
    )
    print(
        "=" * 116
    )


if __name__ == "__main__":
    main()
