# -*- coding: utf-8 -*-
"""
UAGMC Candidate-Specific Effect-Time Master Diagnostic
======================================================

目标
----
这是一份“正式训练之前的最终无训练诊断”，只验证本文最核心的问题：

    对同一个乘客 p，不同候选 departure vertiport k 有不同 access delay T_{p,k}。
    UAGMC 风格方法在 decision time t 使用同一时刻的环境快照 S(t)；
    本项目关注的是：候选 k 是否应该读取/处理到它自己的 effect time
        t + T_{p,k}
    对应的同一组环境变量，再统一比较候选得分。

本脚本不训练任何模型，不改 reward，不改 PPO。

它修复此前 Stage-0~0.8 的主要逻辑漏洞：
1) incoming passenger 不再使用粗粒度 state == "enroute"；
   只把
       state == "enroute" AND sub_state == "to_vertiport"
   视为“正在 ground access、未来会到达 departure vertiport 的 committed passenger”。

2) 对上述 access-stage passenger，剩余 access ETA 使用
       current_timer + 1
   （离散 simulation-step 边界）。
   同时本脚本会在线重新审计 timer countdown 一致性，不盲信旧结论。

3) “真实 future state”只用于 offline oracle diagnostic。
   它可能包含未来尚未 reveal 的 passenger / 后续 policy 行为，绝不当作在线可用信息。

4) 另外单独提供 legal committed-event evidence：
   只使用 decision time 已经存在、已被过去动作 committed 的 access passenger，
   检查这些已知 arrival 是否跨过不同 candidate 的 effect boundary。
   这一部分不读取未来 unrevealed passenger。

5) 不再做旧版 P1/P2 的 service-rate projection，也不再用错误 cohort 证明方法有效。
   本脚本只回答：
       A. 同一乘客的不同机场 effect time 是否显著不同？
       B. 同一组环境变量在各自 effect time 是否已经显著变化？
       C. delay 越长，decision-time snapshot 是否越过时？
       D. 一个 candidate-independent shared future horizon 是否仍与各自 effect time 有明显错位？
       E. 当前已知 committed events 是否真的会跨过 candidate-specific effect boundary？
   通过这些 gate 后，再进入正式训练。

依赖
----
请把本文件放在 UAGMC 根目录，并保留：
    diagnose_uagmc_candidate_temporal_mismatch.py

默认读取历史复现：
    models/final_rl_model.zip
    models/final_vec_normalize.pkl
    train_data/passengers_300.csv

输出
----
diagnostics/uagmc_effect_time_master/<timestamp>/
    summary.json
    candidate_detail.csv
    decision_summary.csv
    horizon_bins.csv
    variable_change_summary.csv
    committed_crossing_summary.csv
    timer_semantics_audit.csv
    resolved_config.json

重要解释
--------
- “oracle effect-time state”是离线事实诊断，不是 proposed online observation。
- “shared horizon”默认取同一乘客所有 candidate access time 的均值，
  只用于检验“统一向未来看”是否仍不能替代 candidate-specific temporal reference。
- 自动 gate 只是工程性的 go/no-go 检查，不是统计学定理；所有原始指标都会完整输出。
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import diagnose_uagmc_candidate_temporal_mismatch as stage0
except Exception as exc:
    raise RuntimeError(
        "无法导入 diagnose_uagmc_candidate_temporal_mismatch.py。\n"
        "请将本脚本与 Stage-0 脚本放在同一 UAGMC 根目录。"
    ) from exc


# =============================================================================
# 变量集合
# =============================================================================

# 不依赖 passenger incoming 语义的“安全变量”。
# 用于回答：即使完全不使用 incoming，effect-time drift 是否仍然存在。
SAFE_KEYS = [
    "waiting",
    "charging_evtols",
    "total_evtols",
    "total_capacity",
    "avg_charge_time",
    "min_charge_time",
    "avg_flight_time",
    "idle_evtols",
]

# 修复 incoming 语义后，用来贴近 UAGMC vertiport aggregate 的完整诊断组。
CORRECTED_KEYS = [
    "waiting",
    "access_incoming",
    "charging_evtols",
    "total_evtols",
    "total_capacity",
    "avg_charge_time",
    "min_charge_time",
    "avg_flight_time",
    "idle_evtols",
]

# 额外记录，但不纳入上面主 drift，避免 ETA=0 时归一化不稳定。
EXTRA_KEYS = [
    "min_access_eta",
    "avg_access_eta",
]


# =============================================================================
# 通用工具
# =============================================================================

def finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for x in values:
        try:
            v = float(x)
        except Exception:
            continue
        if np.isfinite(v):
            out.append(v)
    return out


def mean_or_nan(values: Iterable[Any]) -> float:
    vals = finite(values)
    return float(np.mean(vals)) if vals else float("nan")


def median_or_nan(values: Iterable[Any]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def percentile_or_nan(values: Iterable[Any], q: float) -> float:
    vals = finite(values)
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
    seen: Set[str] = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rankdata_average(values: Sequence[float]) -> np.ndarray:
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


def spearman_rho_p(
    xs: Sequence[float],
    ys: Sequence[float],
) -> Tuple[float, float]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if np.isfinite(float(x)) and np.isfinite(float(y))
    ]

    if len(pairs) < 3:
        return float("nan"), float("nan")

    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)

    # 优先 scipy，输出 p-value。
    try:
        from scipy.stats import spearmanr
        res = spearmanr(x, y)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        rx = _rankdata_average(x)
        ry = _rankdata_average(y)

        if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
            return float("nan"), float("nan")

        return float(np.corrcoef(rx, ry)[0, 1]), float("nan")


def wilson_interval(success: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")

    n = float(total)
    p = float(success) / n

    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / n
            + z * z / (4.0 * n * n)
        )
        / denom
    )

    return max(0.0, center - half), min(1.0, center + half)


def normalized_drift(
    a: Dict[str, float],
    b: Dict[str, float],
    keys: Sequence[str],
) -> float:
    vals: List[float] = []

    for key in keys:
        av = float(a.get(key, 0.0))
        bv = float(b.get(key, 0.0))
        vals.append(abs(bv - av) / (1.0 + abs(av)))

    return float(np.mean(vals)) if vals else 0.0


def any_changed(
    a: Dict[str, float],
    b: Dict[str, float],
    keys: Sequence[str],
    tol: float = 1e-12,
) -> bool:
    return any(
        abs(float(b.get(k, 0.0)) - float(a.get(k, 0.0))) > tol
        for k in keys
    )


def argmin_stable(values: Dict[int, float]) -> int:
    return min(values.items(), key=lambda item: (float(item[1]), int(item[0])))[0]


def safe_ratio(num: float, den: float) -> float:
    if abs(float(den)) <= 1e-12:
        return float("nan")
    return float(num) / float(den)


# =============================================================================
# 正确的 UAGMC passenger / aircraft 语义
# =============================================================================

def evtol_state_name(evtol: Any) -> str:
    try:
        return str(evtol.state.name)
    except Exception:
        return str(getattr(evtol, "state", "UNKNOWN"))


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


def get_person_dict(scenario: Any) -> Dict[str, Any]:
    persons_obj = getattr(scenario, "persons", None)
    raw = getattr(persons_obj, "persons", {}) if persons_obj is not None else {}

    return {
        str(pid): person
        for pid, person in raw.items()
    }


def is_access_committed(person: Any) -> bool:
    """
    核心修复：
    只有真正处于 ground access 阶段的 passenger 才属于 future access arrival。
    """
    return (
        str(getattr(person, "state", "")).lower() == "enroute"
        and str(getattr(person, "sub_state", "")).lower() == "to_vertiport"
    )


def access_remaining_eta(person: Any) -> Optional[float]:
    """
    Stage-0.8 原始数据重新审计后的离散时间语义：
        remaining ETA = current_timer + 1

    若字段异常则返回 None。
    """
    if not is_access_committed(person):
        return None

    try:
        timer = float(getattr(person, "current_timer"))
    except Exception:
        return None

    if not np.isfinite(timer):
        return None

    return max(0.0, timer + 1.0)


def collect_access_events_by_station(
    scenario: Any,
    candidate_ids: Sequence[int],
    exclude_pid: Optional[str] = None,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    返回当前时刻“已经 committed 的 future access arrival”：
        station -> [(pid, remaining_eta), ...]

    不读取未来 passenger。
    """
    candidate_set = {int(v) for v in candidate_ids}
    out: Dict[int, List[Tuple[str, float]]] = {
        int(v): []
        for v in candidate_ids
    }

    excluded = str(exclude_pid) if exclude_pid is not None else None

    for pid, person in get_person_dict(scenario).items():
        if excluded is not None and str(pid) == excluded:
            continue

        if not is_access_committed(person):
            continue

        try:
            vid = int(getattr(person, "origin_vertiport_id"))
        except Exception:
            continue

        if vid not in candidate_set:
            continue

        eta = access_remaining_eta(person)

        if eta is None:
            continue

        out[vid].append((str(pid), float(eta)))

    for vid in out:
        out[vid].sort(key=lambda x: (x[1], x[0]))

    return out


def extract_corrected_state(
    scenario: Any,
    candidate_ids: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    """
    提取同一组 vertiport 环境变量，但修复 passenger incoming 语义。

    waiting:
        真实 vertiport.person_list

    access_incoming:
        仅 state=enroute & sub_state=to_vertiport 且 origin_vertiport_id==vid

    min/avg_access_eta:
        current_timer + 1，只对上述 access-stage passenger

    aircraft/resource channels:
        延续 Stage-0 的直接状态读取。
    """
    persons = get_person_dict(scenario)
    result: Dict[int, Dict[str, float]] = {}

    for vid_int in candidate_ids:
        vid = str(int(vid_int))
        vertiport = scenario.vertiports.vertiport_list[vid]

        waiting = float(len(getattr(vertiport, "person_list", [])))

        access_etas: List[float] = []

        for person in persons.values():
            if not is_access_committed(person):
                continue

            if str(getattr(person, "origin_vertiport_id", "")) != vid:
                continue

            eta = access_remaining_eta(person)

            if eta is not None:
                access_etas.append(float(eta))

        evtols = scenario.vertiports.evtols_at_vertiport.get(vid, [])

        charging_evtols = 0.0
        idle_evtols = 0.0
        total_capacity = 0.0
        remaining_charge_time: List[float] = []
        remaining_flight_time: List[float] = []

        for ev in evtols:
            state_name = evtol_state_name(ev).upper()

            if state_name == "CHARGING":
                charging_evtols += 1.0

            if state_name == "IDLE":
                idle_evtols += 1.0

            try:
                total_capacity += float(ev.spec.capacity)
            except Exception:
                pass

            if hasattr(ev, "remaining_charge_time"):
                try:
                    x = float(ev.remaining_charge_time)
                    if np.isfinite(x):
                        remaining_charge_time.append(x)
                except Exception:
                    pass

            if hasattr(ev, "remaining_flight_time"):
                try:
                    x = float(ev.remaining_flight_time)
                    if np.isfinite(x):
                        remaining_flight_time.append(x)
                except Exception:
                    pass

        result[int(vid_int)] = {
            "waiting": waiting,
            "access_incoming": float(len(access_etas)),
            "charging_evtols": charging_evtols,
            "total_evtols": float(len(evtols)),
            "total_capacity": total_capacity,
            "avg_charge_time": (
                float(np.mean(remaining_charge_time))
                if remaining_charge_time
                else 0.0
            ),
            "min_charge_time": (
                float(np.min(remaining_charge_time))
                if remaining_charge_time
                else 0.0
            ),
            "avg_flight_time": (
                float(np.mean(remaining_flight_time))
                if remaining_flight_time
                else 0.0
            ),
            "idle_evtols": idle_evtols,
            "min_access_eta": (
                float(np.min(access_etas))
                if access_etas
                else 0.0
            ),
            "avg_access_eta": (
                float(np.mean(access_etas))
                if access_etas
                else 0.0
            ),
        }

    return result


# =============================================================================
# Timer semantics 在线审计
# =============================================================================

class AccessTimerAudit:
    """
    只审计 sub_state == to_vertiport。

    若同一 pid 连续两个 simulation snapshot 都仍处于 to_vertiport，
    理论上 timer 每 step 应下降 1。
    """

    def __init__(self):
        self.prev: Dict[str, Tuple[int, float, int]] = {}
        self.rows: List[Dict[str, Any]] = []

    def observe(self, scenario: Any, sim_time: int) -> None:
        current: Dict[str, Tuple[int, float, int]] = {}

        for pid, person in get_person_dict(scenario).items():
            if not is_access_committed(person):
                continue

            try:
                timer = float(getattr(person, "current_timer"))
                vid = int(getattr(person, "origin_vertiport_id"))
            except Exception:
                continue

            if not np.isfinite(timer):
                continue

            current[pid] = (sim_time, timer, vid)

            if pid in self.prev:
                prev_time, prev_timer, prev_vid = self.prev[pid]
                dt = int(sim_time - prev_time)

                if dt > 0 and prev_vid == vid:
                    slope = (timer - prev_timer) / float(dt)

                    self.rows.append(
                        {
                            "pid": pid,
                            "candidate_vertiport": vid,
                            "time0": prev_time,
                            "time1": sim_time,
                            "dt": dt,
                            "timer0": prev_timer,
                            "timer1": timer,
                            "timer_delta": timer - prev_timer,
                            "timer_slope_per_step": slope,
                            "is_near_minus_1": int(abs(slope + 1.0) <= 1e-9),
                        }
                    )

        self.prev = current

    def summary(self) -> Dict[str, Any]:
        slopes = finite(r["timer_slope_per_step"] for r in self.rows)
        good = sum(int(r["is_near_minus_1"]) for r in self.rows)

        return {
            "n_transitions": len(self.rows),
            "median_slope_per_step": median_or_nan(slopes),
            "mean_slope_per_step": mean_or_nan(slopes),
            "fraction_exact_minus_1": (
                float(good) / float(len(self.rows))
                if self.rows
                else float("nan")
            ),
        }


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "UAGMC candidate-specific effect-time master diagnostic "
            "(no training)"
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
        help="例如 0,1；不填则沿用 UAGMC wrapper / Stage-0 自动识别。",
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
        default="diagnostics/uagmc_effect_time_master",
    )

    # 工程 gate，不是统计学结论。
    parser.add_argument(
        "--gate-min-mean-delay-spread",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--gate-min-safe-change-rate",
        type=float,
        default=0.40,
    )

    parser.add_argument(
        "--gate-min-rho-access-safe-drift",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--gate-min-committed-crossing-rate",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--gate-min-timer-consistency",
        type=float,
        default=0.95,
    )

    return parser.parse_args()


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()

    if not project_root.exists():
        raise FileNotFoundError(
            f"项目根目录不存在：{project_root}"
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
    output_root = project_root / args.output_dir / run_stamp
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

    if len(candidate_ids) != int(wrapper.action_space.n):
        raise RuntimeError(
            f"candidate 数量 {len(candidate_ids)} 与 "
            f"action_space.n={wrapper.action_space.n} 不一致。"
        )

    print("=" * 124)
    print(
        "UAGMC CANDIDATE-SPECIFIC EFFECT-TIME MASTER DIAGNOSTIC | NO TRAINING"
    )
    print("=" * 124)
    print(f"Project root : {project_root}")
    print(f"UAGMC root   : {uagmc_root}")
    print(f"Model        : {model_path}")
    print(f"VecNormalize : {vecnorm_path}")
    print(f"Passengers   : {passenger_path}")
    print(f"Candidates   : {candidate_ids}")
    print(f"Device       : {device}")
    print(f"Output       : {output_root}")
    print("-" * 124)

    snapshots: Dict[int, Dict[int, Dict[str, float]]] = {}
    decisions: List[Dict[str, Any]] = []

    timer_audit = AccessTimerAudit()

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

        if sim_time not in snapshots:
            snapshots[sim_time] = extract_corrected_state(
                scenario,
                candidate_ids,
            )

        timer_audit.observe(
            scenario,
            sim_time,
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
            access_times = stage0.estimate_access_times(
                scenario,
                focal_pid,
                candidate_ids,
            )

            committed_events = collect_access_events_by_station(
                scenario,
                candidate_ids,
                exclude_pid=focal_pid,
            )

            if probs is None or len(probs) != len(candidate_ids):
                policy_probs = [float("nan")] * len(candidate_ids)
                policy_margin = float("nan")
            else:
                policy_probs = [float(x) for x in probs]
                ordered = sorted(policy_probs, reverse=True)
                policy_margin = (
                    float(ordered[0] - ordered[1])
                    if len(ordered) >= 2
                    else 1.0
                )

            person = scenario.persons.persons[focal_pid]

            decisions.append(
                {
                    "decision_id": len(decisions),
                    "sim_time": sim_time,
                    "pid": str(focal_pid),
                    "origin_x": float(person.origin_position[0]),
                    "origin_y": float(person.origin_position[1]),
                    "destination_x": float(person.destination_position[0]),
                    "destination_y": float(person.destination_position[1]),
                    "chosen_action_index": action_index,
                    "chosen_vertiport": int(candidate_ids[action_index]),
                    "access_times": {
                        int(k): float(v)
                        for k, v in access_times.items()
                    },
                    "policy_probs": policy_probs,
                    "policy_margin": policy_margin,
                    "now_state": extract_corrected_state(
                        scenario,
                        candidate_ids,
                    ),
                    "committed_events": committed_events,
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
                "[警告] 超过安全 step 上限，强制结束 rollout。"
            )
            break

    final_time = int(
        getattr(
            scenario,
            "time",
            step_count,
        )
    )

    if final_time not in snapshots:
        snapshots[final_time] = extract_corrected_state(
            scenario,
            candidate_ids,
        )

    available_times = sorted(snapshots.keys())

    def snapshot_time(target_time: float) -> Optional[int]:
        target = int(math.ceil(float(target_time)))

        for t in available_times:
            if t >= target:
                return t

        return None

    # =========================================================================
    # Candidate-level：当前 vs 各自 effect time vs shared mean horizon
    # =========================================================================

    candidate_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []

    for decision in decisions:
        access_values = [
            float(v)
            for v in decision["access_times"].values()
        ]

        h_min = min(access_values)
        h_max = max(access_values)
        h_mean = float(np.mean(access_values))
        h_spread = h_max - h_min

        shared_t = snapshot_time(
            float(decision["sim_time"]) + h_mean
        )

        now_waiting: Dict[int, float] = {}
        own_waiting: Dict[int, float] = {}
        shared_waiting: Dict[int, float] = {}

        now_burden: Dict[int, float] = {}
        own_burden: Dict[int, float] = {}
        shared_burden: Dict[int, float] = {}

        now_pressure: Dict[int, float] = {}
        own_pressure: Dict[int, float] = {}
        shared_pressure: Dict[int, float] = {}

        own_safe_drifts: List[float] = []
        own_corrected_drifts: List[float] = []
        own_shared_safe_drifts: List[float] = []

        cross_counts: List[int] = []
        own_minus_shared_cross: List[int] = []

        complete = True

        for action_index, vid in enumerate(candidate_ids):
            vid = int(vid)
            h = float(decision["access_times"][vid])

            own_t = snapshot_time(
                float(decision["sim_time"]) + h
            )

            now_state = decision["now_state"][vid]

            row: Dict[str, Any] = {
                "decision_id": int(decision["decision_id"]),
                "sim_time": int(decision["sim_time"]),
                "pid": str(decision["pid"]),
                "origin_x": float(decision["origin_x"]),
                "origin_y": float(decision["origin_y"]),
                "destination_x": float(decision["destination_x"]),
                "destination_y": float(decision["destination_y"]),
                "candidate_action_index": int(action_index),
                "candidate_vertiport": vid,
                "checkpoint_choice": int(
                    vid == int(decision["chosen_vertiport"])
                ),
                "policy_prob": float(
                    decision["policy_probs"][action_index]
                ),
                "policy_margin": float(
                    decision["policy_margin"]
                ),
                "access_time": h,
                "decision_access_min": h_min,
                "decision_access_max": h_max,
                "decision_access_mean": h_mean,
                "decision_access_spread": h_spread,
                "own_effect_time": (
                    float(decision["sim_time"]) + h
                ),
                "own_snapshot_time": (
                    own_t if own_t is not None else ""
                ),
                "shared_effect_time": (
                    float(decision["sim_time"]) + h_mean
                ),
                "shared_snapshot_time": (
                    shared_t if shared_t is not None else ""
                ),
            }

            for key in SAFE_KEYS + [
                "access_incoming",
                "min_access_eta",
                "avg_access_eta",
            ]:
                row[f"now_{key}"] = float(
                    now_state.get(key, 0.0)
                )

            # -----------------------------
            # Legal committed-event evidence
            # -----------------------------
            events = list(
                decision["committed_events"].get(
                    vid,
                    [],
                )
            )

            own_cross = sum(
                1
                for _, eta in events
                if float(eta) <= h + 1e-12
            )

            shared_cross = sum(
                1
                for _, eta in events
                if float(eta) <= h_mean + 1e-12
            )

            row["committed_access_events_now"] = len(events)
            row["committed_cross_before_own_effect"] = own_cross
            row["committed_cross_before_shared_effect"] = shared_cross
            row["committed_own_minus_shared_cross"] = (
                own_cross - shared_cross
            )
            row["committed_cross_fraction_own"] = (
                float(own_cross) / float(len(events))
                if events
                else 0.0
            )

            cross_counts.append(own_cross)
            own_minus_shared_cross.append(
                own_cross - shared_cross
            )

            # -----------------------------
            # Offline oracle evidence
            # -----------------------------
            if own_t is None or shared_t is None:
                row["future_available"] = 0
                complete = False
                candidate_rows.append(row)
                continue

            own_state = snapshots[own_t][vid]
            shared_state = snapshots[shared_t][vid]

            row["future_available"] = 1

            for key in SAFE_KEYS + [
                "access_incoming",
                "min_access_eta",
                "avg_access_eta",
            ]:
                row[f"own_{key}"] = float(
                    own_state.get(key, 0.0)
                )
                row[f"shared_{key}"] = float(
                    shared_state.get(key, 0.0)
                )
                row[f"delta_now_to_own_{key}"] = (
                    float(own_state.get(key, 0.0))
                    - float(now_state.get(key, 0.0))
                )
                row[f"delta_shared_to_own_{key}"] = (
                    float(own_state.get(key, 0.0))
                    - float(shared_state.get(key, 0.0))
                )

            safe_drift = normalized_drift(
                now_state,
                own_state,
                SAFE_KEYS,
            )

            corrected_drift = normalized_drift(
                now_state,
                own_state,
                CORRECTED_KEYS,
            )

            shared_safe_drift = normalized_drift(
                shared_state,
                own_state,
                SAFE_KEYS,
            )

            shared_corrected_drift = normalized_drift(
                shared_state,
                own_state,
                CORRECTED_KEYS,
            )

            row["safe_drift_now_to_own"] = safe_drift
            row["corrected_drift_now_to_own"] = corrected_drift
            row["safe_drift_shared_to_own"] = shared_safe_drift
            row["corrected_drift_shared_to_own"] = shared_corrected_drift

            row["safe_any_change_now_to_own"] = int(
                any_changed(
                    now_state,
                    own_state,
                    SAFE_KEYS,
                )
            )

            row["corrected_any_change_now_to_own"] = int(
                any_changed(
                    now_state,
                    own_state,
                    CORRECTED_KEYS,
                )
            )

            row["safe_any_change_shared_to_own"] = int(
                any_changed(
                    shared_state,
                    own_state,
                    SAFE_KEYS,
                )
            )

            own_safe_drifts.append(safe_drift)
            own_corrected_drifts.append(corrected_drift)
            own_shared_safe_drifts.append(shared_safe_drift)

            now_waiting[vid] = float(
                now_state["waiting"]
            )
            own_waiting[vid] = float(
                own_state["waiting"]
            )
            shared_waiting[vid] = float(
                shared_state["waiting"]
            )

            now_burden[vid] = (
                float(now_state["waiting"])
                + float(now_state["access_incoming"])
            )
            own_burden[vid] = (
                float(own_state["waiting"])
                + float(own_state["access_incoming"])
            )
            shared_burden[vid] = (
                float(shared_state["waiting"])
                + float(shared_state["access_incoming"])
            )

            now_pressure[vid] = (
                now_burden[vid]
                - float(now_state["idle_evtols"])
            )
            own_pressure[vid] = (
                own_burden[vid]
                - float(own_state["idle_evtols"])
            )
            shared_pressure[vid] = (
                shared_burden[vid]
                - float(shared_state["idle_evtols"])
            )

            candidate_rows.append(row)

        decision_row: Dict[str, Any] = {
            "decision_id": int(decision["decision_id"]),
            "sim_time": int(decision["sim_time"]),
            "pid": str(decision["pid"]),
            "chosen_vertiport": int(
                decision["chosen_vertiport"]
            ),
            "policy_margin": float(
                decision["policy_margin"]
            ),
            "access_delay_min": h_min,
            "access_delay_max": h_max,
            "access_delay_mean": h_mean,
            "access_delay_spread": h_spread,
            "candidate_count": len(candidate_ids),
            "mean_own_safe_drift": mean_or_nan(
                own_safe_drifts
            ),
            "mean_own_corrected_drift": mean_or_nan(
                own_corrected_drifts
            ),
            "mean_shared_to_own_safe_drift": mean_or_nan(
                own_shared_safe_drifts
            ),
            "committed_cross_count_mean": mean_or_nan(
                cross_counts
            ),
            "committed_cross_count_max": (
                max(cross_counts)
                if cross_counts
                else 0
            ),
            "committed_cross_count_range": (
                max(cross_counts) - min(cross_counts)
                if cross_counts
                else 0
            ),
            "candidate_specific_boundary_diff": int(
                any(x != 0 for x in own_minus_shared_cross)
            ),
            "complete_oracle": int(complete),
        }

        if complete and len(own_waiting) == len(candidate_ids):
            now_best_waiting = argmin_stable(
                now_waiting
            )
            own_best_waiting = argmin_stable(
                own_waiting
            )
            shared_best_waiting = argmin_stable(
                shared_waiting
            )

            now_best_burden = argmin_stable(
                now_burden
            )
            own_best_burden = argmin_stable(
                own_burden
            )
            shared_best_burden = argmin_stable(
                shared_burden
            )

            now_best_pressure = argmin_stable(
                now_pressure
            )
            own_best_pressure = argmin_stable(
                own_pressure
            )
            shared_best_pressure = argmin_stable(
                shared_pressure
            )

            decision_row.update(
                {
                    "now_best_waiting": now_best_waiting,
                    "own_effect_best_waiting": own_best_waiting,
                    "shared_horizon_best_waiting": shared_best_waiting,
                    "flip_now_to_own_waiting": int(
                        now_best_waiting
                        != own_best_waiting
                    ),
                    "flip_shared_to_own_waiting": int(
                        shared_best_waiting
                        != own_best_waiting
                    ),
                    "now_best_burden": now_best_burden,
                    "own_effect_best_burden": own_best_burden,
                    "shared_horizon_best_burden": shared_best_burden,
                    "flip_now_to_own_burden": int(
                        now_best_burden
                        != own_best_burden
                    ),
                    "flip_shared_to_own_burden": int(
                        shared_best_burden
                        != own_best_burden
                    ),
                    "now_best_pressure": now_best_pressure,
                    "own_effect_best_pressure": own_best_pressure,
                    "shared_horizon_best_pressure": shared_best_pressure,
                    "flip_now_to_own_pressure": int(
                        now_best_pressure
                        != own_best_pressure
                    ),
                    "flip_shared_to_own_pressure": int(
                        shared_best_pressure
                        != own_best_pressure
                    ),
                }
            )

        decision_rows.append(
            decision_row
        )

    complete_candidate_rows = [
        r
        for r in candidate_rows
        if int(r.get("future_available", 0)) == 1
    ]

    complete_decision_rows = [
        r
        for r in decision_rows
        if int(r.get("complete_oracle", 0)) == 1
    ]

    # =========================================================================
    # 多维度汇总
    # =========================================================================

    access_all = [
        float(r["access_time"])
        for r in complete_candidate_rows
    ]

    safe_drift_all = [
        float(r["safe_drift_now_to_own"])
        for r in complete_candidate_rows
    ]

    corrected_drift_all = [
        float(r["corrected_drift_now_to_own"])
        for r in complete_candidate_rows
    ]

    shared_safe_drift_all = [
        float(r["safe_drift_shared_to_own"])
        for r in complete_candidate_rows
    ]

    rho_access_safe, p_access_safe = spearman_rho_p(
        access_all,
        safe_drift_all,
    )

    rho_access_corrected, p_access_corrected = spearman_rho_p(
        access_all,
        corrected_drift_all,
    )

    # same passenger 内 access spread 与平均 effect-time drift
    spread_vals = [
        float(r["access_delay_spread"])
        for r in complete_decision_rows
    ]

    decision_mean_drift = [
        float(r["mean_own_safe_drift"])
        for r in complete_decision_rows
    ]

    rho_spread_drift, p_spread_drift = spearman_rho_p(
        spread_vals,
        decision_mean_drift,
    )

    # -------------------------------------------------------------------------
    # Horizon bins：用实际样本分位数，避免硬编码旧环境区间。
    # -------------------------------------------------------------------------

    horizon_bin_rows: List[Dict[str, Any]] = []

    if complete_candidate_rows:
        horizons = np.asarray(
            access_all,
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
                    for r in complete_candidate_rows
                    if lo <= float(r["access_time"]) <= hi
                ]
            else:
                rows = [
                    r
                    for r in complete_candidate_rows
                    if lo <= float(r["access_time"]) < hi
                ]

            if not rows:
                continue

            horizon_bin_rows.append(
                {
                    "horizon_lo": lo,
                    "horizon_hi": hi,
                    "n": len(rows),
                    "mean_access_time": mean_or_nan(
                        r["access_time"]
                        for r in rows
                    ),
                    "mean_safe_drift_now_to_own": mean_or_nan(
                        r["safe_drift_now_to_own"]
                        for r in rows
                    ),
                    "mean_corrected_drift_now_to_own": mean_or_nan(
                        r["corrected_drift_now_to_own"]
                        for r in rows
                    ),
                    "safe_change_rate_now_to_own": mean_or_nan(
                        r["safe_any_change_now_to_own"]
                        for r in rows
                    ),
                    "corrected_change_rate_now_to_own": mean_or_nan(
                        r["corrected_any_change_now_to_own"]
                        for r in rows
                    ),
                    "mean_safe_drift_shared_to_own": mean_or_nan(
                        r["safe_drift_shared_to_own"]
                        for r in rows
                    ),
                    "shared_horizon_mismatch_rate": mean_or_nan(
                        r["safe_any_change_shared_to_own"]
                        for r in rows
                    ),
                    "committed_cross_before_own_mean": mean_or_nan(
                        r["committed_cross_before_own_effect"]
                        for r in rows
                    ),
                    "committed_cross_positive_rate": mean_or_nan(
                        int(r["committed_cross_before_own_effect"] > 0)
                        for r in rows
                    ),
                }
            )

    # -------------------------------------------------------------------------
    # Per-variable change rates
    # -------------------------------------------------------------------------

    variable_change_rows: List[Dict[str, Any]] = []

    for key in SAFE_KEYS + [
        "access_incoming",
        "min_access_eta",
        "avg_access_eta",
    ]:
        rows = complete_candidate_rows
        deltas = [
            float(r[f"delta_now_to_own_{key}"])
            for r in rows
        ]

        changed = [
            int(abs(x) > 1e-12)
            for x in deltas
        ]

        lo, hi = wilson_interval(
            sum(changed),
            len(changed),
        )

        variable_change_rows.append(
            {
                "variable": key,
                "n": len(rows),
                "change_rate": mean_or_nan(changed),
                "change_rate_ci95_lo": lo,
                "change_rate_ci95_hi": hi,
                "mean_abs_change": mean_or_nan(
                    abs(x)
                    for x in deltas
                ),
                "median_abs_change": median_or_nan(
                    abs(x)
                    for x in deltas
                ),
                "p90_abs_change": percentile_or_nan(
                    [abs(x) for x in deltas],
                    90,
                ),
                "mean_signed_change": mean_or_nan(
                    deltas
                ),
            }
        )

    # -------------------------------------------------------------------------
    # Committed crossing summary
    # -------------------------------------------------------------------------

    crossing_rows: List[Dict[str, Any]] = []

    for vid in candidate_ids:
        rows = [
            r
            for r in complete_candidate_rows
            if int(r["candidate_vertiport"]) == int(vid)
        ]

        positive = [
            int(
                int(r["committed_cross_before_own_effect"]) > 0
            )
            for r in rows
        ]

        boundary_diff = [
            int(
                int(r["committed_own_minus_shared_cross"]) != 0
            )
            for r in rows
        ]

        crossing_rows.append(
            {
                "candidate_vertiport": int(vid),
                "n": len(rows),
                "mean_current_committed_access_events": mean_or_nan(
                    r["committed_access_events_now"]
                    for r in rows
                ),
                "mean_cross_before_own_effect": mean_or_nan(
                    r["committed_cross_before_own_effect"]
                    for r in rows
                ),
                "positive_crossing_rate": mean_or_nan(
                    positive
                ),
                "own_vs_shared_boundary_diff_rate": mean_or_nan(
                    boundary_diff
                ),
                "mean_own_minus_shared_cross": mean_or_nan(
                    r["committed_own_minus_shared_cross"]
                    for r in rows
                ),
            }
        )

    # =========================================================================
    # Gate
    # =========================================================================

    timer_summary = timer_audit.summary()

    mean_delay_spread = mean_or_nan(
        r["access_delay_spread"]
        for r in decision_rows
    )

    safe_change_rate = mean_or_nan(
        r["safe_any_change_now_to_own"]
        for r in complete_candidate_rows
    )

    corrected_change_rate = mean_or_nan(
        r["corrected_any_change_now_to_own"]
        for r in complete_candidate_rows
    )

    committed_crossing_rate = mean_or_nan(
        int(r["committed_cross_before_own_effect"] > 0)
        for r in complete_candidate_rows
    )

    shared_horizon_mismatch_rate = mean_or_nan(
        r["safe_any_change_shared_to_own"]
        for r in complete_candidate_rows
    )

    timer_consistency = float(
        timer_summary["fraction_exact_minus_1"]
    )

    gate_items = {
        "G1_same_passenger_candidate_delay_heterogeneity": {
            "metric": mean_delay_spread,
            "threshold": float(
                args.gate_min_mean_delay_spread
            ),
            "pass": bool(
                np.isfinite(mean_delay_spread)
                and mean_delay_spread
                >= args.gate_min_mean_delay_spread
            ),
        },
        "G2_effect_time_safe_state_changes_often": {
            "metric": safe_change_rate,
            "threshold": float(
                args.gate_min_safe_change_rate
            ),
            "pass": bool(
                np.isfinite(safe_change_rate)
                and safe_change_rate
                >= args.gate_min_safe_change_rate
            ),
        },
        "G3_longer_delay_more_snapshot_staleness": {
            "metric": rho_access_safe,
            "threshold": float(
                args.gate_min_rho_access_safe_drift
            ),
            "p_value": p_access_safe,
            "pass": bool(
                np.isfinite(rho_access_safe)
                and rho_access_safe
                >= args.gate_min_rho_access_safe_drift
            ),
        },
        "G4_known_committed_events_cross_effect_boundary": {
            "metric": committed_crossing_rate,
            "threshold": float(
                args.gate_min_committed_crossing_rate
            ),
            "pass": bool(
                np.isfinite(committed_crossing_rate)
                and committed_crossing_rate
                >= args.gate_min_committed_crossing_rate
            ),
        },
        "G5_access_timer_semantics_consistent": {
            "metric": timer_consistency,
            "threshold": float(
                args.gate_min_timer_consistency
            ),
            "pass": bool(
                np.isfinite(timer_consistency)
                and timer_consistency
                >= args.gate_min_timer_consistency
            ),
        },
    }

    overall_pass = all(
        bool(item["pass"])
        for item in gate_items.values()
    )

    # =========================================================================
    # Summary
    # =========================================================================

    delay_spreads = [
        float(r["access_delay_spread"])
        for r in decision_rows
    ]

    safe_change_count = sum(
        int(r["safe_any_change_now_to_own"])
        for r in complete_candidate_rows
    )

    safe_ci_lo, safe_ci_hi = wilson_interval(
        safe_change_count,
        len(complete_candidate_rows),
    )

    waiting_flip_rate = mean_or_nan(
        r.get("flip_now_to_own_waiting", float("nan"))
        for r in complete_decision_rows
    )

    burden_flip_rate = mean_or_nan(
        r.get("flip_now_to_own_burden", float("nan"))
        for r in complete_decision_rows
    )

    pressure_flip_rate = mean_or_nan(
        r.get("flip_now_to_own_pressure", float("nan"))
        for r in complete_decision_rows
    )

    shared_waiting_flip_rate = mean_or_nan(
        r.get("flip_shared_to_own_waiting", float("nan"))
        for r in complete_decision_rows
    )

    shared_burden_flip_rate = mean_or_nan(
        r.get("flip_shared_to_own_burden", float("nan"))
        for r in complete_decision_rows
    )

    shared_pressure_flip_rate = mean_or_nan(
        r.get("flip_shared_to_own_pressure", float("nan"))
        for r in complete_decision_rows
    )

    # 最短 / 最长 horizon bin effect size
    shortest_bin_drift = (
        float(horizon_bin_rows[0]["mean_safe_drift_now_to_own"])
        if horizon_bin_rows
        else float("nan")
    )

    longest_bin_drift = (
        float(horizon_bin_rows[-1]["mean_safe_drift_now_to_own"])
        if horizon_bin_rows
        else float("nan")
    )

    summary = {
        "stage": (
            "Candidate-specific effect-time master diagnostic "
            "(corrected, no training)"
        ),
        "core_question": (
            "For the SAME passenger, do different candidate vertiports have "
            "different access/effect times such that the SAME environment "
            "variables should be temporally aligned to t + T_access(p,k) "
            "before candidate comparison?"
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
        "candidate_vertiports": [
            int(v)
            for v in candidate_ids
        ],
        "simulation_steps": step_count,
        "recorded_decisions": len(decisions),
        "candidate_rows": len(candidate_rows),
        "complete_candidate_rows": len(
            complete_candidate_rows
        ),
        "complete_decisions": len(
            complete_decision_rows
        ),
        "semantic_fixes": {
            "incoming_definition": (
                "state=enroute AND sub_state=to_vertiport only"
            ),
            "remaining_access_eta": (
                "current_timer + 1 simulation step"
            ),
            "oracle_future_use": (
                "offline diagnostic only; never an online input"
            ),
            "legal_committed_evidence": (
                "only passengers already in to_vertiport at decision time"
            ),
        },
        "same_passenger_candidate_delay": {
            "mean_min_access": mean_or_nan(
                r["access_delay_min"]
                for r in decision_rows
            ),
            "mean_max_access": mean_or_nan(
                r["access_delay_max"]
                for r in decision_rows
            ),
            "mean_spread": mean_delay_spread,
            "median_spread": median_or_nan(
                delay_spreads
            ),
            "p90_spread": percentile_or_nan(
                delay_spreads,
                90,
            ),
            "max_spread": (
                max(delay_spreads)
                if delay_spreads
                else float("nan")
            ),
            "fraction_spread_ge_2": mean_or_nan(
                int(x >= 2.0)
                for x in delay_spreads
            ),
            "fraction_spread_ge_5": mean_or_nan(
                int(x >= 5.0)
                for x in delay_spreads
            ),
        },
        "oracle_effect_time_staleness": {
            "mean_safe_drift_now_to_own": mean_or_nan(
                safe_drift_all
            ),
            "median_safe_drift_now_to_own": median_or_nan(
                safe_drift_all
            ),
            "p90_safe_drift_now_to_own": percentile_or_nan(
                safe_drift_all,
                90,
            ),
            "safe_any_change_rate": safe_change_rate,
            "safe_any_change_rate_ci95": [
                safe_ci_lo,
                safe_ci_hi,
            ],
            "mean_corrected_drift_now_to_own": mean_or_nan(
                corrected_drift_all
            ),
            "corrected_any_change_rate": corrected_change_rate,
            "rho_access_vs_safe_drift": rho_access_safe,
            "p_access_vs_safe_drift": p_access_safe,
            "rho_access_vs_corrected_drift": rho_access_corrected,
            "p_access_vs_corrected_drift": p_access_corrected,
            "shortest_horizon_bin_mean_safe_drift": shortest_bin_drift,
            "longest_horizon_bin_mean_safe_drift": longest_bin_drift,
            "long_vs_short_drift_ratio": safe_ratio(
                longest_bin_drift,
                shortest_bin_drift,
            ),
            "rho_same_passenger_delay_spread_vs_mean_safe_drift": rho_spread_drift,
            "p_same_passenger_delay_spread_vs_mean_safe_drift": p_spread_drift,
        },
        "candidate_specific_vs_shared_future_reference": {
            "shared_reference": (
                "same passenger's mean candidate access horizon"
            ),
            "mean_safe_drift_shared_to_own": mean_or_nan(
                shared_safe_drift_all
            ),
            "shared_horizon_safe_mismatch_rate": shared_horizon_mismatch_rate,
            "decision_boundary_diff_rate_known_committed": mean_or_nan(
                r["candidate_specific_boundary_diff"]
                for r in decision_rows
            ),
        },
        "decision_relevance_offline_oracle": {
            "now_to_own_waiting_rank_flip_rate": waiting_flip_rate,
            "now_to_own_corrected_burden_rank_flip_rate": burden_flip_rate,
            "now_to_own_corrected_pressure_rank_flip_rate": pressure_flip_rate,
            "shared_to_own_waiting_rank_flip_rate": shared_waiting_flip_rate,
            "shared_to_own_corrected_burden_rank_flip_rate": shared_burden_flip_rate,
            "shared_to_own_corrected_pressure_rank_flip_rate": shared_pressure_flip_rate,
            "guardrail": (
                "Rank flips are descriptive oracle diagnostics, not proof of "
                "counterfactual ATT-optimal action."
            ),
        },
        "legal_committed_event_evidence": {
            "mean_committed_access_events_now": mean_or_nan(
                r["committed_access_events_now"]
                for r in complete_candidate_rows
            ),
            "mean_events_crossing_before_own_effect": mean_or_nan(
                r["committed_cross_before_own_effect"]
                for r in complete_candidate_rows
            ),
            "positive_crossing_rate": committed_crossing_rate,
            "own_vs_shared_cross_count_diff_rate": mean_or_nan(
                int(r["committed_own_minus_shared_cross"] != 0)
                for r in complete_candidate_rows
            ),
            "interpretation": (
                "This section uses only decision-time known passenger access "
                "commitments; no unrevealed future passenger is counted."
            ),
        },
        "timer_semantics_audit": timer_summary,
        "horizon_bins": horizon_bin_rows,
        "gate": {
            "items": gate_items,
            "overall_pass": overall_pass,
            "status": (
                "READY_FOR_CONTROLLED_TRAINING"
                if overall_pass
                else "HOLD_AND_REVIEW"
            ),
            "note": (
                "These thresholds are engineering go/no-go criteria, not "
                "statistical theorem thresholds. Inspect raw metrics before "
                "making a research claim."
            ),
        },
        "interpretation_guardrails": [
            (
                "Oracle effect-time state comes from the realized checkpoint "
                "trajectory and may contain unrevealed future demand or later "
                "policy actions. It is ONLY evidence that temporal mismatch "
                "exists; it is never legal online information."
            ),
            (
                "Legal committed-event crossing uses only passengers already "
                "in sub_state=to_vertiport at decision time."
            ),
            (
                "The diagnostic does not claim PPO cannot learn delayed return. "
                "It tests whether candidate comparison is performed under a "
                "candidate-correct temporal reference."
            ),
            (
                "A PASS justifies controlled training; it does not prove ATT "
                "improvement. ATT improvement must be established by equal-"
                "information controlled training/evaluation."
            ),
        ],
    }

    # =========================================================================
    # 输出
    # =========================================================================

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_root / "candidate_detail.csv",
        candidate_rows,
    )

    write_csv(
        output_root / "decision_summary.csv",
        decision_rows,
    )

    write_csv(
        output_root / "horizon_bins.csv",
        horizon_bin_rows,
    )

    write_csv(
        output_root / "variable_change_summary.csv",
        variable_change_rows,
    )

    write_csv(
        output_root / "committed_crossing_summary.csv",
        crossing_rows,
    )

    write_csv(
        output_root / "timer_semantics_audit.csv",
        timer_audit.rows,
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
                    "safe_keys": SAFE_KEYS,
                    "corrected_keys": CORRECTED_KEYS,
                    "gate_thresholds": {
                        "mean_delay_spread": args.gate_min_mean_delay_spread,
                        "safe_change_rate": args.gate_min_safe_change_rate,
                        "rho_access_safe_drift": args.gate_min_rho_access_safe_drift,
                        "committed_crossing_rate": args.gate_min_committed_crossing_rate,
                        "timer_consistency": args.gate_min_timer_consistency,
                    },
                }
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    env.close()

    # =========================================================================
    # 终端摘要
    # =========================================================================

    print("\n" + "=" * 124)
    print("MASTER DIAGNOSTIC SUMMARY")
    print("=" * 124)
    print(f"Recorded decisions                    : {len(decisions)}")
    print(f"Complete candidate rows               : {len(complete_candidate_rows)}")
    print(f"Complete decisions                    : {len(complete_decision_rows)}")
    print("-" * 124)
    print("A. SAME PASSENGER / DIFFERENT CANDIDATE EFFECT TIMES")
    print(
        f"Mean access-delay spread               : "
        f"{mean_delay_spread:.6f}"
    )
    print(
        f"P90 access-delay spread                : "
        f"{summary['same_passenger_candidate_delay']['p90_spread']:.6f}"
    )
    print("-" * 124)
    print("B. DECISION-TIME SNAPSHOT -> OWN EFFECT-TIME ORACLE")
    print(
        f"Mean SAFE drift                        : "
        f"{summary['oracle_effect_time_staleness']['mean_safe_drift_now_to_own']:.6f}"
    )
    print(
        f"SAFE any-change rate                   : "
        f"{safe_change_rate:.4f}"
    )
    print(
        f"rho(access, SAFE drift)                : "
        f"{rho_access_safe:.6f}"
    )
    print(
        f"p-value                                : "
        f"{p_access_safe:.6g}"
    )
    print(
        f"Long/short horizon drift ratio         : "
        f"{summary['oracle_effect_time_staleness']['long_vs_short_drift_ratio']:.4f}"
    )
    print("-" * 124)
    print("C. SHARED FUTURE HORIZON -> OWN CANDIDATE EFFECT TIME")
    print(
        f"Mean SAFE drift shared->own            : "
        f"{summary['candidate_specific_vs_shared_future_reference']['mean_safe_drift_shared_to_own']:.6f}"
    )
    print(
        f"Shared-horizon mismatch rate           : "
        f"{shared_horizon_mismatch_rate:.4f}"
    )
    print("-" * 124)
    print("D. LEGAL COMMITTED-EVENT EVIDENCE")
    print(
        f"Committed crossing positive rate       : "
        f"{committed_crossing_rate:.4f}"
    )
    print(
        f"Own-vs-shared crossing diff rate       : "
        f"{summary['legal_committed_event_evidence']['own_vs_shared_cross_count_diff_rate']:.4f}"
    )
    print("-" * 124)
    print("E. TIMER SEMANTICS AUDIT")
    print(
        f"Access timer transitions               : "
        f"{timer_summary['n_transitions']}"
    )
    print(
        f"Fraction slope == -1                   : "
        f"{timer_consistency:.4f}"
    )
    print("-" * 124)
    print("F. OFFLINE ORACLE RANKING RELEVANCE")
    print(
        f"Waiting rank flip now->own             : "
        f"{waiting_flip_rate:.4f}"
    )
    print(
        f"Corrected burden flip now->own         : "
        f"{burden_flip_rate:.4f}"
    )
    print(
        f"Corrected pressure flip now->own       : "
        f"{pressure_flip_rate:.4f}"
    )
    print("-" * 124)

    for name, item in gate_items.items():
        label = "PASS" if item["pass"] else "FAIL"
        print(
            f"{name:<52} : {label} "
            f"(metric={item['metric']:.6f}, "
            f"threshold={item['threshold']:.6f})"
        )

    print("-" * 124)
    print(
        "OVERALL GATE                           : "
        + (
            "READY_FOR_CONTROLLED_TRAINING"
            if overall_pass
            else "HOLD_AND_REVIEW"
        )
    )
    print(
        "注意：future oracle 只用于证明时间参考错位存在；"
        "不能作为在线策略输入。"
    )
    print(f"结果目录：{output_root}")
    print("=" * 124)


if __name__ == "__main__":
    main()
