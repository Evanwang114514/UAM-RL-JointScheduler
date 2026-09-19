# -*- coding: utf-8 -*-
"""
UAGMC 候选动作时间错位诊断（Stage-0，无训练）

用途
----
1. 加载已经复现好的 UAGMC PPO checkpoint 与 VecNormalize；
2. 不训练，只让原策略在固定 passenger trace 上运行；
3. 对每个真实 passenger 决策点计算不同候选 vertiport 的 ground-access time；
4. 保存 decision-time 状态，并读取同一条真实 rollout 上各 candidate 自己 realization horizon 的状态；
5. 输出 access-delay heterogeneity、temporal state drift、ranking flip、regret proxy、checkpoint policy margin；
6. 为后续 U0/U1/U2/U3 正式训练实验提供“问题是否真实存在”的证据。

重要边界
--------
- realization-time future state 是离线诊断 oracle：它来自 checkpoint 之后真实发生的轨迹。
- 这些 future state 可能包含当时尚未 reveal 的 passenger / 后续 policy 行为。
- 因此它绝不能作为在线策略输入，只用于判断 temporal mismatch 是否存在、规模多大。
- 本脚本不修改 UAGMC 环境、不修改 reward、不训练任何模型。

当前脚本按 Traffic-Alpha/UAGMC 官方源码接口编写：
- utilss.make_env.make_env
- utilss.uam_rl_wrapper.UAMRLWrapper
- stable_baselines3 PPO / VecNormalize
- Scenario.vehicles.estimate_travel_time(...)

代码注释按项目要求全部使用中文。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# 通用工具
# ============================================================


def mean_or_nan(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def percentile_or_nan(values: Sequence[float], q: float) -> float:
    vals = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.percentile(vals, q)) if vals else float("nan")


def jsonable(x: Any):
    """把 numpy / Path 等对象转换为 JSON 可序列化格式。"""
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
    """写 CSV；如果没有数据也创建空文件。"""
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


# ============================================================
# UAGMC 根目录 / checkpoint / VecNormalize 自动发现
# ============================================================


def looks_like_uagmc_root(path: Path) -> bool:
    """判断目录是否像 Traffic-Alpha/UAGMC 根目录。"""
    return (
        (path / "utilss" / "make_env.py").exists()
        and (path / "utilss" / "uam_rl_wrapper.py").exists()
        and (path / "at_obj" / "scenario.py").exists()
        and (path / "rl_env" / "observation_encoder.py").exists()
    )


def discover_uagmc_root(project_root: Path, explicit: Optional[str]) -> Path:
    """从项目中自动寻找 UAGMC 源码根目录。"""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.resolve()
        if not looks_like_uagmc_root(p):
            raise FileNotFoundError(
                f"--uagmc-root 不是可识别的 UAGMC 根目录：{p}\n"
                "应至少包含 utilss/make_env.py、utilss/uam_rl_wrapper.py、at_obj/scenario.py"
            )
        return p

    if looks_like_uagmc_root(project_root):
        return project_root

    candidates: List[Path] = []
    for file in project_root.rglob("utilss/uam_rl_wrapper.py"):
        root = file.parent.parent
        if looks_like_uagmc_root(root):
            candidates.append(root.resolve())

    candidates = sorted(
        list({p for p in candidates}),
        key=lambda p: (len(p.parts), str(p).lower()),
    )

    if not candidates:
        raise FileNotFoundError(
            "没有自动发现 UAGMC 源码目录。\n"
            "请重新运行并显式传入：--uagmc-root \"你的UAGMC目录\""
        )

    print("[自动发现] UAGMC 候选目录：")
    for i, p in enumerate(candidates):
        print(f"  [{i}] {p}")
    print(f"[自动选择] {candidates[0]}")
    return candidates[0]


def model_score(path: Path) -> Tuple[int, int, float]:
    """checkpoint 自动选择优先级。"""
    name = path.name.lower()
    score = 0
    if name == "final_rl_model.zip":
        score += 20000
    if "best_model" in name:
        score += 15000
    if "best" in name:
        score += 10000
    if "final" in name:
        score += 8000

    numbers = [int(x) for x in re.findall(r"(\d+)", name)]
    step_hint = max(numbers) if numbers else 0
    return score, step_hint, path.stat().st_mtime


def discover_model(
    uagmc_root: Path,
    project_root: Path,
    explicit: Optional[str],
) -> Path:
    """自动寻找已经复现完成的 PPO checkpoint。"""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"checkpoint 不存在：{p}")
        return p

    # 官方测试脚本默认路径优先。
    standard = uagmc_root / "models" / "final_rl_model.zip"
    if standard.exists():
        print(f"[自动发现] checkpoint：{standard}")
        return standard.resolve()

    candidates = list(uagmc_root.rglob("*.zip"))

    # 如果你的完全复现实验 checkpoint 放在项目其他目录，也一起寻找。
    if uagmc_root != project_root:
        candidates.extend(
            p
            for p in project_root.rglob("*.zip")
            if "uagmc" in str(p).lower()
        )

    candidates = list({p.resolve() for p in candidates if p.is_file()})
    if not candidates:
        raise FileNotFoundError(
            "未发现 PPO .zip checkpoint。请使用 --model 显式指定。"
        )

    candidates.sort(key=model_score, reverse=True)
    print("[自动发现] checkpoint 候选（前 10 个）：")
    for i, p in enumerate(candidates[:10]):
        print(f"  [{i}] {p}")
    print(f"[自动选择] {candidates[0]}")
    return candidates[0]


def discover_vecnorm(
    uagmc_root: Path,
    project_root: Path,
    model_path: Path,
    explicit: Optional[str],
) -> Optional[Path]:
    """自动寻找与 checkpoint 对应的 VecNormalize。"""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"VecNormalize 不存在：{p}")
        return p

    standard = uagmc_root / "models" / "final_vec_normalize.pkl"
    if standard.exists():
        print(f"[自动发现] VecNormalize：{standard}")
        return standard.resolve()

    candidates = [
        p
        for p in uagmc_root.rglob("*.pkl")
        if "vec" in p.name.lower() or "normalize" in p.name.lower()
    ]

    if uagmc_root != project_root:
        candidates.extend(
            p
            for p in project_root.rglob("*.pkl")
            if "uagmc" in str(p).lower()
            and ("vec" in p.name.lower() or "normalize" in p.name.lower())
        )

    candidates = list({p.resolve() for p in candidates if p.is_file()})
    if not candidates:
        print("[警告] 未发现 VecNormalize，将尝试直接使用原始 observation。")
        return None

    model_numbers = set(re.findall(r"\d+", model_path.stem.lower()))

    def score(path: Path):
        s = 0
        if path.parent == model_path.parent:
            s += 20000
        path_numbers = set(re.findall(r"\d+", path.stem.lower()))
        s += 500 * len(model_numbers & path_numbers)
        if "final" in model_path.stem.lower() and "final" in path.stem.lower():
            s += 5000
        if "best" in model_path.stem.lower() and "best" in path.stem.lower():
            s += 5000
        return s, path.stat().st_mtime

    candidates.sort(key=score, reverse=True)
    print("[自动发现] VecNormalize 候选（前 10 个）：")
    for i, p in enumerate(candidates[:10]):
        print(f"  [{i}] {p}")
    print(f"[自动选择] {candidates[0]}")
    return candidates[0]


def discover_passenger_file(
    uagmc_root: Path,
    project_root: Path,
    explicit: Optional[str],
) -> Path:
    """寻找固定 passenger trace。"""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p1 = (uagmc_root / p).resolve()
            p2 = (project_root / p).resolve()
            if p1.exists():
                return p1
            if p2.exists():
                return p2
        elif p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Passenger trace 不存在：{p}")

    # 与官方 test_rl.py 一致，优先 train_data/passengers_300.csv。
    preferred = [
        uagmc_root / "train_data" / "passengers_300.csv",
        uagmc_root / "test_data" / "passengers_300.csv",
        uagmc_root / "passengers.csv",
    ]
    for p in preferred:
        if p.exists():
            print(f"[自动发现] passenger trace：{p}")
            return p.resolve()

    candidates = [
        p for p in uagmc_root.rglob("*.csv") if "passenger" in p.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            "未发现 passenger CSV。请使用 --passengers 显式指定。"
        )

    candidates.sort(key=lambda p: str(p).lower())
    print(f"[自动发现] passenger trace：{candidates[0]}")
    return candidates[0].resolve()


# ============================================================
# VecEnv 解包与 UAGMC 接口
# ============================================================


def unwrap_uagmc_wrapper(vec_env: Any) -> Any:
    """从 VecNormalize -> DummyVecEnv -> Monitor 中找到 UAMRLWrapper。"""
    obj = vec_env

    if hasattr(obj, "venv"):
        obj = obj.venv

    if hasattr(obj, "envs") and obj.envs:
        obj = obj.envs[0]

    visited = set()
    for _ in range(20):
        if id(obj) in visited:
            break
        visited.add(id(obj))

        if (
            hasattr(obj, "state")
            and hasattr(obj, "encoder")
            and hasattr(obj, "decoder")
            and hasattr(obj, "env")
        ):
            return obj

        if hasattr(obj, "env"):
            obj = obj.env
        else:
            break

    raise RuntimeError(
        "无法找到 UAMRLWrapper。请确认复现环境仍使用官方 utilss/uam_rl_wrapper.py。"
    )


def get_candidate_ids(wrapper: Any, explicit: Optional[str]) -> List[int]:
    """读取 ActionDecoder 中真实的候选 departure vertiport。"""
    if explicit:
        return [int(x.strip()) for x in explicit.split(",") if x.strip()]

    decoder = wrapper.decoder
    if hasattr(decoder, "from_vertiports"):
        return [int(x) for x in decoder.from_vertiports]

    n_actions = int(wrapper.action_space.n)
    if n_actions == 2:
        print("[警告] 无法读取 decoder.from_vertiports，按官方默认使用 [0, 1]。")
        return [0, 1]

    raise RuntimeError(
        "无法自动确定 candidate vertiport。请显式传入，例如 --candidates 0,1"
    )


def get_waiting_pid(wrapper: Any) -> Optional[str]:
    """读取当前真正等待 RL 决策的第一个 passenger。"""
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
    完全复用 Scenario.apply_decision 中的物理 ground-access 计算：
    scenario.vehicles.estimate_travel_time(origin, destination)。
    """
    person = scenario.persons.persons[pid]
    result: Dict[int, float] = {}

    for vid in candidate_ids:
        vertiport = scenario.vertiports.vertiport_list[str(vid)]
        pickup_time = scenario.vehicles.estimate_travel_time(
            origin=person.origin_position,
            destination=vertiport.vertiport_position,
        )
        result[int(vid)] = float(pickup_time)

    return result


def get_policy_probs(model: Any, obs: np.ndarray) -> Optional[np.ndarray]:
    """读取 PPO 当前离散动作概率，不改变模型。"""
    try:
        import torch

        with torch.no_grad():
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            distribution = model.policy.get_distribution(obs_tensor)
            base_distribution = getattr(distribution, "distribution", None)
            probs = getattr(base_distribution, "probs", None)
            if probs is None:
                return None
            return probs.detach().cpu().numpy()[0].astype(float)
    except Exception:
        return None


# ============================================================
# 候选站状态提取
# ============================================================


def evtol_state_name(evtol: Any) -> str:
    """兼容枚举和字符串形式的 eVTOL state。"""
    try:
        return str(evtol.state.name)
    except Exception:
        return str(getattr(evtol, "state", "UNKNOWN"))


def extract_vertiport_state(
    scenario: Any,
    candidate_ids: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    """
    提取用于 temporal mismatch 诊断的候选站状态。

    前 8 个量与官方 ObservationEncoder 的 vertiport aggregate 语义对应：
    waiting_cnt, incoming_cnt, charging_evtols, total_evtols,
    total_capacity, avg_charge_time, min_charge_time, avg_flight_time。

    额外增加 idle_evtols，只用于离线解释，不进入 checkpoint policy。
    """
    result: Dict[int, Dict[str, float]] = {}

    for vid_int in candidate_ids:
        vid = str(vid_int)
        vertiport = scenario.vertiports.vertiport_list[vid]

        waiting_cnt = float(len(vertiport.person_list))

        incoming_cnt = 0.0
        incoming_remaining: List[float] = []
        for person in scenario.persons.persons.values():
            if (
                getattr(person, "state", None) == "enroute"
                and str(getattr(person, "origin_vertiport_id", "")) == vid
            ):
                incoming_cnt += 1.0
                timer = getattr(person, "current_timer", None)
                if timer is not None:
                    try:
                        incoming_remaining.append(float(timer))
                    except Exception:
                        pass

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
                    remaining_charge_time.append(float(ev.remaining_charge_time))
                except Exception:
                    pass

            if hasattr(ev, "remaining_flight_time"):
                try:
                    remaining_flight_time.append(float(ev.remaining_flight_time))
                except Exception:
                    pass

        result[vid_int] = {
            "waiting": waiting_cnt,
            "incoming": incoming_cnt,
            "charging_evtols": charging_evtols,
            "total_evtols": float(len(evtols)),
            "total_capacity": total_capacity,
            "avg_charge_time": (
                float(np.mean(remaining_charge_time)) if remaining_charge_time else 0.0
            ),
            "min_charge_time": (
                float(np.min(remaining_charge_time)) if remaining_charge_time else 0.0
            ),
            "avg_flight_time": (
                float(np.mean(remaining_flight_time)) if remaining_flight_time else 0.0
            ),
            "idle_evtols": idle_evtols,
            "min_incoming_passenger_time": (
                float(np.min(incoming_remaining)) if incoming_remaining else 0.0
            ),
            "avg_incoming_passenger_time": (
                float(np.mean(incoming_remaining)) if incoming_remaining else 0.0
            ),
        }

    return result


# ============================================================
# 诊断指标
# ============================================================


def state_drift(now: Dict[str, float], future: Dict[str, float]) -> float:
    """
    对官方 ObservationEncoder 的 8 个 vertiport aggregate 做无权重归一化漂移。
    这只是 descriptive diagnostic，不是 reward，也不是 proposed method。
    """
    keys = [
        "waiting",
        "incoming",
        "charging_evtols",
        "total_evtols",
        "total_capacity",
        "avg_charge_time",
        "min_charge_time",
        "avg_flight_time",
    ]

    values = []
    for key in keys:
        a = float(now.get(key, 0.0))
        b = float(future.get(key, 0.0))
        values.append(abs(b - a) / (1.0 + abs(a)))

    return float(np.mean(values)) if values else 0.0


def burden_proxy(state: Dict[str, float]) -> float:
    """waiting + incoming，仅用于 ranking-flip 诊断。"""
    return float(state.get("waiting", 0.0)) + float(state.get("incoming", 0.0))


def pressure_proxy(state: Dict[str, float]) -> float:
    """
    waiting + incoming - idle，只用于解释当前/未来站点排序是否变化。
    不作为论文最终 baseline，不参与在线动作选择。
    """
    return (
        float(state.get("waiting", 0.0))
        + float(state.get("incoming", 0.0))
        - float(state.get("idle_evtols", 0.0))
    )


def argmin_stable(values: Dict[int, float]) -> int:
    """稳定 argmin；平局取 vertiport id 更小者。"""
    return min(values.items(), key=lambda item: (item[1], item[0]))[0]


# ============================================================
# 命令行参数
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="UAGMC Stage-0 候选动作 temporal mismatch 诊断（无训练）"
    )

    parser.add_argument(
        "--project-root",
        default=str(Path.cwd()),
        help="项目根目录，默认当前 CMD 所在目录。",
    )
    parser.add_argument(
        "--uagmc-root",
        default=None,
        help="UAGMC 源码根目录；不填则自动发现。",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="已复现 PPO checkpoint (.zip)；不填则自动发现。",
    )
    parser.add_argument(
        "--vecnorm",
        default=None,
        help="对应 VecNormalize (.pkl)；不填则自动发现。",
    )
    parser.add_argument(
        "--passengers",
        default=None,
        help="固定 passenger CSV；不填优先官方 train_data/passengers_300.csv。",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="候选 departure vertiport，例如 0,1；不填从 ActionDecoder 读取。",
    )
    parser.add_argument(
        "--to-vertiport",
        type=int,
        default=2,
        help="原始 UAGMC arrival vertiport，默认 2。",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=600,
        help="仿真最长时间，官方 test_rl.py 默认为 600。",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="这里只做 checkpoint 推理，默认 CPU；GPU 已可用时可显式传 cuda。",
    )
    parser.add_argument(
        "--output-dir",
        default="diagnostics/uagmc_candidate_temporal_mismatch",
        help="结果输出目录，相对于项目根目录。",
    )

    return parser.parse_args()


def resolve_device(device: str) -> str:
    """仅做推理；auto 时才自动检测 CUDA。"""
    if device != "auto":
        return device

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ============================================================
# 主流程
# ============================================================


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"项目根目录不存在：{project_root}")

    uagmc_root = discover_uagmc_root(project_root, args.uagmc_root)
    model_path = discover_model(uagmc_root, project_root, args.model)
    vecnorm_path = discover_vecnorm(
        uagmc_root,
        project_root,
        model_path,
        args.vecnorm,
    )
    passenger_path = discover_passenger_file(
        uagmc_root,
        project_root,
        args.passengers,
    )

    # 确保导入的是这一份 UAGMC 源码。
    sys.path.insert(0, str(uagmc_root))

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from utilss.make_env import make_env

    # 官方原始 UAGMC 是 V0/V1 二选一；若你的复现已调整，可通过 --candidates 显式覆盖。
    initial_candidates = (
        [int(x.strip()) for x in args.candidates.split(",") if x.strip()]
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
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False

    device = resolve_device(args.device)
    model = PPO.load(str(model_path), env=env, device=device)

    wrapper = unwrap_uagmc_wrapper(env)
    scenario = wrapper.env
    candidate_ids = get_candidate_ids(wrapper, args.candidates)

    if len(candidate_ids) != int(wrapper.action_space.n):
        raise RuntimeError(
            f"candidate 数量 {len(candidate_ids)} 与 action_space.n={wrapper.action_space.n} 不一致。"
        )

    print("=" * 108)
    print("UAGMC CANDIDATE-RELATIVE TEMPORAL MISMATCH | STAGE-0 | NO TRAINING")
    print("=" * 108)
    print(f"Project root : {project_root}")
    print(f"UAGMC root   : {uagmc_root}")
    print(f"Model        : {model_path}")
    print(f"VecNormalize : {vecnorm_path}")
    print(f"Passengers   : {passenger_path}")
    print(f"Candidates   : {candidate_ids}")
    print(f"To vertiport : {args.to_vertiport}")
    print(f"Device       : {device}")
    print(f"Output       : {output_root}")
    print("-" * 108)

    # snapshots[t] 保存 checkpoint rollout 在 t 时刻真实出现的候选站状态。
    snapshots: Dict[int, Dict[int, Dict[str, float]]] = {}
    decisions: List[Dict[str, Any]] = []

    obs = env.reset()
    done = False
    step_count = 0

    while not done:
        sim_time = int(getattr(scenario, "time", step_count))

        if sim_time not in snapshots:
            snapshots[sim_time] = extract_vertiport_state(scenario, candidate_ids)

        pid = get_waiting_pid(wrapper)
        probs = get_policy_probs(model, obs)

        action, _ = model.predict(obs, deterministic=True)
        action_index = int(np.asarray(action).reshape(-1)[0])

        if pid is not None:
            access_times = estimate_access_times(scenario, pid, candidate_ids)

            if probs is None or len(probs) != len(candidate_ids):
                policy_probs = [float("nan")] * len(candidate_ids)
                policy_margin = float("nan")
            else:
                policy_probs = [float(x) for x in probs]
                ordered = sorted(policy_probs, reverse=True)
                policy_margin = (
                    float(ordered[0] - ordered[1]) if len(ordered) >= 2 else 1.0
                )

            person = scenario.persons.persons[pid]
            decisions.append(
                {
                    "decision_id": len(decisions),
                    "sim_time": sim_time,
                    "pid": pid,
                    "origin_x": float(person.origin_position[0]),
                    "origin_y": float(person.origin_position[1]),
                    "destination_x": float(person.destination_position[0]),
                    "destination_y": float(person.destination_position[1]),
                    "chosen_action_index": action_index,
                    "chosen_vertiport": int(candidate_ids[action_index]),
                    "access_times": access_times,
                    "policy_probs": policy_probs,
                    "policy_margin": policy_margin,
                    "now_state": extract_vertiport_state(scenario, candidate_ids),
                }
            )

        obs, reward, dones, infos = env.step(action)
        done = bool(np.asarray(dones).reshape(-1)[0])
        step_count += 1

        # 防止异常环境无限循环。
        if step_count > args.max_time + 1000:
            print("[警告] 超过安全 step 上限，强制结束 rollout。")
            break

    final_time = int(getattr(scenario, "time", step_count))
    if final_time not in snapshots:
        snapshots[final_time] = extract_vertiport_state(scenario, candidate_ids)

    available_times = sorted(snapshots.keys())

    def future_snapshot_time(target_time: float) -> Optional[int]:
        """
        UAGMC 每步 1 min；access time 可为小数。
        使用不早于真实 realization horizon 的第一个系统快照。
        """
        target = int(math.ceil(target_time))
        for t in available_times:
            if t >= target:
                return t
        return None

    # ========================================================
    # Decision-time vs candidate realization-time 诊断
    # ========================================================

    candidate_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []

    access_spreads: List[float] = []
    mean_drifts: List[float] = []
    waiting_flips: List[float] = []
    burden_flips: List[float] = []
    pressure_flips: List[float] = []
    burden_regrets: List[float] = []
    policy_margins: List[float] = []

    for decision in decisions:
        access_values = list(decision["access_times"].values())
        delay_spread = float(max(access_values) - min(access_values))
        access_spreads.append(delay_spread)
        policy_margins.append(float(decision["policy_margin"]))

        now_waiting: Dict[int, float] = {}
        now_burden: Dict[int, float] = {}
        now_pressure: Dict[int, float] = {}

        future_waiting: Dict[int, float] = {}
        future_burden: Dict[int, float] = {}
        future_pressure: Dict[int, float] = {}
        candidate_drifts: Dict[int, float] = {}

        for action_index, vid in enumerate(candidate_ids):
            access_time = float(decision["access_times"][vid])
            realization_time = float(decision["sim_time"]) + access_time
            future_t = future_snapshot_time(realization_time)

            now_state = decision["now_state"][vid]

            row: Dict[str, Any] = {
                "decision_id": decision["decision_id"],
                "sim_time": decision["sim_time"],
                "pid": decision["pid"],
                "origin_x": decision["origin_x"],
                "origin_y": decision["origin_y"],
                "destination_x": decision["destination_x"],
                "destination_y": decision["destination_y"],
                "candidate_action_index": action_index,
                "candidate_vertiport": vid,
                "checkpoint_choice": int(vid == decision["chosen_vertiport"]),
                "access_time": access_time,
                "access_delay_spread": delay_spread,
                "candidate_realization_time": realization_time,
                "future_snapshot_time": future_t if future_t is not None else "",
                "policy_prob": decision["policy_probs"][action_index],
                "policy_margin": decision["policy_margin"],
            }

            for key, value in now_state.items():
                row[f"now_{key}"] = float(value)

            now_waiting[vid] = float(now_state["waiting"])
            now_burden[vid] = burden_proxy(now_state)
            now_pressure[vid] = pressure_proxy(now_state)

            if future_t is None:
                row["future_available"] = 0
                row["state_drift"] = float("nan")
                candidate_rows.append(row)
                continue

            future_state = snapshots[future_t][vid]
            drift = state_drift(now_state, future_state)
            candidate_drifts[vid] = drift

            future_waiting[vid] = float(future_state["waiting"])
            future_burden[vid] = burden_proxy(future_state)
            future_pressure[vid] = pressure_proxy(future_state)

            row["future_available"] = 1
            row["state_drift"] = drift

            for key, value in future_state.items():
                row[f"future_{key}"] = float(value)
                row[f"delta_{key}"] = float(value) - float(now_state.get(key, 0.0))

            candidate_rows.append(row)

        if len(future_waiting) != len(candidate_ids):
            # episode 尾部如果某些 candidate horizon 已越过结束时间，不把该 decision 用于排名翻转统计。
            continue

        mean_drift = mean_or_nan(list(candidate_drifts.values()))
        mean_drifts.append(mean_drift)

        now_best_waiting = argmin_stable(now_waiting)
        future_best_waiting = argmin_stable(future_waiting)

        now_best_burden = argmin_stable(now_burden)
        future_best_burden = argmin_stable(future_burden)

        now_best_pressure = argmin_stable(now_pressure)
        future_best_pressure = argmin_stable(future_pressure)

        flip_waiting = int(now_best_waiting != future_best_waiting)
        flip_burden = int(now_best_burden != future_best_burden)
        flip_pressure = int(now_best_pressure != future_best_pressure)

        waiting_flips.append(float(flip_waiting))
        burden_flips.append(float(flip_burden))
        pressure_flips.append(float(flip_pressure))

        chosen = int(decision["chosen_vertiport"])
        burden_regret_proxy = float(
            future_burden[chosen] - min(future_burden.values())
        )
        burden_regrets.append(burden_regret_proxy)

        decision_rows.append(
            {
                "decision_id": decision["decision_id"],
                "sim_time": decision["sim_time"],
                "pid": decision["pid"],
                "chosen_vertiport": chosen,
                "access_delay_min": float(min(access_values)),
                "access_delay_max": float(max(access_values)),
                "access_delay_spread": delay_spread,
                "policy_margin": float(decision["policy_margin"]),
                "mean_candidate_state_drift": mean_drift,
                "now_best_waiting": now_best_waiting,
                "future_best_waiting": future_best_waiting,
                "flip_waiting": flip_waiting,
                "now_best_burden": now_best_burden,
                "future_best_burden": future_best_burden,
                "flip_burden": flip_burden,
                "now_best_pressure": now_best_pressure,
                "future_best_pressure": future_best_pressure,
                "flip_pressure": flip_pressure,
                "checkpoint_future_burden_regret_proxy": burden_regret_proxy,
            }
        )

    # ========================================================
    # 按 access-delay spread 分桶
    # ========================================================

    spread_bin_rows: List[Dict[str, Any]] = []

    if decision_rows:
        spreads = np.asarray(
            [float(row["access_delay_spread"]) for row in decision_rows],
            dtype=float,
        )
        quantile_edges = np.unique(
            np.quantile(spreads, [0.0, 0.25, 0.50, 0.75, 1.0])
        )

        for i in range(len(quantile_edges) - 1):
            lo = float(quantile_edges[i])
            hi = float(quantile_edges[i + 1])

            if i == len(quantile_edges) - 2:
                rows = [
                    row
                    for row in decision_rows
                    if lo <= float(row["access_delay_spread"]) <= hi
                ]
            else:
                rows = [
                    row
                    for row in decision_rows
                    if lo <= float(row["access_delay_spread"]) < hi
                ]

            if not rows:
                continue

            spread_bin_rows.append(
                {
                    "spread_lo": lo,
                    "spread_hi": hi,
                    "n": len(rows),
                    "mean_state_drift": mean_or_nan(
                        [float(row["mean_candidate_state_drift"]) for row in rows]
                    ),
                    "waiting_flip_rate": mean_or_nan(
                        [float(row["flip_waiting"]) for row in rows]
                    ),
                    "burden_flip_rate": mean_or_nan(
                        [float(row["flip_burden"]) for row in rows]
                    ),
                    "pressure_flip_rate": mean_or_nan(
                        [float(row["flip_pressure"]) for row in rows]
                    ),
                    "mean_checkpoint_future_burden_regret_proxy": mean_or_nan(
                        [
                            float(row["checkpoint_future_burden_regret_proxy"])
                            for row in rows
                        ]
                    ),
                }
            )

    # ========================================================
    # 输出
    # ========================================================

    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "stage": "Stage-0 no-training temporal mismatch diagnostic",
        "project_root": str(project_root),
        "uagmc_root": str(uagmc_root),
        "model": str(model_path),
        "vecnormalize": str(vecnorm_path) if vecnorm_path else None,
        "passenger_trace": str(passenger_path),
        "candidate_vertiports": candidate_ids,
        "to_vertiport": args.to_vertiport,
        "device": device,
        "simulation_steps": step_count,
        "recorded_decisions": len(decisions),
        "analyzable_decisions": len(decision_rows),
        "access_delay": {
            "mean_spread": mean_or_nan(access_spreads),
            "median_spread": percentile_or_nan(access_spreads, 50),
            "p90_spread": percentile_or_nan(access_spreads, 90),
            "max_spread": float(np.max(access_spreads)) if access_spreads else float("nan"),
        },
        "temporal_mismatch": {
            "mean_candidate_state_drift": mean_or_nan(mean_drifts),
            "median_candidate_state_drift": percentile_or_nan(mean_drifts, 50),
            "p90_candidate_state_drift": percentile_or_nan(mean_drifts, 90),
            "waiting_rank_flip_rate": mean_or_nan(waiting_flips),
            "burden_rank_flip_rate": mean_or_nan(burden_flips),
            "pressure_rank_flip_rate": mean_or_nan(pressure_flips),
            "checkpoint_future_burden_regret_proxy_mean": mean_or_nan(
                burden_regrets
            ),
        },
        "checkpoint_policy": {
            "mean_margin": mean_or_nan(policy_margins),
            "median_margin": percentile_or_nan(policy_margins, 50),
            "p10_margin": percentile_or_nan(policy_margins, 10),
        },
        "spread_bins": spread_bin_rows,
        "interpretation_guardrails": [
            "future state 是 checkpoint 实际后续 rollout 的 oracle diagnostic，不是在线可用信息。",
            "ranking flip 使用 waiting / burden / pressure proxy，只用于暴露 temporal mismatch，不等于真实 ATT counterfactual optimum。",
            "该实验只回答‘问题是否存在’，不证明 candidate-relative 方法一定优于 absolute-event 方法。",
            "只有 Stage-0 发现明显 mismatch 后，才进入后续 U0/U1/U2/U3 控制训练。",
        ],
    }

    write_csv(output_root / "candidate_detail.csv", candidate_rows)
    write_csv(output_root / "decision_summary.csv", decision_rows)
    write_csv(output_root / "delay_spread_bins.csv", spread_bin_rows)

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(summary), f, ensure_ascii=False, indent=2)

    with (output_root / "resolved_config.json").open("w", encoding="utf-8") as f:
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

    print("\n" + "=" * 108)
    print("STAGE-0 DIAGNOSTIC SUMMARY")
    print("=" * 108)
    print(f"Recorded decisions             : {len(decisions)}")
    print(f"Analyzable decisions           : {len(decision_rows)}")
    print(f"Mean access-delay spread       : {summary['access_delay']['mean_spread']:.4f} min")
    print(f"P90 access-delay spread        : {summary['access_delay']['p90_spread']:.4f} min")
    print(f"Mean candidate state drift     : {summary['temporal_mismatch']['mean_candidate_state_drift']:.4f}")
    print(f"Waiting ranking flip rate      : {summary['temporal_mismatch']['waiting_rank_flip_rate']:.4f}")
    print(f"Burden ranking flip rate       : {summary['temporal_mismatch']['burden_rank_flip_rate']:.4f}")
    print(f"Pressure ranking flip rate     : {summary['temporal_mismatch']['pressure_rank_flip_rate']:.4f}")
    print(f"Checkpoint mean policy margin  : {summary['checkpoint_policy']['mean_margin']:.4f}")
    print(
        "Future burden regret proxy    : "
        f"{summary['temporal_mismatch']['checkpoint_future_burden_regret_proxy_mean']:.4f}"
    )
    print("-" * 108)
    print("注意：future state 是离线诊断 oracle，不能作为在线策略输入。")
    print(f"结果目录：{output_root}")
    print("=" * 108)


if __name__ == "__main__":
    main()
