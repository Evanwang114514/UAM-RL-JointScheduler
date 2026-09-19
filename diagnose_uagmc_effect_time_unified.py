# -*- coding: utf-8 -*-
r"""
UAGMC Unified Candidate-Specific Effect-Time Diagnostic
=======================================================

This file replaces the earlier chain of Stage-0 / 0.5 / 0.6 / 0.7 / 0.8
diagnostic scripts with ONE corrected diagnostic.

RETAINED LOGIC
--------------
1. Same-passenger candidate access/effect-time heterogeneity.
2. Decision-time -> each candidate's OWN effect-time state drift.
3. Candidate-independent shared future horizon -> own effect-time mismatch.
4. Legal committed-event crossing using ONLY passengers already committed to
   ground access at decision time.
5. Correct incoming semantics:
       state == "enroute" AND sub_state == "to_vertiport"
6. Correct discrete remaining access ETA:
       current_timer + 1
7. Online timer-countdown audit.
8. Offline-oracle ranking-change diagnostics, clearly marked as OFFLINE ONLY.
9. Optional exact Longest-Queue / VertiSync-simple reposition comparison using
   the SAME training-time reposition patch.

DISCARDED / SUPERSEDED LOGIC
----------------------------
The following older diagnostic ideas are intentionally NOT implemented:
- strict timer-only "Committed-only -> Oracle" as a claim that it reconstructs
  the future state;
- arrival-only peeling as a method-validity proof;
- historical-service-rate P2 peeling;
- committed-cohort oracle as a proof that the online representation is correct;
- the old broad definition "state == enroute" for passenger incoming;
- current_timer interpreted without the +1 discrete-step correction.

Why:
The corrected master diagnostic explicitly treats future realized state as an
OFFLINE oracle only. It tests whether temporal mismatch exists and whether
decision-time-known committed events cross candidate-specific boundaries.
It does NOT claim to reconstruct the counterfactual future online.

MODES
-----
Base UAGMC:
    python diagnose_uagmc_effect_time_unified.py --mode base

Exact N=16 reposition comparison:
    python diagnose_uagmc_effect_time_unified.py --mode reposition

The reposition mode still requires the TRAINING implementation:
    train_uagmc_reposition_lq_vs_vertisync_800k.py
because the aircraft reposition policy must be exactly the one used in training,
not reimplemented in this diagnostic.

NO TRAINING occurs in this file.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import sys
import time
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# Standalone helpers retained from Stage-0
# =============================================================================

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


def resolve_device(device: str) -> str:
    """仅做推理；auto 时才自动检测 CUDA。"""
    if device != "auto":
        return device

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# =============================================================================
# Logic provenance / retirement manifest
# =============================================================================

LOGIC_STATUS = {
    "retained": [
        "candidate-specific access/effect-time heterogeneity",
        "decision-time -> own-effect-time SAFE/CORRECTED state drift",
        "shared-horizon -> own-horizon mismatch",
        "legal committed access-event crossing",
        "timer countdown audit",
        "offline oracle ranking-change diagnostic",
        "exact training-time reposition policy adapter",
    ],
    "discarded_or_superseded": [
        "old committed-vs-oracle projection as future-state reconstruction",
        "arrival-only peeling as method-validity evidence",
        "historical service-rate peeling (P2)",
        "committed-cohort oracle as proof of online correctness",
        "state==enroute as incoming-passenger definition",
        "current_timer without +1 discrete-step correction",
    ],
}


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

def _core_parse_args():
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

def _core_main() -> None:
    args = _core_parse_args()

    project_root = Path(args.project_root).expanduser().resolve()

    if not project_root.exists():
        raise FileNotFoundError(
            f"项目根目录不存在：{project_root}"
        )

    uagmc_root = discover_uagmc_root(
        project_root,
        args.uagmc_root,
    )

    model_path = discover_model(
        uagmc_root,
        project_root,
        args.model,
    )

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

    device = resolve_device(args.device)

    model = PPO.load(
        str(model_path),
        env=env,
        device=device,
    )

    wrapper = unwrap_uagmc_wrapper(env)
    scenario = wrapper.env

    candidate_ids = get_candidate_ids(
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

        focal_pid = get_waiting_pid(
            wrapper
        )

        probs = get_policy_probs(
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
            access_times = estimate_access_times(
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





# =============================================================================
# Unified wrapper: base mode + exact reposition mode
# =============================================================================

VALID_REPOSITION_METHODS = ("longest_queue", "vertisync_simple")
DEFAULT_REPOSITION_STEPS = (800_000,)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("integer list cannot be empty")
    return vals


def _parse_methods(text: str) -> List[str]:
    vals = [x.strip().lower() for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("method list cannot be empty")
    bad = [x for x in vals if x not in VALID_REPOSITION_METHODS]
    if bad:
        raise ValueError(
            f"Unknown reposition method(s): {bad}; "
            f"valid={VALID_REPOSITION_METHODS}"
        )
    return vals


def _auto_find_reposition_run_root() -> Path:
    serial = ROOT / "serial_runs"
    patterns = (
        "uagmc_reposition_LQ_vs_VertiSync_N16_seed*_800k_*",
        "*reposition*LQ*VertiSync*N16*800k*",
    )

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
            "Cannot auto-detect LQ-vs-VertiSync N16 run under serial_runs/. "
            "Use --run-root explicitly."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def _discover_reposition_checkpoint(
    run_root: Path,
    method: str,
    step: int,
) -> Tuple[Path, Path]:
    ckpt = run_root / method / "checkpoints"
    model = ckpt / f"uam_ppo_{int(step)}_steps.zip"
    vec = ckpt / f"uam_ppo_vecnormalize_{int(step)}_steps.pkl"

    if not model.exists():
        raise FileNotFoundError(model)
    if not vec.exists():
        raise FileNotFoundError(vec)

    return model.resolve(), vec.resolve()


class _FixedFleetMakeEnvPatch:
    """
    Redirect utilss.make_env.make_env to make_env_fleet.make_env while the
    corrected diagnostic is running. The reposition rule itself is installed
    from the exact training file.
    """

    def __init__(self, fleet_size: int = 16):
        self.fleet_size = int(fleet_size)
        self.legacy_module = None
        self.old_make_env = None

    def install(self) -> None:
        legacy_module = importlib.import_module("utilss.make_env")
        fleet_module = importlib.import_module("utilss.make_env_fleet")

        self.legacy_module = legacy_module
        self.old_make_env = legacy_module.make_env
        fleet_make_env = fleet_module.make_env

        fleet_size = self.fleet_size

        def fixed_fleet_make_env(*args, **kwargs):
            kwargs = dict(kwargs)
            kwargs.update(
                fleet_mode="conserved_closed_loop",
                fleet_size=fleet_size,
                fleet_assertions=True,
            )
            return fleet_make_env(*args, **kwargs)

        legacy_module.make_env = fixed_fleet_make_env

    def restore(self) -> None:
        if self.legacy_module is not None and self.old_make_env is not None:
            self.legacy_module.make_env = self.old_make_env


def _result_dirs(base: Path) -> Set[Path]:
    if not base.exists():
        return set()
    return {
        p.resolve()
        for p in base.iterdir()
        if p.is_dir() and (p / "summary.json").exists()
    }


def _new_result_dir(base: Path, before: Set[Path]) -> Path:
    after = _result_dirs(base)
    new_dirs = sorted(
        after - before,
        key=lambda p: p.stat().st_mtime,
    )
    if new_dirs:
        return new_dirs[-1]

    all_dirs = sorted(
        after,
        key=lambda p: p.stat().st_mtime,
    )
    if not all_dirs:
        raise RuntimeError(
            f"Diagnostic finished but no summary.json exists under {base}"
        )
    return all_dirs[-1]


def _run_core_with_argv(argv: Sequence[str]) -> None:
    old_argv = list(sys.argv)
    try:
        sys.argv = [str(Path(__file__).resolve())] + list(argv)
        _core_main()
    finally:
        sys.argv = old_argv


def _get_nested(data: Dict[str, Any], *keys: str, default=float("nan")):
    obj: Any = data
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def _flatten_summary(
    label: str,
    step: int,
    result_dir: Path,
) -> Dict[str, Any]:
    summary = json.loads(
        (result_dir / "summary.json").read_text(encoding="utf-8")
    )

    return {
        "label": label,
        "train_step": int(step),
        "result_dir": str(result_dir),
        "recorded_decisions": summary.get("recorded_decisions"),
        "complete_candidate_rows": summary.get("complete_candidate_rows"),
        "complete_decisions": summary.get("complete_decisions"),

        "mean_delay_spread": _get_nested(
            summary, "same_passenger_candidate_delay", "mean_spread"
        ),
        "p90_delay_spread": _get_nested(
            summary, "same_passenger_candidate_delay", "p90_spread"
        ),

        "mean_safe_drift_now_to_own": _get_nested(
            summary, "oracle_effect_time_staleness",
            "mean_safe_drift_now_to_own"
        ),
        "safe_change_rate": _get_nested(
            summary, "oracle_effect_time_staleness",
            "safe_any_change_rate"
        ),
        "rho_access_safe_drift": _get_nested(
            summary, "oracle_effect_time_staleness",
            "rho_access_vs_safe_drift"
        ),
        "p_access_safe_drift": _get_nested(
            summary, "oracle_effect_time_staleness",
            "p_access_vs_safe_drift"
        ),
        "long_short_drift_ratio": _get_nested(
            summary, "oracle_effect_time_staleness",
            "long_vs_short_drift_ratio"
        ),

        "mean_safe_drift_shared_to_own": _get_nested(
            summary, "candidate_specific_vs_shared_future_reference",
            "mean_safe_drift_shared_to_own"
        ),
        "shared_horizon_mismatch_rate": _get_nested(
            summary, "candidate_specific_vs_shared_future_reference",
            "shared_horizon_safe_mismatch_rate"
        ),
        "known_boundary_diff_rate": _get_nested(
            summary, "candidate_specific_vs_shared_future_reference",
            "decision_boundary_diff_rate_known_committed"
        ),

        "committed_crossing_rate": _get_nested(
            summary, "legal_committed_event_evidence",
            "positive_crossing_rate"
        ),
        "own_shared_cross_diff_rate": _get_nested(
            summary, "legal_committed_event_evidence",
            "own_vs_shared_cross_count_diff_rate"
        ),

        "timer_transitions": _get_nested(
            summary, "timer_semantics_audit", "n_transitions"
        ),
        "timer_fraction_exact_minus_1": _get_nested(
            summary, "timer_semantics_audit", "fraction_exact_minus_1"
        ),

        "waiting_rank_flip_now_to_own": _get_nested(
            summary, "decision_relevance_offline_oracle",
            "now_to_own_waiting_rank_flip_rate"
        ),
        "burden_rank_flip_now_to_own": _get_nested(
            summary, "decision_relevance_offline_oracle",
            "now_to_own_corrected_burden_rank_flip_rate"
        ),
        "pressure_rank_flip_now_to_own": _get_nested(
            summary, "decision_relevance_offline_oracle",
            "now_to_own_corrected_pressure_rank_flip_rate"
        ),

        "gate_status": _get_nested(
            summary, "gate", "status", default="UNKNOWN"
        ),
        "gate_pass": _get_nested(
            summary, "gate", "overall_pass", default=False
        ),
    }


def _build_pairwise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_key = {
        (str(r["label"]), int(r["train_step"])): r
        for r in rows
    }

    fields = (
        "mean_safe_drift_now_to_own",
        "safe_change_rate",
        "rho_access_safe_drift",
        "long_short_drift_ratio",
        "mean_safe_drift_shared_to_own",
        "shared_horizon_mismatch_rate",
        "known_boundary_diff_rate",
        "committed_crossing_rate",
        "own_shared_cross_diff_rate",
        "waiting_rank_flip_now_to_own",
        "burden_rank_flip_now_to_own",
        "pressure_rank_flip_now_to_own",
    )

    out: Dict[str, Any] = {
        "interpretation": (
            "sync_minus_lq is purely descriptive. A larger temporal-mismatch "
            "metric is not automatically better or worse."
        ),
        "steps": {},
    }

    for step in sorted({int(r["train_step"]) for r in rows}):
        lq = by_key.get(("longest_queue", step))
        sy = by_key.get(("vertisync_simple", step))
        if lq is None or sy is None:
            continue

        row: Dict[str, Any] = {}
        for field in fields:
            try:
                lv = float(lq.get(field))
                sv = float(sy.get(field))
                delta = sv - lv
            except Exception:
                lv = sv = delta = float("nan")

            row[field] = {
                "longest_queue": lv,
                "vertisync_simple": sv,
                "sync_minus_lq": delta,
            }

        out["steps"][str(step)] = row

    return out


def _build_zip(root: Path, name: str) -> Path:
    zpath = root / name
    with zipfile.ZipFile(
        zpath,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p == zpath:
                continue
            if p.suffix.lower() in {".csv", ".json", ".txt"}:
                z.write(p, p.relative_to(root).as_posix())
    return zpath


def _write_unified_logic_manifest(path: Path) -> None:
    _write_json(
        path,
        {
            "logic_status": LOGIC_STATUS,
            "scientific_guardrails": [
                "No training is performed.",
                "Oracle future state is offline diagnostic only.",
                "Known committed-event evidence uses decision-time-known "
                "ground-access passengers only.",
                "The diagnostic demonstrates temporal mismatch, not ATT gain "
                "or counterfactual optimality.",
            ],
        },
    )


def _run_base(args) -> int:
    output_base = Path(args.output_dir)
    if not output_base.is_absolute():
        output_base = (Path(args.project_root).expanduser().resolve() / output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    before = _result_dirs(output_base)

    core_argv = [
        "--project-root", str(args.project_root),
        "--max-time", str(args.max_time),
        "--device", str(args.device),
        "--output-dir", str(output_base),
        "--to-vertiport", str(args.to_vertiport),
        "--gate-min-mean-delay-spread", str(args.gate_min_mean_delay_spread),
        "--gate-min-safe-change-rate", str(args.gate_min_safe_change_rate),
        "--gate-min-rho-access-safe-drift", str(args.gate_min_rho_access_safe_drift),
        "--gate-min-committed-crossing-rate", str(args.gate_min_committed_crossing_rate),
        "--gate-min-timer-consistency", str(args.gate_min_timer_consistency),
    ]

    if args.uagmc_root:
        core_argv += ["--uagmc-root", str(args.uagmc_root)]
    if args.model:
        core_argv += ["--model", str(args.model)]
    if args.vecnorm:
        core_argv += ["--vecnorm", str(args.vecnorm)]
    if args.passengers:
        core_argv += ["--passengers", str(args.passengers)]
    if args.candidates:
        core_argv += ["--candidates", str(args.candidates)]

    _run_core_with_argv(core_argv)

    result_dir = _new_result_dir(output_base, before)
    _write_unified_logic_manifest(result_dir / "logic_manifest.json")
    bundle = _build_zip(
        result_dir,
        "UPLOAD_THIS_effect_time_unified.zip",
    )

    print("\n" + "=" * 124)
    print("UNIFIED BASE DIAGNOSTIC COMPLETE")
    print("=" * 124)
    print(f"Result : {result_dir}")
    print("UPLOAD THIS FILE TO CHATGPT:")
    print(bundle)
    print("=" * 124)
    return 0


def _run_reposition(args) -> int:
    run_root = (
        Path(args.run_root).expanduser()
        if args.run_root
        else _auto_find_reposition_run_root()
    )
    if not run_root.is_absolute():
        run_root = (ROOT / run_root).resolve()
    else:
        run_root = run_root.resolve()

    if not run_root.exists():
        raise FileNotFoundError(run_root)

    passenger_file = (
        Path(args.passengers).expanduser()
        if args.passengers
        else ROOT / "train_data" / "passengers_300.csv"
    )
    if not passenger_file.is_absolute():
        passenger_file = (ROOT / passenger_file).resolve()
    else:
        passenger_file = passenger_file.resolve()

    if not passenger_file.exists():
        raise FileNotFoundError(passenger_file)

    train_file = ROOT / "train_uagmc_reposition_lq_vs_vertisync_800k.py"
    if not train_file.exists():
        raise FileNotFoundError(
            "Reposition mode requires exact training implementation: "
            f"{train_file}"
        )

    methods = _parse_methods(args.methods)
    steps = sorted(set(_parse_int_list(args.steps)))

    output_root = run_root / "effect_time_unified_compare"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_unified_logic_manifest(output_root / "logic_manifest.json")

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    sys.path.insert(0, str(ROOT))
    trainmod = importlib.import_module(
        "train_uagmc_reposition_lq_vs_vertisync_800k"
    )

    for step in steps:
        for method in methods:
            try:
                model, vec = _discover_reposition_checkpoint(
                    run_root,
                    method,
                    step,
                )

                # Use the EXACT policy installation function used in training.
                patch_info = trainmod.install_reposition_patch(method)

                method_base = (
                    output_root
                    / method
                    / f"step_{int(step):07d}"
                )
                method_base.mkdir(parents=True, exist_ok=True)
                before = _result_dirs(method_base)

                make_env_patch = _FixedFleetMakeEnvPatch(
                    fleet_size=int(args.fleet_size)
                )

                try:
                    make_env_patch.install()

                    core_argv = [
                        "--project-root", str(ROOT),
                        "--uagmc-root", str(ROOT),
                        "--model", str(model),
                        "--vecnorm", str(vec),
                        "--passengers", str(passenger_file),
                        "--candidates", "0,1",
                        "--to-vertiport", "2",
                        "--max-time", str(args.max_time),
                        "--device", str(args.device),
                        "--output-dir", str(method_base),
                        "--gate-min-mean-delay-spread",
                        str(args.gate_min_mean_delay_spread),
                        "--gate-min-safe-change-rate",
                        str(args.gate_min_safe_change_rate),
                        "--gate-min-rho-access-safe-drift",
                        str(args.gate_min_rho_access_safe_drift),
                        "--gate-min-committed-crossing-rate",
                        str(args.gate_min_committed_crossing_rate),
                        "--gate-min-timer-consistency",
                        str(args.gate_min_timer_consistency),
                    ]

                    print("\n" + "#" * 124)
                    print(
                        f"UNIFIED REPOSITION DIAGNOSTIC | "
                        f"{method} @ {int(step):,}"
                    )
                    print(f"model       : {model}")
                    print(f"vecnormalize: {vec}")
                    print(
                        f"fleet       : conserved_closed_loop, "
                        f"N={int(args.fleet_size)}"
                    )
                    print(f"patch       : {patch_info}")
                    print("#" * 124)

                    _run_core_with_argv(core_argv)

                finally:
                    make_env_patch.restore()

                result_dir = _new_result_dir(method_base, before)
                _write_unified_logic_manifest(
                    result_dir / "logic_manifest.json"
                )

                rows.append(
                    _flatten_summary(
                        method,
                        step,
                        result_dir,
                    )
                )

            except Exception as exc:
                err = {
                    "method": method,
                    "step": int(step),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(err)
                print(
                    f"\n[ERROR] {method}@{int(step):,}: {repr(exc)}",
                    flush=True,
                )
                if not args.continue_on_error:
                    write_csv(output_root / "errors.csv", errors)
                    raise

    rows.sort(
        key=lambda r: (
            int(r["train_step"]),
            str(r["label"]),
        )
    )

    write_csv(output_root / "method_summary.csv", rows)
    write_csv(output_root / "errors.csv", errors)
    _write_json(
        output_root / "pairwise_comparison.json",
        _build_pairwise(rows),
    )

    _write_json(
        output_root / "comparison_manifest.json",
        {
            "mode": "reposition",
            "run_root": str(run_root),
            "methods": methods,
            "steps": steps,
            "fleet_mode": "conserved_closed_loop",
            "fleet_size": int(args.fleet_size),
            "passenger_file": str(passenger_file),
            "max_time": int(args.max_time),
            "logic_status": LOGIC_STATUS,
            "no_training": True,
        },
    )

    bundle = _build_zip(
        output_root,
        "UPLOAD_THIS_effect_time_unified_compare.zip",
    )

    print("\n" + "=" * 124)
    print("UNIFIED REPOSITION DIAGNOSTIC COMPLETE")
    print("=" * 124)
    for row in rows:
        print(
            f"{row['label']:<18} @ {int(row['train_step']):>8,d} | "
            f"safe_change={float(row.get('safe_change_rate', float('nan'))):.4f} | "
            f"rho={float(row.get('rho_access_safe_drift', float('nan'))):.4f} | "
            f"shared={float(row.get('shared_horizon_mismatch_rate', float('nan'))):.4f} | "
            f"cross={float(row.get('committed_crossing_rate', float('nan'))):.4f} | "
            f"gate={row.get('gate_status')}"
        )
    print("-" * 124)
    print("UPLOAD THIS FILE TO CHATGPT:")
    print(bundle)
    print("=" * 124)

    return 0 if not errors else 2


def _unified_parse_args():
    p = argparse.ArgumentParser(
        description=(
            "One corrected UAGMC effect-time diagnostic replacing the earlier "
            "Stage-0/0.5/0.6/0.7/0.8 script chain."
        )
    )

    p.add_argument(
        "--mode",
        choices=["base", "reposition"],
        default="base",
    )

    # Shared corrected-master settings.
    p.add_argument("--project-root", default=str(Path.cwd()))
    p.add_argument("--uagmc-root", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--vecnorm", default=None)
    p.add_argument("--passengers", default=None)
    p.add_argument("--candidates", default=None)
    p.add_argument("--to-vertiport", type=int, default=2)
    p.add_argument("--max-time", type=int, default=600)
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
    )
    p.add_argument(
        "--output-dir",
        default="diagnostics/uagmc_effect_time_unified",
    )

    # Reposition mode.
    p.add_argument("--run-root", default=None)
    p.add_argument(
        "--methods",
        default="longest_queue,vertisync_simple",
    )
    p.add_argument(
        "--steps",
        default="800000",
    )
    p.add_argument(
        "--fleet-size",
        type=int,
        default=16,
        help=(
            "Must match the reposition training run. "
            "Historical LQ-vs-VertiSync experiment uses N=16."
        ),
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    # Engineering go/no-go thresholds copied from corrected master.
    p.add_argument(
        "--gate-min-mean-delay-spread",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--gate-min-safe-change-rate",
        type=float,
        default=0.40,
    )
    p.add_argument(
        "--gate-min-rho-access-safe-drift",
        type=float,
        default=0.10,
    )
    p.add_argument(
        "--gate-min-committed-crossing-rate",
        type=float,
        default=0.10,
    )
    p.add_argument(
        "--gate-min-timer-consistency",
        type=float,
        default=0.95,
    )

    return p.parse_args()


def main() -> int:
    args = _unified_parse_args()

    print("=" * 124)
    print("UAGMC UNIFIED EFFECT-TIME DIAGNOSTIC | NO TRAINING")
    print("=" * 124)
    print(f"Mode : {args.mode}")
    print(
        "Retired logic: old committed/oracle projection, arrival/service "
        "peeling, cohort-oracle proof."
    )
    print(
        "Retained logic: corrected effect-time master + timer audit + "
        "optional exact reposition adapter."
    )
    print("=" * 124)

    if args.mode == "base":
        return _run_base(args)

    if args.mode == "reposition":
        return _run_reposition(args)

    raise ValueError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
