#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UAGMC checkpoint learning-curve evaluator
=========================================

用途
----
对已经完成的 serial training 实验做纯评估，不训练：

1. 自动扫描:
      serial_runs/.../seed_*/checkpoints/model_*_steps.zip
   并匹配:
      vecnormalize_*_steps.pkl

2. 同时评估官方/原始参考模型:
      models/final_rl_model.zip
      models/final_vec_normalize.pkl

3. 在完全相同 passenger trace + eval seeds 上使用 deterministic policy，
   输出随训练步数变化的:
      ATT
      Access time (= time_stats["to_vertiport"])
      AWT         (= time_stats["wait_uam"])
      AFT         (= time_stats["fly"])
      P50 / P90 / P95 / max travel time
      completion rate
      action share

4. 生成:
      episode_metrics.csv
      curve_by_train_seed.csv
      curve_across_train_seeds.csv
      official_reference.csv
      att_curve.png
      awt_curve.png
      access_curve.png
      aft_curve.png
      completion_curve.png
      summary.txt

重要口径
--------
ATT:
    mean(end_time - start_time) over completed passengers.

这与官方 UAGMC Scenario 中 passenger travel record 的时间口径一致。
原始 UAGMC 在到达目标 vertiport 后即结束 passenger travel record，因此这里:
    - Access / AWT / AFT 可以直接分解；
    - 没有 final egress ground time，脚本不会伪造该指标；
    - residual = ATT - Access - AWT - AFT 作为一致性诊断输出。

推荐放置位置
------------
E:\\Study Files\\github\\UAM-predict\\UAGMC-main\\
与 utilss/, models/, train_encoder.py 同级。

典型运行
--------
python eval_uagmc_serial_learning_curve.py ^
  --run-root "serial_runs\\uagmc_gpu_10m_20260918_003515"

默认:
    - 扫描全部 seed
    - 每 100k checkpoint 评估一次
    - eval seeds = 123,124,125
    - passenger trace = train_data/passengers_300.csv
    - deterministic evaluation
    - CPU 推理（MLP PPO 通常 CPU 更快；不改变策略）

如果想评估每个 50k checkpoint:
python eval_uagmc_serial_learning_curve.py ^
  --run-root "serial_runs\\uagmc_gpu_10m_20260918_003515" ^
  --eval-every 50000

只看部分训练 seed:
python eval_uagmc_serial_learning_curve.py ^
  --run-root "serial_runs\\uagmc_gpu_10m_20260918_003515" ^
  --train-seeds 0,2,4

只用一个 eval seed 快速检查:
python eval_uagmc_serial_learning_curve.py ^
  --run-root "serial_runs\\uagmc_gpu_10m_20260918_003515" ^
  --eval-seeds 123

说明
----
- 本脚本不训练。
- 每个 checkpoint 必须使用它自己的 VecNormalize。
- 官方原模型作为 reference 单独输出，不伪装成某个训练 step。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parent

# UAGMC 源码导入必须从项目根目录进行。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utilss.make_env import make_env  # noqa: E402


# =============================================================================
# 数据结构
# =============================================================================

@dataclass(frozen=True)
class ModelSpec:
    source_group: str          # "serial" or "official_original"
    train_seed: Optional[int]
    train_step: Optional[int]
    model_path: Path
    vecnormalize_path: Optional[Path]
    label: str


# =============================================================================
# 基础工具
# =============================================================================

def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for x in str(text).split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    if not out:
        raise ValueError("整数列表不能为空。")
    return out


def as_float(x: Any, default=float("nan")) -> float:
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return default


def finite_mean(values: Iterable[Any]) -> float:
    arr = np.asarray([as_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def finite_std(values: Iterable[Any], ddof: int = 1) -> float:
    arr = np.asarray([as_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    if len(arr) == 1:
        return 0.0
    return float(arr.std(ddof=ddof))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            serial = {}
            for k, v in row.items():
                if isinstance(v, (list, tuple, dict, np.ndarray)):
                    serial[k] = json.dumps(
                        v.tolist() if isinstance(v, np.ndarray) else v,
                        ensure_ascii=False,
                    )
                else:
                    serial[k] = v
            w.writerow(serial)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_step(path: Path) -> Optional[int]:
    m = re.search(r"model_(\d+)_steps", path.stem)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d+)_steps", path.stem)
    if m:
        return int(m.group(1))

    return None


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested

    # SB3 MLP policy inference is usually cheaper on CPU.
    return "cpu"


# =============================================================================
# UAGMC terminal passenger capture
# =============================================================================

def find_scenario(env: Any):
    """
    Descend through Monitor / UAMRLWrapper until the official Scenario is found.
    """
    obj = env
    seen = set()

    for _ in range(30):
        if id(obj) in seen:
            break
        seen.add(id(obj))

        if all(
            hasattr(obj, key)
            for key in ("person_travel_records", "persons", "finished_ids")
        ):
            return obj

        if hasattr(obj, "scenario"):
            sc = getattr(obj, "scenario")
            if all(
                hasattr(sc, key)
                for key in ("person_travel_records", "persons", "finished_ids")
            ):
                return sc

        if hasattr(obj, "env"):
            obj = obj.env
            continue

        if hasattr(obj, "unwrapped") and obj.unwrapped is not obj:
            obj = obj.unwrapped
            continue

        break

    raise RuntimeError(
        "无法在 wrapper stack 中定位 UAGMC Scenario。"
    )


class TerminalPassengerCapture(gym.Wrapper):
    """
    必须在 DummyVecEnv 自动 reset 前复制 terminal passenger records。
    """

    def _snapshot(self) -> Dict[str, Any]:
        scenario = find_scenario(self.env)

        persons = (
            getattr(
                getattr(scenario, "persons", None),
                "persons",
                {},
            )
            or {}
        )
        records = (
            getattr(scenario, "person_travel_records", {})
            or {}
        )
        finished_ids = set(
            getattr(scenario, "finished_ids", [])
            or []
        )

        rows: List[Dict[str, Any]] = []

        for pid, person in persons.items():
            recs = records.get(pid, []) or []
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
                    "pid": str(pid),
                    "finished": bool(
                        pid in finished_ids
                        or end_time is not None
                    ),
                    "method": getattr(person, "method", None),
                    "state": getattr(person, "state", None),
                    "sub_state": getattr(person, "sub_state", None),
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

        return {
            "time": getattr(scenario, "time", None),
            "n_persons": len(persons),
            "n_finished_ids": len(finished_ids),
            "rows": rows,
        }

    def step(self, action):
        result = self.env.step(action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            if terminated or truncated:
                info = dict(info)
                info["uagmc_terminal_snapshot"] = self._snapshot()
            return obs, reward, terminated, truncated, info

        obs, reward, done, info = result
        if done:
            info = dict(info)
            info["uagmc_terminal_snapshot"] = self._snapshot()
        return obs, reward, done, info


# =============================================================================
# 模型扫描
# =============================================================================

def resolve_run_root(text: str) -> Path:
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    else:
        p = p.resolve()

    if not p.exists():
        raise FileNotFoundError(
            f"run root 不存在：{p}"
        )
    return p


def discover_serial_models(
    run_root: Path,
    wanted_train_seeds: Optional[Sequence[int]],
    eval_every: int,
    max_step: Optional[int],
) -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    wanted = (
        set(int(x) for x in wanted_train_seeds)
        if wanted_train_seeds is not None
        else None
    )

    for seed_dir in sorted(
        run_root.glob("seed_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    ):
        try:
            train_seed = int(seed_dir.name.split("_")[-1])
        except Exception:
            continue

        if wanted is not None and train_seed not in wanted:
            continue

        ckpt_dir = seed_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue

        for model_path in sorted(ckpt_dir.glob("model_*_steps.zip")):
            step = checkpoint_step(model_path)
            if step is None:
                continue

            if max_step is not None and step > max_step:
                continue

            if eval_every > 0 and step % eval_every != 0:
                continue

            vec_path = ckpt_dir / f"vecnormalize_{step:07d}_steps.pkl"

            if not vec_path.exists():
                raise FileNotFoundError(
                    f"模型存在但对应 VecNormalize 缺失：\n"
                    f"model={model_path}\n"
                    f"vec={vec_path}"
                )

            specs.append(
                ModelSpec(
                    source_group="serial",
                    train_seed=train_seed,
                    train_step=step,
                    model_path=model_path.resolve(),
                    vecnormalize_path=vec_path.resolve(),
                    label=f"seed{train_seed}_{step}",
                )
            )

    if not specs:
        raise RuntimeError(
            "没有找到满足条件的 serial checkpoints。"
        )

    return specs


def discover_official_reference(
    model_text: str,
    vec_text: str,
) -> Optional[ModelSpec]:
    model_path = Path(model_text).expanduser()
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()

    if not model_path.exists():
        warnings.warn(
            f"官方参考模型不存在，跳过：{model_path}"
        )
        return None

    vec_path = Path(vec_text).expanduser()
    if not vec_path.is_absolute():
        vec_path = (ROOT / vec_path).resolve()

    if not vec_path.exists():
        raise FileNotFoundError(
            f"官方参考模型 VecNormalize 不存在：{vec_path}"
        )

    return ModelSpec(
        source_group="official_original",
        train_seed=None,
        train_step=None,
        model_path=model_path,
        vecnormalize_path=vec_path,
        label="official_final",
    )


# =============================================================================
# 环境 + 评价
# =============================================================================

def make_eval_env(
    passenger_file: Path,
    max_time: int,
    candidates: Sequence[int],
    to_vertiport: int,
    vec_path: Optional[Path],
    monitor_dir: Path,
):
    def _build():
        base = make_env(
            max_time=max_time,
            log_dir=monitor_dir,
            env_index=0,
            candidate_from_vertiports=list(candidates),
            to_vertiport=to_vertiport,
            person_spawn_file=str(passenger_file),
            enable_logger=False,
        )()
        return TerminalPassengerCapture(base)

    raw = DummyVecEnv([_build])

    if vec_path is None:
        warnings.warn(
            "未提供 VecNormalize；如果模型训练时使用了 VecNormalize，"
            "评价结果无效。",
            RuntimeWarning,
        )
        return raw

    env = VecNormalize.load(
        str(vec_path),
        raw,
    )
    env.training = False
    env.norm_reward = False
    return env


def metrics_from_snapshot(
    snapshot: Dict[str, Any],
    actions: Counter,
    candidates: Sequence[int],
    total_reward: float,
    steps: int,
) -> Dict[str, Any]:
    rows = snapshot.get("rows", [])

    completed = [
        r
        for r in rows
        if bool(r.get("finished"))
        and np.isfinite(
            as_float(
                r.get("travel_time", np.nan)
            )
        )
    ]

    travel = np.asarray(
        [
            as_float(r.get("travel_time", np.nan))
            for r in completed
        ],
        dtype=float,
    )
    travel = travel[np.isfinite(travel)]

    def component(key: str) -> float:
        x = np.asarray(
            [
                as_float(r.get(key, np.nan))
                for r in completed
            ],
            dtype=float,
        )
        x = x[np.isfinite(x)]
        return float(x.mean()) if len(x) else float("nan")

    att = (
        float(travel.mean())
        if len(travel)
        else float("nan")
    )
    access = component("access_time")
    awt = component("wait_uam_time")
    aft = component("fly_time")

    residual = float("nan")
    if all(
        math.isfinite(x)
        for x in (att, access, awt, aft)
    ):
        residual = att - access - awt - aft

    n_persons = int(
        snapshot.get(
            "n_persons",
            len(rows),
        )
    )

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
        "N": n_persons,
        "N_finished": int(len(completed)),
        "completion_rate": (
            float(len(completed) / n_persons)
            if n_persons > 0
            else float("nan")
        ),
        "episode_reward": float(total_reward),
        "episode_steps": int(steps),
        "terminal_scenario_time": snapshot.get("time"),
        "n_policy_actions": int(sum(actions.values())),
    }

    denom = max(
        1,
        int(sum(actions.values())),
    )

    for action_idx, vp in enumerate(candidates):
        count = int(actions.get(action_idx, 0))
        out[f"action_{action_idx}_vp_{vp}_count"] = count
        out[f"action_{action_idx}_vp_{vp}_share"] = (
            count / denom
        )

    return out


def run_one_episode(
    spec: ModelSpec,
    passenger_file: Path,
    eval_seed: int,
    max_time: int,
    candidates: Sequence[int],
    to_vertiport: int,
    deterministic: bool,
    device: str,
    monitor_dir: Path,
) -> Dict[str, Any]:
    seed_all(eval_seed)

    env = make_eval_env(
        passenger_file=passenger_file,
        max_time=max_time,
        candidates=candidates,
        to_vertiport=to_vertiport,
        vec_path=spec.vecnormalize_path,
        monitor_dir=monitor_dir,
    )

    model = PPO.load(
        str(spec.model_path),
        env=env,
        device=device,
    )

    # VecEnv seeds are applied at reset.
    try:
        env.seed(eval_seed)
    except Exception:
        pass

    obs = env.reset()
    done = np.array([False], dtype=bool)

    actions: Counter = Counter()
    total_reward = 0.0
    n_steps = 0
    terminal_snapshot = None

    while not bool(done[0]):
        action, _ = model.predict(
            obs,
            deterministic=deterministic,
        )

        a = int(
            np.asarray(action)
            .reshape(-1)[0]
        )
        actions[a] += 1

        obs, reward, done, infos = env.step(action)

        total_reward += as_float(
            np.asarray(reward).reshape(-1)[0]
        )
        n_steps += 1

        if (
            infos
            and isinstance(infos[0], dict)
            and "uagmc_terminal_snapshot" in infos[0]
        ):
            terminal_snapshot = infos[0][
                "uagmc_terminal_snapshot"
            ]

        if n_steps > max_time + 100:
            env.close()
            raise RuntimeError(
                f"episode 超过预期 horizon："
                f"steps={n_steps}, max_time={max_time}"
            )

    if terminal_snapshot is None:
        env.close()
        raise RuntimeError(
            "终止时未捕获 passenger snapshot。"
            "请检查 UAGMC wrapper/API 是否变化。"
        )

    metrics = metrics_from_snapshot(
        terminal_snapshot,
        actions=actions,
        candidates=candidates,
        total_reward=total_reward,
        steps=n_steps,
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
            "vecnormalize_path": (
                str(spec.vecnormalize_path)
                if spec.vecnormalize_path
                else ""
            ),
            "eval_seed": int(eval_seed),
            "passenger_file": str(passenger_file),
            "deterministic": bool(deterministic),
        }
    )

    env.close()
    return metrics


# =============================================================================
# 汇总
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
    "episode_reward",
    "episode_steps",
    "n_policy_actions",
]


def group_rows(
    rows: Sequence[Dict[str, Any]],
    key_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

    for row in rows:
        key = tuple(row.get(k) for k in key_fields)
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []

    action_share_cols = sorted(
        {
            k
            for row in rows
            for k in row.keys()
            if k.startswith("action_")
            and k.endswith("_share")
        }
    )

    for key, group in sorted(
        groups.items(),
        key=lambda kv: tuple(
            -1 if x is None else x
            for x in kv[0]
        ),
    ):
        summary = dict(
            zip(
                key_fields,
                key,
            )
        )
        summary["n_evals"] = len(group)

        for col in METRIC_COLUMNS + action_share_cols:
            vals = [
                as_float(r.get(col, np.nan))
                for r in group
            ]
            summary[f"{col}_mean"] = finite_mean(vals)
            summary[f"{col}_std"] = finite_std(vals)

        out.append(summary)

    return out


def make_seed_curve(
    episode_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    serial = [
        r
        for r in episode_rows
        if r["source_group"] == "serial"
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
    seed_curve: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    by_step: Dict[int, List[Dict[str, Any]]] = {}

    for row in seed_curve:
        step = int(row["train_step"])
        by_step.setdefault(
            step,
            [],
        ).append(row)

    out: List[Dict[str, Any]] = []

    # 这里跨 seed 统计的是先对 eval seeds 求均值后的 train-seed means，
    # 避免把一个 seed 的多个 eval episodes 当成独立训练 seed。
    metric_mean_cols = sorted(
        {
            k
            for row in seed_curve
            for k in row.keys()
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
            vals = [
                as_float(x.get(col, np.nan))
                for x in group
            ]
            base = col[:-5]  # remove "_mean"
            row[f"{base}_seed_mean"] = finite_mean(vals)
            row[f"{base}_seed_std"] = finite_std(vals)

        out.append(row)

    return out


def make_official_summary(
    episode_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    official = [
        r
        for r in episode_rows
        if r["source_group"] == "official_original"
    ]

    if not official:
        return []

    return group_rows(
        official,
        key_fields=(
            "source_group",
            "model_label",
        ),
    )


# =============================================================================
# 绘图
# =============================================================================

def make_curve_plot(
    across_curve: List[Dict[str, Any]],
    official_summary: List[Dict[str, Any]],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if not across_curve:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"[PLOT] matplotlib unavailable: {exc}"
        )
        return

    x = np.asarray(
        [r["train_step"] for r in across_curve],
        dtype=float,
    )
    y = np.asarray(
        [
            as_float(
                r.get(f"{metric}_seed_mean", np.nan)
            )
            for r in across_curve
        ],
        dtype=float,
    )
    e = np.asarray(
        [
            as_float(
                r.get(f"{metric}_seed_std", np.nan)
            )
            for r in across_curve
        ],
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
        label="Serial checkpoints: mean ± SD across train seeds",
    )

    if official_summary:
        ref = as_float(
            official_summary[0].get(
                f"{metric}_mean",
                np.nan,
            )
        )
        if math.isfinite(ref):
            ax.axhline(
                ref,
                linestyle="--",
                label="Official original final model",
            )

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"UAGMC learning curve: {metric}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


# =============================================================================
# 报告
# =============================================================================

def make_text_report(
    out_dir: Path,
    run_root: Path,
    passenger_file: Path,
    eval_seeds: Sequence[int],
    seed_curve: List[Dict[str, Any]],
    across_curve: List[Dict[str, Any]],
    official_summary: List[Dict[str, Any]],
) -> None:
    lines: List[str] = [
        "=" * 100,
        "UAGMC CHECKPOINT LEARNING-CURVE EVALUATION",
        "=" * 100,
        "",
        f"run root       : {run_root}",
        f"passenger file : {passenger_file}",
        f"eval seeds     : {list(eval_seeds)}",
        "",
        "Metric semantics",
        "-" * 100,
        "ATT        = mean(end_time - start_time) over completed passengers",
        "AGT_access = time_stats['to_vertiport'] (origin -> chosen departure vertiport)",
        "AWT        = time_stats['wait_uam']",
        "AFT        = time_stats['fly']",
        "UAGMC source evaluation has no final egress-ground component here.",
        "",
    ]

    if official_summary:
        ref = official_summary[0]
        lines += [
            "Official original reference",
            "-" * 100,
            (
                f"ATT={as_float(ref.get('ATT_mean')):.4f} | "
                f"Access={as_float(ref.get('AGT_access_mean')):.4f} | "
                f"AWT={as_float(ref.get('AWT_mean')):.4f} | "
                f"AFT={as_float(ref.get('AFT_mean')):.4f} | "
                f"completion={100*as_float(ref.get('completion_rate_mean')):.2f}%"
            ),
            "",
        ]

    if across_curve:
        lines += [
            "Across-train-seed curve",
            "-" * 100,
        ]

        for row in across_curve:
            lines.append(
                f"{int(row['train_step']):>8,d} | "
                f"ATT={as_float(row.get('ATT_seed_mean')):.4f}"
                f"±{as_float(row.get('ATT_seed_std')):.4f} | "
                f"Access={as_float(row.get('AGT_access_seed_mean')):.4f} | "
                f"AWT={as_float(row.get('AWT_seed_mean')):.4f} | "
                f"AFT={as_float(row.get('AFT_seed_mean')):.4f} | "
                f"completion={100*as_float(row.get('completion_rate_seed_mean')):.2f}% | "
                f"n_seeds={int(row['n_train_seeds'])}"
            )

    lines += [
        "",
        "Important",
        "-" * 100,
        "1. Compare checkpoints only when completion rate is comparable/high.",
        "2. Use ATT as the main control-performance curve; AWT/Access/AFT diagnose where the change comes from.",
        "3. Official original model is a horizontal reference, not assigned an artificial training step.",
        "4. These curves evaluate the UAGMC source completion-time definition; they are not the later full door-to-door metric with final egress.",
        "",
    ]

    (out_dir / "summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# CLI / 主流程
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate serial UAGMC checkpoints + original final model "
            "and build ATT/AWT/access/flight learning curves."
        )
    )

    p.add_argument(
        "--run-root",
        required=True,
        help=(
            "Serial experiment root，例如 "
            "serial_runs/uagmc_gpu_10m_20260918_003515"
        ),
    )

    p.add_argument(
        "--train-seeds",
        default=None,
        help="只评估指定 train seeds，例如 0,2,4；默认全部。",
    )

    p.add_argument(
        "--eval-every",
        type=int,
        default=100_000,
        help="checkpoint 间隔筛选；默认每100k。设50000评估全部50k checkpoint。",
    )

    p.add_argument(
        "--max-step",
        type=int,
        default=2_000_000,
    )

    p.add_argument(
        "--eval-seeds",
        default="123,124,125",
        help="固定评价随机种子，默认123,124,125。",
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
        "--official-model",
        default="models/final_rl_model.zip",
    )

    p.add_argument(
        "--official-vecnorm",
        default="models/final_vec_normalize.pkl",
    )

    p.add_argument(
        "--skip-official",
        action="store_true",
    )

    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="只做推理；默认CPU。",
    )

    p.add_argument(
        "--stochastic",
        action="store_true",
        help="默认 deterministic；加此开关则用 stochastic policy。",
    )

    p.add_argument(
        "--output-dir",
        default=None,
        help="默认 <run-root>/learning_curve_eval",
    )

    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="任何 checkpoint 评价失败立即终止；默认记录error后继续。",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    run_root = resolve_run_root(args.run_root)

    passenger_file = Path(args.passenger_file).expanduser()
    if not passenger_file.is_absolute():
        passenger_file = (ROOT / passenger_file).resolve()
    else:
        passenger_file = passenger_file.resolve()

    if not passenger_file.exists():
        raise FileNotFoundError(
            f"passenger trace 不存在：{passenger_file}"
        )

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else run_root / "learning_curve_eval"
    )
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    monitor_dir = output_dir / "_monitor"
    monitor_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_seeds = (
        parse_int_list(args.train_seeds)
        if args.train_seeds
        else None
    )
    eval_seeds = parse_int_list(
        args.eval_seeds
    )
    candidates = parse_int_list(
        args.candidates
    )
    device = choose_device(
        args.device
    )

    serial_specs = discover_serial_models(
        run_root=run_root,
        wanted_train_seeds=train_seeds,
        eval_every=int(args.eval_every),
        max_step=int(args.max_step),
    )

    specs = list(serial_specs)

    if not args.skip_official:
        official = discover_official_reference(
            args.official_model,
            args.official_vecnorm,
        )
        if official is not None:
            specs.append(official)

    manifest = {
        "run_root": str(run_root),
        "passenger_file": str(passenger_file),
        "eval_seeds": eval_seeds,
        "train_seeds": train_seeds,
        "eval_every": int(args.eval_every),
        "max_step": int(args.max_step),
        "candidates": candidates,
        "to_vertiport": int(args.to_vertiport),
        "max_time": int(args.max_time),
        "deterministic": not bool(args.stochastic),
        "device": device,
        "n_serial_models": len(serial_specs),
        "official_included": any(
            s.source_group == "official_original"
            for s in specs
        ),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    (output_dir / "eval_manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 118)
    print("UAGMC SERIAL CHECKPOINT LEARNING-CURVE EVALUATION")
    print("=" * 118)
    print(f"Run root        : {run_root}")
    print(f"Passenger trace : {passenger_file}")
    print(f"Serial models   : {len(serial_specs)}")
    print(f"Eval seeds      : {eval_seeds}")
    print(f"Eval every      : {int(args.eval_every):,}")
    print(f"Policy mode     : {'STOCHASTIC' if args.stochastic else 'DETERMINISTIC'}")
    print(f"Device          : {device}")
    print(f"Output          : {output_dir}")
    print("=" * 118)

    episode_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    total_jobs = len(specs) * len(eval_seeds)
    job = 0

    for spec in specs:
        for eval_seed in eval_seeds:
            job += 1

            step_text = (
                f"{spec.train_step:,}"
                if spec.train_step is not None
                else "OFFICIAL"
            )

            print(
                f"[{job:>4}/{total_jobs}] "
                f"{spec.label:<24} "
                f"step={step_text:<10} "
                f"eval_seed={eval_seed}",
                flush=True,
            )

            try:
                row = run_one_episode(
                    spec=spec,
                    passenger_file=passenger_file,
                    eval_seed=eval_seed,
                    max_time=int(args.max_time),
                    candidates=candidates,
                    to_vertiport=int(args.to_vertiport),
                    deterministic=not bool(args.stochastic),
                    device=device,
                    monitor_dir=monitor_dir,
                )
                episode_rows.append(row)

                print(
                    "      "
                    f"ATT={row['ATT']:.4f} | "
                    f"Access={row['AGT_access']:.4f} | "
                    f"AWT={row['AWT']:.4f} | "
                    f"AFT={row['AFT']:.4f} | "
                    f"finished={row['N_finished']}/{row['N']}",
                    flush=True,
                )

            except Exception as exc:
                err = {
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
                    "eval_seed": eval_seed,
                    "error": repr(exc),
                }
                error_rows.append(err)

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
        error_rows,
    )

    seed_curve = make_seed_curve(
        episode_rows
    )
    across_curve = make_across_seed_curve(
        seed_curve
    )
    official_summary = make_official_summary(
        episode_rows
    )

    write_csv(
        output_dir / "curve_by_train_seed.csv",
        seed_curve,
    )
    write_csv(
        output_dir / "curve_across_train_seeds.csv",
        across_curve,
    )
    write_csv(
        output_dir / "official_reference.csv",
        official_summary,
    )

    make_curve_plot(
        across_curve,
        official_summary,
        metric="ATT",
        ylabel="ATT (min)",
        output_path=output_dir / "att_curve.png",
    )
    make_curve_plot(
        across_curve,
        official_summary,
        metric="AWT",
        ylabel="AWT (min)",
        output_path=output_dir / "awt_curve.png",
    )
    make_curve_plot(
        across_curve,
        official_summary,
        metric="AGT_access",
        ylabel="Access time (min)",
        output_path=output_dir / "access_curve.png",
    )
    make_curve_plot(
        across_curve,
        official_summary,
        metric="AFT",
        ylabel="Flight time (min)",
        output_path=output_dir / "aft_curve.png",
    )
    make_curve_plot(
        across_curve,
        official_summary,
        metric="completion_rate",
        ylabel="Completion rate",
        output_path=output_dir / "completion_curve.png",
    )

    make_text_report(
        out_dir=output_dir,
        run_root=run_root,
        passenger_file=passenger_file,
        eval_seeds=eval_seeds,
        seed_curve=seed_curve,
        across_curve=across_curve,
        official_summary=official_summary,
    )

    print("\n" + "=" * 118)
    print("DONE")
    print("=" * 118)
    print(f"Episode metrics : {output_dir / 'episode_metrics.csv'}")
    print(f"Per-seed curve  : {output_dir / 'curve_by_train_seed.csv'}")
    print(f"Cross-seed curve: {output_dir / 'curve_across_train_seeds.csv'}")
    print(f"Official ref    : {output_dir / 'official_reference.csv'}")
    print(f"Report          : {output_dir / 'summary.txt'}")
    print(f"Errors          : {output_dir / 'errors.csv'}")
    print("=" * 118)

    return 0 if not error_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
