# -*- coding: utf-8 -*-
"""
UAM Single 69.6M Priority Matrix
================================

目标
----
在固定 S3 单侧物理、responsive Longest-Queue 飞机调度、同一 PPO 配置下，
按科学优先级依次完成：

P0  基线与历史缺口修复        4.8M
P1  五族核心机制矩阵         26.4M
P2  多 train-seed 强确认       7.2M
P3  2M 统一长时域训练         24.0M
P4  AI 热点轻量探针            7.2M
总计                         69.6M requested PPO transitions

重要原则
--------
1. 只使用 S3 Single：passenger 由 PPO 决策，aircraft 保持原 responsive LQ。
2. reward / demand / fleet / topology / charging / pad / turnaround 均不改。
3. 严禁读取 unrevealed future passenger；所有 temporal/event 特征仅来自当前
   observation 中已经 committed / 可由物理确定的量。
4. 所有 literature 名称均为 mechanism/style inspired，不是原论文完整复现。
5. P3 的 2M 长训全部从 0 开始，使用完整 2M linear-LR horizon；绝不机械
   resume 600k checkpoint。
6. 默认等待现有 100.2M next-gen 实验产生 RUN_COMPLETE.json 且对应 Python
   进程退出后，才启动本文件，从而保证两套大训练绝不并发。

默认服务器用法
--------------
CUDA_VISIBLE_DEVICES=0 python -u train_uam_single70m_priority.py \
    --output-root serial_runs/uam_single70m_priority_20260928 \
    --wait-for-nextgen-root serial_runs/uam_nextgen100m_formal_20260927

仅查看计划
----------
python train_uam_single70m_priority.py --plan-only

本脚本依赖同目录下已经验证的：
- train_uam_60m_literature_matrix_v3_1.py
- train_uam_60m_jointfirst_v4_deferred_joint_eval.py
- train_uam_60m_jointfirst_v5_minimal_reposition.py
- train_uam_s3_20x600k_litupgrade.py
- train_uam_nextgen_100m_matrix.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import train_uam_60m_literature_matrix_v3_1 as v3
import train_uam_60m_jointfirst_v4_deferred_joint_eval as v4
import train_uam_60m_jointfirst_v5_minimal_reposition as v5
import train_uam_s3_20x600k_litupgrade as up
import train_uam_nextgen_100m_matrix as ng


# =============================================================================
# 冻结实验常量
# =============================================================================

ROOT = Path(__file__).resolve().parent
ENV_KEY = "S3"
PROFILE = ng.P40
SHORT_STEPS = 600_000
LONG_STEPS = 2_000_000
CHECKPOINT_INTERVAL = 50_000
EVAL_SEEDS = (123, 124, 125)
MAX_TIME = 2_500

# P0：最优先修复历史证据缺口。
P0_RUNS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("UAGMC_SOURCE", (2, 3, 4, 5)),
    ("S_EVENTGRAPH_GAT", (2, 3)),
    ("CURRENT", (4, 5)),
)

# P1：20 个五族核心方法 + 2 个 WM constituent controls。
FAMILY_METHODS: Dict[str, Tuple[str, ...]] = {
    "S": (
        "S_EVENTGRAPH_GAT",
        "S_GATV2_EVENT",
        "S_TGAT_EVENT",
        "S_TDM_EVENT_FUSION",
    ),
    "AC": (
        "A_ICM_AC",
        "A_FACMAC_MIXER",
        "A_QPLEX_DUPLEX",
        "A_ANCHOR_QUOTIENT",
    ),
    "AC+WM": (
        "AW_ICM_VE",
        "AW_ICM_VALUECTRL",
        "AW_FACMAC_VE",
        "AW_FACMAC_VALUECTRL",
    ),
    "S+AC": (
        "SA_TGAT_FACMAC",
        "SA_GRAPHMIXER_QPLEX",
        "SA_GATV2_QPLEX",
        "SA_TDMFUSION_ANCHORQ",
    ),
    "S+WM": (
        "SW_TDMFUSION_VE",
        "SW_TDMFUSION_VALUECTRL",
        "SW_GATV2_VE",
        "SW_GATV2_VALUECTRL",
    ),
}
P1_CORE_METHODS: Tuple[str, ...] = tuple(
    m for family in ("S", "AC", "AC+WM", "S+AC", "S+WM")
    for m in FAMILY_METHODS[family]
)
P1_WM_CONTROLS: Tuple[str, ...] = (
    "W_EVENTQ_VE_EMA",
    "W_VALUE_CONTROLLABLE",
)
P1_METHODS: Tuple[str, ...] = P1_CORE_METHODS + P1_WM_CONTROLS
P1_SEEDS = (4, 5)

# P2：由 P1 自动选出五族 winner + 一个全局 runner-up。
P2_SEEDS = (6, 7)
P2_N_METHODS = 6

# P3：三个固定 under-training / positive-interaction 候选，
# 再加 P1 自动得到的 AC、AC+WM、S+WM 三个族 winner。
P3_FIXED_METHODS: Tuple[str, ...] = (
    "W_DREAMER_BALANCED",
    "S_GATV2_EVENT",
    "SA_TGAT_FACMAC",
)
P3_SEEDS = (8, 9)

# P4：最后才跑的热点探针。
P4_METHODS: Tuple[str, ...] = (
    "A_CCT_LITE",
    "SA_CCT_LITE",
    "S_SMDP_DURATION",
    "S_ASYNC_HORIZON_PHASE",
    "S_SUCCESSOR_RESOURCE",
    "SW_SUCCESSOR_VE",
)
P4_SEEDS = (10, 11)

# 总 requested PPO transitions = 69.6M。
P0_BUDGET = sum(len(seeds) for _, seeds in P0_RUNS) * SHORT_STEPS
P1_BUDGET = len(P1_METHODS) * len(P1_SEEDS) * SHORT_STEPS
P2_BUDGET = P2_N_METHODS * len(P2_SEEDS) * SHORT_STEPS
P3_BUDGET = 6 * len(P3_SEEDS) * LONG_STEPS
P4_BUDGET = len(P4_METHODS) * len(P4_SEEDS) * SHORT_STEPS
TOTAL_BUDGET = P0_BUDGET + P1_BUDGET + P2_BUDGET + P3_BUDGET + P4_BUDGET
assert TOTAL_BUDGET == 69_600_000, TOTAL_BUDGET


# =============================================================================
# 新增组合：尽量做“减法实验”，避免再次堆成 model zoo
# =============================================================================

NEW_COMBOS_70M: Dict[str, Tuple[str, ...]] = {
    # AC + WM：同一 AC，比较 VE-only 与 VE+controllability WM。
    "AW_ICM_VE": (
        "A_ICM_AC",
        "W_EVENTQ_VE_EMA",
    ),
    "AW_FACMAC_VE": (
        "A_FACMAC_MIXER",
        "W_EVENTQ_VE_EMA",
    ),

    # S + AC：专门检验 temporal/action reference-frame compatibility。
    "SA_GATV2_QPLEX": (
        "S_GATV2_EVENT",
        "A_QPLEX_DUPLEX",
    ),
    "SA_TDMFUSION_ANCHORQ": (
        "S_TDM_EVENT_FUSION",
        "A_ANCHOR_QUOTIENT",
    ),

    # S + WM：2x2 中的 VE-only 两格；ValueCtrl 两格沿用 next-gen。
    "SW_TDMFUSION_VE": (
        "S_TDM_EVENT_FUSION",
        "W_EVENTQ_VE_EMA",
    ),
    "SW_GATV2_VE": (
        "S_GATV2_EVENT",
        "W_EVENTQ_VE_EMA",
    ),

    # 热点组合。
    "SA_CCT_LITE": (
        "S_TDM_EVENT_FUSION",
        "A_CCT_LITE",
    ),
    "SW_SUCCESSOR_VE": (
        "S_SUCCESSOR_RESOURCE",
        "W_EVENTQ_VE_EMA",
    ),
}


METHOD_META_70M: Dict[str, Dict[str, str]] = {
    "UAGMC_SOURCE": {
        "family": "BASELINE",
        "role": "source 6-frame temporal observation only",
        "literature": "UAGMC/source-style 6-frame TemporalLSTM baseline",
    },
    "AW_ICM_VE": {
        "family": "AC+WM",
        "role": "ICM controllability + value-equivalent future only",
        "literature": "ICM + Value Equivalence inspired controlled interaction",
    },
    "AW_FACMAC_VE": {
        "family": "AC+WM",
        "role": "factorized utility + value-equivalent future only",
        "literature": "FACMAC-style utility + Value Equivalence inspired",
    },
    "SA_GATV2_QPLEX": {
        "family": "S+AC",
        "role": "candidate-relative temporal relation + centered candidate advantage",
        "literature": "GATv2-style + QPLEX-style controlled interaction",
    },
    "SA_TDMFUSION_ANCHORQ": {
        "family": "S+AC",
        "role": "effect-time common/event context + absolute-anchor action residual",
        "literature": "project TDM/event fusion + quotient/anchor residual",
    },
    "SW_TDMFUSION_VE": {
        "family": "S+WM",
        "role": "effect-time temporal/event context + value-equivalent future",
        "literature": "project temporal alignment + Value Equivalence inspired",
    },
    "SW_GATV2_VE": {
        "family": "S+WM",
        "role": "dynamic candidate relation + value-equivalent future",
        "literature": "GATv2-style + Value Equivalence inspired",
    },
    "A_CCT_LITE": {
        "family": "AC-HOTSPOT",
        "role": "candidate effect-state residual against a shared committed-future anchor",
        "literature": "causal/counterfactual intervention inspired; project CCT-lite",
    },
    "SA_CCT_LITE": {
        "family": "S+AC-HOTSPOT",
        "role": "aligned TDM/event context + CCT-lite candidate residual",
        "literature": "project CCT-lite controlled interaction",
    },
    "S_SMDP_DURATION": {
        "family": "S-HOTSPOT",
        "role": "explicit action-duration/effect-time features",
        "literature": "SMDP / temporal abstraction inspired representation",
    },
    "S_ASYNC_HORIZON_PHASE": {
        "family": "S-HOTSPOT",
        "role": "multi-horizon future timing normalized by each candidate effect duration",
        "literature": "asynchronous decision / relative horizon-phase inspired",
    },
    "S_SUCCESSOR_RESOURCE": {
        "family": "S-HOTSPOT",
        "role": "discounted committed-future resource occupancy features",
        "literature": "successor-feature / predictive representation inspired",
    },
    "SW_SUCCESSOR_VE": {
        "family": "S+WM-HOTSPOT",
        "role": "successor-like committed resources + value-equivalent future",
        "literature": "predictive representation + Value Equivalence inspired",
    },
}


# =============================================================================
# 工具函数
# =============================================================================

def canonical(x: Any) -> str:
    return str(x).strip().upper()


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "1.0"}


def fmean(xs: Iterable[Any]) -> float:
    vals = []
    for x in xs:
        try:
            y = float(x)
        except Exception:
            continue
        if math.isfinite(y):
            vals.append(y)
    return float(np.mean(vals)) if vals else float("nan")


def fstd(xs: Iterable[Any]) -> float:
    vals = []
    for x in xs:
        try:
            y = float(x)
        except Exception:
            continue
        if math.isfinite(y):
            vals.append(y)
    return float(np.std(vals)) if vals else float("nan")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    v3.write_csv(path, [dict(r) for r in rows])


def parse_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# =============================================================================
# 服务器互斥 / 等待逻辑
# =============================================================================

def _python_pids_running_script(script_name: str) -> List[int]:
    """Linux /proc 下只识别真实 Python 进程，避免把 tmux/bash 命令误判为训练。"""
    proc = Path("/proc")
    if not proc.exists():
        return []

    out: List[int] = []
    me = os.getpid()
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid == me:
            continue
        try:
            raw = (p / "cmdline").read_bytes()
            parts = [
                x.decode("utf-8", errors="ignore")
                for x in raw.split(b"\x00")
                if x
            ]
        except Exception:
            continue
        if not parts:
            continue
        exe = Path(parts[0]).name.lower()
        if "python" not in exe:
            continue
        if any(Path(arg).name == script_name for arg in parts[1:]):
            out.append(pid)
    return sorted(out)


def ensure_no_duplicate_70m() -> None:
    name = Path(__file__).name
    pids = _python_pids_running_script(name)
    if pids:
        raise RuntimeError(
            f"检测到另一个 {name} 正在运行，PID={pids}；为避免双开，本次退出。"
        )


def wait_for_nextgen_complete(
    target_root: Path,
    *,
    poll_seconds: int,
) -> None:
    """必须同时满足完成标记存在 + 100M Python 进程退出，才允许启动 70M。"""
    target_root = target_root.expanduser().resolve()
    marker = target_root / "RUN_COMPLETE.json"

    print("=" * 120, flush=True)
    print("WAIT GATE | 等待现有 100.2M 完整结束后再启动 Single 69.6M", flush=True)
    print(f"nextgen root : {target_root}", flush=True)
    print(f"marker       : {marker}", flush=True)
    print("=" * 120, flush=True)

    while True:
        obj = parse_json_if_exists(marker)
        marker_ok = bool(obj and str(obj.get("status", "")).upper() == "COMPLETE")
        pids = _python_pids_running_script("train_uam_nextgen_100m_matrix.py")

        if marker_ok and not pids:
            print(
                f"[WAIT GATE PASS] {datetime.now().isoformat(timespec='seconds')} | "
                "100.2M 已完整结束，且训练 Python 进程已退出。",
                flush=True,
            )
            return

        print(
            f"[WAIT] {datetime.now().isoformat(timespec='seconds')} | "
            f"RUN_COMPLETE={marker_ok} | nextgen_pids={pids or 'none'} | "
            f"{poll_seconds}s 后复查",
            flush=True,
        )
        time.sleep(max(30, int(poll_seconds)))


def _physical_gpu_compute_pids(gpu_index: int) -> List[int]:
    """
    返回指定物理 GPU 上的 compute PID。
    查询失败时抛异常，避免在多人服务器上“看不清资源却自动开训”。
    """
    try:
        gpu_rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        index_to_uuid: Dict[int, str] = {}
        for line in gpu_rows.splitlines():
            if not line.strip():
                continue
            left, right = [x.strip() for x in line.split(",", 1)]
            index_to_uuid[int(left)] = right
        if int(gpu_index) not in index_to_uuid:
            raise RuntimeError(
                f"nvidia-smi 中不存在 physical GPU index={gpu_index}"
            )
        target_uuid = index_to_uuid[int(gpu_index)]

        app_rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        pids: List[int] = []
        for line in app_rows.splitlines():
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2 or parts[0] != target_uuid:
                continue
            try:
                pids.append(int(parts[1]))
            except Exception:
                pass
        return sorted(set(pids))
    except Exception as exc:
        raise RuntimeError(
            f"无法可靠查询 GPU 占用，拒绝自动启动 formal training: {exc}"
        ) from exc


def wait_for_gpu_idle(
    gpu_index: int,
    *,
    poll_seconds: int,
) -> None:
    """
    100M 完成后再等待指定物理 GPU 没有任何 compute process。
    这样即使两轮之间有其他用户抢占该卡，也不会自动撞上去。
    """
    print("=" * 120, flush=True)
    print(
        f"GPU GATE | 等待 physical GPU {gpu_index} 完全空闲后再启动 69.6M",
        flush=True,
    )
    print("=" * 120, flush=True)

    while True:
        pids = _physical_gpu_compute_pids(int(gpu_index))
        if not pids:
            print(
                f"[GPU GATE PASS] {datetime.now().isoformat(timespec='seconds')} | "
                f"GPU {gpu_index} 无 compute process。",
                flush=True,
            )
            return

        print(
            f"[GPU WAIT] {datetime.now().isoformat(timespec='seconds')} | "
            f"GPU {gpu_index} compute_pids={pids} | {poll_seconds}s 后复查",
            flush=True,
        )
        time.sleep(max(30, int(poll_seconds)))


# =============================================================================
# 70M 扩展 extractor
# =============================================================================

class SourceUAGMCExtractor(BaseFeaturesExtractor):
    """
    UAGMC_SOURCE 专用 extractor。

    输入环境仍是同一个 S3 LiteratureObservationWrapper，但这里只读取最前面的
    base slice；该 slice 就是原始 UAGMC wrapper 给出的 6-frame source observation。
    forward 结构对齐 utilss.encoding.TemporalLSTMExtractor：
        Linear(frame) -> LSTM(last) -> Linear(output)
    因此不会把 resource/tau/event/future projection 偷带进 source baseline。
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        layout: Optional[Dict[str, Any]] = None,
        method_id: str = "UAGMC_SOURCE",
        env_key: str = "S3",
    ):
        if layout is None:
            raise ValueError("layout required")
        super().__init__(observation_space, features_dim)
        self.slices = dict(layout["slices"])
        self.frame_encoder = nn.Linear(v3.SINGLE_FRAME_DIM, int(features_dim))
        self.lstm = nn.LSTM(
            input_size=int(features_dim),
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.output_proj = nn.Linear(128, int(features_dim))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        lo, hi = self.slices["base"]
        base = observations.float()[:, int(lo):int(hi)]
        frames = base.reshape(
            base.shape[0],
            v3.NUM_FRAMES,
            v3.SINGLE_FRAME_DIM,
        )
        x = self.frame_encoder(frames)
        x, _ = self.lstm(x)
        return self.output_proj(x[:, -1, :])


class Single70Extractor(ng.NextGenExtractor):
    """
    NextGenExtractor + 1 个 source-UAGMC baseline + 4 个轻量热点表示。

    这些新增模块对所有方法都实例化，因此不会因为“是否选中某个卡片”而
    偷偷改变可用参数模块集合。真正 forward 时仍由 method/component 决定。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        K = v3.N_CANDIDATES
        P = v3.PROJECT_PER_CAND

        # ------------------------------------------------------------------
        # A_CCT_LITE：
        # 不做“完整 CCT”或隐藏未来 rollout。
        # srac = candidate-own-effect projection - shared projection；
        # tdm 则是在 observation 构造阶段已经加入 focal passenger +1 后的
        # effect-time queue/supply/pressure。这里组合两者形成局部 intervention
        # residual，不在归一化后的张量上手工加 raw +1。
        # ------------------------------------------------------------------
        self.cct_lite = nn.Sequential(
            nn.Linear(K * (P + v3.TDM_FEATURES_PER_CAND), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # S_SMDP_DURATION：
        # 不改 PPO 为真正 SMDP optimizer，只显式编码 action/effect duration。
        # 每 candidate = TDM 5维 + tau/60 + log(1+tau) + exp(-tau/60)。
        # ------------------------------------------------------------------
        self.smdp_duration = nn.Sequential(
            nn.Linear(K * (v3.TDM_FEATURES_PER_CAND + 3), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # S_ASYNC_HORIZON_PHASE：
        # 不直接使用经 VecNormalize 处理后的 event ETA 去和 raw tau 做单位运算。
        # 改为在固定 raw horizon {0,5,15,30,60} 与 raw tau 之间计算相对 phase，
        # 再用该 phase 对 gamma committed-future projections 做 effect-centered
        # attention/gating。这样保持单位严格一致。
        # ------------------------------------------------------------------
        self.async_horizon_enc = nn.Sequential(
            nn.Linear(P, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
        )
        self.async_horizon_score = nn.Sequential(
            nn.Linear(35, 48),  # 32-d state + phase/abs-phase/pre-post
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.async_horizon_out = nn.Sequential(
            nn.Linear(32 * K, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # S_SUCCESSOR_RESOURCE：
        # 用现有 gamma multi-horizon committed projection 构造 successor-like
        # resource features；不预测 unrevealed demand，也不声称完整 Successor
        # Representation。每 candidate 同时保留 discounted occupancy 与相对 h=0
        # 的 discounted change。
        # ------------------------------------------------------------------
        self.successor_resource = nn.Sequential(
            nn.Linear(K * 2 * P, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def _cct_lite_latent(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        K = v3.N_CANDIDATES
        P = v3.PROJECT_PER_CAND

        srac = self._slice(obs, "srac").reshape(B, K, P)
        tdm = self._slice(obs, "tdm").reshape(
            B, K, v3.TDM_FEATURES_PER_CAND
        )

        # srac 提供 candidate-own effect 相对 shared anchor 的变化；
        # tdm 提供已经包含 focal passenger +1 的 post-action effect state。
        x = torch.cat([srac, tdm], dim=-1).flatten(1)
        return self.cct_lite(x)

    def _smdp_duration_latent(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        K = v3.N_CANDIDATES

        tdm = self._slice(obs, "tdm").reshape(
            B, K, v3.TDM_FEATURES_PER_CAND
        )
        tau = self._slice(obs, "tau").reshape(B, K).clamp_min(0.0)
        duration = torch.stack(
            [
                tau / 60.0,
                torch.log1p(tau) / math.log(61.0),
                torch.exp(-tau / 60.0),
            ],
            dim=-1,
        )
        return self.smdp_duration(
            torch.cat([tdm, duration], dim=-1).flatten(1)
        )

    def _async_horizon_phase_latent(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        K = v3.N_CANDIDATES
        H = len(v3.GAMMA_HORIZONS)
        P = v3.PROJECT_PER_CAND

        g = self._slice(obs, "gamma").reshape(B, K, H, P)
        tau = self._slice(obs, "tau").reshape(B, K).clamp_min(1e-3)
        horizons = torch.as_tensor(
            list(v3.GAMMA_HORIZONS),
            dtype=obs.dtype,
            device=obs.device,
        ).view(1, 1, H)

        phase = (
            horizons - tau.unsqueeze(-1)
        ) / (tau.unsqueeze(-1) + 1e-3)
        pre_post = torch.tanh(phase)

        z = self.async_horizon_enc(g)
        feat = torch.cat(
            [
                z,
                phase.unsqueeze(-1),
                phase.abs().unsqueeze(-1),
                pre_post.unsqueeze(-1),
            ],
            dim=-1,
        )
        score = self.async_horizon_score(feat).squeeze(-1)
        weight = torch.softmax(score, dim=-1)
        pooled = (z * weight.unsqueeze(-1)).sum(dim=2)
        return self.async_horizon_out(pooled.flatten(1))

    def _successor_resource_latent(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        K = v3.N_CANDIDATES
        H = len(v3.GAMMA_HORIZONS)
        P = v3.PROJECT_PER_CAND

        g = self._slice(obs, "gamma").reshape(B, K, H, P)
        horizons = torch.as_tensor(
            list(v3.GAMMA_HORIZONS),
            dtype=obs.dtype,
            device=obs.device,
        )

        # 每 5 分钟约 0.95 的 feature-discount，只作为 successor-like probe。
        weights = torch.pow(
            torch.full_like(horizons, 0.95),
            horizons / 5.0,
        )
        weights = weights / weights.sum().clamp_min(1e-6)

        occupancy = (g * weights.view(1, 1, H, 1)).sum(dim=2)
        delta = (
            (g - g[:, :, 0:1, :])
            * weights.view(1, 1, H, 1)
        ).sum(dim=2)

        return self.successor_resource(
            torch.cat([occupancy, delta], dim=-1).flatten(1)
        )

    def component_latent(
        self,
        obs: torch.Tensor,
        component: str,
    ) -> torch.Tensor:
        c = canonical(component)
        if c == "A_CCT_LITE":
            return self._cct_lite_latent(obs)
        if c == "S_SMDP_DURATION":
            return self._smdp_duration_latent(obs)
        if c == "S_ASYNC_HORIZON_PHASE":
            return self._async_horizon_phase_latent(obs)
        if c == "S_SUCCESSOR_RESOURCE":
            return self._successor_resource_latent(obs)
        return super().component_latent(obs, c)

# =============================================================================
# 注册 / 描述 / build_model
# =============================================================================

def install_70m_registry() -> None:
    # 先装 V5 / 20-cell / next-gen 的所有已经验证 hook 与 registry。
    ng.install_nextgen_hooks()

    # 再注册本轮新增组合。
    for name, comps in NEW_COMBOS_70M.items():
        v4.COMBO_REGISTRY[canonical(name)] = tuple(
            canonical(c) for c in comps
        )

    # 最后替换 extractor/build_model/description，Single 物理与 evaluator 不改。
    v3.build_model = build_model_70m
    v3.build_callbacks = v4.build_callbacks
    v3.method_description = method_description_70m


def components_for_70m(method_id: str) -> Tuple[str, ...]:
    m = canonical(method_id)
    if m == "UAGMC_SOURCE":
        return (m,)
    return tuple(v4.components_for(m))


def infer_family_70m(method_id: str) -> str:
    m = canonical(method_id)
    if m in METHOD_META_70M:
        return METHOD_META_70M[m]["family"]
    try:
        return ng.infer_family(m)
    except Exception:
        return v4.method_family(m)


def method_description_70m(method_id: str) -> Dict[str, Any]:
    m = canonical(method_id)
    meta = METHOD_META_70M.get(m)

    if meta is None:
        try:
            return ng.method_description(m)
        except Exception:
            meta = {
                "family": infer_family_70m(m),
                "role": "existing validated mechanism card",
                "literature": "existing project card",
            }

    return {
        "method_id": m,
        "family": meta["family"],
        "components": list(components_for_70m(m)),
        "role": meta["role"],
        "literature_or_role": meta["literature"],
        "style_inspired_not_verbatim": True,
    }


def build_model_70m(
    *,
    env,
    env_key: str,
    method_id: str,
    profile: v3.SpeedProfile,
    seed: int,
    run_dir: Path,
    device: str,
):
    if str(env_key).upper() != ENV_KEY:
        raise ValueError(
            f"本文件只允许 Single S3，收到 env_key={env_key}"
        )

    extractor_cls = (
        SourceUAGMCExtractor
        if canonical(method_id) == "UAGMC_SOURCE"
        else Single70Extractor
    )

    policy_kwargs: Dict[str, Any] = dict(
        features_extractor_class=extractor_cls,
        features_extractor_kwargs=dict(
            features_dim=128,
            layout=v3.GLOBAL_LAYOUT,
            method_id=method_id,
            env_key=env_key,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    m = canonical(method_id)
    if v4.custom_method_needs_aux(m):
        algo_cls = v4.DiscoveryAuxPPO
    elif m in v4.NATIVE_WM_METHODS:
        algo_cls = v3.LiteratureWorldModelPPO
    else:
        algo_cls = PPO

    return algo_cls(
        policy="MlpPolicy",
        env=env,
        learning_rate=v3.base.linear_schedule(v3.INITIAL_LR),
        n_steps=int(profile.n_steps),
        batch_size=int(profile.batch_size),
        n_epochs=v3.N_EPOCHS,
        gamma=v3.GAMMA,
        gae_lambda=v3.GAE_LAMBDA,
        clip_range=v3.CLIP_RANGE,
        ent_coef=v3.ENT_COEF,
        vf_coef=v3.VF_COEF,
        max_grad_norm=v3.MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        seed=int(seed),
        verbose=1,
        device=device,
        tensorboard_log=str(run_dir / "tb"),
    )


# =============================================================================
# 单 cell 生命周期
# =============================================================================

def run_cell(
    *,
    experiment_root: Path,
    phase: str,
    method: str,
    seed: int,
    timesteps: int,
    eval_seeds: Sequence[int],
    device: str,
    fail_fast: bool,
) -> Path:
    subroot = experiment_root / phase / f"seed{int(seed)}"
    subroot.mkdir(parents=True, exist_ok=True)

    v3.write_json(
        subroot / "seed_manifest.json",
        {
            "phase": phase,
            "env_key": ENV_KEY,
            "train_seed": int(seed),
            "method_id": canonical(method),
            "requested_timesteps": int(timesteps),
            "eval_seeds": list(eval_seeds),
            "profile": {
                "name": PROFILE.name,
                "n_envs": PROFILE.n_envs,
                "n_steps": PROFILE.n_steps,
                "batch_size": PROFILE.batch_size,
                "rollout": PROFILE.rollout,
            },
            "aircraft_control": "original responsive Longest-Queue",
        },
    )

    print(
        f"\n[{phase}] {ENV_KEY}__{canonical(method)} | "
        f"seed={seed} | steps={timesteps:,}",
        flush=True,
    )

    v3.safe_train_and_optional_eval(
        root=subroot,
        env_key=ENV_KEY,
        method_id=canonical(method),
        requested_steps=int(timesteps),
        train_seed=int(seed),
        eval_seeds=eval_seeds,
        profile=PROFILE,
        device=device,
        max_time=MAX_TIME,
        defer_eval=False,
        fail_fast=bool(fail_fast),
    )
    v3.build_master(subroot)
    return subroot / v3.cell_id(ENV_KEY, canonical(method))


# =============================================================================
# 结果读取与动态选择
# =============================================================================

def curve_metrics(run_dir: Path) -> Dict[str, float]:
    rows = v3.read_csv(run_dir / "analysis" / "checkpoint_curve.csv")
    pts: List[Tuple[int, float]] = []

    for r in rows:
        try:
            step = int(float(r.get("train_step", 0)))
            att = float(r.get("ATT_mean", "nan"))
        except Exception:
            continue
        if (
            truthy(r.get("all_full_completion"))
            and math.isfinite(att)
        ):
            pts.append((step, att))

    pts.sort(key=lambda x: x[0])
    if not pts:
        return {
            "best": float("nan"),
            "mean": float("nan"),
            "late3": float("nan"),
            "final": float("nan"),
            "collapse": float("nan"),
        }

    vals = np.asarray([x[1] for x in pts], dtype=float)
    return {
        "best": float(np.min(vals)),
        "mean": float(np.mean(vals)),
        "late3": float(np.mean(vals[-3:])),
        "final": float(vals[-1]),
        "collapse": float(vals[-1] - np.min(vals)),
    }


def method_seed_metrics(
    root: Path,
    *,
    phase: str,
    method: str,
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    out = []
    for seed in seeds:
        run_dir = (
            root
            / phase
            / f"seed{int(seed)}"
            / v3.cell_id(ENV_KEY, canonical(method))
        )
        met = curve_metrics(run_dir)
        out.append(
            {
                "method": canonical(method),
                "seed": int(seed),
                **met,
            }
        )
    return out


def robust_summary(
    root: Path,
    *,
    phase: str,
    method: str,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    rs = method_seed_metrics(
        root,
        phase=phase,
        method=method,
        seeds=seeds,
    )
    valid = [
        r for r in rs
        if finite(r.get("late3")) and finite(r.get("final"))
    ]
    if len(valid) != len(seeds):
        return {
            "method": canonical(method),
            "n_valid": len(valid),
            "late3_mean": float("inf"),
            "worst_final": float("inf"),
            "final_mean": float("inf"),
            "final_std": float("inf"),
            "best_mean": float("inf"),
            "collapse_mean": float("inf"),
        }

    finals = [float(r["final"]) for r in valid]
    return {
        "method": canonical(method),
        "n_valid": len(valid),
        "late3_mean": fmean(r["late3"] for r in valid),
        "worst_final": max(finals),
        "final_mean": fmean(finals),
        "final_std": fstd(finals),
        "best_mean": fmean(r["best"] for r in valid),
        "collapse_mean": fmean(r["collapse"] for r in valid),
    }


def robust_rank_key(rec: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    """
    预注册式选择顺序：
    1) Late3 mean
    2) worst-seed Final
    3) Final mean
    4) train-seed Final std
    Best 不参与主排名，只保留为 reachability 诊断。
    """
    return (
        float(rec["late3_mean"]),
        float(rec["worst_final"]),
        float(rec["final_mean"]),
        float(rec["final_std"]),
    )


def select_p2_methods(root: Path) -> Dict[str, Any]:
    family_winners: Dict[str, str] = {}
    all_rows: List[Dict[str, Any]] = []

    for family in ("S", "AC", "AC+WM", "S+AC", "S+WM"):
        rows = [
            {
                "family": family,
                **robust_summary(
                    root,
                    phase="P1",
                    method=m,
                    seeds=P1_SEEDS,
                ),
            }
            for m in FAMILY_METHODS[family]
        ]
        rows.sort(key=robust_rank_key)

        if not rows or not math.isfinite(float(rows[0]["late3_mean"])):
            raise RuntimeError(
                f"P1 的 {family} 族没有完整的 seed4/5 有效结果，禁止自动进入 P2。"
            )

        family_winners[family] = str(rows[0]["method"])
        all_rows.extend(rows)

    selected = set(family_winners.values())
    runner_candidates = [
        r for r in all_rows
        if str(r["method"]) not in selected
        and math.isfinite(float(r["late3_mean"]))
    ]
    runner_candidates.sort(key=robust_rank_key)
    if not runner_candidates:
        raise RuntimeError("无法选择 P2 全局 runner-up")

    runner_up = str(runner_candidates[0]["method"])
    methods = list(family_winners.values()) + [runner_up]
    if len(set(methods)) != 6:
        raise RuntimeError(f"P2 选择出现重复：{methods}")

    result = {
        "criterion": [
            "late3_mean",
            "worst_seed_final",
            "final_mean",
            "final_std",
        ],
        "best_is_reachability_only": True,
        "family_winners": family_winners,
        "runner_up": runner_up,
        "methods": methods,
        "all_p1_core_rows": sorted(
            all_rows,
            key=lambda r: (str(r["family"]),) + robust_rank_key(r),
        ),
    }
    write_json(root / "P2_selection.json", result)
    write_csv(root / "P1_robust_ranking.csv", result["all_p1_core_rows"])
    return result


def select_p3_methods(root: Path, p2_selection: Dict[str, Any]) -> Dict[str, Any]:
    winners = dict(p2_selection["family_winners"])

    dynamic = [
        str(winners["AC"]),
        str(winners["AC+WM"]),
        str(winners["S+WM"]),
    ]
    methods = list(P3_FIXED_METHODS) + dynamic

    if len(set(methods)) != 6:
        raise RuntimeError(
            f"P3 固定候选与动态 winner 重复，需人工检查：{methods}"
        )

    result = {
        "fixed": list(P3_FIXED_METHODS),
        "dynamic": {
            "AC": dynamic[0],
            "AC+WM": dynamic[1],
            "S+WM": dynamic[2],
        },
        "methods": methods,
        "reason": (
            "固定三个来自上一轮 under-training / positive-interaction 信号；"
            "其余三个来自 P1 五族矩阵的 AC、AC+WM、S+WM robust winner。"
        ),
        "long_run_rule": (
            "全部从0开始2M；SB3 linear LR 以完整2M horizon定义；"
            "绝不从600k checkpoint机械resume。"
        ),
    }
    write_json(root / "P3_selection.json", result)
    return result


# =============================================================================
# Phase runner
# =============================================================================

def mark_phase(
    root: Path,
    phase: str,
    status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    obj = {
        "phase": phase,
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        obj.update(extra)
    write_json(root / phase / f"{status}.json", obj)


def run_p0(
    root: Path,
    *,
    eval_seeds: Sequence[int],
    device: str,
    fail_fast: bool,
) -> None:
    mark_phase(root, "P0", "STARTED", {"budget": P0_BUDGET})
    for method, seeds in P0_RUNS:
        for seed in seeds:
            run_cell(
                experiment_root=root,
                phase="P0",
                method=method,
                seed=seed,
                timesteps=SHORT_STEPS,
                eval_seeds=eval_seeds,
                device=device,
                fail_fast=fail_fast,
            )
    mark_phase(root, "P0", "COMPLETE", {"budget": P0_BUDGET})


def run_p1(
    root: Path,
    *,
    eval_seeds: Sequence[int],
    device: str,
    fail_fast: bool,
) -> None:
    mark_phase(root, "P1", "STARTED", {"budget": P1_BUDGET})

    # 方法优先于 seed：高优先级方法先拿到完整 seed4/5 配对证据。
    for method in P1_METHODS:
        for seed in P1_SEEDS:
            run_cell(
                experiment_root=root,
                phase="P1",
                method=method,
                seed=seed,
                timesteps=SHORT_STEPS,
                eval_seeds=eval_seeds,
                device=device,
                fail_fast=fail_fast,
            )

    mark_phase(root, "P1", "COMPLETE", {"budget": P1_BUDGET})


def run_p2(
    root: Path,
    *,
    eval_seeds: Sequence[int],
    device: str,
    fail_fast: bool,
) -> Dict[str, Any]:
    selection = select_p2_methods(root)
    mark_phase(
        root,
        "P2",
        "STARTED",
        {"budget": P2_BUDGET, "selection": selection["methods"]},
    )

    for method in selection["methods"]:
        for seed in P2_SEEDS:
            run_cell(
                experiment_root=root,
                phase="P2",
                method=method,
                seed=seed,
                timesteps=SHORT_STEPS,
                eval_seeds=eval_seeds,
                device=device,
                fail_fast=fail_fast,
            )

    mark_phase(
        root,
        "P2",
        "COMPLETE",
        {"budget": P2_BUDGET, "selection": selection["methods"]},
    )
    return selection


def run_p3(
    root: Path,
    *,
    p2_selection: Dict[str, Any],
    eval_seeds: Sequence[int],
    device: str,
    fail_fast: bool,
) -> Dict[str, Any]:
    selection = select_p3_methods(root, p2_selection)
    mark_phase(
        root,
        "P3",
        "STARTED",
        {"budget": P3_BUDGET, "selection": selection["methods"]},
    )

    for method in selection["methods"]:
        for seed in P3_SEEDS:
            run_cell(
                experiment_root=root,
                phase="P3",
                method=method,
                seed=seed,
                timesteps=LONG_STEPS,
                eval_seeds=eval_seeds,
                device=device,
                fail_fast=fail_fast,
            )

    mark_phase(
        root,
        "P3",
        "COMPLETE",
        {"budget": P3_BUDGET, "selection": selection["methods"]},
    )
    return selection


def run_p4(
    root: Path,
    *,
    eval_seeds: Sequence[int],
    device: str,
    fail_fast: bool,
) -> None:
    mark_phase(root, "P4", "STARTED", {"budget": P4_BUDGET})

    for method in P4_METHODS:
        for seed in P4_SEEDS:
            run_cell(
                experiment_root=root,
                phase="P4",
                method=method,
                seed=seed,
                timesteps=SHORT_STEPS,
                eval_seeds=eval_seeds,
                device=device,
                fail_fast=fail_fast,
            )

    mark_phase(root, "P4", "COMPLETE", {"budget": P4_BUDGET})


# =============================================================================
# Plan / manifest
# =============================================================================

def budget_manifest() -> Dict[str, Any]:
    return {
        "experiment": "UAM_SINGLE_69P6M_PRIORITY_MATRIX",
        "physics": {
            "env": "S3",
            "topology": "T2",
            "fleet": 40,
            "turnaround_min": 1.0,
            "charging_scale": 1.25,
            "charger_capacity": 5,
            "pad_separation_min": 0.25,
            "aircraft_control": "responsive Longest-Queue",
            "reward": "r_t=-N_active(t)*dt",
        },
        "profile": {
            "name": PROFILE.name,
            "n_envs": PROFILE.n_envs,
            "n_steps": PROFILE.n_steps,
            "batch_size": PROFILE.batch_size,
            "rollout": PROFILE.rollout,
        },
        "eval_seeds": list(EVAL_SEEDS),
        "budget": {
            "P0": P0_BUDGET,
            "P1": P1_BUDGET,
            "P2": P2_BUDGET,
            "P3": P3_BUDGET,
            "P4": P4_BUDGET,
            "TOTAL": TOTAL_BUDGET,
        },
        "P0": {
            "purpose": "baseline/historical-gap repair",
            "runs": [
                {"method": m, "seeds": list(seeds), "steps": SHORT_STEPS}
                for m, seeds in P0_RUNS
            ],
        },
        "P1": {
            "purpose": "five-family mechanism matrix + WM constituent controls",
            "families": {
                k: list(v) for k, v in FAMILY_METHODS.items()
            },
            "wm_controls": list(P1_WM_CONTROLS),
            "seeds": list(P1_SEEDS),
            "steps": SHORT_STEPS,
        },
        "P2": {
            "purpose": "robustness confirmation",
            "selection": "five family winners + one global runner-up from P1",
            "seeds": list(P2_SEEDS),
            "steps": SHORT_STEPS,
        },
        "P3": {
            "purpose": "under-training vs bad representation",
            "fixed": list(P3_FIXED_METHODS),
            "dynamic": "P1 robust winners from AC / AC+WM / S+WM",
            "seeds": list(P3_SEEDS),
            "steps": LONG_STEPS,
            "from_scratch": True,
        },
        "P4": {
            "purpose": "lightweight AI-hotspot probes after core science closes",
            "methods": list(P4_METHODS),
            "seeds": list(P4_SEEDS),
            "steps": SHORT_STEPS,
        },
    }


def print_plan() -> None:
    m = budget_manifest()
    print("=" * 120)
    print("UAM SINGLE 69.6M PRIORITY MATRIX")
    print("=" * 120)
    for p in ("P0", "P1", "P2", "P3", "P4"):
        print(f"{p}: {m['budget'][p] / 1e6:.1f}M")
    print(f"TOTAL: {m['budget']['TOTAL'] / 1e6:.1f}M")
    print("\nP1 families:")
    for family, methods in FAMILY_METHODS.items():
        print(f"  {family:<6s}: {', '.join(methods)}")
    print(f"  WM ctrl: {', '.join(P1_WM_CONTROLS)}")
    print("\nP4 hotspot probes:")
    for method in P4_METHODS:
        print(f"  - {method}")
    print("=" * 120)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="UAM Single 69.6M priority-ordered mechanism experiment"
    )
    ap.add_argument(
        "--output-root",
        default="serial_runs/uam_single70m_priority_20260928",
    )
    ap.add_argument(
        "--wait-for-nextgen-root",
        default="serial_runs/uam_nextgen100m_formal_20260927",
        help=(
            "默认必须等该 100.2M root 的 RUN_COMPLETE.json=COMPLETE 且 "
            "train_uam_nextgen_100m_matrix.py 进程退出后才开始。"
        ),
    )
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument(
        "--physical-gpu",
        type=int,
        default=0,
        help="多人服务器安全门：100M结束后等待该物理GPU无compute进程。",
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    ap.add_argument(
        "--eval-seeds",
        default="123,124,125",
    )
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument(
        "--no-wait",
        action="store_true",
        help="仅人工确认旧训练已结束后才可使用；默认不要加。",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="默认 formal run fail-fast；只有明确需要时才允许失败后继续。",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    print_plan()
    if args.plan_only:
        return 0

    ensure_no_duplicate_70m()

    if not args.no_wait:
        wait_for_nextgen_complete(
            Path(args.wait_for_nextgen_root),
            poll_seconds=int(args.poll_seconds),
        )
    else:
        active = _python_pids_running_script(
            "train_uam_nextgen_100m_matrix.py"
        )
        if active:
            raise RuntimeError(
                f"--no-wait 但检测到 100M 仍在运行 PID={active}，拒绝启动。"
            )

    # 多人服务器资源门：只要指定 GPU 还有任何 compute process，就继续等待。
    if args.device != "cpu" and torch.cuda.is_available():
        wait_for_gpu_idle(
            int(args.physical_gpu),
            poll_seconds=int(args.poll_seconds),
        )

    install_70m_registry()

    eval_seeds = [
        int(x.strip())
        for x in str(args.eval_seeds).split(",")
        if x.strip()
    ]
    if not eval_seeds:
        raise ValueError("eval seeds 不能为空")

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    root = Path(args.output_root).expanduser()
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    else:
        root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    manifest = budget_manifest()
    manifest.update(
        {
            "created": datetime.now().isoformat(timespec="seconds"),
            "root": str(root),
            "device": device,
            "wait_for_nextgen_root": (
                None if args.no_wait else str(Path(args.wait_for_nextgen_root))
            ),
            "fail_fast": not bool(args.continue_on_error),
            "physical_gpu_gate": (
                None if args.device == "cpu" else int(args.physical_gpu)
            ),
        }
    )
    write_json(root / "experiment_manifest.json", manifest)

    fail_fast = not bool(args.continue_on_error)

    print("\n" + "#" * 120)
    print(f"START SINGLE 69.6M | root={root} | device={device}")
    print("#" * 120, flush=True)

    try:
        run_p0(
            root,
            eval_seeds=eval_seeds,
            device=device,
            fail_fast=fail_fast,
        )
        run_p1(
            root,
            eval_seeds=eval_seeds,
            device=device,
            fail_fast=fail_fast,
        )
        p2_selection = run_p2(
            root,
            eval_seeds=eval_seeds,
            device=device,
            fail_fast=fail_fast,
        )
        p3_selection = run_p3(
            root,
            p2_selection=p2_selection,
            eval_seeds=eval_seeds,
            device=device,
            fail_fast=fail_fast,
        )
        run_p4(
            root,
            eval_seeds=eval_seeds,
            device=device,
            fail_fast=fail_fast,
        )
    except Exception:
        write_json(
            root / "RUN_FAILED.json",
            {
                "status": "FAILED",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "traceback": traceback.format_exc(),
            },
        )
        raise

    write_json(
        root / "RUN_COMPLETE.json",
        {
            "status": "COMPLETE",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "total_requested_timesteps": TOTAL_BUDGET,
            "p2_selection": p2_selection,
            "p3_selection": p3_selection,
        },
    )

    print("\n" + "#" * 120)
    print("DONE | UAM SINGLE 69.6M PRIORITY MATRIX")
    print(f"root={root}")
    print("#" * 120, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
