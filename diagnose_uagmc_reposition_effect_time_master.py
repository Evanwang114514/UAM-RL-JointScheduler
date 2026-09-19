# -*- coding: utf-8 -*-
"""
UAGMC reposition policies: full NO-TRAINING effect-time diagnostic
=================================================================

Purpose
-------
Reuse the EXISTING corrected master diagnostic:

    diagnose_uagmc_effect_time_master.py

without rewriting its metrics, and run the exact same diagnostic suite on:

    1) longest_queue
    2) vertisync_simple

for the N=16 conserved-fleet environment.

The aircraft reposition logic is NOT reimplemented here. This script imports:

    train_uagmc_reposition_lq_vs_vertisync_800k.py

and calls its exact:

    install_reposition_patch(method)

so the diagnostic environment uses the same LQ / VertiSync-simple rule that
was used during training.

"NO TRAINING" means:
    - no PPO update
    - no model.learn(...)
    - no parameter modification
    - only deterministic checkpoint rollout + offline diagnostics

By default each method is diagnosed at its final 800k passenger-PPO checkpoint.

The master diagnostic suite remains exactly the previous corrected suite:
    A. same-passenger candidate access/effect-time heterogeneity
    B. decision-time -> own-effect-time SAFE-state drift
    C. delay vs snapshot staleness
    D. shared-horizon -> candidate-own-horizon mismatch
    E. legal committed-event crossing
    F. access timer semantics audit
    G. offline oracle ranking flips
    H. per-variable change rates
    I. horizon-bin diagnostics
    J. engineering gate / READY_FOR_CONTROLLED_TRAINING

Requirements
------------
Put these files together in UAGMC-main:

    diagnose_uagmc_candidate_temporal_mismatch.py
    diagnose_uagmc_effect_time_master.py
    train_uagmc_reposition_lq_vs_vertisync_800k.py
    diagnose_uagmc_reposition_effect_time_master.py   <-- this file

Typical usage
-------------
    cd /d "E:\\Study Files\\github\\UAM-predict\\UAGMC-main"
    conda activate uam5070
    python diagnose_uagmc_reposition_effect_time_master.py

It auto-detects the newest:
    serial_runs/uagmc_reposition_LQ_vs_VertiSync_N16_seed1_800k_*

Default:
    methods      = longest_queue,vertisync_simple
    checkpoint   = 800000
    passengers   = train_data/passengers_300.csv
    max_time     = 600
    device       = cpu

Optional multiple checkpoints:
    python diagnose_uagmc_reposition_effect_time_master.py ^
      --steps 200000,400000,600000,800000

Outputs
-------
<run-root>/effect_time_master_compare/
    comparison_manifest.json
    method_summary.csv
    pairwise_comparison.json
    summary.txt
    longest_queue/
        step_0800000/
            <timestamp>/
                summary.json
                candidate_detail.csv
                decision_summary.csv
                horizon_bins.csv
                variable_change_summary.csv
                committed_crossing_summary.csv
                timer_semantics_audit.csv
                resolved_config.json
    vertisync_simple/
        ...
    UPLOAD_THIS_effect_time_diagnostic.zip

Important interpretation
------------------------
The future/oracle state remains OFFLINE DIAGNOSTIC ONLY.
It is not legal online policy input.

A PASS means that the temporal mismatch mechanism exists strongly enough to
justify controlled method experiments. It does NOT prove an ATT improvement.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MASTER_FILE = ROOT / "diagnose_uagmc_effect_time_master.py"
STAGE0_FILE = ROOT / "diagnose_uagmc_candidate_temporal_mismatch.py"
TRAIN_FILE = ROOT / "train_uagmc_reposition_lq_vs_vertisync_800k.py"
PASSENGER_DEFAULT = ROOT / "train_data" / "passengers_300.csv"

VALID_METHODS = ("longest_queue", "vertisync_simple")
DEFAULT_STEPS = [800_000]


# =============================================================================
# Generic utilities
# =============================================================================

def jsonable(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


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
            out = {}
            for k, v in row.items():
                if isinstance(v, (dict, list, tuple)):
                    out[k] = json.dumps(v, ensure_ascii=False)
                else:
                    out[k] = v
            writer.writerow(out)


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("integer list cannot be empty")
    return vals


def parse_methods(text: str) -> List[str]:
    vals = [x.strip().lower() for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("method list cannot be empty")

    bad = [x for x in vals if x not in VALID_METHODS]
    if bad:
        raise ValueError(
            f"Unknown method(s): {bad}. Valid methods: {VALID_METHODS}"
        )
    return vals


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def fmt(x: Any, digits: int = 4) -> str:
    v = as_float(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


# =============================================================================
# Training run / checkpoint discovery
# =============================================================================

def auto_find_run_root() -> Path:
    serial = ROOT / "serial_runs"

    patterns = [
        "uagmc_reposition_LQ_vs_VertiSync_N16_seed*_800k_*",
        "*reposition*LQ*VertiSync*N16*800k*",
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
            "Cannot auto-detect the LQ-vs-VertiSync training run under "
            "serial_runs/. Use --run-root explicitly."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def discover_checkpoint(
    run_root: Path,
    method: str,
    step: int,
) -> Tuple[Path, Path]:
    ckpt = run_root / method / "checkpoints"

    model = ckpt / f"uam_ppo_{step}_steps.zip"
    vec = ckpt / f"uam_ppo_vecnormalize_{step}_steps.pkl"

    if not model.exists():
        raise FileNotFoundError(
            f"Missing model for {method}@{step}: {model}"
        )

    if not vec.exists():
        raise FileNotFoundError(
            f"Missing VecNormalize for {method}@{step}: {vec}"
        )

    return model.resolve(), vec.resolve()


# =============================================================================
# Exact environment adapter
# =============================================================================

class MakeEnvPatch:
    """
    Temporarily redirect diagnose_uagmc_effect_time_master.py's import:

        from utilss.make_env import make_env

    to the N=16 make_env_fleet factory.

    The reposition destination rule itself is installed by the exact training
    file before this patch is used.
    """

    def __init__(self):
        self.legacy_module = None
        self.old_make_env = None

    def install(self) -> None:
        legacy_module = importlib.import_module("utilss.make_env")
        fleet_module = importlib.import_module("utilss.make_env_fleet")

        self.legacy_module = legacy_module
        self.old_make_env = legacy_module.make_env
        fleet_make_env = fleet_module.make_env

        def fixed16_make_env(*args, **kwargs):
            kwargs = dict(kwargs)
            kwargs.update(
                fleet_mode="conserved_closed_loop",
                fleet_size=16,
                fleet_assertions=True,
            )
            return fleet_make_env(*args, **kwargs)

        legacy_module.make_env = fixed16_make_env

    def restore(self) -> None:
        if self.legacy_module is not None and self.old_make_env is not None:
            self.legacy_module.make_env = self.old_make_env


# =============================================================================
# Run the ORIGINAL master diagnostic once
# =============================================================================

def newest_summary_dir(base: Path, before: set[Path]) -> Path:
    after = {
        p.resolve()
        for p in base.iterdir()
        if p.is_dir() and (p / "summary.json").exists()
    } if base.exists() else set()

    new_dirs = sorted(
        after - before,
        key=lambda p: p.stat().st_mtime,
    )

    if new_dirs:
        return new_dirs[-1]

    # Fallback: master may have reused second-level timestamp only in unusual
    # cases. Pick newest valid result.
    all_dirs = sorted(
        [
            p.resolve()
            for p in base.iterdir()
            if p.is_dir() and (p / "summary.json").exists()
        ],
        key=lambda p: p.stat().st_mtime,
    ) if base.exists() else []

    if not all_dirs:
        raise RuntimeError(
            f"Master diagnostic finished but no summary.json found under {base}"
        )

    return all_dirs[-1]


def run_master_once(
    *,
    method: str,
    step: int,
    model_path: Path,
    vec_path: Path,
    passenger_file: Path,
    max_time: int,
    device: str,
    run_root: Path,
) -> Path:
    # Import exact aircraft-reposition implementation.
    trainmod = importlib.import_module(
        "train_uagmc_reposition_lq_vs_vertisync_800k"
    )

    # IMPORTANT: same function used during training.
    patch_info = trainmod.install_reposition_patch(method)

    master = importlib.import_module(
        "diagnose_uagmc_effect_time_master"
    )

    relative_base = (
        Path("serial_runs")
        / run_root.name
        / "effect_time_master_compare"
        / method
        / f"step_{step:07d}"
    )

    absolute_base = ROOT / relative_base
    absolute_base.mkdir(parents=True, exist_ok=True)

    before = {
        p.resolve()
        for p in absolute_base.iterdir()
        if p.is_dir() and (p / "summary.json").exists()
    }

    env_patch = MakeEnvPatch()
    old_argv = list(sys.argv)

    try:
        env_patch.install()

        sys.argv = [
            str(MASTER_FILE),
            "--project-root",
            str(ROOT),
            "--uagmc-root",
            str(ROOT),
            "--model",
            str(model_path),
            "--vecnorm",
            str(vec_path),
            "--passengers",
            str(passenger_file),
            "--candidates",
            "0,1",
            "--to-vertiport",
            "2",
            "--max-time",
            str(max_time),
            "--device",
            device,
            "--output-dir",
            relative_base.as_posix(),
        ]

        print("\n" + "#" * 128)
        print(
            f"MASTER NO-TRAINING DIAGNOSTIC | "
            f"method={method} | checkpoint={step:,}"
        )
        print("#" * 128)
        print(f"model       : {model_path}")
        print(f"vecnormalize: {vec_path}")
        print(f"env         : conserved_closed_loop, N=16")
        print(f"patch       : {patch_info}")
        print("#" * 128)

        master.main()

    finally:
        sys.argv = old_argv
        env_patch.restore()

    return newest_summary_dir(absolute_base, before)


# =============================================================================
# Summary extraction
# =============================================================================

def get_nested(data: Dict[str, Any], *keys: str, default=float("nan")):
    obj: Any = data
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def flatten_master_summary(
    *,
    method: str,
    step: int,
    result_dir: Path,
) -> Dict[str, Any]:
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "method": method,
        "train_step": step,
        "result_dir": str(result_dir),
        "summary_json": str(summary_path),
        "recorded_decisions": summary.get("recorded_decisions"),
        "complete_candidate_rows": summary.get("complete_candidate_rows"),
        "complete_decisions": summary.get("complete_decisions"),

        # A. Same passenger / candidate delay heterogeneity
        "mean_delay_spread": get_nested(
            summary,
            "same_passenger_candidate_delay",
            "mean_spread",
        ),
        "p90_delay_spread": get_nested(
            summary,
            "same_passenger_candidate_delay",
            "p90_spread",
        ),
        "fraction_spread_ge_2": get_nested(
            summary,
            "same_passenger_candidate_delay",
            "fraction_spread_ge_2",
        ),
        "fraction_spread_ge_5": get_nested(
            summary,
            "same_passenger_candidate_delay",
            "fraction_spread_ge_5",
        ),

        # B. Decision-time -> own effect-time
        "mean_safe_drift_now_to_own": get_nested(
            summary,
            "oracle_effect_time_staleness",
            "mean_safe_drift_now_to_own",
        ),
        "safe_change_rate": get_nested(
            summary,
            "oracle_effect_time_staleness",
            "safe_any_change_rate",
        ),
        "mean_corrected_drift_now_to_own": get_nested(
            summary,
            "oracle_effect_time_staleness",
            "mean_corrected_drift_now_to_own",
        ),
        "corrected_change_rate": get_nested(
            summary,
            "oracle_effect_time_staleness",
            "corrected_any_change_rate",
        ),
        "rho_access_safe_drift": get_nested(
            summary,
            "oracle_effect_time_staleness",
            "rho_access_vs_safe_drift",
        ),
        "p_access_safe_drift": get_nested(
            summary,
            "oracle_effect_time_staleness",
            "p_access_vs_safe_drift",
        ),
        "long_short_drift_ratio": get_nested(
            summary,
            "oracle_effect_time_staleness",
            "long_vs_short_drift_ratio",
        ),
        "rho_delay_spread_mean_safe_drift": get_nested(
            summary,
            "oracle_effect_time_staleness",
            "rho_same_passenger_delay_spread_vs_mean_safe_drift",
        ),

        # C. Shared future -> own candidate-specific future
        "mean_safe_drift_shared_to_own": get_nested(
            summary,
            "candidate_specific_vs_shared_future_reference",
            "mean_safe_drift_shared_to_own",
        ),
        "shared_horizon_mismatch_rate": get_nested(
            summary,
            "candidate_specific_vs_shared_future_reference",
            "shared_horizon_safe_mismatch_rate",
        ),
        "known_boundary_diff_rate": get_nested(
            summary,
            "candidate_specific_vs_shared_future_reference",
            "decision_boundary_diff_rate_known_committed",
        ),

        # D. Legal committed-event evidence
        "mean_committed_events_now": get_nested(
            summary,
            "legal_committed_event_evidence",
            "mean_committed_access_events_now",
        ),
        "mean_cross_before_own": get_nested(
            summary,
            "legal_committed_event_evidence",
            "mean_events_crossing_before_own_effect",
        ),
        "committed_crossing_rate": get_nested(
            summary,
            "legal_committed_event_evidence",
            "positive_crossing_rate",
        ),
        "own_shared_cross_diff_rate": get_nested(
            summary,
            "legal_committed_event_evidence",
            "own_vs_shared_cross_count_diff_rate",
        ),

        # E. Timer audit
        "timer_transitions": get_nested(
            summary,
            "timer_semantics_audit",
            "n_transitions",
        ),
        "timer_fraction_exact_minus_1": get_nested(
            summary,
            "timer_semantics_audit",
            "fraction_exact_minus_1",
        ),

        # F. Offline oracle ranking relevance
        "waiting_rank_flip_now_to_own": get_nested(
            summary,
            "decision_relevance_offline_oracle",
            "now_to_own_waiting_rank_flip_rate",
        ),
        "burden_rank_flip_now_to_own": get_nested(
            summary,
            "decision_relevance_offline_oracle",
            "now_to_own_corrected_burden_rank_flip_rate",
        ),
        "pressure_rank_flip_now_to_own": get_nested(
            summary,
            "decision_relevance_offline_oracle",
            "now_to_own_corrected_pressure_rank_flip_rate",
        ),
        "waiting_rank_flip_shared_to_own": get_nested(
            summary,
            "decision_relevance_offline_oracle",
            "shared_to_own_waiting_rank_flip_rate",
        ),

        # Gate
        "gate_status": get_nested(
            summary,
            "gate",
            "status",
            default="UNKNOWN",
        ),
        "gate_pass": get_nested(
            summary,
            "gate",
            "overall_pass",
            default=False,
        ),
    }


def build_pairwise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    by_key = {
        (str(r["method"]), int(r["train_step"])): r
        for r in rows
    }

    out: Dict[str, Any] = {
        "interpretation": (
            "Positive delta means VertiSync-simple metric is numerically larger "
            "than Longest Queue. This is descriptive, not automatically better/worse."
        ),
        "steps": {},
    }

    compare_fields = [
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
    ]

    steps = sorted({int(r["train_step"]) for r in rows})

    for step in steps:
        lq = by_key.get(("longest_queue", step))
        sy = by_key.get(("vertisync_simple", step))

        if lq is None or sy is None:
            continue

        d: Dict[str, Any] = {
            "longest_queue_result": lq["result_dir"],
            "vertisync_simple_result": sy["result_dir"],
        }

        for field in compare_fields:
            lv = as_float(lq.get(field))
            sv = as_float(sy.get(field))
            d[field] = {
                "longest_queue": lv,
                "vertisync_simple": sv,
                "sync_minus_lq": (
                    sv - lv
                    if math.isfinite(lv) and math.isfinite(sv)
                    else float("nan")
                ),
            }

        out["steps"][str(step)] = d

    return out


# =============================================================================
# Human-readable report
# =============================================================================

def write_report(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    errors: Sequence[Dict[str, Any]],
) -> None:
    lines = [
        "=" * 124,
        "UAGMC N=16 | LONGEST QUEUE vs VERTISYNC-SIMPLE",
        "FULL EFFECT-TIME MASTER DIAGNOSTIC | NO TRAINING",
        "=" * 124,
        "",
        "Same corrected diagnostic as the previous UAGMC-source experiment.",
        "Only the aircraft empty-reposition rule/environment trajectory differs.",
        "",
    ]

    for row in rows:
        lines += [
            "-" * 124,
            f"{row['method']} @ {int(row['train_step']):,}",
            "-" * 124,
            f"decisions                    : {row.get('recorded_decisions')}",
            f"mean candidate delay spread  : {fmt(row.get('mean_delay_spread'), 6)}",
            f"mean SAFE drift now->own     : {fmt(row.get('mean_safe_drift_now_to_own'), 6)}",
            f"SAFE any-change rate         : {fmt(row.get('safe_change_rate'), 4)}",
            f"rho(access, SAFE drift)      : {fmt(row.get('rho_access_safe_drift'), 6)}",
            f"long/short drift ratio       : {fmt(row.get('long_short_drift_ratio'), 4)}",
            f"mean SAFE drift shared->own  : {fmt(row.get('mean_safe_drift_shared_to_own'), 6)}",
            f"shared-horizon mismatch      : {fmt(row.get('shared_horizon_mismatch_rate'), 4)}",
            f"known boundary diff rate     : {fmt(row.get('known_boundary_diff_rate'), 4)}",
            f"committed crossing rate      : {fmt(row.get('committed_crossing_rate'), 4)}",
            f"own/shared crossing diff     : {fmt(row.get('own_shared_cross_diff_rate'), 4)}",
            f"timer exact -1 fraction      : {fmt(row.get('timer_fraction_exact_minus_1'), 4)}",
            f"waiting rank flip now->own   : {fmt(row.get('waiting_rank_flip_now_to_own'), 4)}",
            f"burden rank flip now->own    : {fmt(row.get('burden_rank_flip_now_to_own'), 4)}",
            f"pressure rank flip now->own  : {fmt(row.get('pressure_rank_flip_now_to_own'), 4)}",
            f"gate                         : {row.get('gate_status')}",
            f"result dir                   : {row.get('result_dir')}",
            "",
        ]

    lines += [
        "=" * 124,
        "INTERPRETATION GUARDRAILS",
        "=" * 124,
        "1. This script does not train either model.",
        "2. Future/oracle state is offline diagnostic evidence only.",
        "3. Legal committed-event evidence uses only already-committed access passengers.",
        "4. A gate PASS proves neither ATT superiority nor counterfactual optimality.",
        "5. Compare LQ vs Sync descriptively: different aircraft rules induce different realized state trajectories.",
        "",
        f"errors: {len(errors)}",
    ]

    if errors:
        for e in errors:
            lines.append(
                f"  - {e.get('method')}@{e.get('step')}: {e.get('error')}"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Bundle
# =============================================================================

def build_zip(output_root: Path) -> Path:
    zip_path = output_root / "UPLOAD_THIS_effect_time_diagnostic.zip"

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        for path in sorted(output_root.rglob("*")):
            if not path.is_file():
                continue
            if path == zip_path:
                continue
            if path.suffix.lower() not in {
                ".csv",
                ".json",
                ".txt",
                ".png",
            }:
                continue
            z.write(
                path,
                path.relative_to(output_root).as_posix(),
            )

    return zip_path


# =============================================================================
# CLI / main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run the existing corrected UAGMC effect-time master diagnostic "
            "on Longest Queue and VertiSync-simple N=16 environments."
        )
    )

    p.add_argument(
        "--run-root",
        default=None,
        help="Default: newest LQ-vs-VertiSync N16 800k run.",
    )
    p.add_argument(
        "--methods",
        default="longest_queue,vertisync_simple",
    )
    p.add_argument(
        "--steps",
        default="800000",
        help="Default 800000. Example: 200000,400000,600000,800000",
    )
    p.add_argument(
        "--passengers",
        default=str(PASSENGER_DEFAULT),
    )
    p.add_argument(
        "--max-time",
        type=int,
        default=600,
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Fail loudly if any dependency is missing.
    for required in (
        MASTER_FILE,
        STAGE0_FILE,
        TRAIN_FILE,
    ):
        if not required.exists():
            raise FileNotFoundError(
                f"Required file is missing: {required}"
            )

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

    passenger_file = Path(args.passengers).expanduser()
    if not passenger_file.is_absolute():
        passenger_file = (ROOT / passenger_file).resolve()
    else:
        passenger_file = passenger_file.resolve()

    if not passenger_file.exists():
        raise FileNotFoundError(passenger_file)

    methods = parse_methods(args.methods)
    steps = sorted(set(parse_int_list(args.steps)))

    output_root = run_root / "effect_time_master_compare"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": (
            "Reuse the corrected UAGMC candidate-specific effect-time master "
            "diagnostic for two aircraft reposition policies."
        ),
        "run_root": str(run_root),
        "methods": methods,
        "steps": steps,
        "passenger_file": str(passenger_file),
        "max_time": int(args.max_time),
        "device": args.device,
        "environment": {
            "fleet_mode": "conserved_closed_loop",
            "fleet_size": 16,
            "initial_allocation_at_reset": {"0": 12, "1": 4},
            "candidates": [0, 1],
            "destination": 2,
        },
        "diagnostic_source": str(MASTER_FILE),
        "reposition_logic_source": str(TRAIN_FILE),
        "same_master_metrics_as_original_source_diagnostic": True,
        "no_training": True,
    }
    write_json(
        output_root / "comparison_manifest.json",
        manifest,
    )

    print("=" * 128)
    print("UAGMC REPOSITION EFFECT-TIME MASTER DIAGNOSTIC | NO TRAINING")
    print("=" * 128)
    print(f"Run root   : {run_root}")
    print(f"Methods    : {methods}")
    print(f"Steps      : {steps}")
    print(f"Passengers : {passenger_file}")
    print(f"Max time   : {args.max_time}")
    print(f"Device     : {args.device}")
    print("=" * 128)

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for step in steps:
        for method in methods:
            try:
                model, vec = discover_checkpoint(
                    run_root,
                    method,
                    step,
                )

                result_dir = run_master_once(
                    method=method,
                    step=step,
                    model_path=model,
                    vec_path=vec,
                    passenger_file=passenger_file,
                    max_time=int(args.max_time),
                    device=args.device,
                    run_root=run_root,
                )

                row = flatten_master_summary(
                    method=method,
                    step=step,
                    result_dir=result_dir,
                )
                rows.append(row)

            except Exception as exc:
                error = {
                    "method": method,
                    "step": step,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(error)

                print(
                    f"\n[ERROR] {method}@{step:,}: {repr(exc)}",
                    flush=True,
                )

                if not args.continue_on_error:
                    write_csv(
                        output_root / "errors.csv",
                        errors,
                    )
                    raise

    rows.sort(
        key=lambda r: (
            int(r["train_step"]),
            str(r["method"]),
        )
    )

    write_csv(
        output_root / "method_summary.csv",
        rows,
    )
    write_csv(
        output_root / "errors.csv",
        errors,
    )

    pairwise = build_pairwise(rows)
    write_json(
        output_root / "pairwise_comparison.json",
        pairwise,
    )

    write_report(
        output_root / "summary.txt",
        rows,
        errors,
    )

    bundle = build_zip(output_root)

    print("\n" + "=" * 128)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 128)

    for row in rows:
        print(
            f"{row['method']:<18} @ {int(row['train_step']):>8,d} | "
            f"safe_change={fmt(row.get('safe_change_rate'), 4)} | "
            f"rho={fmt(row.get('rho_access_safe_drift'), 4)} | "
            f"shared_mismatch={fmt(row.get('shared_horizon_mismatch_rate'), 4)} | "
            f"crossing={fmt(row.get('committed_crossing_rate'), 4)} | "
            f"waiting_flip={fmt(row.get('waiting_rank_flip_now_to_own'), 4)} | "
            f"gate={row.get('gate_status')}"
        )

    print("-" * 128)
    print(f"Summary CSV : {output_root / 'method_summary.csv'}")
    print(f"Pairwise    : {output_root / 'pairwise_comparison.json'}")
    print(f"Report      : {output_root / 'summary.txt'}")
    print(f"Errors      : {output_root / 'errors.csv'}")
    print("-" * 128)
    print("UPLOAD THIS FILE TO CHATGPT:")
    print(bundle)
    print("=" * 128)

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
