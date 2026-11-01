"""只读复算一小时诊断与五千万训练结果；不加载模型、不启动训练。"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import re
import statistics

import numpy as np
import pandas as pd
from scipy import stats


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def csv(path, frame):
    frame.to_csv(path, index=False, encoding="utf-8-sig", mode="x")


def paired(values):
    """训练种子配对差为独立单位；小样本区间只作探索性描述。"""
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    half = float(stats.t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    null = [abs(float(np.mean(x * signs))) for signs in itertools.product([-1, 1], repeat=n)]
    p = sum(v >= abs(mean) - 1e-12 for v in null) / len(null)
    loo = [float(np.delete(x, i).mean()) for i in range(n)] if n > 1 else [mean]
    return dict(n=n, mean=mean, sd=sd, ci95_low=mean-half, ci95_high=mean+half,
                wins=int((x < 0).sum()), ties=int((x == 0).sum()), minimum=float(x.min()),
                maximum=float(x.max()), signflip_p_two_sided=p,
                loo_min=min(loo), loo_max=max(loo), per_seed=x.tolist())


def cohort(case):
    if case.startswith("PT_J_"):
        return "PT_J"
    if case.startswith("PT_S_"):
        return "PT_S"
    if case.startswith("SOURCE_"):
        return "SOURCE"
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.input_root.resolve(), args.out.resolve()
    if out.exists():
        raise RuntimeError("派生分析目录已存在；为保留结果请另选新目录")
    out.mkdir(parents=True)
    train = root / "train/uam_gap_factors_48run_50m_v1"
    diag = root / "diag/uam_system_gap_1h_20261101"
    models = pd.read_csv(train / "MODEL_RESULTS.csv")
    plan = read(train / "SUITE_PLAN.json")
    cells, episodes, curves, timing, initials = [], [], [], [], []
    verification = []
    for _, row in models.iterrows():
        folder = train / "cells" / row.id
        task = read(folder / "TASK.json")
        result = read(folder / "RESULT.json")
        end = read(folder / "TRAIN_END.json")
        manifest = read(folder / "MODEL_MANIFEST.json")
        initial = read(folder / "INITIAL_POLICY_SHA.json")
        initials.append(dict(id=row.id, k=row.k, seed=row.seed, control=row.control,
                             network=row.network, sha256=initial["sha256"]))
        meta = dict(id=row.id, k=row.k, seed=row.seed, control=row.control, arm=row.arm, network=row.network)
        for path in sorted((folder / "evaluation").glob("*.json")):
            item = read(path)
            for e in item["episodes"]:
                term = e["terminal_resource"]
                queues = [int(v) for v in e["final_station_queues"].values()]
                unfinished = e["generated"] - e["finished"]
                # 明确限定为截止时的可观察类别，不把它宣称为所有时刻的唯一死锁原因。
                seat_tail = (unfinished > 0 and sum(queues) == unfinished and max(queues, default=0) < 4
                             and sum(term.get("aircraft_states", {}).values()) == term.get("aircraft_states", {}).get("IDLE", 0))
                trace = e["decision_trace"]
                area = e["attribution_ledger"]["person_time_pre"]
                rpt = e["restricted_person_time_per_generated"]
                if abs(area / e["generated"] - rpt) > 1e-8:
                    raise AssertionError("人时账本与RPT不一致：" + row.id)
                if trace and abs(sum(a["unfinished_before"] for a in trace)-area) > 1e-8:
                    raise AssertionError("已保存的逐分钟记录与账本不一致：" + row.id)
                # 中间检查点未全部保存分钟轨迹，不能将缺失的时段分解写成零。
                first = (sum(a["unfinished_before"] for a in trace if a["time"] < 300) / e["generated"]
                         if trace else float("nan"))
                episodes.append(dict(meta, step=item["step"], trace=e["trace"], eval_seed=e["eval_seed"],
                                     rpt=rpt, completion=e["completion"], full=e["full"], att=e["ATT"],
                                     unfinished=unfinished, seat_tail=seat_tail,
                                     restricted_blocked=term["restricted_executor_blocked"],
                                     area_before_300=first, area_after_300=rpt-first,
                                     applied_return=e["attribution_ledger"]["applied_return"],
                                     person_time_pre=e["attribution_ledger"]["person_time_pre"],
                                     person_time_post=e["attribution_ledger"]["person_time_post"],
                                     mean_wait_completed=e["AWT_completed_only"],
                                     effective_return_choices=e["control_diagnostics"]["effective_return_choices"]))
        curve = pd.read_csv(folder / "LEARNING_CURVE.csv")
        for key, value in meta.items():
            curve[key] = value
        curves.append(curve)
        times = pd.read_csv(folder / "TIMING.csv")
        for key, value in meta.items():
            times[key] = value
        timing.append(times)
        cells.append(dict(meta, steps=end["steps"], train_seconds=end["wall_seconds"],
                          eval_seconds=result["eval_seconds"], parameters=manifest["parameters"],
                          ppo=json.dumps(manifest["ppo"], sort_keys=True),
                          script_hash=manifest["code_sha256"]))
    ep = pd.DataFrame(episodes)
    cf, tm, cell = pd.concat(curves), pd.concat(timing), pd.DataFrame(cells)
    # 先在每个模型内部汇总四条顺序，再跨独立训练种子比较。
    late = ep[(ep.step >= .8 * 1003520) & ep.trace.ne("original")]
    final = ep[(ep.step == 1003520) & ep.trace.ne("original")]
    recomputed = late.groupby("id").agg(late_rpt=("rpt", "mean"), late_completion=("completion", "mean"),
                    late_full_fraction=("full", "mean"), late_blocked_fraction=("restricted_blocked", "mean"))
    for name in recomputed.columns:
        error = (models.set_index("id")[name] - recomputed[name]).abs().max()
        verification.append(dict(check="重算" + name, max_absolute_error=error, passed=error < 1e-9))
        if error >= 1e-9:
            raise AssertionError(name)
    checks = pd.DataFrame(initials).groupby(["k", "seed", "control", "network"]).sha256.nunique()
    verification.append(dict(check="同种子同骨架初始参数一致", passed=bool((checks == 1).all())))
    if not (checks == 1).all():
        raise AssertionError("初始模型不一致")
    results = models.merge(final.groupby("id").agg(final_area_before_300=("area_before_300", "mean"),
                           final_area_after_300=("area_after_300", "mean")), on="id")
    results = results.merge(late.groupby("id").agg(seat_tail_fraction=("seat_tail", "mean")), on="id")
    summary = results.groupby(["k", "control", "arm"]).agg(n=("seed", "size"),
                         rpt=("late_rpt", "mean"), sd=("late_rpt", "std"),
                         completion=("late_completion", "mean"), full=("late_full_fraction", "mean"),
                         blocked=("late_blocked_fraction", "mean"), tail=("seat_tail_fraction", "mean"),
                         final_before300=("final_area_before_300", "mean"), final_after300=("final_area_after_300", "mean"),
                         rollback=("original_best_to_late", "mean")).reset_index()
    pe = pd.read_csv(train / "PAIRED_EFFECTS.csv")
    contrasts = []
    for (k, control, contrast), frame in pe.groupby(["k", "control", "contrast"]):
        contrasts.append(dict(k=k, control=control, contrast=contrast, **paired(frame.sort_values("seed").delta_rpt)))
    cs = pd.DataFrame(contrasts)
    # 所有K3的七项对比作为同一探索性族；校正不把小样本变成高功效。
    mask = cs.k.eq(3)
    order = cs[mask].sort_values("signflip_p_two_sided").index
    running = 0.
    for i, ix in enumerate(order):
        running = max(running, min(1., cs.loc[ix, "signflip_p_two_sided"] * (len(order)-i)))
        cs.loc[ix, "holm_p_family14"] = running
    sj = []
    for arm, part in results[results.k.eq(3)].groupby("arm"):
        pivot = part.pivot(index="seed", columns="control", values="late_rpt")
        sj.append(dict(arm=arm, contrast="Joint减Single", **paired(pivot.J3-pivot.LQ)))
    rules = []
    for path in (train / "cells").glob("RULE*/RULES_END.json"):
        data = read(path)
        for item in data["results"]:
            e = item["metrics"]
            rules.append(dict(k=data["task"]["k"], trace=item["trace"], rule=item["rule"],
                              rpt=e["restricted_person_time_per_generated"], full=e["full"], completion=e["completion"]))
    rules = pd.DataFrame(rules)
    rule_summary = rules[rules.trace.ne("original")].groupby(["k", "rule"]).agg(rpt=("rpt", "mean"),
                    full=("full", "mean"), completion=("completion", "mean")).reset_index()
    references = rule_summary[rule_summary.rule.eq("READY_QUEUE_SINGLE")].set_index("k").rpt
    summary["gap_ready"] = summary.rpt - summary.k.map(references)
    perf = cell.groupby(["control", "network"]).agg(n=("id", "size"), steps=("steps", "sum"),
                     train_seconds=("train_seconds", "sum"), eval_seconds=("eval_seconds", "sum")).reset_index()
    perf["weighted_train_fps"] = perf.steps / perf.train_seconds
    perf["train_eval_fps"] = perf.steps / (perf.train_seconds+perf.eval_seconds)
    # 采样包含策略推理、进程通信和物理仿真，不冒称纯物理时间。
    tp = tm.groupby(["control", "network"])[["sampling_seconds", "update_seconds", "prepare_rpc_seconds", "finish_rpc_seconds", "reset_rpc_seconds", "resume_save_seconds"]].sum().reset_index()
    tp["update_share_sample_plus_update"] = tp.update_seconds / (tp.update_seconds+tp.sampling_seconds)
    tp["rpc_share_of_sampling"] = (tp.prepare_rpc_seconds+tp.finish_rpc_seconds+tp.reset_rpc_seconds) / tp.sampling_seconds
    csv(out / "训练逐模型.csv", results)
    csv(out / "训练分组.csv", summary)
    csv(out / "训练配对差_全部.csv", pe)
    csv(out / "训练配对统计.csv", cs)
    csv(out / "单侧联合配对.csv", pd.DataFrame(sj))
    csv(out / "规则逐顺序.csv", rules)
    csv(out / "规则分组.csv", rule_summary)
    csv(out / "训练逐评估记录.csv", ep)
    csv(out / "完整学习曲线.csv", cf)
    csv(out / "训练耗时逐模型.csv", cell)
    csv(out / "训练耗时分组.csv", perf)
    csv(out / "采样更新时间.csv", tp)

    # 诊断的顺序重复不是独立训练；保留案例平均与每条记录。
    tables = [pd.read_csv(diag / name) for name in ["01_完整回放.csv", "02_同等尾批处理.csv", "03_保人数交换.csv", "04_候选成本归因.csv", "05_完整预测修正.csv"]]
    base, tail, swaps, branch, corrections = tables
    for df in tables:
        df["cohort"] = df.case.map(cohort)
    base_only = base[base["mode"].eq("BASE")]
    case_base = base_only.groupby(["case", "cohort"]).agg(base_rpt=("rpt", "mean"), completion=("completion", "mean"),
                                full=("full", "mean")).reset_index()
    case_tail = tail[tail["mode"].eq("BASE")].groupby("case").agg(tail_rpt=("repaired_rpt", "mean"),
                                tail_gain=("gain", "mean"), tail_full=("repaired_full", "mean"))
    case_swap = swaps.groupby("case").agg(swap_best=("best_rpt", "mean"), swap_gain=("gain", "mean"),
                                swap_wait_gain=("wait_occupancy_gain", "mean"), swap_ground_gain=("ground_occupancy_gain", "mean"))
    case_overview = case_base.merge(case_tail, on="case").merge(case_swap, on="case")
    overview = case_overview.groupby("cohort").agg(n_cases=("case", "size"), base_rpt=("base_rpt", "mean"),
                    completion=("completion", "mean"), full=("full", "mean"), tail_rpt=("tail_rpt", "mean"),
                    tail_gain=("tail_gain", "mean"), swap_best=("swap_best", "mean"), swap_gain=("swap_gain", "mean"),
                    wait_gain=("swap_wait_gain", "mean"), ground_gain=("swap_ground_gain", "mean")).reset_index()
    # 两个同背景比较：完整闭环修正，不把局部赢家的收益加起来。
    corr = []
    for mode in ["KNOWN_BASE", "KNOWN_READY"]:
        for _, row in case_overview[case_overview.cohort.isin(["PT_S", "PT_J"])].iterrows():
            raw = corrections[(corrections.case == row["case"]) & corrections["mode"].eq(mode)]
            after = tail[(tail.case == row["case"]) & tail["mode"].eq(mode)]
            corr.append(dict(case=row["case"], cohort=row.cohort, mode=mode,
                             raw_delta=raw.rpt.mean()-row.base_rpt,
                             equal_tail_delta=after.repaired_rpt.mean()-row.tail_rpt))
    corr = pd.DataFrame(corr)
    corr_stats = []
    for (group, mode), part in corr.groupby(["cohort", "mode"]):
        for metric in ["raw_delta", "equal_tail_delta"]:
            corr_stats.append(dict(cohort=group, mode=mode, metric=metric, **paired(part.sort_values("case")[metric])))
    # 候选窗口以状态为单位汇总，再按训练模型聚合，窗口长度绝不伪增样本。
    branch["positive"] = branch.offline_local_gain > 0
    branch["known_positive"] = branch.known_forecast_realized_gain > 0
    branch["known_negative"] = branch.known_forecast_realized_gain < 0
    groupcols = ["stage", "continuation", "horizon", "case", "cohort"]
    cb = branch.groupby(groupcols).agg(states=("index", "size"), gain=("offline_local_gain", "mean"),
             positive=("positive", "mean"), focal=("focal_gain", "mean"), existing=("existing_gain", "mean"),
             future=("future_gain", "mean"), harms_focal=("every_system_best_harms_focal", "mean"),
             personal_conflict=("personal_system_optima_disjoint", "mean"), rank_reversal=("optimal_sets_60_240_disjoint", "mean"),
             p_gain=("passenger_change_fixed_air_gain", "mean"), a_gain=("aircraft_change_fixed_passenger_gain", "mean"),
             beyond_single=("joint_beyond_best_single_gain", "mean"),
             known_gain=("known_forecast_realized_gain", "mean"),
             known_positive=("known_positive", "mean"), known_negative=("known_negative", "mean")).reset_index()
    bg = cb.groupby(["stage", "continuation", "horizon", "cohort"]).mean(numeric_only=True).reset_index()
    # 排名反转在三个窗口重复记录，只取一个窗口计数；各分量均为事后局部窗口人分钟。
    known_reversals = []
    for path in (diag / "cases").glob("*/*/A_BRANCHES/state_*.json"):
        data = read(path)
        for continuation in {r["continuation"] for r in data["rows"]}:
            for horizon in (60, 120, 240):
                scores = {scope: {tuple(r["action"]): r["snapshots"][str(horizon)]["total"]
                                  for r in data["rows"] if r["scope"] == scope and r["continuation"] == continuation}
                          for scope in ["COMMITTED", "REALIZED"]}
                best = {s: {a for a,v in values.items() if v == min(values.values())} for s,values in scores.items()}
                known_reversals.append(dict(case=path.parts[-4], trace=path.parts[-3], index=data["index"],
                       continuation=continuation, horizon=horizon,
                       disjoint=not bool(best["COMMITTED"] & best["REALIZED"])))
    kr = pd.DataFrame(known_reversals)
    kr["cohort"] = kr.case.map(cohort)
    krs = kr.groupby(["continuation", "horizon", "cohort"]).disjoint.agg(["mean", "size"]).reset_index()
    csv(out / "诊断逐模型.csv", case_overview)
    csv(out / "诊断总览.csv", overview)
    csv(out / "合法预测修正逐模型.csv", corr)
    csv(out / "合法预测修正统计.csv", pd.DataFrame(corr_stats))
    csv(out / "局部候选逐模型.csv", cb)
    csv(out / "局部候选分组.csv", bg)
    csv(out / "已知与真实未来排序差.csv", krs)
    sessions = read(diag / "SESSIONS.json")
    # 首尾时间来自用户提供的正式日志，单模型计时与评估计时来自包内逐格结果。
    suite_wall = 9*3600 + 56*60 + 23
    totalsteps = int(cell.steps.sum())
    speed = dict(total_steps=totalsteps, suite_wall_seconds=suite_wall,
                 suite_fps=totalsteps/suite_wall, train_seconds=float(cell.train_seconds.sum()),
                 eval_seconds=float(cell.eval_seconds.sum()),
                 train_fps=totalsteps/cell.train_seconds.sum(),
                 train_eval_fps=totalsteps/(cell.train_seconds.sum()+cell.eval_seconds.sum()),
                 diagnostic_active_wall=sum(s["elapsed_seconds"] for s in sessions),
                 training18h_at_observed_suite_fps=64800*totalsteps/suite_wall)
    result = dict(input_manifest=read(root / "INPUT_MANIFEST.json"), verification=verification,
                  ppo=plan["ppo"], speed=speed, train_summary=summary.to_dict("records"),
                  paired=cs.to_dict("records"), joint_single=sj, rules=rule_summary.to_dict("records"),
                  diag_overview=overview.to_dict("records"), corrections=corr_stats,
                  candidate_summary=bg.to_dict("records"), known_rank=krs.to_dict("records"),
                  diagnostic_session_end=read(diag / "SESSION_END.json"),
                  late_episode_count=len(late), late_all_full=int(late.full.sum()),
                  late_tail=int(late.seat_tail.sum()), late_blocked=int(late.restricted_blocked.sum()),
                  final_episode_count=len(final), final_all_full=int(final.full.sum()),
                  software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    save(out / "分析复算.json", result)
    save(out / "分析校验.json", verification)
    plot(out, results, cf)
    print(json.dumps(clean(dict(speed=speed, verification=verification,
                   train_summary=summary.to_dict("records"), paired=cs[cs.k.eq(3)].to_dict("records"),
                   rules=rule_summary.to_dict("records"), diag_overview=overview.to_dict("records"),
                   corrections=corr_stats)), ensure_ascii=True, indent=2))


def plot(out, models, curves):
    """只画未经平滑的真实曲线与训练种子散点；不把删失ATT画成零。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000"]
    arms = ["L00", "L10_TIME", "L01_BATCH", "L11_BOTH", "T11_BOTH"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
        for ax, control in zip(axes, ["LQ", "J3"]):
            for i, arm in enumerate(arms):
                data = models[(models.k == 3) & (models.control == control) & (models.arm == arm)]
                ax.plot(np.full(len(data), i), data.late_rpt, "o", color=colors[i], fillstyle="none")
                ax.plot(i, data.late_rpt.mean(), marker="_", markersize=24, color=colors[i])
            ax.set(title=f"K3 {control}: four training seeds", ylabel="Late validation restricted time (min/person)")
            ax.set_xticks(range(5), ["Base", "Time", "Batch", "Both", "Transformer"], rotation=15)
            ax.grid(axis="y", alpha=.2)
        fig.savefig(out / "01_seed_points.png", dpi=180, facecolor="white")
        plt.close(fig)
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained", sharex=True, sharey=True)
        for row, control in enumerate(["LQ", "J3"]):
            for col, seed in enumerate([1701, 1702, 1703, 1704]):
                ax = axes[row, col]
                for i, arm in enumerate(arms):
                    d = curves[(curves.k == 3) & (curves.control == control) & (curves.seed == seed) & (curves.arm == arm)].sort_values("step")
                    ax.plot(d.step/1e6, d.rpt, color=colors[i], linestyle=["-", "--", "-.", ":", "-"][i], label=arm)
                ax.set(title=f"{control}, seed {seed}", xlabel="Checkpoint label (million)", ylabel="Original-order RPT (min/person)")
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7)
        fig.savefig(out / "02_all_k3_curves.png", dpi=150, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
