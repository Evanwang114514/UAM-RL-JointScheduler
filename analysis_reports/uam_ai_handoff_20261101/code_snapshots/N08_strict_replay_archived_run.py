"""严格E1既有权重的本地无训练诊断；所有输出只写入独立工作区。"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import html
import importlib.util
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import random
import statistics
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile

EXPECTED = "f94383c22ce38454fc142992f640498e3c948dbea531b5d0f8cadabc1c1968b0"
SUITE = "uam_strictE1_LQ_fleet40m_methods60m_v1"
ENTRY = "train_uam_strictE1_LQ_fleet40m_methods60m_ONEFILE.py"
STEPS = (983040, 1003520)
WORKSPACE = Path(__file__).resolve().parents[2]
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_output(path):
    p = Path(path).resolve()
    if not p.is_relative_to(WORKSPACE):
        raise RuntimeError("诊断输出不在独立工作区内")
    return p


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, obj):
    p = safe_output(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".pending")
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def cell_id(k, seed):
    return f"P2__K{k}__N12__UAGMC_SOURCE_L__s{seed}"


def prepare(archive, out):
    out = safe_output(out)
    if (out / "PREPARE_END.json").exists():
        return read(out / "PREPARE_END.json")
    digest = sha(archive)
    if digest != EXPECTED:
        raise RuntimeError("模型包与服务器已确认SHA256不一致")
    wanted = {cell_id(k, seed) for k in (2, 3, 4) for seed in (601, 602, 603, 604)}
    inventory, models, norms, seen, selected = [], set(), set(), set(), []
    extracted = 0
    # 只提取作者源码、源UAGMC的配套权重与对应记录；不使用extractall。
    with tarfile.open(archive, "r|gz") as tf:
        for item in tf:
            rel = PurePosixPath(item.name)
            if rel.is_absolute() or ".." in rel.parts or any(":" in x for x in rel.parts):
                raise RuntimeError("压缩包包含路径越界")
            key = rel.as_posix()
            if key in seen or item.issym() or item.islnk():
                raise RuntimeError("压缩包包含重复路径或链接：" + key)
            seen.add(key)
            if not item.isfile():
                continue
            inventory.append(dict(path=key, size=item.size))
            if key.endswith(".zip"):
                models.add(key[:-4])
            if key.endswith(".pkl"):
                norms.add(key[:-4])
            parts = rel.parts
            current = parts[3] if len(parts) > 3 and parts[:3] == ("serial_runs", SUITE, "cells") else None
            take = key == ENTRY or (len(parts) > 2 and parts[:2] == ("serial_runs", SUITE) and parts[2] == "_author_source" and rel.suffix in (".py", ".csv", ".json"))
            if current in wanted:
                name = rel.name
                take = (name in ("TASK.json", "MODEL_MANIFEST.json", "STRICT_ENV_MANIFEST.json", "TRAIN_END.json")
                        or ("resume" in parts and any(name == f"step_{s:09d}{ext}" for s in STEPS for ext in (".zip", ".pkl", ".json")))
                        or ("evaluation" in parts and any(name == f"step_{s:09d}.json" for s in STEPS)))
            if not take:
                continue
            if item.size > 100 * 1024**2:
                raise RuntimeError("单个提取条目超过限制")
            p = safe_output(out / "input" / Path(*parts))
            p.parent.mkdir(parents=True, exist_ok=True)
            stream = tf.extractfile(item)
            content = stream.read()
            if len(content) != item.size:
                raise RuntimeError("压缩包条目不完整")
            if p.exists() and p.read_bytes() != content:
                raise RuntimeError("既有隔离输入副本不同，拒绝覆盖")
            if not p.exists():
                p.write_bytes(content)
            extracted += len(content)
            if p.suffix in (".zip", ".pkl"):
                selected.append(dict(path=str(p), sha256=hashlib.sha256(content).hexdigest(), size=len(content)))
    if len(models) != 200 or models != norms:
        raise RuntimeError("全包模型和归一化未达到200组同名配对")
    root = out / "input" / "serial_runs" / SUITE
    jobs = []
    for k in (2, 3, 4):
        for seed in (601, 602, 603, 604):
            folder = root / "cells" / cell_id(k, seed)
            task_record = read(folder / "TASK.json")
            for step in STEPS:
                for ext in (".zip", ".pkl"):
                    p = folder / "resume" / f"step_{step:09d}{ext}"
                    if not p.is_file():
                        raise RuntimeError("主诊断缺配对文件：" + str(p))
                    if ext == ".zip":
                        with zipfile.ZipFile(p) as z:
                            bad = z.testzip()
                            if bad:
                                raise RuntimeError("模型内部CRC失败：" + bad)
            jobs.append(dict(id=cell_id(k, seed), task=task_record["task"], config=task_record["config"],
                             root=str(root), runtime=str(out / "input" / ENTRY), out=str(out / "N0" / cell_id(k, seed))))
    result = dict(archive=str(archive), archive_sha256=digest, all_models=len(models), all_norms=len(norms),
                  paired_cells=len({PurePosixPath(x).parts[3] for x in models}), selected_models=24,
                  selected_norms=24, extracted_bytes=extracted, jobs=jobs, weights=selected,
                  new_training_transitions=0)
    save(out / "archive_inventory.json", inventory)
    save(out / "PREPARE_END.json", result)
    print("[包核验] 全部200组配对；主诊断24组权重/归一化CRC通过", flush=True)
    return result


def runtime(job):
    os.environ.update(UAM_STRICT_MODE="E1", UAM_STRICT_FLEET="12", UAM_STRICT_LOG_EVENTS="1")
    p = safe_output(job["out"])
    p.mkdir(parents=True, exist_ok=True)
    os.chdir(p)
    spec = importlib.util.spec_from_file_location("strict_replay_runtime", job["runtime"])
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    m.install_learning_classes()
    m.materialize_source(Path(job["root"]))
    m.activate_source(Path(job["root"]), job["task"]["k"])
    from loguru import logger
    logger.remove()
    logging.disable(logging.CRITICAL)
    m.torch.set_num_threads(1)
    return m


def load_pair(m, job, step):
    p = Path(job["root"]) / "cells" / job["id"] / "resume" / f"step_{step:09d}"
    model = m.SourcePPO.load(str(p) + ".zip", device="cpu")
    class OwnNorm(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod in ("__main__", "__mp_main__") and name in m.__dict__:
                return m.__dict__[name]
            return super().find_class(mod, name)
    with Path(str(p) + ".pkl").open("rb") as f:
        norm = OwnNorm(f).load()
    norm.training = False
    norm.norm_reward = False
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    def forbidden(*a, **kw):
        raise RuntimeError("无训练诊断禁止调用learn/train/optimizer.step")
    model.learn = model.train = forbidden
    if hasattr(model.policy, "optimizer"):
        model.policy.optimizer.step = forbidden
    return model, norm


def physical(env, m):
    s = env.scenario
    current = s.persons.persons[s.waiting_decisions[0]] if s.waiting_decisions else None
    rows = []
    for vid in m.CANDIDATES[env.k]:
        v = s.vertiports.vertiport_list[str(vid)]
        local = s.vertiports.evtols_at_vertiport.get(str(vid), [])
        incoming = [dict(id=str(p.id), eta=float(p.current_timer or 0)) for p in s.persons.persons.values()
                    if str(getattr(p, "origin_vertiport_id", "")) == str(vid)
                    and getattr(p, "state", "") == "enroute" and getattr(p, "sub_state", "") == "to_vertiport"]
        planes = [dict(id=str(e.id), state=e.state.name, battery=float(e.battery_kwh),
                       ready=max(0., float(e.spec.battery_capacity_kwh - e.battery_kwh)) / max(1e-8, float(e.spec.charge_rate_kwh_per_min))) for e in local]
        returns = [dict(id=str(e.id), eta=float(e.remaining_time), battery=float(e.battery_kwh))
                   for e in s._all_evtols.values() if e.state.name == "FLYING" and not e.passenger_ids
                   and str(e.target_vertiport_id) == str(vid)]
        access = float(s.vehicles.estimate_travel_time(current.origin_position, v.vertiport_position)) if current else None
        rows.append(dict(station=vid, queue=list(v.person_list), incoming=incoming, aircraft=planes,
                         empty_returns=returns, access=access))
    return dict(time=s.time, focal=str(current.id) if current else None, stations=rows,
                finished=len(set(s.finished_ids)), generated=len(s.persons.persons))


def rule_choice(env, m, name, rng):
    """作者规则直接调用封存函数；READY评分逐项沿用既有120M实现。"""
    s = env.scenario
    if not s.waiting_decisions:
        return 0
    if name.startswith("AUTHOR_"):
        if not hasattr(env, "_diagnostic_author_rules"):
            env._diagnostic_author_rules = {}
        cache = env._diagnostic_author_rules
        if name not in cache:
            cache[name] = m.AuthorFunctions(name.split("_", 1)[1])
        return m.adapter_decision(s, env.k, cache[name], m.np)[0]
    if name != "READY_QUEUE_LQ":
        return m.rule_action(env, name, rng)
    person = s.persons.persons[s.waiting_decisions[0]]
    scores = []
    for vid in m.CANDIDATES[env.k]:
        v = s.vertiports.vertiport_list[str(vid)]
        access = float(s.vehicles.estimate_travel_time(person.origin_position, v.vertiport_position))
        distance = s.vertiports._calculate_distance(v.vertiport_position, s.vertiports.vertiport_list['2'].vertiport_position)
        flight = distance / 200. * 60.
        local = s.vertiports.evtols_at_vertiport.get(str(vid), [])
        ready = [max(0., e.spec.battery_capacity_kwh - e.battery_kwh) / max(e.spec.charge_rate_kwh_per_min, 1e-8)
                 for e in local if not e.passenger_ids and e.state.name in ('IDLE', 'CHARGING')]
        queued = len(v.person_list)
        incoming = sum(str(getattr(p, 'origin_vertiport_id', '')) == str(vid)
                       and getattr(p, 'sub_state', '') == 'to_vertiport'
                       and getattr(p, 'state', '') == 'enroute'
                       and max(0., float(p.current_timer or 0.)) <= access for p in s.persons.persons.values())
        cycle = 2. * flight + 2. * distance * .55 / 5.32
        wait = min(ready, default=flight + distance * .55 / 5.32)
        wait += ((queued + incoming) // 4) * cycle / max(1, len(ready))
        scores.append(max(access, wait) + flight)
    return min(range(env.k), key=lambda i:(scores[i], i))


def replay(m, model, norm, job, seed, capture=False, rule=None, forced=None, until=None):
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(seed)
    root = Path(job["root"])
    trace = root / "_author_source" / m.source_snapshot()["commit"] / "train_data" / "passengers_300.csv"
    env = m.SourceEnv(root, job["task"]["k"], 600, str(trace))
    obs, _ = env.reset(seed=seed)
    decisions, ledger, counts, area = [], [], [0] * env.k, 0.
    started = time.monotonic()
    decision_index = 0
    rng = random.Random(seed)
    try:
        with m.torch.no_grad():
            while True:
                s = env.scenario
                t = int(s.time)
                unfinished = len(set(s.persons.persons) - set(s.finished_ids))
                area += unfinished
                if rule:
                    action = rule_choice(env, m, rule, rng)
                else:
                    normalized = norm.normalize_obs(obs.copy())
                    action, _ = model.predict(normalized, deterministic=True)
                    action = int(action)
                if s.waiting_decisions:
                    if forced and decision_index in forced:
                        action = int(forced[decision_index])
                    counts[action] += 1
                    if capture:
                        normalized = norm.normalize_obs(obs.copy())
                        dist = model.policy.get_distribution(m.torch.as_tensor(normalized[None], device="cpu"))
                        snap = physical(env, m)
                        decisions.append(dict(index=decision_index, action=action, **snap,
                                              probabilities=dist.distribution.probs[0].cpu().tolist(),
                                              raw_observation=obs.tolist(), normalized_observation=normalized.tolist()))
                    decision_index += 1
                ledger.append(dict(time=t, unfinished=unfinished))
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated or (until is not None and env.scenario.time >= until):
                    break
        result = m.episode_metrics(env, counts, area)
        result.update(eval_seed=seed, rule=rule, wall_seconds=time.monotonic() - started,
                      ledger=ledger, decisions=decisions, new_training_transitions=0)
        return clean(result)
    finally:
        env.close()


def worker(jobfile):
    job = read(jobfile)
    m = runtime(job)
    out = Path(job["out"])
    if job.get("phase") == "n1":
        model, norm = load_pair(m, job, STEPS[-1])
        episodes = []
        for rule in ("AUTHOR_SPF", "AUTHOR_STTF", "AUTHOR_QTTI", "READY_QUEUE_LQ"):
            for seed in (123, 124):
                ep = replay(m, model, norm, job, seed, rule=rule)
                save(out / f"{rule}_s{seed}.json", ep)
                episodes.append(dict(rule=rule, eval_seed=seed, full=ep['full'], co=ep['ATT_completed_only'],
                                     rpt=ep['restricted_person_time_per_generated'], completion=ep['completion']))
        save(out / "N1_END.json", dict(k=job['task']['k'], episodes=episodes, new_training_transitions=0))
        return
    if job.get("phase") == "n2":
        branch_worker(m, job)
        return
    if job.get('phase') == 'n3':
        offline_worker(m, job)
        return
    checks = []
    for step in STEPS:
        model, norm = load_pair(m, job, step)
        old_path = Path(job["root"]) / "cells" / job["id"] / "evaluation" / f"step_{step:09d}.json"
        # 倒数第二个resume不是周期评估点，不能伪造不存在的历史参考。
        original = read(old_path) if old_path.is_file() else None
        if step == STEPS[-1] and original is None:
            raise RuntimeError("最终保存点缺少原评估参考")
        for seed in (123, 124):
            ep = replay(m, model, norm, job, seed, capture=step == STEPS[-1])
            old = next(x for x in original["episodes"] if x["eval_seed"] == seed and x["mode"] == "DET") if original else None
            diff = {}
            for key in ("generated", "finished", "ATT_completed_only", "restricted_person_time_per_generated"):
                a, b = ep.get(key), old.get(key) if old else None
                diff[key] = (abs(a - b) if a is not None and b is not None else (0 if a == b else None)) if old else None
            ok = all(x is not None and x < 1e-6 for x in diff.values()) if old else None
            checks.append(dict(step=step, eval_seed=seed, reference_available=old is not None, matched=ok, differences=diff,
                               finished=ep["finished"], co=ep["ATT_completed_only"], rpt=ep["restricted_person_time_per_generated"]))
            save(out / f"step_{step:09d}_s{seed}.json", ep)
            print("[复评]", job["id"], step, seed, ("一致" if ok else "不一致") if old else "无历史参考的新复评", ep["finished"], flush=True)
            if ok is False:
                save(out / "REPLAY_MISMATCH.json", checks[-1])
                raise RuntimeError("复评与原记录不一致，停止该模型后续分支")
    save(out / "N0_END.json", dict(id=job["id"], checks=checks, matched=True, new_training_transitions=0))


def branch_worker(m, job):
    """先固定分层样本，再进行同状态分支；克隆参考动作必须复现原轨迹成本。"""
    model, norm = load_pair(m, job, STEPS[-1])
    out = Path(job['out'])
    n0 = read(Path(job['n0']) / f"step_{STEPS[-1]:09d}_s123.json")
    records = n0['decisions']
    rng = random.Random(20261101 + job['task']['seed'])
    chosen = {}
    for group in range(3):
        ids = [x['index'] for x in records if group * 100 <= x['index'] < (group + 1) * 100]
        for index in rng.sample(ids, min(8, len(ids))):
            chosen[index] = 'phase_random'
    # 补样仅用于机制探索，不并入总体错误率。
    enriched = [x['index'] for x in records if x['index'] not in chosen and
                x['action'] != min(range(len(x['stations'])), key=lambda i:(x['stations'][i]['access'], i))]
    for index in rng.sample(enriched, min(12, len(enriched))):
        chosen[index] = 'access_disagreement'
    save(out / 'FROZEN_SAMPLE.json', dict(samples=chosen, random_seed=20261101 + job['task']['seed'],
                                        rule='每早中晚阶段8个随机状态，另补12个与最近接驳分歧状态',
                                        labels_viewed_before_selection=False, one_realized_future=True))
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(123)
    root = Path(job['root'])
    trace = root / '_author_source' / m.source_snapshot()['commit'] / 'train_data/passengers_300.csv'
    env = m.SourceEnv(root, job['task']['k'], 600, str(trace))
    obs, _ = env.reset(seed=123)
    samples, index = {}, 0
    with m.torch.no_grad():
        while True:
            if env.scenario.waiting_decisions:
                if index in chosen:
                    recorded = records[index]
                    if env.scenario.time != recorded['time'] or not m.np.array_equal(obs, m.np.asarray(recorded['raw_observation'], dtype=obs.dtype)):
                        raise RuntimeError('原决策输入无法精确重建')
                    clone = copy.deepcopy(env)
                    if clone.raw.env is not clone.scenario:
                        raise RuntimeError('克隆未保持wrapper与scenario的共享引用')
                    samples[index] = (clone, obs.copy(), random.getstate(), m.np.random.get_state())
                index += 1
            action, _ = model.predict(norm.normalize_obs(obs.copy()), deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
    env.close()
    rows = []
    ticks = 0
    for index in sorted(samples):
        template, observation, py_state, np_state = samples[index]
        state = records[index]
        for action in range(job['task']['k']):
            branch = copy.deepcopy(template)
            random.setstate(py_state)
            m.np.random.set_state(np_state)
            raw = observation.copy()
            initial_ids = set(branch.scenario.persons.persons)
            focal = state['focal']
            cost = [0., 0., 0.]
            snapshots = {}
            train_return = 0.
            reward_sum = 0.
            start_t = branch.scenario.time
            try:
                with m.torch.no_grad():
                    for offset in range(120):
                        s = branch.scenario
                        unfinished = set(s.persons.persons) - set(s.finished_ids)
                        cost[0] += int(focal in unfinished)
                        cost[1] += len((unfinished & initial_ids) - {focal})
                        cost[2] += len(unfinished - initial_ids)
                        a = action if offset == 0 else int(model.predict(norm.normalize_obs(raw.copy()), deterministic=True)[0])
                        raw, reward, terminated, truncated, _ = branch.step(a)
                        ticks += 1
                        train_return += .99**offset * float(reward)
                        reward_sum += float(reward)
                        horizon = offset + 1
                        if horizon in (15, 60, 120) or terminated or truncated:
                            snapshots[str(horizon)] = dict(cost=sum(cost), focal=cost[0], existing_other=cost[1],
                                                           future=cost[2], finished=len(set(s.finished_ids)),
                                                           source_discounted_return=train_return, source_return=reward_sum)
                        if terminated or truncated:
                            break
                if action == state['action']:
                    for h, snap in snapshots.items():
                        expected = sum(x['unfinished'] for x in n0['ledger'] if start_t <= x['time'] < start_t + int(h))
                        if abs(snap['cost'] - expected) > 1e-6:
                            raise RuntimeError('参考动作克隆分支未复现原后续成本')
                rows.append(dict(index=index, time=start_t, layer=chosen[index], original_action=state['action'],
                                 candidate=action, probability=state['probabilities'][action], snapshots=snapshots))
            finally:
                branch.close()
        save(out / 'BRANCH_PROGRESS.json', dict(states_done=len({r['index'] for r in rows}), total_states=len(samples),
                                               branches=len(rows), simulated_ticks=ticks))
    save(out / 'N2_END.json', dict(id=job['id'], eval_seed=123, states=len(samples), branches=len(rows),
                                 simulated_ticks=ticks, rows=rows, reference_clone_check_passed=True,
                                 information_scope='真实未来仅由环境逐步揭示，标签是离线事后比较；不是在线已知未来',
                                 new_training_transitions=0))


def offline_worker(m, job):
    """真实未来只用于离线可行方案搜索，不称数学最优或在线收益保证。"""
    from stable_baselines3.common.utils import set_random_seed
    out = Path(job['out'])
    root = Path(job['root'])
    trace = root / '_author_source' / m.source_snapshot()['commit'] / 'train_data/passengers_300.csv'
    os.environ['UAM_STRICT_LOG_EVENTS'] = '0'
    started = time.monotonic()
    def rollout(tape=None):
        set_random_seed(123)
        env = m.SourceEnv(root, job['task']['k'], 600, str(trace))
        env.reset(seed=123)
        counts, area, index, actual = [0] * env.k, 0., 0, []
        try:
            while True:
                s = env.scenario
                area += len(set(s.persons.persons) - set(s.finished_ids))
                a = 0
                if s.waiting_decisions:
                    a = int(tape[index]) if tape is not None else rule_choice(env, m, 'READY_QUEUE_LQ', None)
                    actual.append(a)
                    index += 1
                    counts[a] += 1
                _, _, terminated, truncated, _ = env.step(a)
                if terminated or truncated:
                    break
            ep = clean(m.episode_metrics(env, counts, area))
            if index != 300:
                raise RuntimeError('离线方案未覆盖300次合法乘客决策')
            return ep, actual
        finally:
            env.close()
    history, candidates, cache = [], [], {}
    best_ep = None
    best_tape = None
    def score(ep):
        # 可行性阶段与完整ATT优化明确区分；不把未完成者CO当全体ATT。
        return (300 - ep['finished'], ep['ATT'] if ep['full'] else ep['restricted_person_time_per_generated'])
    def evaluate(tape, label):
        nonlocal best_ep, best_tape
        key = tuple(tape)
        if key in cache:
            return cache[key]
        ep, actions = rollout(tape)
        cache[key] = ep
        if best_ep is None or score(ep) < score(best_ep):
            best_ep, best_tape = ep, actions
            save(out / 'INCUMBENT.json', dict(episode=ep, actions=actions, source=label))
        history.append(dict(candidate=len(history) + 1, proposal=label, finished=ep['finished'], full=ep['full'],
                            att=ep['ATT'], rpt=ep['restricted_person_time_per_generated'],
                            incumbent_att=best_ep['ATT'], incumbent_finished=best_ep['finished']))
        return ep
    ready_ep, ready_tape = rollout()
    candidates.append(('READY_QUEUE_LQ', ready_tape))
    reference = read(Path(job['rules_dir']) / 'READY_QUEUE_LQ_s123.json')
    if abs(ready_ep['restricted_person_time_per_generated'] - reference['restricted_person_time_per_generated']) > 1e-6:
        raise RuntimeError('离线搜索的规则暖启动与N1不一致')
    for seed in (601, 602, 603, 604):
        old = read(Path(job['n0']).parent / cell_id(job['task']['k'], seed) / 'step_001003520_s123.json')
        actions = [x['action'] for x in old['decisions']]
        ep = evaluate(actions, f'UAGMC_s{seed}')
        if abs(ep['restricted_person_time_per_generated'] - old['restricted_person_time_per_generated']) > 1e-6:
            raise RuntimeError('固定动作序列未复现原模型完整回放')
        candidates.append((f'UAGMC_s{seed}', actions))
    evaluate(ready_tape, 'READY_QUEUE_LQ')
    # 只改变合法乘客选择，飞机始终仍由原LQ执行；不允许不足4人发机。
    for label, actions in candidates:
        pool = []
        for a in range(job['task']['k']):
            ids = [i for i, x in enumerate(actions) if x == a]
            remainder = len(ids) % 4
            pool.extend(ids[-remainder:] if remainder else [])
        for dest in range(job['task']['k']):
            proposal = actions.copy()
            for i in pool:
                proposal[i] = dest
            evaluate(proposal, label + '_full_batch_repair')
    rng = random.Random(20261101 + job['task']['k'])
    for attempt in range(160):
        proposal = list(best_tape)
        if attempt % 4 != 3:
            i = rng.randrange(300)
            other = [j for j in range(300) if proposal[j] != proposal[i]]
            if not other:
                continue
            j = rng.choice(other)
            proposal[i], proposal[j] = proposal[j], proposal[i]
            label = 'count_preserving_swap'
        else:
            src = rng.randrange(job['task']['k'])
            options = [i for i, a in enumerate(proposal) if a == src]
            if len(options) < 4:
                continue
            dest = rng.choice([a for a in range(job['task']['k']) if a != src])
            for i in rng.sample(options, 4):
                proposal[i] = dest
            label = 'four_passenger_move'
        evaluate(proposal, label)
        if attempt % 10 == 0:
            save(out / 'SEARCH_PROGRESS.json', dict(attempt=attempt, evaluated=len(history), incumbent_att=best_ep['ATT']))
    save(out / 'N3_END.json', dict(k=job['task']['k'], eval_seed=123, candidates=len(history), episode=best_ep,
                                 actions=best_tape, history=history, wall_seconds=time.monotonic() - started,
                                 full_future_information=True, same_lq_control=True, global_optimality_certified=False,
                                 no_claim_of_online_learnability=True, new_training_transitions=0))


def write_csv(path, rows):
    """汇总表只写独立诊断目录；空值不伪装成零。"""
    path = safe_output(path)
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(columns, rows):
    def fmt(value):
        if value is None:
            return '—'
        if isinstance(value, float):
            return f'{value:.3f}'
        return str(value).replace('|', '/')
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
                       '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
                      ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in rows])


def report(out):
    """保留每个模型及抽样层；本次不做小样本显著性筛选和总体因果分摊。"""
    state = read(out / 'PREPARE_END.json')
    integrity = [dict(path=x['path'], unchanged=sha(x['path']) == x['sha256']) for x in state['weights']]
    if not all(x['unchanged'] for x in integrity):
        raise RuntimeError('隔离输入的权重或归一化在回放后发生变化')
    save(out / 'POST_INPUT_HASH_CHECK.json', integrity)
    if not all(len(read(out / f'N{i}_SUMMARY.json')) == (12 if i in (0, 2) else 3) for i in range(4)):
        raise RuntimeError('诊断阶段尚未完整结束')
    episodes, models, rules, costs, states, offline, history, examples, audits = [], [], [], [], [], [], [], [], []
    for job in state['jobs']:
        k, seed = job['task']['k'], job['task']['seed']
        pair = []
        time_values, nonzero_time, real_ready_states, exact_input_groups = 0, 0, 0, {}
        records = None
        for step in STEPS:
            for es in (123, 124):
                ep = read(out / 'N0' / job['id'] / f'step_{step:09d}_s{es}.json')
                episodes.append(dict(id=job['id'], k=k, train_seed=seed, step=step, eval_seed=es,
                                     finished=ep['finished'], completion=ep['completion'], full_att=ep['ATT'],
                                     completed_only_att=ep['ATT_completed_only'], rpt=ep['restricted_person_time_per_generated'],
                                     historical_reference=step == STEPS[-1]))
                if step != STEPS[-1]:
                    continue
                pair.append(ep)
                frame = 4 + 8*k
                for d in ep['decisions']:
                    for f in range(6):
                        for a in range(k):
                            for channel in (5, 6, 7):
                                value = d['raw_observation'][f*frame + 4 + 8*a + channel]
                                time_values += 1
                                nonzero_time += int(abs(value) > 1e-12)
                    real_ready_states += int(any(x['ready'] > 0 for s in d['stations'] for x in s['aircraft'])
                                             or any(x['eta'] > 0 for s in d['stations'] for x in s['empty_returns']))
                if es == 123:
                    records = ep['decisions']
        models.append(dict(id=job['id'], k=k, train_seed=seed,
                           rpt=statistics.mean(e['restricted_person_time_per_generated'] for e in pair),
                           completion=statistics.mean(e['completion'] for e in pair),
                           completed_only_att=statistics.mean(e['ATT_completed_only'] for e in pair)))
        audits.append(dict(id=job['id'], inspected_decisions=600, inspected_time_values=time_values,
                           nonzero_time_values=nonzero_time, physical_pending_states=real_ready_states))
        result = read(out / 'N2' / job['id'] / 'N2_END.json')
        groups = {}
        for row in result['rows']:
            groups.setdefault(row['index'], []).append(row)
            for h, values in row['snapshots'].items():
                costs.append(dict(id=job['id'], k=k, train_seed=seed, index=row['index'], layer=row['layer'],
                                  action=row['candidate'], original_action=row['original_action'], horizon=int(h), **values))
        for index, rows in groups.items():
            d = records[index]
            original = next(r for r in rows if r['candidate'] == r['original_action'])
            sets = {}
            for h in ('15', '60', '120'):
                low = min(r['snapshots'][h]['cost'] for r in rows)
                sets[h] = {r['candidate'] for r in rows if r['snapshots'][h]['cost'] == low}
            best_cost = min(r['snapshots']['120']['cost'] for r in rows)
            best = [r for r in rows if r['snapshots']['120']['cost'] == best_cost]
            focal_low = min(r['snapshots']['120']['focal'] for r in rows)
            focal_set = {r['candidate'] for r in rows if r['snapshots']['120']['focal'] == focal_low}
            reward_high = max(r['snapshots']['120']['source_discounted_return'] for r in rows)
            reward_set = {r['candidate'] for r in rows if abs(r['snapshots']['120']['source_discounted_return'] - reward_high) < 1e-7}
            chosen = min(best, key=lambda r:(r['snapshots']['120']['focal'], r['candidate']))
            delta = {key: original['snapshots']['120'][key] - chosen['snapshots']['120'][key]
                     for key in ('cost', 'focal', 'existing_other', 'future')}
            assert abs(delta['cost'] - delta['focal'] - delta['existing_other'] - delta['future']) < 1e-7
            row = dict(id=job['id'], k=k, train_seed=seed, index=index, time=d['time'], layer=original['layer'],
                       original_action=original['original_action'], selected_probability=d['probabilities'][original['original_action']],
                       best_actions=','.join(map(str, sorted(sets['120']))), better_action=chosen['candidate'],
                       cost_gap_120=delta['cost'], focal_gain=delta['focal'], existing_other_gain=delta['existing_other'],
                       future_gain=delta['future'], all_tied_15=len(sets['15']) == k,
                       strict_flip_15_120=not bool(sets['15'] & sets['120']),
                       strict_flip_60_120=not bool(sets['60'] & sets['120']),
                       focal_system_disjoint=not bool(focal_set & sets['120']),
                       every_system_best_hurts_focal=bool(delta['cost'] > 0 and all(r['snapshots']['120']['focal'] > original['snapshots']['120']['focal'] for r in best)),
                       source_person_time_disjoint=not bool(reward_set & sets['120']),
                       any_pending_supply=any(x['ready'] > 0 for s in d['stations'] for x in s['aircraft']) or any(x['eta'] > 0 for s in d['stations'] for x in s['empty_returns']))
            states.append(row)
            key = tuple(d['raw_observation'][:6*(4+8*k)])
            exact_input_groups.setdefault(key, []).append(sets['120'])
            if row['every_system_best_hurts_focal']:
                examples.append(dict(**row, stations=d['stations'], original=original['snapshots']['120'],
                                     alternative=chosen['snapshots']['120']))
        audits[-1]['exact_repeated_input_groups_n2'] = sum(len(v) > 1 for v in exact_input_groups.values())
        audits[-1]['exact_input_disjoint_label_pairs_n2'] = sum(not bool(a & b) for v in exact_input_groups.values()
                                                              for i, a in enumerate(v) for b in v[i+1:])
    for summary in read(out / 'N1_SUMMARY.json'):
        for e in summary['episodes']:
            rules.append(dict(k=summary['k'], **e))
    for result in read(out / 'N3_SUMMARY.json'):
        k = result['k']
        ep = result['episode']
        ready = next(h for h in result['history'] if h['proposal'] == 'READY_QUEUE_LQ')
        ready_repairs = [h['rpt'] for h in result['history'] if h['proposal'] == 'READY_QUEUE_LQ_full_batch_repair' and h['full']]
        repaired = min(ready_repairs) if ready_repairs else (ready['att'] if ready['full'] else None)
        offline.append(dict(k=k, eval_seed=123, source_rpt=statistics.mean(h['rpt'] for h in result['history'] if h['proposal'].startswith('UAGMC_s') and 'repair' not in h['proposal']),
                            ready_rpt=ready['rpt'], ready_finished=ready['finished'], ready_repaired_full_att=repaired,
                            offline_att=ep['ATT'], offline_finished=ep['finished'], offline_wait=ep['AWT'],
                            extra_beyond_ready_repair=repaired-ep['ATT'] if repaired is not None else None,
                            unique_search_candidates=result['candidates']))
        history.extend(dict(k=k, **h) for h in result['history'])
    summaries = []
    for k in (2, 3, 4):
        ms = [m for m in models if m['k'] == k]
        summaries.append(dict(k=k, n_train_models=4, mean_rpt=statistics.mean(m['rpt'] for m in ms),
                              sd_rpt=statistics.stdev(m['rpt'] for m in ms), min_rpt=min(m['rpt'] for m in ms), max_rpt=max(m['rpt'] for m in ms),
                              mean_completion=statistics.mean(m['completion'] for m in ms)))
    local = []
    for k in (2, 3, 4):
        for layer in ('phase_random', 'access_disagreement'):
            selected = [s for s in states if s['k'] == k and s['layer'] == layer]
            local.append(dict(k=k, layer=layer, states=len(selected), better_states=sum(s['cost_gap_120'] > 0 for s in selected),
                              mean_local_gap=statistics.mean(s['cost_gap_120'] for s in selected),
                              all_tied_15=sum(s['all_tied_15'] for s in selected), strict_flip_60_120=sum(s['strict_flip_60_120'] for s in selected),
                              focal_system_disjoint=sum(s['focal_system_disjoint'] for s in selected),
                              every_best_hurts_focal=sum(s['every_system_best_hurts_focal'] for s in selected),
                              source_cost_disjoint=sum(s['source_person_time_disjoint'] for s in selected)))
    for name, rows in [('逐回放48条.csv', episodes), ('逐模型12条.csv', models), ('规则逐回放24条.csv', rules),
                       ('局部分支成本.csv', costs), ('逐状态诊断432条.csv', states), ('离线全程方案.csv', offline),
                       ('离线搜索全部候选.csv', history), ('输入时间通道审计.csv', audits), ('分层局部统计.csv', local)]:
        write_csv(out / name, rows)
    save(out / '局部外部成本案例.json', sorted(examples, key=lambda e:e['cost_gap_120'], reverse=True)[:20])
    save(out / 'SUMMARY.json', dict(model_summary=summaries, local_summary=local, offline_summary=offline,
                                   new_training_transitions=0, parameters_unchanged=True, input_sha256=EXPECTED))
    text_report = build_report(out, summaries, models, rules, local, offline, audits, states, examples)
    (out / '无训练诊断_结论与证据.md').write_text(text_report, encoding='utf-8')
    render_report(out, text_report)
    save(out / 'END.json', dict(status='COMPLETE', models=12, model_replays=48, online_rule_replays=24,
                               sampled_states=len(states), local_branches=sum(len(d['rows']) for d in read(out / 'N2_SUMMARY.json')),
                               local_simulated_minutes=sum(d['simulated_ticks'] for d in read(out / 'N2_SUMMARY.json')),
                               offline_unique_candidates=sum(d['candidates'] for d in read(out / 'N3_SUMMARY.json')),
                               offline_extra_warm_replays=3, new_training_transitions=0,
                               report=str(out / '无训练诊断_结论与证据.html')))
    package = out / '无训练诊断_代码与结果_DATAONLY.tar.gz'
    with tarfile.open(package, 'w:gz') as tar:
        for folder in (Path(__file__).parent, out):
            for path in sorted(folder.rglob('*')):
                if not path.is_file() or 'input' in path.relative_to(folder).parts or path == package or path.suffix in ('.pending', '.pyc'):
                    continue
                prefix = 'code' if folder == Path(__file__).parent else 'results'
                tar.add(path, arcname=prefix + '/' + path.relative_to(folder).as_posix(), recursive=False)
    print('[报告完成]', out / '无训练诊断_结论与证据.html', flush=True)


def build_report(out, summaries, models, rules, local, offline, audits, states, examples):
    total_random = [s for s in states if s['layer'] == 'phase_random']
    parts = ['# 严格E1既有模型：无训练决策诊断',
             '日期：2026-11-01。性质：探索性、同模拟器干预；不是新训练，不是最终论文结论。',
             '## 先读这五点',
             '1. 模型包已修复并核验。12个源UAGMC-L最终模型的24次回放，与归档的生成数、完成数、完成者ATT、固定边界人时逐项完全一致。另24次末期保存点回放无对应历史评估，单独保留。',
             '2. 现有READY在线规则在三个机场规模的固定边界人时均优于源模型均值；因此，源模型落后不等于找到了值得新AI解决的缺口。',
             '3. 不改物理、不改LQ飞机控制，真未来离线搜索找到三个规模均300/300完成的方案；但相对补齐尾批后的READY，剩余改善不大且仅一条需求实现。',
             '4. 局部决策中确实有成本可改进状态，也有“当前乘客略多等、其他人明显少等”的冲突。15分钟大多看不出候选差异；这支持检查长一点的决策成本，不等于证明时间表示是主因。',
             '5. 现在没有充分证据宣称：普通PPO/Transformer无法弥补gap，或缺口已能按原因可靠排名。暂不启动训练。',
             '## 一、范围、合同与计量边界',
             '本包总100个cell：P1机队扫描40个、P2方法比较60个。本次主对象只是P2中K2/K3/K4、固定12架、seed601—604的12个UAGMC_SOURCE_L。原训练合同为5×4096、batch512、gamma0.99、约1.00352M；SOURCE奖励。不是10月8日40架/修正奖励的冻结合同。模型为LSTM，不冒称已核验所有Transformer基线。',
             'SOURCE数据只说明旧目标下的决策；奖励已修正的30M模型及Joint模型尚未在本次加载。本次LQ控制不变，没有进行Joint与Single的因果对照。',
             '原评估硬边界为600、实际产生601次离散计数。RPT=该固定边界内累计未完成人分钟/300；越低越好。未全部完成时全体ATT为空，不用完成者ATT冒充。终点为源模拟器航空到达，不补算未建模末段地面。时间分量和travel的约1分钟离散残差保留，不塞入等待。',
             '## 二、先把源模型和在线规则摆清楚',
             '先对每个训练模型的两次评估取均值，再列4个模型的均值/SD。两次评估不是两次训练，也不是独立需求；同一固定CSV下只改变回放随机初始化。小样本仅描述，不做显著性筛选。',
             md_table(['K','训练模型数','源模型RPT均值','跨训练seed SD','最小','最大','完成率'],
                      [[s['k'],4,s['mean_rpt'],s['sd_rpt'],s['min_rpt'],s['max_rpt'],f"{s['mean_completion']*100:.2f}%"] for s in summaries]),
             md_table(['K','训练seed','模型RPT','完成率'], [[m['k'],m['train_seed'],m['rpt'],f"{100*m['completion']:.2f}%"] for m in models])]
    rule_rows = []
    for k in (2,3,4):
        for name in ('AUTHOR_SPF','AUTHOR_STTF','AUTHOR_QTTI','READY_QUEUE_LQ'):
            es = [r for r in rules if r['k'] == k and r['rule'] == name]
            rule_rows.append([k,name,statistics.mean(r['rpt'] for r in es),f"{100*statistics.mean(r['completion'] for r in es):.2f}%"])
    parts += [md_table(['K','合法在线规则','RPT','完成率'],rule_rows),
              'READY不是数学下界：它是已有实现的服务就绪/队列启发式。作者SPF/STTF/QTTI直接调用封存函数；READY逐项沿用既有120M评分。全部共享物理、需求及LQ控制。没有读取未揭示请求。这里尚未运行合法在线滚动搜索，因此不能把“胜过这四种规则”写成胜过所有强解析方法。',
              '## 三、全程剩余空间：先区分尾批可行性和等待改善',
              '下表只比较eval123这一相同初始化；不要与上表两次回放均值混算。离线搜索先重放4个模型和READY动作序列，再把各站最后余数乘客集中为可满座批次，最后做160次有限交换/四人迁移提案。所有方案仍要走真实接驳、四人满座、飞机充电往返；没有让零散乘客凭空完成。',
              md_table(['K','源模型RPT/4seed均值','READY RPT','READY完成人数','READY尾批修补后ATT','离线最终ATT','修补后额外改善'],
                       [[r['k'],r['source_rpt'],r['ready_rpt'],r['ready_finished'],r['ready_repaired_full_att'],r['offline_att'],r['extra_beyond_ready_repair']] for r in offline]),
              '三个离线最终方案均300/300完成。K2 READY本来全部完成，修补列等于原ATT；K3/K4尾批修补列是在该类有限提案中事后选的最好完整方案。它改变少数末期去向，也可能影响别人的等待，所以只是有次序的可恢复成本对照，不是互斥、可加总的因果分摊。',
              '搜索可看真实完整未来，仅为可行改善证据。不是全局最优、不是性能保证，也不证明这些改善能被合法在线信息学到。538个不同候选方案属于同一需求上的自适应搜索，不能当538个独立样本。',
              '较谨慎的判断：旧源模型相对READY的差距明确；补齐尾批后的READY与本次有限搜索方案之间，已找到约0.57—1.85分钟改善，约1%—3%。这不是剩余空间的上限，更好的离线方案或其他需求可能不同；同样不能把旧源模型到离线方案的整个差距都包装成新AI缺口。',
              '## 四、不是只看终点：432个同状态候选分支',
              '每模型分早/中/晚随机抽8个，共24个普通状态；另补12个与最近接驳规则分歧的状态。随机层和补样层分开。每个状态尝试全部K个候选、后续继续同一冻结策略；共1296个分支、155520仿真分钟。每个原动作克隆均通过15/60/120分钟成本逐项一致检查。',
              '成本按当前乘客、其他已生成乘客、之后新生成乘客拆开；三者严格相加。真实未来只由环境逐步揭示给策略，但候选标签是事后比较，不能直接作为在线可得特征。局部成本差不是全程最优Q，也不能跨状态相加。',
              md_table(['K','样本层','状态数','120min存在更好候选','平均局部差/人分钟','15min全部候选并列','60→120严格排序冲突','个体/系统最优集合不交','每个系统最佳都使当前人更差','源回报/人时最优不交'],
                       [[r['k'],r['layer'],r['states'],r['better_states'],r['mean_local_gap'],r['all_tied_15'],r['strict_flip_60_120'],r['focal_system_disjoint'],r['every_best_hurts_focal'],r['source_cost_disjoint']] for r in local]),
              f"普通随机层总计{len(total_random)}状态，补样层不混入频率估计。所有比率仍是这12个模型、这一固定需求上的描述；相邻状态和同模型分支相关，不构造伪大样本置信区间。",
              '本次432个状态的全部候选在15分钟累计未完成人时上并列。该指标只在完成人数发生变化时区分动作；当前乘客尚需接驳和至少约19分钟载客飞行，因此过短窗口天然难以区分。本结果首先是诊断窗口不足，而不是网络没有学会时序的证据。“短窗口看不到”不等于“最优动作被反转”。严格排序冲突用两个窗口的最优动作集合不相交定义，处理了并列。源回报比较采用原gamma0.99的未归一化SOURCE回报，不等同完整训练中VecNormalize后的优势或PPO更新因果效果。']
    if examples:
        ex = max(examples, key=lambda e:e['cost_gap_120'])
        parts += ['### 一条确实存在的外部成本案例',
                  f"{ex['id']}，决策序号{ex['index']}，时间{ex['time']}。原选候选{ex['original_action']}，替换候选{ex['better_action']}。所有120分钟系统最佳候选都让当前乘客的人时更高，排除了挑并列解制造冲突。",
                  md_table(['120分钟成本','原动作','替换动作','减少量'],[[key,ex['original'][key],ex['alternative'][key],ex['original'][key]-ex['alternative'][key]] for key in ('cost','focal','existing_other','future')]),
                  '为便于阅读，这里展示的是严格冲突案例中系统成本差最大的一条，不能代表平均效果；总体分层统计见上表。这说明“只让当前人快”不一定等于“系统少等”；但不能仅靠这个例子宣布交通文献普遍忽略外部性。详细队列、已承诺ETA、飞机状态保留在局部外部成本案例.json。']
    parts += ['## 五、时间信息：确认了接口缺失，没有确认主因',
              f"逐一审计{sum(a['inspected_decisions'] for a in audits)}次决策的6帧源输入，共{sum(a['inspected_time_values'] for a in audits)}个充电均值/最短充电/飞行剩余时间值；非零值{sum(a['nonzero_time_values'] for a in audits)}个。与此同时，真实物理记录中存在充电或返航尚未就绪的状态。",
              '因此“通道运行时确实没有传入这些时间”已经验证，不再只是静态猜想。但还没有隔离：补回时间能改变多少候选排序；模型历史是否已经补偿了这部分；源时间信息缺失解释了多少ATT。不能写成STIN/Transformer把ETA压没了——当前源风格模型是LSTM，信息首先就没进来。',
              f"在432个带分支标签的样本里，同一模型的完整6帧源输入精确重复组为{sum(a['exact_repeated_input_groups_n2'] for a in audits)}，同输入但120分钟最佳集合不交的状态对为{sum(a['exact_input_disjoint_label_pairs_n2'] for a in audits)}。这里只是精确重合筛查；没有反例不证明观测充分，也不能用“相似状态”冒充严格混叠证明。",
              '## 六、假说现在如何处理',
              md_table(['问题','本次证据','处理'],[
                  ['可提升空间是否存在','离线同物理、同LQ找到更低人时且全部完成的方案','成立于这条需求；不等于足够大的在线可学gap'],
                  ['尾批是否是重要竞争解释','K3/K4 READY修补末期去向就恢复完整并降低RPT','优先单独解释，不能全算时序'],
                  ['个体与系统代价是否冲突','同状态、相同后续策略的成本分解有严格冲突案例','可保留为候选机制，须补全程干预和强规则对照'],
                  ['短窗口是否会误判信息价值','15分钟大量并列，较长窗口能区分','支持扩大诊断窗口，不直接证明表征失败'],
                  ['时间通道是否确实缺失','运行时全零，物理剩余时间非零','事实成立；性能因果未定'],
                  ['源目标是否可能与人时排序不一致','局部SOURCE回报与人时的最佳集合可比较','旧奖励竞争解释；新PT奖励需单独确认'],
                  ['普通PPO/Transformer是否不能补齐','仅冻结LSTM源模型的事后比较','不能下这个结论'],
                  ['原因能否可靠排名','缺少匹配全程修补及独立需求，候选机制重叠','暂不能报原因百分比或主因冠军'],
                  ['Joint卡死是否由协作失败导致','本包全部Single-LQ','本次没有检验，不外推'],
              ]),
              '## 七、下一步仍先无训练，不再立即排新模型',
              '第一，当前eval123上已对4个源模型和READY分别执行了相同尾批修补，不重复该工作；后续把这个匹配对照扩到第二初始化和独立合法需求，再看去掉尾批效应后的差距。同时保留不允许事后改去向的原版。这里的修补仅是诊断，不改正式服务规则。',
              '第二，在相同抽样状态上比较合法在线候选评分/短滚动搜索与真未来评分，检验当前承诺信息能回收多少收益；必须封住未揭示CSV，并计运行时间。若强在线解析已回收大部分，就不应再为同一损失开发复杂RL。',
              '第三，优先在已修正奖励的30M Single/Joint源模型上复查上述决策冲突，再决定这是旧奖励、尾批还是仍存在的系统协调问题。没有对应模型包，本次不能完成。',
              '第四，只有在独立需求与多个已训模型上，修正奖励、排除尾批、匹配强在线对照后仍有稳定且实际有意义的剩余gap，才设计少量训练：回答新增信息能否被原基座利用，或有限预算策略为何学不到。若gap很小或只靠真未来，应收缩/更换问题，不继续堆结构。',
              '目前还没有证明新AI必要，也没有通过无训练完成可排名归因。这是本轮明确的边界，不用“找到一些异常”替代前置证明。',
              '## 八、复核材料与执行审计',
              '输入SHA256：`' + EXPECTED + '`。文件943124945字节，200个zip与200个pkl同名配对；24组被选权重内部CRC通过，全部48个权重/归一化文件回放后哈希不变。原E盘包、项目未改；没有SSH，没有启动训练。',
              '实际执行：N0 48次模型整程回放；N1 24次在线规则整程；N2 432状态/1296候选分支；N3 538个不同离线候选整程+3次READY暖启动复核。训练新增0步。离线搜索动作序列已保留，可逐步复评。',
              '初始N0曾因错误要求983040必须有历史周期评估而停止；随后确认该点为resume点、并非周期评估点，修正读取逻辑后全部完成。旧N0_BLOCKED.json和worker.log原样保留，代表已解决的工具问题，不代表最终存在复评不一致。',
              '文件：逐回放48条.csv、逐模型12条.csv、规则逐回放24条.csv、逐状态诊断432条.csv、局部分支成本.csv、离线搜索全部候选.csv、离线全程方案.csv、输入时间通道审计.csv、POST_INPUT_HASH_CHECK.json、各N阶段原记录；代码run.py与PLAN.md。',
              '统计方法借用了Scientific Agent Skills的独立单位、探索/验证区分与不夸大因果原则；未调用其Python3.12统计示例或安装依赖，当前回放用用户既有Python3.11环境，汇总用标准库描述统计。过程参考：Kassis, Agarwal, He, Patel & Brueckner (2026), [Scientific Agent Skills](https://arxiv.org/abs/2609.00065)，当前记录为2026-09-02修订版；这不是支持UAM机制的领域证据。']
    return '\n\n'.join(parts) + '\n'


def render_report(out, markdown):
    """输出独立可打开页面；所有文字转义，不依赖外部脚本或网络。"""
    blocks = []
    for block in markdown.split('\n\n'):
        if block.startswith('|'):
            lines = block.splitlines()
            rows = [line.strip('| ').split(' | ') for line in lines]
            header = '<tr>' + ''.join('<th>' + html.escape(x) + '</th>' for x in rows[0]) + '</tr>'
            body = ''.join('<tr>' + ''.join('<td>' + html.escape(x) + '</td>' for x in row) + '</tr>' for row in rows[2:])
            blocks.append('<div class="scroll"><table>' + header + body + '</table></div>')
        elif block.startswith('#'):
            level = len(block) - len(block.lstrip('#'))
            blocks.append(f'<h{level}>' + html.escape(block[level:].strip()) + f'</h{level}>')
        else:
            value = html.escape(block)
            # 仅将报告固定的公开过程参考转换为链接。
            value = value.replace('[Scientific Agent Skills](https://arxiv.org/abs/2609.00065)', '<a href="https://arxiv.org/abs/2609.00065">Scientific Agent Skills</a>')
            blocks.append('<p>' + value + '</p>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>严格E1 无训练诊断</title><style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f4f7fa;color:#17283b;line-height:1.8}main{max-width:1120px;margin:32px auto;padding:38px;background:white}h1{font-size:28px}h2{margin-top:44px;border-bottom:2px solid #dbe6ee;padding-bottom:8px}h3{margin-top:30px}p{overflow-wrap:anywhere}table{border-collapse:collapse;font-size:14px;width:100%;margin:15px 0}th,td{border:1px solid #d9e3eb;padding:9px 12px;text-align:left;white-space:nowrap}th{background:#eaf2f8}tr:nth-child(even){background:#f7fafc}.scroll{overflow:auto}a{color:#075d9a}@media(max-width:800px){main{margin:0;padding:20px}}</style><main>' + ''.join(blocks) + '</main></html>'
    (out / '无训练诊断_结论与证据.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive")
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase", choices=("prepare", "n0", "n1", "n2", "n3", "report"), default="n0")
    parser.add_argument("--worker")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    out = safe_output(args.out)
    if args.worker:
        worker(args.worker)
        return
    if args.phase == 'report':
        report(out)
        return
    state = prepare(Path(args.archive), out)
    if args.phase == "prepare":
        return
    results = []
    jobs = state['jobs']
    if args.phase in ('n1', 'n3'):
        jobs = [j for j in jobs if j['task']['seed'] == 601]
    if args.phase in ('n1', 'n2', 'n3') and len(read(out / 'N0_SUMMARY.json')) != 12:
        raise RuntimeError('12个模型N0未通过，不启动后续诊断')
    for index, original_job in enumerate(jobs[:args.limit]):
        job = dict(original_job)
        if args.phase != 'n0':
            job.update(phase=args.phase, n0=original_job['out'], rules_dir=str(out / 'N1' / original_job['id']),
                       out=str(out / args.phase.upper() / original_job['id']))
        target = Path(job["out"])
        endname = args.phase.upper() + '_END.json'
        if (target / endname).exists():
            results.append(read(target / endname))
            continue
        save(target / "JOB.json", job)
        log = target / "worker.log"
        print(f"[{args.phase.upper()} {index+1}/{min(args.limit,len(jobs))}] {job['id']}", flush=True)
        with log.open("a", encoding="utf-8") as f:
            process = subprocess.run([sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--out", str(out),
                                      "--worker", str(target / "JOB.json")], stdout=f, stderr=subprocess.STDOUT,
                                     cwd=target, timeout=1200 if args.phase in ('n2', 'n3') else 180)
        if process.returncode:
            save(out / (args.phase.upper() + '_BLOCKED.json'), dict(id=job["id"], returncode=process.returncode, log=str(log)))
            raise RuntimeError("模型复评失败，请检查" + str(log))
        results.append(read(target / endname))
        save(out / "progress.json", dict(phase=args.phase.upper(), completed_models=len(results), total_models=min(args.limit,len(jobs)),
                                        new_training_transitions=0))
    save(out / (args.phase.upper() + '_SUMMARY.json'), results)
    print('[' + args.phase.upper() + '完成]', len(results), '模型/场景', flush=True)


if __name__ == "__main__":
    main()
