#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""固定 E1/K3，冻结三类策略，给 S、AC 与决策型世界模型做无训练方向筛查。"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import io
import itertools
import json
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
import pickle
import random
import re
import shutil
from statistics import mean
import sys
import tarfile
import time
import traceback
import types


VERSION = 'UAM_HARL_TRANSFER_REPLAY_20261109_V1'
ARCHIVE = 'uam_core_replay_12models_20261109_v1.tar.gz'
ARCHIVE_SHA = '5ddb179c3dc692d42d99f615eaa9142c6ec2038dc22030a3e95c0b2eac1b5dc9'
ENTRY = 'train_uam_happo_mappo_dreamer_140m_ONEFILE.py'
CORE = 'serial_runs/uam_core_author_20261106_v1'
PIN = 'b1af98b0dbab72a2eee9d160751cd09aedbb8ce2'
ARMS = ('HAPPO_RANDOM', 'MAPPO_MATCHED', 'PPO_TF')
SEEDS = (2301, 2302)
TRACES = ('order_4301', 'order_4302', 'order_8301', 'order_8302')
HORIZONS = (5, 15, 30)
EPS = 5  # 固定窗口的系统人分钟差门槛；不是显著性检验。
DESIGN_SEED = 6110927
SOURCES = {
    'HARL': 'https://jmlr.org/papers/v25/23-0488.html',
    '牛鞭论文': 'https://doi.org/10.1177/10591478241305863',
    'ASE': 'https://proceedings.mlr.press/v235/triantafyllou24a.html',
    'CED': 'https://proceedings.mlr.press/v267/triantafyllou25a.html',
    '价值等价': 'https://arxiv.org/abs/2011.03506',
}


class BudgetStop(Exception):
    pass


def guard(job):
    if time.monotonic() >= job['deadline'] or (Path(job['out']) / 'STOP_REQUESTED').exists():
        raise BudgetStop('到达墙钟预算或收到暂停标记')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.pending')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)


def stable_seed(*parts):
    blob = '|'.join(map(str, (DESIGN_SEED, *parts))).encode('utf-8')
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], 'little')


def cell(arm, seed):
    return f'WM_MATCHED__E1__K3__{arm}__s{seed}'


def prepare_inputs(archive, out):
    # 模型与原入口含 pickle/可执行源码，只允许已核验的自有模型包，不接受陌生包。
    if digest(archive) != ARCHIVE_SHA:
        raise ValueError('模型包 SHA-256 与已核验版本不同；不加载模型或源码')
    allowed_cells = {cell(a, s) for a in ARMS for s in SEEDS}
    allowed = {ENTRY, 'REPLAY_MANIFEST.json', f'{CORE}/BANK.json'}
    allowed.update(f'{CORE}/bank/{t}.csv' for t in TRACES)
    input_dir = out / 'input'
    input_dir.mkdir(exist_ok=True)
    seen = set()
    with tarfile.open(archive, 'r|gz') as tf:
        for member in tf:
            name = member.name
            parts = PurePosixPath(name).parts
            selected = name in allowed or (
                len(parts) == 5 and '/'.join(parts[:3]) == f'{CORE}/cells'
                and parts[3] in allowed_cells
                and (parts[-1] in {'TASK.json', 'END.json', 'RESUME.json', 'CONFIG.json'}
                     or re.fullmatch(r'(?:RESUME|R_[0-9]+_[0-9]+)(?:\.norm)?\.(?:zip|pkl)', parts[-1])))
            if not selected:
                continue
            if not member.isfile() or name in seen or member.size > 100 * 1024**2:
                raise ValueError(f'归档白名单成员不安全或重复：{name}')
            target = input_dir.joinpath(*parts)
            if not target.resolve().is_relative_to(input_dir.resolve()) or target.is_symlink():
                raise ValueError(f'输入缓存存在路径重定向：{name}')
            seen.add(name)
            with tf.extractfile(member) as f:
                data = f.read()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.read_bytes() == data:
                continue
            tmp = target.with_name(target.name + '.extract.pending')
            if tmp.is_symlink():
                raise ValueError('输入临时路径是链接')
            tmp.write_bytes(data)
            os.replace(tmp, target)
    if allowed - seen:
        raise ValueError(f'模型包缺少固定输入：{sorted(allowed - seen)}')
    archived = {x['cell']: x for x in read(input_dir / 'REPLAY_MANIFEST.json')['models']}
    result = {}
    for seed, arm in itertools.product(SEEDS, ARMS):
        ident = cell(arm, seed)
        folder = input_dir / CORE / 'cells' / ident
        task, meta, end = (read(folder / n) for n in ('TASK.json', 'RESUME.json', 'END.json'))
        if (task['id'] != ident or task['seed'] != seed or task['arm'] != arm
                or task['environment'] != 'E1' or task['k'] != 3 or task['fleet'] != 12
                or task['control'] != 'J3' or task['reward_mode'] != 'PT_POST'
                or task['method'] != 'J3__M0_BASE' or task['network'] != 'TRANSFORMER'
                or not task.get('space', False) or task.get('temporal', False)
                or end['status'] != 'COMPLETE' or meta['real_steps'] != 1_003_520
                or archived[ident]['steps'] != 1_003_520):
            raise ValueError(f'模型不在同一 E1/K3、J3、约一百万步的比较合同：{ident}')
        for key in ('model', 'norm'):
            target = folder / meta[key]
            relative = target.relative_to(input_dir).as_posix()
            listed = archived[ident]['assets'][key]
            if (target.parent != folder or relative not in seen or relative != listed['path']
                    or meta[key + '_sha'] != listed['sha256'] or digest(target) != listed['sha256']):
                raise ValueError(f'模型与归一化的哈希或路径不匹配：{ident}/{key}')
        result[ident] = dict(folder=str(folder), task=task, pointer=meta)
    bank = {v['name']: v for v in read(input_dir / CORE / 'BANK.json')}
    trace_paths = {}
    for name in TRACES:
        path = input_dir / CORE / 'bank' / (name + '.csv')
        if digest(path) != bank[name]['sha256']:
            raise ValueError(f'固定需求哈希不符：{name}')
        trace_paths[name] = str(path)
    return input_dir / ENTRY, result, trace_paths


def archive_runtime(entry):
    # 只加载可信归档里的冻结环境定义；删除训练入口，绝不启动旧训练矩阵。
    source = Path(entry).read_text(encoding='utf-8-sig')
    tree = ast.parse(source, filename=str(entry))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.If) and '__name__' in ast.unparse(n.test))]
    module = types.ModuleType('uam_archive_core_replay')
    module.__file__ = str(entry)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(entry), 'exec'), module.__dict__)
    return module, module.base_module()


def read_cpu_pickle(path, torch):
    # 作者快照用普通 pickle 保存，训练设备是 CUDA；重定向底层张量存储到 CPU。
    class CPUReader(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ('torch.storage', '_load_from_bytes'):
                return lambda blob: torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
            return super().find_class(module, name)

    with Path(path).open('rb') as f:
        return CPUReader(f).load()


class FrozenActor:
    """按固定 HARL 前向结构读入权重，不创建优化器，不依赖另外下载 HARL。"""

    def __init__(self, weights, args, torch):
        if (args['activation_func'] != 'relu' or args['use_naive_recurrent_policy']
                or args['use_recurrent_policy'] or args['hidden_sizes'] != [128, 128]
                or not args['use_feature_normalization']):
            raise ValueError('快照不是已核验的 HARL 两层前馈离散策略')
        self.torch = torch
        self.w = {k: v.detach().cpu() for k, v in weights.items()}
        self.norms = ('base.feature_norm', 'base.mlp.fc.2', 'base.mlp.fc.5')
        self.linears = ('base.mlp.fc.0', 'base.mlp.fc.3', 'act.action_out.linear')
        expected = {f'{k}.{end}' for k in self.norms + self.linears for end in ('weight', 'bias')}
        if set(self.w) != expected or tuple(self.w[self.linears[-1] + '.weight'].shape) != (3, 128):
            raise ValueError(f'HARL 快照结构不同，拒绝用近似前向：{sorted(set(self.w) ^ expected)}')
        if any(not torch.isfinite(v).all() for v in self.w.values()):
            raise ValueError('HARL 快照权重非有限')

    def probs(self, vector):
        f, w = self.torch.nn.functional, self.w

        def ln(x, key):
            b = w[key + '.bias']
            return f.layer_norm(x, b.shape, w[key + '.weight'], b)

        def linear(x, key):
            return f.linear(x, w[key + '.weight'], w[key + '.bias'])

        x = ln(vector, self.norms[0])
        for index in range(2):
            x = ln(f.relu(linear(x, self.linears[index])), self.norms[index + 1])
        return self.torch.softmax(linear(x, self.linears[-1]), dim=-1)


class Policies:
    def __init__(self, records, base, pack):
        import numpy as np
        import torch
        from stable_baselines3 import PPO
        self.torch, self.np = torch, np
        self.models = {}
        base.torch_features(pack)
        for key, rec in records.items():
            directory, meta = Path(rec['folder']), rec['pointer']
            norm = read_cpu_pickle(directory / meta['norm'], torch)
            norm.training = False
            norm.norm_reward = False
            if rec['task']['arm'] == 'PPO_TF':
                net = PPO.load(directory / meta['model'], device='cpu').policy
                net.set_training_mode(False)
                actors = None
            else:
                cfg = read(directory / 'CONFIG.json')
                if cfg['author_commit'] != PIN or cfg['arm'] != rec['task']['arm']:
                    raise ValueError('HARL 提交或模型类型不匹配')
                data = read_cpu_pickle(directory / meta['model'], torch)
                if data['real_steps'] != meta['real_steps'] or len(data['actors']) != 2:
                    raise ValueError('HARL 快照步数或角色数不符')
                actors = tuple(FrozenActor(w, cfg['args'], torch) for w in data['actors'])
                net = None
            self.models[key] = (norm, net, actors)

    def probs(self, model, obs, phase):
        norm, net, actors = self.models[model]
        if int(round(float(obs[-8]))) != phase:
            raise ValueError('策略输入与乘客/飞机阶段不一致')
        x = norm.normalize_obs(self.np.asarray(obs, self.np.float32)[None])
        with self.torch.no_grad():
            x = self.torch.as_tensor(x, dtype=self.torch.float32, device='cpu')
            p = (net.get_distribution(x).distribution.probs if net is not None else actors[phase].probs(x))
        v = tuple(float(i) for i in p[0].tolist())
        if len(v) != 3 or not all(math.isfinite(x) and x >= 0 for x in v) or abs(sum(v) - 1) > 1e-5:
            raise ValueError('冻结策略输出不是有效的三分类概率')
        return v


def frozen_draw(probs, u):
    cumulative = 0.
    for i, prob in enumerate(probs):
        cumulative += prob
        if u < cumulative:
            return i
    return len(probs) - 1


def probs_info(p):
    return dict(probs=list(p), entropy=-sum(x * math.log(x) for x in p if x > 0),
                argmax=max(range(len(p)), key=p.__getitem__))


def clone_env(original, m):
    # 既有克隆修复还不够：这里多包了一层空返门，必须一起重绑到克隆对象。
    src = original.env
    if original.phase != 0 or getattr(src, '_generator', None) is not None:
        raise ValueError('不能克隆暂停在 prepare/finish 中间的环境')
    cloned = copy.deepcopy(original, {id(m): m})
    env, gate, scenario = cloned.env, cloned.gate, cloned.env.scenario
    if env is src or scenario is src.scenario or env.raw.env is not scenario:
        raise ValueError('克隆环境/底层 scenario 没有隔离')
    if getattr(env, 'legacy', None) is not None and env.legacy.scenario is not scenario:
        raise ValueError('克隆 legacy 仍绑定原 scenario')
    fn = getattr(src._old_dispatch, '__func__', None)
    if fn is None or getattr(src._old_dispatch, '__self__', None) is not src.scenario:
        raise ValueError('旧空返入口绑定无法核验')
    env._old_dispatch = types.MethodType(fn, scenario)
    gate.env = env
    gate.original = env._dispatch_hook
    scenario._strict_dispatch_returns = types.MethodType(gate.hook, scenario)
    if (scenario._strict_dispatch_returns.__self__ is not scenario
            or scenario._strict_dispatch_returns.__func__.__self__ is not gate
            or gate.original.__self__ is not env):
        raise ValueError('克隆空返入口重绑失败')
    fn = getattr(env, '_generator_function', None)
    if fn is not None and (getattr(fn, '__self__', None) is not scenario
                           and (getattr(fn, '__self__', None) is not None or getattr(fn, '__closure__', None))):
        raise ValueError('克隆分钟推进函数仍引用外部状态')
    return cloned


def physical(env):
    s = env.env.scenario
    aircraft = []
    for key, e in s._all_evtols.items():
        aircraft.append((str(key), e.state.name, str(e.current_vertiport_id),
                         str(getattr(e, 'target_vertiport_id', '')), round(float(e.battery_kwh), 9),
                         round(float(getattr(e, 'remaining_time', 0) or 0), 9), tuple(map(str, e.passenger_ids))))
    persons = []
    for key, p in s.persons.persons.items():
        persons.append((str(key), p.state, p.sub_state, str(getattr(p, 'origin_vertiport_id', '')),
                        round(float(getattr(p, 'current_timer', 0) or 0), 9)))
    queues = [(str(k), tuple(map(str, v.person_list))) for k, v in s.vertiports.vertiport_list.items()]
    return (float(s.time), tuple(sorted(aircraft)), tuple(sorted(persons)), tuple(sorted(queues)),
            tuple(sorted(map(str, s.finished_ids))), tuple(map(str, s.waiting_decisions)))


def rng_get(policies):
    return random.getstate(), policies.np.random.get_state(), policies.torch.get_rng_state()


def rng_set(policies, state):
    random.setstate(state[0])
    policies.np.random.set_state(state[1])
    policies.torch.set_rng_state(state[2])


def active_info(env, m):
    s = env.env.scenario
    active = bool(s.waiting_decisions)
    ready = sum(e.state.name == 'IDLE' and not e.passenger_ids
                for e in s.vertiports.evtols_at_vertiport.get('2', []))
    queues = sum(bool(s.vertiports.vertiport_list[str(v)].person_list) for v in m.CANDIDATES[3])
    return active, ready, queues


def legal_post(env, m):
    _, ready, queues = active_info(env, m)
    mask = tuple(i for i, v in enumerate(env.env._legal) if v > .5)
    if not mask:
        raise ValueError('实际飞机合法集合为空')
    return mask, bool(ready and queues and not env.gate.release and not env.gate.remaining)


def step_minute(ctx, env, obs, mode, tape, forced=None):
    s, policies, base, m = env.env.scenario, ctx['policies'], ctx['base'], ctx['m']
    tick = int(s.time)
    u = tape[tick]
    active = bool(s.waiting_decisions)
    if forced is not None:
        p = forced[0]
    elif mode == 'READY':
        p = ctx['core'].ready_choice(env.env, m)
    else:
        p = frozen_draw(policies.probs(mode, obs, 0), u[0])
    post, reward, term, trunc, info = env.step(int(p))
    if reward or term or trunc or info['physical_increment'] != 0:
        raise ValueError('乘客准备阶段不应推进物理时钟')
    legal, meaningful = legal_post(env, m)
    if forced is not None:
        a = forced[1]
        if a not in legal or (not active and p != 0):
            raise ValueError('反事实要求强制一个不存在的动作')
    elif mode == 'READY':
        a = base.lq_target(env.env)
    else:
        a = frozen_draw(policies.probs(mode, post, 1), u[1])
    old_events = len(env.gate.events)
    obs, _, term, trunc, info = env.step(int(a))
    if info['physical_increment'] != 1 or int(s.time) != tick + 1:
        raise ValueError('飞机阶段没有恰好推进一个物理分钟')
    row = env.ledger['decision_log'][-1]
    actual = tuple(sorted((str(y['aircraft']), str(y['target']))
                          for e in env.gate.events[old_events:] if e['kind'] == '实际空返'
                          for y in e['executed']))
    detail = dict(p=int(row['p']), requested=int(a), applied=int(row['applied_target']), legal=list(legal),
                  meaningful=meaningful, passenger_active=active, actual_dispatch=actual)
    if forced is not None and (detail['p'], detail['applied']) != tuple(forced):
        raise ValueError('强制动作被物理投影改写')
    return obs, bool(term or trunc), detail


def supply_proxy(env, m):
    # 同一组已知事件：比较决策时的快照与按当前乘客实际入队时刻切片。
    # 两个分数都是未训练的近似代理，不把飞行到站时间当作实际可服务时间标签。
    s = env.env.scenario
    if not s.waiting_decisions:
        return None
    focal = s.persons.persons[s.waiting_decisions[0]]
    rows = []
    for vid in map(str, m.CANDIDATES[3]):
        v = s.vertiports.vertiport_list[vid]
        access = float(s.vehicles.estimate_travel_time(focal.origin_position, v.vertiport_position))
        km = s.vertiports._calculate_distance(v.vertiport_position, s.vertiports.vertiport_list['2'].vertiport_position)
        flight = float(km) / 200 * 60
        charged = []
        for e in s.vertiports.evtols_at_vertiport.get(vid, []):
            if e.passenger_ids or e.state.name not in ('IDLE', 'CHARGING'):
                continue
            residual = max(0., float(e.spec.battery_capacity_kwh - e.battery_kwh))
            charged.append(0. if e.state.name == 'IDLE' else residual / max(float(e.spec.charge_rate_kwh_per_min), 1e-9))
        arrivals = [max(0., float(p.current_timer or 0)) for p in s.persons.persons.values()
                    if p.state == 'enroute' and p.sub_state == 'to_vertiport'
                    and str(getattr(p, 'origin_vertiport_id', '')) == vid]
        # 同一函数只替换需求与运力的读取时钟；不改动作权限、奖励和需求。
        cycle = 2 * flight + 2 * float(km) * .55 / 5.32

        def score(t):
            prior = len(v.person_list) + sum(tau <= t for tau in arrivals)
            ready = sum(tau <= t for tau in charged)
            wait = min((max(0., tau - t) for tau in charged), default=cycle / 2.)
            return access + flight + wait + (prior // 4) * cycle / max(1, ready)

        rows.append(dict(station=vid, access=access, queued=len(v.person_list),
                         committed_by_access=sum(tau <= access for tau in arrivals),
                         station_ready_times=charged, now=score(0.), aligned=score(access)))
    return rows


def enumeration(ctx, template, obs, models, tape, rng):
    s = template.env.scenario
    u = tape[int(s.time)]
    p_legal = (0, 1, 2) if s.waiting_decisions else (0,)
    info = {}
    for mode in ('READY', *models):
        p_probs = None if mode == 'READY' else ctx['policies'].probs(mode, obs, 0)
        p = (ctx['core'].ready_choice(template.env, ctx['m']) if mode == 'READY' else frozen_draw(p_probs, u[0]))
        info[mode] = dict(p=p if len(p_legal) > 1 else 0, passenger=probs_info(p_probs) if p_probs is not None else None)
    choices = []
    for p in p_legal:
        rng_set(ctx['policies'], rng)
        branch = clone_env(template, ctx['m'])
        try:
            post, *_ = branch.step(p)
            legal, meaningful = legal_post(branch, ctx['m'])
            for mode, row in info.items():
                if row['p'] != p:
                    continue
                prob = None if mode == 'READY' else ctx['policies'].probs(mode, post, 1)
                a = ctx['base'].lq_target(branch.env) if prob is None else frozen_draw(prob, u[1])
                lq = ctx['base'].lq_target(branch.env)
                applied = a if a in legal else lq if lq in legal else legal[0]
                row.update(requested=a, projected=applied, aircraft=probs_info(prob) if prob is not None else None,
                           legal=list(legal), meaningful=meaningful,
                           illegal_mass=(sum(prob[i] for i in range(3) if i not in legal) if prob is not None else None))
            # 无可执行目标时用一个物理代表，同时保留该角色实际请求与投影。
            if meaningful:
                choices.extend((p, a) for a in legal)
            else:
                lq = ctx['base'].lq_target(branch.env)
                choices.append((p, lq if lq in legal else legal[0]))
            for row in info.values():
                if row['p'] == p:
                    row['pair'] = [p, row['projected'] if meaningful else choices[-1][1]]
        finally:
            # 克隆中的物理分钟生成器只被推进到准备阶段，不继续使用。
            branch.close()
    rng_set(ctx['policies'], rng)
    return choices, info


def alternate_suffix(template, rng_seed):
    # 这是真值标签的敏感性试验，不是给选择器的预测。只重排后 30 分钟 OD，保留到达时刻。
    s = template.env.scenario
    if s.persons.spawn_file is None or not isinstance(s.persons.file_data, dict):
        raise ValueError('固定文件需求模式未就绪')
    data = copy.deepcopy(s.persons.file_data)
    times = sorted(k for k in data if s.time < k <= s.time + HORIZONS[-1])
    positions = [(t, i) for t in times for i in range(len(data[t]))]
    od = [data[t][i] for t, i in positions]
    shuffled = od.copy()
    random.Random(rng_seed).shuffle(shuffled)
    for (t, i), val in zip(positions, shuffled):
        data[t][i] = val
    return data, sum(x != y for x, y in zip(od, shuffled))


def rollout(ctx, job, template, obs, tape, rng, mode, choice=None, scope='real', suffix=None):
    rng_set(ctx['policies'], rng)
    env = clone_env(template, ctx['m'])
    s = env.env.scenario
    if scope != 'real':
        if s.persons.spawn_file is None:
            raise ValueError('不能在非文件需求模式屏蔽未揭示请求')
        s.persons.file_data = {} if scope == 'committed' else copy.deepcopy(suffix)
    focal = str(s.waiting_decisions[0]) if s.waiting_decisions else None
    initial = set(map(str, s.persons.persons))
    tick = int(s.time)
    start_cost = env.ledger['area']
    initial_person_cost = env.ledger['individual'].copy()
    checkpoints, events, waits, fingerprint = {}, [], {}, None
    first = None
    try:
        for offset in range(1, HORIZONS[-1] + 1):
            if offset % 5 == 1:
                guard(job)
            obs, done, detail = step_minute(ctx, env, obs, mode, tape, choice if offset == 1 else None)
            if offset == 1:
                first = detail
            if focal is not None and focal in s.persons.persons:
                p = s.persons.persons[focal]
                phase = str(p.sub_state)
                if phase not in ('None', 'unknown'):
                    waits.setdefault(phase, int(s.time) - tick)
            if detail['actual_dispatch']:
                events.extend([offset, a, v] for a, v in detail['actual_dispatch'])
            if offset in HORIZONS or done:
                costs = {p: int(v - initial_person_cost.get(p, 0)) for p, v in env.ledger['individual'].items()
                         if v != initial_person_cost.get(p, 0)}
                total = int(env.ledger['area'] - start_cost)
                if sum(costs.values()) != total:
                    raise ValueError('逐人人时与分支积分不守恒')
                checkpoints[str(offset)] = dict(cost=total, focal=costs.get(focal, 0),
                    other_known=sum(v for p, v in costs.items() if p != focal and p in initial),
                    unrevealed=sum(v for p, v in costs.items() if p not in initial),
                    finished=len(s.finished_ids), generated=len(s.persons.persons))
            if done:
                break
        if offset != HORIZONS[-1]:
            # 只有所有已生成乘客已完成时才能零填充；其他提前停止视作不可比。
            if len(s.finished_ids) != len(s.persons.persons) or scope != 'committed':
                raise ValueError('分支在预定窗口前非吸收终止，拒绝补造结果')
            for h in HORIZONS:
                checkpoints.setdefault(str(h), dict(checkpoints[str(offset)], absorbed=True))
        fingerprint = hashlib.sha256(repr(physical(env)).encode('utf-8')).hexdigest()
        return dict(points=checkpoints, first=first, dispatch=events, focal_observed_stage_time=waits,
                    endpoint=fingerprint, person_time=costs, simulated_minutes=offset)
    finally:
        env.close()


def cost(result, h=30):
    return result['points'][str(h)]['cost']


def pick(values, preferred=None):
    return min(values, key=lambda k: (values[k], k != preferred, k))


def diagnose_state(ctx, job, env, obs, tape, rng, models, ident, stratum):
    initial = physical(env)
    scores = supply_proxy(env, ctx['m'])
    choices, policy = enumeration(ctx, env, obs, models, tape, rng)
    real, known, alt = {}, {}, {}
    suffix, suffix_changed = alternate_suffix(env, stable_seed(ident, 'suffix'))
    for choice in choices:
        guard(job)
        real[choice] = rollout(ctx, job, env, obs, tape, rng, 'READY', choice)
        known[choice] = rollout(ctx, job, env, obs, tape, rng, 'READY', choice, 'committed')
        if suffix_changed:
            alt[choice] = rollout(ctx, job, env, obs, tape, rng, 'READY', choice, 'alternate', suffix)
    ref = tuple(policy['READY']['pair'])
    check = rollout(ctx, job, env, obs, tape, rng, 'READY')
    for key in ('points', 'endpoint', 'person_time', 'dispatch'):
        if check[key] != real[ref][key]:
            raise ValueError(f'共同状态事实重放与同动作干预不一致：{ident}/{key}')
    for choice, branch in known.items():
        if branch['points']['30']['unrevealed'] or branch['points']['30']['generated'] != len(initial[2]):
            raise ValueError('已承诺分支出现未揭示乘客或初始队列丢失')
        if branch['first']['actual_dispatch'] != real[choice]['first']['actual_dispatch']:
            raise ValueError('首分钟调机受未来分支干预影响，停止解释')
    if physical(env) != initial:
        raise ValueError('干预回放污染共同起点')
    costs = {k: cost(v) for k, v in real.items()}
    known_costs = {k: cost(v) for k, v in known.items()}
    alt_costs = {k: cost(v) for k, v in alt.items()}
    joint_best = pick(costs, ref)
    known_best = pick(known_costs, ref)
    alt_best = pick(alt_costs, ref) if alt else None
    # AC 只对实际执行不同、且在两个乘客动作下都存在的目标构造真正的 2×2。
    interaction = []
    for p0, p1 in itertools.combinations(sorted({p for p, _ in choices}), 2):
        common = sorted({a for p, a in choices if p == p0} & {a for p, a in choices if p == p1})
        for a0, a1 in itertools.combinations(common, 2):
            cells = ((p0, a0), (p0, a1), (p1, a0), (p1, a1))
            ds = [real[k]['first']['actual_dispatch'] for k in cells]
            if not all(ds) or ds[0] == ds[1] or ds[2] == ds[3]:
                continue
            c00, c01, c10, c11 = [costs[k] for k in cells]
            d0, d1 = c10 - c00, c11 - c01
            interaction.append(dict(actions=cells, costs=[c00, c01, c10, c11], interaction=d1-d0,
                reverses_passenger_rank=(d0 >= EPS and d1 <= -EPS) or (d1 >= EPS and d0 <= -EPS)))
    rows = []
    for model, info in policy.items():
        actual = tuple(info['pair'])
        fixed_air = {k: v for k, v in costs.items() if k[1] == actual[1]}
        real_dispatch = real[actual]['first']['actual_dispatch']
        fixed_p = {k: v for k, v in costs.items() if k[0] == actual[0] and
                   (k == actual or (real_dispatch and real[k]['first']['actual_dispatch']
                                    and real_dispatch != real[k]['first']['actual_dispatch']))}
        p_choice, a_choice = pick(fixed_air, actual), pick(fixed_p, actual)
        options = [min(fixed_air.values()), min(fixed_p.values())]
        supported = len(fixed_air) > 1 and len(fixed_p) > 1
        p_id = info['p']
        s_now = s_aligned = None
        if scores is not None and len(fixed_air) > 1:
            s_now = min(fixed_air, key=lambda x: (scores[x[0]]['now'], x[0] != p_id, x))
            s_aligned = min(fixed_air, key=lambda x: (scores[x[0]]['aligned'], x[0] != p_id, x))
        rows.append(dict(arm=model.split('__')[-2] if model != 'READY' else model, **info,
                         p_options=len(fixed_air), aircraft_options=len(fixed_p),
                         passenger_regret=costs[actual] - costs[p_choice],
                         aircraft_regret=costs[actual] - costs[a_choice],
                         joint_regret=costs[actual] - costs[joint_best],
                         both_needed=bool(supported and costs[actual] - min(options) < EPS
                                          and costs[actual] - costs[joint_best] >= EPS),
                         total_regret_by_horizon={str(h): cost(real[actual], h)-min(cost(v, h) for v in real.values()) for h in HORIZONS},
                         focal_saved=real[actual]['points']['30']['focal'] - real[joint_best]['points']['30']['focal'],
                         other_known_saved=real[actual]['points']['30']['other_known'] - real[joint_best]['points']['30']['other_known'],
                         later_passengers_saved=real[actual]['points']['30']['unrevealed'] - real[joint_best]['points']['30']['unrevealed'],
                         S_changed=s_now is not None and s_now != s_aligned,
                         S_snapshot_choice=s_now, S_aligned_choice=s_aligned,
                         S_gain_over_now=None if s_now is None else costs[s_now]-costs[s_aligned],
                         S_gain_over_actor=None if s_now is None else costs[actual]-costs[s_aligned]))
    # 只在预登记的常规乘客窗口里做续策敏感性，不从结果最好的状态反选样本。
    continuation = []
    if stratum == 'P' and len(choices) > 1:
        for model in models[:2]:
            actual = tuple(policy[model]['pair'])
            # 没有实际空返权限时，规范候选可能合并了多个请求；不拿规范请求冒充策略原请求。
            if actual[1] != policy[model]['projected']:
                continue
            common = {k: v for k, v in costs.items() if k[1] == actual[1] and k[0] != actual[0]}
            if not common:
                continue
            alt_p = min(common)  # 字典序固定替代，不用事后成本挑赢家。
            b = rollout(ctx, job, env, obs, tape, rng, model)
            b_forced = rollout(ctx, job, env, obs, tape, rng, model, actual)
            for key in ('points', 'person_time', 'endpoint', 'dispatch'):
                if b[key] != b_forced[key]:
                    raise ValueError(f'冻结模型续策事实重放不一致：{ident}/{model}/{key}')
            candidate = rollout(ctx, job, env, obs, tape, rng, model, alt_p)
            continuation.append(dict(model=model, factual=actual, alternative=alt_p,
                gain_shared_ready=costs[actual]-costs[alt_p], gain_own_policy=cost(b)-cost(candidate),
                factual_replay_equal=True, simulated_minutes=sum(x['simulated_minutes'] for x in (b, b_forced, candidate))))
    for branch_set in (real, known, alt):
        if len({(v['points']['30']['generated'], v['simulated_minutes']) for v in branch_set.values()}) > 1 and branch_set is not known:
            raise ValueError('同一未来下的分支需求数或窗口长度不匹配')
    branch_rows = []
    for pair in choices:
        row = dict(pair=pair)
        for label, table in (('real', real), ('committed', known), ('alternate', alt)):
            if pair in table:
                row[label] = {k: v for k, v in table[pair].items() if k != 'person_time'}
        branch_rows.append(row)
    diffs = {p: real[ref]['person_time'].get(p, 0) - real[joint_best]['person_time'].get(p, 0)
             for p in sorted(set(real[ref]['person_time']) | set(real[joint_best]['person_time']))}
    result = dict(id=ident, tick=initial[0], stratum=stratum, policy=rows, supply_proxy=scores,
        joint_best=joint_best, committed_best=known_best, alternate_best=alt_best,
        committed_real_regret=costs[known_best]-costs[joint_best],
        committed_alt_regret=None if not alt else alt_costs[known_best]-alt_costs[alt_best],
        robust_residual_same_best=bool(alt and joint_best == alt_best and costs[known_best]-costs[joint_best] >= EPS
                                        and alt_costs[known_best]-alt_costs[alt_best] >= EPS),
        alternate_changed_requests=suffix_changed, clean_2x2=interaction, branches=branch_rows,
        continuation_sensitivity=continuation, READY_vs_ex_post_individual=diffs, reference_replay_equal=True,
        simulated_minutes=sum(v['simulated_minutes'] for t in (real, known, alt) for v in t.values())
                          + check['simulated_minutes'] + sum(x['simulated_minutes'] for x in continuation))
    rng_set(ctx['policies'], rng)
    return result


def init_context(job):
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT
    import torch
    torch.set_num_threads(1)
    core, base = archive_runtime(job['entry'])
    runtime_dir = Path(job['out']) / 'runtime' / str(os.getpid())
    runtime_dir.mkdir(parents=True, exist_ok=True)
    task = job['records'][cell(ARMS[0], SEEDS[0])]['task']
    m, _ = base.runtime(runtime_dir, task)
    phase = base.gym_class(m)
    env = phase(runtime_dir, task, job['trace_path'], 600, 123, False)
    try:
        policies = Policies(job['records'], base, env.pack)
    finally:
        env.close()
    _CONTEXT = dict(core=core, base=base, m=m, task=task, phase=phase,
                    runtime_dir=runtime_dir, policies=policies)
    return _CONTEXT


_CONTEXT = None


def collect(ctx, job, anchor, tape):
    # 用分层蓄水池挑共同状态；样本量不随看到的收益扩大，记录完整机会分母。
    env = ctx['phase'](ctx['runtime_dir'], ctx['task'], job['trace_path'], 600, 123, False)
    policies = ctx['policies']
    random.seed(123)
    policies.np.random.seed(123)
    policies.torch.manual_seed(123)
    rng = random.Random(stable_seed(job['name'], anchor, 'sampling'))
    reservoir, counts = {}, {}
    try:
        obs, _ = env.reset(seed=123)
        while not env.done and int(env.env.scenario.time) <= 269:
            guard(job)
            tick = int(env.env.scenario.time)
            active, ready, queues = active_info(env, ctx['m'])
            if 30 <= tick < 240:
                slot = (0 if tick < 135 else 1)
                groups = []
                if active:
                    groups.append('P')
                if ready and queues >= 2:
                    groups.append('BOTH' if active else 'A')
                for group in groups:
                    key = f'{slot}_{group}'
                    counts[key] = counts.get(key, 0) + 1
                    if rng.randrange(counts[key]) == 0:
                        if key in reservoir:
                            reservoir[key][0].close()
                        reservoir[key] = (clone_env(env, ctx['m']), obs.copy(), rng_get(policies))
            obs, _, _ = step_minute(ctx, env, obs, anchor, tape)
        return reservoir, counts
    except BaseException:
        for snapshot, _, _ in reservoir.values():
            snapshot.close()
        raise
    finally:
        env.close()


def worker(job):
    path = Path(job['out']) / 'blocks' / (job['name'] + '.json')
    result = read(path) if path.is_file() else dict(status='PARTIAL', block=job['name'], seed=job['seed'],
        trace=job['trace'], bank_role='历史开发' if job['trace'].startswith('order_43') else '历史确认重排；本轮不冒称全新分布',
        rows=[], opportunity_minutes={}, errors=[], new_training_transitions=0)
    if result['status'] == 'COMPLETE':
        return result['status'], len(result['rows'])
    models = [cell(a, job['seed']) for a in ARMS]
    seen = {x['id'] for x in result['rows']}
    tape_rng = random.Random(stable_seed(job['name'], 'paired-actions'))
    tape = [(tape_rng.random(), tape_rng.random()) for _ in range(601)]
    try:
        guard(job)
        ctx = init_context(job)
        # 固定 READY 和交替学习策略两类前缀，避免只在某一策略常到达的状态里比较。
        for anchor in ('READY', models[job['alternate_anchor']]):
            guard(job)
            snapshots, counts = collect(ctx, job, anchor, tape)
            result['opportunity_minutes'][anchor] = counts
            try:
                for key in sorted(snapshots, key=lambda k: (k.split('_')[1] != 'BOTH', k)):
                    guard(job)
                    env, obs, state = snapshots[key]
                    ident = f'{job["name"]}__{anchor}__{key}__t{int(env.env.scenario.time)}'
                    if ident in seen:
                        continue
                    try:
                        row = diagnose_state(ctx, job, env, obs, tape, state, models, ident, key.split('_')[1])
                    except BudgetStop:
                        raise
                    except Exception:
                        result['errors'].append(dict(id=ident, traceback=traceback.format_exc()))
                        save(path, result)
                        continue
                    result['rows'].append(dict(row, anchor=anchor))
                    save(path, result)
            finally:
                for env, _, _ in snapshots.values():
                    env.close()
        result['status'] = 'COMPLETE' if len(result['rows']) == sum(len(c) for c in result['opportunity_minutes'].values()) else 'INCOMPLETE_CASES'
    except BudgetStop:
        result['status'] = 'PARTIAL_BUDGET'
    except Exception:
        result['status'] = 'FAILED_ENGINEERING'
        result['errors'].append(dict(id=job['name'], traceback=traceback.format_exc()))
    save(path, result)
    return result['status'], len(result['rows'])


def summarize(out, planned, elapsed):
    blocks = [read(p) for p in sorted((out / 'blocks').glob('*.json'))]
    pairs = [(b, row) for b in blocks for row in b['rows']]
    statuses = {b['block']: b['status'] for b in blocks}
    errors = sum(len(b['errors']) for b in blocks)
    overall = 'COMPLETE' if len(statuses) == planned and all(x == 'COMPLETE' for x in statuses.values()) else 'PARTIAL'
    if not pairs and errors:
        overall = 'FAILED_NO_VALID_STATE'
    # 每块和训练种子分别汇总，探针和需求顺序不伪装成独立训练复制。
    model_rows = []
    for b, row in pairs:
        for p in row['policy']:
            model_rows.append(dict(block=b['block'], seed=b['seed'], trace=b['trace'], anchor=row['anchor'],
                state=row['id'], stratum=row['stratum'], tick=row['tick'],
                **{k: v for k, v in p.items() if isinstance(v, (int, float, str, bool)) or v is None}))
    if model_rows:
        with (out / '逐决策.csv').open('w', newline='', encoding='utf-8-sig') as f:
            fields = list(dict.fromkeys(k for x in model_rows for k in x))
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(model_rows)
    directions = {'S': {}, 'AC': {}, 'WM': {}}
    for seed in SEEDS:
        subset = [(b, row) for b, row in pairs if b['seed'] == seed]
        # 一个共同状态内先对两个主基座汇总，不把同状态重复计作独立证据。
        s = []
        for b, row in subset:
            valid = [p for p in row['policy'] if p['arm'] in ARMS[:2] and p['S_changed']]
            if len(valid) == 2:
                s.append((b['trace'], mean(p['S_gain_over_now'] for p in valid),
                          mean(p['S_gain_over_actor'] for p in valid)))
        a = [(b['trace'], any(x['reverses_passenger_rank'] for x in row['clean_2x2']))
             for b, row in subset if row['clean_2x2']]
        w = [(b['trace'], row['robust_residual_same_best'], row['committed_real_regret'], row['committed_alt_regret'])
             for b, row in subset if row['alternate_changed_requests'] and len(row['branches']) > 1]
        directions['S'][seed] = dict(states=len(s), traces=len({x[0] for x in s}),
            average_gain_vs_snapshot=mean(x[1] for x in s) if s else None,
            average_gain_vs_actors=mean(x[2] for x in s) if s else None,
            material_gain_states=sum(x[1] >= EPS and x[2] >= EPS for x in s))
        directions['AC'][seed] = dict(states=len(a), traces=len({x[0] for x in a}),
                                      material_rank_reversals=sum(x[1] for x in a))
        directions['WM'][seed] = dict(states=len(w), traces=len({x[0] for x in w}),
            stable_residual_states=sum(x[1] for x in w),
            real_residual_states=sum(x[2] >= EPS for x in w), alternate_residual_states=sum(x[3] >= EPS for x in w))
    flags = {}
    for method, data in directions.items():
        if not all(row['states'] >= 4 and row['traces'] >= 2 for row in data.values()):
            flags[method] = '覆盖不足，不作阴性结论'
            continue
        if method == 'S':
            passed = all(row['material_gain_states'] >= 2 and row['average_gain_vs_snapshot'] > 0
                         and row['average_gain_vs_actors'] > 0 for row in data.values())
            flags[method] = '可进小规模匹配消融：生效时间对齐' if passed else '本轮不支持扩大训练'
        elif method == 'AC':
            passed = all(row['material_rank_reversals'] >= 2 for row in data.values())
            flags[method] = '优先核验角色条件反馈，随后小规模消融' if passed else '交互证据未达到本轮门槛'
        else:
            passed = all(row['stable_residual_states'] >= 2 for row in data.values())
            flags[method] = '只进入候选排序残差的可预测性诊断；未证明世界模型有效' if passed else '暂不投入世界模型训练'
    minutes = sum(row['simulated_minutes'] for _, row in pairs)
    arms_by_seed = {}
    for seed in SEEDS:
        for arm in (*ARMS, 'READY'):
            selected = [x for x in model_rows if x['arm'] == arm and x['seed'] == seed]
            p = [x for x in selected if x['p_options'] > 1]
            a = [x for x in selected if x['aircraft_options'] > 1]
            arms_by_seed[f'{seed}__{arm}'] = dict(states=len(selected), p_options=len(p), aircraft_options=len(a),
                mean_p_regret=mean(x['passenger_regret'] for x in p) if p else None,
                mean_aircraft_regret=mean(x['aircraft_regret'] for x in a) if a else None,
                mean_joint_regret=mean(x['joint_regret'] for x in selected) if selected else None,
                both_actions_needed=sum(x['both_needed'] for x in selected))
    report = dict(status=overall, version=VERSION, blocks_planned=planned, blocks=statuses,
        valid_common_states=len(pairs), historical_or_current_case_errors=errors,
        frozen_seeds=list(SEEDS), frozen_training_steps_per_model=1_003_520,
        branch_physical_minutes=minutes, elapsed_seconds=round(elapsed, 2),
        directions=directions, suggested_next=flags, arms_by_seed=arms_by_seed, new_training_transitions=0,
        interpretation='筛查结果，不是新算法性能证据。全量终点 ATT 沿用旧评估，本轮窗口积分不能混排；共同续策下比较的局部机会不是全局最优。',
        references=SOURCES)
    save(out / 'SESSION_END.json', report)
    text = ['# HAPPO／MAPPO 接入方向：无训练筛查', '',
            f'状态：{overall}；有效共同状态 {len(pairs)}；计划块 {planned}，已读块 {len(blocks)}。',
            f'分支模拟 {minutes} 物理分钟；墙钟 {elapsed / 60:.1f} 分钟；新增训练步数 **0**。', '',
            '## 建议', '']
    text.extend(f'- **{method}**：{value}。' for method, value in flags.items())
    text += ['', '## 判断依据', '',
             '| 训练种子 | S 改变选择窗口 | S 同时比快照和策略好至少5人分钟 | AC 可比较窗口 | AC 名次翻转窗口 | WM 可比较窗口 | 两种续段均有同向剩余损失 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        s, a, w = (directions[x][seed] for x in ('S', 'AC', 'WM'))
        text.append(f'| {seed} | {s["states"]} | {s["material_gain_states"]} | {a["states"]} | {a["material_rank_reversals"]} | {w["states"]} | {w["stable_residual_states"]} |')
    text += ['', '分方法、分种子的错选人时见 `SESSION_END.json`；逐窗口见 `逐决策.csv` 与 `blocks/`。',
             '', '## 口径', '',
             '- E1／三个候选出发站／12架；三个方法各两个配对种子、每模型 1,003,520 步。',
             '- READY 与学习策略各提供共同状态；每块分层抽样，状态与需求轨迹不当作独立训练种子。',
             '- 分支用相同需求、合法动作和共同随机数；使用 5／15／30 分钟系统人时，另记逐人和实际空返。',
             '- 同时比较只保留已揭示请求、真实续段、保持到达时刻的反事实 OD 重排。重排只用于标签，绝不给策略输入。',
             '- 固定某角色的行动、改变另一角色，以及比较不同固定续策，借鉴 ASE/CED；不声称做了其完整中介归因。',
             '- 部分运行或缺少可控机会时优先补回放；不依据本轮自动启动训练。', '', '## 来源', '']
    text += [f'- [{k}]({v})' for k, v in SOURCES.items()]
    (out / '方向筛查.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    return report


def lock_output(out):
    handle = (out / 'RUN.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('同一输出目录已有诊断运行') from None
    return handle


def pack(out):
    target = out / f'{out.name}_DATAONLY.tar.gz'
    tmp = target.with_name(target.name + '.pending')
    with tarfile.open(tmp, 'w:gz', compresslevel=3) as tf:
        for path in (out / 'PLAN.json', out / 'SESSION_END.json', out / '方向筛查.md', out / '逐决策.csv',
                     *sorted((out / 'blocks').glob('*.json'))):
            if path.is_file() and not path.is_symlink():
                tf.add(path, arcname=out.name + '/' + path.relative_to(out).as_posix(), recursive=False)
    os.replace(tmp, target)
    print('[数据包]', target, flush=True)


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent, help='包含 experiment_archives 的本地项目')
    parser.add_argument('--archive', type=Path, help='已核验模型包的可选实际位置')
    parser.add_argument('--minutes', type=float, default=30., help='墙钟预算，默认30分钟')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2, help='最多两个单线程CPU工作进程')
    parser.add_argument('--run-name', default='uam_happo_mappo_transfer_30min_20261109_v1')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.minutes <= 180 or not re.fullmatch(r'uam_[a-zA-Z0-9_]{4,90}', args.run_name):
        parser.error('分钟预算或结果目录名无效')
    if not all(Path(sys.executable).resolve().parent.name.lower() != x for x in ('windowsapps', 'anaconda2', 'miniconda3')):
        parser.error('请使用项目独立环境，例如 conda activate uam5070')
    project = args.project.resolve()
    archive = (args.archive or (project / 'experiment_archives' / ARCHIVE)).resolve(strict=True)
    out = project / 'workspace' / 'artifacts' / args.run_name
    if out.resolve() != out or not out.is_relative_to(project):
        raise ValueError('输出目录不得重定向')
    if shutil.disk_usage(project).free < 2 * 1024**3:
        raise RuntimeError('项目磁盘剩余不足 2 GiB，不解包或开始回放')
    out.mkdir(parents=True, exist_ok=True)
    with lock_output(out):
        if (out / 'PLAN.json').exists() and not args.resume:
            raise RuntimeError('已有诊断数据；请用 --resume 或新的 --run-name')
        if (out / 'STOP_REQUESTED').exists():
            raise RuntimeError('检测到本诊断的 STOP_REQUESTED；确认可恢复后由用户移除，再加 --resume')
        (out / 'blocks').mkdir(exist_ok=True)
        plan = dict(version=VERSION, script_sha256=digest(__file__), archive_sha256=ARCHIVE_SHA,
            train_seeds=list(SEEDS), trace_names=list(TRACES), arms=list(ARMS), horizons=list(HORIZONS),
            order_seed=DESIGN_SEED, epsilon_person_minutes=EPS, new_training_transitions=0, references=SOURCES)
        if (out / 'PLAN.json').exists() and read(out / 'PLAN.json') != plan:
            raise ValueError('代码或科学合同已变化；不能混合续跑')
        save(out / 'PLAN.json', plan)
        entry, records, bank = prepare_inputs(archive, out)
        # 先将每种子/需求作成完整块，再固定混排；预算切断不把缺格标为失败。
        schedule = []
        randomizer = random.Random(DESIGN_SEED)
        # 两个预算波次各含两种子与两需求组；不能让预算提前结束只留下一个种子。
        for wave in range(2):
            phase = [(i, seed, TRACES[i]) for seed in SEEDS for i in (wave, wave + 2)]
            randomizer.shuffle(phase)
            schedule.extend(phase)
        jobs = [dict(name=f's{seed}__{trace}', seed=seed, trace=trace, trace_path=bank[trace],
                     alternate_anchor=(i + SEEDS.index(seed)) % 2, out=str(out), entry=str(entry), records=records,
                     deadline=start + args.minutes * 60 - 8) for i, seed, trace in schedule]
        print(f'[诊断] {len(jobs)}块，{args.workers}个单线程CPU工作进程，约{args.minutes:g}分钟，训练步数0', flush=True)
        if args.workers == 1:
            for i, job in enumerate(jobs, 1):
                status, n = worker(job)
                print(f'[{i}/{len(jobs)}] {job["name"]} {status} 有效状态={n}', flush=True)
                if status == 'PARTIAL_BUDGET':
                    break
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                pending = {pool.submit(worker, j): j for j in jobs}
                try:
                    for i, future in enumerate(as_completed(pending), 1):
                        status, n = future.result()
                        print(f'[{i}/{len(jobs)}] {pending[future]["name"]} {status} 有效状态={n}', flush=True)
                except KeyboardInterrupt:
                    (out / 'STOP_REQUESTED').touch(exist_ok=True)
                    for future in pending:
                        future.cancel()
                    print('[暂停] 保留已完成共同状态；移除本次STOP_REQUESTED后可续跑', flush=True)
        report = summarize(out, len(jobs), time.monotonic() - start)
        pack(out)
        print('[结果]', report['status'], report['suggested_next'], flush=True)
        if report['status'] == 'FAILED_NO_VALID_STATE':
            return 2
    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[_key] = '1'
    raise SystemExit(main())
