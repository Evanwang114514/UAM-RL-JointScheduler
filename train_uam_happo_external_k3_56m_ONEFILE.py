#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""K3 时变 HAPPO 外接观测：七臂、四个配对种子；历史原版只读。"""
import argparse
import hashlib
import inspect
import json
import math
import os
import pickle
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import time
import traceback
import types
from pathlib import Path

for _thread_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_key] = '1'
sys.dont_write_bytecode = True

RUN = 'uam_happo_external_k3_28run_56m_v1'
CASE = 'E1_K3_VARYING'
OLD = 'uam_happo_env5_44run_88m_v2'
OLD_SHA = 'f5ba14c61a52658708b2205d3e511120351e7d2a65053448c828831e87f73df7'
ENTRY_SHA = '27fbe3d361b190be970fce37eb65ef98d159e4fbc52c6952299ef55c9322baa7'
HARL_SHA = 'b1af98b0dbab72a2eee9d160751cd09aedbb8ce2'
SEEDS = (2101, 2102, 2103, 2104)
ARMS = ('H1_INFO', 'H2_S', 'H3_S_AC', 'H4_S_WM', 'H5_S_CF', 'H6_S_CED', 'H7_S_SET')
PARENT = dict(zip(ARMS, ('H0_HISTORICAL', 'H1_INFO', 'H2_S', 'H2_S', 'H2_S', 'H5_S_CF', 'H2_S')))
STEPS = 1_996_800
ROLLOUT = 10_240
PLANNED = len(ARMS) * len(SEEDS) * STEPS
CAP = 56_000_000
CF_H = 30
MEDIATOR_H = 8
READY_H = 60
WIDTH = 45
HALT = False


def stop_signal(*_):
    global HALT
    HALT = True


def module_from_archive(path, project):
    """只读自己归档中的冻结入口；不提取、不执行旧矩阵 main。"""
    from ast import If, parse, unparse
    if path.is_symlink() or not path.is_file():
        raise ValueError('缺少已验证的 88M DATAONLY 归档')
    digest = hashlib.file_digest
    with path.open('rb') as f:
        if digest(f, 'sha256').hexdigest() != OLD_SHA:
            raise ValueError('88M 数据包哈希不一致；禁止混用历史合同')
    names = ['ENTRYPOINT.py', 'PLAN.json', 'BANK.json', 'MODEL_LEVEL_SUMMARY.json']
    names += [f'cells/{CASE}__HAPPO_RANDOM__s{s}/{k}.json' for s in SEEDS for k in ('CONFIG', 'END')]
    with tarfile.open(path, 'r:gz') as tar:
        payload = {}
        for name in names:
            member = tar.getmember(f'{OLD}/{name}')
            if not member.isfile() or member.size > 3_000_000:
                raise ValueError(f'归档部件不合法：{name}')
            with tar.extractfile(member) as f:
                payload[name] = f.read()
    source = payload.pop('ENTRYPOINT.py')
    if hashlib.sha256(source).hexdigest() != ENTRY_SHA:
        raise ValueError('冻结环境入口与旧训练实际使用的源不一致')
    historical = {k: json.loads(v) for k, v in payload.items()}
    plan = historical['PLAN.json']
    if plan['entrypoint_sha256'] != ENTRY_SHA or plan['author_harl_commit'] != HARL_SHA:
        raise ValueError('旧合同不符')
    tree = parse(source)
    tree.body = [node for node in tree.body if not (isinstance(node, If) and '__name__' in unparse(node.test))]
    entry = types.ModuleType('uam_external_frozen_environment')
    entry.__file__ = __file__
    sys.modules[entry.__name__] = entry
    exec(compile(tree, '<verified_88m_environment>', 'exec'), vars(entry))
    entry.PROJECT, entry.RUN_NAME, entry.CAP = project, RUN, CAP
    return entry, historical


def make_plan(hist):
    tasks = []
    for seed in SEEDS:
        old = next(x for x in hist['PLAN.json']['tasks']
                   if x['case'] == CASE and x['arm'] == 'HAPPO_RANDOM' and x['seed'] == seed)
        assert old['steps'] == STEPS and old['fleet'] == 12 and old['control'] == 'J3'
        group = list(ARMS)
        random.Random(14112026 + seed).shuffle(group)
        for arm in group:
            tasks.append(dict(old, id=f'{CASE}__{arm}__s{seed}', arm=arm, parent=PARENT[arm],
                              network='HARL_MLP_EXTERNAL_AIRCRAFT_OBS', run_order=len(tasks)+1))
    assert len(tasks) == 28 and PLANNED == 55_910_400 and STEPS // ROLLOUT == 195
    return dict(run=RUN, tasks=tasks, baseline='旧 88M 四种子，不安排新训练',
                online_planned=PLANNED, cap_including_recovery=CAP, seed_order_key=14112026,
                module_width=WIDTH, cf_minutes=CF_H, mediator_minutes=MEDIATOR_H, readiness_minutes=READY_H,
                historical_archive_sha256=OLD_SHA, source_sha256=ENTRY_SHA, harl_commit=HARL_SHA,
                entry_sha256=hash_file(__file__), comparisons=PARENT,
                note='时变 K3；固定机队 12；真实物理分钟计步；离线分支另计')


def hash_file(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify_vendor(path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != HARL_SHA or git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('HARL 须为指定提交的干净作者副本')


def replay_reset(env):
    # 保证重建的共同前缀不受上一条分支消耗全局随机数的影响。
    import numpy as np
    random.seed(123)
    np.random.seed(123)
    return env.reset(seed=123)


def charge_delay(plane, station, rates, after_energy=0.0):
    # 飞机换站后按目标站充电率；保留充电状态检查的分钟离散化。
    deficit = max(0., float(plane.spec.battery_capacity_kwh) - float(plane.battery_kwh) + after_energy)
    return math.ceil(deficit / float(rates[int(station)])) + int(deficit > 1e-9)


def packet(phase, stations, rates):
    """只读当前已生成对象、物理计时器和合法候选；不读需求表或未来策略。"""
    import numpy as np
    s = phase.env.scenario
    points = s.vertiports.vertiport_list
    lookup = {str(v): j for j, v in enumerate(stations)}
    eta = [[] for _ in stations]
    ready = [[] for _ in stations]
    pending = [0] * 3
    idle = sorted((e for e in s.vertiports.evtols_at_vertiport.get('2', [])
                   if e.state.name == 'IDLE' and not e.passenger_ids), key=lambda e: str(e.id))
    mask = tuple(i for i, value in enumerate(phase.env._legal) if value > .5)
    for p in s.persons.persons.values():
        j = lookup.get(str(p.origin_vertiport_id))
        if j is not None:
            if p.state == 'enroute' and p.sub_state == 'to_vertiport':
                eta[j].append(math.ceil(max(0., float(p.current_timer or 0))) + 1)
            elif p.state == 'assigned' and p.method == 'UAM':
                eta[j].append(math.ceil(max(0., float(p.t_drive_pickup or 0))) + 2)
    for e in s._all_evtols.values():
        flying = e.state.name == 'FLYING'
        j = lookup.get(str(e.target_vertiport_id if flying else e.current_vertiport_id))
        if j is None:
            continue
        if e.state.name == 'IDLE' and not e.passenger_ids:
            ready[j].append(0)
        elif flying or e.state.name == 'CHARGING':
            flight = math.ceil(max(0., float(e.remaining_time))) if flying else 0
            ready[j].append(flight + charge_delay(e, stations[j], rates))
            pending[j] += 1
    a, q, raw = [], [], []
    for j, v in enumerate(stations):
        travel = READY_H
        q.append(len(points[str(v)].person_list))
        if idle:
            e = idle[0]
            d = s.vertiports._calculate_distance(points['2'].vertiport_position, points[str(v)].vertiport_position)
            energy = float(d) * e.spec.energy_consumption_kwh_per_km
            travel = 1 + math.ceil(d / e.spec.max_speed * 60.) + charge_delay(e, v, rates, energy)
        a.append(float(travel))
        # 两组看到完全相同的分箱事件与时间；只有查询锚点变动。
        bands = (4, 8, 16, 32)
        raw.append([q[j]/30, sum(t==0 for t in ready[j])/12, pending[j]/12,
                    min(travel, 60)/40, float(j in mask), len(idle)/12]
                   + [sum(t<=b for t in eta[j])/30 for b in bands]
                   + [sum(t<=b for t in ready[j])/12 for b in bands])
    def tokens(aligned):
        rows = []
        for j, travel in enumerate(a):
            t = min(READY_H, travel) if aligned else 12
            rows.append(raw[j] + [float(j==c) for c in range(3)] +
                        [sum(r<=t for r in eta[j])/30, sum(r<=t for r in ready[j])/12])
        return np.asarray(rows, dtype=np.float32)
    active = bool(idle and any(q) and len(mask)>=2 and not phase.gate.release and not phase.gate.remaining)
    return tokens(False), tokens(True), a, mask, active


def schedule(env, old, tape, stations, rates, forced=None, donor=None, labels=False):
    """外部回放工具；强制动作只用于造标签和诊断，绝不走训练缓冲区。"""
    s = env.env.scenario
    t = int(s.time)
    u, v = tape[t]
    focal = str(s.waiting_decisions[0]) if s.waiting_decisions else None
    p = min(2,int(u/.30*3)) if u < .30 else old.ready_choice(env.env, env.env._three_m)
    if focal is None:
        p = 0
    if donor is not None:
        if donor['focal'] != focal:
            return None
        p = donor['p']
    obs, reward, term, trunc, info = env.step(p)
    if reward or term or trunc or info['physical_increment'] != 0:
        raise ValueError('乘客阶段不得推进分钟')
    legal = tuple(i for i, x in enumerate(env.env._legal) if x > .5)
    hub = s.vertiports.evtols_at_vertiport.get('2', [])
    eligible = bool(any(e.state.name=='IDLE' and not e.passenger_ids for e in hub)
                    and any(s.vertiports.vertiport_list[str(x)].person_list for x in stations)
                    and not env.gate.remaining and not env.gate.release)
    if not legal:
        raise ValueError('空合法候选集合')
    ref = old.base_for_lq(env.env)
    if ref not in legal:
        ref = legal[0]
    a = ref
    # v 是独立均匀数；探索分支重新缩放到 [0,1)。
    if v < .55:
        a = legal[min(len(legal)-1, int(v/.55*len(legal)))]
    if donor is not None:
        if donor['eligible'] != eligible or donor['a'] not in legal:
            return None
        a = donor['a']
    if forced is not None:
        if forced not in legal:
            return None
        a = forced
    feature = packet(env, stations, rates) if labels else None
    off = len(env.gate.events)
    _, _, _, _, info = env.step(a)
    assert info['physical_increment'] == 1 and int(s.time) == t+1
    sig = tuple(sorted((z['aircraft'], z['target']) for ev in env.gate.events[off:]
                       if ev['kind']=='实际空返' for z in ev['executed']))
    return dict(t=t, p=p, a=a, focal=focal, eligible=eligible, sig=sig, legal=legal,
                ref=ref, feature=feature)


def branch(make, old, tape, stations, rates, t, a, donor=None, policy=None):
    """从头重建前缀，避免复制暂停中的生成器；分支用按分钟索引的共同随机数。"""
    env = make()
    steps = 0
    try:
        obs, _ = replay_reset(env)
        start_area = None
        first = None
        downstream = []
        ready = {}
        start_people = {}
        for minute in range(t + (CF_H if donor is not None or policy is not None else READY_H)):
            if minute == t:
                start_area = env.ledger['area']
                start_people = dict(env.ledger['individual'])
            if policy is None:
                use = donor[minute-t-1] if donor is not None and t < minute < t+MEDIATOR_H else None
                d = schedule(env, old, tape, stations, rates, a if minute==t else None, use, minute==t)
            else:
                p = policy(obs, 0, tape[minute][0])
                post, _, _, _, _ = env.step(p)
                legal = tuple(j for j, v in enumerate(env.env._legal) if v>.5)
                sampled = policy(post, 1, tape[minute][1])
                action = a if minute==t and a is not None else sampled
                if minute==t and action not in legal and a is not None:
                    return None, steps
                off = len(env.gate.events)
                f = packet(env, stations, rates) if minute==t else None
                obs, _, _, _, info = env.step(action)
                assert info['physical_increment']==1
                sig = tuple(sorted((z['aircraft'], z['target']) for ev in env.gate.events[off:]
                                   if ev['kind']=='实际空返' for z in ev['executed']))
                d = dict(a=action, legal=legal, sig=sig, feature=f)
            if d is None:
                return None, steps
            steps += 1
            if minute == t:
                first = d
            if t < minute < t+MEDIATOR_H:
                downstream.append({k:v for k,v in d.items() if k not in ('feature',)})
            if minute == t+CF_H-1:
                cost = float(env.ledger['area'] - start_area)
                pressure = [len(env.env.scenario.vertiports.vertiport_list[str(v)].person_list)/30 for v in stations]
                people = {key:value-start_people.get(key,0) for key,value in env.ledger['individual'].items()
                          if value>start_people.get(key,0)}
            if minute >= t:
                # 只追踪首分钟实际飞往该目标的飞机；未观测到就绪不造标签。
                fleet = env.env.scenario._all_evtols
                for e in fleet.values():
                    if (str(e.id), str(e.current_vertiport_id)) in first['sig'] and e.state.name=='IDLE' and not e.passenger_ids:
                        ready.setdefault(str(e.id), minute+1-t)
            if env.done and minute < t+CF_H-1:
                raise ValueError('标签窗口早于预定终点结束')
            if env.done:
                break
        return dict(first=first, cost=cost, pressure=pressure, ready=ready, downstream=downstream,people=people), steps
    finally:
        env.close()


def contexts(base, m):
    from at_obj.vertiport.vertiport_spec import VERTIPORT_CHARGE_RATE
    return tuple(m.CANDIDATES[3]), {int(k):float(v) for k,v in VERTIPORT_CHARGE_RATE.items()}


def known_routes(base, old):
    old.base_for_lq = base.lq_target


def labels_make(old, base, m, root, task, bank):
    import numpy as np
    Phase = base.gym_class(m)
    stations, rates = contexts(base, m)
    source = bank['trace']
    make = lambda: Phase(root, task, source, 500, 123, False)
    records, simulated = [], 0
    counters = dict(scanned=0, eligible=0, distinct=0, ced_supported=0)
    for pattern in range(10):
        if HALT or (root/'STOP_REQUESTED').exists():
            raise InterruptedError('离线标签阶段收到停止请求')
        rnd = random.Random(48001+pattern)
        tape = [(rnd.random(), rnd.random()) for _ in range(501)]
        env = make()
        chosen = []
        try:
            replay_reset(env)
            for minute in range(276):
                d = schedule(env, old, tape, stations, rates)
                simulated += 1
                counters['scanned'] += 1
                if 36<=minute<276 and d['eligible'] and len(d['legal'])>=2:
                    counters['eligible'] += 1
                    chosen.append(d)
        finally:
            env.close()
        # 每个到达区段预先等配额，按固定伪随机顺序选，不以收益大小筛样本。
        for lo in (36, 96, 156, 216):
            rows = [r for r in chosen if lo<=r['t']<lo+60]
            for d in rnd.sample(rows, min(6, len(rows))):
                t, ref = d['t'], d['ref']
                results = {}
                for a in d['legal']:
                    r, work = branch(make, old, tape, stations, rates, t, a)
                    simulated += work
                    if r is not None:
                        results[a] = r
                if ref not in results or len({r['first']['sig'] for r in results.values()})<2:
                    continue
                counters['distinct'] += 1
                factual = results[ref]
                x0, x1, travel, _, _ = factual['first']['feature']
                replay, work = branch(make, old, tape, stations, rates, t, ref, factual['downstream'])
                simulated += work
                if (replay is None or replay['cost']!=factual['cost'] or replay['people']!=factual['people']
                        or replay['first']['sig']!=factual['first']['sig']):
                    raise ValueError('事实动作和后续行为未能自复演')
                rows_out = []
                sigs = {}
                for a in (ref, *(v for v in results if v!=ref)):
                    target = results[a]
                    sig = target['first']['sig']
                    # 指令别名不增加 inverse 分类样本。
                    if sig in sigs:
                        continue
                    sigs[sig] = a
                    mid, work = branch(make, old, tape, stations, rates, t, ref, target['downstream'])
                    simulated += work
                    if a!=ref and sig==factual['first']['sig']:
                        continue
                    total = target['cost']-factual['cost']
                    path = None if mid is None else mid['cost']-factual['cost']
                    if path is not None and a!=ref and sig!=factual['first']['sig']:
                        counters['ced_supported'] += 1
                    candidate_delays = [target['ready'][eid] for eid,vid in sig
                                        if vid==str(stations[a]) and eid in target['ready']]
                    ready = min(candidate_delays,default=None)
                    # 用延后真实队列变化作逆动力学输入，不把调机目的地或航班号当答案。
                    delta_pressure = [u-v for u,v in zip(target['pressure'],factual['pressure'])]
                    rows_out.append(dict(a=a, total=total, path=path, physical=None if path is None else total-path,
                                         y=delta_pressure, wm=None if ready is None else ready-travel[a],
                                         sig=sig, pressure=target['pressure']))
                records.append(dict(pattern=pattern, minute=t, ref=ref, x0=x0, x1=x1, rows=rows_out))
    if not records:
        raise ValueError('没有找到实际空返不同的有效候选')
    return dict(rows=records, counters=counters, offline_simulator_minutes=simulated,
                fit_patterns=list(range(8)), audit_patterns=[8,9], source_sha256=bank['training_sha256'],
                definition=f'反事实 {CF_H} 分钟；交换随后 {MEDIATOR_H-1} 分钟的决策，其后自然执行；就绪最多 {READY_H} 分钟')


def fit_modules(data):
    """CPU 小型外部编码器；主训练只加载导出的只读 NumPy 参数。"""
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    torch.set_num_threads(1)
    train = [r for r in data['rows'] if r['pattern'] in data['fit_patterns']]
    audit = [r for r in data['rows'] if r['pattern'] in data['audit_patterns']]
    if len(train)<12:
        raise ValueError(f'可训练的不同执行状态太少：{len(train)}，不以零特征代替')

    class Encoder(nn.Module):
        def __init__(self, attention):
            super().__init__()
            self.attention = attention
            self.inp = nn.Linear(19, 12)
            if attention:
                self.qkv = nn.Linear(12, 36, bias=False)
                self.mix = nn.Linear(12, 12)
                self.hidden = nn.Linear(12, 12)
            else:
                self.mix = nn.Linear(24,24)
                self.hidden = nn.Linear(24,12)
            self.out = nn.Linear(12,8)
            self.recon = nn.Linear(8,19)
            self.inverse = nn.Sequential(nn.Linear(27,18),nn.Tanh(),nn.Linear(18,3))

        def forward(self, x):
            h = torch.tanh(self.inp(x))
            if self.attention:
                q,k,v = self.qkv(h).chunk(3,dim=-1)
                mix = ((q @ k.transpose(-2,-1))/math.sqrt(12)).softmax(-1) @ v
                h = self.hidden(torch.tanh(h + self.mix(mix)))
            else:
                h = self.hidden(torch.tanh(self.mix(torch.cat((h,h.mean(-2,keepdim=True).expand_as(h)),dim=-1))))
            return torch.tanh(self.out(torch.tanh(h)))

    xt = torch.tensor(np.stack([np.stack((r['x0'],r['x1'])) for r in train]), dtype=torch.float32)
    indices, effects, targets, valid = [], [], [], []
    for i, r in enumerate(train):
        y_counts = {}
        for z in r['rows']:
            if z['sig']:
                y = tuple(round(v,8) for v in z['y'])
                y_counts[y] = y_counts.get(y,0)+1
        supported = [z for z in r['rows'] if z['sig'] and y_counts[tuple(round(v,8) for v in z['y'])]==1]
        labels = sorted(z['a'] for z in supported)
        for z in supported:
            if len(labels)<2:
                continue
            indices.append(i); effects.append(z['y']); targets.append(z['a'])
            valid.append([j in labels for j in range(3)])
    use_ac = len(indices)>=12 and len(set(targets))>=2
    models = {}
    for name, attention, ac in (('shared',False,False), ('ac',False,True), ('set',True,False)):
        if ac and not use_ac:
            continue
        torch.manual_seed(55104)
        net = Encoder(attention)
        optim = torch.optim.Adam(net.parameters(), lr=.006)
        weight = torch.tensor([1.,1.,1.,1.,.25,1.,2.,2.,2.,2.,2.,2.,2.,2.,.25,.25,.25,3.,3.])
        for _ in range(75):
            latent = net(xt)
            loss = ((net.recon(latent)-xt).square()*weight).mean()
            if ac:
                # 保留共同事实重构；仅增加对真实执行结果的逆动力学约束。
                xx = latent[indices,1].reshape(-1,24)
                y = torch.tensor(effects, dtype=torch.float32)
                logits = net.inverse(torch.cat((xx, y), dim=-1))
                loss = loss + .1 * F.cross_entropy(logits.masked_fill(~torch.tensor(valid), -1e8), torch.tensor(targets))
            optim.zero_grad(); loss.backward(); optim.step()
        models[name] = dict(weights={k: v.detach().numpy() for k,v in net.state_dict().items()
                                      if not k.startswith(('recon.', 'inverse.'))},
                            parameters=sum(v.numel() for k,v in net.named_parameters()
                                           if not k.startswith(('recon.', 'inverse.'))),
                            fit_loss=float(loss.detach()))

    def ridge(target, scale, minimum):
        x, y, z, yy = [], [], [], []
        for split, xx, ty in ((train,x,y),(audit,z,yy)):
            for r in split:
                for e in r['rows']:
                    if e[target] is None or e['a']==r['ref'] and target in ('total','path'):
                        continue
                    xx.append(candidate_features(r['x1'], e['a'])); ty.append(e[target]/scale)
        if len(x)<minimum:
            return None, dict(n=len(x), holdout_n=len(z), status='INSUFFICIENT_SUPPORT')
        X, Y = np.asarray(x), np.asarray(y)
        # 固定正则，不按确认集调参。
        w = np.linalg.solve(X.T@X+.07*np.eye(X.shape[1]), X.T@Y)
        err = np.asarray(z) @ w - yy if z else []
        return w, dict(n=len(x), holdout_n=len(z), mae=float(np.mean(np.abs(err))*scale) if z else None,
                       target_span=float(np.max(np.abs(Y))*scale),
                       zero_reference_mae=float(np.mean(np.abs(yy))*scale) if z else None)

    aux, support = {}, {}
    for key, unit, minimum in (('wm',20.,12),('total',90.,12),('path',90.,12)):
        w, report = ridge(key,unit,minimum)
        support[key] = report
        if w is not None:
            aux[key] = w
    if 'path' in aux and support['path'].get('target_span',0)==0:
        aux.pop('path')
        support['path']['status']='NO_NONZERO_PATH_SIGNAL'
    return dict(models=models, aux=aux, support=support, inverse_count=len(indices),
                data_counts=data['counters'], offline_simulator_minutes=data['offline_simulator_minutes'])


def candidate_features(x, a):
    import numpy as np
    # 候选自身、相对其他站及简单交互；训练与部署完全同一顺序。
    x=np.asarray(x)
    return np.r_[1., x[a], x.mean(0), x[a]*x.mean(0)]


def encode_numpy(x, model):
    import numpy as np
    w = model['weights']
    def linear(h, name):
        return h @ w[name+'.weight'].T + w.get(name+'.bias',0.)
    h = np.tanh(linear(x,'inp'))
    if 'qkv.weight' in w:
        q,k,v = np.split(linear(h,'qkv'),3,axis=-1)
        s = (q@k.T)/math.sqrt(12)
        a = np.exp(s-s.max(-1,keepdims=True)); a /= a.sum(-1,keepdims=True)
        h = linear(np.tanh(h+linear(a@v,'mix')),'hidden')
    else:
        h = linear(np.tanh(linear(np.concatenate((h,np.broadcast_to(h.mean(0),h.shape)),axis=-1),'mix')),'hidden')
    return np.tanh(linear(np.tanh(h),'out'))


def module_features(arm, feature, frozen, ref):
    import numpy as np
    x0, x1, travel, legal, active = feature
    x = x0 if arm=='H1_INFO' else x1
    model = 'ac' if arm=='H3_S_AC' else 'set' if arm=='H7_S_SET' else 'shared'
    z = np.zeros((3,15),dtype=np.float32)
    z[:,:8] = encode_numpy(x,frozen['models'][model])
    z[:,8:10] = x[:,17:19]
    z[:,10] = x[:,4]
    z[:,11] = np.minimum(travel,60)/40
    if active:
        for a in legal:
            k = candidate_features(x,a)
            if arm=='H4_S_WM':
                z[a,12] = np.clip(k@frozen['aux']['wm'], -1.,1.)
            if arm in ('H5_S_CF','H6_S_CED') and a!=ref:
                z[a,13] = np.clip(k@frozen['aux']['total'], -3.,3.)
            if arm=='H6_S_CED' and a!=ref:
                z[a,14] = np.clip(k@frozen['aux']['path'], -3.,3.)
    return z.ravel()


def augmented_phase(base, m, frozen):
    import gymnasium as gym
    import numpy as np
    Original = base.gym_class(m)
    stations, rates = contexts(base,m)

    class Augmented(Original):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.original_width = self.observation_space.shape[0]
            self.observation_space = gym.spaces.Box(-np.inf,np.inf,(self.original_width+WIDTH,),np.float32)

        def vector(self):
            x = super().vector()
            if self.phase!=1:
                return np.concatenate((x,np.zeros(WIDTH,dtype=np.float32)))
            current = packet(self,stations,rates)
            legal = current[3]
            ref = base.lq_target(self.env)
            extra = module_features(self.task['arm'],current,frozen,ref if ref in legal else legal[0])
            if not np.isfinite(extra).all():
                raise ValueError('外接观测非有限')
            return np.concatenate((x, extra))

    return Augmented


def auxiliary(root, old, hist, vendor):
    path = root/'FROZEN_AUX.pkl'
    record = old.read(root/'AUX_MANIFEST.json')
    if record:
        if hash_file(path)!=record['sha256']:
            raise ValueError('冻结外接参数哈希不符')
        return record
    base = old.base_module()
    task = next(x for x in hist['PLAN.json']['tasks'] if x['case']==CASE and x['arm']=='HAPPO_RANDOM')
    m,native = base.runtime(root,task)
    known_routes(base,old)
    V = old.read(root/'BANK.json')['VARYING']
    old.validate_job_bank(dict(trace=V['trace'],trace_sha256=V['training_sha256'],bank=V['bank']),native)
    data = labels_make(old,base,m,root,task,V)
    old.save(root/'BRANCH_LABELS.json',data)
    fitted = fit_modules(data)
    old.atomic_pickle(path,fitted)
    available = [a for a in ARMS if any((
        a in ('H1_INFO','H2_S','H7_S_SET'),
        a=='H3_S_AC' and 'ac' in fitted['models'],
        a=='H4_S_WM' and 'wm' in fitted['aux'],
        a=='H5_S_CF' and 'total' in fitted['aux'],
        a=='H6_S_CED' and {'total','path'}<=fitted['aux'].keys()
    ))]
    record = dict(sha256=hash_file(path), labels_sha256=hash_file(root/'BRANCH_LABELS.json'),
                  available=available, support=fitted['support'], inverse_count=fitted['inverse_count'],
                  parameters={k:v['parameters'] for k,v in fitted['models'].items()},
                  data_counts=fitted['data_counts'], offline_simulator_minutes=data['offline_simulator_minutes'])
    old.save(root/'AUX_MANIFEST.json',record)
    return record


def diagnostics(old, base, m, root, job, probabilities):
    """固定状态采样规则，对每个最终模型回放，单独输出解释结果和仿真开销。"""
    stations,rates = contexts(base,m)
    Phase = base.gym_class(m)
    item = job['task']
    traces = [b for b in job['bank'] if b['name'].startswith('order_83')]
    rows, work = [], 0
    for seed,b in enumerate(traces,901):
        rng = random.Random(seed)
        tape = [(rng.random(),rng.random()) for _ in range(601)]
        make = lambda: Phase(root,item,b['path'],600,123,False)
        env = make()
        possible = []
        try:
            obs,_ = replay_reset(env)
            for t in range(276):
                p = probabilities(obs,0,tape[t][0])
                post,_,_,_,_ = env.step(p)
                feature = packet(env,stations,rates)
                a = probabilities(post,1,tape[t][1])
                obs,_,_,_,_ = env.step(a)
                work += 1
                if 36<=t<276 and feature[4] and a in feature[3]:
                    possible.append((t,a,feature[3]))
        finally:
            env.close()
        covered = []
        for lo in (36,96,156,216):
            candidates = [r for r in possible if lo<=r[0]<lo+60]
            if not candidates:
                continue
            t,a,legal = rng.choice(candidates)
            ref,spent = branch(make,old,tape,stations,rates,t,None,policy=probabilities)
            work += spent
            if ref is None or ref['first']['a']!=a:
                raise ValueError('抽样动作与反事实事实前缀不一致')
            results = []
            observed = set()
            for j in (a,*(v for v in legal if v!=a)):
                b2,spent = branch(make,old,tape,stations,rates,t,j,policy=probabilities)
                work += spent
                if b2 is None:
                    continue
                sig = b2['first']['sig']
                if j==a and (sig!=ref['first']['sig'] or b2['cost']!=ref['cost']):
                    raise ValueError('事实回放没有复演')
                if sig not in observed:
                    observed.add(sig)
                    people=set(ref['people'])|set(b2['people'])
                    effects=[b2['people'].get(k,0)-ref['people'].get(k,0) for k in people]
                    if sum(effects)!=b2['cost']-ref['cost']:
                        raise ValueError('反事实乘客时间之和与系统时间差不一致')
                    results.append(dict(action=j,dispatch=sig,cost=b2['cost'],
                                        helped=sum(x<0 for x in effects),harmed=sum(x>0 for x in effects),
                                        saved=sum(-x for x in effects if x<0),added=sum(x for x in effects if x>0)))
            _,aligned,travel,_,_=ref['first']['feature']
            covered.append(dict(minute=t, executed_candidates=len(results), factual_action=a,
                                estimated_ready_minutes=travel, committed_at_ready=aligned[:,17:19],
                                regret=ref['cost']-min(r['cost'] for r in results) if len(results)>=2 else None,
                                candidates=results))
        rows.append(dict(trace=b['name'],seed=seed, eligible=len(possible), rows=covered))
    old.save(root/'cells'/item['id']/'CF_DIAGNOSTICS.json',
             dict(status='COMPLETE', rows=rows, offline_simulator_minutes=work, new_training_transitions=0,
                  definition='同需求与冻结策略；单次合法调机干预后允许自然反应；30分钟局部代价，不是线上成绩'))


def train(old,root,job):
    import cloudpickle
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    # spawn 子进程需按值接收归档中动态装入的环境封装。
    cloudpickle.register_pickle_by_value(old)
    base = old.base_module()
    vendor,args,author_train = base.harl_prepare(dict(harl_dir=job['vendor_dir']))
    from harl.algorithms.actors.happo import HAPPO
    from harl.algorithms.critics.v_critic import VCritic
    from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
    from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
    from harl.common.valuenorm import ValueNorm
    for obj in (HAPPO,VCritic,OnPolicyActorBuffer,OnPolicyCriticBufferEP,ValueNorm,author_train):
        if not Path(inspect.getfile(obj)).resolve().is_relative_to(vendor):
            raise ValueError('训练器或组件没有来自锁定的 HARL 作者目录')
    item=job['task']
    folder=root/'cells'/item['id']
    started=time.time()
    if args!=job['historical_args']:
        raise ValueError('HARL 优化参数与 88M 合同不一致')
    if hash_file(root/'FROZEN_AUX.pkl')!=job['aux_sha']:
        raise ValueError('外接模块参数与任务合同不一致')
    with (root/'FROZEN_AUX.pkl').open('rb') as f:
        frozen=pickle.load(f)
    m,native=base.runtime(root,item)
    old.validate_job_bank(job,native)
    factory = augmented_phase(base,m,frozen)
    base.gym_class = lambda unused: factory
    random.seed(item['seed']); np.random.seed(item['seed']); torch.manual_seed(item['seed'])
    def maker(rank):
        def construct():
            return factory(root,item,job['trace'],500,item['seed']*10+rank,False)
        return construct
    vector=SubprocVecEnv([maker(i) for i in range(5)], start_method='spawn')
    norm=None
    try:
        meta=old.read(folder/'RESUME.json',{})
        record={}
        if meta:
            for k in ('model','norm'):
                p=folder/meta[k]
                if Path(meta[k]).name!=meta[k] or p.is_symlink() or hash_file(p)!=meta[k+'_sha']:
                    raise ValueError('恢复件路径或哈希不符')
            with (folder/meta['model']).open('rb') as f:
                record=pickle.load(f)
            norm=VecNormalize.load(str(folder/meta['norm']),vector)
        else:
            norm=VecNormalize(vector,norm_obs=True,norm_reward=False)
        space=norm.observation_space
        original=gym.spaces.Box(-np.inf,np.inf,(space.shape[0]-WIDTH,),np.float32)
        action=gym.spaces.Discrete(3)
        device=torch.device('cuda')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用，不切 CPU')
        # 只扩大飞机 actor 的输入空间；乘客与 critic 仍走旧维度。
        actors=[HAPPO(args,original,action,device),HAPPO(args,space,action,device)]
        critic=VCritic(args,original,device); vn=ValueNorm(1,device=device)
        buffers=[OnPolicyActorBuffer(args,z,action) for z in (original,space)]
        cb=OnPolicyCriticBufferEP(args,original)
        runner=types.SimpleNamespace(algo_args={'train':args},actor=actors,critic=critic,
                   actor_buffer=buffers,critic_buffer=cb,value_normalizer=vn,state_type='EP',
                   fixed_order=False,num_agents=2,action_aggregation='prod')
        if record:
            for i,a in enumerate(actors):
                a.actor.load_state_dict(record['actors'][i]); a.actor_optimizer.load_state_dict(record['optimizers'][i])
            critic.critic.load_state_dict(record['critic'])
            critic.critic_optimizer.load_state_dict(record['critic_optimizer'])
            vn.load_state_dict(record['value_norm'])
            for k,v in record['value_norm_statistics'].items():
                getattr(vn,k).copy_(v.to(device))
            old.rng_restore(record['rng'])
        real=int(record.get('real_steps',0)); updates=int(record.get('updates',0)); initial=real
        if real<0 or real>STEPS or real!=updates*ROLLOUT:
            raise ValueError('恢复累计步数不等于完整 rollout 数')
        obs=norm.reset(); zeros=np.zeros((5,1,128),np.float32)
        ones=np.ones((5,1),np.float32); masks=ones.copy()
        old.save(folder/'CONFIG.json', dict(author_commit=HARL_SHA,args=args,arm=item['arm'],aux_sha256=job['aux_sha'],
                 actor_widths=[original.shape[0],space.shape[0]],critic_width=original.shape[0],
                 update='原作者 train 函数原样调用；其余为 UAM 采集封装', restart='完整优化状态，物理 episode 重启'))
        def snapshot():
            stem=f'R_{real:09d}_{updates:06d}'; model=folder/(stem+'.pkl'); normalizer=folder/(stem+'.norm.pkl')
            tmp=normalizer.with_name(normalizer.name+'.pending')
            norm.save(str(tmp))
            os.replace(tmp,normalizer)
            fields=dict(real_steps=real,updates=updates,actors=[a.actor.state_dict() for a in actors],
                        optimizers=[a.actor_optimizer.state_dict() for a in actors],critic=critic.critic.state_dict(),
                        critic_optimizer=critic.critic_optimizer.state_dict(),value_norm=vn.state_dict(),
                        value_norm_statistics={k:getattr(vn,k).detach().cpu() for k in ('running_mean','running_mean_sq','debiasing_term')},
                        rng=old.rng_record())
            old.atomic_pickle(model,old.cpu_checkpoint(fields))
            out=dict(model=model.name,norm=normalizer.name,real_steps=real,model_sha=hash_file(model),norm_sha=hash_file(normalizer))
            old.save(folder/'RESUME.json',out)
            protected={out['model'],out['norm']}
            # 只处理本轮本格自身生成的非代表性滚动件；保留两代和评估点。
            kept={str(p.stem) for p in (folder/'checkpoints').glob('*.json')}
            generations=sorted(folder.glob('R_?????????_??????.pkl'))
            for p in generations[:-2]:
                if p.name.split('_')[1] not in kept and p.name not in protected and not p.is_symlink():
                    n=p.with_suffix('.norm.pkl')
                    p.unlink()
                    if n.is_file() and not n.is_symlink():
                        n.unlink()
            return out
        def eval_factory(_):
            def predict(x,carry):
                phase=int(round(float(x[-(WIDTH+8)])))
                z=norm.normalize_obs(np.asarray(x)[None])
                if phase==0:
                    z=z[:,:-WIDTH]
                with torch.no_grad():
                    a,_,_=actors[phase].get_actions(z,zeros[:1],ones[:1])
                return int(a.detach().cpu().numpy()[0,0]), None
            return predict
        def evaluate_due(pointer):
            if real not in job['marks'] or (folder/'eval'/f'{real:09d}.json').is_file():
                return
            old.save(folder/'checkpoints'/f'{real:09d}.json',pointer)
            for a in actors: a.prep_rollout()
            norm.training=False
            try:
                old.evaluate(base,m,root,item,job['bank'],eval_factory,real,final=real==STEPS)
            finally:
                norm.training=True
        if meta:
            evaluate_due(meta)
        while real<STEPS and not old.stopping(root):
            if shutil.disk_usage(root).free<12*1024**3:
                break
            for buffer in buffers: buffer.step=0; buffer.factor=None
            cb.step=0
            for a in actors: a.prep_rollout(); a.lr_decay(real,STEPS)
            critic.prep_rollout(); critic.lr_decay(real,STEPS)
            for k in range(args['episode_length']):
                pre=obs[:,:-WIDTH].copy()
                with torch.no_grad():
                    values,_=critic.get_values(pre,zeros,masks)
                    p,lp,_=actors[0].get_actions(pre,zeros,masks)
                buffers[0].obs[k]=pre; buffers[0].masks[k]=masks; cb.share_obs[k]=pre
                mid,reward,done,_=norm.step(p.detach().cpu().numpy().ravel())
                if any(done) or np.any(reward):
                    raise ValueError('乘客阶段不能提前推进分钟或写入奖励')
                with torch.no_grad():
                    a,la,_=actors[1].get_actions(mid,zeros,masks)
                buffers[1].obs[k]=mid; buffers[1].masks[k]=masks
                nxt,reward,done,info=norm.step(a.detach().cpu().numpy().ravel())
                if any(x['physical_increment']!=1 for x in info):
                    raise ValueError('物理步计数不符')
                nm=(~done).astype(np.float32)[:,None]
                bad=np.asarray([[0. if x.get('TimeLimit.truncated',False) else 1.] for x in info],np.float32)
                next_pre=nxt[:,:-WIDTH].copy()
                for b,z,choice,lp0 in zip(buffers,(next_pre,nxt),(p,a),(lp,la)):
                    b.insert(z,zeros,choice.detach().cpu().numpy(),lp0.detach().cpu().numpy(),nm,ones)
                cb.insert(next_pre,zeros,values.detach().cpu().numpy(),reward[:,None],nm,bad)
                obs=nxt; masks=nm
            with torch.no_grad(): bootstrap,_=critic.get_values(obs[:,:-WIDTH].copy(),zeros,masks)
            cb.compute_returns(bootstrap.detach().cpu().numpy(),vn)
            for a in actors: a.prep_training()
            critic.prep_training()
            am,cm=author_train(runner)
            real+=ROLLOUT; updates+=1
            old.save(folder/'LIVE.json',dict(real_steps=real,updates=updates,seconds=time.time()-started,
                         real_fps=(real-initial)/max(1,time.time()-started)))
            pointer=snapshot()
            factors=[b.factor for b in buffers if b.factor is not None]
            with (folder/'UPDATE_AUDIT.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(old.plain(dict(real_steps=real,actor_metrics=am,critic_metrics=cm,
                         factor_nontrivial_fraction=max(float(np.mean(np.abs(z-1)>1e-5)) for z in factors))),ensure_ascii=False)+'\n')
            evaluate_due(pointer)
            for b in buffers: b.after_update()
            cb.after_update()
        if real==STEPS:
            final=old.read(folder/'eval'/f'{real:09d}.json')
            old.save(folder/'EVALUATION.json',dict(final=final,real_steps=real,weights_unchanged=True))
            if not old.stopping(root) and not (folder/'CF_DIAGNOSTICS.json').is_file():
                norm.training=False
                for a in actors: a.prep_rollout()
                def probabilities(x,role,u):
                    z=norm.normalize_obs(np.asarray(x)[None])
                    if role==0: z=z[:,:-WIDTH].copy()
                    with torch.no_grad():
                        pol=actors[role].actor
                        h=pol.base(torch.as_tensor(z,**pol.tpdv))
                        probs=pol.act.get_logits(h).softmax(-1).detach().cpu().numpy()[0]
                    return min(2,int(np.searchsorted(np.cumsum(probs),u,side='right')))
                try:
                    diagnostics(old,base,m,root,job,probabilities)
                except Exception:
                    old.save(folder/'CF_DIAGNOSTICS.json',dict(status='ENGINEERING_FAILED',traceback=traceback.format_exc()))
        old.save(folder/'END.json',dict(status='COMPLETE' if real==STEPS else 'PARTIAL',real_steps=real,
                                        seconds=time.time()-started,updates=updates))
    finally:
        (norm if norm is not None else vector).close()


def summarize(old,hist,root,tasks):
    out = []
    for item in tasks:
        folder=root/'cells'/item['id']
        if not old.complete_cell(folder):
            continue
        rows = old.read(folder/'EVALUATION.json')['final']
        final = [r['metrics'] for r in rows if r['trace'].startswith('order_83')]
        by = []
        for p in sorted((folder/'eval').glob('*.json')):
            dev = [r['metrics']['fixed_cutoff_cost'] for r in old.read(p) if r['trace'].startswith('order_43')]
            if len(dev)!=8:
                raise ValueError('开发评估数不符')
            by.append(dict(steps=int(p.stem),cost=statistics.mean(dev)))
        if len(final)!=12 or len(by)!=8:
            raise ValueError('正式评估矩阵覆盖不符')
        late=statistics.mean(r['cost'] for r in by[-5:])
        out.append(dict(arm=item['arm'],seed=item['seed'],late_cost=late,
                        final_cost=statistics.mean(m['fixed_cutoff_cost'] for m in final),
                        wait=statistics.mean(m['phases']['wait']/m['generated'] for m in final),
                        completion=statistics.mean(m['completion'] for m in final),
                        p95=statistics.mean(m['p95_cutoff'] for m in final),
                        full_att=statistics.mean(m['full_att'] for m in final) if all(m['full_att'] is not None for m in final) else None,
                        late_minus_best=late-min(r['cost'] for r in by), development_curve=by,
                        cf_status=old.read(folder/'CF_DIAGNOSTICS.json',{}).get('status','NOT_RUN')))
    h0 = [dict(seed=r['seed'],arm='H0_HISTORICAL',late_cost=r['late_cost'],final_cost=r['final_cost'],
               wait=r['final_wait_per_request'],completion=r['final_completion'])
          for r in hist['MODEL_LEVEL_SUMMARY.json']['models'] if r['case']==CASE and r['arm']=='HAPPO_RANDOM']
    idx={(r['arm'],r['seed']):r for r in out+h0}
    pairs={}
    for a,b in PARENT.items():
        pairs[f'{a}_minus_{b}']=[dict(seed=s,**{key:idx[a,s][key]-idx[b,s][key]
                                for key in ('late_cost','final_cost','wait','completion','p95')
                                if key in idx[a,s] and key in idx[b,s]})
                                for s in SEEDS if (a,s) in idx and (b,s) in idx]
    old.save(root/'MODEL_LEVEL_SUMMARY.json',dict(models=out, historical_h0=h0, paired=pairs,
             boundary='历史 H0 非同时期新训；开发与固定确认分开；训练种子才是独立单位'))


def pack(old,project,root,tasks):
    out=project/'experiment_archives'
    if out.is_symlink():
        raise ValueError('归档目录不得为链接')
    out.mkdir(exist_ok=True)
    for kind in ('DATAONLY','FINAL_REPLAY_MODELS'):
        dst=out/f'{RUN}_{kind}.tar.gz'
        tmp=dst.with_name(dst.name+f'.{os.getpid()}.pending')
        if dst.is_symlink() or tmp.is_symlink():
            raise ValueError('禁止向链接归档写入')
        with tmp.open('xb') as sink, tarfile.open(fileobj=sink,mode='w:gz',compresslevel=1) as tar:
            tar.add(__file__,arcname=RUN+'/ENTRYPOINT.py',recursive=False)
            def add(p):
                if p.is_file() and p.resolve()==p and p.is_relative_to(root):
                    tar.add(p,arcname=RUN+'/'+p.relative_to(root).as_posix(),recursive=False)
            for p in sorted(root.glob('*.json')):
                if kind=='DATAONLY' or p.name in ('PLAN.json','BANK.json','AUX_MANIFEST.json','END.json'):
                    add(p)
            for p in sorted((root/'bank'/'VARYING').glob('*.csv')):
                add(p)
            if kind=='DATAONLY':
                for area in ('cells','budget_attempts'):
                    for p in sorted((root/area).rglob('*')):
                        if p.suffix in ('.json','.jsonl','.log'):
                            add(p)
            else:
                add(root/'FROZEN_AUX.pkl')
                for item in tasks:
                    d=root/'cells'/item['id']
                    rec=old.read(d/'RESUME.json')
                    if not rec:
                        continue
                    for name in (rec['model'],rec['norm'],'RESUME.json','CONFIG.json','TASK.json'):
                        if Path(name).name!=name or not (d/name).is_file() or (d/name).is_symlink():
                            raise ValueError(f'模型归档配套不完整：{d/name}')
                        add(d/name)
        os.replace(tmp,dst)
        old.save(root/f'{kind}_PACKAGE.json',dict(path=str(dst),sha256=hash_file(dst),bytes=dst.stat().st_size))
        print('[结果包]',dst,flush=True)


def run_worker(old,root,path):
    job=old.read(path)
    task=job['task']; d=root/'cells'/task['id']
    if path.resolve()!=d/'JOB.json' or task['arm'] not in ARMS or task['steps']!=STEPS:
        raise ValueError('worker 的任务路径与合同不符')
    verify_vendor(Path(job['vendor_dir']))
    try:
        with old.GpuLease(root,3.0):
            train(old,root,job)
    except InterruptedError:
        old.save(d/'END.json',dict(status='PARTIAL',real_steps=old.resume_real(d),reason='停止请求'))
    except BaseException:
        old.save(d/'END.json',dict(status='ENGINEERING_FAILED',real_steps=max(
            old.resume_real(d),old.read(d/'LIVE.json',{}).get('real_steps',0)),traceback=traceback.format_exc()))
        raise


def main():
    p=argparse.ArgumentParser(description='K3 时变七臂 28 格，55,910,400 在线步，历史 H0 不重训')
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--baseline-archive',type=Path)
    p.add_argument('--vendor-dir',type=Path)
    p.add_argument('--plan',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--pack-only',action='store_true')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    project=a.project.resolve(strict=True)
    archive=(a.baseline_archive or project/'experiment_archives'/f'{OLD}_DATAONLY.tar.gz').absolute()
    vendor=(a.vendor_dir or project/'workspace/vendor/HARL').resolve()
    root=project/'serial_runs'/RUN
    if any(x.is_symlink() for x in (root,root.parent)):
        raise ValueError('拒绝通过链接访问运行目录')
    old,hist=module_from_archive(archive,project)
    plan=make_plan(hist)
    tasks=plan['tasks']
    if a.plan:
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        return
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,old.stop_signal if a.worker else stop_signal)
    if a.worker:
        run_worker(old,root,a.worker)
        return
    if not a.pack_only:
        if not os.environ.get('CUDA_VISIBLE_DEVICES','').isdecimal():
            p.error('请选择单卡 CUDA_VISIBLE_DEVICES=数字')
        verify_vendor(vendor)
    elif not (root/'PLAN.json').is_file():
        p.error('没有可打包结果')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'PARENT.lock').is_symlink():
        raise ValueError('父锁不得为链接')
    with old.lock_file(root/'PARENT.lock'):
        saved=old.read(root/'PLAN.json',None)
        if saved is not None and (saved!=plan or not (a.resume or a.pack_only)):
            raise ValueError('旧合同不同，或未指定 --resume')
        if saved is None:
            old.save(root/'PLAN.json',plan)
        for f in (root/'budget_attempts').glob('*.json'):
            z=old.read(f)
            if not z['closed']:
                if z['pid'] is None or old.alive(z['pid']):
                    raise ValueError('上次 worker 是否退出尚未确认；不重开或打包')
                old.finish_attempt(root,f,-999)
        if a.pack_only:
            summarize(old,hist,root,tasks)
            pack(old,project,root,tasks)
            return
        if (root/'STOP_REQUESTED').exists():
            print('存在 STOP_REQUESTED；确认停止后先移走标记再续训。',flush=True)
            return
        if shutil.disk_usage(root).free<12*1024**3:
            raise RuntimeError('可用磁盘低于 12 GiB，不启动')
        banks=old.demand_make(root)
        bank=banks['VARYING']; ref=hist['BANK.json']['VARYING']
        if bank['training_sha256']!=ref['training_sha256'] or (
            {b['name']:b['sha256'] for b in bank['bank']}!={b['name']:b['sha256'] for b in ref['bank']}):
            raise ValueError('训练或旧确认需求不一致')
        print(f'[合同] 七臂、28 格、{PLANNED:,} 在线步；先冻结共同离线模块',flush=True)
        old.save(root/'RUNNING.json',dict(phase='freeze_offline_modules',pid=os.getpid(),task=None,time=time.time()))
        try:
            fitted=auxiliary(root,old,hist,vendor)
        except InterruptedError:
            old.save(root/'RUNNING.json',dict(phase='partial_offline',task=None,pid=None,time=time.time()))
            return
        if HALT:
            return
        print('[模块]',json.dumps(fitted,ensure_ascii=False),flush=True)
        for task in tasks:
            if HALT or old.stopping(root) or shutil.disk_usage(root).free<12*1024**3:
                break
            d=root/'cells'/task['id']
            old.save(d/'TASK.json',task)
            if old.complete_cell(d):
                continue
            if task['arm'] not in fitted['available']:
                old.save(d/'END.json',dict(status='INSUFFICIENT_SUPPORT',real_steps=0,reason='冻结共同样本缺可验证监督'))
                continue
            if old.read(d/'END.json',{}).get('status')=='ENGINEERING_FAILED' and not a.retry_failed:
                continue
            pr=old.aggregate(root,tasks)
            pending=sum(max(0,STEPS-old.resume_real(root/'cells'/k['id'])) for k in tasks
                        if k['arm'] in fitted['available'] and not old.complete_cell(root/'cells'/k['id']))
            if pr['budget_charge_upper_bound']+pending+ROLLOUT>CAP:
                old.save(root/'BUDGET_STOP.json',dict(pending=pending,progress=pr))
                break
            previous=f'cells/{CASE}__HAPPO_RANDOM__s{task["seed"]}'
            cfg=hist[previous+'/CONFIG.json']
            if cfg['author_commit']!=HARL_SHA or hist[previous+'/END.json']['real_steps']!=STEPS:
                raise ValueError('旧 H0 对齐检验未通过')
            job=dict(task=task,steps=STEPS,root=str(root),trace=bank['trace'],trace_sha256=bank['training_sha256'],
                    bank=bank['bank'],marks=old.eval_marks(STEPS,ROLLOUT),
                    historical_args=cfg['args'],vendor_dir=str(vendor),aux_sha=fitted['sha256'])
            old.save(d/'JOB.json',job)
            start=old.resume_real(d)
            old.save(d/'LIVE.json',dict(real_steps=start))
            attempt=root/'budget_attempts'/f'{time.time_ns()}_{task["id"]}.json'
            fields=dict(id=task['id'],pid=None,initial_real=start,loss_bound=ROLLOUT,closed=False,
                        observed_increment=0,unlogged_upper_bound=ROLLOUT)
            old.save(attempt,fields)
            print(f'[训练 {task["run_order"]}/28] {task["id"]}',flush=True)
            with (d/'worker.log').open('a',encoding='utf-8') as log:
                try:
                    proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--project',str(project),
                               '--baseline-archive',str(archive),'--worker',str(d/'JOB.json')],
                               cwd=project,stdout=log,stderr=subprocess.STDOUT)
                except OSError:
                    old.save(attempt,dict(fields,closed=True,unlogged_upper_bound=0,returncode=-1))
                    old.save(d/'END.json',dict(status='ENGINEERING_FAILED',real_steps=start,traceback=traceback.format_exc()))
                    continue
                old.save(attempt,dict(fields,pid=proc.pid))
                while proc.poll() is None:
                    old.save(root/'RUNNING.json',dict(phase='train_or_eval',task=task,pid=proc.pid,
                             worker_progress=old.read(d/'LIVE.json',{}),time=time.time()))
                    if HALT and not (root/'STOP_REQUESTED').exists():
                        old.save(root/'STOP_REQUESTED',dict(time=time.time()))
                    time.sleep(4)
            old.finish_attempt(root,attempt,proc.returncode)
            old.aggregate(root,tasks)
        progress=old.aggregate(root,tasks)
        summarize(old,hist,root,tasks)
        status='COMPLETE' if progress['trained']==28 else 'PARTIAL'
        old.save(root/'END.json',dict(status=status,progress=progress,time=time.time()))
        old.save(root/'RUNNING.json',dict(phase=status.lower(),task=None,pid=None,time=time.time()))
        pack(old,project,root,tasks)


if __name__=='__main__':
    main()
