"""只读复算三组已完成归档；不加载策略、不运行仿真、不修改原始数据。"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import platform
import statistics as st
import tarfile
import time
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
PACKAGES = {
    'stage': 'uam_stage_stin_24run_60m_v1_DATAONLY.tar.gz',
    'space': 'uam_space_time_16run_48m_v1_DATAONLY.tar.gz',
    'diag': 'uam_threeway_diagnostics_20261104_v1_FROZEN_DATAONLY.tar.gz',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def stats(values):
    x = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not x:
        return dict(n=0, mean=None, sd=None, min=None, max=None, values=[])
    return dict(n=len(x), mean=st.mean(x), sd=st.stdev(x) if len(x) > 1 else None,
                min=min(x), max=max(x), values=x)


def quantile(x, q):
    x = sorted(x)
    if not x:
        return None
    at = (len(x) - 1) * q
    low = int(at)
    return x[low] + (x[min(low + 1, len(x) - 1)] - x[low]) * (at - low)


def paired(rows, a, b, key, control=None):
    """先在每个训练模型内汇总回放，再按训练种子配对；正数表示a成本更高。"""
    index = {(r['seed'], r['arm']): r for r in rows if control is None or r.get('control') == control}
    seeds = sorted({seed for seed, arm in index if (seed, a) in index and (seed, b) in index})
    delta = [index[(seed, a)][key] - index[(seed, b)][key] for seed in seeds]
    return dict(a=a, b=b, metric=key, seeds=seeds, **stats(delta),
                a_lower=sum(v < -1e-9 for v in delta), equal=sum(abs(v) <= 1e-9 for v in delta))


def group_stats(rows, groups, metrics):
    out = []
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(g) for g in groups)].append(row)
    for key, part in sorted(grouped.items(), key=lambda z: str(z[0])):
        out.append(dict(zip(groups, key), models=len(part),
                        **{m: stats(r.get(m) for r in part) for m in metrics}))
    return out


def people_pair(a, b):
    """收益只来自同一请求集合的截止前积分，不把未完成者当完整ATT。"""
    if a.get('request_identity') != b.get('request_identity'):
        raise ValueError('请求身份、OD或到达时间不同，拒绝逐人配对')
    ids = set(a['individual']) | set(b['individual'])
    gain = [a['individual'].get(p, 0) - b['individual'].get(p, 0) for p in ids]
    return dict(net=sum(gain), benefit=sum(v > 0 for v in gain), harm=sum(v < 0 for v in gain),
                unchanged=sum(v == 0 for v in gain), gross_gain=sum(max(0, v) for v in gain),
                gross_harm=sum(max(0, -v) for v in gain), worst_harm=max([0] + [-v for v in gain]))


def metric_digest(m, keep_people=True):
    if 'individual' not in m:
        return {k: v for k, v in m.items() if not isinstance(v, (list, dict))} | {
            'trace': m.get('trace'), 'trace_sha256': m.get('trace_sha256'),
            'final_station_queues': m.get('final_station_queues'),
            'control_diagnostics': m.get('control_diagnostics')}
    assert abs(sum(m['individual'].values()) - m['area']) < 1e-7
    assert abs(m['area'] / m['generated'] - m['fixed_cutoff_cost']) < 1e-7
    if m['unfinished']:
        assert m['full_att'] is None
    keys = ('area', 'generated', 'unfinished', 'completion', 'fixed_cutoff_cost', 'full_att',
            'cutoff', 'projection_count', 'decision_count', 'effectless_air', 'phases')
    out = {k: m.get(k) for k in keys}
    out['p95_exposure'] = quantile(list(m['individual'].values()), .95)
    out['max_exposure'] = max(m['individual'].values())
    logs = m.get('decision_log', [])
    meaningful = [r for r in logs if r.get('meaningful')]
    before = [r for r in meaningful if r['time'] < 300]
    out['meaningful_opportunities'] = len(meaningful)
    out['opportunities_before300'] = len(before)
    out['target_choice_opportunities'] = sum(len(r.get('legal', [])) > 1 for r in meaningful)
    out['positive_waits'] = sum(r.get('delay', 0) > 0 for r in meaningful)
    out['wait1'] = sum(r.get('delay') == 1 for r in meaningful)
    out['wait3'] = sum(r.get('delay') == 3 for r in meaningful)
    out['wait_minutes_requested'] = sum(r.get('delay', 0) for r in meaningful)
    out['gate_pending_or_other_ineffective'] = m.get('effectless_air')
    events = m.get('dispatch_events', m.get('gate_events', []))
    out['dispatch_batches'] = sum('executed' in event for event in events)
    out['dispatch_aircraft'] = sum(len(event.get('executed', [])) for event in events)
    points = m.get('area_curve', {})
    out['area300'] = points.get('300')
    if keep_people:
        out['individual'] = m['individual']
        out['request_identity'] = m['request_identity']
    contexts = m.get('contexts', [])
    if contexts:
        out['context_counts'] = dict(total=len(contexts),
            meaningful=sum(c['meaningful'] for c in contexts),
            meaningful_before300=sum(c['meaningful'] and c['time'] < 300 for c in contexts),
            meaningful_after300=sum(c['meaningful'] and c['time'] >= 300 for c in contexts),
            meaningful_active=sum(c['meaningful'] and c['active'] for c in contexts),
            meaningful_no_new_passenger=sum(c['meaningful'] and not c['active'] for c in contexts))
        out['contexts'] = [{k: c[k] for k in ('time', 'active', 'p', 'air', 'legal', 'meaningful',
                             'available', 'unfinished', 'state_sha', 'generation_ended')} for c in contexts]
    return out


def load_package(base, kind):
    """逐成员读取归档，不解压到项目，不执行归档中的任何源码。"""
    path = base / PACKAGES[kind]
    start = time.time()
    out = dict(cells={}, root={}, csv={}, entries=[], archive=dict(name=path.name, bytes=path.stat().st_size,
                                                                sha256=sha(path)))
    counts = collections.Counter()
    seen = set()
    with tarfile.open(path, 'r|gz') as tf:
        for member in tf:
            p = PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts or member.issym() or member.islnk():
                raise ValueError('归档含重定向或越界成员：' + member.name)
            if not member.isfile():
                continue
            if member.name in seen:
                raise ValueError('归档有重名文件：' + member.name)
            seen.add(member.name)
            counts[p.suffix] += 1
            out['archive']['uncompressed_bytes'] = out['archive'].get('uncompressed_bytes', 0) + member.size
            parts = p.parts
            if 'cells' in parts:
                j = parts.index('cells')
                case = parts[j + 1]
                cell = out['cells'].setdefault(case, dict(evaluations=[], curves=[]))
                if kind == 'diag':
                    trace = parts[j + 2]
                    stream = cell.setdefault('traces', {}).setdefault(trace, dict(states=[]))
                    if p.name not in ('BASELINE.json', 'SPACE_TIME.json', 'LEGAL_ONLINE.json', 'END.json'):
                        continue
                    d = json.load(tf.extractfile(member))
                    if p.name == 'BASELINE.json':
                        stream['baseline'] = metric_digest(d)
                    elif p.name == 'SPACE_TIME.json':
                        stream['space_time'] = d
                    elif p.name == 'LEGAL_ONLINE.json':
                        stream['online'] = [dict(mode=r['mode'], decisions=r['decisions'],
                                                 metrics=metric_digest(r['metrics']), gain=r['gain']) for r in d['results']]
                    elif 'states' in parts:
                        stream['states'].append(d)
                    else:
                        stream['end'] = d
                    continue
                if p.name.endswith('.json'):
                    if p.name in ('RESULT.json', 'NEW_ORDER_CONFIRMATION.json', 'DECISION_QUALITY.json',
                                  'TRAIN_END.json', 'END.json', 'INITIAL_POLICY_SHA.json', 'MODEL_MANIFEST.json',
                                  'TASK.json', 'EVALUATION.json', 'RULES_END.json', 'STAGE_EVAL_END.json'):
                        d = json.load(tf.extractfile(member))
                        if p.name == 'EVALUATION.json':
                            cell['task'] = d['task']
                            cell['late'] = [dict(step=cp['real_steps'], episodes=[dict(trace=e['trace'], sha=e['trace_sha256'],
                                                   metrics=metric_digest(e['metrics'])) for e in cp['episodes']])
                                            for cp in d['late_checkpoints']]
                            cell['final'] = [dict(trace=e['trace'], sha=e['trace_sha256'], metrics=metric_digest(e['metrics'])) for e in d['final']]
                            cell['real_steps'] = d['real_steps']
                        elif p.name == 'NEW_ORDER_CONFIRMATION.json':
                            cell['new_confirmation'] = [metric_digest(e) for e in d['episodes']]
                        elif p.name == 'RULES_END.json':
                            cell['rules'] = [dict(trace=e['trace'], rule=e['rule'], metrics=metric_digest(e['metrics'])) for e in d['results']]
                        elif p.name == 'MODEL_MANIFEST.json':
                            cell['manifest'] = {k: v for k, v in d.items() if k not in ('config', 'schema')}
                        else:
                            cell[p.stem] = d
                    elif '/evaluation/' in member.name and p.name.startswith('step_'):
                        d = json.load(tf.extractfile(member))
                        cell['curves'].append(dict(step=d['step'], episodes=[metric_digest(e) for e in d['episodes']]))
                elif p.name == 'progress.csv':
                    raw = tf.extractfile(member).read().decode('utf-8-sig')
                    entries = list(csv.DictReader(io.StringIO(raw)))
                    fps = [float(r['time/fps']) for r in entries if r.get('time/fps')]
                    cell['logged_fps'] = stats(fps)
                continue
            if p.suffix == '.csv' and p.name in ('MODEL_RESULTS.csv', 'RUN_STATUS.csv', 'PAIRED_EFFECTS.csv'):
                raw = tf.extractfile(member).read().decode('utf-8-sig')
                out['csv'][p.stem] = list(csv.DictReader(io.StringIO(raw)))
            elif p.suffix == '.json':
                out['root'][p.name] = json.load(tf.extractfile(member))
            elif p.name == 'ENTRYPOINT.py' or p.name.startswith('diagnose_uam_'):
                blob = tf.extractfile(member).read()
                out['entries'].append(dict(member=member.name, sha256=hashlib.sha256(blob).hexdigest()))
    out['archive'].update(files=len(seen), types=dict(counts), read_seconds=time.time() - start,
                          tar_scan_finished=True, unsafe_members=0, duplicate_members=0)
    return out


def avg(items, key):
    return st.mean([r[key] for r in items])


def stage_summary(data):
    rows, curve_rows, quality, rules, episodes = [], [], [], [], []
    for case, cell in data['cells'].items():
        if 'rules' in cell:
            rules.extend(cell['rules'])
        if 'RESULT' not in cell:
            continue
        d = cell['RESULT']; task = d['task']; curves = sorted(cell['curves'], key=lambda x: x['step'])
        late = [e for cp in curves if cp['step'] >= .8 * 2437120 for e in cp['episodes'] if str(e.get('trace')).startswith('order_430')]
        final = [e for cp in curves if cp['step'] == 2437120 for e in cp['episodes'] if str(e.get('trace')).startswith('order_430')]
        conf = cell['new_confirmation']
        late_recomputed = avg(late, 'restricted_person_time_per_generated')
        assert abs(late_recomputed - d['main_late_confirmation']['rpt']) < 1e-8
        assert len(late) == 44 and len(final) == 4 and len(conf) == 4
        seconds = cell['TRAIN_END']['wall_seconds']
        r = dict(id=case, seed=task['seed'], arm=task['arm'], control=task['control'],
                 late=late_recomputed, final=avg(final, 'restricted_person_time_per_generated'),
                 confirm=avg(conf, 'restricted_person_time_per_generated'),
                 late_completion=avg(late, 'completion'), final_completion=avg(final, 'completion'),
                 confirm_completion=avg(conf, 'completion'), late_full=st.mean(e['full'] for e in late),
                 confirm_full=st.mean(e['full'] for e in conf), seconds=seconds, real_fps=2437120 / seconds,
                 parameters=cell['manifest'].get('parameters'),
                 best=d['original_curve_stability']['best_rpt'], late_original=d['original_curve_stability']['late_rpt'],
                 retreat=d['original_curve_stability']['best_to_late_ratio'],
                 late_sd=d['original_curve_stability']['late_sd'],
                 late_slope=d['original_curve_stability']['late_slope_per_million'],
                 initial=cell.get('INITIAL_POLICY_SHA', {}).get('sha256'))
        rows.append(r)
        for cp in curves:
            for trace in ('original', 'bank'):
                part = [e for e in cp['episodes'] if e['trace'] == 'original'] if trace == 'original' else [e for e in cp['episodes'] if str(e['trace']).startswith('order_430')]
                if part:
                    curve_rows.append(dict(id=case, arm=r['arm'], control=r['control'], seed=r['seed'], step=cp['step'],
                                           scope=trace, cost=avg(part, 'restricted_person_time_per_generated'), completion=avg(part, 'completion')))
        for scope, part in [('final_validation', final), ('confirmation', conf)]:
            episodes.extend(dict(id=case, arm=r['arm'], control=r['control'], seed=r['seed'], scope=scope, **e) for e in part)
        for q in cell.get('DECISION_QUALITY', {}).get('rows', []):
            quality.append(dict(id=case, arm=r['arm'], control=r['control'], seed=r['seed'], **q))
    assert len(rows) == 24
    groups = group_stats(rows, ['control', 'arm'], ['late', 'final', 'confirm', 'late_completion', 'confirm_completion',
                         'late_full', 'confirm_full', 'retreat', 'late_sd', 'real_fps', 'parameters'])
    comparisons = [paired(rows, a, b, metric, 'J3') for a, b in [('STIN_ON','STIN_OFF'), ('CONDITIONAL','SPLIT'),
                   ('SPLIT','JOINT'), ('CONDITIONAL','JOINT'), ('STIN_OFF','JOINT'), ('STIN_ON','JOINT')]
                   for metric in ('late', 'confirm')]
    for metric in ('late', 'confirm'):
        j = {r['seed']:r for r in rows if r['arm']=='JOINT' and r['control']=='J3'}
        s = {r['seed']:r for r in rows if r['arm']=='JOINT' and r['control']=='LQ'}
        differences = [j[k][metric]-s[k][metric] for k in sorted(j)]
        comparisons.append(dict(a='Joint', b='Single_LQ', metric=metric, seeds=sorted(j), **stats(differences), a_lower=sum(x<0 for x in differences)))
    # 同一共同READY状态及相同后续控制器下，检查候选代价是否可跨模型复用。
    common = collections.defaultdict(list)
    for q in quality:
        if q['origin'] == 'common_ready' and q['control'] == 'J3':
            common[(q['trace'],q['state_sha'])].append(q)
    common_audit = []
    for (trace, state), qs in common.items():
        costs = collections.defaultdict(set)
        for q in qs:
            for branch in q['branches']:
                costs[tuple(branch['action'])].add(branch['cost'])
        common_audit.append(dict(trace=trace,state_sha=state,models=len(qs),candidate_cost_inconsistencies=sum(len(v)>1 for v in costs.values())))
    q_per_model=[]
    for r in rows:
        own = [q for q in quality if q['id']==r['id'] and q['origin']=='own']
        common_q = [q for q in quality if q['id']==r['id'] and q['origin']=='common_ready']
        q_per_model.append(dict(id=r['id'],arm=r['arm'],control=r['control'],seed=r['seed'],
             own_regret=st.mean(q['loss']/300 for q in own) if own else None,
             common_regret=st.mean(q['loss']/300 for q in common_q) if common_q else None,
             common_errors=sum(q['loss']>0 for q in common_q), common_cases=len(common_q),
             own_cases=len(own)))
    rule_groups=[]
    for scope, names in [('validation', [f'order_{i}' for i in range(4301,4305)]), ('confirmation',[f'order_{i}' for i in range(6501,6505)])]:
        for rule in sorted({r['rule'] for r in rules}):
            part=[r['metrics'] for r in rules if r['rule']==rule and r['trace'] in names]
            rule_groups.append(dict(scope=scope,rule=rule,episodes=len(part),cost=avg(part,'restricted_person_time_per_generated'),completion=avg(part,'completion'),full=sum(e['full'] for e in part)))
    return dict(rows=rows, groups=groups, comparisons=comparisons, curves=curve_rows,
                episodes=episodes, rules=rule_groups, quality=quality, quality_per_model=q_per_model, common_state_audit=common_audit)


def space_summary(data):
    rows, episodes, pairs = [], [], []
    for case, cell in data['cells'].items():
        if 'final' not in cell:
            continue
        task = cell['task']; late=[e['metrics'] for cp in cell['late'] for e in cp['episodes']]
        final=[e['metrics'] for e in cell['final'][:4]]; conf=[e['metrics'] for e in cell['final'][4:]]
        assert cell['real_steps']==3010560 and len(final)==4 and len(conf)==4
        r=dict(id=case,arm=task['arm'],seed=task['seed'],late=avg(late,'fixed_cutoff_cost'),
               final=avg(final,'fixed_cutoff_cost'),confirm=avg(conf,'fixed_cutoff_cost'),
               late_completion=avg(late,'completion'),final_completion=avg(final,'completion'),
               confirm_completion=avg(conf,'completion'),late_full=st.mean(e['unfinished']==0 for e in late),
               confirm_full=st.mean(e['unfinished']==0 for e in conf),
               real_fps=3010560/cell['TRAIN_END']['seconds'],seconds=cell['TRAIN_END']['seconds'],
               late_checkpoints=[cp['step'] for cp in cell['late']])
        for key in ('positive_waits','wait1','wait3','wait_minutes_requested','meaningful_opportunities','target_choice_opportunities','dispatch_batches','dispatch_aircraft','p95_exposure','max_exposure','projection_count'):
            r['confirm_'+key]=avg(conf,key)
        rows.append(r)
        for e in cell['final']:
            episodes.append(dict(id=case,arm=r['arm'],seed=r['seed'],trace=e['trace'],sha=e['sha'],
                                 scope='validation' if int(e['trace'].split('_')[1])<7000 else 'confirmation',**e['metrics']))
    assert len(rows)==16
    groups=group_stats(rows,['arm'],['late','final','confirm','late_completion','confirm_completion','late_full','confirm_full',
               'real_fps','confirm_positive_waits','confirm_wait1','confirm_wait3','confirm_meaningful_opportunities',
               'confirm_dispatch_batches','confirm_dispatch_aircraft','confirm_p95_exposure','confirm_max_exposure','confirm_projection_count'])
    contrasts=[paired(rows,a,b,m) for a,b in [('S1_T0','S0_T0'),('S0_T1','S0_T0'),('S1_T1','S1_T0'),('S1_T1','S0_T1'),('S1_T1','S0_T0')] for m in ('late','confirm')]
    by={(r['seed'],r['arm']):r for r in rows}
    interaction={m:stats(by[(seed,'S1_T1')][m]-by[(seed,'S1_T0')][m]-by[(seed,'S0_T1')][m]+by[(seed,'S0_T0')][m] for seed in (2001,2002,2003,2004)) for m in ('late','confirm')}
    index={(e['seed'],e['arm'],e['trace']):e for e in episodes}
    for e in episodes:
        if e['arm']=='S0_T0':continue
        ref=index[(e['seed'],'S0_T0',e['trace'])]
        assert ref['sha']==e['sha']
        result=people_pair(ref,e)
        pairs.append(dict(seed=e['seed'],arm=e['arm'],trace=e['trace'],scope=e['scope'],
                          cost_gain=result['net']/300,completion_delta=e['completion']-ref['completion'],**result))
    per_model_pairs=[]
    for r in rows:
        for scope in ('validation','confirmation'):
            part=[p for p in pairs if p['seed']==r['seed'] and p['arm']==r['arm'] and p['scope']==scope]
            if part:per_model_pairs.append(dict(seed=r['seed'],arm=r['arm'],scope=scope,
                **{key:avg(part,key) for key in ('cost_gain','benefit','harm','gross_gain','gross_harm','worst_harm','completion_delta')}))
    return dict(rows=rows,groups=groups,contrasts=contrasts,interaction=interaction,episodes=episodes,
                individual_pairs=pairs,individual_model_pairs=per_model_pairs)


def case_meta(case):
    if case=='READY_J3':return dict(family='READY',seed=None,arm='READY',control='J3')
    parts=case.split('__');seed=int(parts[-1].removeprefix('s'))
    if seed>=1900:return dict(family='stage',seed=seed,arm=parts[-2],control=parts[-3])
    return dict(family='credit',seed=seed,arm=parts[-2],control=parts[-3])


def diag_summary(data):
    baseline_rows, online_rows, state_rows, candidate_rows, job_rows = [], [], [], [], []
    for case, cell in data['cells'].items():
        # 运行时会生成一个s0辅助目录，它不是已登记的诊断单元。
        traces={k:v for k,v in cell.get('traces',{}).items() if 'baseline' in v}
        if not traces:
            continue
        meta=case_meta(case)
        for trace, stream in traces.items():
            b=stream['baseline'];end=stream['end'];n=b['generated']
            assert end['status']=='COMPLETE'
            job_rows.append(dict(case=case,trace=trace,**meta,**end))
            baseline_rows.append(dict(case=case,trace=trace,**meta,**{k:v for k,v in b.items() if k not in ('individual','request_identity','contexts')}))
            for mode in stream.get('online',[]):
                met=mode['metrics'];g=people_pair(b,met)
                assert abs(g['net']-mode['gain']['total_gain'])<1e-7
                decisions=mode['decisions'];valid=[x for x in decisions if 'choice' in x]
                natural={c['time']:c['air'] for c in b.get('contexts',[])}
                early=(b['area300']-met['area300'])/n if b['area300'] is not None and met['area300'] is not None else None
                online_rows.append(dict(case=case,trace=trace,**meta,mode=mode['mode'],
                    gain=g['net']/n,before300_gain=early,after300_gain=g['net']/n-early if early is not None else None,
                    completion_delta=met['completion']-b['completion'],completion=met['completion'],
                    benefited=g['benefit'],harmed=g['harm'],worst_harm=g['worst_harm'],gross_gain=g['gross_gain'],gross_harm=g['gross_harm'],
                    decisions=len(valid),agreements=sum(x['half_draws_agree'] for x in valid),
                    real_changes=sum(x['choice']!={'target':natural.get(x['time']), 'delay':0} for x in valid),
                    choices=[x['choice'] for x in valid],opportunity_times=[x['time'] for x in valid]))
            for state in stream['states']:
                ctx=state['context']; outputs=state['outputs'];ref=outputs[0];ref_cost=ref['cost'];plan=ref['plan']
                assert abs(ref_cost-b['area'])<1e-7 and state['reference_matched']
                subsets={'space':[r for r in outputs if r['plan']['delay']==0],
                         'time':[r for r in outputs if r['plan']['target']==plan['target']], 'both':outputs}
                result=dict(case=case,trace=trace,**meta,time=ctx['time'],active=ctx['active'],
                            available=ctx['available'],unfinished=ctx['unfinished'],legal=len(ctx['legal']),
                            statesha=ctx['state_sha'],candidate_count=len(outputs))
                for name, part in subsets.items():
                    winner=min(part,key=lambda x:(x['cost'],x['plan']!=plan,str(x['plan'])))
                    result[name+'_gain']=(ref_cost-winner['cost'])/n
                    result[name+'_candidates']=len(part)
                    result[name+'_winner']=winner['plan']
                    result[name+'_gross_harm']=winner['gain']['lost_person_minutes']
                    result[name+'_harmed']=winner['gain']['harmed']
                ranked=stream.get('space_time',{}).get('states',[])
                rank=next((x for x in ranked if x['time']==ctx['time']),None)
                if rank:
                    result['ranking_top_changed']=rank['ranking_60'][0]!=rank['ranking_full'][0]
                # 短窗口相同分数不能误称为错误排序；同时列乐观、悲观的并列决策代价。
                short_best=max(o['gain_60'] for o in outputs)
                short_ties=[o for o in outputs if abs(o['gain_60']-short_best)<1e-7]
                full_best=min(o['cost'] for o in outputs)
                result['short_ties']=len(short_ties)
                result['short_best_case_regret']=(min(o['cost'] for o in short_ties)-full_best)/n
                result['short_worst_case_regret']=(max(o['cost'] for o in short_ties)-full_best)/n
                result['short_identified_wrong']=result['short_best_case_regret']>1e-7
                best_both=min(outputs,key=lambda o:(o['cost'],o['plan']!=plan))
                result['best_both_before300_gain']=best_both['gain_before300']/n if best_both['gain_before300'] is not None else None
                result['before300']=ctx['time']<300
                state_rows.append(result)
                for o in outputs:
                    candidate_rows.append(dict(case=case,trace=trace,**meta,time=ctx['time'],target=o['plan']['target'],delay=o['plan']['delay'],
                        gain=(ref_cost-o['cost'])/n, gain60=o['gain_60']/n,gain120=o['gain_120']/n,
                        before300_gain=o['gain_before300']/n if o['gain_before300'] is not None else None,
                        before300=ctx['time']<300,completion=o['completion'],harmed=o['gain']['harmed']))
    # 每模型先平均两个顺序；不把状态数量多的模型赋予更多统计权重。
    oracle_models=[]
    for case in sorted({r['case'] for r in state_rows}):
        part=[r for r in state_rows if r['case']==case]
        for scope in ('all','before300','after300','no_new_passenger'):
            current=[r for r in part if scope=='all' or scope=='before300' and r['time']<300 or scope=='after300' and r['time']>=300 or scope=='no_new_passenger' and not r['active']]
            if not current:continue
            oracle_models.append(dict(case=case,**case_meta(case),scope=scope,states=len(current),
                **{k:avg(current,k) for k in ('space_gain','time_gain','both_gain')},
                positive_space=st.mean(r['space_gain']>0 for r in current),
                positive_time=st.mean(r['time_gain']>0 for r in current),
                rank_change=st.mean(r.get('ranking_top_changed',False) for r in current),
                strict_short_wrong=st.mean(r['short_identified_wrong'] for r in current),
                short_best_case_regret=st.mean(r['short_best_case_regret'] for r in current)))
    online_models=[]
    for case in sorted({r['case'] for r in online_rows}):
        for mode in ('space','time','both'):
            part=[r for r in online_rows if r['case']==case and r['mode']==mode]
            if not part:continue
            online_models.append(dict(case=case,**case_meta(case),mode=mode,
                 **{k:avg(part,k) for k in ('gain','before300_gain','after300_gain','completion_delta','benefited','harmed','worst_harm','gross_gain','gross_harm')},
                 agreements=sum(r['agreements'] for r in part),decisions=sum(r['decisions'] for r in part),
                 episodes=len(part),episode_wins=sum(r['gain']>0 for r in part)))
    # 成本分解说明差距出现在哪个环节，不把会计分解误称为因果份额。
    phase_models=[]
    for case in sorted({r['case'] for r in baseline_rows}):
        part=[r for r in baseline_rows if r['case']==case]
        phase_models.append(dict(case=case,**case_meta(case),
            cost=avg(part,'fixed_cutoff_cost'),
            **{phase:st.mean(r['phases'].get(phase,0)/r['generated'] for r in part)
               for phase in ('ground','wait','flight')}))
    return dict(jobs=job_rows,baselines=baseline_rows,states=state_rows,candidates=candidate_rows,
                baseline_phase_models=phase_models,
                baseline_phase_groups=group_stats(phase_models,['family','arm','control'],['cost','ground','wait','flight']),
                oracle_models=oracle_models,oracle_groups=group_stats(oracle_models,['family','arm','scope'],['space_gain','time_gain','both_gain','positive_space','positive_time','rank_change','strict_short_wrong','short_best_case_regret']),
                online_rows=online_rows,online_models=online_models,
                online_groups=group_stats(online_models,['family','arm','mode'],['gain','before300_gain','after300_gain','completion_delta','benefited','harmed','worst_harm']),
                total_branch_calls=sum(r.get('branch_calls',2) for r in job_rows),
                total_sampled_states=sum(r.get('sampled_states',0) for r in job_rows),
                wall_seconds=data['root']['FROZEN_END.json']['wall_seconds'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archives',type=Path,required=True);args=ap.parse_args()
    # 分析口径在计算组间结果之前落盘，不追加按结果挑选的新主指标。
    save(HERE/'ANALYSIS_PLAN.json',dict(
        primary='每组沿用原固定后期窗口的全体生成乘客截止人时；先模型内平均，再训练种子配对',
        secondary='最终验证、固定新增顺序确认、完成率、个体收益损失、真实运行时间',
        contrasts_stage=['STIN_ON-STIN_OFF','CONDITIONAL-SPLIT','SPLIT-JOINT','CONDITIONAL-JOINT','Joint-Single'],
        contrasts_space='空间控制与等待控制2x2配对效应及交互',
        interpretation='探索性分析；四训练种子，保留逐种子结果；不使用大样本显著性，不挑Best',
        missing='未完成者ATT为空；不删除坏种子；诊断状态不作为独立训练复制',
        boundaries='不同控制权限、训练步数及确定/随机评估协议不合并成同一排行榜',
        execution='纯标准库只读归档复算，不运行任何策略或仿真，不加载pickle或模型权重'))
    packages={}
    for kind in PACKAGES:
        print('读取',kind,flush=True);packages[kind]=load_package(args.archives,kind)
        print('读取完成',kind,packages[kind]['archive'],flush=True)
    result=dict(stage=stage_summary(packages['stage']),space=space_summary(packages['space']),diag=diag_summary(packages['diag']))
    audit=dict(python=platform.python_version(),inputs={k:d['archive'] for k,d in packages.items()},
               source_entries={k:d['entries'] for k,d in packages.items()},
               raw_packages_unchanged=True,new_training_transitions=0,verification='ANALYZED；数值复算不是策略重跑复现',
               integrity_checks=['tar逐成员扫描完成','归档成员无重名或重定向','逐人人时守恒','三组主要口径及种子层级隔离','60M官方44次后期回放汇总复算一致'])
    # 完整逐人中间结果留在本地；仓库版去掉重复原始轨迹，便于后续聊天AI读取。
    save(HERE/'analysis_details.json',result)
    result['stage'].pop('episodes',None)
    result['space'].pop('episodes',None)
    result['stage']['quality']=[{k:v for k,v in row.items() if k!='branches'} for row in result['stage']['quality']]
    save(HERE/'analysis_summary.json',result);save(HERE/'INPUT_AUDIT.json',audit)
    for k in ('stage','space'):
        print(k,'组汇总',json.dumps(result[k]['groups'],ensure_ascii=False),flush=True)
    print('诊断规模',len(result['diag']['jobs']),len(result['diag']['states']),result['diag']['total_branch_calls'],flush=True)
    print('输出',HERE/'analysis_summary.json',flush=True)


if __name__=='__main__':
    main()
