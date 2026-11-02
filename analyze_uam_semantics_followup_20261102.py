#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只读结果包复算：决策语义、预测建议、逐选择损失和历史吞吐；不加载模型。"""
from __future__ import annotations
import argparse, csv, hashlib, io, itertools, json, math, statistics, tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class SafeArchive:
    """只转换明确授权包中的JSON/CSV，禁止解压入口、执行脚本或反序列化模型。"""
    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_symlink() or not self.path.is_file(): raise ValueError('结果包不是普通文件')
        self.tar = tarfile.open(path, 'r:gz'); members = self.tar.getmembers()
        if len(members) > 12000 or sum(m.size for m in members) > 2 * 1024**3: raise ValueError('包超出明确扫描上限')
        self.members = {}
        for m in members:
            parts = PurePosixPath(m.name).parts
            if not parts or m.name.startswith('/') or '..' in parts or '\\' in m.name or m.issym() or m.islnk():
                raise ValueError('包成员路径异常')
            if m.name in self.members: raise ValueError('包成员重名')
            self.members[m.name] = m
        self.root = PurePosixPath(members[0].name).parts[0]
    def blob(self, name):
        m = self.members[name]
        if not m.isfile() or m.size > 32 * 1024**2: raise ValueError('只读取有界普通成员')
        return self.tar.extractfile(m).read()
    def data(self, name):
        def unique(pairs):
            result = {}
            for k, v in pairs:
                if k in result: raise ValueError('JSON键重复')
                result[k] = v
            return result
        return json.loads(self.blob(name).decode('utf-8-sig'), object_pairs_hook=unique,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError('非有限JSON常数')))
    def at(self, relative): return self.data(self.root + '/' + relative)
    def close(self): self.tar.close()


def csv_out(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)


def summary(values):
    x = list(values)
    if not x: return dict(n=0)
    return dict(n=len(x), mean=statistics.mean(x), sd=statistics.stdev(x) if len(x) > 1 else None,
                median=statistics.median(x), minimum=min(x), maximum=max(x),
                leave_one_mean_range=[min(statistics.mean(x[:i] + x[i+1:]) for i in range(len(x))),
                                      max(statistics.mean(x[:i] + x[i+1:]) for i in range(len(x)))] if len(x) > 1 else None)


def signed_reference(values):
    from scipy import stats
    x = list(values); result = summary(x); n = len(x)
    if n < 2: return result
    mu = statistics.mean(x); se = statistics.stdev(x) / math.sqrt(n)
    result['t_reference_ci95'] = [mu - stats.t.ppf(.975, n - 1) * se, mu + stats.t.ppf(.975, n - 1) * se]
    # 只对有正有负可能的预测增益作参考，不检验按最小候选构造的非负空间。
    result['exact_sign_flip_p'] = sum(abs(sum(s * v for s, v in zip(signs, x))) >= abs(sum(x)) - 1e-8
                                      for signs in itertools.product((-1, 1), repeat=n)) / 2**n
    result['positive_replicates'] = sum(v > 0 for v in x)
    result['raw_values'] = x
    result['shapiro_reference_p'] = float(stats.shapiro(x).pvalue) if 3 <= n <= 5000 and len(set(x)) > 1 else None
    result['independence_limit'] = '只有四个独立训练种子；验证顺序和抽样状态不扩充独立样本量。'
    return result


def analyze(diag_path, training_path, models_path, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    diag = SafeArchive(diag_path); training = SafeArchive(training_path); models = SafeArchive(models_path)
    try:
        end, manifest, audit = diag.at('SESSION_END.json'), diag.at('MANIFEST.json'), diag.at('统计与完整性边界.json')
        if end['status'] != 'COMPLETE' or len(end['results']) != 26 or any(r['status'] != 'COMPLETE' for r in end['results']):
            raise ValueError('不是完整的26单元诊断')
        if hashlib.sha256(diag.blob(diag.root + '/ENTRYPOINT.py')).hexdigest() != manifest['script_sha']:
            raise ValueError('诊断入口与清单hash不一致')
        original_plan = training.at('SUITE_PLAN.json'); model_plan = models.at('SUITE_PLAN.json')
        if original_plan != model_plan: raise ValueError('原记录包与模型包训练合同不同')
        bank_hash = {b['name']: b['sha256'] for b in original_plan['config']['bank']}
        if any(bank_hash[b['name']] != b['sha256'] for b in manifest['bank']): raise ValueError('需求银行改变')
        model_matches = 0
        for c in manifest['cases']:
            cell = c['task']['id']
            for key, suffix in (('model_sha', 'zip'), ('norm_sha', 'pkl')):
                member = f'{models.root}/cells/{cell}/checkpoints/step_002396160.{suffix}'
                if hashlib.sha256(models.blob(member)).hexdigest() != c[key]: raise ValueError('模型配套hash不符')
                model_matches += 1
        states = []; baseline = []; opportunities = []; chains = []; counters = []
        for name in diag.members:
            if '/state_' in name and name.endswith('.json'):
                states.append(dict(case=name.split('/')[3], **diag.data(name)))
            elif '/jobs/baseline/' in name and name.endswith('/own_BASE.json'):
                e = diag.data(name); case = name.split('/')[3]; bank = name.split('/')[-2]
                baseline.append(dict(case=case, bank=bank, **e['metrics']))
                op = e['events']['decision_opportunities']; ch = e['events']['aircraft_chains']
                opportunities.append(dict(case=case, bank=bank, ticks=len(op),
                    passenger_decisions=sum(r['passenger_active'] for r in op),
                    pre_effective=sum(r['before']['effective'] for r in op),
                    dispatch_effective=sum(r['dispatch']['effective'] for r in op),
                    no_passenger_effective=sum(r['dispatch']['effective'] and not r['passenger_active'] for r in op),
                    pre_post_change=sum(r['before'] != r['dispatch'] for r in op)))
                chains += [dict(case=case, bank=bank, **v) for v in ch]
            elif name.endswith('/COUNTERS.json'): counters.append(diag.data(name))
        if len(states) != 104 or len(baseline) != 34: raise ValueError('实际覆盖与设计不符')
        state_rows = []; forecast_rows = []; holds = []; examples = []; common = defaultdict(set)
        for s in states:
            if not s['clone_reference_passed']: raise ValueError('克隆校验未通过')
            if s['origin'] == 'common_ready': common[(s['bank'], s['index'])].add(s['state_sha'])
            kind = 'READY' if s['case'] == 'READY_K3' else 'J' if '__J3__' in s['case'] else 'S'
            seed = None if kind == 'READY' else int(s['case'].rsplit('__s', 1)[1])
            for r in s['results']:
                rows = s['rows'][r['continuation']]; reference = next(v for v in rows if v['action'] == r['reference'])
                winner = min(rows, key=lambda v: (v['total'], v['action'] != r['reference'], tuple(v['action'])))
                loss = reference['total'] - winner['total']
                if loss != r['avoidable_person_minutes']: raise ValueError('候选损失复算不一致')
                if sum(r['phase_gains'].values()) != loss or r['focal_gain'] + r['existing_gain'] + r['future_gain'] != loss:
                    raise ValueError('人时拆分不守恒')
                state_rows.append(dict(case=s['case'], kind=kind, seed=seed, bank=s['bank'], index=s['index'], time=s['time'],
                    origin=s['origin'], follow=r['continuation'], loss=loss, normal_gain=r['before_absorbing_gain'],
                    tail_gain=r['tail_absorbing_gain'], focal_gain=r['focal_gain'], existing_gain=r['existing_gain'], future_gain=r['future_gain'],
                    wait_gain=r['phase_gains']['wait'], candidate_span=r['candidate_span'], near_best=r['near_best'],
                    passenger_only=r['passenger_only_gain'], aircraft_only=r['aircraft_only_gain'], joint_extra=r['joint_extra_beyond_best_single'],
                    flip60=r['temporal'][4]['short_full_optima_disjoint'], flip120=r['temporal'][5]['short_full_optima_disjoint'],
                    remaining_reference=r['remaining_reference'], remaining_winner=r['remaining_winner']))
            f = s.get('forecast')
            if f and f['status'] == 'COMPLETE':
                r = s['results'][0]; rows = s['rows'][r['continuation']]
                ref = next(v for v in rows if v['action'] == r['reference']); picked = next(v for v in rows if v['action'] == f['picked'])
                if ref['total'] - picked['total'] != f['realized_full_gain']: raise ValueError('预测实际收益复算不一致')
                forecast_rows.append(dict(case=s['case'], kind=kind, seed=seed, bank=s['bank'], index=s['index'],
                    gain=f['realized_full_gain'], gain120=f['realized_120_gain'], oracle120=f['realized_120_oracle_gain'],
                    normal_gain=ref['before_absorbing_area'] - picked['before_absorbing_area'],
                    tail_gain=ref['tail_absorbing_area'] - picked['tail_absorbing_area'], agree=f['split_draws_agree'],
                    remaining_reference=ref['remaining'], remaining_picked=picked['remaining']))
            if s.get('persistence_stress'):
                holds.append(dict(case=s['case'], seed=seed, bank=s['bank'], index=s['index'], **s['persistence_stress']))
            if kind == 'READY' and s['origin'] == 'own':
                examples.append(dict(bank=s['bank'], index=s['index'], **s['results'][0], forecast=f))
        if any(len(v) != 1 for v in common.values()): raise ValueError('共同状态不一致')
        group = defaultdict(list); model_group = defaultdict(list)
        for r in state_rows:
            group[(r['kind'], r['origin'], r['follow'])].append(r)
            model_group[(r['case'], r['kind'], r['seed'], r['origin'], r['follow'])].append(r)
        model_rows = []
        for (case, kind, seed, origin, follow), rows in model_group.items():
            model_rows.append(dict(case=case, kind=kind, seed=seed, origin=origin, follow=follow, states=len(rows),
                **{k: statistics.mean(r[k] for r in rows) for k in ('loss', 'normal_gain', 'tail_gain', 'focal_gain', 'existing_gain', 'future_gain',
                                                                   'wait_gain', 'near_best', 'passenger_only', 'aircraft_only', 'joint_extra', 'flip60', 'flip120')}))
        grouped = []
        for (kind, origin, follow), rows in group.items():
            values = [r for r in model_rows if (r['kind'], r['origin'], r['follow']) == (kind, origin, follow)]
            grouped.append(dict(kind=kind, origin=origin, follow=follow, states=len(rows), models=len(values),
                **{k: summary(r[k] for r in values) for k in ('loss', 'normal_gain', 'tail_gain', 'passenger_only', 'aircraft_only', 'joint_extra')},
                flip60=sum(r['flip60'] for r in rows), flip120=sum(r['flip120'] for r in rows)))
        fore_model = []
        for case in sorted({r['case'] for r in forecast_rows}):
            rows = [r for r in forecast_rows if r['case'] == case]
            fore_model.append(dict(case=case, kind=rows[0]['kind'], seed=rows[0]['seed'], states=len(rows),
                gain=statistics.mean(r['gain'] for r in rows), normal_gain=statistics.mean(r['normal_gain'] for r in rows),
                gain120=statistics.mean(r['gain120'] for r in rows), positive=sum(r['gain'] > 0 for r in rows),
                negative=sum(r['gain'] < 0 for r in rows), agree=sum(r['agree'] for r in rows)))
        fore_stats = {kind: signed_reference(r['gain'] for r in fore_model if r['kind'] == kind) for kind in ('J', 'S')}
        # 两项预测增益参考检验作Holm校正，不对按最小值构造的空间作显著性检验。
        ordered = sorted(fore_stats, key=lambda k: fore_stats[k]['exact_sign_flip_p']); running = 0.
        for rank, key in enumerate(ordered):
            running = max(running, (2 - rank) * fore_stats[key]['exact_sign_flip_p'])
            fore_stats[key]['holm_p'] = min(1., running)
        timing_group = []
        for kind in ('J', 'S', 'READY'):
            rows = [r for r in opportunities if ('READY' if r['case'] == 'READY_K3' else 'J' if '__J3__' in r['case'] else 'S') == kind]
            timing_group.append(dict(kind=kind, episodes=len(rows), ticks=sum(r['ticks'] for r in rows),
                pre_effective=sum(r['pre_effective'] for r in rows), effective=sum(r['dispatch_effective'] for r in rows),
                no_passenger_effective=sum(r['no_passenger_effective'] for r in rows),
                effective_per_episode=statistics.mean(r['dispatch_effective'] for r in rows)))
        baseline_group = []
        for kind in ('J', 'S', 'READY'):
            for net in (('READY',) if kind == 'READY' else ('L', 'T')):
                rows = [r for r in baseline if (r['case'] == 'READY_K3' if kind == 'READY' else
                    ('__J3__' in r['case']) == (kind == 'J') and r['case'] != 'READY_K3' and f'__{net}_G099' in r['case'])]
                by_case = defaultdict(list)
                for r in rows: by_case[r['case']].append(r['restricted_person_time_per_generated'])
                baseline_group.append(dict(kind=kind, network=net, rpt=summary(statistics.mean(v) for v in by_case.values()),
                    full=sum(r['full'] for r in rows), episodes=len(rows), completion=statistics.mean(r['completion'] for r in rows)))
        train_ends = [training.data(name) for name in training.members if name.endswith('/TRAIN_END.json')]
        result_ends = [training.data(name) for name in training.members if name.endswith('/EVAL_END.json')]
        root_end = training.at('END.json')
        train_seconds = sum(r['wall_seconds'] for r in train_ends); steps = sum(r['steps'] for r in train_ends)
        fps_by_group = []
        for control in ('J3', 'LQ'):
            for net in ('LSTM', 'TRANSFORMER'):
                rows = [r for r in train_ends if r['task']['control'] == control and r['task']['network'] == net and r['task']['gamma'] == .99]
                fps_by_group.append(dict(control=control, network=net, steps=sum(r['steps'] for r in rows),
                    wall_seconds=sum(r['wall_seconds'] for r in rows), fps=sum(r['steps'] for r in rows) / sum(r['wall_seconds'] for r in rows)))
        runtime = dict(diag_wall_seconds=end['wall_seconds'], full_episodes=sum(c['full_episodes'] for c in counters),
            branch_calls=sum(c['branch_calls'] for c in counters), simulated_minutes=sum(c['simulated_minutes'] for c in counters),
            sum_worker_seconds=sum(c['session_wall_seconds'] for c in counters), training_steps=steps,
            training_seconds=train_seconds, pure_training_fps=steps/train_seconds,
            previous_total_wall_seconds=root_end.get('wall_seconds'),
            previous_whole_fps=steps/root_end['wall_seconds'], fps_groups=fps_by_group,
            planned_new_steps=24 * 119 * 20480, new_work_cap=60_000_000,
            planned_matrix='信用三臂＋STIN开关＋单侧，共六条件×四种子；用户最终要求六千万上限',
            training_hours_at_fps={str(fps): (24 * 119 * 20480) / fps / 3600 for fps in (900, 1200, 1300, 1387)})
        integrity = dict(status='ANALYZED', archive_sha256=sha(diag_path), source_training_sha256=sha(training_path),
            source_models_sha256=sha(models_path), model_pair_hash_matches=model_matches, states=len(states),
            state_comparisons=len(state_rows), forecast_states=len(forecast_rows), common_states=sum(s['origin']=='common_ready' for s in states),
            all_reference_clones_passed=True, common_physical_hash_passed=True,
            original_records_unchanged=True, new_training_transitions=0, local_model_loads=0,
            scientific_scope='冻结仿真中的条件干预；不是现实因果普适结论，也不证明PPO理论学不来。')
        diagnostics = dict(chain_count=len(chains), completed_services=sum(c['service'] is not None for c in chains),
            command_to_arrival=summary(c['arrival']-c['command_tick'] for c in chains if c['arrival'] is not None),
            command_to_ready=summary(c['ready']-c['command_tick'] for c in chains if c['ready'] is not None),
            command_to_service=summary(c['service']-c['command_tick'] for c in chains if c['service'] is not None),
            forecast_half_agree=sum(r['agree'] for r in forecast_rows), holds=holds,
            all_branch_samples_at_passenger_events=True,
            air_only_sampling_gap='本轮深分支只在乘客索引采样，遗漏无新乘客但有飞机选择的机会；不能据此宣布飞机侧作用很小。')
        for name, data in (('逐状态复算.json', state_rows), ('逐模型复算.json', model_rows), ('分组结果.json', grouped),
                           ('预测逐状态.json', forecast_rows), ('预测逐模型.json', fore_model), ('预测不确定性.json', fore_stats),
                           ('基准结果.json', baseline_group), ('时序与覆盖.json', dict(groups=timing_group, diagnostics=diagnostics)),
                           ('耗时核算.json', runtime), ('完整性核查.json', integrity), ('READY案例.json', examples)):
            save(out / name, data)
        for name, data in (('逐状态复算.csv', state_rows), ('逐模型复算.csv', model_rows), ('预测逐模型.csv', fore_model),
                           ('决策机会.csv', opportunities), ('飞机事件链.csv', chains)):
            csv_out(out / name, data)
        print(json.dumps(dict(integrity=integrity, forecast=fore_stats, timing=timing_group, baseline=baseline_group, runtime=runtime), ensure_ascii=False, indent=2))
    finally: diag.close(); training.close(); models.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archives', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    analyze(a.archives/'uam_decision_semantics_20261102_v1_DATAONLY.tar.gz',
            a.archives/'uam_credit_horizon_32run_18h_v1_DATAONLY.tar.gz',
            a.archives/'uam_credit_horizon_32run_18h_v1_FINAL_REPLAY_MODELS.tar.gz', a.out)
