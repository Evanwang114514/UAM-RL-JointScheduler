#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只读汇总诊断与18小时训练归档；不加载模型、不调用训练或仿真。"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, itertools, json, math, statistics, sys, tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
import numpy as np
import scipy
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Archive:
    """归档只作为数据读取，拒绝路径逃逸、链接和重复成员。"""
    def __init__(self,path):
        self.path=Path(path);self.data={};self.logs={};self.members=[]
        self.sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with tarfile.open(self.path,'r:gz') as tf:
            for m in tf:
                p=PurePosixPath(m.name)
                if p.is_absolute() or '..' in p.parts or m.issym() or m.islnk():raise ValueError(m.name)
                if not m.isfile():continue
                if m.name in self.members:raise ValueError('重复成员：'+m.name)
                self.members.append(m.name)
                if m.size>40*1024**2:raise ValueError('异常大记录：'+m.name)
                if m.name.endswith('.json'):
                    self.data[m.name]=json.load(tf.extractfile(m))
                elif m.name.endswith('train.log'):
                    self.logs[m.name]=tf.extractfile(m).read().decode('utf-8',errors='replace')
        self.root=self.members[0].split('/')[0]
    def get(self,path):return self.data[self.root+'/'+path]


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2),encoding='utf-8')


def describe(values):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)];n=len(x)
    if not n:return dict(n=0)
    sd=float(np.std(x,ddof=1)) if n>1 else None
    ci=float(stats.t.ppf(.975,n-1))*sd/math.sqrt(n) if n>1 else None
    return dict(n=n,mean=float(x.mean()),sd=sd,median=float(np.median(x)),minimum=float(x.min()),maximum=float(x.max()),
                ci95_t=[float(x.mean()-ci),float(x.mean()+ci)] if ci is not None else None,
                values=x.tolist())


def contrast(label,diffs):
    d=np.array(diffs,dtype=float);point=float(d.mean())
    # 四个训练种子只支持很粗的精确配对检验；不把回放次数当样本数。
    distribution=[abs(float(np.mean(d*np.array(s)))) for s in itertools.product((-1,1),repeat=len(d))]
    p=sum(x>=abs(point)-1e-12 for x in distribution)/len(distribution)
    loo=[float(np.delete(d,i).mean()) for i in range(len(d))]
    return dict(label=label,**describe(d),wins_lower=int(sum(d<0)),ties=int(sum(d==0)),
                exact_two_sided_signflip_p=p,leave_one_seed_out_means=loo,
                signed_dz=point/float(np.std(d,ddof=1)) if np.std(d,ddof=1)>0 else None)


def training(a,out):
    plan=a.get('SUITE_PLAN.json');tasks=plan['tasks'];rows=[];curves=[];episodes=[];initial=defaultdict(set)
    for t in tasks:
        prefix=a.root+'/cells/'+t['id']+'/'
        r=a.data[prefix+'RESULT.json'];end=a.data[prefix+'TRAIN_END.json'];m=a.data[prefix+'MODEL_MANIFEST.json']
        assert end['steps']==plan['config']['actual_steps'] and r['task']==t
        assert m['code_sha256']==plan['script_sha256'] and m['experimental_gamma']==t['gamma']
        assert m['reward_normalization_gamma']==.99 and m['ppo']['gamma']==t['gamma']
        initial[(t['seed'],t['control'],t['network'])].add(a.data[prefix+'INITIAL_POLICY_SHA.json']['sha256'])
        late=r['main_late_confirmation'];final=r['final_confirmation'];s=r['original_curve_stability']
        row=dict(**t,late_rpt=late['rpt'],late_completion=late['completion'],late_full_fraction=late['full_episode_fraction'],
            final_rpt=final['rpt'],final_completion=final['completion'],final_full_fraction=final['full_episode_fraction'],
            best_original=s['best_rpt'],late_original=s['late_rpt'],late_sd_original=s['late_sd'],
            late_slope_original=s['late_slope_per_million'],rollback_original=s['best_to_late_ratio'],
            train_seconds=end['wall_seconds'],eval_seconds=r['eval_seconds'],steps=end['steps'],parameters=m['parameters'])
        rows.append(row)
        for name,v in a.data.items():
            if not name.startswith(prefix+'evaluation/') or 'episodes' not in v:continue
            for e in v['episodes']:
                item=dict(**t,record=name.split(prefix)[1],**e)
                episodes.append(item)
            original=[e for e in v['episodes'] if e['trace']=='original' and e['eval_seed']==123]
            if original:curves.append(dict(**t,step=v.get('step',int(Path(name).stem.split('_')[-1])),rpt=original[0]['restricted_person_time_per_generated'],completion=original[0]['completion']))
    assert len(rows)==32 and all(len(v)==1 for v in initial.values())
    groups=[]
    for control,net,gamma in itertools.product(('J3','LQ'),('LSTM','TRANSFORMER'),(.99,1.)):
        items=sorted([r for r in rows if (r['control'],r['network'],r['gamma'])==(control,net,gamma)],key=lambda r:r['seed'])
        groups.append(dict(control=control,network=net,gamma=gamma,rpt=describe([r['late_rpt'] for r in items]),
            final=describe([r['final_rpt'] for r in items]),completion=describe([r['late_completion'] for r in items]),
            full_fraction=describe([r['late_full_fraction'] for r in items]),rollback=describe([r['rollback_original'] for r in items]),
            slope=describe([r['late_slope_original'] for r in items]),models=items))
    by={(r['control'],r['network'],r['gamma'],r['seed']):r for r in rows};contrasts=[]
    for control,net in itertools.product(('J3','LQ'),('LSTM','TRANSFORMER')):
        contrasts.append(contrast(f'{control}/{net}:gamma1-gamma099',[by[control,net,1.,s]['late_rpt']-by[control,net,.99,s]['late_rpt'] for s in range(1801,1805)]))
    for control,gamma in itertools.product(('J3','LQ'),(.99,1.)):
        contrasts.append(contrast(f'{control}/gamma{gamma}:Transformer-LSTM',[by[control,'TRANSFORMER',gamma,s]['late_rpt']-by[control,'LSTM',gamma,s]['late_rpt'] for s in range(1801,1805)]))
    for net,gamma in itertools.product(('LSTM','TRANSFORMER'),(.99,1.)):
        contrasts.append(contrast(f'{net}/gamma{gamma}:Joint-Single',[by['J3',net,gamma,s]['late_rpt']-by['LQ',net,gamma,s]['late_rpt'] for s in range(1801,1805)]))
    ordered=sorted(range(len(contrasts)),key=lambda i:contrasts[i]['exact_two_sided_signflip_p']);last=0.
    for rank,i in enumerate(ordered):
        last=max(last,min(1.,(len(contrasts)-rank)*contrasts[i]['exact_two_sided_signflip_p']));contrasts[i]['holm_p_family12']=last
    # 原顺序完整曲线与确认集分别保存，不能把两种银行拼成一条曲线。
    main=[];last=[]
    late_start=.8*plan['config']['actual_steps']
    for e in episodes:
        if e['trace']=='original':continue
        step=int(Path(e['record']).stem.split('_')[-1])
        if step>=late_start:main.append(e)
        if step==plan['config']['actual_steps']:last.append(e)
    reconstruction=[]
    for row in rows:
        es=[e for e in main if e['id']==row['id']]
        assert len(es)==40
        delta=float(np.mean([e['restricted_person_time_per_generated'] for e in es]))-row['late_rpt']
        assert abs(delta)<1e-9
        reconstruction.append(dict(id=row['id'],episodes=len(es),reconstruction_difference=delta))
    failures=[]
    for e in main:
        if e['full']:continue
        tr=e.get('terminal_resource',{});qs=e.get('terminal_queues',tr.get('queues',{}));unfinished=e.get('unfinished_states',[])
        tail=bool(qs) and sum(qs.values())==e['generated']-e['finished'] and max(qs.values())<4 and set(tr.get('aircraft_states',{}))=={'IDLE'}
        failures.append(dict(id=e['id'],seed=e['seed'],control=e['control'],network=e['network'],gamma=e['gamma'],
                            tail=tail,blocked=tr.get('restricted_executor_blocked',False),missing=e['generated']-e['finished']))
    timing=dict(train_seconds=sum(r['train_seconds'] for r in rows),eval_seconds=sum(r['eval_seconds'] for r in rows),
                suite_wall_seconds=a.get('END.json')['wall_seconds'])
    timing['pure_training_fps']=sum(r['steps'] for r in rows)/timing['train_seconds']
    timing['suite_fps']=sum(r['steps'] for r in rows)/timing['suite_wall_seconds']
    components=[]
    for control,net,gamma in itertools.product(('J3','LQ'),('LSTM','TRANSFORMER'),(.99,1.)):
        ss=[e for e in main if (e['control'],e['network'],e['gamma'])==(control,net,gamma)]
        if not all('restricted_components_per_generated' in e for e in ss):
            # 新训练归档没有同口径的状态占用分解，不拿已完成乘客的均值冒充全体分解。
            components.append(dict(control=control,network=net,gamma=gamma,available=False,missing_records=sum('restricted_components_per_generated' not in e for e in ss)))
            continue
        per_seed=[]
        for seed in range(1801,1805):
            es=[e for e in ss if e['seed']==seed]
            per_seed.append(dict(seed=seed,**{k:float(np.mean([e['restricted_components_per_generated'][k] for e in es])) for k in ('ground','wait','flight','other')}))
        components.append(dict(control=control,network=net,gamma=gamma,**{k:describe([r[k] for r in per_seed]) for k in ('ground','wait','flight','other')}))
    result=dict(plan=plan,progress=a.get('progress.json'),groups=groups,contrasts=contrasts,timing=timing,
                late_episodes=len(main),late_full=sum(e['full'] for e in main),final_episodes=len(last),final_full=sum(e['full'] for e in last),
                incomplete_categories=Counter('tail' if e['tail'] else 'blocked' if e['blocked'] else 'other' for e in failures),
                initial_pairs_identical=all(len(v)==1 for v in initial.values()),reconstruction=reconstruction,components=components)
    save(out/'training_analysis.json',result);save(out/'training_models.json',rows);save(out/'training_curves.json',curves)
    save(out/'training_late_episode_records.json',main);save(out/'training_failure_classes.json',failures)
    for g in groups:print('TRAIN_GROUP',g['control'],g['network'],g['gamma'],'RPT',round(g['rpt']['mean'],4),'SD',round(g['rpt']['sd'],4),'FULL',round(g['full_fraction']['mean'],4),'VALUES',g['rpt']['values'])
    for c in contrasts:print('CONTRAST',c['label'],round(c['mean'],4),c['wins_lower'],c['exact_two_sided_signflip_p'])
    print('TRAIN_COUNTS',json.dumps(clean({k:result[k] for k in ('late_episodes','late_full','final_episodes','final_full','incomplete_categories','timing')})))
    return result,rows,curves


def metric_fields(m,prefix):
    r={prefix+'_rpt':m['restricted_person_time_per_generated'],prefix+'_full':m['full'],prefix+'_finished':m['finished']}
    r.update({prefix+'_'+k:v for k,v in m.get('restricted_components_per_generated',{}).items()})
    return r


def diagnosis(a,out):
    manifest=a.get('MANIFEST.json');cases={c['case']:c for c in manifest['cases']};rows=[];branches=[];audit=[];counters=[]
    for name,o in a.data.items():
        if name.endswith('/COUNTERS.json'):counters.append(o)
        if '/cases/' in name and name.endswith('/BASE.json'):
            case,trace=name.split('/cases/')[1].split('/')[:2];c=cases[case];prefix=name[:-len('BASE.json')]
            tail=a.data[prefix+'TAIL_BASE.json'];swap=a.data[prefix+'SWAPS.json']
            row=dict(case=case,control='READY' if c['controller']=='READY' else c['task']['control'],
                arm='READY' if c['controller']=='READY' else c['task']['arm'],seed=c['training_seed'],trace=trace,
                **metric_fields(o['metrics'],'base'),**metric_fields(tail['best']['metrics'],'tail'),**metric_fields(swap['best']['metrics'],'swap'),
                tail_changed=tail['best']['changed'],swap_pairs=len(swap['trials']))
            row.update(tail_gain=row['base_rpt']-row['tail_rpt'],swap_gain=row['tail_rpt']-row['swap_rpt'])
            rows.append(row)
            assert swap['count_preserved'] and swap['complete']
        if '/C_BRANCHES/state_' not in name:continue
        case,trace=name.split('/cases/')[1].split('/')[:2];c=cases[case]
        assert o['all_references_passed']
        for follow in sorted({v['continuation'] for v in o['rows']}):
            candidates=[v for v in o['rows'] if v['continuation']==follow];actions=[v['action'] for v in candidates]
            if o['original_reference'] not in actions:
                raise ValueError(json.dumps(dict(case=case,trace=trace,follow=follow,reference=o['original_reference'],actions=actions),ensure_ascii=True))
            ref=actions.index(o['original_reference'])
            for h in (60,120,240):
                costs=[v['snapshots'][str(h)] for v in candidates];known=[v['committed_snapshots'][str(h)] for v in candidates]
                def best(field,data=costs):
                    vals=[v[field] for v in data];low=min(vals)
                    return {i for i,x in enumerate(vals) if abs(x-low)<1e-8}
                def choose(indices):return min(indices,key=lambda i:(i!=ref,i))
                winner=choose(best('total'));pick=choose(best('total',known));rr=costs[ref];ww=costs[winner]
                key=dict(case=case,control='READY' if c['controller']=='READY' else c['task']['control'],
                    arm='READY' if c['controller']=='READY' else c['task']['arm'],seed=c['training_seed'],trace=trace,
                    index=o['index'],time=o['time'],period='early' if o['index']<100 else 'middle' if o['index']<200 else 'late',continuation=follow,horizon=h)
                ponly={i for i,act in enumerate(actions) if act[1]==actions[ref][1]};aonly={i for i,act in enumerate(actions) if act[0]==actions[ref][0]}
                pbest=min(costs[i]['total'] for i in ponly);abest=min(costs[i]['total'] for i in aonly)
                rec=dict(**key,candidates=len(costs),local_gain=rr['total']-ww['total'],focal_gain=rr['focal']-ww['focal'],
                    existing_gain=rr['existing']-ww['existing'],future_gain=rr['future']-ww['future'],
                    passenger_only_gain=rr['total']-pbest,aircraft_only_gain=rr['total']-abest,
                    joint_extra_gain=min(pbest,abest)-ww['total'],
                    discount_disjoint=not bool(best('discounted_099')&best('total')),
                    post_discount_disjoint=not bool(best('reward_post_g099')&best('reward_post_g100')),
                    pre_post_disjoint=not bool(best('total')&best('reward_post_g100')),
                    committed_disjoint=not bool(best('total',known)&best('total')),
                    committed_realized_gain=rr['total']-costs[pick]['total'],
                    committed_pick_regret=costs[pick]['total']-ww['total'],
                    discounted_pick_regret=costs[choose(best('reward_post_g099'))]['reward_post_g100']-min(v['reward_post_g100'] for v in costs),
                    after_cutoff_max=max(v['after_train_cutoff500'] for v in costs),
                    after_cutoff_gain=rr['after_train_cutoff500']-ww['after_train_cutoff500'])
                for h2 in (60,120,240):
                    if h2>=h:continue
                    other=[v['snapshots'][str(h2)] for v in candidates]
                    rec[f'horizon{h2}_disjoint']=not bool(best('total',other)&best('total'))
                branches.append(rec)
        audit.append(dict(case=case,trace=trace,index=o['index'],references_passed=o['all_references_passed']))
    groups=[]
    for control,arm in sorted({(r['control'],r['arm']) for r in rows}):
        subset=[r for r in rows if (r['control'],r['arm'])==(control,arm)];models=[]
        for seed in sorted({r['seed'] for r in subset},key=lambda v:-1 if v is None else v):
            ss=[r for r in subset if r['seed']==seed]
            models.append(dict(seed=seed,**{k:float(np.mean([r[k] for r in ss])) for k in ('base_rpt','tail_rpt','swap_rpt','tail_gain','swap_gain','base_full','tail_full','swap_full')}))
        groups.append(dict(control=control,arm=arm,models=models,**{k:describe([r[k] for r in models]) for k in ('base_rpt','tail_rpt','swap_rpt','tail_gain','swap_gain','base_full','tail_full','swap_full')}))
    rank_groups=[]
    fields=('local_gain','focal_gain','existing_gain','future_gain','passenger_only_gain','aircraft_only_gain','joint_extra_gain',
        'discount_disjoint','post_discount_disjoint','pre_post_disjoint','committed_disjoint','committed_realized_gain','committed_pick_regret',
        'discounted_pick_regret','after_cutoff_max','after_cutoff_gain','horizon60_disjoint','horizon120_disjoint')
    for control,arm,follow,h in sorted({(r['control'],r['arm'],r['continuation'],r['horizon']) for r in branches}):
        subset=[r for r in branches if (r['control'],r['arm'],r['continuation'],r['horizon'])==(control,arm,follow,h)]
        models=[]
        for seed in sorted({r['seed'] for r in subset},key=lambda v:-1 if v is None else v):
            ss=[r for r in subset if r['seed']==seed]
            models.append(dict(seed=seed,states=len(ss),**{k:float(np.mean([r[k] for r in ss if k in r])) for k in fields if any(k in r for r in ss)}))
        rank_groups.append(dict(control=control,arm=arm,continuation=follow,horizon=h,states=len(subset),models=models,
            **{k:describe([r[k] for r in models if k in r]) for k in fields if any(k in r for r in models)}))
    counts=dict(cases=len(cases),base_episodes=len(rows),states=len(audit),branch_windows=len(branches),
        full_episodes=sum(r.get('full_episodes',0) for r in counters),candidate_calls=sum(r.get('candidate_calls',0) for r in counters),
        simulated_minutes=sum(r.get('completed_simulated_minutes',0) for r in counters),
        worker_seconds=sum(r.get('worker_elapsed_seconds',0) for r in counters),wall_seconds=a.get('SESSION_END.json')['wall_seconds'],
        retained_old_failure_records=sum(n.endswith('/FAILED.json') for n in a.data))
    counts['summed_subprocess_wall_seconds']=sum(r.get('elapsed_seconds',0) for r in a.get('SESSION_END.json')['results'])
    counts['branch_candidate_rows_before_window_expansion']=sum(len(v['rows']) for k,v in a.data.items() if '/C_BRANCHES/state_' in k)
    validation_groups=[]
    for control,arm in sorted({(r['control'],r['arm']) for r in rows}):
        subset=[r for r in rows if (r['control'],r['arm'])==(control,arm) and r['trace']!='original'];models=[]
        for seed in sorted({r['seed'] for r in subset},key=lambda v:-1 if v is None else v):
            ss=[r for r in subset if r['seed']==seed]
            models.append(dict(seed=seed,**{k:float(np.mean([r[k] for r in ss])) for k in ('base_rpt','tail_rpt','swap_rpt','tail_gain','swap_gain','base_full','tail_full','swap_full')}))
        validation_groups.append(dict(control=control,arm=arm,models=models,**{k:describe([r[k] for r in models]) for k in ('base_rpt','tail_rpt','swap_rpt','tail_gain','swap_gain','base_full','tail_full','swap_full')}))
    result=dict(counts=counts,progress=a.get('progress.json'),session=a.get('SESSION_END.json'),groups=groups,validation_groups=validation_groups,rank_groups=rank_groups,
                all_reference_checks_passed=all(r['references_passed'] for r in audit),manifest=manifest)
    save(out/'diagnosis_analysis.json',result);save(out/'diagnosis_episodes.json',rows);save(out/'diagnosis_branch_windows.json',branches)
    for g in groups:print('DIAG_GROUP',g['control'],g['arm'],'base/tail/swap',*[round(g[k]['mean'],4) for k in ('base_rpt','tail_rpt','swap_rpt')],'SWAP',g['swap_gain']['values'])
    for g in rank_groups:
        if g['horizon']==240 and g['arm'] in ('L00','L11_BOTH','T11_BOTH','READY'):
            print('RANK',g['control'],g['arm'],g['continuation'],'gain',round(g['local_gain']['mean'],2),'discount',round(g['post_discount_disjoint']['mean'],3),'known',round(g['committed_disjoint']['mean'],3),'future',round(g['future_gain']['mean'],2),'after500',round(g['after_cutoff_max']['mean'],2))
    print('DIAG_COUNTS',json.dumps(counts))
    return result,rows,branches


def plots(out,groups,curves,diag):
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharey=True,layout='constrained')
    colors=['#0072B2','#D55E00','#0072B2','#D55E00'];marks=['o','s','o','s']
    for ax,control in zip(axs,('J3','LQ')):
        gg=[g for g in groups if g['control']==control]
        for i,g in enumerate(gg):
            vals=g['rpt']['values'];ax.scatter(np.full(4,i)+np.linspace(-.09,.09,4),vals,color=colors[i],marker=marks[i],s=38)
            ax.plot([i-.2,i+.2],[np.mean(vals)]*2,color='black',lw=2)
        ax.set_xticks(range(4),[('LSTM' if g['network']=='LSTM' else 'Transformer')+'\nγ='+str(g['gamma']) for g in gg])
        ax.set_title('联合控制' if control=='J3' else '单侧乘客控制＋飞机规则');ax.set_ylabel('固定601分钟边界的人时成本（分钟/人，低更好）');ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'training_seed_points.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(14,7),sharex=True,sharey=True,layout='constrained')
    for ax,g in zip(axs.ravel(),groups):
        for s in range(1801,1805):
            data=sorted([r for r in curves if (r['control'],r['network'],r['gamma'],r['seed'])==(g['control'],g['network'],g['gamma'],s)],key=lambda r:r['step'])
            ax.plot([r['step']/1e6 for r in data],[r['rpt'] for r in data],lw=1,label=str(s))
        ax.set_title(f"{g['control']} {g['network']} γ={g['gamma']}");ax.grid(alpha=.15)
    axs[0,0].legend(title='训练种子',fontsize=8);fig.supxlabel('训练步数（百万）；全部为原顺序回放');fig.supylabel('固定边界人时成本（分钟/人）')
    fig.savefig(out/'training_all_curves.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    subset=[g for g in diag if g['arm'] in ('L00','L11_BOTH','T11_BOTH','READY')]
    for i,g in enumerate(subset):
        vals=[g[k]['mean'] for k in ('base_rpt','tail_rpt','swap_rpt')]
        ax.plot(range(3),vals,marker='o',label=g['control']+' '+g['arm'])
    ax.set_xticks(range(3),['原模型原决策','事后统一处理尾批','再做保总人数的顺序交换'])
    ax.set_ylabel('固定边界人时成本（分钟/人，低更好）');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2)
    fig.savefig(out/'diagnosis_feasible_improvement.png',dpi=170);plt.close(fig)


def supplement(train,diag,out,t):
    old=Archive(train.path.parent/'uam_gap_factors_48run_50m_v1_DATAONLY.tar.gz')
    old_plan=old.get('SUITE_PLAN.json');new_plan=train.get('SUITE_PLAN.json')
    assert old_plan['script_sha256']==new_plan['frozen_50m_sha256']
    old_bank={x['name']:x['sha256'] for x in old_plan['config']['bank']}
    new_bank={x['name']:x['sha256'] for x in new_plan['config']['bank']}
    assert old_bank==new_bank
    matches=0
    for name,v in diag.data.items():
        if '/cases/' not in name or not name.endswith('/BASE.json'):continue
        case,trace=name.split('/cases/')[1].split('/')[:2]
        if case=='READY_K3':continue
        original=old.get('cells/'+case+'/evaluation/step_001003520.json')
        es=[e for e in original['episodes'] if e['trace']==trace and e['eval_seed']==123]
        assert len(es)==1 and abs(es[0]['restricted_person_time_per_generated']-v['metrics']['restricted_person_time_per_generated'])<1e-7
        matches+=1
    assert matches==120
    save(out/'SOURCE_VALIDATION.json',dict(training_models=32,training_steps_and_hashes_passed=True,late_metric_reconstructions=32,
        final_replays_match_original_records=matches,bank_sha_match=True,all_branch_references_passed=True,
        assertion_scope='重新汇总与源记录核对；不是重新运行模型认证'))
    rule_data=old.get('cells/RULES__E1__K3/RULES_END.json')['results']
    rules=[]
    for name in sorted({r['rule'] for r in rule_data}):
        es=[r for r in rule_data if r['rule']==name and r['trace']!='original']
        rules.append(dict(rule=name,rpt=float(np.mean([r['metrics']['restricted_person_time_per_generated'] for r in es])),
                          full_fraction=float(np.mean([r['metrics']['full'] for r in es])),
                          by_trace={r['trace']:r['metrics']['restricted_person_time_per_generated'] for r in es}))
    save(out/'matched_rule_baselines.json',dict(source=str(old.path),sha256=old.sha,bank_hashes_match=True,baseline_rows=rules))
    sys.path.insert(0,'C:/Users/w/.agents/skills/statistical-analysis/scripts')
    from assumption_checks import check_normality
    checks=[dict(contrast=c['label'],**check_normality(c['values'],name=c['label'],plot=False)) for c in t['contrasts']]
    save(out/'statistical_assumptions.json',dict(checks=checks,note='每项只有4个差值，正态检验无能力证实正态；不据此挑检验、删种子。t区间仅参考，精确符号翻转也依赖可交换/对称假设。'))
    path=train.path.parent/'uam_credit_horizon_32run_18h_v1_FINAL_REPLAY_MODELS.tar.gz'
    with tarfile.open(path,'r:gz') as tf:
        members=[m.name for m in tf if m.isfile()]
    zips={n[:-4] for n in members if n.endswith('.zip')};pkls={n[:-4] for n in members if n.endswith('.pkl')}
    assert len(zips)==len(pkls)==32 and zips==pkls
    save(out/'model_package_inventory.json',dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        models=32,normalizations=32,pairing_passed=True,model_loading_performed=False))
    print('RULES',json.dumps(rules));print('COMPONENTS',json.dumps(t['components']))
    return rules


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    base=Path('E:/Study Files/github/UAM-predict/UAGMC-main/experiment_archives')
    train=Archive(base/'uam_credit_horizon_32run_18h_v1_DATAONLY.tar.gz');diag=Archive(base/'uam_50m_credit_diag_1h_20261102_v1_IMPORT_FIXED_DATAONLY.tar.gz')
    t,rows,curves=training(train,out);d,episodes,branches=diagnosis(diag,out);plots(out,t['groups'],curves,d['groups'])
    save(out/'ANALYSIS_MANIFEST.json',dict(date='2026-11-02',status='ANALYZED',python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
        inputs=[dict(path=str(a.path),sha256=a.sha,members=len(a.members)) for a in (train,diag)],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_model_load=True,new_training_transitions=0,analysis_unit='每臂4个独立训练种子；回放先在模型内汇总；对比均为探索性',
        primary_contrasts='四项gamma配对；其余网络和控制对比为次要；保守地对12项共同Holm校正',
        preflight=dict(cpu_percent=5.3,free_ram_gib=19.58,disk_free_gib=22.46,gpu_need='none',gpu_allocations=0,threads=1)))
    supplement(train,diag,out,t)


if __name__=='__main__':main()
