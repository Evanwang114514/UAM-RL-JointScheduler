#!/usr/bin/env python3
"""只读取冻结归档；重算 112 块或 CPU 回放外接模型，不训练。"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import random
import statistics as st
import sys
import tarfile

EXT = 'uam_happo_external_k3_28run_56m_v1'
BASE = 'uam_happo_env5_44run_88m_v2'
CROSS = 'uam_cross_followup_20261111_v1'
EXT_CODE = '2a545b7a0eec7e15de31ea0953f98b8072739226ee81106efab6b6f76bfecaf0'
EXT_PACK = 'aefbf4eeb8a63924b6d2ec6fc42a3f35ae293ffddf8720bf73e38daaa5e183aa'
BASE_PACK = 'f5ba14c61a52658708b2205d3e511120351e7d2a65053448c828831e87f73df7'
EXT_DATA = '1fe8a8567fa2867c70331e4eabf45a20a769000927ef7e23e0f575f6f1969194'
HARL_HASH = {
    'harl/algorithms/actors/happo.py': 'fcaa0d7e2de253d37f8f3efd167aebb03ac2cf189984e96d84b4f28a3011b000',
    'harl/models/policy_models/stochastic_policy.py': 'c8beff42a63bcb38b202ad9b000c4a685ef5973ed0244e7fb9245a0d319f84d5',
    'harl/models/base/act.py': 'b47b91aee18bd067b650e14e27cb68789692cb98fbf402f232119aa5bde8f99b'}
WINDOWS = ('30', '60', '120', 'to_cutoff')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'归档或代码不是普通文件：{path}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def member(tar, name):
    x = tar.getmember(name)
    if not x.isfile() or x.size > 25 * 2**20:
        raise ValueError('归档条目类型或大小异常：'+name)
    with tar.extractfile(x) as f:
        return f.read()


def save(path, obj):
    # 只创建派生结果，不修改任何实验原包。
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def label(x):
    for part in ('HAPPO_RANDOM', 'MAPPO_MATCHED', 'PPO_TF'):
        if '__'+part+'__' in x: return part
    return x


def aggregate_model(rows):
    # 同一块中的多个候选不是独立样本；先对块平均，再按训练 seed 列出。
    fields = ('mass_on_compared_pair','conditional_wrong_probability','conditional_expected_extra_person_min')
    out = {}
    for group, data in sorted(rows.items()):
        blocks = {b:{k:st.mean(r[k] for r in vals if r[k] is not None)
                    for k in fields if any(r[k] is not None for r in vals)}
                  for b, vals in data.items()}
        out['/'.join(map(str,group))] = dict(blocks=len(blocks),tuples=sum(map(len,data.values())),
            **{k:st.mean(v[k] for v in blocks.values() if k in v)
               for k in fields if any(k in v for v in blocks.values())})
    return out


def cross(archive):
    results = defaultdict(Counter)
    sensitivity = defaultdict(Counter)
    quality = defaultdict(lambda: defaultdict(list))
    paired_quality = defaultdict(lambda: defaultdict(list))
    blocks = []
    seen = set()
    end = truth = None
    stages = {}
    with tarfile.open(archive, 'r|gz') as tar:
        for item in tar:
            p = PurePosixPath(item.name)
            if p.parts[0] != CROSS or '..' in p.parts or p in seen:
                raise ValueError('归档路径或重名异常')
            seen.add(p)
            if p.name in {'PLAN.json','SESSION_END.json', 'KNOWN_ANSWERS.json','STATUS.json'}:
                if not item.isfile() or item.size > 2**20: raise ValueError('状态条目异常')
                with tar.extractfile(item) as f: rec=json.load(f)
                if p.name=='PLAN.json':
                    stages={x['id']:{'补齐旧块':'old','新顺序复验':'new'}[x['stage']]
                            for x in rec['planned_blocks']}
                elif p.name=='SESSION_END.json': end=rec
                elif p.name=='KNOWN_ANSWERS.json': truth=rec
                else:
                    assert rec['status']=='COMPLETE'
                    blocks.append(rec['block'])
                continue
            if not (len(p.parts)==4 and p.parts[1]=='blocks' and p.name.startswith('CROSS__') and p.name.endswith('.json.gz')):
                continue
            if not item.isfile() or item.size>2**20: raise ValueError('分支条目异常')
            with tar.extractfile(item) as f, gzip.GzipFile(fileobj=f) as zf:
                raw=zf.read(8*2**20+1)
                if len(raw)>8*2**20: raise ValueError('分支解压超限')
                x=json.loads(raw)
            # 原顺序库里也有 order_frozen_*；必须按预注册任务分批。
            stage=stages[p.parts[2]]
            seed=x['seed']; anchor=label(x['anchor'])
            cells=[c for s in x['selections'] if s['stage']=='both_enroute'
                   for c in s['cells'] if c['continuation']=='READY']
            counts=results[stage,seed,anchor]
            counts['pairs']+=1; counts['eligible']+=bool(cells)
            counts['flipped']+=any(c['comparisons'][c['primary_window']]['material_flip'] for c in cells)
            counts['rectangles']+=len(cells)
            counts['flip_rectangles']+=sum(c['comparisons'][c['primary_window']]['material_flip'] for c in cells)
            for c in cells:
                group=sensitivity[stage,seed]
                for w in WINDOWS:
                    if w in c['comparisons']:
                        group[w+'_n']+=1; group[w+'_flip']+=c['comparisons'][w]['material_flip']
                group['short_cutoff']+=c['primary_window']!='120'
                if c['number_of_futures']>1:
                    group['suffix_n']+=1
                    group['suffix_original_flip']+=c['comparisons'][c['primary_window']]['material_flip']
                    group['suffix_average_flip']+=c['average_future_interaction']['material_flip']
                    group['suffix_all_same_flip']+=bool(c['same_material_direction_every_future'])
                if c['known_only'] is not None:
                    group['known_n']+=1; group['known_flip']+=c['known_only']['material_flip']
                flip=c['comparisons'][c['primary_window']]['material_flip']
                for q in c['model_rank_quality']:
                    if q['conditional_expected_extra_person_min'] is not None:
                        quality[stage,seed,anchor,'flip' if flip else 'no_flip',label(q['model'])][p.parts[2]].append(q)
                # 同矩形、同历史下比较，避免模型有效概率覆盖不同。
                for h in (0,1):
                    by={label(q['model']):q for q in c['model_rank_quality'] if q['history']==h}
                    a,b=(by.get(k,{}) for k in ('HAPPO_RANDOM','MAPPO_MATCHED'))
                    k='conditional_expected_extra_person_min'
                    if a.get(k) is not None and b.get(k) is not None:
                        paired_quality[stage,seed,anchor,'flip' if flip else 'no_flip'][p.parts[2]].append(a[k]-b[k])
    assert end['status']=='COMPLETE' and len(blocks)==end['planned_blocks']==112 and set(blocks)==set(stages)
    assert all(v['status']=='KNOWN_ANSWER_UNIT_CHECKS_PASS' for v in truth.values())
    for s in end['report']:
        k=('new' if s['stage']=='新顺序复验' else 'old',s['seed'])
        total=sum((v for g,v in results.items() if g[:2]==k),Counter())
        for a,b in (('pairs','early_action_pairs'),('eligible','primary_eligible_early_pairs'),
                    ('flipped','primary_flipped_early_pairs'),('rectangles','primary_rectangles'),
                    ('flip_rectangles','primary_material_flips')):
            assert total[a]==s[b],(k,a,total[a],s[b])
    key=lambda d:{'/'.join(map(str,k)):v for k,v in sorted(d.items())}
    return dict(archive_sha256=digest(archive),complete_blocks=len(blocks),stages=end['report'],
                anchors=key(results),horizons=key(sensitivity),rank_quality=aggregate_model(quality),
                paired_expected_extra_happo_minus_mappo=key({
                    g:dict(blocks=len(v),mean=st.mean(st.mean(x) for x in v.values()))
                    for g,v in paired_quality.items()}),new_training_transitions=0)


def replay(args,out):
    # 仅反序列化自己已核验的冻结模型，不接收任意第三方 pickle。
    import pickle
    import numpy as np
    import torch
    import gymnasium as gym
    from stable_baselines3.common.vec_env import VecNormalize
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    project=args.project.resolve()
    vendor=args.harl.resolve()
    for f,expected in HARL_HASH.items():
        if digest(vendor/f)!=expected: raise ValueError('回放 HARL 文件与锁定快照不符：'+f)
    archives=project/'experiment_archives'
    archive=archives/'uam_post_happo_replay_v1'/'external_BEST_LAST_MID.tar.gz'
    if digest(archive)!=EXT_PACK: raise ValueError('56M 代表模型包哈希不符')
    if args.external_source:
        source=args.external_source.resolve()
        if digest(source)!=EXT_CODE: raise ValueError('56M 源码字节与封存版本不符')
    else:
        # 读取真实运行的封存入口，避免 Windows checkout 的换行改变源码哈希。
        with tarfile.open(archive,'r:gz') as tf:
            src=member(tf,EXT+'/ENTRYPOINT.py')
        if sha(src)!=EXT_CODE: raise ValueError('封存入口源码哈希不符')
        source=out/'FROZEN_ENTRYPOINT.py'
        with source.open('xb') as f: f.write(src)
    spec=importlib.util.spec_from_file_location('uam_module_frozen_56m',source)
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    base_pack=archives/(BASE+'_DATAONLY.tar.gz')
    data_pack=archives/(EXT+'_DATAONLY.tar.gz')
    if digest(base_pack)!=BASE_PACK or digest(data_pack)!=EXT_DATA:
        raise ValueError('88M 或 56M 数据包哈希不符')
    old,_=mod.module_from_archive(base_pack,project)
    base=old.base_module()
    sys.path.insert(0,str(vendor))
    from harl.algorithms.actors.happo import HAPPO
    def read(t,p): return member(t,EXT+'/'+p)
    results=[]; skipped=[]
    with tarfile.open(archive,'r:gz') as tar:
        sel=json.loads(read(tar,'REPLAY_SELECTION.json'))
        manifest=json.loads(read(tar,'AUX_MANIFEST.json'))
        frozen=read(tar,'FROZEN_AUX.pkl')
        if sha(frozen)!=manifest['sha256']: raise ValueError('模块参数哈希不符')
        with tarfile.open(data_pack,'r:gz') as tf:
            label_data=read(tf,'BRANCH_LABELS.json')
        if sha(label_data)!=manifest['labels_sha256']: raise ValueError('分支标签哈希不符')
        labels=json.loads(label_data)
        support={}
        for key in ('wm','total','path','physical'):
            support[key]={}
            for split in ('fit','audit'):
                values=[x[key] for row in labels['rows'] if row['pattern'] in labels[split+'_patterns']
                        for x in row['rows'] if x[key] is not None and (key=='wm' or x['a']!=row['ref'])]
                support[key][split]=dict(n=len(values),distinct=sorted(set(values)))
        aux=pickle.loads(frozen)
        task0=next(r for r in sel['rows'] if r['status']=='COMPLETE')
        item0=json.loads(read(tar,'cells/'+task0['cell']+'/TASK.json'))
        m,_=base.runtime(out,item0); Phase=mod.augmented_phase(base,m,aux); stations,rates=mod.contexts(base,m)
        bank=json.loads(read(tar,'BANK.json'))['VARYING']['bank']
        chosen=next(x for x in bank if x['name']==args.trace)
        trace=out/(args.trace+'.csv')
        data=read(tar,'bank/VARYING/'+trace.name)
        if sha(data)!=chosen['sha256']: raise ValueError('确认轨迹哈希不符')
        with trace.open('xb') as f: f.write(data)
        for r in sel['rows']:
            cell=r['cell']; folder='cells/'+cell
            if r['status']!='COMPLETE':
                skipped.append(dict(cell=cell,reason=r['status'])); continue
            item=json.loads(read(tar,folder+'/TASK.json'))
            if args.arm and item['arm'] not in args.arm: continue
            conf=json.loads(read(tar,folder+'/CONFIG.json'))
            point=next(x for x in r['selections'] if args.role in x['roles'])
            meta=json.loads(read(tar,point['pointer']))
            assert conf['aux_sha256']==manifest['sha256'] and point['step']==meta['real_steps']
            state,norm=(read(tar,folder+'/'+meta[k]) for k in ('model','norm'))
            if sha(state)!=meta['model_sha'] or sha(norm)!=meta['norm_sha']: raise ValueError('代表模型或归一化哈希不符')
            norm=pickle.loads(norm); assert isinstance(norm,VecNormalize); norm.training=False
            state=pickle.loads(state); cfg=conf['args']; widths=conf['actor_widths']
            space=lambda w:gym.spaces.Box(-np.inf,np.inf,(w,),np.float32)
            actors=[HAPPO(cfg,space(w),gym.spaces.Discrete(3),torch.device('cpu')) for w in widths]
            if len(actors)!=len(state['actors']): raise ValueError('actor 数不一致')
            for a,s in zip(actors,state['actors']): a.actor.load_state_dict(s); a.prep_rollout()
            arm=item['arm']
            parent={'H1_INFO':None,'H2_S':'H1_INFO'}.get(arm,'H2_S')
            def probs(x,role):
                normed=norm.normalize_obs(np.asarray(x)[None])
                if role==0: normed=normed[:,:widths[0]]
                act=actors[role].actor
                with torch.no_grad():
                    t=torch.as_tensor(normed,**act.tpdv)
                    return act.act.get_logits(act.base(t)).softmax(-1).cpu().numpy()[0]
            def pick(q,u): return min(2,int(np.searchsorted(np.cumsum(q),u,side='right')))
            def policy(x,role,u): return pick(probs(x,role),u)
            make=lambda:Phase(out,item,str(trace),600,123,False)
            rng=random.Random(901); tape=[(rng.random(),rng.random()) for _ in range(601)]
            env=make(); opportunities=[]; changed=0
            try:
                obs,_=mod.replay_reset(env)
                for t in range(240):
                    p=policy(obs,0,tape[t][0]); post,_,_,_,_=env.step(p)
                    feature=mod.packet(env,stations,rates); legal=feature[3]
                    q=probs(post,1); a=pick(q,tape[t][1])
                    if feature[4] and a in legal and t>=36:
                        ref=base.lq_target(env.env); ref=ref if ref in legal else legal[0]
                        new=(np.zeros(mod.WIDTH,np.float32) if parent is None else
                             mod.module_features(parent,feature,aux,ref))
                        replacement=post.copy(); replacement[-mod.WIDTH:]=new
                        swapped=probs(replacement,1)
                        tv=float(np.abs(q-swapped).sum()/2)
                        wm_constant=None; wm_variation=None
                        if arm=='H4_S_WM':
                            # 保留同样的候选掩码；只把预测换成训练池常数基线。
                            y=np.asarray(support['wm']['fit']['distinct'],dtype=float)
                            if len(y)==1:
                                flat=post[-mod.WIDTH:].reshape(3,15)
                                wm_variation=float(np.max(np.abs(flat[list(legal),12]*20-y[0])))
                                z=post.copy(); z[-mod.WIDTH:].reshape(3,15)[list(legal),12]=y[0]/20
                                wm_constant=float(np.abs(q-probs(z,1)).sum()/2)
                        changed+=bool(np.max(np.abs(post[-mod.WIDTH:]-new))>1e-8)
                        opportunities.append(dict(t=t,a=a,legal=list(legal),tv=tv,
                            feature_linf=float(np.max(np.abs(post[-mod.WIDTH:]-new))),
                            commitment_signal_diff=bool(np.any(feature[0][:,17:19]!=feature[1][:,17:19])),
                            wm_constant_tv=wm_constant,wm_max_residual_deviation_minutes=wm_variation,
                            greedy_changed=int(np.argmax(q))!=int(np.argmax(swapped)),
                            same_draw_changed=a!=pick(swapped,tape[t][1]),q=q.tolist(),replacement_p=swapped.tolist()))
                    obs,_,_,_,_=env.step(a)
            finally: env.close()
            # 按时间分层抽样，不按 TV 或是否获益挑状态。保留别名、拒绝不等飞机批次。
            branches=[]; mod.CF_H=args.horizon
            slots=sorted(set(round((len(opportunities)-1)*(j+.5)/args.states) for j in range(args.states))) if opportunities and args.states else []
            for ix in slots:
                x=opportunities[ix]; t=x['t']; a=x['a']; candidates={}; natural,_=mod.branch(
                    make,old,tape,stations,rates,t,None,policy=policy)
                for k in x['legal']:
                    z,_=mod.branch(make,old,tape,stations,rates,t,k,policy=policy)
                    if z is not None:
                        candidates[k]=(z['cost'],z['first']['sig'])
                if (not natural or natural['first']['a']!=a or a not in candidates
                    or candidates[a]!=(natural['cost'],natural['first']['sig'])):
                    raise RuntimeError('事实动作或首个实际执行不可复演')
                sig=candidates[a][1]; batch=sorted(x[0] for x in sig)
                do={k:c for k,(c,s) in candidates.items() if batch and sorted(x[0] for x in s)==batch}
                if len({candidates[k][1] for k in do})<2:
                    branches.append(dict(t=t,status='NO_SAME_BATCH_ALTERNATIVE')); continue
                best=min(do.values())
                def regret(probs):
                    mass=sum(probs[k] for k in do)
                    return [mass,sum(probs[k]*(c-best) for k,c in do.items())/mass if mass>1e-12 else None]
                q,tq=regret(x['q']),regret(x['replacement_p'])
                branches.append(dict(t=t,status='COMPLETE',costs=do,conditional_expected_regret=q[1],
                    replacement_conditional_expected_regret=tq[1],pair_mass=q[0],replacement_mass=tq[0]))
            result=dict(cell=cell,arm=arm,seed=item['seed'],step=point['step'],trace=args.trace,
                        replacement=parent or 'ZERO_OOD_SENSITIVITY_ONLY',horizon=args.horizon,
                        eligible=len(opportunities),feature_nonzero_difference=changed,states=opportunities,branches=branches)
            save(out/(cell+'.json'),result)
            results.append({k:v for k,v in result.items() if k!='states'} | dict(
                mean_tv=st.mean(x['tv'] for x in opportunities) if opportunities else None,
                same_draw_changes=sum(x['same_draw_changed'] for x in opportunities),
                greedy_changes=sum(x['greedy_changed'] for x in opportunities),
                query_signal_changes=sum(x['commitment_signal_diff'] for x in opportunities),
                wm_constant_mean_tv=st.mean(y) if (y:=[x['wm_constant_tv'] for x in opportunities if x['wm_constant_tv'] is not None]) else None,
                wm_max_residual_deviation_minutes=max(y) if (y:=[x['wm_max_residual_deviation_minutes'] for x in opportunities if x['wm_max_residual_deviation_minutes'] is not None]) else None))
            print(cell,'覆盖',len(opportunities),'特征不同',changed,flush=True)
    assert not torch.cuda.is_initialized()
    return dict(archive_sha256=EXT_PACK,label_data_sha256=manifest['labels_sha256'],
                label_support=support,inverse_count=manifest['inverse_count'],results=results,skipped=skipped,
                scope='同权重、不同通道输入的局部敏感性；不可代替独立训练消融或在线全局收益',new_training_transitions=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('cross','replay'))
    p.add_argument('--project',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--harl',type=Path); p.add_argument('--external-source',type=Path)
    p.add_argument('--arm',action='append'); p.add_argument('--role',choices=('LAST','BEST','MID'),default='LAST')
    p.add_argument('--trace',choices=tuple('order_830'+str(i) for i in range(1,5)),default='order_8301')
    p.add_argument('--states',type=int,default=2); p.add_argument('--horizon',type=int,choices=(30,60,120),default=120)
    args=p.parse_args()
    if not 0<=args.states<=12: p.error('--states 必须为 0–12')
    if args.mode=='replay' and args.harl is None:
        p.error('replay 需要 --harl 指向锁定的作者源码目录')
    if args.out.exists() or args.out.is_symlink(): p.error('输出目录必须新建')
    args.out.mkdir(parents=True,exist_ok=False)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
    if args.mode=='cross':
        a=args.project/'experiment_archives'/(CROSS+'_DATAONLY.tar.gz')
        result=cross(a)
    else:
        result=replay(args,args.out.resolve())
    save(args.out/'AUDIT.json',result)


if __name__=='__main__':
    main()
