"""大规模冻结策略诊断：只回放、分支及解析控制，不更新任何模型参数。"""
from __future__ import annotations
import argparse
import copy
import csv
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
from types import MethodType
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

_script = Path(__file__).resolve()
WS = _script.parents[2] if _script.parent.parent.name == 'projects' else _script.parent / 'workspace'
STRICT = 'uam_strictE1_LQ_fleet40m_methods60m_v1'
PT = 'uam_E1_reward_charge_28run_30m_v1'
STRICT_ENTRY = 'train_uam_strictE1_LQ_fleet40m_methods60m_ONEFILE.py'
PT_ENTRY = 'train_uam_E1_reward_charge_attribution_30m_ONEFILE.py'
EXPECTED_STRICT = 'f94383c22ce38454fc142992f640498e3c948dbea531b5d0f8cadabc1c1968b0'
STEP = 1003520
METHODS = ('UAGMC_SOURCE_L','UAGMC_OD_FOCAL','AC_ICM','S_EFFECT_EVENT','SA_EFFECT_ICM')
BRANCH_TRACES = ('original','order_4201','order_4204')
HORIZONS = (60,120,240)
CLONE_PROTOCOL = 'dispatch_rebind_v2_20261101'
PHASE_DIRS = {'branches':'branches_v2','intervene':'intervene_v2','tail':'tail_v2'}
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'


def safe(path):
    p = Path(path).resolve()
    if not p.is_relative_to(WS):
        raise RuntimeError('输出越过独立工作区')
    return p


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x


def save(path,x):
    p=safe(path); p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.pending')
    temp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(p)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with safe(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


class PauseRequested(Exception):
    """只中断本轮任务；已完成的独立格保留。"""


def check_pause(job):
    root=Path(job.get('run_root',Path(job['out']).parents[2]))
    if (root/'STOP_REQUESTED').exists():raise PauseRequested('用户请求暂停本轮诊断')


def prepare(archive,out,family):
    marker=out/f'INPUT_{family}.json'
    if marker.exists():return read(marker)
    archive=Path(archive).resolve(); before=archive.stat(); digest=sha(archive)
    if family=='strict' and digest!=EXPECTED_STRICT:raise RuntimeError('旧100M模型包哈希不同')
    suite,entry=(STRICT,STRICT_ENTRY) if family=='strict' else (PT,PT_ENTRY)
    extracted=[]; seen=set()
    with tarfile.open(archive,'r|gz') as tf:
        for item in tf:
            rel=PurePosixPath(item.name)
            if rel.is_absolute() or '..' in rel.parts or any(':' in p for p in rel.parts) or item.issym() or item.islnk():
                raise RuntimeError('归档包含不安全路径')
            if item.name in seen:raise RuntimeError('归档路径重复')
            seen.add(item.name)
            if not item.isfile():continue
            parts=rel.parts
            # 两种归档前缀均规范化为本轮独立input目录。
            if suite in parts:
                sub=PurePosixPath(*parts[parts.index(suite)+1:])
                if not sub.parts:continue
                take=(sub.parts[0]=='_author_source' and rel.suffix in ('.py','.csv','.json'))
                if sub.parts[0]=='cells' and len(sub.parts)>2:
                    cell=sub.parts[1]
                    selected=(cell.startswith('P2__') if family=='strict' else cell.startswith(('CORE__','BOUNDARY__')))
                    take=selected and (rel.name in ('TASK.json','TRAIN_END.json','MODEL_MANIFEST.json') or
                        ('resume' in sub.parts and rel.name in (f'step_{STEP:09d}.zip',f'step_{STEP:09d}.pkl')) or
                        ('evaluation' in sub.parts and rel.name==f'step_{STEP:09d}.json'))
                target=out/'input'/family/suite/Path(*sub.parts)
            elif rel.name==entry:
                take=True;target=out/'input'/family/entry
            else:continue
            if not take:continue
            if item.size>100*1024**2:raise RuntimeError('单条目过大')
            data=tf.extractfile(item).read()
            if len(data)!=item.size:raise RuntimeError('归档条目截断')
            p=safe(target);p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists() and p.read_bytes()!=data:raise RuntimeError('独立输入已存在不同内容')
            if not p.exists():p.write_bytes(data)
            extracted.append(dict(path=str(p),size=len(data),sha256=hashlib.sha256(data).hexdigest()))
            if p.suffix=='.zip':
                with zipfile.ZipFile(p) as z:
                    if z.testzip():raise RuntimeError('模型CRC失败')
    if archive.stat().st_size!=before.st_size or archive.stat().st_mtime_ns!=before.st_mtime_ns:
        raise RuntimeError('归档可能仍在下载，不使用半成品')
    root=out/'input'/family/suite
    jobs=[]
    for folder in sorted((root/'cells').iterdir()):
        task=read(folder/'TASK.json')['task']
        for ext in ('.zip','.pkl'):
            if not (folder/'resume'/f'step_{STEP:09d}{ext}').is_file():raise RuntimeError('模型归一化配对缺失')
        jobs.append(dict(id=folder.name,task=task,root=str(root),runtime=str(out/'input'/family/entry),family=family))
    expected=60 if family=='strict' else 28
    if len(jobs)!=expected:raise RuntimeError(f'模型数不是{expected}，实际{len(jobs)}')
    result=dict(archive=str(archive),sha256=digest,bytes=before.st_size,jobs=jobs,extracted=extracted)
    save(marker,result)
    return result


def make_bank(out,root):
    marker=out/'BANK.json'
    if marker.exists():return read(marker)
    original=next((Path(root)/'_author_source').glob('*/train_data/passengers_300.csv'))
    with original.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if len(rows)!=300:raise RuntimeError('需求数不是300')
    bank=[]
    od_keys=('origin_x','origin_y','dest_x','dest_y')
    for seed in (None,4201,4202,4203,4204):
        ods=[tuple(r[k] for k in od_keys) for r in rows]
        if seed is not None:random.Random(seed).shuffle(ods)
        name='original' if seed is None else f'order_{seed}'
        path=safe(out/'bank'/f'{name}.csv');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
            for row,od in zip(rows,ods):writer.writerow(dict(row,**dict(zip(od_keys,od))))
        bank.append(dict(name=name,path=str(path),sha256=sha(path),scope='原OD联合多重集和到达时刻不变，仅重排顺序',confirmation=seed==4204))
    save(marker,bank);return bank


def load_runtime(job,work):
    os.chdir(work)
    os.environ.update(UAM_STRICT_MODE='E1',UAM_STRICT_FLEET='12',UAM_STRICT_LOG_EVENTS='0')
    name='frozen_large_runtime'
    spec=importlib.util.spec_from_file_location(name,job['runtime'])
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    if job['family']=='strict':
        m.install_learning_classes();m.materialize_source(Path(job['root']));m.activate_source(Path(job['root']),job['task']['k'])
    else:
        t=job['task']
        os.environ.update(UAM_ATTR_REWARD=t['reward_mode'],UAM_ATTR_TIME=t['time_mode'],UAM_ATTR_AIR_OVERRIDE='POLICY',
                          UAM_ATTR_PHASE='eval')
        # 只调用冻结类初始化，不调用训练/评估批处理入口。
        m.initialize_full_runtime();m.materialize_source(Path(job['root']));m.activate_source(Path(job['root']),t['k'])
    from loguru import logger
    logger.remove();logging.disable(logging.CRITICAL);m.torch.set_num_threads(1)
    return m


def load_model(m,job):
    path=Path(job['root'])/'cells'/job['id']/'resume'/f'step_{STEP:09d}'
    cls=m.SourcePPO if job['family']=='strict' else m.FullPPO
    model=cls.load(str(path)+'.zip',device='cpu')
    class NormLoader(pickle.Unpickler):
        def find_class(self,module,name):
            if module in ('__main__','__mp_main__') and hasattr(m,name):return getattr(m,name)
            return super().find_class(module,name)
    with Path(str(path)+'.pkl').open('rb') as f:norm=NormLoader(f).load()
    norm.training=False;norm.norm_reward=False;model.policy.set_training_mode(False)
    for p in model.policy.parameters():p.requires_grad_(False)
    def forbidden(*args,**kwargs):raise RuntimeError('无训练运行禁止参数更新')
    model.learn=model.train=forbidden
    for opt in (getattr(model.policy,'optimizer',None),getattr(model,'aux_optimizer',None)):
        if opt is not None:opt.step=forbidden
    return model,norm


def env_new(m,job,trace,seed):
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(seed)
    t=job['task'];root=Path(job['root'])
    env=m.SourceEnv(root,t['k'],600,trace) if job['family']=='strict' else m.FullEnv(root,t['k'],600,trace,t['method'],'E1')
    obs,_=env.reset(seed=seed)
    return env,obs


def physical(env,m):
    s=env.scenario; focal=s.persons.persons[s.waiting_decisions[0]] if s.waiting_decisions else None
    stations=[]
    for vid in m.CANDIDATES[env.k]:
        v=s.vertiports.vertiport_list[str(vid)]
        incoming=[dict(id=str(p.id),eta=float(p.current_timer or 0)) for p in s.persons.persons.values()
                  if str(getattr(p,'origin_vertiport_id',''))==str(vid) and p.state=='enroute' and p.sub_state=='to_vertiport']
        planes=[dict(id=str(e.id),state=e.state.name,ready=max(0.,e.spec.battery_capacity_kwh-e.battery_kwh)/max(e.spec.charge_rate_kwh_per_min,1e-9))
                for e in s.vertiports.evtols_at_vertiport.get(str(vid),[])]
        returning=[dict(id=str(e.id),eta=float(e.remaining_time)) for e in s._all_evtols.values()
                   if e.state.name=='FLYING' and not e.passenger_ids and str(e.target_vertiport_id)==str(vid)]
        access=float(s.vehicles.estimate_travel_time(focal.origin_position,v.vertiport_position)) if focal else 0.
        stations.append(dict(station=vid,queue=list(v.person_list),incoming=incoming,planes=planes,returns=returning,access=access))
    return dict(time=int(s.time),focal=str(focal.id) if focal else None,stations=stations)


def ready(env,m):
    if not env.scenario.waiting_decisions:return 0
    s=env.scenario;p=s.persons.persons[s.waiting_decisions[0]];scores=[]
    for vid in m.CANDIDATES[env.k]:
        v=s.vertiports.vertiport_list[str(vid)]
        access=float(s.vehicles.estimate_travel_time(p.origin_position,v.vertiport_position))
        dist=s.vertiports._calculate_distance(v.vertiport_position,s.vertiports.vertiport_list['2'].vertiport_position)
        flight=dist/200.*60.
        supply=[max(0.,e.spec.battery_capacity_kwh-e.battery_kwh)/max(e.spec.charge_rate_kwh_per_min,1e-9)
                for e in s.vertiports.evtols_at_vertiport.get(str(vid),[]) if not e.passenger_ids and e.state.name in ('IDLE','CHARGING')]
        incoming=sum(str(getattr(p2,'origin_vertiport_id',''))==str(vid) and p2.state=='enroute' and p2.sub_state=='to_vertiport'
                     and max(0.,float(p2.current_timer or 0))<=access for p2 in s.persons.persons.values())
        cycle=2*flight+2*dist*.55/5.32
        wait=min(supply,default=flight+dist*.55/5.32)+(len(v.person_list)+incoming)//4*cycle/max(1,len(supply))
        scores.append(max(access,wait)+flight)
    return min(range(env.k),key=lambda i:(scores[i],i))


def advance(m,job,env,obs,model,norm,mode='MODEL',forced=None):
    """同一分钟的乘客和飞机决策接口保持源运行时，强制动作必须合法。"""
    if job['family']=='strict':
        if forced is not None:p=int(forced[0])
        elif mode=='READY':p=ready(env,m)
        elif mode.startswith('AUTHOR_'):
            name=mode.split('_',1)[1]
            if not hasattr(env,'_diag_authors'):env._diag_authors={}
            if name not in env._diag_authors:env._diag_authors[name]=m.AuthorFunctions(name)
            p=m.adapter_decision(env.scenario,env.k,env._diag_authors[name],m.np)[0] if env.scenario.waiting_decisions else 0
        elif not env.scenario.waiting_decisions:p=0
        else:p=int(model.predict(norm.normalize_obs(obs.copy()),deterministic=True)[0])
        new,reward,a,b,_=env.step(p)
        return new,bool(a or b),(p,-1),float(reward)
    pre=norm.normalize_obs(obs[None])
    if forced is not None:p=int(forced[0])
    elif mode=='READY':p=ready(env,m)
    else:p=int(model.policy.sample_passenger(m.torch.as_tensor(pre,device='cpu'),True)[0][0])
    pending_raw=env.prepare_step(p)
    if forced is not None and forced[1]>=0:
        a=int(forced[1])
        legal=pending_raw[env.schema.return_mask]
        if legal[a]<=.5:raise ValueError('非法飞机候选')
    elif mode in ('READY','AIR_LQ'):a=-1
    else:
        pending=norm.normalize_obs(pending_raw[None]);pending[:,:env.schema.pack]=pre[:,:env.schema.pack]
        a=int(model.policy.sample_return(m.torch.as_tensor(pending,device='cpu'),m.torch.as_tensor([p],device='cpu'),True)[0])
    new,reward,done,_=env.finish_step(a)
    return new,bool(done),(p,a),float(reward)


def metrics(m,job,env,counts,area):
    ep=m.episode_metrics(env,counts,area) if job['family']=='strict' else m._base_episode_metrics(env,counts,area)
    # 大诊断保留必要明细，避免重复复制逐飞机日志和全量控制日志。
    for key in ('aircraft_events','resource_timeline','decision_trace','control_trace'):ep.pop(key,None)
    s=env.scenario;missing=set(s.persons.persons)-set(s.finished_ids)
    ep['unfinished_states']=[dict(id=str(pid),born=s.persons.persons[pid].spawn_time,state=s.persons.persons[pid].state,
                                  substate=s.persons.persons[pid].sub_state,station=str(s.persons.persons[pid].origin_vertiport_id)) for pid in sorted(missing)]
    ep['terminal_queues']={str(v):len(s.vertiports.vertiport_list[str(v)].person_list) for v in m.CANDIDATES[env.k]}
    return clean(ep)


def run_episode(m,job,model,norm,trace,seed,mode='MODEL',capture=False,tape=None,correction=None):
    env,obs=env_new(m,job,trace,seed);area=0.;counts=[0]*env.k;index=0;decisions=[];actions=[];ledger=[];corrections=[]
    started=time.monotonic()
    try:
        with m.torch.no_grad():
            while True:
                s=env.scenario;area+=len(set(s.persons.persons)-set(s.finished_ids));ledger.append(dict(time=s.time,unfinished=len(set(s.persons.persons)-set(s.finished_ids))))
                if int(s.time)%30==0:check_pause(job)
                active=bool(s.waiting_decisions);forced=None
                if active and tape is not None:forced=(tape[index],-1)
                if active and correction and index%15==7:
                    costs=legal_costs(m,job,env,obs,model,norm,120,correction)
                    if job['family']=='strict':prior=int(model.predict(norm.normalize_obs(obs.copy()),deterministic=True)[0])
                    else:prior=int(model.policy.sample_passenger(m.torch.as_tensor(norm.normalize_obs(obs[None]),device='cpu'),True)[0][0])
                    p=min(costs,key=lambda a:(costs[a],a!=prior,a));forced=(p,-1)
                    corrections.append(dict(index=index,costs=costs,chosen=p,prior=prior))
                snap=physical(env,m) if active and capture else None
                oldobs=obs.copy()
                obs,done,action,reward=advance(m,job,env,obs,model,norm,mode,forced)
                if active:
                    counts[action[0]]+=1;actions.append(action[0])
                    if capture:decisions.append(dict(index=index,action=list(action),observation=oldobs.tolist(),**snap))
                    index+=1
                if done:break
        ep=metrics(m,job,env,counts,area)
        return dict(metrics=ep,decisions=decisions,actions=actions,ledger=ledger,corrections=corrections,
                    wall_seconds=time.monotonic()-started,new_training_transitions=0)
    finally:env.close()


def clone_env(env):
    if getattr(env,'_generator',None) is not None:raise RuntimeError('不可克隆暂停中的generator')
    result=copy.deepcopy(env)
    if result.raw.env is not result.scenario:raise RuntimeError('克隆scenario引用不一致')
    if hasattr(result,'_dispatch_hook') and getattr(env,'_old_dispatch',None) is not None:
        # FullEnv的调度入口是MethodType(已绑定wrapper的方法, scenario)。
        # deepcopy只复制外层scenario，不复制内层方法的wrapper所有者。
        # 两层都显式重绑，否则分支会读取原环境的_priority或改动原环境。
        original_dispatch=env._old_dispatch
        function=getattr(original_dispatch,'__func__',None)
        if function is None:raise RuntimeError('未识别的原始返航调度绑定，拒绝猜测')
        result._old_dispatch=MethodType(function,result.scenario)
        result.scenario._strict_dispatch_returns=MethodType(result._dispatch_hook,result.scenario)
        hook=result.scenario._strict_dispatch_returns
        if hook.__self__ is not result.scenario or getattr(hook.__func__,'__self__',None) is not result:
            raise RuntimeError('克隆调度入口未同时绑定克隆wrapper与scenario')
        if result._old_dispatch.__self__ is not result.scenario:
            raise RuntimeError('克隆LQ入口未绑定克隆scenario')
        if getattr(result,'legacy',None) is not None and result.legacy.scenario is not result.scenario:
            raise RuntimeError('克隆legacy与scenario引用不一致')
    bound=getattr(result,'_generator_function',None)
    if bound is not None:
        owner=getattr(bound,'__self__',None)
        if owner is not None and owner is not result.scenario:
            raise RuntimeError('克隆分钟函数绑定了原scenario')
        if owner is None and getattr(bound,'__closure__',None):
            raise RuntimeError('分钟函数存在未核验的闭包状态')
    return result


def hide_future(env):
    # 清除分支模拟器尚未揭示的文件请求；已存在的乘客和物理事件完全保留。
    env.scenario.persons.file_data={}
    if env.scenario.persons.spawn_file is None:raise RuntimeError('请求关闭未保持文件生成模式')


def branch(m,job,template,observation,model,norm,action,horizon,scope='REALIZED',continuation='MODEL',future_rows=None,audit=False):
    env=clone_env(template);obs=observation.copy();initial=set(env.scenario.persons.persons)
    focal=env.scenario.waiting_decisions[0] if env.scenario.waiting_decisions else None
    if scope=='COMMITTED':hide_future(env)
    elif scope=='BOOTSTRAP':
        hide_future(env);env.scenario.persons.file_data=future_rows
    cost=[0.,0.,0.];snapshots={};ticks=0;audit_trace=[]
    try:
        with m.torch.no_grad():
            for offset in range(horizon):
                if offset%60==0:check_pause(job)
                s=env.scenario;u=set(s.persons.persons)-set(s.finished_ids)
                tick=int(s.time);unfinished_count=len(u)
                cost[0]+=int(focal in u);cost[1]+=len((u&initial)-{focal});cost[2]+=len(u-initial)
                obs,done,actual,reward=advance(m,job,env,obs,model,norm,continuation,action if offset==0 else None)
                if audit:audit_trace.append(dict(time=tick,unfinished=unfinished_count,action=list(actual)))
                ticks+=1
                if offset+1 in HORIZONS or done or offset+1==horizon:
                    snapshots[str(offset+1)]=dict(total=sum(cost),focal=cost[0],existing=cost[1],future=cost[2],finished=len(set(s.finished_ids)))
                if done:break
        return dict(snapshots=snapshots,ticks=ticks,audit_trace=audit_trace) if audit else dict(snapshots=snapshots,ticks=ticks)
    finally:env.close()


def legal_costs(m,job,env,obs,model,norm,horizon,objective):
    # 只替换当前乘客选择，飞机继续原冻结控制器；不读取真实未来请求。
    py=random.getstate();npstate=m.np.random.get_state();scores={}
    for p in range(env.k):
        random.setstate(py);m.np.random.set_state(npstate)
        x=branch(m,job,env,obs,model,norm,(p,-1),horizon,'COMMITTED','MODEL')['snapshots'][str(horizon)]
        scores[p]=x['focal'] if objective=='SELF' else x['total']
    random.setstate(py);m.np.random.set_state(npstate)
    return scores


def baseline_worker(m,job,model,norm,out):
    results=[]
    for b in job['bank']:
        for seed in ((123,124) if b['name']=='original' else (123,)):
            key=f"{b['name']}_s{seed}"
            path=out/(key+'.json')
            if path.exists():result=read(path)
            else:
                capture=job['branch_model'] and b['name'] in BRANCH_TRACES and seed==123
                result=run_episode(m,job,model,norm,b['path'],seed,capture=capture)
                result.update(trace=b['name'],eval_seed=seed);save(path,result)
            if b['name']=='original':
                old=read(Path(job['root'])/'cells'/job['id']/'evaluation'/f'step_{STEP:09d}.json')
                ref=next(e for e in old['episodes'] if e['eval_seed']==seed and e.get('mode','DET')=='DET')
                for field in ('generated','finished','ATT_completed_only','restricted_person_time_per_generated'):
                    if abs(result['metrics'][field]-ref[field])>1e-6:raise RuntimeError('最终模型历史复评不一致：'+field)
            results.append(dict(trace=b['name'],eval_seed=seed,metrics=result['metrics'],seconds=result['wall_seconds']))
    save(out/'END.json',dict(id=job['id'],family=job['family'],task=job['task'],results=results,new_training_transitions=0))


def branch_worker(m,job,model,norm,out):
    rows=[];state_count=0;total_ticks=0
    for b in job['bank']:
        if b['name'] not in BRANCH_TRACES:continue
        base=read(Path(job['baseline'])/(b['name']+'_s123.json'))
        records=base['decisions'];rng=random.Random(20261101+job['task']['seed'])
        selected=sorted(i for phase in range(3) for i in rng.sample([r['index'] for r in records if 100*phase<=r['index']<100*(phase+1)],6))
        save(out/(b['name']+'_SAMPLE.json'),dict(indices=selected,selection='每早中晚各6随机状态，候选标签生成前固定'))
        env,obs=env_new(m,job,b['path'],123);index=0;saved={}
        try:
            with m.torch.no_grad():
                while True:
                    if env.scenario.waiting_decisions:
                        if index in selected:
                            rec=records[index]
                            if not m.np.array_equal(obs,m.np.asarray(rec['observation'],dtype=obs.dtype)):raise RuntimeError('状态重建失败')
                            saved[index]=(clone_env(env),obs.copy(),random.getstate(),m.np.random.get_state())
                        index+=1
                    obs,done,_,_=advance(m,job,env,obs,model,norm)
                    if done:break
        finally:env.close()
        for index,(template,raw,py,npstate) in saved.items():
            record=records[index];state_count+=1
            joint=bool(getattr(template,'joint',False))
            original=(record['action'][0],record['action'][1] if joint else -1)
            actions=[(p,a) for p in range(template.k) for a in (range(template.k) if joint else (-1,))]
            # 先验证原动作，再计算干预候选；校验不通过时不积累伪反事实标签。
            if original not in actions:raise RuntimeError('原动作不在候选空间')
            actions=[original]+[a for a in actions if a!=original]
            reference_checked=False
            for action in actions:
                for scope in ('REALIZED','COMMITTED'):
                    random.setstate(py);m.np.random.set_state(npstate)
                    reference=(scope=='REALIZED' and action==original)
                    try:result=branch(m,job,template,raw,model,norm,action,240,scope,audit=reference)
                    except ValueError as error:
                        if str(error)!='非法飞机候选':raise
                        continue
                    total_ticks+=result['ticks']
                    if scope=='REALIZED' and action==original:
                        for horizon,values in result['snapshots'].items():
                            expected=sum(x['unfinished'] for x in base['ledger'] if record['time']<=x['time']<record['time']+int(horizon))
                            if values['total']!=expected:
                                first=next((dict(time=x['time'],clone_unfinished=x['unfinished'],baseline_unfinished=y['unfinished'])
                                            for x in result.get('audit_trace',[]) for y in base['ledger']
                                            if x['time']==y['time'] and x['unfinished']!=y['unfinished']),None)
                                save(out/'REFERENCE_MISMATCH.json',dict(protocol=CLONE_PROTOCOL,id=job['id'],trace=b['name'],index=index,
                                    action=list(action),horizon=int(horizon),clone_cost=values['total'],baseline_cost=expected,
                                    first_divergence=first,audit_trace=result.get('audit_trace',[])))
                                raise RuntimeError('克隆参考成本不等于原轨迹；详细差异已保存REFERENCE_MISMATCH.json')
                        reference_checked=True
                    result.pop('audit_trace',None)
                    rows.append(dict(trace=b['name'],index=index,time=record['time'],original=list(original),action=list(action),
                                     scope=scope,physical=record['stations'],**result))
            if not reference_checked:raise RuntimeError('原动作未进入合法候选或未核验，不接受该状态')
            save(out/'PROGRESS.json',dict(states=state_count,branches=len(rows),simulated_minutes=total_ticks))
            template.close()
    save(out/'END.json',dict(id=job['id'],family=job['family'],task=job['task'],states=state_count,rows=rows,
                              simulated_minutes=total_ticks,clone_protocol=CLONE_PROTOCOL,new_training_transitions=0))


def rule_worker(m,job,model,norm,out):
    rows=[]
    modes=('AUTHOR_SPF','AUTHOR_STTF','AUTHOR_QTTI','READY') if job['family']=='strict' else ('READY','AIR_LQ')
    for b in job['bank']:
        for mode in modes:
            result=run_episode(m,job,model,norm,b['path'],123,mode)
            save(out/(b['name']+'_'+mode+'.json'),result)
            rows.append(dict(trace=b['name'],mode=mode,metrics=result['metrics'],seconds=result['wall_seconds']))
    save(out/'END.json',dict(id=job['id'],family=job['family'],task=job['task'],results=rows,clone_protocol=CLONE_PROTOCOL,new_training_transitions=0))


def intervention_worker(m,job,model,norm,out):
    rows=[]
    for b in job['bank']:
        if b['name'] not in BRANCH_TRACES:continue
        for objective in ('SELF','SYSTEM'):
            # 固定每15次乘客决策检查一次，不用事后Oracle标签挑触发点。
            result=run_episode(m,job,model,norm,b['path'],123,correction=objective)
            save(out/(b['name']+'_'+objective+'.json'),result)
            rows.append(dict(trace=b['name'],mode=objective,metrics=result['metrics'],seconds=result['wall_seconds'],corrections=len(result['corrections'])))
    save(out/'END.json',dict(id=job['id'],family=job['family'],task=job['task'],results=rows,clone_protocol=CLONE_PROTOCOL,new_training_transitions=0))


def tail_worker(m,job,model,norm,out):
    """事后仅改变各站最后余数乘客的去向；不放宽满座约束，不作为在线策略。"""
    rows=[]
    for b in job['bank']:
        if b['name'] not in BRANCH_TRACES:continue
        sources=[('MODEL',Path(job['baseline'])/(b['name']+'_s123.json'))]
        for label in ('SELF','SYSTEM'):
            p=Path(job['intervention'])/(b['name']+'_'+label+'.json')
            if p.exists():sources.append((label,p))
        for label,path in sources:
            original=read(path);tape=original['actions']
            unmodified=run_episode(m,job,model,norm,b['path'],123,tape=tape)
            if abs(unmodified['metrics']['restricted_person_time_per_generated']-original['metrics']['restricted_person_time_per_generated'])>1e-6:
                raise RuntimeError('不修改的乘客动作序列无法复现，尾批诊断停止')
            pool=[]
            for a in range(job['task']['k']):
                ids=[i for i,x in enumerate(tape) if x==a];r=len(ids)%4
                if r:pool.extend(ids[-r:])
            trials=[]
            for target in range(job['task']['k']) if pool else ():
                proposal=tape.copy()
                for i in pool:proposal[i]=target
                result=run_episode(m,job,model,norm,b['path'],123,tape=proposal)
                trials.append(dict(target=target,moved=sum(a!=b2 for a,b2 in zip(tape,proposal)),metrics=result['metrics']))
            if not trials:trials=[dict(target=None,moved=0,metrics=original['metrics'])]
            save(out/(b['name']+'_'+label+'.json'),dict(original=original['metrics'],trials=trials,privileged_hindsight=True))
            rows.append(dict(trace=b['name'],mode=label,original=original['metrics'],trials=trials))
    save(out/'END.json',dict(id=job['id'],family=job['family'],task=job['task'],results=rows,clone_protocol=CLONE_PROTOCOL,new_training_transitions=0))


def worker(jobpath):
    job=read(jobpath);out=Path(job['out']);m=load_runtime(job,out);model,norm=load_model(m,job)
    function={'baseline':baseline_worker,'branches':branch_worker,'rules':rule_worker,'intervene':intervention_worker,'tail':tail_worker}[job['phase']]
    function(m,job,model,norm,out)


def launch(job):
    out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    try:check_pause(job)
    except PauseRequested:return dict(id=job['id'],phase=job['phase'],status='paused')
    if (out/'END.json').exists():
        if job['phase'] in PHASE_DIRS and read(out/'END.json').get('clone_protocol')!=CLONE_PROTOCOL:
            raise RuntimeError('诊断格版本不同，不覆盖旧记录')
        return dict(id=job['id'],phase=job['phase'],status='existing')
    save(out/'JOB.json',dict(job,diagnostic_script_sha256=sha(__file__)))
    with (out/'worker.log').open('a',encoding='utf-8') as f:
        p=subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker',str(out/'JOB.json')],
                         stdout=f,stderr=subprocess.STDOUT,timeout=7200,cwd=out)
    if p.returncode==130:return dict(id=job['id'],phase=job['phase'],status='paused')
    if p.returncode:raise RuntimeError(str(out/'worker.log'))
    return dict(id=job['id'],phase=job['phase'],status='done')


def pipeline(out,records,bank,phases,workers):
    for phase in phases:
        jobs=[]
        for record in records:
            for base in record['jobs']:
                t=base['task'];family=base['family']
                branch_model=(family=='strict' and t['method']=='UAGMC_SOURCE_L') or (family=='pt' and t['arm'] in ('S_PT_RAW','J_PT_RAW') and t['k']==3)
                selected=True
                if phase in ('branches','tail'):selected=branch_model
                elif phase=='rules':selected=(family=='strict' and t['method']=='UAGMC_SOURCE_L' and t['seed']==601) or (family=='pt' and t['arm'] in ('S_PT_RAW','J_PT_RAW'))
                elif phase=='intervene':selected=branch_model
                if not selected:continue
                if phase!='baseline' and not (out/'baseline'/family/base['id']/'END.json').exists():raise RuntimeError('对应模型baseline尚未结束')
                jobs.append(dict(base,bank=bank,phase=phase,branch_model=branch_model,
                                 baseline=str(out/'baseline'/family/base['id']),out=str(out/PHASE_DIRS.get(phase,phase)/family/base['id']),run_root=str(out)))
                jobs[-1]['intervention']=str(out/PHASE_DIRS['intervene']/family/base['id'])
        rng=random.Random(20261101);rng.shuffle(jobs)
        completed=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fs={pool.submit(launch,j):j for j in jobs}
            for f in as_completed(fs):
                j=fs[f]
                try:r=f.result();completed.append(r);print(f"[{phase} {len(completed)}/{len(jobs)}] {j['family']} {j['id']}",flush=True)
                except Exception as error:
                    save(out/('FAILURE_v2.json' if phase in PHASE_DIRS else 'FAILURE.json'),dict(job=j,error=str(error),clone_protocol=CLONE_PROTOCOL));raise
                save(out/'progress.json',dict(phase=phase,result_directory=PHASE_DIRS.get(phase,phase),clone_protocol=CLONE_PROTOCOL,
                                              completed=len(completed),total=len(jobs),new_training_transitions=0))
        if any(r['status']=='paused' for r in completed):
            save(out/'SESSION_PAUSED.json',dict(phase=phase,status='PAUSED_BY_USER',new_training_transitions=0))
            return False
        save(out/(phase+'_SUMMARY.json'),completed)
    return True


def summarize_pack(out):
    """服务器自动生成可核对汇总并打数据包，不下载或删除模型。"""
    registered=[j for p in sorted(out.glob('INPUT_*.json')) for j in read(p)['jobs']]
    n_branch=sum((j['family']=='strict' and j['task']['method']=='UAGMC_SOURCE_L') or
                 (j['family']=='pt' and j['task'].get('arm') in ('S_PT_RAW','J_PT_RAW') and j['task']['k']==3) for j in registered)
    n_rules=sum((j['family']=='strict' and j['task']['method']=='UAGMC_SOURCE_L' and j['task']['seed']==601) or
                (j['family']=='pt' and j['task'].get('arm') in ('S_PT_RAW','J_PT_RAW')) for j in registered)
    for phase,expected in dict(baseline=len(registered),rules=n_rules,branches=n_branch,intervene=n_branch,tail=n_branch).items():
        files=list((out/PHASE_DIRS.get(phase,phase)).glob('*/*/END.json'))
        if len(files)!=expected:raise RuntimeError(f'{phase}尚未完成：{len(files)}/{expected}，不打完整完成包')
        if phase in PHASE_DIRS and any(read(p).get('clone_protocol')!=CLONE_PROTOCOL for p in files):
            raise RuntimeError('克隆协议不匹配，不将旧分支混入新报告')
    baseline=[];rules=[];interventions=[];local=[];tails=[]
    for path in sorted((out/'baseline').glob('*/*/END.json')):
        d=read(path);t=d['task']
        for r in d['results']:
            e=r['metrics']
            baseline.append(dict(family=d['family'],id=d['id'],k=t['k'],train_seed=t['seed'],method=t['method'],
                                 arm=t.get('arm',''),trace=r['trace'],eval_seed=r['eval_seed'],rpt=e['restricted_person_time_per_generated'],
                                 completion=e['completion'],full_att=e['ATT'],co_att=e['ATT_completed_only'],seconds=r['seconds']))
    for phase,target in (('rules',rules),('intervene',interventions)):
        for path in sorted((out/PHASE_DIRS.get(phase,phase)).glob('*/*/END.json')):
            d=read(path)
            for r in d['results']:
                e=r['metrics']
                target.append(dict(family=d['family'],id=d['id'],k=d['task']['k'],train_seed=d['task']['seed'],
                                   trace=r['trace'],mode=r['mode'],rpt=e['restricted_person_time_per_generated'],
                                   completion=e['completion'],full_att=e['ATT'],seconds=r['seconds']))
    for path in sorted((out/PHASE_DIRS['branches']).glob('*/*/END.json')):
        d=read(path);groups={}
        for r in d['rows']:groups.setdefault((r['trace'],r['index']),[]).append(r)
        for (trace,index),rows in groups.items():
            actual={tuple(r['action']):r for r in rows if r['scope']=='REALIZED'}
            known={tuple(r['action']):r for r in rows if r['scope']=='COMMITTED'}
            original=tuple(rows[0]['original'])
            for h in HORIZONS:
                key=str(h)
                choose=lambda field:min(known,key=lambda a:(known[a]['snapshots'][key][field],a!=original,a))
                system=choose('total');selfish=choose('focal')
                best=min(r['snapshots'][key]['total'] for r in actual.values())
                base=actual[original]['snapshots'][key]['total']
                local.append(dict(family=d['family'],id=d['id'],k=d['task']['k'],trace=trace,index=index,horizon=h,
                                  original_cost=base,offline_local_gap=base-best,
                                  committed_system_gain=base-actual[system]['snapshots'][key]['total'],
                                  committed_self_gain=base-actual[selfish]['snapshots'][key]['total'],
                                  legal_candidates=len(actual)))
    for path in sorted((out/PHASE_DIRS['tail']).glob('*/*/END.json')):
        d=read(path)
        for r in d['results']:
            full=[x['metrics']['ATT'] for x in r['trials'] if x['metrics']['full']]
            tails.append(dict(family=d['family'],id=d['id'],k=d['task']['k'],trace=r['trace'],mode=r['mode'],
                              original_rpt=r['original']['restricted_person_time_per_generated'],
                              best_repaired_rpt=min(x['metrics']['restricted_person_time_per_generated'] for x in r['trials']),
                              best_full_att=min(full) if full else None,privileged_hindsight=True))
    for name,rows in (('01_全部模型复评.csv',baseline),('02_在线规则与飞机替换.csv',rules),('03_全程定向干预.csv',interventions),
                      ('04_同状态合法预测与事后差距.csv',local),('05_尾批竞争解释.csv',tails)):
        csv_write(out/name,rows)
    groups=[]
    for family in sorted({r['family'] for r in baseline}):
        for k in (2,3,4):
            for method in sorted({r['method'] for r in baseline if r['family']==family and r['k']==k}):
                subset=[r for r in baseline if r['family']==family and r['k']==k and r['method']==method and r['eval_seed']==123]
                for scope in ('original','reordered'):
                    selected=[r for r in subset if (r['trace']=='original')==(scope=='original')]
                    if not selected:continue
                    groups.append(dict(family=family,k=k,method=method,scope=scope,episodes=len(selected),
                                       rpt_mean=statistics.mean(r['rpt'] for r in selected),
                                       completion_mean=statistics.mean(r['completion'] for r in selected)))
    csv_write(out/'06_描述性分组.csv',groups)
    missing_pt=not (out/'INPUT_pt.json').exists()
    report='\n\n'.join(['# 大规模无训练诊断：执行汇总',
        f"模型回放{len(baseline)}次；规则与飞机替换{len(rules)}次；全程干预{len(interventions)}次；局部状态窗口记录{len(local)}条；尾批对照{len(tails)}条。新增训练0步。",
        '30M人时奖励模型：'+('未提供，不得用旧奖励结论替代。' if missing_pt else '已加载，具体复评是否通过见各格END和worker日志。'),
        '克隆协议：'+CLONE_PROTOCOL+'。旧branches/intervene/tail目录及FAILURE.json只作历史审计，不纳入本次汇总。',
        '## 解读边界',
        '原CSV＋4种同OD多重集重排只检验条件顺序稳健性，不声称跨分布泛化。统计先按训练模型、trace分别汇总，不把状态或窗口当独立训练seed。',
        'COMMITTED分支清除了尚未揭示请求，是无新到达近似预测，不是完全准确的真实未来。REALIZED分支是事后参照，不能直接作为在线已知信息。',
        'SELF和SYSTEM全程干预同次数、同候选、同120分钟预算；只改变候选代价目标，平局保留原策略动作。尾批对照使用完整动作序列，仅作离线诊断，不作为合法在线算法。',
        '## 下一步训练准入',
        '需人工核对：人时奖励后剩余差距、相对强在线解析、非尾批收益、定向修正全程收益，以及确认顺序4204上的方向。没有这些证据，不自动挑方法、不自动启动训练。',
        '本页是自动执行摘要；最终研究结论、机制主次及几千万步训练方案应在读完上述逐格数据后单独给出。',
        '过程参考：Scientific Agent Skills, https://arxiv.org/abs/2609.00065；只借用设计与统计独立单位原则，不作为UAM机制证据。'])
    (out/'执行汇总.md').write_text(report,encoding='utf-8')
    body=''.join('<p>'+html.escape(x).replace('\n','<br>')+'</p>' for x in report.split('\n\n'))
    (out/'执行汇总.html').write_text('<!doctype html><meta charset="utf-8"><title>无训练诊断汇总</title><style>body{max-width:1000px;margin:35px auto;font:17px/1.9 system-ui;padding:20px}p{overflow-wrap:anywhere}</style>'+body,encoding='utf-8')
    changed=[]
    for marker in out.glob('INPUT_*.json'):
        for r in read(marker)['extracted']:
            if Path(r['path']).suffix in ('.zip','.pkl') and sha(r['path'])!=r['sha256']:changed.append(r['path'])
    if changed:raise RuntimeError('回放后输入权重哈希变化')
    save(out/'SESSION_END.json',dict(status='COMPLETE',model_episodes=len(baseline),rule_episodes=len(rules),intervention_episodes=len(interventions),
                                   branch_window_records=len(local),missing_pt_models=missing_pt,clone_protocol=CLONE_PROTOCOL,
                                   weights_unchanged=True,new_training_transitions=0))
    package=out/'uam_large_notrain_20261101_DATAONLY.tar.gz'
    with tarfile.open(package,'w:gz') as tar:
        for p in sorted(out.rglob('*')):
            if p.is_file() and 'input' not in p.relative_to(out).parts and p!=package and p.suffix!='.pending':
                tar.add(p,arcname='uam_large_notrain_20261101/'+p.relative_to(out).as_posix(),recursive=False)
        tar.add(Path(__file__),arcname='code/'+Path(__file__).name,recursive=False)
    print('[完成数据包]',package,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out');parser.add_argument('--strict-archive');parser.add_argument('--pt-archive');parser.add_argument('--worker')
    parser.add_argument('--phase',default='baseline,rules,branches,intervene,tail');parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args.worker);return
    out=safe(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.resume and (out/'STOP_REQUESTED').exists():(out/'STOP_REQUESTED').unlink()
    if args.phase=='summary':summarize_pack(out);return
    records=[]
    if args.strict_archive:records.append(prepare(args.strict_archive,out,'strict'))
    if args.pt_archive:records.append(prepare(args.pt_archive,out,'pt'))
    if not records:raise RuntimeError('必须明确至少一个输入归档')
    bank=make_bank(out,records[0]['jobs'][0]['root'])
    save(out/'SESSION_STARTED_v2.json',dict(clone_protocol=CLONE_PROTOCOL,script_sha256=sha(__file__),
        phase_directories=PHASE_DIRS,baseline_reused=True,rules_reused=True,
        note='旧FAILURE.json保留作历史；本版分支异常另记FAILURE_v2.json',new_training_transitions=0))
    completed=pipeline(out,records,bank,args.phase.split(','),min(2,max(1,args.workers)))
    if completed and 'tail' in args.phase.split(','):summarize_pack(out)


if __name__=='__main__':
    try:main()
    except PauseRequested:raise SystemExit(130)
