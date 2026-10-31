#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""五千万上限的缺口归因实验；本文件包含校验固定的历史运行时，不依赖其他矩阵入口。"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')
import ast, base64, hashlib, zlib, io
_FROZEN_ATTR_SHA = __FROZEN_SHA__
_FROZEN_ATTR_B85 = __FROZEN_PAYLOAD__
_frozen = zlib.decompress(base64.b85decode(_FROZEN_ATTR_B85))
if hashlib.sha256(_frozen).hexdigest()!=_FROZEN_ATTR_SHA:
    raise RuntimeError('封存运行时校验失败，不混用物理或训练合同')
exec(compile(_frozen,'<frozen_30m_without_main>','exec'),globals())
del _frozen

GAP_VERSION='UAM_GAP_FACTORS_50M_20261101_V1'
ATTR_ROOT='uam_gap_factors_48run_50m_v1'
ATTR_CAP=HARD_CAP=50_000_000
ATTR_STEPS=STEPS=FULL_ACTUAL_STEPS=1_003_520
ATTR_PLANNED=48*ATTR_STEPS
GAP_INIT_PID=None
GAP_DIAG_SHA='19434fe6744be50a8e9ab7eda54dcb5801e86c43fea7f21cd6ceba9f97db098f'
GAP_ARMS={'L00':(False,False,'LSTM'),'L10_TIME':(True,False,'LSTM'),
          'L01_BATCH':(False,True,'LSTM'),'L11_BOTH':(True,True,'LSTM'),
          'T11_BOTH':(True,True,'TRANSFORMER')}

def gap_registry():
    """先完整核心因子块，再穿插普通Transformer对照；不按结果选下一组。"""
    rng=random.Random(20261101);tasks=[]
    def block(stage,k,seed,arms):
        cells=[(control,arm) for control in ('LQ','J3') for arm in arms]
        rng.shuffle(cells)
        for control,arm in cells:
            timing,batch,network=GAP_ARMS[arm]
            tasks.append(dict(id=f'{stage}__E1__K{k}__{control}__{arm}__s{seed}',stage=stage,
                environment='E1',k=k,fleet=12,seed=seed,arm=arm,control=control,
                reward_mode='PT_POST',time_mode='RAW',variant='M0_BASE',
                method=('J3' if control=='J3' else 'SINGLE')+'__M0_BASE',
                timing=timing,batch=batch,network=network))
    for seed in (1701,1702):block('P1_FACTORS',3,seed,list(GAP_ARMS)[:4])
    for seed in (1701,1702):block('P2_GENERIC',3,seed,['T11_BOTH'])
    for seed in (1703,1704):block('P3_REPLICATION',3,seed,list(GAP_ARMS)[:4])
    for seed in (1703,1704):block('P4_GENERIC',3,seed,['T11_BOTH'])
    for k in (2,4):block('P5_BOUNDARY',k,1705,['L00','L11_BOTH'])
    for i,t in enumerate(tasks,1):t['run_order']=i
    assert len(tasks)==48 and len({t['id'] for t in tasks})==48
    assert ATTR_PLANNED==48_168_960 and ATTR_PLANNED<ATTR_CAP
    return tasks

def gap_extra(s,k,focal,timing,batch):
    """只读取已生成乘客和已承诺飞机；不接触file_data或未揭示请求。"""
    if not timing and not batch:return np.zeros(6*k,np.float32)
    incoming_by_station={str(v):[] for v in CANDIDATES[k]}
    # 一次扫描已生成乘客，不为每个候选反复扫描，也不做嵌套仿真。
    for p in s.persons.persons.values():
        vid=str(getattr(p,'origin_vertiport_id',''))
        if vid in incoming_by_station and p.state=='enroute' and p.sub_state=='to_vertiport':
            incoming_by_station[vid].append(max(0.,float(p.current_timer or 0.)))
    values=[]
    for vid in CANDIDATES[k]:
        v=s.vertiports.vertiport_list[str(vid)]
        incoming=incoming_by_station[str(vid)]
        q=len(v.person_list)
        # 601只表示当前无已承诺供给/到达；不存在“已知需等601分钟”的含义。
        times=[0.]*4
        if timing:
            local=[e for e in s.vertiports.evtols_at_vertiport.get(str(vid),[])
                   if not e.passenger_ids and e.state.name in ('IDLE','CHARGING')]
            avg_charge,_=charging_pair(local)
            # ready_descriptor包含既有返航空机及充电的离散估计，非真实未来排队等待。
            ready=ready_descriptor(s,vid)
            returning=[max(0.,float(e.remaining_time)) for e in s._all_evtols.values()
                if e.state.name=='FLYING' and not e.passenger_ids and str(e.target_vertiport_id)==str(vid)]
            # 不另开焦点OD或access捷径，避免把独立OD通路的效果混称为时间效果。
            times=[x/60. for x in [ready,avg_charge,min(incoming) if incoming else 601.,min(returning) if returning else 601.]]
        residues=[(q%4)/3.,((q+len(incoming))%4)/3.] if batch else [0.,0.]
        values.extend(times+residues)
    if len(values)!=6*k or not all(math.isfinite(x) for x in values):
        raise AssertionError('合法信息描述符维数/数值异常')
    return np.asarray(values,np.float32)

class GapSchema:
    """所有臂同一维数；时间/余数关闭时是零槽位，权限位不进入编码器。"""
    def __init__(self,k):
        self.k=int(k);self.frame=4+8*self.k;self.history=6*self.frame
        self.extra=slice(self.history,self.history+6*self.k)
        self.passenger_active=self.extra.stop;self.pack=self.passenger_active+1
        self.return_mask=slice(2*self.pack,2*self.pack+self.k);self.dim=2*self.pack+self.k
    def manifest(self):
        return dict(k=self.k,frame=self.frame,history=self.history,extra=[self.extra.start,self.extra.stop],
            pack=self.pack,dim=self.dim,fields_per_station=['已承诺供给ready估计','本站平均charge','最近已承诺乘客到达','最近已承诺空机到达','队列mod4','队列加在途乘客mod4'],
            boundary='新增信息只接当前pre/dispatch残差，不倒改原六帧历史；无真实未来请求',
            original='源六帧原样；零槽L00是容量匹配的源策略对照')

_gap_old_setup=attr_setup
_gap_old_initialize=attr_initialize
def gap_initialize():
    global GAP_INIT_PID,FullSchema,MinimalSchema,FullEnv,FullFeatures,FullPolicy
    if GAP_INIT_PID==os.getpid():return
    _gap_old_initialize()
    OldEnv,OldPolicy=FullEnv,FullPolicy
    from torch import nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    class GapEnv(OldEnv):
        def __init__(self,*args,**kwargs):
            self.gap_timing=os.environ.get('UAM_GAP_TIME')=='1'
            self.gap_batch=os.environ.get('UAM_GAP_BATCH')=='1'
            super().__init__(*args,**kwargs)
        def _events_and_pack(self,history,focal=None):
            if focal is None and self.scenario.waiting_decisions:
                focal=self.scenario.persons.persons[self.scenario.waiting_decisions[0]]
            history=np.asarray(history,np.float32).reshape(-1).copy()
            if len(history)!=self.schema.history:raise AssertionError('六帧源历史维数改变')
            extra=gap_extra(self.scenario,self.k,focal,self.gap_timing,self.gap_batch)
            return np.concatenate([history,extra,np.asarray([float(bool(self.scenario.waiting_decisions))],np.float32)])
    class PlainTemporalTransformer(nn.Module):
        """普通六帧注意力对照，不声称新算法，也不声称穷尽Transformer调参空间。"""
        def __init__(self,frame):
            super().__init__();self.input=nn.Linear(frame,128)
            self.position=nn.Parameter(torch.zeros(1,6,128))
            layer=nn.TransformerEncoderLayer(128,4,256,dropout=0.,batch_first=True,activation='relu')
            self.encoder=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        def forward(self,x):return self.encoder(self.input(x)+self.position)[:,-1]
    class GapFeatures(BaseFeaturesExtractor):
        def __init__(self,observation_space,k=2,method='SINGLE__M0_BASE',environment='E1'):
            super().__init__(observation_space,128)
            from utilss.encoding import TemporalLSTMExtractor
            self.k,self.method,self.environment=k,method,environment
            self.variant='M0_BASE';self.schema=GapSchema(k)
            self.network=os.environ.get('UAM_GAP_NETWORK','LSTM')
            if self.network=='LSTM':
                self.temporal=TemporalLSTMExtractor(gym.spaces.Box(-np.inf,np.inf,(6,self.schema.frame),np.float32),features_dim=128)
            elif self.network=='TRANSFORMER':self.temporal=PlainTemporalTransformer(self.schema.frame)
            else:raise RuntimeError('未知网络臂')
            # 新的线性残差不消耗原策略初始化随机序列；L00时严格为零输出。
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(911_101+k)
                self.extra_net=nn.Linear(6*k,128,bias=False)
        def _encode(self,obs,start):
            d=self.schema
            return self.temporal(obs[:,start:start+d.history].reshape(-1,6,d.frame))+self.extra_net(obs[:,start+d.extra.start:start+d.extra.stop])
        def forward(self,obs):return self._encode(obs,0)
        def dispatch_features(self,obs):return self._encode(obs,self.schema.pack)
        def auxiliary_parameters(self):return []
    class GapPolicy(OldPolicy):
        def _build(self,lr_schedule):
            features=self.features_extractor;extra=features._modules.pop('extra_net')
            try:super()._build(lr_schedule)
            finally:features.add_module('extra_net',extra)
            # 将残差参数加入同一个PPO优化器；没有辅助loss或额外更新步。
            self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)
    for cls in (GapEnv,PlainTemporalTransformer,GapFeatures,GapPolicy):
        cls.__qualname__=cls.__name__;cls.__module__=__name__;globals()[cls.__name__]=cls
    FullSchema=MinimalSchema=GapSchema;FullEnv=GapEnv;FullFeatures=GapFeatures;FullPolicy=GapPolicy
    GAP_INIT_PID=os.getpid()

initialize_full_runtime=gap_initialize
def gap_setup(task,root,phase):
    os.environ.update(UAM_GAP_TIME=str(int(task.get('timing',False))),UAM_GAP_BATCH=str(int(task.get('batch',False))),
                      UAM_GAP_NETWORK=task.get('network','LSTM'))
    return _gap_old_setup(task,root,phase)
attr_setup=gap_setup

def gap_update_manifest(folder,task):
    p=Path(folder)/'MODEL_MANIFEST.json';m=read_json(p,{})
    m.update(version=GAP_VERSION,original_reward=False,reward_mode='PT_POST',time_mode='RAW',arm=task['arm'],
        timing=task['timing'],batch=task['batch'],network=task['network'],auxiliary='无',
        architecture='源六帧时序编码器＋当前合法描述符线性残差；普通Transformer为替换对照',
        physical_changes='无；满4座、12机队、V2返回权限与充电均不变',
        source_runtime_sha256=_FROZEN_ATTR_SHA,diagnostic_sha256=GAP_DIAG_SHA,entrypoint_sha256=digest(SCRIPT),
        causal_limit='特征组织/可得信息与有限预算效应，不证明源PPO或全部Transformer在原则上不能学习',
        gamma_warning='继承.99；不冒充早期40机队gamma1合同')
    save_json(p,m)
attr_update_manifest=gap_update_manifest

def gap_bank(root,native):
    folder=root/'bank';folder.mkdir(parents=True,exist_ok=True)
    source=Path(native)/'train_data/passengers_300.csv'
    with source.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    if fields is None or len(rows)!=300:raise RuntimeError('源300人CSV不符，不换需求凑结果')
    # 到达时刻列原样固定；整条OD元组一起打乱，不分别打乱起点/终点。
    spatial=[x for x in fields if x.lower() not in ('time','timestamp','arrival_time','arrival','request_time','id','person_id')]
    if not spatial:raise RuntimeError('源OD列无法辨认')
    bank=[dict(name='original',path=str(source),sha256=digest(source),role='历史原顺序参考')]
    for seed in (4301,4302,4303,4304):
        shuffled=[{k:r[k] for k in spatial} for r in rows];random.Random(seed).shuffle(shuffled)
        result=[dict(r,**od) for r,od in zip(rows,shuffled)]
        path=folder/f'order_{seed}.csv'
        # 新bank首次创建后只校验，不在恢复时覆盖。
        text=io.StringIO(newline='');w=csv.DictWriter(text,fieldnames=fields);w.writeheader();w.writerows(result)
        blob=text.getvalue().encode('utf-8')
        if path.exists() and path.read_bytes()!=blob:raise RuntimeError('确认bank改变')
        if not path.exists():path.write_bytes(blob)
        bank.append(dict(name=f'order_{seed}',path=str(path),sha256=digest(path),role='代码冻结后的条件顺序确认；非新分布'))
    save_json(root/'BANK.json',bank)
    return bank

def gap_eval(task,root,config):
    directory=gap_setup(task,root,'eval');import pickle
    started=time.perf_counter();points=[];bank=config['bank']
    for marker in sorted((directory/'checkpoints').glob('step_*.json')):
        cp=read_json(marker);step=int(cp['step']);dest=directory/'evaluation'/f'step_{step:09d}.json'
        existing=read_json(dest) if config['resume'] else None
        if existing:points.append(existing);continue
        model=FullPPO.load(marker.parent/Path(cp['model']).name,device=config['eval_device'])
        model.policy.set_training_mode(False);frozen=policy_sha(model.policy)
        with (marker.parent/Path(cp['vecnorm']).name).open('rb') as f:norm=pickle.load(f)
        norm.training=False;norm.norm_reward=False
        if norm.observation_space.shape!=(GapSchema(task['k']).dim,):raise RuntimeError('模型归一化维数不匹配')
        late=step>=.8*ATTR_STEPS
        sources=bank if late else bank[:1];episodes=[]
        for trace in sources:
            for seed in ((123,124) if trace['name']=='original' and step==ATTR_STEPS else (123,)):
                cfg=dict(config,trace=trace['path'],_capture_lifecycle=step==ATTR_STEPS,_final_checkpoint=step==ATTR_STEPS)
                os.environ['UAM_ATTR_TRACE_SHA']=trace['sha256']
                m=full_replay(model,norm,root,task,cfg,seed,True,probe=False)
                m.update(trace=trace['name'],trace_sha256=trace['sha256'])
                episodes.append(m)
        if policy_sha(model.policy)!=frozen:raise AssertionError('评估改变权重')
        current=dict(task=task,step=step,episodes=episodes,weights_unchanged=True)
        save_json(dest,current);points.append(current)
        save_json(directory/'LIVE.json',dict(phase='eval',checkpoint=step,time=now(),pid=os.getpid()))
        print('[评估]',task['id'],step,'回放数',len(episodes),flush=True);del model,norm
    points.sort(key=lambda p:p['step'])
    if not points or points[-1]['step']!=ATTR_STEPS:raise RuntimeError('最终模型缺失，不报完成')
    def collapse(es):
        return dict(rpt=mean([e['restricted_person_time_per_generated'] for e in es]),
            completion=mean([e['completion'] for e in es]),full_episode_fraction=mean([float(e['full']) for e in es]),
            blocked_fraction=mean([float(e['terminal_resource']['restricted_executor_blocked']) for e in es]),
            full_att=mean([e['ATT'] for e in es]) if all(e['full'] for e in es) else None,
            episodes=len(es),note='未全部完成则全体ATT=null；RPT为600分钟固定边界诊断')
    confirm=lambda p:[e for e in p['episodes'] if e['trace']!='original']
    late=[p for p in points if p['step']>=.8*ATTR_STEPS]
    late_confirm=[e for p in late for e in confirm(p)]
    final=points[-1];curves=[dict(step=p['step'],**collapse([e for e in p['episodes'] if e['trace']=='original' and e['eval_seed']==123])) for p in points]
    save_csv(directory/'LEARNING_CURVE.csv',curves)
    original_late=[p for p in curves if p['step']>=.8*ATTR_STEPS];best=min(curves,key=lambda p:p['rpt'])
    xs=[p['step']/1e6 for p in original_late];ys=[p['rpt'] for p in original_late]
    mx,my=mean(xs),mean(ys);den=sum((x-mx)**2 for x in xs)
    stability=dict(best_rpt=best['rpt'],best_step=best['step'],best_completion=best['completion'],
        late_rpt=my,late_sd=sd(ys),late_slope_per_million=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den if den else None,
        best_to_late_ratio=(my-best['rpt'])/best['rpt'] if best['rpt'] else None,
        scope='仅同一原顺序eval123曲线；不与新确认bank混成一条曲线；Best非选模型规则')
    result=dict(task=task,main_late_confirmation=collapse(late_confirm),final_confirmation=collapse(confirm(final)),
        final_original=collapse([e for e in final['episodes'] if e['trace']=='original']),
        original_curve_stability=stability,
        eval_seconds=time.perf_counter()-started,checkpoint_scope='末20%固定窗口；不按Best选模型',
        new_training_transitions=0,bank_scope='同OD联合多重集和原时刻的四次重排；训练seed是独立单位')
    save_json(directory/'RESULT.json',result);save_json(directory/'EVAL_END.json',dict(step=ATTR_STEPS,time=now()))

def gap_rules(task,root,config):
    directory=gap_setup(task,root,'rules');results=[]
    for trace in config['bank']:
        for mode in ('AUTHOR_SPF','AUTHOR_STTF','AUTHOR_QTTI','READY_QUEUE_SINGLE'):
            cfg=dict(config,trace=trace['path'],_capture_lifecycle=False,_final_checkpoint=False)
            os.environ['UAM_ATTR_TRACE_SHA']=trace['sha256']
            m=full_replay(None,None,root,task,cfg,123,True,rule=mode,probe=False)
            results.append(dict(trace=trace['name'],rule=mode,metrics=m))
    save_json(directory/'RULES_END.json',dict(task=task,results=results,new_training_transitions=0))

def gap_aggregate(root,tasks):
    rows=[];status=[];initial=defaultdict(set)
    for t in tasks:
        d=root/'cells'/t['id'];tr=read_json(d/'TRAIN_END.json');r=read_json(d/'RESULT.json')
        fail=list(d.glob('FAILED_*.json'));live=read_json(d/'LIVE.json',{})
        status.append(dict(**t,trained=bool(tr),evaluated=bool(r),failed=bool(fail),steps=int(tr['steps']) if tr else int(live.get('steps',0))))
        if r:
            m=r['main_late_confirmation'];f=r['final_confirmation']
            rows.append(dict(**t,late_rpt=m['rpt'],late_completion=m['completion'],late_full_fraction=m['full_episode_fraction'],
                late_att=m['full_att'],late_blocked_fraction=m['blocked_fraction'],final_rpt=f['rpt'],final_completion=f['completion'],final_att=f['full_att'],
                parameters=read_json(d/'MODEL_MANIFEST.json',{}).get('parameters'),original_late_sd=r['original_curve_stability']['late_sd'],
                original_best_to_late=r['original_curve_stability']['best_to_late_ratio']))
        ini=read_json(d/'INITIAL_POLICY_SHA.json')
        if ini:initial[(t['k'],t['seed'],t['control'],t['network'])].add(ini['sha256'])
    if any(len(v)>1 for v in initial.values()):raise AssertionError('同网络同seed因子臂初始权重不一致')
    by={(r['k'],r['seed'],r['control'],r['arm']):r for r in rows};paired=[]
    for k,seed,control in sorted({(r['k'],r['seed'],r['control']) for r in rows}):
        for label,a,b in [('TIME_ONLY','L10_TIME','L00'),('BATCH_ONLY','L01_BATCH','L00'),
            ('BOTH','L11_BOTH','L00'),('TIME_WHEN_BATCH','L11_BOTH','L01_BATCH'),
            ('BATCH_WHEN_TIME','L11_BOTH','L10_TIME'),('GENERIC_NETWORK','T11_BOTH','L11_BOTH')]:
            x,y=by.get((k,seed,control,a)),by.get((k,seed,control,b))
            if x and y:paired.append(dict(k=k,seed=seed,control=control,contrast=label,delta_rpt=x['late_rpt']-y['late_rpt'],delta_completion=x['late_completion']-y['late_completion']))
        names=['L00','L10_TIME','L01_BATCH','L11_BOTH']
        if all((k,seed,control,a) in by for a in names):
            r=[by[k,seed,control,a] for a in names]
            paired.append(dict(k=k,seed=seed,control=control,contrast='TIME_X_BATCH',
                delta_rpt=r[3]['late_rpt']-r[1]['late_rpt']-r[2]['late_rpt']+r[0]['late_rpt'],delta_completion=None))
    measured,upper=budget_account(root)
    p=dict(total=48,trained=sum(x['trained'] for x in status),evaluated=sum(x['evaluated'] for x in status),
        failed=sum(x['failed'] for x in status),completed_training_transitions=sum(x['steps'] for x in status if x['trained']),
        measured_training_work=measured,interrupted_unlogged_upper_bound=upper,
        planned_actual_budget=ATTR_PLANNED,budget_cap=ATTR_CAP,time=now())
    save_csv(root/'RUN_STATUS.csv',status);save_csv(root/'MODEL_RESULTS.csv',rows);save_csv(root/'PAIRED_EFFECTS.csv',paired)
    save_json(root/'progress.json',p);return p

def gap_pack_models(root,tasks):
    """额外保存最终模型和配对归一化，避免只有DATAONLY而无法再次回放。"""
    archive=root.parent.parent/'experiment_archives'/(root.name+'_FINAL_REPLAY_MODELS.tar.gz')
    temporary=archive.with_name(archive.name+'.tmp')
    if archive.is_symlink() or archive.resolve().parent!=archive.parent.resolve():raise RuntimeError('回放包路径重定向')
    if archive.exists():
        with tarfile.open(archive,'r:gz') as prior:
            contract=json.load(prior.extractfile(root.name+'/SUITE_PLAN.json'))
            if contract.get('script_sha256')!=digest(SCRIPT):raise RuntimeError('同名回放包合同不同，不覆盖')
        return str(archive)
    archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(temporary,'w:gz') as tar:
        tar.add(SCRIPT,arcname=root.name+'/ENTRYPOINT.py',recursive=False)
        for f in [root/'SUITE_PLAN.json',root/'BANK.json',root/'TASK_REGISTRY.json',root/'END.json']+list((root/'bank').glob('*.csv')):
            if f.is_file():tar.add(f,arcname=root.name+'/'+f.relative_to(root).as_posix(),recursive=False)
        for t in tasks:
            d=root/'cells'/t['id'];cp=d/'checkpoints'/f'step_{ATTR_STEPS:09d}.json'
            if not cp.exists():raise RuntimeError('最终模型指针缺失，不报可回放包完成')
            meta=read_json(cp)
            for f in [cp,d/'TASK.json',d/'MODEL_MANIFEST.json',d/'TRAIN_END.json',
                      cp.parent/Path(meta['model']).name,cp.parent/Path(meta['vecnorm']).name]:
                if not f.is_file() or f.is_symlink() or not f.resolve().is_relative_to(root.resolve()):
                    raise RuntimeError('最终回放模型缺失或路径重定向')
                tar.add(f,arcname=root.name+'/'+f.relative_to(root).as_posix(),recursive=False)
    os.replace(temporary,archive)
    save_json(root/'FINAL_REPLAY_PACKAGE.json',dict(path=str(archive),sha256=digest(archive),models=len(tasks),time=now()))
    print('[最终模型与归一化包]',archive,flush=True);return str(archive)

def gap_resource_check(root,device,min_free_gib):
    """只拒绝启动自己的新格，不终止别人；用户仍须先查看top/nvidia-smi。"""
    attr_resource_check(root,device,min_free_gib)
    if os.name!='nt' and hasattr(os,'getloadavg'):
        cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
        if os.getloadavg()[0]>.85*cpus:raise RuntimeError('CPU当前负载过高，不启动自己的新训练')
    if device.startswith('cuda'):
        query=subprocess.run(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
            '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        if query.returncode:raise RuntimeError('无法核验GPU，停止自己的新格')
        free,util=[float(x) for x in query.stdout.strip().split(',')]
        if free<4096 or util>=60:raise RuntimeError('GPU可用不足4GiB或当前利用率>=60%；先观察其他用户占用，再恢复')

def gap_main():
    ap=argparse.ArgumentParser(description='48.169M固定缺口归因；50M总上限；无世界模型/多智能体新算法')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--retry-failed',action='store_true')
    ap.add_argument('--device',default='cuda');ap.add_argument('--eval-device',default='cpu')
    ap.add_argument('--phase',choices=('all','train','eval','rules','pack'),default='all')
    ap.add_argument('--timeout-hours',type=float,default=4.);ap.add_argument('--min-free-gib',type=float,default=10.)
    ap.add_argument('--plan',action='store_true');ap.add_argument('--root');ap.add_argument('--worker',choices=('train','eval','rules'));ap.add_argument('--task')
    args=ap.parse_args();tasks=gap_registry()
    if args.plan:
        print(json.dumps(dict(version=GAP_VERSION,tasks=tasks,planned_actual_budget=ATTR_PLANNED,budget_cap=ATTR_CAP,reserve=ATTR_CAP-ATTR_PLANNED),ensure_ascii=False,indent=2));return
    project=Path(__file__).resolve().parent;root=Path(args.root).resolve() if args.root else project/'serial_runs'/ATTR_ROOT
    if root.name!=ATTR_ROOT or not root.is_relative_to(project/'serial_runs') or root.is_symlink():raise RuntimeError('只能使用本项目新私有实验目录')
    if args.worker:
        record=read_json(args.task);task,config=record['task'],record['config']
        os.environ['UAM_ATTR_TRACE_SHA']=digest(config['trace'])
        dict(train=attr_train,eval=gap_eval,rules=gap_rules)[args.worker](task,root,config);return
    root.mkdir(parents=True,exist_ok=True);lock=(root/'PARENT_LOCK').open('a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0)
        if lock.read(1)==b'':lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read_json(root/'SUITE_PLAN.json')
    if old and old['script_sha256']!=digest(SCRIPT):raise RuntimeError('代码合同改变，不混入旧实验')
    if old and not args.resume and args.phase!='pack':raise RuntimeError('已有结果；请--resume，不清空')
    if not old and (root/'cells').exists() and any((root/'cells').iterdir()):raise RuntimeError('已有cell缺合同，保留待查')
    native=materialize_source(root);bank=gap_bank(root,native)
    config=dict(version=GAP_VERSION,device=args.device,eval_device=args.eval_device,
        trace=str(native/'train_data/passengers_300.csv'),bank=bank,train_horizon=500,eval_horizon=600,
        actual_steps=ATTR_STEPS,requested_steps=1_000_000,resume=args.resume,retry_failed=args.retry_failed,timeout_hours=args.timeout_hours)
    if old:
        for key in ('device','eval_device','trace','bank','train_horizon','eval_horizon','actual_steps'):
            if old['config'][key]!=config[key]:raise RuntimeError('恢复合同变化：'+key)
    else:
        save_json(root/'SUITE_PLAN.json',dict(version=GAP_VERSION,script_sha256=digest(SCRIPT),
            frozen_runtime_sha256=_FROZEN_ATTR_SHA,diagnostic_sha256=GAP_DIAG_SHA,config=config,tasks=tasks,
            ppo=SOURCE_PPO,planned_actual_budget=ATTR_PLANNED,budget_cap=ATTR_CAP,
            main='末20%检查点×四条新条件顺序，先模型内平均，再按独立训练seed配对',
            comparisons='LSTM时间×批余数2×2；同信息普通Transformer；不靠更换物理造增益'))
    save_json(root/'TASK_REGISTRY.json',tasks)
    print('[预算]',ATTR_PLANNED,'/50,000,000；48格；预留',ATTR_CAP-ATTR_PLANNED,flush=True)
    try:
        if args.phase=='pack':attr_pack(root);return
        if args.phase in ('all','rules'):
            for k in (3,2,4):
                t=dict(id=f'RULES__E1__K{k}',environment='E1',k=k,fleet=12,seed=123,control='LQ',
                    method='SINGLE__M0_BASE',variant='M0_BASE',arm='L00',reward_mode='PT_POST',time_mode='RAW',timing=False,batch=False,network='LSTM')
                attr_launch('rules',t,root,config)
        for i,t in enumerate(tasks,1):
            if args.phase in ('all','train') and not (root/'cells'/t['id']/'TRAIN_END.json').exists():
                gap_resource_check(root,args.device,args.min_free_gib)
                print(f'[训练 {i}/48]',t['id'],flush=True);attr_launch('train',t,root,config)
            if args.phase in ('all','eval') and (root/'cells'/t['id']/'TRAIN_END.json').exists():attr_launch('eval',t,root,config)
            gap_aggregate(root,tasks)
        p=gap_aggregate(root,tasks)
        rules_ok=all((root/'cells'/f'RULES__E1__K{k}'/'RULES_END.json').exists() for k in (2,3,4))
        save_json(root/'END.json',dict(status='COMPLETE' if p['evaluated']==48 and p['failed']==0 and rules_ok else 'PARTIAL',progress=p,time=now(),
            note='K3每臂4训练seed；K2/K4只有单seed边界，不声称全环境规律'))
        if read_json(root/'END.json')['status']=='COMPLETE':gap_pack_models(root,tasks)
        attr_pack(root);print('[结束]',p,flush=True)
    except KeyboardInterrupt:
        gap_aggregate(root,tasks);save_json(root/'INTERRUPTED.json',dict(time=now(),note='同代码--resume；保留全部已完成格'));raise SystemExit(130)
    finally:
        status=read_json(root/'END.json',{}).get('status','PARTIAL')
        save_json(root/'RUNNING.json',dict(phase='complete' if status=='COMPLETE' else 'idle_partial',pid=None,task=None,time=now()))
        lock.close()

if __name__=='__main__':gap_main()
