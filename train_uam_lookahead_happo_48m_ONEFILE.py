#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TMLR Lookahead 从论文复实现：原版、1/2/4 轮及两个定向消融。"""
import argparse
import copy
import hashlib
import json
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

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
RUN='uam_lookahead_happo_k3_24run_48m_v1'
OLD='uam_happo_env5_44run_88m_v2'
OLD_SHA='f5ba14c61a52658708b2205d3e511120351e7d2a65053448c828831e87f73df7'
ENTRY_SHA='27fbe3d361b190be970fce37eb65ef98d159e4fbc52c6952299ef55c9322baa7'
HARL='b1af98b0dbab72a2eee9d160751cd09aedbb8ce2'
CASE='E1_K3_UNIFORM'
SEEDS=(2101,2102,2103,2104)
STEPS=1_996_800
ROLLOUT=10_240
CAP=48_000_000
SHADOW_H=20
SHADOW_N=128
WARM=2
RING=8
SAMPLES=2048
HALT=False

# 元组是虚拟更新轮数、真实环境用不用影子、价值估计是否用 V-trace。
ARMS={
    'L0_HAPPO':(0,False,False),
    'L1_FULL':(1,True,True),
    'L2_SHADOW_2':(2,True,True),
    'L3_SHADOW_4':(4,True,True),
    'L4_DISCARD_SHADOW':(1,False,True),
    'L5_NO_VTRACE':(1,True,False),
}


def frozen(project,archive):
    import ast
    if not archive.is_file() or archive.is_symlink():
        raise FileNotFoundError(archive)
    with archive.open('rb') as f:
        digest=hashlib.sha256()
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
        if digest.hexdigest()!=OLD_SHA:
            raise ValueError('需要经核对的 88M DATAONLY 归档')
    with tarfile.open(archive) as tar:
        with tar.extractfile(OLD+'/ENTRYPOINT.py') as f: raw=f.read()
        with tar.extractfile(OLD+'/PLAN.json') as f: plan=json.load(f)
        with tar.extractfile(OLD+'/BANK.json') as f: bank=json.load(f)
        cfg={}
        # HARL 配置不随工况变化，用旧时变四格核对原参数。
        for s in SEEDS:
            with tar.extractfile(f'{OLD}/cells/E1_K3_VARYING__HAPPO_RANDOM__s{s}/CONFIG.json') as f:
                cfg[s]=json.load(f)['args']
    if (hashlib.sha256(raw).hexdigest()!=ENTRY_SHA or plan['entrypoint_sha256']!=ENTRY_SHA
            or plan['author_harl_commit']!=HARL):
        raise ValueError('历史入口或作者版本不符')
    tree=ast.parse(raw)
    tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and '__name__' in ast.unparse(n.test))]
    old=types.ModuleType('uam_lookahead_frozen_env5')
    old.__file__=__file__
    sys.modules[old.__name__]=old
    exec(compile(tree,'<verified_88m_environment>','exec'),vars(old))
    old.PROJECT,old.RUN_NAME,old.CAP=project,RUN,CAP
    return old,plan,bank,cfg


def make_plan(old,plan):
    tasks=[]
    for s in SEEDS:
        # K3 均匀原 88M 只训练了 UAGMC-L；取原 J3 动作与奖励合同，换回原均匀需求。
        anchor=next(x for x in plan['tasks'] if x['id']==f'E1_K3_VARYING__HAPPO_RANDOM__s{s}')
        if anchor['steps']!=STEPS or anchor['control']!='J3' or anchor['fleet']!=12:
            raise ValueError('原 HAPPO 合同不符')
        arms=list(ARMS)
        random.Random(14112027+s).shuffle(arms)
        for a in arms:
            depth,use_shadow,vt=ARMS[a]
            tasks.append(dict(anchor,id=f'{CASE}__{a}__s{s}',case=CASE,arrival='UNIFORM',arm=a,
                              shadow_rounds=depth,shadow_behavior=use_shadow,vtrace=vt,run_order=len(tasks)+1))
    assert len(tasks)==24 and sum(t['steps'] for t in tasks)==47_923_200
    return dict(revision=RUN,seed_block_order_key=14112027,case=CASE,tasks=tasks,source_sha256=ENTRY_SHA,
                entrypoint_sha256=old.sha(__file__),author_harl_commit=HARL,online_budget=47_923_200,
                budget_cap=CAP,rollout=ROLLOUT,shadow_horizon=SHADOW_H,shadow_trajectories=SHADOW_N,
                shadow_minibatches=10,shadow_entropy=0.001,
                warmup_rounds=WARM,replay_per_round=SAMPLES,replay_blocks=RING,
                vtrace=dict(rho=1,c=1),model_steps_per_round=48,model_batch=256,
                policy='正文每轮刷新影子；深度 2/4 在本轮连续更新；不读未来需求',
                arms={a:list(v) for a,v in ARMS.items()},old_archive_sha256=OLD_SHA,
                reporting='确认集固定；真实步、虚拟步、拟合更新分别计；不宣称原论文精确同结果')


def stop(*_):
    global HALT
    HALT=True


def verify_vendor(path):
    def read(*a):
        return subprocess.check_output(['git','-C',str(path),*a],text=True).strip()
    if read('rev-parse','HEAD')!=HARL or read('status','--porcelain','--untracked-files=all'):
        raise ValueError('HARL 必须是指定且干净的提交；不会改作者目录')


def vtrace(reward,value,bootstrap,continues,discount,logp,enabled=True):
    """价值校正用联合比率；返回不含角色比率的 Q−V，供独立 PPO 权重使用。"""
    import numpy as np
    if not all(np.isfinite(x).all() for x in (reward,value,bootstrap,logp)):
        raise ValueError('V-trace 输入包含非有限数值')
    ratio=np.exp(np.clip(logp,-40,40))
    c=np.minimum(1.,ratio) if enabled else np.ones_like(ratio)
    corrected=np.empty_like(value)
    delta=reward+discount*bootstrap-value
    carry=np.zeros_like(value[0])
    for i in range(len(value)-1,-1,-1):
        carry=c[i]*delta[i]+discount[i]*continues[i]*c[i]*carry
        corrected[i]=value[i]+carry
    # 切开 episode 后的 reset 样本；最后一个 rollout 步使用真实 next value。
    nxt=bootstrap.copy()
    nxt[:-1]=np.where(continues[:-1],corrected[1:],bootstrap[:-1])
    advantage=reward+discount*nxt-value
    if not np.isfinite(corrected).all() or not np.isfinite(advantage).all():
        raise ValueError('V-trace 目标非有限')
    return corrected,advantage,ratio


def ratio_parts(new,anchor,own_beh,prefix):
    """PPO 裁剪只看新/旧正式策略，旧正式/行为是外部权重。"""
    return new-anchor,anchor-own_beh+prefix


def numeric_checks():
    # 不需要环境、PyTorch 或 GPU 的客观真值：边界、比率及条件联合概率。
    import numpy as np
    r=np.array([[[1.]],[[2.]],[[4.]]])
    v=np.zeros_like(r); d=np.full_like(r,.5)
    cont=np.array([[[1]],[[0]],[[1]]],dtype=bool); d[1]=0
    unit=np.zeros_like(r)
    a,b,_=vtrace(r,v,v,cont,d,unit)
    assert np.allclose(a.ravel(),[2,2,4]) and np.allclose(b.ravel(),[2,2,4])
    a,_,_=vtrace(r,v,v,cont,d,np.log(np.full_like(r,.5)))
    assert np.allclose(a.ravel(),[.75,1,2])
    a,_,_=vtrace(r,v,v,cont,d,np.log(np.full_like(r,.5)),False)
    assert np.allclose(a.ravel(),[2,2,4])
    sen=[.7,.3]; other=[.55,.45]; post=[[.8,.2],[.4,.6]]
    assert abs(sum(sen[i]*post[i][j]*(other[i]/sen[i]) for i in range(2) for j in range(2))-1)<1e-12
    a,b=ratio_parts(np.log(.31),np.log(.27),np.log(.22),np.log(.63/.53))
    assert abs(np.exp(a+b)-(.31/.22)*(.63/.53))<1e-12
    # 论文任务的解析最优及最佳回应，不冒称训练复现论文 Figure 3。
    f=lambda a,b:a**3-2*(a-b)**2+3*a
    assert f(1,1)==4 and f(.4,.4)>f(.4,.65)


def policy_logits(actor,obs,zeros,masks,actions):
    """通过作者接口拿分布；不重写 actor 网络。"""
    _,ent,dist=actor.evaluate_actions(obs,zeros,actions,masks,None,None)
    if not hasattr(dist,'logits'):
        raise ValueError('本轮必须是 K3 单离散 Categorical')
    return dist.logits,ent


def offpolicy_update(actors,critic,cb,vn,buffs,adv,anchor,order,np,torch):
    """本项目的新顺序更新层，不修改 vendor；保留作者 actor、critic、优化器。"""
    advantage=(adv-adv.mean())/(adv.std()+1e-5)
    flat=advantage.reshape(-1,1)
    prefix=np.zeros_like(flat)
    report=[]
    for i in order:
        actor,b=actors[i],buffs[i]
        n=flat.shape[0]
        obs=b.obs[:-1].reshape(n,-1)
        zs=b.rnn_states[:-1].reshape(n,*b.rnn_states.shape[2:])
        act=b.actions.reshape(n,1).astype(np.int64)
        masks=b.masks[:-1].reshape(n,1)
        mu=b.action_log_probs.reshape(n,1)
        pi0=anchor[i].reshape(n,1)
        sums=[]
        # 前序权重当前角色训练期间固定；不把联合 V-trace 比率重复乘到目标。
        w=np.exp(np.clip(pi0-mu+prefix,-10,10)).astype(np.float32)
        width=n//actor.actor_num_mini_batch
        if n%actor.actor_num_mini_batch:
            raise ValueError('正式 minibatch 不整除')
        for _ in range(actor.ppo_epoch):
            for ix in torch.randperm(n).numpy().reshape(-1,width):
                raw,ent,_=actor.evaluate_actions(obs[ix],zs[ix],act[ix],masks[ix],None,None)
                target=(torch.as_tensor(flat[ix],**actor.tpdv))
                ratio=(raw-torch.as_tensor(pi0[ix],**actor.tpdv)).clamp(-20,20).exp()
                weight=torch.as_tensor(w[ix],**actor.tpdv)
                term=torch.minimum(ratio*target,ratio.clamp(1-actor.clip_param,1+actor.clip_param)*target)
                loss=-(weight*term).mean()
                actor.actor_optimizer.zero_grad(set_to_none=True)
                (loss-actor.entropy_coef*ent).backward()
                grad=torch.nn.utils.clip_grad_norm_(actor.actor.parameters(),actor.max_grad_norm,
                                                    error_if_nonfinite=True)
                actor.actor_optimizer.step()
                sums.append([loss.item(),ent.item(),grad.item()])
        with torch.no_grad():
            new=[]
            for j in range(0,n,2048):
                z=slice(j,j+2048)
                lp,_,_=actor.evaluate_actions(obs[z],zs[z],act[z],masks[z],None,None)
                new.append(lp.cpu().numpy())
        prefix+=np.concatenate(new)-mu
        report.append(dict(role=i,losses=np.mean(sums,axis=0),mean_weight=float(w.mean()),
                           max_weight=float(w.max())))
    cm=critic.train(cb,vn)
    return report,cm


def wm_model(dim,torch):
    from torch import nn
    class Dynamics(nn.Module):
        """共享当前状态代理；单独预测乘客选站后观测与整分钟转移、奖励。"""
        def __init__(self):
            super().__init__()
            def net(ni,no):
                return nn.Sequential(nn.Linear(ni,192),nn.SiLU(),nn.Linear(192,192),
                                     nn.SiLU(),nn.Linear(192,no))
            core=dim-8
            self.post=net(dim+3,2*core+3)
            self.next=net(dim+6,2*(core+1))

        def post_parameters(self,x,p):
            return self.post(torch.cat((x,nn.functional.one_hot(p,3).float()),-1))

        def next_parameters(self,x,p,a):
            return self.next(torch.cat((x,nn.functional.one_hot(p,3).float(),
                                        nn.functional.one_hot(a,3).float()),-1))
    return Dynamics()


def gaussian(params,torch):
    mean,log_std=params.chunk(2,dim=-1)
    return mean,log_std.clamp(-5,1)


def fit_world(model,opt,ring,norm,torch,np,rng,device):
    data={k:np.concatenate([block[k] for block in ring])
          for k in ('x','post','next','p','executed_p','a','r','done','age')}
    # 不把自动 reset 引入学习目标；留出真实样本由固定下标切出。
    ids=np.arange(len(data['r']))
    train=ids[ids%8!=0]; valid=ids[ids%8==0]
    if not len(train) or not len(valid):
        raise ValueError('真实回放池不足')
    pack=norm.observation_space.shape[0]-8
    reward_unit=100.
    def tx(x,dtype=torch.float32):
        return torch.as_tensor(x,dtype=dtype,device=device)
    def err(index):
        x=tx(norm.normalize_obs(data['x'][index]))
        before=tx(norm.normalize_obs(data['post'][index])[:,:pack])
        after=tx(norm.normalize_obs(data['next'][index])[:,:pack])
        p,a=[tx(data[k][index],torch.long) for k in ('p','a')]
        reward=tx(data['r'][index]/reward_unit).reshape(-1)
        mask=tx(~data['done'][index]).reshape(-1)
        post_parameters=model.post_parameters(x,p)
        u,lu=gaussian(post_parameters[:,:-3],torch)
        logits=post_parameters[:,-3:]
        v,lv=gaussian(model.next_parameters(x,p,a),torch)
        def nll(y,z,l):
            return .5*((y-z)*torch.exp(-l)).square()+l
        e_post=nll(before,u,lu).mean()
        e_next=(nll(after,v[:,:pack],lv[:,:pack]).mean(-1)*mask).sum()/mask.sum().clamp_min(1.)
        e_rew=nll(reward,v[:,-1],lv[:,-1]).mean()
        applied=tx(data['executed_p'][index],torch.long)
        e_applied=torch.nn.functional.cross_entropy(logits,applied)
        return e_post+e_next+e_rew+e_applied,(before,u,after,v,mask,reward,x,logits,applied)
    model.train()
    for _ in range(48):
        ix=rng.choice(train,256,replace=len(train)<256)
        loss,_=err(ix)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        opt.step()
    model.eval()
    with torch.no_grad():
        loss,(post,post_hat,nxt,y,mask,r,x,logits,applied)=err(valid[:1024])
        report=dict(nll=float(loss),post_rmse=float((post-post_hat).square().mean().sqrt()),
                    next_rmse=float((((nxt-y[:,:pack]).square().mean(-1)*mask).sum()
                                    /mask.sum().clamp_min(1)).sqrt()),
                    persistence_rmse=float((((nxt-x[:,:pack]).square().mean(-1)*mask).sum()
                                    /mask.sum().clamp_min(1)).sqrt()),
                    executed_passenger_accuracy=float((logits.argmax(-1)==applied).float().mean()),
                    reward_mae=float((r-y[:,-1]).abs().mean()*reward_unit),n=len(valid[:1024]))
    return data,report


def shadow_team(real,vc,vn):
    # deepcopy 整个训练对象以保持参数与各自优化器引用；任何虚拟更新都不写回。
    return copy.deepcopy((real,vc,vn))


def value_probe(vn):
    return {k:getattr(vn,k).detach().cpu() for k in ('running_mean','running_mean_sq','debiasing_term')}


def runner_for(args,actors,critic,an,cb,vn):
    return types.SimpleNamespace(algo_args={'train':args},actor=actors,critic=critic,actor_buffer=an,
                                 critic_buffer=cb,value_normalizer=vn,state_type='EP',
                                 fixed_order=False,num_agents=2,action_aggregation='prod')


def imagine(team,args,space,action_space,model,norm,dataset,depth,author,Buffer,CB,np,torch,rng,device):
    actors,critic,vn=team
    shape=dict(args,episode_length=SHADOW_H,n_rollout_threads=SHADOW_N)
    for a in actors:
        a.actor_num_mini_batch=10
        a.entropy_coef=.001
    critic.critic_num_mini_batch=10
    bs=[Buffer(shape,space,action_space) for _ in actors]; cb=CB(shape,space)
    runner=runner_for(shape,actors,critic,bs,cb,vn)
    zero=np.zeros((SHADOW_N,args['recurrent_n'],args['hidden_sizes'][-1]),np.float32)
    ones=np.ones((SHADOW_N,1),np.float32)
    d=space.shape[0]; core=d-8
    starts=np.arange(len(dataset['r']))
    starts=starts[(starts%8!=0)&(dataset['age']<480)&(~dataset['done'])]
    if not len(starts):
        raise RuntimeError('没有合法的虚拟起点')
    def array(t): return t.detach().cpu().numpy()
    def sample(param):
        mean,log_std=gaussian(param,torch)
        return mean+torch.randn_like(mean)*log_std.exp()
    raw_tail=np.zeros((4,d),np.float32)
    raw_tail[1:,-8]=1.
    raw_tail[1:,-7:-4]=np.eye(3)
    tails=norm.normalize_obs(raw_tail)[:,-8:]
    reports=[]
    for _ in range(depth):
        obs=norm.normalize_obs(dataset['x'][rng.choice(starts,SHADOW_N,replace=True)])
        for a in actors: a.prep_rollout()
        critic.prep_rollout()
        reward_avg=[]
        with torch.no_grad():
            for k in range(SHADOW_H):
                values,_=critic.get_values(obs,zero,ones)
                p,lp,_=actors[0].get_actions(obs,zero,ones)
                x=torch.as_tensor(obs,device=device,dtype=torch.float32); pi=p.long().flatten()
                pp=model.post_parameters(x,pi)
                y=sample(pp[:,:-3]).clamp(-10.,10.)
                # 无待选乘客时真实环境把请求动作映射为 0；在影子内用学习到的执行类别。
                executed=torch.distributions.Categorical(logits=pp[:,-3:]).sample()
                mid=np.concatenate((array(y),tails[1+array(executed)]),axis=-1)
                a,la,_=actors[1].get_actions(mid,zero,ones)
                out=sample(model.next_parameters(x,pi,a.long().flatten()))
                nxt=np.concatenate((array(out[:,:core].clamp(-10.,10.)),np.repeat(tails[:1],SHADOW_N,axis=0)),axis=-1)
                r=array((out[:,-1:]*100.).clamp(-300.,0.))
                reward_avg.append(float(r.mean()))
                bs[0].obs[k]=obs; bs[1].obs[k]=mid; cb.share_obs[k]=obs
                for b,act,logp in zip(bs,(p,a),(lp,la)):
                    b.insert(nxt,zero,array(act),array(logp),ones,ones)
                cb.insert(nxt,zero,array(values),r,ones,ones)
                obs=nxt
            v,_=critic.get_values(obs,zero,ones)
            cb.compute_returns(array(v),vn)
        for a in actors: a.prep_training()
        critic.prep_training()
        am,cm=author(runner)
        reports.append(dict(mean_imagined_reward=float(np.mean(reward_avg)),actor=am,critic=cm))
    return reports


def probe_logits(actors,bs,torch,ix):
    out=[]
    with torch.no_grad():
        for a,b in zip(actors,bs):
            n=b.actions.shape[0]*b.actions.shape[1]
            z=b.rnn_states[:-1].reshape(n,*b.rnn_states.shape[2:])
            logits,_=policy_logits(a,b.obs[:-1].reshape(n,-1)[ix],z[ix],
                                   b.masks[:-1].reshape(n,1)[ix],b.actions.reshape(n,1)[ix])
            out.append(logits.detach().cpu().numpy())
    return out


def open_pointer(old,folder):
    ptr=old.read(folder/'RESUME.json',{})
    for k in ('model','norm','extra'):
        if k not in ptr:
            continue
        file=folder/ptr[k]
        if file.name!=ptr[k] or file.is_symlink() or old.sha(file)!=ptr[k+'_sha']:
            raise RuntimeError('恢复代校验失败：'+k)
    return ptr


def train(old,config,job):
    import cloudpickle
    import numpy as np, torch, gymnasium as gym
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    root=Path(job['root']); item=job['task']; folder=root/'cells'/item['id']; started=time.monotonic()
    base=old.base_module()
    cloudpickle.register_pickle_by_value(old)
    _,args,author=base.harl_prepare(dict(harl_dir=job['vendor_dir']))
    if old.plain(args)!=config[item['seed']] or args['episode_length']*args['n_rollout_threads']!=ROLLOUT:
        raise RuntimeError('HARL 参数未与旧合同完全对齐')
    if args['use_recurrent_policy'] or args['use_naive_recurrent_policy']:
        raise ValueError('此离线顺序修正只实现冻结的前馈 HAPPO')
    from harl.algorithms.actors.happo import HAPPO
    from harl.algorithms.critics.v_critic import VCritic
    from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer as Buffer
    from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP as CB
    from harl.common.valuenorm import ValueNorm
    m,native=base.runtime(root,item); Phase=base.gym_class(m)
    old.validate_job_bank(job,native)
    old.case_rules(base,m,root,item,job['bank'])
    seed=item['seed']; torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(job['memory_fraction'],0)
    def maker(rank):
        def constructor(): return Phase(root,item,job['trace'],500,seed*10+rank,False)
        return constructor
    vector=SubprocVecEnv([maker(i) for i in range(args['n_rollout_threads'])],start_method='spawn')
    try:
        ptr=open_pointer(old,folder)
        norm=VecNormalize.load(str(folder/ptr['norm']),vector) if ptr else VecNormalize(vector,norm_obs=True,norm_reward=False)
        return train_loop(old,base,m,job,folder,root,args,HAPPO,VCritic,ValueNorm,Buffer,CB,author,ptr,norm,
                          gym,np,torch,started)
    finally:
        vector.close()


def train_loop(old,base,m,job,folder,root,args,HAPPO,VCritic,ValueNorm,Buffer,CB,author,ptr,norm,
               gym,np,torch,started):
    item=job['task']; space=norm.observation_space; device=torch.device('cuda')
    action_space=gym.spaces.Discrete(3)
    actors=[HAPPO(args,space,action_space,device=device) for _ in range(2)]
    critic=VCritic(args,space,device=device); vn=ValueNorm(1,device=device)
    depth,shadow_beh,vt=ARMS[item['arm']]
    # 初始化模型不消耗正式策略的 RNG；L0 与其余各臂冷启动完全同口径。
    rs=old.rng_record()
    model=wm_model(space.shape[0],torch).to(device) if depth else None
    opt=torch.optim.Adam(model.parameters(),lr=3e-4) if model else None
    old.rng_restore(rs)
    ring=[]; rng=np.random.default_rng(61811+item['seed']); real=updates=imag=fit_steps=0
    if ptr:
        with (folder/ptr['model']).open('rb') as f: state=pickle.load(f)
        for i,a in enumerate(actors):
            a.actor.load_state_dict(state['actors'][i]); a.actor_optimizer.load_state_dict(state['optimizers'][i])
        critic.critic.load_state_dict(state['critic']); critic.critic_optimizer.load_state_dict(state['critic_optimizer'])
        vn.load_state_dict(state['value_norm'])
        for k,v in state['value_norm_statistics'].items(): getattr(vn,k).copy_(v.to(device))
        real,updates=state['real_steps'],state['updates']
        if depth:
            with (folder/ptr['extra']).open('rb') as f: e=pickle.load(f)
            model.load_state_dict(e['model']); opt.load_state_dict(e['optim']); ring=e['ring']
            rng.bit_generator.state=e['rng']; imag,fit_steps=e['imag'],e['fit_steps']
        old.rng_restore(state['rng'])
    initial=real; n=args['n_rollout_threads']; t=args['episode_length']
    bs=[Buffer(args,space,action_space) for _ in actors]; cb=CB(args,space)
    runner=runner_for(args,actors,critic,bs,cb,vn)
    z=np.zeros((n,args['recurrent_n'],args['hidden_sizes'][-1]),np.float32)
    ones=np.ones((n,1),np.float32); masks=ones.copy(); obs=norm.reset()
    age=np.zeros(n,np.int32)
    old.save(folder/'CONFIG.json',dict(args=args,method=item['arm'],author_commit=HARL,depth=depth,
             behavior='shadow' if shadow_beh else 'formal_old',vtrace=vt,shadow_steps_per_round=SHADOW_H*SHADOW_N,
             resume='恢复策略、所有优化器、模型回放池、随机状态、归一化；仿真从新 episode 开始。'))
    def array(tensor): return tensor.detach().cpu().numpy()
    def snapshot():
        # 两代交替写：中断时不会覆盖 RESUME 指向的上一代；里程碑硬链接不受轮换影响。
        tag=f'ROLL_{updates%2}'
        model_path=folder/(tag+'.pkl'); norm_path=folder/(tag+'.norm.pkl')
        tmp=norm_path.with_name(norm_path.name+'.pending')
        norm.save(str(tmp)); os.replace(tmp,norm_path)
        old.atomic_pickle(model_path,old.cpu_checkpoint(dict(real_steps=real,updates=updates,
           actors=[a.actor.state_dict() for a in actors],optimizers=[a.actor_optimizer.state_dict() for a in actors],
           critic=critic.critic.state_dict(),critic_optimizer=critic.critic_optimizer.state_dict(),
           value_norm=vn.state_dict(),value_norm_statistics=value_probe(vn),rng=old.rng_record())))
        meta=dict(real_steps=real,updates=updates,model=model_path.name,norm=norm_path.name,
                  model_sha=old.sha(model_path),norm_sha=old.sha(norm_path),target_real_steps=STEPS,
                  imagined_steps=imag,model_updates=fit_steps)
        if depth:
            extra=folder/(tag+'.extra.pkl')
            old.atomic_pickle(extra,old.cpu_checkpoint(dict(model=model.state_dict(),optim=opt.state_dict(),
                ring=ring,rng=rng.bit_generator.state,imag=imag,fit_steps=fit_steps)))
            meta.update(extra=extra.name,extra_sha=old.sha(extra))
        old.save(folder/'RESUME.json',meta)
        return meta
    def factory(_):
        def predict(x,carry):
            role=int(round(float(x[-8])))
            with torch.no_grad():
                a,_,_=actors[role].get_actions(norm.normalize_obs(np.asarray(x)[None]),z[:1],ones[:1])
            return int(array(a)[0,0]),None
        return predict
    def capture(meta):
        for mark in job['marks']:
            out=folder/'eval'/f'{mark:09d}.json'
            if real<mark or out.is_file():
                continue
            if real!=mark:
                raise RuntimeError('旧评估点缺失，不用更晚模型冒充')
            checkpoint={k:v for k,v in meta.items() if k not in ('extra','extra_sha')}
            for role,suffix in [('model','.pkl'),('norm','.norm.pkl')]:
                dst=folder/(f'R_{real:09d}'+suffix)
                if not dst.exists(): os.link(folder/meta[role],dst)
                if dst.is_symlink() or old.sha(dst)!=meta[role+'_sha']:
                    raise RuntimeError('里程碑权重与当前代不一致')
                checkpoint[role]=dst.name
            old.save(folder/'checkpoints'/f'{real:09d}.json',checkpoint)
            for a in actors: a.prep_rollout()
            was=norm.training; norm.training=False
            try:
                old.evaluate(base,m,root,item,job['bank'],factory,real,final=real==STEPS)
            finally:
                norm.training=was

    if ptr: capture(ptr)
    while real<STEPS and not old.stopping(root):
        if shutil.disk_usage(root).free<job['min_free_gib']*1024**3:
            old.save(root/'DISK_STOP.json',dict(real_steps=real,task=item['id'],free=shutil.disk_usage(root).free))
            old.save(root/'STOP_REQUESTED',dict(reason='磁盘余量不足，自动停止'))
            break
        for a in actors: a.prep_rollout(); a.lr_decay(real,STEPS)
        critic.prep_rollout(); critic.lr_decay(real,STEPS)
        active=bool(depth and updates>=WARM)
        shadow=None; audit={}
        if active:
            old.save(folder/'LIVE.json',dict(phase='fit_and_imagine',real_steps=real,updates=updates,
                                             imagined_steps=imag,model_updates=fit_steps))
            rs=old.rng_record()
            dataset,report=fit_world(model,opt,ring,norm,torch,np,rng,device)
            shadow=shadow_team(actors,critic,vn)
            audit.update(model=report,shadow=imagine(shadow,args,space,action_space,model,norm,dataset,depth,
                                                      author,Buffer,CB,np,torch,rng,device))
            old.rng_restore(rs)
            imag+=depth*SHADOW_H*SHADOW_N; fit_steps+=48
        behavior=shadow[0] if active and shadow_beh else actors
        for a in behavior: a.prep_rollout()
        for b in bs: b.step=0; b.factor=None
        cb.step=0
        anchor=np.zeros((2,t,n,1),np.float32)
        ends=np.full((t,n,1),np.nan,np.float32)
        chosen=set(rng.choice(t*n,SAMPLES,replace=False).tolist()) if depth else set()
        per_time={}
        for ix in sorted(chosen): per_time.setdefault(ix//n,[]).append(ix%n)
        block={name:[] for name in ('x','post','next','p','executed_p','a','r','done','age')}
        for k in range(t):
            ix=per_time.get(k)
            if ix is not None: x=norm.get_original_obs()[ix].copy()
            with torch.no_grad():
                vals,_=critic.get_values(obs,z,masks)
                p,lp,_=behavior[0].get_actions(obs,z,masks)
                if behavior is not actors:
                    anchor[0,k]=array(actors[0].evaluate_actions(obs,z,array(p),masks)[0])
                else: anchor[0,k]=array(lp)
            mid,nothing,done,_=norm.step(array(p).ravel())
            if np.any(nothing) or np.any(done): raise RuntimeError('选站准备推进了物理分钟')
            if ix is not None: post=norm.get_original_obs()[ix].copy()
            with torch.no_grad():
                a,la,_=behavior[1].get_actions(mid,z,masks)
                if behavior is not actors:
                    anchor[1,k]=array(actors[1].evaluate_actions(mid,z,array(a),masks)[0])
                else: anchor[1,k]=array(la)
            nxt,r,done,infos=norm.step(array(a).ravel())
            if any(info.get('physical_increment')!=1 for info in infos): raise RuntimeError('真实步计数不符')
            real+=n
            nm=(~done).astype(np.float32)[:,None]
            bad=np.asarray([[0. if info.get('TimeLimit.truncated',False) else 1.] for info in infos],np.float32)
            if active:
                for j,info in enumerate(infos):
                    if done[j] and not bad[j,0]:
                        terminal=np.asarray(info['terminal_observation'])[None]
                        with torch.no_grad(): terminal_v,_=critic.get_values(terminal,z[:1],ones[:1])
                        ends[k,j]=vn.denormalize(array(terminal_v))[0]
            bs[0].obs[k]=obs; bs[1].obs[k]=mid; cb.share_obs[k]=obs
            bs[0].masks[k]=masks; bs[1].masks[k]=masks
            for b,act,log in zip(bs,(p,a),(lp,la)): b.insert(nxt,z,array(act),array(log),nm,ones)
            cb.insert(nxt,z,array(vals),r[:,None],nm,bad)
            if ix is not None:
                ps=np.argmax(post[:,-7:-4],axis=1)
                for key,val in zip(block,(x,post,norm.get_original_obs()[ix],array(p).ravel()[ix],ps,array(a).ravel()[ix],
                                           r[ix],done[ix],age[ix])):
                    block[key].append(val.copy())
            age=np.where(done,0,age+1)
            obs=nxt; masks=nm
            if (k+1)%256==0:
                elapsed=max(1,time.monotonic()-started)
                old.save(folder/'LIVE.json',dict(phase='collect_real',real_steps=real,updates=updates,seconds=elapsed,
                         real_fps=(real-initial)/elapsed,imagined_steps=imag,model_updates=fit_steps))
        with torch.no_grad(): v_next,_=critic.get_values(obs,z,masks)
        if depth:
            ring=(ring+[{key:np.concatenate(vals) for key,vals in block.items()}])[-RING:]
        ix=np.linspace(0,n*t-1,256,dtype=int)
        if active:
            old_logits=probe_logits(actors,bs,torch,ix); shadow_logits=probe_logits(shadow[0],bs,torch,ix)
        for a in actors: a.prep_training()
        critic.prep_training()
        if active:
            cb.value_preds[-1]=array(v_next)
            vals=vn.denormalize(cb.value_preds)
            cont=cb.masks[1:].astype(bool)
            discount=args['gamma']*(cont|(~cb.bad_masks[1:].astype(bool)))
            bootstrap=np.where(np.isfinite(ends),ends,vals[1:])
            gain=(anchor-np.stack([b.action_log_probs for b in bs])).sum(axis=0)
            cb.returns[:-1],adv,ratio=vtrace(cb.rewards,vals[:-1],bootstrap,cont,discount,gain,enabled=vt)
            order=torch.randperm(2).tolist()
            am,cm=offpolicy_update(actors,critic,cb,vn,bs,adv,anchor,order,np,torch)
            new=probe_logits(actors,bs,torch,ix)
            def kl(p,q): return float(np.mean(np.sum(np.exp(p)*(p-q),axis=-1)))
            audit.update(actor_order=order,kl_new_vs_old=[kl(a,b) for a,b in zip(new,old_logits)],
                 kl_new_vs_shadow=[kl(a,b) for a,b in zip(new,shadow_logits)],
                 kl_gap=[kl(a,b)-kl(a,c) for a,b,c in zip(new,old_logits,shadow_logits)],
                 joint_rho_mean=float(ratio.mean()),joint_rho_above_one=float((ratio>1).mean()),
                 joint_rho_relative_ess=float(ratio.mean()**2/np.square(ratio.astype(np.float64)).mean()))
        else:
            cb.compute_returns(array(v_next),vn); am,cm=author(runner)
        updates+=1
        audit.update(real_steps=real,updates=updates,imagined_steps=imag,model_updates=fit_steps,actor=am,critic=cm)
        with (folder/'UPDATE_AUDIT.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(old.plain(audit),ensure_ascii=False,allow_nan=False)+'\n')
        meta=snapshot()
        elapsed=max(1,time.monotonic()-started)
        old.save(folder/'LIVE.json',dict(phase='trained_rollout',real_steps=real,updates=updates,seconds=elapsed,
                 real_fps=(real-initial)/elapsed,imagined_steps=imag,model_updates=fit_steps))
        capture(meta)
        for b in bs: b.after_update()
        cb.after_update()
    if real==STEPS:
        final=old.read(folder/'eval'/f'{STEPS:09d}.json')
        if final is None: raise RuntimeError('最终确认未完成')
        old.save(folder/'EVALUATION.json',dict(final=final,real_steps=real,weights_unchanged=True))
    old.save(folder/'END.json',dict(status='COMPLETE' if real==STEPS else 'PARTIAL',real_steps=real,
                  seconds=time.monotonic()-started,updates=updates,imagined_steps=imag,model_updates=fit_steps))


def report(old,root,tasks):
    models=[]
    for item in tasks:
        folder=root/'cells'/item['id']
        if not old.complete_cell(folder): continue
        late=[]; best=[]
        for p in sorted((folder/'eval').glob('*.json')):
            rows=old.read(p)
            values=[x['metrics']['fixed_cutoff_cost'] for x in rows if x['trace'].startswith('order_43')]
            if len(values)!=8 or any(x is None for x in values):
                raise ValueError('开发验证覆盖不全：'+str(p))
            score=statistics.mean(values)
            mark=int(p.stem)
            best.append((score,mark))
            if mark>=.8*STEPS: late.append(score)
        end=old.read(folder/'EVALUATION.json')['final']
        final=[x['metrics'] for x in end if x['trace'].startswith('order_83')]
        if len(final)!=12 or not late: raise ValueError('晚期或固定确认覆盖不全')
        m=lambda field: statistics.mean(x[field] for x in final)
        cost,best_mark=min(best)
        models.append(dict(id=item['id'],arm=item['arm'],seed=item['seed'],late_cost=statistics.mean(late),
            best_dev_cost=cost,best_mark=best_mark,final_cost=m('fixed_cutoff_cost'),
            final_completion=m('completion'),final_att=m('full_att') if all(x['full_att'] is not None for x in final) else None,
            final_wait=statistics.mean(x['phases'].get('wait',0)/x['generated'] for x in final),
            retention_cost=late[-1]-cost,imagined_steps=old.read(folder/'END.json')['imagined_steps']))
    contrasts=[]
    for arm,other in [('L1_FULL','L0_HAPPO'),('L1_FULL','L2_SHADOW_2'),('L1_FULL','L3_SHADOW_4'),
                      ('L1_FULL','L4_DISCARD_SHADOW'),('L1_FULL','L5_NO_VTRACE')]:
        idx={(x['arm'],x['seed']):x for x in models}
        paired=[]
        for s in SEEDS:
            if (arm,s) not in idx or (other,s) not in idx: continue
            a,b=idx[arm,s],idx[other,s]
            paired.append(dict(seed=s,**{k:a[k]-b[k] for k in
                           ('late_cost','final_cost','final_completion','final_wait','retention_cost')}))
        deltas=[x['final_cost'] for x in paired]
        contrasts.append(dict(left=arm,right=other,n=len(paired),paired=paired,
                   mean_final_delta=statistics.mean(deltas) if deltas else None,
                   sd_final_delta=statistics.stdev(deltas) if len(deltas)>1 else None,
                   improved=sum(x<0 for x in deltas)))
    result=dict(models=models,contrasts=contrasts,sign='成本负差表示左组更好；训练 seed 是独立单位')
    old.save(root/'MODEL_LEVEL_SUMMARY.json',result)
    return result


def pack(old,project,root,result):
    out=project/'experiment_archives'
    if out.is_symlink(): raise ValueError('归档路径为链接')
    out.mkdir(exist_ok=True)
    index={}; files=set()
    for row in result['models']:
        folder=root/'cells'/row['id']; checks=folder/'checkpoints'
        mids=sorted(int(p.stem) for p in checks.glob('*.json'))
        mid=min(mids,key=lambda k:(abs(k-STEPS/2),k))
        index[row['id']]=dict(best=row['best_mark'],mid=mid,last=STEPS)
        files.update(folder/x for x in ('TASK.json','CONFIG.json'))
        for mark in set(index[row['id']].values()):
            meta_file=checks/f'{mark:09d}.json'
            meta=old.read(meta_file)
            files.add(meta_file)
            for role in ('model','norm'):
                p=folder/meta[role]
                if p.name!=meta[role] or p.is_symlink() or old.sha(p)!=meta[role+'_sha']:
                    raise RuntimeError('回放归档校验失败：'+str(p))
                files.add(p)
    old.save(root/'REPLAY_SELECTION.json',dict(selection=index,policy='开发集成本最优 / 约50% / 最终；同权重只存一次'))
    for label in ('DATAONLY','BEST_LAST_MID_REPLAY_MODELS'):
        dest=out/f'{RUN}_{label}.tar.gz'; temp=dest.with_name(dest.name+f'.{os.getpid()}.pending')
        if dest.is_symlink(): raise RuntimeError('归档目标不能为链接')
        with temp.open('xb') as s, tarfile.open(mode='w:gz',fileobj=s,compresslevel=1) as archive:
            def add(p):
                if p.is_symlink() or any(x.is_symlink() for x in p.parents if x!=root):
                    raise RuntimeError('拒绝归档链接：'+str(p))
                archive.add(p,arcname=RUN+'/'+p.relative_to(root).as_posix(),recursive=False)
            archive.add(__file__,arcname=RUN+'/ENTRYPOINT.py',recursive=False)
            for p in sorted(root.glob('*.json')):
                if not p.name.endswith('_PACKAGE.json'): add(p)
            for p in sorted((root/'bank'/'UNIFORM').glob('*.csv')): add(p)
            if label=='DATAONLY':
                for area in ('rules','budget_attempts','cells'):
                    for p in sorted((root/area).rglob('*')):
                        if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.log') and not p.name.startswith('RESOURCE_WAIT'):
                            add(p)
            else:
                for p in sorted(files): add(p)
        with tarfile.open(temp) as archive:
            names=archive.getnames()
        if RUN+'/PLAN.json' not in names or len(names)!=len(set(names)):
            raise RuntimeError('归档缺少合同或存在重名')
        os.replace(temp,dest)
        sha=old.sha(dest)
        side=dest.with_name(dest.name+'.sha256')
        tmp=side.with_name(side.name+f'.{os.getpid()}.pending')
        with tmp.open('x',encoding='ascii') as f: f.write(f'{sha}  {dest.name}\n')
        os.replace(tmp,side)
        old.save(root/f'{label}_PACKAGE.json',dict(path=str(dest),sha256=sha,bytes=dest.stat().st_size))
        print('[归档]',dest,flush=True)


def worker(old,root,job_path,cfg):
    path=job_path.resolve(); job=old.read(path); cell=root/'cells'/job['task']['id']
    if path!=cell/'JOB.json' or job['root']!=str(root) or job_path.is_symlink():
        raise ValueError('worker任务路径不属于本组')
    try:
        with old.GpuLease(root,job['gpu_reserve_gib']):
            train(old,cfg,job)
        return 0
    except (InterruptedError,KeyboardInterrupt):
        old.save(cell/'END.json',dict(status='PARTIAL',real_steps=old.resume_real(cell),reason='主动停止'))
        return 0
    except Exception:
        error=traceback.format_exc()
        print(error,flush=True)
        old.save(cell/'END.json',dict(status='ENGINEERING_FAILED',traceback=error,
                 real_steps=max(old.resume_real(cell),old.read(cell/'LIVE.json',{}).get('real_steps',0))))
        return 1


def main():
    p=argparse.ArgumentParser(description='Lookahead-HAPPO：6臂×4训练种子，47,923,200真实步')
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--source-archive',type=Path)
    p.add_argument('--vendor-dir',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--plan',action='store_true')
    p.add_argument('--verify-math',action='store_true')
    p.add_argument('--pack-only',action='store_true')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    numeric_checks()
    if args.verify_math:
        print('[通过] V-trace、终止切分、联合似然与论文玩具最优点')
        return 0
    project=args.project.resolve()
    archive=(args.source_archive or project/'experiment_archives'/f'{OLD}_DATAONLY.tar.gz').absolute()
    vendor=(args.vendor_dir or project/'workspace/vendor/HARL').resolve()
    old,frozen_plan,old_bank,cfg=frozen(project,archive)
    plan=make_plan(old,frozen_plan)
    if args.plan:
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        return 0
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,old.stop_signal if args.worker else stop)
    root=project/'serial_runs'/RUN
    if root.is_symlink() or root.parent.is_symlink(): raise RuntimeError('实验路径不能是链接')
    if args.worker: return worker(old,root,args.worker,cfg)
    if not args.pack_only:
        verify_vendor(vendor)
        if not os.environ.get('CUDA_VISIBLE_DEVICES','').isdecimal():
            p.error('显式用 CUDA_VISIBLE_DEVICES 指定单卡')
    if args.pack_only and not (root/'PLAN.json').is_file(): raise FileNotFoundError(root/'PLAN.json')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'PARENT.lock').is_symlink(): raise ValueError('父锁不能是链接')
    with old.lock_file(root/'PARENT.lock'):
        current=old.read(root/'PLAN.json')
        if current is not None and (current!=plan or not(args.resume or args.pack_only)):
            raise RuntimeError('已有合同不同或未指定 --resume；拒绝覆盖')
        if current is None: old.save(root/'PLAN.json',plan)
        tasks=plan['tasks']; attempts=root/'budget_attempts'
        for fn in sorted(attempts.glob('*.json')):
            rec=old.read(fn)
            if not rec['closed']:
                if rec['pid'] is None or old.alive(rec['pid']):
                    raise RuntimeError('未确认旧 worker 退出：'+str(fn))
                old.finish_attempt(root,fn,-999)
        if args.pack_only:
            pack(old,project,root,report(old,root,tasks))
            return 0
        ds=old.demand_make(root)['UNIFORM']
        ref=old_bank['UNIFORM']
        if ds['training_sha256']!=ref['training_sha256'] or [x['sha256'] for x in ds['bank']]!=[x['sha256'] for x in ref['bank']]:
            raise RuntimeError('训练或评估需求银行发生变化')
        print(f'[合同] 六臂×四种子，{plan["online_budget"]:,}真实步；计数不包含模型虚拟步',flush=True)
        for item in tasks:
            if HALT or old.stopping(root) or shutil.disk_usage(root).free<12*1024**3: break
            folder=root/'cells'/item['id']
            if old.complete_cell(folder): continue
            if old.read(folder/'END.json',{}).get('status')=='ENGINEERING_FAILED' and not args.retry_failed: continue
            progress=old.aggregate(root,tasks)
            future=sum(STEPS-old.resume_real(root/'cells'/x['id']) for x in tasks if not old.complete_cell(root/'cells'/x['id']))
            if progress['budget_charge_upper_bound']+future+ROLLOUT>CAP:
                old.save(root/'BUDGET_STOP.json',dict(progress=progress,remaining=future,loss_bound=ROLLOUT))
                break
            for path in (root/'cells',folder,attempts):
                if path.is_symlink(): raise RuntimeError('拒绝写入链接目录：'+str(path))
            old.save(folder/'TASK.json',item)
            old.save(folder/'JOB.json',dict(root=str(root),task=item,steps=STEPS,trace=ds['trace'],
                     trace_sha256=ds['training_sha256'],bank=ds['bank'],vendor_dir=str(vendor),
                     marks=old.eval_marks(STEPS,ROLLOUT),gpu_reserve_gib=3.,memory_fraction=.10,min_free_gib=12.))
            rec=dict(id=item['id'],pid=None,initial_real=old.resume_real(folder),loss_bound=ROLLOUT,
                     closed=False,observed_increment=0,unlogged_upper_bound=ROLLOUT)
            fn=attempts/f'{time.time_ns()}_{item["id"]}.json'; old.save(fn,rec)
            print(f'[训练 {item["run_order"]}/24] {item["id"]}',flush=True)
            with (folder/'worker.log').open('a',encoding='utf-8') as log:
                cmd=[sys.executable,'-B','-u',__file__,'--project',str(project),'--source-archive',str(archive),
                     '--worker',str(folder/'JOB.json')]
                try:
                    process=subprocess.Popen(cmd,cwd=project,stdout=log,stderr=subprocess.STDOUT)
                except OSError:
                    old.save(folder/'END.json',dict(status='ENGINEERING_FAILED',traceback=traceback.format_exc(),
                             real_steps=rec['initial_real']))
                    old.save(fn,dict(rec,closed=True,unlogged_upper_bound=0,returncode=-1))
                    continue
                old.save(fn,dict(rec,pid=process.pid))
                while process.poll() is None:
                    if HALT and not (root/'STOP_REQUESTED').exists():
                        old.save(root/'STOP_REQUESTED',dict(requested_at=time.time()))
                    old.save(root/'RUNNING.json',dict(phase='train_or_eval',task=item,pid=process.pid,time=time.time(),
                             worker_progress=old.read(folder/'LIVE.json',{}),
                             resource_wait=old.read(root/'RESOURCE_WAIT.json',{})))
                    time.sleep(4)
            old.finish_attempt(root,fn,process.returncode)
            old.aggregate(root,tasks)
            if process.returncode: print('[本格失败，后续格仍继续]',item['id'],flush=True)
        progress=old.aggregate(root,tasks); done=(progress['trained']==len(tasks))
        old.save(root/'RUNNING.json',dict(phase='complete' if done else 'partial',pid=None,task=None,time=time.time()))
        old.save(root/'END.json',dict(status='COMPLETE' if done else 'PARTIAL',progress=progress,time=time.time()))
        result=report(old,root,tasks)
        if result['models'] and shutil.disk_usage(root).free>=12*1024**3: pack(old,project,root,result)
        print(f'[结束] {progress["trained"]}/24完整；{progress["failed"]}工程失败',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
