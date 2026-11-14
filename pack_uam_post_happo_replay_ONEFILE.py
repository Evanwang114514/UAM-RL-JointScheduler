#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读选择训练快照；分批打包，不删除、不重训、不加载模型。"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import statistics
import tarfile
from pathlib import Path

RUNS = {
    'external': ('uam_happo_external_k3_28run_56m_v1', 24, 'train_uam_happo_external_k3_56m_ONEFILE.py'),
    'env': ('uam_happo_env5_44run_88m_v2', 44, 'train_uam_happo_env5_80m_v2_ONEFILE.py'),
    'a2po': ('uam_a2po_k3_20run_40m_v3', 20, 'train_uam_a2po_k3_v3_ONEFILE.py'),
    'ace': ('uam_ace_k3_16run_32m_v2', 16, 'train_uam_ace_k3_32m_v2_ONEFILE.py'),
    'bppo': ('uam_bppo_harl_matched_16run_20m_v1', 5, 'train_uam_bppo_20m_ONEFILE.py'),
}
OUT = 'uam_post_happo_replay_v1'
REFERENCE = 'uam_core_replay_12models_20261109_v1.tar.gz'
REFERENCE_SHA = '5ddb179c3dc692d42d99f615eaa9142c6ec2038dc22030a3e95c0b2eac1b5dc9'
# 已核对的本地 12 模型包内，四个 BPPO 终点的整对哈希。
BPPO_LOCAL = {
    '39442c9e10a30fa2ac7042812784d2613135fba3b6d79f95890a9934aa52e8e0': 'f5cea8ec22bbd8bff405ce207fb6acf2e8d66f9a01c59c0849ecbbab306f387a',
    '939828c4cd2c0af18b750b4a54913c1884ce87f5430e9ad1f4e4621862b41f58': '65920f6a434f37ea81f242e2df7d9e510b729d34f6abd9c3fe6bfb4f7775d831',
    '98c25bd6fd6e1590a0408685f433cd7061d84d9ded13f758f573ea9dc37bf959': 'b12a53d98a5a39cc9c9db126ed923b9af51b4dae0482184e4a2060adcafa1123',
    'a800d7efb05399a5ff3d2197e1adf2683dd76ffb304ad036b517ba2f4116a984': '1d9f649dbb21c4d410b519471bc6a8aae2100ea5dc1adf7110f581e43714a6b6',
}


def confined(p, parent):
    if not p.is_relative_to(parent) or p.resolve()!=p:
        raise ValueError(f'路径逃逸或重定向：{p}')
    return p


def strict_file(p, parent):
    confined(p,parent)
    if not p.is_file() or p.stat().st_size==0:
        raise FileNotFoundError(p)
    return p


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(p, parent):
    return json.loads(strict_file(p,parent).read_text(encoding='utf-8-sig'))


def step_number(v):
    if type(v) is not int or not 0<v<100_000_000:
        raise ValueError(f'非法训练步数：{v}')
    return v


def pointer_files(folder, meta):
    if not isinstance(meta,dict):
        raise ValueError('无配套 checkpoint 元数据')
    files=[]
    for key in ('model','norm','rng'):
        if key=='rng' and key not in meta:
            continue
        name=meta[key]
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',name) or name in ('.','..'):
            raise ValueError(f'非法文件名：{name}')
        f=strict_file(folder/name,folder)
        expected=meta[key+'_sha']
        if not re.fullmatch(r'[0-9a-f]{64}',expected):
            raise ValueError(f'无有效哈希：{f}')
        files.append((f,expected))
    return files


def dev_scores(folder, marks):
    # 只按 4301–4304 开发需求和两次策略回放选 Best，不看 830x。
    target={(f'order_{s}',p) for s in range(4301,4305) for p in (123,124)}
    scores={}
    for step in marks:
        path=folder/'eval'/f'{step:09d}.json'
        if not path.is_file():
            continue
        rows=[r for r in read(path,folder) if str(r.get('trace','')).startswith('order_43')]
        if (len(rows)!=8 or {(r.get('trace'),r.get('policy_seed')) for r in rows}!=target
                or any(r.get('checkpoint')!=step or r.get('environment_seed')!=123 for r in rows)):
            raise ValueError(f'开发评估覆盖不匹配：{path}')
        costs=[r['metrics']['fixed_cutoff_cost'] for r in rows]
        if any(not math.isfinite(x) for x in costs):
            raise ValueError(f'开发指标非有限：{path}')
        scores[step]=statistics.mean(costs)
    return scores


def collect(project, label, root, dedup):
    run, expected, entry=RUNS[label]
    plan=read(root/'PLAN.json',root)
    finished=read(root/'END.json',root)
    files, hashes, inline = {}, {}, {}
    manifest=dict(run=run, expected_completed=expected, run_status=finished['status'], rows=[],
                  selection='开发集最低固定截止累计人时 BEST；最近 50% 的已评估点 MID；实际完成末代 LAST。')

    def add(p,digest=None):
        strict_file(p,root)
        name=p.relative_to(root).as_posix()
        files[name]=p
        if digest:
            if name in hashes and hashes[name]!=digest:
                raise ValueError(f'同一文件有不同哈希：{p}')
            hashes[name]=digest
        return name

    for f in ('PLAN.json','BANK.json','END.json','progress.json','AUX_MANIFEST.json','BASELINE_CONTRACT.json'):
        if (root/f).exists():
            add(root/f)
    for p in sorted((root/'bank').rglob('*.csv')):
        add(p)
    if label=='external':
        aux=read(root/'AUX_MANIFEST.json',root)
        add(root/'FROZEN_AUX.pkl',aux['sha256'])
        manifest['required_baseline_data']='uam_happo_env5_44run_88m_v2_DATAONLY.tar.gz'

    # 只归档与 PLAN 哈希一致的实际训练入口，不把本机新版当成旧版。
    want=plan.get('entrypoint_sha256',plan.get('entry_sha256'))
    source=confined(project/entry,project)
    raw=source.read_bytes() if source.is_file() and source.stat().st_size<=2_000_000 else b''
    if raw and hashlib.sha256(raw).hexdigest()==want:
        inline['ENTRYPOINT.py']=raw
    else:
        archives=project/'experiment_archives'
        archive=confined(archives/f'{run}_DATAONLY.tar.gz',project)
        if label=='bppo' and not archive.is_file():
            archive=confined(archives/f'{run}_PARTIAL_DATAONLY.tar.gz',project)
        if archive.is_file():
            with tarfile.open(archive,'r:gz') as tar:
                try:
                    info=tar.getmember(run+'/ENTRYPOINT.py')
                except KeyError:
                    info=None
                if info and info.isfile() and info.size<=2_000_000:
                    raw=tar.extractfile(info).read()
                    if hashlib.sha256(raw).hexdigest()==want:
                        inline['ENTRYPOINT.py']=raw
    manifest['entrypoint']={'expected_sha256':want,'status':'INCLUDED' if inline else 'MISSING_EXACT_ENTRY'}

    completed=0
    for task in plan['tasks']:
        cell=task['id']
        if not re.fullmatch(r'[A-Za-z0-9_]+',cell):
            raise ValueError(f'非法实验格：{cell}')
        folder=confined(root/'cells'/cell,root)
        status=read(folder/'END.json',root) if (folder/'END.json').exists() else {}
        row=dict(cell=cell, status=status.get('status','PENDING'), selections=[])
        manifest['rows'].append(row)
        if row['status']!='COMPLETE':
            continue
        completed+=1
        for name in ('END.json','CONFIG.json','TASK.json','RESUME.json'):
            if (folder/name).is_file():
                add(folder/name)
        meta_files={int(p.stem):p for p in sorted((folder/'checkpoints').glob('*.json'))
                    if re.fullmatch(r'[0-9]{9}',p.stem)}
        # 即使模型文件已不存在，仍先按评估选真正 Best，不用次优冒充。
        eval_steps={int(p.stem) for p in (folder/'eval').glob('*.json') if re.fullmatch(r'[0-9]{9}',p.stem)}
        scores=dev_scores(folder,sorted(eval_steps))
        last=step_number(status.get('real_steps'))
        wanted={'LAST':last}
        if scores:
            if len(scores)!=8 or last not in scores:
                raise ValueError(f'{cell} 的完整开发曲线不是八点，不能可靠选 Best')
            wanted.update(BEST=min(scores,key=lambda x:(scores[x],x)),
                          MID=min(scores,key=lambda x:(abs(x-last/2),x)))
        else:
            row['note']='缺开发评估，无法指定 BEST/MID'
        for step in sorted(set(wanted.values())):
            rec=dict(step=step, roles=[k for k,v in wanted.items() if v==step], dev_cost=scores.get(step))
            row['selections'].append(rec)
            checkpoint=meta_files.get(step)
            pointer=folder/'RESUME.json' if step==last else checkpoint
            if step in scores:
                add(folder/'eval'/f'{step:09d}.json')
            # 如果最后一次恢复指针不可用，尝试相同步数的评估快照，绝不换成别的步数。
            choices=[p for p in dict.fromkeys((pointer,checkpoint if step==last else None)) if p]
            rec['status']='MISSING_OR_INVALID'
            for p in choices:
                try:
                    meta=read(p,folder)
                    if step_number(meta.get('real_steps'))!=step:
                        raise ValueError('记录步数不一致')
                    if (label=='bppo' and dedup and 'rng' not in meta
                            and BPPO_LOCAL.get(meta['model_sha'])==meta['norm_sha']):
                        rec.update(status='ALREADY_LOCAL',reference=REFERENCE,reference_sha256=REFERENCE_SHA,
                                   paths=[f'serial_runs/{run}/cells/{cell}/{meta[k]}' for k in ('model','norm')])
                    else:
                        parts=pointer_files(folder,meta)
                        rec.update(status='SELECTED',paths=[add(f,h) for f,h in parts])
                    rec['pointer']=add(p)
                    rec.pop('reason',None)
                    break
                except (OSError,ValueError,KeyError) as e:
                    rec['reason']=str(e)
            if not choices:
                rec['reason']='所选步数没有 checkpoint 指针'
    manifest['completed']=completed
    if completed!=expected:
        raise ValueError(f'{run} 完成数为 {completed}，与本次核对的 {expected} 不同；请先核对现场')
    manifest['raw_bytes']=sum(f.stat().st_size for f in files.values())+sum(map(len,inline.values()))
    return files,hashes,inline,manifest


def publish(out,label,files,hashes,inline,manifest):
    # 所有权重只作字节复制，不反序列化；缺件明确报告，原结果只读。
    for name,want in hashes.items():
        if sha(files[name])!=want:
            raise ValueError(f'原文件哈希不符，不发布 {label}：{name}')
    out.mkdir(parents=True,exist_ok=True)
    prefix=manifest['run']
    target=out/f'{label}_BEST_LAST_MID.tar.gz'
    side=out/f'{label}_BEST_LAST_MID.tar.gz.sha256'
    for p in (target,side):
        confined(p,out)
        if p.exists():
            raise FileExistsError(f'已存在，不覆盖：{p}')
    pend=out/f'.{target.name}.{os.getpid()}.pending'
    with pend.open('xb') as f, tarfile.open(fileobj=f,mode='w:gz',compresslevel=1) as t:
        payload=dict(inline,**{'REPLAY_SELECTION.json':json.dumps(manifest,ensure_ascii=False,indent=2).encode()})
        for name,raw in payload.items():
            info=tarfile.TarInfo(f'{prefix}/{name}')
            info.size=len(raw)
            t.addfile(info,io.BytesIO(raw))
        for name,path in sorted(files.items()):
            t.add(path,arcname=f'{prefix}/{name}',recursive=False)
    # 同路径原子发布，不覆盖已有结果。
    digest=sha(pend)
    os.link(pend,target)
    pend.unlink()
    with side.open('x',encoding='ascii') as f:
        f.write(f'{digest}  {target.name}\n')
    print(f'[结果包] {target} ({target.stat().st_size/1024**2:.1f} MiB)',flush=True)


def verify_local(folder):
    failures=0
    for label,(run,_,_) in RUNS.items():
        target=folder/f'{label}_BEST_LAST_MID.tar.gz'
        try:
            checksum=target.with_name(target.name+'.sha256').read_text(encoding='ascii').split()
            if len(checksum)!=2 or checksum[1]!=target.name or sha(target)!=checksum[0]:
                raise ValueError('归档 SHA-256 不符')
            with tarfile.open(target,'r:gz') as tar:
                members=set(tar.getnames())
                manifest=json.load(tar.extractfile(f'{run}/REPLAY_SELECTION.json'))
                hits=[j for row in manifest['rows'] for j in row['selections']]
                for j in hits:
                    if j['status']=='SELECTED' and any(f'{run}/{name}' not in members for name in j['paths']):
                        raise ValueError('所选成员不全')
                if any(j['status']=='ALREADY_LOCAL' for j in hits) and sha(folder.parent/REFERENCE)!=REFERENCE_SHA:
                    raise ValueError('去重依赖的旧 12 模型包校验失败')
                missing=sum(j['status']=='MISSING_OR_INVALID' for j in hits)
                unknown=sum('note' in row for row in manifest['rows'])
                print(f'[校验通过] {label}: {len(hits)} 个去重节点；缺件 {missing}，不可选 Best/Mid 的格子 {unknown}；入口 {manifest["entrypoint"]["status"]}')
        except (OSError,ValueError,KeyError,tarfile.TarError) as exc:
            print(f'[校验失败] {label}: {exc}')
            failures+=1
    return failures


def main():
    p=argparse.ArgumentParser(description='按实验分包备份 Best、Last 与约 50% 步数快照')
    p.add_argument('--project',type=Path,default=Path.cwd())
    p.add_argument('--run',choices=tuple(RUNS)+('all',),default='all')
    p.add_argument('--pack',action='store_true')
    p.add_argument('--max-gib',type=float,default=8)
    p.add_argument('--verify-local',type=Path,help='Windows 下载后校验各包，不读取原实验')
    p.add_argument('--dedup-bppo-core',action='store_true',help='仅当本地已保留已核对的旧 12 模型包时使用')
    a=p.parse_args()
    if a.verify_local:
        raise SystemExit(int(bool(verify_local(a.verify_local))))
    import fcntl
    if not math.isfinite(a.max_gib) or a.max_gib<=0:
        p.error('--max-gib 须为正数')
    project=a.project.resolve(strict=True)
    if not project.is_relative_to(Path.home().resolve()):
        raise ValueError('只能在自己 home 内的项目运行')
    out=confined(project/'experiment_archives'/OUT,project)
    selected=tuple(RUNS) if a.run=='all' else (a.run,)
    with contextlib.ExitStack() as stack:
        all_runs={}
        for label in selected:
            root=confined(project/'serial_runs'/RUNS[label][0],project)
            lock=confined(root/'PARENT.lock',root)
            with_lock=stack.enter_context(lock.open('rb'))
            try:
                fcntl.flock(with_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError as e:
                raise RuntimeError(f'实验还持有写锁，等退出再归档：{root.name}') from e
            all_runs[label]=collect(project,label,root,a.dedup_bppo_core)
        total=0
        for label,(_,_,_,m) in all_runs.items():
            total+=m['raw_bytes']
            rec=[i for row in m['rows'] for i in row['selections']]
            counts={k:sum(i['status']==k for i in rec) for k in ('SELECTED','ALREADY_LOCAL','MISSING_OR_INVALID')}
            missing=[f"{r['cell']}:{'/'.join(i['roles'])}@{i['step']}" for r in m['rows'] for i in r['selections']
                     if i['status']=='MISSING_OR_INVALID']
            unknown=[r['cell'] for r in m['rows'] if r.get('note')]
            print(f"{label}: {m['completed']} 已完成格，{counts}；原始文件 {m['raw_bytes']/1024**3:.3f} GiB；入口 {m['entrypoint']['status']}",flush=True)
            if missing:
                print('  缺件：',', '.join(missing),flush=True)
            if unknown:
                print('  无法选 BEST/MID：',', '.join(unknown),flush=True)
        print(f'原始文件总量 {total/1024**3:.3f} GiB（未压缩；实际下载以归档大小为准）',flush=True)
        if not a.pack:
            return
        reserve=total+max(total//20,64*1024**2)
        if reserve>a.max_gib*1024**3 or shutil.disk_usage(project).free<reserve+12*1024**3:
            raise RuntimeError('预估包超出限额，或打包后磁盘不足 12 GiB；未创建包')
        # 把超限/已存在挡在所有打包操作之前。
        for label in selected:
            for suffix in ('.tar.gz','.tar.gz.sha256'):
                dest=confined(out/f'{label}_BEST_LAST_MID{suffix}',out)
                if dest.exists():
                    raise FileExistsError(f'已存在，不覆盖：{dest}；请只选尚未打包的一组')
        for label,args in all_runs.items():
            publish(out,label,*args)


if __name__=='__main__':
    main()
