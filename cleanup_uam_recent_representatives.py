#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只清理白名单旧训练格的已识别中间模型；先清单，后逐文件确认删除。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import zipfile

ROOT=Path('/home/SY/uam/UAGMC-main')
RUNS=(
 'uam_UAGMC_minfix_240m_v1','uam_E01_S_AC_198m_v1',
 'uam_E1_reward_charge_28run_30m_v1','uam_gap_factors_48run_50m_v1',
 'uam_credit_horizon_32run_18h_v1','uam_stage_stin_24run_60m_v1',
 'uam_stin_E01_completion_16run_v1')
STEP=re.compile(r'step_(\d+)\.(zip|pkl)$')
RNG=re.compile(r'rng_(\d+)\.pt$')
ENTRYPOINTS=('train_uam_UAGMC_minfix_240m_ONEFILE.py','train_uam_E01_S_AC_198m_ONEFILE.py',
 'train_uam_E1_reward_charge_attribution_30m_ONEFILE.py','train_uam_gap_factors_50m_ONEFILE.py',
 'train_uam_credit_horizon_18h_ONEFILE.py','train_uam_stage_credit_18h_ONEFILE.py','train_uam_stin_E01_completion_ONEFILE.py')


def read(p):
    if not p.is_file(): return {}
    safe(p)
    if p.stat().st_size>(64 if p.name=='PLAN.json' else 8)*1024**2: raise RuntimeError('JSON异常过大：'+str(p))
    return json.loads(p.read_text(encoding='utf-8-sig'))


def safe(p):
    if not p.is_absolute() or not p.is_relative_to(ROOT): raise RuntimeError('路径越界')
    current=ROOT
    for part in ('.',)+p.relative_to(ROOT).parts:
        current=current/part
        s=current.lstat()
        if stat.S_ISLNK(s.st_mode) or s.st_dev!=ROOT.stat().st_dev: raise RuntimeError('软链接或跨设备：'+str(current))
    s=p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid() or s.st_nlink!=1:
        raise RuntimeError('属主、硬链接或文件类型待查：'+str(p))
    return dict(size=s.st_size,inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def idle():
    # 只检查自己账号的进程；不读取环境变量，不终止任何进程。
    if not Path('/proc').is_dir(): raise RuntimeError('仅允许在指定Linux服务器执行')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid(): continue
        try:
            if proc.stat().st_uid!=os.getuid(): continue
            command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except FileNotFoundError: continue
        except PermissionError: raise RuntimeError('不能核验同账号进程：'+proc.name)
        if any(name in command for name in RUNS+ENTRYPOINTS) and 'python' in command.lower():
            raise RuntimeError('目标旧实验仍有进程引用，停止：'+proc.name)


def dump(p,value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def audit():
    idle(); targets=[]; retained=[]; skipped=[]; counts={}
    for run in RUNS:
        directory=ROOT/'serial_runs'/run
        if not directory.is_dir() or directory.is_symlink(): continue
        progress=read(directory/'progress.json')
        n=progress.get('total'); finished=progress.get('trained',progress.get('completed'))
        if not isinstance(n,int) or n<=0 or finished!=n:
            skipped.append(dict(path=str(directory),reason='没有全轮训练完成证据')); continue
        counts[run]=0
        for cell in sorted((directory/'cells').iterdir()):
            if not cell.is_dir() or cell.is_symlink(): continue
            if not (cell/'checkpoints').is_dir(): continue
            try:
                task=read(cell/'TASK.json')
                if task.get('id')!=cell.name: raise RuntimeError('任务归属说明不匹配')
                if not read(cell/'TRAIN_END.json'): raise RuntimeError('缺少训练结束记录')
                if not (read(cell/'EVAL_END.json') or read(cell/'STAGE_EVAL_END.json')):
                    raise RuntimeError('缺少评估结束记录')
                groups={}
                for sub in ('checkpoints','resume'):
                    folder=cell/sub
                    if not folder.is_dir(): continue
                    if folder.is_symlink(): raise RuntimeError('模型目录是链接')
                    for p in folder.iterdir():
                        match=STEP.fullmatch(p.name) or RNG.fullmatch(p.name)
                        if match:
                            safe(p); groups.setdefault(int(match[1]),[]).append(p)
                complete={s:files for s,files in groups.items() if any(p.suffix=='.zip' for p in files) and any(p.suffix=='.pkl' for p in files)}
                if not complete: raise RuntimeError('没有完整模型与归一化配套')
                latest=max(complete)
                if latest!=max(groups): raise RuntimeError('最大步数模型配套不完整')
                middle=min(complete,key=lambda s:(abs(s-latest*.5),s))
                keep_steps={latest,middle}
                for name in ('CELL_SUMMARY.json','RESULT.json'):
                    record=read(cell/name)
                    for key in ('best_step','best_checkpoint'):
                        value=record.get(key)
                        if type(value) is int and value in complete: keep_steps.add(value)
                # 不加载pickle或神经网络；保留模型必须通过ZIP完整性及内部步数核验。
                preserve=[]
                for step in keep_steps:
                    for p in complete[step]:
                        if p.suffix=='.zip':
                            with zipfile.ZipFile(p) as z:
                                if z.getinfo('data').file_size>32*1024**2 or z.testzip() is not None: raise RuntimeError('保留模型ZIP损坏')
                                if json.loads(z.read('data')).get('num_timesteps')!=step: raise RuntimeError('模型内部步数不符')
                        preserve.append(dict(path=str(p.relative_to(ROOT)),meta=safe(p),sha256=sha(p)))
                # 未配套的步数保留待查；只删除完整组内的白名单文件。
                remove=[p for step,files in complete.items() if step not in keep_steps for p in files]
                for p in remove:
                    targets.append(dict(path=str(p.relative_to(ROOT)),meta=safe(p),cell=str(cell.relative_to(ROOT))))
                retained.extend(preserve); counts[run]+=len(remove)
            except Exception as error:
                skipped.append(dict(path=str(cell),reason=str(error)))
    out=ROOT/'cleanup_audits'/('representatives_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    if (ROOT/'cleanup_audits').is_symlink(): raise RuntimeError('审计目录是链接，停止')
    out.mkdir(parents=True,exist_ok=False)
    plan=dict(root=str(ROOT),script_sha256=sha(Path(__file__).resolve()),targets=targets,retained=retained,skipped=skipped,
        candidate_count=len(targets),candidate_logical_GiB=sum(x['meta']['size'] for x in targets)/1024**3,run_counts=counts,
        rule='每格保留最终、约中期、元数据明确记录的最佳；其他路径和不完整模型组不删')
    dump(out/'PLAN.json',plan)
    for filename,rows in (('拟删除逐文件清单.csv',targets),('保留逐文件清单.csv',retained)):
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f); w.writerow(('项目内相对路径','字节数'))
            w.writerows((r['path'],r['meta']['size']) for r in rows)
    dump(out/'SUMMARY.json',{k:v for k,v in plan.items() if k not in ('targets','retained')})
    print(json.dumps(dict(plan=str(out/'PLAN.json'),files=len(targets),logical_GiB=plan['candidate_logical_GiB'],skipped_cells=len(skipped)),ensure_ascii=False,indent=2))
    print('请先查看同目录拟删除/保留CSV；确认后以 --apply PLAN.json 执行。')


def apply(path):
    safe(path)
    if not path.is_relative_to(ROOT/'cleanup_audits'): raise RuntimeError('清单目录不符')
    plan=read(path)
    if plan.get('root')!=str(ROOT) or plan.get('script_sha256')!=sha(Path(__file__).resolve()): raise RuntimeError('项目或代码版本不符')
    idle()
    for r in plan['retained']:
        p=ROOT/r['path']
        if safe(p)!=r['meta'] or sha(p)!=r['sha256']: raise RuntimeError('保留件发生变化：'+str(p))
    pending=[]
    for r in plan['targets']:
        p=ROOT/r['path']
        parts=p.relative_to(ROOT).parts
        if len(parts)!=6 or parts[0]!='serial_runs' or parts[1] not in RUNS or parts[2]!='cells' or parts[4] not in ('resume','checkpoints'):
            raise RuntimeError('目标不在允许的逐格模型目录')
        if not (STEP.fullmatch(p.name) or RNG.fullmatch(p.name)): raise RuntimeError('非白名单文件')
        if p.exists():
            if safe(p)!=r['meta']: raise RuntimeError('目标发生变化：'+str(p))
            pending.append(r)
    print('逐文件清单：',path.parent/'拟删除逐文件清单.csv')
    print('本次剩余：',len(pending),'份，逻辑GiB：',round(sum(r['meta']['size'] for r in pending)/1024**3,3))
    print('删除的中间模型不可从DATAONLY恢复；结果、最终/代表模型、归档、环境均不删。')
    answer='DELETE '+str(len(pending))
    if input('确认请原样输入 '+answer+'：')!=answer: print('取消，未删除'); return
    deleted=0; total=0
    with (path.parent/'DELETE_VERIFIED.jsonl').open('a',encoding='utf-8') as log:
        for r in pending:
            if deleted%100==0: idle()
            p=ROOT/r['path']
            if safe(p)!=r['meta']: raise RuntimeError('删除前目标变化，停止')
            # 仅逐个解除本清单中的普通文件；绝不递归删除目录。
            p.unlink(); deleted+=1; total+=r['meta']['size']
            log.write(json.dumps(dict(path=r['path'],bytes=r['meta']['size'],time=time.time()),ensure_ascii=False)+'\n'); log.flush()
            if deleted%1000==0: print('[已删除]',deleted,'/',len(pending),flush=True)
    dump(path.parent/'DELETE_END.json',dict(status='COMPLETE',deleted_this_run=deleted,logical_GiB=total/1024**3,time=time.time()))
    print('完成：本次删除',deleted,'份中间模型，逻辑GiB',round(total/1024**3,3))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--apply',type=Path); args=parser.parse_args()
    if Path.cwd()!=ROOT or ROOT.is_symlink() or (ROOT/'serial_runs').is_symlink(): raise RuntimeError('请进入指定项目真实目录，不允许链接路径')
    if args.apply: apply(args.apply.absolute())
    else: audit()


if __name__=='__main__': main()
