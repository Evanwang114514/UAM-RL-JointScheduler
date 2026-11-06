#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修复既有三线实验的两个工程故障；不是新矩阵，不改变科学合同。"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
import ast
import hashlib
import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
import sysconfig
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
REVISION = 'UAM_ENGINEERING_REPAIR_20261107_V1'
ENTRY = {
    'core': ('train_uam_happo_mappo_dreamer_140m_ONEFILE.py',
             'f59eff09873a0703e81ec0a9bfbe8e1511a2d65abb58500ea4289c8f858f12ac'),
    'dima': ('train_uam_dima_25m_ONEFILE.py',
             '5a7c4ec7ec6bfcdccdf8594316d6f82d7b8b9e82f290b1961e7f535670563ba0'),
}
SOURCE_ERROR = '导入了项目旧环境，而不是内嵌作者环境'
DIMA_RELATIVE = 'agent/world_models/world_model_env.py'
DIMA_OLD = "                info['av_action'][dead] = av_action"
DIMA_NEW = (
    '                # 重置与预测的掩码统一类型和设备；0/1值、形状及动作权限不变。\n'
    "                info['av_action'][dead] = av_action.to(info['av_action'])"
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            result.update(block)
    return result.hexdigest()


def checked(path, root, exists=True):
    """不接受越界、符号链接或通过中间目录重定向的实验文件。"""
    path, root = Path(path).absolute(), Path(root).absolute()
    relative = path.relative_to(root)
    cursor = root
    if cursor.is_symlink():
        raise RuntimeError('项目根目录不能是符号链接')
    for part in relative.parts:
        if part in ('.', '..'):
            raise RuntimeError('路径不允许回退')
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError('拒绝链接路径：' + str(cursor))
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('路径越过项目范围')
    if exists and not path.exists():
        raise RuntimeError('缺少文件：' + str(path))
    return path


def read(path, default=None):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError('拒绝读取链接：' + str(path))
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else default


def write_json(path, value):
    write_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


def write_bytes(path, value):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError('拒绝覆盖链接：' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending_' + str(os.getpid()))
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError('保留未完成写入，需人工核对：' + str(temporary))
    with temporary.open('xb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def backup_once(path, destination):
    """只备份小型元数据；不复制、修改或删除模型与归一化文件。"""
    if not Path(path).is_file():
        return
    if destination.is_symlink():
        raise RuntimeError('备份不能是链接')
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open('rb') as source, destination.open('xb') as target:
            shutil.copyfileobj(source, target)


def record_tool(repair):
    target = repair / 'REPAIR_TOOL.py'
    if target.exists() and digest(target) != digest(__file__):
        raise RuntimeError('同一修复版本的工具源码发生变化，保留旧记录，不静默覆盖')
    backup_once(Path(__file__), target)


def load_entry(project, suite):
    filename, wanted = ENTRY[suite]
    path = checked(Path(project) / filename, project)
    if digest(path) != wanted:
        raise RuntimeError('原矩阵入口已不是本次核验的版本；不绕过合同：' + str(path))
    spec = importlib.util.spec_from_file_location('uam_repair_reference_' + suite, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    root = checked(Path(project) / 'serial_runs' / ('uam_' + suite + '_author_20261106_v1'), project)
    expected = module.plan_make()
    expected['entrypoint_sha256'] = wanted
    if read(checked(root / 'PLAN.json', project)) != expected:
        raise RuntimeError('既有PLAN与原矩阵合同不同；拒绝续接新设计')
    return module, root, expected


def own_workers(entry, cell=None):
    """只检查本账号进程；不终止进程，也不读取其他用户的命令。"""
    if sys.platform != 'linux':
        raise RuntimeError('正式修复只在用户自己的Linux服务器执行')
    result = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            if folder.stat().st_uid != os.getuid():
                continue
            arguments = (folder / 'cmdline').read_bytes().decode('utf-8', errors='replace').split('\0')
            # 按独立参数识别原入口，避免把包含命令文本的bash启动器误判成训练。
            if str(entry) not in arguments:
                continue
            if cell and not any(str(cell) in value for value in arguments):
                continue
            result.append(int(folder.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise RuntimeError('无法核验自己的进程，不冒险并发修复：' + folder.name) from error
    return result


def dima_text_patch(text):
    if text.count(DIMA_OLD + '\n') != 1 or DIMA_NEW in text:
        raise RuntimeError('DIMA补丁定位不唯一或源码不是预期版本')
    result = text.replace(DIMA_OLD + '\n', DIMA_NEW + '\n', 1)
    ast.parse(result)
    return result


def patch_dima(project, module, root, vendor):
    """只改作者适配沙盒，备份放在原打包规则已排除的adapted_author目录内。"""
    adapted = checked(root / 'adapted_author' / module.SOURCE_PINS['dima'], project)
    target = checked(adapted / DIMA_RELATIVE, project)
    manifest_path = checked(adapted / 'UAM_ADAPTATION.json', project)
    repair = checked(root / 'engineering_repair' / REVISION, project, exists=False)
    record_tool(repair)
    record_path = repair / 'DIMA_PATCH.json'
    record = read(record_path)
    manifest = read(manifest_path)
    if record:
        if digest(target) != record['after_sha256']:
            raise RuntimeError('已登记的DIMA修复与现有源码不符，不覆盖')
        for relative, wanted in manifest['adapted_hashes'].items():
            if digest(checked(adapted / relative, project)) != wanted:
                raise RuntimeError('适配源码校验不符：' + relative)
        print('[DIMA补丁] 已完成且校验通过，不重复修改', flush=True)
        return record
    # 作者原件、发布入口及已有适配的每个哈希都必须先通过。
    vendor = checked(vendor, project)
    for relative, wanted in module.SOURCE_HASHES['dima'].items():
        if digest(checked(vendor / relative, project)) != wanted:
            raise RuntimeError('锁定的DIMA作者原件不符：' + relative)
    for relative, wanted in manifest['adapted_hashes'].items():
        if digest(checked(adapted / relative, project)) != wanted:
            raise RuntimeError('已有适配沙盒被改动，不静默接着改：' + relative)
    if DIMA_RELATIVE not in module.SOURCE_HASHES['dima']:
        raise RuntimeError('原作者清单未覆盖需要修复的文件')
    if digest(target) != module.SOURCE_HASHES['dima'][DIMA_RELATIVE]:
        raise RuntimeError('目标文件不再是未改动的作者world_model_env.py')
    before = target.read_bytes()
    after = dima_text_patch(before.decode('utf-8-sig')).encode('utf-8')
    backups = checked(root / 'adapted_author' / '_engineering_backups' / REVISION, project, exists=False)
    backup_once(target, backups / 'world_model_env.py')
    backup_once(manifest_path, backups / 'UAM_ADAPTATION.before.json')
    backup_once(root / 'SOURCE_ADAPTATION.json', backups / 'SOURCE_ADAPTATION.before.json')
    updated = json.loads(json.dumps(manifest))
    updated['adapted_hashes'][DIMA_RELATIVE] = hashlib.sha256(after).hexdigest()
    item = dict(revision=REVISION, file=DIMA_RELATIVE,
                before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=hashlib.sha256(after).hexdigest(),
                before=DIMA_OLD.strip(), after=DIMA_NEW,
                meaning='只统一0/1可用动作掩码的dtype/device，不改动作集合、损失或更新次数')
    updated.setdefault('engineering_patches', []).append(item)
    updated['purpose'] = manifest.get('purpose', '') + '；另记录可用动作掩码dtype工程修复'
    write_json(repair / 'DIMA_PATCH_PREPARED.json', dict(**item, state='PREPARED', time=time.time()))
    write_bytes(target, after)
    write_json(manifest_path, updated)
    write_json(root / 'SOURCE_ADAPTATION.json', updated)
    # 若上述原子写入之间进程中断，PREPARED与原件备份都保留，不假装完整成功。
    write_json(record_path, dict(**item, state='APPLIED', time=time.time(),
        entrypoint_unchanged=True, plan_unchanged=True, new_training_transitions=0))
    print('[DIMA补丁] dtype修复及适配哈希已登记；原作者仓库、PLAN和模型均未改', flush=True)
    return read(record_path)


def pointer_check(folder, metadata):
    steps = int(metadata['real_steps'])
    if int(metadata['decision_records']) != steps * 2:
        raise RuntimeError('物理步数与两阶段记录数不符')
    for name, key in (('model', 'model_sha'), ('norm', 'norm_sha')):
        filename = metadata[name]
        if Path(filename).name != filename:
            raise RuntimeError('恢复件文件名越界')
        file = checked(folder / filename, folder)
        if digest(file) != metadata[key]:
            raise RuntimeError('模型/归一化与元数据校验不符：' + filename)
    return metadata


def select_checkpoints(folder, job):
    """各预注册时点只取第一次达到阈值的存档，不按成绩挑模型。"""
    resume = pointer_check(folder, read(checked(folder / 'RESUME.json', folder)))
    target = int(job['steps'])
    if not target <= int(resume['real_steps']) < target + 10240:
        raise RuntimeError('未满训或超出一次合法rollout，禁止补评估冒充满训')
    train_end = read(checked(folder / 'TRAIN_END.json', folder))
    if train_end.get('status') != 'COMPLETE' or int(train_end['real_steps']) != int(resume['real_steps']):
        raise RuntimeError('TRAIN_END与恢复点不能证明已满训')
    pointers = []
    for path in sorted(folder.glob('CP_*.json')) + [folder / 'RESUME.json']:
        pointers.append((path.name, pointer_check(folder, read(path))))
    selected = []
    for mark in job['marks']:
        # 最终模型必须用learn返回后保存的RESUME，而非最后一次更新前由callback保存的CP。
        if mark == job['marks'][-1]:
            selected.append(dict(mark=int(mark), pointer='RESUME.json', meta=resume))
            continue
        eligible = [(int(meta['real_steps']), name, meta) for name, meta in pointers
                    if mark <= int(meta['real_steps']) < mark + 10240]
        if not eligible:
            raise RuntimeError('缺少固定时点评估模型，保留缺口，不重训填表：' + str(mark))
        _, name, meta = min(eligible, key=lambda value: (value[0], value[1] == 'RESUME.json', value[1]))
        selected.append(dict(mark=int(mark), pointer=name, meta=meta))
    return resume, selected


def validate_job(project, module, root, plan, cell_id):
    expected = next((item for item in plan['tasks'] if item['id'] == cell_id and item['arm'] == 'PPO_TF'), None)
    if expected is None:
        raise RuntimeError('只允许修复原合同中的PPO_TF，不接受任意任务')
    folder = checked(root / 'cells' / cell_id, project)
    job = read(checked(folder / 'JOB.json', project))
    if job['task'] != expected or Path(job['root']).resolve() != root.resolve() or job['steps'] != expected['steps']:
        raise RuntimeError('任务说明与PLAN不符')
    if job['marks'] != module.eval_marks(job['steps'], 10240):
        raise RuntimeError('预注册评估时点不符')
    for bank in job['bank']:
        if digest(checked(bank['path'], root)) != bank['sha256']:
            raise RuntimeError('需求银行校验不符：' + bank['name'])
    checked(job['trace'], root)
    return folder, job


def ppo_worker(project, cell_id):
    module, root, plan = load_entry(project, 'core')
    folder, job = validate_job(project, module, root, plan, cell_id)
    repair = checked(root / 'engineering_repair' / REVISION / 'ppo' / cell_id, project, exists=False)
    lock = module.lock_file(repair / 'WORKER.lock')
    try:
        if own_workers(project / ENTRY['core'][0], folder):
            raise RuntimeError('该cell仍有原训练worker；不并发读写')
        end = read(folder / 'END.json', {})
        if end.get('status') == 'COMPLETE' and (folder / 'EVALUATION.json').is_file():
            return
        if SOURCE_ERROR not in end.get('traceback', ''):
            raise RuntimeError('不是已定位的PPO评估来源错误，保留待查')
        resume, selected = select_checkpoints(folder, job)
        for name in ('END.json', 'LIVE.json', 'EVALUATION.json', 'TRAIN_END.json', 'RESUME.json'):
            backup_once(folder / name, repair / 'before' / name)
        model_hashes = {meta['model']: meta['model_sha'] for meta in [resume] + [row['meta'] for row in selected]}
        norm_hashes = {meta['norm']: meta['norm_sha'] for meta in [resume] + [row['meta'] for row in selected]}
        # 此进程不调用learn或train_ppo；只在全新进程中加载一次封存环境，保留全部来源校验。
        base = module.base_module()
        m, native = base.runtime(root, job['task'])
        import numpy as np
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        probe = base.gym_class(m)(root, job['task'], job['trace'], 500, 123, False)
        try:
            base.torch_features(probe.pack)
        finally:
            probe.close()
        manifest = dict(revision=REVISION, state='EVALUATING', new_training_transitions=0,
            source_runtime_loads=1, evaluation_device='cpu', entrypoint_sha256=ENTRY['core'][1],
            model_sha256=model_hashes, normalization_sha256=norm_hashes, selections=selected,
            boundary='同需求、同随机策略评估、同固定存档；训练与归一化原文件只读')
        write_json(repair / 'EVALUATION_REPAIR.json', manifest)
        final = None
        for row in selected:
            mark, metadata = row['mark'], row['meta']
            with (folder / metadata['norm']).open('rb') as stream:
                norm = pickle.load(stream)
            norm.training = False
            norm.norm_reward = False
            model = PPO.load(folder / metadata['model'], device='cpu')
            def factory(seed):
                def predict(obs, carry):
                    action, _ = model.predict(norm.normalize_obs(np.asarray(obs)[None]), deterministic=False)
                    return int(action[0]), None
                return predict
            destination = folder / 'eval' / f'{mark:09d}.json'
            backup_once(destination, repair / 'before_eval' / destination.name)
            result = module.evaluate(base, m, root, job['task'], job['bank'], factory, mark,
                                     final=mark == job['marks'][-1])
            for episode in result:
                episode['actual_checkpoint'] = int(metadata['real_steps'])
                episode['engineering_repair_revision'] = REVISION
            module.save(destination, result)
            module.save(folder / 'checkpoints' / f'{mark:09d}.json', dict(metadata, requested_mark=mark))
            if mark == job['marks'][-1]:
                final = result
            print('[PPO只补评估]', cell_id, mark, '实际存档', metadata['real_steps'], flush=True)
        module.decision_probe(base, m, root, job['task'], job, factory)
        for name, wanted in {**model_hashes, **norm_hashes}.items():
            if digest(folder / name) != wanted:
                raise RuntimeError('补评估期间原模型文件变化，拒绝标记完成')
        if final is None:
            raise RuntimeError('缺少最终评估')
        module.save(folder / 'EVALUATION.json', dict(final=final, real_steps=resume['real_steps'],
            weights_unchanged=True, normalization_files_unchanged=True, new_training_transitions=0,
            repair_revision=REVISION, selected_checkpoints=selected))
        previous_live = read(folder / 'LIVE.json', {})
        module.save(folder / 'LIVE.json', dict(previous_live, real_steps=resume['real_steps'],
            phase='evaluation_complete', new_training_transitions=0, repair_revision=REVISION))
        module.save(folder / 'END.json', dict(status='COMPLETE', real_steps=resume['real_steps'],
            seconds=read(folder / 'TRAIN_END.json').get('seconds'), evaluated=True,
            new_training_transitions=0, repair_revision=REVISION,
            prior_error_preserved=str(repair / 'before' / 'END.json')))
        manifest.update(state='COMPLETE', weights_unchanged=True, time=time.time())
        write_json(repair / 'EVALUATION_REPAIR.json', manifest)
    finally:
        lock.close()


def refresh_if_idle(module, root, plan, do_pack=False):
    """原父进程活着时不争写总进度；下一次原聚合会自然读到补齐的cell。"""
    try:
        handle = module.lock_file(root / 'PARENT.lock')
    except (OSError, BlockingIOError):
        if do_pack:
            raise RuntimeError('原实验父进程还在运行，等结束后再打包')
        return False
    try:
        if do_pack and own_workers(Path(module.__file__)):
            raise RuntimeError('仍有本实验worker，不在运行中打包')
        progress = module.aggregate(root, plan['tasks'])
        module.summary(root, plan['tasks'])
        complete = progress['trained'] == len(plan['tasks']) and progress['evaluated'] == len(plan['tasks'])
        if complete:
            backup_once(root / 'END.json', root / 'engineering_repair' / REVISION / 'ROOT_END.before.json')
            module.save(root / 'END.json', dict(status='COMPLETE', progress=progress,
                repair_revision=REVISION, time=time.time()))
        if do_pack:
            module.pack(root)
            archive = module.PROJECT / 'experiment_archives' / (root.name + '_ENGINEERING_REPAIR.json')
            write_json(archive, dict(revision=REVISION, repair_script_sha256=digest(__file__),
                entrypoint_sha256=ENTRY[module.SUITE][1], new_training_transitions=0, time=time.time()))
        return True
    finally:
        handle.close()


def follow_ppo(project, follow, poll_seconds):
    module, root, plan = load_entry(project, 'core')
    repair = checked(root / 'engineering_repair' / REVISION, project, exists=False)
    record_tool(repair)
    lock = module.lock_file(repair / 'PPO_FOLLOW.lock')
    attempted, results = set(), []
    tasks = [item for item in plan['tasks'] if item['arm'] == 'PPO_TF']
    try:
        while True:
            unresolved = []
            for item in tasks:
                folder = root / 'cells' / item['id']
                end = read(folder / 'END.json', {})
                if end.get('status') == 'COMPLETE' and (folder / 'EVALUATION.json').is_file():
                    continue
                unresolved.append(item['id'])
                signature = (item['id'], digest(folder / 'END.json') if (folder / 'END.json').is_file() else '')
                if signature in attempted or end.get('status') != 'ENGINEERING_FAILED':
                    continue
                if SOURCE_ERROR not in end.get('traceback', ''):
                    continue
                # 未满训的格绝不补训；等待原训练完成或让用户单独决定。
                try:
                    _, job = validate_job(project, module, root, plan, item['id'])
                    select_checkpoints(folder, job)
                except Exception as error:
                    attempted.add(signature)
                    results.append(dict(id=item['id'], state='NEEDS_REVIEW', error=str(error)))
                    print('[保留待查]', item['id'], str(error), flush=True)
                    continue
                if own_workers(project / ENTRY['core'][0], folder):
                    continue
                attempted.add(signature)
                log = repair / 'ppo' / item['id'] / 'evaluation_worker.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                print('[启动独立CPU补评估]', item['id'], flush=True)
                with log.open('ab') as stream:
                    result = subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                        '--project', str(project), '--mode', 'ppo', '--ppo-worker', item['id']],
                        cwd=project, stdout=stream, stderr=subprocess.STDOUT)
                results.append(dict(id=item['id'], state='COMPLETE' if result.returncode == 0 else 'FAILED',
                    returncode=result.returncode, log=str(log), new_training_transitions=0))
                print('[补评估结果]', results[-1], flush=True)
            completed = sum(read(root / 'cells' / item['id'] / 'END.json', {}).get('status') == 'COMPLETE'
                            for item in tasks)
            write_json(repair / 'PPO_PROGRESS.json', dict(revision=REVISION, total_ppo=len(tasks),
                completed_ppo=completed, unresolved=unresolved, attempts=results,
                new_training_transitions=0, time=time.time()))
            refresh_if_idle(module, root, plan)
            if not follow or completed == len(tasks):
                break
            if not own_workers(project / ENTRY['core'][0]):
                print('[补评估退出] 原主进程已结束；未解决的格保留待查，不自动补训练', flush=True)
                break
            if (repair / 'STOP_PPO_REPAIR').exists():
                print('[补评估退出] 收到仅针对修复工具的停止标记', flush=True)
                break
            time.sleep(max(5., poll_seconds))
    finally:
        lock.close()


def resume_dima(project, vendor, patch_only, clear_stop):
    module, root, plan = load_entry(project, 'dima')
    entry = project / ENTRY['dima'][0]
    if not patch_only:
        expected_environment = project / 'workspace/envs/uam_dima_20261106'
        if Path(sys.prefix).resolve() != expected_environment.resolve():
            raise RuntimeError('恢复训练必须使用DIMA自己的独立环境Python，不使用主组环境')
    handle = module.lock_file(root / 'PARENT.lock')
    try:
        if own_workers(entry):
            raise RuntimeError('DIMA仍有自己的父进程或worker，不并发修改')
        patch_dima(project, module, root, vendor)
        if patch_only:
            return
        # 先核验独立环境和动态库，再响应用户明确给出的解除暂停请求。
        library = Path(sysconfig.get_paths()['purelib']) / 'nvidia' / 'nvjitlink' / 'lib'
        if not library.resolve().is_relative_to(expected_environment.resolve()):
            raise RuntimeError('动态库路径不在DIMA独立环境内')
        if not (library / 'libnvJitLink.so.12').is_file():
            raise RuntimeError('当前Python不是已修复的DIMA独立环境，未找到其nvJitLink库')
        if (root / 'STOP_REQUESTED').exists():
            if not clear_stop:
                raise RuntimeError('DIMA有暂停标记；明确加--clear-stop才恢复')
            stopped = checked(root / 'STOP_REQUESTED', project)
            backup_once(stopped, root / 'engineering_repair' / REVISION / 'STOP_REQUESTED.before')
            stopped.unlink()
        # 只增加本独立环境的动态库搜索路径，不安装或修改任何环境。
        environment = os.environ.copy()
        environment['LD_LIBRARY_PATH'] = str(library) + (':' + environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else '')
        if not environment.get('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('恢复DIMA前必须显式设置CUDA_VISIBLE_DEVICES')
        command = [sys.executable, '-B', '-u', str(entry), '--resume', '--retry-failed', '--device', 'cuda',
                   '--root', str(root), '--vendor-dir', str(vendor)]
        write_json(root / 'engineering_repair' / REVISION / 'DIMA_RESUME.json', dict(command=command,
            plan_unchanged=True, entrypoint_unchanged=True, budget_cap=plan['budget_cap'], time=time.time(),
            boundary='按原RESUME恢复；恢复点后的重复采样仍由原预算账本计费，不清零失败记录'))
    finally:
        handle.close()
    print('[恢复DIMA] 使用原入口、原预算、原恢复点；显存不足仍按原规则排队', flush=True)
    os.execve(sys.executable, command, environment)


def main():
    parser = argparse.ArgumentParser(description='既有PPO评估与DIMA掩码工程修复，不另建训练矩阵')
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--mode', choices=('ppo', 'dima', 'pack'), required=True)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30.)
    parser.add_argument('--ppo-worker')
    parser.add_argument('--vendor-dir', type=Path)
    parser.add_argument('--patch-only', action='store_true')
    parser.add_argument('--clear-stop', action='store_true')
    args = parser.parse_args()
    project = args.project.absolute()
    checked(project, project)
    if args.ppo_worker:
        if args.mode != 'ppo':
            raise RuntimeError('补评估worker参数与mode不符')
        ppo_worker(project, args.ppo_worker)
    elif args.mode == 'ppo':
        follow_ppo(project, args.follow, args.poll_seconds)
    elif args.mode == 'dima':
        vendor = args.vendor_dir.absolute() if args.vendor_dir else project / 'workspace/vendor/DIMA_20261106'
        resume_dima(project, vendor, args.patch_only, args.clear_stop)
    else:
        for suite in ('core', 'dima'):
            module, root, plan = load_entry(project, suite)
            refresh_if_idle(module, root, plan, do_pack=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
