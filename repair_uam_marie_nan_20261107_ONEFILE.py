#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仅修既有MARIE实验的数值故障；校验、保留恢复点，再调用原入口。"""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

sys.dont_write_bytecode = True
REVISION = 'UAM_MARIE_CRITIC_MATH_20261107_V1'
ENTRY_NAME = 'train_uam_marie_25m_ONEFILE.py'
ENTRY_SHA = '4c76ac5ebb1a523214a45c2dd162e5af33603f0cbaabd1429dd03005d2177aef'
RUN_NAME = 'uam_marie_author_20261106_v1'
FAILED_CELL = 'CORE__E1__K3__MARIE_SHORT_HORIZON__s2301'
MIN_RESUME_STEPS = 86673
CRITIC_FILE = 'networks/dreamer/critic.py'
LEARNER_FILE = 'agent/learners/DreamerLearner.py'
RESERVE_GIB = 12.0


def safe_path(project, path, must_exist=True):
    """逐层拒绝符号链接，目标必须留在自己的项目目录。"""
    project, path = Path(project).absolute(), Path(path).absolute()
    relative = path.relative_to(project)
    cursor = project
    if cursor.is_symlink():
        raise RuntimeError('项目目录不能是符号链接')
    for part in relative.parts:
        if part in ('.', '..'):
            raise RuntimeError('不接受路径回退')
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError('拒绝符号链接：' + str(cursor))
    if not path.resolve().is_relative_to(project.resolve()):
        raise RuntimeError('路径越出项目')
    if must_exist and not path.exists():
        raise RuntimeError('缺少文件：' + str(path))
    return path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    if path.is_symlink():
        raise RuntimeError('拒绝读取链接：' + str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_bytes(path, data):
    if path.is_symlink():
        raise RuntimeError('拒绝覆盖链接：' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.pending_' + str(os.getpid()))
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def write_json(path, data):
    write_bytes(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


def preserve(source, target):
    """保留原始小文件，已有副本不覆盖。"""
    if source.is_symlink() or target.is_symlink():
        raise RuntimeError('备份源或目标为符号链接')
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as reader, target.open('xb') as writer:
            shutil.copyfileobj(reader, writer)


def once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError('补丁定位不是唯一匹配：' + before.strip()[:100])
    return text.replace(before, after, 1)


def method_replace(text, method, before, after):
    """只改锁定类内指定方法，避免替换其他同名数值操作。"""
    learner = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'DreamerLearner')
    node = next(n for n in learner.body if isinstance(n, ast.FunctionDef) and n.name == method)
    lines = text.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [once(''.join(lines[node.lineno - 1:node.end_lineno]), before, after)]
    return ''.join(lines)


FINITE_HELPERS = '''
def _uam_marie_finite(label, value):
    # 不替换或跳过异常数值；错误按原入口记为失败。
    if not bool(torch.isfinite(value.detach()).all()):
        raise RuntimeError("[MARIE数值保护] 非有限值：" + label)


def _uam_marie_restore_guard(learner):
    # 反序列化不会重跑构造函数；恢复作者启用的异常检测。
    pid = _uam_marie_os.getpid()
    if getattr(learner, "_uam_marie_finite_checked_pid", None) == pid:
        return
    torch.autograd.set_detect_anomaly(True)
    for label, obj in vars(learner).items():
        if isinstance(obj, torch.nn.Module):
            # 不误判注意力中合法的负无穷掩码缓冲区。
            for name, value in obj.named_parameters():
                _uam_marie_finite("参数/" + label + "/" + name, value)
        elif isinstance(obj, torch.optim.Optimizer):
            for state in obj.state.values():
                for name, value in state.items():
                    if torch.is_tensor(value):
                        _uam_marie_finite("优化器/" + label + "/" + name, value)
    learner._uam_marie_finite_checked_pid = pid
    print("[MARIE数值修复] 参数和优化器状态有限；Critic使用数学注意力后端", flush=True)


'''


def patch_text(critic, learner):
    critic = once(critic, 'from networks.transformer.layers import AttentionEncoder\n',
                  'from networks.transformer.layers import AttentionEncoder\n'
                  'from torch.nn.attention import SDPBackend, sdpa_kernel\n')
    lines, count = [], 0
    for line in critic.splitlines(keepends=True):
        if line.startswith(('        attn_embeds = F.relu(self._attention_stack(embeds).view(',
                            '        critic_feat = F.relu(self._attention_stack(embeds).view(')):
            lines.extend(('        # 只在价值网络使用同公式的数学后端。\n',
                          '        with sdpa_kernel(SDPBackend.MATH):\n', '    ' + line))
            count += 1
        else:
            lines.append(line)
    if count != 2:
        raise RuntimeError('两类价值网络的注意力定位不符')
    critic = ''.join(lines)
    learner = once(learner, 'import sys\n', 'import sys\nimport os as _uam_marie_os\n')
    learner = once(learner, '\ndef orthogonal_init(', '\n' + FINITE_HELPERS + 'def orthogonal_init(')
    # 原入口加载存档后，先用params向控制器同步参数，再开始任何真实采样。
    learner = method_replace(learner, 'params', '    def params(self):\n',
                             '    def params(self):\n        _uam_marie_restore_guard(self)\n')
    # 同一个方法内还有minibatch的价值归一化分支；只定位rollout之后的第一处分支。
    anchor = '        if self.use_valuenorm:\n            B, N = old_values.shape[:-1]\n'
    learner = method_replace(learner, 'train_agent_with_transformer', anchor,
        '        _uam_marie_finite("想象状态特征/critic", critic_feat)\n'
        '        _uam_marie_finite("想象状态特征/actor", actor_feat)\n'
        '        _uam_marie_finite("想象回报目标", returns)\n'
        '        _uam_marie_finite("旧价值估计", old_values)\n' + anchor)
    learner = method_replace(learner, 'train_agent_with_transformer', "        wandb.log({'Agent/Returns': returns.mean()})\n",
        '        _uam_marie_finite("策略优势", adv)\n' + "        wandb.log({'Agent/Returns': returns.mean()})\n")
    learner = method_replace(learner, 'apply_optimizer', '        opt.zero_grad()\n',
        '        _uam_marie_finite("损失/" + type(model).__name__, loss)\n        opt.zero_grad()\n')
    learner = method_replace(learner, 'apply_optimizer', 'clip_grad_norm_(model.parameters(), grad_clip)',
                             'clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)')
    compile(critic, CRITIC_FILE, 'exec')
    compile(learner, LEARNER_FILE, 'exec')
    return {CRITIC_FILE: critic.encode('utf-8'), LEARNER_FILE: learner.encode('utf-8')}


def load_entry(project):
    entry = safe_path(project, project / ENTRY_NAME)
    if sha(entry) != ENTRY_SHA:
        raise RuntimeError('MARIE原入口SHA不符；不覆盖原文件或绕过原计划校验')
    spec = importlib.util.spec_from_file_location('uam_marie_original_for_repair', entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    root = safe_path(project, project / 'serial_runs' / RUN_NAME)
    expected = module.plan_make()
    expected['entrypoint_sha256'] = ENTRY_SHA
    if read(safe_path(project, root / 'PLAN.json')) != expected:
        raise RuntimeError('MARIE既有PLAN与原入口不匹配，拒绝恢复')
    return module, root, entry


def patch_plan(project, module, root, vendor):
    """对照封存作者文件，准备允许中断后重做的确定性补丁。"""
    pinned = module.SOURCE_HASHES['marie']
    adapted = safe_path(project, root / 'adapted_author' / module.SOURCE_PINS['marie'])
    manifest = safe_path(project, adapted / 'UAM_ADAPTATION.json')
    backup = safe_path(project, root / 'adapted_author' / '_engineering_backups' / REVISION, False)
    old_manifest = backup / 'UAM_ADAPTATION.before.json'
    baseline = read(old_manifest if old_manifest.is_file() else manifest)
    for relative, wanted in pinned.items():
        if sha(safe_path(project, vendor / relative)) != wanted:
            raise RuntimeError('锁定的作者原件不符：' + relative)
    patched = patch_text(*(safe_path(project, vendor / key).read_text(encoding='utf-8-sig')
                           for key in (CRITIC_FILE, LEARNER_FILE)))
    after = json.loads(json.dumps(baseline))
    changes = []
    for relative, data in patched.items():
        if baseline['adapted_hashes'].get(relative) != pinned[relative]:
            raise RuntimeError('本次需要修复的作者文件有未识别的旧修改：' + relative)
        new_sha = hashlib.sha256(data).hexdigest()
        after['adapted_hashes'][relative] = new_sha
        changes.append(dict(file=relative, before_sha256=pinned[relative], after_sha256=new_sha))
    after.setdefault('engineering_patches', []).append(dict(revision=REVISION, files=changes,
        purpose='Critic数学注意力后端与有限值保护；不改模型、目标、训练更新或环境'))
    if read(manifest) not in (baseline, after):
        raise RuntimeError('适配清单有未知修改，保留待查')
    for relative, wanted in baseline['adapted_hashes'].items():
        digest = sha(safe_path(project, adapted / relative))
        if digest not in (wanted, after['adapted_hashes'][relative]):
            raise RuntimeError('适配源码有未知修改，保留待查：' + relative)
    return adapted, manifest, backup, baseline, after, patched, changes


def install_patch(project, module, root, vendor, audit):
    adapted, manifest, backup, baseline, after, patched, changes = patch_plan(project, module, root, vendor)
    old_manifest = backup / 'UAM_ADAPTATION.before.json'
    if not old_manifest.exists():
        preserve(manifest, old_manifest)
    for relative in patched:
        preserve(safe_path(project, vendor / relative), safe_path(project, backup / relative, False))
    write_json(audit / 'PATCH.json', dict(revision=REVISION, state='PREPARED', files=changes, time=time.time()))
    for relative, data in patched.items():
        target = safe_path(project, adapted / relative)
        if sha(target) != after['adapted_hashes'][relative]:
            write_bytes(target, data)
    write_json(manifest, after)
    write_json(safe_path(project, root / 'SOURCE_ADAPTATION.json', False), after)
    # 直接调用原入口自己的源码校验；不降低其完整性要求。
    if module.wm_source(root, vendor) != adapted:
        raise RuntimeError('原入口重新核验的作者适配路径不符')
    write_json(audit / 'PATCH.json', dict(revision=REVISION, state='APPLIED', files=changes, time=time.time(),
        entrypoint_unchanged=True, plan_unchanged=True, new_training_transitions=0))
    print('[MARIE修复] 作者适配副本已修复，原PLAN、入口和作者仓库不变', flush=True)


def protect_resume(project, root, audit):
    cell = safe_path(project, root / 'cells' / FAILED_CELL)
    status = read(safe_path(project, cell / 'END.json'))
    if status.get('status') == 'COMPLETE' and (cell / 'EVALUATION.json').is_file():
        print('[MARIE恢复] 首格已完整结束，原入口会跳过', flush=True)
        return None
    pointer_path = safe_path(project, cell / 'RESUME.json')
    pointer = read(pointer_path)
    filename = pointer['full_file']
    real = int(pointer['real_steps'])
    if real < MIN_RESUME_STEPS or int(pointer['decision_records']) != real * 2:
        raise RuntimeError('缺少足够步数的完整恢复点；不从头训练或回退到更早记录')
    if filename != f'RESUME_FULL_{real:09d}.pkl':
        raise RuntimeError('恢复文件名与计步不符')
    source = safe_path(project, cell / 'replay' / filename)
    if sha(source) != pointer['full_sha256']:
        raise RuntimeError('完整恢复文件哈希不符，不启动续训')
    # 只保护最初的修复前恢复点，后续重启不积累多个大回放备份。
    original_pointer = audit / 'RESUME.before.json'
    initial = read(original_pointer) if original_pointer.exists() else pointer
    initial_name = initial['full_file']
    if Path(initial_name).name != initial_name:
        raise RuntimeError('保护文件名越界')
    protected = safe_path(project, cell / 'replay' / '_engineering_backup' / REVISION / initial_name, False)
    protected.parent.mkdir(parents=True, exist_ok=True)
    if protected.exists():
        if sha(protected) != initial['full_sha256']:
            raise RuntimeError('保护副本与最初恢复记录不一致')
    else:
        if initial_name != filename or initial['full_sha256'] != pointer['full_sha256']:
            raise RuntimeError('最初的恢复保护副本丢失；不静默替换')
        # 硬链接不复制大文件；原入口轮换时不影响这个独立名称。
        os.link(source, protected)
    preserve(pointer_path, original_pointer)
    preserve(cell / 'END.json', audit / 'CELL_END.before.json')
    write_json(audit / 'RECOVERY.json', dict(real_steps=real, original=str(source), protected=str(protected),
        sha256=pointer['full_sha256'], protected_sha256=initial['full_sha256'],
        boundary='恢复完整训练状态，从新仿真回合继续；不声称位级续接', time=time.time()))
    print('[MARIE恢复] 原存档哈希通过，已保护完整恢复点：', real, '步', flush=True)
    return real


def proc_start(pid):
    return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()[19]


def own_processes(root, entry):
    """只识别当前账号、当前入口和当前实验根目录；不发送进程信号。"""
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if str(entry) not in argv:
                if entry.name not in argv or not (proc / 'cwd').resolve().is_relative_to(entry.parent.resolve()):
                    continue
            if '--worker' in argv:
                job = Path(argv[argv.index('--worker') + 1])
                if not job.is_absolute() or not job.resolve().is_relative_to(root.resolve()):
                    continue
            elif '--root' in argv:
                if Path(argv[argv.index('--root') + 1]).resolve() != root.resolve():
                    continue
            found.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def pause_and_lock(module, root, entry, timeout):
    marker = root / 'STOP_REQUESTED'
    if marker.is_symlink():
        raise RuntimeError('暂停标记不能是符号链接')
    live = own_processes(root, entry)
    if live and not marker.exists():
        with marker.open('x', encoding='utf-8') as stream:
            stream.write(f'{REVISION} {os.getpid()} {time.time_ns()}\n')
        print('[MARIE恢复] 已请求旧MARIE正常退出，仅等待该实验：', live, flush=True)
    marker_state = (marker.stat().st_mtime_ns, marker.read_bytes()) if marker.exists() else None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not own_processes(root, entry):
            try:
                lock = module.lock_file(root / 'PARENT.lock')
            except BlockingIOError:
                lock = None
            if lock is not None:
                if not own_processes(root, entry):
                    return lock, marker_state
                lock.close()
        time.sleep(3)
    raise RuntimeError('旧MARIE尚未正常退出；保留暂停标记和存档，不强杀或重复启动')


def preflight(project, root):
    env = safe_path(project, project / 'workspace/envs/uam_marie_20261106')
    if Path(sys.prefix).resolve() != env.resolve():
        raise RuntimeError('必须使用MARIE独立环境Python')
    if importlib.metadata.version('torch') != '2.4.1+cu121':
        raise RuntimeError('Torch不是本轮核验的2.4.1+cu121；不自动安装或换版本')
    library = safe_path(project, Path(sysconfig.get_paths()['purelib']) / 'nvidia/nvjitlink/lib')
    if not (library / 'libnvJitLink.so.12').is_file():
        raise RuntimeError('缺少MARIE独立环境里的nvJitLink库')
    if shutil.disk_usage(root).free < 12 * 1024**3:
        raise RuntimeError('项目剩余磁盘不足12GiB，不动现有进程')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('本轮必须显式指定CUDA_VISIBLE_DEVICES=0')
    output = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free,memory.total',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    free, total = [float(v.strip()) / 1024 for v in output.strip().split(',')]
    if total < RESERVE_GIB + 1.0:
        raise RuntimeError('所选GPU容量低于保留的安全预算')
    print(f'[资源预检] GPU 0 空闲 {free:.2f}/{total:.2f} GiB；MARIE申请 {RESERVE_GIB:g} GiB，继续使用原共享准入', flush=True)
    return library


def main():
    parser = argparse.ArgumentParser(description='修复并恢复原MARIE矩阵，不新开实验')
    parser.add_argument('--resume', action='store_true', required=True)
    parser.add_argument('--stop-timeout', type=float, default=900.0)
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('正式恢复由用户在服务器终端执行')
    if not 30 <= args.stop_timeout <= 3600:
        raise RuntimeError('旧进程退出等待须在30到3600秒之间')
    project = Path(__file__).resolve().parent
    module, root, entry = load_entry(project)
    vendor = safe_path(project, project / 'workspace/vendor/MARIE_20261106')
    audit = safe_path(project, root / 'engineering_repair' / REVISION, False)
    own_lock = module.lock_file(audit / 'REPAIR.lock')
    parent_lock = None
    try:
        latest_launch = audit / 'LAUNCH.json'
        if latest_launch.is_file():
            old_launch = read(latest_launch)
            pid = old_launch['pid']
            if pid in own_processes(root, entry) and proc_start(pid) == old_launch['proc_start']:
                print('[MARIE恢复] 修复后的原入口已经在运行，不重复启动', flush=True)
                return
        library = preflight(project, root)
        patch_plan(project, module, root, vendor)
        parent_lock, marker_state = pause_and_lock(module, root, entry, args.stop_timeout)
        protected_tool = safe_path(project, audit / 'REPAIR_TOOL.py', False)
        if protected_tool.is_file() and sha(protected_tool) != sha(__file__):
            raise RuntimeError('已登记的修复工具与当前版本不符，不覆盖')
        preserve(Path(__file__), protected_tool)
        restored = protect_resume(project, root, audit)
        install_patch(project, module, root, vendor, audit)
        environment = os.environ.copy()
        environment['LD_LIBRARY_PATH'] = str(library) + (':' + environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else '')
        command = [sys.executable, '-B', '-u', str(entry), '--resume', '--retry-failed', '--device', 'cuda',
                   '--gpu-reserve-gib', str(RESERVE_GIB), '--root', str(root), '--vendor-dir', str(vendor)]
        marker = safe_path(project, root / 'STOP_REQUESTED', False)
        current_marker = (marker.stat().st_mtime_ns, marker.read_bytes()) if marker.exists() else None
        if current_marker != marker_state:
            raise RuntimeError('修复期间出现新的暂停指令，保留不启动')
        write_json(latest_launch, dict(revision=REVISION, pid=os.getpid(), proc_start=proc_start(os.getpid()),
            command=command, restored_steps=restored, gpu_reserve_gib=RESERVE_GIB, time=time.time()))
        if marker.exists():
            preserve(marker, audit / f'STOP_REQUESTED.{time.time_ns()}.before')
            marker.unlink()
    finally:
        if parent_lock is not None:
            parent_lock.close()
        own_lock.close()
    print('[MARIE启动] 原入口从完整恢复点继续；不改主组、DIMA、预算或环境', flush=True)
    os.chdir(project)
    os.execve(sys.executable, command, environment)


if __name__ == '__main__':
    main()
