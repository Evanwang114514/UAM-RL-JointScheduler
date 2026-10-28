#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仅清理经归档核验的H150旧中间检查点；不连接服务器、不碰其他实验。

默认仅列清单；--apply也必须在交互终端再次确认。
保留首点、中点、末四点、dense最优点及其模型/归一化/恢复状态。
不加载任何模型，不导入torch，不修改训练合同。
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import tarfile
import time

PROJECT = Path('/home/SY/uam/UAGMC-main')
NAME = 'uam_H150_20261015_v2'
ARCHIVE_SHA = '3613171791f119f398ef694506df11becd2a9de43311aeb886c8578fdac7da8e'
CODE_SHA = '208a7f2711222e82edaf71ddac31e7b8a94fc5fabc74ce97afa0c2f54d33eef9'
TOTAL = 696320


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def owned(path, uid, directory=False):
    """拒绝链接、其他账号文件和预料之外的文件类型。"""
    info = path.lstat()
    wanted = stat.S_ISDIR if directory else stat.S_ISREG
    if path.is_symlink() or not wanted(info.st_mode) or info.st_uid != uid:
        raise RuntimeError(f'属主、类型或符号链接核验失败，未删除：{path}')
    return info


def no_active_training(uid):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != uid:
                continue
            command = (proc/'cmdline').read_bytes()
        except (OSError, PermissionError):
            continue
        if b'train_uam_H150_how_why_ONEFILE.py' in command:
            raise RuntimeError('发现H150训练/评估进程，停止清理；没有终止该进程')


def retained_steps(steps, dense):
    ordered = sorted(steps)
    usable = []
    for row in dense:
        try:
            value, step = float(row['ATT']), int(row['step'])
        except (ValueError, TypeError, KeyError):
            continue
        if math.isfinite(value) and row.get('full', '').lower() in ('true', '1') and step in steps:
            usable.append((value, step))
    if not ordered or not usable or ordered[-1] != TOTAL:
        return set(ordered)
    return {ordered[0], ordered[len(ordered)//2], *ordered[-4:], min(usable)[1]}


def fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_size, info.st_mtime_ns, info.st_nlink)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='通过核验、列出清单且手动确认后才删除')
    args = parser.parse_args()
    if os.name != 'posix':
        raise RuntimeError('此工具仅供指定Linux服务器使用，拒绝在Windows本地删除')
    import pwd
    uid = os.getuid()
    if pwd.getpwuid(uid).pw_name != 'SY' or uid == 0:
        raise RuntimeError('必须使用SY本人会话，不使用root或sudo')
    root = PROJECT/'serial_runs'/NAME
    archive = PROJECT/'experiment_archives'/f'{NAME}_DATAONLY.tar.gz'
    for folder in (PROJECT, PROJECT/'serial_runs', root, root/'cells'):
        owned(folder, uid, True)
        if folder.resolve() != folder:
            raise RuntimeError(f'路径存在链接或重定向，停止：{folder}')
    owned(archive, uid)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise RuntimeError('归档包与本地已核验原包哈希不同，未删除；请人工核对，不绕过此检查')
    no_active_training(uid)
    owned(root/'PLAN.json', uid)
    owned(root/'progress.json', uid)
    plan, progress = read_json(root/'PLAN.json'), read_json(root/'progress.json')
    if plan.get('version') != 'H150_20261015_V2' or plan.get('code_sha256') != CODE_SHA:
        raise RuntimeError('运行版本不是已核验的H150归档版本，未删除')
    if any(progress.get(k) != 150 for k in ('planned', 'trained', 'evaluated')):
        raise RuntimeError('H150尚未确认150格训练和评估均结束，未删除')

    targets, keep_report, skipped = [], [], []
    with tarfile.open(archive, 'r:gz') as bundle:
        saved = json.load(bundle.extractfile(f'{NAME}/progress.json'))
        saved_plan = json.load(bundle.extractfile(f'{NAME}/PLAN.json'))
        if saved_plan.get('code_sha256') != CODE_SHA:
            raise RuntimeError('包内计划不匹配，未删除')
        for record in saved['rows']:
            cell = record['cell']
            if '/' in cell or '\\' in cell or cell.startswith('.'):
                raise RuntimeError('任务名非法，未删除')
            # 评估不完整的负结果整格保留，便于继续归因。
            if record.get('training') != 'COMPLETE' or record.get('evaluation') != 'COMPLETE' or record.get('status') != 'VALID':
                skipped.append(cell); continue
            directory = root/'cells'/cell
            checkpoint = directory/'checkpoints'
            owned(directory, uid, True); owned(checkpoint, uid, True)
            end_path = directory/'train_end.json'; owned(end_path, uid)
            end = read_json(end_path)
            if end.get('status') != 'COMPLETE' or end.get('actual_steps') != TOTAL:
                raise RuntimeError(f'格子训练状态不匹配，停止：{cell}')
            manifests = {}
            for marker in checkpoint.glob('checkpoint_*.json'):
                owned(marker, uid); info = read_json(marker); step = int(info['step'])
                if not info.get('ready') or marker.name != f'checkpoint_{step}.json':
                    raise RuntimeError(f'检查点登记异常：{marker}')
                expected = dict(model=f'checkpoint_{step}.model.zip', norm=f'checkpoint_{step}.norm.pkl', state=f'checkpoint_{step}.state.pt')
                if any(info.get(k) != v for k,v in expected.items()):
                    raise RuntimeError(f'文件命名不符合我们源码，未删除：{marker}')
                manifests[step] = expected
            if set(manifests) != set(range(40960, TOTAL+1, 40960)):
                raise RuntimeError(f'检查点登记不是原17点，停止：{cell}')
            dense_text = bundle.extractfile(f'{NAME}/cells/{cell}/analysis/dense_curve.csv').read().decode('utf-8-sig')
            keep = retained_steps(set(manifests), list(csv.DictReader(io.StringIO(dense_text))))
            keep_report.append(dict(cell=cell, keep=sorted(keep)))
            for step, files in manifests.items():
                # 保留点的三份文件必须齐全，否则不开始整个清理。
                for filename in files.values():
                    path = checkpoint/filename
                    if step in keep:
                        owned(path, uid); continue
                    if not path.exists() and not path.is_symlink():
                        continue
                    info = owned(path, uid)
                    if info.st_nlink != 1:
                        raise RuntimeError(f'发现硬链接，停止：{path}')
                    targets.append(dict(path=str(path), cell=cell, step=step, bytes=info.st_size,
                        allocated=info.st_blocks*512, fingerprint=fingerprint(info)))

    print('归属核验通过：固定项目路径＋原归档SHA256＋原代码版本＋150格登记＋逐文件SY属主。')
    print('只处理H150；不处理其他账号/其他目录，不处理120M、240M或当前198M。')
    for item in targets:
        print(f"[拟删除] {item['allocated']/1048576:.2f} MiB\t{item['path']}")
    print(f"合计{len(targets)}个冗余文件，预计分配空间{sum(t['allocated'] for t in targets)/1073741824:.2f} GiB。")
    print(f'保留{len(keep_report)}格关键点；另外完整保留{len(skipped)}格非VALID结果。')
    if not targets:
        print('没有待清理文件，未删除。'); return
    if not args.apply:
        print('仅清单模式，未删除；确认后使用相同命令加--apply。'); return
    if not sys.stdin.isatty():
        raise RuntimeError('必须在交互终端人工确认，不能后台自动删除')
    answer = input('确认仅删除上列H150冗余中间模型，请输入 DELETE_H150_MIDDLE：').strip()
    if answer != 'DELETE_H150_MIDDLE':
        print('已取消，未删除。'); return
    no_active_training(uid)
    for item in targets:
        path = Path(item['path'])
        owned(path.parent, uid, True)
        if path.resolve().parent != path.parent or fingerprint(owned(path, uid)) != tuple(item['fingerprint']):
            raise RuntimeError(f'确认期间文件或路径变化，未开始删除：{path}')
    audit_dir = PROJECT/'cleanup_audits'
    if audit_dir.exists(): owned(audit_dir, uid, True)
    else: audit_dir.mkdir()
    audit = audit_dir/f'H150_middle_{time.time_ns()}.json'
    with audit.open('x', encoding='utf-8') as stream:
        json.dump(dict(targets=targets, kept=keep_report, skipped=skipped, archive_sha256=ARCHIVE_SHA), stream, ensure_ascii=False, indent=2)
    before = os.statvfs(PROJECT); removed = 0
    for item in targets:
        path = Path(item['path'])
        if fingerprint(owned(path, uid)) != tuple(item['fingerprint']):
            raise RuntimeError(f'文件发生变化，已停止；此前已删{removed}个，详见清单：{path}')
        path.unlink(); removed += 1
    after = os.statvfs(PROJECT)
    print(f'完成：删除{removed}个H150冗余模型/归一化/恢复文件；保留全部JSON/CSV/日志、原始包及关键检查点。')
    print(f'当前文件系统可用空间：{after.f_bavail*after.f_frsize/1073741824:.2f} GiB；清单：{audit}')
    print('删除的中间模型无法由DATAONLY包恢复；当前198M与240M恢复件未改动。')


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(f'[停止] {error}', file=sys.stderr); sys.exit(1)
