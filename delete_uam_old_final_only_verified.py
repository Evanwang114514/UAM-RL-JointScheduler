#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只删除20261101_211457_1905767审计包列明的中间模型；不递归删除目录。"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time

if os.name == "posix":
    import fcntl


PROJECT = Path("/home/SY/uam/UAGMC-main")
AUDIT = PROJECT / "cleanup_audits/old_final_only_20261101_211457_1905767"
PACKAGE = AUDIT / "old_final_only_20261101_211457_1905767_AUDIT_ONLY.tar.gz"
PACKAGE_SHA = "f05ff46d6a5227c0def8ef92ac422c65404cbfb41a9ec64f11c4f649abd04d77"
AUDITOR_SHA = "f0f853c281dfd09e07aac37e696f3d590320e67ca87981a5975710ccb9f0e5ad"
EXPECTED_FILES = 27386
EXPECTED_BYTES = 113694308016
EXPECTED_KEEP = 1436
PROCESS_PROTOCOL = "confirm_exited_pid_v2_20261101"
GROUPS = {
    "uam_100m_two_tracks_v1": 304,
    "uam_100m_two_tracks_BC_v2": 4462,
    "uam_21x7_700k_METHOD_FIRST_V2": 4998,
    "uam_round3_uagmc_causal_V1": 4488,
    "uam_80m_attribution_V4_FIXED": 3906,
    "uam_stage12_v2_20261007_033409": 2460,
    "uam_K2K3_supp60m_focus_v2": 4032,
    "uam500m_supplement70m_s3": 2736,
}
MODEL_RE = re.compile(r"(?:uam_ppo_(?:vecnormalize_)?\d+_steps\.(?:zip|pkl)|"
                      r"checkpoint_\d+\.(?:model\.zip|norm\.pkl|state\.pt))\Z")
FINGERPRINT = {"device": "st_dev", "inode": "st_ino", "uid": "st_uid",
               "nlink": "st_nlink", "bytes": "st_size", "mtime_ns": "st_mtime_ns"}
FILE_HASHES = {
    "全部模型处置清单.csv": "553a29c2157e8c33661eeff779ca39cc939e75285965d7ad9dd1ecafb96b35ff",
    "归属证据匹配.csv": "557de913b2cd1ee7830c45eef259cd033af7198e78d0595e8d6ce59e8534a418",
    "拟删除逐文件清单.csv": "0d1bda9dc32c1d98a00e8e93e48f1ceb068917c40eba086285d734995008f451",
    "本地归档证据索引.json": "810544c2223735ea7ee10cb5d43268e3525d456448c5ea0ce0a9e1a3d9c7cf45",
    "清点汇总.json": "b092f4803d71f6cb4d7f6ff4cbf36f6882eeb7696709bc1447823d664b85d052",
}
PROCESS_DETAILS = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_package(path):
    """审计包及其中五份清单均绑定本次已在本地核验的哈希。"""
    if digest(path.read_bytes()) != PACKAGE_SHA:
        raise RuntimeError("审计包不是本次核验原件，未删除")
    blobs = {}
    with tarfile.open(path, "r:gz") as bundle:
        for member in bundle:
            if (not member.isfile() or member.name not in FILE_HASHES or member.name in blobs
                    or member.size > 30 * 1024**2):
                raise RuntimeError("审计包成员异常，未删除")
            raw = bundle.extractfile(member).read()
            if digest(raw) != FILE_HASHES[member.name]:
                raise RuntimeError("包内清单哈希不同，未删除")
            blobs[member.name] = raw
    if set(blobs) != set(FILE_HASHES):
        raise RuntimeError("审计材料不完整，未删除")
    def rows(name):
        return list(csv.DictReader(io.StringIO(blobs[name].decode("utf-8-sig"))))
    return (rows("拟删除逐文件清单.csv"), rows("全部模型处置清单.csv"),
            rows("归属证据匹配.csv"), json.loads(blobs["本地归档证据索引.json"]),
            json.loads(blobs["清点汇总.json"]))


def validate_manifest(candidates, all_rows):
    """仅接受这八批已核验实验的明确文件；禁止根目录通配或路径扩张。"""
    paths = [row["path"] for row in candidates]
    all_paths = [row["path"] for row in all_rows]
    if len(paths) != EXPECTED_FILES or len(set(paths)) != len(paths) or len(set(all_paths)) != len(all_paths):
        raise RuntimeError("文件数量或唯一性核验失败，未删除")
    if set(paths) != {row["path"] for row in all_rows if row["action"] == "CANDIDATE"}:
        raise RuntimeError("两份逐文件清单不一致，未删除")
    if sum(int(row["bytes"]) for row in candidates) != EXPECTED_BYTES:
        raise RuntimeError("候选总大小与本地核验值不同，未删除")
    counts = {name: 0 for name in GROUPS}
    for row in candidates:
        name = row["experiment"]
        if name not in GROUPS or row["action"] != "CANDIDATE":
            raise RuntimeError("候选超出八批实验白名单，未删除")
        root = PurePosixPath("/home/SY/uam/UAGMC-main/serial_runs") / name
        cell = PurePosixPath(row["cell"])
        path = PurePosixPath(row["path"])
        if cell.is_absolute() or ".." in cell.parts or ".." in path.parts:
            raise RuntimeError("清单含越界路径，未删除")
        if str(root / str(cell) / "checkpoints" / path.name) != row["path"] or not MODEL_RE.fullmatch(path.name):
            raise RuntimeError("不是源码约定的checkpoint文件，未删除")
        if int(row["uid"]) != 998 or int(row["nlink"]) != 1:
            raise RuntimeError("属主或链接数不符合原审计，未删除")
        counts[name] += 1
    if counts != GROUPS:
        raise RuntimeError("逐实验数量与本地核验值不同，未删除")
    keep = {row["path"] for row in all_rows if row["action"] == "KEEP_FINAL"}
    if len(keep) != EXPECTED_KEEP or keep & set(paths):
        raise RuntimeError("最终模型保留清单异常，未删除")


def current_fingerprint(row, auditor):
    path = Path(row["path"])
    root = PROJECT / "serial_runs" / row["experiment"]
    info = auditor.file_info(path, root, os.getuid(), root.stat().st_dev)
    if any(getattr(info, attribute) != int(row[key]) for key, attribute in FINGERPRINT.items()):
        raise RuntimeError("文件自审计后已变化，停止：" + str(path))
    return info


def safe_open(path, flags):
    descriptor = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        os.close(descriptor)
        raise RuntimeError("清理日志或锁文件不安全，未删除")
    return descriptor


def prior_intents(journal, original):
    """中断后仅允许跳过已有同清单删除意向、且确已消失的文件。"""
    result = set()
    if not journal.exists():
        return result
    info = journal.lstat()
    if journal.is_symlink() or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_size > 64 * 1024**2:
        raise RuntimeError("历史清理日志不安全，停止")
    for line in journal.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("package_sha256") != PACKAGE_SHA:
            raise RuntimeError("历史清理日志属于其他清单，停止")
        if entry.get("action") == "intent":
            row = entry["row"]
            if row["path"] not in original or row != original[row["path"]]:
                raise RuntimeError("历史日志中的文件指纹不同，停止")
            result.add(row["path"])
    return result


def group_key(row):
    """把原清单中的模型/归一化/状态文件归入同一保存点，用于恢复被中断的删除。"""
    name = PurePosixPath(row["path"]).name
    match = re.fullmatch(r"uam_ppo_(?:vecnormalize_)?(\d+)_steps\.(?:zip|pkl)", name)
    family = "pair"
    if match is None:
        match = re.fullmatch(r"checkpoint_(\d+)\.(?:model\.zip|norm\.pkl|state\.pt)", name)
        family = "triple"
    if match is None:
        raise RuntimeError("无法确认成组保存点，停止")
    return row["experiment"], row["cell"], family, int(match[1])


def check_current_candidates(fresh_rows, remaining, missing):
    """只允许同清单已记录删除造成的半组残余；不能借恢复扩大到待查模型。"""
    fresh = {row["path"] for row in fresh_rows if row["action"] == "CANDIDATE"}
    wanted = {row["path"] for row in remaining}
    verified = {(row["experiment"], row["cell"]) for row in fresh_rows if row["action"] == "KEEP_FINAL"}
    partial = {group_key(row) for row in missing}
    resumable = {row["path"] for row in remaining if group_key(row) in partial and
                 (row["experiment"], row["cell"]) in verified}
    if not fresh <= wanted or not wanted - fresh <= resumable:
        raise RuntimeError("当前可删文件与已批准清单不同，未删除；不自动扩大或缩小范围")


def inspect_process(process, uid):
    """仅忽略已证实退出的PID；仍存在但无法读取的进程继续拦截。"""
    last_error = ""
    for _ in range(2):
        try:
            before = process.stat()
            if before.st_uid != uid:
                return "other", []
            arguments = (process / "cmdline").read_bytes().split(b"\0")
            after = process.stat()
            if after.st_uid != before.st_uid or after.st_ino != before.st_ino:
                last_error = "读取期间PID身份变化"
                continue
            return "ok", [item.decode("utf-8", "replace") for item in arguments if item]
        except OSError as error:
            last_error = type(error).__name__ + ": " + str(error)
            try:
                # cmdline失败不等于进程退出，必须再次确认进程根目录是否已消失。
                process.stat()
            except (FileNotFoundError, ProcessLookupError):
                return "gone", []
            except OSError as probe_error:
                last_error += "; 根目录复查 " + type(probe_error).__name__
                break
    return "unreadable", [last_error]


def precise_process_references(run_name, script_names, proc_root=None):
    """保持原引用判据，修复PID退出竞态；不打印可能含敏感参数的完整命令行。"""
    PROCESS_DETAILS.clear()
    active, unreadable = [], []
    root = Path("/proc") if proc_root is None else proc_root
    for process in root.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        status, values = inspect_process(process, os.getuid())
        pid = int(process.name)
        if status == "unreadable":
            unreadable.append(pid)
            PROCESS_DETAILS[pid] = values[0]
        elif status == "ok" and (any(run_name in item for item in values) or
                                  any(Path(item).name in script_names for item in values)):
            active.append(pid)
            PROCESS_DETAILS[pid] = "仍在运行且引用本批旧实验"
    return active, unreadable


def process_block_message(name, active, unreadable):
    return ("旧实验进程检查未通过：" + name + "; active_pids=" + str(active) +
            "; unreadable_pids=" + str(unreadable) + "; 原因=" +
            json.dumps({pid: PROCESS_DETAILS.get(pid, "") for pid in active + unreadable}, ensure_ascii=False))


def no_old_processes(auditor, evidence):
    """确认后和清理期间再次检查，避免用户停留在确认提示时新启动旧实验。"""
    for name in GROUPS:
        scripts = {item["name"] for item in evidence[name].get("scripts", [])}
        active, unreadable = auditor.process_references(name, scripts)
        if active or unreadable:
            raise RuntimeError(process_block_message(name, active, unreadable))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="核验后交互确认才执行删除")
    args = parser.parse_args()
    if os.name != "posix" or not Path("/proc").is_dir() or os.getuid() != 998:
        raise RuntimeError("仅供原SY账号在指定Linux服务器执行，不使用sudo，不在Windows删除")
    import pwd
    if pwd.getpwuid(os.getuid()).pw_name != "SY":
        raise RuntimeError("账号不匹配，未删除")
    for path in (PROJECT, PROJECT / "serial_runs", PROJECT / "cleanup_audits", AUDIT):
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or path.is_symlink() or path.resolve() != path
                or info.st_uid != os.getuid() or info.st_dev != PROJECT.stat().st_dev):
            raise RuntimeError("固定项目或审计路径异常，未删除：" + str(path))
    source = PROJECT / "audit_uam_old_final_only.py"
    if source.is_symlink() or digest(source.read_bytes()) != AUDITOR_SHA:
        raise RuntimeError("原只读审计脚本已变化，停止；不换用未经核验代码")
    spec = importlib.util.spec_from_file_location("verified_uam_auditor", source)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    # 原审计文件和其哈希保持不动；仅在本进程中修正/proc采样的退出竞态。
    auditor.process_references = precise_process_references
    auditor.file_info(PACKAGE, AUDIT, os.getuid(), AUDIT.stat().st_dev)
    candidates, all_rows, proofs, evidence, summary = load_package(PACKAGE)
    validate_manifest(candidates, all_rows)
    lock = safe_open(AUDIT / "DELETE_VERIFIED.lock", os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = AUDIT / "DELETE_VERIFIED.jsonl"
        original = {row["path"]: row for row in candidates}
        intents = prior_intents(journal, original)
        remaining, missing = [], []
        for row in candidates:
            path = Path(row["path"])
            if not path.exists() and not path.is_symlink():
                if row["path"] not in intents:
                    raise RuntimeError("文件在本工具清理前已消失，需重新核对：" + str(path))
                missing.append(row)
            else:
                current_fingerprint(row, auditor)
                remaining.append(row)
        for row in all_rows:
            if row["action"] == "KEEP_FINAL":
                current_fingerprint(row, auditor)
        for row in proofs:
            if row["experiment"] not in GROUPS:
                continue
            path = Path(row["server_record"])
            root = PROJECT / "serial_runs" / row["experiment"]
            auditor.file_info(path, root, os.getuid(), root.stat().st_dev)
            if digest(path.read_bytes()) != row["server_sha256"]:
                raise RuntimeError("实验归属记录已变化，未删除：" + str(path))
        # 再次运行同版只读归属判断，核验最终模型、K2K3源码/计划及活动进程。
        fresh_rows = []
        for name in GROUPS:
            result, rows, _, issues = auditor.audit_run(PROJECT, name, evidence[name])
            if issues:
                raise RuntimeError("路径仍有待查项，未删除：" + name + "; " + str(issues[:3]))
            if result.get("active_pids") or result.get("unreadable_processes"):
                raise RuntimeError(process_block_message(name, result.get("active_pids", []),
                                                        result.get("unreadable_processes", [])))
            fresh_rows.extend(rows)
            print("[复核通过]", name, flush=True)
        check_current_candidates(fresh_rows, remaining, missing)
        parents = {str(Path(row["path"]).parent): Path(row["path"]).parent.stat() for row in remaining}
        print("\n完整逐文件清单：", AUDIT / "拟删除逐文件清单.csv")
        print("原清单27386份；本次剩余", len(remaining), "份；先前已删除", len(missing), "份。")
        print("本次逻辑GiB：", round(sum(int(row["bytes"]) for row in remaining) / 1024**3, 3))
        print("保留1436份最终模型配套和全部待查/结果数据；删除的中间模型不能再从DATAONLY包恢复。")
        if not args.apply:
            print("只复核，未删除；需要--apply并交互确认。")
            return
        if not remaining:
            print("同清单已全部处理，无需再次删除。")
            return
        if not os.isatty(0) or input("确认请原样输入 DELETE 27386：").strip() != "DELETE 27386":
            raise RuntimeError("未取得交互确认，本次未删除")
        no_old_processes(auditor, evidence)
        descriptor = safe_open(journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        removed = 0
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            for row in remaining:
                current_fingerprint(row, auditor)
                path = Path(row["path"])
                folder = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    expected = parents[str(path.parent)]
                    actual = os.fstat(folder)
                    if (actual.st_dev, actual.st_ino, actual.st_uid) != (expected.st_dev, expected.st_ino, os.getuid()):
                        raise RuntimeError("删除前父目录发生变化，停止：" + str(path.parent))
                    info = os.stat(path.name, dir_fd=folder, follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode) or any(
                            getattr(info, attribute) != int(row[key]) for key, attribute in FINGERPRINT.items()):
                        raise RuntimeError("删除前文件指纹变化，停止：" + str(path))
                    handle.write(json.dumps(dict(package_sha256=PACKAGE_SHA, action="intent", row=row,
                                                 process_protocol=PROCESS_PROTOCOL),
                                            ensure_ascii=False) + "\n")
                    handle.flush()
                    # 只删除经过确认的单个文件；不删除目录、JSON、CSV或最终模型。
                    os.unlink(path.name, dir_fd=folder)
                    removed += 1
                finally:
                    os.close(folder)
                if removed % 1000 == 0:
                    os.fsync(handle.fileno())
                    print("[已删除]", removed, "/", len(remaining), flush=True)
                    no_old_processes(auditor, evidence)
            os.fsync(handle.fileno())
        for row in all_rows:
            if row["action"] == "KEEP_FINAL":
                current_fingerprint(row, auditor)
        end = dict(status="COMPLETE", deleted_this_run=removed, previously_deleted=len(missing),
                   package_sha256=PACKAGE_SHA, final_files_retained=1436,
                   process_protocol=PROCESS_PROTOCOL, delete_code_sha256=digest(Path(__file__).read_bytes()),
                   logical_bytes_deleted_this_run=sum(int(row["bytes"]) for row in remaining))
        end_path = AUDIT / ("DELETE_END_" + time.strftime("%Y%m%d_%H%M%S") + "_" + str(os.getpid()) + ".json")
        with end_path.open("x", encoding="utf-8") as handle:
            json.dump(end, handle, ensure_ascii=False, indent=2)
        print("[完成]", json.dumps(end, ensure_ascii=False))
        print("[审计记录]", end_path)
    finally:
        os.close(lock)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit("停止：" + type(error).__name__ + "：" + str(error) +
                         "。未扩大范围；若已进入删除阶段，已完成部分记录在DELETE_VERIFIED.jsonl。")
