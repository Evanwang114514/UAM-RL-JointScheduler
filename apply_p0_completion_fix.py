# -*- coding: utf-8 -*-
# Apply P0 fixes: completion termination, ATT-aligned reward, reset seed propagation,
# request-time travel records, and 10000-min safety guards.

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

RUNNER_FILES = [
    "train_uagmc_45x800k_formal.py",
    "train_uagmc_45x800k_formal_JOINTFIX.py",
    "train_uagmc_6x6_800k.py",
    "train_uagmc_E0_E2_E6_effect_time_700k.py",
    "train_uagmc_E3_E6_obs_topology_matrix_800k_FORMAL.py",
    "train_uagmc_E3_E4_E5_serial_1m.py",
]


def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak_p0")
    if not bak.exists():
        shutil.copy2(path, bak)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n == 0:
        if new in text:
            return text
        raise RuntimeError(f"Cannot find expected block for {label}")
    if n != 1:
        raise RuntimeError(f"Expected one block for {label}, found {n}")
    return text.replace(old, new, 1)


def patch_scenario(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text

    text = re.sub(r"max_time:\s*int\s*=\s*450,", "max_time: int = 10000,", text, count=1)

    reset_old = '''    def reset(self, seed=None):
        super().reset(seed=seed)

        self.time = 0
        self.persons.reset()
'''
    reset_new = '''    def reset(self, seed=None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(int(seed))

        self.time = 0
        self.persons.reset()

        # IDs repeat for the fixed trace; never carry travel records across episodes.
        self.person_travel_records = {}
'''
    text = replace_once(text, reset_old, reset_new, "Scenario.reset")

    step_old = '''        logger.info(f"===== STEP {self.time} START =====")

        # 1. Spawn 新乘客
'''
    step_new = '''        logger.info(f"===== STEP {self.time} START =====")

        # Person-minutes in [t, t+1): every already-generated unfinished passenger counts.
        active_count_start = sum(
            1
            for person in self.persons.persons.values()
            if str(getattr(person, "state", "")).lower() != "finished"
        )
        reward = -float(active_count_start)

        # 1. Spawn 新乘客
'''
    text = replace_once(text, step_old, step_new, "reward insertion")

    reward_old = '''        # =========================
        # 6. Reward
        # =========================
        reward = 0.0
        if self.finished_ids:
            total_travel_time = 0.0
            for pid in self.finished_ids:
                last = self.person_travel_records[pid][-1]
                total_travel_time += (last["end_time"] - last["start_time"])

            avg_travel_time = total_travel_time / len(self.finished_ids)
            reward -= avg_travel_time


        terminated = False

        truncated = self.time >= self.max_time


        logger.info(f"[STEP END] time={self.time} reward={reward}")

        self.time += 1

        return self.get_state(), reward, terminated, truncated, {}
'''
    reward_new = '''        # =========================
        # 6. Episode termination
        # =========================
        # Spawn logic is inclusive (time <= passenger_generation_end_time).
        generation_done = self.time >= self.passenger_generation_end_time

        n_spawned = len(self.persons.persons)
        n_finished = len(set(self.finished_ids))

        all_finished = (
            generation_done
            and n_spawned > 0
            and n_finished >= n_spawned
            and len(self.waiting_decisions) == 0
        )

        # Natural task success: every generated passenger has completed.
        terminated = bool(all_finished)

        # max_time is ONLY a safety/deadlock guard.
        truncated = bool(self.time >= self.max_time and not terminated)

        info = {
            "generation_done": bool(generation_done),
            "n_spawned": int(n_spawned),
            "n_finished": int(n_finished),
            "completion_rate": (
                float(n_finished) / float(n_spawned)
                if n_spawned > 0 else 0.0
            ),
            "hard_guard_hit": bool(truncated),
        }

        logger.info(
            f"[STEP END] time={self.time} reward={reward} "
            f"finished={n_finished}/{n_spawned} "
            f"terminated={terminated} truncated={truncated}"
        )

        self.time += 1

        return self.get_state(), reward, terminated, truncated, info
'''
    text = replace_once(text, reward_old, reward_new, "reward/termination")

    start_old = '''                "start_time": self.time,
                "end_time": None
'''
    start_new = '''                "start_time": float(
                    getattr(person, "spawn_time", self.time)
                ),
                "end_time": None
'''
    text = replace_once(text, start_old, start_new, "travel start time")

    if text != orig:
        backup(path)
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_wrapper(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    old = '''    def reset(self, *, seed=None, options=None):
        self.state = self.env.reset()
'''
    new = '''    def reset(self, *, seed=None, options=None):
        # Propagate Gym/SB3 seed into Scenario.reset.
        self.state = self.env.reset(seed=seed)
'''
    text = replace_once(text, old, new, "UAMRLWrapper.reset")
    if text != orig:
        backup(path)
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_make_env(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    text = re.sub(r"max_time:\s*int\s*=\s*420,", "max_time: int = 10000,", text, count=1)
    if text != orig:
        backup(path)
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_runner(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    orig = text
    text, _ = re.subn(
        r"(?m)^MAX_TIME\s*=\s*(?:500|600)\s*$",
        "MAX_TIME = 10000",
        text,
        count=1,
    )
    if text != orig:
        backup(path)
        path.write_text(text, encoding="utf-8")
        return True
    return False


def check(root: Path) -> int:
    scenario = root / "at_obj" / "scenario.py"
    wrapper = root / "utilss" / "uam_rl_wrapper.py"
    s = scenario.read_text(encoding="utf-8")
    w = wrapper.read_text(encoding="utf-8")

    tests = [
        ("completion termination", "all_finished" in s and "generation_done" in s),
        ("active-count reward", "active_count_start" in s),
        ("request-time ATT start", 'getattr(person, "spawn_time"' in s),
        ("Scenario Python RNG seed", "random.seed(int(seed))" in s),
        ("wrapper seed propagation", "self.env.reset(seed=seed)" in w),
    ]

    print("P0 CHECK")
    ok_all = True
    for name, ok in tests:
        print(f"  {'OK' if ok else 'FAIL':<4} {name}")
        ok_all = ok_all and ok

    for rel in RUNNER_FILES:
        p = root / rel
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            m = re.search(r"(?m)^MAX_TIME\s*=\s*(\d+)\s*$", txt)
            if m:
                val = int(m.group(1))
                ok = val >= 10000
                print(f"  {'OK' if ok else 'FAIL':<4} {rel}: MAX_TIME={val}")
                ok_all = ok_all and ok

    return 0 if ok_all else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()

    if args.check:
        return check(root)

    changed = []
    targets = [
        (root / "at_obj" / "scenario.py", patch_scenario),
        (root / "utilss" / "uam_rl_wrapper.py", patch_wrapper),
        (root / "utilss" / "make_env_fleet.py", patch_make_env),
    ]

    for path, fn in targets:
        if not path.exists():
            raise FileNotFoundError(path)
        if fn(path):
            changed.append(str(path.relative_to(root)))

    for rel in RUNNER_FILES:
        p = root / rel
        if p.exists() and patch_runner(p):
            changed.append(rel)

    print("P0 patch applied.")
    if changed:
        print("Changed:")
        for x in changed:
            print("  -", x)
    else:
        print("No files changed (already patched or unmatched).")

    print("Backups use suffix .bak_p0")
    print("Run check:")
    print("  python apply_p0_completion_fix.py --root . --check")
    return check(root)


if __name__ == "__main__":
    raise SystemExit(main())
