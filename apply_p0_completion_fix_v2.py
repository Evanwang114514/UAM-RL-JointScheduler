# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

HARD_GUARD = 10_000


def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak_p0_v2")
    if not bak.exists():
        shutil.copy2(path, bak)


def write_if_changed(path: Path, old: str, new: str) -> bool:
    if old == new:
        return False
    backup(path)
    path.write_text(new, encoding="utf-8")
    return True


def patch_scenario(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text

    text = re.sub(
        r"(?m)(^\s*max_time:\s*int\s*=\s*)\d+(\s*,\s*$)",
        rf"\g<1>{HARD_GUARD}\g<2>",
        text,
        count=1,
    )

    if "random.seed(int(seed))" not in text:
        needle = '''    def reset(self, seed=None):
        super().reset(seed=seed)

        self.time = 0
        self.persons.reset()
'''
        replacement = '''    def reset(self, seed=None):
        super().reset(seed=seed)

        # Scenario/Person/eVTOL code uses Python's random module.
        if seed is not None:
            random.seed(int(seed))

        self.time = 0
        self.persons.reset()

        # Passenger IDs repeat across fixed-trace episodes.
        self.person_travel_records = {}
'''
        if needle not in text:
            raise RuntimeError("Scenario.reset layout changed; patch manually.")
        text = text.replace(needle, replacement, 1)

    text = re.sub(
        r"\n\s*for pid in self\.persons\.persons\.keys\(\):\n"
        r"\s*if pid not in self\.person_travel_records:\n"
        r"\s*self\.person_travel_records\[pid\] = \[\]\n",
        "\n",
        text,
        count=1,
    )

    if "active_count_start" not in text:
        needle = '        logger.info(f"===== STEP {self.time} START =====")\n\n        # 1. Spawn 新乘客\n'
        replacement = '''        logger.info(f"===== STEP {self.time} START =====")

        # Person-minutes in [t,t+1): already-generated unfinished passengers.
        active_count_start = sum(
            1
            for person in self.persons.persons.values()
            if str(getattr(person, "state", "")).lower() != "finished"
        )
        reward = -float(active_count_start)

        # 1. Spawn 新乘客
'''
        if needle not in text:
            raise RuntimeError("Scenario.step start marker changed.")
        text = text.replace(needle, replacement, 1)

    if 'getattr(person, "spawn_time", self.time)' not in text:
        old = '"start_time": self.time,\n                "end_time": None'
        new = '''"start_time": float(
                    getattr(person, "spawn_time", self.time)
                ),
                "end_time": None'''
        if old not in text:
            raise RuntimeError("Travel-record start_time assignment changed.")
        text = text.replace(old, new, 1)

    if "all_finished" not in text or "hard_guard_hit" not in text:
        pattern = re.compile(
            r"\n\s*# =========================\n"
            r"\s*# 6\. Reward\n"
            r"\s*# =========================\n"
            r".*?"
            r"\s*return self\.get_state\(\), reward, terminated, truncated, \{\}\n",
            re.DOTALL,
        )
        replacement = '''
        # =========================
        # 6. Natural completion / safety truncation
        # =========================
        # Passenger spawning is inclusive while
        # self.time <= passenger_generation_end_time.
        generation_done = self.time >= self.passenger_generation_end_time

        n_spawned = len(self.persons.persons)
        n_finished = len(set(self.finished_ids))

        all_finished = (
            generation_done
            and n_spawned > 0
            and n_finished >= n_spawned
            and len(self.waiting_decisions) == 0
        )

        terminated = bool(all_finished)

        # max_time is ONLY a runaway/deadlock safety guard.
        truncated = bool(
            self.time >= self.max_time
            and not terminated
        )

        info = {
            "generation_done": bool(generation_done),
            "n_spawned": int(n_spawned),
            "n_finished": int(n_finished),
            "completion_rate": (
                float(n_finished) / float(n_spawned)
                if n_spawned > 0
                else 0.0
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
        text2, n = pattern.subn(replacement, text, count=1)
        if n != 1:
            raise RuntimeError("Old Scenario reward/termination block not found.")
        text = text2

    return write_if_changed(path, original, text)


def patch_wrapper(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text

    if "self.env.reset(seed=seed)" not in text:
        old = '''    def reset(self, *, seed=None, options=None):
        self.state = self.env.reset()
'''
        new = '''    def reset(self, *, seed=None, options=None):
        # Propagate Gym/SB3 seed into Scenario.reset.
        self.state = self.env.reset(seed=seed)
'''
        if old not in text:
            raise RuntimeError("UAMRLWrapper.reset layout changed.")
        text = text.replace(old, new, 1)

    return write_if_changed(path, original, text)


def patch_make_env(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    text = re.sub(
        r"(?m)(^\s*max_time:\s*int\s*=\s*)\d+(\s*,\s*$)",
        rf"\g<1>{HARD_GUARD}\g<2>",
        text,
        count=1,
    )
    return write_if_changed(path, original, text)


def check(root: Path) -> int:
    scenario = root / "at_obj" / "scenario.py"
    wrapper = root / "utilss" / "uam_rl_wrapper.py"
    factory = root / "utilss" / "make_env_fleet.py"

    s = scenario.read_text(encoding="utf-8")
    w = wrapper.read_text(encoding="utf-8")
    f = factory.read_text(encoding="utf-8")

    tests = [
        ("natural completion", "all_finished" in s and "generation_done" in s),
        ("hard guard info", '"hard_guard_hit": bool(truncated)' in s),
        ("-N active reward", "reward = -float(active_count_start)" in s),
        ("spawn-time ATT start", 'getattr(person, "spawn_time", self.time)' in s),
        ("records cleared on reset", "self.person_travel_records = {}" in s),
        ("Python random seeded", "random.seed(int(seed))" in s),
        ("wrapper seed forwarded", "self.env.reset(seed=seed)" in w),
        ("Scenario guard=10000", bool(re.search(r"max_time:\s*int\s*=\s*10000", s))),
        ("factory guard=10000", bool(re.search(r"max_time:\s*int\s*=\s*10000", f))),
    ]

    print("=" * 78)
    print("P0 COMPLETION FIX CHECK")
    print("=" * 78)
    ok_all = True
    for name, ok in tests:
        print(f"{'OK' if ok else 'FAIL':<5} {name}")
        ok_all = ok_all and ok
    print("=" * 78)

    if ok_all:
        print("P0 source checks passed.")
        print("Start training in a NEW Python process after applying this patch.")
        return 0

    print("P0 source checks FAILED.")
    return 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    scenario = root / "at_obj" / "scenario.py"
    wrapper = root / "utilss" / "uam_rl_wrapper.py"
    factory = root / "utilss" / "make_env_fleet.py"

    for path in (scenario, wrapper, factory):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.check:
        return check(root)

    changed = []
    if patch_scenario(scenario):
        changed.append(str(scenario.relative_to(root)))
    if patch_wrapper(wrapper):
        changed.append(str(wrapper.relative_to(root)))
    if patch_make_env(factory):
        changed.append(str(factory.relative_to(root)))

    print("P0 patch applied.")
    if changed:
        print("Changed:")
        for name in changed:
            print("  -", name)
    else:
        print("No changes needed; source already looks patched.")

    print("Backups use suffix: .bak_p0_v2")
    print()
    return check(root)


if __name__ == "__main__":
    raise SystemExit(main())
