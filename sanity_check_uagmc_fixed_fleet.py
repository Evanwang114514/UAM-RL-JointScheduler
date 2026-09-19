# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utilss.make_env_fleet import make_env

ROOT = Path(__file__).resolve().parent


def find_scenario(env: Any):
    obj = env
    seen = set()

    for _ in range(40):
        if id(obj) in seen:
            break
        seen.add(id(obj))

        if all(
            hasattr(obj, key)
            for key in (
                "person_travel_records",
                "persons",
                "finished_ids",
                "vertiports",
                "_all_evtols",
            )
        ):
            return obj

        if hasattr(obj, "scenario"):
            nxt = getattr(obj, "scenario")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue

        if hasattr(obj, "env"):
            nxt = getattr(obj, "env")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue

        if hasattr(obj, "unwrapped"):
            nxt = getattr(obj, "unwrapped")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue

        break

    raise RuntimeError(
        "Cannot locate UAGMC Scenario. "
        f"Stopped at {type(obj).__module__}.{type(obj).__name__}"
    )


def unwrap_step(step_out):
    if not isinstance(step_out, tuple):
        raise RuntimeError(f"Unexpected step return type: {type(step_out)}")

    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        return obs, reward, bool(terminated), bool(truncated), info

    if len(step_out) == 4:
        obs, reward, done, info = step_out
        return obs, reward, bool(done), False, info

    raise RuntimeError(f"Unexpected step tuple length: {len(step_out)}")


def waiting_decision_count(env: Any, scenario: Any) -> int:
    obj = env
    seen = set()

    for _ in range(30):
        if id(obj) in seen:
            break
        seen.add(id(obj))

        state = getattr(obj, "state", None)
        if isinstance(state, dict):
            waiting = state.get("waiting_decisions")
            if waiting is not None:
                try:
                    return len(waiting)
                except Exception:
                    pass

        waiting = getattr(obj, "waiting_decisions", None)
        if waiting is not None:
            try:
                return len(waiting)
            except Exception:
                pass

        if hasattr(obj, "env"):
            nxt = getattr(obj, "env")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue

        break

    waiting = getattr(scenario, "waiting_decisions", None)
    if waiting is not None:
        try:
            return len(waiting)
        except Exception:
            pass

    return 0


def run_one(fleet_size: int, passenger_file: Path, max_time: int, seed: int):
    env = make_env(
        max_time=max_time,
        log_dir=ROOT / "logs" / "fixed_fleet_sanity",
        env_index=fleet_size,
        person_spawn_file=str(passenger_file),
        candidate_from_vertiports=[0, 1],
        to_vertiport=2,
        enable_logger=False,
        fleet_mode="conserved_closed_loop",
        fleet_size=fleet_size,
        fleet_assertions=True,
    )()

    try:
        env.reset(seed=seed)
        scenario = find_scenario(env)

        if not hasattr(scenario, "get_fixed_fleet_diagnostics"):
            raise RuntimeError(
                "Located object is not ConservedFleetScenario: "
                f"{type(scenario).__module__}.{type(scenario).__name__}"
            )

        initial_ids = set(scenario._all_evtols.keys())
        initial_home = dict(getattr(scenario, "_fixed_home", {}))

        if len(initial_ids) != fleet_size:
            raise AssertionError(
                f"Initial fleet count {len(initial_ids)} != {fleet_size}"
            )

        if set(initial_home.keys()) != initial_ids:
            raise AssertionError(
                "Initial home mapping does not cover the fixed fleet exactly."
            )

        terminated = False
        truncated = False
        steps = 0
        action_toggle = 0

        while not (terminated or truncated):
            if waiting_decision_count(env, scenario) > 0:
                action = action_toggle % 2
                action_toggle += 1
            else:
                action = 0

            step_out = env.step(action)
            _, _, terminated, truncated, _ = unwrap_step(step_out)

            current_ids = set(scenario._all_evtols.keys())
            if current_ids != initial_ids:
                raise AssertionError(
                    "Fixed-fleet ID set changed during episode. "
                    f"missing={sorted(initial_ids-current_ids)}, "
                    f"added={sorted(current_ids-initial_ids)}"
                )

            if len(current_ids) != fleet_size:
                raise AssertionError(
                    f"Fleet count changed: {len(current_ids)} != {fleet_size}"
                )

            if dict(getattr(scenario, "_fixed_home", {})) != initial_home:
                raise AssertionError("Aircraft home mapping changed during episode.")

            scenario._assert_fixed_fleet()

            steps += 1
            if steps > max_time + 50:
                raise RuntimeError(
                    f"Sanity loop exceeded horizon: steps={steps}, max_time={max_time}"
                )

        diag = scenario.get_fixed_fleet_diagnostics()

        if int(diag["fleet_size"]) != fleet_size:
            raise AssertionError(
                f"Diagnostic fleet size {diag['fleet_size']} != {fleet_size}"
            )

        if int(diag.get("spawned_at_reset", -1)) != fleet_size:
            raise AssertionError(
                "Fixed fleet was not created exactly once at reset: "
                f"{diag.get('spawned_at_reset')} != {fleet_size}"
            )

        return diag

    finally:
        try:
            env.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="UAGMC conserved fixed-fleet sanity test"
    )
    parser.add_argument("--fleet-sizes", default="4,6,8,10")
    parser.add_argument(
        "--passenger-file",
        default="train_data/passengers_300.csv",
    )
    parser.add_argument("--max-time", type=int, default=600)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    passenger_file = Path(args.passenger_file).expanduser()
    if not passenger_file.is_absolute():
        passenger_file = (ROOT / passenger_file).resolve()

    if not passenger_file.exists():
        raise FileNotFoundError(f"Passenger file not found: {passenger_file}")

    sizes = [
        int(x.strip())
        for x in str(args.fleet_sizes).split(",")
        if x.strip()
    ]

    if not sizes or any(x <= 0 for x in sizes):
        raise ValueError(f"Invalid fleet sizes: {sizes}")

    print("=" * 120)
    print("UAGMC FIXED-FLEET SANITY")
    print("=" * 120)
    print(f"Passenger file : {passenger_file}")
    print(f"Fleet sizes    : {sizes}")
    print(f"Seed           : {args.seed}")
    print(f"Max time       : {args.max_time}")
    print("=" * 120)

    for size in sizes:
        d = run_one(size, passenger_file, args.max_time, args.seed)
        print(
            f"fleet={size:>3} | "
            f"allocation={d.get('initial_allocation')} | "
            f"service_arrivals={d.get('service_arrivals')} | "
            f"repo_dep={d.get('reposition_departures')} | "
            f"repo_arr={d.get('reposition_arrivals')} | "
            f"states={d.get('state_counts')} | "
            f"locations={d.get('current_location_counts')} | PASS"
        )

    print("=" * 120)
    print("ALL FIXED-FLEET SANITY CHECKS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()
