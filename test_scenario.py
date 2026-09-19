import random
from policy.nearest_vertiport_policy import NearestVertiportPolicy
from policy.queue_aware_vertiport_policy import QueueAwareVertiportPolicy
from policy.simple_queue_policy import QueueOnlyVertiportPolicy
from at_obj.scenario import Scenario

# 不实用 loggur
import os
os.environ["LOGURU_AUTOINIT"] = "false"


import logging


def print_evtol_status(state):
    print("  [eVTOL status]")
    for ev_id, ev in state["evtols"].items():
        print(
            f"    eVTOL {ev_id}: "
            f"state={ev['state']}, "
            f"vertiport={ev['current_vertiport']}, "
            f"battery={ev['battery_kwh']:.1f}, "
            f"remaining_time={ev['remaining_time']}"
        )


def print_vertiport_status(state):
    print("  [Vertiport status]")
    for vid, v in state["vertiports"].items():
        print(
            f"    Vertiport {vid}: "
            f"waiting={v['wait_person']}, "
            f"queue={v['person_list']}"
        )

        # 输出该 vertiport 的 eVTOL 状态
        print("      EVTOLs at this vertiport:")
        for e in v["evtols"]:
            print(
                f"        eVTOL {e['id']}: "
                f"state={e['state']}, "
                f"battery_pct={e['battery_pct']:.1f}, "
                f"passengers={e['passengers']}, "
                #f"remaining_time={e['remaining_time']},"
                f"current_vertiport={e['current_vertiport']}"
            )



def print_person_status(state):
    print("  [Passenger status]")
    for pid, p in state["persons"].items():
        print(
            f"    Person {pid}: "
            f"state={p['state']}, "
            f"sub_state={p['sub_state']}, "
            f"method={p['method']}, "
            f"action={p['action']}"
        )

# passenger 状态更新 当分配好之后，更新状态要 1. 是不是到达了 vertiport 2. 是不是
def test_scenario_env():
    # ========================
    # 1. 初始化环境
    # ========================
    env = Scenario(max_time=600, person_spawn_file = 'train_data/passengers_200.csv')
    state = env.reset()
    #  策略
    policy = QueueOnlyVertiportPolicy(
        candidate_from_vertiports=[0, 1],
        to_vertiport=2,
        greedy_prob=1.0,
    )



    print("=" * 60)
    print("Initial state")
    print("=" * 60)
    print(f"Time: {state['time']}")
    print(f"Initial waiting passengers: {state['waiting_decisions']}")
    print(f"Number of vertiports: {len(state['vertiports'])}")
    print(f"Number of eVTOLs: {len(state['evtols'])}")
    print_evtol_status(state)
    print_vertiport_status(state)
    print("=" * 60)

    # ========================
    # 2. 推进仿真
    # ========================
    for t in range(1, 600): # 最大仿真时间 420 分钟 推进的这个要修改
        print("\n" + "#" * 60)
        print(f"STEP {t}")
        print("#" * 60)

        waiting = state["waiting_decisions"]

        actions = {}
        actions = policy.act(env, state)



        state, reward, terminated, done, info = env.step(actions)


        # ===== 推进 1 分钟 =====
        state, reward, terminated, done, info = env.step()

        # ===== 详细状态输出 =====
        print(f"\n[After step {t}] Time = {state['time']}")
        print(f"\n[After step {t}] reward = {reward}")
        print(f"Waiting decisions: {state['waiting_decisions']}")
        print(f"Finished passengers: {env.finished_ids}")

        print_person_status(state)
        print_vertiport_status(state)
        print_evtol_status(state)

        if done:
            print("\nSimulation reached max time.")
            break

    print("\n" + "=" * 60)
    print("Final state summary")
    print("=" * 60)
    print(f"Total finished passengers: {len(env.finished_ids)}")
    print(f"Remaining waiting passengers: {state['waiting_decisions']}")
    print_evtol_status(state)
    print_vertiport_status(state)


if __name__ == "__main__":
    test_scenario_env()
