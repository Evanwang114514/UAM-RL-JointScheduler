"""
Author: PangAY
Date: 2025-12-29
Purpose: 生成 UAM 测试用乘客数据 CSV 文件
         约束：每分钟最多只有一个乘客
"""

import csv
import random

# =========================
# 配置
# =========================
NUM_PASSENGERS = 100       # 生成乘客数量
MAX_SPAWN_TIME = 300       # 最大生成时间（分钟）
X_RANGE = (5, 10)         # 起点和终点 x 坐标范围
Y_RANGE = (0, 10)         # 起点和终点 y 坐标范围
OUTPUT_FILE = "passengers.csv"

assert NUM_PASSENGERS <= MAX_SPAWN_TIME + 1, \
    "NUM_PASSENGERS 不能大于 MAX_SPAWN_TIME + 1（否则无法保证每分钟最多1人）"

# =========================
# 生成乘客数据
# =========================
def generate_passenger_data(num_passengers: int):
    passengers = []

    # 🔑 关键：无放回采样时间
    spawn_times = random.sample(
        range(0, MAX_SPAWN_TIME + 1),
        num_passengers
    )
    spawn_times.sort()  # 可选：让 CSV 时间有序，方便调试

    for i, spawn_time in enumerate(spawn_times):
        pid = f"P{i}"

        origin_x = random.randint(*X_RANGE)
        origin_y = random.randint(*Y_RANGE)
        dest_x = random.randint(*X_RANGE)
        dest_y = random.randint(*Y_RANGE)

        passengers.append({
            "id": pid,
            "time": spawn_time,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "dest_x": dest_x,
            "dest_y": dest_y
        })

    return passengers

# =========================
# 写入 CSV
# =========================
def save_to_csv(passengers, filename: str):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["id", "time", "origin_x", "origin_y", "dest_x", "dest_y"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for p in passengers:
            writer.writerow(p)

    print(f"Generated {len(passengers)} passengers")
    print(f"Each minute has at most ONE passenger")
    print(f"Saved to {filename}")

# =========================
# 主函数
# =========================
if __name__ == "__main__":
    passengers = generate_passenger_data(NUM_PASSENGERS)
    save_to_csv(passengers, OUTPUT_FILE)
