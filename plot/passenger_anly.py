# %%
import re
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns

# ======================
# Plot style (same as yours)
# ======================
plt.style.use('ieee')
sns.set_palette('colorblind')

# ======================
# Log file
# ======================
log_file = "./scenario_passenger.log"

# ======================
# Regex
# ======================
stats_pattern = re.compile(
    r"\[PASSENGER_STATS\] pid=(\S+) "
    r"to_vertiport_time=(\d+) "
    r"wait_uam_time=(\d+) "
    r"fly_time=(\d+) "
    r"total=(\d+)"
)

# ======================
# Read log & aggregate by pid
# ======================
passenger_final = {}

with open(log_file, "r") as f:
    for line in f:
        m = stats_pattern.search(line)
        if m:
            pid, to_v, wait, fly, total = m.groups()
            # 每次覆盖，最终保留最后一次（完成后的）
            passenger_final[pid] = {
                "to_vertiport": int(to_v),
                "wait_uam": int(wait),
                "fly": int(fly),
                "total": int(total),
            }

print(f"Total passengers counted: {len(passenger_final)}")

# ======================
# Convert to numpy
# ======================
to_vert = np.array([v["to_vertiport"] for v in passenger_final.values()])
wait_uam = np.array([v["wait_uam"] for v in passenger_final.values()])
fly = np.array([v["fly"] for v in passenger_final.values()])
total = np.array([v["total"] for v in passenger_final.values()])

# 防御：避免除 0
total[total == 0] = 1

# ======================
# Compute ratios
# ======================
to_vert_ratio = to_vert / total
wait_ratio = wait_uam / total
fly_ratio = fly / total

# 取平均（240 个乘客）
avg_to_vert = to_vert_ratio.mean()
avg_wait = wait_ratio.mean()
avg_fly = fly_ratio.mean()

# ======================
# Plot: stacked bar (log-based)
# ======================
def plot_stacked_bar_log(ratios, labels, output_file):
    plt.figure(figsize=(5, 3.6))
    plt.ylabel("Ratio of Total Travel Time", fontsize=12)
    plt.xlabel("All Passengers", fontsize=12)

    bottom = 0
    for ratio, label in zip(ratios, labels):
        plt.bar(
            ["Passengers"],
            [ratio],
            bottom=bottom,
            label=label
        )
        bottom += ratio

    plt.ylim(0, 1)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

plot_stacked_bar_log(
    ratios=[avg_to_vert, avg_wait, avg_fly],
    labels=["To Vertiport", "Waiting for UAM", "Flying"],
    output_file="passenger_time_ratio_log.png"
)

# ======================
# Extra: Wait vs Total scatter (IEEE)
# ======================
plt.figure(figsize=(4.5, 3.5))
plt.scatter(total, wait_ratio, alpha=0.6)
plt.xlabel("Total Travel Time", fontsize=11)
plt.ylabel("Waiting Time Ratio", fontsize=11)
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("wait_ratio_vs_total.png", dpi=300)
plt.show()

# %%
