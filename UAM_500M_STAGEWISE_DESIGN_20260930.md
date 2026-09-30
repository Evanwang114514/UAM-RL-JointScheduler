# UAM 500M 分阶段实验设计与运行合同（2026-09-30）

## 结论先行

这不是一次预先填满 500M 的固定矩阵。代码把 500M 作为**累计实际训练步数上限**，先做冻结锚点，再围绕 S、AC、S+AC 的已知失败机制筛选，之后才给 J 结构与正式长训预算。没有通过筛选的方向不自动吃完预算。WM 不再是第四条主线。

现有 50M `PRE500M_FINAL_FREEZE.json` 尚未在本地仓库找到。正式训练因此被代码禁止；必须先拿到 F3–F8 完整结果，确认 load、batch、demand protocol、J4 物理语义与评估边界。还须先预注册 `FINAL_TEST_BANK.json`（每个 trace 的绝对路径与 SHA256），脚本仅核验文件身份并锁入 suite 合同，训练/选模不会读取其乘客内容；训练/验证与该 bank 的路径不能重合。已参与 F5 选择的所谓 test traces 在本编排器中一律称 **validation traces**。

## 文献到实验的对应关系

| 研究问题 | 文献启发 | 本项目实现与严格边界 | 与谁比较 |
|---|---|---|---|
| S 进入好 basin 后漂移 | [SPR 的 EMA target encoder](https://arxiv.org/abs/2007.05929)、[V-JEPA 的 masked latent 表示](https://openreview.net/forum?id=WFYbBOEOtv) | `S_EMA` 对当前已观测 committed-event latent 做慢速教师蒸馏；`S_MASKED_EVENT_EMA` 只掩盖**当前已知事件**后恢复其 latent。不是原文完整复现，不预测未揭示乘客或未来策略动作 | `S_TDM_EVENT_FUSION`、`SHARED` |
| AC 逆动力学标签噪声 | [ICM 逆动力学可控表示](https://proceedings.mlr.press/v70/pathak17a.html)、[动作冗余研究](https://proceedings.mlr.press/v161/baram21a.html) | `AC_RELEVANT_ICM` 从真实 step 后 `v5_rl_choice_active` 监督逆动力学：飞机选择真实生效时学四类；未生效时只学乘客两类。该标签**只用于训练后的辅助损失**，绝不注入在线 actor 观测 | `A_ICM_AC` |
| S+AC 更新互扰 | [PCGrad 指出的梯度冲突问题](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)、[PPG 的策略/辅助阶段隔离](https://proceedings.mlr.press/v139/cobbe21a.html) | `SA_ALTERNATE` 隔 rollout 开启 ICM 辅助更新，`SA_EMA_RELEVANT`/`SA_MASKED_RELEVANT` 测试稳定的 S 辅助与有效 AC 目标的组合。**当前没有声称实现 PCGrad 或完整 PPG**；若后续 gradient-cosine 诊断支持冲突，再另立精确 PCGrad 对照 | `SA_TDMFUSION_ICM`、S-only、AC-only |
| Joint 依赖与 credit | [MAPPO 的 centralized critic 经验](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html)、[MAT 的自回归联合动作](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html)、[2026 稀疏动作依赖](https://proceedings.mlr.press/v337/ding26a.html)、[2025 多智能体顺序效应分解](https://proceedings.mlr.press/v267/triantafyllou25a.html) | `C1` 对同一表示匹配 J0/J3/J4；`J4C_S`/`J4C_AC`/`J4C_SA` 是额外条件式 J4 actor，飞机分支条件于乘客动作，critic 与物理保持原样。只是启发式迁移，不叫完整 MAPPO/MAT | 原 J4、J3、J0 |

**特别警告：**现有 V5 的 `v5_rl_choice_active` 是**执行后**才知道的事实，不能未经论证就拿它在执行前给 J4 actor 做 action mask。这个实现只在辅助逆动力学中用它，避免未来泄漏和先知式 credit。若 F6 后希望做真正 PPO relevance mask，先要证明决策前可从现有观测准确判定同一等价类；否则不能做。

[ACAC（ICML 2025）](https://proceedings.mlr.press/v267/jung25a.html) 针对异步 macro-action 历史与 centralized critic 的对齐，提供了更深的 J 研究方向，但其异步执行设定不等于本项目的 V5 单时刻联合动作。故当前代码**没有**冒充 ACAC 复现或改 GAE；如未来要引入，需先定义 passenger/aircraft 各自真实 decision event、effect event 和独立轨迹，再做 matched 对照。[2025 的 JEPA-for-RL 研究](https://arxiv.org/abs/2504.16591) 也说明 latent 辅助值得筛查，但本代码只采用已知 committed-event 的当前掩码蒸馏，不做未来 observation 的模型滚动预测。

**撞车边界：**[IJCAI 2025 的 Asynchronous Credit Assignment](https://www.ijcai.org/proceedings/2025/20) 已提出 virtual synchrony 与动作交互价值分解；[UAI 2026 的稀疏动作依赖](https://proceedings.mlr.press/v337/ding26a.html) 已给出动作条件策略的理论框架。因此不能把“异步信用分配”或“乘客先、飞机后条件决策”本身写成原创。本文可验证的具体增量必须落在 UAM 已承诺事件的**生效时间对齐、物理相关性及训练稳定性**的耦合上。当前 V5 两个控制位仍在同一决策步输出，所谓 J4C 只是因子分解对照，不是物理异步 MARL 算法。

## 预算与次序

| 阶段 | 上限 | 作用 | 固定首批计划 |
|---|---:|---|---:|
| A | 30M | 六个冻结锚点，四个 paired replicate | 29.4912M |
| B | 105M | S/AC/S+AC 修复筛选→晋级→长训 | B1=7.3728M；B2/B3 依验证结果选拔 |
| C | 85M | J0/J3/J4/条件式 J4 的结构对照→晋级 | C1=14.7456M；C2/C3 依结果选拔 |
| D | 240M | 最多七个 finalist 的 10-seed 长训、最多两个加独立 seed | D/D2 需选拔文件 |
| RESERVE | 40M | 异常重跑、明确机制诊断、必要的确认 | 不自动使用 |

脚本只在请求的阶段运行，阶段失败不会自动切到下一个阶段。B2/B3/C2/C3/D/D2 需要研究者先查看完整 learning curve 后填写 `uam500m_selection_example.json` 的副本。选拔标准预注册为：独立训练 replicate 的 late-window ATT 优先，cross-seed CV 理想 ≤10%，mean Best→Late degradation 理想 ≤10%，单 seed catastrophic rebound 尽量 <20%，且所有 episode 全完成。Best 仅证明 reachability。heldout trace/评估 seed 不计作独立训练样本。

晋级阶段会自动附带**同 seed、同训练长度**的对照：B2 带 S/AC/S+AC 原版；B3/C2/C3 带每个候选的最近原版；D 带 UAGMC_SOURCE、CURRENT、SHARED 及各 finalist 的原版；D2 带该候选的最近原版。D 最多三个新候选，并要求“候选+对照”总数不超过七个。这样不把不同 horizon 的历史模型当成正式 paired baseline。

完整 F3–F8 冻结报告中的 `frozen_ppo` 会决定 10×2048 或其他已选 profile、batch、load 和 demand；脚本不会按方法单独调这些参数，也不会修改 `-N_active*Δt`、T2、40 架、S3 充电/周转/pad 物理。当前候选 `10×2048` 不是硬编码结论。

## Windows CMD：先审计划与烟测

```cmd
cd /d "E:\Study Files\github\UAM-predict\UAGMC-main"
conda activate uam5070
python train_uam_500m_stagewise.py --stage A --plan-only
python train_uam_500m_stagewise.py --smoke --smoke-rollouts 3 --device cuda --freeze "路径\PRE500M_FINAL_FREEZE.json" --suite-root "serial_runs\uam500m_stagewise_MAIN"
```

最终测试集清单格式为 `{"traces":[{"path":"E:\\...\\final_seed900.csv","sha256":"该文件的64位SHA256"}]}`，先用 CMD 的 `certutil -hashfile "E:\...\final_seed900.csv" SHA256` 获取哈希。该文件应在任何正式方法选择前写定并保存；**不能把 F5 已看过的 trace 填进去**。以下正式命令均需追加 `--final-test-bank "路径\FINAL_TEST_BANK.json"`。

`--smoke` 跑 3 个完整 global rollout，单独保存在 `smoke` 子目录，不计正式账本。正式训练前必须在**同一冻结文件、同一 profile、同一 suite-root**获得完整训练吞吐 ≥1000 SPS；否则自动拒绝正式运行。先在共享服务器执行 `top` 与 `nvidia-smi`，选空闲 GPU，自己 home 下运行；代码绝不调用 sudo、killall 或 GPU reset。服务器上的文件路径需要确保冻结 JSON 与 demand trace 全部位于自己的目录。

## Windows CMD：正式分阶段执行

```cmd
python train_uam_500m_stagewise.py --stage A --freeze "路径\PRE500M_FINAL_FREEZE.json" --final-test-bank "路径\FINAL_TEST_BANK.json" --suite-root "serial_runs\uam500m_stagewise_MAIN" --device cuda
python train_uam_500m_stagewise.py --stage A --freeze "路径\PRE500M_FINAL_FREEZE.json" --final-test-bank "路径\FINAL_TEST_BANK.json" --suite-root "serial_runs\uam500m_stagewise_MAIN" --evaluate-only
python train_uam_500m_stagewise.py --stage B1 --freeze "路径\PRE500M_FINAL_FREEZE.json" --final-test-bank "路径\FINAL_TEST_BANK.json" --suite-root "serial_runs\uam500m_stagewise_MAIN" --device cuda
python train_uam_500m_stagewise.py --stage B1 --freeze "路径\PRE500M_FINAL_FREEZE.json" --final-test-bank "路径\FINAL_TEST_BANK.json" --suite-root "serial_runs\uam500m_stagewise_MAIN" --evaluate-only
python train_uam_500m_stagewise.py --stage C1 --freeze "路径\PRE500M_FINAL_FREEZE.json" --final-test-bank "路径\FINAL_TEST_BANK.json" --suite-root "serial_runs\uam500m_stagewise_MAIN" --device cuda
python train_uam_500m_stagewise.py --stage C1 --freeze "路径\PRE500M_FINAL_FREEZE.json" --final-test-bank "路径\FINAL_TEST_BANK.json" --suite-root "serial_runs\uam500m_stagewise_MAIN" --evaluate-only
```

之后复制 `uam500m_selection_example.json` 为新的选择文件，**按验证曲线填写** B2/B3/C2/C3/D/D2。晋级命令示例：

```cmd
python train_uam_500m_stagewise.py --stage B2 --selection-file "我的晋级选择.json" --freeze "路径\PRE500M_FINAL_FREEZE.json" --final-test-bank "路径\FINAL_TEST_BANK.json" --suite-root "serial_runs\uam500m_stagewise_MAIN" --device cuda
python train_uam_500m_stagewise.py --stage D --selection-file "我的晋级选择.json" --freeze "路径\PRE500M_FINAL_FREEZE.json" --final-test-bank "路径\FINAL_TEST_BANK.json" --suite-root "serial_runs\uam500m_stagewise_MAIN" --device cuda
```

同一命令重跑会跳过已完整 cell；失败 cell 默认跳过，可加 `--retry-failed` 从最近**模型+VecNormalize 成对 checkpoint**继续。此续训保留 PPO 模型、PPO optimizer 与归一化器；新加的辅助 optimizer 暂未单独序列化，会在恢复时重建，环境 episode 也从头开始，故只是统计续训，不承诺 bitwise 相同。两次相同错误或训练吞吐低于阈值会停止整批，避免系统性故障浪费算力。训练每 50k 保存；`--evaluate-only` 在训练结束后回放，不中途打断 GPU 训练。

```cmd
python train_uam_500m_stagewise.py --stage A --freeze "路径\PRE500M_FINAL_FREEZE.json" --final-test-bank "路径\FINAL_TEST_BANK.json" --suite-root "serial_runs\uam500m_stagewise_MAIN" --device cuda --resume-suite --retry-failed
python train_uam_500m_stagewise.py --stage A --freeze "路径\PRE500M_FINAL_FREEZE.json" --final-test-bank "路径\FINAL_TEST_BANK.json" --suite-root "serial_runs\uam500m_stagewise_MAIN" --device cuda --resume-cell "A__J4__S_TDM_EVENT_FUSION__s401" --retry-failed
```

## 已执行的本地技术验证与仍未验证部分

- 两个 Python 文件 `py_compile` 通过，A 阶段计划能正确列出 24 个 cell、29.4912M。
- `SA_EMA_RELEVANT`、`S_MASKED_EVENT_EMA`、`J4C_S` 各完成 20,480 步的本地 V5/CUDA 烟测，没有形状或联合动作错误。
- 本地 RTX 5070 Laptop 的三次单-rollout完整训练 SPS 分别约 **786、835、857**；虽然 SB3 rollout 显示约 1004–1069 fps，但加上 PPO/辅助更新和启动开销后**尚未达到正式目标 1000 SPS**。不能声称已具备 500M 正式吞吐；需要在实际服务器用 3-rollout pilot 再判定。
- 50M 的完整冻结文件、跨 seed 结果、真正 untouched FINAL_TEST bank 均未在本地验证。正式训练被 fail-closed 阻止；当前代码不是论文结果，也不意味着新方法有效。

## 边界与下一次审查

B/C 阶段的 `summary.json` 是**每个训练 seed 一个**。论文汇总必须先对每个模型的多个 validation traces 求均值，再跨训练 seed 报 mean/SD/CV/worst/range、每条曲线的 late slope 与 Best→Late 回弹。D 之前还应复核纯 ICM 多 seed long-run、J4 credit relevance、F3 load 是否确实选在有辨别力的中高负载。

本设计不包含未经证实的 PCGrad、真实 predecision action mask、完整 JEPA、完整 MAT/MAPPO，也不把 post-step info 给执行时策略。上述候选若要进入 500M，只能作为后续单独审计的扩展，不可借此报告冒称已实现。
