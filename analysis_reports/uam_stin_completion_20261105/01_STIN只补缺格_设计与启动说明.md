# STIN只补缺格：原环境与完整周期环境

2026年11月5日。只交付代码与补齐设计，没有启动本地或服务器训练。

## 先说核查结果

| 环境与控制 | 已完成什么 | 本轮处理 |
| --- | --- | --- |
| 原环境E0，乘客学习、飞机沿用源环境规则 | 没有找到与近期STIN合同匹配的开启／关闭四种子成对结果 | 补8格 |
| 完整周期E1，乘客学习、飞机按最长队列规则调回 | 原60M只有普通网络的单侧参照，不是STIN开启／关闭 | 补8格 |
| 完整周期E1，乘客和飞机都学习，简称J3联合控制 | 开／关×种子1901—1904，8格均已满训及完成评估 | 只读核查，**不重训** |

已经核对本地`uam_stage_stin_24run_60m_v1_DATAONLY.tar.gz`中的训练结束、评估结束、模型说明、结果和四条额外顺序验证记录。旧8格每格均为2,437,120步；并不是只看到总状态就宣布完成。服务器最终模型是否仍在，由新脚本启动时读取原目录再核查；DATAONLY没有模型，不能据此保证模型文件存在。

“完整周期”指近期固定12架飞机的严格E1版本，不是更早40架机队的S3。本轮不扩到2／4机场，不加等待动作、预测、奖励消融或新算法。

## 新增训练到底是什么

脚本：`train_uam_stin_E01_completion_ONEFILE.py`。

原环境先跑，再跑完整周期环境。每个环境的四个种子内配对随机排列STIN开启和关闭，保持同一训练长度、归一化、奖励、观测、PPO参数与评估顺序。

- 新增16格，每格2,437,120步，共38,993,920真实环境步；上限40M，余量仅容纳中断尾段，不追加方法。
- 原来的联合8格只是核查，不计作本轮新训练。若记录缺失或合同不符，明确报告缺口，不为填表自动再训八格。
- 历史STIN的每格步数不缩短。不能拿旧长训练的中途存档冒充本轮短训练结果。
- STIN关闭仍读取同样六帧，保留逐帧编码与平均聚合，只旁路位置编码和跨帧Transformer块；不是删除历史信息。
- 两臂都完整初始化同一结构、分配相同参数，dropout沿用旧60M的0，避免把采样噪声差异当机制差异；关闭版的部分参数不参与前向，因此实际计算量不相同。
- 沿用近期5环境×4096步、batch512、10轮、gamma=.99、整段线性学习率、PT_POST人时奖励等真实旧合同，不混用10月初另一套冻结配置。

这里的E0是源环境物理下的受控消融。为了和近期结果匹配，继续使用上述奖励与实现兼容处理，**不宣称忠实复刻原论文某张消融表的数字或CLS网络**。

## 评估与完整性

沿用500分钟训练边界、600分钟评估边界。主结果仍是每个模型后20%固定存档的四条验证顺序；6501—6504只作额外固定顺序验证，已经被查看，不能再称未触碰最终测试。

保留完整曲线、最终逐乘客／飞机事件、完成率和全体截止人时成本。只有全员完成时才填完整ATT；未完成不能只挑已完成者平均。不开新的深分支诊断，避免这次“补消融”变成另一套大诊断。

新增结果保存到：`serial_runs/uam_stin_E01_completion_16run_v1`。

主要文件：`progress.json`、`STIN_COVERAGE.json`、`LEGACY_JOINT_AUDIT.json`、`MODEL_RESULTS.csv`、`PAIRED_EFFECTS.csv`、逐格模型与归一化。

训练完整、评估缺失时，只补评估。已有完整格直接跳过；每格异常独立保留，继续其他格。恢复模型、优化器、归一化与随机状态，环境从新episode重启，不声称逐位连续。不会因为GPU某一刻利用率高就退出整套，但启动前仍须用户查看共享资源。

预计约8—11小时；来自历史STIN约1300步/秒量级与关闭版、单侧控制更轻的粗估，不是本次实测。单会话上限18小时，中断后原命令加`--resume --retry-failed`即可接续。与热点组并行会更慢，预计时间不包含竞争带来的额外开销。

## 五段操作命令

### 1．Windows CMD上传

```cmd
cd /d "E:\Study Files\github\UAM-predict\UAGMC-main"
scp -O -P 10111 "train_uam_stin_E01_completion_ONEFILE.py" SY@22.tcp.vip.cpolar.cn:/home/SY/uam/UAGMC-main/
```

### 2．服务器直接正式运行

先查看`top`和`nvidia-smi`。不另设smoke test，不修改已有环境，不安装新依赖。脚本内含原子保存、合同检查、单格失败隔离和断点恢复。

```bash
cd /home/SY/uam/UAGMC-main
top -b -n 1 | head -n 12
nvidia-smi
```

确认资源允许后：

```bash
cd /home/SY/uam/UAGMC-main
mkdir -p /home/SY/wyw/logs
if pgrep -u "$(id -u)" -f '[t]rain_uam_stin_E01_completion_ONEFILE[.]py([[:space:]]|$)' >/dev/null; then
    echo "STIN补齐组已在运行，不重复启动"
else
    nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        /home/SY/wyw/conda/envs/uam_server/bin/python -B -u \
        /home/SY/uam/UAGMC-main/train_uam_stin_E01_completion_ONEFILE.py \
        --resume --retry-failed --device cuda --eval-device cpu --wall-hours 18 \
        >> /home/SY/wyw/logs/uam_stin_completion.log 2>&1 < /dev/null &
    echo "STIN补齐父进程PID：$!"
fi
```

### 3．服务器完成后打包

脚本收尾自动打包。若只需重新生成包，不会重新训练：

```bash
cd /home/SY/uam/UAGMC-main
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/SY/wyw/conda/envs/uam_server/bin/python -B -u \
    train_uam_stin_E01_completion_ONEFILE.py --resume --phase pack
ls -lh experiment_archives/uam_stin_E01_completion_16run_v1*tar.gz
```

`DATAONLY`包含本轮结果和旧联合八格的只读核验记录；`FINAL_REPLAY_MODELS`包含新增16格最终模型和配套归一化。旧联合八格模型继续保留在原目录，不重复拷贝或删除。

### 4．Windows CMD下载

```cmd
cd /d "E:\Study Files\github\UAM-predict\UAGMC-main"
if not exist "experiment_archives" mkdir "experiment_archives"
scp -O -P 10111 SY@22.tcp.vip.cpolar.cn:/home/SY/uam/UAGMC-main/experiment_archives/uam_stin_E01_completion_16run_v1_DATAONLY.tar.gz "experiment_archives/"
scp -O -P 10111 SY@22.tcp.vip.cpolar.cn:/home/SY/uam/UAGMC-main/experiment_archives/uam_stin_E01_completion_16run_v1_FINAL_REPLAY_MODELS.tar.gz "experiment_archives/"
```

### 5．本地包地址

都在`E:\Study Files\github\UAM-predict\UAGMC-main\experiment_archives`内。本轮只需把DATAONLY交回分析；需要后续模型回放时再附最终模型包。不会下载到Windows默认下载目录。

## 验证与工具披露

代码交付只做标准库合同、来源hash与语法检查，没有运行UAM、Torch/JAX推理、smoke或训练。实验设计规范用于配对种子、开关对照和正确区分训练复制与多次回放；运行安全规范用于独立目录、显式设备、日志与恢复，不代表服务器资源已获实时验证。

工具参考：[Kassis等，Scientific Agent Skills，2026](https://doi.org/10.48550/arXiv.2609.00065)。本轮没有新增外部文献主张或方法。
