# 检索登记与阅读入口

2026-11-05｜迁移库2025—2026；诊断库包含必要早期基础。以下22项为本轮定向核验、并进入报告的去重文献清单，不是六数据库全量命中数。

完整交付：

1. [两类文献与源码核验](01_两类文献与源码核验.md)：来源、机制、真实源码入口、许可证/复现限制、采纳与排除理由。
2. [当前实验结束后的诊断与条件训练方案](02_当前实验结束后的诊断与条件训练方案.md)：每个假说怎样支持/否定、回放控制、条件训练与停止规则。

## 来源状态

检索技能的六库聚合入口自动寻找凭据文件，未执行。采用公开会议/期刊论文页、作者全文、官方仓库和源文件定向核验；不编造API命中数、相关度分数、被引量或被引作者排名。OpenReview直接论坛页多次出现浏览器验证，改用正式会议页、官方全文或期刊目录；部分原始文件读取失败已在方法卡标记。没有执行下载代码，不能声称完成复现。

## 一、迁移文献登记

| 编号 | 完整题目 | 发表版本 | 原始来源 | 处理 |
| --- | --- | --- | --- | --- |
| T1 | Mastering Diverse Domains through World Models | Nature 2025 | [作者仓库](https://github.com/danijar/dreamerv3) | 现有DreamerV3先分析 |
| T2 | Decentralized Transformers with Centralized Aggregation are Sample-Efficient Multi-Agent World Models | TMLR 2025；预印本2024 | [全文](https://arxiv.org/html/2406.15836)、[代码](https://github.com/breez3young/MARIE) | MARIE条件备选 |
| T3 | Revisiting Multi-Agent World Modeling from a Diffusion-Inspired Perspective | NeurIPS 2025 | [论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5e85dc044c7e774ebc1ce44963b4755c-Abstract-Conference.html)、[代码](https://github.com/breez3young/DIMA) | DIMA条件备选，授权未核清 |
| T4 | Agent-Centric Actor-Critic for Asynchronous Multi-Agent Reinforcement Learning | ICML 2025 | [论文](https://proceedings.mlr.press/v267/jung25a.html)、[代码](https://github.com/LGAI-Research/acac) | ACAC仅真正异步时考虑 |
| T5 | Empowering Multi-Robot Cooperation via Sequential World Models | ICLR 2026 | [论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/16415eed5a0a121bfce79924db05d3fe-Abstract-Conference.html)、[代码](https://github.com/zhaozijie2022/seqwm) | SeqWM暂缓迁移 |
| T6 | Learning Disentangled Multi-Agent World Model for Decentralized Control | ICML 2026 | [论文](https://proceedings.mlr.press/v306/xue26e.html)、[代码](https://github.com/DiXue98/DMAWM) | DMAWM执行信息条件未对上，不优先 |
| T7 | MA-WAM: Multi-Agent World-Action Model for Test-Time Planning | 2026年9月预印本 | [论文](https://arxiv.org/abs/2609.31281)、[代码](https://github.com/ma-wam/MA-WAM) | 借评估设计，不整套迁移 |
| T8 | TIGER-MARL: Enhancing Multi-Agent Reinforcement Learning with Temporal Information through Graph-based Embeddings and Representations | L4DC 2026 | [论文](https://proceedings.mlr.press/v331/gupta26a.html) | 不重开时间编码线 |
| T9 | Learning and Planning Multi-Agent Tasks via an MoE-based World Model | NeurIPS 2025 | [论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a2d96d2eb2902043c2db705ca03e9a2-Abstract-Conference.html) | M3W多任务前提不符 |
| T10 | Bootstrap Off-policy with World Model | NeurIPS 2025 | [论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c290dcfbb071656287ee373cdbe8a259-Abstract-Conference.html) | BOOM暂无对应必要性 |

T8—T10只作书目与范围筛选，没有完整源码审计。HAPPO是已在本项目迁移的既有控制，发表年份早于这次热点窗口，不冒称2025—2026新热点。

## 二、诊断与设计文献登记

| 编号 | 完整题目 | 年份/版本 | 原始来源 | 用途 |
| --- | --- | --- | --- | --- |
| D1 | Diagnosing Bottlenecks in Deep Q-learning Algorithms | ICML 2019 | [论文](https://proceedings.mlr.press/v97/fu19a.html) | 已知答案的单元测试 |
| D2 | Agent-Specific Effects: A Causal Effect Propagation Analysis in Multi-Agent MDPs | ICML 2024 | [论文](https://proceedings.mlr.press/v235/triantafyllou24a.html)、[代码](https://github.com/stelios30/agent-specific-effects) | 主体后续行为传播 |
| D3 | Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making | ICML 2025 | [论文](https://proceedings.mlr.press/v267/triantafyllou25a.html)、[代码](https://github.com/stelios30/cf-effect-decomposition) | 有明确定义的路径分解 |
| D4 | Counterfactual Off-Policy Evaluation with Gumbel-Max Structural Causal Models | ICML 2019 | [论文](https://proceedings.mlr.press/v97/oberst19a/oberst19a.pdf)、[代码](https://github.com/clinicalml/gumbel-max-scm) | 外生随机输入与识别边界 |
| D5 | Counterfactual Shapley Credit Assignment | RLJ/RLC 2026 | [论文](https://rlj.cs.umass.edu/2026/papers/Paper45.pdf) | 小窗口动作信用与已知原因恢复 |
| D6 | Empirical Design in Reinforcement Learning | JMLR 2024 | [论文](https://jmlr.org/papers/v25/23-0183.html) | 配对、复制和偏差控制 |
| D7 | Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO | ICLR 2020版本 | [论文](https://arxiv.org/abs/2005.12729) | 算法与实现交叉核查 |
| D8 | Deep Reinforcement Learning at the Edge of the Statistical Precipice | NeurIPS 2021 | [论文](https://arxiv.org/abs/2108.13264) | 区间与小样本评价 |
| D9 | MACCA: Offline Multi-agent Reinforcement Learning with Causal Credit Assignment | TMLR 2025 | [全文](https://openreview.net/pdf/7d36f52116c6b858648f33fd3728eb79a748bf3c.pdf)、[代码](https://github.com/ziyan-wang98/MACCA) | 奖励生成条件，不作直接交通真值 |
| D10 | Aligning Credit for Multi-Agent Cooperation via Model-based Counterfactual Imagination | AAMAS 2024 | [论文](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p281.pdf) | 模型反事实与真实效果分开 |
| D11 | A Causality-Inspired Spatial-Temporal Return Decomposition Approach for Multi-Agent Reinforcement Learning | RLJ/RLC 2026 | [论文](https://rlj.cs.umass.edu/2026/papers/Paper83.pdf) | CAST的已知生成机制检验，不改UAM奖励 |
| D12 | The Trace Is the State: Exact Credit Assignment for LLM Agent Teams | 2026预印本，当前版本 | [论文](https://arxiv.org/abs/2603.06859) | C3后续样本核验思路，不照搬LLM状态前提 |

## 三、归纳与阅读顺序

**概览：** 10项迁移线索和12项诊断/设计资料，均按本次问题定向筛选，不代表热点总体分布。

**材料显示的方法变化：** 世界模型论文同时涉及主体联合依赖、顺序条件化、集中训练/分散执行和测试时规划；这些是不同前提，不应统一当成“更会预测”。这不是对全领域热度的统计判断。

**主要主题：** 联合动力学（T2/T3/T5），真实决策时钟（T4），想象与执行一致性（T1/T6），候选排序（T7），因果干预（D2—D5），实验可信性（D1/D6—D8）。

**标题关键词频数：** 本轮没有按六库输出建立可比较语料，不提供伪精确频数。**录用论文被引排名、第一作者被引排名：** 未获取，均不报告。

**建议阅读顺序：** D1先学怎样检验诊断器；D6固定比较合同；D2/D3明确反事实对象；T2/T3比较完整方法与核心机制；T7借鉴同状态排序及计算成本对照。D5作为小窗口信用分解补充，不先做整场长轨迹归因。

所有UAM应用、筛选和下一轮规模都是报告提出的方案，不是原论文已验证的UAM事实。没有使用外部模型上传项目资料，没有连接服务器，没有读取或修改旧实验数据。
