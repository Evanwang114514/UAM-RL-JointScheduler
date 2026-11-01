# GPT聊天版：UAM代码与数据入口（2026-11-01）

先读总报告，不要从旧文献报告恢复已经暂存的S/AC/WM模型竞赛。当前任务是50M归因结果，不是追加训练。

## 1．推荐阅读顺序

1. [AI交接总报告（纯文本）](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/AI_HANDOFF_UAM_20261101.md)；重点第0/1/8/9节。
2. [20条结论状态账本](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/claim_ledger.csv)；区分事实、假说、负结果和未知。
3. [全部训练与诊断清单](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/experiment_inventory.csv)；8条训练系列＋9条诊断记录，不是17项独立训练。
4. [来源与内容SHA256](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/sources.json)、[权重库存边界](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/model_inventory.json)。
5. [轻量派生证据ZIP下载](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/UAM_HANDOFF_EVIDENCE_20261101.zip)。

ZIP包含本次报告、原样历史自写报告、逐模型/配对/曲线CSV、诊断成本表和代码快照；不含原始DATAONLY合集、权重、Conda、外部论文全文。不能仅凭ZIP重新运行策略。GPT若不能解压或抓取二进制，直接读下面的单文件raw链接；也可由用户上传ZIP。

## 2．实验代码：固定版本优先

下面根目录脚本固定在本次交接前的main commit `42109a01575aeeb1a5cf62514ef4c77e0d89d2be`。只读；执行旧模型仍应对照当轮归档代码/hash，不能把最新main自动替换进旧运行。

| 实验 | 脚本（GitHub查看） | 原始文本 |
| --- | --- | --- |
| T00 | [train_uam_source_K234_5methods_4seeds_60m_ONEFILE.py](https://github.com/Evanwang114514/UAM-RL-JointScheduler/blob/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_source_K234_5methods_4seeds_60m_ONEFILE.py) | [raw](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_source_K234_5methods_4seeds_60m_ONEFILE.py) |
| RULES | [run_uam_source_K234_SPF_STTF_QTTI_ONEFILE.py](https://github.com/Evanwang114514/UAM-RL-JointScheduler/blob/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/run_uam_source_K234_SPF_STTF_QTTI_ONEFILE.py) | [raw](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/run_uam_source_K234_SPF_STTF_QTTI_ONEFILE.py) |
| T01 | [train_uam_strictE1_LQ_fleet40m_methods60m_ONEFILE.py](https://github.com/Evanwang114514/UAM-RL-JointScheduler/blob/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_strictE1_LQ_fleet40m_methods60m_ONEFILE.py) | [raw](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_strictE1_LQ_fleet40m_methods60m_ONEFILE.py) |
| T02 | [train_uam_E01_full5_4seeds_120m_ONEFILE.py](https://github.com/Evanwang114514/UAM-RL-JointScheduler/blob/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_E01_full5_4seeds_120m_ONEFILE.py) | [raw](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_E01_full5_4seeds_120m_ONEFILE.py) |
| T04 | [train_uam_UAGMC_minfix_240m_ONEFILE.py](https://github.com/Evanwang114514/UAM-RL-JointScheduler/blob/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_UAGMC_minfix_240m_ONEFILE.py) | [raw](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_UAGMC_minfix_240m_ONEFILE.py) |
| T05 | [train_uam_E01_S_AC_198m_ONEFILE.py](https://github.com/Evanwang114514/UAM-RL-JointScheduler/blob/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_E01_S_AC_198m_ONEFILE.py) | [raw](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_E01_S_AC_198m_ONEFILE.py) |
| T06 | [train_uam_E1_reward_charge_attribution_30m_ONEFILE.py](https://github.com/Evanwang114514/UAM-RL-JointScheduler/blob/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_E1_reward_charge_attribution_30m_ONEFILE.py) | [raw](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_E1_reward_charge_attribution_30m_ONEFILE.py) |
| T07 | [train_uam_gap_factors_50m_ONEFILE.py](https://github.com/Evanwang114514/UAM-RL-JointScheduler/blob/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_gap_factors_50m_ONEFILE.py) | [raw](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/train_uam_gap_factors_50m_ONEFILE.py) |
| N05/N06 | [diagnose_uam_gap_notrain_ONEFILE.py](https://github.com/Evanwang114514/UAM-RL-JointScheduler/blob/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/diagnose_uam_gap_notrain_ONEFILE.py) | [raw](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/42109a01575aeeb1a5cf62514ef4c77e0d89d2be/diagnose_uam_gap_notrain_ONEFILE.py) |

- [N09实际运行v2诊断代码](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/code_snapshots/diagnose_uam_large_notrain_20261101_ONEFILE.py)，SHA256 `c47d3980fab1abc68d73e97f26da62b28192c528c89f34d948b47d812dd6950f`。
- [N08严格E1分支回放归档入口](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/code_snapshots/N08_strict_replay_archived_run.py)：取自其代码与结果包的code/run.py，可能依赖原归档工作目录，勿当自足新矩阵直接启动。
- [N04评估效率当前解释入口](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/code_snapshots/diagnose_uam_eval_replay_ONEFILE.py)、[N02 Oracle当前解释入口](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/code_snapshots/run_uam_future_oracle_bounded.py)：未在本轮认证它们与当时执行全文逐字一致。
- [48M当前解释用脚本](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/code_snapshots/train_uam_E01_share_ac_diagnosis_48m_ONEFILE.py)：未获得旧执行全文/hash认证，不能冒称原运行逐字版。

本轮不修改这些训练脚本，不启动/停止服务器任务，不新增实验。50M代码本地已核SHA为`f9a8e8b0998cc74a37f6aaa7fae99a21c29e712e9cc32f21507eeba65ef2e077`，服务器未在本轮现场比对。

## 3．可以直接读取的证据表

| ID | 文件 | CSV数据行数 | 支持的报告 |
| --- | --- | ---: | --- |
| D01 | [data/240m/model_results.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/240m/model_results.csv) | 468 | S08 |
| D02 | [data/240m/paired_effects.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/240m/paired_effects.csv) | 540 | S08 |
| D03 | [data/240m/rule_results.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/240m/rule_results.csv) | 33 | S08 |
| D04 | [data/240m/audit.json](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/240m/audit.json) | JSON | S08 |
| D05 | [data/240m/learning_curves.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/240m/learning_curves.csv) | 7956 | S08 |
| D06 | [data/30m/model_results.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/30m/model_results.csv) | 28 | S15 |
| D07 | [data/30m/paired_effects.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/30m/paired_effects.csv) | 38 | S15 |
| D08 | [data/30m/rules.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/30m/rules.csv) | 18 | S15 |
| D09 | [data/30m/frozen_reward_pairs.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/30m/frozen_reward_pairs.csv) | 20 | S15 |
| D10 | [data/30m/audit.json](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/30m/audit.json) | JSON | S15 |
| D11 | [data/30m/learning_curves.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/30m/learning_curves.csv) | 588 | S15 |
| D12 | [data/large_notrain/baseline_episodes.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/large_notrain/baseline_episodes.csv) | 528 | S16 |
| D13 | [data/large_notrain/rule_paired_effects.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/large_notrain/rule_paired_effects.csv) | 160 | S16 |
| D14 | [data/large_notrain/intervention_pairs.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/large_notrain/intervention_pairs.csv) | 120 | S16 |
| D15 | [data/large_notrain/tail_cases.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/large_notrain/tail_cases.csv) | 180 | S16 |
| D16 | [data/large_notrain/branch_costs.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/large_notrain/branch_costs.csv) | 3240 | S16 |
| D17 | [data/large_notrain/summary.json](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/large_notrain/summary.json) | JSON | S16 |
| D18 | [data/literature/faculty_audit_544.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/literature/faculty_audit_544.csv) | 544 | L01,L02 |
| D19 | [data/literature/thu_papers_66.json](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/literature/thu_papers_66.json) | JSON | L01 |
| D20 | [data/literature/traffic_papers.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/literature/traffic_papers.csv) | 83 | L03,L05 |
| D21 | [data/literature/ai_papers.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/literature/ai_papers.csv) | 93 | L04,L05 |
| D22 | [data/strict_replay/branch_costs.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/strict_replay/branch_costs.csv) | 3888 | S14 |
| D23 | [data/strict_replay/model_results.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/strict_replay/model_results.csv) | 12 | S14 |
| D24 | [data/strict_replay/states.csv](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/data/strict_replay/states.csv) | 432 | S14 |

主表先按训练模型聚合；不要把分支行、eval seed、checkpoint或同OD重排当独立训练seed。CSV保持原报告数值，本轮只复制索引，没有重算或改写原数。

## 4．历史报告（旧建议不自动生效）

| 来源ID | 原报告 | 阅读/版本边界 |
| --- | --- | --- |
| S01 | [严格E1一亿步_E0与解析基线对照_20261021](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S01_strict100.md) | 已读取主要结果；本轮未重训或重算全部原始记录 |
| S02 | [论文与官方源码核验_关键入口错误](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S02_source_audit.md) | 核对关键入口、固定源版本和复现边界 |
| S03 | [00_两轮补分析_核心结论与下一步_20261028](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S03_120m48m_summary.md) | 核心结论全文复读 |
| S04 | [01_120M完整五方法_结果与联合服务失配分析_20261028](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S04_120m.md) | 本轮重读核心章节；未重复整轮分析 |
| S05 | [02_48M_Shared_AC_配对诊断与机制验证_20261028](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S05_48m.md) | 本轮重读方法与核心证据；执行代码未封存问题保留 |
| S06 | [自严格E1一亿以来_按方法逐环境机场seed重审_20261029](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S06_method_review.md) | 重读覆盖和核心结论；逐seed附表作为原报告保留 |
| S07 | [研究结论与数据证据总账_20261028](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S07_old_ledger.md) | 历史账本，不覆盖更新的奖励归因结论 |
| S08 | [01_240M系统分析_先看结论再看方法](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S08_240m.md) | 核心证据重读；配套逐模型、规则、配对CSV公开 |
| S09 | [02_后续无训练诊断与条件性训练](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S09_240m_old_followup.md) | 旧建议仅保留，不自动恢复为当前任务 |
| S10 | [诊断包分析与1500万步铺路_20261030](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S10_eval_efficiency.md) | 评估耗时和完成情况核心证据 |
| S11 | [01_无训练诊断_明确结果与尚未解决的问题](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S11_gap_notrain.md) | 已有记录诊断与未完成分支的边界 |
| S12 | [E01_Trace与Oracle_系统分析_20261024](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S12_trace_oracle.md) | 读取核心结果；Oracle不作为合法在线基线 |
| S13 | [无训练短诊断_结论与下一步](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S13_temporal_screen.md) | 人工trace短筛，不将标记直接称交通损失 |
| S14 | [无训练诊断_结论与证据](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S14_strict_replay.md) | 旧SOURCE模型分支；不推广为PT已验证 |
| S15 | [01_30M归因分析_结论与证据](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S15_30m.md) | 关键奖励转折；原CSV与多顺序统计隔离 |
| S16 | [01_大规模无训练诊断_结果与竞争解释](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S16_large_notrain.md) | 最新N09主要分析及竞争解释 |
| S17 | [02_五千万训练设计_时间与批服务归因](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S17_50m_design.md) | 当前固定设计；没有新增运行或修改 |
| S18 | [05_近期2亿步_198run固定实施方案](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/S18_198m_design.md) | 设计不是最终结果；本地缺最终包 |
| L01 | [01_扩展教师与文献完整搜索报告](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/L01_thu_search.md) | 历史检索报告；本轮不刷新人员、文献或引文 |
| L02 | [02_查找过的全部教师_544条记录](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/L02_thu_people.md) | 544条覆盖记录不等于544名去重教师 |
| L03 | [01_交通UAM文献与研究缺口_完整版](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/L03_traffic_literature.md) | 保留检索层级；不是全部全文精读 |
| L04 | [02_AI前沿与四篇研究路线_完整版](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/L04_ai_literature.md) | 历史How工具箱，不自动启动新模型 |
| L05 | [研究范围与进度](https://raw.githubusercontent.com/Evanwang114514/UAM-RL-JointScheduler/main/analysis_reports/uam_ai_handoff_20261101/evidence_reports/L05_literature_scope.md) | 文献数量、阅读层级和会议版缺全文限制 |

## 5．原始包在哪里：本地/服务器地址不是公网

下列原始tar.gz**本轮没有上传GitHub，公开URL为空**。本轮公开的是派生证据ZIP和以上单文件CSV/JSON/报告。要审核更细的原始事件或重放模型，由用户从下面本地目录上传对应原包；不要说已从GitHub读取了这些原包。

本地统一目录：`E:\Study Files\github\UAM-predict\UAGMC-main\experiment_archives`。服务器历史项目：`/home/SY/uam/UAGMC-main`，常见归档目录`experiment_archives`。本轮未连接服务器，不认证路径当前存在。

| ID | 原始文件名 | 本地字节数 | 状态 |
| --- | --- | ---: | --- |
| T00 | `uam_source_K234_5methods_4seeds_60m_v1_DATAONLY.tar.gz` | 22016376 | 完整/已分析 |
| T01 | `uam_strictE1_LQ_fleet40m_methods60m_v1_DATAONLY.tar.gz` | 71687699 | 完整/已分析 |
| T02 | `uam_E01_full5_4seeds_120m_v1_DATAONLY.tar.gz` | 38734217 | 完整/已分析 |
| T03 | `uam_E01_share_ac_diagnosis_48m_v1_DATAONLY.tar.gz` | 11751873 | 完整/已分析；热启动 |
| T04 | `uam_UAGMC_minfix_240m_v1_DATAONLY.tar.gz` | 77633149 | 完整/已分析 |
| T05 | `uam_E01_S_AC_198m_v1_DATAONLY.tar.gz` | 尚未收到 | 最终包未收到；最后明确到191启动 |
| T06 | `uam_E1_reward_charge_28run_30m_v1_DATAONLY.tar.gz` | 7530860 | 完整/已分析 |
| T07 | `uam_gap_factors_48run_50m_v1_DATAONLY.tar.gz` | 尚未收到 | 用户16:48快照19训19评；第20格进行中 |
| N02 | `uam_E01_trace_oracle_20261024_DATAONLY.tar.gz` | 1981190 | 完成/已分析 |
| N04 | `uam_eval_diag_20261030_DATAONLY.tar.gz` | 528575 | 完成/已分析 |
| N05 | `uam_gap_notrain_20261030_local_16725_28016_DATAONLY.tar.gz` | 1505763 | 已有记录重复分析 |
| N06 | `uam_gap_notrain_20261031_full_v2_DATAONLY.tar.gz` | 4088727 | 部分分支失败，报告保留边界 |
| N09 | `uam_large_notrain_20261101_DATAONLY.tar.gz` | 18077066 | v2完成/已分析 |

N09服务器历史原包位于：`workspace/artifacts/uam_large_notrain_20261101_server_v1/uam_large_notrain_20261101_DATAONLY.tar.gz`。N09/30M/240M最关键派生表已在第3节公开，可据此阅读交接结论；它们不等价于完整原包。

严格E1回放必须用本地带`VERIFIED`的943,124,945字节包；无VERIFIED的427,556,864字节包是截断遗留。48M当时权重库存为0，198M最终包未收到；不因缺包重训。

## 6．接手限制

优先复核当前50M的最终包（尚未收到），按TIME×BATCH及普通Transformer预设对照分析。不要恢复完整S/AC、WM/MARL、调场景等暂存路线；不要将旧SOURCE巨大联合损失当正确目标后的可用headroom；不要从时序输入为零直接声称时序是性能主因。用户自行执行Windows CMD上传和服务器命令，AI不代连。

本目录是追加交接，不覆盖旧原报告/实验，GitHub main可变化；需要固定引用时在本次交接提交的commit下打开同一路径。
