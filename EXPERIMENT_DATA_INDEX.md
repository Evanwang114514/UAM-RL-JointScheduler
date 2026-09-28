# 矩阵实验数据索引（2026-09-28）

本索引对应项目根目录下的 `experiment_archives/`。归档文件统一使用
`expNN_实验标识_预算__内容.扩展名`；无学习规则基线使用 `baselineNN_`。
`__raw.tar.gz` 是对当时本机 `serial_runs/` 中整个实验目录的只读打包；
`__analysis.zip` 只保存结果表、日志和评估明细，除非明确标为 `models`，不含模型权重。
所有归档的 SHA-256 见 `EXPERIMENT_ARCHIVE_SHA256.txt`。

归档只复制或移动了数据包，原始 `serial_runs/` 目录仍保留。归档目录被 Git 忽略，
GitHub 同步的是本索引与校验表，不是数十 GB 的训练数据。

| 编号 | 实验入口 | 归档文件 | 本机覆盖与限制 |
|---|---|---|---|
| exp01 | `train_uagmc_reposition_lq_vs_vertisync_800k.py` | `exp01_lq_vs_vertisync_1p6m__analysis.zip` | 从历史总包拆出的分析记录；本机未找到模型目录。 |
| exp02 | `train_uagmc_E3_E4_E5_serial_1m.py` | `exp02_e3e5_complexity_3m__analysis.zip` | E3/E4/E5 分析记录；本机未找到模型目录。 |
| exp03 | `train_uagmc_E0_E2_E6_effect_time_700k.py` | `exp03_effect_time_4p2m__analysis.zip` | E0/E2/E3/E4/E5/E6 分析记录；本机未找到模型目录。 |
| exp04 | `train_uagmc_E3_E6_obs_topology_matrix_800k_FORMAL.py` | `exp04_obs_topology_25p6m__raw.tar.gz`、`exp04_obs_topology_25p6m__analysis.zip` | 32 格本机原始目录与分析包；原始包含 checkpoint。 |
| exp05 | `train_uagmc_6x6_800k.py` | `exp05_uagmc_6x6_28p8m__raw.tar.gz`、`exp05_uagmc_6x6_28p8m__analysis.zip` | 36 格本机原始目录与分析包；原始包含 checkpoint。 |
| exp06 | `train_uagmc_45x800k_formal_JOINTFIX.py` | `exp06_formal45_36m__raw.tar.gz`、`exp06_formal45_36m__analysis.zip` | 45 格本机原始目录与分析包；原始包含 checkpoint。 |
| exp07 | `train_uam_7x12_600k_v2.py` | `exp07_uam_7x12_50p4m__raw.tar.gz`、`exp07_uam_7x12_50p4m__single36_models.zip` | 原始包保存本机目录；模型包仅覆盖 36 格 Single。本机汇总为 47 格，不能称为完整 84 格备份。 |
| exp08 | `train_uam_60m_literature_matrix_v3_1.py` | `exp08_literature_v3_57p6m__raw.tar.gz`、`exp08_literature_v3_57p6m__phase_b_delta.zip`、`exp08_literature_v3_57p6m__final_delta.zip` | 原始包保存本机目录；本机 `matrix_master.csv` 为 29 格，不能称为完整 96 格备份。 |
| exp09 | `train_uam_60m_jointfirst_v4_deferred_joint_eval.py` | `exp09_jointfirst_v4_60m__raw.tar.gz`、`exp09_jointfirst_v4_60m__single33_models.zip`、`exp09_jointfirst_v4_60m__joint_partial_analysis.zip` | 原始包含 100 个 cell 文件夹及模型文件；本机 `matrix_master.csv` 仅汇总 43 格，Joint 分析包为局部结果。 |
| exp10 | `train_uam_60m_jointfirst_v5_minimal_reposition.py` | `exp10_jointfirst_v5_60m__analysis_full.zip` | 100 格训练、100 格有效评估；包内没有模型或 checkpoint，本机未找到原始模型目录。 |
| exp11 | `train_uam_s3_20x600k_litupgrade.py` | `exp11_s3_litupgrade20_12m__analysis.zip` | 20 格分析结果；包内没有模型。 |
| exp12 | `train_uam_nextgen_100m_matrix.py` | `exp12_nextgen_100p2m__analysis_full.tar.gz` | 167 个 cell summary 和 `RUN_COMPLETE.json`；包内没有模型或 checkpoint。 |
| exp13 | `train_uam_single70m_priority.py` | `exp13_single_priority_69p6m__analysis_snapshot_20260928.tar.gz` | 2026-09-28 新包含 66 个 cell summary；P0/P1/P2 有阶段完成标记，P3 仅见已启动。无全局 `RUN_COMPLETE.json`、模型或 checkpoint；虽然原包名含 COMPLETED，仍不能称为最终完整归档。 |
| baseline01 | `run_uagmc_E0_E2_E6_rule_baselines.py` | `baseline01_e0e6_rules__analysis.zip` | 从历史总包拆出的规则基线分析记录。 |
| baseline02 | `run_uagmc_6env_6baseline_36runs.py` | `baseline02_6env6rules__raw.tar.gz` | 本机完整原始结果目录。 |
| baseline03 | `run_formal25_analytical_baselines.py` | `baseline03_formal25__raw.tar.gz`、`baseline03_formal25__analysis.zip` | 本机完整原始结果目录与原有小结果包。 |

`legacy_mixed_20260923__analysis.zip` 是原来的历史混合总包，包含上述早期实验及其他旧快照。
它保留作溯源，不是任何一个矩阵的唯一归档。

## 尚需从服务器补齐

- exp10/V5、exp11、exp12 的模型权重和 checkpoint：本机分析包均不含这些文件。
- exp13 的最终完成包和模型文件：现有文件只是 66 格快照。
- exp07、exp08、exp09 若服务器上有比本机更完整的后续结果，需要单独核对，不可用本机归档代表全部计划格子。

本索引只记录已实查的本机文件。服务器目录尚未成功连接核验。
