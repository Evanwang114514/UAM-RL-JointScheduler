# 三组实验合并分析交付

先读 `01_三组实验合并分析与后续安排.html`，第1节是简明结论，第7—8节是后续诊断与条件性训练设计。Markdown是相同报告的可编辑文字版，不是另一份报告。

本目录不含新训练/回放入口，也不启动服务器任务。`analyze_packages.py`只从用户已有压缩包流式读取JSON/CSV并复算统计，不加载模型、pickle或归档源码，不解压改写原包。无第三方库、无新增训练步。

复核文件：

- `INPUT_AUDIT.json`：输入大小、SHA256、逐成员读取结果及包内实验入口哈希。
- `ANALYSIS_PLAN.json`：分析口径和比较单位。它是本次回顾分析的事前计算约束，不冒称在原实验采集前注册。
- `analysis_summary.json`：逐模型、配对差值、分支统计、逐人配对。诊断状态不是新的独立训练复制。
- `analysis_details.json`：本地完整逐人中间记录；分析时生成，但不重复推送GitHub。需要时从原包复算。
- `validate_report.py`：核心数字、表格、输出边界和HTML结构的本地检查。
- `render_report.py`：报告Markdown到HTML的机械转换。

本地Windows CMD只读复算（不是服务器实验提交命令）：

```cmd
cd /d "E:\Study Files\github\UAM-predict\UAGMC-main"
conda activate uam5070
python -B analysis_reports\uam_three_batch_review_20261104\analyze_packages.py --archives "E:\Study Files\github\UAM-predict\UAGMC-main\experiment_archives"
python -B analysis_reports\uam_three_batch_review_20261104\render_report.py
python -B analysis_reports\uam_three_batch_review_20261104\validate_report.py
```

原始三个数据包位于本地项目的 `experiment_archives`，不推送大型归档到GitHub。网页聊天AI读取本报告的仓库版后，若需重新统计原始记录，仍需用户附加原包。模型包和对应归一化参数另行保留；DATAONLY不能替代模型回放所需文件。

方法解释上最重要的限制：两组训练的权限、决策接口、评估动作采样及窗口不同，不跨组做因果排名；未完成者完整ATT为空，主表是截止成本；STIN共同物理状态仅4个；合法在线修正最多两次，不是完整MPC；短窗口错选率不等于PPO错误率。
