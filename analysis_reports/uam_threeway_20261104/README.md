# 本轮交付索引

请先看两份报告的“一页总览”，再看命令，不必先读代码。

- [旧实验回顾与文献筛选](01_旧实验回顾与文献筛选_先读结论.md)：什么真的试过、什么只是旧卡片名、为什么选两条路线、其他文献为什么暂缓。
- [三组实验设计](02_三组实验设计_普通训练热点训练与诊断.md)：40h两路线迁移、14h普通训练、综合诊断、指标与不同结果的后续处理。
- [完整上传、并行启动与下载命令](03_上传并行启动与下载命令.md)：Windows CMD与服务器Linux分开。

离线浏览版：[回顾与文献HTML](01_旧实验回顾与文献筛选_先读结论.html)、[实验设计HTML](02_三组实验设计_普通训练热点训练与诊断.html)。

正式入口在仓库根目录：

- [train_uam_hot_transfer_40h_ONEFILE.py](../../train_uam_hot_transfer_40h_ONEFILE.py)
- [train_uam_space_time_14h_ONEFILE.py](../../train_uam_space_time_14h_ONEFILE.py)
- [diagnose_uam_threeway_comprehensive_ONEFILE.py](../../diagnose_uam_threeway_comprehensive_ONEFILE.py)

GitHub主仓库：<https://github.com/Evanwang114514/UAM-RL-JointScheduler>。数据／模型不提交GitHub；实验结束自动打包到服务器`experiment_archives`，由用户下载到本地项目同名目录。

本交付只有静态／标准库测试，不是服务器端运行验收。热方法为DreamerV3及HAPPO，两对照为普通PPO和匹配MAPPO；DIMA等列入详细筛选但不硬开第三条。
