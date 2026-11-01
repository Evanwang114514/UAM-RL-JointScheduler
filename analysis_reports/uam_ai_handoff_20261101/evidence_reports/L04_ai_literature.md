# AI/RL前沿与UAGMC、RadioUNet、RadioDiff研究路线复审

检索截止：2026-10-29。核心问题：不再追逐模块名，怎样把新的AI进展变成一个可归因、可负担的交通贡献？

## 先给结论

**最值得借鉴的变化不是“PPO已经过时”，而是从直接输出动作，转向利用结构、少量执行期计算和可校准的候选价值比较。** 对我们目前K2/K3/K4的小离散动作问题，优先级是：

1. **结构化决策／短程搜索＋决策相关尾值。** 先用真实已知机制，学习不能解析决定的部分。
2. **训练稳定性和基座保真。** 先确认新模型能复现旧策略，再谈增量。
3. **异步信用分配。** 只有确实存在不同决策时刻/持续动作时才适用；“效果延迟”不能直接等同于“异步训练”。
4. **离线到在线、专家蒸馏。** 可降低计算，但已是成熟范式，不能单独撑新意。
5. **生成式/Flow策略。** 当输出变成多步、可变长、多峰调度方案时才更值得；眼下少数离散候选先穷举。

不优先：为了热点引入LLM/VLA、把任务改成语言推理、全量视频世界模型、元学习一个新RL算法。它们在别处的高水平结果不是我们可直接复制的收益。

## 1. 工作量与来源边界

- AI首轮四个关键词查询得到OpenAlex/arXiv/Crossref各100条原始命中，含大量不相关结果。
- 进一步读取ICML2025、ICML2026官方PMLR目录，按明确主题词得到697条题名候选并保存官方摘要页面。
- NeurIPS2025主会目录按同类主题得到306条题名候选，人工挑选22条读取官方摘要；另核验ICLR、IJCAI、CoRL、Nature等代表工作。
- 最终列入**93条AI/RL文献**；它们完成的是摘要级筛选，**不声称93篇全文精读**。其中FastTD3仅列技术报告；正式状态与预印本分开。
- 渠道为ICML2025/2026共65条、NeurIPS2025共22条，其余6条分别来自ICLR、IJCAI、CoRL、Nature及技术报告；这解释了文献池的偏向，不宣称每个顶会已被穷尽。
- 这是相关性驱动的目的性取样，偏重ICML，不能据计数宣称整个AI领域热点的市场份额。
- 高频引用只作为关注线索；未统一取得的引用数不编造，不提供伪“高被引榜”。

四篇研究路线的状态必须单列：

| 目标 | 核实身份与日期 | 本轮实际读取 | 边界 |
|---|---|---|---|
| UAGMC会议版 | ICC2025，3095–3100，DOI 10.1109/ICC52391.2025.11160730 | 官方作者书目、DOI记录和公开索引摘要 | **没有取得会议全文，不能称已精读完成** |
| UAGMC期刊版 | T-ITS2026，DOI 10.1109/TITS.2026.3680351；arXiv:2601.21316 | 作者稿的问题、方法、实验、消融、敏感性及结论章节 | 不是出版社最终PDF逐字校对；实现差异须再对真实源码 |
| RadioUNet | TWC2021；arXiv始于2019 | 公开作者稿的任务、数据、架构、训练、迁移、对照与复杂度章节 | “real-life”迁移实验使用高精度仿真代理，不等于真实测量全验证 |
| RadioDiff | TCCN2025卷期，DOI含2024，2024已在线 | 作者接受稿的方法、数据、结果、AFT消融及限制章节 | dynamic为环境快照，不能套成时序闭环世界模型 |

RadioUNet与RadioDiff作为用户指定的研究范例单列，不凑进2025/2026前沿论文数量。四篇中会议全文仍有缺口；用户已同意以UAGMC期刊全文为主、不以会议全文缺失阻塞本轮交付。会议内容仍不由期刊细节倒填。

## 2. UAGMC两版：值得学什么，也不能误解什么

### 2.1 可核实的演进

[会议版作者书目](https://wangmn.cn/publications/) 将其列为ICC2025；[期刊作者稿](https://arxiv.org/html/2601.21316) 把已有UAGMC-L与Attention版UAGMC-A区分。因会议全文未取得，下面不虚构其训练表或逐项新增内容。

期刊路线可以压缩为：**选站中的异构信息→多源编码→时序整合→PPO决策→规则与网络消融→需求/座位敏感性。** 这不是发明PPO，而是把任务结构与证据组织成闭环。

### 2.2 三个决定我们研究起点的事实

1. 期刊主模型是Attention版，不是近期本项目常用的LSTM源风格对照。
2. 文本明确有充电与满座出发，也明确后续仍需研究完整任务调度；因此不能说原文“完全没有充电”，也不能把其未来工作当成全球无人做过。
3. 原文表2使用3座、600时步等设置；本项目近期4座和其他训练合同不能直接当作论文数字的严格复现。

### 2.3 关键数值只用于理解原文，不与我们的ATT混排

| 期刊表3方法 | ATT/min |
|---|---:|
| SPF | 133.65 |
| QTTI | 51.87 |
| UAGMC-L | 48.69 |
| UAGMC-A | 46.63 |

来源为作者稿表3；各表对“±”的统计说明并不统一，不能读成我们的四训练seed SD。[原文实验部分](https://arxiv.org/html/2601.21316#S5) <!--ref:T006--><!--anchor:section:5-->

**我们的判断：** 值得学的是任务—结构—消融的对接，而非把其较弱解析参照直接沿用到2027。原文等待近似与实验环境的细节应通过代码核查；不能从公式简化直接推断公开模拟器必然使用同一近似，更不能由其复杂度描述推出所有在线优化都不可用。

更值得学的是原文表4的证据结构：没有两个核心组件时ATT为63.07，仅MSCE为60.57，仅STIN为56.31，二者同用为46.63。它至少把“输入组织”和“时间处理”拆成四格，而不是只展示最终网络赢了；但这些数值不能替代我们自己的同合同、多训练seed交互验证。[UAGMC表4](https://arxiv.org/html/2601.21316#S5.T4)

### 2.4 这给我们什么新的要求

- 论文不能只以“击败UAGMC-L”为终点，应包含A版或清楚解释为何只复现L版。
- 若我们要研究已承诺飞机供给，必须展示它改变了哪些候选动作排序；多读几个资源属性不构成自动增益。
- 若引用其未来工作，应同时承认VertiSync、eVTOL-VSP与联运调度已有覆盖。
- 先完成模型/环境合同对照；不要把源码接口问题包装成整个领域表示理论的缺陷。

## 3. RadioUNet→RadioDiff：真正值得学的范本

### 3.1 RadioUNet不是随手拿U-Net套数据

[RadioUNet作者稿](https://arxiv.org/abs/1911.09002) 把城市几何与发射机位置组织为图像到无线图的映射，构建RadioMapSeer，区分有/无测量输入，并研究粗仿真到精仿真的适配。架构利用平移结构，验证按地图分离，兼顾精度与推理成本。<!--ref:RadioUNet--><!--anchor:section:3-->

**对我们最有价值的类比：** 不是原始UNet有多新，而是输入表示尊重问题结构，模型学到的函数不依赖单一地图；对应到UAM，我们应研究相同决策接口是否跨K、需求和控制权限复用，而非只在一条训练trace上比数字。

### 3.2 RadioDiff补的是哪一层

[RadioDiff接受稿](https://unicxidian.org/publication/radiodiff/RadioDiff.pdf) 采用条件潜在扩散，区分静/动态障碍输入，用AFT增强特征，并比较精度、图像结构和成本；不是只把一个backbone改名。原文表II的静态NMSE从RadioUNet的0.0074降至0.0049，表IV推理时间则从0.056秒增至0.6秒。<!--ref:RadioDiff--><!--anchor:section:V-->

**对我们的迁移判断：** 新方法可以更慢，但必须证明在业务决策时限内值得；“更复杂模型更准确”不够，需要性能—成本曲线。上述数值来自同篇论文内部，不跨RadioUNet原文的硬件/划分拼接。

### 3.3 两点必须批判性看待

1. **“连续输出/输出不在输入里，所以必须是生成式”不是普遍数学定理。** 回归同样能预测连续场；应把RadioDiff的任务重述视为作者建模论证，其方法效果仍由实验支持，不能照抄成“不用生成模型必然错”。
2. **RadioUNet原文已经研究车辆与输入不完整。** 因而不能把两篇演进讲成“旧方法从不考虑动态、新方法首次考虑动态”；新意需要放在具体方法与比较条件上。

此外，RadioDiff的dynamic主要是快照障碍变化，不是持续在线控制；其训练有大量监督仿真标签，我们是动作会改变后续数据的RL系统。这是迁移中最大的不同。

还要分清**研究范式递进**与**同一团队的版本升级**：两篇作者群不同；本报告使用“RadioUNet→RadioDiff”表达同一问题上不同方法范式的比较，不宣称它们是一篇论文的直接会刊扩展。

### 3.4 能复制的是研究逻辑，不是扩散模块

| 范例中的动作 | 我们可以学什么 | 我们不能怎么写 |
|---|---|---|
| 把物理问题重新表达为适合AI学习的对象 | 将未来供需写成候选动作的边际服务代价 | “我们把UAM变成生成式，所以先进” |
| 显式规定输入条件与数据边界 | 只用已揭示请求和已承诺任务 | 用服务器掌握的真实未来请求做在线推演 |
| 用同数据对比不同模型范式 | 相同环境/trace/控制权限比较规则、优化、RL | 以更强资源或特权信息击败旧方法 |
| 针对缺陷做组件消融 | 预测哪个状态、哪类错误应被修复 | 全模型偶然更好就反推所有模块都有效 |
| 展示成本与限制 | 比较相同墙钟预算下ATT收益 | 忽略训练和执行期分支搜索成本 |

### 3.5 回答你的核心疑问：借用成熟AI，为什么有的工作仍有价值？

**借用AI本身不是问题，缺少独立问题与可信增量才是问题。** 我们不必发明一个通用新RL优化器；但必须说明旧方法在哪类合法决策状态下信息不够、哪种任务结构能把新方法的能力用出来，以及强规则/优化为什么没有以更低成本达到同样效果。

对应的UAM研究路线应是：**识别可重复的协调损失→把损失转成可学习/可计算的候选代价→选择适合这种输出的AI→用错误类型和成本对照证实。** 顺序不能反过来：先选2026热点，再找一个环境让它显得必要。

如果普通解析已经解决同一问题，就承认不需要复杂AI；如果简单残差已足够，就把贡献放在交通机制与决策接口，而不是强称新网络。这不是要求每篇都证明理论定理，而是要求贡献不只依赖模型名称和单次低Best。

## 4. 2025–2026最值得我们关注的八条AI路线

### 路线A：结构化RL，让控制约束进入动作产生过程｜P0

[Structured RL for Combinatorial Decision-Making，NeurIPS2025](https://papers.nips.cc/paper_files/paper/2025/hash/fe100e5e7bca984803f0e4c6e94a6a88-Abstract-Conference.html) 将组合优化层嵌入actor；它是“控制本身成为方法的一部分”的直接参照。<!--ref:A093--><!--anchor:section:Abstract-->

**适配：** 神经网络给候选的边际成本，解析层选合法的乘客/飞机组合；小K先做精确枚举，不必先引入可微求解器。**风险：** “神经成本＋优化层”本身已有人做；我们的增量必须是服务承诺、时序交互或可解释成本分解。

### 路线B：执行期计算／搜索，不再期待一次前向解决全部问题｜P0

[Breaking the Performance Ceiling，NeurIPS2025](https://papers.nips.cc/paper_files/paper/2025/hash/2226e1facc28d7cc36cb1f9d1d82b991-Abstract-Conference.html) 研究推理策略对复杂协同RL的提升；[Search in Model-Based RL，ICML2026](https://proceedings.mlr.press/v306/chang26a.html) 则指出即使模型准确，搜索也可能因价值过估计而变差。

**适配：** 先枚举合法候选，短程传播已知承诺，用保守校准尾值排序；只在动作难分时增加分支。**反证：** 同预算的普通rollout一样好，或搜索反而利用错误尾值。不能因为有模拟器，就声称搜索一定改善。

### 路线C：决策导向学习，修正排序而非完整预测世界｜P0/P1

[Decision-Focused Learning via Tangent-Space Projection，ICML2026](https://proceedings.mlr.press/v306/lee26q.html) 分析预测误差中对决策有意义的分量；[Value-Aligned WM，ICML2026](https://proceedings.mlr.press/v306/jiang26ai.html) 将价值相关性与模型学习相连。

**适配：** 学习候选总乘客时间差、较优动作或尾值，而非重建每架飞机的所有未来轨迹。**限制：** PEAR的连续优化正则性条件不自动适用于离散满座出发；这里只借鉴目标选择，不能直接套其梯度定理。

### 路线D：基座保护、残差价值标定与可塑性｜P0诊断

[SimBa，ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/741aab8b41a2987867acc9939ad50383-Abstract-Conference.html)、[SimbaV2，ICML2025](https://proceedings.mlr.press/v267/lee25u.html) 关注稳定的网络尺度；[DAWN，ICML2026](https://proceedings.mlr.press/v306/ma26s.html) 分析残差RL中的critic冷启动与尺度失配；[C-CHAIN，ICML2025](https://proceedings.mlr.press/v267/tang25g.html) 研究输出churn与可塑性。

**适配：** 记录固定状态的动作分布漂移、特征尺度、候选差距；新支路关闭时严格复现旧策略。**限制：** DAWN主要是连续动作残差，不能把其有界动作修正原样当成离散logit安全保证；持续学习理论也不能直接诊断我们固定任务中的所有seed波动。

### 路线E：真实异步动作的信用分配｜有条件P1

[ACAC，ICML2025](https://proceedings.mlr.press/v267/jung25a.html) 使用agent-centric历史与异步GAE；[Asynchronous Credit Assignment，IJCAI2025](https://ijcai.org/proceedings/2025/20) 使用虚拟同步代理与乘性价值分解。

**适配前提：** 乘客决策与飞机可调时刻真不同，且动作持续时间与训练样本对应清楚。**不能偷换：** 现在每分钟共同采样两个head、只是飞机有时无效，并不自动等于这些论文的异步宏动作问题。先核对轨迹语义，再决定移植。

### 路线F：Flow/扩散策略、动作块与冻结基座操纵｜中期P2

[FQL，ICML2025](https://proceedings.mlr.press/v267/park25f.html) 结合表达力强的流策略与一步actor；[Direct FQL，ICML2026](https://proceedings.mlr.press/v306/cao26f.html) 改进流策略优化；[RTC，NeurIPS2025](https://papers.nips.cc/paper_files/paper/2025/hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html) 处理动作块的异步执行；[DSRL，CoRL2025作者项目页](https://diffusion-steering.github.io/) 在冻结扩散基座的潜在空间训练控制。

**适配：** 如果未来输出整个短时调度序列，多峰可行计划与已承诺前缀会使它们更有意义。**当前判断：** K≤4的少数离散动作先枚举，Flow不是默认首选；不可把RTC里的连续动作“冻结前缀”不加检查套到乘客不可逆分配。

### 路线G：离线到在线与蒸馏｜省算力工具，不单独作贡献

[OPT，ICML2025](https://proceedings.mlr.press/v267/shin25c.html) 指出离线预训练可能在在线阶段因价值误差受损；与交通侧MIP-BC-RL近邻一起，说明“先学专家”既有潜力也有分布偏移问题。

**适配：** 从合法解析/有限窗口求解器生成训练标签，监督候选价值或排序，再用在线数据修正。**边界：** 如果专家看到真实未来，只能作为特权训练教师单列，并证明部署输入不泄漏；最好先做非特权教师公平对照。

### 路线H：自动发现RL、RLVR与大模型推理｜了解但不主攻

[DiscoRL，Nature2025](https://www.nature.com/articles/s41586-025-09761-x) 通过大量环境经验元优化更新规则，是“跳出PPO”的代表方向；VinePPO等语言推理工作说明价值估计和信用分配仍在快速发展。

**适配判断：** 我们可以借鉴固定状态诊断、分支回报和任务相关预测；不应在一个UAM模拟器里照做昂贵的跨环境算法发现，也不应把可逐分钟计算的人时成本改为语言任务的稀疏对错奖励。

## 5. 十二篇优先文献：每篇只讲它改变哪个决定

| 优先级 | 论文 | 核心思路 | 本项目最小可迁移对象 | 必须对照/不应迁移的部分 |
|---|---|---|---|---|
| P0 | SRL，NeurIPS2025 | 把组合优化放入actor | 学边际成本、解析选合法组合 | 普通枚举/同预算MPC；不借“优化层”宣称首创 |
| P0 | Inference Strategies，NeurIPS2025 | 执行期多次尝试突破一次前向上限 | 小分支候选比较 | 相同时间预算，不比较特权未来 |
| P0 | Search Difficulty，ICML2026 | 搜索会放大价值偏差 | 尾值保守校准与集成 | 无尾值短视推演、普通单critic |
| P0 | Snapshot of Influence，NeurIPS2025 | 追溯PPO更新与最近buffer数据影响 | 固定状态的局部训练归因 | 局部解释不当作全训练因果定理 |
| P0 | DAWN，ICML2026 | 残差critic冷启动/尺度匹配 | 基座数据作价值锚点 | 残差关闭等价、同信息普通基座 |
| P0 | SimBa/SimbaV2，2025 | 简洁通路与规范化稳定优化 | 作为现代强网络对照 | 标准化不同可能改变训练合同，单列而非偷偷替换 |
| P1 | ACAC，ICML2025 | 分别处理异步历史与GAE | 真异步决策轨迹 | 不能只把双head换名ACAC |
| P1 | VSP/MVD，IJCAI2025 | 依赖当前正在执行动作分配信用 | 共同背景＋候选交互项 | 乘性分解与简单加法/条件head分别比较 |
| P1 | TAWM，ICML2025 | 显式条件化时间步长 | 未来若有不同控制间隔 | Δt是采样间隔，不等于我们动作生效时间 |
| P1 | PEAR，ICML2026 | 决策相关误差投影 | 排序/边际成本监督 | 离散队列任务不直接继承连续优化结论 |
| P1 | C-CHAIN/P3O，ICML2025 | 减少churn或重生并恢复知识 | 检查/修复后期漂移 | 若无漂移而仅跨seed不同，不应先重置模型 |
| P2 | FQL/RTC/DSRL，2025 | 多峰策略、动作块、冻结基座操纵 | 多步可变长调度提案 | 小动作枚举优先，不盲目扩散化 |

完整引用、作者、年份、官方链接与阅读层级见文末93条表；未通读全文的条目不填虚构超参数或“作者说的不足”。

## 6. 五个构思经过反方检验后的取舍

| 构思 | 吸引力 | 最接近已有工作/风险 | 取舍 |
|---|---|---|---|
| 承诺日历＋候选边际成本＋短程分支 | 最贴近已有环境与S/AC，数据与计算可控 | MPC、SRL、rollout都很成熟；必须证明独特的供需承诺交互 | **主候选** |
| 自适应动作时钟＋异步信用 | 可能减少无效飞机决策 | ACAC/VSP已直接研究；现有每分钟合同未必是真异步 | **先诊断，暂不大训** |
| 满座汇聚—分散＋可计算协调准则 | 交通意义强，可解释K非单调 | 批排队与池化理论很老；可能只有配置问题 | **作为独立机制候选** |
| Flow生成整段计划再修复 | AI较新，适合多峰计划 | 当前动作太小；可行性修复/采样成本难解释 | **后期输出复杂后再做** |
| 纯稳定化修UAGMC | 最快落地 | SimBa、DAWN、可塑性已有大量方法；容易退回基线修补 | **工程必要，但不独占论文主线** |

选择依据是问题契合与可证伪性，不是哪个词更热门。以上新意尚未获独立同行认证；若核心近邻完全覆盖，需重新收缩，而不是改名。

## 7. 一套把How和Why放在一起的最小实验逻辑

### 7.1 无训练部分先回答“信息有没有价值”

1. 从E0/E1的已有checkpoint抽状态，先按训练seed和episode分层，避免只挑失败状态。
2. 对每个合法候选做相同后续策略的分支回放，获得局部候选代价差；保持外生随机条件可配对。
3. 分支规划若要模拟未揭示请求，必须从合法条件模型采样；**不能读取真实测试未来CSV**。真未来版本只作独立Oracle。
4. 比较原观测、增加ready特征、时间对齐和候选差分的简单探针：看代价加权排序错误，不只看MSE。
5. 如果新增信息不改善独立状态排序，就停止对应模型扩张；如果改善排序但RL变差，才优先查融合/优化。

### 7.2 训练部分使用可归因小因子，而非新方法大全

共同基座保持一致；S=服务承诺时间对齐，AC=同背景候选代价差。构成{无S无AC、有S无AC、无S有AC、有S有AC}四臂，另保留不经复杂融合的源基座。

需要同一状态特征集合与相同网络规模的对照，才能区分“信息更多”与“组织方式更好”。辅助目标的更新是否碰actor，应单独开/关而非与表示一起改变。

若四臂代价分别为L00、L10、L01、L11（越低越好），报告交互量L11−L10−L01＋L00；负值支持该尺度上的额外组合收益，但必须按独立训练seed给不确定性，不以单次负值宣布普适协同。

### 7.3 必须多一层“非UAGMC验证”

同样的候选成本接口接到另一种基座，或接到解析短程规划；不要求都赢，但需要解释在哪些控制机会、哪些服务时差下有效。

若增益只在UAGMC-L而不在A版/强规则出现，文章应诚实定位为特定基座适配，不能写成普适交通决策方法。

### 7.4 指标与比较

- 保留用户关心的Best ATT，但Best必须在验证集选checkpoint，再到独立测试回放；直接用同一测试曲线的最小值排名会产生选择偏差。
- 同时报告平台均值、跨seed分布、单run漂移、完成率与未完成处理；completed-only和全体ATT分列。
- 独立单位是训练模型，不是乘客、检查点、回放seed；同一模型多trace不能伪增n。
- 主要目标仍是ATT，不添加平衡/好看曲线奖励；风险、尾部和成本只作额外诊断或明确另立实验。
- 训练步数、数据量、辅助更新次数与执行期计算都记录；“同一步数”不等于“同计算成本”。

## 8. 理论高度应该怎么提升

**可做的：** 给出一个有物理含义的状态/候选抽象，证明它保留什么、忽略什么；给出误差如何影响候选排序的条件；用真实回放测量条件出现频率，再用干预验证机制。

**不该做的：** 把通用POMDP、ICM逆动力学、普通残差或2ε排序界改名；把“有因果图”当识别保证；把“WM预测准”当策略一定更好。

三个层次应分开：

1. **定义/恒等式：** 全体乘客时间等于未完成乘客数的时间积分，需要正确边界和结尾结算；它不是算法创新。
2. **假设下性质：** 某承诺摘要在限定状态/任务下足以区分候选，或误差小于决策margin时不选错；要写出可检验前提。
3. **新增研究贡献：** 为什么这种摘要/分解在在线批服务与机队回流耦合中有效、何时不够，以及同计算预算的实证增量。

如果只能证实第1层，而第2/3层没有新内容，就不应堆定理包装；清晰的交通机制结果仍优于形式复杂但无法验证的“理论高度”。

## 9. 与交通报告合并后的最终路线

**建议从“UAGMC→更多模块”转成“合法已知承诺→可比较的候选服务代价→受预算约束的协调决策”。**

UAGMC保留为重要基线；S/AC保留为时间背景与边际作用的分工；WM只估计未解析尾部；现代AI借鉴结构化RL、决策导向学习和执行期搜索，但先允许“简单解析已经足够”的反结论。

这条路线不保证录用，也尚未证明全球新颖；它比只修一篇文章更独立，同时比直接引入大模型更接近我们的代码、数据与计算能力。

## 附录A：全部93条AI/RL文献

每条保留官方页面/可核验链接、作者、年份、主题、阅读级别与中文迁移提示；逐条归纳不转载整段英文摘要，也不等于完成全文评审。

| ID | 年份／渠道 | 文献与作者 | 中文逐条归纳／适用边界 | 研究用途与优先级 | 阅读 |
|---|---|---|---|---|---|
| A001 | 2025／ICML 2025 | [Accurate and Efficient World Modeling with Masked Latent Transformers](https://proceedings.mlr.press/v267/burchi25a.html)<br>Maxime Burchi; Radu Timofte | 用掩码潜在Transformer改善世界模型压缩的信息损失；可借鉴任务相关信息保留，不应为低维UAM重建像素。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A002 | 2025／ICML 2025 | [Symmetric Reinforcement Learning Loss for Robust Learning on Diverse Tasks and Model Scales](https://proceedings.mlr.press/v267/byun25a.html)<br>Ju-Seung Byun; Andrew Perrault | 针对RL更新中的噪声与方向不确定性提出对称损失；需要先确认我们的不稳定是否来自相同优化机制。 | P2边界：定理假设与约束需独立核验；不能悄悄改变主ATT目标。 | A-primary |
| A003 | 2025／ICML 2025 | [Bellman Unbiasedness: Toward Provably Efficient Distributional Reinforcement Learning with General Value Function Approximation](https://proceedings.mlr.press/v267/cho25a.html)<br>Taehyun Cho; Seungyub Han; Seokhun Ju; Dohyeong Kim; Kyungjae Lee; Jungwoo Lee | 分析分布式RL中可学习统计量的Bellman无偏性与样本效率；理论有有限时域MDP和函数类条件。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A004 | 2025／ICML 2025 | [Improving Transformer World Models for Data-Efficient RL](https://proceedings.mlr.press/v267/dedieu25a.html)<br>Antoine Dedieu; Joseph Ortiz; Xinghua Lou; Carter Wendelken; J Swaroop Guntupalli; Wolfgang Lehrach; Miguel Lazaro-Gredilla; Kevin Patrick Murphy | 先建立强无模型基座，再增加数据高效Transformer世界模型；值得借鉴逐步增量，而非只拿最终大系统比较。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A005 | 2025／ICML 2025 | [Graph World Model](https://proceedings.mlr.press/v267/feng25p.html)<br>Tao Feng; Yexin Wu; Guanyu Lin; Jiaxuan You | 以通用消息传递组织图结构和多模态世界状态；飞机—任务关系可借鉴，但通用图模型不是直接收益保证。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A006 | 2025／ICML 2025 | [Sample Complexity of Distributionally Robust Off-Dynamics Reinforcement Learning with Online Interaction](https://proceedings.mlr.press/v267/he25n.html)<br>Yiting He; Zhishuai Liu; Weixin Wang; Pan Xu | 研究训练与部署动力学不同且只能在线交互的鲁棒学习；固定E0/E1公平比较不应被偷偷替换成另一问题。 | P2边界：定理假设与约束需独立核验；不能悄悄改变主ATT目标。 | A-primary |
| A007 | 2025／ICML 2025 | [Neurosymbolic World Models for Sequential Decision Making](https://proceedings.mlr.press/v267/hernandez-cano25a.html)<br>Leonardo Hernandez Cano; Maxine Perroni-Scharf; Neil Dhir; Arun Ramamurthy; Armando Solar-Lezama | 学习显式有限状态机以表达区域动力学结构，再辅助策略优化；适合思考已知阶段与未知残差的分工。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A008 | 2025／ICML 2025 | [Offline-to-Online Reinforcement Learning with Classifier-Free Diffusion Generation](https://proceedings.mlr.press/v267/huang25v.html)<br>Xiao Huang; Xu Liu; Enze Zhang; Tong Yu; Shuai Li | 用无分类器扩散生成缓解离线到在线的数据分布差；当前应先检验普通解析教师和回放是否足够。 | P2：更适合多峰/多步输出；当前少数离散候选先枚举。 | A-primary |
| A009 | 2025／ICML 2025 | [PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable Reinforcement Learning](https://proceedings.mlr.press/v267/huang25ai.html)<br>Dongchi Huang; Jiaqi Wang; Yang Li; Chunhe Xia; Tianle Zhang; Kaige Zhang | 训练期特权信息帮助部分可观测安全RL；若使用真未来教师必须单列，部署时输入不能泄漏。 | P2边界：定理假设与约束需独立核验；不能悄悄改变主ATT目标。 | A-primary |
| A010 | 2025／ICML 2025 | [Reflect-then-Plan: Offline Model-Based Planning through a Doubly Bayesian Lens](https://proceedings.mlr.press/v267/jeong25a.html)<br>Jihwan Jeong; Xiaoyu Wang; Jingmin Wang; Scott Sanner; Pascal Poupart | 以双层贝叶斯信念组织离线模型规划，并在部署观察后更新动力学信念；可借鉴不确定性适配，不冒称已知事件也需预测。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A011 | 2025／ICML 2025 | [Agent-Centric Actor-Critic for Asynchronous Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v267/jung25a.html)<br>Whiyoung Jung; Sunghoon Hong; Deunsol Yoon; Kanghoon Lee; Woohyung Lim | 按agent独立编码异步宏动作历史，再聚合critic并调整优势；效果延迟不自动等于论文中的决策异步。 | P1有条件：只有真实异步决策/持续动作才适用；与效果延迟分开。 | A-primary |
| A012 | 2025／ICML 2025 | [A Forget-and-Grow Strategy for Deep Reinforcement Learning Scaling in Continuous Control](https://proceedings.mlr.press/v267/kang25c.html)<br>Zilin Kang; Chenyuan Hu; Yu Luo; Zhecheng Yuan; Ruijie Zheng; Huazhe Xu | 通过遗忘与增长缓解连续控制中的早期经验偏置；只在实际测到相同学习障碍时考虑，不直接重置正常平台。 | P1诊断：先测固定状态输出漂移与学习能力；不预设其为波动主因。 | A-primary |
| A013 | 2025／ICML 2025 | [Categorical Distributional Reinforcement Learning with Kullback-Leibler Divergence: Convergence and Asymptotics](https://proceedings.mlr.press/v267/kastner25a.html)<br>Tyler Kastner; Mark Rowland; Yunhao Tang; Murat A Erdogdu; Amir-Massoud Farahmand | 为KL损失的类别分布式RL给收敛与渐近分析；分布式价值与离散动作是两回事。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A014 | 2025／ICML 2025 | [VinePPO: Refining Credit Assignment in RL Training of LLMs](https://proceedings.mlr.press/v267/kazemnejad25a.html)<br>Amirhossein Kazemnejad; Milad Aghajohari; Eva Portelance; Alessandro Sordoni; Siva Reddy; Aaron Courville; Nicolas Le Roux | 检验语言推理中价值估计和中间步骤信用问题；可借鉴分支回报诊断，不把语言稀疏奖替换ATT人时。 | P2跨域：语言/生成推理的反馈语义不同；只借鉴信用诊断，不改ATT目标。 | A-primary |
| A015 | 2025／ICML 2025 | [Reward-free World Models for Online Imitation Learning](https://proceedings.mlr.press/v267/li25af.html)<br>Shangzhe Li; Zhiao Huang; Hao Su | 在潜在空间学习无需重建的动力学，配合逆soft-Q实现在线模仿；专家数据和任务目标仍需公平对齐。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A016 | 2025／ICML 2025 | [Continual Reinforcement Learning by Planning with Online World Models](https://proceedings.mlr.press/v267/liu25p.html)<br>Zichen Liu; Guoji Fu; Chao Du; Wee Sun Lee; Min Lin | 在线浅层模型结合MPC应对持续任务变化和遗忘；其不遗忘性质不能直接解释固定任务的所有seed差异。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A017 | 2025／ICML 2025 | [Distributionally Robust Multi-Agent Reinforcement Learning for Dynamic Chute Mapping](https://proceedings.mlr.press/v267/liu25ad.html)<br>Guangyi Liu; Suzan Iloglu; Michael Caldara; Joseph W Durham; Michael M. Zavlanos | 以组分布鲁棒学习处理仓储分流率波动；与机场分配有结构相似性，但不可为了借方法改我们的主优化目标。 | P2边界：定理假设与约束需独立核验；不能悄悄改变主ATT目标。 | A-primary |
| A018 | 2025／ICML 2025 | [Regret-Free Reinforcement Learning for Temporal Logic Specifications](https://proceedings.mlr.press/v267/majumdar25a.html)<br>R Majumdar; Mahmoud Salamati; Sadegh Soudjani | 将时序逻辑控制归约为可学习的reach-avoid问题并给遗憾保证；安全规范不是乘客ATT的替代指标。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A019 | 2025／ICML 2025 | [Beyond CVaR: Leveraging Static Spectral Risk Measures for Enhanced Decision-Making in Distributional Reinforcement Learning](https://proceedings.mlr.press/v267/moghimi25a.html)<br>Mehrdad Moghimi; Hyejin Ku | 使用静态谱风险测度扩展分布式RL风险决策；适合独立尾部诊断，不悄悄变更平均ATT目标。 | P2边界：定理假设与约束需独立核验；不能悄悄改变主ATT目标。 | A-primary |
| A020 | 2025／ICML 2025 | [Time-Aware World Model for Adaptive Prediction and Control](https://proceedings.mlr.press/v267/nhu25a.html)<br>Anh N Nhu; Sanghyun Son; Ming Lin | 显式以采样时间间隔为条件学习多频动力学；采样间隔与动作真正产生服务的时刻需要区分。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A021 | 2025／ICML 2025 | [Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning](https://proceedings.mlr.press/v267/omura25a.html)<br>Motoki Omura; Kazuki Ota; Takayuki Osa; Yusuke Mukuta; Tatsuya Harada | 在连续actor-critic中渐变使用最优Bellman算子以兼顾学习速度和过估计；不能原样当小离散PPO升级包。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A022 | 2025／ICML 2025 | [Activation by Interval-wise Dropout: A Simple Way to Prevent Neural Networks from Plasticity Loss](https://proceedings.mlr.press/v267/park25b.html)<br>Sangyeon Park; Isaac Han; Seungwon Oh; Kyungjoong Kim | 按激活区间施加不同dropout以维持可塑性；先测死神经元或适应能力，再决定是否需要这种干预。 | P1诊断：先测固定状态输出漂移与学习能力；不预设其为波动主因。 | A-primary |
| A023 | 2025／ICML 2025 | [Flow Q-Learning](https://proceedings.mlr.press/v267/park25f.html)<br>Seohong Park; Qiyang Li; Sergey Levine | 以流模型表达离线行为，同时用一步actor学价值改进，避免递归反传；少数离散动作不必因此生成化。 | P2：更适合多峰/多步输出；当前少数离散候选先枚举。 | A-primary |
| A024 | 2025／ICML 2025 | [Scaling Laws for Pre-training Agents and World Models](https://proceedings.mlr.press/v267/pearce25a.html)<br>Tim Pearce; Tabish Rashid; David Bignell; Raluca Georgescu; Sam Devlin; Katja Hofmann | 量化模仿和世界模型预训练的数据—模型—计算关系；其幂律不是我们的PPO步数普适预算公式。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A025 | 2025／ICML 2025 | [General agents need world models](https://proceedings.mlr.press/v267/richens25a.html)<br>Jonathan Richens; Tom Everitt; David Abel | 在限定泛化任务类下讨论策略隐含预测模型的必要性；不等于每个实际任务都要显式训练大世界模型。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A026 | 2025／ICML 2025 | [A Reduction Framework for Distributionally Robust Reinforcement Learning under Average Reward](https://proceedings.mlr.press/v267/roch25a.html)<br>Zachary Andrew Roch; George K. Atia; Yue Wang | 把鲁棒平均收益问题归约到折扣问题以便学习；长期平均收益与有限人时清空任务要分开。 | P2边界：定理假设与约束需独立核验；不能悄悄改变主ATT目标。 | A-primary |
| A027 | 2025／ICML 2025 | [Value-Based Deep RL Scales Predictably](https://proceedings.mlr.press/v267/rybkin25a.html)<br>Oleh Rybkin; Michal Nauman; Preston Fu; Charlie Victor Snell; Pieter Abbeel; Sergey Levine; Aviral Kumar | 用更新/数据比描述价值学习的数据—计算前沿；可指导预算记录，不据此保证长训性能单调。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A028 | 2025／ICML 2025 | [Online Pre-Training for Offline-to-Online Reinforcement Learning](https://proceedings.mlr.press/v267/shin25c.html)<br>Yongjae Shin; Jeonghye Kim; Whiyoung Jung; Sunghoon Hong; Deunsol Yoon; Youngsoo Jang; Geon-Hyeong Kim; Jongseong Chae; Youngchul Sung; Kanghoon Lee; Woohyung Lim | 增加在线预训练环节修复离线价值估计的分布偏移；对解析教师后再RL的路线尤其值得检查。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A029 | 2025／ICML 2025 | [Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](https://proceedings.mlr.press/v267/tang25g.html)<br>Hongyao Tang; Johan Obando-Ceron; Pablo Samuel Castro; Aaron Courville; Glen Berseth | 从输出churn与NTK秩下降解释持续学习可塑性，提出近似抑制策略；适合固定状态漂移诊断而非直接宣布归因。 | P1诊断：先测固定状态输出漂移与学习能力；不预设其为波动主因。 | A-primary |
| A030 | 2025／ICML 2025 | [What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](https://proceedings.mlr.press/v267/vafa25a.html)<br>Keyon Vafa; Peter G. Chang; Ashesh Rambachan; Sendhil Mullainathan | 以归纳偏置探针检验基础模型是否真正学到所假设的结构；预测准确与学到正确机制不能混同。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A031 | 2025／ICML 2025 | [Trajectory World Models for Heterogeneous Environments](https://proceedings.mlr.press/v267/yin25f.html)<br>Shaofeng Yin; Jialong Wu; Siqiao Huang; Xingjian Su; Xu He; Jianye Hao; Mingsheng Long | 用异构环境轨迹和灵活传感/动作结构预训练动力学；可借鉴跨K接口，但大规模数据前提不能省略。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A032 | 2025／ICML 2025 | [World Model Implanting for Test-time Adaptation of Embodied Agents](https://proceedings.mlr.press/v267/yoo25a.html)<br>Minjong Yoo; Jinwoo Jang; Sihyung Yoon; Honguk Woo | 检索并在测试期组合领域世界模型与语言推理；当前低维、已知物理UAM不宜直接照搬整套大模型。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A033 | 2025／ICML 2025 | [Minimax Optimal Regret Bound for Reinforcement Learning with Trajectory Feedback](https://proceedings.mlr.press/v267/zhang25l.html)<br>Zihan Zhang; Yuxin Chen; Jason D. Lee; Simon Shaolei Du; Ruosong Wang | 分析只能观察整段累计奖励时的最优遗憾；我们可逐分钟算人时，无需人为丢弃已有稠密反馈。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A034 | 2025／ICML 2025 | [Logarithmic Regret for Online KL-Regularized Reinforcement Learning](https://proceedings.mlr.press/v267/zhao25ac.html)<br>Heyang Zhao; Chenlu Ye; Wei Xiong; Quanquan Gu; Tong Zhang | 分析KL正则在线学习的效率与遗憾；正则化改变了目标，不能把保证无条件套到任意PPO实现。 | P2跨域：语言/生成推理的反馈语义不同；只借鉴信用诊断，不改ATT目标。 | A-primary |
| A035 | 2025／ICML 2025 | [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://proceedings.mlr.press/v267/zhou25t.html)<br>Gaoyue Zhou; Hengkai Pan; Yann Lecun; Lerrel Pinto | 用冻结视觉特征学习动力学，并在测试期规划而不重建像素；可学其任务相关预测思想，不必引入视觉主干。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A036 | 2025／ICML 2025 | [Stay Hungry, Keep Learning: Sustainable Plasticity for Deep Reinforcement Learning](https://proceedings.mlr.press/v267/zhou25am.html)<br>Huaicheng Zhou; Zifeng Zhuang; Donglin Wang | 通过模型重生并恢复已学知识维持持续学习能力；应分别测可达性和保留性，避免重置毁掉好策略。 | P1诊断：先测固定状态输出漂移与学习能力；不预设其为波动主因。 | A-primary |
| A037 | 2026／ICML 2026 | [Intrinsic Credit Assignment for Long Horizon Interaction](https://proceedings.mlr.press/v306/auzina26a.html)<br>Ilze Amanda Auzina; Joschka Strüber; Sergio Hernández-Gutiérrez; Shashwat Goel; Ameya Prabhu; Matthias Bethge | 以语言模型对目标信念的变化提供中间信用；适用于信息搜寻交互，不应转为我们的人工辅助奖励。 | P2跨域：语言/生成推理的反馈语义不同；只借鉴信用诊断，不改ATT目标。 | A-primary |
| A038 | 2026／ICML 2026 | [Efficient and Uncertainty-Aware Diffusion Framework for Offline-to-Online Reinforcement Learning](https://proceedings.mlr.press/v306/bui26a.html)<br>Ha Manh Bui; Metod Jazbec; Eric Nalisnick; Anqi Liu | 离线阶段蒸馏快速扩散策略和转移模型，再度量分布迁移不确定性；可参考教师偏移检查，当前不优先扩散。 | P2：更适合多峰/多步输出；当前少数离散候选先枚举。 | A-primary |
| A039 | 2026／ICML 2026 | [Direct Flow Q-Learning](https://proceedings.mlr.press/v306/cao26f.html)<br>Shicheng Cao; Jingrui Jia; Wenyu Li; Feng Duan; Tao Zhang; Shengbo Eben Li | 不用递归反传或一步蒸馏，直接向流速度场注入末端Q梯度；保留迭代表达力但仍需解释执行成本。 | P2：更适合多峰/多步输出；当前少数离散候选先枚举。 | A-primary |
| A040 | 2026／ICML 2026 | [The Surprising Difficulty of Search in Model-Based Reinforcement Learning](https://proceedings.mlr.press/v306/chang26a.html)<br>Wei-Di Chang; Mikael Henaff; Brandon Amos; Gregory Dudek; Scott Fujimoto | 发现准确模型上的搜索也会失败，价值过估计比预测精度更关键；直接约束我们短程搜索加尾值的论证。 | P0：优先核对小动作适配、成本对齐、排序误差或基座保护。 | A-primary |
| A041 | 2026／ICML 2026 | [Policy-Driven World Model Adaptation for Robust Offline Model-based Reinforcement Learning](https://proceedings.mlr.press/v306/chen26fl.html)<br>Jiayu Chen; Le Xu; Aravind Venugopal; Jeff Schneider | 使离线世界模型随策略目标适配，缓解预测似然与控制收益不一致；先把已知物理与待学部分分开。 | P2边界：定理假设与约束需独立核验；不能悄悄改变主ATT目标。 | A-primary |
| A042 | 2026／ICML 2026 | [Balancing Plasticity and Stability with Fast and Slow Successor Features](https://proceedings.mlr.press/v306/chua26a.html)<br>Raymond Chua; Doina Precup; Blake Aaron Richards | 快慢后继特征平衡连续环境漂移中的适应与稳定；当前若没有环境漂移，只借鉴而不直接套结论。 | P1诊断：先测固定状态输出漂移与学习能力；不预设其为波动主因。 | A-primary |
| A043 | 2026／ICML 2026 | [Multivariate Distributional Reinforcement Learning Using Sliced Divergences](https://proceedings.mlr.press/v306/debes26a.html)<br>Baptiste Debes; Tinne Tuytelaars | 通过投影后的散度学习多维回报分布并分析收缩；额外分量可作诊断，不能替换冻结的单目标ATT。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A044 | 2026／ICML 2026 | [Compositional Planning with Jumpy World Models](https://proceedings.mlr.press/v306/farebrother26a.html)<br>Jesse Farebrother; Matteo Pirotta; Andrea Tirinzoni; Marc G Bellemare; Alessandro Lazaric; Ahmed Touati | 学习预训练策略的多步占据动力学以支持组合规划；可以启发跨事件跳跃，但不是任意未来轨迹真值。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A045 | 2026／ICML 2026 | [Learning Task-Sufficient World Models by Synergizing Agentic Exploration and Structured Modeling](https://proceedings.mlr.press/v306/feng26aa.html)<br>Fan Feng; Yujia Zheng; Minghao Fu; Yongqiang Chen; Guangyi Chen; Kevin Patrick Murphy; Biwei Huang; Kun Zhang | 结构化模型与主动探测共同学习任务充分表示；充分性必须相对具体决策任务检验，不靠表示完整性自证。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A046 | 2026／ICML 2026 | [Scalable Simulation-Based Model Inference with Test-Time Complexity Control](https://proceedings.mlr.press/v306/gloeckler26a.html)<br>Manuel Gloeckler; J.P. Manzano-Patrón; Stamatios Sotiropoulos; Cornelius Schröder; Jakob H. Macke | 推断仿真模型结构及参数的联合后验，并允许测试期调复杂度；研究对象主要是模型选择，不是直接的UAM调度算法。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A047 | 2026／ICML 2026 | [Secure Multi-agent Reinforcement Learning for Service Systems with Affinity and Byzantine Nodes: Stability Analysis and Protection Design](https://proceedings.mlr.press/v306/jiang26u.html)<br>Yifan Jiang; Pan Jiasheng; Mengtian Li; Li Jin | 在服务网络与拜占庭节点下分析分散actor-critic的排队稳定；安全通信假设与我们的同账号多进程无关。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A048 | 2026／ICML 2026 | [Boosting World Models Learning via Latent-Space Value Alignment](https://proceedings.mlr.press/v306/jiang26ai.html)<br>Xingyu Jiang; Yuheng Pan; Mukang You; Xiuhui Zhang; Ning Gao; Guanwei Yan; Hao Li; Yue Deng | 将概率动力学拟合与价值相关潜在学习结合；只预测影响候选排序的尾部，避免完整重建成为无关负担。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A049 | 2026／ICML 2026 | [Mixture of Horizons in Action Chunking](https://proceedings.mlr.press/v306/jing26f.html)<br>Dong Jing; Gang Wang; Jiaqi Liu; Weiliang Tang; Zelong Sun; Yunchao Yao; Zhenyu Wei; Yunhui Liu; Zhiwu Lu; Mingyu Ding | 混合不同动作块时域以兼顾全局前瞻与局部精度；当前不可逆逐客分配需要额外可行性处理。 | P2：更适合多峰/多步输出；当前少数离散候选先枚举。 | A-primary |
| A050 | 2026／ICML 2026 | [Credit Assignment via Neural Manifold Noise Correlation](https://proceedings.mlr.press/v306/kang26n.html)<br>Byungwoo Kang; Maceo Richards; Bernardo L. Sabatini | 用低维神经流形噪声估计神经元/突触信用；这里的信用不是多agent贡献，降低迁移优先级。 | P3不同义：研究神经元/突触信用，不等于乘客—飞机agent信用。 | A-primary |
| A051 | 2026／ICML 2026 | [Decision-Focused Learning via Tangent-Space Projection of Prediction Error](https://proceedings.mlr.press/v306/lee26q.html)<br>Junhyeong Lee; Sangjin Jin; Yongjae Lee | 在局部活跃约束稳定等条件下提取决策相关预测误差；离散组批队列不能直接继承连续优化梯度公式。 | P0：优先核对小动作适配、成本对齐、排序误差或基座保护。 | A-primary |
| A052 | 2026／ICML 2026 | [Causal Flow Q-Learning for Robust Offline Reinforcement Learning](https://proceedings.mlr.press/v306/li26ja.html)<br>Mingxuan Li; Junzhe Zhang; Elias Bareinboim | 考虑示范者与学习者观测不同形成的潜在混杂，鲁棒优化离线流策略；不存在同类混杂时不强造因果故事。 | P2：更适合多峰/多步输出；当前少数离散候选先枚举。 | A-primary |
| A053 | 2026／ICML 2026 | [Implicit Action Chunking for Smooth Continuous Control](https://proceedings.mlr.press/v306/liang26s.html)<br>Bosun Liang; Shuo Pei; Zirui Chen; Chuanzhi Fan; Chen Sun; Yuankai Wu; Huachun Tan; Yong Wang | 用双窗口平滑连续控制，不扩大显式动作块输出维度；机场离散选择不能以数值平滑替代合法决策。 | P2：更适合多峰/多步输出；当前少数离散候选先枚举。 | A-primary |
| A054 | 2026／ICML 2026 | [Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization](https://proceedings.mlr.press/v306/luo26u.html)<br>Yifu Luo; Haoyuan Sun; Xinhao Hu; Penghui Du; Keyu Fan; Bo Li; Sinan Du; Xu Wan; Zhiyu Chen; Bo Xia; Yongzhe Chang; Kai Wu; Kun Gai; Tiantian Zhang; Xueqian Wang | 把图像生成流过程的强化学习更新从步级转为块级；生成去噪步骤并不等于实际飞机运行阶段。 | P2跨域：语言/生成推理的反馈语义不同；只借鉴信用诊断，不改ATT目标。 | A-primary |
| A055 | 2026／ICML 2026 | [SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning](https://proceedings.mlr.press/v306/luo26ah.html)<br>Lirui Luo; Guoxi Zhang; Hongming Xu; Cong Fang; Qing Li | 从谱可塑性解释MoE持续RL退化并给可计算代理；先有MoE和相同失败证据才值得迁移。 | P1诊断：先测固定状态输出漂移与学习能力；不预设其为波动主因。 | A-primary |
| A056 | 2026／ICML 2026 | [What Makes Value Learning Efficient in Residual Reinforcement Learning?](https://proceedings.mlr.press/v306/ma26s.html)<br>Guozheng Ma; Lu Li; Haoyu Wang; Zixuan Liu; Pierre-Luc Bacon; Dacheng Tao | 识别残差critic冷启动及尺度失配，以基座数据锚定和规范化修复；连续残差保证不能直接套离散logit。 | P0：优先核对小动作适配、成本对齐、排序误差或基座保护。 | A-primary |
| A057 | 2026／ICML 2026 | [Causal-JEPA: Learning World Models through Object-Level Latent Masking](https://proceedings.mlr.press/v306/nam26c.html)<br>Heejeong Nam; Quentin Le Lidec; Lucas Maes; Yann Lecun; Randall Balestriero | 对象级潜在遮挡迫使模型利用交互关系，减少快捷预测；这种反事实式训练查询本身不是因果识别证明。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A058 | 2026／ICML 2026 | [Fast and Highly Expressive Policy Learning for Offline Reinforcement Learning via Bootstrapped Flow Q-Learning](https://proceedings.mlr.press/v306/nguyen26p.html)<br>Thanh Xuan Nguyen; Tri Ton; Hongbin Choe; Tung Minh Luu; Chang D. Yoo | 以自举流学习实现一步动作生成，降低迭代采样和蒸馏负担；先与简单离散候选枚举比复杂度。 | P2：更适合多峰/多步输出；当前少数离散候选先枚举。 | A-primary |
| A059 | 2026／ICML 2026 | [Test-Time Graph Search for Goal-Conditioned Reinforcement Learning](https://proceedings.mlr.press/v306/opryshko26a.html)<br>Evgenii Opryshko; Junwei Quan; Claas A Voelcker; Yilun Du; Igor Gilitschenski | 利用离线状态图与局部价值几何在测试期选择子目标；可借鉴轻量规划，但机场调度状态未必有相同几何。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A060 | 2026／ICML 2026 | [Scalable Reinforcement Learning via Adaptive Batch Scaling](https://proceedings.mlr.press/v306/park26w.html)<br>Jongchan Park | 随训练中非平稳程度变化调整有效batch；属于优化器实验，不能与新表示一起改动后归因给表示。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A061 | 2026／ICML 2026 | [Parallel Stochastic Gradient-Based Planning for World Models](https://proceedings.mlr.press/v306/psenka26a.html)<br>Michael Psenka; Michael Rabbat; Aditi S. Krishnapriyan; Yann Lecun; Amir Bar | 以虚拟状态和软动力学约束并行规划，缓解长链梯度；精确离散仿真不一定需要可微的像素规划器。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A062 | 2026／ICML 2026 | [Behavior-Invariant Task Representation Learning with Transformer-based World Models for Offline Meta-Reinforcement Learning](https://proceedings.mlr.press/v306/qian26o.html)<br>Fuyuan Qian; Menglong Zhang; Song Wang; Quanying Liu | 抽取行为不变的任务表示并用Transformer模型处理离线元RL分布迁移；跨任务前提与固定trace实验不同。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A063 | 2026／ICML 2026 | [Focus-Then-Contact: Speeding Up Robotic Contact-Rich Task Learning with Affordance-Guided Real-World Residual Reinforcement Learning](https://proceedings.mlr.press/v306/qiao26f.html)<br>Guanren Qiao; Ruixiang Ouyang; Sheng Xu; Ruixing Jin; Yueci Deng; Yunxin Tai; Kui Jia; Guiliang Liu | 用残差策略和可供性引导加速真实接触控制；其额外机器人奖励不应照搬到乘客ATT合同。 | P1诊断：先测固定状态输出漂移与学习能力；不预设其为波动主因。 | A-primary |
| A064 | 2026／ICML 2026 | [Preserving Plasticity in Continual Learning via Dynamical Isometry](https://proceedings.mlr.press/v306/rosseau26a.html)<br>Andries Rosseau; Robert Müller; Ann Nowe | 以层间Jacobian近等距关系研究持续学习的可塑性；可借鉴梯度流检查，但不能凭理论名字宣布我们的故障。 | P1诊断：先测固定状态输出漂移与学习能力；不预设其为波动主因。 | A-primary |
| A065 | 2025／ICML 2025 | [Hyperspherical Normalization for Scalable Deep Reinforcement Learning](https://proceedings.mlr.press/v267/lee25u.html)<br>Hojoon Lee; Youngdo Lee; Takuma Seno; Donghu Kim; Peter Stone; Jaegul Choo | 超球面约束权重与特征范数，结合分布价值及尺度处理稳定SAC；作为现代强基座而非无条件替换原PPO。 | P0：优先核对小动作适配、成本对齐、排序误差或基座保护。 | A-primary |
| A066 | 2025／ICLR 2025 | [SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/741aab8b41a2987867acc9939ad50383-Abstract-Conference.html)<br>Hojoon Lee et al. | 输入规范化、残差通路与层规范化形成简单可扩展RL网络；适合作为复杂网络是否必要的强对照。 | P0：优先核对小动作适配、成本对齐、排序误差或基座保护。 | A-primary |
| A067 | 2025／IJCAI 2025 | [Asynchronous Credit Assignment for Multi-Agent Reinforcement Learning](https://ijcai.org/proceedings/2025/20)<br>Yongheng Liang; Hejun Wu; Haitao Wang; Hao Cai | 用虚拟同步代理和乘性价值分解处理真实异步动作依赖；需核对持续动作与经验时间语义。 | P1有条件：只有真实异步决策/持续动作才适用；与效果延迟分开。 | A-primary |
| A068 | 2025／CoRL 2025（作者官方代码确认） | [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://diffusion-steering.github.io/)<br>Andrew Wagenmaker et al. | 冻结扩散行为基座，在潜在空间训练操纵策略；可学基座保护原则，不要求当前改成扩散策略。 | P2：更适合多峰/多步输出；当前少数离散候选先枚举。 | A-primary |
| A069 | 2025／Nature | [Discovering state-of-the-art reinforcement learning algorithms](https://www.nature.com/articles/s41586-025-09761-x)<br>Junhyuk Oh et al. | 跨环境元优化发现RL更新规则并测试未见环境；发现阶段算力巨大，更适合作为趋势而非当前实现任务。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A070 | 2026／ICLR 2026 | [One-Step Flow Q-Learning: Addressing the Diffusion Policy Bottleneck in Offline Reinforcement Learning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/14f75513f0f1ca01de1e826b52e6b840-Abstract-Conference.html)<br>Xuan Thanh Nguyen; Chang Yoo | 以平均速度场学习一步流动作生成，绕开迭代去噪瓶颈；对多峰连续动作更自然。 | P2：更适合多峰/多步输出；当前少数离散候选先枚举。 | A-primary |
| A071 | 2025／技术报告／预印本；未核实正式会刊 | [FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control](https://arxiv.org/abs/2505.22642)<br>Younggyo Seo et al. | 高吞吐TD3针对大量并行的人形连续控制；公开技术报告不冒充已核实的会议论文。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A072 | 2025／NeurIPS 2025 | [Forecasting in Offline Reinforcement Learning for Non-stationary Environments](https://papers.nips.cc/paper_files/paper/2025/hash/0bc91addac6b72fb970c678a936dacf0-Abstract-Conference.html)<br>Ada, Suzan Ece; Martius, Georg; Ugur, Emre; Oztop, Erhan | 通过候选状态生成应对非平稳离线环境中的偏移；只有测试确有动力学变化时才是直接问题。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A073 | 2025／NeurIPS 2025 | [Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2025/hash/0c8d49f2744d8ee4d5948621d6bb5bc0-Abstract-Conference.html)<br>Anand, Emile; Karmarkar, Ishani; Qu, Guannan | 用抽样均值场近似缓解大量agent的联合空间爆炸；我们机场数不等于agent数，小规模无需先用近似。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A074 | 2025／NeurIPS 2025 | [Continuous Q-Score Matching: Diffusion Guided Reinforcement Learning for Continuous-Time Control](https://papers.nips.cc/paper_files/paper/2025/hash/1286e85106eb9c52eb77f617596dd917-Abstract-Conference.html)<br>HUA, Chengxiu; Gu, Jiawen; Tang, Yushun | 由连续时间Q的鞅条件连接扩散策略得分与动作梯度；连续随机系统假设不直接适合分钟级离散组批。 | P2：更适合多峰/多步输出；当前少数离散候选先枚举。 | A-primary |
| A075 | 2025／NeurIPS 2025 | [Mind the GAP! The Challenges of Scale in Pixel-based Deep Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2025/hash/1d5dc0384137d46a197c484873f5457f-Abstract-Conference.html)<br>Sokar, Ghada; Castro, Pablo Samuel | 发现像素encoder到全连接层的瓶颈限制扩展，以全局平均池化改进；此池化不是交通资源池化理论。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A076 | 2025／NeurIPS 2025 | [Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions](https://papers.nips.cc/paper_files/paper/2025/hash/1d96679f099976d88744c3ad48311def-Abstract-Conference.html)<br>Long, Kehan; Cortes, Jorge; Atanasov, Nikolay | 从价值函数加动力学残差构造广义Lyapunov证书；排队稳定与单run学习曲线稳定不是同一概念。 | P2边界：定理假设与约束需独立核验；不能悄悄改变主ATT目标。 | A-primary |
| A077 | 2025／NeurIPS 2025 | [Approximating Shapley Explanations in Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2025/hash/204e1588746cd47df9f61f7acd14014c-Abstract-Conference.html)<br>Beechey, Daniel; Şimşek, Özgür | 近似Shapley解释并处理时序依赖以降低RL解释成本；解释归因不等同于已识别的交通因果机制。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A078 | 2025／NeurIPS 2025 | [Breaking the Performance Ceiling in Reinforcement Learning requires Inference Strategies](https://papers.nips.cc/paper_files/paper/2025/hash/2226e1facc28d7cc36cb1f9d1d82b991-Abstract-Conference.html)<br>Chalumeau, Felix; Rajaonarivonivelomanantsoa, Daniel; de Kock, Ruan John; Formanek, Juan; Abramowitz, Sasha; Mahjoub, Omayma; Khlifi, Wiem; Du Toit, Simon; Nessir, Louay; Shabe, Refiloe; Fokam, Arnol; Singh, Siddarth; Mbou Sob, Ulrich Armel; Pretorius, Arnu | 利用执行期多次尝试突破一次前向策略的平台；需与相同计算预算的普通搜索比较。 | P0：优先核对小动作适配、成本对齐、排序误差或基座保护。 | A-primary |
| A079 | 2025／NeurIPS 2025 | [Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2025/hash/22f6de59f132ad379e65fcef966c1593-Abstract-Conference.html)<br>Korkmaz, Ezgi | 研究训练中反向作用式动作经验以改善探索效率；与候选动作共享背景的代价差分应区分。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A080 | 2025／NeurIPS 2025 | [Real-Time Execution of Action Chunking Flow Policies](https://papers.nips.cc/paper_files/paper/2025/hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html)<br>Black, Kevin; Galliker, Manuel; Levine, Sergey | 在不重训的情况下异步执行Flow/扩散动作块，处理延迟与块边界；实际乘客承诺不能被随意重写。 | P2：更适合多峰/多步输出；当前少数离散候选先枚举。 | A-primary |
| A081 | 2025／NeurIPS 2025 | [Stable Gradients for Stable Learning at Scale in Deep Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2025/hash/32375260090404f907ceae19f3564a7e-Abstract-Conference.html)<br>Creus Castanyer, Roger; Obando Ceron, Johan; Li, Lu; Bacon, Pierre-Luc; Berseth, Glen; Courville, Aaron; Castro, Pablo Samuel | 把网络扩展退化定位到非平稳与梯度路径问题，并提出简单干预；适合先测后修而非先堆模块。 | P1诊断：先测固定状态输出漂移与学习能力；不预设其为波动主因。 | A-primary |
| A082 | 2025／NeurIPS 2025 | [A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2025/hash/377d0752059d3d4686aa021b664a25dd-Abstract-Conference.html)<br>Hu, Yuzheng; Wu, Fan; Ye, Haotian; Forsyth, David; Zou, James; Jiang, Nan; Ma, Jiaqi; Zhao, Han | 将PPO某检查点的行为联系到最近buffer中的样本；局部数据归因不代表全训练路径的因果结论。 | P0诊断：可参考局部PPO数据归因；不等于全训练因果追责。 | A-primary |
| A083 | 2025／NeurIPS 2025 | [Learning and Planning Multi-Agent Tasks via an MoE-based World Model](https://papers.nips.cc/paper_files/paper/2025/hash/3a2d96d2eb2902043c2db705ca03e9a2-Abstract-Conference.html)<br>Zhao, Zijie; Zhao, Zhongyue; Xu, Kaixuan; Fu, Yuqian; Chai, Jiajun; Zhu, Yuanheng; Zhao, Dongbin | 按任务动力学的相似组用MoE世界模型支持多任务规划；固定E1未必需要混合专家。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A084 | 2025／NeurIPS 2025 | [Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2025/hash/3e8d9bf1dd1eb9d3d9d500fb3543c87b-Abstract-Conference.html)<br>Li, Simin; Mao, Zihao; Li, Hanxiao; Jing, Zonglei; bian, Zhuohang; Guo, Jun; Wang, Li; Han, Zhuoran; Xu, Ruixiao; Yu, Xin; Ma, Chengdong; Ma, Yuqing; An, Bo; Yang, Yaodong; Lv, Weifeng; Liu, Xianglong | 大规模比较MARL合作性能、扰动稳健性与恢复能力；提醒平均奖励高不等于稳健，但不据其数量推我们结论。 | P2边界：定理假设与约束需独立核验；不能悄悄改变主ATT目标。 | A-primary |
| A085 | 2025／NeurIPS 2025 | [A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2025/hash/c7f812431607ac5a17973bbf79733013-Abstract-Conference.html)<br>Liu, Anjie; Wang, Jianhong; Kaski, Samuel; Wang, Jun; Yang, Mengyue | 用多agent影响图分析定向干预和协调机制；因果/交互图已有直接近邻，需要避免仅改名重复。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A086 | 2025／NeurIPS 2025 | [Confounding Robust Deep Reinforcement Learning: A Causal Approach](https://papers.nips.cc/paper_files/paper/2025/hash/c9a177f06ecd9892cd1664d89994fb58-Abstract-Conference.html)<br>Li, Mingxuan; Zhang, Junzhe; Bareinboim, Elias | 针对带隐藏混杂的离线经验构建鲁棒DQN；可控模拟器中若无同类偏差，不应套因果结论。 | P2边界：定理假设与约束需独立核验；不能悄悄改变主ATT目标。 | A-primary |
| A087 | 2025／NeurIPS 2025 | [Role-aware Multi-agent Reinforcement Learning for Coordinated Emergency Traffic Control](https://papers.nips.cc/paper_files/paper/2025/hash/cb22704ebc9bc4e0ef875d1959651012-Abstract-Conference.html)<br>Cheng, Ming; Chen, Hao; Li, Zhiqing; Wang, Jia; Wang, Senzhang | 按应急交通关系动态分配车辆/信号角色，用异构时序图协同；是任务结构驱动MARL的交通近邻。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A088 | 2025／NeurIPS 2025 | [Online Optimization for Offline Safe Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2025/hash/cdc099a583d760335ee9b5ab21626a48-Abstract-Conference.html)<br>Chemingui, Yassine; Deshwal, Aryan; Fern, Alan; Nguyen-Tang, Thanh; Doppa, Jana | 把离线安全RL转为极小极大与在线优化组合；安全累计约束的保证不等于ATT最优保证。 | P2边界：定理假设与约束需独立核验；不能悄悄改变主ATT目标。 | A-primary |
| A089 | 2025／NeurIPS 2025 | [Trajectory Graph Learning: Aligning with Long Trajectories in Reinforcement Learning Without Reward Design](https://papers.nips.cc/paper_files/paper/2025/hash/d3f39e51f5f634fb16cc3e658f8512b9-Abstract-Conference.html)<br>Li, Yunfan; Liu, Eric; Yang, Lin | 用专家标注轨迹图保持长程模仿一致性；可辅助教师计划学习，不替换已明确的人时目标。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A090 | 2025／NeurIPS 2025 | [Gymnasium: A Standard Interface for Reinforcement Learning Environments](https://papers.nips.cc/paper_files/paper/2025/hash/d7ff1795e8527f6443371c3933bdb52b-Abstract-Datasets_and_Benchmarks_Track.html)<br>Towers, Mark; Kwiatkowski, Ariel; Balis, John; De Cola, Gianluca; Deleu, Tristan; Goulão, Manuel; Andreas, Kallinteris; Krimmel, Markus; KG, Arjun; Perez-Vicente, Rodrigo; Terry, J; Pierré, Andrea; Schulhoff, Sander; Tai, Jun Jet; Tan, Hannah; Younis, Omar G. | 规范环境与算法接口，支持更可靠复现；终止与截断语义是基础校验，不是新交通方法。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A091 | 2025／NeurIPS 2025 | [Zero-shot World Models via Search in Memory](https://papers.nips.cc/paper_files/paper/2025/hash/da7d7341051b4e0444cc96c78f1e4df9-Abstract-Conference.html)<br>Malato, Federico; Hautamäki, Ville | 通过记忆检索近似潜在动力学而不额外训练；准确性仍需检验，不能当精确物理的默认替代。 | P1/P2：只学未解析、任务相关尾部；已知物理优先算，不做全量重建。 | A-primary |
| A092 | 2025／NeurIPS 2025 | [FANS: A Flatness-Aware Network Structure for Generalization in Offline Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2025/hash/db1d5c63576587fc1d40d33a75190c71-Abstract-Conference.html)<br>Wang, Da; Ma, Yi; Guo, Ting; Tang, Hongyao; Wei, Wei; Liang, Jiye | 以结构引导较平坦的离线策略解改善泛化；若迁移应与同数据普通网络比较。 | P2：为机制/训练对照提供线索；完整迁移仍需原文与源码核验。 | A-primary |
| A093 | 2025／NeurIPS 2025 | [Structured Reinforcement Learning for Combinatorial Decision-Making](https://papers.nips.cc/paper_files/paper/2025/hash/fe100e5e7bca984803f0e4c6e94a6a88-Abstract-Conference.html)<br>Hoppe, Heiko; Baty, Léo; Bouvier, Louis; Parmentier, Axel; Schiffer, Maximilian | 把组合优化层放入actor，用可训练损失连接神经成本和合法决策；是控制—方法深度融合的强直接参照。 | P0：优先核对小动作适配、成本对齐、排序误差或基座保护。 | A-primary |

## 附录B：主题、年份与来源统计

统计的是本轮目的性文献池，不用于推断全领域发表比例。

| 年份 | 条目数 |
|---|---:|
| 2025 | 64 |
| 2026 | 29 |

| 主题 | 条目数 |
|---|---:|
| 规划与任务相关模型 | 26 |
| 其他RL与决策学习 | 25 |
| 优化稳定性与基座保护 | 13 |
| 风险、鲁棒性与理论 | 12 |
| 生成策略与离线到在线 | 11 |
| 异步决策与信用 | 6 |

题名高频词（去除通用停用词）：offline=12；online=11；agent=10；multi=9；robust=7；control=7；flow=7；efficient=6；planning=6；time=6；plasticity=6；diffusion=5。

引用快照：OpenAlex，2026-10-29；引用量不是质量评分，2026论文受时间窗口影响明显。

取得可用OpenAlex引用快照：75/93条；以下为可匹配子集的前5，不是完整领域高被引榜。

| 文献 | 第一作者 | 引用快照 |
|---|---|---:|
| [Gymnasium: A Standard Interface for Reinforcement Learning Environments](https://papers.nips.cc/paper_files/paper/2025/hash/d7ff1795e8527f6443371c3933bdb52b-Abstract-Datasets_and_Benchmarks_Track.html) | Towers, Mark | 43 |
| [Discovering state-of-the-art reinforcement learning algorithms](https://www.nature.com/articles/s41586-025-09761-x) | Junhyuk Oh et al. | 17 |
| [Real-Time Execution of Action Chunking Flow Policies](https://papers.nips.cc/paper_files/paper/2025/hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html) | Black, Kevin | 12 |
| [Asynchronous Credit Assignment for Multi-Agent Reinforcement Learning](https://ijcai.org/proceedings/2025/20) | Yongheng Liang | 6 |
| [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://proceedings.mlr.press/v267/zhou25t.html) | Gaoyue Zhou | 5 |

与题名相近但不完全匹配者不自动合并；预印本与会刊引用可能分裂，具体原始定位保留在JSON。

## 未完成项与研究诚信

**UAGMC会议版全文精读尚未完成**；其公开获取限制已记录，本轮按用户意见以期刊全文为主，会议—期刊逐页差异不下结论。其他三份用户指定范例已取得公开作者稿，核读范围明确，不把接受稿自动说成最终版。

本次为AI辅助、有界文献扫描与研究构思；未上传私有训练数据到外部研究服务，未运行新的训练，未改动项目环境或声称外部专家审核通过。所有“我们建议”均为待检验设计，不是本项目已证实结论。
