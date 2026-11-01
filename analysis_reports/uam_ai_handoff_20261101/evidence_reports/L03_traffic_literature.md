# 交通/UAM文献复审：哪些路已经走过，我们还值得做什么？

检索截止：2026-10-29。面向ITSC2027；兼顾T-ITS、T-VT、TR-C。本文是文献与研究设计报告，不是新实验结果。未连接服务器、未启动训练、未改物理环境。

## 先读这六条结论

1. **你的担忧成立：仅把UAGMC搬到闭环环境、再增加一个老表示模块，作为主贡献偏弱。** 闭环运营、空机调机、充电与联合优化均已有直接研究，不能再笼统声称这些问题未解决。
2. **但“已有闭环研究”不等于“我们的在线联合问题没有空间”。** 要把问题收窄到：乘客选站改变未来工作量、飞机调机改变未来供给，如何在已承诺事件和有限计算时间下作出协调决策。
3. **不能只打败UAGMC或作者三条规则。** 同权限的滚动优化、短视边际成本规则、普通RL和现代结构化RL都可能比它们更有挑战性。
4. **最值得推进的是“决策导向的短程推演＋只学未解析部分”，而不是再训练一个庞大未来预测器。** 这是本报告的迁移判断，不是已经证明的新方法，也不是全球首次声明。
5. **S/AC可以保留，但身份要变。** S用于把供需放到同一服务时刻比较；AC用于估计候选动作相对同一背景的边际代价；创新必须落在这个接口如何改变可验证的交通决策，而非模块名字。
6. **先证明存在“可被决策改善的损失”，再投入新训练。** 若强在线规则已经接近可控的短程参考、候选动作代价差也很小，应该缩小主张，不能强行维护复杂方法。

## 1. 本轮检索到底完成了什么

| 项目 | 本轮范围与边界 |
|---|---|
| 广检 | 交通首轮三个查询，OpenAlex/arXiv/Crossref各75条原始命中；其中有明显无关条目，均未自动纳入 |
| 目标期刊补检 | T-ITS、T-VT、TR-C三个期刊，分别查询UAM、vertiport、air taxi、eVTOL、drone fleet scheduling、air-ground transportation；1个查询失败，保留失败记录 |
| 条目清单 | **83条筛选后文献记录**：57条题名直接涉及UAM/AAM/vertiport/air-taxi/eVTOL，26条为车辆或无人机协同等邻接交通研究 |
| 阅读层级 | 53条有公开索引摘要，2条另核出版页面摘要，20条仅书目；另8条来自同日既有ITSC逐条报告，保留历史阅读等级，不冒充本轮重新通读 |
| 目标渠道占比 | T-ITS 18条、TR-C 16条、T-VT 6条、ITSC 8条，共48条；其余35条是其他会刊或邻接背景，不冒充全部来自四大目标渠道 |
| 重点复核 | UAGMC期刊作者稿、VertiSync及Fair-CoPlan的相关方法/实验/结论章节；若出版社全文不可用，明确只据作者稿或摘要 |
| 年份 | 以2025–2026正式出版/可得日期为主；DOI含2024但2025卷期的条目保留双日期，不称“2025首次提出” |
| 引用量 | 有OpenAlex快照者记录检索日与来源；没有者留空，不把Crossref计数和OpenAlex计数混成排名 |

**83条是可核对的文献池，不是83篇全文精读。** 元数据条目只证明论文存在和主题范围，不能用来断言其具体算法或局限。所有条目、链接、作者、阅读层级见文末完整表与CSV。

这里统计的是**文献记录，不是83项独立研究证据**：例如UAGMC会议与期刊属于同一研究线，不能当两次独立复证。仅有题录的20条不参与方法优劣或缺口结论；重要判断主要落在已获取摘要和重点章节的近邻文献上。

这是有明确范围的研究扫描，不是穷尽式系统综述；检索语言、数据库收录和获取权限均有偏差。Semantic Scholar限流、DBLP部分失败，未绕过限制；使用可用的公开索引及官方论文页面补足。没有据“不曾搜到”推出“世界上无人做过”。

## 2. 四个目标渠道分别在解决什么

| 渠道 | 本轮可见的代表问题 | 对我们最有用的启发 | 不能据此推出什么 |
|---|---|---|---|
| ITSC | 多运营商公平冲突消解、闭排队网络机队规模、空域分区、运行验证 | 一个明确运营矛盾、可解释方法、强对照和边界，就能形成完整工作 | 不能从少量已录用论文预测录用率或推断审稿统一偏好 |
| T-ITS | 选站、吞吐量、联合重平衡、地空联运、服务能力与需求 | 需要分清有限时域ATT、长期吞吐稳定性和服务可用性 | 不能把吞吐最优直接称为我们的ATT最优 |
| T-VT | 大量UAM通信/轨迹/能效，也有车辆—无人机协同和调度 | 学习它如何把特定系统结构接到算法，而非照搬通信目标 | “写UAM”的通信论文不自动是乘客调度的SOTA |
| TR-C | 多模式动态任务分配、充电—电网耦合、滚动运营、优化与学习混合 | 真实需求、明确决策权限、求解质量与运行时间共同形成证据 | 不能认为模型越复杂、机制越多，就越接近期刊要求 |

本轮四类目标渠道的全部命中与未纳入条目也保存在原始检索记录中；没有为了各渠道数量好看而给不相关的通信文章分配同等权重。

## 3. 最接近我们的文献：必须先承认它们解决了什么

### 3.1 闭环与补能：不是我们的独占新问题

| 论文 | 已覆盖内容 | 对我们的约束与剩余可问问题 |
|---|---|---|
| [VertiSync，T-ITS，在线2025/卷期2026](https://arxiv.org/html/2503.17313) | 载客服务与空机重平衡联合排程，包含安全间隔与能源；在网络可逆/对称、机队足够等条件下分析吞吐 | 不能说前人没有闭环；它的长期吞吐保证不直接回答逐客选站的有限时域door-to-door ATT |
| [eVTOL-VSP，Transportmetrica A，2026](https://doi.org/10.1080/23249935.2026.2625143) | 航班分配、飞行器路径、充电、起降一体化；时空状态网络、MILP与ADMM分解 | “加入完整生命周期”不够新；应核对在线信息、乘客分流权限与计算预算是否相同 |
| [On-Demand UAM Scheduling with Operational Considerations，2025](https://doi.org/10.2514/1.I011460) | 电量、停放、补能、吞吐与异构机队；遗传算法结合滚动调度 | 优化并非只能离线；必须与可用的滚动求解方法比，不宜只击败简单规则 |
| [Smart charging，TR-C，2025](https://doi.org/10.1016/j.trc.2025.105281) | 由机队调度得到可充电时间窗，再考虑电网动态及不确定性 | 充电是已有成熟主题；我们的“未来供给”应首先尊重任务承诺与可行时间窗 |

**重要区别：** 飞机有生命周期，是物理模型；策略是否利用服务就绪信息，是决策问题。前者已经广泛研究，后者必须用同状态动作分支和相对强基线证明，而不是由前者自动推出。

### 3.2 地空联运与联合分配：不能把问题简化成两篇文章的拼接

[Li等，TR-C 2025](https://doi.org/10.1016/j.trc.2025.105111) 已把联运服务方案、资源检索、动态任务分配与车辆重平衡放在同一系统中。其具体任务和我们不同，但足以否定“地面＋空中＋重平衡没人做”的宽泛表述。<!--ref:T024--><!--anchor:section:Abstract-->

[An MILP-Based Framework for Multi-Modal Transportation with Air Taxis，2025](https://doi.org/10.2514/6.2025-3781) 将地空交通匹配与路径优化结合；[Utilizing eVTOL Aircraft to Alleviate Traffic Congestion，T-ITS 2025](https://doi.org/10.1109/TITS.2025.3553817) 研究站点、运营时段和转移客流的联动。它们的存在要求我们准确说明：哪些决策是给定的、哪些由策略控制，而非只强调“联合”。

**我们的可保留小切口：** 同一在线状态下，选站为若干分钟后的机场增加乘客，调机为另一个时刻增加运力；两者的收益不能只按当前队列相加。这个交互是否足够强、能否被小模型利用，仍需项目数据验证。

### 3.3 优化＋学习、因果、图网络：这些也已经有人做

| 最接近的反例 | 已有方向 | 因而不能作为我们的独立创新 |
|---|---|---|
| [Coordinated drone delivery via hybrid training，TR-C 2026](https://doi.org/10.1016/j.trc.2026.105857) | MIP专家轨迹→行为克隆→RL，处理异构配送单位 | “用解析解教RL”本身不是新意；是否能处理在线信息与逐客承诺才是区别 |
| [Causal-aware fleet repositioning，TR-C 2026](https://www.sciencedirect.com/science/article/pii/S0968090X26002445) | 供给影响需求实现，使用反事实需求预测辅助调机 | “因果＋重定位”不是新意；我们的固定外生需求不能被擅自改成内生需求 |
| [Path-pool Transformer，TR-C 2025](https://doi.org/10.1016/j.trc.2025.105165) | 为动态无人机配送组织候选路径与学习决策 | “Transformer＋候选集”不够；必须说明候选如何受已承诺服务日历约束 |
| [Learning Joint Rebalancing and Dynamic Pricing，T-ITS 2025](https://doi.org/10.1109/TITS.2025.3582639) | 图上的层级策略结合优化，研究重平衡与定价 | 联合控制不是普遍无人区；不能把别人的定价权限加入我们主实验 |

这些是**不同场景中的近邻或反例**，不是说它们已经解决我们的全部问题；也不能因任务不同就不比较其核心思想。

### 3.4 ITSC近年的小切口：值得学习证据组织

[Fair-CoPlan，ITSC2025](https://arxiv.org/html/2508.14380v1) 用容量许可—运营商提案—公平消解组织方法，比较的是明确的公平/延误权衡；[Optimal Fleet Sizing，ITSC2025官方摘要](https://its.papercept.net/conferences/scripts/abstract.pl?ConfID=91&Number=473) 使用闭Jackson网络研究服务可用性和利润。两者都不要求重新发明整个RL领域，却都不是“换个模块看ATT”。

对我们而言，一项有用贡献可以是：**识别一个反直觉的供需协调条件，给出可计算决策量，再证明它在不同强基座和需求回放中都有效。** 这是研究判断，不是会议录用保证。

### 3.5 作者承认的不足，与我们提出的研究边界，必须分开

| 来源 | 证据地位 | 能用于下一步的内容 | 不能作出的跳跃 |
|---|---|---|---|
| UAGMC期刊结论 | 作者明确未来工作 | 完整任务排程、落地冲突、现实空域和通信约束仍待进一步扩展 | 不能据此声称整个UAM领域都没有闭环运营 |
| [VertiSync结论](https://arxiv.org/html/2503.17313) | 作者明确计算改进方向 | 考虑神经近似与列生成等缩减求解成本 | 因而“用神经网络加速它”本身也不足以当未经研究的新题 |
| eVTOL-VSP、滚动UAM排程 | 摘要可见的建模/求解范围 | 用来核对我们的动作权限、资源约束和执行时限 | 未取得全文的地方，不虚构作者承认了某个不足 |
| 联运动态调度、因果调机 | 与本项目的条件差异 | 核对是否能在线利用同一已知承诺、是否有内生需求或额外定价权 | “目标不同”不自动等于我们有创新，也不允许强行改需求去适配 |

因此，本报告的结论是**找到了需要验证的接口问题，而不是已经认证了一个全球无人解决的空白**。尤其不建议从某篇文章的Future Work直接摘一个词就开大矩阵。

## 4. 五个候选缺口：哪些保留，哪些降级

以下均是限定检索范围内的候选，不是“现有文献都没有解决”的认证。

| 候选问题 | 与已有研究相比究竟缺什么 | 最大反对意见 | 决定是否继续的证据 | 优先级 |
|---|---|---|---|---|
| G1：已承诺供需的候选动作边际服务成本 | 不只预测平均等待；在同一背景下比较“这个乘客＋这次调机”对既有乘客的影响 | 这可能只是普通短程MPC/rollout换名字 | 必须超过同预算的解析推演，并证明学习项只修补解析项的系统误差 | **P0** |
| G2：满座批服务下的汇聚—分散临界条件 | 地面绕路、等凑批和空机循环之间何时该集中、何时该分散 | 批服务与资源池化理论很成熟；机场增加趋势可能是配置混杂 | 保持旧机场/资源可嵌入的对照，分离凑批等待和缺机等待，检验阈值能预测较优动作 | **P0诊断/P1论文** |
| G3：真正有效控制机会的自适应计算 | 容量不可调时不浪费联合搜索；候选接近时增加分支计算 | 动作mask、事件触发、计算分配都已有研究 | 相同计算时间下，减少无效推理且保留/改善ATT；与普通mask和固定深度搜索分开 | **P1** |
| G4：同一交通机制跨优化器/基座复用 | 不把修复限定于UAGMC，比较actor、Q-learning、解析策略的同一接口 | 通用性可能只来自更多输入或计算 | 至少两个不同基座有配对增益，且最强简单规则无法解释全部增量 | **作为验证要求** |
| G5：真实不确定性下保守的尾值校正 | 不预测已知事件，只处理尚未决定的未来需求/策略反应 | 鲁棒MPC、分布式RL、决策导向学习都已有成熟方法 | 校准的误差/排序改善必须对应闭环ATT改善，而非只降低预测MSE | **P1，有数据再上** |

**不建议继续当主贡献：** 首次闭环、首次Joint、首次GAT/ICM/WM、UAGMC在某个新增环境变差、加更多物理约束使问题更难。这些可以是设置、动机或消融，不是足够清晰的独立研究突破。

## 5. 最推荐的论文问题，不再寄生于UAGMC

暂定题眼：**面向已承诺服务的在线工作量—运力协同决策**。

通俗表述：乘客和飞机都还在路上，当前队列很短不代表乘客到站时服务快；我们比较的是“候选动作真正生效时，系统还要为谁等待”，并只对解析模型无法决定的后续影响学习修正。

### 5.1 方法分工

1. **解析层：** 传播当前已经确定的接驳、载客、充电、返航任务，保留明确的服务日历与资源守恒。
2. **候选层：** 对合法的乘客/飞机动作做相同预算的短程分支；比较总乘客时间，而非队列平衡分数。
3. **学习层：** 估计短程窗口之外、或未来尚未决定行为产生的尾部差异；不重新学习已能精确算出的飞行时间。
4. **保护层：** 当修正没有可靠证据时退回强基座；阈值须在验证集冻结，不能逐测试场景挑胜者。
5. **S/AC：** S组织时间与承诺背景，AC表示候选相对这个背景的边际影响；二者服务于同一代价比较，不是两个loss硬拼。

**这仍是候选方案。** 若它只是标准rollout/残差critic在小场景上的复现，应如实降级；真正可主张的新意需要来自承诺结构、批服务交互、误差界或可复用接口中的至少一个实质性结果。

### 5.2 一个可以诚实证明、但不能吹成全新定理的性质

对固定已知状态、同一后续策略，设候选代价为Q(a)，估计满足所有候选的误差不超过ε。选择估计代价最小的动作，真实候选损失至多比最好候选高2ε；若最好与次好差距大于2ε，排序不会选错。

这是标准近似贪心比较的直接推论，**不是我们的新理论**。它有用之处在于指导测量：关心候选代价差和误差，而非只关心全局预测MSE；若后续策略或预测模型改变，必须把误差另计，不能宣称整个闭环单调改善。

## 6. 我们到底需要打败谁

| 对照层 | 必须回答的问题 | 公平边界 |
|---|---|---|
| 作者SPF/STTF/QTTI | 与源研究有什么关系 | 忠实原版与扩站规则分别标注，不能把扩站版称原文结果 |
| 项目最强在线解析规则 | 学习是否真的带来交通收益 | 同需求、同动作权限、同已知信息；规则也是算法，不叫理论下界 |
| 同预算短程rollout/MPC | 新方法是否只多用了仿真计算 | 对齐每次决策时间、分支数和可看未来范围 |
| UAGMC-L与UAGMC-A | 是不是只击败了较弱的LSTM变体 | 不能将近期LSTM实现当成期刊Attention主模型 |
| 普通离散actor-critic/Q-learning | 收益是否依赖特定优化器 | 新算法单列合同，先统一评估，不偷调每组物理 |
| 现代结构化RL/相关学习—优化混合 | 是否超越现有方法范式 | 简化或移植时命名为adapted/inspired，不伪称完整原版 |
| 知道真未来的离线参考 | 问题还有多大潜在空间 | 仅作特权参考；有限预算求到的可行解不是已证最优上界/下界 |

不存在一篇跨权限、跨目标都可直接称为唯一SOTA的论文。对我们的投稿，**最强同问题在线非学习算法＋代表性现代学习算法**比单一UAGMC更重要。

## 7. 利用现有E0/E1先做的五项判别

1. **按强规则重新画差距。** 每个模型先对共同回放汇总，再跨训练seed比较；分开Best可达性、后期平台与独立测试表现。
2. **检查K的比较是否真是“只多一个机场”。** 原机场坐标、需求、总机队、初始飞机分布、容量与动作可忽略性均须逐项核对；否则不能把非单调归因给机场数。
3. **等待互斥分解。** 仅缺乘客批次、仅缺就绪飞机、两者都缺、两者皆有却未服务；不要重复加总重叠等待。
4. **同状态分支。** 用合法可得信息，比较候选动作对未来累计未完成人时的差异；先验证有没有可改善动作，而非先预设S应有效。
5. **网络更新影响。** 固定状态比较PPO前后、aux前后动作概率与代价加权错误；相关梯度不等于已证因果，需配对关闭相应更新复核。

没有原始归档的新数据，本报告不重算或新增“已发现规律”。既有240M/198M结果尚未在本轮取得，不能提前判定哪个修复成功。

## 8. “机场越多反而越差”能否成为独立小切口

有潜力，但目前不能直接成立。若新增机场真的只是扩展可选动作，旧策略在新环境中仍能原样运行，且其他物理和需求完全不变，那么最优策略至少可以不使用新机场，最优值不会因为多一个选项而变差。

因此，观测到变差至少有三种竞争解释：**环境同时改变了；策略没有学会忽略额外选择；分流产生了新的组批/供给碎片化但现有策略未协调好。** 这不是“增加机场必然物理有害”的证明。

可检验设计：先验证旧策略嵌入（新站动作永不选）能否重现旧轨迹；再看强规则、普通RL和协调方法增加站点后各自的损失；最后把损失定位到满座批服务还是机队周转。不要为了得到趋势移动机场或调需求。

## 9. 优先级与止损条件

| 阶段 | 先做什么 | 继续的最低理由 | 停止或转向的情形 |
|---|---|---|---|
| 无训练诊断 | 同状态候选分支、等待分解、K配置核对 | 有稳定出现的高代价误选状态，且合法信息能改善排序 | 差异只在极少数尾批或指标删失中出现 |
| 小规模方法检验 | 解析推演、学习尾值、两者组合，保留普通residual | 组合在独立需求回放中有增量，增量不能由额外计算解释 | 简单规则/同预算搜索完全达到效果 |
| 机制验证 | 去时间对齐、去候选差分、去基座保护，做小因子交互 | 去掉对应结构产生预测中的特定错误 | 只有全模型偶然低Best，无法解释组成作用 |
| 投稿确认 | 两个强基座、独立训练seed、合法测试需求 | 稳健相对增益及计算—性能曲线，适用边界可复核 | 只对UAGMC-L一次seed有效 |

不建议现在再开启几十种热点模型的大矩阵；先用这套漏斗决定“是否值得训练”，再决定预算。

## 10. 反方审查：我们不能强行解释什么

- **闭环并不新。** 若文章主要还是E0→E1退化，最强反对意见就是研究对象过窄且贡献寄生于基线。
- **理论不能装饰。** 旧排队论、2ε界或通用MARL定理不能改名成我们首次提出；需要解释它对动作选择有何可计算影响。
- **新AI不保证有效。** 小离散动作不天然需要扩散采样，精确仿真不天然需要学习全世界模型。
- **无显著规律也有意义，但不足以自动形成方法论文。** 可复现的配置/评价修正应先做好，再判断是否有独立交通结论。
- **局限来自哪一方要分清。** 本文提出的“未联合优化某个变量”若原文没明确说，只是我们依据所见模型的迁移边界，不冒充作者未来工作。

### 关键文献之间的条件差异

| 文献对 | 表面冲突 | 本报告的处理 |
|---|---|---|
| UAGMC vs VertiSync | 一个选站，一个做网络排程，谁更强？ | 目标/权限/资源模型不同，不可跨论文数字排名；共同环境适配后才比较 |
| UAGMC vs 联运动态调度 | 地空协调是不是尚未研究？ | 广义联运已有；UAGMC聚焦在线选站，不能把它的自述推广成整个领域空白 |
| 闭环MILP vs 学习式在线控制 | 有优化解是不是没必要RL？ | 看在线信息与求解时限；小问题可能确实无需RL，这个反例必须允许 |
| 因果调机 vs 我们AC | 都叫“动作影响”是否相同？ | 内生需求因果识别与固定需求下候选成本差分不同，不混用理论保证 |
| 满座效率 vs 乘客等待 | 集中分流和均匀分流哪种更好？ | 由组批、访问距离、就绪供给共同决定，不先给平衡奖励来预设答案 |

以上为有范围的候选对比较，不是对全语料所有论文两两查尽。研究者仍需确认这些区分是否足以支撑最终论文。

## 11. 最终建议

**不要放弃S/AC，但停止把“换一套表示修UAGMC”当完整故事。** 让S/AC成为一个面向服务承诺的决策接口，在强解析/优化和至少两个学习基座上证明增量；用“什么时候值得协调、为什么改善、要花多少计算”构成文章。

若只能做一个方向，优先G1；若无训练分支发现G1没有明显空间，优先研究G2的运营规律，而不是继续抽卡找网络。

## 附录A：完整83条文献清单

下面的完整表由保留的元数据和逐条中文归纳生成；M=题录，A-index=公开索引摘要，A-publisher=另核出版摘要，A-primary=官方摘要。重点章节核读另见正文；历史F/A/M不升级成本轮全文阅读。表内“可迁移点”是我们的初筛建议，不是原作者承认的不足。

| ID | 年份／渠道 | 文献与作者 | 中文逐条归纳／适用边界 | 研究用途与优先级 | 阅读 |
|---|---|---|---|---|---|
| T001 | 2026／IEEE Transactions on Intelligent Transportation Systems | [Integrated Network Design and Demand Estimation for Advanced Air Mobility Using an Extended Multiple-Allocation p-Hub Median Framework](https://doi.org/10.1109/tits.2026.3697571)<br>Mahdi Monemi Nodehi; Yantong Dou; Joseph Post; Yu Zhang | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T002 | 2026／IEEE Transactions on Intelligent Transportation Systems | [Trajectory Optimization of eVTOL Vehicles for Urban Air Mobility Using Uniform Trigonometrization Method](https://doi.org/10.1109/tits.2026.3670893)<br>Kshitij Mall; Rishikesh Gadre; Yufei Wu; Zhenbo Wang; Daniel A. DeLaurentis | 比较eVTOL巡航、下降和着陆轨迹的数值优化；可提示飞行时间的物理边界，但不是乘客选站或机队调机方法。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T003 | 2025／IEEE Transactions on Intelligent Transportation Systems | [The State of Urban Air Mobility Research: An Assessment of Challenges and Opportunities](https://doi.org/10.1109/tits.2024.3511453)<br>Sheikh Shahriar Ahmed; Grigorios Fountas; Virginie Lurkin; Panagiotis Ch. Anastasopoulos; Yu Zhang; Michel Bierlaire; Fred Mannering | 综述需求、公众接受、基础设施和运行管理；可界定领域已覆盖的问题，不能把综述中的每项机会当成尚未解决的具体命题。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T004 | 2026／IEEE Transactions on Intelligent Transportation Systems | [Demand Modeling for Advanced Air Mobility: Challenges, Opportunities, and Future Directions](https://doi.org/10.1109/tits.2026.3671002)<br>Kamal Acharya; Waleed Raza; Katherine Vasiloff; Zhenbo Wang; Liang Sun; Houbing Herbert Song | 整理AAM需求建模的计量、离散选择、仿真与学习路线；提醒我们固定需求trace只覆盖运营决策，不代表完整市场需求。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T005 | 2026／IEEE Transactions on Intelligent Transportation Systems | [Throughput Maximizing Takeoff Scheduling for eVTOL Vehicles in On-Demand Urban Air Mobility Systems](https://doi.org/10.1109/tits.2025.3640561)<br>Milad Pooladsanj; Ketan Savla; Petros A. Ioannou | VertiSync联合载客与重平衡，研究安全和能源约束下吞吐临界；其保证依赖网络和机队条件，不能直接当有限时域ATT最优。 | P0：最接近运营/方法边界；先核对决策权限、在线信息和运行时间。 | A-index |
| T006 | 2026／IEEE Transactions on Intelligent Transportation Systems | [Heterogeneous Vertiport Selection Optimization for On-Demand Air Taxi Services: A Deep Reinforcement Learning Approach](https://doi.org/10.1109/tits.2026.3680351)<br>Aoyu Pang; Maonan Wang; Zifan Sha; Wenwei Yue; Changle Li; Chung Shue Chen; Man-On Pun | 把多源交通信息与时间历史编码用于逐客出发站选择；其Attention主模型、源LSTM和我们闭环移植必须分别比较。 | P0：最接近运营/方法边界；先核对决策权限、在线信息和运行时间。 | A-index |
| T007 | 2025／IEEE Transactions on Intelligent Transportation Systems | [Real-Time Avoidance of Obstacles and Emergent Geo-Fences for Urban Air Mobility Using Deep Reinforcement Learning](https://doi.org/10.1109/tits.2025.3589322)<br>Mingcheng Zhang; Bizhao Pang; Chao Yan; Mir Feroskhan; Chen Lv | 利用带记忆的深度RL处理障碍和突发禁飞区；属于航迹避障，不能直接证明乘客—飞机协调表示有效。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T008 | 2026／IEEE Transactions on Intelligent Transportation Systems | [Cable or Locker: Coordination of the Urban Dual-Mode Drone Delivery Service](https://doi.org/10.1109/tits.2026.3656094)<br>Zhi Pei; Yu Zhang; Shanshan Lv; Yangchen Mao | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T009 | 2026／IEEE Transactions on Intelligent Transportation Systems | [A Learning-Enhanced Method for the Multi-Visit Vehicle-Drone Routing Problem With Continuous Locations](https://doi.org/10.1109/tits.2026.3716617)<br>Zongcheng Zhang; Yanru Chen; Shuang Liang; Mujin Gao; Bo Du; Mohamed Ismail Mohamed Wahab | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T010 | 2025／IEEE Transactions on Intelligent Transportation Systems | [Multi-Objective Multi-Drone Collaborative Routing Problem With Heterogeneous Delivery and Pickup Service](https://doi.org/10.1109/tits.2025.3548426)<br>Fangyu Hong; Guohua Wu; Yalin Wang; Qizhang Luo; Ling Wang; Jianmai Shi | 联合多无人机与智能柜，建模取送和即时配送的多目标路径；可借鉴异构任务约束组织，不迁移其目标权重。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T011 | 2026／IEEE Transactions on Intelligent Transportation Systems | [Relational Trajectory-Entropy Augmented MADRL for Truck-Drone Collaborative Deliver With Mobile Charging Optimization](https://doi.org/10.1109/tits.2025.3607709)<br>Huixian Qiu; Zheng Luo; Yijing Huang; Jianqiang Li; Chengwen Luo | 以关系图和轨迹熵辅助多智能体卡车—无人机移动充电协作；其额外探索奖励不能悄悄加入我们的固定ATT合同。 | P1：对照承诺、资源或调度语义；目标不同不能直接搬成绩。 | A-index |
| T012 | 2026／IEEE Transactions on Intelligent Transportation Systems | [Urban-Rural Bus-Assisted Drone Logistics Delivery With Time Window Considerations](https://doi.org/10.1109/tits.2026.3705834)<br>Yu Jiang; Chen Guo; Ye Yuan; Mengmeng Liu; Qingwen Xue | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T013 | 2026／IEEE Transactions on Intelligent Transportation Systems | [Reinforcement Learning and Exact Optimization for Traffic-Weighted Hybrid Truck-Drone Last-Mile Delivery With Deadlines](https://doi.org/10.1109/tits.2026.3677411)<br>Abdullahi Sani Shuaibu; Ashraf Sharif Mahmoud; Tarek Rahil Sheltami; Anis Elgabli | 将RL与小规模精确MILP并列处理带期限的卡车—无人机配送；值得借鉴可解小实例的最优参照与规模边界。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T014 | 2025／IEEE Transactions on Intelligent Transportation Systems | [Joint Optimization of Electric Bus Infrastructure Planning, Fleet Composition, and Charging Schedule Considering Multi-Gun Charging and Compatibility](https://doi.org/10.1109/tits.2025.3526844)<br>Mingye Zhang; Min Yang; Renjie Zhang; Rui Peng | 联合电动公交机队、充电基础设施和多枪兼容调度；说明供给就绪时间需尊重资源可行性，但当前不用因此新增充电机制。 | P1：对照承诺、资源或调度语义；目标不同不能直接搬成绩。 | A-index |
| T015 | 2026／IEEE Transactions on Intelligent Transportation Systems | [Optimization of Flight Paths for eVTOL Aircraft in Complex Urban Airspace](https://doi.org/10.1109/tits.2026.3686959)<br>Hang Zhou; Feng-Yang Zhao; Yun-Kai Kou; Xiao-Bing Hu | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T016 | 2025／IEEE Transactions on Intelligent Transportation Systems | [Utilizing eVTOL Aircraft to Alleviate Traffic Congestion on an Arterial Road](https://doi.org/10.1109/tits.2025.3553817)<br>Bang-Kai Xiong; Rui Jiang; Kai Wang; Xinmin Tang; Ying Shang; Xiaobo Qu | 联合站点、运营时段和转移比例缓解道路瓶颈；其货币化综合成本与我们的固定拓扑ATT不同。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T017 | 2026／IEEE Transactions on Intelligent Transportation Systems | [Analyzing the Impact of Service Stages on Demand and Modal Shift for Urban Air Mobility: A Case Study of Seoul Metropolitan Area](https://doi.org/10.1109/tits.2026.3698394)<br>Hyunsoo Kim; Kwanjung Yee | 按UAM发展阶段分析交通方式转移和分航线需求；可用来说明结论的需求适用范围，不用于训练中调需求造趋势。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T018 | 2025／IEEE Transactions on Intelligent Transportation Systems | [Learning Joint Rebalancing and Dynamic Pricing Policies for Autonomous Mobility-on-Demand](https://doi.org/10.1109/tits.2025.3582639)<br>Xinling Li; Carolin Schmidt; Daniele Gammelli; Filipe Rodrigues | 图上的层级策略结合优化，联合自动出行系统重平衡与定价；提示联合控制已有成熟近邻，价格权不是我们当前动作。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T019 | 2025／Transportation Research Part C: Emerging Technologies | [Integrating urban air mobility into the power grid through smart charging solutions](https://doi.org/10.1016/j.trc.2025.105281)<br>Jiaman Wu; Shangqing Cao; Mark Hansen; Marta C. González | 由机队调度得到充电时间窗，再与电网约束和不确定性协调；可借鉴先承诺服务、再处理可行资源窗口。 | P1：对照承诺、资源或调度语义；目标不同不能直接搬成绩。 | A-index |
| T020 | 2026／Transportation Research Part C: Emerging Technologies | [Resilient multi-agent reinforcement learning for centralised tactical conflict resolution under uncertain perturbations and non-cooperative traffic in urban air mobility](https://doi.org/10.1016/j.trc.2026.105542)<br>Rodolphe Fremond; Yan Xu; Junjie Zhao; Antonios Tsourdos; Gokhan Inalhan | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T021 | 2026／Transportation Research Part C: Emerging Technologies | [Integrated take-off management and trajectory optimization for merging control in urban air mobility corridors](https://doi.org/10.1016/j.trc.2025.105370)<br>Yingqi Liu; Tianlu Pan; Jingjun Tan; Renxin Zhong; Can Chen | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T022 | 2026／Transportation Research Part C: Emerging Technologies | [U-Aerodrome: Data-driven and risk-bounded airspace reconfiguration for safe integration of urban air mobility at aerodrome](https://doi.org/10.1016/j.trc.2025.105506)<br>Xinting Hu; Bizhao Pang; Sameer Alam; Mir Feroskhan | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T023 | 2026／Transportation Research Part C: Emerging Technologies | [Real-time traffic simulation and management for large-scale urban air mobility: Integrating route guidance and collision avoidance](https://doi.org/10.1016/j.trc.2025.105477)<br>Canqiang Weng; Can Chen; Jingjun Tan; Tianlu Pan; Renxin Zhong | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T024 | 2025／Transportation Research Part C: Emerging Technologies | [On-demand ridesharing based on dynamic scheduling in urban air mobility](https://doi.org/10.1016/j.trc.2025.105111)<br>Shanghan Li; Tengfei Zhang; Yiyong Xiao; Daqing Li | 联运系统统一服务方案、动态任务分配与重平衡；它直接限制了我们对首次地空联合调度的宽泛主张。 | P0：最接近运营/方法边界；先核对决策权限、在线信息和运行时间。 | A-index |
| T025 | 2026／Transportation Research Part C: Emerging Technologies | [A branch-and-cut algorithm for routing a heterogeneous drone-ground vehicle fleet to deliver time-sensitive products](https://doi.org/10.1016/j.trc.2025.105454)<br>Sayed Hamid Hosseini Dolatabadi; Tanveer Hossain Bhuiyan; Waquar Kaleem; Anirudh Subramanyam; Taner Cokyasar | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T026 | 2026／Transportation Research Part C: Emerging Technologies | [Metro-drone collaborative delivery: Integrated hub activation and multi-trip scheduling under time windows](https://doi.org/10.1016/j.trc.2026.105831)<br>Shuo Sun; Laijun Zhao; Changzhi Wu; Huiyong Li; Yifei Wang | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T027 | 2026／Transportation Research Part C: Emerging Technologies | [Coordinated drone delivery via hybrid training: From optimization to reinforcement learning](https://doi.org/10.1016/j.trc.2026.105857)<br>Lizhi Bi; Bin Shuai; Zhanru Liu; Yichen Sun | 出版摘要讨论MIP专家、行为克隆与RL串联的异构配送；学习优化专家不是独立新意，在线信息差异才需进一步核验。 | P0：最接近运营/方法边界；先核对决策权限、在线信息和运行时间。 | A-publisher |
| T028 | 2025／Transportation Research Part C: Emerging Technologies | [The Heterogeneous-Fleet Electric Vehicle Routing Problem with Nonlinear Charging Functions](https://doi.org/10.1016/j.trc.2024.104932)<br>Weiquan Wang; Yossiri Adulyasak; Jean-François Cordeau; Guannan He | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T029 | 2026／Transportation Research Part C: Emerging Technologies | [Causal-aware deep reinforcement learning framework to predictively optimize for-hire vehicle fleet repositioning](https://doi.org/10.1016/j.trc.2026.105756)<br>Rong Cao; David Z.W. Wang; Jintao Ke; Yiming Yan | 出版摘要以供给影响需求的内生反馈和反事实预测辅助调机；我们的固定外生请求不直接满足同一因果设定。 | P1边界：内生需求反事实与本项目固定外生请求不同；先核对因果假设。 | A-publisher |
| T030 | 2026／Transportation Research Part C: Emerging Technologies | [Slope-dependent energy optimization for cooperative truck-drone delivery and pickup routing with time windows](https://doi.org/10.1016/j.trc.2026.105594)<br>Ying Wang; Taofeng Ye; Jiajun Sun; Jicong Duan | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T031 | 2025／Transportation Research Part C: Emerging Technologies | [Integrating battery-related decisions into truck-drone tandem delivery problem with limited battery resources](https://doi.org/10.1016/j.trc.2025.105082)<br>Zhongshan Liu; Bin Yu; Tingting Chen; Li Zhang | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T032 | 2026／Transportation Research Part C: Emerging Technologies | [Dynamic switching models for truck-only delivery and drone-assisted truck delivery under demand uncertainty](https://doi.org/10.1016/j.trc.2026.105807)<br>Jiaqing Lu; Qianwen (Vivian) Guo; Dian Sheng; Shumin Chen; Paul Schonfeld | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T033 | 2025／Transportation Research Part C: Emerging Technologies | [Path pool based transformer model in reinforcement framework for dynamic urban drone delivery problem](https://doi.org/10.1016/j.trc.2025.105165)<br>Chuankai Xiang; Yanfang Mo; Wei Liu; Zhibin Wu; Lishuai Li | 用候选路径池和Transformer支持动态无人机任务分配与路径决策；应学习候选结构，而非仅借Transformer名称。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T034 | 2026／Transportation Research Part C: Emerging Technologies | [Integrated optimization of ride-pooling and shared micro-mobility services with meeting points](https://doi.org/10.1016/j.trc.2025.105452)<br>Tingting Dong; Yulong Hu; Zemin Wang; Sen Li | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T035 | 2026／IEEE Transactions on Vehicular Technology | [Air-Ground Integrated UAV-Assisted VANET for Last-Mile Delivery](https://doi.org/10.1109/tvt.2026.3739703)<br>Muhammad Umar Farooq Qaisar; Lin Zhang; Paolo Bellavista; Shehzad Ashraf Chaudhry; Chang Liu; Guangjie Han | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T036 | 2026／IEEE Transactions on Vehicular Technology | [Joint Multi-Vehicle Routing and Multi-Drone Scheduling for a Hybrid Drone-Assisted Last-Mile Delivery System in Smart Cities](https://doi.org/10.1109/tvt.2026.3692486)<br>Yuanhang Qi; Xumin Huang; Gewen Huang; Junhao Wu; Shengli Xie | 联合普通车辆、携机车辆路径及多无人机发射调度；是异构控制交互的邻接证据，不是相同客运ATT实验。 | P1：对照承诺、资源或调度语义；目标不同不能直接搬成绩。 | A-index |
| T037 | 2026／IEEE Transactions on Vehicular Technology | [Federated Edge Learning Based Multi-Drone Optimal Delivery Routing Approach](https://doi.org/10.1109/tvt.2026.3715448)<br>Jie Zhang; Mengying Zhang; Junhua Zhou; Shiyao Zhang | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T038 | 2026／IEEE Transactions on Vehicular Technology | [Energy-Efficient Optimization for Integrated Sensing and Communication Assisted Cell Free Urban Air Mobility](https://doi.org/10.1109/tvt.2025.3626568)<br>Chunyu Pan; Zhonghao Luo; Kai Wang; Daquan Feng; Xinwei Yue | 优化空地通信与感知的能效和精度；与交通资源调度信息需求有关，但不是乘客服务SOTA。 | P2邻接：通信/感知或轨迹指标不同；不作为ATT调度SOTA。 | A-index |
| T039 | 2025／IEEE Transactions on Vehicular Technology | [Cooperative Urban Air Mobility Trajectory Design for Power and AoI Optimization: A Multi-Agent Reinforcement Learning Approach](https://doi.org/10.1109/tvt.2025.3561155)<br>Hyeonsu Kim; Pyae Sone Aung; Md. Shirajum Munir; Walid Saad; Choong Seon Hong | 多智能体同时考虑航迹、功耗与信息年龄；AoI衡量状态新鲜度，不等于乘客等待或动作生效延迟。 | P2邻接：通信/感知或轨迹指标不同；不作为ATT调度SOTA。 | A-index |
| T040 | 2026／IEEE Transactions on Vehicular Technology | [Graph-Transferable Rebalancing Policy for City-wide Mobility-on-Demand Systems based on Physics-Informed Neural Networks](https://doi.org/10.1109/tvt.2026.3737260)<br>Hanwen Zhang; Jicheng Chen; Hui Zhang | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T041 | 2025／Journal of Aerospace Information Systems | [On-Demand Urban Air Mobility Scheduling with Operational Considerations](https://doi.org/10.2514/1.i011460)<br>Jaeyoul Ko; Jaemyung Ahn | 电量、停放、充电和吞吐共同进入滚动调度；摘要报告其测试实例可在分钟级求解，不能笼统说优化都无法在线。 | P0：最接近运营/方法边界；先核对决策权限、在线信息和运行时间。 | A-index |
| T042 | 2025／Transportation Research Part E Logistics and Transportation Review | [Urban air mobility network design and operations strategy in an urban agglomeration](https://doi.org/10.1016/j.tre.2025.104316)<br>Ziyu Wang; Di Lv; Shuai Jia; Kai Wang; Xiaobo Qu | 未取得摘要；只按题名列作书目候选，不推断方法或作者局限。 | M待核：仅按题名定位；尚无方法证据，不推断作者局限。 | M |
| T043 | 2026／Scientific Reports | [Intelligent scheduling and resource allocation for urban air mobility networks based on graph neural networks](https://doi.org/10.1038/s41598-026-48143-9)<br>Xiaoqi Xu; Y Q Zhao; Peilong Zhang | 异构图预测结合确定性约束修复，处理UAM资源与排程；是学习输出经物理可行层执行的直接近邻。 | P1：对照承诺、资源或调度语义；目标不同不能直接搬成绩。 | A-index |
| T044 | 2025／Scientific Reports | [Routing and scheduling optimization for urban air mobility fleet management using quantum annealing](https://doi.org/10.1038/s41598-025-86843-w)<br>Renichiro Haba; Takuya Mano; Ryosuke Ueda; Genichiro Ebe; Kohei Takeda; Masayoshi Terabe; Masayuki Ohzeki | 把大机队无冲突路径选择写成图上的组合优化，并考察量子退火；重点是问题结构，不是要求本项目使用量子计算。 | P1：对照承诺、资源或调度语义；目标不同不能直接搬成绩。 | A-index |
| T045 | 2025／AIAA SciTech Forum | [Separation Assurance in Urban Air Mobility Systems Using Shared Scheduling Protocols](https://doi.org/10.2514/6.2025-2116)<br>Surya Murthy; Tyler Ingebrand; Sophia Smith; Ufuk Topcu; Peng Wei; Natasha A. Neogi | 借共享排程协议协调走廊交汇处的速度和时间以保间隔；其通信协议类比不自动适用于载客组批。 | P1：对照承诺、资源或调度语义；目标不同不能直接搬成绩。 | A-index |
| T046 | 2025／Conference proceedings/Conference proceedings - IEEE International Conference on Systems, Man, and Cybernetics | [Conflict-Free Flight Scheduling Using Strategic Demand Capacity Balancing for Urban Air Mobility Operations](https://doi.org/10.1109/smc58881.2025.11343076)<br>Vahid Hemmati; Yonas Ayalew; Ahmad Reza Mohammadi; Reza Ahmari; Parham Mohsenzadeh Kebria; Abdollah Homaifar; Mehrdad Saif | 从两机延迟出发避冲突扩展到多机出发时刻优化；可借鉴由局部交互构建全局排程的证据组织。 | P1：对照承诺、资源或调度语义；目标不同不能直接搬成绩。 | A-index |
| T047 | 2025／AIAA AVIATION Forum | [Rule-Based Departure Sequencing in a Multi-Vertiport Network to Maximize Throughput](https://doi.org/10.2514/6.2025-3308)<br>Anubhav Halder; Farhan Gandhi | 细化站坪预约、充电和旅客处理后比较规则式出发排序；提示强规则与高保真仿真本身就可能产生显著收益。 | P1：对照承诺、资源或调度语义；目标不同不能直接搬成绩。 | A-index |
| T048 | 2025／Sustainability | [Efficient Urban Air Mobility Vertiport Operational Plans Considering On-Ground Traffic Environment](https://doi.org/10.3390/su17115054)<br>Jaekyun Lee; Uwon Huh; Peng Wei; Kyowon Song | 把起降场运营与地面交通仿真连接，考察到达和周转不确定性；适合定义系统边界，而非当前必须加料。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T049 | 2026／Journal of Transport Geography | [An optimization framework for the design and operation of efficient urban air mobility systems: An application in the Île-de-France region](https://doi.org/10.1016/j.jtrangeo.2026.104611)<br>Sam Randeraad; Marta Ribeiro; Jan Anne Annema; Gonçalo Homem de Almeida Correia | 联合站点可行性、需求—供给反馈与运营设计；宏观设计目标与我们固定站点的在线选站不同。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T050 | 2025／AIAA SciTech Forum | [Energy-Aware Strategic Traffic Management for Urban Air Mobility](https://doi.org/10.2514/6.2025-0580)<br>Abenezer G. Taye; Shulu Chen; Peng Wei | 离线路径能耗与在线容量/需求进入MILP，选择地面延迟与速度；可借鉴已知物理计算与在线决策分工。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T051 | 2025／AIAA SciTech Forum | [Multi-Agent Integrated Path Planning and Scheduling in Advanced Air Mobility](https://doi.org/10.2514/6.2025-1352)<br>Dhananjay Tiwari; Salar Basiri; Srinivasa M. Salapaka | 以设施位置—路径模型组织多机联合路径和时序，并加入避冲突；不能把其运输成本当door-to-door ATT。 | P1：对照承诺、资源或调度语义；目标不同不能直接搬成绩。 | A-index |
| T052 | 2026／IET Intelligent Transport Systems | [GRID‐Routing: A Proposal for Massive Traffic Management in Urban Air Mobility](https://doi.org/10.1049/itr2.70336)<br>Víctor Blázquez Milara; Iker Nebot Amoriza; Aurelio Bermúdez; Rafael Casado; Enrique Hernández Orallo; Carlos T. Calafate | 用规则空域网格与明确飞行计划管理大规模交通；主要解决空域组织，不直接解决有限飞机何时服务乘客。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T053 | 2025／Transportation Research Interdisciplinary Perspectives | [A clustering-metaheuristic-simulation approach to determine air taxi operating site location](https://doi.org/10.1016/j.trip.2025.101330)<br>Varshini Priyaa Senthilnathan; Mohanapriya Singaravelu; Suchithra Rajendran; Sharan Srinivas | 聚类产生候选站点，遗传算法与仿真共同筛站；提醒站点变化本身就是实验变因，不应混入方法优劣比较。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T054 | 2026／Transportmetrica A Transport Science | [Optimising eVTOL vehicle scheduling problem in on-demand urban air mobility system with capacitated vertiports](https://doi.org/10.1080/23249935.2026.2625143)<br>Dongbo Guo; Enjian Yao; Rongseng Chen; Tianyu Zhang; Yongxiang Zhang | 将航班分配、机队路径、充电和起降整合为时空状态MILP并分解求解；是完整运营已有研究的直接证据。 | P0：最接近运营/方法边界；先核对决策权限、在线信息和运行时间。 | A-index |
| T055 | 2025／Eng—Advances in Engineering | [Vertiports: The Infrastructure Backbone of Advanced Air Mobility—A Review](https://doi.org/10.3390/eng6050093)<br>Paola Di Mascio; Giulia Del Serrone; Laura Moretti | 综述起降场设计、布局和联运接口；提供物理模型边界，不足以证明某个表示模块的新颖性。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T056 | 2025／AIAA SciTech Forum | [How to Integrate Vertiport Operations Into the Airspace? Introducing the EUREKA Project](https://doi.org/10.2514/6.2025-2370)<br>Bianca Isabella Schuchardt; Thu Hang Vu; Alberto Mangas; César Nava; Giovanni Riccardi; Marie Kolago; Abul Hassan Abdullah; Paolo Cappello; Marsel Omeri | EUREKA把到离场、协同管理、异常和网络容量连起来；系统级验证重要，但不应因此一次扩张全部环境。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T057 | 2025／Drones | [UAV Operations and Vertiport Capacity Evaluation with a Mixed-Reality Digital Twin for Future Urban Air Mobility Viability](https://doi.org/10.3390/drones9090621)<br>Junjie Zhao; Zhang Zheng Wen; Krishnakanth Mohanta; Stefan Subasu; Rodolphe Fremond; Yu Su; Ruechuda Kallaka; Antonios Tsourdos | 以数字孪生和实体无人机代理检验起降场运行与异常；现实性证据不同于仅在仿真中增加参数数量。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T058 | 2025／IEEE/AIAA Digital Avionics Systems Conference (DASC) | [Geometric Vertiport Localization: UAM Procedural Envelopes in a Cityscape](https://doi.org/10.1109/dasc66011.2025.11257220)<br>Chi Kien Huynh; Billy Josefsson; Anastasia Lemetti; Valentin Polishchuk; Leonid Sedov | 依据城市几何和程序包络筛选合法起降场位置；这里的定位是设施选址，不是无线定位，也不是ATT调度算法。 | P2背景：这里是设施几何选址；不误归为无线定位，也不作ATT调度SOTA。 | A-index |
| T059 | 2026／AIAA SciTech Forum | [Systematic Multi-Criteria Approach to Determine Suitable Vertiport Locations for Advanced Air Mobility](https://doi.org/10.2514/6.2026-2276)<br>Melissa Lepe; Jacqueline L. Huynh | 多准则筛选起降场并考虑社会环境约束；站点质量与运营策略不能混作同一个性能因素。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T060 | 2025／AIAA SciTech Forum | [Data-Driven Approach Toward Vertiport Placement for Urban Air Mobility Operations: Case Studies in South Korea](https://doi.org/10.2514/6.2025-2784)<br>Shinkook Cha; J.S. Lee; Chungho Park; Yujin Kim; Young-Jae Lee; Junghyun A. Kim | 用数据聚类选择起降场，再与既有交通作情景比较；有助于外部场景验证，不用于事后调整主benchmark。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T061 | 2026／Transportation Research Record Journal of the Transportation Research Board | [Simulation-Based Capacity Analysis of Cross-Dock eVTOL Vertiports for Cargo Operations](https://doi.org/10.1177/03611981261461483)<br>Matthew Perrett; Nathan N. Huynh | 仿真比较货运交叉转运站固定/共用门的容量；说明共享资源布局会改变瓶颈，和学习网络收益需分离。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T062 | 2025／AIAA AVIATION Forum | [AirTaxiSim: A Simulator for Autonomous Air Taxis](https://doi.org/10.2514/6.2025-3349)<br>Ayoosh Bansal; Mikael Yeghiazaryan; Hyung Jin Yoon; Duo Wang; Ashik E Rasul; Chuyuan Tao; Yang Zhao; Tianyi Zhu; Oswin So; Chuchu Fan; Petros G. Voulgaris; Naira Hovakimyan; Lui Sha | 端到端空中出租车仿真强调自主控制和复杂城市环境；主要是模拟平台，不是我们的运营优化冠军基线。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T063 | 2025／CEAS Aeronautical Journal | [A method to determine the availability and performance of fixed-wing air taxis in real weather conditions for regional air mobility within Germany](https://doi.org/10.1007/s13272-025-00807-4)<br>Lutz Wortmann; Stefan Hagemann; Eike Stumpf | 引入真实气象和航迹评估区域固定翼空中出租车可用性；用于限定适用边界，不要求当前修改为天气任务。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T064 | 2025／OR Spectrum | [Planning an airport shuttle network with air taxis using choice-based optimization](https://doi.org/10.1007/s00291-024-00801-y)<br>Thomas Hagspihl; Rainer Kolisch; S. Schiffels | 把旅客交通方式选择放进机场接驳枢纽选址；旅客行为和服务供给有联动，但决策时标与我们不同。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T065 | 2025／Integrated Communications, Navigation and Surveillance Conference (ICNS) | [Development and Evaluation via Fast-Time Simulation of an Air Taxi Shuttle Service for the City of Frankfurt Connecting Airport and Trade Fair](https://doi.org/10.1109/icns65417.2025.10976853)<br>Fabian Morscheck; Christian Kallies; Enno Nagel; Rostislav Karásek | 利用历史空中交通与快速仿真评估法兰克福接驳走廊；强调真实运行冲突与运营假设的一致性。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T066 | 2025／AIAA AVIATION Forum | [An MILP-Based Framework for Multi-Modal Transportation with Air Taxis](https://doi.org/10.2514/6.2025-3781)<br>Shuting Yang; Yixuan Dai; Anne X. Li; Dengfeng Sun | MILP结合地面候选路径与地空车辆匹配；是多模式联合分配已被研究的直接近邻。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T067 | 2025／AIAA AVIATION Forum | [Operational Air Taxi Flight Routes in a Metropolitan Region](https://doi.org/10.2514/6.2025-3240)<br>Jordan K. Kam; Matthias Cassanova; Vishwanath Bulusu; Alexandre M. Bayen; Raja Sengupta | 在既有受控空域中设计并评估可操作的空中出租车航线；可学习现实约束说明，不直接移植航线目标。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T068 | 2025／Sustainable Futures | [The potential role of air taxis in reducing emissions and enhancing mobility: A case study of the Philippines](https://doi.org/10.1016/j.sftr.2025.100586)<br>Arthur Dela Peña | 调查与情景建模评估出行和减排潜力；依赖采用率等假设，不是已实现的运营性能收益。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T069 | 2025／International Conference on Communications | [V2X-Enabled Air-Ground Traffic Coordination for Enhancing On-Demand Air-Taxi Mobility](https://doi.org/10.1109/icc52391.2025.11160730)<br>Aoyu Pang; Maonan Wang; Wenwei Yue; Man-On Pun; Chung Shue Chen | UAGMC会议摘要强调V2X支持下逐客登机站选择；会议全文未取得，不能倒填期刊中的具体结构和实验表。 | P0：最接近运营/方法边界；先核对决策权限、在线信息和运行时间。 | A-index |
| T070 | 2026／arXiv | [Distance-based subsidy rate design to incentivize ride-hail access to advanced air mobility hubs](http://arxiv.org/abs/2606.21390v1)<br>Zhenglei Ji; Hai Yang; Joseph Y. J. Chow | 设计按距离的补贴协调网约车接驳与AAM运营利润；已有联运激励研究，但我们不增加价格控制权。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T071 | 2025／Aerospace | [Enhancing Urban Air Mobility Scheduling Through Declarative Reasoning and Stakeholder Modeling](https://doi.org/10.3390/aerospace12070605)<br>Jeongseok Kim; Kangjin Kim | 把起降场任务写成资源约束项目排程，使用ASP表达利益相关方要求；说明可行性组织并非只有MILP或RL。 | P1：对照承诺、资源或调度语义；目标不同不能直接搬成绩。 | A-index |
| T072 | 2025／Journal of Urban Mobility | [Willingness to use urban air mobility (UAM) as an airport shuttle across levels of automation](https://doi.org/10.1016/j.urbmob.2025.100162)<br>Soyeon Kim; Kaihan Zhang | 用问卷与有序logit比较不同自动化等级的使用意愿；用于解释部署背景，不提供调度收益证据。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T073 | 2026／Journal of Urban Mobility | [Urban air mobility vertiports: A bibliometric analysis and scoping review of physical site selection, design, and integration with ground transport](https://doi.org/10.1016/j.urbmob.2026.100265)<br>Chuancheng Li; Yin Zhang; Qunfeng Ji | 以范围综述梳理选址、设施设计和地面联运的交叉；综述样本范围不等于运营算法的完整排行榜。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T074 | 2026／Journal of Urban Mobility | [A support to vertiport micro-location selection at airport for urban air mobility airport shuttle service](https://doi.org/10.1016/j.urbmob.2025.100181)<br>Bojana Mirković; Nikola Ivanov; Dušan Crnogorac; Matija Sindik | 关注机场内起降场微观位置，而非仅区域中心点；提醒接驳距离差异可解释部分ATT变化。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T075 | 2026／AIAA AVIATION 2026 Forum | [Probabilistic Risk Modeling for Urban Air Mobility (UAM) Vertiport Operations With Non-Cooperative Airspace Users](https://doi.org/10.2514/6.2026-4535)<br>Suhwan Oh; Hyoungkeun Yoon; Taegyeong Yun; Seonghyun Lee; Yonghee Jang; Joonseo Kim; Junghyun A. Kim | 以随机轨迹和环境几何评估起降场附近非合作目标风险；安全评估与乘客效率是不同维度。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | A-index |
| T076 | 2025／ITSC 2025 | [Fair-CoPlan: Negotiated Flight Planning with Fair Deconfliction for Urban Air Mobility](https://arxiv.org/html/2508.14380v1)<br>Nicole Fronda; Phil Smith; Bardh Hoxha; Yash Vardhan Pant; Houssam Abbas | 分阶段分配容量、提交航班并公平消解冲突；值得学习明确权衡和可解释小切口，不要求所有指标同时最优。 | P1：对照承诺、资源或调度语义；目标不同不能直接搬成绩。 | 本轮复核元数据；历史阅读F |
| T077 | 2025／ITSC 2025 | [Optimal Fleet Sizing for On-Demand Urban Air Mobility Services Using Queueing-Theoretical Approach](https://its.papercept.net/conferences/scripts/abstract.pl?ConfID=91&Number=473)<br>Byeong Tak Park | 闭Jackson网络分析机队规模、服务可用性与收益；池化直觉需要具体网络假设，不直接变成我们的ATT单调定理。 | P1：对照承诺、资源或调度语义；目标不同不能直接搬成绩。 | 本轮复核元数据；历史阅读A |
| T078 | 2025／ITSC 2025 | [Sphereabout: A Spherical Intersection Design for Networked Urban Air Mobility](https://arxiv.org/html/2507.01610v1)<br>Farzan Moosavi; Bilal Farooq | 研究球形空中交汇结构；空域几何设计与运营决策需分层，不能直接作为选站策略比较。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | 本轮复核元数据；历史阅读F |
| T079 | 2025／ITSC 2025 | [Airspace Partitioning for Urban Air Mobility: A Macroscopic Approach with a Realistic Network](https://portal.fis.tum.de/en/publications/airspace-partitioning-for-urban-air-mobility-a-macroscopic-approa/)<br>Magdalena Peksa; Mehdi Keyvan-Ekbatani; Klaus Bogenberger | 真实网络上的宏观空域分区；可用于理解规模化约束，当前不为增加复杂度而改拓扑。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | 本轮复核元数据；历史阅读A |
| T080 | 2025／ITSC 2025 | [Operational Validation of a SWIM-Based UAM Traffic Management System: Lessons from Korea’s Grand Challenge](https://its.papercept.net/conferences/scripts/abstract.pl?ConfID=91&Number=348)<br>Kwangchun Kang; Suyoung Shin; Hyoungkeun Yoon; Taegyeong Yun; Changhwan Heo | 报告韩国运行验证中的交通管理接口与经验；可学习验证链和可部署性，而非只比较仿真奖励。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | 本轮复核元数据；历史阅读A |
| T081 | 2025／ITSC 2025 | [PRREACH: Probabilistic Risk Assessment Using Reachability for UAV Control](https://arxiv.org/html/2509.04451)<br>Nicole Fronda; Hariharan Narayanan; Sadia Afrin Ananna; Steven Weber; Houssam Abbas | 用可达性方法评估概率风险；安全证明的条件不能不经检查迁移到乘客—飞机调度。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | 本轮复核元数据；历史阅读F |
| T082 | 2025／ITSC 2025 | [A Third-Party Risk-Aware Decision-Making Method for UAVs in Dynamic Traffic Environments](https://dblp.org/db/conf/itsc/itsc2025.html)<br>Jiahui Xu; Yong Wang; Jiaru Zhong; Yiming Shu; Jiwei Tang; Chen Sun | 题名关注动态交通中的第三方风险；未取得摘要，不据此解释其方法或局限。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | 本轮复核元数据；历史阅读M |
| T083 | 2025／ITSC 2025 | [Pathfinders in the Sky: Formal Decision-Making Models for Collaborative Air Traffic Control in Convective Weather](https://arxiv.org/abs/2505.01804)<br>Jimin Choi; Kartikeya Anand; Husni R. Idris; Huy T. Tran; Max Z. Li | 形式化协作空管和对流天气决策；主要属于异常与安全管理，和当前固定天气主实验分开。 | P2背景：用于界定需求、网络或运行边界；不要求把这些机制加入主环境。 | 本轮复核元数据；历史阅读A |

## 附录B：检索量、引用量与主题统计

统计的是本轮目的性文献池，不用于推断全领域发表比例。

| 年份 | 条目数 |
|---|---:|
| 2025 | 45 |
| 2026 | 38 |

| 主题 | 条目数 |
|---|---:|
| UAM空域、仿真与其他 | 27 |
| 邻接交通：车辆／无人机协同 | 26 |
| UAM网络与需求 | 14 |
| UAM运营与调度 | 13 |
| UAM补能与能源 | 3 |

题名高频词（去除通用停用词）：mobility=47；urban=40；drone=16；optimization=15；demand=14；delivery=14；multi=14；scheduling=12；traffic=11；vertiport=10；routing=9；approach=8。

引用快照：OpenAlex，2026-10-29；引用量不是质量评分，2026论文受时间窗口影响明显。

取得可用OpenAlex引用快照：75/83条；以下为可匹配子集的前5，不是完整领域高被引榜。

| 文献 | 第一作者 | 引用快照 |
|---|---|---:|
| [The State of Urban Air Mobility Research: An Assessment of Challenges and Opportunities](https://doi.org/10.1109/tits.2024.3511453) | Sheikh Shahriar Ahmed | 50 |
| [The Heterogeneous-Fleet Electric Vehicle Routing Problem with Nonlinear Charging Functions](https://doi.org/10.1016/j.trc.2024.104932) | Weiquan Wang | 36 |
| [Integrating battery-related decisions into truck-drone tandem delivery problem with limited battery resources](https://doi.org/10.1016/j.trc.2025.105082) | Zhongshan Liu | 28 |
| [Urban air mobility network design and operations strategy in an urban agglomeration](https://doi.org/10.1016/j.tre.2025.104316) | Ziyu Wang | 28 |
| [Multi-Objective Multi-Drone Collaborative Routing Problem With Heterogeneous Delivery and Pickup Service](https://doi.org/10.1109/tits.2025.3548426) | Fangyu Hong | 25 |

与题名相近但不完全匹配者不自动合并；预印本与会刊引用可能分裂，具体原始定位保留在JSON。

## 研究诚信与交付边界

本文由AI辅助检索、整理和综合判断；不是外部独立同行评审，也没有作者本人确认。交付附件提供书目、中文归纳、检索说明、阅读范围、失败记录和来源链接；完整原始响应、完整摘要和作者稿仅保留在本地阅读缓存，不再批量打包分发。83条记录中未获取全文者仍需后续核验；尤其不能据它们写“全部现有文献都没解决”。本轮没有自动修改任何既有实验或宣称提高录用概率。
