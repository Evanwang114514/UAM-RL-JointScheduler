# 清华相关团队与2024—2026文献｜按S／AC主线重新筛选的扩展报告

本轮交付五份报告：本文件为完整搜索与迁移综合；02为所有教师核对记录；03为论文设计；04为实验研究路线；05为可执行的约2亿步固定实施设计，另交单文件代码与命令。**没有启动正式训练，没有连接服务器，也没有联系老师；有限本地接口检查不算新实证结果。**

## 0．最重要的结论

1. 研究问题不应继续写成“让UAGMC输”：应研究**闭环需求与运力分配中，候选动作在不同时间生效，如何组织与其价值相关的承诺信息**；UAGMC是必须如实保留的基线与代码来源。
2. 主线仍是S／AC／S＋AC。S是**按候选自身生效时刻重构表示**；AC是**保留背景、分离实际执行影响**，不能退化为“加ETA”和“一步ICM”。
3. 新增电子系文献中，**DyPS（KDD 2024）与CoopRide（KDD 2025）**很值得看：分别触及共享方式的条件性和交通中的需求—供给协作，但不能照搬其角色划分、奖励或大规模多智能体假设。[DyPS原文](https://fi.ee.tsinghua.edu.cn/public/publications/0261cfca-f04c-11f0-b382-2aaffa21d846.pdf)；[CoopRide原文](https://fi.ee.tsinghua.edu.cn/public/publications/785c48e0-f1d2-11f0-b382-2aaffa21d846.pdf)。<!--ref:N01--><!--anchor:section:Abstract--><!--ref:N02--><!--anchor:section:Abstract-->
4. 李秀方向的STAR、CIR、DIMA，以及于超方向的ARSQ，对“表示—动作—优化”的拆解仍较直接；**CIR是约束初始表示的TD方法，不是多步逆动力学方法**，这处容易误读，必须明确纠正。
5. 深研院不能只看数据／电子信息：物流运筹、电动车能源、海洋机器人也有相关成果；但“领域使用RL”不等于“能直接修复我们的S／AC”。
6. 下一步优先做**同信息对照＋实际执行链＋有限分支验证**；不是马上从这份文献池里再选几十个模型开训。

## 1．范围、工作量与不能夸大的地方

| 范围 | 官方名录记录 | 本轮处理 | 可核对结果 |
|---|---:|---|---|
| 自动化系 | 110 | 复用上一版逐人主页材料，按本轮Q&A重新筛选 | 110条有既有页面材料；不是110次新HTTP访问 |
| 电子工程系 | 160 | 从公开教师平台名录逐条获取主页 | 160条获取成功 |
| 深圳国际研究生院全院 | 274 | 从全院教师名录逐条获取主页，不限电子信息 | 273条成功、1条超时 |
| 合计 | **544条核对记录** | 姓名、机构、链接、方向线索、筛选与论文阶段分别记录 | **不是544位去重后的独立教师** |

名录范围包括SIGS各研究院／相关学科公开列出的教师；它不保证穷尽兼职、刚入职、尚未入网的人员。院系主页可访问不代表信息完备，更不代表已经逐篇读完该老师近三年论文。

本轮教师逐条初筛完成后，先锁定约12条重点研究路线，再进入论文检索；路线中的合作老师不能重复算成多个独立团队。所有未入选人员也在报告02，不隐去不相关方向或失败记录。

原文献池48条全部保留并重新分级；新条目另用N编号。**48条不表示本轮新增48次全文阅读**；部分条目仍只有出版社摘要、作者页或元数据。阅读深度逐条公开。

### 官方入口与检索路径

- 电子系：[院系教师入口](https://www.ee.tsinghua.edu.cn/ryqk/teacher/xxgdzyjs/js2.htm)、[公开教师平台](https://web.ee.tsinghua.edu.cn/)；使用页面实际调用的公开目录接口，四页共160条。
- SIGS：[全院教师入口](https://www.sigs.tsinghua.edu.cn/7644/list.htm)；使用前端公开teacherHome查询取得274条，不使用私人数据库。
- 自动化系：既有110条官方个人页材料及其原链接全部保留于报告02，本轮重新评价与S／AC的关系。
- 论文优先使用arXiv、PMLR、OpenReview、正式会议／出版社及作者公开论文；GitHub仅用于确认作者代码与实现线索。
- paper-search多索引脚本启动因本地缺少requests失败；未更改环境安装依赖，改用公开网页与论文原站。**不能宣称DBLP、OpenAlex、Semantic Scholar、Crossref等所有API均已系统跑通。**
- IEEE部分页面触发验证；ScienceDirect部分直接打开403；相关条目只按已取得的出版社索引／摘要／作者稿报告。没有突破访问限制，也没有把摘要称作全文。

## 2．依据你的Q&A，搜索方向如何调整

| 原先容易跑偏的方向 | 本轮纠正 | 对筛选的影响 |
|---|---|---|
| 机场越多效果越差就是论文问题 | 机场数量只作规模／边界，不作预设主故事 | 不专找“扩机场必坏”的文献 |
| 物理异步＝反馈延迟＝异步梯度 | 三者分开；飞机效应时间不能被通信延迟论文直接证明 | 延迟RL降为诊断参照 |
| 任何Shared都属于S | 参数共享、公共背景共享、候选作用时间对齐是三件事 | DyPS用于共享条件，不代替S |
| 逆模型预测动作准确＝AC成立 | 记录实际执行、保留不可控但价值相关的背景 | ICM只是一种辅助实现 |
| 拼接两个模块＝S＋AC创新 | 检验对齐背景与候选影响的交互 | 必做2×2而不只Base→S→SAC |
| 改联合时钟就解决S／AC | 时钟改变策略权限，应单独诊断 | 不把新时钟藏进方法对照 |
| 世界模型预测越多越好 | 已承诺过程直接计算，未决竞争才考虑预测 | 大WM／视觉生成降级 |
| 教师有热门论文就值得全迁移 | 先检查动作、目标、信息和算力是否匹配 | 多篇LLM／机器人生成策略只保留观察 |

## 3．建议持续跟踪的12条研究路线

这里的优先级是**对当前项目的帮助**，不是教师学术水平、招生概率或导师推荐排序；未核实招生名额，也没有联系任何人。

| 路线 | 代表老师／合作线 | 为什么留下 | 本轮取舍 |
|---|---|---|---|
| 1．动作条件价值与策略优化 | 于超 | ARSQ、生成策略优化、训练系统 | ARSQ优先；流策略不硬塞给离散选站 |
| 2．表示、世界模型与稳定性 | 李秀 | STAR、CIR、MARIE、DIMA | 表示与优化诊断最直接；不全量复制WM |
| 3．规划与模型搜索 | 杨君 | MAZero、鲁棒RL、分布式价值 | 有界分支与规划对照优先 |
| 4．无效／冗余动作 | 赵千川 | NPM等关注动作结构与学习效率 | 实际执行等价须验证，不能只看当前未移动 |
| 5．仿真预算与序贯决策 | 贾庆山 | OCBA采样、查询—策略失配、偏好与多智能体 | 用于减少无效验证与澄清训练目标 |
| 6．交通时空资源分配 | 李勇／徐丰力／王寰东 | DyPS、CoopRide及动态建模 | 新增重点；共同团队／合著关系不重复计证据 |
| 7．控制与延迟表示 | 王学谦（SIGS）／李志恒；谷源涛为延迟理论关联线 | 现实控制、编码与反馈延迟 | 与物理生效时间不同，不能抢走S主线 |
| 8．交通学习系统验证 | 封硕／李力／胡坚明等 | 密集学习、交通协作与关键状态验证 | 用于机制证据与保留强基座，不改需求抽样制造成绩 |
| 9．强数学／运筹参照 | 杨朋／戚铭尧／张灿荣／王凌 | 仓储、港口、配送中的任务—资源协作 | 作为强规则／优化与RL交互的重要参照 |
| 10．优化与训练机制 | 丁文伯／丁宁／汪玉 | RKHS策略牛顿、熵机制、RL系统 | 小型诊断优先；非参数和LLM方法不叫PPO即插即用升级 |
| 11．信息时效与网络控制 | 陈巍／周盛／牛志升 | 时效性、通信／计算资源调度 | AoI／模型漂移不等于ATT，不改交通目标 |
| 12．物理与学习结合 | 许银亮；曲钧天／曹顺翔为跨领域补线 | EV安全RL、决策导向预测、模型控制 | 展示全SIGS补查的价值；硬约束和已知物理不要丢给黑盒 |

其他已检索或保留观察的老师包括范文慧、游科友、宋士吉、袁春、陈鑫磊、李徵、曾龙、张涛、肖喜、江奔奔、黄高、唐彦嵩、姚权铭等；逐人的“主页阶段／论文阶段”见报告02，不能把这一名单都说成已完成同等深度的近三年文献综述。

## 4．最值得转成具体工作的六组线索

### 4.1 S：候选自己的时刻，比泛泛增加信息更值得检验

研究对象不是“知道更多未来”，而是同一合法事件在**自己的服务相关时刻**是否更有决策价值。比较Shared、同信息NOALIGN和自己的ETA查询；三组都得到候选ETA和同一C_t。

DIMA／MARIE帮助思考交互事件与模型组织，STAR帮助思考状态—动作表示；它们没有替我们证明自己的ETA优于一个学得好的普通事件编码器。先用有限分支的成本排序看增量，再投入RL。

### 4.2 AC：先核验执行，再谈信用

NPM适合提醒我们检查请求动作、执行器映射及长期等价；ARSQ有助于条件动作关系，但它们均不支持“一个无效飞机bit就可以随便删”。

本项目建议保留完整公共背景，再表达候选导致的可计算状态差。一步逆动力学可能学到即时记账，而不是几十分钟后的服务收益；应通过实际动作链与局部分支辨别。

### 4.3 S＋AC：共享必须有条件，不能把共享这个词当贡献

DyPS按行为角色动态调整参数共享，说明固定分组存在适用边界；这与我们的共同时间背景不同。可借鉴“哪些情况下应共享／分离”的设计问题，**不必把每个机场或每架飞机变成一个agent**。

CoopRide将状态协作、统一动作表达和合作强度联合考虑；最有用的是需求—容量关系进入表示与动作含义。其元梯度合作奖励会改变训练目标，不能悄悄移植到固定ATT实验。

### 4.4 为什么复杂结构反而差：先分离表示与优化

CIR针对TD表示约束，熵机制论文研究推理语言模型优化；它们的任务和理论条件都不同于我们的PPO。可借鉴通道尺度、表示漂移、策略熵及辅助更新前后KL的测量，不能直接宣称已经找到了UAM病因。

同样信息若能改善独立排序探针、却不能改善RL，优先查通路和优化；若新信息本身不改变动作排序，继续加网络可能只是增加学习成本。

### 4.5 WM：保留“决策导向”，放弃“预测一切”

DynaDiff及EV／风电的决策导向预测提供跨环境动态与预测—决策结合线索；但已知飞行／充电规则应直接计算。WM只在未决竞争、模型失配或估值确有问题时作为小辅助，不成为第三个平行大骨干。

### 4.6 解析参照不是可选项

配送匹配、订单—机器人协调、港口调度与EV安全控制提醒我们：很多结构适合由合法规则、匹配或小规划解决。应与同信息、同权限的强在线规则比较，并报告计算量；赢弱RL不够。

全未来Oracle是有意越过在线信息边界的诊断参照，不能和在线方法并列宣称可部署；可行Oracle解是可实现成本参照，除非有数学证明，不称最优下界。

## 5．阅读优先级与下一步使用方式

| 级别 | 建议阅读对象 | 读完要产出什么 |
|---|---|---|
| P0：近期直接使用 | STAR、DIMA、ARSQ、NPM、MAZero、DyPS、CoopRide（结构部分） | 一页“能迁移／不能迁移”、一个对照或诊断，不自动新开模型 |
| P1：机制诊断／后备 | CIR、MARIE、OCBA、Dense learning、DEER/MRAC、熵机制、CSET、决策导向预测／安全RL | 明确成立条件、数据需求与额外代价 |
| P2：观察／低适配 | 连续Flow、LLM自博弈、视觉生成、RKHS大规模二阶、量子电池／软机器人专用控制 | 保留可追踪卡片，暂不占主训练预算 |
| 基础而非新成果 | 2023多智能体延迟奖励等 | 解释概念边界，不充数为2024—2026新论文 |

P0不是“已证明能降低ATT”；它只是值得先转成一个**可证伪检查**。详细卡片保留原论文核心、本项目推断、适配风险和源码读取状态。

## 6．必须并列考虑的文献张力

下表是跨任务的启示张力，不代表这些论文彼此矛盾，也没有经过作者确认；`scholar_confirmation=pending`。

| 一种启示 | 另一种启示 | 本项目如何检验 |
|---|---|---|
| 共享降低样本与计算需求 | 异质行为可能需要动态共享或分离 | 同骨架下Shared与候选时刻／条件残差配对 |
| 状态—动作表示有助于控制 | 把不可控背景删掉可能损失价值信息 | 保留背景，单独增加Δ，不用逆模型筛掉一切背景 |
| 模型规划可能提高样本效率 | 模型偏差与成本可能抵消收益 | 先用已知仿真器有界分支，再考虑学习小模型 |
| 熵塌缩可能限制探索 | 低熵也可能是合理确定性决策 | 同时看合法机会、动作代价和性能，不把熵本身当目标 |
| 更丰富观测理论上允许更优策略 | 有限训练预算下可能更难优化 | 信息探针、函数保真、优化干扰三路分开 |
| 反事实编辑可减混杂 | 即时奖励相同不保证长程价值相同 | 只用物理合法状态对，保持后续策略和评估边界明确 |

## 7．本轮纠错与核验边界

1. EE王学谦与SIGS王学谦不是本轮可以合并的同一教师档案；机器人延迟RL论文只归SIGS线。
2. EE两个张超、跨系杨帆，以及张晓平、陈鑫磊等英文重名均按机构与主页消歧，不凭英文姓名认领成果。
3. MARIE正式为TMLR 2025，不写成ICLR；同一工作不同arXiv题名／版本不算多篇新成果。
4. 谷源涛的延迟奖励论文正式为L4DC 2023，列基础补读，不塞入2024—2026成果统计。
5. 曲钧天相关2024软机械臂正文与同题封面条目分开：正文DOI为`10.1002/aisy.202400112`，封面不算第二篇方法论文。
6. CSET在PMLR有正式书目与摘要；本轮PDF下载未完成，作者Yong Li的具体单位未从PDF首部完成独立复核。**可用其研究内容作外部相关工作，不把这条算作已经消歧的李勇团队成果。**
7. 部分2026期刊卷期晚于本轮检索时点，但已有预印本／早期在线；卡片分开标注，不把未来卷期等同已正式出刊。
8. 公开UAM主分支入口本轮尝试获取失败；只能确认已读本地120M源码中的配置，不能声称其已与GitHub或服务器逐字一致。
9. 不报告未经来源支持的引用次数，不以搜索排序、明星教师或热门关键词代替项目适配性。
10. 本轮没有完成全球新颖性排查，没有实测论文代码，不宣称所有教师所有近三年成果均已穷尽。

## 8．交付后应如何推进

**先做：**吸收正在运行／已完成的48M与240M证据，补齐执行与信息诊断；根据报告04决定是否开启约48M同信息机制包。

**随后做：**若S或执行影响确有独立／交互增量，锁定方法与分析，再开约72M独立确认；若只有S有效，收缩论文，不保全所有模块。

**暂时不做：**新加pad／天气／新机队、全量Flow／LLM／世界模型、重新搜索一整套PPO参数、用新联合时钟掩盖原比较失配。

教师联系应在形成一张可靠的机制图、一个小型复现和清楚问题之后进行；本报告不是套磁邮件，不包含招生或录用保证。

## 9．逐篇文献卡片

下文由已核对书目与本轮新增记录生成；P编号为旧池重评，N编号为新增。每条“迁移”均为本项目假说，而不是原论文已经在UAM验证的结论。代码可访问、静态阅读与实际复现严格分开。


**本版卡片共66条：旧池48条＋新增18条；新增中17条位于2024—2026窗口，另1条为2023基础补读。66条包括方法、系统、运筹、评论及低适配记录，不等于66篇全都直接有用的RL方法，也不等于66篇全文精读。**

### P01｜Learning from Suboptimal Data in Continuous Control via Auto-Regressive Soft Q-Network

**ICML 2025；arXiv:2502.00288；路线：于超；当前优先级：P0。** [原始来源](https://arxiv.org/abs/2502.00288)<!--ref:P01--><!--anchor:section:Abstract-->

- 原文核心：ARSQ用自回归维度优势表达动作分量依赖，并结合连续动作的粗到细离散化。
- 本项目迁移假说：借鉴条件动作价值：乘客候选先确定后，飞机价值修正依赖该候选；共享公共背景，而不是两个独立头各学各的。
- 风险／不照搬部分：我们的动作本来是离散的，不必照搬分箱；它使用离线/次优数据，额外数据与更新成本必须单列。
- 阅读层级：方法＋消融（含共享/分离骨干）。
- 源码：[作者仓库](https://github.com/HosnLS/ARSQ)；已查看README及目录；未运行、未逐文件审计算法。
- 重评：原优先级P0→本轮P0；近期机制接口／判别诊断。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P02｜ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning

**NeurIPS 2025；arXiv:2505.22094；路线：于超；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2505.22094)<!--ref:P02--><!--anchor:section:Abstract-->

- 原文核心：让流匹配策略可用在线强化学习微调，兼顾少步生成和策略优化。
- 本项目迁移假说：可借鉴保留强策略再微调的思路；目前不建议把少量机场分类动作换成流模型。
- 风险／不照搬部分：主要面对连续机器人动作与生成策略；不是直接替换SB3 PPO即可提升ATT。
- 阅读层级：摘要/项目页。
- 源码：[作者仓库](https://github.com/ReinFlow/ReinFlow)；已查看README及目录；未运行、未逐文件审计算法。
- 重评：原优先级P2→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P03｜SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling

**ICLR 2026（作者项目页）；arXiv:2509.25756；路线：于超、丁文伯合作；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2509.25756)<!--ref:P03--><!--anchor:section:Abstract-->

- 原文核心：将流生成过程组织成可优化的序列结构，处理策略梯度传播与参数化稳定性。
- 本项目迁移假说：提示我们检查新增网络通路是否难优化，而不是把“信息多却变差”直接解释为信息无用。
- 风险／不照搬部分：连续流策略的技术不能直接当作离散UAM的更强PPO；当前不占主要实验预算。
- 阅读层级：摘要/作者项目页。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P04｜RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation

**arXiv 2025:2509.15965；个人主页列OSDI 2026；路线：于超；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2509.15965)<!--ref:P04--><!--anchor:section:Abstract-->

- 原文核心：研究强化学习训练工作流与分布式执行效率，而非提出交通决策表示。
- 本项目迁移假说：适合将来做训练、采样、推理分阶段计时与资源调度时参考。
- 风险／不照搬部分：系统吞吐收益不能直接迁移为本项目FPS或ATT收益；本轮未独立核验OSDI论文集条目。
- 阅读层级：摘要/作者主页。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P05｜MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs

**ICLR 2026；arXiv:2510.15414；路线：于超、陈鑫磊合作；当前优先级：P2。** [原始来源](https://thu-nics.github.io/MARSHAL/)<!--ref:P05--><!--anchor:section:Abstract-->

- 原文核心：研究多智能体推理中的回合级反馈与不同角色的优势处理。
- 本项目迁移假说：可检查乘客/飞机两个分支的优势尺度和有效反馈是否严重不同。
- 风险／不照搬部分：LLM博弈中的自博弈不等于共同ATT目标；不能由此给UAM增加对抗目标或新奖励。
- 阅读层级：摘要/项目页。
- 源码：未核验可用官方实现；未核验官方代码。
- 版本与消歧：早期题名MARS，与MARSHAL按同一工作去重。。
- 重评：原优先级P1→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P06｜RE-PO: Robust Enhanced Policy Optimization as a General Framework for LLM Alignment

**ICLR 2026；arXiv:2509.24159；路线：于超；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2509.24159)<!--ref:P06--><!--anchor:section:Abstract-->

- 原文核心：在偏好优化中估计和处理标签可靠性，面向大语言模型对齐。
- 本项目迁移假说：若将来用近似求解器产生偏好标签，可考虑标签不确定性。
- 风险／不照搬部分：不是通用PPO升级版；我们有明确数学ATT时，不应人为制造人类偏好学习问题。
- 阅读层级：摘要/引言。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P07｜STEAM: Self-Supervised Temporal Ensemble Advantage Modeling for Real-World Robot Learning

**arXiv 2026:2606.29834；2026-10-08修订版；路线：于超、陈鑫磊合作；当前优先级：P1。** [原始来源](https://arxiv.org/abs/2606.29834)<!--ref:P07--><!--anchor:section:Abstract-->

- 原文核心：用时序进度相关的自监督优势模型与集成降低机器人学习中的评价难度。
- 本项目迁移假说：可做服务进度/就绪顺序的辅助探针，检验表示是否理解有用的阶段变化。
- 风险／不照搬部分：论文也承认时间偏移不总等于任务进度，视觉可能漏掉隐状态；更不能把进度分数当ATT奖励。
- 阅读层级：方法＋局限性。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P08｜OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control

**IEEE RA-L 9(3):2838–2844，2024；路线：于超；当前优先级：P2。** [原始来源](https://ieeexplore.ieee.org/document/10409589/)<!--ref:P08--><!--anchor:section:Abstract-->

- 原文核心：提供GPU并行无人机仿真与强化学习基准，属于研究基础设施。
- 本项目迁移假说：可以借鉴性能计时和任务定义方式，不需要为UAM调度迁移物理引擎。
- 风险／不照搬部分：飞控动力学与机场运营不是同一层问题；框架并行速度不能直接套给我们的Python仿真。
- 阅读层级：IEEE摘要＋作者PDF。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P09｜RLinf-USER: A Unified and Extensible System for Real-World Online Policy Learning in Embodied AI

**arXiv 2026:2602.07837；路线：于超、陈鑫磊合作；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2602.07837)<!--ref:P09--><!--anchor:section:Abstract-->

- 原文核心：统一真实机器人在线学习中的数据、策略和执行流水线。
- 本项目迁移假说：属于长期工程参考，不作为当前S/AC机制候选。
- 风险／不照搬部分：系统论文不等于对ATT更优的算法；无需为研究框架重构整个现有项目。
- 阅读层级：摘要/引言。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P10｜Temporal Difference Learning with Constrained Initial Representations

**Information Sciences 755:123774，2026；arXiv:2602.11800；路线：李秀；当前优先级：P1。** [原始来源](https://arxiv.org/abs/2602.11800)<!--ref:P10--><!--anchor:section:Abstract-->

- 原文核心：CIR约束初始表示的尺度，并用保留信息的结构改善TD学习稳定性。
- 本项目迁移假说：保留UAGMC原通路，新增就绪信息先限定单位和数值范围；优先检验小残差，而非替换全部骨干。
- 风险／不照搬部分：理论条件不覆盖任意神经PPO；默认网络较大，部分困难DMC任务并不占优，不能称普适稳定修复。
- 阅读层级：方法＋实验局限。
- 源码：[作者仓库](https://github.com/dmksjfl/CIR)；已查看README及目录；未运行、未逐文件审计算法。
- 版本与消歧：Crossref卷期标2026年11月，晚于本轮日期；公开预印本已存在，按2026公开成果保留，不称11月已过去。。
- 重评：原优先级P0→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P11｜Enhancing Visual Reinforcement Learning with State–Action Representation

**Knowledge-Based Systems 304:112487，2024；路线：李秀；当前优先级：P0。** [原始来源](https://www.sciencedirect.com/science/article/pii/S0950705124011213)<!--ref:P11--><!--anchor:section:Abstract-->

- 原文核心：STAR将状态—动作表示及其动态关系用于价值学习。
- 本项目迁移假说：做“仅critic接收S/AC信息”与“actor/critic均接收”的小对照，区分估值收益与策略表示干扰。
- 风险／不照搬部分：视觉RL与我们的结构化状态不同；不能把其理论和实验直接等同于本项目辅助梯度问题已被证明。
- 阅读层级：摘要＋引言/讨论。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P0→本轮P0；近期机制接口／判别诊断。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P12｜Off-policy reinforcement learning algorithms can be sample-efficient for continuous control via sample multiple reuse

**Information Sciences 666:120371，2024；路线：李秀；当前优先级：P2。** [原始来源](https://www.sciencedirect.com/science/article/pii/S0020025524002846)<!--ref:P12--><!--anchor:section:Abstract-->

- 原文核心：SMR通过同批数据的多次利用提高off-policy训练样本效率。
- 本项目迁移假说：提醒我们同时核算环境步数、优化次数和墙钟时间，不把更多更新伪装成等算力。
- 风险／不照搬部分：不能照搬成给PPO无限增加epochs；分布偏移和策略更新约束不同。
- 阅读层级：摘要/出版社介绍。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P13｜Decentralized Transformers with Centralized Aggregation are Sample-Efficient Multi-Agent World Models

**TMLR，2025；arXiv:2406.15836；路线：李秀；当前优先级：P1。** [原始来源](https://arxiv.org/abs/2406.15836)<!--ref:P13--><!--anchor:section:Abstract-->

- 原文核心：MARIE用局部动态模型与集中聚合协调多智能体世界模型。
- 本项目迁移假说：借鉴“公共背景统一聚合，局部动作影响分开表达”，不把S和AC做成两条互不交流的支路。
- 风险／不照搬部分：不能把可精确计算的已承诺飞机过程全部改为预测；聚合数据源曾误标ICLR，本报告以TMLR为准。
- 阅读层级：正式论文PDF/方法。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P14｜Revisiting Multi-Agent World Modeling from a Diffusion-Inspired Perspective

**NeurIPS 2025；arXiv:2505.20922；路线：李秀；当前优先级：P0。** [原始来源](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5e85dc044c7e774ebc1ce44963b4755c-Abstract-Conference.html)<!--ref:P14--><!--anchor:section:Abstract-->

- 原文核心：DIMA逐次揭示各智能体动作来组织状态预测，减少一次处理整个联合转移的负担。
- 本项目迁移假说：迁移顺序条件分解：先形成共同承诺状态，再加入乘客动作，最后加入飞机动作的条件增量。
- 风险／不照搬部分：顺序条件化不是因果识别；完整扩散世界模型代价较高，且不是所有任务都比其他WM更快。
- 阅读层级：方法＋计算成本附录。
- 源码：[作者仓库](https://github.com/breez3young/DIMA)；已查看README及目录；未运行、未逐文件审计算法。
- 重评：原优先级P0→本轮P0；近期机制接口／判别诊断。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P15｜Normalization Enhances Generalization in Visual Reinforcement Learning

**AAMAS 2024；arXiv首次2023；路线：李秀、李志恒合作；当前优先级：P1。** [原始来源](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1137.pdf)<!--ref:P15--><!--anchor:section:Abstract-->

- 原文核心：通过适当归一化改善视觉干扰下的泛化，不依赖额外大型结构。
- 本项目迁移假说：先排查历史、当前OD、ready-time各分支量纲和尺度，采用合法物理单位，避免信息支路抢占数值尺度。
- 风险／不照搬部分：视觉风格归一化不等同于物理时间对齐；不能随意打乱真实时间量来做增强。
- 阅读层级：正式论文/引言。
- 源码：[作者仓库](https://github.com/lilucse/Normalization-Enhances-Generalization-in-Visual-Reinforcement-Learning)；已查看README及目录；未运行、未逐文件审计算法。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P16｜Efficient Multi-agent Reinforcement Learning by Planning

**ICLR 2024；arXiv:2405.11778；路线：杨君；当前优先级：P0。** [原始来源](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d74e6bfe9ce029526e69db14d2c281ec-Abstract-Conference.html)<!--ref:P16--><!--anchor:section:Abstract-->

- 原文核心：MAZero结合局部动态、集中价值与采样规划，而非枚举所有联合行动序列。
- 本项目迁移假说：我们K≤4、两分支联合候选较少，可只对少量关键状态做有时限的分支回放，用作诊断或小型规划器。
- 风险／不照搬部分：规划只能读取当前已知承诺；长期rollout策略依赖和模型误差仍在，不是全局最优上界。
- 阅读层级：方法§4及搜索设计。
- 源码：[作者仓库](https://github.com/liuqh16/MAZero)；已查看README及目录；未运行、未逐文件审计算法。
- 重评：原优先级P1→本轮P0；近期机制接口／判别诊断。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P17｜Single-Trajectory Distributionally Robust Reinforcement Learning

**ICML 2024；初稿2023；路线：杨君；当前优先级：P1。** [原始来源](https://proceedings.mlr.press/v235/liang24d.html)<!--ref:P17--><!--anchor:section:Abstract-->

- 原文核心：研究单轨迹数据下分布鲁棒Q学习的估计与收敛。
- 本项目迁移假说：可用于未来需求分布转移的理论阅读。
- 风险／不照搬部分：分布鲁棒不是跨训练seed稳定性的同义词，也不能在冻结主目标下悄悄换成最坏分布目标。
- 阅读层级：摘要/论文元数据。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P18｜Episodic Novelty Through Temporal Distance

**ICLR 2025；arXiv:2501.15418；路线：杨君、赵千川合作；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2501.15418)<!--ref:P18--><!--anchor:section:Abstract-->

- 原文核心：ETD用时序距离形成探索相关的新颖性信号。
- 本项目迁移假说：只借用时序距离作为表征诊断，检查相似当前快照是否对应不同未来服务进度。
- 风险／不照搬部分：原算法有内在探索奖励，我们不能照搬到固定ATT主合同；时序可达距离也不等于飞机服务就绪时间。
- 阅读层级：摘要/方法概述。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P19｜DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning

**JAIR 83，2025；arXiv:2004.14547；路线：杨君、赵千川合作；当前优先级：P1。** [原始来源](https://arxiv.org/abs/2004.14547v3)<!--ref:P19--><!--anchor:section:Abstract-->

- 原文核心：通过回报分布建模讨论风险敏感决策。
- 本项目迁移假说：可分析价值误差与尾部服务风险，但主比较仍保持原平均ATT。
- 风险／不照搬部分：初稿2020，不能包装成2025才提出的算法；使用CVaR会改变目标；也不要与其他作者同名DSAC混同。
- 阅读层级：摘要/版本记录。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P20｜No Prior Mask: Eliminate Redundant Action for Deep Reinforcement Learning

**AAAI 2024；arXiv:2312.06258；路线：赵千川；当前优先级：P0。** [原始来源](https://ojs.aaai.org/index.php/AAAI/article/view/29652)<!--ref:P20--><!--anchor:section:Abstract-->

- 原文核心：NPM通过动作转移相似性估计识别冗余动作，减少无意义的动作区分。
- 本项目迁移假说：优先审查飞机分支：不同原始动作经fallback后是否确实产生相同物理转移、承诺和reward；先做精确等价类。
- 风险／不照搬部分：一步没移动不等于长期无效；论文折扣价值界含(1−γ)分母，不能直接用于我们的γ=1。
- 阅读层级：方法＋Lemma 1条件。
- 源码：[作者仓库](https://github.com/zhongdy15/npm)；已查看README及目录；未运行、未逐文件审计算法。
- 重评：原优先级P0→本轮P0；近期机制接口／判别诊断。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P21｜Query-Policy Misalignment in Preference-Based Reinforcement Learning

**ICLR 2024；路线：贾庆山；当前优先级：P1。** [原始来源](https://proceedings.iclr.cc/paper_files/paper/2024/file/c703e3320a0d0f1ed2e2b95964249c17-Paper-Conference.pdf)<!--ref:P21--><!--anchor:section:Abstract-->

- 原文核心：指出对奖励模型最有信息的查询，不一定对正在优化的策略最有用。
- 本项目迁移假说：支持区分“预测更准确”与“选动作更准确”：S/WM探针应看动作排序及错误代价，不只看MSE。
- 风险／不照搬部分：这是偏好查询场景的结论，不能直接当作我们的辅助损失干扰已被证实。
- 阅读层级：论文摘要/问题定义。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P22｜An OCBA-Based Method for Efficient Sample Collection in Reinforcement Learning

**IEEE TASE 21(3):3615–3626，2024；DOI首次在线2023；路线：贾庆山；当前优先级：P1。** [原始来源](https://ieeexplore.ieee.org/document/10148580/)<!--ref:P22--><!--anchor:section:Abstract-->

- 原文核心：围绕计算预算分配提高强化学习样本收集效率。
- 本项目迁移假说：当需要多次分支回放时，先把计算用在动作差距小、排序不确定且会影响总等待的状态。
- 风险／不照搬部分：IEEE全文访问受限；本轮核对作者、年份和题名，算法细节未按全文审计，不将其当已就绪实现方案。
- 阅读层级：注册书目＋作者方向；全文待取。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P23｜Multi-Agent Reinforcement Learning via Agent-Specific Preference

**arXiv 2026:2608.08604；路线：贾庆山；当前优先级：P1。** [原始来源](https://arxiv.org/abs/2608.08604)<!--ref:P23--><!--anchor:section:Abstract-->

- 原文核心：MAGPIE利用智能体特定偏好与聚合关系帮助多智能体学习。
- 本项目迁移假说：作为公共目标和局部反馈如何衔接的对照阅读。
- 风险／不照搬部分：我们的ATT已知，不应改成偏好奖励；未核实正式期刊，不采用聚合网站的TASE标签。
- 阅读层级：摘要/元数据。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P24｜On Sampling Efficiency Optimization in Constrained Reinforcement Learning

**IEEE AIM 2024，966–971；路线：贾庆山；当前优先级：P1。** [原始来源](https://ras.papercept.net/images/temp/AIM/files/0220.pdf)<!--ref:P24--><!--anchor:section:Abstract-->

- 原文核心：围绕受约束学习的采样效率，讨论性能与可行性评估的计算分配。
- 本项目迁移假说：对人工漏斗、置信不足状态优先追加回放有启发。
- 风险／不照搬部分：预算分配思想不是新的物理控制器；需要保留独立验证，不能只给看起来好的候选追加然后直接宣布最好。
- 阅读层级：作者PDF。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P25｜Heterogeneous Multi-Agent Reinforcement Learning for Zero-Shot Scalable Collaboration

**Neurocomputing 648:130716，2025；arXiv:2404.03869；路线：范文慧；当前优先级：P1。** [原始来源](https://www.sciencedirect.com/science/article/pii/S0925231225013888)<!--ref:P25--><!--anchor:section:Abstract-->

- 原文核心：SHPPO让大部分参数共享，同时用受控的异构层表达角色差异。
- 本项目迁移假说：候选机场编码器共享背景，乘客/飞机头保留角色相关的小差异，比较完全分离与共享的增量。
- 风险／不照搬部分：机场候选数增加不是智能体数量增加；论文零样本扩智能体结论不能直接当作K2→K4保证。
- 阅读层级：方法＋消融。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P26｜OdML: An optimization-driven meta-learning framework for multi-task approximate model predictive control

**Applied Soft Computing 185:113882，2025；路线：范文慧；当前优先级：P1。** [原始来源](https://www.sciencedirect.com/science/article/pii/S1568494625011950)<!--ref:P26--><!--anchor:section:Abstract-->

- 原文核心：把近似MPC策略的在线适配组织为优化问题，辅以约束处理。
- 本项目迁移假说：适合长期考虑“学习快速候选＋小型可行性优化”的架构。
- 风险／不照搬部分：不是RL算法的直接增量；当前环境固定，不需要为了元学习而重新造大量变化场景。
- 阅读层级：出版社摘要＋引言/节选。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P27｜A Matching Algorithm with Reinforcement Learning and Decoupling Strategy for Order Dispatching in On-Demand Food Delivery

**Tsinghua Science and Technology 29(2):386–399，2024；路线：王凌；当前优先级：P1。** [原始来源](https://www.sciopen.com/article/10.26599/TST.2023.9010069)<!--ref:P27--><!--anchor:section:Abstract-->

- 原文核心：将复杂派单拆为学习排序与可执行匹配，减少直接优化整个组合动作的负担。
- 本项目迁移假说：飞机调度可以先学习候选优先级或边际评分，再由固定可行执行器落实；不必让策略承担全部搜索。
- 风险／不照搬部分：原文配送约束、目标和订单揭示方式不同；需固定UAM控制权限，不能让某方案额外改变运力。
- 阅读层级：出版社摘要/介绍。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P0→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P28｜Order Dispatching Via GNN-Based Optimization Algorithm for On-Demand Food Delivery

**IEEE TITS 25(10):13147–13162，2024；路线：王凌；当前优先级：P1。** [原始来源](https://ieeexplore.ieee.org/document/10509849/)<!--ref:P28--><!--anchor:section:Abstract-->

- 原文核心：题名与作者记录明确连接GNN、运筹优化和动态派单。
- 本项目迁移假说：保留为“网络缩小选择范围＋优化器执行”的重点待精读来源。
- 风险／不照搬部分：本轮只完成作者官网与Crossref/DBLP书目交叉核验，IEEE全文受限；不据二手摘要承诺算法细节或收益。
- 阅读层级：注册书目/作者清单；全文待取。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P29｜Reinforcement Learning-Based Dynamic Order Recommendation for On-Demand Food Delivery

**Tsinghua Science and Technology 29(2):356–367，2024；路线：王凌；当前优先级：P1。** [原始来源](https://www.sciopen.com/article/10.26599/TST.2023.9010041)<!--ref:P29--><!--anchor:section:Abstract-->

- 原文核心：用时序策略组织动态推荐/分配顺序，处理不同执行者之间的冲突。
- 本项目迁移假说：对乘客决策和飞机优先级的执行次序可作交通系统类比。
- 风险／不照搬部分：派单冲突和订单数量不是ATT；不能直接搬论文的商业目标或奖励。
- 阅读层级：出版社摘要/介绍。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P30｜Deep reinforcement learning for order-tote-robot coordination in multi-tote storage and retrieval systems with sequential picking stations

**European Journal of Operational Research，2026，在线6月16日；路线：杨朋；当前优先级：P1。** [原始来源](https://www.sciencedirect.com/science/article/pii/S0377221726005370)<!--ref:P30--><!--anchor:section:Abstract-->

- 原文核心：将订单、料箱和机器人协同视为耦合序列决策；论文摘要指出更多工作站不保证性能持续改善。
- 本项目迁移假说：很适合启发“需求分散、服务资源分散和执行次序”共同决定收益，而不是只看资源节点数量。
- 风险／不照搬部分：仓储结果不能证明我们的K3极值正确；仍须在固定trace与几何下检查组批和机队分布。
- 阅读层级：出版社摘要＋介绍。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P0→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P31｜Reinforcement learning for pod retrieval scheduling and robot dispatching in the multi-deep compact robotic mobile fulfillment systems

**Expert Systems with Applications 327:132847，2026；路线：杨朋、戚铭尧合作；当前优先级：P1。** [原始来源](https://www.sciencedirect.com/science/article/pii/S0957417426017604)<!--ref:P31--><!--anchor:section:Abstract-->

- 原文核心：把货架检索顺序与机器人分派联动，处理阻挡等资源依赖。
- 本项目迁移假说：启发真正的C×M：表示不是单独多加特征，而是进入“选任务—选资源”的条件决策链。
- 风险／不照搬部分：主要读到出版社摘要、引言与章节节选，未拿到付费完整方法；不声称已可完整复现。
- 阅读层级：出版社摘要＋引言/节选。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P0→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P32｜Omni-scale Learning-based Sequential Decision Framework for Order Fulfillment of Tote-handling Robotic Systems

**arXiv 2026:2605.08758；路线：杨朋；当前优先级：P1。** [原始来源](https://arxiv.org/abs/2605.08758)<!--ref:P32--><!--anchor:section:Abstract-->

- 原文核心：面向订单—料箱—机器人序列协同，考虑不同规模下的学习式决策。
- 本项目迁移假说：可以围绕共享候选编码、条件决策和跨规模失效建立对照。
- 风险／不照搬部分：序列结构的优势必须和同信息/同搜索成本基线比较；不是给每个K单独改环境。
- 阅读层级：摘要/问题结构。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P33｜Integrated Scheduling Optimization for Automated Container Terminal: A Reinforcement Learning-Based Approach

**IEEE TITS 26(7):10019–10035，2025；路线：张灿荣；当前优先级：P1。** [原始来源](https://ieeexplore.ieee.org/document/10981459/)<!--ref:P33--><!--anchor:section:Abstract-->

- 原文核心：题名与正式书目确认研究自动化码头多设备集成调度及强化学习。
- 本项目迁移假说：值得优先获取全文，用于强解析/小规模精确求解参照及学习与优化混合的交通叙事。
- 风险／不照搬部分：本轮确认作者官网CV和注册书目，IEEE全文受限；未将二手摘要中的数值改进写入证据。
- 阅读层级：注册书目/作者CV；全文待取。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P34｜A delay-robust method for enhanced real-time reinforcement learning

**Neural Networks 181:106769，2025；在线2024；路线：王学谦、李志恒合作；当前优先级：P1。** [原始来源](https://www.global-optimization.com/boyuan/Mypaper/NN-2025.pdf)<!--ref:P34--><!--anchor:section:Abstract-->

- 原文核心：MRAC/MISMDP研究实时观察—计算—执行中的时延以及需要保留的最小信息。
- 本项目迁移假说：用于区分“当前观测缺了什么历史/承诺信息”与“模型只需估计未来服务能力”。
- 风险／不照搬部分：实时计算时延不等于飞机几十分钟后才生效的运营时延；若仿真阻塞执行，不能假装有同一种问题。
- 阅读层级：作者全文/问题定义与方法。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P0→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P35｜Delay-Aware Reinforcement Learning With Encoder-Enhanced State Representations

**IEEE TAI 7(8):4476–4490，2026；DEER初稿2024；路线：王学谦、李志恒合作；当前优先级：P1。** [原始来源](https://ieeexplore.ieee.org/abstract/document/11396950)<!--ref:P35--><!--anchor:section:Abstract-->

- 原文核心：DEER用编码器组织延迟观测与历史动作，压缩恢复决策所需上下文。
- 本项目迁移假说：启发用合法已承诺事件和历史动作形成固定维上下文，避免泛化地预测整个未来。
- 风险／不照搬部分：同样需区分观测延迟与运营生效延迟；初稿与正式版按一组工作去重，但不假定每个细节完全相同。
- 阅读层级：IEEE元数据/摘要＋arXiv:2406.03102方法。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P36｜ReDiffuser: Reliable Decision-Making Using a Diffuser with Confidence Estimation

**ICML 2024；路线：李徵（关联补查，不归给王学谦）；当前优先级：P2。** [原始来源](https://proceedings.mlr.press/v235/he24e.html)<!--ref:P36--><!--anchor:section:Abstract-->

- 原文核心：对离线扩散式决策加入置信估计，调节规划可靠性。
- 本项目迁移假说：借鉴“不可靠状态少信复杂模块”的门控思想，而非直接换成Diffuser。
- 风险／不照搬部分：置信估计不是安全认证；需要覆盖充分的离线数据。李徵2025加入SIGS，不能把2024论文算作其现任组当年产出。
- 阅读层级：正式论文摘要/PDF。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P37｜Curse of rarity for autonomous vehicles

**Nature Communications 15:4808，2024，Comment；路线：封硕；当前优先级：P2。** [原始来源](https://www.nature.com/articles/s41467-024-49194-0)<!--ref:P37--><!--anchor:section:Abstract-->

- 原文核心：讨论自动驾驶中关键稀有场景难以被普通数据规模覆盖的问题。
- 本项目迁移假说：先统计我们的高动作代价差状态占比，不能假定平均ATT已充分揭示机制。
- 风险／不照搬部分：这是一篇评论文章，不是新调度算法；ATT密集损失与稀有安全事件不完全同类。
- 阅读层级：原文。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P38｜Adaptive Testing Environment Generation for Connected and Automated Vehicles With Dense Reinforcement Learning

**IEEE TITS 26(4):5135–5145，2025；初稿2024；路线：封硕、胡坚明、张毅合作；当前优先级：P1。** [原始来源](https://www.jingxuanyang.com/file_upload/2025-DenseRL-TITS.pdf)<!--ref:P38--><!--anchor:section:Abstract-->

- 原文核心：通过密集反馈和自适应场景生成，把测试计算集中于重要场景。
- 本项目迁移假说：可另建诊断状态库，专门复核组批临界、就绪排序反转和跨站空闲—积压并存。
- 风险／不照搬部分：诊断库不是主评估分布；不能在偏置测试集上赢了就声称整体ATT改善。
- 阅读层级：作者PDF/方法。
- 源码：[作者仓库](https://github.com/THU-AI-Testing/Adaptive-Testing-DenseRL)；已查看README及目录；未运行、未逐文件审计算法。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P39｜Breaking through safety performance stagnation of autonomous vehicles with dense learning

**Nature Communications 17:3163，2026；路线：封硕；当前优先级：P1。** [原始来源](https://www.nature.com/articles/s41467-026-69761-x)<!--ref:P39--><!--anchor:section:Abstract-->

- 原文核心：利用更有信息的事件组织学习，并让修正策略在关键状态介入。
- 本项目迁移假说：保留UAGMC强基座，只在有充分证据的高代价状态启用小残差；同时测覆盖率与误介入代价。
- 风险／不照搬部分：安全事件任务不同；选择性修正本身也不是我们的新颖性，需要证明UAM特有的就绪/组批机制。
- 阅读层级：结果＋方法＋讨论/局限。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P0→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P40｜Policy Gradient Methods for the Cost-Constrained LQR: Strong Duality and Global Convergence

**arXiv 2024:2406.03734；路线：游科友；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2406.03734)<!--ref:P40--><!--anchor:section:Abstract-->

- 原文核心：研究成本约束LQR下策略优化的对偶和收敛结构。
- 本项目迁移假说：适合学习怎样把假设、可行域和理论结论写清楚。
- 风险／不照搬部分：LQR是线性二次结构，不能把其全局收敛结论移植给非线性组批UAM或神经PPO。
- 阅读层级：摘要/定理适用问题。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P41｜Asynchronous Parallel Policy Gradient Methods for the Linear Quadratic Regulator

**arXiv 2024:2407.03233；路线：游科友；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2407.03233)<!--ref:P41--><!--anchor:section:Abstract-->

- 原文核心：研究并行优化工作者异步更新时的策略梯度。
- 本项目迁移假说：仅在训练系统确实采用异步梯度时有直接用处。
- 风险／不照搬部分：异步优化不等于乘客和飞机的异步生效，不能拿题名替我们的物理时序失配背书。
- 阅读层级：摘要/问题定义。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P42｜Deep Reinforcement Learning for Traveling Purchaser Problems

**arXiv 2024:2404.02476；2025期刊信息另需核验版本；路线：游科友、宋士吉合作；当前优先级：P1。** [原始来源](https://arxiv.org/abs/2404.02476)<!--ref:P42--><!--anchor:section:Abstract-->

- 原文核心：把路线选择与采购子问题拆开，同时依据共同目标评价。
- 本项目迁移假说：学习最值得学习的顺序/评分，让可精确求解的资源子问题交给数学方法，是轻量C×M的有用范式。
- 风险／不照搬部分：静态完整实例并不符合在线乘客揭示边界；本报告以公开预印本方法为准，不混入同名版本的未核实期刊细节。
- 阅读层级：预印本摘要/结构。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P43｜Solving Continual Offline Reinforcement Learning with Decision Transformer

**arXiv 2024:2401.08478；路线：袁春；当前优先级：P1。** [原始来源](https://arxiv.org/abs/2401.08478)<!--ref:P43--><!--anchor:section:Abstract-->

- 原文核心：研究连续任务的离线决策学习中如何平衡旧知识保留和新任务适应。
- 本项目迁移假说：可以借鉴保留基座知识与小适配支路；用于理解旧好策略是否被新增训练破坏。
- 风险／不照搬部分：跨任务遗忘与同一PPO run的late drift不同，不能仅靠这篇论文诊断原因。
- 阅读层级：摘要/结构概述。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P44｜Preference Optimization for Combinatorial Optimization Problems

**arXiv 2025:2505.08735；路线：袁春；当前优先级：P1。** [原始来源](https://arxiv.org/abs/2505.08735)<!--ref:P44--><!--anchor:section:Abstract-->

- 原文核心：利用组合优化解之间的优劣关系进行学习，结合搜索产生训练信号。
- 本项目迁移假说：同状态分支回放的ATT差可构成可核验排序标签，用于诊断或小型残差监督，而不是向LLM索要偏好。
- 风险／不照搬部分：生成标签的额外计算要计费；用全未来离线最优产生的标签不能冒充在线算法可用信息。
- 阅读层级：摘要/问题结构。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P45｜A Bi-Level Network-Wide Cooperative Driving Approach Including Deep Reinforcement Learning-Based Routing

**IEEE TIV 9(1):1243–1259，2024；DOI在线2023；路线：李力；当前优先级：P1。** [原始来源](https://doi.org/10.1109/TIV.2023.3305818)<!--ref:P45--><!--anchor:section:Abstract-->

- 原文核心：正式题名连接网络级路由与双层协同驾驶。
- 本项目迁移假说：保留为交通系统中“高层路由—低层执行协同”叙事的待精读参考。
- 风险／不照搬部分：当前只交叉核验作者主页和书目信息，不把未获全文的方法细节作为确定的实现依据。
- 阅读层级：作者清单/书目；全文待取。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P1→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P46｜Scalable UAV Multi-Hop Networking via Multi-Agent Reinforcement Learning with Large Language Models

**arXiv 2025:2505.08448，2026年修订；路线：陈鑫磊；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2505.08448)<!--ref:P46--><!--anchor:section:Abstract-->

- 原文核心：把无人机网络的多智能体分组、局部职责与高层指导结合。
- 本项目迁移假说：可参考不同角色的交互分解，但当前先用明确数学结构，不引入LLM。
- 风险／不照搬部分：通信覆盖、吞吐和分组奖励不同于统一ATT；不能照搬奖励分解和行为约束后仍称同目标公平比较。
- 阅读层级：摘要/版本记录。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P47｜Query-Efficient Video Adversarial Attack with Stylized Logo on Service Computing

**arXiv 2024:2408.12099，2026修订；作者标注TDSC接收；路线：肖喜；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2408.12099)<!--ref:P47--><!--anchor:section:Abstract-->

- 原文核心：将强化学习用于有限查询预算下的参数选择，是AI安全中的RL应用。
- 本项目迁移假说：主要用于确认这一研究线确有近年RL工作；对UAM只保留预算有限的序列选择这一较远联系。
- 风险／不照搬部分：对抗视频与交通运营的目标完全不同，不作为当前S/AC算法候选；也不与同名美国博士生混淆。
- 阅读层级：作者预印本摘要/版本记录。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P2；保留追踪，当前不占主预算。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### P48｜A Two-Stage Reinforcement Learning Approach for Robot Navigation in Long-range Indoor Dense Crowd Environments

**IEEE IROS 2024:5489–5496；路线：曾龙、张涛合作；当前优先级：P1。** [原始来源](https://ieeexplore.ieee.org/document/10801711/)<!--ref:P48--><!--anchor:section:Abstract-->

- 原文核心：作者主页与DBLP确认在复杂机器人导航中使用两阶段RL。
- 本项目迁移假说：作为机器人分层决策方向的储备研究线，暂不直接引入UAM。
- 风险／不照搬部分：IEEE正文受限，当前只核对题名、作者和会议；不能凭“两阶段”题名推断具体策略结构。
- 阅读层级：作者官网＋DBLP；全文待取。
- 源码：未核验可用官方实现；未核验官方代码。
- 重评：原优先级P2→本轮P1；条件适配／诊断支持。
- 材料来源：复用上一版书目及阅读材料，按本轮QA重新评价；非本轮全部重新全文阅读。

### N01｜DyPS: Dynamic Parameter Sharing in Multi-Agent Reinforcement Learning for Spatio-Temporal Resource Allocation

**KDD 2024；DOI:10.1145/3637528.3672052；路线：李勇／电子系FIB相关线；当前优先级：P0。** [原始来源](https://fi.ee.tsinghua.edu.cn/public/publications/0261cfca-f04c-11f0-b382-2aaffa21d846.pdf)<!--ref:N01--><!--anchor:section:Abstract-->

- 原文核心：从行为轨迹提取角色特征，动态调整同角色智能体的参数共享。
- 本项目迁移假说：研究哪些背景应共享、哪些候选需要条件残差，不必增加机场级agent。
- 风险／不照搬部分：参数共享不等于候选时间对齐；动态角色、自监督和更新成本是额外因素。
- 阅读层级：原稿摘要、方法结构与实验设定；非逐项证明审查。
- 源码：[作者仓库](https://github.com/tsinghua-fib-lab/DyPS)；已读README、run入口片段；核心algo获取超时；未运行。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N02｜CoopRide: Cooperate All Grids in City-Scale Ride-Hailing Dispatching with Multi-Agent Reinforcement Learning

**KDD 2025；DOI:10.1145/3690624.3709205；路线：李勇／电子系FIB相关线；当前优先级：P0。** [原始来源](https://fi.ee.tsinghua.edu.cn/public/publications/785c48e0-f1d2-11f0-b382-2aaffa21d846.pdf)<!--ref:N02--><!--anchor:section:Abstract-->

- 原文核心：结合图式协作状态、统一动作表达与合作强度学习。
- 本项目迁移假说：让需求、容量及实际候选执行关系共同进入表示，作为控制—方法耦合参照。
- 风险／不照搬部分：元梯度合作奖励不可偷偷移入固定ATT比较；网格不是机场的同构复制。
- 阅读层级：原文§3.2—3.4与问题设定；未复算实验。
- 源码：[作者仓库](https://github.com/tsinghua-fib-lab/CoopRide)；已读README、run_CoTa配置／奖励片段；核心algo获取失败；未运行。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N03｜Reward-Preserving Counterfactual State Editing for Offline Reinforcement Learning

**ICML 2026；PMLR 306:125273–125296；路线：外部相关工作；Yong Li单位未独立核完；当前优先级：P1。** [原始来源](https://proceedings.mlr.press/v306/wang26s.html)<!--ref:N03--><!--anchor:section:Abstract-->

- 原文核心：以奖励保持约束组织反事实状态编辑，缓解离线序列决策混杂。
- 本项目迁移假说：构造物理合法、表面相似而候选价值不同的状态对，检验表示混叠。
- 风险／不照搬部分：即时奖励保持不保证长期价值保持；不随机编辑出不可能状态，不直接认领为清华李勇成果。
- 阅读层级：正式书目／摘要；PDF下载超时，未读全文。
- 源码：未核验可用官方实现；未核验官方代码。
- 教师归属：尚未完成PDF单位消歧，不计为某导师已确认个人成果。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N04｜Generative Adaptation of Dynamics to Environmental Shifts via Weight-space Diffusion

**ICML 2026；PMLR 306:70643–70670；路线：王寰东／李勇合作线；廖庆民相关合著；当前优先级：P1。** [原始来源](https://proceedings.mlr.press/v306/li26gl.html)<!--ref:N04--><!--anchor:section:Abstract-->

- 原文核心：DynaDiff在权重空间适配环境变化，并约束功能行为一致。
- 本项目迁移假说：只有确有动态模型失配时考虑小模型校准或决策相关预测。
- 风险／不照搬部分：不应用大扩散模型预测已知飞行／充电；环境变化和方法变化分开计。
- 阅读层级：PMLR书目、摘要；非全文。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N05｜The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models

**arXiv 2025:2505.22617；路线：丁宁及合作者；当前优先级：P1。** [原始来源](https://arxiv.org/abs/2505.22617)<!--ref:N05--><!--anchor:section:Abstract-->

- 原文核心：分析推理模型RL中的策略熵演化及优势与log-prob的统计关系。
- 本项目迁移假说：同时测量熵、有效选择机会与动作成本，检查过早确定化。
- 风险／不照搬部分：LLM经验关系不是UAM定律；低熵不自动表示塌缩，不直接改PPO冒充纯表示消融。
- 阅读层级：原稿摘要、结构及熵／协方差段落；未复现。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N06｜Policy Newton Algorithm in Reproducing Kernel Hilbert Space

**arXiv 2025:2506.01597；路线：丁文伯；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2506.01597)<!--ref:N06--><!--anchor:section:Abstract-->

- 原文核心：用三次正则与表示定理，将RKHS二阶策略更新转成有限维问题。
- 本项目迁移假说：作为跳出PPO的后备路线；先固定表示，再公平比较优化器。
- 风险／不照搬部分：维数随轨迹数据增长；不是SB3即插即用升级，离散适配和计算成本另验。
- 阅读层级：原PDF首页摘要与清华单位；未审查收敛证明。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N07｜FedTeddi: Temporal Drift and Divergence Aware Scheduling for Timely Federated Edge Learning

**arXiv 2025:2509.07342；路线：陈巍／周盛／牛志升合作线；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2509.07342)<!--ref:N07--><!--anchor:section:Abstract-->

- 原文核心：结合时序漂移与分布差异，安排联邦边缘学习资源。
- 本项目迁移假说：借鉴信息相关性随时间变化，而非等权处理全部事件。
- 风险／不照搬部分：模型漂移不等于飞机生效时间；不把交通目标改成AoI。
- 阅读层级：原站作者、摘要、投稿说明；非全文。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N08｜Electric Vehicles Management in Distribution Network: A Data-Efficient Bi-Level Safe Deep Reinforcement Learning Method

**IEEE TPWRS 40(1)，2025；早期在线2024；文献号10509804；路线：许银亮合作线；当前优先级：P1。** [原始来源](https://ieeexplore.ieee.org/abstract/document/10509804)<!--ref:N08--><!--anchor:section:Abstract-->

- 原文核心：面向EV管理的数据高效双层安全DRL；具体安全构造尚未核完。
- 本项目迁移假说：可计算的可行性留在执行层，学习高层选择。
- 风险／不照搬部分：主要核实出版社索引与作者单位；不能认领其安全保证或直接移植连续电网动作。
- 阅读层级：出版社索引／作者单位；正文触发验证。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N09｜Objective-driven charging control and learning-based incentive pricing for demand response of electric vehicles

**Energy 338，2025，138804；路线：许银亮合作线；当前优先级：P1。** [原始来源](https://www.sciencedirect.com/science/article/pii/S0360544225044469)<!--ref:N09--><!--anchor:section:Abstract-->

- 原文核心：结合充电控制、学习式激励及优化分解。
- 本项目迁移假说：学习不确定部分，已知约束与执行分配仍计算。
- 风险／不照搬部分：定价不是固定ATT选站；不能引入新的激励奖励。
- 阅读层级：出版社可检索摘要／结论片段；直接页403，非全文。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N10｜A double deep reinforcement learning-based adaptive framework for decision-optimal wind power interval prediction

**Energy 329，2025，136661；路线：许银亮／郭烨相关能源线；当前优先级：P1。** [原始来源](https://www.sciencedirect.com/science/article/pii/S0360544225023035)<!--ref:N10--><!--anchor:section:Abstract-->

- 原文核心：让区间预测服务于后续决策，而非只追求预测MSE。
- 本项目迁移假说：若需要小WM，用候选排序与决策损失评价其价值。
- 风险／不照搬部分：风电不是UAM；预测改善不自动证明ATT改善。
- 阅读层级：出版社摘要／结论片段；直接页403，非完整方法。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N11｜A novel operation method for renewable building by combining distributed DC energy system and deep reinforcement learning

**Applied Energy 353，2024，122188；在线2023；路线：建筑能源相关线；Yi Zhang同名仍需补核；当前优先级：P2。** [原始来源](https://www.sciencedirect.com/science/article/pii/S0306261923015520)<!--ref:N11--><!--anchor:section:Abstract-->

- 原文核心：将建筑分布式直流能源系统与DRL控制结合。
- 本项目迁移假说：说明物理与学习分工值得关注，全院搜索不应限电子信息。
- 风险／不照搬部分：具体Yi Zhang身份未完成PDF消歧；不直接认领为某位老师的已确认成果。
- 阅读层级：出版社书目／摘要；非全文。
- 源码：未核验可用官方实现；未核验官方代码。
- 教师归属：尚未完成PDF单位消歧，不计为某导师已确认个人成果。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N12｜Model predictive control of fluid–structure interaction via Koopman-based reduced-order model

**Journal of Fluid Mechanics 1026，2026，A32；路线：曹顺翔；当前优先级：P2。** [原始来源](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/model-predictive-control-of-fluidstructure-interaction-via-koopmanbased-reducedorder-model/2829DE4F0B9D0D6539DD1C2461B6112B)<!--ref:N12--><!--anchor:section:Abstract-->

- 原文核心：用Koopman降阶模型与MPC控制流固耦合系统。
- 本项目迁移假说：作为可控复杂度模型／规划参照。
- 风险／不照搬部分：这是模型控制而非RL；UAM多数承诺动力学已有解析规则，不必重学。
- 阅读层级：出版社摘要、作者及SIGS海洋工程单位。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N13｜A Cable-Actuated Soft Manipulator for Dexterous Grasping Based on Deep Reinforcement Learning

**Advanced Intelligent Systems，2024；正文DOI:10.1002/aisy.202400112；路线：曲钧天；SIGS王学谦等合作者；当前优先级：P2。** [原始来源](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202400112)<!--ref:N13--><!--anchor:section:Abstract-->

- 原文核心：面向软机械臂的深度强化学习连续控制。
- 本项目迁移假说：作为时序与部分观测控制的邻域案例。
- 风险／不照搬部分：连续软体控制不等于离散机场决策；同题202470046封面不重复算方法论文。
- 阅读层级：出版社摘要、作者单位及正文／封面消歧。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N14｜Reinforcement Learning for Charging Optimization of Inhomogeneous Dicke Quantum Batteries

**arXiv 2025:2511.12176；路线：江奔奔／吴任冰等合作；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2511.12176)<!--ref:N14--><!--anchor:section:Abstract-->

- 原文核心：比较不同观测条件下的量子电池充电策略。
- 本项目迁移假说：仅作观测价值需分条件测量的邻域案例。
- 风险／不照搬部分：量子电池不是eVTOL电池；不因标题charging就接入主训练。
- 阅读层级：原站作者／摘要。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N15｜Parallel Bayesian optimization using satisficing Thompson sampling for fast charging design of lithium-ion batteries

**Engineering Applications of Artificial Intelligence，2025；10.1016/j.engappai.2025.110868；路线：江奔奔；当前优先级：P2。** [原始来源](https://www.sciencedirect.com/science/article/pii/S0952197625008681)<!--ref:N15--><!--anchor:section:Abstract-->

- 原文核心：用满足性并行贝叶斯优化减少充电设计的信息采集成本。
- 本项目迁移假说：借鉴预算分配／满足性门槛组织实验漏斗。
- 风险／不照搬部分：不是RL算法；不能反复查看最终测试自适应挑赢家。
- 阅读层级：出版社摘要／书目。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N16｜Human-in-the-loop Online Rejection Sampling for Robotic Manipulation

**arXiv 2025:2510.26406；路线：唐彦嵩合作线；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2510.26406)<!--ref:N16--><!--anchor:section:Abstract-->

- 原文核心：针对在线微调的价值误差和稀疏监督，采用人参与的拒绝采样。
- 本项目迁移假说：用合法规则／有限搜索标签诊断坏动作与数据质量。
- 风险／不照搬部分：人类或专家标签是额外信息与成本；不与纯在线同预算成绩混写。
- 阅读层级：原站作者／摘要；非全文。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N17｜Multi-Agent Reinforcement Learning with Reward Delays

**L4DC 2023；PMLR 211:692–704【窗口外基础】；路线：谷源涛；当前优先级：基础。** [原始来源](https://proceedings.mlr.press/v211/zhang23c.html)<!--ref:N17--><!--anchor:section:Abstract-->

- 原文核心：处理不同智能体和时刻的奖励到达延迟。
- 本项目迁移假说：区分奖励反馈、观测和动作物理生效三种延迟。
- 风险／不照搬部分：不是2024—2026新成果，不直接证明service-ready错位。
- 阅读层级：PMLR书目／摘要与作者单位。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

### N18｜Absolute Zero: Reinforced Self-play Reasoning with Zero Data

**arXiv 2025:2505.03335；路线：黄高合作线；当前优先级：P2。** [原始来源](https://arxiv.org/abs/2505.03335)<!--ref:N18--><!--anchor:section:Abstract-->

- 原文核心：通过可验证任务与自博弈组织推理学习。
- 本项目迁移假说：研究流程可借鉴自动提出／验证候选，不引入主方法。
- 风险／不照搬部分：推理任务生成不等于固定需求UAM控制；硬接会改变训练分布。
- 阅读层级：原站作者／摘要；非全文。
- 源码：未核验可用官方实现；未核验官方代码。
- 材料来源：本轮扩展检索；访问／阅读层级见卡片。

## 10．审计与披露

保留逐人记录、书目JSON、阶段锁与检索输出索引；原始整页缓存留在本地复核，不随分享包整批分发。教师材料按获取与阅读层级标注，没有模拟独立多模型复核，也未将任何条目标成用户本人已读。文献事实、迁移假说与建议分别列出。


## 方法与软件披露

学术研究技能用于分离来源事实、迁移假说与建议；实验设计技能用于完整S×AC、配对区块、预先固定比较与筛选／确认边界。由同一AI辅助整理，未执行独立多模型审查或实证验证，没有把任何教师材料标为用户已读。技能参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065)；本轮核对原站最新为v2，参考链接不固定版本后缀。该软件参考不计入66条项目文献卡。
