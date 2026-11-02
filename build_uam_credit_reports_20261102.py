#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用已经核验的派生JSON生成三份中文报告及两个独立交付包。"""
import hashlib, html, json, shutil, zipfile
from pathlib import Path
import markdown

PROJECT=Path(__file__).resolve().parent
DATA=PROJECT.parents[1]/'artifacts/uam-credit-analysis/20261102_v1'
OUT=PROJECT/'analysis_reports/uam_credit_results_20261102'
OUT.mkdir(parents=True,exist_ok=True)

def read(name):return json.loads((DATA/name).read_text(encoding='utf8'))
T=read('training_analysis.json');D=read('diagnosis_analysis.json');RULE=read('matched_rule_baselines.json')

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'

def f(x,d=2):return f'{x:.{d}f}'
def pct(x):return f'{100*x:.1f}%'
def control(x):return '联合J' if x=='J3' else '单侧S' if x=='LQ' else '强规则'
def net(x):return 'LSTM' if x=='LSTM' else 'Transformer'

SOFTWARE='''
### 分析流程说明

本报告按统计分析、实验设计与学术研究验证流程区分证据、推测和建议，按训练种子聚合，未将多次回放当独立训练样本。图中显示真实种子和完整曲线，不以最低检查点选择赢家。使用现有Python 3.11环境与NumPy/SciPy，不安装技能示例要求的另一套Python 3.12环境；算法和实际软件版本另存清单。未加载模型、未运行新回放、未连接服务器，新增训练0步。

流程工具参考：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065)。该文仅用于说明分析流程，不是UAM机制或方法有效性的证据。
'''

FALLACIES=table(['已检查的解释风险','本轮处理'],[
('总体与分组反转','分单侧/联合、网络、gamma列出，不用总平均遮盖反转'),
('从组均值推个体','逐训练种子、逐回放保留；不把整体均值说成每名乘客都改善'),
('筛选人群偏差','模型按登记清单全取，不只取低分/坏种子'),
('碰撞变量偏差','不只选全完成场；完成率和成本并列保留'),
('忽略基率','给出总回放数、全完成数、尾批和其他未完成数量'),
('均值回归','不以选中最坏模型后的回升证明修复'),
('幸存者偏差','未完成ATT保持空值，不用已完成人均ATT冒充全体'),
('多重寻找效应','12项配对差全部报告并作保守Holm校正'),
('分析路径自由度','新增探索明确标注；原预注册口径与敏感性口径分开'),
('相关当因果','局部候选收益、跨模型差异、整场干预三类证据分开'),
('反向因果','事件时间顺序记录不足时不将相关性认作时序致因')])


def diagnostic_report():
    main=D['validation_groups'];ranks=[g for g in D['rank_groups'] if g['horizon']==240 and g['continuation']=='BASE']
    rank_table=table(['控制/模型','局部最好可减少人时','.99与1排序不同','60与240分钟排序不同','删去后续请求与真回放排序不同'],[
        (control(g['control'])+' '+g['arm'],f(g['local_gain']['mean']),pct(g['post_discount_disjoint']['mean']),pct(g['horizon60_disjoint']['mean']),pct(g['committed_disjoint']['mean'])) for g in ranks])
    return f'''# 报告一：无训练诊断结果——有局部空间，但交通共同缺口尚未完全立住

2026-11-02｜先看本页前两节即可掌握重点；后面是数据和边界。

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent；统计与图形辅助流程。
- Origin Mode：validate；Origin Date：2026-11-02。
- Verification Status：ANALYZED；Version Label：credit_diag_analysis_v1。
- 对象：旧50M中40个K3模型，种子1701–1704，加1个强规则。**不是刚训练完的1801–1804模型。**
- 来源：本地IMPORT_FIXED_DATAONLY归档，原包不修改；未重新仿真。

## 先用白话说结论

1. **普通模型还没做到最好。** 统一事后处理尾批后，再交换两名较早乘客的站点选择、保持每站总人数不变，仍能改善部分整场结果。这说明问题不只是“最后几个人凑不满4座”。
2. **但不能据此说找到了强解析也普遍解决不了的交通问题。** 强READY规则在同样12次随机交换中没有进一步改善，处理尾批后仍比这些普通模型强。
3. **局部看，强READY也会选得不够好。** 18个抽样状态中，穷举候选的240分钟成本，平均还可省89.56人·分钟。这只是局部窗口的可改空间，尚未证明整场也能省这么多。
4. **“gamma太短视”不是主要解释。** .99和1的排序有变化，但换120/240分钟窗口、或删掉未来请求带来的排序变化通常更大；新的训练实验也没有从gamma=1得到普遍收益。
5. 下一步最该补的是：**在最新普通模型和强规则上，把局部更优候选真的接回整场，确认收益是否存在、能否重复，以及主要发生在谁的等待上。**

## 这次到底跑了多大

- 41个控制器：40个训练模型＋1个READY。READY不按种子复制成4个独立模型。
- 3条顺序：original、4301、4302；原OD组合和到达时刻不变，仅重排请求顺序。不是3个新城市，也不是全新需求分布。
- 123条原始回放；2058次整场仿真调用；7068次候选分支调用；450个抽样决策状态；2646条“状态×后续策略×窗口”记录。
- 成功66/66工作单元；权重未更新；450个状态的参考克隆校验均通过。包里还保留66份旧FAILED记录，这是修复前的历史，**不能据此把修复后的成功结果判为失败**。
- 修复后本次实际墙钟1744.26秒，即**29分04秒**；两个CPU工作进程的耗时之和3467.57秒。配置上限一小时，不代表一定用满一小时。COUNTERS里的1459秒是部分内部计时，不能当整套时长。

## 1．先隔离尾批，再看正常阶段还有没有空间

先解释表里的名字：L00＝普通LSTM，不加新描述符；L10_TIME＝只加合法时间描述符；L01_BATCH＝只加“离凑满4座还差多少”的描述符；L11_BOTH＝两者都有；T11_BOTH＝同样信息，历史网络改成普通Transformer。这里BATCH是乘客凑批，不是PPO训练的小批量。READY是不用训练、用合法队列/服务就绪信息估计候选服务代价的在线强规则。

“统一尾批”是事后把最后零星乘客重新分配后，从头重放，并非运行中发现卡住就删除乘客；有真实未来特权，只用于诊断。交换只取前240名乘客，最后60名不动，保持各站人数不变；飞机策略仍会对新状态作反应，因此也不能叫“飞机轨迹固定的纯乘客效应”。

原诊断汇总使用3条顺序；为避免训练原顺序混入，本报告主表另列**只用4301、4302两条验证顺序**的分层敏感性分析，不冒充新增独立确认。先在每个训练模型内平均两次，再平均4个训练种子。original在完整补充表保留。单位是固定601分钟截止下的人时/生成乘客，低更好。脚本名义600分钟，实际推进601个一分钟步；源代码以航空到达为终点，未建模的目的地末段地面时间不补造。

{table(['控制/模型','原回放','统一尾批后','再交换后','交换额外改善'],[(control(g['control'])+' '+g['arm'],f(g['base_rpt']['mean']),f(g['tail_rpt']['mean']),f(g['swap_rpt']['mean']),f(g['swap_gain']['mean'])) for g in main])}

例如联合L11从60.87降到56.64，主要是尾批处理；再从56.64降到55.20，才是本次有限顺序交换额外找到的1.45分钟/人。强规则为54.82→51.98→51.98。**不能把前面4.23分钟全算成新时序方法可获得的收益。**

所有统一尾批与交换后样本均全体完成。交换的最好结果允许保持原样，因此改善非负部分来自搜索定义；“4个种子都非负”不能直接当统计显著性。12次未找到改进也不能证明没有改进。

## 2．候选排序告诉我们什么

这里“排序不同”指两套成本的最优动作集合没有交集，平局不强行算错。下表使用3条顺序，模型每臂4种子，每模型18个状态；READY只有18个状态。成本是**人·分钟/一个局部决策**，不是分钟/人，不能与上一表直接相加。

{rank_table}

最值得保留的现象：同一状态下，短窗口认为好的动作，长窗口可能不再好。但窗口变化、删请求、不同后续策略是不同干预，不能把它们排成可相加的“因果占比”。

### 不要把“只模拟已承诺”直接当作合法预测器

COMMITTED的实现是删掉尚未出现的请求，而不是从合法需求模型预测后续。满4座的环境里，删掉请求可能人为阻断后续凑批。因此它与真实回放排序不同，既可能因为未来需求重要，也可能因为这个近似世界的尾批结构不对。

联合L11在240分钟、沿原策略继续时，按COMMITTED选动作，在真实回放下平均反而**增加32.61人·分钟**；普通Transformer对应增加16.11。不能直接把这种标签喂给新模型。真实未来版本只作Oracle（事后知道未来的诊断参照），不能进入在线策略输入。

## 3．这是不是“联合协作”问题

在联合L11的240分钟窗口，允许改乘客侧时平均可省218.89人·分钟，只改飞机侧31.11，同时改两侧相对最好单侧的额外收益8.33。联合Transformer分别为160.61、37.89、8.67。

这些数**不是可相加归因百分比**，也不排除飞机状态限制了乘客选择。但当前证据不支持“主要损失必然来自多智能体信用分配”。若现在直接换复杂MARL（多智能体强化学习），仍有再次抽模型的风险。

“人·分钟”表示人数乘等待时间，例如10人各少等1分钟就是10人·分钟；不能读成“每人少等10分钟”。

## 4．关于时序，目前哪些说法不能写成结论

- 后续未出现乘客常贡献较大的局部成本差，但他们的人数和暴露时间也更多。未经人数/等待暴露量归一化，不能说“未来需求解释了90%的问题”。
- PRE/POST是每分钟更新前/更新后计人数。240分钟的无折扣最优集合，本轮主表中几乎不因这一步变化而改变。它不是当前最大的排序来源。
- `after_train_cutoff500`只能说明局部窗口有一部分延伸到500分钟以后。它没有直接提供“去掉截断会带来多少整场改善”的因果答案。
- 本轮没有直接检验“STIN把不同候选的ETA压平，进而造成ATT损失”。这是另一个需要观测和同状态干预验证的假说，不因时序看起来重要就自动成立。

## 5．下一步保留与暂缓

保留：正常阶段可重复损失、决策顺序与承诺窗口、强规则也存在的局部空间，以及单侧/联合同权限对照。

暂缓：gamma=1作为主要修复；COMMITTED标签直接监督；“PPO/Transformer原理上学不会”的叙事；为找好结果改座位、充电或机场数；直接上WM（世界模型）或MARL。它们是后续工具，不是现在已确定的答案。

## 附录：包括original的完整敏感性表

{table(['控制/模型','原回放','统一尾批','再交换','四种子交换增益'],[(control(g['control'])+' '+g['arm'],f(g['base_rpt']['mean']),f(g['tail_rpt']['mean']),f(g['swap_rpt']['mean']),' / '.join(f(x) for x in g['swap_gain']['values'])) for g in D['groups']])}

![有限交换与尾批处理的分阶段对照](evidence/diagnosis_feasible_improvement.png)

该图使用3条顺序；主表只用4301/4302，二者口径已分开。强规则不是数学最优。参考记录见`diagnosis_episodes.json`和`diagnosis_branch_windows.json`，本报告不由局部窗口收益推算整场ATT。

## 解释风险检查

11/11类已检查，详见下表。整体证据边界为CAUTION：有限候选、有限状态、已有验证顺序，缺少最新模型上的全程验证，不支持普适因果或投稿结论。

{FALLACIES}
{SOFTWARE}
'''


def training_report():
    groups=T['groups'];ready=next(r['rpt'] for r in RULE['baseline_rows'] if r['rule']=='READY_QUEUE_SINGLE')
    score=table(['控制','普通网络','γ','后期成本均值±种子SD','与强READY差','全完成场比例','最终存档成本'],[
        (control(g['control']),net(g['network']),str(g['gamma']),f(g['rpt']['mean'])+' ± '+f(g['rpt']['sd']),f(g['rpt']['mean']-ready),pct(g['full_fraction']['mean']),f(g['final']['mean'])) for g in groups])
    primary=T['contrasts'][:4]
    return f'''# 报告二：18小时训练结果——gamma=1没有修复问题，普通模型的能力还不能被否定

2026-11-02｜32个模型全部完成；实际76,677,120步。

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent；Origin Mode：validate。
- Origin Date：2026-11-02；Verification Status：ANALYZED；Version Label：credit_train_analysis_v1。
- 来源：32run DATAONLY，强规则取同物理、同验证银行的旧50M RULES_END记录；银行SHA一致。
- 主指标逐条从40次后期回放重新汇总，与32份RESULT逐模型一致；未重新训练或执行策略。

## 先用白话说重点

1. **“把未来看得更远就能解决”这条简单解释没有成立。** γ=1在四个分组的平均成本都更高；单侧LSTM四个种子全部变差，平均增加7.37分钟/人。
2. **普通PPO＋Transformer不能被当成已经排除的弱基线。** 联合Transformer、γ=.99平均57.64，是本轮最好组；其中一个种子54.79，已经低于强READY的56.16，但其他种子没稳定复现。
3. **联合不是必然更好或必然更坏。** γ=.99时，联合LSTM与单侧LSTM几乎相当；联合Transformer平均更好，但只有3/4种子同方向。
4. **模型并非一直乱飘。** 大部分完整曲线从很差的初期降到较集中的平台。更准确的问题是“增量收益依赖种子和配置”，而不是全部数据都没有规律。
5. **现在最需要检查的是这些最新模型具体错在哪。** 当前诊断分析的是旧170x模型，不能用它直接宣布最新180x模型也有相同的正常阶段缺陷。

## 缩写一次解释清楚

- S/单侧：学习乘客选哪个出发站，飞机按既定规则调度；这里不是旧文献中的时间模块S。
- J/Joint：乘客站点和飞机返回方向都进入联合控制；J3是项目控制结构名称，不等于完整MAPPO/HAPPO算法。
- LSTM：顺序读取历史的普通网络。Transformer：普通一层注意力历史网络；本轮不是特殊新结构。
- γ/gamma：对较远成本的折扣。0.99会逐渐降低远期权重，1不降低。本轮逐分钟奖励没变，**优化目标的时间权重确实变了**，所以事先登记为实验因素。
- RPT：截至固定时间，所有已生成但未完成人数的积分÷生成总人数，本文称“固定边界人时成本”。完成后等于相应全体旅时；没完成时不能称完整ATT。
- ATT：全体生成乘客的平均旅时。未全完成的回放保持空值，不用已完成者均值代替。

本轮脚本名义评估600分钟，实际为601个一分钟步。源代码终点是航空到达，未建模的目的地末段地面时间不补造，不冒称覆盖现实完整door-to-door全过程。

## 1．全部八组结果

每组4个独立训练种子1801–1804；每模型训练2,396,160步。主结果先在最后20%的10个固定存档×4条验证顺序内平均，再跨4个训练种子平均。SD表示这4个种子之间的差异，不是40次回放的标准误。

{score}

强规则READY为**56.1633**。源规则SPF/STTF/QTTI分别66.66/65.49/67.55，本轮有不少模型比它们好；但是不能因此回避更强的READY。READY本身也只有1/4验证回放全完成，所以它也不是完美系统或数学下界。

训练、奖励、信息都沿用新近的K3、12架飞机、4座满批E1；5环境×4096步、batch512、10 epochs。**不是10月8日旧40架机队/gamma1冻结协议**。所有组都有相同合法时间和凑批描述符，没有用真实未来，没有辅助损失。不同网络参数数目并非严格相等，详见逐模型清单；同种子同网络的两gamma组初始参数SHA均一致。

## 2．主问题：把γ改为1，究竟怎么样

下表正数表示γ=1更差；四个差值依次对应1801、1802、1803、1804。

{table(['控制/网络','γ1−γ.99均值','四个配对差','γ1更好的种子数','95% t参考区间'],[(c['label'].split(':')[0],f(c['mean']),' / '.join(f(x) for x in c['values']),str(c['wins_lower'])+'/4',f(c['ci95_t'][0])+' 至 '+f(c['ci95_t'][1])) for c in primary])}

这直接反驳了“只要不折扣，就应该普遍更好”的当前方案。但不证明远期后果不重要，也不证明γ=1永远不好。价值估计、GAE=.95、奖励归一化gamma固定.99、500分钟训练截断和更高回报方差仍可能影响学习；**目前没有数据把这些原因排好主次**。不建议下一步再扫一排gamma碰运气。

统计边界：每个比较只有4个训练种子。精确双侧符号翻转检验的最小p值只有.125，不能凭4/4就写“显著”。原预注册主比较是4项gamma配对，其原始p均不显著；附录再将含次要对比的12项统一做更保守的Holm校正，仍均不显著。t区间只是分布假设下的参考。正态筛查与留一种子敏感性均保存，不按筛查结果临时换检验。跨控制、跨网络不保证初始权重相同，仅gamma内部检查到同初始化。

## 3．普通Transformer和联合控制的含义

在γ=.99下，联合Transformer比联合LSTM平均低1.50分钟/人，3/4种子更好；单侧Transformer反而平均高0.55，只有1/4更好。**普通注意力网络不是处处都赢，也不是毫无用处。**

同为γ=.99，联合LSTM比单侧低0.18，2/4种子更好，几乎没有清楚优势；联合Transformer比单侧低2.23，3/4更好。γ=1时联合相对单侧差距更大，但单侧自己明显退步，不能把这个差距全部解读成“协作能力增强”。

联合Transformer、γ=.99的四个成绩为60.17、54.79、58.06、57.55。它的平均值仍比READY高1.48；一个种子胜出说明“普通方法存在可达性”，不足以证明稳定超越，也不足以宣布必须专门WM/MARL才能解决。

![八组逐种子成本，横线为组均值](evidence/training_seed_points.png)

## 4．完整曲线是不是还在乱飘

![32个独立模型的完整原顺序曲线，不平滑、不删初期高点](evidence/training_all_curves.png)

所有原始曲线保留。多数模型后期仍在改善，不能笼统说都发生后期崩溃；单侧LSTM、γ=1的均值平台较高，反而可能“稳定地差”。Best低于后期平均，也部分受从48个存档中取最低值的选择偏差影响。

{table(['控制/网络/γ','原顺序Best→后期回退均值','后期斜率（分钟/百万步）'],[(control(g['control'])+' '+net(g['network'])+' '+str(g['gamma']),pct(g['rollback']['mean']),f(g['slope']['mean'])) for g in groups])}

曲线图只用original、eval123，与主表4301–4304不同，**不拼成一条学习曲线**。新2.396M与旧1M使用不同全程学习率衰减和不同训练种子，不能用新中途存档对旧终点，直接归因“多训这些步就带来多少改善”。

## 5．完成率：大部分不是所谓联合永久卡死

1280次后期确认回放中，只有140次全完成，1140次未完成：

- **1111次**：未完成人全部在队列、各站不足4人、所有飞机IDLE，属于当前满座与有限动作集合下的尾批残留。
- **26次**：记录显示受限执行器阻塞，需要逐例查看队列、飞机位置和合法返回机会。
- **3次**：其他未完成状态，不凭总表判断永久卡死。

因此不能将89.1%的未完成场都叫“联合协调失败”。但尾批残留也不能忽略：它真实影响固定截止成本，并让全体ATT无法计算。最终模型128次确认回放仅20次全完成。

新训练包没有全体乘客同口径的ground/wait/flight状态占用积分；不能拿已完成乘客的分项均值替代它们，从而宣称增加的7.37分钟全部来自等待。**这一分解必须用下一轮无训练回放补上。**

## 6．耗时与模型是否保留

整套实际**16小时09分53秒**，在18小时预算内；纯训练合计15小时21分23秒，评估41分15秒，其余为保存/调度等。纯训练加权1387 FPS，整套含评估约1318 FPS。这比计划保守1200 FPS略快，不是任务少跑。

最终模型包已确认32份模型＋32份归一化文件一一配对。这里只验证文件配套，没有重新加载执行。原始三个压缩包、32条曲线、坏种子和所有未完成记录均保留。

## 附录A：32个种子成绩，不只给平均

{table(['控制/网络/γ','1801','1802','1803','1804'],[(control(g['control'])+' '+net(g['network'])+' '+str(g['gamma']),*[f(x) for x in g['rpt']['values']]) for g in groups])}

## 附录B：全部配对比较

{table(['对比（前者减后者）','均值差','更低/4种子','双侧精确p','Holm p（12项）'],[(c['label'],f(c['mean']),str(c['wins_lower']),f(c['exact_two_sided_signflip_p'],3),f(c['holm_p_family12'],3)) for c in T['contrasts']])}

## 解释风险检查

11/11类已检查。数据完整性通过、统计主指标重新汇总一致；“训练能否重复”“理论不可学”“交通文献共同缺口”均未被本次运行完整性检查证明。

{FALLACIES}
{SOFTWARE}
'''


def next_report():
    return '''# 报告三：下一轮无训练实验——先把局部改进接回整场，再决定是否值得训练

2026-11-02｜仅设计，尚未运行；不自动开始新训练。建议先做一小时上限的一组CPU诊断。

## Material Passport

- Origin Skill：academic-research-suite、experimental-design。
- Origin Mode：plan；Origin Date：2026-11-02。
- Verification Status：UNVERIFIED（尚未采集新实验结果）。
- Version Label：next_notrain_credit_bridge_v1。
- 输入：刚完成的32份2.396M最终模型及对应归一化；同源READY规则；原银行与新封存确认顺序。

## 一句话目的

**看清楚强普通模型和强规则还在哪些正常决策上真有损失；确认这种损失是不是能转成整场、可重复的改进。**

不再先假设“时间特征不够”“PPO太短视”“必须世界模型/多智能体”。先解决上两份报告暴露的证据断层：旧170x诊断与最新180x训练没有对接，局部窗口赢家也不一定是整场赢家。

## 先冻结四个问题，不许看完结果再换题

| 假说 | 哪个结果支持继续 | 哪个结果会推翻/降低优先级 |
| --- | --- | --- |
| H1：正常阶段存在共同交通空间 | 最新普通模型和READY都能在同权限下得到重复的整场改善 | 改善只出现在弱模型，READY整场几乎不变 |
| H2：真正需要考虑的是动作承诺顺序，而不只是站点总供需 | 每站总人数不变、最后60人不变，改变早中期承诺顺序仍降低非尾批成本 | 改善完全来自改总流量或尾批救援 |
| H3：短窗口局部赢家不能稳定接回整场 | 在相同状态同一后续策略下，60/120/240分钟与整场候选排序反复改变 | 窗口与整场高度一致，简单候选成本足够 |
| H4：决策触发和合法动作机会造成部分损失 | 坏决策集中在有/无有效调机机会、乘客到站前后等可复核事件条件 | 这些分组没差别，不再把它们当主要故事 |

这些是假说，不是本轮已经确认的论文创新。机场数、座位数、充电参数不变，也不把这些参数扫一遍当主任务。

## 0．设计原则与信息边界

- 冻结网络、归一化统计和优化器；新增训练0步。模型不调用learn、train或optimizer.step。
- 物理、需求、奖励、动作权限不改；单侧仍只决定乘客，飞机规则原样。READY也不能获得模型没有的额外调机权限。
- 先用original、4301、4302复核旧结果，再使用**代码与判定门槛冻结后才打开**的确认顺序。若先看过，改称验证，不冒充独立测试。
- 每个模型独立统计；回放、状态、分支是嵌套重复测量，不能伪增训练种子数。READY只有一个控制器。
- 真实未来可以给离线Oracle做“有没有空间”的诊断，但不能进入在线策略输入。删掉未来请求的COMMITTED仅作近似失效对照，不当正确教师。
- 预先留住所有原始、统一尾批、未完成、超时和失败记录。不只挑最坏模型/最坏状态。

## 第一阶段：把最新模型接上，先校准尾批和口径

**对象**：全部32个模型＋READY；原有3条顺序，共99条基准回放。

每条同时保存逐分钟的人时积分及ground、wait、flight、other四项，加和严格等于主成本；逐乘客事件保存出生、选择、到站、上机和完成时刻；飞机保存可用、被调、抵达、可服务时刻。记录模型输出、实际执行动作、无效/自动决定的飞机动作和未触发决策的区间。

对每条复用相同的事后尾批处理，从头回放，不删除乘客、不允许半座起飞、不赋予额外救援动作。原结果永远保留为主口径，尾批后结果只作隔离诊断。

对已证实的吸收状态（需求结束、所有飞机IDLE、所有剩余人在各站不足4人的队列、没有其他合法推进动作），可直接计算延长截止到900/1200分钟的成本敏感性：原积分＋未完成人数×延长分钟数。**这不是换主指标挑赢家，也不必让模拟器空转等待**；未满足全部条件的记录不得套用公式。

这一阶段能回答：最新γ=.99普通模型是否已经消除了大部分旧模型正常阶段差距；gamma=1的额外损失到底出现在哪个时间段、哪个环节。只能在有完整积分后解释，不能从已完成者均值反推。

## 第二阶段：相同状态、相同后续策略，比较窗口赢家与整场赢家

优先16个γ=.99模型（2网络×2控制×4种子）＋READY，共17个控制器。γ=1先只做第一阶段压力对照，不为它继续扩大搜索。

每个控制器×3条顺序，分别在前100、中100、后100名请求内固定随机抽2个决策，合计**306个状态**。不按失败分数抽样；额外反例只放单独案例库，不混进发生率统计。

对每个合法候选，从一模一样的克隆状态、相同外生条件回放到原601分钟截止。一遍回放同时读出60/120/240分钟和整场成本。固定两种后续策略：原策略、共同READY；二者分别比较自己的参考动作，不跨后续策略拼成本。READY本身只保留一个参考后续，避免重复计样本。

若联合最多3×3个候选、单侧3个候选，该部分REALIZED分支最多约3510次候选调用（包含参考候选），有效动作不足时更少；克隆校验等额外调用单独计数。

**核心输出**：

1. 局部赢家接到整场后是否仍更好；全程成本从同一前缀加上分支后缀得到，不把重叠局部收益加起来。
2. 代价加权排序错误：不是只数选错几次，而是这一次选错多花了多少人时。
3. 当前乘客、已出现的其他乘客、尚未出现乘客各自的受益/受损；额外报告每组人数和等待暴露量，避免“未来人多所以贡献大”的假象。
4. 乘客动作、飞机动作、两者一起改的局部额外收益；明确不把三项相加当100%归因。
5. 同样的合法ETA/ready特征是否对应不同的整场最佳候选；先检查已有特征能否区分，而不是看见时间就宣布STIN有问题。

## 第三阶段：专门检查“顺序”和“触发时机”，不扩成新方法大全

在第一阶段的尾批等价锚点上，做预先固定的早中期站点选择交换：每站总人数不变，最后60位不动。普通模型先12次；**对强READY重点补到最多96个预定方案**，其中一半为单次交换，一半为两对互不重叠的交换。有限预算允许不满额，但完整记录尝试数。

单次交换不能改善、两次协调交换能改善，只是“至少两步配合可能重要”的证据。需要看完整人时和成本分项，而不是只看它凑齐了最后一架飞机。飞机策略会正常反应，因此这仍不是把飞机动作完全冻结的纯顺序因果分解。

决策日志同时回答两个此前容易漏的问题：

- **决策本身的时间**：谁在什么时候获得决策机会？飞机就绪后是否必须等下一名乘客才触发调机？被忽略/自动执行的动作有多少？
- **动作带来的时间含义**：这名乘客在多久后到站，这架飞机多久后可服务，在中间窗口谁占用/释放了资源？成本差在当前、到站前后、首个批次还是后续循环出现？

本轮只记录、克隆和配对，不偷偷增加主动调度频率。若发现动作权限/触发合同缺陷，单列工程问题；之后即使修复，也给所有方法相同权限，不能冒充AI创新。

## 规模、时间和优先级

上一轮66单元、450状态、2058整场、7068候选调用、293万模拟分钟，在2个CPU工作进程下用了29分04秒。它只能作同运行时同机器的近似标尺，不能拿训练FPS直接换算诊断速度。

本轮优先级：**99条基准与尾批 → 306状态的整场分支 → READY加强交换 → 新确认顺序**。预计约45–75分钟；建议设置60分钟会话硬上限、最后5分钟不再开新单元。全程分支可能更贵，实际以首批单元计时更新准入，未跑完写PARTIAL，不把“预算到时”写成“实验完整”。

如果需要严格限制在一小时，宁可完整完成较少的预先分层单元，再续跑；不临时缩短某些模型窗口，不只留下最好的案例。所有控制器各至少覆盖一个完整种子块后，才增加确认样本。

新确认顺序不要作为事后“再看哪个顺序能赢”的工具。若预算不够，它保留为下一会话；不能拿只在已有430x上反复调出来的现象宣布泛化。

## 预先写好的后续路线

| 结束后的结果 | 下一步做什么 | 明确不做什么 |
| --- | --- | --- |
| 最新普通模型已接近READY，正常阶段改进很小 | 承认大部分是基线/训练问题，停止当前复杂表示路线 | 不声称PPO学不会，不立刻再换模型 |
| 模型能改，READY整场不能稳定改 | 作为学习实现差距，先做简单校准或蒸馏小验证；暂不立交通共同缺口 | 不把UAGMC缺点包装成当前UAM文献普遍缺点 |
| READY和最新模型都有正常阶段整场空间，且单步/短窗很难找出 | 才保留“承诺顺序/后续资源窗口”的候选交通问题，再查最近邻文献 | 不预设必须S/AC、WM或MARL |
| 差距主要由决策触发或合法执行器限制带来 | 统一接口合同、给所有对照相同权限，再核剩余空间 | 不把多给控制权的收益当网络创新 |
| 真实未来Oracle能改，合法信息版本不能 | 分开测需求不确定性与模拟近似误差；可考虑合法条件采样/决策相关预测 | 不拿真实测试未来训练或在线选动作 |
| 合法当前信息的简单代价已经能解释并改善 | 先实现这个简单控制器作强基线；若学习只省计算再研究蒸馏 | 不为新颖而换大模型 |
| 只有尾批能改，正常部分无重复收益 | 归为有限动作/收尾问题，当前题眼降级 | 不改座位或奖励造出新的大gap |

## 什么结果才值得再花几千万训练

以下是预先登记的项目推进门槛，不是显著性标准或事后保证：

1. 至少两个验证顺序、至少3/4训练种子，出现同类正常阶段可行改善；强READY也有整场收益，或者明确研究的是“普通学习怎样接近解析”的另一类问题。
2. 希望全程成本改善达到约**0.5分钟/人以上**，且不是靠降低完成率、只救尾批或改变动作权限。没有达到，也如实报告，不临时降低门槛宣布胜利。
3. 干预能解释损失发生的决策条件与时间段；未解释部分单列，不把局部不相加指标变成一张假的百分比归因饼图。
4. 有一个不偷看未来、可能被学习的输入→输出目标；简单规则/普通PPO＋Transformer不能已在相同信息和算力下轻松做到。注意这里只能排除**已运行的配置和预算**，不能证明算法原理上不能。
5. 设计只针对这一项未解决问题的少量训练对照，先短验证，再确认；不要重新开WM、MARL、S/AC、若干Transformer的一排候选。

如果问题主要是候选后续成本，WM可以只预测决策相关的有限后果，而非整个世界；如果只改任一侧都不行、联合改才有稳定额外收益，才有理由研究协作结构。二者都需要本轮证据触发，而不是现在就选定论文方法。

## 交付与执行边界

本轮交付的是下一步诊断设计，不包含已运行的新实验，也没有自动启动训练。后续实现仍为一份独立诊断入口，继承已修复的作者环境隔离：先生成封存作者源码，再导入；保留路径真实性检查。每个模型基准须与本次32run最终记录复现一致，克隆参考候选成本须与原轨迹一致，任一失败都不能继续把标签交给训练。保留冻结模型、原始指标、版本/hash、逐单元计时、断点恢复、失败隔离及DATAONLY包。用户自行上传服务器、看top/nvidia-smi后启动。
''' + SOFTWARE


STYLE='''body{margin:0;background:#f5f6f8;color:#192533;font-family:"Microsoft YaHei",Arial,sans-serif;line-height:1.85}main{max-width:1120px;margin:24px auto;background:white;padding:42px 50px}h1{font-size:28px;line-height:1.5}h2{margin-top:36px;font-size:22px;border-bottom:1px solid #ccd5df;padding-bottom:8px}h3{font-size:18px}p,li{font-size:16px}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:14px;width:100%}th,td{border:1px solid #cbd5e1;padding:8px 12px;text-align:left;white-space:normal}th{background:#e8eff6}tr:nth-child(even){background:#f5f8fb}img{max-width:100%;height:auto}a{color:#07588a}code{background:#eef2f6;padding:2px 4px;overflow-wrap:anywhere}nav{background:#edf3f8;padding:10px 16px}pre{white-space:pre-wrap}@media(max-width:700px){main{margin:0;padding:22px 16px}h1{font-size:24px}}'''

def publish(name,text):
    text=text.strip()+'\n'
    (OUT/(name+'.md')).write_text(text,encoding='utf8')
    body=markdown.markdown(text,extensions=['tables','fenced_code','toc'])
    title=html.escape(text.splitlines()[0].lstrip('# '))
    (OUT/(name+'.html')).write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+STYLE+'</style><main><nav><a href="index.html">返回三份报告目录</a></nav>'+body+'</main></html>',encoding='utf8')

def main():
    assert all(r['tail_full'] and r['swap_full'] for r in read('diagnosis_episodes.json'))
    names=['01_无训练诊断分析_局部空间与整场证据','02_十八小时训练分析_普通网络与折扣','03_下一轮无训练设计_把局部接回整场']
    for name,content in zip(names,[diagnostic_report(),training_report(),next_report()]):publish(name,content)
    evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
    public=['ANALYSIS_MANIFEST.json','training_analysis.json','training_models.json','training_curves.json','diagnosis_analysis.json',
        'matched_rule_baselines.json','statistical_assumptions.json','model_package_inventory.json','SOURCE_VALIDATION.json','training_seed_points.png','training_all_curves.png','diagnosis_feasible_improvement.png']
    for name in public:shutil.copy2(DATA/name,evidence/name)
    # 逐状态明细仅放本地报告包；GitHub只同步聚合证据，不上传模型或原始归档。
    packs=PROJECT.parents[1]/'artifacts/uam-credit-analysis/20261102_delivery';packs.mkdir(parents=True,exist_ok=True)
    specs=[('01_诊断分析报告包_20261102.zip',names[0],['diagnosis_analysis.json','diagnosis_episodes.json','diagnosis_branch_windows.json','diagnosis_feasible_improvement.png']),
           ('02_训练分析报告包_20261102.zip',names[1],['training_analysis.json','training_models.json','training_curves.json','training_failure_classes.json','matched_rule_baselines.json','statistical_assumptions.json','model_package_inventory.json','training_seed_points.png','training_all_curves.png'])]
    inventory=[]
    for pack,name,files in specs:
        with zipfile.ZipFile(packs/pack,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for ext in ('.html','.md'):z.write(OUT/(name+ext),name+ext)
            for file in files:z.write(DATA/file,'evidence/'+file)
            z.write(DATA/'ANALYSIS_MANIFEST.json','evidence/ANALYSIS_MANIFEST.json')
            z.write(DATA/'SOURCE_VALIDATION.json','evidence/SOURCE_VALIDATION.json')
            z.write(PROJECT/'analyze_uam_credit_results_20261102.py','analysis_source/analyze_uam_credit_results_20261102.py')
            z.writestr('index.html','<!doctype html><meta charset="utf-8"><h1>报告包</h1><p><a href="'+name+'.html">打开本包分析报告</a></p><p>证据和逐模型明细保存在evidence目录。</p>')
        with zipfile.ZipFile(packs/pack) as z:assert z.testzip() is None
        inventory.append(dict(name=pack,bytes=(packs/pack).stat().st_size,sha256=hashlib.sha256((packs/pack).read_bytes()).hexdigest(),zip_integrity='PASS'))
    text='# 本轮三份报告\n\n2026-11-02。诊断与训练分别交付，第三份仅为下一步无训练设计。\n\n'+''.join(f'- [{n}]({n}.html)\n' for n in names)+'\n原始压缩包位于本地项目experiment_archives，没有上传模型到GitHub。报告以固定边界人时成本为主，未完成全体ATT保持空值。\n'
    publish('README',text)
    shutil.copy2(OUT/'README.html',OUT/'index.html')
    (OUT/'DELIVERY_INVENTORY.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(reports=[str(OUT/(n+'.html')) for n in names],packs=[str(packs/r['name']) for r in inventory]),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
