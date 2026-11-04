"""核对报告核心数字与HTML结构；不运行策略、不导入机器学习库。"""
from pathlib import Path
from html.parser import HTMLParser
import json
import math
import statistics as st
from decimal import Decimal, ROUND_HALF_UP

ROOT = Path(__file__).resolve().parent
SUMMARY = json.loads((ROOT / 'analysis_summary.json').read_text(encoding='utf-8'))
AUDIT = json.loads((ROOT / 'INPUT_AUDIT.json').read_text(encoding='utf-8'))
REPORT = (ROOT / '01_三组实验合并分析与后续安排.md').read_text(encoding='utf-8')


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7), (actual, expected)


def group(family, arm, control=None):
    rows = SUMMARY[family]['groups']
    return next(r for r in rows if r['arm'] == arm and (control is None or r['control'] == control))


checks = []
assert len(SUMMARY['stage']['rows']) == 24
assert len(SUMMARY['space']['rows']) == 16
assert len(SUMMARY['diag']['jobs']) == 82
assert len(SUMMARY['diag']['states']) == 285
assert len(SUMMARY['diag']['candidates']) == 2445
assert SUMMARY['diag']['total_branch_calls'] == 8795
assert AUDIT['new_training_transitions'] == 0
assert all(r['tar_scan_finished'] and not r['unsafe_members'] and not r['duplicate_members']
           for r in AUDIT['inputs'].values())
checks.append('三组规模、只读边界与归档成员检查')

on = group('stage', 'STIN_ON', 'J3')
off = group('stage', 'STIN_OFF', 'J3')
close(on['confirm']['mean'] - off['confirm']['mean'], 7.003333333333332)
close(on['late']['mean'] - off['late']['mean'], 6.73939393939394)
assert all(a > b for a, b in zip(on['confirm']['values'], off['confirm']['values']))
assert all(a > b for a, b in zip(on['late']['values'], off['late']['values']))
close(on['confirm_completion']['mean'], off['confirm_completion']['mean'])
quality = SUMMARY['stage']['quality_per_model']
on_quality = sorted([r for r in quality if r['arm'] == 'STIN_ON'], key=lambda r: r['seed'])
off_quality = sorted([r for r in quality if r['arm'] == 'STIN_OFF'], key=lambda r: r['seed'])
close(st.mean(r['common_regret'] for r in on_quality), .9633333333333334)
close(st.mean(r['common_regret'] for r in off_quality), 1.7683333333333333)
assert all(a['common_regret'] < b['common_regret'] for a, b in zip(on_quality, off_quality))
assert len(SUMMARY['stage']['common_state_audit']) == 4
assert all(r['candidate_cost_inconsistencies'] == 0 for r in SUMMARY['stage']['common_state_audit'])
checks.append('STIN整段与共同状态方向反转')

base = group('space', 'S0_T0')
both = group('space', 'S1_T1')
close(both['confirm']['mean'] - base['confirm']['mean'], 4.5841666666666665)
assert all(a > b for a, b in zip(both['confirm']['values'], base['confirm']['values']))
close(base['confirm_dispatch_batches']['mean'], 71.5625)
close(both['confirm_dispatch_batches']['mean'], 56.3125)
close(both['confirm_dispatch_aircraft']['mean'], 73.375)
pairs = [r for r in SUMMARY['space']['individual_model_pairs']
         if r['scope'] == 'confirmation' and r['arm'] == 'S1_T1']
close(st.mean(r['gross_harm'] - r['gross_gain'] for r in pairs) / 300, 4.5841666666666665)
checks.append('空间等待四臂、批次变化及逐人人时守恒')

online = SUMMARY['diag']['online_models']
close(sum(r['agreements'] for r in online) / sum(r['decisions'] for r in online), 89 / 156)
for family, arm, expected in [('credit', 'T_G099', 1.0616666666666668), ('stage', 'JOINT', -.695)]:
    rows = [r for r in online if r['family'] == family and r['arm'] == arm and r['mode'] == 'space']
    close(st.mean(r['gain'] for r in rows), expected)
states = SUMMARY['diag']['states']
assert sum(r['before300'] for r in states) == 210
assert sum(r['before300'] and r['active'] for r in states) == 207
assert all(r['short_best_case_regret'] <= r['short_worst_case_regret'] + 1e-7 for r in states)
checks.append('合法在线修正、并列动作和正常运行覆盖')

phase = SUMMARY['diag']['baseline_phase_groups']
pon = next(r for r in phase if r['family'] == 'stage' and r['arm'] == 'STIN_ON')
poff = next(r for r in phase if r['family'] == 'stage' and r['arm'] == 'STIN_OFF')
close(pon['cost']['mean'] - poff['cost']['mean'], 6.923333333333333)
close(pon['wait']['mean'] - poff['wait']['mean'], 6.357916666666666)
checks.append('STIN成本环节分解')

assert 24 * 2437120 + 16 * 3010560 == 106659840
assert 8 * 1003520 == 8028160
assert 8 * 3010560 == 24084480
for label in ('D1', 'D2', 'D3', '截止成本', '不是PPO本身的错误率', '没有创建或启动', '8,028,160'):
    assert label in REPORT, label
assert 'STIN_ON-STIN_OFF' in (ROOT / 'ANALYSIS_PLAN.json').read_text(encoding='utf-8')

# 人工报告中的核心表格必须与复算数据按同样的小数位一致。
for family, arm, control in [('stage', 'STIN_ON', 'J3'), ('stage', 'STIN_OFF', 'J3'),
                            ('stage', 'CONDITIONAL', 'J3'), ('stage', 'SPLIT', 'J3'),
                            ('stage', 'JOINT', 'J3'), ('stage', 'JOINT', 'LQ'),
                            ('space', 'S0_T0', None), ('space', 'S1_T0', None),
                            ('space', 'S0_T1', None), ('space', 'S1_T1', None)]:
    row = group(family, arm, control)
    for key in ('late', 'confirm'):
        # 报告按十进制四舍五入，避免64.195的二进制浮点表示使末位误报。
        rounded = lambda value: str(Decimal(str(value)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
        expected = f"{rounded(row[key]['mean'])}±{rounded(row[key]['sd'])}"
        assert expected in REPORT, (family, arm, expected)
checks.append('正文核心数字及后续训练预算')


class Structure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links = set(), []
        self.tables = self.closed_tables = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            assert attrs['id'] not in self.ids
            self.ids.add(attrs['id'])
        if tag == 'a':
            self.links.append(attrs.get('href', ''))
        if tag == 'table':
            self.tables += 1

    def handle_endtag(self, tag):
        if tag == 'table':
            self.closed_tables += 1


page = (ROOT / '01_三组实验合并分析与后续安排.html').read_text(encoding='utf-8')
parser = Structure()
parser.feed(page)
assert parser.tables == parser.closed_tables and parser.tables >= 12
assert all(link[1:] in parser.ids for link in parser.links if link.startswith('#'))
assert '<html lang="zh-CN">' in page and '\ufffd' not in page
checks.append('HTML编码、表格和目录跳转结构')

result = dict(status='PASS', checks=checks, policy_rerun=False, new_training_transitions=0,
              scope='本地归档统计及文档一致性检查，不是策略复现实验')
(ROOT / 'REPORT_CHECK.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
