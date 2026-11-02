#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把已撰写的中文报告机械排版为HTML，并登记派生证据与代码hash；不执行模型。"""
import hashlib
import html
import json
import re
from pathlib import Path

import markdown

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / 'analysis_reports/uam_semantics_followup_20261102'
STYLE = '''
body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f3f6f8;color:#18303e;margin:0;line-height:1.8}
main{max-width:1120px;margin:30px auto;background:white;padding:38px 48px;border-radius:14px}
h1{font-size:30px;line-height:1.45}h2{font-size:23px;margin-top:42px;border-bottom:1px solid #d4dfe7;padding-bottom:8px}h3{margin-top:30px}
a{color:#126b8e}table{border-collapse:collapse;width:100%;display:block;overflow-x:auto;font-size:14px;margin:22px 0}
th,td{border:1px solid #d5e0e8;padding:10px 12px;min-width:90px;vertical-align:top}th{background:#e7f0f6}tr:nth-child(even){background:#f7fafc}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#102a38;color:#e4eef5;padding:18px;border-radius:8px;font-size:13px}
code{font-family:Consolas,monospace;overflow-wrap:anywhere}p,li{overflow-wrap:anywhere}nav{font-size:14px;padding-bottom:18px;border-bottom:1px solid #ddd}
@media(max-width:760px){main{padding:20px;margin:10px}h1{font-size:24px}}@media print{main{margin:0;padding:0}nav{display:none}table{display:table;font-size:11px}}
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rendered = []
    for path in sorted(OUT.glob('*.md')):
        text = path.read_text(encoding='utf-8')
        # 对应的报告都有HTML版本；证据/代码链接维持原后缀。
        text = re.sub(r'\]\(([^)]+)\.md\)', r'](\1.html)', text)
        content = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'])
        title = text.splitlines()[0].lstrip('# ')
        target = path.with_suffix('.html')
        page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        page += '<title>' + html.escape(title) + '</title><style>' + STYLE + '</style><main>'
        page += '<nav><a href="index.html">交付目录</a> · 2026-11-03 · 已有结果复算／新训练待运行</nav>' + content + '</main></html>'
        target.write_text(page, encoding='utf-8'); rendered.append(str(target.relative_to(PROJECT)))
    (OUT / 'index.html').write_bytes((OUT / 'README.html').read_bytes())
    files = [PROJECT / 'train_uam_stage_credit_18h_ONEFILE.py', PROJECT / 'analyze_uam_semantics_followup_20261102.py',
             PROJECT / 'tests/test_stage_credit_contract.py'] + [p for p in OUT.rglob('*') if p.is_file() and p.name != 'DELIVERY_MANIFEST.json']
    manifest = dict(date='2026-11-03', status='ANALYZED_AND_CODE_READY_NOT_TRAINED',
        training_plan=dict(cells=24, transitions_per_cell=2437120, rollouts_per_cell=119,
                           planned_actual_transitions=58490880, measured_work_cap=60000000,
                           wall_hour_cap=18, expected_hours='14–16，尚未实测新逻辑'),
        implementation_validation=dict(pure_contract_tests=10, passed=10, local_torch_imports=0,
                                       local_model_loads=0, local_replays=0, new_training_transitions=0,
                                       warning='纯检查不等于已验证GPU训练；正式服务端循环有内置断言'),
        raw_archives=['uam_decision_semantics_20261102_v1_DATAONLY.tar.gz',
                      'uam_credit_horizon_32run_18h_v1_DATAONLY.tar.gz',
                      'uam_credit_horizon_32run_18h_v1_FINAL_REPLAY_MODELS.tar.gz'],
        files=[dict(path=p.relative_to(PROJECT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
    (OUT / 'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(rendered=rendered, manifest_files=len(files), code_sha256=sha(files[0])), ensure_ascii=False))


if __name__ == '__main__':
    main()
