"""机械渲染本目录的一份合并报告；只使用标准库，不运行任何实验。"""
from pathlib import Path
import html
import re

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / '01_三组实验合并分析与后续安排.md'


def inline(value):
    value = html.escape(value)
    value = re.sub(r'`([^`]+)`', r'<code>\1</code>', value)
    value = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', value)
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', value)


def render(text):
    lines = text.splitlines()
    body, toc = [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            body.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>')
        elif re.match(r'^#{1,3} ', line):
            level = len(line.split(' ')[0])
            label = line[level + 1:]
            anchor = 'section-' + str(len(toc))
            toc.append((level, label, anchor))
            body.append(f'<h{level} id="{anchor}">{inline(label)}</h{level}>')
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+', c) for c in cells):
                    rows.append(cells)
                i += 1
            body.append('<div class="table-wrap"><table>')
            for row_index, cells in enumerate(rows):
                tag = 'th' if row_index == 0 else 'td'
                body.append('<tr>' + ''.join(f'<{tag}>{inline(c)}</{tag}>' for c in cells) + '</tr>')
            body.append('</table></div>')
            i -= 1
        elif line.strip():
            body.append('<p>' + inline(line) + '</p>')
        i += 1
    nav = ''.join(f'<a href="#{anchor}">{html.escape(label)}</a>'
                  for level, label, anchor in toc if level == 2)
    css = '''body{margin:0;background:#f3f6f8;color:#24343f;font:17px/1.85 system-ui,"Microsoft YaHei",sans-serif}
    main{max-width:1060px;margin:auto;padding:32px 44px;background:white}h1{font-size:30px;line-height:1.5}
    h2{font-size:24px;margin-top:42px;border-bottom:2px solid #bed2dd;padding-bottom:8px}h3{font-size:20px;margin-top:28px}
    p{margin:14px 0}strong{color:#144d63}a{color:#126485;overflow-wrap:anywhere}code{font-family:Consolas,monospace;font-size:.9em;overflow-wrap:anywhere}
    details{background:#edf4f7;padding:12px 18px;border-radius:8px}summary{cursor:pointer;font-weight:600}nav a{display:block;padding:3px 0}
    .table-wrap{overflow:auto;margin:18px 0}table{width:100%;border-collapse:collapse;font-size:15px;line-height:1.65}
    th,td{border:1px solid #cbd8df;padding:10px;vertical-align:top;overflow-wrap:anywhere}th{background:#e8f1f5}tr:nth-child(even){background:#f7fafb}
    pre{padding:16px;background:#18303e;color:#edf7fc;overflow:auto}.note{font-size:14px;color:#526978}
    @media(max-width:700px){main{padding:20px}body{font-size:16px}h1{font-size:25px}h2{font-size:22px}}
    @media print{body{background:white}main{padding:0}details{display:none}h2,h3{break-after:avoid}tr{break-inside:avoid}}
    '''
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>三组实验合并分析与后续安排</title><style>' + css + '</style></head><body><main>'
            '<p class="note">数据统计复算完成；未启动新训练或服务器任务。正文第一节是简明结论。</p>'
            '<details><summary>展开目录</summary><nav>' + nav + '</nav></details>'
            + ''.join(body) + '</main></body></html>')


if __name__ == '__main__':
    target = REPORT.with_suffix('.html')
    target.write_text(render(REPORT.read_text(encoding='utf-8')), encoding='utf-8')
    print(target)
