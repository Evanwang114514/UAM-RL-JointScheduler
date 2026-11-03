"""将本目录Markdown机械渲染为可离线阅读的HTML；不运行实验。"""
import html
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def inline(text):
    text=html.escape(text)
    text=re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',text)

def render(text):
    lines=text.splitlines();out=[];toc=[];i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            out.append('<pre><code>'+html.escape('\n'.join(code))+'</code></pre>')
        elif re.match(r'^#{1,3} ',line):
            level=len(line.split(' ')[0]);title=line[level+1:];anchor='s'+str(len(toc))
            toc.append((level,title,anchor));out.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>')
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',c) for c in cells):rows.append(cells)
                i+=1
            out.append('<div class="table"><table>')
            for n,cells in enumerate(rows):
                tag='th' if n==0 else 'td';out.append('<tr>'+''.join(f'<{tag}>{inline(c)}</{tag}>' for c in cells)+'</tr>')
            out.append('</table></div>');i-=1
        elif re.match(r'^[-*] |^\d+\. ',line):out.append('<p class="item">'+inline(line)+'</p>')
        elif line.strip():out.append('<p>'+inline(line)+'</p>')
        i+=1
    nav=''.join(f'<a class="level{l}" href="#{a}">{html.escape(t)}</a>' for l,t,a in toc if l<=2)
    css='body{font:17px/1.85 system-ui,"Microsoft Yahei",sans-serif;color:#24333f;background:#f4f7f9;margin:0}main{max-width:1100px;margin:auto;background:white;padding:40px 48px}h1{font-size:30px}h2{font-size:24px;margin-top:45px;border-bottom:2px solid #b7cfd8}h3{font-size:20px}a{color:#12617b}nav{background:#edf4f7;padding:18px}nav a{display:block}nav .level1{font-weight:bold}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border:1px solid #cbd9de;padding:10px;vertical-align:top}th{background:#e7f1f5}tr:nth-child(even){background:#f6f9fb}.table{overflow:auto}pre{background:#172a38;color:#eef6fa;padding:18px;overflow:auto;line-height:1.6}code{font-family:Consolas,monospace}p.item{margin-left:12px}strong{color:#153d51}@media(max-width:700px){main{padding:20px}body{font-size:16px}}'
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UAM三组实验交付</title><style>'+css+'</style><main><nav>'+nav+'</nav>'+''.join(out)+'</main></html>'

if __name__=='__main__':
    for path in ROOT.glob('0[123]_*.md'):
        path.with_suffix('.html').write_text(render(path.read_text(encoding='utf-8')),encoding='utf-8')
        print(path.with_suffix('.html').name)
