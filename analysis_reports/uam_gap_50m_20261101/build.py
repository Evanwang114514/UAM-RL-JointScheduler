"""机械封装入口和已核对的30M运行时；不执行任何训练。"""
from pathlib import Path
import ast,base64,zlib,hashlib,subprocess

here=Path(__file__).resolve().parent
project=Path(r'E:/Study Files/github/UAM-predict/UAGMC-main')
source=project/'train_uam_E1_reward_charge_attribution_30m_ONEFILE.py'
raw=source.read_bytes()
# 只用已经入库的真实入口；本地若改变，不能偷偷换运行时。
main=subprocess.run(['git','-C',str(project),'show','origin/main:train_uam_E1_reward_charge_attribution_30m_ONEFILE.py'],capture_output=True,check=True).stdout
if raw!=main:
    # Windows换行不改变程序语义；其余文本差异必须拒绝。
    assert raw.decode('utf-8-sig').replace('\r\n','\n')==main.decode('utf-8-sig').replace('\r\n','\n'),'本地与已读取main源入口不同'
s=raw.decode('utf-8-sig');lines=s.splitlines(keepends=True);tree=ast.parse(s)
nodes=[n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'"]
assert len(nodes)==1
n=nodes[0];sealed=''.join(lines[:n.lineno-1]+lines[n.end_lineno:]).encode('utf-8')
template=(here/'train_template.py').read_text(encoding='utf-8')
final=template.replace('__FROZEN_SHA__',repr(hashlib.sha256(sealed).hexdigest())).replace('__FROZEN_PAYLOAD__',repr(base64.b85encode(zlib.compress(sealed,9)).decode()))
ast.parse(final)
name='train_uam_gap_factors_50m_ONEFILE.py'
for target in (here/name,project/name):target.write_text(final,encoding='utf-8',newline='\n')
print('生成',name,'bytes',len(final.encode()),'sha256',hashlib.sha256(final.encode()).hexdigest())
