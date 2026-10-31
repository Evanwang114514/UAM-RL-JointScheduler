"""纯静态和描述符单元检查，不导入torch、不创建仿真、不启动PPO。"""
import ast, unittest, random, math, io, csv, tempfile
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np

HERE=Path(__file__).resolve().parent
TREE=ast.parse((HERE/'train_template.py').read_text(encoding='utf-8'))
NAMES={'gap_registry','gap_extra','GapSchema','gap_bank'}
CODE=ast.Module(body=[n for n in TREE.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in NAMES],type_ignores=[])
NS=dict(random=random,math=math,np=np,Path=Path,io=io,csv=csv,ATTR_STEPS=1_003_520,ATTR_PLANNED=48_168_960,ATTR_CAP=50_000_000,
 GAP_ARMS={'L00':(False,False,'LSTM'),'L10_TIME':(True,False,'LSTM'),'L01_BATCH':(False,True,'LSTM'),'L11_BOTH':(True,True,'LSTM'),'T11_BOTH':(True,True,'TRANSFORMER')})
exec(compile(CODE,'<仅静态定义>','exec'),NS)

class Check(unittest.TestCase):
 def test_budget(self):
  ts=NS['gap_registry']();self.assertEqual(len(ts),48);self.assertEqual(len({x['id'] for x in ts}),48)
  self.assertEqual(len([x for x in ts if x['k']==3]),40)
  self.assertEqual(48*1_003_520,48_168_960)
 def test_factorial(self):
  ts=NS['gap_registry']()
  for seed in (1701,1702,1703,1704):
   for c in ('LQ','J3'):
    self.assertEqual({x['arm'] for x in ts if x['k']==3 and x['seed']==seed and x['control']==c},set(NS['GAP_ARMS']))
 def test_order(self):
  ts=NS['gap_registry']();self.assertTrue(all(t['stage']=='P1_FACTORS' for t in ts[:16]))
  self.assertEqual(sum(t['network']=='TRANSFORMER' for t in ts[:20]),4)
  self.assertTrue(all(t['stage']=='P5_BOUNDARY' for t in ts[-8:]))
 def test_schema(self):
  for k in (2,3,4):
   d=NS['GapSchema'](k);self.assertEqual(d.dim,2*d.pack+k)
   self.assertEqual(d.extra.stop-d.extra.start,6*k)
   self.assertEqual(d.return_mask.stop,d.dim)
 def test_information_and_zero_sham(self):
  class Hidden:
   @property
   def file_data(self):raise AssertionError('非法访问未揭示需求')
  p=N(origin_vertiport_id='0',state='enroute',sub_state='to_vertiport',current_timer=5.)
  people=Hidden();people.persons={'p':p}
  vs={str(i):N(person_list=list(range(i+3)),vertiport_position=(i,0)) for i in (0,1)}
  s=N(persons=people,vertiports=N(vertiport_list=vs,evtols_at_vertiport={}),vehicles=N(estimate_travel_time=lambda a,b:7.),_all_evtols={})
  NS.update(CANDIDATES={2:(0,1)},charging_pair=lambda _: (0.,0.),ready_descriptor=lambda s,v:601.)
  z=NS['gap_extra'](s,2,N(origin_position=(0,0)),False,False)
  self.assertTrue(np.array_equal(z,np.zeros(12)))
  b=NS['gap_extra'](s,2,None,False,True).reshape(2,6)
  self.assertTrue(np.array_equal(b[:,:4],np.zeros((2,4))))
  self.assertAlmostEqual(b[0,4],1.);self.assertAlmostEqual(b[0,5],0.)
  t=NS['gap_extra'](s,2,N(origin_position=(0,0)),True,False).reshape(2,6)
  self.assertTrue(np.array_equal(t[:,4:],np.zeros((2,2))))
  self.assertAlmostEqual(t[0,2],5/60)
 def test_bank(self):
  import hashlib
  NS['digest']=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
  NS['save_json']=lambda *a:None
  with tempfile.TemporaryDirectory(prefix='uam_gap_test_',dir=HERE) as name:
   root=Path(name);(root/'train_data').mkdir()
   rows=[dict(id=f'P{i}',time=i,origin_x=i%11,origin_y=i%7,dest_x=i%3,dest_y=i%13) for i in range(300)]
   f=root/'train_data/passengers_300.csv'
   with f.open('w',newline='',encoding='utf-8') as stream:
    w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   bank=NS['gap_bank'](root,root);self.assertEqual(len(bank),5)
   for item in bank[1:]:
    with open(item['path'],encoding='utf-8') as stream:rs=list(csv.DictReader(stream))
    self.assertEqual([int(r['time']) for r in rs],list(range(300)))
    self.assertEqual([r['id'] for r in rs],[r['id'] for r in rows])
    od=lambda r:tuple(str(r[k]) for k in ('origin_x','origin_y','dest_x','dest_y'))
    self.assertEqual(sorted(map(od,rs)),sorted(map(od,rows)))
 def test_no_auxiliary_and_no_hidden_schema(self):
  source=ast.unparse(TREE)
  extra=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='gap_extra')
  self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='file_data' for n in ast.walk(extra)))
  self.assertIn('return []',source)
  self.assertIn("reward_mode='PT_POST'",source)

if __name__=='__main__':unittest.main(verbosity=2)
