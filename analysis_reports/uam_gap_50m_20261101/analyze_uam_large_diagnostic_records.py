"""只读分析已完成的无训练数据，不载入模型、不调用训练。"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, statistics, tarfile
from collections import defaultdict, Counter
from pathlib import Path, PurePosixPath

def mean(v):
    return statistics.mean(v) if v else None

def describe(v):
    v=[float(x) for x in v if x is not None and math.isfinite(float(x))]
    return dict(n=len(v), mean=mean(v), sd=statistics.stdev(v) if len(v)>1 else None,
                min=min(v) if v else None,max=max(v) if v else None,values=v)

def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();p=Path(a.archive);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    data={};tables={};code={};ledgers={};channels=Counter()
    with tarfile.open(p,'r:gz') as tar:
        for m in tar:
            parts=PurePosixPath(m.name).parts
            if m.issym() or m.islnk() or '..' in parts or m.name.startswith('/'):
                raise RuntimeError('不安全的归档成员')
            if not m.isfile():continue
            if m.name.endswith('/END.json') and any('/'+v+'/' in m.name for v in ['baseline','rules','branches_v2','intervene_v2','tail_v2']):
                data[m.name]=json.load(tar.extractfile(m))
            elif '/baseline/' in m.name and any(m.name.endswith('/'+q+'_s123.json') for q in ('original','order_4201','order_4204')):
                episode=json.load(tar.extractfile(m))
                if episode.get('decisions'):
                    ledgers[(parts[2],parts[3],episode['trace'])]={int(r['time']):r['unfinished'] for r in episode['ledger']}
                    for r in episode['decisions']:
                        k=len(r['stations']);frame=4+8*k;obs=r['observation'];channels['decisions']+=1
                        for j in range(6):
                            for i in range(k):
                                for f in (5,6,7):
                                    channels['raw_time_fields']+=1
                                    channels['nonzero_raw_time_fields']+=abs(obs[j*frame+4+8*i+f])>1e-9
                        channels['states_with_positive_committed_return']+=any(any(x['eta']>0 for x in v['returns']) for v in r['stations'])
                        channels['states_with_positive_committed_access']+=any(any(x['eta']>0 for x in v['incoming']) for v in r['stations'])
            elif len(parts)==2 and m.name.endswith('.csv'):
                blob=tar.extractfile(m).read();tables[parts[-1]]=list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig'))))
                (out/parts[-1]).write_bytes(blob)
            elif m.name.endswith('SESSION_END.json') or m.name.endswith('BANK.json'):
                data[parts[-1]]=json.load(tar.extractfile(m))
            elif m.name.endswith('.py'):
                blob=tar.extractfile(m).read();code[m.name]=hashlib.sha256(blob).hexdigest()
    ends={phase:[d for n,d in data.items() if '/'+phase+'/' in n] for phase in ['baseline','rules','branches_v2','intervene_v2','tail_v2']}
    assert len(ends['baseline'])==88 and len(ends['branches_v2'])==20
    assert data['SESSION_END.json']['clone_protocol']=='dispatch_rebind_v2_20261101'
    base={};base_rows=[];rules=[];inter=[];tails=[];branch=[]
    for d in ends['baseline']:
        t=d['task']
        for r in d['results']:
            m=r['metrics'];key=(d['family'],d['id'],r['trace'],r['eval_seed']);assert key not in base;base[key]=m
            assert (m['ATT'] is not None)==m['full']
            base_rows.append(dict(family=d['family'],id=d['id'],k=t['k'],seed=t['seed'],arm=t.get('arm',t['method']),trace=r['trace'],eval_seed=r['eval_seed'],rpt=m['restricted_person_time_per_generated'],completion=m['completion'],full=m['full'],att=m['ATT']))
    for d in ends['rules']:
        for r in d['results']:
            b=base[d['family'],d['id'],r['trace'],123];m=r['metrics']
            rules.append(dict(family=d['family'],id=d['id'],k=d['task']['k'],seed=d['task']['seed'],arm=d['task'].get('arm',d['task']['method']),trace=r['trace'],mode=r['mode'],base_rpt=b['restricted_person_time_per_generated'],rpt=m['restricted_person_time_per_generated'],gain=b['restricted_person_time_per_generated']-m['restricted_person_time_per_generated'],completion=m['completion'],full=m['full'],att=m['ATT']))
    for phase in ['intervene_v2','tail_v2','branches_v2']:
        for d in ends[phase]:d['task'].setdefault('control','LQ')
    for d in ends['intervene_v2']:
        for r in d['results']:
            b=base[d['family'],d['id'],r['trace'],123];m=r['metrics']
            inter.append(dict(family=d['family'],id=d['id'],k=d['task']['k'],seed=d['task']['seed'],control=d['task']['control'],trace=r['trace'],mode=r['mode'],base_rpt=b['restricted_person_time_per_generated'],rpt=m['restricted_person_time_per_generated'],gain=b['restricted_person_time_per_generated']-m['restricted_person_time_per_generated'],completion=m['completion'],base_completion=b['completion'],att=m['ATT']))
    for d in ends['tail_v2']:
        for r in d['results']:
            vals=[x['metrics'] for x in r['trials']];b=r['original'];valid=[x['ATT'] for x in vals if x['full']]
            tails.append(dict(family=d['family'],id=d['id'],k=d['task']['k'],seed=d['task']['seed'],control=d['task']['control'],trace=r['trace'],mode=r['mode'],base_rpt=b['restricted_person_time_per_generated'],repaired_rpt=min(x['restricted_person_time_per_generated'] for x in vals),best_full_att=min(valid) if valid else None,full_repair=bool(valid),base_full=b['full'],gain=b['restricted_person_time_per_generated']-min(x['restricted_person_time_per_generated'] for x in vals)))
    ledger_checks=0
    for d in ends['branches_v2']:
        assert d['clone_protocol']=='dispatch_rebind_v2_20261101'
        groups=defaultdict(list)
        for r in d['rows']:groups[r['trace'],r['index']].append(r)
        assert len(groups)==54
        for (trace,index),rs in groups.items():
            rr={tuple(r['action']):r for r in rs if r['scope']=='REALIZED'}
            cc={tuple(r['action']):r for r in rs if r['scope']=='COMMITTED'}
            original=tuple(rs[0]['original']);assert original in rr and set(rr)==set(cc)
            for h in ['60','120','240']:
                costs={act:r['snapshots'][h] for act,r in rr.items()};pred={act:r['snapshots'][h] for act,r in cc.items()}
                for v in costs.values():assert abs(v['total']-sum(v[x] for x in ['focal','existing','future']))<1e-6
                def pick(source,field):return min(source,key=lambda act:(source[act][field],act!=original,act))
                best=pick(costs,'total');sys=pick(pred,'total');self_=pick(pred,'focal');b=costs[original]
                tape=ledgers[d['family'],d['id'],trace];start=int(rs[0]['time'])
                assert b['total']==sum(tape.get(q,0) for q in range(start,start+int(h)))
                ledger_checks+=1
                po=[v['total'] for act,v in costs.items() if act[1]==original[1]]
                ao=[v['total'] for act,v in costs.items() if act[0]==original[0]]
                row=dict(family=d['family'],id=d['id'],k=d['task']['k'],seed=d['task']['seed'],control=d['task']['control'],trace=trace,index=index,time=rs[0]['time'],horizon=int(h),candidates=len(costs),original=str(original),best=str(best),system=str(sys),self_action=str(self_),gap=b['total']-costs[best]['total'],system_gain=b['total']-costs[sys]['total'],self_gain=b['total']-costs[self_]['total'],system_excess=costs[sys]['total']-costs[best]['total'],system_optimal=costs[sys]['total']==costs[best]['total'],self_optimal=costs[self_]['total']==costs[best]['total'],passenger_only_gap=b['total']-min(po) if po else None,air_only_gap=b['total']-min(ao) if ao else None,focal_gain=b['focal']-costs[best]['focal'],existing_gain=b['existing']-costs[best]['existing'],future_gain=b['future']-costs[best]['future'])
                branch.append(row)
    write_csv(out/'07_逐回放基线.csv',base_rows);write_csv(out/'08_规则配对增量.csv',rules);write_csv(out/'09_全程干预配对.csv',inter);write_csv(out/'10_尾批逐项.csv',tails);write_csv(out/'11_候选成本与分支归因.csv',branch)
    summary={}
    def grouped(rows,keys,fields):
        groups=defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in keys)].append(r)
        ans=[]
        for key,rs in sorted(groups.items(),key=lambda x:str(x[0])):
            # 先在同一训练模型内平均回放；不将状态、窗口或测试顺序冒充独立训练复制。
            models=defaultdict(list)
            for r in rs:models[r['id']].append(r)
            item=dict(zip(keys,key));item['rows']=len(rs);item['models']=len(models)
            for field in fields:item[field]=describe([mean([r[field] for r in vs if r[field] is not None]) for vs in models.values()])
            ans.append(item)
        return ans
    summary['baseline']=grouped([r for r in base_rows if r['eval_seed']==123],['family','k','arm'],['rpt','completion','full'])
    summary['rules']=grouped(rules,['family','k','arm','mode'],['rpt','gain','completion','full'])
    summary['intervene']=grouped(inter,['family','k','control','mode'],['rpt','gain','completion'])
    summary['tail']=grouped(tails,['family','k','control','mode'],['repaired_rpt','gain','full_repair'])
    summary['branch']=grouped(branch,['family','k','control','horizon'],['gap','system_gain','self_gain','system_optimal','passenger_only_gap','air_only_gap','focal_gain','existing_gain','future_gain'])
    summary['confirmation_intervene']=grouped([r for r in inter if r['trace']=='order_4204'],['family','k','control','mode'],['rpt','gain','completion'])
    summary['original_intervene']=grouped([r for r in inter if r['trace']=='original'],['family','k','control','mode'],['rpt','gain','completion'])
    summary['validation']=dict(archive=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),phase_models={k:len(v) for k,v in ends.items()},end=data['SESSION_END.json'],archived_code=code,branch_states=len(branch)//3,cohort_identity='PASS',independent_ledger_checks=ledger_checks,independent_ledger_mismatches=0,raw_channel_audit=dict(channels),old_v1_excluded=True)
    summary['paired_tail_intervention']=[]
    tm={(r['family'],r['id'],r['trace'],r['mode']):r for r in tails}
    for family,k,control in sorted({(r['family'],r['k'],r['control']) for r in tails}):
        for mode in ['SELF','SYSTEM']:
            ds=defaultdict(list)
            for r in tails:
                if (r['family'],r['k'],r['control'],r['mode'])!=(family,k,control,mode):continue
                b=tm[family,r['id'],r['trace'],'MODEL']
                ds[r['id']].append(b['repaired_rpt']-r['repaired_rpt'])
            summary['paired_tail_intervention'].append(dict(family=family,k=k,control=control,mode=mode,gain=describe([mean(v) for v in ds.values()])))
    summary['branch_opportunities']=[]
    for family,k,control,horizon in sorted({(r['family'],r['k'],r['control'],r['horizon']) for r in branch}):
        rs=[r for r in branch if (r['family'],r['k'],r['control'],r['horizon'])==(family,k,control,horizon)]
        summary['branch_opportunities'].append(dict(family=family,k=k,control=control,horizon=horizon,states=len(rs),positive=sum(r['gap']>0 for r in rs),system_harm=sum(r['system_gain']<0 for r in rs),system_improve=sum(r['system_gain']>0 for r in rs),best_harms_focal=sum(r['gap']>0 and r['focal_gain']<0 for r in rs)))
    (out/'分析汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ['baseline','rules','intervene','tail','branch','confirmation_intervene']:
        print('\n',name)
        for r in summary[name]:
            print({k:({q:round(v[q],3) if isinstance(v[q],(float,int)) else v[q] for q in ['mean','sd','min','max']} if isinstance(v,dict) else v) for k,v in r.items() if k!='rows'})

if __name__=='__main__':main()
