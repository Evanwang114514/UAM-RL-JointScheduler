# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, gc, hashlib, json, math, shutil, traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import train_uam_7x12_600k_v2 as exp

try:
    import run_uagmc_6env_6baseline_36runs as strongbase
except Exception as exc:
    strongbase = None
    STRONGBASE_IMPORT_ERROR = repr(exc)
else:
    STRONGBASE_IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parent
DEFAULT_ENVS = ("S1","S2","S3")
DEFAULT_METHODS = tuple(f"M{i}" for i in range(12))
VALIDATION_SEEDS = (123,124,125)
TEST_SEEDS = (223,224,225)
BASELINE_METHODS = ("STTF","QTTI2","ECTF")
UNKNOWN_EVENT_PENALTY_MIN = 60.0

def fnum(x: Any, default=float("nan")) -> float:
    try:
        y=float(np.asarray(x).reshape(-1)[0])
        return y if math.isfinite(y) else default
    except Exception:
        return default

def fmean(xs: Iterable[Any]) -> float:
    a=np.asarray([fnum(x) for x in xs],dtype=float)
    a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")

def fstd(xs: Iterable[Any], ddof=0) -> float:
    a=np.asarray([fnum(x) for x in xs],dtype=float)
    a=a[np.isfinite(a)]
    if not len(a): return float("nan")
    if len(a)<=ddof: return 0.0
    return float(a.std(ddof=ddof))

def as_bool(x: Any) -> bool:
    if isinstance(x,bool): return x
    return str(x).strip().lower() in ("1","true","yes","y","t")

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding="utf-8")

def write_csv(path: Path, rows: Sequence[Dict[str,Any]]) -> None:
    rows=list(rows); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8-sig"); return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader()
        for r in rows:
            out={}
            for k in fields:
                v=r.get(k,"")
                if isinstance(v,np.ndarray): v=v.tolist()
                if isinstance(v,(list,tuple,dict)): v=json.dumps(v,ensure_ascii=False,default=str)
                out[k]=v
            w.writerow(out)

def read_csv(path: Path) -> List[Dict[str,Any]]:
    if not path.exists(): return []
    with path.open("r",newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def parse_names(text: str, allowed: Sequence[str]) -> List[str]:
    xs=[x.strip().upper() for x in str(text).split(",") if x.strip()]
    bad=[x for x in xs if x not in allowed]
    if bad: raise ValueError(f"unsupported={bad}; allowed={list(allowed)}")
    return xs

def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]

def stable_hash(vec: np.ndarray, decimals: int) -> str:
    return hashlib.sha1(np.round(np.asarray(vec,dtype=np.float64),decimals).tobytes()).hexdigest()[:20]

def normalized_auc(xs,ys) -> float:
    if not ys: return float("nan")
    if len(xs)<2: return float(ys[0])
    x=np.asarray(xs,dtype=float); y=np.asarray(ys,dtype=float)
    o=np.argsort(x); x=x[o]; y=y[o]; span=float(x[-1]-x[0])
    if span <= 0:
        return float(np.mean(y))
    # NumPy 2.x removed np.trapz; np.trapezoid is the supported replacement.
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        # Fallback for older NumPy.
        trapezoid = np.trapz
    return float(trapezoid(y, x) / span)

def corrcoef_safe(a,b) -> float:
    x=np.asarray(a,dtype=float); y=np.asarray(b,dtype=float)
    m=np.isfinite(x)&np.isfinite(y); x=x[m]; y=y[m]
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12: return float("nan")
    return float(np.corrcoef(x,y)[0,1])

def find_run_root(arg: Optional[str]) -> Path:
    if arg and str(arg).lower()!="auto":
        p=Path(arg).expanduser()
        if not p.is_absolute(): p=ROOT/p
        p=p.resolve()
        if not p.exists(): raise FileNotFoundError(p)
        return p
    cs=sorted((ROOT/"serial_runs").glob("uam_7x12_600k_seed1_*"),key=lambda p:p.stat().st_mtime,reverse=True)
    for p in cs:
        if (p/"S1__M0").exists(): return p.resolve()
    raise FileNotFoundError("Pass --run-root explicitly.")

def cell_dir(root:Path,e:str,m:str)->Path: return root/f"{e}__{m}"

def preflight(run_root,envs,methods):
    exp.assert_p0_patch()
    missing=[]; available=[]
    for e in envs:
        for m in methods:
            p=cell_dir(run_root,e,m)/"analysis"/"checkpoint_curve.csv"
            (available if p.exists() else missing).append(f"{e}__{m}")
    return {
        "run_root":str(run_root),"ZERO_PPO_TRAINING":True,
        "strong_baseline_available":strongbase is not None,
        "strong_baseline_import_error":STRONGBASE_IMPORT_ERROR,
        "available_cells":available,"missing_cells":missing
    }

def select_best_checkpoint(run_root,e,m):
    d=cell_dir(run_root,e,m); rows=read_csv(d/"analysis"/"checkpoint_curve.csv")
    c=[]
    for r in rows:
        att=fnum(r.get("ATT_mean"))
        if as_bool(r.get("all_full_completion")) and math.isfinite(att):
            c.append((att,int(fnum(r.get("train_step"),0)),r))
    if not c: return None
    _,step,row=min(c,key=lambda z:(z[0],z[1]))
    model,vec=exp.checkpoint_paths(d,step)
    if not model.exists() or not vec.exists(): return None
    return {
        "env_key":e,"method":m,"method_name":exp.METHOD_NAMES.get(m,m),
        "selected_step":step,"validation_ATT_mean":fnum(row.get("ATT_mean")),
        "validation_ATT_std":fnum(row.get("ATT_std")),
        "model_path":str(model),"vec_path":str(vec),"cell_dir":str(d)
    }

def collect_selected(run_root,envs,methods):
    out=[]
    for e in envs:
        for m in methods:
            x=select_best_checkpoint(run_root,e,m)
            if x: out.append(x)
    return out

def baseline_stage_for_env(e):
    s=exp.ENV_SPECS[e]
    return ("E4" if s.physical_stage=="E4" else "E6",float(s.pad_separation),int(s.charger_capacity))

def run_exact_baselines(out_root,envs,seeds,continue_on_error):
    if strongbase is None: raise RuntimeError(str(STRONGBASE_IMPORT_ERROR))
    raw=[]; errors=[]; total=len(envs)*len(BASELINE_METHODS)*len(seeds); idx=0
    for e in envs:
        stage,pad,cap=baseline_stage_for_env(e)
        for method in BASELINE_METHODS:
            for seed in seeds:
                idx+=1
                rd=out_root/"01_exact_baselines"/"runs"/f"{e}__{method}__seed{seed}"; rd.mkdir(parents=True,exist_ok=True)
                print(f"[baseline {idx:02d}/{total:02d}] {e}/{method}/seed{seed}",flush=True)
                try:
                    exp.configure_worker_physics()
                    r=dict(strongbase.run_one(
                        stage=stage,topology=exp.TOPOLOGY,method=method,seed=int(seed),
                        fleet_size=exp.FLEET_SIZE,pad_separation=pad,charger_capacity=cap,
                        max_time=exp.HARD_GUARD,unknown_penalty=UNKNOWN_EVENT_PENALTY_MIN,run_dir=rd))
                    r.update({"env_key":e,"physical_stage":stage,"turnaround_min":exp.TURNAROUND_MIN,
                              "charge_rate_scale":exp.CHARGE_RATE_SCALE,"charger_capacity":cap,
                              "pad_separation_min":pad,"eval_seed":int(seed)})
                    raw.append(r); write_csv(out_root/"01_exact_baselines"/"raw_results.csv",raw)
                except Exception as exc:
                    errors.append({"env_key":e,"method":method,"eval_seed":seed,"error":repr(exc),"traceback":traceback.format_exc()})
                    write_csv(out_root/"01_exact_baselines"/"errors.csv",errors)
                    if not continue_on_error: raise
                finally:
                    try: exp.core.restore_process_patches()
                    except Exception: pass
                    gc.collect()
    g=defaultdict(list)
    for r in raw: g[(str(r["env_key"]),str(r["method"]))].append(r)
    agg=[]
    metrics=("ATT","AWT","AGT_access","AFT","travel_p90","travel_p95","completion_rate","system_person_minutes_per_passenger")
    for (e,m),rs in sorted(g.items()):
        row={"env_key":e,"method":m,"n_eval_seeds":len(rs),
             "all_full_completion":all(fnum(x.get("completion_rate"),0)>=.999999 for x in rs)}
        for k in metrics:
            row[k+"_mean"]=fmean(x.get(k) for x in rs); row[k+"_std"]=fstd((x.get(k) for x in rs),ddof=1)
        agg.append(row)
    write_csv(out_root/"01_exact_baselines"/"aggregate_results.csv",agg)
    write_csv(out_root/"01_exact_baselines"/"errors.csv",errors)
    return raw,agg

def compute_stability(out_root,run_root,envs,methods):
    rows=[]
    for e in envs:
        for m in methods:
            curve=read_csv(cell_dir(run_root,e,m)/"analysis"/"checkpoint_curve.csv")
            if not curve: continue
            allr=sorted(curve,key=lambda r:int(fnum(r.get("train_step"),0)))
            valid=[r for r in allr if as_bool(r.get("all_full_completion")) and math.isfinite(fnum(r.get("ATT_mean")))]
            if not valid:
                rows.append({"env_key":e,"method":m,"n_checkpoints":len(allr),"n_valid_checkpoints":0,"invalid_checkpoint_rate":1.0}); continue
            steps=[int(fnum(r["train_step"])) for r in valid]; atts=[fnum(r["ATT_mean"]) for r in valid]
            bi=int(np.argmin(atts)); best=atts[bi]; final=atts[-1]; last3=atts[-3:]; post=atts[bi:]
            late=[fnum(r["ATT_mean"]) for r in valid if int(fnum(r["train_step"]))>=400000]
            rows.append({
                "env_key":e,"method":m,"method_name":exp.METHOD_NAMES.get(m,m),
                "n_checkpoints":len(allr),"n_valid_checkpoints":len(valid),
                "invalid_checkpoint_rate":1-len(valid)/max(1,len(allr)),
                "best_step":steps[bi],"best_ATT":best,"final_step":steps[-1],"final_ATT":final,
                "best_to_final_abs":final-best,"best_to_final_pct":(final-best)/best if best>0 else float("nan"),
                "last3_ATT_mean":fmean(last3),"last3_ATT_std":fstd(last3,ddof=1),
                "late400k_ATT_mean":fmean(late),"late400k_ATT_std":fstd(late,ddof=1),
                "checkpoint_AUC_normalized":normalized_auc(steps,atts),
                "collapse_threshold_1p5x_best":1.5*best,
                "collapse_count_after_best":sum(x>1.5*best for x in post),
                "collapse_rate_after_best":sum(x>1.5*best for x in post)/max(1,len(post)),
                "post_best_n":len(post)
            })
    write_csv(out_root/"03_stability"/"stability_summary.csv",rows)
    return rows

def run_independent_tests(out_root,selected,test_seeds,continue_on_error):
    raw=[]; errors=[]; total=len(selected)*len(test_seeds); idx=0
    for s in selected:
        e=str(s["env_key"]); m=str(s["method"]); step=int(s["selected_step"]); d=Path(s["cell_dir"])
        for seed in test_seeds:
            idx+=1
            print(f"[test {idx:03d}/{total:03d}] {e}/{m}/{step//1000}k seed={seed}",flush=True)
            try:
                r=dict(exp.evaluate_checkpoint(
                    env_key=e,method=m,model_path=Path(s["model_path"]),vec_path=Path(s["vec_path"]),
                    train_step=step,eval_seed=int(seed),run_dir=d))
                r["selection_validation_ATT_mean"]=s["validation_ATT_mean"]
                r["selection_validation_seeds"]=list(VALIDATION_SEEDS)
                r["independent_test"]=True
                raw.append(r); write_csv(out_root/"02_independent_test"/"raw_results.csv",raw)
            except Exception as exc:
                errors.append({"env_key":e,"method":m,"train_step":step,"test_seed":seed,
                               "error":repr(exc),"traceback":traceback.format_exc()})
                write_csv(out_root/"02_independent_test"/"errors.csv",errors)
                if not continue_on_error: raise
    g=defaultdict(list)
    for r in raw: g[(str(r["env_key"]),str(r["method"]))].append(r)
    agg=[]
    metrics=("ATT","AWT","AGT_access","AFT","travel_p90","travel_p95","completion_rate",
             "system_person_minutes_per_passenger","mean_abs_supply_demand_gap",
             "mean_late_supply","mean_early_supply","late_supply_rate")
    for (e,m),rs in sorted(g.items()):
        row={"env_key":e,"method":m,"method_name":exp.METHOD_NAMES.get(m,m),
             "selected_step":int(rs[0]["train_step"]),"n_test_seeds":len(rs),
             "all_full_completion":all(bool(x.get("valid_full_completion")) for x in rs)}
        for k in metrics:
            row[k+"_mean"]=fmean(x.get(k) for x in rs); row[k+"_std"]=fstd((x.get(k) for x in rs),ddof=1)
        row["validation_ATT_mean"]=fnum(rs[0].get("selection_validation_ATT_mean"))
        row["test_minus_validation_ATT"]=fnum(row.get("ATT_mean"))-row["validation_ATT_mean"]
        agg.append(row)
    write_csv(out_root/"02_independent_test"/"aggregate_results.csv",agg)
    write_csv(out_root/"02_independent_test"/"errors.csv",errors)
    return raw,agg

def load_selected_eval(out_root,s,seed):
    e=str(s["env_key"]); m=str(s["method"])
    raw=exp.build_eval_env(env_key=e,method=m,run_dir=out_root/"_diagnostic_monitor"/f"{e}__{m}__seed{seed}")
    env=VecNormalize.load(str(s["vec_path"]),raw); env.training=False; env.norm_reward=False
    model=PPO.load(str(s["model_path"]),env=env,device="cpu"); model.policy.set_training_mode(False)
    try: env.seed(int(seed))
    except Exception: pass
    return env,model,env.reset()

def close_eval(env):
    try: env.close()
    except Exception: pass
    try: exp.core.restore_process_patches()
    except Exception: pass
    gc.collect()

def pressure_from_project(p):
    return float(p[1])/(1.0+max(0.0,float(p[5])))

def collect_rank_inversion(out_root,selected_map,envs,seeds,continue_on_error):
    raw_rows=[]; errors=[]; vectors=defaultdict(list)
    for e in envs:
        s=selected_map.get((e,"M0"))
        if not s: continue
        spec=exp.ENV_SPECS[e]; layout=exp.make_layout(spec); visible_hi=int(layout["slices"]["focal"][1])
        for seed in seeds:
            print(f"[rank] {e}/M0 seed={seed}",flush=True)
            env=None
            try:
                env,model,obs=load_selected_eval(out_root,s,int(seed)); done=np.asarray([False]); di=0
                while not bool(done[0]):
                    scenario=exp.mx.find_scenario(env); person=exp._focal_person(env,scenario)
                    if person is not None:
                        p0s=[]; pes=[]; hs=[]; cp=[]; ep=[]
                        for vid in exp.CANDIDATES:
                            h=max(0.0,exp._access_time(scenario,person,int(vid)))
                            p0=exp._project(scenario,spec,int(vid),0.0); pe=exp._project(scenario,spec,int(vid),h)
                            hs.append(h); p0s.append(p0); pes.append(pe); cp.append(pressure_from_project(p0)); ep.append(pressure_from_project(pe))
                        ca=int(np.argmin(cp)); ea=int(np.argmin(ep)); probs=exp.policy_probs(model,obs)
                        try: original=np.asarray(env.get_original_obs(),dtype=float).reshape(1,-1)[0]
                        except Exception: original=np.asarray(obs,dtype=float).reshape(1,-1)[0]
                        vis=original[:visible_hi].astype(float,copy=True)
                        row={
                            "env_key":e,"seed":int(seed),"decision_idx":di,"sim_time":fnum(getattr(scenario,"time",np.nan)),
                            "current_argmin":ca,"effect_argmin":ea,"rank_flip":int(ca!=ea),
                            "access_v0":hs[0],"access_v1":hs[1],
                            "current_pressure_v0":cp[0],"current_pressure_v1":cp[1],
                            "effect_pressure_v0":ep[0],"effect_pressure_v1":ep[1],
                            "effect_margin_abs":abs(ep[0]-ep[1]),
                            "policy_p0":fnum(probs[0] if len(probs)>0 else np.nan),
                            "policy_p1":fnum(probs[1] if len(probs)>1 else np.nan),
                            "visible_hash_exact4":stable_hash(vis,4),"visible_hash_coarse1":stable_hash(vis,1),
                            "current_project_v0":p0s[0],"current_project_v1":p0s[1],
                            "effect_project_v0":pes[0],"effect_project_v1":pes[1]
                        }
                        raw_rows.append(row); vectors[e].append({**row,"_visible":vis,"_future_vec":np.asarray(list(pes[0])+list(pes[1]),dtype=float)}); di+=1
                    action,_=model.predict(obs,deterministic=True); obs,_,done,_=env.step(action)
            except Exception as exc:
                errors.append({"env_key":e,"seed":seed,"error":repr(exc),"traceback":traceback.format_exc()})
                if not continue_on_error: raise
            finally:
                if env is not None: close_eval(env)
    write_csv(out_root/"04_rank_inversion"/"raw_decisions.csv",raw_rows); write_csv(out_root/"04_rank_inversion"/"errors.csv",errors)
    g=defaultdict(list)
    for r in raw_rows: g[str(r["env_key"])].append(r)
    summary=[]
    for e,rs in sorted(g.items()):
        summary.append({"env_key":e,"n_decisions":len(rs),"rank_flip_rate":fmean(r["rank_flip"] for r in rs),
                        "mean_effect_margin_abs":fmean(r["effect_margin_abs"] for r in rs),
                        "mean_current_margin_abs":fmean(abs(fnum(r["current_pressure_v0"])-fnum(r["current_pressure_v1"])) for r in rs)})
    write_csv(out_root/"04_rank_inversion"/"summary.csv",summary)
    return raw_rows,summary,vectors

def group_conflicting_pairs(records,hash_key,label,max_pairs=200):
    groups=defaultdict(list)
    for r in records: groups[str(r[hash_key])].append(r)
    out=[]
    for sig,rs in groups.items():
        if len(rs)<2 or len(set(int(r["effect_argmin"]) for r in rs))<2: continue
        pair=None
        for i in range(len(rs)):
            for j in range(i+1,len(rs)):
                if int(rs[i]["effect_argmin"])!=int(rs[j]["effect_argmin"]):
                    pair=(rs[i],rs[j]); break
            if pair: break
        if not pair: continue
        a,b=pair; va=np.asarray(a["_visible"]); vb=np.asarray(b["_visible"]); fa=np.asarray(a["_future_vec"]); fb=np.asarray(b["_future_vec"])
        out.append({"pair_type":label,"env_key":a["env_key"],"signature":sig,"group_size":len(rs),
                    "a_seed":a["seed"],"a_decision_idx":a["decision_idx"],"a_effect_action":a["effect_argmin"],
                    "a_policy_p0":a["policy_p0"],"a_policy_p1":a["policy_p1"],
                    "b_seed":b["seed"],"b_decision_idx":b["decision_idx"],"b_effect_action":b["effect_argmin"],
                    "b_policy_p0":b["policy_p0"],"b_policy_p1":b["policy_p1"],
                    "visible_l2":float(np.linalg.norm(va-vb)),"future_l2":float(np.linalg.norm(fa-fb)),
                    "policy_prob_l1":abs(fnum(a["policy_p0"])-fnum(b["policy_p0"]))+abs(fnum(a["policy_p1"])-fnum(b["policy_p1"]))})
        if len(out)>=max_pairs: break
    return out

def nearest_conflicting_pairs(records,max_pairs=100):
    if len(records)<2: return []
    X=np.stack([np.asarray(r["_visible"],dtype=float) for r in records]); y=np.asarray([int(r["effect_argmin"]) for r in records])
    mu=X.mean(0,keepdims=True); sd=X.std(0,keepdims=True); sd[sd<1e-6]=1.0; Z=(X-mu)/sd
    pairs={}
    for i in range(len(records)):
        idx=np.flatnonzero(y!=y[i])
        if not len(idx): continue
        d2=np.mean((Z[idx]-Z[i])**2,axis=1); j=int(idx[int(np.argmin(d2))]); a,b=sorted((i,j))
        if (a,b) in pairs: continue
        ra,rb=records[a],records[b]
        pairs[(a,b)]={"pair_type":"nearest_conflicting_oracle","env_key":ra["env_key"],
                      "a_seed":ra["seed"],"a_decision_idx":ra["decision_idx"],"a_effect_action":ra["effect_argmin"],
                      "a_policy_p0":ra["policy_p0"],"a_policy_p1":ra["policy_p1"],
                      "b_seed":rb["seed"],"b_decision_idx":rb["decision_idx"],"b_effect_action":rb["effect_argmin"],
                      "b_policy_p0":rb["policy_p0"],"b_policy_p1":rb["policy_p1"],
                      "standardized_visible_rmse":float(math.sqrt(float(np.min(d2)))),
                      "raw_visible_l2":float(np.linalg.norm(X[a]-X[b])),
                      "future_l2":float(np.linalg.norm(np.asarray(ra["_future_vec"])-np.asarray(rb["_future_vec"]))),
                      "policy_prob_l1":abs(fnum(ra["policy_p0"])-fnum(rb["policy_p0"]))+abs(fnum(ra["policy_p1"])-fnum(rb["policy_p1"]))}
    out=list(pairs.values()); out.sort(key=lambda r:(fnum(r["standardized_visible_rmse"]),-fnum(r["future_l2"])))
    return out[:max_pairs]

def analyze_aliasing(out_root,vectors):
    pairs=[]; summary=[]
    for e,rs in sorted(vectors.items()):
        exact=group_conflicting_pairs(rs,"visible_hash_exact4","exact_round4")
        coarse=group_conflicting_pairs(rs,"visible_hash_coarse1","coarse_round1")
        nearest=nearest_conflicting_pairs(rs); pairs+=exact+coarse+nearest
        eg=defaultdict(list); cg=defaultdict(list)
        for r in rs: eg[r["visible_hash_exact4"]].append(r); cg[r["visible_hash_coarse1"]].append(r)
        def counts(g):
            d=c=0
            for gr in g.values():
                if len(gr)>=2:
                    d+=1
                    if len(set(int(x["effect_argmin"]) for x in gr))>=2: c+=1
            return d,c
        ed,ec=counts(eg); cd,cc=counts(cg)
        summary.append({"env_key":e,"n_records":len(rs),"exact_round4_duplicate_groups":ed,"exact_round4_conflicting_groups":ec,
                        "coarse_round1_duplicate_groups":cd,"coarse_round1_conflicting_groups":cc,
                        "n_exact_conflict_pairs_saved":len(exact),"n_coarse_conflict_pairs_saved":len(coarse),
                        "nearest_conflict_best_distance":fnum(nearest[0].get("standardized_visible_rmse")) if nearest else float("nan"),
                        "exact_pair_found":bool(exact),
                        "interpretation":"Exact natural alias pair found." if exact else "No exact natural duplicate found; nearest/conflicting pairs are only candidates for a later controlled synthetic intervention."})
    write_csv(out_root/"05_state_aliasing"/"pairs.csv",pairs); write_csv(out_root/"05_state_aliasing"/"summary.csv",summary)
    return pairs,summary

def wm_selected_prediction(model,obs,action_index):
    ext=getattr(model.policy,"features_extractor",None)
    if not isinstance(ext,exp.FactorialExtractor): raise TypeError(type(ext))
    if not bool(getattr(ext,"use_wm",False)): raise ValueError("method has no WM")
    with torch.no_grad():
        x=torch.as_tensor(obs,device=model.device,dtype=torch.float32); resource=ext._slice(x,"resource"); mean,std=ext._world_predictions(resource)
    dim=int(exp.RESOURCE_DIM); lo=int(action_index)*dim; hi=lo+dim
    return mean[:,lo:hi].detach().cpu().numpy().reshape(-1),std[:,lo:hi].detach().cpu().numpy().reshape(-1)

def target_resource(env_key,obs):
    lo,hi=exp.make_layout(exp.ENV_SPECS[env_key])["slices"]["resource"]
    return np.asarray(obs,dtype=float).reshape(1,-1)[0][int(lo):int(hi)]

def run_wm_calibration(out_root,selected_map,envs,seeds,continue_on_error):
    wm_methods=[m for m in DEFAULT_METHODS if exp.METHOD_FLAGS[m][2]]; raw=[]; errors=[]
    for e in envs:
        for m in wm_methods:
            s=selected_map.get((e,m))
            if not s: continue
            for seed in seeds:
                print(f"[wm-cal] {e}/{m} seed={seed}",flush=True); env=None
                try:
                    env,model,obs=load_selected_eval(out_root,s,int(seed)); done=np.asarray([False]); si=0
                    while not bool(done[0]):
                        action,_=model.predict(obs,deterministic=True); ai=int(np.asarray(action).reshape(-1)[0]); pm,ps=wm_selected_prediction(model,obs,ai)
                        nxt,_,done,_=env.step(action)
                        if not bool(done[0]):
                            tgt=target_resource(e,nxt); err=tgt-pm; ae=np.abs(err); sig=np.maximum(ps,1e-6)
                            raw.append({"env_key":e,"method":m,"selected_step":int(s["selected_step"]),"seed":int(seed),"step_idx":si,"action":ai,
                                        "mae":float(np.mean(ae)),"rmse":float(np.sqrt(np.mean(err**2))),"pred_std_mean":float(np.mean(ps)),
                                        "coverage_1sigma":float(np.mean(ae<=sig)),"coverage_2sigma":float(np.mean(ae<=2*sig)),
                                        "gaussian_nll_proxy":float(np.mean(np.log(sig)+.5*(err/sig)**2)),
                                        "abs_error_by_dim":ae.tolist(),"pred_std_by_dim":ps.tolist()})
                        obs=nxt; si+=1
                except Exception as exc:
                    errors.append({"env_key":e,"method":m,"seed":seed,"error":repr(exc),"traceback":traceback.format_exc()})
                    if not continue_on_error: raise
                finally:
                    if env is not None: close_eval(env)
    write_csv(out_root/"06_wm_calibration"/"raw_one_step.csv",raw); write_csv(out_root/"06_wm_calibration"/"errors.csv",errors)
    g=defaultdict(list)
    for r in raw: g[(str(r["env_key"]),str(r["method"]))].append(r)
    summary=[]
    for (e,m),rs in sorted(g.items()):
        flat_e=[]; flat_s=[]
        for r in rs: flat_e.extend(r["abs_error_by_dim"]); flat_s.extend(r["pred_std_by_dim"])
        summary.append({"env_key":e,"method":m,"selected_step":int(rs[0]["selected_step"]),"n_transitions":len(rs),
                        "one_step_MAE_mean":fmean(r["mae"] for r in rs),"one_step_MAE_std":fstd((r["mae"] for r in rs),ddof=1),
                        "one_step_RMSE_mean":fmean(r["rmse"] for r in rs),"pred_std_mean":fmean(r["pred_std_mean"] for r in rs),
                        "corr_transition_std_vs_MAE":corrcoef_safe([r["pred_std_mean"] for r in rs],[r["mae"] for r in rs]),
                        "corr_dimension_std_vs_abs_error":corrcoef_safe(flat_s,flat_e),
                        "coverage_1sigma_mean":fmean(r["coverage_1sigma"] for r in rs),"coverage_2sigma_mean":fmean(r["coverage_2sigma"] for r in rs),
                        "gaussian_nll_proxy_mean":fmean(r["gaussian_nll_proxy"] for r in rs),
                        "calibration_scope":"CURRENT one-step normalized-resource WM only; NOT effect-time residual WM"})
    write_csv(out_root/"06_wm_calibration"/"summary.csv",summary)
    return raw,summary

def build_master(out_root,baseline_agg,test_agg,stability,rank_summary,alias_summary,wm_summary):
    b=defaultdict(list); t=defaultdict(list)
    for r in baseline_agg: b[str(r["env_key"])].append(r)
    for r in test_agg: t[str(r["env_key"])].append(r)
    rm={str(r["env_key"]):r for r in rank_summary}; am={str(r["env_key"]):r for r in alias_summary}
    stage=[]
    for e in DEFAULT_ENVS:
        bs=[r for r in b.get(e,[]) if math.isfinite(fnum(r.get("ATT_mean")))]
        ts=[r for r in t.get(e,[]) if math.isfinite(fnum(r.get("ATT_mean")))]
        bb=min(bs,key=lambda r:fnum(r["ATT_mean"])) if bs else {}; m0=next((r for r in ts if str(r.get("method"))=="M0"),{}); br=min(ts,key=lambda r:fnum(r["ATT_mean"])) if ts else {}
        stage.append({"env_key":e,"best_analytic_method":bb.get("method",""),"best_analytic_test_ATT":fnum(bb.get("ATT_mean")),
                      "M0_independent_test_ATT":fnum(m0.get("ATT_mean")),"best_RL_method_independent_test":br.get("method",""),
                      "best_RL_independent_test_ATT":fnum(br.get("ATT_mean")),"rank_flip_rate":fnum(rm.get(e,{}).get("rank_flip_rate")),
                      "exact_alias_pair_found":am.get(e,{}).get("exact_pair_found",""),
                      "nearest_alias_distance":fnum(am.get(e,{}).get("nearest_conflict_best_distance"))})
    write_csv(out_root/"MASTER_stage_comparison.csv",stage)
    sm={(str(r["env_key"]),str(r["method"])):r for r in stability}; long=[]
    for r in test_agg:
        s=sm.get((str(r["env_key"]),str(r["method"])),{})
        long.append({**r,"best_to_final_pct_existing_curve":fnum(s.get("best_to_final_pct")),"last3_ATT_mean_existing_curve":fnum(s.get("last3_ATT_mean")),
                     "checkpoint_AUC_normalized":fnum(s.get("checkpoint_AUC_normalized")),"collapse_rate_after_best":fnum(s.get("collapse_rate_after_best"))})
    write_csv(out_root/"MASTER_RL_independent_test_plus_stability.csv",long); write_csv(out_root/"MASTER_WM_calibration.csv",list(wm_summary))

def parse_args():
    p=argparse.ArgumentParser(description="ZERO-training UAM evidence collector")
    p.add_argument("--run-root",default="auto")
    p.add_argument("--envs",default="S1,S2,S3")
    p.add_argument("--methods",default=",".join(DEFAULT_METHODS))
    p.add_argument("--validation-seeds",default="123,124,125")
    p.add_argument("--test-seeds",default="223,224,225")
    p.add_argument("--output-root",default=None)
    p.add_argument("--continue-on-error",action="store_true")
    p.add_argument("--skip-baselines",action="store_true")
    p.add_argument("--skip-tests",action="store_true")
    p.add_argument("--skip-rank",action="store_true")
    p.add_argument("--skip-wm",action="store_true")
    p.add_argument("--no-zip",action="store_true")
    return p.parse_args()

def main():
    args=parse_args()
    envs=parse_names(args.envs,DEFAULT_ENVS); methods=parse_names(args.methods,DEFAULT_METHODS)
    vseeds=parse_ints(args.validation_seeds); tseeds=parse_ints(args.test_seeds); run_root=find_run_root(args.run_root)
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_root:
        out_root=Path(args.output_root).expanduser()
        if not out_root.is_absolute(): out_root=ROOT/out_root
        out_root=out_root.resolve()
    else:
        out_root=(ROOT/"serial_runs"/f"uam_no_train_evidence_{stamp}").resolve()
    out_root.mkdir(parents=True,exist_ok=True)

    manifest={"experiment":"UAM_NO_TRAIN_EVIDENCE","created":datetime.now().isoformat(timespec="seconds"),
              "ZERO_PPO_TRAINING":True,"run_root":str(run_root),"envs":envs,"methods":methods,
              "validation_seeds_for_checkpoint_selection":vseeds,"independent_test_seeds":tseeds,
              "baseline_methods":list(BASELINE_METHODS),"baseline_seeds":vseeds,
              "physics":{"topology":exp.TOPOLOGY,"fleet_size":exp.FLEET_SIZE,"turnaround_min":exp.TURNAROUND_MIN,
                         "charge_rate_scale":exp.CHARGE_RATE_SCALE,"charger_capacity":exp.CHARGER_CAPACITY,
                         "S1_pad":exp.ENV_SPECS["S1"].pad_separation,"S2_pad":exp.ENV_SPECS["S2"].pad_separation,
                         "S3_pad":exp.ENV_SPECS["S3"].pad_separation,"hard_guard":exp.HARD_GUARD},
              "rank_inversion_workload":"projected_waiting/(1+projected_serviceable_supply)",
              "wm_calibration_scope":"existing one-step normalized-resource WM; no retraining"}
    write_json(out_root/"manifest.json",manifest)
    write_json(out_root/"00_preflight.json",preflight(run_root,envs,methods))

    selected=collect_selected(run_root,envs,methods); write_csv(out_root/"selected_checkpoints.csv",selected)
    selected_map={(str(r["env_key"]),str(r["method"])):r for r in selected}
    stability=compute_stability(out_root,run_root,envs,methods)

    baseline_raw=[]; baseline_agg=[]
    if not args.skip_baselines:
        try: baseline_raw,baseline_agg=run_exact_baselines(out_root,envs,vseeds,bool(args.continue_on_error))
        except Exception:
            write_json(out_root/"01_exact_baselines"/"fatal_error.json",{"traceback":traceback.format_exc()})
            if not args.continue_on_error: raise

    test_raw=[]; test_agg=[]
    if not args.skip_tests:
        try: test_raw,test_agg=run_independent_tests(out_root,selected,tseeds,bool(args.continue_on_error))
        except Exception:
            write_json(out_root/"02_independent_test"/"fatal_error.json",{"traceback":traceback.format_exc()})
            if not args.continue_on_error: raise

    rank_raw=[]; rank_summary=[]; vectors={}; alias_pairs=[]; alias_summary=[]
    if not args.skip_rank:
        try:
            rank_raw,rank_summary,vectors=collect_rank_inversion(out_root,selected_map,envs,tseeds,bool(args.continue_on_error))
            alias_pairs,alias_summary=analyze_aliasing(out_root,vectors)
        except Exception:
            write_json(out_root/"04_rank_inversion"/"fatal_error.json",{"traceback":traceback.format_exc()})
            if not args.continue_on_error: raise

    wm_raw=[]; wm_summary=[]
    if not args.skip_wm:
        try: wm_raw,wm_summary=run_wm_calibration(out_root,selected_map,envs,tseeds,bool(args.continue_on_error))
        except Exception:
            write_json(out_root/"06_wm_calibration"/"fatal_error.json",{"traceback":traceback.format_exc()})
            if not args.continue_on_error: raise

    build_master(out_root,baseline_agg,test_agg,stability,rank_summary,alias_summary,wm_summary)
    done={"status":"SUCCESS","ZERO_PPO_TRAINING":True,"output_root":str(out_root),"selected_checkpoints":len(selected),
          "baseline_rows":len(baseline_raw),"independent_test_rows":len(test_raw),"stability_rows":len(stability),
          "rank_rows":len(rank_raw),"alias_pairs":len(alias_pairs),"wm_calibration_rows":len(wm_raw)}
    write_json(out_root/"RUN_COMPLETE.json",done)
    zp=None
    if not args.no_zip: zp=shutil.make_archive(str(out_root),"zip",root_dir=out_root)
    print("\n"+"="*110); print("ZERO-TRAINING EVIDENCE COLLECTION DONE"); print("Output:",out_root)
    if zp: print("ZIP:",zp)
    print("="*110)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
