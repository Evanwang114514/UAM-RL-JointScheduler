# -*- coding: utf-8 -*-
"""
UAGMC official-source GPU serial matrix: 5 seeds x 2M = 10M by default.

Place next to:
  train_encoder.py
  diagnose_uagmc_candidate_temporal_mismatch.py
  diagnose_uagmc_effect_time_master.py

Run:
  python run_uagmc_gpu_10m_serial.py

What changes relative to local train_encoder.py:
  - PPO device -> cuda
  - PPO seed -> experiment seed
  - PPO.learn total_timesteps -> per-run budget
Everything else is left to train_encoder.py.

Passive training diagnostics every 50k:
  checkpoint model + VecNormalize, PPO scalars, stochastic action shares,
  deterministic probe on already-seen observations, collapse flags.

After each seed finishes training, selected saved checkpoints are evaluated by
our corrected effect-time Master Diagnostic. Those extra rollouts occur AFTER
training and therefore do not perturb that seed's training trajectory.
"""
from __future__ import annotations

import argparse, csv, json, math, os, random, runpy, subprocess, sys, time, traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_MAX_BUDGET = 10_000_000


def jdump(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    def conv(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, dict): return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [conv(v) for v in x]
        try: json.dumps(x); return x
        except Exception: return repr(x)
    path.write_text(json.dumps(conv(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def jload(path: Path, default=None):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return default


def append_csv(path: Path, row: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {}
    for k, v in row.items():
        if isinstance(v, (list, tuple, dict, np.ndarray)):
            serial[k] = json.dumps(v if not isinstance(v, np.ndarray) else v.tolist(), ensure_ascii=False)
        else:
            serial[k] = v
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(serial.keys()))
        if not exists: w.writeheader()
        w.writerow(serial); f.flush()


def read_csv(path: Path):
    if not path.exists(): return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sf(x, default=float("nan")):
    try: return float(x)
    except Exception: return default


def fmt(x, n=4):
    try:
        x = float(x)
        return f"{x:.{n}f}" if math.isfinite(x) else "-"
    except Exception: return "-"


def int_list(s: str):
    out = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not out: raise ValueError("empty integer list")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="train_encoder.py")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--steps-per-run", type=int, default=2_000_000)
    p.add_argument("--max-total-budget", type=int, default=DEFAULT_MAX_BUDGET)
    p.add_argument("--diag-interval", type=int, default=50_000)
    p.add_argument("--effect-diag-steps", default="500000,1000000,1500000,2000000")
    p.add_argument("--effect-diag-script", default="diagnose_uagmc_effect_time_master.py")
    p.add_argument("--no-effect-diagnostic", action="store_true")
    p.add_argument("--output-root", default=None)
    p.add_argument("--resume-root", default=None)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--worker-seed", type=int)
    p.add_argument("--worker-run-dir")
    return p.parse_args()


def worker(a):
    import torch, stable_baselines3
    from stable_baselines3 import PPO as OriginalPPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize

    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requested but torch.cuda.is_available() is False")
    if a.worker_seed is None or not a.worker_run_dir:
        raise ValueError("worker requires --worker-seed and --worker-run-dir")

    seed = int(a.worker_seed)
    run_dir = Path(a.worker_run_dir).resolve(); run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "checkpoints"; ckpt.mkdir(exist_ok=True)
    source = (ROOT / a.source).resolve()
    if not source.exists(): raise FileNotFoundError(source)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    class Diag(BaseCallback):
        def __init__(self):
            super().__init__(verbose=1)
            self.next = int(a.diag_interval)
            self.recent = deque(maxlen=4096)
            self.counts = None; self.total = 0
            self.csv = run_dir / "training_diagnostics.csv"

        def _on_training_start(self):
            n = getattr(self.model.action_space, "n", None)
            if n is not None: self.counts = np.zeros(int(n), dtype=np.int64)
            jdump(run_dir / "runtime_config.json", {
                "seed": seed, "device": str(self.model.device),
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "torch_cuda": torch.version.cuda, "sb3": stable_baselines3.__version__,
                "steps_per_run": int(a.steps_per_run), "diag_interval": int(a.diag_interval),
                "n_envs": int(getattr(self.training_env, "num_envs", 1)),
                "n_steps": int(getattr(self.model, "n_steps", -1)),
                "batch_size": int(getattr(self.model, "batch_size", -1)),
                "n_epochs": int(getattr(self.model, "n_epochs", -1)),
                "gamma": sf(getattr(self.model, "gamma", float("nan"))),
                "gae_lambda": sf(getattr(self.model, "gae_lambda", float("nan"))),
                "policy": type(self.model.policy).__name__,
                "extractor": type(self.model.policy.features_extractor).__name__,
            })

        def logv(self, k): return getattr(self.model.logger, "name_to_value", {}).get(k, float("nan"))

        def record(self):
            acts = self.locals.get("actions")
            if acts is not None and self.counts is not None:
                x = np.asarray(acts).reshape(-1).astype(np.int64)
                x = x[(x >= 0) & (x < len(self.counts))]
                if x.size:
                    self.counts += np.bincount(x, minlength=len(self.counts))[:len(self.counts)]
                    self.total += int(x.size)
            obs = self.locals.get("new_obs")
            if isinstance(obs, np.ndarray) and obs.dtype != object:
                for i in range(obs.shape[0]): self.recent.append(np.asarray(obs[i]).copy())

        def probe(self):
            out = {"det_probe_n":0,"det_action_counts":[],"det_action_shares":[],
                   "det_top_share":float("nan"),"prob_margin_mean":float("nan"),
                   "prob_entropy_mean":float("nan")}
            if not self.recent: return out
            try: obs = np.stack(list(self.recent))
            except Exception: return out
            pp, aa = [], []
            try:
                with torch.no_grad():
                    for s in range(0, len(obs), 512):
                        t = torch.as_tensor(obs[s:s+512], dtype=torch.float32, device=self.model.device)
                        d = self.model.policy.get_distribution(t).distribution
                        p = d.probs.detach().cpu().numpy(); pp.append(p); aa.append(np.argmax(p, axis=1))
                p = np.concatenate(pp); act = np.concatenate(aa).astype(np.int64)
                c = np.bincount(act, minlength=p.shape[1]); sh = c / max(1, int(c.sum()))
                ps = np.sort(p, axis=1); margin = ps[:,-1]-ps[:,-2] if p.shape[1]>=2 else np.ones(len(p))
                ent = -np.sum(p*np.log(np.clip(p,1e-12,1.0)), axis=1)
                out.update(det_probe_n=len(act), det_action_counts=c.tolist(), det_action_shares=sh.tolist(),
                           det_top_share=float(np.max(sh)), prob_margin_mean=float(np.mean(margin)),
                           prob_entropy_mean=float(np.mean(ent)))
            except Exception as e: out["probe_error"] = repr(e)
            return out

        def save_vec(self, path):
            try: vn = self.model.get_vec_normalize_env()
            except Exception: vn = None
            if vn is not None: vn.save(str(path)); return True
            cur = self.training_env
            for _ in range(8):
                if isinstance(cur, VecNormalize): cur.save(str(path)); return True
                cur = getattr(cur, "venv", None)
                if cur is None: break
            return False

        def checkpoint(self, target):
            pr = self.probe()
            counts = self.counts.copy() if self.counts is not None else np.zeros(0,dtype=np.int64)
            shares = counts/max(1,int(counts.sum())) if counts.size else np.zeros(0)
            stem = ckpt / f"model_{target:07d}_steps"; self.model.save(str(stem))
            vp = ckpt / f"vecnormalize_{target:07d}_steps.pkl"; vs = self.save_vec(vp)
            kl=sf(self.logv("train/approx_kl")); cf=sf(self.logv("train/clip_fraction")); ev=sf(self.logv("train/explained_variance"))
            flags=[]
            if math.isfinite(sf(pr.get("det_top_share"))) and sf(pr.get("det_top_share"))>=0.98: flags.append("DET_COLLAPSE_GE_98PCT")
            if math.isfinite(sf(pr.get("prob_entropy_mean"))) and sf(pr.get("prob_entropy_mean"))<=0.10: flags.append("DET_ENTROPY_LE_0.10")
            if math.isfinite(cf) and cf>=0.30: flags.append("CLIP_FRACTION_GE_0.30")
            if math.isfinite(kl) and kl>=0.05: flags.append("KL_GE_0.05")
            row={"seed":seed,"target_steps":target,"actual_steps":int(self.num_timesteps),
                 "ppo_updates":int(getattr(self.model,"_n_updates",-1)),
                 "learning_rate":float(self.model.policy.optimizer.param_groups[0]["lr"]),
                 "approx_kl":kl,"clip_fraction":cf,"explained_variance":ev,
                 "policy_gradient_loss":sf(self.logv("train/policy_gradient_loss")),
                 "value_loss":sf(self.logv("train/value_loss")),"entropy_loss":sf(self.logv("train/entropy_loss")),
                 "train_action_counts":counts.tolist(),"train_action_shares":shares.tolist(),"interval_action_n":self.total,
                 **pr,"flags":flags,"model_path":str(stem.with_suffix(".zip")),
                 "vecnormalize_path":str(vp) if vs else ""}
            append_csv(self.csv,row)
            print("\n"+"="*120)
            print(f"TRAIN DIAG seed={seed} target={target:,} actual={int(self.num_timesteps):,}")
            print(f"PPO KL={fmt(kl,6)} clipF={fmt(cf)} EV={fmt(ev)} lr={row['learning_rate']:.3e}")
            print(f"STOCH share={[round(float(x),4) for x in shares.tolist()]}")
            print(f"DET   share={[round(float(x),4) for x in pr.get('det_action_shares',[])]} H={fmt(pr.get('prob_entropy_mean'))} margin={fmt(pr.get('prob_margin_mean'))}")
            print(f"FLAGS {flags if flags else ['OK']}")
            print("="*120+"\n", flush=True)
            if self.counts is not None: self.counts[:] = 0
            self.total = 0

        def _on_step(self):
            self.record()
            while int(self.num_timesteps) >= self.next:
                self.checkpoint(self.next); self.next += int(a.diag_interval)
            return True

    class WrappedPPO(OriginalPPO):
        def __init__(self,*x,**kw):
            kw["device"]="cuda"; kw["seed"]=seed
            super().__init__(*x,**kw)
            print(f"[SERIAL] seed={seed} PPO device={self.device}", flush=True)
        def learn(self,total_timesteps,callback=None,*x,**kw):
            print(f"[SERIAL] source requested {int(total_timesteps):,}; override -> {int(a.steps_per_run):,}", flush=True)
            d=Diag()
            cb=d if callback is None else (list(callback)+[d] if isinstance(callback,(list,tuple)) else [callback,d])
            return super().learn(total_timesteps=int(a.steps_per_run),callback=cb,*x,**kw)

    jdump(run_dir/"run_request.json", {"seed":seed,"source":str(source),"steps":int(a.steps_per_run),
          "device":"cuda","diag_interval":int(a.diag_interval),"started":datetime.now().isoformat(),
          "controlled_overrides":["device=cuda",f"seed={seed}",f"total_timesteps={int(a.steps_per_run)}"]})

    old = stable_baselines3.PPO; status="success"; err=None; t0=time.perf_counter()
    try:
        stable_baselines3.PPO = WrappedPPO
        argv=sys.argv[:]; sys.argv=[str(source)]
        try: runpy.run_path(str(source),run_name="__main__")
        finally: sys.argv=argv
    except BaseException:
        status="error"; err=traceback.format_exc(); raise
    finally:
        stable_baselines3.PPO=old
        jdump(run_dir/"worker_end.json",{"seed":seed,"status":status,"wall_seconds":time.perf_counter()-t0,"error":err,"ended":datetime.now().isoformat()})
    return 0


def latest_summary(base: Path):
    hits=list(base.rglob("summary.json")) if base.exists() else []
    return max(hits,key=lambda p:p.stat().st_mtime) if hits else None


def effect_diags(root: Path, run_dir: Path, seed: int, steps: List[int], script_name: str):
    script=(ROOT/script_name).resolve()
    if not script.exists(): print(f"[WARN] no {script}"); return
    for st in steps:
        m=run_dir/"checkpoints"/f"model_{st:07d}_steps.zip"
        v=run_dir/"checkpoints"/f"vecnormalize_{st:07d}_steps.pkl"
        if not (m.exists() and v.exists()): print(f"[WARN] missing checkpoint seed={seed} step={st}"); continue
        out=run_dir/"effect_time_diagnostics"/f"step_{st:07d}"; out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(script),"--project-root",str(ROOT),"--uagmc-root",str(ROOT),"--model",str(m),"--vecnorm",str(v),"--device","cuda","--output-dir",str(out)]
        print(f"\n[EFFECT DIAG] seed={seed} step={st:,}",flush=True)
        p=subprocess.run(cmd,cwd=str(ROOT),check=False)
        sp=latest_summary(out); s=jload(sp,{}) if sp else {}
        stale=s.get("oracle_effect_time_staleness",{}) if isinstance(s,dict) else {}
        delay=s.get("same_passenger_candidate_delay",{}) if isinstance(s,dict) else {}
        legal=s.get("legal_committed_event_evidence",{}) if isinstance(s,dict) else {}
        gate=s.get("gate",{}) if isinstance(s,dict) else {}
        append_csv(root/"effect_time_trained_summary.csv",{"seed":seed,"step":st,"returncode":p.returncode,
            "gate_status":gate.get("status"),"gate_pass":gate.get("overall_pass"),"mean_delay_spread":delay.get("mean_spread"),
            "safe_change_rate":stale.get("safe_any_change_rate"),"rho_access_safe_drift":stale.get("rho_access_vs_safe_drift"),
            "mean_safe_drift":stale.get("mean_safe_drift_now_to_own"),"committed_crossing_rate":legal.get("positive_crossing_rate"),
            "summary_json":str(sp) if sp else ""})


def parent(a):
    seeds=int_list(a.seeds); total=len(seeds)*int(a.steps_per_run)
    if total>int(a.max_total_budget):
        raise RuntimeError(f"budget exceeded: {len(seeds)} x {a.steps_per_run:,} = {total:,} > {a.max_total_budget:,}")
    if not (ROOT/a.source).exists(): raise FileNotFoundError(ROOT/a.source)
    if a.resume_root: exp=Path(a.resume_root).resolve()
    elif a.output_root: exp=(ROOT/a.output_root).resolve() if not Path(a.output_root).is_absolute() else Path(a.output_root).resolve()
    else: exp=ROOT/"serial_runs"/f"uagmc_gpu_10m_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp.mkdir(parents=True,exist_ok=True)
    ed=[x for x in int_list(a.effect_diag_steps) if 0<x<=int(a.steps_per_run)]
    jdump(exp/"experiment_manifest.json",{"seeds":seeds,"steps_per_run":a.steps_per_run,"total_budget":total,
          "max_budget":a.max_total_budget,"device":"cuda","serial":True,"diag_interval":a.diag_interval,
          "effect_diag_steps":ed,"source":str((ROOT/a.source).resolve())})
    print("="*120); print("UAGMC GPU SERIAL MATRIX")
    print(f"seeds={seeds} | steps/seed={a.steps_per_run:,} | total={total:,}/{a.max_total_budget:,}")
    print(f"train diag every {a.diag_interval:,}; effect diag steps={ed if not a.no_effect_diagnostic else 'OFF'}")
    print(f"output={exp}"); print("="*120)
    failed=False
    for i,seed in enumerate(seeds,1):
        rd=exp/f"seed_{seed}"; end=jload(rd/"worker_end.json",{})
        if not (isinstance(end,dict) and end.get("status")=="success"):
            rd.mkdir(parents=True,exist_ok=True)
            env=os.environ.copy(); env["PYTHONHASHSEED"]=str(seed); env.setdefault("CUDA_VISIBLE_DEVICES","0")
            cmd=[sys.executable,str(Path(__file__).resolve()),"--worker","--source",a.source,
                 "--steps-per-run",str(a.steps_per_run),"--diag-interval",str(a.diag_interval),
                 "--worker-seed",str(seed),"--worker-run-dir",str(rd)]
            print(f"\n### RUN {i}/{len(seeds)} seed={seed} ###",flush=True)
            t0=time.perf_counter(); p=subprocess.run(cmd,cwd=str(ROOT),env=env,check=False); wall=time.perf_counter()-t0
            status="success" if p.returncode==0 and jload(rd/"worker_end.json",{}).get("status")=="success" else "error"
            append_csv(exp/"serial_status.csv",{"order":i,"seed":seed,"steps":a.steps_per_run,"status":status,"returncode":p.returncode,"wall_seconds":wall,"run_dir":str(rd)})
            if status!="success":
                failed=True
                if not a.continue_on_error: return 1
                continue
        else: print(f"[SKIP] seed={seed} already success",flush=True)
        if not a.no_effect_diagnostic:
            done=rd/"effect_time_done.json"
            if not done.exists():
                effect_diags(exp,rd,seed,ed,a.effect_diag_script)
                jdump(done,{"seed":seed,"steps":ed,"done":datetime.now().isoformat()})
            else: print(f"[SKIP] seed={seed} effect diagnostics done",flush=True)
    # final checkpoint summary
    out=exp/"final_training_summary.csv"
    if out.exists(): out.unlink()
    for seed in seeds:
        rows=read_csv(exp/f"seed_{seed}"/"training_diagnostics.csv")
        if not rows: continue
        r=rows[-1]
        append_csv(out,{"seed":seed,"target_steps":r.get("target_steps"),"actual_steps":r.get("actual_steps"),
            "approx_kl":r.get("approx_kl"),"clip_fraction":r.get("clip_fraction"),"explained_variance":r.get("explained_variance"),
            "det_action_shares":r.get("det_action_shares"),"det_top_share":r.get("det_top_share"),
            "prob_margin_mean":r.get("prob_margin_mean"),"prob_entropy_mean":r.get("prob_entropy_mean"),"flags":r.get("flags")})
    jdump(exp/"experiment_end.json",{"status":"completed_with_errors" if failed else "success","total_budget":total,"ended":datetime.now().isoformat()})
    print(f"\nDONE: {exp}")
    return 1 if failed else 0


def main():
    a=parse_args()
    return worker(a) if a.worker else parent(a)

if __name__=="__main__":
    raise SystemExit(main())
