#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, concurrent.futures as cf, importlib.util, json, math, re, statistics, sys, threading, time, urllib.request
from pathlib import Path
from collections import Counter

def load_jsonl(path):
    out=[]
    with open(path,"r",encoding="utf-8") as f:
        for l in f:
            if l.strip(): out.append(json.loads(l))
    return out

def load_dataset(path):
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj,list):return obj
    if isinstance(obj,dict):
        for k in ["data","records","samples","spectra","items","dataset"]:
            if isinstance(obj.get(k),list):return obj[k]
        vals=list(obj.values())
        if vals and all(isinstance(x,dict) for x in vals):return vals
    raise ValueError("unsupported dataset")

def gid(x):
    for k in ["identifier","id","Identifier","spectrum_id","spec_id"]:
        if isinstance(x,dict) and x.get(k) is not None:return str(x[k])
    return None

def sf(x,d=None):
    try:return float(x)
    except:return d

def first(d,keys,default=None):
    for k in keys:
        if isinstance(d,dict) and d.get(k) is not None:return d[k]
    return default

def zmargin(row):
    z=sf(row.get("z_margin"))
    if z is not None:return z
    scores=row.get("preranker_scores") or row.get("scores") or []
    xs=[sf(x) for x in scores]; xs=[x for x in xs if x is not None]
    if len(xs)<2:return math.inf
    mu=statistics.fmean(xs); sd=math.sqrt(max(statistics.fmean((x-mu)**2 for x in xs),1e-12))
    return (xs[0]-xs[1])/sd

def cformula(c):return str(first(c,["formula","Formula","molecular_formula"],"") or "").replace(" ","")
def cadduct(c):return re.sub(r"\s+","",str(first(c,["best_adduct","BestAdduct","adduct"],"") or ""))
def cppm(c):return sf(first(c,["ppm","ppm_error","mass_error_ppm","abs_ppm"]))

def precursor(row):
    for obj in [row,row.get("metadata",{}),row.get("precursor",{})]:
        v=first(obj,["precursor_mz","precursorMz","mz","m/z"])
        if sf(v) is not None:return sf(v)
    return None

def polarity(row):
    for obj in [row,row.get("metadata",{})]:
        v=first(obj,["polarity","ion_mode","ionization_mode","mode"])
        if v is not None:
            s=str(v).strip().lower()
            if s in ["+","positive","pos","p"]:return "positive"
            if s in ["-","negative","neg","n"]:return "negative"
            return str(v)
    ad=str(row.get("metadata",{}).get("adduct","")).strip()
    if ad.endswith("+"):return "positive"
    if ad.endswith("-"):return "negative"
    return "unknown"

def peak_text(row,n):
    spec=row.get("spectrum",{})
    ps=[]
    mzs=spec.get("mzs"); ints=spec.get("intensities")
    if isinstance(mzs,list) and isinstance(ints,list):
        ps=[(sf(m),sf(i)) for m,i in zip(mzs,ints)]
        ps=[(m,i) for m,i in ps if m is not None and i is not None]
    if not ps:return "Unavailable"
    q=sorted(ps,key=lambda x:x[1],reverse=True)[:n]
    mx=max(i for _,i in q) or 1
    q=sorted(q)
    return ", ".join(f"{m:.4f}:{100*i/mx:.1f}" for m,i in q)

FIELDS=[
 ("Formula",["Formula","formula","molecular_formula"]),("BestAdduct",["BestAdduct","best_adduct","adduct"]),
 ("ppm",["ppm","ppm_error","mass_error_ppm","abs_ppm"]),("DBE",["DBE","dbe"]),
 ("V3Rank",["V3Rank","v3_rank"]),("OldRank",["OldRank","old_rank","OLDRank"]),
 ("DirectRank",["DirectRank","direct_rank"]),("FRank",["FRank","f_rank"]),("Support",["Support","support"])
]
def fmt(c,label):
    a=[label]
    for name,keys in FIELDS:
        v=first(c,keys)
        if v is not None and str(v)!="":a.append(f"{name}: {v}")
    return "\n".join(a)

SYSTEM="""你是一名LC-MS/MS中性分子式候选纠错专家。
前置XGBoost已经对候选进行了全局排序。你的任务不是重新做全局排序，而是判断是否有足够证据用XGB Rank2覆盖XGB Rank1。
如果证据不足、两个候选都可疑或无法明确支持Rank2，优先KEEP。只有Rank2证据明确更强时才SWAP。
不得查询数据库，不得生成新候选，不输出推理过程。只输出严格JSON：{"decision":"KEEP"}或{"decision":"SWAP"}。"""

def messages(ds,row,n):
    sl=row["shortlist"]; z=zmargin(row); pmz=precursor(ds)
    u=f"""Precursor m/z: {"Unavailable" if pmz is None else f"{pmz:.6f}"}
Polarity: {polarity(ds)}
XGB normalized margin z: {z:.6f}

MS2 peaks (m/z:relative_intensity, max=100):
{peak_text(ds,n)}

{fmt(sl[0],"XGB Rank1")}
{fmt(sl[1],"XGB Rank2")}

请判断是否应该用Rank2覆盖Rank1。证据不足时KEEP。只输出严格JSON。"""
    return [{"role":"system","content":SYSTEM},{"role":"user","content":u}]

def post(url,payload,timeout):
    req=urllib.request.Request(url,data=json.dumps(payload,ensure_ascii=False).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer EMPTY"},method="POST")
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())

def parse(content):
    s=(content or "").strip()
    for txt in [s]+(re.findall(r"\{.*?\}",s,re.S)[:1]):
        try:
            o=json.loads(txt); d=str(o.get("decision","")).upper()
            if d in ["KEEP","SWAP"]:return d
        except:pass
    return None

def infer(api,model,msg,timeout,retries):
    err=""
    for i in range(retries+1):
        try:
            o=post(api.rstrip("/")+"/chat/completions",
                   {"model":model,"messages":msg,"temperature":0,"max_tokens":32},timeout)
            raw=o["choices"][0]["message"].get("content","")
            d=parse(raw)
            return {"status":"ok" if d else "parse_error","decision":d or "KEEP","raw":raw}
        except Exception as e:
            err=repr(e); time.sleep(min(2**i,8))
    return {"status":"request_error","decision":"KEEP","error":err}

def collect(args):
    rows=load_jsonl(args.shortlist_jsonl); dm={gid(x):x for x in load_dataset(args.dataset) if gid(x)}
    done={}
    op=Path(args.output); op.parent.mkdir(parents=True,exist_ok=True)
    if op.exists() and not args.overwrite:
        for x in load_jsonl(op): done[gid(x)]=x
    jobs=[]
    for r in rows:
        sid=gid(r); z=zmargin(r)
        if sid and sid not in done and len(r.get("shortlist") or [])>=2 and math.isfinite(z) and z<args.max_gate and sid in dm:
            jobs.append((r,dm[sid]))
    print(f"rows={len(rows)} eligible={len(jobs)} done={len(done)} max_gate={args.max_gate}")
    lock=threading.Lock(); ctr=Counter(); start=time.time()
    def work(p):
        r,ds=p; q=infer(args.api_base,args.model,messages(ds,r,args.prompt_max_peaks),args.timeout,args.retries)
        return {"identifier":gid(r),"z_margin":zmargin(r),**q}
    with op.open("a" if op.exists() and not args.overwrite else "w",encoding="utf-8") as f:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i,res in enumerate(ex.map(work,jobs),1):
                ctr[res["status"]]+=1; ctr[res["decision"]]+=1
                with lock:f.write(json.dumps(res,ensure_ascii=False)+"\n");f.flush()
                if i%args.progress_every==0 or i==len(jobs):
                    print(f"[{i}/{len(jobs)}] {i/max(time.time()-start,1e-9):.2f}/s {dict(ctr)}",flush=True)

def load_base(path):
    sp=importlib.util.spec_from_file_location("base",path); m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m

def fvalid(base,formula):
    try:
        c=base.parse_formula(formula)
        if not c:return 0
        d=base.calc_dbe(c)
        return int(base.dbe_valid(d) and base.nitrogen_rule_valid(c) and base.ratio_valid(c))
    except:return 0

def metrics(rows,idxs,base,ppm5):
    N=len(rows); c=Counter()
    for r,idx in zip(rows,idxs):
        sl=r.get("shortlist") or []
        if not sl:continue
        order=list(range(len(sl)))
        if idx==1 and len(order)>1:order[0],order[1]=1,0
        gold=str(r.get("true_formula") or "").replace(" ","")
        preds=[cformula(sl[i]) for i in order]
        c["t1"]+=int(gold in preds[:1]);c["t3"]+=int(gold in preds[:3]);c["t5"]+=int(gold in preds[:5])
        top=sl[order[0]]; p=cppm(top)
        c["mv"]+=int(p is not None and abs(p)<=ppm5);c["fv"]+=fvalid(base,cformula(top))
        c["aa"]+=int(bool(r.get("true_adduct")) and cadduct(top)==re.sub(r"\s+","",str(r.get("true_adduct") or "")))
    return {"N":N,"Top-1 Accuracy":c["t1"]/N,"Top-3 Accuracy":c["t3"]/N,"Top-5 Accuracy":c["t5"]/N,
            "Mass-valid Rate":c["mv"]/N,"Formula Validity":c["fv"]/N,"Adduct Accuracy":c["aa"]/N}

def evaluate(rows,preds,t,base,ppm5,r2ppm):
    idx=[]; st=Counter()
    for r in rows:
        sl=r.get("shortlist") or []; z=zmargin(r); fi=0
        if len(sl)>=2 and math.isfinite(z) and z<t:
            st["gated"]+=1; p=preds.get(gid(r))
            if p:
                st["pred"]+=1; d=str(p.get("decision","KEEP")).upper()
                if d=="SWAP":
                    allow=True
                    if r2ppm is not None:
                        pp=cppm(sl[1]); allow=(pp is not None and abs(pp)<=r2ppm)
                    if allow:
                        fi=1; st["swaps"]+=1
                        gold=str(r.get("true_formula") or "").replace(" ","")
                        a=cformula(sl[0])==gold;b=cformula(sl[1])==gold
                        if (not a) and b:st["benefit"]+=1
                        elif a and (not b):st["harm"]+=1
                        else:st["neutral"]+=1
                    else:st["guard_block"]+=1
        idx.append(fi)
    b=metrics(rows,[0]*len(rows),base,ppm5); f=metrics(rows,idx,base,ppm5)
    return {"threshold":t,"baseline":b,"final":f,"delta_top1_pp":100*(f["Top-1 Accuracy"]-b["Top-1 Accuracy"]),**st}

def score(args):
    rows=load_jsonl(args.shortlist_jsonl); pred={gid(x):x for x in load_jsonl(args.predictions)}
    if args.expected_n and len(rows)!=args.expected_n:raise SystemExit(f"expected {args.expected_n}, got {len(rows)}")
    base=load_base(args.base_file)
    ts=[float(x) for x in args.thresholds.split(",")] if args.sweep else [args.gate_threshold]
    rr=[evaluate(rows,pred,t,base,args.mass_valid_ppm,args.r2_max_abs_ppm) for t in ts]
    print(f"{'thr':>6} {'gate':>6} {'swap':>6} {'benefit':>8} {'harm':>6} {'block':>6} {'Top1':>8} {'delta':>8}")
    for r in rr:
        print(f"{r['threshold']:6.3f} {r.get('gated',0):6d} {r.get('swaps',0):6d} {r.get('benefit',0):8d} {r.get('harm',0):6d} {r.get('guard_block',0):6d} {100*r['final']['Top-1 Accuracy']:8.2f} {r['delta_top1_pp']:8.3f}")
    best=max(rr,key=lambda r:(r["final"]["Top-1 Accuracy"],-r.get("harm",0),-r.get("swaps",0),-r["threshold"]))
    print("\nBEST:",best["threshold"])
    for lab,m in [("XGB",best["baseline"]),("XGB+LLM",best["final"])]:
        print(lab, {k:(round(v*100,2) if isinstance(v,float) and k!="N" else v) for k,v in m.items()})
    print("benefit/harm/neutral =",best.get("benefit",0),best.get("harm",0),best.get("neutral",0))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps({"best_threshold":best["threshold"],"best":best,"all":rr},ensure_ascii=False,indent=2),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    c=sub.add_parser("collect");c.add_argument("--shortlist-jsonl",required=True);c.add_argument("--dataset",required=True)
    c.add_argument("--api-base",default="http://127.0.0.1:19998/v1");c.add_argument("--model",required=True);c.add_argument("--max-gate",type=float,default=.30)
    c.add_argument("--prompt-max-peaks",type=int,default=30);c.add_argument("--workers",type=int,default=32);c.add_argument("--timeout",type=float,default=120)
    c.add_argument("--retries",type=int,default=3);c.add_argument("--progress-every",type=int,default=50);c.add_argument("--overwrite",action="store_true");c.add_argument("--output",required=True);c.set_defaults(func=collect)
    s=sub.add_parser("score");s.add_argument("--shortlist-jsonl",required=True);s.add_argument("--predictions",required=True);s.add_argument("--base-file",default="./eval_final_candidate_union_2000.py")
    s.add_argument("--expected-n",type=int);s.add_argument("--mass-valid-ppm",type=float,default=5);s.add_argument("--r2-max-abs-ppm",type=float,default=None)
    s.add_argument("--sweep",action="store_true");s.add_argument("--thresholds",default=".05,.075,.10,.125,.15,.175,.20,.25,.30");s.add_argument("--gate-threshold",type=float)
    s.add_argument("--output-json");s.set_defaults(func=score)
    a=ap.parse_args();a.func(a)
if __name__=="__main__":main()
