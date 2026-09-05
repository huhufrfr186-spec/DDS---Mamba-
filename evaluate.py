"""Sequence-level OPE metrics and bootstrap confidence intervals.

Use this for development and to validate raw prediction files. For final
leaderboard claims, also run the benchmark owner's exact evaluator.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax0, ay0, ax1, ay1 = a[:,0],a[:,1],a[:,0]+a[:,2],a[:,1]+a[:,3]; bx0,by0,bx1,by1=b[:,0],b[:,1],b[:,0]+b[:,2],b[:,1]+b[:,3]
    inter=np.maximum(0,np.minimum(ax1,bx1)-np.maximum(ax0,bx0))*np.maximum(0,np.minimum(ay1,by1)-np.maximum(ay0,by0)); union=a[:,2]*a[:,3]+b[:,2]*b[:,3]-inter
    return inter/np.maximum(union,1e-6)


def sequence_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str,float]:
    n=min(len(pred),len(gt)); pred,gt=pred[:n],gt[:n]; overlap=iou_xywh(pred,gt); centre=np.linalg.norm((pred[:,:2]+pred[:,2:]/2)-(gt[:,:2]+gt[:,2:]/2),axis=1); scale=np.sqrt(np.maximum(gt[:,2]*gt[:,3],1e-6))
    return {"success":float(np.mean([np.mean(overlap>=t) for t in np.arange(0, 1.01, .05)])),"precision":float(np.mean(centre<=20)),"normalized_precision":float(np.mean(centre/scale<=.5))}


def bootstrap(values: np.ndarray, rounds: int, seed: int) -> tuple[float,float,float]:
    rng=np.random.default_rng(seed); mean=float(values.mean()); sample=rng.integers(0,len(values),(rounds,len(values))); means=values[sample].mean(1); lo,hi=np.quantile(means,[.025,.975]); return mean,float(lo),float(hi)


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--pred-dir",type=Path,required=True); p.add_argument("--gt-dir",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--bootstrap",type=int,default=10000); p.add_argument("--seed",type=int,default=20260711); a=p.parse_args(); rows=[]
    for pred_path in sorted(a.pred_dir.glob("*.txt")):
        gt_path=a.gt_dir/(pred_path.stem+".txt")
        if not gt_path.exists(): continue
        rows.append({"sequence":pred_path.stem,**sequence_metrics(np.loadtxt(pred_path,delimiter=","),np.loadtxt(gt_path,delimiter=","))})
    if not rows: raise FileNotFoundError("No matching prediction/ground-truth txt pairs")
    report={"sequences":rows,"aggregate":{}}
    for metric in ("success","precision","normalized_precision"):
        report["aggregate"][metric]=dict(zip(("mean","ci95_low","ci95_high"),bootstrap(np.array([r[metric] for r in rows]),a.bootstrap,a.seed)))
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2)+"\n")
if __name__ == "__main__": main()
