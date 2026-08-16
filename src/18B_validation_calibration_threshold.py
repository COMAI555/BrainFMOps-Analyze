#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation-only calibration and threshold locking for BrainFMOps."""
import argparse, hashlib, json, logging, os, random, sys
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0

REQ={"image_id","subject_id","relative_path","class_label","binary_label","partition","is_valid_image"}

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def parse_bool(s):
    return s if pd.api.types.is_bool_dtype(s) else s.astype(str).str.lower().str.strip().isin({"true","1","yes","y"})

def load_manifest(path,root):
    df=pd.read_csv(path); missing=sorted(REQ-set(df.columns))
    if missing: raise ValueError("Missing columns: "+", ".join(missing))
    df["partition"]=df["partition"].astype(str).str.lower().str.strip()
    if (df["partition"]!="validation").any(): raise ValueError("Manifest contains non-validation rows")
    df["binary_label"]=pd.to_numeric(df["binary_label"],errors="coerce")
    df["is_valid_image"]=parse_bool(df["is_valid_image"])
    df=df[df["binary_label"].isin([0,1]) & df["is_valid_image"]].copy(); df["binary_label"]=df["binary_label"].astype(int)
    df["absolute_path"]=df["relative_path"].map(lambda x:str((Path(x) if Path(x).is_absolute() else root/Path(x)).resolve()))
    bad=~df["absolute_path"].map(lambda x:Path(x).is_file())
    if bad.any(): raise FileNotFoundError(f"{int(bad.sum())} images are missing")
    return df.reset_index(drop=True)

class DS(Dataset):
    def __init__(self,df,size):
        self.df=df; self.tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r["absolute_path"]) as im: x=self.tf(im.convert("RGB"))
        return {"image":x,"label":torch.tensor(int(r["binary_label"])),"subject_id":str(r["subject_id"]),"image_id":str(r["image_id"]),"class_label":str(r["class_label"]),"relative_path":str(r["relative_path"])}

def build_model(dropout):
    m=efficientnet_b0(weights=None); n=m.classifier[1].in_features; m.classifier=nn.Sequential(nn.Dropout(dropout),nn.Linear(n,2)); return m

def load_model(path,device):
    c=torch.load(path,map_location=device,weights_only=False); d=float(c.get("training_config",{}).get("dropout",0.30)); m=build_model(d); m.load_state_dict(c["model_state_dict"]); m.to(device).eval(); return m,c

def infer(model,loader,device):
    rows=[]
    with torch.inference_mode():
        for b,batch in enumerate(loader,1):
            p=torch.softmax(model(batch["image"].to(device,non_blocking=True)),1)[:,1].cpu().numpy()
            for i,v in enumerate(p): rows.append({"image_id":batch["image_id"][i],"subject_id":batch["subject_id"][i],"original_class":batch["class_label"][i],"true_label":int(batch["label"][i]),"probability_dementia":float(v),"relative_path":batch["relative_path"][i]})
            if b%100==0: logging.info("Processed %d batches",b)
    return pd.DataFrame(rows)

def metric_row(y,p,t):
    pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); sens=tp/(tp+fn); spec=tn/(tn+fp)
    return {"threshold":float(t),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"sensitivity":float(sens),"specificity":float(spec),"f1":float(f1_score(y,pred,zero_division=0)),"youden_j":float(sens+spec-1),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-root",type=Path,required=True); ap.add_argument("--validation-manifest",type=Path,required=True); ap.add_argument("--checkpoint",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--image-size",type=int,default=224); ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--num-workers",type=int,default=4); ap.add_argument("--bins",type=int,default=10); ap.add_argument("--target-sensitivity",type=float,default=0.80); ap.add_argument("--random-seed",type=int,default=42); args=ap.parse_args()
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(out/"calibration_console.log",encoding="utf-8")],force=True)
    random.seed(args.random_seed); np.random.seed(args.random_seed); torch.manual_seed(args.random_seed)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); logging.info("Device: %s",device)
    df=load_manifest(args.validation_manifest.resolve(),args.dataset_root.resolve())
    loader=DataLoader(DS(df,args.image_size),batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=torch.cuda.is_available(),persistent_workers=args.num_workers>0)
    model,ckpt=load_model(args.checkpoint.resolve(),device); img=infer(model,loader,device)
    consistency=img.groupby("subject_id")["true_label"].nunique();
    if (consistency>1).any(): raise ValueError("Inconsistent labels within subject")
    subj=img.groupby("subject_id",as_index=False).agg(true_label=("true_label","first"),original_class=("original_class","first"),image_count=("image_id","count"),probability_dementia=("probability_dementia","mean"),median_probability=("probability_dementia","median"))
    y=subj["true_label"].to_numpy(int); p=subj["probability_dementia"].to_numpy(float)
    table=pd.DataFrame([metric_row(y,p,t) for t in np.arange(0.05,0.951,0.01)])
    selected={"youden_j":table.loc[table["youden_j"].idxmax()].to_dict(),"maximum_f1":table.loc[table["f1"].idxmax()].to_dict(),"maximum_balanced_accuracy":table.loc[table["balanced_accuracy"].idxmax()].to_dict()}
    elig=table[table["sensitivity"]>=args.target_sensitivity]
    if not elig.empty: selected[f"sensitivity_at_least_{args.target_sensitivity:.2f}"]=elig.sort_values(["specificity","balanced_accuracy","threshold"],ascending=[False,False,False]).iloc[0].to_dict()
    selected["default_0_50"]=table.iloc[(table["threshold"]-0.5).abs().argmin()].to_dict()
    edges=np.linspace(0,1,args.bins+1); ids=np.digitize(p,edges[1:-1],right=True); bins=[]; ece=0.; mce=0.
    for i in range(args.bins):
        m=ids==i
        if not m.any(): bins.append({"bin":i+1,"lower":edges[i],"upper":edges[i+1],"count":0,"mean_probability":np.nan,"observed_frequency":np.nan,"gap":np.nan}); continue
        conf=float(p[m].mean()); obs=float(y[m].mean()); gap=abs(conf-obs); ece+=m.mean()*gap; mce=max(mce,gap); bins.append({"bin":i+1,"lower":edges[i],"upper":edges[i+1],"count":int(m.sum()),"mean_probability":conf,"observed_frequency":obs,"gap":gap})
    eps=1e-6; clipped=np.clip(p,eps,1-eps); logits=np.log(clipped/(1-clipped)).reshape(-1,1); lr=LogisticRegression(penalty=None,solver="lbfgs",max_iter=1000).fit(logits,y)
    locked={k:float(v["threshold"]) for k,v in selected.items()}
    report={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"scientific_rule":"Thresholds selected using validation data only","device":str(device),"checkpoint_epoch":ckpt.get("epoch"),"validation_images":len(img),"validation_subjects":len(subj),"roc_auc":float(roc_auc_score(y,p)),"pr_auc":float(average_precision_score(y,p)),"brier_score":float(brier_score_loss(y,p)),"ece":float(ece),"mce":float(mce),"calibration_slope":float(lr.coef_[0,0]),"calibration_intercept":float(lr.intercept_[0]),"selected_threshold_details":selected,"locked_thresholds":locked,"checkpoint_sha256":sha256_file(args.checkpoint.resolve()),"validation_manifest_sha256":sha256_file(args.validation_manifest.resolve())}
    img.to_csv(out/"validation_image_predictions.csv",index=False,encoding="utf-8-sig"); subj.to_csv(out/"validation_subject_predictions.csv",index=False,encoding="utf-8-sig"); table.to_csv(out/"validation_threshold_table.csv",index=False,encoding="utf-8-sig"); bdf=pd.DataFrame(bins); bdf.to_csv(out/"calibration_bins.csv",index=False,encoding="utf-8-sig")
    (out/"calibration_threshold_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"); (out/"locked_thresholds.json").write_text(json.dumps({"source":"validation_set_only","thresholds":locked},indent=2),encoding="utf-8")
    vb=bdf.dropna(); plt.figure(figsize=(6,5)); plt.plot(vb["mean_probability"],vb["observed_frequency"],marker="o",label="Model"); plt.plot([0,1],[0,1],"--",label="Perfect"); plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency"); plt.title("Validation Subject-level Reliability Diagram"); plt.legend(); plt.tight_layout(); plt.savefig(out/"subject_level_reliability_diagram.png",dpi=300); plt.close()
    plt.figure(figsize=(8,5));
    for c in ["sensitivity","specificity","balanced_accuracy","f1"]: plt.plot(table["threshold"],table[c],label=c)
    plt.xlabel("Threshold"); plt.ylabel("Metric"); plt.title("Validation Threshold Performance"); plt.legend(); plt.tight_layout(); plt.savefig(out/"validation_threshold_performance.png",dpi=300); plt.close()
    print("\n"+"="*88); print("BRAINF MOPS VALIDATION CALIBRATION COMPLETE"); print("="*88); print(f"Device               : {device}"); print(f"Validation subjects  : {len(subj)}"); print(f"ROC-AUC              : {report['roc_auc']:.4f}"); print(f"PR-AUC               : {report['pr_auc']:.4f}"); print(f"Brier score          : {report['brier_score']:.4f}"); print(f"ECE                  : {ece:.4f}"); print(f"Calibration slope    : {report['calibration_slope']:.4f}"); print(f"Calibration intercept: {report['calibration_intercept']:.4f}")
    for k,v in locked.items(): print(f"{k:24s}: {v:.4f}")
    print(f"Output directory     : {out}")
    return 0
if __name__=="__main__": raise SystemExit(main())
