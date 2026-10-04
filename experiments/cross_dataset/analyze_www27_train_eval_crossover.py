#!/usr/bin/env python3
"""Cross training geometry with inference geometry using fixed checkpoints."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import torch
import torch.nn.functional as F
from analyze_group_metrics import build_model

TRAIN = {"dot": "SASRec", "joint": "GeometrySASRec", "stopgrad_joint": "StopGradGeometrySASRec"}

def score(seq, items, mode):
    if mode in {"sequence", "joint"}: seq = F.normalize(seq, dim=-1)
    if mode in {"item", "joint"}: items = F.normalize(items, dim=-1)
    return seq @ items.T

def evaluate(model, test_data, mode):
    device = next(model.parameters()).device; hits10 = ndcg10 = hits20 = ndcg20 = total = 0
    with torch.no_grad():
        for batched in test_data:
            interaction, history, users, positives = batched[0].to(device), batched[1], torch.as_tensor(batched[2],device=device).long(), torch.as_tensor(batched[3],device=device).long()
            logits = score(model.forward(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN]), model.item_embedding.weight, mode)
            logits[:,0] = -torch.inf
            if history is not None: logits[history] = -torch.inf
            rank = (logits[users] > logits[users, positives].unsqueeze(1)).sum(1).float() + 1
            hits10 += (rank<=10).sum().item(); hits20 += (rank<=20).sum().item()
            ndcg10 += ((rank<=10).float()/torch.log2(rank+1)).sum().item(); ndcg20 += ((rank<=20).float()/torch.log2(rank+1)).sum().item(); total += len(rank)
    return {"recall@10":hits10/total,"ndcg@10":ndcg10/total,"recall@20":hits20/total,"ndcg@20":ndcg20/total}

def checkpoint(root, dataset, seed, variant):
    files=sorted((root/dataset/f"seed{seed}"/variant).glob("*.pth"))
    if not files: raise FileNotFoundError(f"missing {dataset} seed{seed} {variant}")
    return str(files[-1])

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--seeds",default="2023,2024,2025"); p.add_argument("--checkpoint_root",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--gpu_id",type=int,default=1); p.add_argument("--hidden_size",type=int,default=256); p.add_argument("--inner_size",type=int,default=1024); p.add_argument("--max_item_list_length",type=int,default=50); p.add_argument("--eval_batch_size",type=int,default=512); a=p.parse_args()
    a.n_layers=2; a.n_heads=2; a.hidden_dropout_prob=.5; a.attn_dropout_prob=.5; a.learning_rate=.001; a.train_batch_size=1024; a.temperature=10.; a.score_geometry="both"; a.sequence_norm_power=1.; a.item_norm_power=1.
    rows=[]
    for seed in map(int,a.seeds.split(',')):
      a.seed=seed
      for train_variant, model_name in TRAIN.items():
        _,_,_,_,test,model=build_model(model_name, checkpoint(a.checkpoint_root,a.dataset,seed,train_variant),a)
        for infer in ("dot","sequence","item","joint"): rows.append({"dataset":a.dataset,"seed":seed,"train_geometry":train_variant,"infer_geometry":infer,**evaluate(model,test,infer)})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader();w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")
if __name__=="__main__": main()
