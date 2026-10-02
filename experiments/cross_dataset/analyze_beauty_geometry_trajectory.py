#!/usr/bin/env python3
"""Record training-time geometry diagnostics for controlled SASRec scores.

This is a mechanism probe, not a model-selection run.  It evaluates every
epoch on the validation split and records diagnostics on one *fixed training
batch*: the median raw sequence norm, the entropy of the training softmax, and
the fraction of the sequence-output gradient lying in the radial direction.
The latter is computed with ``autograd.grad`` and never updates the model.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

# The script is executed by path from ``experiments/cross_dataset``.  Add the
# repository root explicitly so local model modules resolve on every host.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_logger, init_seed

from models import get_model_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["dot", "joint"])
    parser.add_argument("--dataset", default="Beauty")
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def config_dict(args, model_name, temperature):
    return {
        "seed": args.seed,
        "reproducibility": True,
        "data_path": "./dataset/",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"], "item": ["item_id"]},
        "MAX_ITEM_LIST_LENGTH": 50,
        "train_batch_size": 1024,
        "eval_batch_size": 512,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        # Explicitly disable RecBole's default sampled-negative setting: the
        # controlled SASRec runs optimise full-softmax CE.
        "train_neg_sample_args": None,
        "epochs": args.epochs,
        "eval_step": 1,
        # We need the whole trajectory rather than an early-stopped endpoint.
        "stopping_step": args.epochs + 1,
        "show_progress": False,
        "verbose": False,
        "topk": [5, 10, 20],
        "metrics": ["Recall", "NDCG"],
        "valid_metric": "NDCG@10",
        "eval_args": {
            "split": {"LS": "valid_and_test"},
            "order": "TO",
            "group_by": "user",
            "mode": {"valid": "full", "test": "full"},
        },
        "checkpoint_dir": "./ckpt/beauty_geometry_trajectory",
        "use_gpu": torch.cuda.is_available(),
        "hidden_size": 256,
        "n_layers": 2,
        "n_heads": 2,
        "inner_size": 1024,
        "hidden_dropout_prob": 0.5,
        "attn_dropout_prob": 0.5,
        "loss_type": "CE",
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        "temperature": temperature,
        "score_geometry": "both",
        "sequence_norm_power": 1.0,
        "item_norm_power": 1.0,
    }


def logits_for(model, sequence_output):
    if hasattr(model, "_full_scores"):
        return model._full_scores(sequence_output)
    return torch.matmul(sequence_output, model.item_embedding.weight.transpose(0, 1))


def diagnostics(model, interaction):
    """Return raw norm, CE entropy and radial gradient energy on fixed data."""
    model.eval()
    item_seq = interaction[model.ITEM_SEQ]
    item_seq_len = interaction[model.ITEM_SEQ_LEN]
    pos_items = interaction[model.POS_ITEM_ID]

    sequence_output = model.forward(item_seq, item_seq_len)
    logits = logits_for(model, sequence_output)
    loss = model.loss_fct(logits, pos_items)
    gradient = torch.autograd.grad(loss, sequence_output, only_inputs=True)[0]
    raw_norm = torch.linalg.vector_norm(sequence_output, dim=-1)
    unit = sequence_output / raw_norm.unsqueeze(-1).clamp_min(1e-12)
    radial_component = (gradient * unit).sum(dim=-1).square()
    grad_energy = gradient.square().sum(dim=-1).clamp_min(1e-12)
    radial_fraction = (radial_component / grad_energy).mean()
    with torch.no_grad():
        probability = torch.softmax(logits, dim=-1)
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return {
        "sequence_norm_median": raw_norm.median().item(),
        "sequence_norm_mean": raw_norm.mean().item(),
        "radial_gradient_fraction": radial_fraction.item(),
        "softmax_entropy": entropy.item(),
    }


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    model_name = "SASRec" if args.variant == "dot" else "GeometrySASRec"
    temperature = 1.0 if args.variant == "dot" else 10.0
    config = Config(model=get_model_class(model_name), dataset=args.dataset,
                    config_dict=config_dict(args, model_name, temperature))
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)
    model = get_model_class(model_name)(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], model_name)(config, model)

    # The batch is captured before optimisation and is never used as an update.
    fixed_interaction = next(iter(train_data)).to(config["device"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "seed", "epoch", "valid_ndcg_at_10",
                  "sequence_norm_median", "sequence_norm_mean",
                  "radial_gradient_fraction", "softmax_entropy"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        def callback(epoch, valid_score):
            row = {"variant": args.variant, "seed": args.seed, "epoch": epoch + 1,
                   "valid_ndcg_at_10": float(valid_score)}
            row.update(diagnostics(model, fixed_interaction))
            writer.writerow(row)
            handle.flush()
            print(
                "TRAJECTORY " + " ".join(
                    f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in row.items()
                ),
                flush=True,
            )

        trainer.fit(train_data, valid_data, verbose=False, saved=False,
                    show_progress=False, callback_fn=callback)


if __name__ == "__main__":
    main()
