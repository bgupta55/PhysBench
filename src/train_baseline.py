"""
train_baseline.py — Train SmallCNN on CIFAR-10 (seeds 0-9).

Usage:
    python train_baseline.py [--seeds 0 1 2 3 4 5 6 7 8 9] [--out_dir PATH]

Seeds 0-4: original paper seeds.
Seeds 5-9: Experiment C (additional seeds for statistical power).
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from model import SmallCNN
from data_utils import get_data

EPOCHS = 25
BATCH  = 64

# Use MPS (Apple Silicon GPU) if available, else CPU
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
print(f"[train_baseline] device={DEVICE}")


def train_one(seed, Xtr, ytr, Xte, yte):
    torch.manual_seed(seed)
    model = SmallCNN().to(DEVICE)
    Xtr_d, ytr_d = Xtr.to(DEVICE), ytr.to(DEVICE)
    Xte_d, yte_d = Xte.to(DEVICE), yte.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for ep in range(EPOCHS):
        idx = torch.randperm(len(Xtr_d))
        for i in range(0, len(Xtr_d), BATCH):
            b = idx[i:i + BATCH]
            opt.zero_grad()
            out = model(Xtr_d[b].contiguous())
            loss = lossf(out, ytr_d[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte_d).argmax(1) == yte_d).float().mean().item()
    return model.cpu(), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), ".."))
    args = parser.parse_args()

    ckpt_dir    = os.path.join(args.out_dir, "ckpt")
    results_dir = os.path.join(args.out_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    Xtr, ytr, Xte, yte = get_data(train_per_class=300, test_per_class=100, seed=0)
    log = {}
    t0  = time.time()

    for s in args.seeds:
        ckpt_path = os.path.join(ckpt_dir, f"model_seed{s}.pt")
        if os.path.exists(ckpt_path):
            # load to get acc for logging
            m = SmallCNN()
            m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            m.eval()
            with torch.no_grad():
                acc = (m(Xte).argmax(1) == yte).float().mean().item()
            log[str(s)] = {"clean_test_acc": acc, "train_time_s": 0.0, "cached": True}
            print(f"seed {s}: already cached — clean_test_acc={acc:.4f}")
            continue

        t1 = time.time()
        model, acc = train_one(s, Xtr, ytr, Xte, yte)
        torch.save(model.state_dict(), ckpt_path)
        elapsed = time.time() - t1
        log[str(s)] = {"clean_test_acc": acc, "train_time_s": elapsed}
        print(f"seed {s}: clean_test_acc={acc:.4f}  ({elapsed:.1f}s)")

    log["total_time_s"] = time.time() - t0
    out_path = os.path.join(results_dir, "baseline.json")
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved baseline.json  (total {log['total_time_s']:.1f}s)")


if __name__ == "__main__":
    main()
