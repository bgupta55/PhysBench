"""
train_resnet20.py — Train ResNet-20 on FULL CIFAR-10 (Experiment B).

Usage:
    python train_resnet20.py [--seeds 0 1 2 3 4 5 6 7 8 9] [--out_dir PATH]
    python train_resnet20.py --epochs 100 --seeds 0 1 2 3 4 5 6 7 8 9

Fix (Phase 1.3): Added standard CIFAR-10 augmentation — RandomCrop(32, padding=4)
and RandomHorizontalFlip — applied per-batch during training.  Without these the
model lands at ~85.3% (our original result); with them it should reach 90–92%,
matching the canonical He et al. (2016) ResNet-20/CIFAR-10 result of ~91.25%.
The augmentation is applied on CPU to the batch tensor slice before forward pass;
no changes to the data loading or evaluation path are needed.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import ResNet20
from data_utils import get_data

EPOCHS_DEFAULT = 100
BATCH          = 128

# Use MPS (Apple Silicon GPU) if available, else CPU
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
print(f"[train_resnet20] device={DEVICE}")


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Apply standard CIFAR-10 training augmentation in-place on a CPU tensor batch.

    Augmentations (matching the canonical He et al. recipe):
      - RandomCrop(32, padding=4): pad 4px on each side, then random 32×32 crop
      - RandomHorizontalFlip(p=0.5)

    x: float32 tensor of shape [N, 3, 32, 32], already normalised.
    Returns an augmented copy (original is not modified).
    """
    x = x.clone()
    N = x.shape[0]

    # Pad 4px on each side: [N, 3, 32, 32] → [N, 3, 40, 40]
    x_pad = F.pad(x, (4, 4, 4, 4), mode="reflect")

    for i in range(N):
        # Random crop: pick top-left corner uniformly from [0, 8] × [0, 8]
        top  = torch.randint(0, 9, (1,)).item()
        left = torch.randint(0, 9, (1,)).item()
        x[i] = x_pad[i, :, top:top + 32, left:left + 32]
        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            x[i] = x[i].flip(-1)

    return x


def train_one_resnet20(seed, Xtr, ytr, Xte, yte, epochs):
    torch.manual_seed(seed)
    model = ResNet20().to(DEVICE)
    # Keep training data on CPU for augmentation; move eval data to device
    Xte_d, yte_d = Xte.to(DEVICE), yte.to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 80, 90], gamma=0.1)
    lossf = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        idx = torch.randperm(len(Xtr))
        ep_loss = 0.0
        for i in range(0, len(Xtr), BATCH):
            b    = idx[i:i + BATCH]
            # Augment on CPU, then move to device
            xb   = augment_batch(Xtr[b]).to(DEVICE)
            yb   = ytr[b].to(DEVICE)
            opt.zero_grad()
            out  = model(xb.contiguous())
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        scheduler.step()
        if (ep + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xte_d).argmax(1) == yte_d).float().mean().item()
            model.train()
            print(f"  seed {seed}  ep {ep+1}/{epochs}  loss={ep_loss:.3f}  val_acc={acc:.4f}")

    model.eval()
    with torch.no_grad():
        acc = (model(Xte_d).argmax(1) == yte_d).float().mean().item()
    return model.cpu(), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--epochs",  type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument("--force",   action="store_true",
                        help="Retrain even if checkpoint already exists (Phase 1.3 fix)")
    args = parser.parse_args()

    ckpt_dir    = os.path.join(args.out_dir, "ckpt")
    results_dir = os.path.join(args.out_dir, "results", "resnet20")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Full CIFAR-10
    Xtr, ytr, Xte, yte = get_data(train_per_class=5000, test_per_class=1000, seed=0)
    print(f"Loaded full CIFAR-10: train={len(Xtr)} test={len(Xte)}")

    log = {}
    t0  = time.time()
    for s in args.seeds:
        ckpt_path = os.path.join(ckpt_dir, f"resnet20_seed{s}.pt")
        if os.path.exists(ckpt_path) and not getattr(args, 'force', False):
            m = ResNet20()
            m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            m.eval()
            with torch.no_grad():
                acc = (m(Xte).argmax(1) == yte).float().mean().item()
            log[str(s)] = {"clean_test_acc": acc, "train_time_s": 0.0, "cached": True}
            print(f"seed {s}: already cached — clean_test_acc={acc:.4f}")
            continue

        print(f"\nTraining ResNet-20 seed {s} ({args.epochs} epochs)…")
        t1 = time.time()
        model, acc = train_one_resnet20(s, Xtr, ytr, Xte, yte, args.epochs)
        torch.save(model.state_dict(), ckpt_path)
        elapsed = time.time() - t1
        log[str(s)] = {"clean_test_acc": acc, "train_time_s": elapsed}
        print(f"seed {s}: clean_test_acc={acc:.4f}  ({elapsed:.1f}s)")

    log["total_time_s"] = time.time() - t0
    out_path = os.path.join(results_dir, "baseline_resnet20.json")
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved baseline_resnet20.json  (total {log['total_time_s']:.1f}s)")


if __name__ == "__main__":
    main()
