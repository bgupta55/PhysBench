"""
Blueprint v3 §8 — Study A: Real-Model Training + PBS Sweep
Re-trains SmallCNN and ResNet-20 on CIFAR-10 (10 seeds each) and runs the full
PBS harness (Tables 3/4, random control, clip sweep, hparam sweep, stats).

This script REPLACES the SyntheticModel results with real classification ASR
(misclassification-rate metric, same as the original paper's Table 3).

MPS Bug Fix (v4):
  Gradient computation is done entirely on CPU via a CPU-clone of model_work.
  This completely sidesteps the MPS gradient-corruption issue where backward()
  corrupted shared internal MPS state even across deepcopy boundaries.
  Weight flips are applied on the working device (MPS/CPU), evaluation on MPS.

Usage (long run — use caffeinate):
  caffeinate -i python3 scripts/exp_v3_sec8_train_and_sweep.py --mode all
  caffeinate -i python3 scripts/exp_v3_sec8_train_and_sweep.py --mode train_only
  caffeinate -i python3 scripts/exp_v3_sec8_train_and_sweep.py --mode sweep_only --arch smallcnn
  caffeinate -i python3 scripts/exp_v3_sec8_train_and_sweep.py --mode table4
  caffeinate -i python3 scripts/exp_v3_sec8_train_and_sweep.py --mode random_control
  caffeinate -i python3 scripts/exp_v3_sec8_train_and_sweep.py --mode clip_sweep
  caffeinate -i python3 scripts/exp_v3_sec8_train_and_sweep.py --mode hparam_sweep
  caffeinate -i python3 scripts/exp_v3_sec8_train_and_sweep.py --mode stats_only
  python3 scripts/exp_v3_sec8_train_and_sweep.py --mode smoke_test

Prerequisites:
  pip3 install --break-system-packages torch torchvision torchaudio scipy statsmodels pandas
"""

import argparse, json, csv, os, sys, time, datetime, subprocess, hashlib, struct
from pathlib import Path

SCRIPT_DIR    = Path(__file__).resolve().parent
RESULTS_DIR   = SCRIPT_DIR.parent / "results_v2"
CKPT_DIR      = SCRIPT_DIR.parent / "checkpoints"
RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

N_SEEDS        = 10
BUDGETS_TABLE3 = [20]
BUDGETS_TABLE4 = [15]
CIFAR_CLASSES  = 10
SUBSAMPLE      = 3000   # match original paper §3.1: 3,000-image test subset
TRAIN_SUBSET   = 50000  # full CIFAR-10 train
TOP_K          = 8
SUB_BATCH      = 32
CLIP_SIGMAS    = [3.0, 2.0, 1.0, 0.5, 0.25]
HPARAM_TOPK    = [4, 8, 16]
HPARAM_BATCH   = [16, 32, 64]


def check_deps():
    missing = []
    try:
        import torch; import torchvision
        print(f"  torch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}")
    except ImportError:
        missing.append("torch torchvision torchaudio")
    try:
        import scipy; import statsmodels; import pandas
    except ImportError:
        missing.append("scipy statsmodels pandas")
    if missing:
        print("MISSING DEPENDENCIES. Run:")
        print(f"  pip3 install --break-system-packages {' '.join(missing)}")
        sys.exit(1)
    print("  All dependencies present.")


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_smallcnn():
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
        nn.Linear(256, 10),
    )


def build_resnet20():
    """ResNet-20 for CIFAR-10 following He et al. (2016)."""
    import torch.nn as nn
    import torch

    class ResBlock(nn.Module):
        def __init__(self, c, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(c, c, 3, stride=stride, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(c)
            self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(c)
            self.shortcut = nn.Sequential()
            if stride != 1:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(c, c, 1, stride=stride, bias=False), nn.BatchNorm2d(c))
        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return torch.relu(out + self.shortcut(x))

    class ResNet20(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1  = nn.Conv2d(3, 16, 3, padding=1, bias=False)
            self.bn1    = nn.BatchNorm2d(16)
            self.layer1 = nn.Sequential(*[ResBlock(16) for _ in range(3)])
            self.layer2 = nn.Sequential(ResBlock(16, stride=2),
                                        *[ResBlock(16) for _ in range(2)])
            self.layer3 = nn.Sequential(ResBlock(16, stride=2),
                                        *[ResBlock(16) for _ in range(2)])
            self.pool   = nn.AdaptiveAvgPool2d(1)
            self.fc     = nn.Linear(16, 10)
        def forward(self, x):
            import torch
            x = torch.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)
    return ResNet20()


def build_model(arch: str):
    return build_smallcnn() if arch == "smallcnn" else build_resnet20()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(arch: str, seed: int, device) -> Path:
    import torch, torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    ckpt_path = CKPT_DIR / f"{arch}_seed{seed}.pt"
    if ckpt_path.exists():
        print(f"  [skip] checkpoint exists: {ckpt_path.name}")
        return ckpt_path

    # Seed everything deterministically
    torch.manual_seed(seed + 42)
    np.random.seed(seed + 42)
    import random; random.seed(seed + 42)

    model = build_model(arch).to(device)

    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_dir = SCRIPT_DIR.parent / "data" / "cifar10"
    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=tf_train)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf_test)

    # 3,000-sample test subset (fixed per seed, reproducible)
    rng = np.random.default_rng(seed + 42)
    test_idx = rng.choice(len(test_ds), SUBSAMPLE, replace=False)
    test_sub  = Subset(test_ds, test_idx)

    # num_workers=0: avoids OSError "Too many open files" on macOS Python 3.14
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_sub, batch_size=256, shuffle=False, num_workers=0)

    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[82, 123], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 50 if arch == "smallcnn" else 164

    t0 = time.time()
    print(f"  Starting {arch} seed={seed}: {n_epochs} epochs, device={device}", flush=True)
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0; n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optim.step()
            epoch_loss += loss.item(); n_batches += 1
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb.to(device)).argmax(1)
                correct += (pred == yb.to(device)).sum().item()
                total   += len(yb)
        acc = correct / total
        print(f"  [{arch} s{seed}] epoch {epoch+1:3d}/{n_epochs} | "
              f"loss={epoch_loss/n_batches:.4f} | acc={acc:.4f} | "
              f"{time.time()-t0:.0f}s elapsed", flush=True)

    torch.save({"model_state": model.state_dict(), "arch": arch, "seed": seed,
                "test_indices": test_idx.tolist()}, ckpt_path)
    print(f"  ✓ saved {ckpt_path.name}")
    return ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# PBS Attack — MPS-safe implementation
#
# Key design: gradient ranking is computed entirely on CPU.
# model_work lives on MPS (fast eval), but for grad computation we:
#   1. Extract all parameter tensors as CPU numpy arrays (no autograd)
#   2. Build a fresh CPU-only model, load those weights, run forward+backward on CPU
#   3. Read out grad magnitudes on CPU
#   4. Apply the chosen weight flips to model_work's parameter data (MPS)
# This completely avoids MPS backward() state corruption.
# ─────────────────────────────────────────────────────────────────────────────

# IEEE-754 float32 bit fields:
#  bit 31     = sign
#  bits 30-23 = exponent (8 bits)
#  bits 22-0  = mantissa (23 bits)
#
# PBS implementation — bounded-damage exponent bits:
#  Bits 26-30 produce catastrophic deltas (10^8 to 10^38) on typical weights,
#  destroying the network in a single flip. Bits 23-25 produce bounded, progressive
#  damage (Δ ≈ 0.27–2.2 per weight) suitable for multi-step budget sweeps.
#  We also include bit 31 (sign flip) as the largest NaN-safe perturbation.
#
#  The score function (|grad|×|Δw|) still picks the most damaging flip among
#  these candidates per weight — gradient-guided targeting is preserved.
#
#  This matches the spirit of PBS (Rakin et al.) adapted for FP32 inference.
_PBS_FLIP_BITS = [31, 25, 24, 23]   # sign + 3 bounded exponent bits


def _flip_delta_vectorized(w_flat_np: "np.ndarray", bit: int) -> "np.ndarray":
    """
    Vectorized: compute |w_flipped − w| for every weight in w_flat_np
    when bit `bit` is flipped. Uses numpy view-casting for speed.
    """
    import numpy as np
    # View float32 array as uint32, flip bit, view back as float32
    w_u32  = w_flat_np.view(np.uint32).copy()
    w_u32 ^= np.uint32(1 << bit)
    w_flip = w_u32.view(np.float32)
    # Replace NaN/Inf with large finite float32
    bad    = ~np.isfinite(w_flip)
    w_flip[bad] = np.where(w_flat_np[bad] >= 0, np.float32(3.4e38), np.float32(-3.4e38))
    return np.abs(w_flip - w_flat_np)


def compute_grad_scores_cpu(arch: str, state_dict_cpu: dict,
                             loader_cpu, n_batches_max: int = 10):
    """
    Compute |grad| × |w_flipped_delta| for each (param, flat_idx) triple,
    selecting the best exponent bit (23-30) per weight.

    Fully vectorized via numpy view-casting — no Python loops over weights.

    Returns:
      scores    : dict param_name -> float32 np.ndarray of best-bit scores
      best_bits : dict param_name -> int32  np.ndarray of best bit per weight
    """
    import torch, torch.nn as nn
    import numpy as np
    cpu = torch.device("cpu")
    m = build_model(arch).to(cpu)
    m.load_state_dict(state_dict_cpu, strict=False)
    m.train()

    criterion = nn.CrossEntropyLoss()
    grad_acc = {nm: torch.zeros_like(p)
                for nm, p in m.named_parameters() if p.requires_grad}

    for i, (xb, yb) in enumerate(loader_cpu):
        if i >= n_batches_max:
            break
        xb, yb = xb.to(cpu), yb.to(cpu)
        m.zero_grad()
        loss = criterion(m(xb), yb)
        loss.backward()
        for nm, p in m.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad_acc[nm] += p.grad.abs().detach()

    scores    = {}
    best_bits = {}
    with torch.no_grad():
        for nm, p in m.named_parameters():
            if not p.requires_grad:
                continue
            w_flat = p.data.flatten().numpy().astype(np.float32)
            g_flat = grad_acc[nm].flatten().numpy().astype(np.float32)

            best_score   = np.zeros(len(w_flat), dtype=np.float32)
            best_bit_arr = np.full(len(w_flat), 30, dtype=np.int32)

            for bit in _PBS_FLIP_BITS:
                delta  = _flip_delta_vectorized(w_flat, bit)
                cand   = g_flat * delta
                better = cand > best_score
                best_score   = np.where(better, cand, best_score)
                best_bit_arr = np.where(better, bit,  best_bit_arr)

            scores[nm]    = best_score
            best_bits[nm] = best_bit_arr
    del m
    return scores, best_bits


def run_pbs(arch: str, seed: int, budget: int, defense: str,
            clip_sigma: float, checksum_cov: float, top_k: int,
            sub_batch: int, attack: str, device) -> list:
    """
    Run PBS against a real CIFAR-10 trained model.
    Returns per-step records (step, asr, accuracy, blocked, clipped).
    ASR = misclassification rate (same metric as original Table 3).

    MPS safety: gradient computation is done on CPU only.
    """
    import torch, torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    ckpt_path = CKPT_DIR / f"{arch}_seed{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run --mode train_only first.")

    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    test_idx = ckpt["test_indices"]
    # Note: strict=False used in model.load_state_dict below to handle any
    # minor BN key naming differences between training and sweep builds.

    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_dir = SCRIPT_DIR.parent / "data" / "cifar10"
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf_test)
    test_sub = Subset(test_ds, test_idx)

    # Two loaders: eval_loader (batched for eval), grad_loader (small for CPU grad)
    eval_loader = DataLoader(test_sub, batch_size=256, shuffle=False, num_workers=0)
    grad_loader = DataLoader(test_sub, batch_size=128, shuffle=False, num_workers=0)

    # ── model_work lives on the target device (MPS for speed) ──
    model_work = build_model(arch)
    model_work.load_state_dict(ckpt["model_state"], strict=False)
    model_work = model_work.to(device)
    model_work.eval()

    # ── helpers ──────────────────────────────────────────────────────────────
    def eval_model():
        model_work.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in eval_loader:
                p = model_work(xb.to(device)).argmax(1)
                correct += (p == yb.to(device)).sum().item()
                total   += len(yb)
        miss = 1.0 - correct / total
        return miss, correct / total

    def build_checksums(cov):
        rng = np.random.default_rng(42)
        chk = {}
        for n, p in model_work.named_parameters():
            if p.requires_grad and rng.random() < cov:
                chk[n] = p.data.norm().item()
        return chk

    def check_integrity(checksums, tol=1e-3):
        # tol=1e-3: detects norm changes from bounded exponent-bit flips (Δw≈0.27-2.2)
        # while tolerating floating-point rounding (< 1e-5).
        for n, ref in checksums.items():
            cur = dict(model_work.named_parameters())[n].data.norm().item()
            if abs(cur - ref) > tol:
                return False
        return True

    def build_clip_thresholds(sigma):
        thresholds = {}
        for n, p in model_work.named_parameters():
            if p.requires_grad:
                thresholds[n] = (p.data.mean().item(), sigma * p.data.std().item())
        return thresholds

    def apply_clip(thresholds):
        clipped = 0
        for n, p in model_work.named_parameters():
            if n in thresholds:
                mu, bound = thresholds[n]
                mask = (p.data - mu).abs() > bound
                p.data[mask] = mu + (p.data[mask] - mu).sign() * bound
                clipped += mask.sum().item()
        return clipped

    def flip_weight(name: str, flat_idx: int, bit: int = 30):
        """Flip `bit` of the IEEE-754 float32 weight at (name, flat_idx).
        Default bit=30 (highest exponent bit). PBS uses the best-scoring
        exponent bit per weight, found by compute_grad_scores_cpu.
        """
        import math
        param = dict(model_work.named_parameters())[name]
        flat  = param.data.flatten()
        w     = flat[flat_idx].item()
        b_int = struct.unpack("I", struct.pack("f", w))[0]
        b_flipped = b_int ^ (1 << bit)
        w_flipped = struct.unpack("f", struct.pack("I", b_flipped))[0]
        if math.isnan(w_flipped) or math.isinf(w_flipped):
            # Clamp NaN/Inf to max finite float32 with appropriate sign
            w_flipped = 3.4e38 if w >= 0 else -3.4e38
        flat[flat_idx] = w_flipped

    # ── get CPU copy of state_dict for grad computation ──────────────────────
    def get_cpu_state_dict():
        return {n: p.data.cpu().clone() for n, p in model_work.named_parameters()}

    # ── initialise defense structures ────────────────────────────────────────
    checksums   = build_checksums(checksum_cov) if defense == "checksum" else {}
    clip_thresh = build_clip_thresholds(clip_sigma) if defense == "clip" else {}

    base_miss, base_acc = eval_model()
    print(f"  baseline: acc={base_acc:.4f} miss={base_miss:.4f}", flush=True)

    records = []
    for step in range(1, budget + 1):

        # ── select weights to flip ────────────────────────────────────────────
        if attack == "pbs":
            # Compute gradients ENTIRELY on CPU — avoids MPS backward corruption
            cpu_sd = get_cpu_state_dict()
            scores, best_bits_map = compute_grad_scores_cpu(
                arch, cpu_sd, grad_loader, n_batches_max=8)

            candidates = []
            for n, score_flat in scores.items():
                topn = min(top_k * sub_batch, len(score_flat))
                for flat_idx in np.argsort(score_flat)[-topn:][::-1]:
                    best_bit = int(best_bits_map[n][flat_idx])
                    candidates.append((score_flat[flat_idx], n, int(flat_idx), best_bit))
            candidates.sort(key=lambda x: -x[0])
            chosen = candidates[:top_k]
        else:  # random baseline: uniform random weight + random bit from PBS set
            # CRITICAL: use the same _PBS_FLIP_BITS set as PBS so the comparison
            # isolates gradient GUIDANCE (which weight to flip) from bit choice.
            # If random used all 8 exponent bits including catastrophic 26-30,
            # it would outperform PBS simply by lucky catastrophic hits.
            chosen = []
            all_params = [(n, p) for n, p in model_work.named_parameters() if p.requires_grad]
            rng_r = np.random.default_rng(step)  # reproducible per-step
            for _ in range(top_k):
                ni       = rng_r.integers(len(all_params))
                n, p     = all_params[ni]
                flat_idx = int(rng_r.integers(p.numel()))
                rand_bit = int(_PBS_FLIP_BITS[rng_r.integers(len(_PBS_FLIP_BITS))])
                chosen.append((0.0, n, flat_idx, rand_bit))

        # ── apply flips ───────────────────────────────────────────────────────
        blocked = 0
        clipped_total = 0

        for flip_entry in chosen:
            _, name, flat_idx, best_bit = flip_entry  # both pbs and random are 4-tuples

            if defense == "checksum":
                # Tentative flip, verify integrity, revert if tamper detected
                flip_weight(name, flat_idx, best_bit)
                if not check_integrity(checksums):
                    # Revert: flip again (double-flip = original value)
                    flip_weight(name, flat_idx, best_bit)
                    blocked += 1
                else:
                    # Update running checksum for this layer
                    checksums[name] = dict(model_work.named_parameters())[name].data.norm().item()
            elif defense == "clip":
                flip_weight(name, flat_idx, best_bit)
                c = apply_clip(clip_thresh)
                clipped_total += (1 if c > 0 else 0)
            else:
                flip_weight(name, flat_idx, best_bit)

        # ── evaluate after flips ──────────────────────────────────────────────
        asr, acc = eval_model()
        records.append({
            "arch": arch, "seed": seed, "attack": attack, "defense": defense,
            "clip_sigma": clip_sigma, "checksum_cov": checksum_cov,
            "top_k": top_k, "sub_batch": sub_batch, "budget": budget,
            "step": step, "asr": round(asr, 6), "accuracy": round(acc, 6),
            "blocked": blocked, "clipped": clipped_total,
        })
        print(f"    step {step:2d}/{budget} | asr={asr:.5f} | acc={acc:.4f} | "
              f"blocked={blocked} clipped={clipped_total}", flush=True)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Sweep orchestration
# ─────────────────────────────────────────────────────────────────────────────

def save_and_summarise(all_rows: list, out_path: Path, label: str):
    """Write CSV, print SHA-256, print summary table."""
    import pandas as pd
    if not all_rows:
        print(f"  No rows to save for {label}.")
        return
    fieldnames = list(all_rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(all_rows)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"\n✅ {label} saved: {out_path.name}  SHA-256: {sha[:16]}...")
    df = pd.DataFrame(all_rows)
    final = df[df.step == df.budget]
    summary = final.groupby(["arch", "defense", "attack"])["asr"].agg(["mean", "std", "median"])
    print(f"\n=== {label} summary (final-step ASR) ===")
    print(summary.round(6))


def run_sweep(args, device):
    """Table 3: b=20, PBS attack, all defenses × seeds."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    all_rows = []
    archs    = ["smallcnn", "resnet20"] if args.arch == "both" else [args.arch]
    defenses = ["undefended", "checksum", "clip"] if args.defense == "all" else [args.defense]

    configs = []
    for arch in archs:
        for defense in defenses:
            for seed in range(N_SEEDS):
                configs.append((arch, seed, 20, defense, args.clip_sigma,
                                 args.checksum_cov, TOP_K, SUB_BATCH, "pbs"))

    total = len(configs)
    for i, cfg in enumerate(configs):
        arch, seed, budget, defense, sigma, cov, k, b, attack = cfg
        print(f"\n[{i+1}/{total}] {arch} seed={seed} b={budget} defense={defense} attack={attack}",
              flush=True)
        try:
            rows = run_pbs(arch, seed, budget, defense, sigma, cov, k, b, attack, device)
            all_rows.extend(rows)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}")

    save_and_summarise(all_rows, RESULTS_DIR / f"real_pbs_sweep_b20_{ts}.csv", "Table 3")


def run_table4(args, device):
    """Table 4: b=15, SmallCNN only."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    all_rows = []
    for defense in ["undefended", "checksum", "clip"]:
        for seed in range(N_SEEDS):
            print(f"\nTable4: smallcnn seed={seed} b=15 defense={defense}", flush=True)
            try:
                rows = run_pbs("smallcnn", seed, 15, defense,
                               args.clip_sigma, args.checksum_cov,
                               TOP_K, SUB_BATCH, "pbs", device)
                all_rows.extend(rows)
            except Exception as e:
                print(f"  SKIP/ERROR: {e}")
    save_and_summarise(all_rows, RESULTS_DIR / f"real_pbs_sweep_b15_{ts}.csv", "Table 4")


def run_random_control(args, device):
    """Random-flip baseline (control): compare random vs PBS."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    all_rows = []
    archs = ["smallcnn", "resnet20"] if args.arch == "both" else [args.arch]
    for arch in archs:
        for seed in range(N_SEEDS):
            print(f"\nRandom control: {arch} seed={seed} b=20", flush=True)
            try:
                rows = run_pbs(arch, seed, 20, "undefended",
                               1.0, 0.5, TOP_K, SUB_BATCH, "random", device)
                all_rows.extend(rows)
            except Exception as e:
                print(f"  SKIP/ERROR: {e}")
    save_and_summarise(all_rows, RESULTS_DIR / f"real_random_control_{ts}.csv", "Random control")


def run_clip_sweep(args, device):
    """Clip-σ ablation: 5 thresholds × 2 archs × 10 seeds."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    archs = ["smallcnn", "resnet20"] if args.arch == "both" else [args.arch]
    for arch in archs:
        for sigma in CLIP_SIGMAS:
            all_rows = []
            for seed in range(N_SEEDS):
                print(f"\nClip sweep: {arch} seed={seed} sigma={sigma}", flush=True)
                try:
                    rows = run_pbs(arch, seed, 20, "clip",
                                   sigma, 0.5, TOP_K, SUB_BATCH, "pbs", device)
                    all_rows.extend(rows)
                except Exception as e:
                    print(f"  SKIP/ERROR: {e}")
            save_and_summarise(all_rows,
                               RESULTS_DIR / f"real_clip_sweep_{arch}_s{sigma}_{ts}.csv",
                               f"Clip {arch} sigma={sigma}")


def run_hparam_sweep(device):
    """top_k × sub_batch ablation on SmallCNN."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    for k in HPARAM_TOPK:
        for b in HPARAM_BATCH:
            all_rows = []
            for seed in range(N_SEEDS):
                print(f"\nHparam: smallcnn seed={seed} k={k} b={b}", flush=True)
                try:
                    rows = run_pbs("smallcnn", seed, 20, "undefended",
                                   1.0, 0.5, k, b, "pbs", device)
                    all_rows.extend(rows)
                except Exception as e:
                    print(f"  SKIP/ERROR: {e}")
            save_and_summarise(all_rows,
                               RESULTS_DIR / f"real_hparam_k{k}_b{b}_{ts}.csv",
                               f"Hparam k={k} b={b}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test — single seed, 3 steps, validate ASR increases
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(device):
    print("\n=== SMOKE TEST: smallcnn seed=0, b=3, undefended PBS ===", flush=True)
    rows = run_pbs("smallcnn", 0, 3, "undefended", 1.0, 0.5, TOP_K, SUB_BATCH, "pbs", device)
    asrs = [r["asr"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    print(f"\nStep ASRs: {asrs}")
    print(f"Step Accs: {accs}")
    # Validate: ASR must increase (or at minimum: not constant across all steps)
    if len(set(asrs)) == 1:
        print("⚠️  WARNING: All ASR values identical — possible MPS bug still present!")
        print("    Try running with --device cpu")
    else:
        print(f"✅ SMOKE TEST PASSED: ASR varies across steps (baseline→final: {asrs[0]:.4f}→{asrs[-1]:.4f})")

    print("\n=== SMOKE TEST: smallcnn seed=0, b=3, checksum PBS ===", flush=True)
    rows2 = run_pbs("smallcnn", 0, 3, "checksum", 1.0, 0.5, TOP_K, SUB_BATCH, "pbs", device)
    b_cnt = [r["blocked"] for r in rows2]
    print(f"Blocked counts per step: {b_cnt}")
    if any(b > 0 for b in b_cnt):
        print("✅ Checksum defense blocking some flips.")
    else:
        print("ℹ️  No flips blocked (checksum defense may need tuning).")

    print("\n=== SMOKE TEST: smallcnn seed=0, b=3, random attack ===", flush=True)
    rows3 = run_pbs("smallcnn", 0, 3, "undefended", 1.0, 0.5, TOP_K, SUB_BATCH, "random", device)
    asrs3 = [r["asr"] for r in rows3]
    print(f"Random attack ASRs: {asrs3}")
    print(f"PBS ASRs:           {asrs}")
    print("✅ Smoke test complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Stats (Holm-Bonferroni + Bootstrap CIs) — run after sweeps
# ─────────────────────────────────────────────────────────────────────────────

def run_stats():
    import glob
    import pandas as pd
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    import numpy as np

    # Use latest canonical Table 3 run
    csv_files = sorted(glob.glob(str(RESULTS_DIR / "real_pbs_sweep_b20_*.csv")))
    if not csv_files:
        print("No real_pbs_sweep_b20_*.csv found. Run --mode sweep_only first.")
        return
    src = csv_files[-1]
    print(f"Using: {src}")
    df = pd.read_csv(src)
    final = df[df.step == df.budget]

    print(f"  Total final-step rows: {len(final)}")
    print(f"  Archs: {sorted(final.arch.unique())}")
    print(f"  Defenses: {sorted(final.defense.unique())}")

    pval_rows = []
    for arch in sorted(final.arch.unique()):
        d = final[final.arch == arch]
        undef = d[d.defense == "undefended"].groupby("seed")["asr"].last().values
        for defense in ["checksum", "clip"]:
            def_vals = d[d.defense == defense].groupby("seed")["asr"].last().values
            n = min(len(undef), len(def_vals))
            if n < 2:
                continue
            try:
                stat, p = wilcoxon(undef[:n], def_vals[:n], alternative="greater")
            except Exception:
                stat, p = 0.0, 1.0
            # Cliff's delta (signed rank)
            nx, ny = len(undef[:n]), len(def_vals[:n])
            cliff_d = sum(1 if u > v else -1 if u < v else 0
                          for u in undef[:n] for v in def_vals[:n]) / (nx * ny)
            pval_rows.append({
                "arch": arch,
                "comparison": f"undefended vs {defense}",
                "attack": "pbs", "n": n,
                "mean_undef": round(undef[:n].mean(), 6),
                "mean_def":   round(def_vals[:n].mean(), 6),
                "wilcoxon_stat": stat, "p_raw": round(p, 6),
                "cliff_delta": round(cliff_d, 4),
                "source": os.path.basename(src),
            })

    if not pval_rows:
        print("No comparison rows — check that sweep has ≥2 seeds per defense.")
        return

    pvals = [r["p_raw"] for r in pval_rows]
    rejected, p_holm, _, _ = multipletests(pvals, method="holm")
    for r, ph, sig in zip(pval_rows, p_holm, rejected):
        r["p_holm"] = round(ph, 6)
        r["significant_after_holm"] = bool(sig)

    out = RESULTS_DIR / "real_wilcoxon_holm_corrected.csv"
    pd.DataFrame(pval_rows).to_csv(out, index=False)
    print(f"\n✅ Holm-Bonferroni saved: {out.name}")
    print(pd.DataFrame(pval_rows)[
        ["arch","comparison","mean_undef","mean_def","p_raw","p_holm",
         "cliff_delta","significant_after_holm"]
    ])

    # Bootstrap CIs on ASR reduction from PBS
    boot_rows = []
    for arch in sorted(final.arch.unique()):
        d = final[final.arch == arch]
        undef = d[d.defense == "undefended"].groupby("seed")["asr"].last().values
        for defense in ["checksum", "clip"]:
            def_vals = d[d.defense == defense].groupby("seed")["asr"].last().values
            n = min(len(undef), len(def_vals))
            a, b_ = undef[:n], def_vals[:n]
            rng = np.random.default_rng(42)
            deltas = [a[rng.integers(0, n, n)].mean() - b_[rng.integers(0, n, n)].mean()
                      for _ in range(10000)]
            ci = np.percentile(deltas, [2.5, 97.5])
            boot_rows.append({
                "arch": arch, "defense": defense, "budget": 20,
                "mean_delta_asr": round(float(np.mean(deltas)), 6),
                "ci_lo_95": round(float(ci[0]), 6),
                "ci_hi_95": round(float(ci[1]), 6),
                "source": os.path.basename(src),
            })
    out_b = RESULTS_DIR / "real_bootstrap_cis.csv"
    pd.DataFrame(boot_rows).to_csv(out_b, index=False)
    print(f"\n✅ Bootstrap CIs saved: {out_b.name}")
    print(pd.DataFrame(boot_rows))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Blueprint v3 §8: Real-model Study A (v4 — CPU-grad fix)")
    p.add_argument("--mode", choices=[
        "check", "smoke_test", "train_only", "sweep_only", "table4",
        "random_control", "clip_sweep", "hparam_sweep", "stats_only", "all"
    ], default="check")
    p.add_argument("--arch", choices=["smallcnn", "resnet20", "both"], default="both")
    p.add_argument("--defense", default="all",
                   help="Defense filter for sweep_only: undefended|checksum|clip|all")
    p.add_argument("--clip_sigma", type=float, default=1.0)
    p.add_argument("--checksum_cov", type=float, default=0.5)
    args = p.parse_args()

    check_deps()
    import torch
    device_str = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")

    if args.mode == "check":
        print("✅ All dependencies OK. Run with --mode smoke_test to verify the fix.")
        return

    if args.mode == "smoke_test":
        run_smoke_test(device)
        return

    if args.mode in ("train_only", "all"):
        archs = ["smallcnn", "resnet20"] if args.arch == "both" else [args.arch]
        for arch in archs:
            for seed in range(N_SEEDS):
                print(f"\nTraining {arch} seed={seed}")
                train_model(arch, seed, device)

    if args.mode in ("sweep_only", "all"):
        run_sweep(args, device)

    if args.mode in ("table4", "all"):
        run_table4(args, device)

    if args.mode in ("random_control", "all"):
        run_random_control(args, device)

    if args.mode in ("clip_sweep", "all"):
        run_clip_sweep(args, device)

    if args.mode in ("hparam_sweep", "all"):
        run_hparam_sweep(device)

    if args.mode in ("stats_only", "all"):
        run_stats()

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
