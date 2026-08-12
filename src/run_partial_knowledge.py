"""
run_partial_knowledge.py — Partial-knowledge adaptive attacker sweep.

Sweeps the fraction of the protection mask that the attacker knows:
  0%, 25%, 50%, 75%, 100%

At each fraction, the attacker probabilistically zeroes out gradients at
known-protected positions (sampled from the true mask without replacement).
At 0% the attacker has no mask knowledge (same as non-adaptive).
At 100% it's the full white-box adaptive attacker from Experiment A.

n=10 seeds, SmallCNN, 20-flip budget.

Output: results/partial_knowledge_results.json
"""
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from model import SmallCNN
from data_utils import get_data
from attack_defense import (get_quant_state, apply_state_to_model,
                              build_protection_mask, ATTACKABLE_LAYERS_SMALL)
from quant import dequantize_tensor, flip_bit

# Force CPU — MPS memory may be occupied by concurrent ResNet-20 retrain.
DEVICE      = torch.device("cpu")
FRACTIONS   = [0.0, 0.25, 0.50, 0.75, 1.00]
MAX_FLIPS   = 20
TOPK        = 8
BITS        = 8


def pbs_partial_knowledge(model, state, X_atk, y_atk, X_eval, y_eval,
                           protect_mask, known_fraction, seed):
    """PBS where attacker knows `known_fraction` of the protection mask."""
    torch.manual_seed(seed + int(known_fraction * 1000))

    # Build the attacker's partial knowledge: sample `known_fraction` of
    # protected positions as actually-known (others treated as unprotected)
    known_mask = {}
    for name, m in protect_mask.items():
        flat     = m.flatten()
        prot_idx = flat.nonzero(as_tuple=False).flatten().tolist()
        k = int(len(prot_idx) * known_fraction)
        rng = np.random.default_rng(seed + 42)
        if k > 0 and len(prot_idx) > 0:
            known_idx = rng.choice(prot_idx, size=k, replace=False).tolist()
        else:
            known_idx = []
        known_flat = torch.zeros_like(flat, dtype=torch.bool)
        for i in known_idx:
            known_flat[i] = True
        known_mask[name] = known_flat.view(m.shape)

    traj = {"flip_idx": [], "asr": [], "n_effective_flips": []}
    n_eff = 0

    for step in range(1, MAX_FLIPS + 1):
        apply_state_to_model(model, state)
        model.zero_grad()
        for name in state:
            dict(model.named_parameters())[name].requires_grad_(True)
        F.cross_entropy(model(X_atk.contiguous()), y_atk).backward()

        candidates = []
        for name in state:
            grad = dict(model.named_parameters())[name].grad
            if grad is None:
                continue
            fg = grad.abs().flatten().clone()
            # Zero out known-protected gradients
            if name in known_mask:
                fg[known_mask[name].flatten()] = 0.0
            k = min(TOPK, (fg > 0).sum().item())
            k = max(k, 1)
            for idx in torch.topk(fg, k).indices.tolist():
                candidates.append((name, idx))

        if not candidates:
            break

        best = None
        for name, idx in candidates:
            flat_q   = state[name]["q"].flatten()
            orig_val = int(flat_q[idx].item())
            for bp in range(BITS):
                flat_q[idx] = flip_bit(orig_val, bp, BITS)
                apply_state_to_model(model, state)
                with torch.no_grad():
                    l = F.cross_entropy(model(X_atk), y_atk).item()
                if best is None or l > best[0]:
                    best = (l, name, idx, bp, int(flat_q[idx].item()))
                flat_q[idx] = orig_val
            apply_state_to_model(model, state)

        if best is None:
            break

        _, name, idx, bp, new_val = best
        orig_val_rb = int(state[name]["q"].flatten()[idx].item())
        state[name]["q"].flatten()[idx] = new_val

        # Defense: checksum rollback (true protected mask, not attacker's view)
        protected = bool(protect_mask[name].flatten()[idx].item()) if name in protect_mask else False
        if protected:
            state[name]["q"].flatten()[idx] = orig_val_rb
        else:
            n_eff += 1

        apply_state_to_model(model, state)
        with torch.no_grad():
            asr = (model(X_eval).argmax(1) != y_eval).float().mean().item()
        traj["flip_idx"].append(step)
        traj["asr"].append(asr)
        traj["n_effective_flips"].append(n_eff)
        if asr >= 0.90:
            break

    return traj


def main():
    out_dir     = os.path.join(os.path.dirname(SRC_DIR), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file    = os.path.join(out_dir, "partial_knowledge_results.json")

    Xtr, ytr, Xte, yte = get_data(300, 100, 0)

    # Per-seed, per-fraction results
    all_results = {str(f): [] for f in FRACTIONS}

    for seed in range(10):
        ckpt = os.path.join(os.path.dirname(SRC_DIR), "ckpt", f"model_seed{seed}.pt")
        X_atk  = Xtr[seed * 32 : seed * 32 + 32].to(DEVICE)
        y_atk  = ytr[seed * 32 : seed * 32 + 32].to(DEVICE)
        X_eval = Xte[:200].to(DEVICE)
        y_eval = yte[:200].to(DEVICE)

        m_ref = SmallCNN()
        m_ref.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m_ref.to(DEVICE).eval()
        state_ref    = get_quant_state(m_ref, bits=BITS)
        protect_mask = build_protection_mask(state_ref, protect_fraction=0.5, seed=seed)

        for frac in FRACTIONS:
            m = SmallCNN()
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            m.to(DEVICE).eval()
            state = get_quant_state(m, bits=BITS)
            apply_state_to_model(m, state)
            traj  = pbs_partial_knowledge(m, state, X_atk, y_atk, X_eval, y_eval,
                                           protect_mask=protect_mask,
                                           known_fraction=frac, seed=seed)
            final_asr = traj["asr"][-1] if traj["asr"] else 0.0
            all_results[str(frac)].append(final_asr)
            print(f"  seed={seed}  frac={frac:.2f}  ASR={final_asr:.3f}")

    # Aggregate per fraction
    aggregate = {}
    for f in FRACTIONS:
        vals = all_results[str(f)]
        aggregate[str(f)] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "per_seed": vals,
        }

    print("\n=== PARTIAL KNOWLEDGE AGGREGATE ===")
    for f in FRACTIONS:
        a = aggregate[str(f)]
        print(f"  known_frac={f:.2f}  ASR={a['mean']:.3f} ± {a['std']:.3f}")

    with open(out_file, "w") as fh:
        json.dump({"per_seed": all_results, "aggregate": aggregate}, fh, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
