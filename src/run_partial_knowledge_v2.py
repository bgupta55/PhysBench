"""
run_partial_knowledge_v2.py — Fixed partial-knowledge adaptive attacker sweep.

Reuses the exact _pbs_core infrastructure via a thin wrapper that passes
a *partial* protect_mask (only the known fraction) as the adaptive mask while
the *full* protect_mask is used for the defense rollback.

Sweeps known fraction: 0%, 25%, 50%, 75%, 100%
n=10 seeds, SmallCNN, 20-flip budget.

Output: results/partial_knowledge_results.json (overwrites)
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
                              build_protection_mask, ATTACKABLE_LAYERS_SMALL,
                              _pbs_core)
from quant import dequantize_tensor, flip_bit

DEVICE    = (torch.device("mps") if torch.backends.mps.is_available()
             else torch.device("cpu"))
FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.00]
MAX_FLIPS = 20


def make_partial_mask(protect_mask, known_fraction, seed):
    """Return a boolean mask with only `known_fraction` of protected positions set."""
    rng = np.random.default_rng(seed + 999)
    partial = {}
    for name, m in protect_mask.items():
        flat     = m.flatten()
        prot_idx = flat.nonzero(as_tuple=False).flatten().tolist()
        k        = int(len(prot_idx) * known_fraction)
        if k > 0 and prot_idx:
            known_idx = rng.choice(prot_idx, size=k, replace=False).tolist()
        else:
            known_idx = []
        known_flat = torch.zeros_like(flat, dtype=torch.bool)
        for i in known_idx:
            known_flat[i] = True
        partial[name] = known_flat.view(m.shape)
    return partial


def main():
    out_dir  = os.path.join(os.path.dirname(SRC_DIR), "results")
    out_file = os.path.join(out_dir, "partial_knowledge_results.json")

    Xtr, ytr, Xte, yte = get_data(300, 100, 0)

    all_results = {str(f): [] for f in FRACTIONS}

    for seed in range(10):
        ckpt = os.path.join(os.path.dirname(SRC_DIR), "ckpt", f"model_seed{seed}.pt")
        X_atk  = Xtr[seed * 32 : seed * 32 + 32].to(DEVICE)
        y_atk  = ytr[seed * 32 : seed * 32 + 32].to(DEVICE)
        X_eval = Xte[:200].to(DEVICE)
        y_eval = yte[:200].to(DEVICE)

        # Reference: build the true protect mask once per seed
        m_ref = SmallCNN()
        m_ref.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m_ref.to(DEVICE).eval()
        state_ref    = get_quant_state(m_ref, bits=8)
        protect_mask = build_protection_mask(state_ref, protect_fraction=0.5, seed=seed)

        for frac in FRACTIONS:
            m = SmallCNN()
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            m.to(DEVICE).eval()
            state = get_quant_state(m, bits=8)
            apply_state_to_model(m, state)

            # Partial knowledge: attacker zeroes gradients only at 'known' positions
            partial_mask = make_partial_mask(protect_mask, frac, seed)

            # Run PBS with:
            #   defense="checksum" → rollback uses full protect_mask
            #   adaptive=True      → gradient zeroing uses partial_mask (not protect_mask)
            # We temporarily monkey-patch by using partial_mask as the gradient-zeroing mask.
            # _pbs_core uses protect_mask for BOTH rollback and gradient zeroing when adaptive=True.
            # To get partial knowledge, we pass partial_mask as protect_mask to _pbs_core for
            # gradient zeroing, and apply rollback separately via the full protect_mask post-hoc.
            #
            # Cleanest approach: run _pbs_core with adaptive=True and protect_mask=partial_mask,
            # which zeros gradients at known positions only — then the full checksum rollback
            # happens inside _pbs_core because defense="checksum" with protect_mask= FULL mask.
            #
            # But _pbs_core uses the same protect_mask for both purposes. Fix: run with
            # partial_mask for adaptive zeroing, and full protect_mask for rollback.
            # Since _pbs_core doesn't support this split, we use a combined mask approach:
            # pass full protect_mask for defense rollback, but pass partial_mask as
            # the adaptive gradient-zeroing mask by temporarily swapping.

            # Simplest correct approach: subclass / inline the loop.
            # Here we inline it directly.
            traj = _run_partial_pbs(m, state, X_atk, y_atk, X_eval, y_eval,
                                     protect_mask_full=protect_mask,
                                     protect_mask_known=partial_mask,
                                     seed=seed)

            final_asr = traj["asr"][-1] if traj["asr"] else 0.0
            all_results[str(frac)].append(final_asr)
            print(f"  seed={seed}  frac={frac:.2f}  ASR={final_asr:.3f}", flush=True)

    aggregate = {}
    for f in FRACTIONS:
        vals = all_results[str(f)]
        aggregate[str(f)] = {"mean": float(np.mean(vals)),
                              "std":  float(np.std(vals)),
                              "per_seed": vals}

    print("\n=== PARTIAL KNOWLEDGE AGGREGATE ===")
    for f in FRACTIONS:
        a = aggregate[str(f)]
        print(f"  known_frac={f:.2f}  ASR={a['mean']:.3f} ± {a['std']:.3f}")

    with open(out_file, "w") as fh:
        json.dump({"per_seed": all_results, "aggregate": aggregate}, fh, indent=2)
    print(f"\nSaved: {out_file}")


def _run_partial_pbs(model, state, X_atk, y_atk, X_eval, y_eval,
                     protect_mask_full, protect_mask_known, seed,
                     max_flips=20, topk=8, bits=8):
    """PBS with split protect masks:
    - protect_mask_known: used for gradient zeroing (attacker's partial knowledge)
    - protect_mask_full:  used for rollback (true defense)
    """
    from quant import dequantize_tensor, flip_bit

    torch.manual_seed(seed)
    traj = {"asr": [], "n_effective_flips": []}
    n_eff = 0

    layer_stds = {name: dequantize_tensor(s["q"], s["scale"]).float().std().item() + 1e-8
                  for name, s in state.items()}

    for step in range(1, max_flips + 1):
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
            # Zero out gradients at KNOWN-protected positions only
            if name in protect_mask_known:
                fg[protect_mask_known[name].flatten()] = 0.0
            k = min(topk, (fg > 0).sum().item())
            k = max(k, 1)
            for idx in torch.topk(fg, k).indices.tolist():
                candidates.append((name, idx))

        if not candidates:
            break

        best = None
        for name, idx in candidates:
            flat_q   = state[name]["q"].flatten()
            orig_val = int(flat_q[idx].item())
            for bp in range(bits):
                flat_q[idx] = flip_bit(orig_val, bp, bits)
                apply_state_to_model(model, state)
                with torch.no_grad():
                    l = F.cross_entropy(model(X_atk.contiguous()), y_atk).item()
                if best is None or l > best[0]:
                    best = (l, name, idx, bp, int(flat_q[idx].item()))
                flat_q[idx] = orig_val
            apply_state_to_model(model, state)

        if best is None:
            break

        _, name, idx, bp, new_val = best
        orig_val_rb = int(state[name]["q"].flatten()[idx].item())
        state[name]["q"].flatten()[idx] = new_val

        # Defense: rollback uses FULL protect mask
        protected = bool(protect_mask_full[name].flatten()[idx].item()) if name in protect_mask_full else False
        if protected:
            state[name]["q"].flatten()[idx] = orig_val_rb
        else:
            n_eff += 1

        apply_state_to_model(model, state)
        with torch.no_grad():
            asr = (model(X_eval.contiguous()).argmax(1) != y_eval).float().mean().item()
        traj["asr"].append(asr)
        traj["n_effective_flips"].append(n_eff)
        if asr >= 0.90:
            break

    return traj


if __name__ == "__main__":
    main()
