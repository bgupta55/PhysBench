"""
run_adaptive.py — Experiment A: Adaptive attacker.

Usage:
    python run_adaptive.py <seed> [--out_dir PATH]

Produces results/adaptive/adaptive_seed{seed}.json with trajectories for:
  - none       (undefended)
  - checksum   (non-adaptive attacker, original)
  - checksum_adaptive  (attacker knows the mask, excludes protected weights)
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import SmallCNN
from data_utils import get_data
from attack_defense import (get_quant_state, apply_state_to_model,
                             progressive_bit_search,
                             progressive_bit_search_adaptive,
                             build_protection_mask)

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))


def run_adaptive(seed, out_dir):
    ckpt_dir        = os.path.join(out_dir, "ckpt")
    results_dir     = os.path.join(out_dir, "results", "adaptive")
    os.makedirs(results_dir, exist_ok=True)

    Xtr, ytr, Xte, yte = get_data(300, 100, 0)
    X_atk  = Xtr[seed * 32:(seed * 32 + 32)].to(DEVICE)
    y_atk  = ytr[seed * 32:(seed * 32 + 32)].to(DEVICE)
    X_eval = Xte[:200].to(DEVICE)
    y_eval = yte[:200].to(DEVICE)

    ckpt_path = os.path.join(ckpt_dir, f"model_seed{seed}.pt")
    m = SmallCNN()

    results = {}

    # ── Condition 1: undefended ──────────────────────────────────────
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.to(DEVICE).eval()
    state = get_quant_state(m, bits=8)
    apply_state_to_model(m, state)
    with torch.no_grad():
        clean_acc = (m(X_eval).argmax(1) == y_eval).float().mean().item()

    t0   = time.time()
    traj = progressive_bit_search(m, state, X_atk, y_atk, X_eval, y_eval,
                                   max_flips=20, topk=8, defense=None,
                                   protect_mask=None, clip_c=3.0,
                                   target_asr=0.90, seed=seed)
    traj["elapsed_s"]            = time.time() - t0
    traj["post_quant_clean_acc"] = clean_acc
    results["none"] = traj
    print(f"  seed={seed}  none            final_asr={traj['asr'][-1]:.3f}  "
          f"elapsed={traj['elapsed_s']:.1f}s")

    # ── Conditions 2 & 3: checksum (non-adaptive vs adaptive) ────────
    for adaptive in [False, True]:
        m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        m.to(DEVICE).eval()
        state        = get_quant_state(m, bits=8)
        apply_state_to_model(m, state)
        protect_mask = build_protection_mask(state, protect_fraction=0.5, seed=seed)
        with torch.no_grad():
            clean_acc = (m(X_eval).argmax(1) == y_eval).float().mean().item()

        fn  = progressive_bit_search_adaptive if adaptive else progressive_bit_search
        lbl = "checksum_adaptive" if adaptive else "checksum"
        t0  = time.time()
        traj = fn(m, state, X_atk, y_atk, X_eval, y_eval,
                  max_flips=20, topk=8, defense="checksum",
                  protect_mask=protect_mask, clip_c=3.0,
                  target_asr=0.90, seed=seed)
        traj["elapsed_s"]            = time.time() - t0
        traj["post_quant_clean_acc"] = clean_acc
        results[lbl] = traj
        print(f"  seed={seed}  {lbl:<22} final_asr={traj['asr'][-1]:.3f}  "
              f"elapsed={traj['elapsed_s']:.1f}s")

    out_path = os.path.join(results_dir, f"adaptive_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → saved {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), ".."))
    args = parser.parse_args()
    run_adaptive(args.seed, args.out_dir)


if __name__ == "__main__":
    main()
