"""
run_targeted.py — Experiment D: Targeted attack variant.

Tests 3 (source, target) class pairs per seed:
  - (0=airplane → 8=ship)
  - (3=cat     → 5=dog)
  - (9=truck   → 1=automobile)

Usage:
    python run_targeted.py <seed> [--out_dir PATH]

Produces results/targeted/targeted_seed{seed}.json.
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
                             progressive_bit_search_targeted,
                             build_protection_mask)

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

CLASS_PAIRS = [
    (0, 8, "airplane→ship"),
    (3, 5, "cat→dog"),
    (9, 1, "truck→automobile"),
]


def run_targeted(seed, out_dir):
    ckpt_dir    = os.path.join(out_dir, "ckpt")
    results_dir = os.path.join(out_dir, "results", "targeted")
    os.makedirs(results_dir, exist_ok=True)

    Xtr, ytr, Xte, yte = get_data(300, 100, 0)
    X_atk  = Xtr[seed * 32:(seed * 32 + 32)].to(DEVICE)
    y_atk  = ytr[seed * 32:(seed * 32 + 32)].to(DEVICE)
    X_eval = Xte[:200].to(DEVICE)
    y_eval = yte[:200].to(DEVICE)

    ckpt_path = os.path.join(ckpt_dir, f"model_seed{seed}.pt")
    m = SmallCNN()

    results = {}
    for src, tgt, pair_label in CLASS_PAIRS:
        for defense in [None, "checksum", "clip"]:
            m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            m.to(DEVICE).eval()
            state = get_quant_state(m, bits=8)
            apply_state_to_model(m, state)
            with torch.no_grad():
                clean_acc = (m(X_eval).argmax(1) == y_eval).float().mean().item()

            protect_mask = None
            if defense == "checksum":
                protect_mask = build_protection_mask(state, protect_fraction=0.5, seed=seed)

            t0   = time.time()
            traj = progressive_bit_search_targeted(
                m, state, X_atk, y_atk, X_eval, y_eval,
                source_class=src, target_class=tgt,
                max_flips=20, topk=8, defense=defense,
                protect_mask=protect_mask, clip_c=3.0,
                target_asr=0.70, seed=seed,
                targeted_lambda=0.5
            )
            traj["elapsed_s"]            = time.time() - t0
            traj["post_quant_clean_acc"] = clean_acc
            traj["source_class"]         = src
            traj["target_class"]         = tgt
            traj["pair_label"]           = pair_label

            def_key  = defense if defense else "none"
            result_key = f"{pair_label}_{def_key}"
            results[result_key] = traj
            final_tasr = traj["targeted_asr"][-1] if traj["targeted_asr"] else 0.0
            print(f"  seed={seed}  {pair_label}  defense={def_key}  "
                  f"targeted_asr={final_tasr:.3f}  asr={traj['asr'][-1]:.3f}  "
                  f"elapsed={traj['elapsed_s']:.1f}s")

    out_path = os.path.join(results_dir, f"targeted_seed{seed}.json")
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
    run_targeted(args.seed, args.out_dir)


if __name__ == "__main__":
    main()
