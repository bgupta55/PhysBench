"""
run_sweep.py — Parameter sweep (coverage, clip_c) for one SmallCNN seed.

Usage:
    python run_sweep.py <seed> [--out_dir PATH]

Produces results/sweep_seed{seed}.json.
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
                             progressive_bit_search, build_protection_mask)

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))


def run_sweep(seed, out_dir):
    ckpt_dir    = os.path.join(out_dir, "ckpt")
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    Xtr, ytr, Xte, yte = get_data(300, 100, 0)
    X_atk  = Xtr[seed * 32:seed * 32 + 32].to(DEVICE)
    y_atk  = ytr[seed * 32:seed * 32 + 32].to(DEVICE)
    X_eval = Xte[:200].to(DEVICE)
    y_eval = yte[:200].to(DEVICE)

    ckpt_path = os.path.join(ckpt_dir, f"model_seed{seed}.pt")
    m = SmallCNN()

    conditions = [
        ("checksum", 0.25, None),
        ("checksum", 0.75, None),
        ("clip",     None, 1.0),
        ("clip",     None, 2.0),
    ]

    results = {}
    for defense, coverage, clip_c in conditions:
        m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        m.to(DEVICE).eval()
        state        = get_quant_state(m, bits=8)
        apply_state_to_model(m, state)
        protect_mask = build_protection_mask(state, coverage, seed) if defense == "checksum" else None
        t0           = time.time()
        traj         = progressive_bit_search(m, state, X_atk, y_atk, X_eval, y_eval,
                                              max_flips=15, topk=8, defense=defense,
                                              protect_mask=protect_mask,
                                              clip_c=clip_c if clip_c else 3.0,
                                              target_asr=0.90, seed=seed)
        traj["elapsed_s"] = time.time() - t0
        key               = f"{defense}_{coverage if coverage else clip_c}"
        results[key]      = traj
        print(f"  seed={seed}  {key}  final_asr={traj['asr'][-1]:.3f}  "
              f"elapsed={traj['elapsed_s']:.1f}s")

    out_path = os.path.join(results_dir, f"sweep_seed{seed}.json")
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
    run_sweep(args.seed, args.out_dir)


if __name__ == "__main__":
    main()
