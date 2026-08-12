"""
run_resnet20_seed.py — Experiment B: Attack/defense grid on ResNet-20 / full CIFAR-10.

Usage:
    python run_resnet20_seed.py <seed> [--out_dir PATH]

Produces results/resnet20/resnet20_seed{seed}.json.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import ResNet20
from data_utils import get_data
from attack_defense import (get_quant_state, apply_state_to_model,
                             progressive_bit_search, build_protection_mask,
                             ATTACKABLE_LAYERS_RESNET20)

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))


def run_resnet20_seed(seed, out_dir):
    ckpt_dir        = os.path.join(out_dir, "ckpt")
    results_dir     = os.path.join(out_dir, "results", "resnet20")
    os.makedirs(results_dir, exist_ok=True)

    Xtr, ytr, Xte, yte = get_data(5000, 1000, 0)
    X_atk  = Xtr[seed * 32:(seed * 32 + 32)].to(DEVICE)
    y_atk  = ytr[seed * 32:(seed * 32 + 32)].to(DEVICE)
    X_eval = Xte[:1000].to(DEVICE)
    y_eval = yte[:1000].to(DEVICE)

    ckpt_path = os.path.join(ckpt_dir, f"resnet20_seed{seed}.pt")
    m = ResNet20()
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.to(DEVICE).eval()

    results = {}
    for defense in [None, "checksum", "clip"]:
        m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        m.to(DEVICE).eval()
        state = get_quant_state(m, bits=8, attackable_layers=ATTACKABLE_LAYERS_RESNET20)
        apply_state_to_model(m, state)
        with torch.no_grad():
            clean_acc = (m(X_eval).argmax(1) == y_eval).float().mean().item()

        protect_mask = None
        if defense == "checksum":
            protect_mask = build_protection_mask(state, protect_fraction=0.5, seed=seed)

        t0   = time.time()
        traj = progressive_bit_search(m, state, X_atk, y_atk, X_eval, y_eval,
                                       max_flips=20, topk=8, defense=defense,
                                       protect_mask=protect_mask, clip_c=3.0,
                                       target_asr=0.90, seed=seed)
        traj["elapsed_s"]            = time.time() - t0
        traj["post_quant_clean_acc"] = clean_acc
        key                          = defense if defense else "none"
        results[key]                 = traj
        print(f"  seed={seed}  defense={key}  elapsed={traj['elapsed_s']:.1f}s  "
              f"final_asr={traj['asr'][-1]:.3f}")

    out_path = os.path.join(results_dir, f"resnet20_seed{seed}.json")
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
    run_resnet20_seed(args.seed, args.out_dir)


if __name__ == "__main__":
    main()
