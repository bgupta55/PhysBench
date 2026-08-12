#!/usr/bin/env python3
"""
run_all_experiments.py — Master runner for all PhysBench experiments.

Runs, in order:
  0. Install dependencies (if needed)
  1. Train SmallCNN seeds 0-9 (Exp C = seeds 5-9 new)
  2. Run attack/defense grid seeds 0-9 (run_seed + run_sweep)
  3. Experiment A: Adaptive attacker, seeds 0-9
  4. Experiment B: ResNet-20 training + attack grid (optional, can be slow)
  5. Experiment D: Targeted attack, seeds 0-9
  6. Aggregate results
  7. Generate figures

Usage:
    cd experiments/physbench
    python run_all_experiments.py
    python run_all_experiments.py --skip_resnet20          # skip Exp B (slow)
    python run_all_experiments.py --seeds 0 1 2 3 4        # only 5 seeds
    python run_all_experiments.py --resnet20_epochs 30     # shorter Exp B

All paths are relative to this file's parent directory (experiments/physbench/).
"""
import argparse
import subprocess
import sys
import os
import time

HERE    = os.path.dirname(os.path.abspath(__file__))
SRC     = os.path.join(HERE, "src")
PYTHON  = sys.executable


def run(cmd, cwd=SRC, label=""):
    """Run a shell command, streaming output, and raise on failure."""
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"  $ {cmd}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"  ✓ done in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Run all PhysBench experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                        help="Seeds to run (default: 0-9)")
    parser.add_argument("--skip_resnet20", action="store_true",
                        help="Skip Experiment B (ResNet-20), which is time-intensive")
    parser.add_argument("--skip_targeted", action="store_true",
                        help="Skip Experiment D (targeted attack)")
    parser.add_argument("--resnet20_seeds", type=int, nargs="+", default=list(range(5)),
                        help="Seeds for ResNet-20 training (default: 0-4)")
    parser.add_argument("--resnet20_epochs", type=int, default=100,
                        help="Epochs for ResNet-20 training (default: 100; use 30 for a quick run)")
    args = parser.parse_args()

    out_dir = HERE  # experiments/physbench/
    seeds_str  = " ".join(str(s) for s in args.seeds)
    rn_seeds_s = " ".join(str(s) for s in args.resnet20_seeds)

    log = {}
    grand_t0 = time.time()

    # ─────────────────────────────────────────────────────────────────────
    # Step 0: Install dependencies
    # ─────────────────────────────────────────────────────────────────────
    run(f"{PYTHON} -m pip install torch torchvision scipy matplotlib Pillow --quiet",
        cwd=HERE, label="Step 0: Install dependencies")

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Train SmallCNN seeds 0-9
    # ─────────────────────────────────────────────────────────────────────
    log["train_baseline"] = run(
        f"{PYTHON} train_baseline.py --seeds {seeds_str} --out_dir {out_dir}",
        label="Step 1: Train SmallCNN (all seeds)")

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Run seed attack/defense grid + parameter sweep
    # ─────────────────────────────────────────────────────────────────────
    t_seeds = 0.0
    for s in args.seeds:
        t_seeds += run(
            f"{PYTHON} run_seed.py {s} --out_dir {out_dir}",
            label=f"Step 2a: run_seed seed={s}")
    log["run_seed_all"] = t_seeds

    t_sweep = 0.0
    for s in args.seeds:
        t_sweep += run(
            f"{PYTHON} run_sweep.py {s} --out_dir {out_dir}",
            label=f"Step 2b: run_sweep seed={s}")
    log["run_sweep_all"] = t_sweep

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Experiment A — Adaptive attacker
    # ─────────────────────────────────────────────────────────────────────
    t_adap = 0.0
    for s in args.seeds:
        t_adap += run(
            f"{PYTHON} run_adaptive.py {s} --out_dir {out_dir}",
            label=f"Step 3: Exp A (adaptive) seed={s}")
    log["exp_a_adaptive"] = t_adap

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Experiment B — ResNet-20
    # ─────────────────────────────────────────────────────────────────────
    if not args.skip_resnet20:
        log["train_resnet20"] = run(
            f"{PYTHON} train_resnet20.py --seeds {rn_seeds_s} "
            f"--epochs {args.resnet20_epochs} --out_dir {out_dir}",
            label=f"Step 4a: Exp B — Train ResNet-20 ({args.resnet20_epochs} epochs)")

        t_rn = 0.0
        for s in args.resnet20_seeds:
            t_rn += run(
                f"{PYTHON} run_resnet20_seed.py {s} --out_dir {out_dir}",
                label=f"Step 4b: Exp B — attack/defense ResNet-20 seed={s}")
        log["exp_b_resnet20_attack"] = t_rn
    else:
        print("\n[SKIP] Experiment B (ResNet-20) skipped via --skip_resnet20")

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Experiment D — Targeted attack
    # ─────────────────────────────────────────────────────────────────────
    if not args.skip_targeted:
        t_tgt = 0.0
        for s in args.seeds:
            t_tgt += run(
                f"{PYTHON} run_targeted.py {s} --out_dir {out_dir}",
                label=f"Step 5: Exp D (targeted) seed={s}")
        log["exp_d_targeted"] = t_tgt
    else:
        print("\n[SKIP] Experiment D (targeted) skipped via --skip_targeted")

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Aggregate
    # ─────────────────────────────────────────────────────────────────────
    log["aggregate"] = run(
        f"{PYTHON} aggregate.py --seeds {seeds_str} --out_dir {out_dir}",
        label="Step 6: Aggregate all results")

    # ─────────────────────────────────────────────────────────────────────
    # Step 7: Figures
    # ─────────────────────────────────────────────────────────────────────
    log["make_figures"] = run(
        f"{PYTHON} make_figures.py --out_dir {out_dir}",
        label="Step 7: Generate figures")

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    total = time.time() - grand_t0
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE  total={total:.1f}s ({total/60:.1f} min)")
    print(f"{'='*60}")
    for k, v in log.items():
        print(f"  {k:<30} {v:.1f}s")
    print(f"\n  Results: {out_dir}/results/")
    print(f"  Figures: {out_dir}/figures/")


if __name__ == "__main__":
    main()
