"""
make_figures.py — Regenerate all figures from results.

Usage:
    python make_figures.py [--out_dir PATH]

Reads results/summary.json + per-seed JSONs.
Writes figures/* (PDF + PNG, 200 dpi).

Generates:
  Fig 1:  ASR trajectory (original seeds)
  Fig 2:  Defense comparison bar chart
  Fig 3:  Clean-accuracy retention
  Fig 4:  Architecture harness diagram
  Fig 5:  Adaptive attacker 3-bar comparison (Exp A)
  Fig 6:  Seeds 0-9 ASR trajectory (Exp C, if data present)
  Fig 7:  ResNet-20 defense comparison (Exp B, if data present)
  Fig 8:  Targeted attack comparison (Exp D, if data present)
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


DEFENSES = ["none", "checksum", "clip"]
LABELS   = {"none": "Undefended", "checksum": "Checksum (ours)", "clip": "Range-clip (ours)"}
COLORS   = {"none": "#d62728",    "checksum": "#1f77b4",          "clip": "#2ca02c"}
BUDGET   = 20


def savefig(fig, path_no_ext, dpi=200):
    fig.savefig(path_no_ext + ".pdf")
    fig.savefig(path_no_ext + ".png", dpi=dpi)
    plt.close(fig)
    print(f"  saved {os.path.basename(path_no_ext)}.pdf/png")


def make_figures(out_dir):
    results_dir = os.path.join(out_dir, "results")
    fig_dir     = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(summary_path):
        print("No summary.json found; run aggregate.py first.")
        return

    summary     = json.load(open(summary_path))
    seeds_used  = summary.get("seeds_used", list(range(5)))
    all_results = {s: json.load(open(os.path.join(results_dir, f"seed{s}.json")))
                   for s in seeds_used
                   if os.path.exists(os.path.join(results_dir, f"seed{s}.json"))}

    if not all_results:
        print("No per-seed results found.")
        return

    x = np.arange(1, BUDGET + 1)

    # Helper: pad traj to BUDGET length
    def pad(lst):
        lst = list(lst)
        while len(lst) < BUDGET:
            lst.append(lst[-1])
        return np.array(lst[:BUDGET])

    # ── Figure 1: ASR trajectory ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for d in DEFENSES:
        traj = np.array([pad(all_results[s][d]["asr"]) for s in seeds_used])
        mean, std = traj.mean(0), traj.std(0)
        ax.plot(x, mean, label=LABELS[d], color=COLORS[d],
                marker="o", markersize=3, linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, color=COLORS[d], alpha=0.15)
    ax.set_xlabel("Number of bit flips (progressive bit search)")
    ax.set_ylabel("Attack success rate (ASR)")
    ax.set_title(f"Tier-0 PBS ASR vs. flip budget (n={len(seeds_used)} seeds)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, "fig_asr_trajectory"))

    # ── Figure 2: defense comparison bar chart ────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 4))
    means = [summary["asr_at_budget"][d]["mean"] * 100 for d in DEFENSES]
    stds  = [summary["asr_at_budget"][d]["std"]  * 100 for d in DEFENSES]
    bars  = ax.bar([LABELS[d] for d in DEFENSES], means, yerr=stds, capsize=4,
                   color=[COLORS[d] for d in DEFENSES])
    ax.set_ylabel("ASR at 20-flip budget (%)")
    ax.set_title(f"Defense comparison, Tier 0 (n={len(seeds_used)} seeds)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=15)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, "fig_defense_comparison"))

    # ── Figure 3: clean-accuracy retention ───────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for d in DEFENSES:
        traj = np.array([pad(all_results[s][d]["clean_acc"]) for s in seeds_used])
        mean, std = traj.mean(0), traj.std(0)
        ax.plot(x, mean, label=LABELS[d], color=COLORS[d],
                marker="s", markersize=3, linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, color=COLORS[d], alpha=0.15)
    ax.set_xlabel("Number of bit flips")
    ax.set_ylabel("Clean test accuracy")
    ax.set_title("Clean-accuracy retention under attack")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, "fig_clean_acc"))

    # ── Figure 4: architecture harness diagram ────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    boxes = [
        (0.3, 1.5, 1.8, 1.0, "Quantized\nmodel\n(int8)"),
        (2.6, 1.5, 2.0, 1.0, "Progressive\nbit search\n(attack)"),
        (5.1, 2.6, 2.0, 1.0, "Defense:\ncheck/rollback"),
        (5.1, 0.4, 2.0, 1.0, "Defense:\nrange clip"),
        (7.6, 1.5, 2.1, 1.0, "Evaluation:\nASR, accuracy,\nTier-transfer gap"),
    ]
    for bx, by, w, h, label in boxes:
        box = FancyBboxPatch((bx, by), w, h,
                             boxstyle="round,pad=0.05,rounding_size=0.08",
                             linewidth=1.2, edgecolor="#333333", facecolor="#eef3f8")
        ax.add_patch(box)
        ax.text(bx + w / 2, by + h / 2, label,
                ha="center", va="center", fontsize=8.3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, linewidth=1.1,
                                     color="#333333"))

    arrow(2.1, 2.0, 2.6, 2.0)
    arrow(4.6, 2.3, 5.1, 2.9)
    arrow(4.6, 1.7, 5.1, 1.1)
    arrow(7.1, 3.1, 7.6, 2.3)
    arrow(7.1, 0.9, 7.6, 1.7)
    ax.text(5.05, 3.85, "Tier-0 harness (this study, executed)",
            fontsize=9.5, ha="left", fontweight="bold")
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, "fig_architecture"))

    # ── Figure 5: adaptive attacker (Exp A) ──────────────────────────────
    adap_summary_path = os.path.join(results_dir, "adaptive", "adaptive_summary.json")
    adap_dir          = os.path.join(results_dir, "adaptive")
    if os.path.exists(adap_summary_path):
        adap_summary = json.load(open(adap_summary_path))
        adap_seeds   = adap_summary["seeds_used"]
        adap_res     = {s: json.load(open(os.path.join(adap_dir, f"adaptive_seed{s}.json")))
                        for s in adap_seeds}

        conds  = ["none", "checksum", "checksum_adaptive"]
        labels = {"none": "Undefended",
                  "checksum": "Checksum\n(non-adaptive)",
                  "checksum_adaptive": "Checksum\n(adaptive)"}
        colors = {"none": "#d62728", "checksum": "#1f77b4", "checksum_adaptive": "#ff7f0e"}

        # 3-bar comparison
        fig, ax = plt.subplots(figsize=(5.5, 4))
        means = [adap_summary["asr_at_budget"][c]["mean"] * 100 for c in conds]
        stds  = [adap_summary["asr_at_budget"][c]["std"]  * 100 for c in conds]
        ax.bar([labels[c] for c in conds], means, yerr=stds, capsize=4,
               color=[colors[c] for c in conds])
        ax.set_ylabel("ASR at 20-flip budget (%)")
        ax.set_title(f"Experiment A: Adaptive attacker (n={len(adap_seeds)} seeds)")
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=10)
        fig.tight_layout()
        savefig(fig, os.path.join(fig_dir, "fig_adaptive_comparison"))

        # Trajectory comparison
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for c in conds:
            traj = np.array([pad(adap_res[s][c]["asr"]) for s in adap_seeds])
            mean, std = traj.mean(0), traj.std(0)
            ax.plot(x, mean, label=labels[c], color=colors[c],
                    marker="o", markersize=3, linewidth=1.8)
            ax.fill_between(x, mean - std, mean + std, color=colors[c], alpha=0.15)
        ax.set_xlabel("Number of bit flips")
        ax.set_ylabel("Attack success rate (ASR)")
        ax.set_title("Exp A: Adaptive vs. non-adaptive attacker ASR trajectory")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        savefig(fig, os.path.join(fig_dir, "fig_adaptive_trajectory"))

    # ── Figure 6: n=10 seeds trajectory (Exp C) ──────────────────────────
    seeds_10 = [s for s in range(10)
                if os.path.exists(os.path.join(results_dir, f"seed{s}.json"))]
    if len(seeds_10) >= 6:
        res10 = {s: json.load(open(os.path.join(results_dir, f"seed{s}.json")))
                 for s in seeds_10}
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for d in DEFENSES:
            traj = np.array([pad(res10[s][d]["asr"]) for s in seeds_10])
            mean, std = traj.mean(0), traj.std(0)
            ax.plot(x, mean, label=LABELS[d], color=COLORS[d],
                    marker="o", markersize=3, linewidth=1.8)
            ax.fill_between(x, mean - std, mean + std, color=COLORS[d], alpha=0.15)
        ax.set_xlabel("Number of bit flips")
        ax.set_ylabel("ASR")
        ax.set_title(f"Exp C: ASR trajectory, n={len(seeds_10)} seeds")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        savefig(fig, os.path.join(fig_dir, "fig_n10_asr_trajectory"))

    # ── Figure 7: ResNet-20 (Exp B) ───────────────────────────────────────
    rn_summary_path = os.path.join(results_dir, "resnet20", "resnet20_summary.json")
    if os.path.exists(rn_summary_path):
        rn_summary = json.load(open(rn_summary_path))
        rn_seeds   = rn_summary["seeds_used"]
        rn_res     = {s: json.load(open(os.path.join(results_dir, "resnet20",
                                                      f"resnet20_seed{s}.json")))
                      for s in rn_seeds}

        # Bar chart
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, (title, smry) in zip(axes, [
            (f"SmallCNN (n={len(seeds_used)})", summary),
            (f"ResNet-20 (n={len(rn_seeds)})",  rn_summary)
        ]):
            means = [smry["asr_at_budget"][d]["mean"] * 100 for d in DEFENSES]
            stds  = [smry["asr_at_budget"][d]["std"]  * 100 for d in DEFENSES]
            ax.bar([LABELS[d] for d in DEFENSES], means, yerr=stds, capsize=4,
                   color=[COLORS[d] for d in DEFENSES])
            ax.set_ylabel("ASR at 20-flip budget (%)")
            ax.set_title(title)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis='x', rotation=15)
        fig.suptitle("Exp B: Scale comparison — SmallCNN vs. ResNet-20", y=1.02)
        fig.tight_layout()
        savefig(fig, os.path.join(fig_dir, "fig_scale_comparison"))

        # ResNet-20 trajectory
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for d in DEFENSES:
            traj = np.array([pad(rn_res[s][d]["asr"]) for s in rn_seeds])
            mean, std = traj.mean(0), traj.std(0)
            ax.plot(x, mean, label=LABELS[d], color=COLORS[d],
                    marker="o", markersize=3, linewidth=1.8)
            ax.fill_between(x, mean - std, mean + std, color=COLORS[d], alpha=0.15)
        ax.set_xlabel("Number of bit flips")
        ax.set_ylabel("ASR")
        ax.set_title(f"Exp B: ResNet-20 PBS ASR trajectory (n={len(rn_seeds)} seeds)")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        savefig(fig, os.path.join(fig_dir, "fig_resnet20_trajectory"))

    # ── Figure 8: targeted attack (Exp D) ────────────────────────────────
    tgt_summary_path = os.path.join(results_dir, "targeted", "targeted_summary.json")
    if os.path.exists(tgt_summary_path):
        tgt_summary  = json.load(open(tgt_summary_path))
        pair_labels  = list(tgt_summary["per_pair"].keys())
        def_keys     = ["none", "checksum", "clip"]
        def_labels   = {"none": "Undefended", "checksum": "Checksum", "clip": "Clip"}
        pair_colors  = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        fig, axes = plt.subplots(1, len(pair_labels), figsize=(4 * len(pair_labels), 4),
                                 sharey=True)
        if len(pair_labels) == 1:
            axes = [axes]
        for ax, (pl, pc) in zip(axes, zip(pair_labels, pair_colors)):
            means = [tgt_summary["per_pair"][pl][dk]["mean_targeted_asr"] * 100
                     for dk in def_keys]
            stds  = [tgt_summary["per_pair"][pl][dk]["std_targeted_asr"] * 100
                     for dk in def_keys]
            ax.bar([def_labels[dk] for dk in def_keys], means, yerr=stds, capsize=4,
                   color=[COLORS[dk] for dk in def_keys])
            ax.set_title(pl, fontsize=9)
            ax.set_ylabel("Targeted ASR (%)" if ax is axes[0] else "")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis='x', rotation=15)
        fig.suptitle("Exp D: Targeted attack targeted-ASR at 20-flip budget", y=1.02)
        fig.tight_layout()
        savefig(fig, os.path.join(fig_dir, "fig_targeted_comparison"))

    print("\nAll figures generated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), ".."))
    args = parser.parse_args()
    make_figures(args.out_dir)


if __name__ == "__main__":
    main()
