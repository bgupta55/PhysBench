"""
Experiment 3 — DRAM Geometry Monte Carlo Sweep
Validates Phase 3.1 of the JSA Remediation Blueprint.

Purpose: Replace a single arbitrary geometry configuration with a distribution
swept over literature-informed parameter ranges (DeepHammer / TRRespass).

Deliverables:
  results/frr_distribution.csv   — per-trial FRR_PQC and FRR_DNN values
  results/frr_distribution.png   — histogram figure for paper
  results/exp3_summary.json      — aggregate statistics

Run:
  python scripts/exp3_geometry_sweep.py
"""
import json, hashlib, datetime, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Simulation parameters ─────────────────────────────────────────────────────
# FIPS 203 ML-KEM-512 real decapsulation key size (Phase 1.1)
DK_SIZE      = 1632
# DNN weight buffer size, ≈ SmallCNN — per original paper's Table 9 reference
DNN_BUF_SIZE = 601_472
PAGE_SIZE    = 256 * 1024   # 256 KB huge page

# Parameter ranges grounded in published Rowhammer profiling literature:
# DeepHammer (Yuan et al., USENIX Security 2020) — victim-row offsets ±1/±2,
#   aggressor counts 1–8; TRRespass (Frigo et al., S&P 2020) — similar ranges.
# Row buffer sizes from JEDEC DDR4/LPDDR4 specs: 1 KB, 2 KB, 4 KB banks.
AGGRESSOR_COUNTS  = [1, 2, 4, 8]          # from DeepHammer Table 2
VICTIM_OFFSETS    = [-2, -1, 1, 2]        # ± rows from aggressor, DeepHammer §4
ROW_SIZES_BYTES   = [1024, 2048, 4096]    # JEDEC LPDDR4/DDR4 bank-row sizes

N_TRIALS = 10_000
SEED     = 42


def vectorized_frr(dk_offset, dnn_offset, aggressor_positions, victim_offset, row_size, page_size):
    """
    Compute fraction of bytes of dk/dnn buffer that fall within at least one victim row.
    Vectorized over byte range using numpy boolean masks.
    """
    dk_bytes  = np.arange(dk_offset, dk_offset + DK_SIZE)
    dnn_len   = min(DNN_BUF_SIZE, page_size)
    dnn_bytes = np.arange(dnn_offset, dnn_offset + dnn_len)

    dk_in  = np.zeros(DK_SIZE,  dtype=bool)
    dnn_in = np.zeros(dnn_len,  dtype=bool)

    for a in aggressor_positions:
        v_start = a + victim_offset * row_size
        v_end   = v_start + row_size
        dk_in  |= (dk_bytes  >= v_start) & (dk_bytes  < v_end)
        dnn_in |= (dnn_bytes >= v_start) & (dnn_bytes < v_end)

    return dk_in.sum() / DK_SIZE, dnn_in.sum() / dnn_len


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR.parent
        ).decode().strip()
    except Exception:
        return "N/A"


def main():
    rng = np.random.default_rng(seed=SEED)

    # Sample configuration parameters for each trial
    aggressor_counts = rng.choice(AGGRESSOR_COUNTS, size=N_TRIALS)
    victim_offsets_v = rng.choice(VICTIM_OFFSETS,   size=N_TRIALS)
    row_sizes        = rng.choice(ROW_SIZES_BYTES,  size=N_TRIALS)

    frr_pqc = np.zeros(N_TRIALS)
    frr_dnn = np.zeros(N_TRIALS)

    print(f"[Exp3] Running {N_TRIALS:,} Monte Carlo geometry trials …")
    for i in range(N_TRIALS):
        row = int(row_sizes[i])
        n_agg = int(aggressor_counts[i])
        v_off = int(victim_offsets_v[i])

        # Randomly place dk within page
        dk_offset = int(rng.integers(0, max(1, PAGE_SIZE - DK_SIZE)))
        dnn_offset = int(rng.integers(0, max(1, PAGE_SIZE - min(DNN_BUF_SIZE, PAGE_SIZE))))

        # Aggressor rows, expressed as byte-start positions
        n_rows_in_page = PAGE_SIZE // row
        agg_rows = rng.integers(0, n_rows_in_page, size=n_agg) * row
        agg_positions = [int(a) for a in agg_rows]

        frr_pqc[i], frr_dnn[i] = vectorized_frr(
            dk_offset, dnn_offset, agg_positions, v_off, row, PAGE_SIZE
        )

        if (i + 1) % 2000 == 0:
            print(f"  … {i+1}/{N_TRIALS} trials done", flush=True)

    df = pd.DataFrame({
        "trial":        range(N_TRIALS),
        "aggressor_count": aggressor_counts,
        "victim_offset":   victim_offsets_v,
        "row_size_bytes":  row_sizes,
        "frr_pqc":      frr_pqc,
        "frr_dnn":      frr_dnn,
    })
    csv_out = RESULTS_DIR / "frr_distribution.csv"
    df.to_csv(csv_out, index=False)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title, color in [
        (axes[0], frr_pqc, "FRR_PQC (ML-KEM-512 dk, 1632 B)", "#2563eb"),
        (axes[1], frr_dnn, "FRR_DNN (SmallCNN buffer, ≤256KB)", "#16a34a"),
    ]:
        ax.hist(col, bins=60, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(col),  color="red",    linestyle="--", linewidth=1.5,
                   label=f"Median = {np.median(col):.3f}")
        ax.axvline(np.percentile(col,  5), color="orange", linestyle=":",  linewidth=1.2,
                   label=f"P5 = {np.percentile(col, 5):.3f}")
        ax.axvline(np.percentile(col, 95), color="orange", linestyle=":",  linewidth=1.2,
                   label=f"P95 = {np.percentile(col,95):.3f}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Fraction of Bytes in Victim Row(s)")
        ax.set_ylabel("Count (of 10,000 trials)")
        ax.legend(fontsize=9)

    fig.suptitle(
        f"DRAM Geometry Monte Carlo Sweep — N={N_TRIALS:,} trials\n"
        "(Row offsets/aggressor counts grounded in DeepHammer/TRRespass literature)",
        fontsize=10,
    )
    plt.tight_layout()
    fig_out = RESULTS_DIR / "frr_distribution.png"
    plt.savefig(fig_out, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    stats = {}
    for name, arr in [("FRR_PQC", frr_pqc), ("FRR_DNN", frr_dnn)]:
        stats[name] = {
            "median":  round(float(np.median(arr)),         4),
            "mean":    round(float(np.mean(arr)),           4),
            "p5":      round(float(np.percentile(arr,  5)), 4),
            "p95":     round(float(np.percentile(arr, 95)), 4),
            "max":     round(float(np.max(arr)),            4),
        }

    summary = {
        "n_trials": N_TRIALS,
        "seed": SEED,
        "dk_size_bytes": DK_SIZE,
        "parameter_ranges": {
            "aggressor_counts":   AGGRESSOR_COUNTS,
            "victim_offsets":     VICTIM_OFFSETS,
            "row_sizes_bytes":    ROW_SIZES_BYTES,
            "literature_sources": ["DeepHammer (Yuan et al., USENIX Sec 2020)",
                                   "TRRespass (Frigo et al., S&P 2020)"],
        },
        "statistics": stats,
        "csv_sha256": hashlib.sha256(open(csv_out, "rb").read()).hexdigest(),
        "git_commit": get_git_commit(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    json.dump(summary, open(RESULTS_DIR / "exp3_summary.json", "w"), indent=2)

    print("\n[Exp3] RESULTS:")
    for name, s in stats.items():
        print(f"  {name:8s}  median={s['median']:.4f}  p5={s['p5']:.4f}  p95={s['p95']:.4f}")
    print(f"\n  Histogram → {fig_out}")
    print(f"  CSV       → {csv_out}")


if __name__ == "__main__":
    main()
