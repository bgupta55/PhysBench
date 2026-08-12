"""
run_pqc_fast.py — Fast PQC co-location geometry simulation (n=10 seeds).

This is a geometry-only version: it does NOT re-run PBS (which is slow),
but uses the existing Tier-0 results from seed{0-9}.json as the T0 baseline
and computes T2-proxy ASR analytically from the Rowhammer reachability fraction.

The Tier-2 proxy ASR is estimated as:
   ASR_T2 ≈ ASR_T0 * FRR_DNN
because constrained PBS is limited to the FRR_DNN fraction of reachable DNN weights;
assuming PBS's gradient-ranked candidates are uniformly distributed (conservative),
the effective flip budget scales linearly with FRR_DNN.

This is explicitly labelled as a proof-of-concept analytical estimate.
Real Delta_k requires running constrained PBS on hardware or in the full sim.
"""
import sys, os, json
import numpy as np
import random

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO    = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from model import SmallCNN
from attack_defense import get_quant_state, ATTACKABLE_LAYERS_SMALL
from data_utils import get_data

import torch

# ── Layout constants (identical to run_pqc_coloc.py) ─────────────────────────
ROW_SIZE         = 64
HAMMERED_ROWS    = 4
PAGE_SIZE        = 256 * 1024
PAGE_START_BYTES = 128 * 1024
PQC_OFFSET_ROWS  = PAGE_SIZE // (2 * ROW_SIZE)
PQC_OFFSET_BYTES = PQC_OFFSET_ROWS * ROW_SIZE
KYBER_KEY_BYTES  = 1024   # 2 × 256 int16 coefficients
CKPT_DIR         = os.path.join(REPO, "ckpt")
RESULTS_DIR      = os.path.join(REPO, "results")
OUT_FILE         = os.path.join(RESULTS_DIR, "pqc_coloc_results.json")


def select_hammered_rows(seed):
    n_rows      = PAGE_SIZE // ROW_SIZE
    pqc_row_end = PQC_OFFSET_ROWS + (KYBER_KEY_BYTES + ROW_SIZE - 1) // ROW_SIZE
    boundary = []
    r = PQC_OFFSET_ROWS - 1
    if 1 <= r < n_rows - 1:
        boundary.append(r)
    r = pqc_row_end + 1
    if 1 <= r < n_rows - 1:
        boundary.append(r)
    rng = random.Random(seed + 9000)
    extras = list(range(2, n_rows - 2, 2))
    rng.shuffle(extras)
    selected, used = list(boundary), set(boundary)
    for r in extras:
        if all(abs(r - s) >= 2 for s in used):
            selected.append(r); used.add(r)
        if len(selected) == HAMMERED_ROWS:
            break
    return sorted(selected[:HAMMERED_ROWS])


def reachable_offsets(hammered_rows):
    n_rows = PAGE_SIZE // ROW_SIZE
    victims = set()
    for ar in hammered_rows:
        if ar - 1 >= 0:   victims.add(ar - 1)
        if ar + 1 < n_rows: victims.add(ar + 1)
    offsets = set()
    for row in victims:
        offsets.update(range(row * ROW_SIZE, (row + 1) * ROW_SIZE))
    return offsets


def state_byte_count(state):
    return sum(s["q"].numel() for s in state.values())


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    Xtr, ytr, Xte, yte = get_data(train_per_class=300, test_per_class=100, seed=0)

    per_seed = []
    for seed in range(10):
        # Load existing T0 results
        t0_path = os.path.join(RESULTS_DIR, f"seed{seed}.json")
        with open(t0_path) as f:
            t0 = json.load(f)
        asr_t0_undef = t0["none"]["asr"][-1]
        asr_t0_chk   = t0["checksum"]["asr"][-1]

        # Geometry
        hammered = select_hammered_rows(seed)
        reachable_all = reachable_offsets(hammered)

        pqc_range        = set(range(PQC_OFFSET_BYTES, PQC_OFFSET_BYTES + KYBER_KEY_BYTES))
        reachable_dnn    = reachable_all - pqc_range
        reachable_pqc    = reachable_all & pqc_range
        n_dnn_page       = PAGE_SIZE - KYBER_KEY_BYTES

        # Load model to get total weight count for DNN
        m = SmallCNN()
        m.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"model_seed{seed}.pt"), map_location="cpu"))
        m.eval()
        state       = get_quant_state(m, bits=8, attackable_layers=ATTACKABLE_LAYERS_SMALL)
        n_total_dnn = state_byte_count(state)

        # FRR_DNN = reachable DNN bytes in page / total attackable DNN bytes
        # (conservative: reachable_dnn bytes are all within the shared page slice)
        frr_dnn = len(reachable_dnn) / n_total_dnn

        # FRR_PQC = reachable PQC bytes / total PQC bytes
        frr_pqc = len(reachable_pqc) / KYBER_KEY_BYTES

        # T2 proxy: linear scaling (conservative lower-bound estimate)
        asr_t2_undef = asr_t0_undef * frr_dnn
        asr_t2_chk   = asr_t0_chk   * frr_dnn
        dk_undef     = asr_t0_undef - asr_t2_undef
        dk_chk       = asr_t0_chk   - asr_t2_chk

        # Avg coefs changed per PQC flip (1 byte = 1 int16 coef, so = 1.0)
        avg_coefs_per_flip = 1.0

        print(f"seed {seed}: FRR_DNN={frr_dnn:.4f}  FRR_PQC={frr_pqc:.4f}  "
              f"Δk_undef={dk_undef:.3f}  Δk_chk={dk_chk:.3f}")

        per_seed.append({
            "seed": seed,
            "hammered_rows":           hammered,
            "frr_dnn":                 frr_dnn,
            "frr_pqc":                 frr_pqc,
            "n_reachable_dnn":         len(reachable_dnn),
            "n_reachable_pqc_bytes":   len(reachable_pqc),
            "asr_t0_undefended":       asr_t0_undef,
            "asr_t0_checksum":         asr_t0_chk,
            "asr_t2_undefended":       asr_t2_undef,
            "asr_t2_checksum":         asr_t2_chk,
            "delta_k_undefended":      dk_undef,
            "delta_k_checksum":        dk_chk,
            "avg_coefs_changed_per_flip": avg_coefs_per_flip,
        })

    def ms(k):
        v = [r[k] for r in per_seed]
        return float(np.mean(v)), float(np.std(v))

    keys = ["frr_dnn", "frr_pqc", "asr_t0_undefended", "asr_t0_checksum",
            "asr_t2_undefended", "asr_t2_checksum",
            "delta_k_undefended", "delta_k_checksum",
            "n_reachable_pqc_bytes"]

    print("\n=== AGGREGATE RESULTS (n=10 seeds) ===")
    for k in keys:
        m, s = ms(k)
        print(f"  {k:42s}: {m:.4f} ± {s:.4f}")

    agg = {k: {"mean": ms(k)[0], "std": ms(k)[1]} for k in keys}
    summary = {
        "config": {
            "row_size_bytes":    ROW_SIZE,
            "page_size_bytes":   PAGE_SIZE,
            "pqc_offset_bytes":  PQC_OFFSET_BYTES,
            "kyber_key_bytes":   KYBER_KEY_BYTES,
            "n_hammered_rows":   HAMMERED_ROWS,
            "method":            "geometry_only_analytical_T2_proxy",
            "note":              "T2 ASR is analytical lower-bound estimate; real Delta_k requires constrained PBS on hardware."
        },
        "per_seed":  per_seed,
        "aggregate": agg,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
