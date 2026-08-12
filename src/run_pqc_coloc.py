"""
Experiment E: PQC Co-location Simulation (Proof-of-Concept)
=============================================================
Simulates a shared DRAM page layout in which DNN weight bytes and a mock
Kyber-512 private-key buffer are interleaved within the same set of DRAM
rows.  This reflects the realistic scenario where the OS heap allocator
places a PQC key buffer on a page that also contains DNN weight data, so
the same Rowhammer aggressor rows can induce bit flips in both targets.

Layout design
-------------
We model a 256 KB "shared page region" extracted from the middle of the
DNN weight flat buffer (~600 KB for SmallCNN).  Within that region we
carve out a 1024-byte slot (the Kyber-512 private key) at PQC_OFFSET_BYTES
from the start of the shared page — exactly halfway through — so hammered
aggressor rows near that boundary straddle DNN weight bytes and PQC key
bytes simultaneously.

For each of N_SEEDS seeds we:
  1. Extract the shared 256 KB page from the DNN weight flat buffer.
  2. Embed the Kyber-512 key at the mid-page offset.
  3. Select HAMMERED_ROWS aggressor rows near the DNN/PQC boundary.
  4. Compute victim rows; partition reachable bytes into DNN vs. PQC.
  5. Run PBS constrained to reachable DNN bytes (Tier-2 proxy ASR).
  6. Run PBS unconstrained (Tier-0 baseline ASR).
  7. Compute Delta_k = ASR(T0) - ASR(T2) for undefended and checksum.
  8. Count reachable PQC key bytes; simulate single-bit flips and count
     how many Kyber coefficients change per flip.

Output: results/pqc_coloc_results.json (per-seed + aggregate).
This is a software-level co-location simulation, NOT hardware-validated.
It is labelled exactly that in the manuscript.
"""

import sys
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F

# ── path setup ─────────────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from model import SmallCNN
from quant import dequantize_tensor, flip_bit
from attack_defense import (
    get_quant_state, apply_state_to_model, build_protection_mask,
    progressive_bit_search, ATTACKABLE_LAYERS_SMALL,
)
from data_utils import get_data

# ── constants ──────────────────────────────────────────────────────────────
N_SEEDS        = 10
ROW_SIZE       = 64           # bytes per simulated DRAM row (64-byte cache line)
HAMMERED_ROWS  = 4            # number of aggressor rows the attacker hammers
MAX_FLIPS      = 20
TOPK           = 8
BITS           = 8

# Shared DRAM page: 256 KB slice from the middle of the DNN weight buffer.
PAGE_SIZE        = 256 * 1024                    # 256 KB = 4096 rows × 64 B
PAGE_START_BYTES = 128 * 1024                    # start 128 KB into weight buf

# PQC key is placed at the exact midpoint of the shared page.
# Aggressor rows near this midpoint will straddle both DNN and PQC bytes.
PQC_OFFSET_ROWS  = PAGE_SIZE // (2 * ROW_SIZE)  # row 2048 out of 4096
PQC_OFFSET_BYTES = PQC_OFFSET_ROWS * ROW_SIZE   # byte 131072 within page

# Kyber-512 private key: 2 polynomials × 256 int16 coefficients = 1024 bytes
KYBER_N_POLYS    = 2
KYBER_N_COEFS    = 256
KYBER_COEF_BYTES = 2
KYBER_KEY_BYTES  = KYBER_N_POLYS * KYBER_N_COEFS * KYBER_COEF_BYTES  # 1024 B
KYBER_COEF_MIN   = -3
KYBER_COEF_MAX   =  3

CKPT_DIR    = os.path.join(os.path.dirname(SRC_DIR), "ckpt")
RESULTS_DIR = os.path.join(os.path.dirname(SRC_DIR), "results")
OUT_FILE    = os.path.join(RESULTS_DIR, "pqc_coloc_results.json")


# ── Kyber key helpers ───────────────────────────────────────────────────────

def make_kyber_key(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(KYBER_COEF_MIN, KYBER_COEF_MAX + 1,
                        size=(KYBER_N_POLYS, KYBER_N_COEFS), dtype=np.int16)


def key_to_bytes(key):
    return key.astype("<i2").tobytes()


def bytes_to_key(b):
    return np.frombuffer(b, dtype="<i2").reshape(KYBER_N_POLYS, KYBER_N_COEFS).copy()


# ── Layout helpers ──────────────────────────────────────────────────────────

def build_shared_page(weight_bytes, pqc_key_bytes):
    """
    Extract PAGE_SIZE bytes from the DNN weight flat buffer and embed the
    PQC key at PQC_OFFSET_BYTES.  Returns the shared page as a bytearray.
    """
    assert PAGE_START_BYTES + PAGE_SIZE <= len(weight_bytes), (
        f"Weight buffer ({len(weight_bytes)} B) too small for "
        f"PAGE_START={PAGE_START_BYTES} + PAGE_SIZE={PAGE_SIZE}"
    )
    assert PQC_OFFSET_BYTES + KYBER_KEY_BYTES <= PAGE_SIZE

    page = bytearray(weight_bytes[PAGE_START_BYTES : PAGE_START_BYTES + PAGE_SIZE])
    page[PQC_OFFSET_BYTES : PQC_OFFSET_BYTES + KYBER_KEY_BYTES] = pqc_key_bytes
    return page


def select_hammered_rows(seed):
    """
    Choose HAMMERED_ROWS aggressor rows near the DNN/PQC boundary so that
    victim rows (aggressor +/- 1) straddle both regions.

    PQC key occupies rows [PQC_OFFSET_ROWS, PQC_OFFSET_ROWS + PQC_ROW_SPAN).
    We pick rows immediately before and after that block, then pad with
    evenly-spaced rows if more are needed.
    """
    n_rows      = PAGE_SIZE // ROW_SIZE  # 4096
    pqc_row_end = PQC_OFFSET_ROWS + (KYBER_KEY_BYTES + ROW_SIZE - 1) // ROW_SIZE

    # Boundary aggressor rows: their victim rows touch the PQC key block
    boundary = []
    # Just before PQC start: victim row = PQC_OFFSET_ROWS  (first PQC row)
    r = PQC_OFFSET_ROWS - 1
    if 1 <= r < n_rows - 1:
        boundary.append(r)
    # Just after PQC end: victim row = pqc_row_end  (first DNN row after key)
    r = pqc_row_end + 1
    if 1 <= r < n_rows - 1:
        boundary.append(r)

    # Pad with evenly-spaced rows (spacing >= 2 enforced)
    rng    = random.Random(seed + 9000)
    extras = list(range(2, n_rows - 2, 2))
    rng.shuffle(extras)

    selected = list(boundary)
    used     = set(selected)
    for r in extras:
        if all(abs(r - s) >= 2 for s in used):
            selected.append(r)
            used.add(r)
        if len(selected) == HAMMERED_ROWS:
            break

    return sorted(selected[:HAMMERED_ROWS])


def reachable_in_page(hammered_rows):
    """
    Return the set of byte offsets (within the shared page) reachable by
    Rowhammer.  Victim rows are aggressor +/- 1.
    """
    n_rows    = PAGE_SIZE // ROW_SIZE
    victim_rs = set()
    for ar in hammered_rows:
        if ar - 1 >= 0:
            victim_rs.add(ar - 1)
        if ar + 1 < n_rows:
            victim_rs.add(ar + 1)
    offsets = set()
    for row in victim_rs:
        offsets.update(range(row * ROW_SIZE, (row + 1) * ROW_SIZE))
    return offsets


# ── Weight-byte mapping ─────────────────────────────────────────────────────

def state_to_bytes(state):
    """Serialize quantized weight state to a flat bytes object (int8, 1 B/weight)."""
    return b"".join(
        state[name]["q"].flatten().numpy().astype(np.int8).tobytes()
        for name in sorted(state.keys())
    )


def state_byte_map(state):
    """
    Map (layer_name, flat_index) -> byte_offset in the full weight flat buffer.
    """
    mapping = {}
    offset  = 0
    for name in sorted(state.keys()):
        for i in range(state[name]["q"].numel()):
            mapping[(name, i)] = offset
            offset += 1
    return mapping


# ── Constrained PBS ─────────────────────────────────────────────────────────

def run_pbs_constrained(model, state, X_atk, y_atk, X_eval, y_eval,
                         reachable_dnn_page_offsets, byte_map,
                         defense=None, protect_mask=None, seed=0):
    """
    PBS restricted to weight elements whose byte offset (in the full weight
    flat buffer) falls within [PAGE_START_BYTES, PAGE_START_BYTES + PAGE_SIZE)
    AND in reachable_dnn_page_offsets (converted to absolute offsets).

    reachable_dnn_page_offsets: set of byte offsets within the shared PAGE
                                 (0-based within page) that are DNN bytes.
    """
    torch.manual_seed(seed)

    # Convert page-relative offsets to absolute weight-buffer offsets
    reachable_abs = {PAGE_START_BYTES + off for off in reachable_dnn_page_offsets}

    # Build (name, idx) -> bool lookup
    is_reachable = {key: (off in reachable_abs) for key, off in byte_map.items()}

    traj = {"flip_idx": [], "asr": [], "n_effective_flips": []}
    n_eff = 0

    def cur_asr():
        with torch.no_grad():
            return (model(X_eval).argmax(1) != y_eval).float().mean().item()

    for step in range(1, MAX_FLIPS + 1):
        apply_state_to_model(model, state)
        model.zero_grad()
        for name in state:
            dict(model.named_parameters())[name].requires_grad_(True)

        F.cross_entropy(model(X_atk), y_atk).backward()

        # Rank reachable candidates by gradient magnitude
        cands = []
        for name in state:
            grad = dict(model.named_parameters())[name].grad
            if grad is None:
                continue
            fg = grad.abs().flatten()
            for idx in range(fg.numel()):
                if is_reachable.get((name, idx), False):
                    cands.append((name, idx, fg[idx].item()))
        if not cands:
            break

        cands.sort(key=lambda x: -x[2])
        top = cands[:TOPK]

        best = None
        for name, idx, _ in top:
            flat_q  = state[name]["q"].flatten()
            orig    = int(flat_q[idx].item())
            for bp in range(BITS):
                flat_q[idx] = flip_bit(orig, bp, BITS)
                apply_state_to_model(model, state)
                with torch.no_grad():
                    loss = F.cross_entropy(model(X_atk), y_atk).item()
                if best is None or loss > best[0]:
                    best = (loss, name, idx, bp, int(flat_q[idx].item()))
                flat_q[idx] = orig
            apply_state_to_model(model, state)

        if best is None:
            break

        _, name, idx, bp, new_val = best
        rollback = int(state[name]["q"].flatten()[idx].item())
        state[name]["q"].flatten()[idx] = new_val

        protected = False
        if defense == "checksum" and protect_mask and name in protect_mask:
            protected = bool(protect_mask[name].flatten()[idx].item())
            if protected:
                state[name]["q"].flatten()[idx] = rollback
            else:
                n_eff += 1
        else:
            n_eff += 1

        apply_state_to_model(model, state)
        asr = cur_asr()
        traj["flip_idx"].append(step)
        traj["asr"].append(asr)
        traj["n_effective_flips"].append(n_eff)
        if asr >= 0.90:
            break

    return traj


# ── PQC impact measurement ──────────────────────────────────────────────────

def measure_pqc_impact(kyber_key, reachable_all_page_offsets):
    """
    Count reachable PQC key bytes and simulate single-bit flips.

    reachable_all_page_offsets: set of byte offsets within the shared page
                                 (0-based) that are reachable by Rowhammer.
    """
    pqc_range = range(PQC_OFFSET_BYTES, PQC_OFFSET_BYTES + KYBER_KEY_BYTES)
    reachable_pqc = [off for off in reachable_all_page_offsets if off in pqc_range]

    n_reachable_bits = len(reachable_pqc) * 8
    n_total_bits     = KYBER_KEY_BYTES * 8
    frr_pqc          = len(reachable_pqc) / KYBER_KEY_BYTES

    orig_bytes = bytearray(key_to_bytes(kyber_key))
    changed_counts = []
    for page_off in reachable_pqc:
        local_off = page_off - PQC_OFFSET_BYTES
        orig_byte = orig_bytes[local_off]
        for bp in range(8):
            corrupted = bytearray(orig_bytes)
            corrupted[local_off] = orig_byte ^ (1 << bp)
            diff = int(np.sum(bytes_to_key(bytes(corrupted)) != kyber_key))
            changed_counts.append(diff)

    avg_changed = float(np.mean(changed_counts)) if changed_counts else 0.0
    return {
        "frr_pqc":                      frr_pqc,
        "n_reachable_pqc_bytes":        len(reachable_pqc),
        "n_reachable_pqc_bits":         n_reachable_bits,
        "n_total_pqc_bits":             n_total_bits,
        "avg_coefs_changed_per_flip":   avg_changed,
        "any_pqc_reachable":            len(reachable_pqc) > 0,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cpu"

    print(f"Shared page: {PAGE_SIZE//1024} KB  "
          f"(bytes {PAGE_START_BYTES}–{PAGE_START_BYTES+PAGE_SIZE} of weight buffer)")
    print(f"PQC key offset within page: byte {PQC_OFFSET_BYTES} "
          f"(row {PQC_OFFSET_ROWS} of {PAGE_SIZE//ROW_SIZE})")
    print(f"Hammered rows per seed: {HAMMERED_ROWS}  |  Row size: {ROW_SIZE} B\n")

    # Load data once (seed=0 fixed, same as run_seed.py convention)
    Xtr, ytr, Xte, yte = get_data(train_per_class=300, test_per_class=100, seed=0)

    results = []

    for seed in range(N_SEEDS):
        print(f"=== Seed {seed} ===")
        torch.manual_seed(seed)

        X_atk  = Xtr[seed * 32 : seed * 32 + 32].to(device)
        y_atk  = ytr[seed * 32 : seed * 32 + 32].to(device)
        X_eval = Xte[:200].to(device)
        y_eval = yte[:200].to(device)

        ckpt_path = os.path.join(CKPT_DIR, f"model_seed{seed}.pt")

        def fresh_model():
            m = SmallCNN().to(device)
            m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            m.eval()
            return m

        # Reference state for layout / mask (not modified by attacks)
        ref_model = fresh_model()
        ref_state = get_quant_state(ref_model, bits=BITS,
                                    attackable_layers=ATTACKABLE_LAYERS_SMALL)
        protect_mask = build_protection_mask(ref_state, protect_fraction=0.50, seed=seed)
        byte_map     = state_byte_map(ref_state)

        # Build shared page and embed Kyber key
        weight_bytes_raw = state_to_bytes(ref_state)
        kyber_key        = make_kyber_key(seed + 100)
        pqc_bytes_raw    = key_to_bytes(kyber_key)
        _shared_page     = build_shared_page(weight_bytes_raw, pqc_bytes_raw)

        # Choose hammered rows and compute reachable sets
        hammered_rows      = select_hammered_rows(seed)
        reachable_all_page = reachable_in_page(hammered_rows)

        # Split into DNN reachable and PQC reachable
        pqc_range            = range(PQC_OFFSET_BYTES, PQC_OFFSET_BYTES + KYBER_KEY_BYTES)
        reachable_dnn_page   = {off for off in reachable_all_page if off not in pqc_range}
        n_dnn_page_bytes     = PAGE_SIZE - KYBER_KEY_BYTES
        frr_dnn              = len(reachable_dnn_page) / n_dnn_page_bytes

        print(f"  Hammered rows: {hammered_rows}")
        print(f"  DNN reachable in page: {len(reachable_dnn_page)}/{n_dnn_page_bytes} B  "
              f"(FRR_DNN = {frr_dnn:.4f})")

        # ── Tier-0 runs ──────────────────────────────────────────────────
        m0u = fresh_model()
        s0u = get_quant_state(m0u, bits=BITS, attackable_layers=ATTACKABLE_LAYERS_SMALL)
        t0u = progressive_bit_search(m0u, s0u, X_atk, y_atk, X_eval, y_eval,
                                     max_flips=MAX_FLIPS, topk=TOPK, bits=BITS,
                                     defense=None, seed=seed)
        asr_t0_undef = t0u["asr"][-1] if t0u["asr"] else 0.0

        m0c = fresh_model()
        s0c = get_quant_state(m0c, bits=BITS, attackable_layers=ATTACKABLE_LAYERS_SMALL)
        t0c = progressive_bit_search(m0c, s0c, X_atk, y_atk, X_eval, y_eval,
                                     max_flips=MAX_FLIPS, topk=TOPK, bits=BITS,
                                     defense="checksum", protect_mask=protect_mask, seed=seed)
        asr_t0_chk = t0c["asr"][-1] if t0c["asr"] else 0.0

        # ── Tier-2 proxy runs (constrained to reachable DNN bytes) ───────
        m2u = fresh_model()
        s2u = get_quant_state(m2u, bits=BITS, attackable_layers=ATTACKABLE_LAYERS_SMALL)
        t2u = run_pbs_constrained(m2u, s2u, X_atk, y_atk, X_eval, y_eval,
                                   reachable_dnn_page_offsets=reachable_dnn_page,
                                   byte_map=byte_map, defense=None, seed=seed)
        asr_t2_undef = t2u["asr"][-1] if t2u["asr"] else 0.0

        m2c = fresh_model()
        s2c = get_quant_state(m2c, bits=BITS, attackable_layers=ATTACKABLE_LAYERS_SMALL)
        t2c = run_pbs_constrained(m2c, s2c, X_atk, y_atk, X_eval, y_eval,
                                   reachable_dnn_page_offsets=reachable_dnn_page,
                                   byte_map=byte_map,
                                   defense="checksum", protect_mask=protect_mask, seed=seed)
        asr_t2_chk = t2c["asr"][-1] if t2c["asr"] else 0.0

        dk_undef = asr_t0_undef - asr_t2_undef
        dk_chk   = asr_t0_chk   - asr_t2_chk

        print(f"  Tier-0  ASR: undef={asr_t0_undef:.3f}  chk={asr_t0_chk:.3f}")
        print(f"  Tier-2p ASR: undef={asr_t2_undef:.3f}  chk={asr_t2_chk:.3f}  "
              f"(Δk_undef={dk_undef:.3f}, Δk_chk={dk_chk:.3f})")

        # ── PQC co-location impact ────────────────────────────────────────
        pqc = measure_pqc_impact(kyber_key, reachable_all_page)
        print(f"  PQC reachable: {pqc['n_reachable_pqc_bytes']}/{KYBER_KEY_BYTES} B  "
              f"(FRR_PQC={pqc['frr_pqc']:.4f})  "
              f"avg_coefs_changed/flip={pqc['avg_coefs_changed_per_flip']:.2f}\n")

        results.append({
            "seed":               seed,
            "hammered_rows":      hammered_rows,
            "frr_dnn":            frr_dnn,
            "n_reachable_dnn":    len(reachable_dnn_page),
            "asr_t0_undefended":  asr_t0_undef,
            "asr_t0_checksum":    asr_t0_chk,
            "asr_t2_undefended":  asr_t2_undef,
            "asr_t2_checksum":    asr_t2_chk,
            "delta_k_undefended": dk_undef,
            "delta_k_checksum":   dk_chk,
            **{f"pqc_{k}": v for k, v in pqc.items()},
        })

    # ── Aggregate ────────────────────────────────────────────────────────────
    keys = ["frr_dnn", "asr_t0_undefended", "asr_t0_checksum",
            "asr_t2_undefended", "asr_t2_checksum",
            "delta_k_undefended", "delta_k_checksum",
            "pqc_frr_pqc", "pqc_n_reachable_pqc_bytes",
            "pqc_avg_coefs_changed_per_flip"]

    def ms(k):
        v = [r[k] for r in results]
        return float(np.mean(v)), float(np.std(v))

    print("=== AGGREGATE RESULTS (n=10 seeds) ===")
    for k in keys:
        m, s = ms(k)
        print(f"  {k:42s}: {m:.4f} ± {s:.4f}")

    agg = {k: {"mean": ms(k)[0], "std": ms(k)[1]} for k in keys}
    summary = {
        "config": {
            "row_size_bytes":    ROW_SIZE,
            "page_size_bytes":   PAGE_SIZE,
            "page_start_bytes":  PAGE_START_BYTES,
            "pqc_offset_bytes":  PQC_OFFSET_BYTES,
            "kyber_key_bytes":   KYBER_KEY_BYTES,
            "n_hammered_rows":   HAMMERED_ROWS,
            "max_flips":         MAX_FLIPS,
            "n_seeds":           N_SEEDS,
        },
        "per_seed":  results,
        "aggregate": agg,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
