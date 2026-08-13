"""
Experiment 2 — Byte↔Coefficient Mapping Analysis
Validates Phase 1.2 of the JSA Remediation Blueprint.

Purpose: Replace the false "1 reachable byte = 1 corrupted coefficient"
assumption with a measured mapping for 12-bit-packed ML-KEM-512 secret key.

Deliverables:
  results/byte_coef_map.csv      — per-byte coefficient-touch counts
  results/exp2_summary.json      — aggregate statistics

Run:
  python scripts/exp2_byte_coef_map.py
"""
import csv, json, hashlib, datetime, subprocess, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

try:
    from kyber_py.ml_kem import ML_KEM_512
    from kyber_py.polynomials.polynomials import PolynomialRing
    import importlib.metadata as _meta
    KYBER_VERSION = _meta.version("kyber-py")
except ImportError as e:
    print(f"[ERROR] {e}"); sys.exit(1)

K_RANK       = 2
N            = 256
BYTES_PER_POLY = 384   # 256 * 12 / 8
PACKED_S_LEN = 768     # 2 * 384

R = PolynomialRing()


def decode_s(packed_s_bytes: bytes):
    """Decode the 768-byte packed secret key into a flat list of 512 coefficients."""
    p0 = R.decode(bytes(packed_s_bytes[:BYTES_PER_POLY]), 12)
    p1 = R.decode(bytes(packed_s_bytes[BYTES_PER_POLY:]), 12)
    return p0.coeffs + p1.coeffs  # 512 coefficients total


def analyze_byte_coef_map(packed_s_bytes: bytes):
    """
    For every byte in packed_s, flip each of its 8 bits one at a time and record
    which coefficient indices change. Returns list of per-byte records.
    """
    results = []
    baseline = decode_s(packed_s_bytes)

    total_bytes = len(packed_s_bytes)
    for byte_idx in range(total_bytes):
        touched_coeffs = set()
        for bit in range(8):
            mutated = bytearray(packed_s_bytes)
            mutated[byte_idx] ^= (1 << bit)
            flipped = decode_s(bytes(mutated))
            for i, (a, b) in enumerate(zip(baseline, flipped)):
                if a != b:
                    touched_coeffs.add(i)

        results.append({
            "byte_idx": byte_idx,
            "poly_idx": 0 if byte_idx < BYTES_PER_POLY else 1,
            "n_coeffs_touched": len(touched_coeffs),
            "coeff_indices": sorted(touched_coeffs),
        })

        if (byte_idx + 1) % 100 == 0:
            print(f"  … processed {byte_idx+1}/{total_bytes} bytes", flush=True)

    return results


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR.parent
        ).decode().strip()
    except Exception:
        return "N/A"


def main():
    print("[Exp2] Generating ML-KEM-512 key pair …")
    ek, dk = ML_KEM_512.keygen()
    packed_s = bytes(dk[:PACKED_S_LEN])

    print(f"[Exp2] Analyzing byte→coefficient mapping for {PACKED_S_LEN} bytes …")
    results = analyze_byte_coef_map(packed_s)

    # Write CSV
    csv_out = RESULTS_DIR / "byte_coef_map.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["byte_idx", "poly_idx", "n_coeffs_touched", "coeff_indices"])
        w.writeheader()
        for r in results:
            w.writerow({**r, "coeff_indices": str(r["coeff_indices"])})

    # Aggregate statistics
    counts = [r["n_coeffs_touched"] for r in results]
    pct_single = sum(1 for c in counts if c == 1) / len(counts) * 100
    pct_double = sum(1 for c in counts if c == 2) / len(counts) * 100
    pct_zero   = sum(1 for c in counts if c == 0) / len(counts) * 100

    # Per-polynomial breakdown
    poly0 = [r for r in results if r["poly_idx"] == 0]
    poly1 = [r for r in results if r["poly_idx"] == 1]
    pct_single_p0 = sum(1 for r in poly0 if r["n_coeffs_touched"] == 1) / len(poly0) * 100
    pct_double_p0 = sum(1 for r in poly0 if r["n_coeffs_touched"] == 2) / len(poly0) * 100
    pct_single_p1 = sum(1 for r in poly1 if r["n_coeffs_touched"] == 1) / len(poly1) * 100
    pct_double_p1 = sum(1 for r in poly1 if r["n_coeffs_touched"] == 2) / len(poly1) * 100

    # Theoretical prediction: in a pure 12-bit packing of 256 coefs into 384 bytes,
    # every 3rd byte straddles two coefs (bytes at positions 1, 4, 7, … within each group of 3)
    # ≈ 1/3 of bytes → 2 coefficients;  2/3 → 1 coefficient; 0 → zero bytes.
    theoretical_single  = 2/3 * 100
    theoretical_double  = 1/3 * 100

    summary = {
        "total_bytes_analyzed": PACKED_S_LEN,
        "pct_bytes_touching_0_coeffs": round(pct_zero, 2),
        "pct_bytes_touching_1_coeff": round(pct_single, 2),
        "pct_bytes_touching_2_coeffs": round(pct_double, 2),
        "theoretical_pct_single": round(theoretical_single, 2),
        "theoretical_pct_double": round(theoretical_double, 2),
        "per_poly": {
            "poly0_pct_single": round(pct_single_p0, 2),
            "poly0_pct_double": round(pct_double_p0, 2),
            "poly1_pct_single": round(pct_single_p1, 2),
            "poly1_pct_double": round(pct_double_p1, 2),
        },
        "csv_sha256": hashlib.sha256(open(csv_out, "rb").read()).hexdigest(),
        "kyber_py_version": KYBER_VERSION,
        "git_commit": get_git_commit(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    json.dump(summary, open(RESULTS_DIR / "exp2_summary.json", "w"), indent=2)

    print(f"\n[Exp2] RESULTS:")
    print(f"  {pct_zero:.1f}% of bytes touch 0 coefficients")
    print(f"  {pct_single:.1f}% of bytes touch exactly 1 coefficient  (theory: {theoretical_single:.1f}%)")
    print(f"  {pct_double:.1f}% of bytes touch exactly 2 coefficients (theory: {theoretical_double:.1f}%)")
    print(f"\n  → This replaces old Table 9 'Coefs at risk per PQC flip: 1.0' row.")
    print(f"\n  Results written to {csv_out}")


if __name__ == "__main__":
    main()
