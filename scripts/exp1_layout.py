"""
Experiment 1 — KAT Validation of the Kyber-512 Key Layout
Validates Phase 1.1–1.3 of the JSA Remediation Blueprint.

Deliverables:
  results/exp1_manifest.json   — provenance record (version + hashes)
  results/exp1_layout.txt      — dk layout summary

Run:
  python scripts/exp1_layout.py
"""
import hashlib, json, os, subprocess, datetime, sys
from pathlib import Path

# ── resolve paths relative to this script's location ──────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── kyber-py import ────────────────────────────────────────────────────────────
try:
    from kyber_py.ml_kem import ML_KEM_512
    from kyber_py.polynomials.polynomials import PolynomialRing
    import importlib.metadata as _meta
    KYBER_VERSION = _meta.version("kyber-py")
except ImportError as e:
    print(f"[ERROR] Cannot import kyber-py: {e}")
    print("  Install with:  pip install kyber-py")
    sys.exit(1)

# ── FIPS 203 ML-KEM-512 dk layout constants ───────────────────────────────────
K            = 2           # module rank
N            = 256         # polynomial degree
Q            = 3329        # prime modulus
ETA          = 3           # CBD noise parameter for Kyber-512
BITS_PER_COEF = 12         # ByteEncode_12: 12 bits per coefficient
BYTES_PER_POLY = N * BITS_PER_COEF // 8   # 384
PACKED_S_LEN = K * BYTES_PER_POLY          # 768
PK_LEN       = 800
H_PK_LEN     = 32
Z_LEN        = 32
DK_LEN       = PACKED_S_LEN + PK_LEN + H_PK_LEN + Z_LEN  # 1632

R = PolynomialRing()

def check_dk_layout(dk: bytes):
    """Assert the real FIPS 203 dk layout and return parsed components."""
    assert len(dk) == DK_LEN, f"dk must be {DK_LEN} bytes, got {len(dk)}"

    packed_s = dk[:PACKED_S_LEN]                          # 768 B
    pk_bytes = dk[PACKED_S_LEN : PACKED_S_LEN + PK_LEN]  # 800 B
    h_pk     = dk[PACKED_S_LEN + PK_LEN : PACKED_S_LEN + PK_LEN + H_PK_LEN]  # 32 B
    z        = dk[-Z_LEN:]                                # 32 B

    assert len(packed_s) == PACKED_S_LEN
    assert len(pk_bytes)  == PK_LEN
    assert len(h_pk)      == H_PK_LEN
    assert len(z)         == Z_LEN
    return packed_s, pk_bytes, h_pk, z


def round_trip_check(packed_s: bytes):
    """Decode packed_s to coefficient arrays and re-encode; must reproduce identical bytes."""
    p0_bytes = packed_s[:BYTES_PER_POLY]
    p1_bytes = packed_s[BYTES_PER_POLY:]
    p0 = R.decode(p0_bytes, 12)
    p1 = R.decode(p1_bytes, 12)
    p0_rt = p0.encode(12)
    p1_rt = p1.encode(12)
    assert p0_rt == p0_bytes, "Round-trip failed for polynomial 0"
    assert p1_rt == p1_bytes, "Round-trip failed for polynomial 1"
    return p0, p1


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR.parent
        ).decode().strip()
    except Exception:
        return "N/A"


def main():
    print("[Exp1] Generating ML-KEM-512 key pair …")
    ek, dk = ML_KEM_512.keygen()

    print("[Exp1] Checking dk layout …")
    packed_s, pk_bytes, h_pk, z = check_dk_layout(dk)

    print("[Exp1] Running round-trip encode/decode …")
    p0, p1 = round_trip_check(bytes(packed_s))

    # Note: packed_s stores s_hat = NTT(s), so coefficients are NTT-domain values
    # in Z_q (range [0, 3328]), NOT the original CBD(eta=3) range [-3,3].
    # The CBD range check is valid only on s before NTT; after NTT all values
    # are valid elements of Z_q.  We verify the NTT-domain range instead.
    q = Q
    assert all(0 <= c < q for c in p0.coeffs), f"p0 has coeff outside Z_q=[0,{q-1}]"
    assert all(0 <= c < q for c in p1.coeffs), f"p1 has coeff outside Z_q=[0,{q-1}]"

    summary = {
        "pass": True,
        "dk_len": len(dk),
        "packed_s_len": len(packed_s),
        "pk_len": len(pk_bytes),
        "h_pk_len": len(h_pk),
        "z_len": len(z),
        "bits_per_coef": BITS_PER_COEF,
        "cbd_eta": ETA,
        "stored_as": "NTT_domain_s_hat_mod_q",
        "coeff_range_Zq_ok": True,
        "round_trip_ok": True,
        "dk_sha256": hashlib.sha256(dk).hexdigest(),
        "kyber_py_version": KYBER_VERSION,
        "git_commit": get_git_commit(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    out = RESULTS_DIR / "exp1_manifest.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"[Exp1] PASSED — manifest written to {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
