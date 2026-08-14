"""
Experiment 4 — Real Fault Injection into Decapsulation
Validates Phase 3.2 of the JSA Remediation Blueprint.

Purpose: Turn "byte is reachable" into an actual measured cryptographic outcome
by running real decapsulation on corrupted keys.  This is the centerpiece new
result of Study B.

Deliverables:
  results/decap_fault_outcomes.csv        — per-trial outcome records
  results/decap_fault_outcomes_table.csv  — per-region breakdown with 95% CIs
  results/exp4_summary.json               — aggregate statistics + provenance

Run (small test, N=100):
  python scripts/exp4_fault_injection.py --trials 100

Run (full, N=10000):
  python scripts/exp4_fault_injection.py --trials 10000

Run with other parameter sets (Experiment 3 cross-param generality):
  python scripts/exp4_fault_injection.py --scheme ML_KEM_768  --out results_v2/exp4_mlkem768.csv
  python scripts/exp4_fault_injection.py --scheme ML_KEM_1024 --out results_v2/exp4_mlkem1024.csv
"""
import csv, json, hashlib, datetime, subprocess, argparse, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── ML-KEM dk layout constants by parameter set ──────────────────────────────
# dk = packed_s || ek || H(ek) || z
# ML-KEM-512:  dk = 768 + 800 + 32 + 32 = 1632
# ML-KEM-768:  dk = 1152 + 1184 + 32 + 32 = 2400
# ML-KEM-1024: dk = 1536 + 1568 + 32 + 32 = 3168
_LAYOUT = {
    "ML_KEM_512":  {"packed_s": 768,  "pk": 800,  "H_pk": 32, "z": 32},
    "ML_KEM_768":  {"packed_s": 1152, "pk": 1184, "H_pk": 32, "z": 32},
    "ML_KEM_1024": {"packed_s": 1536, "pk": 1568, "H_pk": 32, "z": 32},
}

# defaults (overridden by --scheme)
PACKED_S_START = 0
PACKED_S_END   = 768
PK_START       = 768
PK_END         = 768 + 800
H_PK_START     = 768 + 800
H_PK_END       = 768 + 800 + 32
Z_START        = 1632 - 32
Z_END          = 1632
DK_LEN         = 1632


def _get_region(byte_idx: int) -> str:
    """Return which dk sub-field a byte index belongs to."""
    if PACKED_S_START <= byte_idx < PACKED_S_END:
        return "packed_s"
    elif PK_START <= byte_idx < PK_END:
        return "pk"
    elif H_PK_START <= byte_idx < H_PK_END:
        return "H_pk"
    else:
        return "z"


def _force_reject_ciphertext(ct: bytes) -> bytes:
    """
    Produce a deliberately invalid ciphertext guaranteed to trigger implicit
    rejection in FIPS 203 decapsulation.  We flip the last byte of ct — an
    invalid ciphertext causes decaps to return K = PRF(z, ct) instead of the
    real shared secret, which is the FO implicit-rejection output.
    """
    ct_bad = bytearray(ct)
    ct_bad[-1] ^= 0xFF
    return bytes(ct_bad)


def run_trial(trial_id: int, scheme_name: str = "ML_KEM_512"):
    """
    Single fault-injection trial.  Returns a dict with trial metadata and outcome.
    Runs in a subprocess worker — must import kyber_py inside the function.

    Implementation note (kyber-py 1.2.0):
      ML_KEM_512.decaps() performs a FIPS 203 hash check: H(ek) stored in dk[H_PK]
      must match the ek embedded in dk.  Corrupting H_pk or any field that shifts
      this check causes a ValueError before cryptographic processing.
      This is itself a meaningful fault outcome ("validation_fail") — it means
      the implementation would detect the corruption and abort, equivalent to
      a hard fault-detection response.  We classify this as "validation_abort"
      (not "fo_implicit_reject") since the library raises before the FO check runs.
    """
    from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
    import numpy as _np

    _SCHEME_MAP = {
        "ML_KEM_512":  ML_KEM_512,
        "ML_KEM_768":  ML_KEM_768,
        "ML_KEM_1024": ML_KEM_1024,
    }
    scheme = _SCHEME_MAP[scheme_name]
    layout = _LAYOUT[scheme_name]

    ps_end  = layout["packed_s"]
    pk_end  = ps_end + layout["pk"]
    hpk_end = pk_end + layout["H_pk"]
    z_end   = hpk_end + layout["z"]
    dk_len  = z_end

    rng = _np.random.default_rng(seed=trial_id)

    # Fresh key generation + encapsulation
    ek, dk = scheme.keygen()
    K_correct, ct = scheme.encaps(ek)

    # Choose a byte/bit to flip, weighted proportionally to region size
    regions = ["packed_s", "pk", "H_pk", "z"]
    weights = [layout["packed_s"]/dk_len, layout["pk"]/dk_len,
               layout["H_pk"]/dk_len,     layout["z"]/dk_len]
    region = rng.choice(regions, p=weights)

    if region == "packed_s":
        byte_idx = int(rng.integers(0, ps_end))
    elif region == "pk":
        byte_idx = int(rng.integers(ps_end, pk_end))
    elif region == "H_pk":
        byte_idx = int(rng.integers(pk_end, hpk_end))
    else:  # z
        byte_idx = int(rng.integers(hpk_end, z_end))

    bit_idx = int(rng.integers(0, 8))

    # Inject fault
    dk_faulty = bytearray(dk)
    dk_faulty[byte_idx] ^= (1 << bit_idx)
    dk_faulty = bytes(dk_faulty)

    # Derive the implicit-rejection reference output using original dk + corrupt ct
    # Per FIPS 203: when FO check fails, K = PRF(z, ct)
    ct_bad = _force_reject_ciphertext(ct)
    K_reject_ref = scheme.decaps(dk, ct_bad)

    # Run decapsulation with faulty key.
    # kyber-py 1.2.0 enforces FIPS 203 input-validation (hash check of H_pk in dk).
    try:
        K_faulty = scheme.decaps(dk_faulty, ct)
        if K_faulty == K_correct:
            outcome = "no_effect"
        elif K_faulty == K_reject_ref:
            outcome = "fo_implicit_reject"
        else:
            outcome = "silent_wrong_output"
    except ValueError:
        outcome = "validation_abort"

    return {
        "trial":     trial_id,
        "region":    region,
        "byte_idx":  byte_idx,
        "bit_idx":   bit_idx,
        "outcome":   outcome,
    }


def wilson_ci(successes: int, total: int, z: float = 1.96):
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half   = (z * (p * (1 - p) / total + z**2 / (4 * total**2))**0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR.parent
        ).decode().strip()
    except Exception:
        return "N/A"


def main(n_trials: int = 10_000, workers: int = None,
         scheme_name: str = "ML_KEM_512", out_csv: str = None):
    print(f"[Exp4] Running {n_trials:,} fault-injection trials (scheme={scheme_name}) …")
    all_results = []

    import functools
    trial_fn = functools.partial(run_trial, scheme_name=scheme_name)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(trial_fn, i): i for i in range(n_trials)}
        completed = 0
        for fut in as_completed(futures):
            all_results.append(fut.result())
            completed += 1
            if completed % max(1, n_trials // 20) == 0:
                print(f"  … {completed}/{n_trials} ({100*completed//n_trials}%)", flush=True)

    all_results.sort(key=lambda r: r["trial"])

    # Determine output paths
    if out_csv:
        csv_out   = Path(out_csv) if Path(out_csv).is_absolute() else SCRIPT_DIR.parent / out_csv
        table_csv = csv_out.parent / (csv_out.stem + "_table.csv")
        summary_path = csv_out.parent / (csv_out.stem + "_summary.json")
    else:
        csv_out      = RESULTS_DIR / "decap_fault_outcomes.csv"
        table_csv    = RESULTS_DIR / "decap_fault_outcomes_table.csv"
        summary_path = RESULTS_DIR / "exp4_summary.json"

    csv_out.parent.mkdir(parents=True, exist_ok=True)

    # Write raw CSV
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "region", "byte_idx", "bit_idx", "outcome"])
        w.writeheader()
        w.writerows(all_results)

    df = pd.DataFrame(all_results)

    # Per-region breakdown with Wilson 95% CIs
    outcomes   = ["no_effect", "fo_implicit_reject", "silent_wrong_output", "validation_abort"]
    table_rows = []
    for region, grp in df.groupby("region"):
        n = len(grp)
        row = {"region": region, "n_trials": n}
        for outcome in outcomes:
            cnt = (grp["outcome"] == outcome).sum()
            pct = cnt / n * 100
            ci  = wilson_ci(cnt, n)
            row[f"{outcome}_n"]    = int(cnt)
            row[f"{outcome}_pct"]  = round(pct, 2)
            row[f"{outcome}_ci_lo"] = round(ci[0]*100, 2)
            row[f"{outcome}_ci_hi"] = round(ci[1]*100, 2)
        table_rows.append(row)

    pd.DataFrame(table_rows).to_csv(table_csv, index=False)

    # Console summary
    print(f"\n[Exp4/{scheme_name}] Per-region outcome breakdown:")
    print(df.groupby(["region", "outcome"]).size().unstack(fill_value=0).to_string())

    # JSON summary
    import importlib.metadata as _meta
    kyber_ver = _meta.version("kyber-py")

    summary = {
        "scheme":            scheme_name,
        "n_trials":          n_trials,
        "kyber_py_version":  kyber_ver,
        "total_outcomes":    df["outcome"].value_counts().to_dict(),
        "per_region":        {r["region"]: {k:v for k,v in r.items() if k!="region"}
                              for r in table_rows},
        "csv_sha256":        hashlib.sha256(open(csv_out,    "rb").read()).hexdigest(),
        "table_sha256":      hashlib.sha256(open(table_csv,  "rb").read()).hexdigest(),
        "git_commit":        get_git_commit(),
        "timestamp":         datetime.datetime.utcnow().isoformat() + "Z",
    }
    json.dump(summary, open(summary_path, "w"), indent=2)

    print(f"\n  Raw CSV   → {csv_out}")
    print(f"  Table CSV → {table_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int, default=10_000,
                        help="Number of fault-injection trials (default: 10000)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPU cores)")
    parser.add_argument("--scheme",  type=str, default="ML_KEM_512",
                        choices=["ML_KEM_512", "ML_KEM_768", "ML_KEM_1024"],
                        help="ML-KEM parameter set (default: ML_KEM_512)")
    parser.add_argument("--out",     type=str, default=None,
                        help="Override output CSV path (default: results/decap_fault_outcomes.csv)")
    args = parser.parse_args()
    main(n_trials=args.trials, workers=args.workers,
         scheme_name=args.scheme, out_csv=args.out)
