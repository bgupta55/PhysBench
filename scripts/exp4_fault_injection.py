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
"""
import csv, json, hashlib, datetime, subprocess, argparse, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── ML-KEM-512 dk layout constants ────────────────────────────────────────────
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


def run_trial(trial_id: int):
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
    from kyber_py.ml_kem import ML_KEM_512, ml_kem as _mlk
    import numpy as _np

    rng = _np.random.default_rng(seed=trial_id)

    # Fresh key generation + encapsulation
    ek, dk = ML_KEM_512.keygen()
    K_correct, ct = ML_KEM_512.encaps(ek)

    # Choose a byte/bit to flip, weighted proportionally to region size
    # (packed_s: 768/1632, pk: 800/1632, H_pk: 32/1632, z: 32/1632)
    regions = ["packed_s", "pk", "H_pk", "z"]
    weights = [768/1632, 800/1632, 32/1632, 32/1632]
    region = rng.choice(regions, p=weights)

    if region == "packed_s":
        byte_idx = int(rng.integers(PACKED_S_START, PACKED_S_END))
    elif region == "pk":
        byte_idx = int(rng.integers(PK_START, PK_END))
    elif region == "H_pk":
        byte_idx = int(rng.integers(H_PK_START, H_PK_END))
    else:  # z
        byte_idx = int(rng.integers(Z_START, Z_END))

    bit_idx = int(rng.integers(0, 8))

    # Inject fault
    dk_faulty = bytearray(dk)
    dk_faulty[byte_idx] ^= (1 << bit_idx)
    dk_faulty = bytes(dk_faulty)

    # Derive the implicit-rejection reference output using original dk + corrupt ct
    # Per FIPS 203: when FO check fails, K = PRF(z, ct); we compute this by
    # decapsulating with a deliberately invalid ciphertext against the CORRECT dk
    ct_bad = _force_reject_ciphertext(ct)
    K_reject_ref = ML_KEM_512.decaps(dk, ct_bad)

    # Run decapsulation with faulty key.
    # kyber-py 1.2.0 enforces FIPS 203 input-validation (hash check of H_pk in dk)
    # before any cryptographic processing.  A flip in H_pk or pk will trigger
    # this check and raise ValueError — this is classified as "validation_abort"
    # (the implementation self-detects the corruption and refuses to process).
    try:
        K_faulty = ML_KEM_512.decaps(dk_faulty, ct)
        # Classify outcome
        if K_faulty == K_correct:
            outcome = "no_effect"
        elif K_faulty == K_reject_ref:
            outcome = "fo_implicit_reject"
        else:
            outcome = "silent_wrong_output"
    except ValueError:
        # FIPS 203 hash-check aborted decapsulation — corruption was detected
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


def main(n_trials: int = 10_000, workers: int = None):
    print(f"[Exp4] Running {n_trials:,} fault-injection trials …")
    all_results = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_trial, i): i for i in range(n_trials)}
        completed = 0
        for fut in as_completed(futures):
            all_results.append(fut.result())
            completed += 1
            if completed % max(1, n_trials // 20) == 0:
                print(f"  … {completed}/{n_trials} ({100*completed//n_trials}%)", flush=True)

    all_results.sort(key=lambda r: r["trial"])

    # Write raw CSV
    csv_out = RESULTS_DIR / "decap_fault_outcomes.csv"
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

    table_csv = RESULTS_DIR / "decap_fault_outcomes_table.csv"
    pd.DataFrame(table_rows).to_csv(table_csv, index=False)

    # Console summary
    print("\n[Exp4] Per-region outcome breakdown:")
    print(df.groupby(["region", "outcome"]).size().unstack(fill_value=0).to_string())

    # JSON summary
    import importlib.metadata as _meta
    kyber_ver = _meta.version("kyber-py")

    summary = {
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
    json.dump(summary, open(RESULTS_DIR / "exp4_summary.json", "w"), indent=2)

    print(f"\n  Raw CSV   → {csv_out}")
    print(f"  Table CSV → {table_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int, default=10_000,
                        help="Number of fault-injection trials (default: 10000)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPU cores)")
    args = parser.parse_args()
    main(n_trials=args.trials, workers=args.workers)
