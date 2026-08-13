"""
Experiment 7 — Data-Integrity / Provenance Verification
Validates Phase 4.1 (R2-Data.1–R2-Data.4) of the JSA Remediation Blueprint.

Hashes every result CSV, checks for duplicate values between Table 3 / Table 4
re-runs, and produces a provenance manifest that can be cited in the paper.

Deliverables:
  results/provenance_manifest.json           — SHA-256 of every results CSV
  results/table3_vs_table4_comparison.csv    — duplicate-value check
  results/exp7_summary.json

Run:
  python scripts/exp7_provenance_check.py
"""
import hashlib, json, datetime, subprocess, sys, glob
from pathlib import Path

import pandas as pd
import numpy as np

SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def file_sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def count_rows(path: str) -> int:
    try:
        return sum(1 for _ in open(path)) - 1
    except Exception:
        return -1


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR.parent
        ).decode().strip()
    except Exception:
        return "N/A"


def main():
    # ── Hash every results CSV ─────────────────────────────────────────────────
    manifest = []
    csv_paths = sorted(glob.glob(str(RESULTS_DIR / "*.csv")))
    json_paths = sorted(glob.glob(str(RESULTS_DIR / "*.json")))
    png_paths  = sorted(glob.glob(str(RESULTS_DIR / "*.png")))

    for path in csv_paths + json_paths + png_paths:
        name = Path(path).name
        manifest.append({
            "file":     name,
            "sha256":   file_sha256(path),
            "n_rows":   count_rows(path) if path.endswith(".csv") else None,
            "size_kb":  round(Path(path).stat().st_size / 1024, 2),
        })

    manifest_out = RESULTS_DIR / "provenance_manifest.json"
    json.dump({
        "files":      manifest,
        "git_commit": get_git_commit(),
        "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
    }, open(manifest_out, "w"), indent=2)

    print(f"[Exp7] Provenance manifest: {len(manifest)} files hashed")

    # ── Table 3 vs Table 4 duplicate check ────────────────────────────────────
    t3_csvs = sorted(glob.glob(str(RESULTS_DIR / "pbs_sweep_b20_pbs_*.csv")))
    t4_csvs = sorted(glob.glob(str(RESULTS_DIR / "pbs_sweep_b15_pbs_*.csv")))

    if not t3_csvs or not t4_csvs:
        print("\n[Exp7] WARNING: Table 3 (b20) or Table 4 (b15) sweep CSV not found.")
        print("       Run exp6a_pbs_sweep.py with --budget 20 and --budget 15 first.")
        print("       Creating placeholder comparison record.")

        comparison = pd.DataFrame([{
            "seed": "N/A", "arch": "N/A", "defense": "N/A",
            "asr_20flip": "N/A", "asr_15flip": "N/A", "identical": "N/A",
            "note": "Table 3 or Table 4 CSV not yet generated — run Exp6a first",
        }])
    else:
        t3 = pd.read_csv(t3_csvs[-1])
        t4 = pd.read_csv(t4_csvs[-1])

        # Take final-step ASR for each (arch, seed, defense)
        t3_final = t3[t3["step"] == t3["step"].max()]
        t4_final = t4[t4["step"] == t4["step"].max()]

        merge_cols = ["arch", "seed", "defense"]
        comparison = t3_final[merge_cols + ["asr"]].merge(
            t4_final[merge_cols + ["asr"]], on=merge_cols, suffixes=("_20flip", "_15flip")
        )
        comparison["identical"] = (
            comparison["asr_20flip"].round(4) == comparison["asr_15flip"].round(4)
        )

        n_identical = int(comparison["identical"].sum())
        n_total     = len(comparison)
        print(f"\n[Exp7] Table 3 vs Table 4 identity check:")
        print(f"  {n_identical} / {n_total} (arch×seed×defense) pairs identical to 4 d.p.")

        if n_identical == n_total:
            print("  ⚠ ALL values are identical — this likely reflects early saturation.")
            print("    Per blueprint Phase 4.1: state this explicitly in the paper text,")
            print("    citing the per-step logs as evidence of genuine saturation.")
        elif n_identical > 0:
            print(f"  {n_identical} identical rows (likely saturated conditions).")
        else:
            print("  No identical values — Table 3 and Table 4 differ as expected.")

        print(comparison[["arch", "seed", "defense", "asr_20flip", "asr_15flip", "identical"]].to_string(index=False))

    comp_out = RESULTS_DIR / "table3_vs_table4_comparison.csv"
    comparison.to_csv(comp_out, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = {
        "n_files_hashed":         len(manifest),
        "provenance_manifest":    str(manifest_out.name),
        "comparison_csv":         str(comp_out.name),
        "comparison_sha256":      file_sha256(str(comp_out)),
        "git_commit":             get_git_commit(),
        "timestamp":              datetime.datetime.utcnow().isoformat() + "Z",
    }
    json.dump(summary, open(RESULTS_DIR / "exp7_summary.json", "w"), indent=2)

    print(f"\n[Exp7] All provenance records written.")
    print(f"  Manifest → {manifest_out}")
    print(f"  T3 vs T4 → {comp_out}")


if __name__ == "__main__":
    main()
