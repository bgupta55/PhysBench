"""
run_all_study_a_repairs.sh driver — Python wrapper that sequentially
launches all Study A repair experiments (Exp 6a–6f) and logs output.

Usage:
  python scripts/run_study_a_driver.py

Or via shell script run_all_study_a_repairs.sh (which calls this).
"""
import subprocess, sys, os, datetime
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
LOGS_DIR    = SCRIPT_DIR.parent / "logs"
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

python = sys.executable

JOBS = [
    # 6a — Table 3 re-run (20-flip, both architectures, PBS)
    [python, str(SCRIPT_DIR / "exp6a_pbs_sweep.py"),
     "--budget", "20", "--seeds", "10", "--arch", "both", "--attack", "pbs", "--defense", "all"],

    # 6a — Table 4 re-run (15-flip, SmallCNN, PBS)
    [python, str(SCRIPT_DIR / "exp6a_pbs_sweep.py"),
     "--budget", "15", "--seeds", "10", "--arch", "smallcnn", "--attack", "pbs", "--defense", "all"],

    # 6b — Random-flip negative control
    [python, str(SCRIPT_DIR / "exp6a_pbs_sweep.py"),
     "--budget", "20", "--seeds", "10", "--arch", "both", "--attack", "random", "--defense", "all"],

    # 6c — Clip sigma sweep: smallcnn
    *[
        [python, str(SCRIPT_DIR / "exp6a_pbs_sweep.py"),
         "--budget", "20", "--seeds", "10", "--arch", "smallcnn",
         "--attack", "pbs", "--defense", "clip", "--clip-sigma", str(sigma)]
        for sigma in ["3.0", "2.0", "1.0", "0.5", "0.25"]
    ],

    # 6c — Clip sigma sweep: resnet20
    *[
        [python, str(SCRIPT_DIR / "exp6a_pbs_sweep.py"),
         "--budget", "20", "--seeds", "10", "--arch", "resnet20",
         "--attack", "pbs", "--defense", "clip", "--clip-sigma", str(sigma)]
        for sigma in ["3.0", "1.0", "0.25"]   # reduced: 3 thresholds as per blueprint note
    ],

    # 6d — Attack hyperparameter sweep (SmallCNN)
    *[
        [python, str(SCRIPT_DIR / "exp6a_pbs_sweep.py"),
         "--budget", "20", "--seeds", "10", "--arch", "smallcnn",
         "--attack", "pbs", "--defense", "undefended",
         "--top-k", str(k), "--sub-batch", str(b)]
        for k in [4, 8, 16] for b in [16, 32, 64]
    ],

    # 6e — Multiple-comparisons correction
    [python, str(SCRIPT_DIR / "exp6e_correction.py")],

    # 6f — Bootstrap CIs
    [python, str(SCRIPT_DIR / "exp6f_bootstrap.py")],
]


def run_job(cmd, label, log_path):
    print(f"\n{'='*60}")
    print(f"[Driver] Running: {label}")
    print(f"[Driver] Log:     {log_path}")
    print(f"{'='*60}")
    with open(log_path, "w") as lf:
        lf.write(f"# CMD: {' '.join(cmd)}\n")
        lf.write(f"# START: {datetime.datetime.utcnow().isoformat()}Z\n\n")
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(SCRIPT_DIR.parent)
        )
        lf.write(result.stdout)
        lf.write(f"\n# END: {datetime.datetime.utcnow().isoformat()}Z\n")
        lf.write(f"# RETURNCODE: {result.returncode}\n")
    # Also print to console
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"[Driver] WARNING: job exited with code {result.returncode}")
    return result.returncode


def main():
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    print(f"[Driver] Starting Study A repair sweep at {ts}")
    print(f"[Driver] Total jobs: {len(JOBS)}")

    failed = []
    for i, cmd in enumerate(JOBS):
        label    = " ".join(cmd[-6:])  # last few args as label
        log_path = LOGS_DIR / f"studyA_job{i+1:02d}_{ts}.log"
        rc       = run_job(cmd, label, log_path)
        if rc != 0:
            failed.append((i + 1, label))

    print(f"\n[Driver] Done. {len(JOBS) - len(failed)}/{len(JOBS)} jobs succeeded.")
    if failed:
        print(f"[Driver] Failed jobs: {failed}")
    else:
        print("[Driver] All jobs passed.")


if __name__ == "__main__":
    main()
