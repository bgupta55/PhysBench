# JSA SI Remediation — Experiments README
# ========================================
#
# Folder structure:
#   experiments/
#     scripts/          — all Python experiment scripts
#     results/          — all output CSVs, JSONs, PNGs (git-tracked provenance)
#     logs/             — run logs (git-tracked for reproducibility)
#     lattice-estimator/ — clone of https://github.com/malb/lattice-estimator
#     Makefile          — one-command reproducibility driver
#     requirements.txt  — Python dependencies
#     README.md         — this file
#
# Quick start (after installing dependencies per requirements.txt):
#
#   cd experiments/
#   make smoke          # 2-minute smoke test of Exp 4 (N=100 trials)
#   make kat            # Exps 1+2, ~5 minutes
#   make geometry       # Exp 3, ~5-10 minutes
#   make faultinject    # Exp 4, ~30-60 minutes (full 10k trials)
#   make lattice        # Exp 5, minutes–hours depending on SageMath
#   make studyA         # Exps 6a-6f, hours (Study A repairs)
#   make stats          # Exps 6e, 6f, 7 — statistics + provenance
#   make all            # Everything above, in order
#
# For long-running jobs, wrap with caffeinate:
#   caffeinate -i make all > logs/master.log 2>&1 &
#
# Each experiment writes:
#   results/<exp_name>.csv      — raw data (citable with SHA-256)
#   results/<exp_name>.json     — summary statistics + provenance
#   logs/<exp_name>.log         — run log for reproducibility audit
