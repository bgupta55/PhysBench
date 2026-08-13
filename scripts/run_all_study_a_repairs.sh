#!/usr/bin/env bash
# run_all_study_a_repairs.sh
# Sequentially launches all Study A repair experiments (Exp 6a–6f).
# Usage:  caffeinate -i bash scripts/run_all_study_a_repairs.sh
#    or:  bash scripts/run_all_study_a_repairs.sh > logs/study_a_repairs.log 2>&1 &

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."    # cd to experiments/

echo "=== Study A Repairs Driver ==="
echo "Working directory: $(pwd)"
date

python3 scripts/run_study_a_driver.py
echo "=== Driver complete ==="
date
