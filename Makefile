# Makefile for JSA SI Remediation Experiments
# Run from the experiments/ directory.
# Usage:
#   make all               — run all experiments end-to-end
#   make kat               — Exps 1+2 (KAT + byte-coef map), ~5 min
#   make geometry          — Exp 3 (DRAM Monte Carlo), ~5-10 min
#   make faultinject       — Exp 4 (decap fault injection, 10k trials), ~30-60 min
#   make lattice           — Exp 5 (lattice hardness), minutes–hours
#   make studyA            — Exp 6a-6f (Study A repairs), hours
#   make stats             — Exps 6e, 6f, 7 (statistics + provenance), <15 min
#   make smoke             — quick 100-trial smoke test of Exp 4
#   make clean             — remove results/ and logs/

PYTHON := python3
SCRIPTS := scripts

.PHONY: all kat geometry faultinject lattice studyA stats smoke clean

all: kat geometry faultinject lattice studyA stats
	@echo "=== All experiments complete ==="

kat:
	@mkdir -p logs results
	@echo "[make] Experiment 1 — KAT / key layout"
	$(PYTHON) $(SCRIPTS)/exp1_layout.py 2>&1 | tee logs/exp1_layout.log
	@echo "[make] Experiment 2 — byte↔coef map"
	$(PYTHON) $(SCRIPTS)/exp2_byte_coef_map.py 2>&1 | tee logs/exp2_byte_coef_map.log

geometry:
	@mkdir -p logs results
	@echo "[make] Experiment 3 — DRAM geometry Monte Carlo"
	$(PYTHON) $(SCRIPTS)/exp3_geometry_sweep.py 2>&1 | tee logs/exp3_geometry_sweep.log

smoke:
	@mkdir -p logs results
	@echo "[make] Smoke test — Exp 4 with N=100 trials"
	$(PYTHON) $(SCRIPTS)/exp4_fault_injection.py --trials 100 2>&1 | tee logs/exp4_smoke.log

faultinject:
	@mkdir -p logs results
	@echo "[make] Experiment 4 — fault injection (N=10000)"
	$(PYTHON) $(SCRIPTS)/exp4_fault_injection.py --trials 10000 2>&1 | tee logs/exp4_fault_injection.log

lattice:
	@mkdir -p logs results
	@echo "[make] Experiment 5 — lattice hardness estimate"
	$(PYTHON) $(SCRIPTS)/exp5_lattice_hardness.py 2>&1 | tee logs/exp5_lattice_hardness.log

studyA:
	@mkdir -p logs results
	@echo "[make] Experiment 6a–6f — Study A repairs"
	$(PYTHON) $(SCRIPTS)/run_study_a_driver.py 2>&1 | tee logs/study_a_driver.log

stats:
	@mkdir -p logs results
	@echo "[make] Experiments 6e, 6f — stats + correction"
	$(PYTHON) $(SCRIPTS)/exp6e_correction.py   2>&1 | tee logs/exp6e_correction.log
	$(PYTHON) $(SCRIPTS)/exp6f_bootstrap.py    2>&1 | tee logs/exp6f_bootstrap.log
	@echo "[make] Experiment 7 — data-integrity / provenance"
	$(PYTHON) $(SCRIPTS)/exp7_provenance_check.py 2>&1 | tee logs/exp7_provenance.log

clean:
	rm -rf results/* logs/*
	@echo "[make] Cleaned results/ and logs/"
