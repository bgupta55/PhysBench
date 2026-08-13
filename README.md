# PhysBench
PhysBench
# PhysBench — Tier-0 Evaluation Harness for Bit-Flip Attack Defenses

Open code and checkpoints for the paper:

> **Physically Grounded Evaluation of Bit-Flip Attack Defenses: Bridging Simulated and Hardware-Constrained Fault Models**  
> Bhanwar Gupta, Sanjeev Rana  

This repository provides the complete Tier-0 baseline: independently implemented Progressive Bit Search (PBS) attack, two defense mechanism classes, n=10 reproducible seeds, and all raw results used in the paper.

---

## What is in this repo

```
physbench/
├── src/
│   ├── model.py                   # SmallCNN (~75K params) and ResNet-20 (~270K params)
│   ├── data_utils.py              # CIFAR-10 loader, fixed subsample seeding
│   ├── quant.py                   # int8 quantization, dequantization, bit-flip primitive
│   ├── attack_defense.py          # PBS attack + checksum and clip defenses
│   ├── train_baseline.py          # Train SmallCNN, seeds 0–9
│   ├── train_resnet20.py          # Train ResNet-20, seeds 0–9
│   ├── run_seed.py                # Attack/defense grid — SmallCNN, one seed
│   ├── run_resnet20_seed.py       # Attack/defense grid — ResNet-20, one seed
│   ├── run_adaptive.py            # Exp A: adaptive attacker (white-box mask)
│   ├── run_sweep.py               # Exp B: hyperparameter sweep (coverage, clip threshold)
│   ├── run_targeted.py            # Exp D: targeted class-pair attack
│   ├── run_partial_knowledge.py   # Exp B2: partial mask knowledge sweep
│   ├── run_pqc_coloc.py           # Exp C: PQC co-location geometry analysis
│   ├── run_grover_toy.py          # Toy 3-qubit Grover illustration (Qiskit-Aer)
│   ├── run_vqc.py                 # Exp F: bit-flip attack on VQC parameters (future work)
│   ├── aggregate.py               # Aggregate SmallCNN per-seed JSONs → summary.json
│   ├── aggregate_resnet20.py      # Aggregate ResNet-20 per-seed JSONs → resnet20_summary.json
│   └── make_figures.py            # Regenerate all paper figures from results/
├── ckpt/
│   ├── model_seed{0..9}.pt        # SmallCNN checkpoints, seeds 0–9
│   └── resnet20_seed{0..9}.pt     # ResNet-20 checkpoints, seeds 0–9
├── results/
│   ├── summary.json               # Aggregated SmallCNN results (n=10)
│   ├── seed{0..9}.json            # Per-seed SmallCNN attack trajectories
│   ├── sweep_seed{0..9}.json      # Per-seed hyperparameter sweep results
│   ├── adaptive/                  # Experiment A results
│   ├── resnet20/                  # ResNet-20 results (per-seed + summary)
│   ├── targeted/                  # Experiment D results
│   ├── pqc_coloc_results.json     # Experiment C: PQC geometry analysis
│   └── grover_toy_results.json    # Toy Grover circuit output
└── run_all_experiments.py         # One-command full replication
```

---

## Key results (from paper, n=10 seeds each)

| Architecture | Condition | ASR @ 20 flips | vs. undefended |
|---|---|---|---|
| SmallCNN | Undefended | 64.1% ± 1.8% | — |
| SmallCNN | Checksum (50% coverage) | 31.8% ± 2.1% | −32.3 pp, p=0.002, δ=1.0 |
| SmallCNN | Range-clip (3σ) | 63.7% ± 1.3% | null (p=0.188) |
| ResNet-20 | Undefended | 93.4% ± 2.1% | — |
| ResNet-20 | Checksum (50% coverage) | 11.6% ± 1.2% | **−81.8 pp**, p=0.002, δ=1.0 |
| ResNet-20 | Range-clip (3σ) | 93.4% ± 2.1% | null |

**Adaptive attacker** (Exp A): white-box mask knowledge recovers 97.8% of the checksum's protection (63.4% vs. undefended 64.1%).

---

## Setup

Requires Python 3.10+ and PyTorch. A virtual environment with pinned versions is included.

```bash
cd experiments/physbench

# Activate the included venv (Python 3.14, PyTorch 2.13, Qiskit 2.5)
source .venv/bin/activate

# Or create your own:
pip install torch torchvision scipy matplotlib qiskit qiskit-aer
```

Runs on CPU, Apple MPS, or CUDA — device is auto-detected in every script.

---

## Reproduce everything

```bash
cd experiments/physbench
python run_all_experiments.py
```

This runs in order: SmallCNN training → attack/defense grid → adaptive → sweep → targeted → ResNet-20 training → ResNet-20 grid → aggregation → figures.

**Skip the slow ResNet-20 training** (~14 min/seed × 10 seeds) and use the released checkpoints:

```bash
python run_all_experiments.py --skip_resnet20
```

Expected total runtime (Apple M4 Pro MPS, all seeds, with ResNet-20): ~3 hours.

---

## Reproduce individual experiments

```bash
# SmallCNN: train one seed
python src/train_baseline.py --seeds 0

# SmallCNN: attack/defense grid, one seed
python src/run_seed.py 0

# ResNet-20: train one seed (with standard CIFAR-10 augmentation)
python src/train_resnet20.py --seeds 0 --epochs 100

# ResNet-20: attack/defense grid, one seed
python src/run_resnet20_seed.py 0

# Exp A: adaptive attacker
python src/run_adaptive.py

# Exp B: hyperparameter sweep (15-flip budget)
python src/run_sweep.py --seeds 0 1 2 3 4 5 6 7 8 9

# Exp C: PQC co-location geometry analysis
python src/run_pqc_coloc.py

# Exp D: targeted class-pair attack
python src/run_targeted.py

# Aggregate SmallCNN seeds → summary.json
python src/aggregate.py --seeds 0 1 2 3 4 5 6 7 8 9

# Aggregate ResNet-20 seeds → resnet20/resnet20_summary.json
python src/aggregate_resnet20.py

# Regenerate all figures
python src/make_figures.py
```

---

## Experiment descriptions

| Script | Paper section | What it does |
|---|---|---|
| `run_seed.py` | §4–5 (F1, F3) | PBS + checksum + clip on SmallCNN, one seed; outputs per-seed JSON |
| `run_resnet20_seed.py` | §4–5 (F1, F3) | Same grid on ResNet-20; requires `resnet20_seed{N}.pt` in `ckpt/` |
| `run_adaptive.py` | §5.2 (F2) | Adaptive attacker with white-box mask knowledge; all 10 seeds |
| `run_sweep.py` | §5, Tab. 2 | Checksum coverage (25/50/75%) and clip threshold (1σ/2σ/3σ) sweep |
| `run_targeted.py` | §5.4 (F4) | Targeted misclassification: airplane→ship, cat→dog, truck→automobile |
| `run_pqc_coloc.py` | §6.1, Tab. 6 | Memory geometry analysis: FRR_DNN, FRR_PQC, Δk lower bounds |
| `run_grover_toy.py` | §6.4 | 3-qubit Grover toy circuit on qiskit-aer statevector simulator |
| `run_vqc.py` | (future work) | PBS on VQC rotation-angle parameters; infrastructure only |

---

## Metrics

**Protection** `Prot(D, A, M, s)` — ASR reduction from defense at Tier 0 (simulated oracle), holding the tier fixed:

```
Prot = ASR(undefended; A, M, s) − ASR(D; A, M, s)
```

**Tier-transfer gap** `Δk(D, A, M, s)` — ASR drop when attacker is constrained to physical flip set at Tier k, holding the defense fixed:

```
Δk = ASR(Tier 0; D, A, M, s) − ASR(Tier k; D, A, M, s)
```

These two quantities are distinct. All numbers in this repo are **Tier-0 only** (`Δk` not yet measured; requires hardware — see protocol in paper §5).

---

## Fault-model tiers

| Tier | Mechanism | What changes |
|---|---|---|
| 0 | Simulated oracle (this repo) | Python directly modifies PyTorch weight tensors |
| 1 | Clock/voltage glitch | Flip delivery constrained to glitch timing window |
| 2 | CPU-DRAM Rowhammer | Flip delivery constrained to Rowhammer-exploitable rows |
| 3 | GPU-GDDR Rowhammer | Flip delivery constrained to GDDR6 address-mapped rows |

This harness is the **Tier-0 reference artifact**. The `run_pqc_coloc.py` script simulates the geometry of Tier-2 row adjacency analytically but does not execute constrained PBS; hardware-measured `Δk` values require the Tier 1–3 experiments described in §5 of the paper.

---

## Citation

```bibtex
@article{gupta2026physbench,
  title   = {Physically Grounded Evaluation of Bit-Flip Attack Defenses:
             Bridging Simulated and Hardware-Constrained Fault Models},
  author  = {Gupta, Bhanwar and Rana, Sanjeev},
  journal = {Journal of Systems Architecture},
  year    = {2026},
  note    = {Special Issue: Fault-aware security of current and post-quantum
             computing systems}
}
```

---

## License

MIT. See `LICENSE` for details. CIFAR-10 is used under its original terms (academic research only).
