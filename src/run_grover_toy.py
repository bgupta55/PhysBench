"""
Grover Illustration: PBS Candidate Search as an Unstructured Search Problem
============================================================================
This script produces a PURELY ILLUSTRATIVE toy example showing the structural
analogy between PBS's inner candidate-search loop (Algorithm 1, line 7 in the
manuscript) and Grover's unstructured search.

It is NOT a scaled demonstration and makes NO speedup claim.
The "search space" used here (3 qubits = 8 candidates) is trivially small;
the point is only to show the oracle / diffusion circuit structure as it would
map onto the PBS search problem, not to demonstrate a practical advantage.

The script:
  1. Defines a 3-qubit Grover oracle marking one "best flip" candidate.
  2. Runs 1 Grover iteration (optimal for N=8).
  3. Measures the output and confirms the marked item is found with high probability.
  4. Saves a circuit diagram to figures/fig_grover_toy.png.

This produces the figure used in the manuscript's Appendix / Discussion to
accompany the theoretical complexity paragraph about PBS's search loop.

Requirements:
    pip install qiskit qiskit-aer matplotlib pylatexenc
"""

import os
import sys
import json
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

OUT_FILE   = os.path.join(RESULTS_DIR, "grover_toy_results.json")
FIG_FILE   = os.path.join(FIGURES_DIR, "fig_grover_toy.png")


def build_grover_oracle(n_qubits, marked_state_int):
    """
    Build a Grover oracle that flips the phase of the marked computational
    basis state |marked_state_int>.

    In the PBS analogy, the 'marked state' is the index of the candidate
    (weight_index × bit_position) that maximises the substitute-batch loss
    — exactly the element PBS's inner loop is searching for.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PhaseOracle
    except ImportError as e:
        raise ImportError(f"Qiskit not found: {e}\n"
                          "pip install qiskit qiskit-aer") from e

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, name="Oracle")
    # Flip bits that are 0 in the marked state (so the marked state maps to |11...1>)
    for i in range(n_qubits):
        if not (marked_state_int >> i) & 1:
            qc.x(i)
    # Multi-controlled Z (phase flip of |11...1>)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    # Undo the bit flips
    for i in range(n_qubits):
        if not (marked_state_int >> i) & 1:
            qc.x(i)
    return qc


def build_diffusion(n_qubits):
    """Grover diffusion operator (inversion about the mean)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, name="Diffusion")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover_toy(n_qubits=3, marked=5, n_shots=1024):
    """
    Run one full Grover search (initial H + 1 oracle + 1 diffusion)
    on a simulator.  Returns the measurement outcome distribution.

    n_qubits=3  →  N=8 candidates, optimal n_iter=1 for k=1 marked item.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Initial uniform superposition
    qc.h(range(n_qubits))
    qc.barrier()

    # One Grover iteration
    oracle    = build_grover_oracle(n_qubits, marked)
    diffusion = build_diffusion(n_qubits)
    qc.compose(oracle,    inplace=True)
    qc.barrier()
    qc.compose(diffusion, inplace=True)
    qc.barrier()

    qc.measure(range(n_qubits), range(n_qubits))

    sim  = AerSimulator()
    job  = sim.run(transpile(qc, sim), shots=n_shots)
    counts = job.result().get_counts()

    # Convert to probability dict
    probs  = {k: v / n_shots for k, v in counts.items()}
    marked_bitstr = format(marked, f"0{n_qubits}b")  # e.g. "101" for 5
    marked_prob   = probs.get(marked_bitstr, 0.0)

    return qc, counts, probs, marked_bitstr, marked_prob


def save_circuit_diagram(qc, path):
    """Save the circuit diagram as a PNG."""
    try:
        fig = qc.draw("mpl", style="bw")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        fig.clf()
        import matplotlib.pyplot as plt
        plt.close("all")
        print(f"  Circuit diagram saved: {path}")
    except Exception as e:
        print(f"  Could not save circuit diagram ({e}) — continuing without it.")


def main():
    n_qubits = 3
    N        = 2 ** n_qubits     # = 8 candidates
    marked   = 5                  # the "best flip" candidate index
    n_shots  = 2048

    print("=" * 60)
    print("Grover Toy Illustration — PBS candidate search analogy")
    print("=" * 60)
    print(f"  Search space:  N = {N} candidates ({n_qubits} qubits)")
    print(f"  Marked item:   index {marked} ({format(marked, f'0{n_qubits}b')})")
    print(f"  Grover iters:  1  (optimal for N=8, k=1)")
    print(f"  Shots:         {n_shots}")
    print()
    print("PBS analogy:")
    print(f"  Each of the N={N} candidates represents a (weight_idx, bit_pos) pair.")
    print(f"  The oracle marks the pair that maximises substitute-batch loss.")
    print(f"  Grover finds it in O(√N) oracle calls vs. O(N) classical exhaustive search.")
    print(f"  At PBS's actual scale (top-k=8 × 8 bits = 64 candidates) this is")
    print(f"  irrelevant (64 evals is trivially fast); the analogy holds at larger scale.")
    print()

    try:
        qc, counts, probs, marked_bitstr, marked_prob = run_grover_toy(
            n_qubits=n_qubits, marked=marked, n_shots=n_shots
        )
    except ImportError as e:
        print(f"SKIP: {e}")
        print("Install qiskit-aer to run this experiment.")
        return

    print(f"  Marked bitstring: |{marked_bitstr}>")
    print(f"  Measured probability of marked item: {marked_prob:.4f}")
    print(f"  Classical expected (random guess):   {1/N:.4f}")
    print(f"  Grover theoretical (1 iter, N=8):    {np.sin(3*np.pi/8)**2:.4f}")
    print()

    # Top-5 outcomes
    top5 = sorted(probs.items(), key=lambda x: -x[1])[:5]
    print("  Top-5 measurement outcomes:")
    for bitstr, p in top5:
        marker = " ← marked" if bitstr == marked_bitstr else ""
        print(f"    |{bitstr}> : {p:.4f}{marker}")

    # Save circuit diagram
    save_circuit_diagram(qc, FIG_FILE)

    # Save JSON results
    result = {
        "config": {
            "n_qubits":  n_qubits,
            "N":         N,
            "marked":    marked,
            "n_shots":   n_shots,
            "n_grover_iters": 1,
        },
        "marked_bitstr":        marked_bitstr,
        "marked_prob_measured": marked_prob,
        "classical_baseline":   1.0 / N,
        "grover_theoretical":   float(np.sin(3 * np.pi / 8) ** 2),
        "counts":               counts,
        "probs":                {k: float(v) for k, v in probs.items()},
    }
    with open(OUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {OUT_FILE}")
    print()
    print("NOTE: This is purely illustrative.  No speedup claim is made.")
    print("      See manuscript Discussion section for exact framing.")


if __name__ == "__main__":
    main()
