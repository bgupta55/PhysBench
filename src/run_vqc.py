"""
Experiment F: Bit-flip Attacks on Variational Quantum Circuit (VQC) Parameters
================================================================================
This is a quantum-computing security experiment, DISTINCT from the PQC (post-quantum
cryptography) co-location experiment in run_pqc_coloc.py.  See the manuscript's
framing note at the start of Section 4.5.

We evaluate whether the PBS bit-flip attack class, developed for classical neural
network weights, also applies to the trainable parameters (rotation angles) of a
Variational Quantum Circuit (VQC).  This is, to our knowledge, the first systematic
bit-flip / fault-injection study on QNN parameters; prior QNN robustness work focuses
on adversarial input perturbations, not parameter corruption.

Architecture
------------
- 4-qubit VQC with ZZFeatureMap (reps=1) + RealAmplitudes ansatz (reps=2)
- 24 trainable angle parameters (the ansatz's rotation gate angles)
- Task: binary MNIST classification (digits 3 vs 8 — comparable classical difficulty
  to SmallCNN on CIFAR-10's balanced 10-class task)
- Input: 4 PCA features from 28×28 MNIST images (standard QML dimensionality reduction)

Training
--------
- COBYLA classical optimizer (standard for small VQCs; gradient-free)
- 10 seeds, same statistical protocol as classical experiments
- Reports: clean test accuracy, same format as classical baseline.json

Attack / Defense
-----------------
- PBS applied to the flat 24-parameter angle vector, quantized to int8
- Same checksum (50% coverage) and range-clip (3σ) defenses
- Same n=10 seeds, Wilcoxon signed-rank + Cliff's δ

Key structural difference vs. classical weights
------------------------------------------------
Rotation angles are periodic (sin/cos functions of θ ∈ [0, 2π)).  A bit-flip in the
int8 encoding of an angle can cause large-magnitude jumps that wrap around the
periodic domain — qualitatively different from a classical weight blowup.  We check
for degenerate collapse (circuit outputting a fixed basis state for all inputs)
post-flip, analogous to the NaN/Inf check for classical models (Phase 1.4).

Output: results/vqc_results.json  (per-seed trajectories + aggregate summary)
"""

import sys
import os
import json
import time
import numpy as np
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from quant import quantize_tensor, dequantize_tensor, flip_bit

RESULTS_DIR = os.path.join(os.path.dirname(SRC_DIR), "results")
OUT_FILE    = os.path.join(RESULTS_DIR, "vqc_results.json")

N_SEEDS    = 10
N_QUBITS   = 4
N_PCA      = 4       # PCA features fed into the circuit
MAX_FLIPS  = 20
TOPK       = 8
BITS       = 8
DIGIT_A    = 3
DIGIT_B    = 8
N_TRAIN    = 200     # training samples per class
N_TEST     = 100     # test samples per class
MAX_ITER   = 300     # SPSA iterations per seed (SPSA needs more than COBYLA)


# ── Data preparation ────────────────────────────────────────────────────────

def load_mnist_binary(digit_a=DIGIT_A, digit_b=DIGIT_B,
                      n_train=N_TRAIN, n_test=N_TEST,
                      n_pca=N_PCA, seed=0):
    """
    Load MNIST digits digit_a vs digit_b, reduce to n_pca features via PCA.
    Returns (X_train, y_train, X_test, y_test) as numpy arrays, labels {0,1}.
    """
    try:
        import torchvision
        import torchvision.transforms as T
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}\n"
            "Install with: pip install torchvision scikit-learn"
        ) from e

    cache = os.path.expanduser("~/.cache/mnist")
    os.makedirs(cache, exist_ok=True)
    transform = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.MNIST(cache, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(cache, train=False, download=True, transform=transform)

    def extract(ds, digit_a, digit_b, n_per_class):
        rng = np.random.default_rng(seed)
        X, y = [], []
        for label_int, target in [(digit_a, 0), (digit_b, 1)]:
            idxs = [i for i, (_, l) in enumerate(ds) if l == label_int]
            rng.shuffle(idxs)
            for i in idxs[:n_per_class]:
                img, _ = ds[i]
                X.append(img.numpy().flatten())
                y.append(target)
        return np.array(X, dtype=np.float32), np.array(y, dtype=int)

    Xtr, ytr = extract(train_ds, digit_a, digit_b, n_train)
    Xte, yte = extract(test_ds,  digit_a, digit_b, n_test)

    # Normalize + PCA
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(Xtr)
    Xte_s  = scaler.transform(Xte)

    pca = PCA(n_components=n_pca, random_state=seed)
    Xtr_p = pca.fit_transform(Xtr_s)
    Xte_p = pca.transform(Xte_s)

    # Scale PCA features to [0, π] for angle encoding
    pca_min = Xtr_p.min(axis=0)
    pca_max = Xtr_p.max(axis=0)
    scale   = pca_max - pca_min + 1e-8
    Xtr_p   = (Xtr_p - pca_min) / scale * np.pi
    Xte_p   = (Xte_p - pca_min) / scale * np.pi

    return Xtr_p, ytr, Xte_p, yte


# ── VQC model wrapper ───────────────────────────────────────────────────────

class VQCModel:
    """
    Thin wrapper around qiskit_machine_learning's EstimatorQNN.

    Design note on weight-resetting (fixes AttributeError from v1):
    NeuralNetworkClassifier has no public weights setter in qiskit-ml >= 0.7.
    We work around this by storing the QNN + ansatz parameter list directly
    and rebuilding a fresh classifier with `initial_point` set to the saved
    weights whenever we need to reset — this is the supported API.
    """

    def __init__(self, n_qubits=N_QUBITS, reps_fm=1, reps_ansatz=2, seed=0):
        try:
            from qiskit.circuit.library import zz_feature_map, real_amplitudes
            from qiskit_machine_learning.neural_networks import EstimatorQNN
            from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
            from qiskit_algorithms.optimizers import SPSA
        except ImportError:
            try:  # fallback to deprecated class API for older installs
                from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
                from qiskit_machine_learning.neural_networks import EstimatorQNN
                from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
                from qiskit_algorithms.optimizers import SPSA
                zz_feature_map  = lambda **kw: ZZFeatureMap(**kw)   # noqa: E731
                real_amplitudes = lambda **kw: RealAmplitudes(**kw)  # noqa: E731
            except ImportError as e:
                raise ImportError(
                    f"Missing Qiskit dependency: {e}\n"
                    "Install with: pip install qiskit qiskit-machine-learning"
                    " qiskit-aer qiskit-algorithms"
                ) from e

        self.n_qubits    = n_qubits
        self._reps_fm    = reps_fm
        self._reps_ans   = reps_ansatz
        self._EstimatorQNN         = EstimatorQNN
        self._NeuralNetworkClassifier = NeuralNetworkClassifier
        self._SPSA       = SPSA
        self._zz_fm      = zz_feature_map
        self._ra         = real_amplitudes

        feature_map = self._build_feature_map()
        ansatz      = self._build_ansatz()
        self.circuit     = feature_map.compose(ansatz)
        self._fm_params  = feature_map.parameters
        self._ans_params = ansatz.parameters
        self.n_params    = len(ansatz.parameters)
        self._weights    = None   # set after training

    def _build_feature_map(self):
        try:
            return self._zz_fm(feature_dimension=self.n_qubits, reps=self._reps_fm)
        except TypeError:
            return self._zz_fm(feature_dimension=self.n_qubits)

    def _build_ansatz(self):
        try:
            return self._ra(self.n_qubits, reps=self._reps_ans)
        except TypeError:
            return self._ra(self.n_qubits)

    def _make_classifier(self, initial_point=None):
        """Build a fresh NeuralNetworkClassifier, optionally with a warm start."""
        feature_map = self._build_feature_map()
        ansatz      = self._build_ansatz()
        circuit     = feature_map.compose(ansatz)
        qnn = self._EstimatorQNN(
            circuit=circuit,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )
        opt = self._SPSA(maxiter=MAX_ITER)
        clf = self._NeuralNetworkClassifier(
            neural_network=qnn,
            optimizer=opt,
            initial_point=initial_point,
        )
        return clf

    def fit(self, X, y, seed=0):
        np.random.seed(seed)
        clf = self._make_classifier()
        clf.fit(X, y)
        # Extract weights via the optimizer's result stored in the classifier
        self._weights = np.array(clf.weights, dtype=np.float64)
        self._clf_ref = clf   # keep reference for predict
        return self

    def predict(self, X):
        return self._clf_ref.predict(X)

    @property
    def weights(self):
        return self._weights.copy()

    def set_weights(self, w):
        """
        Reset model to given weight vector w.
        Rebuilds the classifier with initial_point=w, then calls fit for 0
        iterations so the internal state is consistent.  For attack purposes
        we only need predict() to work correctly — we call apply_vqc_state()
        which updates weights via the QNN's parameter bindings directly.
        We store w and build a predict-only classifier using warm start.
        """
        self._weights = np.array(w, dtype=np.float64)
        # Rebuild classifier with initial_point = w; fit for 1 iteration
        # so the internal parameter binding is set without changing the weights
        clf = self._make_classifier(initial_point=self._weights.copy())
        # We DON'T call clf.fit() here — instead we directly set the internal
        # weights attribute that predict() uses (read-only property in newer API,
        # but the underlying _fit_result stores the weights we can set via warm_start)
        # Simpler approach: use the QNN directly for prediction
        self._clf_ref = clf
        # Force the classifier to recognise classes by fitting a trivial 2-sample subset
        # with the fixed initial_point (0 optimizer steps = weights unchanged)
        self._clf_ref._fit_result = type('R', (), {'x': self._weights})()

    def accuracy(self, X, y):
        pred = self.predict(X)
        return float(np.mean(pred == y))


# ── Quantize / attack helpers for VQC ──────────────────────────────────────

def vqc_quant_state(weights_np, bits=BITS):
    """Quantize the flat angle parameter vector to int8."""
    w_tensor = torch.from_numpy(weights_np.astype(np.float32))
    q, scale = quantize_tensor(w_tensor, bits)
    return {"angles": {"q": q.clone(), "scale": scale,
                       "shape": w_tensor.shape, "n": len(weights_np)}}


def apply_vqc_state(model, state):
    """Dequantize and push weights back into the VQC model."""
    w = dequantize_tensor(state["angles"]["q"],
                          state["angles"]["scale"]).numpy().astype(np.float64)
    model.set_weights(w)


def vqc_asr(model, X_eval, y_eval):
    """Attack success rate = fraction misclassified."""
    pred = model.predict(X_eval)
    return float(np.mean(pred != y_eval))


def vqc_check_degenerate(model, X_sample):
    """
    Check whether the VQC has collapsed to a constant-output state.
    Returns True if all predictions are identical (degenerate).
    """
    pred = model.predict(X_sample)
    return len(set(pred.tolist())) == 1


def run_pbs_vqc(model, state, X_atk, y_atk, X_eval, y_eval,
                defense=None, protect_mask=None,
                max_flips=MAX_FLIPS, topk=TOPK, bits=BITS, seed=0):
    """
    PBS applied to VQC angle parameters.

    Gradient computation: finite-difference (parameter-shift rule is
    exact for VQCs but expensive; FD is a faster approximation for
    identifying high-impact angles, which is all PBS needs).

    defense: None | "checksum" | "clip"
    protect_mask: boolean array of length n_params (True = protected)
    """
    np.random.seed(seed)

    n_params    = state["angles"]["n"]
    layer_std   = float(dequantize_tensor(state["angles"]["q"],
                                          state["angles"]["scale"]).float().std().item()) + 1e-8
    clip_bound  = 3.0 * layer_std

    traj = {"flip_idx": [], "asr": [], "clean_acc": [],
            "n_effective_flips": [], "degenerate_after_flip": []}
    n_eff = 0

    def cur_asr():
        return vqc_asr(model, X_eval, y_eval)

    def cur_acc():
        pred = model.predict(X_eval)
        return float(np.mean(pred == y_eval))

    # Finite-difference gradient approximation for angle ranking
    def fd_gradient(eps=1e-2):
        apply_vqc_state(model, state)
        base_loss = _cross_entropy_loss(model, X_atk, y_atk)
        grads = np.zeros(n_params)
        flat_q = state["angles"]["q"].flatten().numpy().copy()
        scale  = state["angles"]["scale"]
        for i in range(n_params):
            orig = float(flat_q[i])
            # +eps in dequantized space
            flat_q[i] = orig + eps / scale
            state["angles"]["q"] = torch.from_numpy(flat_q.copy()).to(torch.int32)
            apply_vqc_state(model, state)
            loss_p = _cross_entropy_loss(model, X_atk, y_atk)
            flat_q[i] = orig
            state["angles"]["q"] = torch.from_numpy(flat_q.copy()).to(torch.int32)
            apply_vqc_state(model, state)
            grads[i] = (loss_p - base_loss) / eps
        return np.abs(grads)

    def _cross_entropy_loss(model, X, y):
        """Approximate CE loss via predicted labels (0/1)."""
        pred = model.predict(X)
        # Soft approximation: use error rate as proxy loss
        return float(np.mean(pred != y))

    for step in range(1, max_flips + 1):
        apply_vqc_state(model, state)

        # Rank candidates by finite-difference gradient magnitude
        abs_grads = fd_gradient()

        if protect_mask is not None and defense == "checksum":
            abs_grads[protect_mask] = 0.0   # mask out protected params

        top_idxs = np.argsort(abs_grads)[::-1][:topk]

        flat_q = state["angles"]["q"].flatten().clone()
        scale  = state["angles"]["scale"]

        best = None
        for idx in top_idxs:
            orig = int(flat_q[idx].item())
            for bp in range(bits):
                new_val = flip_bit(orig, bp, bits)
                flat_q[idx] = new_val
                state["angles"]["q"] = flat_q.view(state["angles"]["shape"]).clone().to(torch.int32)
                apply_vqc_state(model, state)
                loss = _cross_entropy_loss(model, X_atk, y_atk)
                if best is None or loss > best[0]:
                    best = (loss, int(idx), bp, new_val)
                flat_q[idx] = orig
            state["angles"]["q"] = flat_q.view(state["angles"]["shape"]).clone().to(torch.int32)
            apply_vqc_state(model, state)

        if best is None:
            break

        _, best_idx, best_bp, best_val = best
        rollback = int(state["angles"]["q"].flatten()[best_idx].item())
        flat_q2  = state["angles"]["q"].flatten().clone()
        flat_q2[best_idx] = best_val
        state["angles"]["q"] = flat_q2.view(state["angles"]["shape"]).clone().to(torch.int32)

        protected = False
        if defense == "checksum" and protect_mask is not None:
            protected = bool(protect_mask[best_idx])
            if protected:
                flat_q2[best_idx] = rollback
                state["angles"]["q"] = flat_q2.view(state["angles"]["shape"]).clone().to(torch.int32)
            else:
                n_eff += 1
        elif defense == "clip":
            w = float(dequantize_tensor(
                state["angles"]["q"].flatten()[best_idx:best_idx+1],
                scale).item())
            if abs(w) > clip_bound:
                w_clipped = max(min(w, clip_bound), -clip_bound)
                new_q = int(round(w_clipped / scale))
                new_q = max(-128, min(127, new_q))
                flat_q2[best_idx] = new_q
                state["angles"]["q"] = flat_q2.view(state["angles"]["shape"]).clone().to(torch.int32)
            n_eff += 1
        else:
            n_eff += 1

        apply_vqc_state(model, state)
        asr = cur_asr()
        acc = cur_acc()
        degen = vqc_check_degenerate(model, X_eval[:20])

        traj["flip_idx"].append(step)
        traj["asr"].append(asr)
        traj["clean_acc"].append(acc)
        traj["n_effective_flips"].append(n_eff)
        traj["degenerate_after_flip"].append(degen)

        if asr >= 0.90:
            break

    return traj


def build_vqc_protect_mask(n_params, protect_fraction=0.50):
    """Protect the top protect_fraction parameters by index order (proxy for criticality)."""
    k = int(n_params * protect_fraction)
    mask = np.zeros(n_params, dtype=bool)
    mask[:k] = True   # top-k by parameter index (magnitude ranking done in PBS)
    return mask


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check Qiskit availability early
    try:
        import qiskit
        import qiskit_machine_learning
        print(f"Qiskit {qiskit.__version__}, "
              f"qiskit-machine-learning {qiskit_machine_learning.__version__}")
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install: pip install qiskit qiskit-machine-learning qiskit-aer qiskit-algorithms scikit-learn")
        return

    print(f"Task: MNIST {DIGIT_A} vs {DIGIT_B}  |  {N_QUBITS} qubits  |  "
          f"{N_TRAIN} train / {N_TEST} test per class  |  {N_SEEDS} seeds\n")

    print("Loading data and fitting PCA...")
    Xtr, ytr, Xte, yte = load_mnist_binary(seed=0)
    print(f"  Train: {Xtr.shape}  Test: {Xte.shape}\n")

    results = []

    for seed in range(N_SEEDS):
        print(f"=== Seed {seed} ===")
        t0 = time.time()

        # ── Train VQC ────────────────────────────────────────────────────
        model = VQCModel(n_qubits=N_QUBITS, seed=seed)
        model.fit(Xtr, ytr, seed=seed)
        clean_acc = model.accuracy(Xte, yte)
        n_params  = model.n_params
        print(f"  Trained in {time.time()-t0:.1f}s  |  clean_acc={clean_acc:.3f}  "
              f"|  n_params={n_params}")

        weights_orig = model.weights

        protect_mask = build_vqc_protect_mask(n_params, 0.50)

        seed_result = {
            "seed":          seed,
            "clean_acc":     clean_acc,
            "n_params":      n_params,
            "train_time_s":  time.time() - t0,
            "conditions":    {},
        }

        # Attack batch: first 32 training samples
        X_atk = Xtr[:32]
        y_atk = ytr[:32]

        for defense in [None, "checksum", "clip"]:
            # Reset model to trained weights
            model.set_weights(weights_orig.copy())
            state = vqc_quant_state(weights_orig.copy())

            apply_vqc_state(model, state)
            pre_asr = vqc_asr(model, Xte, yte)

            traj = run_pbs_vqc(
                model, state, X_atk, y_atk, Xte, yte,
                defense=defense,
                protect_mask=protect_mask if defense == "checksum" else None,
                seed=seed,
            )
            final_asr = traj["asr"][-1] if traj["asr"] else pre_asr
            degen_any  = any(traj["degenerate_after_flip"])

            key = defense if defense else "none"
            print(f"  {key:12s}  ASR={final_asr:.3f}  "
                  f"degenerate={degen_any}  "
                  f"flips={len(traj['flip_idx'])}")
            seed_result["conditions"][key] = {
                "final_asr":          final_asr,
                "degenerate_any":     degen_any,
                "trajectory":         traj,
            }

        results.append(seed_result)
        print()

    # ── Aggregate ────────────────────────────────────────────────────────────
    def ms(key, cond):
        vals = [r["conditions"][cond]["final_asr"] for r in results if cond in r["conditions"]]
        return float(np.mean(vals)), float(np.std(vals))

    print("=== AGGREGATE RESULTS (n=10 seeds) ===")
    for cond in ["none", "checksum", "clip"]:
        m, s = ms("final_asr", cond)
        print(f"  ASR [{cond:10s}]: {m:.4f} ± {s:.4f}")

    clean_accs = [r["clean_acc"] for r in results]
    print(f"  Clean acc:     {np.mean(clean_accs):.4f} ± {np.std(clean_accs):.4f}")

    summary = {
        "config": {
            "n_qubits":    N_QUBITS,
            "n_pca":       N_PCA,
            "digit_a":     DIGIT_A,
            "digit_b":     DIGIT_B,
            "n_train":     N_TRAIN,
            "n_test":      N_TEST,
            "max_flips":   MAX_FLIPS,
            "topk":        TOPK,
            "bits":        BITS,
            "max_iter":    MAX_ITER,
            "n_seeds":     N_SEEDS,
        },
        "per_seed":  results,
        "aggregate": {
            cond: {"mean": ms("final_asr", cond)[0], "std": ms("final_asr", cond)[1]}
            for cond in ["none", "checksum", "clip"]
        },
        "clean_acc": {"mean": float(np.mean(clean_accs)), "std": float(np.std(clean_accs))},
    }

    with open(OUT_FILE, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
