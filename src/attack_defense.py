"""
Attack and defense implementations for PhysBench Tier-0.

Includes:
  - progressive_bit_search: original PBS (non-adaptive, as in paper)
  - progressive_bit_search_adaptive: Experiment A — attacker knows the
      defense and excludes protected weights from candidate selection
  - progressive_bit_search_targeted: Experiment D — targeted class attack

Defenses: critical_subset_checksum, range_clip.
"""
import copy
import torch
import torch.nn.functional as F
from quant import quantize_tensor, dequantize_tensor, flip_bit

# Default attackable layers for SmallCNN
ATTACKABLE_LAYERS_SMALL = ["features.8.weight", "classifier.0.weight", "classifier.2.weight"]

# Attackable layers for ResNet-20: last residual stage + classifier
ATTACKABLE_LAYERS_RESNET20 = [
    "layer3.0.conv1.weight", "layer3.0.conv2.weight",
    "layer3.1.conv1.weight", "layer3.1.conv2.weight",
    "layer3.2.conv1.weight", "layer3.2.conv2.weight",
    "fc.weight",
]


def get_quant_state(model, bits=8, attackable_layers=None):
    layers = attackable_layers or ATTACKABLE_LAYERS_SMALL
    state = {}
    params = dict(model.named_parameters())
    for name in layers:
        if name not in params:
            continue
        w = params[name].data
        q, scale = quantize_tensor(w, bits)
        state[name] = {"q": q.clone(), "scale": scale, "shape": w.shape}
    return state


def apply_state_to_model(model, state):
    params = dict(model.named_parameters())
    for name, s in state.items():
        w = dequantize_tensor(s["q"], s["scale"]).view(s["shape"])
        params[name].data.copy_(w)


def loss_on_batch(model, X, y):
    out = model(X.contiguous())
    return F.cross_entropy(out, y)


def accuracy(model, X, y):
    with torch.no_grad():
        out = model(X.contiguous())
        pred = out.argmax(1)
        return (pred == y).float().mean().item()


def build_protection_mask(state, protect_fraction, seed):
    """Rank by |quantized value| (defender-side proxy for criticality) and
    protect the top `protect_fraction` as checksum-guarded."""
    mask = {}
    for name, s in state.items():
        flat = s["q"].flatten().float().abs()
        k = int(len(flat) * protect_fraction)
        if k == 0:
            mask[name] = torch.zeros_like(s["q"], dtype=torch.bool)
            continue
        thresh = torch.topk(flat, k).values.min()
        mask[name] = (s["q"].abs() >= thresh)
    return mask


# ─────────────────────────────────────────────────────────────
# Helper: shared body for PBS variants
# ─────────────────────────────────────────────────────────────

def _pbs_core(model, state, X_atk, y_atk, X_eval, y_eval,
              max_flips, topk, bits, defense, protect_mask, clip_c,
              target_asr, seed, adaptive=False,
              targeted=False, source_class=None, target_class=None,
              targeted_lambda=0.5):
    """
    Shared implementation for all PBS variants. Returns trajectory dict.

    adaptive=True  → zero out protected-weight gradients before top-k ranking
                     (Experiment A: attacker knows the defense mask)
    targeted=True  → use targeted loss objective on (source_class, target_class)
                     (Experiment D)
    """
    torch.manual_seed(seed)
    traj = {"flip_idx": [], "asr": [], "clean_acc": [], "n_effective_flips": []}
    if targeted:
        traj["targeted_asr"] = []   # fraction of source imgs → target class
    n_effective = 0

    layer_stds = {name: dequantize_tensor(s["q"], s["scale"]).float().std().item() + 1e-8
                  for name, s in state.items()}

    def current_asr():
        with torch.no_grad():
            out = model(X_eval)
            pred = out.argmax(1)
            untargeted = (pred != y_eval).float().mean().item()
            return untargeted

    def current_targeted_asr():
        """Fraction of source-class images classified as target_class."""
        with torch.no_grad():
            mask_src = (y_eval == source_class)
            if mask_src.sum() == 0:
                return 0.0
            out = model(X_eval[mask_src].contiguous())
            pred = out.argmax(1)
            return (pred == target_class).float().mean().item()

    for step in range(1, max_flips + 1):
        apply_state_to_model(model, state)
        model.zero_grad()
        for name in state.keys():
            dict(model.named_parameters())[name].requires_grad_(True)

        # ── compute attacker loss ──────────────────────────────────────
        if targeted:
            # source-class batch → maximize prob of target_class
            mask_src = (y_atk == source_class)
            mask_other = ~mask_src
            loss = torch.tensor(0.0, requires_grad=True)
            if mask_src.sum() > 0:
                out_src = model(X_atk[mask_src].contiguous())
                tgt_labels = torch.full((mask_src.sum(),), target_class, dtype=torch.long, device=X_atk.device)
                loss_src = F.cross_entropy(out_src, tgt_labels)
                loss = loss + loss_src
            if mask_other.sum() > 0:
                out_other = model(X_atk[mask_other].contiguous())
                loss_other = F.cross_entropy(out_other, y_atk[mask_other])
                loss = loss - targeted_lambda * loss_other
        else:
            loss = loss_on_batch(model, X_atk, y_atk)

        loss.backward()

        # ── rank candidates ────────────────────────────────────────────
        candidates = []
        for name in state.keys():
            grad = dict(model.named_parameters())[name].grad
            if grad is None:
                continue
            flat_grad = grad.abs().flatten()

            # Experiment A: zero out gradients at protected positions
            if adaptive and protect_mask is not None and name in protect_mask:
                flat_grad = flat_grad.clone()
                flat_grad[protect_mask[name].flatten()] = 0.0

            k = min(topk, (flat_grad > 0).sum().item())
            k = max(k, 1)
            top_idx = torch.topk(flat_grad, k).indices
            for idx in top_idx.tolist():
                candidates.append((name, idx))

        if not candidates:
            break

        # ── exhaustive bit search within candidates ────────────────────
        best = None
        for name, idx in candidates:
            q = state[name]["q"]
            flat_q = q.flatten()
            orig_val = int(flat_q[idx].item())
            for bitpos in range(bits):
                new_val = flip_bit(orig_val, bitpos, bits)
                flat_q[idx] = new_val
                apply_state_to_model(model, state)
                with torch.no_grad():
                    if targeted:
                        mask_src = (y_atk == source_class)
                        if mask_src.sum() > 0:
                            out_src = model(X_atk[mask_src].contiguous())
                            tgt_lbl = torch.full((mask_src.sum(),), target_class, dtype=torch.long, device=X_atk.device)
                            l = F.cross_entropy(out_src, tgt_lbl).item()
                        else:
                            l = 0.0
                    else:
                        l = loss_on_batch(model, X_atk, y_atk).item()
                if best is None or l > best[0]:
                    best = (l, name, idx, bitpos, new_val)
                flat_q[idx] = orig_val
            apply_state_to_model(model, state)

        if best is None:
            break

        _, name, idx, bitpos, new_val = best
        orig_val_for_rollback = int(state[name]["q"].flatten()[idx].item())
        state[name]["q"].flatten()[idx] = new_val

        # ── defense intervention ───────────────────────────────────────
        protected = False
        if defense == "checksum":
            protected = protect_mask[name].flatten()[idx].item()
            if not protected:
                n_effective += 1
        elif defense == "clip":
            w = dequantize_tensor(state[name]["q"], state[name]["scale"])
            bound = clip_c * layer_stds[name]
            wflat = w.flatten()
            val = wflat[idx].item()
            if abs(val) > bound:
                wflat[idx] = max(min(val, bound), -bound)
                clipped_q = torch.clamp(
                    torch.round(wflat / state[name]["scale"]),
                    -(2**(bits-1)), 2**(bits-1)-1
                ).to(torch.int32)
                state[name]["q"] = clipped_q.view(state[name]["shape"])
            n_effective += 1
        else:
            n_effective += 1

        if defense == "checksum" and protected:
            state[name]["q"].flatten()[idx] = orig_val_for_rollback

        apply_state_to_model(model, state)
        asr = current_asr()
        acc = accuracy(model, X_eval, y_eval)
        traj["flip_idx"].append(step)
        traj["asr"].append(asr)
        traj["clean_acc"].append(acc)
        traj["n_effective_flips"].append(n_effective)
        if targeted:
            traj["targeted_asr"].append(current_targeted_asr())

        if targeted:
            if current_targeted_asr() >= target_asr:
                break
        else:
            if asr >= target_asr:
                break

    return traj


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def progressive_bit_search(model, state, X_atk, y_atk, X_eval, y_eval,
                            max_flips=20, topk=8, bits=8,
                            defense=None, protect_mask=None, clip_c=3.0,
                            target_asr=0.90, seed=0):
    """Original PBS — attacker has no knowledge of the defense mask."""
    return _pbs_core(model, state, X_atk, y_atk, X_eval, y_eval,
                     max_flips=max_flips, topk=topk, bits=bits,
                     defense=defense, protect_mask=protect_mask, clip_c=clip_c,
                     target_asr=target_asr, seed=seed, adaptive=False)


def progressive_bit_search_adaptive(model, state, X_atk, y_atk, X_eval, y_eval,
                                     max_flips=20, topk=8, bits=8,
                                     defense=None, protect_mask=None, clip_c=3.0,
                                     target_asr=0.90, seed=0):
    """
    Experiment A: Adaptive PBS.

    The attacker has white-box knowledge of the protection mask and zeroes
    out gradient magnitude at every protected-weight index before the top-k
    candidate ranking step.  This means the top-k selection never nominates
    a checksum-protected weight, bypassing the checksum defense entirely.
    """
    return _pbs_core(model, state, X_atk, y_atk, X_eval, y_eval,
                     max_flips=max_flips, topk=topk, bits=bits,
                     defense=defense, protect_mask=protect_mask, clip_c=clip_c,
                     target_asr=target_asr, seed=seed, adaptive=True)


def progressive_bit_search_targeted(model, state, X_atk, y_atk, X_eval, y_eval,
                                     source_class, target_class,
                                     max_flips=20, topk=8, bits=8,
                                     defense=None, protect_mask=None, clip_c=3.0,
                                     target_asr=0.70, seed=0,
                                     targeted_lambda=0.5):
    """
    Experiment D: Targeted PBS.

    Maximizes the probability that source_class images are classified as
    target_class, while penalising collateral accuracy drop on other classes.

    Loss = CE(source_batch → target_class)
           - lambda * CE(other_class_batch → true_label)
    """
    return _pbs_core(model, state, X_atk, y_atk, X_eval, y_eval,
                     max_flips=max_flips, topk=topk, bits=bits,
                     defense=defense, protect_mask=protect_mask, clip_c=clip_c,
                     target_asr=target_asr, seed=seed, adaptive=False,
                     targeted=True, source_class=source_class,
                     target_class=target_class, targeted_lambda=targeted_lambda)
