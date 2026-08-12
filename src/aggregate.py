"""
aggregate.py — Aggregate results from all seeds and experiments.

Usage:
    python aggregate.py [--out_dir PATH] [--seeds 0 1 2 3 4 5 6 7 8 9]

Reads:
  results/seed{0..N}.json
  results/baseline.json
  results/sweep_seed{0..N}.json
  results/adaptive/adaptive_seed{0..N}.json
  results/resnet20/resnet20_seed{0..N}.json     (if present)
  results/targeted/targeted_seed{0..N}.json     (if present)

Writes:
  results/summary.json         (original-style + new experiments)
  results/adaptive/adaptive_summary.json
  results/resnet20/resnet20_summary.json        (if data present)
  results/targeted/targeted_summary.json        (if data present)
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats


def wilcoxon_safe(a, b):
    diffs = np.array(a) - np.array(b)
    if np.all(diffs == diffs[0]):
        return None, None
    try:
        w, p = stats.wilcoxon(a, b)
        return float(w), float(p)
    except Exception:
        return None, None


def cliffs_delta(a, b):
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def aggregate_main(seeds, out_dir):
    results_dir = os.path.join(out_dir, "results")
    defenses    = ["none", "checksum", "clip"]
    budget      = 20

    # ── Baseline ────────────────────────────────────────────────────────────
    baseline_path = os.path.join(results_dir, "baseline.json")
    baseline      = json.load(open(baseline_path)) if os.path.exists(baseline_path) else {}

    # ── Per-seed run results ──────────────────────────────────────────────
    available_seeds = [s for s in seeds
                       if os.path.exists(os.path.join(results_dir, f"seed{s}.json"))]
    if not available_seeds:
        print("No seed results found; skipping main summary.")
        all_results = {}
    else:
        all_results = {s: json.load(open(os.path.join(results_dir, f"seed{s}.json")))
                       for s in available_seeds}

    summary = {"seeds_used": available_seeds,
               "clean_test_acc": {}, "asr_at_budget": {},
               "acc_drop_at_budget": {}, "flips_to_50pct_asr": {},
               "significance_vs_none": {}}

    if baseline and available_seeds:
        clean_accs = [baseline[str(s)]["clean_test_acc"]
                      for s in available_seeds if str(s) in baseline]
        summary["clean_test_acc"] = {
            "mean": float(np.mean(clean_accs)), "std": float(np.std(clean_accs)),
            "per_seed": clean_accs
        }

    if all_results:
        for d in defenses:
            # pad short trajectories to budget length
            asrs    = []
            accs    = []
            postqs  = []
            flips50 = []
            for s in available_seeds:
                traj = all_results[s][d]
                asr_traj  = traj["asr"]
                acc_traj  = traj["clean_acc"]
                pq        = traj["post_quant_clean_acc"]
                # Pad to budget length if early exit
                while len(asr_traj) < budget:
                    asr_traj.append(asr_traj[-1])
                    acc_traj.append(acc_traj[-1])
                asrs.append(asr_traj[budget - 1])
                accs.append(acc_traj[budget - 1])
                postqs.append(pq)
                hit = next((i + 1 for i, a in enumerate(traj["asr"]) if a >= 0.5), budget + 1)
                flips50.append(hit)

            drop = [postqs[i] - accs[i] for i in range(len(available_seeds))]
            summary["asr_at_budget"][d] = {
                "mean": float(np.mean(asrs)), "std": float(np.std(asrs)),
                "per_seed": asrs
            }
            summary["acc_drop_at_budget"][d] = {
                "mean": float(np.mean(drop)), "std": float(np.std(drop)),
                "per_seed": drop
            }
            summary["flips_to_50pct_asr"][d] = {
                "mean": float(np.mean(flips50)), "std": float(np.std(flips50)),
                "per_seed": flips50
            }

        none_asr = summary["asr_at_budget"]["none"]["per_seed"]
        for d in ["checksum", "clip"]:
            d_asr  = summary["asr_at_budget"][d]["per_seed"]
            w, p   = wilcoxon_safe(none_asr, d_asr)
            cd     = cliffs_delta(none_asr, d_asr)
            summary["significance_vs_none"][d] = {
                "wilcoxon_stat": w, "p_value": p,
                "cliffs_delta":  float(cd),
                "mean_asr_reduction_pp": float(np.mean(none_asr) - np.mean(d_asr)) * 100
            }

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote summary.json")

    # ── Adaptive summary (Exp A) ─────────────────────────────────────────
    adap_dir = os.path.join(results_dir, "adaptive")
    adap_seeds = [s for s in available_seeds
                  if os.path.exists(os.path.join(adap_dir, f"adaptive_seed{s}.json"))]
    if adap_seeds:
        adap_results = {s: json.load(open(os.path.join(adap_dir, f"adaptive_seed{s}.json")))
                        for s in adap_seeds}
        adap_conds   = ["none", "checksum", "checksum_adaptive"]
        adap_summary = {"seeds_used": adap_seeds, "asr_at_budget": {}}
        for cond in adap_conds:
            asrs = []
            for s in adap_seeds:
                traj     = adap_results[s][cond]
                asr_traj = traj["asr"][:]
                while len(asr_traj) < budget:
                    asr_traj.append(asr_traj[-1])
                asrs.append(asr_traj[budget - 1])
            adap_summary["asr_at_budget"][cond] = {
                "mean": float(np.mean(asrs)), "std": float(np.std(asrs)),
                "per_seed": asrs
            }
        # How much does adaptive recover vs non-adaptive?
        none_a = adap_summary["asr_at_budget"]["none"]["per_seed"]
        cs_a   = adap_summary["asr_at_budget"]["checksum"]["per_seed"]
        ca_a   = adap_summary["asr_at_budget"]["checksum_adaptive"]["per_seed"]
        adap_summary["recovery_analysis"] = {
            "non_adaptive_reduction_pp":
                float(np.mean(none_a) - np.mean(cs_a)) * 100,
            "adaptive_reduction_pp":
                float(np.mean(none_a) - np.mean(ca_a)) * 100,
            "adaptive_recovered_pp":
                float(np.mean(ca_a) - np.mean(cs_a)) * 100,
            "interpretation":
                ("adaptive_close_to_undefended" if np.mean(ca_a) > np.mean(cs_a) + 0.10
                 else "checksum_robust_to_adaptive")
        }
        adap_out = os.path.join(adap_dir, "adaptive_summary.json")
        with open(adap_out, "w") as f:
            json.dump(adap_summary, f, indent=2)
        print("Wrote adaptive_summary.json")

    # ── ResNet-20 summary (Exp B) ─────────────────────────────────────────
    rn_dir   = os.path.join(results_dir, "resnet20")
    rn_seeds = [s for s in seeds
                if os.path.exists(os.path.join(rn_dir, f"resnet20_seed{s}.json"))]
    if rn_seeds:
        rn_results  = {s: json.load(open(os.path.join(rn_dir, f"resnet20_seed{s}.json")))
                       for s in rn_seeds}
        rn_summary  = {"seeds_used": rn_seeds, "asr_at_budget": {},
                       "significance_vs_none": {}}
        for d in defenses:
            asrs = []
            for s in rn_seeds:
                traj     = rn_results[s][d]
                asr_traj = traj["asr"][:]
                while len(asr_traj) < budget:
                    asr_traj.append(asr_traj[-1])
                asrs.append(asr_traj[budget - 1])
            rn_summary["asr_at_budget"][d] = {
                "mean": float(np.mean(asrs)), "std": float(np.std(asrs)),
                "per_seed": asrs
            }
        rn_none_asr = rn_summary["asr_at_budget"]["none"]["per_seed"]
        for d in ["checksum", "clip"]:
            d_asr = rn_summary["asr_at_budget"][d]["per_seed"]
            w, p  = wilcoxon_safe(rn_none_asr, d_asr)
            cd    = cliffs_delta(rn_none_asr, d_asr)
            rn_summary["significance_vs_none"][d] = {
                "wilcoxon_stat": w, "p_value": p,
                "cliffs_delta":  float(cd),
                "mean_asr_reduction_pp": float(np.mean(rn_none_asr) - np.mean(d_asr)) * 100
            }
        rn_out = os.path.join(rn_dir, "resnet20_summary.json")
        with open(rn_out, "w") as f:
            json.dump(rn_summary, f, indent=2)
        print("Wrote resnet20_summary.json")

    # ── Targeted summary (Exp D) ─────────────────────────────────────────
    tgt_dir   = os.path.join(results_dir, "targeted")
    tgt_seeds = [s for s in available_seeds
                 if os.path.exists(os.path.join(tgt_dir, f"targeted_seed{s}.json"))]
    if tgt_seeds:
        from run_targeted import CLASS_PAIRS
        tgt_results = {s: json.load(open(os.path.join(tgt_dir, f"targeted_seed{s}.json")))
                       for s in tgt_seeds}
        tgt_summary = {"seeds_used": tgt_seeds, "per_pair": {}}
        pair_labels = [pl for _, _, pl in CLASS_PAIRS]
        for pl in pair_labels:
            tgt_summary["per_pair"][pl] = {}
            for def_key in ["none", "checksum", "clip"]:
                rk   = f"{pl}_{def_key}"
                tasrs = []
                for s in tgt_seeds:
                    traj = tgt_results[s].get(rk, {})
                    tasr_traj = traj.get("targeted_asr", [0.0])
                    while len(tasr_traj) < budget:
                        tasr_traj.append(tasr_traj[-1])
                    tasrs.append(tasr_traj[min(budget - 1, len(tasr_traj) - 1)])
                tgt_summary["per_pair"][pl][def_key] = {
                    "mean_targeted_asr": float(np.mean(tasrs)),
                    "std_targeted_asr":  float(np.std(tasrs)),
                    "per_seed": tasrs
                }
        tgt_out = os.path.join(tgt_dir, "targeted_summary.json")
        with open(tgt_out, "w") as f:
            json.dump(tgt_summary, f, indent=2)
        print("Wrote targeted_summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), ".."))
    args = parser.parse_args()
    aggregate_main(args.seeds, args.out_dir)


if __name__ == "__main__":
    main()
