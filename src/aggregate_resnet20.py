"""
aggregate_resnet20.py — Recompute resnet20_summary.json from per-seed JSON files.

Usage: python aggregate_resnet20.py [--out_dir PATH]

Run this after all resnet20_seed{0-9}.json files have been written by run_resnet20_seed.py.
"""
import argparse, json, os
import numpy as np
from scipy.stats import wilcoxon

def cliff_delta(x, y):
    """Cliff's delta effect size (x vs y, paired)."""
    n = len(x)
    more = sum(xi > yi for xi, yi in zip(x, y))
    less = sum(xi < yi for xi, yi in zip(x, y))
    return (more - less) / (n * n) if n > 0 else 0.0

def wilcoxon_p(x, y):
    try:
        stat, p = wilcoxon([xi - yi for xi, yi in zip(x, y)])
        return float(stat), float(p)
    except Exception:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), ".."))
    args = parser.parse_args()

    results_dir = os.path.join(args.out_dir, "results", "resnet20")
    per_seed = []
    for s in range(10):
        path = os.path.join(results_dir, f"resnet20_seed{s}.json")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found — skipping seed {s}")
            continue
        per_seed.append((s, json.load(open(path))))

    seeds_used = [s for s, _ in per_seed]
    conditions = ["none", "checksum", "clip"]

    asr_by_cond = {c: [d[c]["asr"][-1] for _, d in per_seed] for c in conditions}
    clean_acc   = [d["none"]["post_quant_clean_acc"] for _, d in per_seed]

    summary = {
        "seeds_used": seeds_used,
        "clean_acc": {"mean": float(np.mean(clean_acc)), "std": float(np.std(clean_acc)),
                      "per_seed": clean_acc},
        "asr_at_budget": {},
        "significance_vs_none": {},
    }

    for c in conditions:
        vals = asr_by_cond[c]
        summary["asr_at_budget"][c] = {
            "mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_seed": vals
        }

    for c in ["checksum", "clip"]:
        none_vals = asr_by_cond["none"]
        c_vals    = asr_by_cond[c]
        stat, p   = wilcoxon_p(none_vals, c_vals)
        cd        = cliff_delta(none_vals, c_vals)
        reduction = float(np.mean(none_vals) - np.mean(c_vals)) * 100
        summary["significance_vs_none"][c] = {
            "wilcoxon_stat": stat, "p_value": p,
            "cliffs_delta": cd,
            "mean_asr_reduction_pp": reduction,
        }

    out_path = os.path.join(results_dir, "resnet20_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {out_path}")
    print(f"\nRESULTS SUMMARY:")
    print(f"  clean_acc:  {summary['clean_acc']['mean']:.4f} ± {summary['clean_acc']['std']:.4f}")
    for c in conditions:
        v = summary["asr_at_budget"][c]
        print(f"  asr_{c:<10}: {v['mean']:.4f} ± {v['std']:.4f}")
    for c in ["checksum", "clip"]:
        v = summary["significance_vs_none"][c]
        print(f"  {c}: delta={v['cliffs_delta']:.2f}  p={v['p_value']}  "
              f"reduction={v['mean_asr_reduction_pp']:.1f}pp")

if __name__ == "__main__":
    main()
