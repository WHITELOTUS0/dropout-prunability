"""Pruning-probe statistics: seed-paired tests of each dropout rate against the p=0 control.

For every sparsity on the sweep grid and every dropout rate p > 0, computes
    delta_k = Acc(s, p; seed k) - Acc(s, 0; seed k)
over seeds and reports the mean, standard deviation and a one-sample t test of delta = 0.
Done for plain magnitude pruning and for magnitude pruning + BatchNorm recalibration,
for the head-dropout and the block-dropout models. Also draws the probe figure
(raw / recalibrated / random) used in the report.

Usage: python analysis/probe_tests.py --runs runs --out figures/analysis
Writes probe_tests.csv, probe_tests.md, probe_head.png, probe_block.png
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

COLORS = {0.0: "#1f6feb", 0.1: "#e0712e", 0.3: "#2a9d5c", 0.5: "#9b4fd1"}
PS = [0.0, 0.1, 0.3, 0.5]


def load_sweeps(runs_dir):
    """Return {(where, p, seed): prune_sweep dict}. The p=0 model is shared by both placements."""
    out = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "p*_s*"))):
        f = os.path.join(d, "prune_sweep.json")
        if not os.path.isdir(d) or not os.path.exists(f):
            continue
        sw = json.load(open(f))
        where = "block" if "blk" in os.path.basename(d) else "head"
        out[(where, float(sw["p"]), int(sw["seed"]))] = sw
    for (where, p, seed), sw in list(out.items()):
        if p == 0.0:  # Dropout(0) and Dropout2d(0) are both the identity: one control serves both
            out[("head", 0.0, seed)] = sw
            out[("block", 0.0, seed)] = sw
    return out


def acc_curve(sw, key):
    return np.array([r["test_acc"] for r in sw[key]])


def paired_table(sweeps):
    rows = []
    S = next(iter(sweeps.values()))["sparsities"]
    for where in ["head", "block"]:
        for key in ["magnitude", "magnitude_bnrecal"]:
            for p in PS[1:]:
                seeds = sorted(k[2] for k in sweeps if k[0] == where and k[1] == p)
                seeds = [s for s in seeds if (where, 0.0, s) in sweeps]
                if len(seeds) < 2:
                    continue
                for i, s in enumerate(S):
                    d = [acc_curve(sweeps[(where, p, k)], key)[i] - acc_curve(sweeps[(where, 0.0, k)], key)[i] for k in seeds]
                    t = stats.ttest_1samp(d, 0.0)
                    rows.append(dict(placement=where, pruning=key, p=p, sparsity=s, n_seeds=len(d),
                                     mean_diff=np.mean(d), sd_diff=np.std(d, ddof=1), t=t.statistic, p_value=t.pvalue,
                                     n_positive=int(sum(x > 0 for x in d))))
    return pd.DataFrame(rows)


def mean_std(sweeps, where, p, key):
    a = np.array([acc_curve(sw, key) for (w, q, _), sw in sweeps.items() if w == where and q == p])
    return a.mean(0), (a.std(0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1]))


def probe_figure(sweeps, where, path):
    S = np.array(next(iter(sweeps.values()))["sparsities"]) * 100
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.9), dpi=200)
    for p in PS:
        if not any(k[0] == where and k[1] == p for k in sweeps):
            continue
        lab = "control" if p == 0 else f"{where} p={p}"
        for key, a in [("magnitude", ax[0]), ("magnitude_bnrecal", ax[1])]:
            m, s = mean_std(sweeps, where, p, key)
            a.errorbar(S, m, yerr=s, color=COLORS[p], marker="o", ms=3, lw=1.5, capsize=2, label=lab)
        m, s = mean_std(sweeps, where, p, "random")
        ax[2].errorbar(S, m, yerr=s, color=COLORS[p], marker="x", ms=3, lw=1, ls="--", capsize=2, label=f"{lab} random")
        m, _ = mean_std(sweeps, where, p, "magnitude_bnrecal")
        ax[2].plot(S, m, color=COLORS[p], lw=1.2, alpha=0.6)
    ax[0].set(title="(a) magnitude pruning, no fine-tuning", xlabel="sparsity (%)", ylabel="test accuracy (%)")
    ax[1].set(title="(b) + BatchNorm recalibration", xlabel="sparsity (%)", ylabel="test accuracy (%)")
    ax[2].set(title="(c) random (dashed) vs magnitude+recal (solid)", xlabel="sparsity (%)", ylabel="test accuracy (%)")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="figures/analysis")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sweeps = load_sweeps(a.runs)
    df = paired_table(sweeps)
    df.to_csv(os.path.join(a.out, "probe_tests.csv"), index=False)
    shown = df[df.sparsity.isin([0.8, 0.9, 0.95])].round(3)
    with open(os.path.join(a.out, "probe_tests.md"), "w") as f:
        f.write(shown.to_markdown(index=False) + "\n")
    print(shown.to_string(index=False))
    for where in ["head", "block"]:
        if any(k[0] == where and k[1] > 0 for k in sweeps):
            probe_figure(sweeps, where, os.path.join(a.out, f"probe_{where}.png"))
    print("written to", a.out)


if __name__ == "__main__":
    main()
