"""Schedule ablation (E1) statistics: one-shot vs iterative vs front-loaded at a matched 20-epoch budget.

Reads every runs/p*_s*/ft_{schedule}_s{sparsity}.json, builds the per-(p, seed, sparsity, schedule)
table, and for every sparsity where both schedules exist on the same (p, seed) pairs reports the
paired difference one-shot minus iterative with a one-sample t test. Also draws the fine-tuning
figure used in the report: (a) accuracy after fine-tuning per schedule, (b) seed-paired differences,
(c) block-dropout recovery relative to each model's own dense accuracy.

Usage: python analysis/schedule_tests.py --runs runs --out figures/analysis
Writes finetune_all.csv, schedule_paired.csv, schedule_paired.md, finetune.png
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
STYLE = {"oneshot": ("--", "s"), "iterative": ("-", "o"), "frontload": (":", "^")}


def load_finetune(runs_dir):
    rows = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "p*_s*"))):
        if not os.path.isdir(d):
            continue
        where = "block" if "blk" in os.path.basename(d) else "head"
        for f in glob.glob(os.path.join(d, "ft_*.json")):
            r = json.load(open(f))
            rows.append(dict(placement=where, p=float(r["p"]), seed=int(r["seed"]), sparsity=float(r["sparsity"]),
                             schedule=r["schedule"], ft_epochs=r["ft_epochs"], dense=r["dense_test_acc"],
                             before_ft=r["stages"][0]["test_acc_before_ft"], after_ft=r["final_test_acc"],
                             eff_sparsity=r["final_sparsity"],
                             step_epochs="+".join(str(s["ft_epochs"]) for s in r["stages"])))
    df = pd.DataFrame(rows)
    # the p=0 control is the same model for both placements; keep one copy under "head"
    df = df[~((df.placement == "block") & (df.p == 0.0))]
    return df.sort_values(["placement", "p", "sparsity", "schedule", "seed"]).reset_index(drop=True)


def paired_schedules(df, a="oneshot", b="iterative"):
    rows = []
    H = df[df.placement == "head"]
    for s in sorted(H.sparsity.unique()):
        diffs = []
        for p in PS:
            for seed in sorted(H.seed.unique()):
                x = H[(H.p == p) & (H.seed == seed) & (H.sparsity == s)]
                if (x.schedule == a).any() and (x.schedule == b).any():
                    diffs.append(x[x.schedule == a].after_ft.iloc[0] - x[x.schedule == b].after_ft.iloc[0])
        if len(diffs) >= 2:
            t = stats.ttest_1samp(diffs, 0.0)
            rows.append(dict(sparsity=s, comparison=f"{a} - {b}", n_pairs=len(diffs), mean_diff=np.mean(diffs),
                             sd_diff=np.std(diffs, ddof=1), t=t.statistic, p_value=t.pvalue,
                             n_positive=int(sum(d > 0 for d in diffs))))
    return pd.DataFrame(rows)


def finetune_figure(df, path):
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.9), dpi=200)
    H = df[df.placement == "head"]
    for p in PS:
        for sch, (ls, mk) in STYLE.items():
            g = H[(H.p == p) & (H.schedule == sch)].groupby("sparsity").after_ft.agg(["mean", "std"])
            if len(g):
                ax[0].errorbar(g.index * 100, g["mean"], yerr=g["std"].fillna(0), color=COLORS[p], ls=ls, marker=mk,
                               ms=3, lw=1.3, capsize=2, label=f"p={p} {sch}")
    ax[0].set(xlabel="sparsity (%)", ylabel="test acc after 20 ft epochs (%)", title="(a) head: one-shot / iterative / front-loaded")
    ax[0].legend(ncol=3, fontsize=5.5)
    for s in sorted(H.sparsity.unique()):
        for p in PS:
            for seed in sorted(H.seed.unique()):
                x = H[(H.p == p) & (H.seed == seed) & (H.sparsity == s)]
                if (x.schedule == "oneshot").any() and (x.schedule == "iterative").any():
                    ax[1].scatter(s * 100 + (p - 0.25) * 6, x[x.schedule == "oneshot"].after_ft.iloc[0] - x[x.schedule == "iterative"].after_ft.iloc[0],
                                  color=COLORS[p], s=12, label=f"p={p}" if seed == 0 and s == 0.9 else None)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xlabel="sparsity (%)", ylabel="one-shot minus iterative (pts)", title="(b) seed-paired schedule difference"); ax[1].legend(fontsize=7)
    B = df[(df.placement == "block") | (df.p == 0.0)]
    for p in PS:
        d = B[(B.p == p) & (B.schedule == "oneshot") & (B.sparsity.isin([0.8, 0.9, 0.95]))].copy()
        if not len(d):
            continue
        d["drop"] = d.after_ft - d.dense
        g = d.groupby("sparsity")["drop"].agg(["mean", "std"])
        ax[2].errorbar(g.index * 100, g["mean"], yerr=g["std"].fillna(0), color=COLORS[p], marker="o", ms=3, lw=1.5, capsize=2,
                       label="control" if p == 0 else f"block p={p}")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set(xlabel="sparsity (%)", ylabel="after ft minus dense (pts)", title="(c) block dropout: one-shot + 20 ft epochs"); ax[2].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="figures/analysis")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = load_finetune(a.runs)
    if df.empty:
        print("no ft_*.json files under", a.runs); return
    df.to_csv(os.path.join(a.out, "finetune_all.csv"), index=False)
    pairs = pd.concat([paired_schedules(df, "oneshot", "iterative"), paired_schedules(df, "oneshot", "frontload"),
                       paired_schedules(df, "frontload", "iterative")], ignore_index=True)
    pairs.to_csv(os.path.join(a.out, "schedule_paired.csv"), index=False)
    with open(os.path.join(a.out, "schedule_paired.md"), "w") as f:
        f.write(pairs.round(3).to_markdown(index=False) + "\n")
    print(pairs.round(3).to_string(index=False))
    finetune_figure(df, os.path.join(a.out, "finetune.png"))
    print("written to", a.out)


if __name__ == "__main__":
    main()
