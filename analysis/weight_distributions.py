"""Weight-distribution analysis (H2): does the dropout rate shift the learned weight magnitudes?

Reads the `weights` block of every runs/p*_s*/prune_sweep.json (log-binned |w| histogram, mean and
median |w|, fraction of weights below thresholds) and produces, per placement and dropout rate,
a seed-mean table and an overlay figure of the histograms with head dropout dotted and block
dropout solid.

Usage: python analysis/weight_distributions.py --runs runs --out figures/analysis
Writes weight_stats.csv, weight_stats.md, weight_hist.png
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {0.0: "#1f6feb", 0.1: "#e0712e", 0.3: "#2a9d5c", 0.5: "#9b4fd1"}


def load_weights(runs_dir):
    """Return list of dicts with placement, p, seed and the sweep's weights block."""
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "p*_s*"))):
        f = os.path.join(d, "prune_sweep.json")
        if not os.path.isdir(d) or not os.path.exists(f):
            continue
        sw = json.load(open(f))
        where = "block" if "blk" in os.path.basename(d) else "head"
        if where == "block" and float(sw["p"]) == 0.0:
            continue  # identical to the head p=0 model
        out.append(dict(placement="control" if float(sw["p"]) == 0.0 else where, p=float(sw["p"]), seed=int(sw["seed"]), w=sw["weights"]))
    return out


def stats_table(recs):
    rows = []
    for r in recs:
        w = r["w"]
        rows.append(dict(placement=r["placement"], p=r["p"], seed=r["seed"], mean_abs=w["mean_abs"], median_abs=w["median_abs"],
                         frac_below_1e_4=w["frac_below"]["0.0001"], frac_below_1e_3=w["frac_below"]["0.001"],
                         frac_below_1e_2=w["frac_below"]["0.01"]))
    df = pd.DataFrame(rows)
    g = df.groupby(["placement", "p"]).agg(n_seeds=("seed", "count"), mean_abs=("mean_abs", "mean"), mean_abs_sd=("mean_abs", "std"),
                                          median_abs=("median_abs", "mean"), frac_below_1e_3=("frac_below_1e_3", "mean"),
                                          frac_below_1e_2=("frac_below_1e_2", "mean")).reset_index()
    return df, g


def hist_figure(recs, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=200)
    edges = np.array(recs[0]["w"]["log_bin_edges"]); centers = np.sqrt(edges[:-1] * edges[1:])
    for placement, ls in [("control", ":"), ("head", ":"), ("block", "-")]:
        for p in sorted({r["p"] for r in recs if r["placement"] == placement}):
            hs = [np.array(r["w"]["hist"], dtype=float) for r in recs if r["placement"] == placement and r["p"] == p]
            h = np.mean(hs, 0); h = h / h.sum()
            ax.plot(centers, h, color=COLORS[p], ls=ls, lw=1.4, label="control" if placement == "control" else f"{placement} p={p}")
    ax.set(xscale="log", xlim=(1e-5, 1), xlabel="|w|", ylabel="fraction of weights", title="Weight magnitudes (seed mean)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="figures/analysis")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    recs = load_weights(a.runs)
    if not recs:
        print("no prune_sweep.json under", a.runs); return
    per_run, summary = stats_table(recs)
    per_run.to_csv(os.path.join(a.out, "weight_stats_per_run.csv"), index=False)
    summary.to_csv(os.path.join(a.out, "weight_stats.csv"), index=False)
    with open(os.path.join(a.out, "weight_stats.md"), "w") as f:
        f.write(summary.round(4).to_markdown(index=False) + "\n")
    print(summary.round(4).to_string(index=False))
    hist_figure(recs, os.path.join(a.out, "weight_hist.png"))
    print("written to", a.out)


if __name__ == "__main__":
    main()
