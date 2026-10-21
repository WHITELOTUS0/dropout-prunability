"""Stage 3: aggregate all runs/p*_s*/ into tables + figures for the report.
Usage: python plots.py --runs runs --out figures
Produces: summary.csv, dense_table.md, acc_vs_sparsity.png, retained_vs_sparsity.png, random_vs_magnitude.png,
          weight_hist.png, per_layer_sparsity.png, layerwise_sensitivity.png, training_curves.png, finetune.png"""
import argparse, glob, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dp.common import load_json

# fixed categorical order, one hue per dropout rate (validated palette, see dataviz/palette.md)
COLORS = {0.0: "#1f6feb", 0.1: "#e0712e", 0.3: "#2a9d5c", 0.5: "#9b4fd1"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "figure.dpi": 150})


def color(p):
    return COLORS.get(p, "#666666")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="figures")
    ap.add_argument("--where", default="head", choices=["head", "block"], help="which dropout placement to plot")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    runs = []
    for d in sorted(glob.glob(os.path.join(a.runs, "p*_s*"))):
        if not os.path.exists(os.path.join(d, "train_log.json")): continue
        tl = load_json(os.path.join(d, "train_log.json"))
        if tl["args"].get("where", "head") != a.where: continue
        ps = load_json(os.path.join(d, "prune_sweep.json")) if os.path.exists(os.path.join(d, "prune_sweep.json")) else None
        ft = [load_json(f) for f in glob.glob(os.path.join(d, "ft_*.json"))]
        runs.append({"dir": d, "p": tl["args"]["p"], "seed": tl["args"]["seed"], "train": tl, "sweep": ps, "ft": ft})
    if not runs:
        print("no runs found"); return
    ps_all = sorted({r["p"] for r in runs})

    # ---- dense table
    rows = []
    for r in runs:
        f = r["train"].get("final", {})
        rows.append({"p": r["p"], "seed": r["seed"], "epochs": len(r["train"]["epochs"]), **{k: f.get(k) for k in
                     ["train_acc", "val_acc", "test_acc", "test_loss", "gen_gap", "train_time_s"]}})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "dense_runs.csv"), index=False)
    g = df.groupby("p")[["train_acc", "test_acc", "test_loss", "gen_gap"]].agg(["mean", "std"]).round(2)
    with open(os.path.join(a.out, "dense_table.md"), "w") as f:
        f.write(df.round(3).to_markdown(index=False) + "\n\n" + g.to_markdown() + "\n")
    print(df.round(2).to_string(index=False))

    # ---- training curves
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for r in runs:
        e = r["train"]["epochs"]; x = [q["epoch"] for q in e]
        ax[0].plot(x, [q["train_loss"] for q in e], color=color(r["p"]), lw=1.5, alpha=0.8, label=f"p={r['p']}" if r["seed"] == min(q["seed"] for q in runs if q["p"] == r["p"]) else None)
        ax[1].plot(x, [q["val_acc"] for q in e], color=color(r["p"]), lw=1.5, alpha=0.8)
    ax[0].set(xlabel="epoch", ylabel="train loss", title="Training loss"); ax[1].set(xlabel="epoch", ylabel="val accuracy (%)", title="Validation accuracy")
    ax[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(a.out, "training_curves.png")); plt.close(fig)

    sw = [r for r in runs if r["sweep"]]
    if sw:
        S = sw[0]["sweep"]["sparsities"]
        def curve(r, key, field="test_acc"):
            return np.array([c[field] for c in r["sweep"][key]])
        def mean_std(p, key, field="test_acc"):
            arr = np.stack([curve(r, key, field) for r in sw if r["p"] == p]); return arr.mean(0), arr.std(0), arr.shape[0]

        # ---- Acc(s,p) and R(s,p), with and without BN recalibration
        for key, fname, title in [("magnitude", "acc_vs_sparsity.png", "Global magnitude pruning, no fine-tuning"),
                                  ("magnitude_bnrecal", "acc_vs_sparsity_bnrecal.png", "Magnitude pruning + BN recalibration")]:
            fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
            for p in ps_all:
                m, s, n = mean_std(p, key); rm, rs, _ = mean_std(p, key, "retained")
                ax[0].errorbar(np.array(S) * 100, m, yerr=s, color=color(p), lw=2, marker="o", ms=4, capsize=2, label=f"p={p} (n={n})")
                ax[1].errorbar(np.array(S) * 100, rm, yerr=rs, color=color(p), lw=2, marker="o", ms=4, capsize=2)
            ax[0].set(xlabel="sparsity (%)", ylabel="test accuracy (%)", title=title)
            ax[1].set(xlabel="sparsity (%)", ylabel="retained accuracy R(s)", title="Retained accuracy Acc(s)/Acc(0)")
            ax[1].axhline(1.0, color="#999", lw=1, ls="--"); ax[0].legend()
            fig.tight_layout(); fig.savefig(os.path.join(a.out, fname)); plt.close(fig)

        # ---- magnitude vs random (B2)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for p in ps_all:
            m, s, _ = mean_std(p, "magnitude"); rm, rs, _ = mean_std(p, "random")
            ax.plot(np.array(S) * 100, m, color=color(p), lw=2, marker="o", ms=4, label=f"p={p} magnitude")
            ax.plot(np.array(S) * 100, rm, color=color(p), lw=1.5, ls=":", marker="x", ms=4, label=f"p={p} random")
        ax.set(xlabel="sparsity (%)", ylabel="test accuracy (%)", title="Magnitude vs random pruning (no fine-tuning)")
        ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(os.path.join(a.out, "random_vs_magnitude.png")); plt.close(fig)

        # ---- summary csv (s*, AUC, prunability)
        srows = []
        for r in sw:
            for key in ["magnitude", "magnitude_bnrecal", "random"]:
                srows.append({"p": r["p"], "seed": r["seed"], "method": key, "dense_test_acc": r["sweep"]["dense"]["test_acc"],
                              **r["sweep"]["summary"][key],
                              **{f"acc@{int(s*100)}": c["test_acc"] for s, c in zip(S, r["sweep"][key])}})
        sdf = pd.DataFrame(srows); sdf.to_csv(os.path.join(a.out, "summary.csv"), index=False)
        agg = sdf[sdf.method == "magnitude"].groupby("p")[["dense_test_acc", "s_star", "auc_retained", "kuhn_prunability"] + [f"acc@{int(s*100)}" for s in S]].agg(["mean", "std"]).round(3)
        with open(os.path.join(a.out, "summary_table.md"), "w") as f:
            f.write(sdf.round(3).to_markdown(index=False) + "\n\n" + agg.to_markdown() + "\n")
        print(sdf[sdf.method == "magnitude"][["p", "seed", "dense_test_acc", "s_star", "auc_retained", "kuhn_prunability", "acc@80", "acc@90", "acc@95"]].round(3).to_string(index=False))

        # ---- weight histograms + near-zero fractions
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for p in ps_all:
            rr = [r for r in sw if r["p"] == p]
            edges = np.array(rr[0]["sweep"]["weights"]["log_bin_edges"]); centers = np.sqrt(edges[1:] * edges[:-1])
            h = np.mean([np.array(r["sweep"]["weights"]["hist"]) for r in rr], 0); h = h / h.sum()
            ax[0].plot(centers, h, color=color(p), lw=2, label=f"p={p}")
            fb = rr[0]["sweep"]["weights"]["frac_below"]; ts = sorted(fb, key=float)
            vals = np.mean([[r["sweep"]["weights"]["frac_below"][t] for t in ts] for r in rr], 0)
            ax[1].plot(range(len(ts)), vals * 100, color=color(p), lw=2, marker="o", label=f"p={p}")
        ax[0].set(xscale="log", xlabel="|w|", ylabel="fraction of weights", title="Weight-magnitude distribution"); ax[0].legend()
        ax[1].set(xticks=range(len(ts)), xticklabels=[f"<{t}" for t in ts], ylabel="% of weights", title="Near-zero weights")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "weight_hist.png")); plt.close(fig)

        # ---- per-layer sparsity implied by the global threshold (at 80% and 90%)
        layers = list(sw[0]["sweep"]["magnitude"][0]["per_layer_sparsity"].keys())
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
        for k, st in enumerate([0.8, 0.9]):
            i = S.index(st)
            for p in ps_all:
                v = np.mean([[r["sweep"]["magnitude"][i]["per_layer_sparsity"][l] for l in layers] for r in sw if r["p"] == p], 0)
                ax[k].plot(range(len(layers)), v * 100, color=color(p), lw=1.5, marker="o", ms=3, label=f"p={p}")
            ax[k].set(title=f"Per-layer sparsity at global {int(st*100)}%", xticks=range(len(layers)), ylabel="layer sparsity (%)" if k == 0 else None)
            ax[k].set_xticklabels([l.replace(".weight", "") for l in layers], rotation=90, fontsize=6)
        ax[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(a.out, "per_layer_sparsity.png")); plt.close(fig)

        # ---- layer-wise sensitivity
        if "layerwise" in sw[0]["sweep"]:
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
            for k, st in enumerate(["0.5", "0.7", "0.9"]):
                for p in ps_all:
                    v = np.mean([[r["sweep"]["layerwise"][l][st] for l in layers] for r in sw if r["p"] == p and "layerwise" in r["sweep"]], 0)
                    ax[k].plot(range(len(layers)), v, color=color(p), lw=1.5, marker="o", ms=3, label=f"p={p}")
                ax[k].set(title=f"One layer pruned to {int(float(st)*100)}%", xticks=range(len(layers)), ylabel="test acc (%)" if k == 0 else None)
                ax[k].set_xticklabels([l.replace(".weight", "") for l in layers], rotation=90, fontsize=6)
            ax[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(a.out, "layerwise_sensitivity.png")); plt.close(fig)

    # ---- E1 fine-tuning: one-shot vs iterative
    ftr = [(r["p"], r["seed"], f) for r in runs for f in r["ft"]]
    if ftr:
        rows = [{"p": p, "seed": sd, "sparsity": f["sparsity"], "schedule": f["schedule"], "dense": f["dense_test_acc"],
                 "before_ft": f["stages"][0]["test_acc_before_ft"], "after_ft": f["final_test_acc"], "eff_sparsity": f["final_sparsity"]} for p, sd, f in ftr]
        fdf = pd.DataFrame(rows).sort_values(["p", "sparsity", "schedule", "seed"]); fdf.to_csv(os.path.join(a.out, "finetune.csv"), index=False)
        with open(os.path.join(a.out, "finetune_table.md"), "w") as f: f.write(fdf.round(2).to_markdown(index=False) + "\n")
        print(fdf.round(2).to_string(index=False))
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for p in ps_all:
            for sch, ls, mk in [("oneshot", "--", "s"), ("iterative", "-", "o")]:
                d = fdf[(fdf.p == p) & (fdf.schedule == sch)].groupby("sparsity")["after_ft"].mean()
                if len(d): ax.plot(d.index * 100, d.values, color=color(p), ls=ls, marker=mk, ms=4, lw=1.8, label=f"p={p} {sch}")
        ax.set(xlabel="sparsity (%)", ylabel="test acc after 20 ft epochs (%)", title="One-shot vs iterative (matched budget)")
        ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(os.path.join(a.out, "finetune.png")); plt.close(fig)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
