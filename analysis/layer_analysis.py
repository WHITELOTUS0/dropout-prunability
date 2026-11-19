"""Layer-wise analysis: where pruning fails and where a global threshold spends its budget.

Three per-layer quantities from runs/p*_s*/prune_sweep.json, averaged over seeds:
  1. sensitivity: accuracy lost when a single layer is pruned to 50/70/90% with every other layer intact
  2. budget: the sparsity each layer receives under one global magnitude threshold at a given target
  3. scale: the ratio of each layer's mean |w| to the control's, which explains 2
and a two-panel figure (sensitivity at 90%, per-layer sparsity at a global 80%) comparing the control
with the block-dropout models.

Usage: python analysis/layer_analysis.py --runs runs --out figures/analysis [--target 0.8] [--compare block:0.3 block:0.5]
Writes layer_sensitivity.csv, layer_budget.csv, layer_scale.csv, layers.png
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {0.0: "#1f6feb", 0.1: "#e0712e", 0.3: "#2a9d5c", 0.5: "#9b4fd1"}


def load_sweeps(runs_dir):
    out = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "p*_s*"))):
        f = os.path.join(d, "prune_sweep.json")
        if not os.path.isdir(d) or not os.path.exists(f):
            continue
        sw = json.load(open(f))
        where = "block" if "blk" in os.path.basename(d) else "head"
        if float(sw["p"]) == 0.0:
            where = "control"
        out.setdefault((where, float(sw["p"])), []).append(sw)
    return out


def layers_of(sw):
    return list(sw["weights"]["per_layer"].keys())


def sensitivity(sweeps):
    rows = []
    for (where, p), sws in sweeps.items():
        for layer in layers_of(sws[0]):
            for s in ["0.5", "0.7", "0.9"]:
                drops = [sw["dense"]["test_acc"] - sw["layerwise"][layer][s] for sw in sws if layer in sw.get("layerwise", {})]
                if drops:
                    rows.append(dict(placement=where, p=p, layer=layer.replace(".weight", ""), layer_sparsity=float(s),
                                     n_seeds=len(drops), acc_drop=np.mean(drops), acc_drop_sd=np.std(drops, ddof=1) if len(drops) > 1 else 0.0))
    return pd.DataFrame(rows)


def budget(sweeps, target):
    rows = []
    for (where, p), sws in sweeps.items():
        i = sws[0]["sparsities"].index(target)
        for layer in layers_of(sws[0]):
            v = [sw["magnitude"][i]["per_layer_sparsity"][layer] for sw in sws]
            rows.append(dict(placement=where, p=p, layer=layer.replace(".weight", ""), numel=sws[0]["weights"]["per_layer"][layer]["numel"],
                             global_target=target, layer_sparsity=np.mean(v)))
    return pd.DataFrame(rows)


def scale(sweeps):
    ctrl = sweeps[("control", 0.0)]
    rows = []
    for (where, p), sws in sweeps.items():
        for layer in layers_of(sws[0]):
            m = np.mean([sw["weights"]["per_layer"][layer]["mean_abs"] for sw in sws])
            m0 = np.mean([sw["weights"]["per_layer"][layer]["mean_abs"] for sw in ctrl])
            rows.append(dict(placement=where, p=p, layer=layer.replace(".weight", ""), mean_abs=m, ratio_to_control=m / m0))
    return pd.DataFrame(rows)


def figure(sens, bud, compare, target, path):
    layers = list(bud[(bud.placement == "control")].layer)
    short = [l.replace("shortcut.0", "sc") for l in layers]
    conds = [("control", 0.0)] + [(c.split(":")[0], float(c.split(":")[1])) for c in compare]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.0), dpi=200)
    x = np.arange(len(layers)); w = 0.8 / len(conds)
    for k, (where, p) in enumerate(conds):
        lab = "control" if where == "control" else f"{where} p={p}"
        d = sens[(sens.placement == where) & (sens.p == p) & (sens.layer_sparsity == 0.9)].set_index("layer").reindex(layers)
        ax[0].bar(x + (k - (len(conds) - 1) / 2) * w, d.acc_drop, w, color=COLORS[p], label=lab)
        b = bud[(bud.placement == where) & (bud.p == p)].set_index("layer").reindex(layers)
        ax[1].bar(x + (k - (len(conds) - 1) / 2) * w, b.layer_sparsity * 100, w, color=COLORS[p], label=lab)
    ax[0].set(ylabel="accuracy drop (pts)", title="(a) sensitivity: each layer pruned alone to 90%")
    ax[1].set(ylabel="layer sparsity (%)", title=f"(b) per-layer sparsity under a global {int(target*100)}% threshold")
    ax[1].axhline(target * 100, color="k", lw=0.7, ls="--")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(short, rotation=70, fontsize=6); a.grid(alpha=0.3, axis="y"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="figures/analysis")
    ap.add_argument("--target", type=float, default=0.8, help="global sparsity for the per-layer budget panel")
    ap.add_argument("--compare", nargs="*", default=["block:0.3", "block:0.5"], help="conditions to draw next to the control, as placement:p")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sweeps = load_sweeps(a.runs)
    if ("control", 0.0) not in sweeps:
        print("no control sweep under", a.runs); return
    sens, bud, sc = sensitivity(sweeps), budget(sweeps, a.target), scale(sweeps)
    sens.to_csv(os.path.join(a.out, "layer_sensitivity.csv"), index=False)
    bud.to_csv(os.path.join(a.out, "layer_budget.csv"), index=False)
    sc.to_csv(os.path.join(a.out, "layer_scale.csv"), index=False)
    top = sens[sens.layer_sparsity == 0.9].sort_values("acc_drop", ascending=False).groupby(["placement", "p"]).head(4)
    print("most sensitive layers at 90% (per condition):"); print(top.round(2).to_string(index=False))
    compare = [c for c in a.compare if (c.split(":")[0], float(c.split(":")[1])) in sweeps]
    figure(sens, bud, compare, a.target, os.path.join(a.out, "layers.png"))
    print("written to", a.out)


if __name__ == "__main__":
    main()
