"""Stage 2: post-training pruning probe (no fine-tuning) + weight / layer analysis.
Usage: python prune_sweep.py --run runs/p0.3_s0
Writes <run>/prune_sweep.json with:
  magnitude: Acc(s) for global magnitude pruning, with and without BN recalibration
  random:    Acc(s) for random unstructured pruning (baseline B2)
  weights:   |w| histogram, near-zero fractions, per-layer sparsity implied by the global threshold
  layerwise: test acc when pruning ONE layer at a time to 50/70/90%
"""
import argparse, copy, os
import torch
from dp.models import ResNet18CIFAR, prunable_params
from dp.data import load_cifar10
from dp.common import get_device, make_gen, evaluate, recalibrate_bn, save_json, load_json
from dp.pruning import global_magnitude_masks, random_masks, layer_masks, apply_masks, sparsity_stats

SPARSITIES = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]
LAYER_SPARSITIES = [0.5, 0.7, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n_train_eval", type=int, default=10000, help="train subset for train-loss/Kuhn prunability")
    ap.add_argument("--skip_layerwise", action="store_true")
    args = ap.parse_args()

    device = get_device(args.device)
    log = load_json(os.path.join(args.run, "train_log.json"))
    p = log["args"]["p"]; seed = log["args"]["seed"]
    data = load_cifar10(args.data, device)
    xtr, ytr = data["train"]; xtr, ytr = xtr[:args.n_train_eval], ytr[:args.n_train_eval]
    xte, yte = data["test"]

    base = ResNet18CIFAR(10, p).to(device)
    base.load_state_dict(torch.load(os.path.join(args.run, "model.pt"), map_location=device))
    base.eval()
    dense_loss, dense_acc = evaluate(base, xte, yte)
    dense_trl, dense_tra = evaluate(base, xtr, ytr)
    print(f"[{os.path.basename(args.run)}] dense test_acc {dense_acc:.2f}")
    res = {"p": p, "seed": seed, "dense": {"test_acc": dense_acc, "test_loss": dense_loss,
                                            "train_acc": dense_tra, "train_loss": dense_trl},
           "sparsities": SPARSITIES, "magnitude": [], "magnitude_bnrecal": [], "random": []}

    def probe(masks, recal):
        m = copy.deepcopy(base)
        apply_masks(m, masks)
        if recal:
            recalibrate_bn(m, xtr, ytr)
        st = sparsity_stats(m)
        tel, tea = evaluate(m, xte, yte)
        trl, tra = evaluate(m, xtr, ytr)
        return {"sparsity_target": None, "sparsity_eff": st["sparsity"], "nonzero": st["nonzero"],
                "test_acc": tea, "test_loss": tel, "train_acc": tra, "train_loss": trl,
                "retained": tea / dense_acc, "per_layer_sparsity": {k: v["sparsity"] for k, v in st["per_layer"].items()}}

    gen = make_gen(device, 1000 + seed)
    for s in SPARSITIES:
        mk = global_magnitude_masks(base, s)
        r = probe(mk, False); r["sparsity_target"] = s; res["magnitude"].append(r)
        r2 = probe(mk, True); r2["sparsity_target"] = s; res["magnitude_bnrecal"].append(r2)
        rr = probe(random_masks(base, s, gen), False); rr["sparsity_target"] = s; res["random"].append(rr)
        print(f"  s={s:.2f} eff={r['sparsity_eff']:.3f}  mag {r['test_acc']:6.2f}  mag+bn {r2['test_acc']:6.2f}  rand {rr['test_acc']:6.2f}", flush=True)

    # summary statistics of the magnitude curve
    def summarize(curve):
        accs = [c["test_acc"] for c in curve]; ss = [c["sparsity_target"] for c in curve]
        within = [s for s, a in zip(ss, accs) if a >= dense_acc - 1.0]
        s_star = max(within) if within else 0.0
        auc = sum((ss[i+1]-ss[i]) * (accs[i+1]+accs[i]) / 2 for i in range(len(ss)-1)) / (ss[-1]-ss[0]) / dense_acc
        # Kuhn-style prunability: largest sparsity whose train loss stays within 10% of dense train loss
        ok = [s for s, c in zip(ss, curve) if c["train_loss"] <= dense_trl * 1.10 + 1e-3]
        return {"s_star": s_star, "auc_retained": auc, "kuhn_prunability": max(ok) if ok else 0.0}
    res["summary"] = {"magnitude": summarize(res["magnitude"]), "magnitude_bnrecal": summarize(res["magnitude_bnrecal"]),
                      "random": summarize(res["random"])}
    print("  summary:", res["summary"]["magnitude"])

    # weight-distribution analysis
    allw = torch.cat([w.detach().abs().flatten().float().cpu() for _, w in prunable_params(base)])
    edges = torch.logspace(-7, 0.5, 60)
    hist = torch.histogram(allw, bins=edges).hist
    res["weights"] = {"log_bin_edges": edges.tolist(), "hist": hist.tolist(),
                      "mean_abs": allw.mean().item(), "median_abs": allw.median().item(),
                      "frac_below": {str(t): (allw < t).float().mean().item() for t in [1e-4, 1e-3, 1e-2, 5e-2]},
                      "per_layer": {n: {"numel": w.numel(), "mean_abs": w.detach().abs().mean().item(),
                                        "std": w.detach().std().item()} for n, w in prunable_params(base)}}

    # layer-wise sensitivity
    if not args.skip_layerwise:
        res["layerwise"] = {}
        for n, w in prunable_params(base):
            res["layerwise"][n] = {}
            for s in LAYER_SPARSITIES:
                m = copy.deepcopy(base); apply_masks(m, layer_masks(base, n, s))
                res["layerwise"][n][str(s)] = evaluate(m, xte, yte)[1]
            print(f"  layer {n:28s} numel={w.numel():8d} " + " ".join(f"s{s}:{res['layerwise'][n][str(s)]:6.2f}" for s in LAYER_SPARSITIES), flush=True)

    save_json(res, os.path.join(args.run, "prune_sweep.json"))


if __name__ == "__main__":
    main()
