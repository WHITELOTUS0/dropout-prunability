"""Extension E1: one-shot vs iterative magnitude pruning at a matched 20-epoch fine-tuning budget.
  one-shot : prune to s, fine-tune 20 epochs
  iterative: 4 equal cumulative steps to s (e.g. 20->40->60->80), 5 fine-tune epochs each
Usage: python finetune.py --run runs/p0.3_s0 --sparsity 0.9 --schedule iterative
Writes <run>/ft_{schedule}_s{sparsity}.json"""
import argparse, copy, math, os, time
import torch
import torch.nn.functional as F
from dp.models import ResNet18CIFAR
from dp.data import load_cifar10, iterate
from dp.common import get_device, set_seed, make_gen, evaluate, save_json, load_json
from dp.pruning import global_magnitude_masks, apply_masks, mask_grads, assert_sparsity, sparsity_stats


def finetune(model, masks, xtr, ytr, xval, yval, epochs, lr, wd, bs, device, gen, use_amp, log):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    steps = math.ceil(xtr.shape[0] / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for ep in range(epochs):
        model.train(); tl, n = 0.0, 0; t0 = time.time()
        for xb, yb in iterate(xtr, ytr, bs, shuffle=True, gen=gen, train=True):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); mask_grads(model, masks)
            scaler.step(opt); scaler.update(); sched.step()
            apply_masks(model, masks)  # pruned weights stay exactly zero
            tl += loss.item() * yb.shape[0]; n += yb.shape[0]
        vl, va = evaluate(model, xval, yval)
        log.append({"train_loss": tl / n, "val_loss": vl, "val_acc": va, "sparsity": sparsity_stats(model)["sparsity"],
                    "time": time.time() - t0})
        print(f"    ft ep {ep+1}/{epochs} loss {tl/n:.3f} val_acc {va:.2f} sparsity {log[-1]['sparsity']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sparsity", type=float, required=True)
    ap.add_argument("--schedule", choices=["oneshot", "iterative"], required=True)
    ap.add_argument("--ft_epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--subset", type=int, default=0)
    args = ap.parse_args()

    device = get_device(args.device)
    tlog = load_json(os.path.join(args.run, "train_log.json"))
    p, seed = tlog["args"]["p"], tlog["args"]["seed"]
    set_seed(seed + 7)
    data = load_cifar10(args.data, device)
    xtr, ytr = data["train"]
    if args.subset: xtr, ytr = xtr[:args.subset], ytr[:args.subset]
    xval, yval = data["val"]; xte, yte = data["test"]
    use_amp = device.type == "cuda"
    gen = make_gen(device, seed + 7)

    model = ResNet18CIFAR(10, p).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run, "model.pt"), map_location=device))
    dense_acc = evaluate(model, xte, yte)[1]
    res = {"p": p, "seed": seed, "sparsity": args.sparsity, "schedule": args.schedule, "ft_epochs": args.ft_epochs,
           "dense_test_acc": dense_acc, "stages": []}
    print(f"[{os.path.basename(args.run)}] {args.schedule} -> {args.sparsity}  dense {dense_acc:.2f}")

    if args.schedule == "oneshot":
        targets, ep_per = [args.sparsity], [args.ft_epochs]
    else:
        targets = [args.sparsity * (i + 1) / args.steps for i in range(args.steps)]
        ep_per = [args.ft_epochs // args.steps] * args.steps
        ep_per[-1] += args.ft_epochs - sum(ep_per)
    masks = None
    for s, e in zip(targets, ep_per):
        masks = global_magnitude_masks(model, s, prev_masks=masks)
        apply_masks(model, masks); assert_sparsity(model, s)
        before = evaluate(model, xte, yte)[1]
        log = []
        finetune(model, masks, xtr, ytr, xval, yval, e, args.lr, args.wd, args.bs, device, gen, use_amp, log)
        assert_sparsity(model, s)
        after_l, after = evaluate(model, xte, yte)
        res["stages"].append({"sparsity": s, "ft_epochs": e, "test_acc_before_ft": before,
                              "test_acc_after_ft": after, "test_loss_after_ft": after_l, "epochs": log})
        print(f"  stage s={s:.2f}: before ft {before:.2f} -> after ft {after:.2f}", flush=True)
    res["final_test_acc"] = res["stages"][-1]["test_acc_after_ft"]
    res["final_sparsity"] = sparsity_stats(model)["sparsity"]
    save_json(res, os.path.join(args.run, f"ft_{args.schedule}_s{args.sparsity}.json"))


if __name__ == "__main__":
    main()
