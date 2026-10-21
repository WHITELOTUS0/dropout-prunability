"""Stage 1: dense training of ResNet-18 + Dropout(p) on CIFAR-10.
Usage: python train.py --p 0.3 --seed 0 --epochs 100 --out runs
Writes runs/p{p}_s{seed}/{model.pt, train_log.json}."""
import argparse, os, time, math
import torch
import torch.nn.functional as F
from dp.models import ResNet18CIFAR, count_params
from dp.data import load_cifar10, iterate
from dp.common import get_device, set_seed, make_gen, evaluate, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.0, help="dropout probability")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./runs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", type=int, default=1, help="mixed precision on CUDA")
    ap.add_argument("--subset", type=int, default=0, help="debug: use only N training images")
    args = ap.parse_args()

    device = get_device(args.device)
    set_seed(args.seed)
    tag = f"p{args.p}_s{args.seed}"
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{tag}] device={device} out={out_dir}")

    data = load_cifar10(args.data, device)
    xtr, ytr = data["train"]
    if args.subset:
        xtr, ytr = xtr[:args.subset], ytr[:args.subset]
    xval, yval = data["val"]; xte, yte = data["test"]

    model = ResNet18CIFAR(10, args.p).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(f"params: {count_params(model):,}")
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=False)
    steps_per_epoch = math.ceil(xtr.shape[0] / args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = make_gen(device, args.seed)

    log = {"args": vars(args), "params": count_params(model), "epochs": []}
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tl, tc, n = 0.0, 0, 0
        te0 = time.time()
        for xb, yb in iterate(xtr, ytr, args.bs, shuffle=True, gen=gen, train=True):
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(xb)
                loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * yb.shape[0]; tc += (out.argmax(1) == yb).sum().item(); n += yb.shape[0]
        if not math.isfinite(tl):
            raise RuntimeError("NaN/Inf training loss")
        vl, va = evaluate(model, xval, yval)
        rec = {"epoch": ep + 1, "train_loss": tl / n, "train_acc": 100 * tc / n, "val_loss": vl, "val_acc": va,
               "lr": sched.get_last_lr()[0], "epoch_time": time.time() - te0}
        log["epochs"].append(rec)
        print(f"ep {ep+1:3d}/{args.epochs} train_loss {rec['train_loss']:.3f} train_acc {rec['train_acc']:.2f} "
              f"val_loss {vl:.3f} val_acc {va:.2f} ({rec['epoch_time']:.1f}s)", flush=True)
        if (ep + 1) % 10 == 0 or ep + 1 == args.epochs:
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
            save_json(log, os.path.join(out_dir, "train_log.json"))

    # final dense metrics (eval mode, dropout off, no augmentation)
    trl, tra = evaluate(model, xtr, ytr)
    tel, tea = evaluate(model, xte, yte)
    log["final"] = {"train_loss": trl, "train_acc": tra, "val_loss": vl, "val_acc": va,
                    "test_loss": tel, "test_acc": tea, "gen_gap": tra - tea, "train_time_s": time.time() - t0}
    print(f"[{tag}] FINAL test_acc {tea:.2f} test_loss {tel:.3f} train_acc {tra:.2f} gap {tra-tea:.2f}")
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    save_json(log, os.path.join(out_dir, "train_log.json"))


if __name__ == "__main__":
    main()
