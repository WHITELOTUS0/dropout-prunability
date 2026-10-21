import json, os, random, time
import numpy as np
import torch
from .data import iterate, normalize


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_gen(device, seed):
    g = torch.Generator(device=device.type)
    g.manual_seed(seed)
    return g


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    """Returns (mean cross-entropy loss, accuracy in %). Dropout disabled (eval mode)."""
    model.eval()
    loss_sum, correct = 0.0, 0
    for xb, yb in iterate(x, y, batch_size, shuffle=False):
        out = model(xb)
        loss_sum += torch.nn.functional.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    n = x.shape[0]
    return loss_sum / n, 100.0 * correct / n


@torch.no_grad()
def recalibrate_bn(model, x, y, batch_size=256, n_batches=40):
    """Re-estimate BatchNorm running stats on (unaugmented) training data after pruning."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats(); m.momentum = None  # cumulative average
    model.train()
    for i, (xb, _) in enumerate(iterate(x, y, batch_size, shuffle=False)):
        if i >= n_batches: break
        model(xb)
    model.eval()


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    with open(path) as f:
        return json.load(f)
