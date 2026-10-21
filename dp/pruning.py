"""Global unstructured magnitude pruning with explicit, persistent masks.
Masks are plain tensors so they can be asserted after every step and re-applied during fine-tuning."""
import torch
from .models import prunable_params


def global_magnitude_masks(model, sparsity, prev_masks=None):
    """Mask that keeps the largest-|w| entries globally so that `sparsity` of all prunable weights are zero.
    Previously pruned entries (prev_masks==0) are treated as magnitude 0, so sparsity is cumulative."""
    params = prunable_params(model)
    if sparsity <= 0:
        return {n: torch.ones_like(w, dtype=torch.bool) for n, w in params}
    mags = []
    for n, w in params:
        m = w.detach().abs().flatten()
        if prev_masks is not None:
            m = m * prev_masks[n].flatten()
        mags.append(m)
    allm = torch.cat(mags)
    k = int(round(sparsity * allm.numel()))
    thr = torch.kthvalue(allm.float().cpu(), k).values.item() if k > 0 else -1.0
    masks = {}
    for n, w in params:
        mag = w.detach().abs()
        if prev_masks is not None:
            mag = mag * prev_masks[n]
        masks[n] = mag > thr
    return masks


def random_masks(model, sparsity, gen):
    params = prunable_params(model)
    masks = {}
    for n, w in params:
        r = torch.rand(w.shape, device=w.device, generator=gen)
        masks[n] = r >= sparsity
    return masks


def layer_masks(model, layer_name, sparsity):
    """Prune a single layer (by magnitude) to `sparsity`; all other layers untouched."""
    masks = {}
    for n, w in prunable_params(model):
        if n == layer_name:
            k = int(round(sparsity * w.numel()))
            thr = torch.kthvalue(w.detach().abs().flatten().float().cpu(), k).values.item() if k > 0 else -1
            masks[n] = w.detach().abs() > thr
        else:
            masks[n] = torch.ones_like(w, dtype=torch.bool)
    return masks


@torch.no_grad()
def apply_masks(model, masks):
    for n, w in prunable_params(model):
        w.mul_(masks[n].to(w.dtype))


def mask_grads(model, masks):
    for n, w in prunable_params(model):
        if w.grad is not None:
            w.grad.mul_(masks[n].to(w.grad.dtype))


@torch.no_grad()
def sparsity_stats(model):
    tot, nz = 0, 0
    per_layer = {}
    for n, w in prunable_params(model):
        c = int((w != 0).sum().item()); t = w.numel()
        per_layer[n] = {"numel": t, "nonzero": c, "sparsity": 1 - c / t}
        tot += t; nz += c
    return {"total": tot, "nonzero": nz, "sparsity": 1 - nz / tot, "per_layer": per_layer}


def assert_sparsity(model, target, tol=0.005):
    s = sparsity_stats(model)["sparsity"]
    assert abs(s - target) <= tol, f"effective sparsity {s:.4f} != target {target:.4f}"
    return s
