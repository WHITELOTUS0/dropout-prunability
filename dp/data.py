"""CIFAR-10 held entirely on the accelerator; augmentation (4-px pad random crop + h-flip) done on-device.
Avoids the CPU dataloader bottleneck on Kaggle (4 vCPUs) and on Apple MPS."""
import numpy as np
import torch
import torchvision

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
VAL_SIZE = 5000
SPLIT_SEED = 0  # fixed train/val split, independent of the training seed


def _to_tensor(ds):
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # uint8 N,3,32,32
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x, y


def load_cifar10(root, device):
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)
    x, y = _to_tensor(tr)
    xt, yt = _to_tensor(te)
    perm = torch.from_numpy(np.random.RandomState(SPLIT_SEED).permutation(len(x)))
    val_idx, tr_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]
    d = {
        "train": (x[tr_idx].to(device), y[tr_idx].to(device)),
        "val": (x[val_idx].to(device), y[val_idx].to(device)),
        "test": (xt.to(device), yt.to(device)),
    }
    return d


def normalize(x_uint8):
    x = x_uint8.float().div_(255.0)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def augment(x_uint8, gen):
    """Random crop with 4-px zero padding and random horizontal flip, batched on-device."""
    n = x_uint8.shape[0]
    dev = x_uint8.device
    padded = torch.nn.functional.pad(x_uint8, (4, 4, 4, 4))
    ox = torch.randint(0, 9, (n,), device=dev, generator=gen)
    oy = torch.randint(0, 9, (n,), device=dev, generator=gen)
    ar = torch.arange(32, device=dev)
    rows = (oy[:, None] + ar[None, :])  # n,32
    cols = (ox[:, None] + ar[None, :])
    out = padded[torch.arange(n, device=dev)[:, None, None, None],
                 torch.arange(3, device=dev)[None, :, None, None],
                 rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(n, device=dev, generator=gen) < 0.5
    out[flip] = out[flip].flip(3)
    return out


def iterate(x, y, batch_size, shuffle, gen=None, train=False):
    n = x.shape[0]
    idx = torch.randperm(n, device=x.device, generator=gen) if shuffle else torch.arange(n, device=x.device)
    for i in range(0, n, batch_size):
        b = idx[i:i + batch_size]
        xb = x[b]
        if train:
            xb = augment(xb, gen)
        yield normalize(xb), y[b]
