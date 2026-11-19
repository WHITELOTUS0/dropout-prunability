# Model card: ResNet-18 (CIFAR variant) with dropout

## Architecture

Input: one 32 x 32 x 3 image, normalized per channel. Output: 10 logits.

| stage | layers | output shape (C x H x W) | weights |
|---|---|---|---|
| stem | 3x3 conv 3 -> 64, BatchNorm, ReLU | 64 x 32 x 32 | 1,728 |
| stage 1 | 2 BasicBlocks, 64 channels | 64 x 32 x 32 | 147,456 |
| stage 2 | 2 BasicBlocks, 128 channels, stride 2, 1x1 shortcut conv | 128 x 16 x 16 | 524,288 |
| stage 3 | 2 BasicBlocks, 256 channels, stride 2, 1x1 shortcut conv | 256 x 8 x 8 | 2,097,152 |
| stage 4 | 2 BasicBlocks, 512 channels, stride 2, 1x1 shortcut conv | 512 x 4 x 4 | 8,388,608 |
| head | global average pool, Dropout(p), linear 512 -> 10 | 10 | 5,120 |

Prunable weights (21 conv and linear tensors): 11,164,352. BatchNorm scale and shift parameters
(9,600) and the classifier bias (10) are never pruned; total 11,173,962 parameters.

A BasicBlock is: conv 3x3, BatchNorm, ReLU, [Dropout2d(p), extension E2 only], conv 3x3, BatchNorm,
add the shortcut, ReLU. Stage 4 holds 75% of all weights in four 2.36M-weight convolutions, which is why
global magnitude pruning removes mostly stage-4 weights up to 70% sparsity.

## The two dropout placements

- **Head dropout** (`--where head`, main study): `nn.Dropout(p)` between the global average pool and the
  classifier. It perturbs only the 512-dimensional input of the 5,130-parameter classifier, under 0.1% of the
  network. The backbone receives no dropout noise directly.
- **Block dropout** (`--where block`, extension E2): `nn.Dropout2d(p)` after the first conv/BN/ReLU of every
  BasicBlock, dropping whole channels of the intermediate feature map. All 16 blocks are perturbed; the stem
  convolution is the only conv that is never followed by dropout.

Dropout is active only in training mode. `Dropout(0)` and `Dropout2d(0)` are both the identity, so the p=0
model is the same control for both placements and is trained once (`runs/p0.0_s{seed}`); the Kaggle runner
copies it to `runs/p0.0blk_s{seed}` so the block plots have a control at every seed.

## Why E2 was planned as a fallback and became the main result

The proposal expected head dropout to be a weak intervention and listed block dropout as the fallback if
head dropout showed no effect. Head dropout did turn out to leave the weight distribution unchanged
(mean |w| 0.0045 to 0.0047 for every p). Block dropout changes it, and in the direction opposite to H2:
mean |w| rises to 0.0054, 0.0063 and 0.0066 at p = 0.1, 0.3 and 0.5, mostly in the large stage-4 layers
(1.5x to 2x the control at p = 0.5) while the stage-1 and stage-2 convolutions shrink slightly (0.82x to 0.92x).
A single global magnitude threshold therefore removes fewer stage-4 weights and more early-layer weights,
and the early layers are the ones that the single-layer sensitivity analysis shows to be fragile.
See `analysis/weight_distributions.py` and `analysis/layer_analysis.py` for the numbers.

## Training recipe (identical for every run)

SGD, momentum 0.9, learning rate 0.1 with per-step cosine annealing, weight decay 5e-4, batch 128,
100 epochs, mixed precision on CUDA, cross-entropy loss. Augmentation: random crop with 4-pixel padding
and horizontal flip, applied on the GPU. Fixed 45,000 / 5,000 train / validation split shared by all runs.
The seed controls initialization, data order and augmentation.
