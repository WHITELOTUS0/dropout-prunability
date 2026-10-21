# Does Regularization Shape Prunability?
Empirical study of dropout rate vs. post-training magnitude-pruning tolerance (ResNet-18, CIFAR-10).
11-785 Introduction to Deep Learning, CMU Africa. Team: Glorry Sibomana, Alia Teto, Ethel Kwagalakwe, Anny Christelle Irakoze.

## Pipeline
| Stage | Script | Output |
|---|---|---|
| 1. Dense training, ResNet-18 + `nn.Dropout(p)` before the classifier | `train.py --p P --seed S --epochs 100` | `runs/pP_sS/model.pt`, `train_log.json` |
| 2. Pruning probe (global magnitude, random baseline, BN-recal, weight stats, layer-wise sensitivity) | `prune_sweep.py --run runs/pP_sS` | `runs/pP_sS/prune_sweep.json` |
| E1. One-shot vs iterative pruning, matched 20-epoch fine-tuning | `finetune.py --run runs/pP_sS --sparsity 0.9 --schedule iterative` | `runs/pP_sS/ft_*.json` |
| 3. Figures and tables | `plots.py --runs runs --out figures` | `figures/*.png`, `summary.csv`, `*.md` tables |

`run_matrix.sh "0.0 0.1 0.3 0.5" "0 1 2" 100 GPU_ID` runs stages 1+2 for a grid and skips finished runs (resumable across Kaggle sessions).

## Setup
Training: SGD momentum 0.9, lr 0.1 cosine-annealed per step, weight decay 5e-4, batch 128, 100 epochs, cross-entropy,
random crop (4-px pad) + horizontal flip, 45k/5k train/val split (fixed split seed), AMP on CUDA.
Data lives on the accelerator and augmentation is batched on-device, so no dataloader workers are needed.
Pruning: global unstructured magnitude pruning over all Conv2d/Linear weights (BatchNorm and biases untouched),
sparsities {0,20,40,60,70,80,90,95}%, explicit boolean masks re-applied after every optimizer step; effective sparsity is asserted.

## Running on Kaggle
1. Code: https://github.com/WHITELOTUS0/dropout-prunability (already set in `kaggle_notebook.ipynb`). Attach your CIFAR-10 tarball dataset and the previous notebook version as Inputs: the first cell copies the tarball and restores trained models so nothing is retrained.
2. New Kaggle notebook → File → Import notebook → upload `kaggle_notebook.ipynb`.
3. Settings: Accelerator **GPU T4 x2**, Internet **On**.
4. Run the pilot cell first (1 epoch, ~1 min) then the matrix cell. Use *Save & Run All (Commit)* for long runs so `/kaggle/working` is preserved;
   download `results_*.tar.gz` from the Output tab and unpack it into `runs/` locally, then run `plots.py`.

## Local smoke test (Apple MPS / CPU)
```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch torchvision numpy pandas matplotlib tabulate
.venv/bin/python train.py --p 0.3 --seed 0 --epochs 2 --subset 5000 --out runs_smoke
.venv/bin/python prune_sweep.py --run runs_smoke/p0.3_s0 --skip_layerwise
.venv/bin/python plots.py --runs runs_smoke --out figures_smoke
```
