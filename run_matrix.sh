#!/usr/bin/env bash
# Run the dense-training matrix, then the pruning sweeps.
# Usage: bash run_matrix.sh "0.0 0.1 0.3 0.5" "0" 100 [GPU_ID]
# On a 2-GPU Kaggle session launch two copies with different GPU_IDs (e.g. p in "0.0 0.1" on GPU 0, "0.3 0.5" on GPU 1).
set -e
PS=${1:-"0.0 0.1 0.3 0.5"}; SEEDS=${2:-"0"}; EPOCHS=${3:-100}; GPU=${4:-0}; WHERE=${WHERE:-head}
SUF=""; [ "$WHERE" = "block" ] && SUF="blk"
export CUDA_VISIBLE_DEVICES=$GPU
for seed in $SEEDS; do for p in $PS; do
  if [ -f runs/p${p}${SUF}_s${seed}/train_log.json ] && grep -q '"final"' runs/p${p}${SUF}_s${seed}/train_log.json; then
    echo "skip p=$p seed=$seed (done)"; else
    python train.py --p $p --seed $seed --epochs $EPOCHS --where $WHERE --out runs 2>&1 | tee runs/train_p${p}${SUF}_s${seed}.log
  fi
  [ -f runs/p${p}${SUF}_s${seed}/prune_sweep.json ] || python prune_sweep.py --run runs/p${p}${SUF}_s${seed} 2>&1 | tee runs/prune_p${p}${SUF}_s${seed}.log
done; done
