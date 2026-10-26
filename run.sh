#!/usr/bin/env bash
# ObjectJEPA: full pipeline on 2 local GPUs.
# Usage: bash run.sh [NUM_VIDEOS]   (default 500 for a smoke run; 0 = all 10k)
# Optional: CONDA_ENV=myenv bash run.sh  activates that conda env first
if [ -n "${CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"  # before set -u: conda's hooks use unset vars
  conda activate "$CONDA_ENV"
fi
set -euo pipefail

LIMIT=${1:-500}
DATA=${DATA:-$HOME/objectjepa_data}
VIDEOS=$DATA/clevrer/train
FEATS=$DATA/features
CKPT=$DATA/checkpoints

# 1. CLEVRER train videos (~10k clips), downloaded once
if [ ! -d "$VIDEOS" ]; then
  mkdir -p "$DATA/clevrer"
  wget -c -O "$DATA/clevrer/video_train.zip" http://data.csail.mit.edu/clevrer/videos/train/video_train.zip
  unzip -q "$DATA/clevrer/video_train.zip" -d "$VIDEOS"
fi

# 2. Frozen V-JEPA 2 features, one process per GPU (each takes half the videos)
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g python extract_features.py --videos "$VIDEOS" --out "$FEATS" \
    --limit "$LIMIT" --shard $g --num_shards 2 > "extract_gpu$g.log" 2>&1 &
done
wait
echo "features: $(ls "$FEATS" | wc -l) videos"

# 3. Predictor training with DDP on both GPUs (--bs is per GPU)
torchrun --nproc_per_node=2 train_predictor.py --features "$FEATS" --out "$CKPT" \
  --epochs 30 --bs 64 --width 384 --depth 6 2>&1 | tee train.log
