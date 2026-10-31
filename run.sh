#!/usr/bin/env bash
# ObjectJEPA: full pipeline (download, frozen V-JEPA 2 features, predictor training).
# Usage: bash run.sh [NUM_VIDEOS]   (default 500 for a smoke run; 0 = all 10k)
# Optional: CONDA_ENV=myenv bash run.sh  activates that conda env first
#           NPROC=k bash run.sh          k parallel processes (default: one per visible CUDA device)
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
NPROC=${NPROC:-$(python -c "import torch; print(max(1, torch.cuda.device_count()))")}

# 1. CLEVRER train videos (~10k clips), downloaded once
if [ ! -d "$VIDEOS" ]; then
  mkdir -p "$DATA/clevrer"
  wget -c -O "$DATA/clevrer/video_train.zip" http://data.csail.mit.edu/clevrer/videos/train/video_train.zip
  unzip -q "$DATA/clevrer/video_train.zip" -d "$VIDEOS"
fi

# 2. Frozen V-JEPA 2 features, one process per device (each takes 1/NPROC of the videos)
for g in $(seq 0 $((NPROC - 1))); do
  CUDA_VISIBLE_DEVICES=$g python extract_features.py --videos "$VIDEOS" --out "$FEATS" \
    --limit "$LIMIT" --shard $g --num_shards "$NPROC" > "extract_$g.log" 2>&1 &
done
wait
echo "features: $(ls "$FEATS" | wc -l) videos"

# 3. Predictor training with DDP, one process per device (--bs is per process)
torchrun --nproc_per_node="$NPROC" train_predictor.py --features "$FEATS" --out "$CKPT" \
  --epochs 30 --bs 64 --width 384 --depth 6 2>&1 | tee train.log
