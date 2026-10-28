"""Stage 1: encode CLEVRER videos with a frozen V-JEPA 2 encoder.

For each video we take (context, target) clip pairs:
  context = 16 frames (every 2nd frame of a 32-frame window)
  target  = the next 16 frames, encoded SEPARATELY so the context
            features never see the future (no leakage).
Patch tokens are average-pooled to a small grid to keep files small.

Output: one .pt file per video with
  ctx: [S, T', P, D] float16   (S = clip pairs, T' = 8, P = pool*pool, D = 1024)
  tgt: [S, T', P, D] float16

Two GPUs: run one process per card, each takes half the videos:
  CUDA_VISIBLE_DEVICES=0 python extract_features.py ... --shard 0 --num_shards 2 &
  CUDA_VISIBLE_DEVICES=1 python extract_features.py ... --shard 1 --num_shards 2 &
"""
import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor


def read_frames(path):
    import av

    with av.open(path) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    return np.stack(frames)  # [T, H, W, 3]


def clip_pairs(frames, window=32, stride=2, starts=(0, 64)):
    pairs = []
    for s in starts:
        if s + 2 * window > len(frames):
            continue
        ctx = frames[s : s + window : stride]
        tgt = frames[s + window : s + 2 * window : stride]
        pairs.append((ctx, tgt))
    return pairs


@torch.no_grad()
def encode_tokens(model, processor, clips, device, frame="squash"):
    """Patch tokens on the full grid: [B, t, side, side, D] float32 on device."""
    videos = [torch.from_numpy(np.ascontiguousarray(c)).permute(0, 3, 1, 2) for c in clips]  # T, C, H, W
    # The default processor center-crops to 256x256, which drops ~40% of the width of a
    # 480x320 CLEVRER frame. "squash" resizes the whole frame to 256x256 instead (the
    # encoder's RoPE assumes a square grid, so a non-square input is not an option).
    kw = {"do_center_crop": False, "size": {"height": 256, "width": 256}} if frame == "squash" else {}
    inputs = processor(videos, return_tensors="pt", **kw).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model.get_vision_features(**inputs)
    h = out if torch.is_tensor(out) else out.last_hidden_state  # [B, N, D]
    B, N, D = h.shape
    t = len(clips[0]) // 2  # tubelet size 2
    side = int(math.isqrt(N // t))
    return h.float().reshape(B, t, side, side, D)


@torch.no_grad()
def encode(model, processor, clips, device, pool, frame="squash"):
    h = encode_tokens(model, processor, clips, device, frame)
    B, t, side, _, D = h.shape
    h = h.permute(0, 1, 4, 2, 3).reshape(B * t, D, side, side)
    h = F.adaptive_avg_pool2d(h, pool)  # [B*t, D, pool, pool]
    h = h.reshape(B, t, D, pool * pool).permute(0, 1, 3, 2)  # [B, t, P, D]
    return h.half().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder containing CLEVRER .mp4 files (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--limit", type=int, default=0, help="max videos (0 = all)")
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--frame", choices=["squash", "crop"], default="squash",
                    help="squash = whole frame resized to 256x256, crop = processor's center crop")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoVideoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    paths = sorted(glob.glob(os.path.join(args.videos, "**", "*.mp4"), recursive=True))
    if args.limit:
        paths = paths[: args.limit]
    paths = paths[args.shard :: args.num_shards]
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(paths)} videos -> {args.out}")

    for i, p in enumerate(paths):
        name = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(args.out, f"{name}.pt")
        if os.path.exists(dst):
            continue
        pairs = clip_pairs(read_frames(p))
        if not pairs:
            continue
        ctx = encode(model, processor, [c for c, _ in pairs], device, args.pool, args.frame)
        tgt = encode(model, processor, [t for _, t in pairs], device, args.pool, args.frame)
        torch.save({"ctx": ctx, "tgt": tgt}, dst)
        if i % 50 == 0:
            print(f"[{i}/{len(paths)}] {name} ctx={tuple(ctx.shape)}", flush=True)


if __name__ == "__main__":
    main()
