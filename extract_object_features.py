"""Stage 3: per-object V-JEPA 2 features (object-centric latents).

Same clip pairs and same frozen encoder as extract_features.py (context and
target still encoded in SEPARATE passes), but instead of pooling patch tokens
to a fixed grid, each object's tokens are averaged inside its mask:

  feat[k, obj] = sum_p w[k, obj, p] * token[k, p] / sum_p w[k, obj, p]

where w is the fraction of patch p covered by the object's mask, averaged over
the 2 frames of tubelet step k.

Mask sources (--masks):
  clevrer : CLEVRER "derender proposals" (per-frame RLE masks with predicted
            color/material/shape). Masks are linked to the annotation's objects
            by their attribute triple, which is unique within a CLEVRER video.
  sam     : SAM 3 tracks from sam_masks.py (--mask_dir); slots = SAM tracks.

Output: one .pt per video with
  ctx, tgt   : [S, T', K, D] float16  (K = MAX_OBJ slots, zero where absent)
  ctx_area, tgt_area : [S, T', K] float16, mask area in patch units (0 = not visible)
  ctx_xy, tgt_xy     : [S, T', K, 2] float16, mask centroid (x, y) in [0, 1]
  objects    : per-slot info ({object_id, color, material, shape} for clevrer,
               {track_id, shape, n_frames, score} for sam)
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from extract_features import clip_pairs, encode_tokens, read_frames
from transformers import AutoModel, AutoVideoProcessor

MAX_OBJ = 8  # CLEVRER videos have 3-6 objects


def rle_decode(rle):
    """COCO RLE (compressed string or uncompressed list) -> bool [H, W]."""
    h, w = json.loads(rle["size"]) if isinstance(rle["size"], str) else rle["size"]
    s, counts, p = rle["counts"], [], 0
    if isinstance(s, list):
        counts, s = s, ""
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in counts:
        if val:
            flat[pos : pos + c] = True
        pos += c
        val = not val
    return flat.reshape(w, h).T  # RLE is column-major


_ANN = {}


def clevrer_masks(root, video_name):
    """Returns (objects, frames) where frames[i] = {slot: rle} for frame i."""
    if not _ANN:
        for a in glob.glob(os.path.join(root, "annotations", "**", "annotation_*.json"), recursive=True):
            _ANN[int(os.path.basename(a)[11:16])] = a
    idx = int(video_name.split("_")[-1])
    objects = json.load(open(_ANN[idx]))["object_property"]
    slot = {(o["color"], o["material"], o["shape"]): i for i, o in enumerate(objects)}
    prop = json.load(open(os.path.join(root, "proposals", f"proposal_{idx:05d}.json")))
    frames = []
    for f in prop["frames"]:
        best = {}  # slot -> (score, rle); keep the best-scoring mask if a triple repeats
        for o in f["objects"]:
            s = slot.get((o["color"], o["material"], o["shape"]))
            if s is not None and o["score"] > best.get(s, (-1, None))[0]:
                best[s] = (o["score"], o["mask"])
        frames.append({s: m for s, (_, m) in best.items()})
    return objects, frames


def sam_masks(mask_dir, video_name):
    d = json.load(open(os.path.join(mask_dir, f"{video_name}.json")))
    return d["objects"], [{int(k): v for k, v in f.items()} for f in d["frames"]]


def mask_weights(frame_masks, frame_ids, n_steps, side, cache):
    """[n_steps, MAX_OBJ, side, side] patch coverage, plus centroids [n_steps, MAX_OBJ, 2]."""
    w = np.zeros((n_steps, MAX_OBJ, side, side), np.float32)
    xy = np.zeros((n_steps, MAX_OBJ, 2), np.float32)
    for k in range(n_steps):
        ids = frame_ids[2 * k : 2 * k + 2]  # tubelet size 2
        for fi in ids:
            for s, rle in frame_masks[fi].items():
                if (fi, s) not in cache:
                    cache[(fi, s)] = rle_decode(rle)
                m = cache[(fi, s)]
                H, W = m.shape  # squash: the whole frame maps onto the side x side grid
                w[k, s] += m.reshape(side, H // side, side, W // side).mean(axis=(1, 3)) / len(ids)
                ys, xs = np.nonzero(m)
                if len(xs):
                    xy[k, s] += np.array([xs.mean() / W, ys.mean() / H]) / len(ids)
    return w, xy


def pool_objects(tokens, w):
    """tokens [t, side, side, D], w [t, K, side, side] -> feats [t, K, D], area [t, K]."""
    wt = torch.from_numpy(w).to(tokens.device)
    area = wt.sum(dim=(2, 3))
    feats = torch.einsum("tkij,tijd->tkd", wt, tokens) / area.clamp_min(1e-6)[..., None]
    return feats, area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--clevrer_root", help="folder with annotations/ and proposals/ (--masks clevrer)")
    ap.add_argument("--mask_dir", help="sam_masks.py output (--masks sam)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--masks", choices=["clevrer", "sam"], default="clevrer")
    ap.add_argument("--list", help="text file with one video name per line (default: all)")
    ap.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoVideoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    paths = sorted(glob.glob(os.path.join(args.videos, "**", "*.mp4"), recursive=True))
    if args.list:
        wanted = {l.strip() for l in open(args.list)}
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in wanted]
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
        frames = read_frames(p)
        if args.masks == "sam":
            objects, frame_masks = sam_masks(args.mask_dir, name)
        else:
            objects, frame_masks = clevrer_masks(args.clevrer_root, name)
        # some CLEVRER proposal files stop early (e.g. 115 of 128 frames): no masks there = not visible
        frame_masks = frame_masks + [{}] * (len(frames) - len(frame_masks))
        assert len(objects) <= MAX_OBJ, (name, len(objects))
        # frame indices of each clip, mirroring clip_pairs(window=32, stride=2, starts=(0, 64))
        idx = clip_pairs(np.arange(len(frames)))
        pairs = clip_pairs(frames)
        if not pairs:
            continue
        out, cache = {}, {}
        for part, j in (("ctx", 0), ("tgt", 1)):
            tok = encode_tokens(model, processor, [pr[j] for pr in pairs], device)  # [S, t, side, side, D]
            S, t, side = tok.shape[:3]
            feats, areas, xys = [], [], []
            for s in range(S):
                w, xy = mask_weights(frame_masks, idx[s][j], t, side, cache)
                f, a = pool_objects(tok[s], w)
                feats.append(f)
                areas.append(a)
                xys.append(torch.from_numpy(xy))
            out[part] = torch.stack(feats).half().cpu()
            out[f"{part}_area"] = torch.stack(areas).half().cpu()
            out[f"{part}_xy"] = torch.stack(xys).half()
        out["objects"] = objects
        torch.save(out, dst)
        if i % 50 == 0:
            vis = (out["ctx_area"] > 0.25).float().sum(-1).mean().item()
            print(f"[{i}/{len(paths)}] {name} ctx={tuple(out['ctx'].shape)} objects={len(objects)} "
                  f"visible/step={vis:.1f}", flush=True)


if __name__ == "__main__":
    main()
