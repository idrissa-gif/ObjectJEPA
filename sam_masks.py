"""Stage 3 (segmentation): SAM 3 object tracks for CLEVRER videos.

Runs the SAM 3 video model (transformers Sam3VideoModel) with text prompts,
one per CLEVRER shape, and saves per-frame object masks as uncompressed RLE.
Only every --stride-th frame is tracked: the clip pairs use every 2nd frame,
so the others are never needed. The output has the same shape as the CLEVRER
proposals (frames[i] = {slot: rle}), so extract_object_features.py --masks sam
can use it directly.

python sam_masks.py --videos clevrer/train --out sam_masks --list val_videos.txt
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from extract_features import read_frames

MAX_OBJ = 8


def rle_encode(mask):
    """bool [H, W] -> uncompressed COCO RLE (column-major counts, starting with 0s)."""
    flat = np.asarray(mask, dtype=bool).T.ravel()
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    edges = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(edges).tolist()
    if flat[0]:
        counts = [0] + counts
    return {"size": list(mask.shape), "counts": counts}


@torch.no_grad()
def track(model, proc, frames, prompts, device):
    sess = proc.init_video_session(video=list(frames), inference_device=device, processing_device="cpu",
                                   video_storage_device="cpu", dtype=torch.bfloat16)
    proc.add_text_prompt(sess, prompts)
    # transformers 5.0.0 bug: Sam3VideoModel caches the whole text-encoder output instead of
    # .pooler_output (the image model does it right). Fill the cache with the tensor first.
    for pid, ids in sess.prompt_input_ids.items():
        sess.prompt_embeddings[pid] = model.detector_model.get_text_features(
            input_ids=ids, attention_mask=sess.prompt_attention_masks[pid]).pooler_output
    per_frame, shape_of = {}, {}
    for out in model.propagate_in_video_iterator(sess):
        r = proc.postprocess_outputs(sess, out)
        for prompt, ids in r["prompt_to_obj_ids"].items():
            for i in ids:
                shape_of.setdefault(int(i), prompt)
        per_frame[out.frame_idx] = {int(i): (m.cpu().numpy(), float(s))
                                    for i, m, s in zip(r["object_ids"], r["masks"], r["scores"])}
    return per_frame, shape_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", help="text file with one video name per line (default: all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--prompts", nargs="+", default=["cube", "sphere", "cylinder"])
    ap.add_argument("--model", default="facebook/sam3")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Sam3VideoModel.from_pretrained(args.model).to(device, dtype=torch.bfloat16).eval()
    proc = Sam3VideoProcessor.from_pretrained(args.model)

    paths = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(args.videos, "**", "*.mp4"), recursive=True)}
    names = [l.strip() for l in open(args.list)] if args.list else sorted(paths)
    if args.limit:
        names = names[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(names)} videos -> {args.out}", flush=True)

    for i, name in enumerate(names):
        dst = os.path.join(args.out, f"{name}.json")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        frames = read_frames(paths[name])
        per_frame, shape_of = track(model, proc, frames[:: args.stride], args.prompts, device)
        # keep the MAX_OBJ tracks seen in the most frames (drops short false positives)
        seen = {}
        for objs in per_frame.values():
            for oid, (_, s) in objs.items():
                seen.setdefault(oid, []).append(s)
        keep = sorted(seen, key=lambda o: -len(seen[o]))[:MAX_OBJ]
        slot = {oid: k for k, oid in enumerate(sorted(keep))}
        out_frames = [{} for _ in range(len(frames))]
        for j, objs in per_frame.items():
            for oid, (m, _) in objs.items():
                if oid in slot:
                    out_frames[j * args.stride][str(slot[oid])] = rle_encode(m)
        objects = [{"track_id": oid, "shape": shape_of.get(oid), "n_frames": len(seen[oid]),
                    "score": float(np.mean(seen[oid]))} for oid in sorted(keep)]
        with open(dst, "w") as f:
            json.dump({"video": name, "stride": args.stride, "objects": objects, "frames": out_frames}, f)
        print(f"[{i}/{len(names)}] {name} tracks={len(seen)} kept={len(keep)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
