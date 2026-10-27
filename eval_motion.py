"""Extra evaluation: does the predictor model motion, or only the static scene?

The official val metric averages L1 over all 64 patches, and most of a CLEVRER
frame is static background. This script re-scores a trained predictor and the
baselines on the SAME val split, separately on "moving" and "static" patches.

A patch is "moving" at target step k if, in pixel space, its cell changed
since the last context frame: mean |gray(target frames of step k) - gray(last
context frame)| over the cell > --thr. The mask comes from the raw video only,
so it does not favour any method. (Squash mode: the whole 480x320 frame is
resized to 256x256 and pooled 16x16 -> 8x8, so patch (i, j) is the cell
rows i*H/8..(i+1)*H/8, cols j*W/8..(j+1)*W/8 of the original frame.)

Reporting only: the official metric and BEATS verdict stay in train_predictor.py.

python eval_motion.py --ckpt ckpt_dir --features feats --videos clevrer/train
"""
import argparse
import glob
import json
import os
import random
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from extract_features import clip_pairs, read_frames
from train_predictor import Predictor, baselines


def motion_score(path, t_tgt, pool):
    """[S, t_tgt, pool*pool] mean abs gray change of each cell vs the last context frame."""
    frames = read_frames(path).astype(np.float32).mean(-1)  # [T, H, W]
    H, W = frames.shape[1:]
    out = []
    for ctx, tgt in clip_pairs(frames):
        d = np.abs(tgt - ctx[-1])  # [16, H, W]
        d = d.reshape(t_tgt, -1, pool, H // pool, pool, W // pool).mean(axis=(1, 3, 5))
        out.append(d.reshape(t_tgt, pool * pool))
    return np.stack(out)


def _job(a):
    return motion_score(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="dir with predictor_best.pt")
    ap.add_argument("--features", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--thr", type=float, nargs="+", default=[2.0, 5.0])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    ck = torch.load(os.path.join(args.ckpt, "predictor_best.pt"), map_location="cpu")
    cfg, s = ck["cfg"], ck["shape"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Predictor(s["D"], s["Tc"], s["Tt"], s["P"], cfg["width"], cfg["depth"]).to(device).eval()
    net.load_state_dict(ck["model"])
    mu, sd = ck["mu"].to(device), ck["sd"].to(device)
    pool = int(round(s["P"] ** 0.5))

    # same val split as train_predictor.py (same seed, same file order)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    files = sorted(glob.glob(os.path.join(args.features, "*.pt")))
    random.shuffle(files)
    val = files[: max(1, int(len(files) * cfg["val_frac"]))]
    videos = {os.path.splitext(os.path.basename(p))[0]: p
              for p in glob.glob(os.path.join(args.videos, "**", "*.mp4"), recursive=True)}
    names = [os.path.splitext(os.path.basename(f))[0] for f in val]
    with Pool(args.workers) as p:
        motion = p.map(_job, [(videos[n], s["Tt"], pool) for n in names], chunksize=4)

    # per-method sums of |error| over moving / static patch slots, for each threshold
    sums = {}
    counts = {thr: [0, 0] for thr in args.thr}
    with torch.no_grad():
        for f, m in zip(val, motion):
            d = torch.load(f, map_location="cpu")
            c = (d["ctx"].float().to(device) - mu) / sd
            t = (d["tgt"].float().to(device) - mu) / sd
            preds = {"predictor": net(c), **baselines(c, t.shape[1])}
            m = torch.from_numpy(m).to(device)  # [S, Tt, P]
            for k, pr in preds.items():
                err = (pr - t).abs().mean(-1)  # [S, Tt, P], L1 averaged over channels
                for thr in args.thr:
                    mov = m > thr
                    a = sums.setdefault((k, thr), [0.0, 0.0])
                    a[0] += err[mov].sum().item()
                    a[1] += err[~mov].sum().item()
            for thr in args.thr:
                mov = (m > thr).sum().item()
                counts[thr][0] += mov
                counts[thr][1] += m.numel() - mov
    res = {}
    for thr in args.thr:
        n_mov, n_stat = counts[thr]
        res[str(thr)] = {
            "frac_moving": n_mov / (n_mov + n_stat),
            **{k: {"moving": sums[(k, thr)][0] / max(1, n_mov), "static": sums[(k, thr)][1] / max(1, n_stat)}
               for k, t in sums if t == thr},
        }
    print(f"{args.ckpt}: {len(val)} val videos")
    for thr, r in res.items():
        print(f"thr {thr}: {100 * r['frac_moving']:.1f}% of patch slots moving")
        for k, v in r.items():
            if k != "frac_moving":
                print(f"  {k:10s} moving {v['moving']:.4f}  static {v['static']:.4f}")
    with open(os.path.join(args.ckpt, "motion_eval.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
