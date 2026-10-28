"""Demo video: object masks + actual vs predicted object paths (+ optional "what if").

Frames 0-31 are observed (context). From frame 32 on, the object predictor rolls
forward (whatif.rollout) and the video shows, for each object:
  solid line  : actual centroid path (from the masks)
  dotted line : predicted path
  with --remove: the removed object is greyed out and the dotted line is the
  counterfactual path of the others; predicted collisions are marked.

python visualize.py --video video_09770 --ckpt ckpt_obj --features obj_features --masks clevrer --out demo.mp4
"""
import argparse
import os

import av
import cv2
import numpy as np
import torch

from extract_features import read_frames
from extract_object_features import clevrer_masks, rle_decode, sam_masks
from whatif import C, PATCH_PX, W_PX, H_PX, load_model, observed_tracks, rollout

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230)]
STEP = 4  # frames per tubelet step


def draw_path(img, pts, color, dotted=False):
    pts = [tuple(int(v) for v in p) for p in pts]
    for a, b in zip(pts[:-1], pts[1:]):
        if dotted:
            cv2.circle(img, b, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, b, 3, color, -1, cv2.LINE_AA)
        else:
            cv2.line(img, a, b, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="e.g. video_09770")
    ap.add_argument("--videos", default=os.path.expanduser("~/objectjepa_data/clevrer/train"))
    ap.add_argument("--clevrer_root", default=os.path.expanduser("~/objectjepa_data/clevrer"))
    ap.add_argument("--mask_dir", default=os.path.expanduser("~/objectjepa_data/sam_masks"))
    ap.add_argument("--masks", choices=["clevrer", "sam"], default="clevrer")
    ap.add_argument("--features", required=True, help="object features made with the same --masks")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--remove", type=int, nargs="*", default=[], help="slot(s) to remove")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    path = next(os.path.join(r, f) for r, _, fs in os.walk(args.videos) for f in fs if f == f"{args.video}.mp4")
    frames = read_frames(path)
    objects, fmasks = (sam_masks(args.mask_dir, args.video) if args.masks == "sam"
                       else clevrer_masks(args.clevrer_root, args.video))
    d = torch.load(os.path.join(args.features, f"{args.video}.pt"), map_location="cpu")
    model, mu, sd, cfg = load_model(args.ckpt)
    real_xy, _, real_vis = observed_tracks(d, cfg["min_area"])
    pred_xy, pred_area, pred_vis = rollout(model, mu, sd, d, cfg["min_area"], removed=args.remove)
    scale = np.array([W_PX, H_PX])
    real_xy, pred_xy = real_xy * scale, pred_xy * scale
    r = np.sqrt(np.maximum(pred_area, 0) * PATCH_PX / np.pi)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with av.open(args.out, "w") as out:
        st = out.add_stream("libx264", rate=args.fps)
        st.width, st.height, st.pix_fmt = W_PX, H_PX, "yuv420p"
        for fi in range(0, len(frames), 2):
            img = np.ascontiguousarray(frames[fi])
            t = fi // STEP
            for s, rle in fmasks[fi].items():
                m = rle_decode(rle).astype(np.uint8)
                color = (128, 128, 128) if s in args.remove else PALETTE[s % len(PALETTE)]
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, color, 2)
            for s in range(len(objects)):
                if s in args.remove:
                    continue
                color = PALETTE[s % len(PALETTE)]
                seen = [p for k, p in enumerate(real_xy[: t + 1, s]) if real_vis[k, s]]
                draw_path(img, seen, color)
                if t >= 8 and pred_vis[t, s]:  # predicted from frame 32 on
                    draw_path(img, pred_xy[8 : t + 1, s], color, dotted=True)
            if t >= 8:  # predicted collisions at this step
                for a in range(len(objects)):
                    for b in range(a + 1, len(objects)):
                        if pred_vis[t, a] and pred_vis[t, b] and \
                                np.linalg.norm(pred_xy[t, a] - pred_xy[t, b]) < C * (r[t, a] + r[t, b]):
                            mid = tuple(int(v) for v in (pred_xy[t, a] + pred_xy[t, b]) / 2)
                            cv2.drawMarker(img, mid, (255, 255, 255), cv2.MARKER_STAR, 18, 2)
            label = "observed" if t < 8 else ("what if: removed slot " + ",".join(map(str, args.remove))
                                              if args.remove else "predicted (dotted) vs actual (solid)")
            cv2.putText(img, f"frame {fi:3d}  {label}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
