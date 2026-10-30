"""Stage 4 tool: "what will happen next?" — forecast collisions after the video ends.

CLEVRER's predictive questions ask which of two collisions happens after the
last frame (frame 127); that event is not in the video or its annotations, so
answering is pure forecasting. The object predictor is rolled out from the LAST
clip (frames 96-127) for 3 calls (frames 128-223), the same 8 real + 24
predicted steps the collision head was trained on, and each pair is scored by
the head's highest contact logit in the first WINDOW predicted steps.

Evaluation (each question: 2 options, exactly 1 correct, chance 0.5): pick the
option whose pair scores higher. Baselines on the same tracks:
  distance_rule     : closest approach (distance / sum of radii) in the same rollout
  constant_velocity : objects keep their last velocity from frames 120-127

WINDOW = 8 steps (32 frames) was picked on the collision head's dev videos
(world model 0.985 at 32 frames, 0.981 at 64, 0.976 at 96).

python predict_next.py --ckpt ckpt_obj --features obj_features --head ckpt_collision --videos val_videos.txt
"""
import argparse
import json
import os

import numpy as np
import torch

from collision_head import PAIRS, head_logits
from whatif import H_PX, PATCH_PX, W_PX, resolve, rollout

WINDOW = 8  # predicted tubelet steps after the video (32 frames)
IDX = {tuple(p.tolist()): i for i, p in enumerate(PAIRS)}


def last_clip(d):
    """The video's last clip (frames 96-127) in the layout rollout() reads as context."""
    return {"ctx": d["tgt"][-1:], "ctx_area": d["tgt_area"][-1:], "ctx_xy": d["tgt_xy"][-1:]}


def contact_ratio(xy, area, vis):
    """[T, 28] centre distance / sum of radii per step and pair (inf where not both visible)."""
    p = xy * np.array([W_PX, H_PX])
    r = np.sqrt(np.maximum(area, 0) * PATCH_PX / np.pi)
    a, b = PAIRS[:, 0].numpy(), PAIRS[:, 1].numpy()
    ratio = np.linalg.norm(p[:, a] - p[:, b], axis=-1) / np.maximum(r[:, a] + r[:, b], 1)
    return np.where(vis[:, a] & vis[:, b], ratio, np.inf)


@torch.no_grad()
def future_scores(model, mu, sd, d, min_area, head=None, device="cpu"):
    """[28] score per slot pair for a collision in the WINDOW steps after the video (higher = more likely):
    the head's max contact logit on the rollout, or minus the closest approach without a head."""
    xy, area, vis, z = rollout(model, mu, sd, last_clip(d), min_area, device=device, return_latents=True)
    if head is None:
        return -contact_ratio(xy, area, vis)[8:8 + WINDOW].min(0)
    return head_logits(head, z, xy, area, vis, True, device)[8:8 + WINDOW].max(0)


def constant_velocity_scores(d, min_area):
    """[28] minus the closest approach when objects keep their velocity from the last two steps."""
    xy, area = d["tgt_xy"][-1].float().numpy(), d["tgt_area"][-1].float().numpy()
    t = np.arange(1, WINDOW + 1)[:, None, None]
    fxy = xy[-1][None] + t * (xy[-1] - xy[-2])[None]
    vis = (area[-1] > min_area)[None] & ((fxy >= 0) & (fxy <= 1)).all(-1)
    return -contact_ratio(fxy, np.repeat(area[-1][None], WINDOW, 0), vis).min(0)


def eval_predictive(model, mu, sd, cfg, feat_dir, questions, videos, head, device="cpu"):
    """Per-question accuracy of each scorer on CLEVRER predictive questions (ties count 0.5)."""
    by_video = {q["video_filename"][:-4]: q for q in questions}
    acc = {k: [] for k in ("world_model", "distance_rule", "constant_velocity")}
    skipped = 0
    for name in videos:
        path = os.path.join(feat_dir, f"{name}.pt")
        qs = [q for q in by_video.get(name, {"questions": []})["questions"] if q["question_type"] == "predictive"]
        if not qs or not os.path.exists(path):
            continue
        d = torch.load(path, map_location="cpu")
        scores = {"world_model": future_scores(model, mu, sd, d, cfg["min_area"], head, device),
                  "distance_rule": future_scores(model, mu, sd, d, cfg["min_area"], None, device),
                  "constant_velocity": constant_velocity_scores(d, cfg["min_area"])}
        for q in qs:
            pairs, truth = [], []
            for ch in q["choices"]:
                r = resolve(ch["program"], d["objects"])
                if len(r) == 2 and all(len(x) == 1 for x in r):
                    pairs.append(tuple(sorted((next(iter(r[0])), next(iter(r[1]))))))
                    truth.append(ch["answer"] == "correct")
            if len(pairs) != 2 or sum(truth) != 1 or any(a == b for a, b in pairs):
                skipped += 1
                continue
            for k, s in scores.items():
                s0, s1 = s[IDX[pairs[0]]], s[IDX[pairs[1]]]
                acc[k].append(0.5 if s0 == s1 else float(truth[0] if s0 > s1 else truth[1]))
    return acc, skipped


def main():
    from scipy.stats import binomtest

    from collision_head import load_head
    from whatif import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--head", required=True, help="collision head checkpoint dir (collision_head.py)")
    ap.add_argument("--questions", default=os.path.expanduser("~/objectjepa_data/clevrer/questions/train.json"))
    ap.add_argument("--videos", required=True, help="text file with one video name per line (use val videos)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mu, sd, cfg = load_model(args.ckpt, device)
    head, _ = load_head(args.head, device)
    acc, skipped = eval_predictive(model, mu, sd, cfg, args.features, json.load(open(args.questions)),
                                   [l.strip() for l in open(args.videos)], head, device)
    n = len(acc["world_model"])
    print(f"predictive questions: {n} (2 options, 1 correct, chance 0.5); skipped {skipped}")
    for k, v in acc.items():
        print(f"  {k:18s} {np.mean(v):.3f}")
    wm, cv = np.array(acc["world_model"]), np.array(acc["constant_velocity"])
    win, loss = int((wm > cv).sum()), int((wm < cv).sum())
    p = binomtest(win, win + loss).pvalue if win + loss else 1.0
    print(f"  world model vs constant velocity: {win} wins / {loss} losses, sign test p = {p:.2g}")
    with open(os.path.join(args.ckpt, f"predict_next_eval{args.tag}.json"), "w") as f:
        json.dump({k: {"accuracy": float(np.mean(v)), "n": len(v)} for k, v in acc.items()} |
                  {"skipped": skipped, "sign_test_vs_cv": {"wins": win, "losses": loss, "p": p},
                   "window_steps": WINDOW, "videos": args.videos, "features": args.features}, f, indent=1)


if __name__ == "__main__":
    main()
