"""Learned "what if" update on top of the collision head.

whatif.py's delta rule keeps the observed collisions and changes only the pairs
where the rollouts with and without the removed object disagree. This replaces
that rule with a small gradient-boosted classifier over the same world-model
signals, per object pair: the head's logits on the observed video, on the
factual rollout and on the counterfactual rollout (max, first contact step,
number of contact steps), and the cf - factual difference.

Deliberately NOT used: anything about the removed object's own contacts in the
real video. A classifier on those cues alone scores 0.88 on dev (a CLEVRER
shortcut: "if x hit a first, a's next collision changes"), and adding the
rollouts on top does not help it, so it would not measure the world model.

Fitted on CLEVRER counterfactual questions of the head's 1,000 DEV videos
(train split; the head never trained on them); the val videos are only scored
by whatif.py.

python whatif_update.py --features obj_features_clevrer --ckpt ckpt_obj_e60_w512d8 --head ckpt_collision \
    --videos dev_videos_head.txt --out ckpt_update
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from collision_head import PAIRS

IDX = {tuple(p.tolist()): i for i, p in enumerate(PAIRS)}


def summary(logit, t0, thr):
    """[28, 3]: max logit, first step above thr (or T), number of steps above thr, from step t0."""
    s = logit[t0:]
    above = s > thr
    first = np.where(above.any(0), above.argmax(0) + t0, logit.shape[0])
    return np.stack([s.max(0), first, above.sum(0)], 1)


def pair_features(lf, lb, lc, thr):
    """[28, 10] features from observed / factual-rollout / counterfactual-rollout logits [T, 28]."""
    return np.concatenate([summary(lf, 0, thr), summary(lb, 8, thr), summary(lc, 8, thr),
                           (lc[8:].max(0) - lb[8:].max(0))[:, None]], 1).astype(np.float32)


def load_update(ckpt_dir):
    with open(os.path.join(ckpt_dir, "update.pkl"), "rb") as f:
        return pickle.load(f)


def update_events(upd, lf, lb, lc, thr, removed=()):
    """Counterfactual collisions (slot pairs) predicted by the learned update."""
    hit = upd.predict(pair_features(lf, lb, lc, thr)).astype(bool)
    return {tuple(PAIRS[j].tolist()) for j in np.nonzero(hit)[0]} - {p for p in IDX if set(p) & set(removed)}


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier

    from whatif import eval_questions, load_model
    from collision_head import load_head

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--ckpt", required=True, help="object predictor checkpoint dir")
    ap.add_argument("--head", required=True, help="collision head checkpoint dir")
    ap.add_argument("--questions", default=os.path.expanduser("~/objectjepa_data/clevrer/questions/train.json"))
    ap.add_argument("--videos", required=True, help="the head's dev videos (never the val videos)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mu, sd, cfg = load_model(args.ckpt, device)
    head, thr = load_head(args.head, device)
    rows = []
    eval_questions(model, mu, sd, cfg, args.features, json.load(open(args.questions)),
                   [l.strip() for l in open(args.videos)], device, head, thr, head_fact=True, rows=rows)
    X = np.stack([pair_features(lf, lb, lc, thr)[IDX[pair]] for lf, lb, lc, pair, _ in rows])
    y = np.array([collide for *_, collide in rows])
    upd = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0).fit(X, y)
    print(f"fitted on {len(y)} options, train accuracy {(upd.predict(X) == y).mean():.3f}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "update.pkl"), "wb") as f:
        pickle.dump(upd, f)


if __name__ == "__main__":
    main()
