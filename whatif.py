"""Stage 4 tool: counterfactual ("what if") rollouts with the object predictor.

Given one video's object latents for its first clip (frames 0-31):
  1. optionally remove objects (their context tokens are hidden from the model),
  2. roll the object predictor forward: each call predicts the next 32 frames,
     and its predictions become the next context (3 calls cover frames 32-127),
  3. detect collisions from predicted screen positions: objects a, b collide
     when |xy_a - xy_b| < C * (r_a + r_b), r from the mask area. C = 1.0 was
     calibrated on 300 val videos against CLEVRER's annotated collisions using
     the real masks (precision 0.73, recall 0.97).

Evaluation on CLEVRER counterfactual questions ("Which event will happen if
the cylinder is removed?"): the question/choice programs are parsed directly
(no LLM), so the score measures the physics, not the language part. Reported
with two ways of using the model and one baseline:
  predictor : events of the counterfactual rollout itself
  delta     : real events, changed only where the rollouts with and without the
              object disagree (cancels the model's systematic rollout error)
  no_change : counterfactual events = real events (no model). Strong, since the
              answer choices never involve the removed object and most other
              collisions do not change.

python whatif.py --ckpt ckpt_obj --features obj_features --questions questions/train.json --videos val_videos.txt
"""
import argparse
import json
import os

import numpy as np
import torch

from train_object_predictor import ObjectPredictor, last_seen

C = 1.0  # collision distance factor, calibrated on CLEVRER masks
W_PX, H_PX = 480, 320
PATCH_PX = (W_PX / 16) * (H_PX / 16)  # pixels per V-JEPA patch (squash, 16x16 grid)


def load_model(ckpt_dir, device="cpu"):
    ck = torch.load(os.path.join(ckpt_dir, "predictor_best.pt"), map_location="cpu")
    s, cfg = ck["shape"], ck["cfg"]
    model = ObjectPredictor(s["D"], s["Tc"], s["Tt"], s["K"], cfg["width"], cfg["depth"]).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, ck["mu"].to(device), ck["sd"].to(device), cfg


@torch.no_grad()
def rollout(model, mu, sd, d, min_area, removed=(), n_calls=3, device="cpu"):
    """Positions/areas/visibility over the whole video, in tubelet steps (4 frames each).

    d: one video's features (from extract_object_features.py). Uses clip pair 0.
    Returns xy [T, K, 2] in [0, 1], area [T, K] in patches, vis [T, K] bool.
    """
    ctx = ((d["ctx"][0].float().to(device) - mu) / sd).unsqueeze(0)  # [1, Tc, K, D]
    area = d["ctx_area"][0].float().to(device).unsqueeze(0)
    xy = d["ctx_xy"][0].float().to(device).unsqueeze(0)
    vis = area > min_area
    for k in removed:
        vis[..., k] = False
    xys, areas, viss = [xy[0]], [area[0]], [vis[0]]
    for _ in range(n_calls):
        pred, pxy = model(ctx, vis, xy)
        Tt = pred.shape[1]
        # objects seen in the context stay present; their size is carried over
        seen = vis.any(1, keepdim=True).expand(-1, Tt, -1)
        a = last_seen(area.unsqueeze(-1), vis).squeeze(-1).expand(-1, Tt, -1)
        ctx, xy, vis, area = pred, pxy, seen, a
        xys.append(xy[0]); areas.append(area[0]); viss.append(vis[0])
    return torch.cat(xys).cpu().numpy(), torch.cat(areas).cpu().numpy(), torch.cat(viss).cpu().numpy()


def collisions(xy, area, vis, c=C):
    """Set of colliding slot pairs (a < b) from tracks xy [T, K, 2], area [T, K], vis [T, K]."""
    p = xy * np.array([W_PX, H_PX])
    r = np.sqrt(np.maximum(area, 0) * PATCH_PX / np.pi)
    T, K = vis.shape
    ev = set()
    for a in range(K):
        for b in range(a + 1, K):
            both = vis[:, a] & vis[:, b]
            dist = np.linalg.norm(p[:, a] - p[:, b], axis=-1)
            if np.any(both & (dist < c * (r[:, a] + r[:, b]))):
                ev.add((a, b))
    return ev


def observed_tracks(d, min_area):
    """Real tracks from the masks over both clip pairs (frames 0-127), same layout as rollout()."""
    parts = [(d[f"{p}_xy"][s], d[f"{p}_area"][s]) for s in range(d["ctx"].shape[0]) for p in ("ctx", "tgt")]
    xy = torch.cat([x for x, _ in parts]).float().numpy()
    area = torch.cat([a for _, a in parts]).float().numpy()
    return xy, area, area > min_area


# --- CLEVRER question programs -> slots -------------------------------------------------------

ATTR = {"filter_color": "color", "filter_material": "material", "filter_shape": "shape"}


def resolve(program, objects):
    """Object references in a CLEVRER program -> list of slot index sets, in order of appearance."""
    refs, cur = [], None
    for i, tok in enumerate(program):
        if tok == "objects":
            cur = set(range(len(objects)))
        elif tok in ATTR and cur is not None:
            val = program[i - 1]
            cur = {k for k in cur if objects[k].get(ATTR[tok]) == val}
        elif tok == "unique" and cur is not None:
            refs.append(cur)
            cur = None
    return refs


def eval_questions(model, mu, sd, cfg, feat_dir, questions, videos, device="cpu"):
    by_video = {q["video_filename"][:-4]: q for q in questions}
    stats = {k: [0, 0] for k in ("predictor", "delta", "no_change")}  # [correct options, total]
    per_q = {k: [0, 0] for k in stats}
    skipped = 0
    for name in videos:
        path = os.path.join(feat_dir, f"{name}.pt")
        if name not in by_video or not os.path.exists(path):
            continue
        d = torch.load(path, map_location="cpu")
        objs = d["objects"]
        fact = collisions(*observed_tracks(d, cfg["min_area"]))
        base = None  # factual rollout, computed once per video
        for q in by_video[name]["questions"]:
            if q["question_type"] != "counterfactual":
                continue
            rem = resolve(q["program"], objs)
            choices = [(resolve(ch["program"], objs), ch["answer"] == "correct") for ch in q["choices"]]
            if len(rem) != 1 or len(rem[0]) != 1 or any(len(r) != 2 or any(len(x) != 1 for x in r) for r, _ in choices):
                skipped += 1  # program shape we do not parse (e.g. non-collision events)
                continue
            x = next(iter(rem[0]))
            negate = q["program"][-1] == "negate"  # "which event will NOT happen": correct = absent
            cf = collisions(*rollout(model, mu, sd, d, cfg["min_area"], removed=[x], device=device))
            if base is None:
                base = collisions(*rollout(model, mu, sd, d, cfg["min_area"], device=device))
            answers = {
                "predictor": cf,
                "delta": (fact - (base - cf)) | (cf - base),
                "no_change": fact,
            }
            for k, ev in answers.items():
                ok_all = True
                for (ra, rb), truth in choices:
                    pair = tuple(sorted((next(iter(ra)), next(iter(rb)))))
                    ok = ((pair in ev) != negate) == truth
                    stats[k][0] += ok
                    stats[k][1] += 1
                    ok_all &= ok
                per_q[k][0] += ok_all
                per_q[k][1] += 1
    return {k: {"per_option": stats[k][0] / max(1, stats[k][1]), "per_question": per_q[k][0] / max(1, per_q[k][1]),
                "n_options": stats[k][1], "n_questions": per_q[k][1]} for k in stats}, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--videos", required=True, help="text file with one video name per line (use val videos)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mu, sd, cfg = load_model(args.ckpt, device)
    questions = json.load(open(args.questions))
    videos = [l.strip() for l in open(args.videos)]
    res, skipped = eval_questions(model, mu, sd, cfg, args.features, questions, videos, device)
    for k, v in res.items():
        print(f"{k:10s} per-option {v['per_option']:.3f}  per-question {v['per_question']:.3f}  "
              f"({v['n_questions']} questions, {v['n_options']} options)")
    print(f"skipped {skipped} questions with an unparsed program")
    with open(os.path.join(args.ckpt, "whatif_eval.json"), "w") as f:
        json.dump({"results": res, "skipped": skipped, "videos": args.videos, "features": args.features}, f, indent=1)


if __name__ == "__main__":
    main()
