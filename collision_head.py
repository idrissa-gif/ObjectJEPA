"""Learned collision read-out ("collision head") on object latents.

The what-if tool reads collisions out of predicted screen positions with a
distance rule (whatif.collisions). This head instead reads them from the object
latents and tracks: for every object pair and tubelet step it predicts whether
the two objects are in contact, trained on CLEVRER's annotated collisions.

Training sequences per TRAIN video (the val videos of the predictors are never used):
  real : the observed latents/tracks of all 32 steps (frames 0-127)
  pred : the object predictor's rollout from clip 0 (8 real context steps, then
         24 predicted steps), so the head learns to read imperfect predictions.
The train videos are split again: --dev videos (default 1,000) pick the epoch and
the threshold; the val videos are only used by whatif.py for the final number.

python collision_head.py --features obj_features_clevrer --ckpt ckpt_obj_e60_w512d8 --val_videos val_videos.txt --out ckpt_collision
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from whatif import C, PATCH_PX, H_PX, W_PX, collisions, load_model, rollout

K = 8
PAIRS = torch.tensor([(a, b) for a in range(K) for b in range(a + 1, K)])  # [28, 2]


def observed(d):
    """Real latents/areas/xy over the 32 steps (frames 0-127)."""
    cat = lambda key: torch.cat([d[f"{p}{key}"][s] for s in range(d["ctx"].shape[0]) for p in ("ctx", "tgt")]).float()
    return cat(""), cat("_area"), cat("_xy")


class CollisionHead(nn.Module):
    def __init__(self, d_in, width=128, n_geo=6):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, width), nn.GELU())
        self.temporal = nn.Sequential(
            nn.Conv1d(3 * width + n_geo, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, 1, 1))

    @staticmethod
    def geometry(xy, area, vis, is_pred):
        """Per pair and step: contact ratio, distance, closing speed, radii, both visible, predicted flag."""
        p = xy * xy.new_tensor([W_PX, H_PX])
        r = (area.clamp_min(0) * PATCH_PX / np.pi).sqrt()
        a, b = PAIRS[:, 0].to(xy.device), PAIRS[:, 1].to(xy.device)
        dist = (p[:, :, a] - p[:, :, b]).norm(dim=-1)  # [B, T, 28]
        rr = r[:, :, a] + r[:, :, b]
        both = (vis[:, :, a] & vis[:, :, b]).float()
        closing = F.pad(dist[:, 1:] - dist[:, :-1], (0, 0, 1, 0))
        flag = is_pred.float().expand_as(dist)
        geo = torch.stack([dist / rr.clamp_min(1), dist / 100, closing / 20, rr / 50, both, flag], -1)
        return geo * both.unsqueeze(-1), both

    def forward(self, z, xy, area, vis, is_pred):  # z [B, T, K, D] normalised; is_pred [B, T, 1]
        h = self.proj(z)
        a, b = PAIRS[:, 0].to(z.device), PAIRS[:, 1].to(z.device)
        ha, hb = h[:, :, a], h[:, :, b]
        geo, both = self.geometry(xy, area, vis, is_pred)
        x = torch.cat([ha + hb, (ha - hb).abs(), ha * hb, geo], -1)  # [B, T, 28, F]
        B, T, P, Fd = x.shape
        logit = self.temporal(x.permute(0, 2, 3, 1).reshape(B * P, Fd, T)).view(B, P, T).transpose(1, 2)
        return logit, both  # [B, T, 28] per-step contact logits, valid mask


def load_head(ckpt_dir, device="cpu"):
    ck = torch.load(os.path.join(ckpt_dir, "head_best.pt"), map_location="cpu")
    head = CollisionHead(ck["d_in"], ck["width"]).to(device).eval()
    head.load_state_dict(ck["head"])
    return head, ck["thr"]


@torch.no_grad()
def head_logits(head, z, xy, area, vis, predicted, device="cpu"):
    """Per-step contact logits [T, 28] (numpy) for one sequence: z [T, K, D] normalised latents;
    steps >= 8 predicted if `predicted`. Steps where a pair is not both visible get -20."""
    T = z.shape[0]
    pr = torch.zeros(1, T, 1, device=device)
    if predicted:
        pr[:, 8:] = 1
    t = lambda x, dt: torch.as_tensor(x).to(device, dt)[None]
    logit, both = head(t(z, torch.float), t(xy, torch.float), t(area, torch.float), t(vis, torch.bool), pr)
    return logit.masked_fill(~both.bool(), -20.0)[0].cpu().numpy()


def logit_events(logit, thr):
    """Colliding slot pairs: any step of [T, 28] logits above thr."""
    return {tuple(PAIRS[j].tolist()) for j in np.nonzero((logit > thr).any(0))[0]}


def head_events(head, thr, z, xy, area, vis, predicted, device="cpu"):
    """Colliding slot pairs in one sequence (see head_logits)."""
    return logit_events(head_logits(head, z, xy, area, vis, predicted, device), thr)


def labels(ann_path, T=32):
    """Per-step contact labels [T, 28] from CLEVRER collisions (step = frame // 4, +-1 step)."""
    y = torch.zeros(T, len(PAIRS))
    idx = {tuple(p.tolist()): i for i, p in enumerate(PAIRS)}
    for c in json.load(open(ann_path))["collision"]:
        a, b = sorted(c["object_ids"])
        if b < K:
            t = c["frame_id"] // 4
            y[max(0, t - 1):t + 2, idx[(a, b)]] = 1
    return y


@torch.no_grad()
def build(files, model, mu, sd, min_area, ann, device):
    """Stack real and predicted sequences for a list of feature files."""
    out = {k: [] for k in ("z", "xy", "area", "vis", "pred", "y")}
    for f in files:
        d = torch.load(f, map_location="cpu")
        z, area, xy = observed(d)
        y = labels(ann[int(os.path.basename(f)[-8:-3])])
        rxy, rarea, rvis, rz = rollout(model, mu, sd, d, min_area, device=device, return_latents=True)
        for zz, xx, aa, vv, pr in (((z - mu.cpu()) / sd.cpu(), xy, area, area > min_area, 0),
                                   (rz, torch.from_numpy(rxy), torch.from_numpy(rarea), torch.from_numpy(rvis), 1)):
            out["z"].append(zz.half()); out["xy"].append(xx.float()); out["area"].append(aa.float())
            out["vis"].append(vv.bool()); out["y"].append(y)
            out["pred"].append(torch.cat([torch.zeros(8), torch.full((24,), float(pr))]).view(32, 1))
    return {k: torch.stack(v) for k, v in out.items()}


def event_prf(logit, both, y, pred, thr, future_only):
    """Pair-level events (any step) vs annotations, on predicted (pred=1) sequences."""
    m = both.bool() & (torch.arange(32, device=logit.device).view(1, 32, 1) >= (8 if future_only else 0))
    sel = pred[:, -1, 0] > 0
    ev = ((logit > thr) & m).any(1)[sel]
    tr = ((y > 0) & m).any(1)[sel]
    tp, P, T = (ev & tr).sum().item(), ev.sum().item(), tr.sum().item()
    p, r = tp / max(1, P), tp / max(1, T)
    return p, r, 2 * p * r / max(1e-9, p + r)


def rule_prf(data, future_only):
    """Same metric for the distance rule (whatif.collisions) on the same predicted sequences."""
    tp = P = T = 0
    for i in torch.nonzero(data["pred"][:, -1, 0] > 0).flatten().tolist():
        t0 = 8 if future_only else 0
        xy, area, vis = (data[k][i, t0:].numpy() for k in ("xy", "area", "vis"))
        ev = collisions(xy, area, vis, C)
        tr = {tuple(PAIRS[j].tolist()) for j in torch.nonzero((data["y"][i, t0:] > 0).any(0)).flatten().tolist()
              if vis[:, PAIRS[j, 0]].any() and vis[:, PAIRS[j, 1]].any()}
        tp, P, T = tp + len(ev & tr), P + len(ev), T + len(tr)
    p, r = tp / max(1, P), tp / max(1, T)
    return p, r, 2 * p * r / max(1e-9, p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--ckpt", required=True, help="object predictor checkpoint dir")
    ap.add_argument("--val_videos", required=True, help="held-out videos, never used here")
    ap.add_argument("--annotations", default=os.path.expanduser("~/objectjepa_data/clevrer/annotations"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="max train videos (smoke test)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--width", type=int, default=128)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ann = {int(os.path.basename(a)[11:16]): a
           for a in glob.glob(os.path.join(args.annotations, "**", "annotation_*.json"), recursive=True)}
    val = {l.strip() for l in open(args.val_videos)}
    files = [f for f in sorted(glob.glob(os.path.join(args.features, "*.pt"))) if os.path.basename(f)[:-3] not in val]
    random.Random(0).shuffle(files)
    dev_files, train_files = files[:args.dev], files[args.dev:]
    if args.limit:
        dev_files, train_files = dev_files[:args.limit // 5], train_files[:args.limit]
    print(f"{len(train_files)} train / {len(dev_files)} dev videos (val videos excluded: {len(val)})", flush=True)
    with open(os.path.join(args.out, "dev_videos.txt"), "w") as f:  # whatif_update.py fits on these
        f.write("".join(os.path.basename(p)[:-3] + "\n" for p in dev_files))

    model, mu, sd, cfg = load_model(args.ckpt, device)
    tr = build(train_files, model, mu, sd, cfg["min_area"], ann, device)
    dv = build(dev_files, model, mu, sd, cfg["min_area"], ann, device)
    print(f"built {len(tr['z'])} train / {len(dv['z'])} dev sequences", flush=True)

    head = CollisionHead(tr["z"].shape[-1], args.width).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(tr["z"]) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    pos = (tr["y"] > 0).float().mean().item()
    pos_w = torch.tensor((1 - pos) / max(pos, 1e-6)).sqrt().to(device)

    def run(data, train):
        head.train(train)
        order = torch.randperm(len(data["z"])) if train else torch.arange(len(data["z"]))
        logits, boths, tot = [], [], 0.0
        for i in range(0, len(order), args.bs):
            idx = order[i:i + args.bs]
            z, xy, area, vis, pr, y = (data[k][idx].to(device) for k in ("z", "xy", "area", "vis", "pred", "y"))
            with torch.set_grad_enabled(train):
                logit, both = head(z.float(), xy, area, vis, pr)
                loss = (F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w, reduction="none") * both).sum() / both.sum().clamp_min(1)
            if train:
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            else:
                logits.append(logit.cpu()); boths.append(both.cpu())
            tot += loss.item() * len(idx)
        return tot / len(order), (torch.cat(logits), torch.cat(boths)) if logits else None

    rule = rule_prf(dv, True)
    print(f"distance rule on dev rollouts, future steps: P {rule[0]:.3f} R {rule[1]:.3f} F1 {rule[2]:.3f}", flush=True)
    best, hist = None, []
    for ep in range(args.epochs):
        tl, _ = run(tr, True)
        vl, (lg, bo) = run(dv, False)
        f1s = {thr: event_prf(lg, bo, dv["y"], dv["pred"], thr, True) for thr in np.arange(-2, 4.01, 0.5)}
        thr, (p, r, f1) = max(f1s.items(), key=lambda kv: kv[1][2])
        hist.append({"epoch": ep, "train_loss": tl, "dev_loss": vl, "thr": float(thr), "P": p, "R": r, "F1": f1})
        print(f"ep {ep:2d} train {tl:.4f} dev {vl:.4f} | future events P {p:.3f} R {r:.3f} F1 {f1:.3f} @thr {thr:+.1f}"
              f"  (rule F1 {rule[2]:.3f})", flush=True)
        if best is None or f1 > best:
            best = f1
            torch.save({"head": head.state_dict(), "thr": float(thr), "width": args.width, "d_in": tr["z"].shape[-1],
                        "predictor": os.path.abspath(args.ckpt)}, os.path.join(args.out, "head_best.pt"))
    json.dump({"history": hist, "rule_future_dev": rule}, open(os.path.join(args.out, "history.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
