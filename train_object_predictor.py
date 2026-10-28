"""Stage 3: object-level JEPA predictor on per-object V-JEPA 2 latents.

Input : ctx [T', K, D] (one token per object and time step, from
        extract_object_features.py), target tgt [T', K, D].
Output: predicted tgt latents per object, plus each object's screen position.

Same recipe as train_predictor.py: frozen encoder, L1 in latent space,
residual on the object's last seen context latent, split by VIDEO with the
same seed (so the val videos are the same as for the frame-level predictor).

Visibility: an object step is visible if its mask covers > --min_area
patches. Context tokens of invisible steps are masked out of attention.
Scored target steps: visible in the target AND seen at least once in the
context (an object that only enters later cannot be predicted).

Baselines per object (official criterion: copy_last, linear, mean):
  copy_last : last seen context latent
  linear    : last + (last - prev) * k, if visible at the last two context steps
  mean      : mean of the visible context latents
Also reported: copy_clip (ctx[k] if visible, else last seen).
Positions (x, y in [0, 1]): predictor head vs copy_last and constant velocity.

python train_object_predictor.py --features obj_features --out ckpt_obj
python train_object_predictor.py --eval_only --out ckpt_obj --eval_features obj_features_sam
"""
import argparse
import glob
import json
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

W_XY, H_XY = 480, 320  # CLEVRER frame size, to report position error in pixels


class ObjectPairs(Dataset):
    def __init__(self, files, min_area):
        self.items = []
        for f in files:
            d = torch.load(f, map_location="cpu")
            for s in range(d["ctx"].shape[0]):
                self.items.append(tuple(d[k][s] for k in ("ctx", "tgt", "ctx_area", "tgt_area", "ctx_xy", "tgt_xy")))
        self.min_area = min_area

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        c, t, ca, ta, cxy, txy = (x.float() for x in self.items[i])
        return c, t, ca > self.min_area, ta > self.min_area, cxy, txy


def last_seen(x, vis):
    """x [B, T, K, ...], vis [B, T, K] -> value at the last visible step [B, 1, K, ...] (0 if never)."""
    T = vis.shape[1]
    step = torch.arange(T, device=vis.device).view(1, T, 1)
    idx = torch.where(vis, step, torch.full_like(step, -1)).amax(1)  # [B, K]
    gather = idx.clamp_min(0).view(idx.shape[0], 1, -1, *([1] * (x.dim() - 3)))
    gather = gather.expand(-1, 1, -1, *x.shape[3:])
    out = x.gather(1, gather)
    return out * (idx >= 0).view(idx.shape[0], 1, -1, *([1] * (x.dim() - 3)))


class ObjectPredictor(nn.Module):
    """Transformer over (visible context object tokens + future query tokens per object)."""

    def __init__(self, d_in, t_ctx, t_tgt, n_slot, width=384, depth=6, heads=None, dropout=0.1):
        super().__init__()
        self.t_tgt = t_tgt
        heads = heads or max(1, width // 64)
        self.inp = nn.Linear(d_in, width)
        self.time_pos = nn.Parameter(torch.zeros(t_ctx + t_tgt, 1, width))
        self.slot_pos = nn.Parameter(torch.zeros(1, n_slot, width))  # object identity tag
        self.query = nn.Parameter(torch.zeros(t_tgt, 1, width))
        for p in (self.time_pos, self.slot_pos, self.query):
            nn.init.trunc_normal_(p, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, d_in)
        self.out_xy = nn.Linear(width, 2)
        for lin in (self.out, self.out_xy):  # residuals start at zero (= copy_last)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, ctx, ctx_vis, ctx_xy):  # ctx [B, Tc, K, D], ctx_vis [B, Tc, K]
        B, Tc, K, _ = ctx.shape
        x = self.inp(ctx)
        q = self.query.expand(-1, K, -1).unsqueeze(0).expand(B, -1, -1, -1)
        x = torch.cat([x, q], 1) + self.time_pos + self.slot_pos  # [B, Tc+Tt, K, W]
        pad = torch.cat([~ctx_vis, ctx_vis.new_zeros(B, self.t_tgt, K)], 1)  # True = ignore
        x = self.blocks(x.flatten(1, 2), src_key_padding_mask=pad.flatten(1))
        x = self.norm(x).view(B, Tc + self.t_tgt, K, -1)[:, Tc:]
        return last_seen(ctx, ctx_vis) + self.out(x), last_seen(ctx_xy, ctx_vis) + self.out_xy(x)


def baselines(ctx, vis, xy, t_tgt):
    last = last_seen(ctx, vis)
    both = (vis[:, -1] & vis[:, -2]).unsqueeze(1).unsqueeze(-1)  # [B, 1, K, 1]
    k = torch.arange(1, t_tgt + 1, device=ctx.device).view(1, -1, 1, 1).float()
    lin = torch.where(both, ctx[:, -1:] + (ctx[:, -1:] - ctx[:, -2:-1]) * k, last)
    w = vis.unsqueeze(-1).float()
    mean = ((ctx * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1)).expand(-1, t_tgt, -1, -1)
    out = {"copy_last": last.expand(-1, t_tgt, -1, -1), "linear": lin, "mean": mean}
    if ctx.shape[1] == t_tgt:
        out["copy_clip"] = torch.where(vis.unsqueeze(-1), ctx, last)
    xy_last = last_seen(xy, vis)
    xy_lin = torch.where(both, xy[:, -1:] + (xy[:, -1:] - xy[:, -2:-1]) * k, xy_last)
    return out, {"copy_last": xy_last.expand(-1, t_tgt, -1, -1), "const_velocity": xy_lin}


def score_mask(ctx_vis, tgt_vis):
    return tgt_vis & ctx_vis.any(1, keepdim=True)  # [B, Tt, K]


@torch.no_grad()
def evaluate(model, loader, device, norm):
    model.eval()
    l1, cos, px, n = {}, {}, {}, 0
    for c, t, cv, tv, cxy, txy in loader:
        c, t, cv, tv, cxy, txy = (x.to(device) for x in (c, t, cv, tv, cxy, txy))
        c, t = norm(c), norm(t)
        m = score_mask(cv, tv)
        pred, pxy = model(c, cv, cxy)
        base, base_xy = baselines(c, cv, cxy, t.shape[1])
        for k, p in {"predictor": pred, **base}.items():
            l1[k] = l1.get(k, 0.0) + (p - t).abs().mean(-1)[m].sum().item()
            cos[k] = cos.get(k, 0.0) + F.cosine_similarity(p, t, dim=-1)[m].sum().item()
        scale = torch.tensor([W_XY, H_XY], device=device)
        for k, p in {"predictor": pxy, **base_xy}.items():
            px[k] = px.get(k, 0.0) + ((p - txy) * scale).norm(dim=-1)[m].sum().item()
        n += m.sum().item()
    model.train()
    res = {f"l1/{k}": v / n for k, v in l1.items()}
    res.update({f"cos/{k}": v / n for k, v in cos.items()})
    res.update({f"px/{k}": v / n for k, v in px.items()})
    res["n_scored"] = n
    return res


def line(res):
    beats = res["l1/predictor"] < min(res["l1/copy_last"], res["l1/linear"], res["l1/mean"])
    clip = res.get("l1/copy_clip")
    return (f"val predictor {res['l1/predictor']:.4f} copy_last {res['l1/copy_last']:.4f} "
            f"linear {res['l1/linear']:.4f} mean {res['l1/mean']:.4f} "
            + (f"copy_clip {clip:.4f} " if clip is not None else "")
            + f"| cos {res['cos/predictor']:.3f} vs {res['cos/copy_last']:.3f} "
            + f"| px {res['px/predictor']:.1f} vs const_vel {res['px/const_velocity']:.1f} copy_last {res['px/copy_last']:.1f} "
            + ("BEATS" if beats else "")
            + ("+clip" if beats and clip is not None and res["l1/predictor"] < clip else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features")
    ap.add_argument("--out", default="ckpt_obj")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--xy_weight", type=float, default=10.0)
    ap.add_argument("--min_area", type=float, default=0.25, help="visible if mask covers > this many patches")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--eval_features", help="also evaluate the best checkpoint on ALL files in this folder")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    if not args.eval_only:
        random.seed(args.seed)  # same split as train_predictor.py
        torch.manual_seed(args.seed)
        files = sorted(glob.glob(os.path.join(args.features, "*.pt")))
        assert files, f"no feature files in {args.features}"
        random.shuffle(files)
        n_val = max(1, int(len(files) * args.val_frac))
        train_ds, val_ds = ObjectPairs(files[n_val:], args.min_area), ObjectPairs(files[:n_val], args.min_area)
        print(f"train {len(train_ds)} pairs / val {len(val_ds)} pairs from {len(files)} videos", flush=True)

        c0 = train_ds[0][0]
        Tc, K, D = c0.shape
        Tt = train_ds[0][1].shape[0]
        # per-channel normalisation from visible training target tokens
        toks = torch.cat([train_ds[i][1][train_ds[i][3]] for i in range(min(len(train_ds), 2000))])
        mu, sd = toks.mean(0).to(device), toks.std(0).clamp_min(1e-4).to(device)
        norm = lambda x: (x - mu) / sd

        train_dl = DataLoader(train_ds, args.bs, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
        val_dl = DataLoader(val_ds, args.bs, num_workers=4)
        model = ObjectPredictor(D, Tc, Tt, K, args.width, args.depth).to(device)
        print(f"object predictor params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        total = args.epochs * len(train_dl)
        warm = max(1, total // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
        )
        best, history = float("inf"), []
        for ep in range(args.epochs):
            run = 0.0
            for c, t, cv, tv, cxy, txy in train_dl:
                c, t, cv, tv, cxy, txy = (x.to(device, non_blocking=True) for x in (c, t, cv, tv, cxy, txy))
                perm = torch.randperm(K, device=device)  # slot order carries no meaning
                c, t, cv, tv, cxy, txy = (x[:, :, perm] for x in (c, t, cv, tv, cxy, txy))
                c, t = norm(c), norm(t)
                m = score_mask(cv, tv)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    pred, pxy = model(c, cv, cxy)
                loss_lat = (pred.float() - t).abs().mean(-1)[m].mean()
                loss_xy = (pxy.float() - txy).abs().sum(-1)[m].mean()
                loss = loss_lat + args.xy_weight * loss_xy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                run += loss_lat.item()
            res = evaluate(model, val_dl, device, norm)
            res.update(epoch=ep, train_l1=run / len(train_dl))
            history.append(res)
            print(f"ep {ep:02d} train {res['train_l1']:.4f} | {line(res)}", flush=True)
            if res["l1/predictor"] < best:
                best = res["l1/predictor"]
                torch.save({"model": model.state_dict(), "mu": mu.cpu(), "sd": sd.cpu(), "cfg": vars(args),
                            "shape": {"Tc": Tc, "Tt": Tt, "K": K, "D": D}},
                           os.path.join(args.out, "predictor_best.pt"))
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
        print(f"best val L1 {best:.4f} -> {args.out}/predictor_best.pt", flush=True)

    if args.eval_features:
        ck = torch.load(os.path.join(args.out, "predictor_best.pt"), map_location="cpu")
        s, cfg = ck["shape"], ck["cfg"]
        model = ObjectPredictor(s["D"], s["Tc"], s["Tt"], s["K"], cfg["width"], cfg["depth"]).to(device)
        model.load_state_dict(ck["model"])
        mu, sd = ck["mu"].to(device), ck["sd"].to(device)
        files = sorted(glob.glob(os.path.join(args.eval_features, "*.pt")))
        ds = ObjectPairs(files, cfg["min_area"])
        res = evaluate(model, DataLoader(ds, 64), device, lambda x: (x - mu) / sd)
        print(f"eval on {args.eval_features} ({len(files)} videos, {res['n_scored']} object steps): {line(res)}")
        name = os.path.basename(os.path.normpath(args.eval_features))
        with open(os.path.join(args.out, f"eval_{name}.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
