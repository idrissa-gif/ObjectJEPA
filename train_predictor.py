"""Stage 2: train a JEPA-style latent predictor on frozen V-JEPA 2 features.

Given context latents ctx [T', P, D] predict future latents tgt [T', P, D].
Loss: L1 in latent space (as in V-JEPA). The encoder stays frozen, so the
targets cannot collapse.

Baselines reported on the same validation split (the success criterion
is to beat them):
  copy_last : repeat the last context step for every future step
  linear    : extrapolate last + (last - prev) * k
  mean      : mean of all context steps

One GPU : python train_predictor.py --features feats
Two GPUs: torchrun --nproc_per_node=2 train_predictor.py --features feats
"""
import argparse
import glob
import json
import math
import os
import random

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class LatentPairs(Dataset):
    def __init__(self, files):
        self.items = []
        for f in files:
            d = torch.load(f, map_location="cpu")
            for s in range(d["ctx"].shape[0]):
                self.items.append((d["ctx"][s], d["tgt"][s]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        c, t = self.items[i]
        return c.float(), t.float()


class Predictor(nn.Module):
    """Transformer over (context tokens + learnable future query tokens)."""

    def __init__(self, d_in, t_ctx, t_tgt, n_patch, width=384, depth=6, heads=None, dropout=0.1):
        super().__init__()
        self.t_tgt, self.n_patch = t_tgt, n_patch
        heads = heads or max(1, width // 64)
        self.inp = nn.Linear(d_in, width)
        self.time_pos = nn.Parameter(torch.zeros(t_ctx + t_tgt, 1, width))
        self.space_pos = nn.Parameter(torch.zeros(1, n_patch, width))
        self.query = nn.Parameter(torch.zeros(t_tgt, n_patch, width))
        for p in (self.time_pos, self.space_pos, self.query):
            nn.init.trunc_normal_(p, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, d_in)
        nn.init.zeros_(self.out.weight)  # residual starts at zero
        nn.init.zeros_(self.out.bias)

    def forward(self, ctx):  # ctx [B, Tc, P, D]
        B, Tc, P, _ = ctx.shape
        x = self.inp(ctx)
        q = self.query.unsqueeze(0).expand(B, -1, -1, -1)
        x = torch.cat([x, q], 1) + self.time_pos + self.space_pos  # [B, Tc+Tt, P, W]
        x = self.blocks(x.flatten(1, 2))
        x = self.out(self.norm(x)).view(B, Tc + self.t_tgt, P, -1)
        # residual on the last observed step: starts as the copy_last baseline
        # and only has to learn the change (motion, collisions)
        return ctx[:, -1:] + x[:, Tc:]


def baselines(ctx, t_tgt):
    last, prev = ctx[:, -1:], ctx[:, -2:-1]
    k = torch.arange(1, t_tgt + 1, device=ctx.device).view(1, -1, 1, 1).float()
    return {
        "copy_last": last.expand(-1, t_tgt, -1, -1),
        "linear": last + (last - prev) * k,
        "mean": ctx.mean(1, keepdim=True).expand(-1, t_tgt, -1, -1),
    }


@torch.no_grad()
def evaluate(model, loader, device, norm):
    model.eval()
    tot = {"predictor": 0.0, "copy_last": 0.0, "linear": 0.0, "mean": 0.0}
    cos = {"predictor": 0.0, "copy_last": 0.0}
    n = 0
    for c, t in loader:
        c, t = norm(c.to(device)), norm(t.to(device))
        preds = {"predictor": model(c), **baselines(c, t.shape[1])}
        for k, p in preds.items():
            tot[k] += F.l1_loss(p, t, reduction="sum").item() / t[0].numel()
            if k in cos:
                cos[k] += F.cosine_similarity(p, t, dim=-1).mean(dim=(1, 2)).sum().item()
        n += c.shape[0]
    model.train()
    res = {f"l1/{k}": v / n for k, v in tot.items()}
    res.update({f"cos/{k}": v / n for k, v in cos.items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_proc = rank == 0
    log = print if main_proc else (lambda *a, **k: None)
    random.seed(args.seed)  # same seed on every rank -> same train/val split
    torch.manual_seed(args.seed)

    files = sorted(glob.glob(os.path.join(args.features, "*.pt")))
    assert files, f"no feature files in {args.features}"
    random.shuffle(files)  # split by VIDEO, so no clip of a val video is seen in training
    n_val = max(1, int(len(files) * args.val_frac))
    train_ds, val_ds = LatentPairs(files[n_val:]), LatentPairs(files[:n_val])
    log(f"train {len(train_ds)} pairs / val {len(val_ds)} pairs from {len(files)} videos")

    c0, t0 = train_ds[0]
    Tc, P, D = c0.shape
    Tt = t0.shape[0]

    # per-channel normalisation from training targets
    stack = torch.stack([train_ds[i][1] for i in range(min(len(train_ds), 2000))])
    mu = stack.mean(dim=(0, 1, 2)).to(device)
    sd = stack.std(dim=(0, 1, 2)).clamp_min(1e-4).to(device)
    norm = lambda x: (x - mu) / sd

    sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if ddp else None
    train_dl = DataLoader(train_ds, args.bs, shuffle=sampler is None, sampler=sampler,
                          num_workers=4, drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, args.bs, num_workers=4)

    net = Predictor(D, Tc, Tt, P, args.width, args.depth).to(device)
    model = DDP(net, device_ids=[device.index]) if ddp else net
    log(f"predictor params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * len(train_dl)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    )

    os.makedirs(args.out, exist_ok=True)
    best, history = float("inf"), []
    for ep in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        run = 0.0
        for c, t in train_dl:
            c, t = norm(c.to(device)), norm(t.to(device))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.l1_loss(model(c).float(), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
        if ddp:
            t_run = torch.tensor(run, device=device)
            dist.all_reduce(t_run)
            run = t_run.item() / world
        if not main_proc:
            continue
        res = evaluate(net, val_dl, device, norm)
        res.update(epoch=ep, train_l1=run / len(train_dl))
        history.append(res)
        beats = res["l1/predictor"] < min(res["l1/copy_last"], res["l1/linear"], res["l1/mean"])
        print(
            f"ep {ep:02d} train {res['train_l1']:.4f} | val predictor {res['l1/predictor']:.4f} "
            f"copy_last {res['l1/copy_last']:.4f} linear {res['l1/linear']:.4f} mean {res['l1/mean']:.4f} "
            f"| cos {res['cos/predictor']:.3f} vs {res['cos/copy_last']:.3f} {'BEATS' if beats else ''}",
            flush=True,
        )
        if res["l1/predictor"] < best:
            best = res["l1/predictor"]
            torch.save(
                {"model": net.state_dict(), "mu": mu.cpu(), "sd": sd.cpu(), "cfg": vars(args),
                 "shape": {"Tc": Tc, "Tt": Tt, "P": P, "D": D}},
                os.path.join(args.out, "predictor_best.pt"),
            )
    if main_proc:
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
        print(f"best val L1 {best:.4f} -> {args.out}/predictor_best.pt")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
