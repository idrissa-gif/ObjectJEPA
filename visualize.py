"""Demo video: object masks + actual vs predicted object paths (+ optional "what if").

Frames 0-31 are observed (context). From frame 32 on, the object predictor rolls
forward (whatif.rollout) and the video shows, for each object (outline and label
in the object's own colour):
  solid line  : actual centroid path (from the masks)
  dotted line : predicted path
  star        : predicted collision
  with --remove: the removed object is faded out of the frames with a dashed
  outline, the dotted lines are the counterfactual paths of the others, and the
  last frames show the answer of the what_if_remove tool (the one the agent uses).
  with --future: the real video plays to its end (frame 127), then the last frame
  freezes and the predictor keeps going (frames 128-159, predict_next.py): moving
  rings are the predicted objects, stars the collisions predicted with score >= 0.5,
  and the end card shows the ranking of the predict_future tool.

python visualize.py --video video_06926 --masks sam --features obj_features_sam --ckpt ckpt_obj --remove 2 --out demo.mp4
python visualize.py --video video_07635 --masks sam --features obj_features_sam --ckpt ckpt_obj --head ckpt_collision --future --out next.mp4
"""
import argparse
import os
import textwrap

import av
import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agent_tools import Scene
from collision_head import PAIRS, head_logits
from predict_next import WINDOW, last_clip
from extract_features import read_frames
from extract_object_features import rle_decode
from whatif import C, PATCH_PX, W_PX, H_PX, collisions, observed_tracks, rollout

# display colours for the CLEVRER colour names (brighter than the rendered ones, for outlines and labels)
COLOURS = {"blue": (50, 115, 255), "brown": (190, 125, 60), "cyan": (0, 225, 225), "gray": (205, 205, 205),
           "green": (70, 210, 80), "purple": (185, 95, 245), "red": (245, 60, 60), "yellow": (250, 225, 40)}
STEP = 4  # frames per tubelet step
CTX = 8  # observed steps (frames 0-31)
SCALE = 2  # output is 2x the 480x320 CLEVRER frames, so the text stays readable
BAR = 112  # height of the legend bar under the video
MAX_JUMP = 60  # px per step; longer jumps in a mask track are detection glitches, not drawn as lines
WHITE, DARK = (255, 255, 255), (24, 24, 28)


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", name), size)


F_LABEL, F_TEXT, F_BOLD, F_BIG = font(15), font(17), font(19, True), font(24, True)


def pt(p):
    return tuple(int(round(v * SCALE)) for v in p)


def draw_path(img, pts, color, dotted=False):
    if not dotted:
        for a, b in zip(pts[:-1], pts[1:]):
            if np.linalg.norm(b - a) < MAX_JUMP:
                cv2.line(img, pt(a), pt(b), color, 3, cv2.LINE_AA)
        return
    for p in pts[1:]:
        cv2.circle(img, pt(p), 6, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, pt(p), 4, color, -1, cv2.LINE_AA)


def dashed(img, cnt, color, on=8, off=6):
    p = cnt[:, 0] * SCALE
    for i in range(0, len(p), on + off):
        cv2.polylines(img, [p[i : i + on]], False, color, 3, cv2.LINE_AA)


def fade_out(frame, mask, keep=0.3):
    """Paint the floor over an object (inpainting), leaving a faint ghost."""
    m = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8))
    inp = cv2.inpaint(frame, m, 5, cv2.INPAINT_TELEA)
    out = frame.copy()
    out[m > 0] = (keep * frame[m > 0] + (1 - keep) * inp[m > 0]).astype(np.uint8)
    return out


def text_layer(img, items):
    """items: (xy, text, font, fill, anchor, box) -- box draws a dark rounded background."""
    base = Image.fromarray(img).convert("RGBA")
    over = Image.new("RGBA", base.size)
    d = ImageDraw.Draw(over)
    for xy, text, f, fill, anchor, box in items:
        if box:
            l, t, r, b = d.textbbox(xy, text, font=f, anchor=anchor)
            d.rounded_rectangle((l - 5, t - 3, r + 5, b + 3), 5, fill=(0, 0, 0, 170))
        d.text(xy, text, font=f, fill=fill, anchor=anchor)
    return np.array(Image.alpha_composite(base, over).convert("RGB"))


def legend(img, y, removed):
    """Line, dots, star and (with --remove) dashed-outline glyphs, returns the text items."""
    x, items = 16, []
    entries = [("line", "actual path (real video)"),
               ("dots", "predicted without it" if removed else "predicted by ObjectJEPA"),
               ("star", "predicted collision")] + ([("dash", "removed object")] if removed else [])
    for kind, label in entries:
        c = (200, 200, 200)
        if kind == "line":
            cv2.line(img, (x, y), (x + 34, y), c, 3, cv2.LINE_AA)
        elif kind == "dots":
            for k in range(3):
                cv2.circle(img, (x + 5 + 12 * k, y), 6, WHITE, -1, cv2.LINE_AA)
                cv2.circle(img, (x + 5 + 12 * k, y), 4, c, -1, cv2.LINE_AA)
        elif kind == "star":
            cv2.drawMarker(img, (x + 17, y), WHITE, cv2.MARKER_STAR, 22, 2, cv2.LINE_AA)
        else:
            for k in range(3):
                cv2.line(img, (x + 12 * k, y), (x + 12 * k + 7, y), WHITE, 3, cv2.LINE_AA)
        items.append(((x + 44, y), label, F_TEXT, (225, 225, 225), "lm", False))
        x += 44 + int(F_TEXT.getlength(label)) + 30
    return items


def answer_lines(sc, removed, real, pred):
    if removed:
        res = sc.what_if_remove(removed)
        lines = [f"Without the {' and the '.join(res['removed'])}:"]
        lines += [f"+ new: {e['description']}" for e in res["newly_happen"]]
        lines += [f"- no longer: {e['description']}" for e in res["no_longer_happen"]]
        if len(lines) == 1:
            lines.append("no change in which objects collide")
        return lines
    # collisions after the observed part only: actual (masks) vs predicted (rollout)
    act, prd = (collisions(x[CTX:], a[CTX:], v[CTX:]) for x, a, v in (real, pred))
    tag = {(1, 1): "\u2713 predicted, and it happens", (1, 0): "\u2717 predicted, but it doesn't happen",
           (0, 1): "\u2717 missed"}
    return ["Collisions after frame 32"] + [f"{tag[e in prd, e in act]}: the {sc.name(e[0])} + the {sc.name(e[1])}"
                                            for e in sorted(act | prd)] + (["none"] if not act | prd else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="e.g. video_09770")
    ap.add_argument("--root", default=os.path.expanduser("~/objectjepa_data"),
                    help="holds clevrer/ (videos, annotations, proposals) and sam_masks/")
    ap.add_argument("--masks", choices=["clevrer", "sam"], default="clevrer")
    ap.add_argument("--features", required=True, help="object features made with the same --masks")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--remove", type=int, nargs="*", default=[], help="slot(s) to remove")
    ap.add_argument("--question", default=None, help="text for the legend bar (default: made from --remove)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--hold", type=float, default=4.0, help="seconds to show the answer at the end")
    ap.add_argument("--future", action="store_true", help="forecast after the end of the video (needs --head)")
    ap.add_argument("--head", default=None, help="collision head checkpoint dir (for --future)")
    args = ap.parse_args()

    sc = Scene(args.video, args.features, args.ckpt, args.masks, root=args.root, head=args.head)
    path = next(os.path.join(r, f) for r, _, fs in os.walk(os.path.join(args.root, "clevrer"))
                for f in fs if f == f"{args.video}.mp4")
    frames = read_frames(path)
    if args.future:
        return future_video(args, sc, frames)
    objects, fmasks, removed = sc.objects, sc.fmasks, args.remove
    colour = [COLOURS.get(o.get("color"), WHITE) for o in objects]
    min_area = sc.cfg["min_area"]
    real = observed_tracks(sc.d, min_area)
    pred = rollout(sc.model, sc.mu, sc.sd, sc.d, min_area, removed=removed)
    scale = np.array([W_PX, H_PX])
    real_xy, real_vis = real[0] * scale, real[2]
    pred_xy, pred_vis = pred[0] * scale, pred[2]
    r = np.sqrt(np.maximum(pred[1], 0) * PATCH_PX / np.pi)
    question = args.question or (f"What if the {' and the '.join(sc.name(s) for s in removed)} were removed?"
                                 if removed else "Where will the objects go? (prediction starts at frame 32)")
    answer = answer_lines(sc, removed, real, pred)

    W, H = W_PX * SCALE, H_PX * SCALE
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with av.open(args.out, "w") as out:
        st = out.add_stream("libx264", rate=args.fps)
        st.width, st.height, st.pix_fmt = W, H + BAR, "yuv420p"
        st.options = {"crf": "18"}

        def emit(img):
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)

        for fi in range(0, len(frames), 2):
            t = fi // STEP
            frame = np.ascontiguousarray(frames[fi])
            for s in removed:
                if s in fmasks[fi]:
                    frame = fade_out(frame, rle_decode(fmasks[fi][s]))
            img = np.zeros((H + BAR, W, 3), np.uint8)
            img[:H] = cv2.resize(frame, (W, H), interpolation=cv2.INTER_CUBIC)
            img[H:] = DARK
            items = []
            for s, rle in fmasks[fi].items():  # outlines + name labels
                m = rle_decode(rle).astype(np.uint8)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if s in removed:
                    for c in cnts:
                        dashed(img, c, WHITE)
                else:
                    cv2.drawContours(img, [c * SCALE for c in cnts], -1, colour[s], 2, cv2.LINE_AA)
                ys, xs = np.nonzero(m)
                lx = min(max(int(xs.mean() * SCALE), 90), W - 90)
                if s in removed:  # below the object, so it doesn't hide the paths of the others
                    items.append(((lx, min(int(ys.max() * SCALE) + 10, H - 40)), f"removed: {sc.name(s)}", F_LABEL,
                                  WHITE, "mt", True))
                else:
                    items.append(((lx, max(int(ys.min() * SCALE) - 8, 52)), sc.name(s), F_LABEL, colour[s], "mb", True))
            for s in range(len(objects)):  # paths
                if s in removed:
                    continue
                seen = [p for k, p in enumerate(real_xy[: t + 1, s]) if real_vis[k, s]]
                draw_path(img, seen, colour[s])
                if t >= CTX and pred_vis[t, s]:
                    draw_path(img, pred_xy[CTX - 1 : t + 1, s], colour[s], dotted=True)
            if t >= CTX:  # predicted collisions at this step
                for a in range(len(objects)):
                    for b in range(a + 1, len(objects)):
                        if pred_vis[t, a] and pred_vis[t, b] and \
                                np.linalg.norm(pred_xy[t, a] - pred_xy[t, b]) < C * (r[t, a] + r[t, b]):
                            mid = pt((pred_xy[t, a] + pred_xy[t, b]) / 2)
                            cv2.drawMarker(img, mid, (0, 0, 0), cv2.MARKER_STAR, 34, 6, cv2.LINE_AA)
                            cv2.drawMarker(img, mid, WHITE, cv2.MARKER_STAR, 34, 3, cv2.LINE_AA)
            # progress bar: observed part in blue, playhead in white
            y0 = H - 8
            cv2.rectangle(img, (0, y0), (W, H), (60, 60, 66), -1)
            cv2.rectangle(img, (0, y0), (W * CTX * STEP // len(frames), H), (50, 115, 255), -1)
            cv2.rectangle(img, (W * fi // len(frames) - 2, y0 - 4), (W * fi // len(frames) + 2, H), WHITE, -1)
            phase = "OBSERVED" if t < CTX else ("WHAT IF  (predicted)" if removed else "PREDICTED by ObjectJEPA")
            items.append(((12, 12), f"{phase}   frame {fi:3d} / {len(frames) - 1}", F_BOLD,
                          (120, 170, 255) if t < CTX else (255, 210, 90), "lt", True))
            items.append(((16, H + 26), question, F_BOLD, WHITE, "lm", False))
            items += legend(img, H + 72, removed)
            img = text_layer(img, items)
            emit(img)

        # end card: the answer on a panel over the darkened last frame
        img[:H] = (img[:H] * 0.3).astype(np.uint8)
        lines = [l for a in answer for l in textwrap.wrap(a, 80, subsequent_indent="    ")]
        y = H // 2 - 17 * len(lines)
        cv2.rectangle(img, (50, y - 70), (W - 50, y + 34 * len(lines) + 10), DARK, -1)
        cv2.rectangle(img, (50, y - 70), (W - 50, y + 34 * len(lines) + 10), (255, 210, 90), 2)
        items = [((W // 2, y - 36), "ObjectJEPA answer", F_BIG, (255, 210, 90), "mm", False)]
        items += [((80, y + 8 + 34 * i), l, F_TEXT if i else F_BOLD, WHITE, "lm", False) for i, l in enumerate(lines)]
        img = text_layer(img, items)
        for _ in range(int(args.hold * args.fps)):
            emit(img)
        for pkt in st.encode():
            out.mux(pkt)
    print(f"wrote {args.out}")


def future_video(args, sc, frames, per_step=4):
    """Real video to the end, then the frozen last frame with the forecast of frames 128-159."""
    objects, fmasks, min_area = sc.objects, sc.fmasks, sc.cfg["min_area"]
    colour = [COLOURS.get(o.get("color"), WHITE) for o in objects]
    scale = np.array([W_PX, H_PX])
    real = observed_tracks(sc.d, min_area)
    real_xy, real_vis = real[0] * scale, real[2]
    xy, area, vis, z = rollout(sc.model, sc.mu, sc.sd, last_clip(sc.d), min_area, return_latents=True)
    fut_xy, fut_vis = xy[CTX - 1:CTX + WINDOW] * scale, vis[CTX - 1:CTX + WINDOW]  # last real step + forecast
    r = np.sqrt(np.maximum(area[CTX:CTX + WINDOW], 0) * PATCH_PX / np.pi)
    ranking = sc.predict_future()["after_the_video"]
    lg = head_logits(sc.head, z, xy, area, vis, True)[CTX:CTX + WINDOW]
    pair_idx = {tuple(p): i for i, p in enumerate(PAIRS.tolist())}
    hits = [(tuple(e["objects"]), int(lg[:, pair_idx[tuple(e["objects"])]].argmax()))  # star where the head is surest
            for e in ranking if e["score"] >= 0.5]
    question = args.question or "What will happen next?"
    n_real, n_all = len(frames), len(frames) + WINDOW * STEP
    W, H = W_PX * SCALE, H_PX * SCALE

    def outlines(img, fi):
        items = []
        for s, rle in fmasks[fi].items():
            m = rle_decode(rle).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, [c * SCALE for c in cnts], -1, colour[s], 2, cv2.LINE_AA)
            ys, xs = np.nonzero(m)
            lx = min(max(int(xs.mean() * SCALE), 90), W - 90)
            items.append(((lx, max(int(ys.min() * SCALE) - 8, 52)), sc.name(s), F_LABEL, colour[s], "mb", True))
        return items

    def chrome(img, items, frame_no, future):
        y0 = H - 8
        cv2.rectangle(img, (0, y0), (W, H), (60, 60, 66), -1)
        cv2.rectangle(img, (0, y0), (W * n_real // n_all, H), (50, 115, 255), -1)
        if future:
            cv2.rectangle(img, (W * n_real // n_all, y0), (W * frame_no // n_all, H), (255, 210, 90), -1)
        x = W * frame_no // n_all
        cv2.rectangle(img, (x - 2, y0 - 4), (x + 2, H), WHITE, -1)
        phase = "PREDICTED after the video ends" if future else "REAL VIDEO"
        items.append(((12, 12), f"{phase}   frame {frame_no:3d}", F_BOLD,
                      (255, 210, 90) if future else (120, 170, 255), "lt", True))
        items.append(((16, H + 26), question, F_BOLD, WHITE, "lm", False))
        items += legend_future(img, H + 72)
        return text_layer(img, items)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with av.open(args.out, "w") as out:
        st = out.add_stream("libx264", rate=args.fps)
        st.width, st.height, st.pix_fmt = W, H + BAR, "yuv420p"
        st.options = {"crf": "18"}

        def emit(img):
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)

        for fi in range(0, n_real, 2):  # the real video, with the actual paths
            t = fi // STEP
            img = np.zeros((H + BAR, W, 3), np.uint8)
            img[:H] = cv2.resize(np.ascontiguousarray(frames[fi]), (W, H), interpolation=cv2.INTER_CUBIC)
            img[H:] = DARK
            items = outlines(img, fi)
            for s in range(len(objects)):
                draw_path(img, [p for k, p in enumerate(real_xy[: t + 1, s]) if real_vis[k, s]], colour[s])
            emit(chrome(img, items, fi, False))

        last = (cv2.resize(np.ascontiguousarray(frames[-1]), (W, H), interpolation=cv2.INTER_CUBIC) * 0.55).astype(np.uint8)
        for k in range(1, WINDOW * per_step + 1):  # the forecast, interpolated inside each 4-frame step
            u = k / per_step  # in steps after the last real step
            i0 = min(int(u), WINDOW - 1)
            img = np.zeros((H + BAR, W, 3), np.uint8)
            img[:H], img[H:] = last, DARK
            items = []
            for s in range(len(objects)):
                draw_path(img, [p for kk, p in enumerate(real_xy[:, s]) if real_vis[kk, s]], colour[s])
                if not fut_vis[1, s]:
                    continue
                draw_path(img, fut_xy[: int(u) + 1, s], colour[s], dotted=True)
                p = fut_xy[i0, s] + (u - i0) * (fut_xy[min(i0 + 1, WINDOW), s] - fut_xy[i0, s])
                cv2.circle(img, pt(p), int(r[min(i0, WINDOW - 1), s] * SCALE), colour[s], 3, cv2.LINE_AA)
                items.append(((int(p[0] * SCALE), max(int((p[1] - r[i0, s]) * SCALE) - 8, 52)), sc.name(s),
                              F_LABEL, colour[s], "mb", True))
            for (a, b), t_hit in hits:
                if u >= t_hit + 1:
                    mid = pt((fut_xy[t_hit + 1, a] + fut_xy[t_hit + 1, b]) / 2)
                    cv2.drawMarker(img, mid, (0, 0, 0), cv2.MARKER_STAR, 40, 7, cv2.LINE_AA)
                    cv2.drawMarker(img, mid, WHITE, cv2.MARKER_STAR, 40, 3, cv2.LINE_AA)
            emit(chrome(img, items, n_real - 1 + int(round(u * STEP)), True))

        answer = ["Forecast after the end of the video (predict_future):"]
        answer += [f"{e['rank']}. {e['description']}  (score {e['score']:.2f})" for e in ranking[:3]] or ["no collision"]
        img[:H] = (img[:H] * 0.3).astype(np.uint8)
        y = H // 2 - 17 * len(answer)
        cv2.rectangle(img, (50, y - 70), (W - 50, y + 34 * len(answer) + 10), DARK, -1)
        cv2.rectangle(img, (50, y - 70), (W - 50, y + 34 * len(answer) + 10), (255, 210, 90), 2)
        items = [((W // 2, y - 36), "ObjectJEPA forecast", F_BIG, (255, 210, 90), "mm", False)]
        items += [((80, y + 8 + 34 * i), l, F_TEXT if i else F_BOLD, WHITE, "lm", False) for i, l in enumerate(answer)]
        img = text_layer(img, items)
        for _ in range(int(args.hold * args.fps)):
            emit(img)
        for pkt in st.encode():
            out.mux(pkt)
    print(f"wrote {args.out}")


def legend_future(img, y):
    x, items = 16, []
    for kind, label in (("line", "actual path"), ("dots", "forecast path"),
                        ("ring", "forecast object"), ("star", "forecast collision")):
        c = (200, 200, 200)
        if kind == "line":
            cv2.line(img, (x, y), (x + 34, y), c, 3, cv2.LINE_AA)
        elif kind == "dots":
            for k in range(3):
                cv2.circle(img, (x + 5 + 12 * k, y), 6, WHITE, -1, cv2.LINE_AA)
                cv2.circle(img, (x + 5 + 12 * k, y), 4, c, -1, cv2.LINE_AA)
        elif kind == "ring":
            cv2.circle(img, (x + 17, y), 10, c, 3, cv2.LINE_AA)
        else:
            cv2.drawMarker(img, (x + 17, y), WHITE, cv2.MARKER_STAR, 22, 2, cv2.LINE_AA)
        items.append(((x + 44, y), label, F_TEXT, (225, 225, 225), "lm", False))
        x += 44 + int(F_TEXT.getlength(label)) + 30
    return items


if __name__ == "__main__":
    main()
