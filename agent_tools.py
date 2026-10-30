"""Stage 4: local tools for the "what if" agent (no API calls here).

The language model (Muse Spark) only turns a question into tool calls and the
tool results into an answer; the physics comes from these tools, which run the
masks + V-JEPA 2 latents + object predictor locally.

Tools (see TOOLS for the JSON schemas given to the model):
  describe_objects(video)            -> slots with shape / colour (/ material)
  observed_collisions(video)         -> collisions seen in the real video
  what_if_remove(video, objects)     -> collisions that disappear / appear when
                                        objects are removed (delta rollout)
  predict_future(video)              -> collisions forecast after the video ends,
                                        ranked by score (predict_next.py)

Objects are referred to by slot number, as returned by describe_objects.
"""
import glob
import json
import os

import numpy as np
import torch

from extract_features import read_frames
from extract_object_features import clevrer_masks, rle_decode, sam_masks
from collision_head import PAIRS, head_logits, load_head, logit_events, observed
from predict_next import future_scores
from whatif_update import load_update, update_events
from whatif import collisions, load_model, observed_tracks, rollout

# mean RGB inside the true CLEVRER masks of each colour (measured on 10 videos)
REF_RGB = {"blue": (53, 68, 108), "brown": (93, 73, 45), "cyan": (53, 108, 108), "gray": (81, 81, 80),
           "green": (46, 80, 37), "purple": (93, 54, 109), "red": (102, 49, 49), "yellow": (123, 119, 61)}
_REF = {k: np.array(v, float) / sum(v) for k, v in REF_RGB.items()}  # chroma, brightness-free


# metal is shiny: uneven brightness inside the mask. Threshold on std/mean of max(R,G,B), set on
# 1,456 objects of val videos 200-500 with CLEVRER masks (accuracy 0.939), not on the SAM videos.
METAL_CV = 0.136


def material_name(frame, mask):
    v = frame[mask].astype(float).max(1)
    return "metal" if v.std() / max(v.mean(), 1.0) > METAL_CV else "rubber"


def colour_name(frame, mask):
    rgb = frame[mask].astype(float).mean(0)
    chroma = rgb / max(rgb.sum(), 1e-6)
    return min(_REF, key=lambda k: np.abs(_REF[k] - chroma).sum())


class Scene:
    """Everything the tools need for one video: masks, object features, predictor."""

    def __init__(self, video, features, ckpt, masks="clevrer",
                 root=os.path.expanduser("~/objectjepa_data"), device="cpu", head=None, update=None):
        self.video, self.masks = video, masks
        self.d = torch.load(os.path.join(features, f"{video}.pt"), map_location="cpu")
        self.model, self.mu, self.sd, self.cfg = load_model(ckpt, device)
        self.device = device
        # optional learned collision read-out (collision_head.py); default: distance rule
        self.head, self.thr = load_head(head, device) if head else (None, None)
        # optional learned what-if update (whatif_update.py, needs the head); default: delta rule
        self.update = load_update(update) if update and head else None
        if masks == "sam":
            self.objects, self.fmasks = sam_masks(os.path.join(root, "sam_masks"), video)
            path = glob.glob(os.path.join(root, "clevrer", "train", "**", f"{video}.mp4"), recursive=True)[0]
            frames = read_frames(path)
            for s, o in enumerate(self.objects):  # SAM gives the shape; estimate colour and material
                area, fi = max((rle_decode(fr[s]).sum(), i) for i, fr in enumerate(self.fmasks) if s in fr)
                m = rle_decode(self.fmasks[fi][s])
                o["color"], o["material"] = colour_name(frames[fi], m), material_name(frames[fi], m)
        else:
            self.objects, self.fmasks = clevrer_masks(os.path.join(root, "clevrer"), video)
        self._base = None

    def name(self, s):
        o = self.objects[s]
        return " ".join(x for x in (o.get("color"), o.get("material"), o.get("shape")) if x)

    def _observed(self):
        """(collision pairs, head logits or None) for the real video."""
        if self.head is None:
            return collisions(*observed_tracks(self.d, self.cfg["min_area"])), None
        z, area, xy = observed(self.d)
        lg = head_logits(self.head, (z.to(self.device) - self.mu) / self.sd, xy, area,
                         area > self.cfg["min_area"], False, self.device)
        return logit_events(lg, self.thr), lg

    def _rollout(self, removed=()):
        """(collision pairs, head logits or None) for a rollout with some objects removed."""
        r = rollout(self.model, self.mu, self.sd, self.d, self.cfg["min_area"], removed=removed,
                    device=self.device, return_latents=self.head is not None)
        if self.head is None:
            return collisions(*r), None
        lg = head_logits(self.head, r[3], *r[:3], True, self.device)
        return logit_events(lg, self.thr), lg

    def _events(self, pairs):
        return [{"objects": [a, b], "description": f"the {self.name(a)} collides with the {self.name(b)}"}
                for a, b in sorted(pairs)]

    # --- tools ---------------------------------------------------------------------------------
    def describe_objects(self):
        return [{"slot": s, "description": self.name(s), **{k: o[k] for k in ("color", "material", "shape") if k in o}}
                for s, o in enumerate(self.objects)]

    def observed_collisions(self):
        return self._events(self._observed()[0])

    def what_if_remove(self, objects):
        removed = [int(x) for x in objects]
        fact, lf = self._observed()
        if self._base is None:
            self._base = self._rollout()
        (base, lb), (cf, lc) = self._base, self._rollout(removed)
        if self.update is not None:
            after = update_events(self.update, lf, lb, lc, self.thr, removed)
        else:
            after = {e for e in (fact - (base - cf)) | (cf - base) if not set(e) & set(removed)}
        return {
            "removed": [self.name(s) for s in removed],
            "collisions_without_them": self._events(after),
            "no_longer_happen": self._events({e for e in fact if not set(e) & set(removed)} - after),
            "newly_happen": self._events(after - fact),
        }

    def predict_future(self):
        s = future_scores(self.model, self.mu, self.sd, self.d, self.cfg["min_area"], self.head, self.device)
        # head logit -> 0..1 score; without a head, closest approach (ratio 1 = touching) -> 0..1
        p = 1 / (1 + np.exp(-s)) if self.head is not None else 1 / (1 + np.exp(4 * (-s - 1.5)))
        order = [j for j in np.argsort(-s, kind="stable") if p[j] >= 0.01]  # raw scores: no ties from rounding
        return {"after_the_video": [e | {"rank": i + 1, "score": round(float(p[j]), 3)}
                                    for i, j in enumerate(order) for e in self._events([tuple(PAIRS[j].tolist())])],
                "note": "ranked from most to least likely to collide in the next 32 frames (score 0-1); "
                        "pairs not listed score < 0.01"}


TOOLS = [
    {"name": "describe_objects",
     "description": "List the objects in the video, with slot numbers and attributes (color, shape, material if known).",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "observed_collisions",
     "description": "Collisions between objects that happen in the real video.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "what_if_remove",
     "description": "Simulate the video with some objects removed from the start, using the learned world model. "
                    "Returns the collisions that would happen, those that no longer happen, and new ones.",
     "parameters": {"type": "object",
                    "properties": {"objects": {"type": "array", "items": {"type": "integer"},
                                               "description": "slot numbers from describe_objects"}},
                    "required": ["objects"]}},
    {"name": "predict_future",
     "description": "Forecast, with the learned world model, which collisions will happen after the video ends. "
                    "Returns object pairs ranked by a score from 0 to 1.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
]


def run_tool(scene, name, args):
    if name == "describe_objects":
        return scene.describe_objects()
    if name == "observed_collisions":
        return scene.observed_collisions()
    if name == "what_if_remove":
        return scene.what_if_remove(args["objects"])
    if name == "predict_future":
        return scene.predict_future()
    raise ValueError(f"unknown tool {name}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--masks", choices=["clevrer", "sam"], default="clevrer")
    ap.add_argument("--remove", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    sc = Scene(a.video, a.features, a.ckpt, a.masks)
    for tool, args in (("describe_objects", {}), ("observed_collisions", {}), ("what_if_remove", {"objects": a.remove})):
        print(f"--- {tool}({args})\n{json.dumps(run_tool(sc, tool, args), indent=1)}")
