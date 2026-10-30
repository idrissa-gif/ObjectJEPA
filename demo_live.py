"""Live demo: ask the agent one "what if" question and show each tool call as it happens.

    python demo_live.py                       # video_06926, question 14, Muse Spark
    python demo_live.py --qid 14 13 --wait    # "what if", then "what will happen next?" on the same scene
    python demo_live.py --qid 15              # another CLEVRER question of the same video
    python demo_live.py --question "What happens if the green sphere is removed?"
    python demo_live.py --offline             # rule-based stand-in, no API call
    python demo_live.py --wait                # load, then wait for Enter (for recording)

Uses agent.answer unchanged; a thin wrapper around the LLM prints the model's tool
calls and the tool results. For a CLEVRER question the ground truth is shown at
the end. The Muse Spark backend reads META_API_KEY from the environment only.
"""
import argparse
import json
import os
import sys
import time

from agent import ChatLLM, OfflineLLM, answer, parse_choices

C = {"dim": "\033[2m", "bold": "\033[1m", "blue": "\033[94m", "orange": "\033[38;5;215m",
     "green": "\033[92m", "red": "\033[91m", "end": "\033[0m"}


def paint(text, *styles):
    return "".join(C[s] for s in styles) + text + C["end"] if C else text


def show_result(name, res):
    if isinstance(res, dict) and "error" in res:
        print(paint(f"    error: {res['error']}", "red"))
    elif name == "describe_objects":
        for o in res:
            print(f"    slot {o['slot']}  {o['description']}")
    elif name == "observed_collisions":
        for e in res or [{"description": "(no collisions)"}]:
            print(f"    • {e['description']}")
    elif name == "what_if_remove":
        new = {tuple(e["objects"]) for e in res["newly_happen"]}
        print(f"    removed: {', '.join(res['removed']) or '(nothing)'}")
        for e in res["collisions_without_them"]:
            if tuple(e["objects"]) in new:
                print(paint(f"    + NEW   {e['description']}", "orange", "bold"))
            else:
                print(f"    • still {e['description']}")
        for e in res["no_longer_happen"]:
            print(paint(f"    − GONE  {e['description']}", "dim"))
    elif name == "predict_future":
        for e in res["after_the_video"][:5]:
            line = f"    {e['rank']}. {e['description']}  (score {e['score']:.2f})"
            print(paint(line, "orange", "bold") if e["rank"] == 1 else line)
        if not res["after_the_video"]:
            print("    (no collision expected)")
    else:
        print(f"    {json.dumps(res)}")


class Narrated:
    """Wraps an llm(messages, tools) callable and prints what goes in and out."""

    def __init__(self, llm, label, pause):
        self.llm, self.label, self.pause, self.seen = llm, label, pause, 0

    def __call__(self, messages, tools):
        for m in messages[self.seen:]:
            if m["role"] == "tool":
                print(paint(f"  ← {m['name']}", "blue"))
                show_result(m["name"], json.loads(m["content"]))
                time.sleep(self.pause)
        print(paint(f"\n{self.label} is thinking…", "dim"))
        t = time.time()
        out = self.llm(messages, tools)
        self.seen = len(messages)
        for c in out.get("tool_calls", []):
            args = json.dumps(c.get("arguments") or {})
            print(paint(f"  → {c['name']}", "orange", "bold") + paint(f" {args}  ({time.time() - t:.1f} s)", "dim"))
        return out


if __name__ == "__main__":
    from agent_tools import Scene

    D = os.path.expanduser("~/objectjepa_data")
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="video_06926")
    ap.add_argument("--qid", type=int, nargs="+", default=[14], help="CLEVRER question id(s) in this video, "
                    "asked one after the other (e.g. 14 13: what if, then what next)")
    ap.add_argument("--question", default=None, help="free-form question instead of --qid")
    ap.add_argument("--masks", choices=["clevrer", "sam"], default="sam")
    ap.add_argument("--features", default=os.path.join(D, "obj_features_sam"))
    ap.add_argument("--ckpt", default=os.path.join(D, "ckpt_obj_e60_w512d8"))
    ap.add_argument("--questions", default=os.path.join(D, "clevrer/questions/train.json"))
    ap.add_argument("--head", default=os.path.join(D, "ckpt_collision"), help="collision head ('' = distance rule)")
    ap.add_argument("--update", default=os.path.join(D, "ckpt_update"), help="learned what-if update ('' = delta rule)")
    ap.add_argument("--offline", action="store_true", help="rule-based parser, no API call")
    ap.add_argument("--base-url", default=os.environ.get("META_BASE_URL", "https://api.meta.ai/v1"))
    ap.add_argument("--model", default=os.environ.get("META_MODEL", "muse-spark-1.3"))
    ap.add_argument("--pause", type=float, default=0.6, help="seconds between printed tool results")
    ap.add_argument("--wait", action="store_true", help="load first, then wait for Enter (start recording meanwhile)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color or not sys.stdout.isatty():
        C = {}

    asks = []  # (question, choices or None, truth or None)
    if args.question:
        asks.append((args.question, None, None))
    else:
        qs = next(v for v in json.load(open(args.questions)) if v["video_filename"] == f"{args.video}.mp4")
        for qid in args.qid:
            q = next(q for q in qs["questions"] if q["question_id"] == qid)
            asks.append((q["question"], [c["choice"] for c in q.get("choices", [])] or None,
                         [c["answer"] == "correct" for c in q.get("choices", [])]))

    print(paint(f"Loading {args.video} ({args.masks.upper()} masks, world model {os.path.basename(args.ckpt)})…", "dim"))
    scene = Scene(args.video, args.features, args.ckpt, args.masks, head=args.head or None, update=args.update or None)
    llm = OfflineLLM() if args.offline else ChatLLM(args.base_url, args.model)
    label = "offline parser" if args.offline else args.model

    for n, (question, choices, truth) in enumerate(asks):
        if args.wait:
            input(paint("Ready. Press Enter to ask the question." if n == 0 else "\nPress Enter for the next question.", "dim"))
            print("\033[2J\033[H" if C else "", end="")  # clear the screen
        used = (getattr(llm, "tokens_in", 0), getattr(llm, "tokens_out", 0))
        print(paint(f"\nQ: {question}", "bold"))
        for i, c in enumerate(choices or []):
            print(f"   {i}. {c}")
        reply = answer(question, scene, Narrated(llm, label, args.pause), choices)

        print(paint(f"\n{label} answers:", "bold"))
        if choices:
            for i, (c, yes) in enumerate(zip(choices, parse_choices(reply, len(choices)))):
                mark = paint("✓", "green") if yes == truth[i] else paint("✗", "red")
                print(f"   {mark} {i}. {'yes' if yes else 'no '}  {c}")
            print(paint(f"   (✓ = matches the CLEVRER ground truth)", "dim"))
        else:
            print("   " + reply.replace("\n", "\n   "))
        if not args.offline:
            t_in, t_out = llm.tokens_in - used[0], llm.tokens_out - used[1]
            print(paint(f"\n{t_in} tokens in, {t_out} out, ≈ ${(t_in * 1.25 + t_out * 4.25) / 1e6:.3f}", "dim"))
