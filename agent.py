"""Stage 4: the "what if" agent loop.

    answer(question, scene, llm, choices=None)

The language model gets the question (and multiple-choice options, if any) and
the tool schemas from agent_tools.TOOLS. It may call tools; their JSON results
are fed back until it answers. `llm(messages, tools)` must return either
{"tool_calls": [{"name": ..., "arguments": {...}}]} or {"content": "..."}.

Backends:
  OfflineLLM : no API. Rule-based stand-in (attribute word matching) used to
               test the whole loop for free and as a baseline for the real model.
  Muse Spark : added once the endpoint is known. The API key is read from an
               environment variable only; it is never written to code or logs.
"""
import json
import re

from agent_tools import TOOLS, run_tool

SYSTEM = (
    "You answer questions about a short physics video of objects sliding and colliding. "
    "Use the tools: describe_objects to see the objects and their slot numbers, observed_collisions "
    "for what really happens, what_if_remove to simulate the video without some objects. "
    "For multiple-choice questions, answer with one line per option: '<option number>: yes' or '<option number>: no'."
)

ATTRS = ("color", "material", "shape")


def match_objects(text, objects):
    """Slots whose attributes all appear in the text, for the most specific description found."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    words |= {w[:-1] for w in words if w.endswith("s")}  # plurals
    best, best_n = [], 0
    for o in objects:
        named = [o[k] for k in ATTRS if k in o and o[k] in words]
        if named and len(named) >= best_n:
            if len(named) > best_n:
                best = []
            best, best_n = best + [o["slot"]], len(named)
    return best


class OfflineLLM:
    """Stand-in for the language model: handles '... if/without <object> ... removed' questions."""

    def __call__(self, messages, tools):
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        question = next(m["content"] for m in messages if m["role"] == "user")
        if not tool_msgs:
            return {"tool_calls": [{"name": "describe_objects", "arguments": {}}]}
        if len(tool_msgs) == 1:
            objects = json.loads(tool_msgs[0]["content"])
            head = question.split("\n")[0]
            m = re.search(r"(?:if|without) (?:the )?(.+?)(?: is removed|,|\?|$)", head.lower())
            target = match_objects(m.group(1), objects) if m else []
            return {"tool_calls": [{"name": "what_if_remove", "arguments": {"objects": target[:1]}}]}
        objects = json.loads(tool_msgs[0]["content"])
        result = json.loads(tool_msgs[-1]["content"])
        happen = {tuple(sorted(e["objects"])) for e in result["collisions_without_them"]}
        negate = " not " in question.split("\n")[0].lower()
        lines = []
        for line in question.split("\n")[1:]:
            mm = re.match(r"(\d+)\. (.+)", line.strip())
            if not mm:
                continue
            parts = re.split(r" collides with | and | collide", mm.group(2).lower())
            refs = [match_objects(p, objects) for p in parts[:2]]
            pair = tuple(sorted((refs[0][0], refs[1][0]))) if all(len(r) == 1 for r in refs) else None
            yes = (pair in happen) != negate if pair else False
            lines.append(f"{mm.group(1)}: {'yes' if yes else 'no'}")
        return {"content": "\n".join(lines)}


def answer(question, scene, llm, choices=None, max_steps=6, log=None):
    if choices:
        question += "\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(choices))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for _ in range(max_steps):
        out = llm(messages, TOOLS)
        if "tool_calls" not in out:
            if log is not None:
                log.extend(messages + [{"role": "assistant", "content": out["content"]}])
            return out["content"]
        messages.append({"role": "assistant", "tool_calls": out["tool_calls"]})
        for call in out["tool_calls"]:
            res = run_tool(scene, call["name"], call.get("arguments", {}))
            messages.append({"role": "tool", "name": call["name"], "content": json.dumps(res)})
    return ""


def parse_choices(text, n):
    """'<i>: yes/no' lines -> list of bools (False when missing)."""
    got = dict(re.findall(r"(\d+)\s*:\s*(yes|no)", text.lower()))
    return [got.get(str(i)) == "yes" for i in range(n)]


if __name__ == "__main__":
    import argparse
    import os

    from agent_tools import Scene

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--masks", choices=["clevrer", "sam"], default="clevrer")
    ap.add_argument("--questions", default=os.path.expanduser("~/objectjepa_data/clevrer/questions/train.json"))
    ap.add_argument("--videos", required=True, help="text file with video names (use val videos)")
    ap.add_argument("--limit", type=int, default=0, help="max questions")
    args = ap.parse_args()
    qs = {q["video_filename"][:-4]: q for q in json.load(open(args.questions))}
    llm, n_opt, ok_opt, n_q, ok_q = OfflineLLM(), 0, 0, 0, 0
    for name in (l.strip() for l in open(args.videos)):
        if not os.path.exists(os.path.join(args.features, f"{name}.pt")) or name not in qs:
            continue
        scene = Scene(name, args.features, args.ckpt, args.masks)
        for q in qs[name]["questions"]:
            if q["question_type"] != "counterfactual":
                continue
            got = parse_choices(answer(q["question"], scene, llm, [c["choice"] for c in q["choices"]]),
                                len(q["choices"]))
            truth = [c["answer"] == "correct" for c in q["choices"]]
            ok = [g == t for g, t in zip(got, truth)]
            n_opt, ok_opt, n_q, ok_q = n_opt + len(ok), ok_opt + sum(ok), n_q + 1, ok_q + all(ok)
            if args.limit and n_q >= args.limit:
                break
        if args.limit and n_q >= args.limit:
            break
    print(f"agent (offline parser, {args.masks} masks): per-option {ok_opt / max(1, n_opt):.3f} "
          f"per-question {ok_q / max(1, n_q):.3f} ({n_q} questions)")
