"""Stage 4: the "what if" agent loop.

    answer(question, scene, llm, choices=None)

The language model gets the question (and multiple-choice options, if any) and
the tool schemas from agent_tools.TOOLS. It may call tools; their JSON results
are fed back until it answers. `llm(messages, tools)` must return either
{"tool_calls": [{"name": ..., "arguments": {...}}]} or {"content": "..."}.

Backends:
  OfflineLLM : no API. Rule-based stand-in (attribute word matching) used to
               test the whole loop for free and as a baseline for the real model.
  ChatLLM    : any OpenAI-compatible /chat/completions endpoint with tool calling
               (Muse Spark). Base URL and model from --base-url/--model or
               META_BASE_URL/META_MODEL; the API key is read from META_API_KEY only
               and is never written to code or logs.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

from agent_tools import TOOLS, run_tool

SYSTEM = (
    "You answer questions about a short physics video of objects sliding and colliding. "
    "Use the tools: describe_objects to see the objects and their slot numbers, observed_collisions "
    "for what really happens, what_if_remove to simulate the video without some objects, predict_future "
    "for what will happen after the video ends. "
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
    """Stand-in for the language model: handles '... if/without <object> ... removed' questions
    and 'what will happen next?' questions."""

    def __call__(self, messages, tools):
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        question = next(m["content"] for m in messages if m["role"] == "user")
        if not tool_msgs:
            return {"tool_calls": [{"name": "describe_objects", "arguments": {}}]}
        predictive = "next" in question.split("\n")[0].lower()  # "What will happen next?"
        if len(tool_msgs) == 1 and predictive:
            return {"tool_calls": [{"name": "predict_future", "arguments": {}}]}
        if len(tool_msgs) == 1:
            objects = json.loads(tool_msgs[0]["content"])
            head = question.split("\n")[0]
            m = re.search(r"(?:if|without) (?:the )?(.+?)(?: is removed|,|\?|$)", head.lower())
            target = match_objects(m.group(1), objects) if m else []
            return {"tool_calls": [{"name": "what_if_remove", "arguments": {"objects": target[:1]}}]}
        objects = json.loads(tool_msgs[0]["content"])
        result = json.loads(tool_msgs[-1]["content"])
        options = []  # (option number, slot pair or None)
        for line in question.split("\n")[1:]:
            mm = re.match(r"(\d+)\. (.+)", line.strip())
            if not mm:
                continue
            parts = re.split(r" collides with | and | collide", mm.group(2).lower())
            refs = [match_objects(p, objects) for p in parts[:2]]
            options.append((mm.group(1), tuple(sorted((refs[0][0], refs[1][0]))) if all(len(r) == 1 for r in refs) else None))
        if predictive:  # exactly one option happens: pick the best-ranked pair
            rank = {tuple(sorted(e["objects"])): e["rank"] for e in result["after_the_video"]}
            best = min(options, key=lambda o: rank.get(o[1], 99))[0] if options else None
            return {"content": "\n".join(f"{i}: {'yes' if i == best else 'no'}" for i, _ in options)}
        happen = {tuple(sorted(e["objects"])) for e in result["collisions_without_them"]}
        negate = " not " in question.split("\n")[0].lower()
        return {"content": "\n".join(f"{i}: {'yes' if ((pair in happen) != negate if pair else False) else 'no'}"
                                      for i, pair in options)}


class ChatLLM:
    """OpenAI-compatible chat backend. Counts tokens so a run can report (and cap) its cost."""

    def __init__(self, base_url=None, model=None, key_env="META_API_KEY"):
        self.url = (base_url or os.environ["META_BASE_URL"]).rstrip("/") + "/chat/completions"
        self.model = model or os.environ["META_MODEL"]
        self.key = os.environ[key_env]
        self.tokens_in = self.tokens_out = 0
        self.first_usage = None

    @staticmethod
    def _wire(m):
        if m["role"] == "assistant" and "tool_calls" in m:
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))}}
                for c in m["tool_calls"]]}
        if m["role"] == "tool":
            return {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
        return m

    def _post(self, body, retries=3):
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        for i in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and i < retries:
                    time.sleep(2 ** i * 5)
                    continue
                raise RuntimeError(f"HTTP {e.code} from {self.url}: {e.read()[:500].decode(errors='replace')}")

    def __call__(self, messages, tools):
        out = self._post({"model": self.model, "temperature": 0,
                          "messages": [self._wire(m) for m in messages],
                          "tools": [{"type": "function", "function": t} for t in tools]})
        usage = out.get("usage") or {}
        self.first_usage = self.first_usage or usage
        self.tokens_in += usage.get("prompt_tokens", 0)
        self.tokens_out += usage.get("completion_tokens", 0)
        msg = out["choices"][0]["message"]
        if msg.get("tool_calls"):
            return {"tool_calls": [{"id": c["id"], "name": c["function"]["name"],
                                    "arguments": json.loads(c["function"].get("arguments") or "{}")}
                                   for c in msg["tool_calls"]]}
        return {"content": msg.get("content") or ""}


def answer(question, scene, llm, choices=None, max_steps=6, log=None):
    if choices:
        question += "\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(choices))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for step in range(max_steps):
        out = llm(messages, TOOLS)
        if "tool_calls" not in out:
            if log is not None:
                log.extend(messages + [{"role": "assistant", "content": out["content"]}])
            return out["content"]
        for i, call in enumerate(out["tool_calls"]):
            call.setdefault("id", f"call_{step}_{i}")
        messages.append({"role": "assistant", "tool_calls": out["tool_calls"]})
        for call in out["tool_calls"]:
            try:
                res = run_tool(scene, call["name"], call.get("arguments", {}))
            except Exception as e:  # bad arguments from the model: tell it instead of crashing
                res = {"error": f"{type(e).__name__}: {e}"}
            messages.append({"role": "tool", "name": call["name"], "tool_call_id": call["id"],
                             "content": json.dumps(res)})
    return ""


def parse_choices(text, n):
    """'<i>: yes/no' lines -> list of bools (False when missing)."""
    got = dict(re.findall(r"(\d+)\s*:\s*(yes|no)", text.lower()))
    return [got.get(str(i)) == "yes" for i in range(n)]


if __name__ == "__main__":
    import argparse

    from agent_tools import Scene

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--masks", choices=["clevrer", "sam"], default="clevrer")
    ap.add_argument("--questions", default=os.path.expanduser("~/objectjepa_data/clevrer/questions/train.json"))
    ap.add_argument("--videos", required=True, help="text file with video names (use val videos)")
    ap.add_argument("--limit", type=int, default=0, help="max questions")
    ap.add_argument("--types", default="counterfactual", help="comma-separated CLEVRER question types "
                    "(counterfactual, predictive)")
    ap.add_argument("--backend", choices=["offline", "chat"], default="offline")
    ap.add_argument("--head", default=None, help="collision head checkpoint dir (default: distance rule)")
    ap.add_argument("--update", default=None, help="learned what-if update dir, needs --head (default: delta rule)")
    ap.add_argument("--base-url", default=None, help="chat backend (default $META_BASE_URL)")
    ap.add_argument("--model", default=None, help="chat backend (default $META_MODEL)")
    ap.add_argument("--price-in", type=float, default=1.25, help="USD per M input tokens")
    ap.add_argument("--price-out", type=float, default=4.25, help="USD per M output tokens")
    ap.add_argument("--max-usd", type=float, default=1.0, help="stop once the estimated cost exceeds this")
    ap.add_argument("--log", default=None, help="write each question's messages to this .jsonl")
    args = ap.parse_args()
    qs = {q["video_filename"][:-4]: q for q in json.load(open(args.questions))}
    llm = OfflineLLM() if args.backend == "offline" else ChatLLM(args.base_url, args.model)
    cost = lambda: (getattr(llm, "tokens_in", 0) * args.price_in + getattr(llm, "tokens_out", 0) * args.price_out) / 1e6
    logf = open(args.log, "w") if args.log else None
    n_opt, ok_opt, n_q, ok_q = 0, 0, 0, 0
    types = args.types.split(",")
    by_type = {t: [0, 0, 0, 0] for t in types}  # options, correct options, questions, correct questions
    for name in (l.strip() for l in open(args.videos)):
        if not os.path.exists(os.path.join(args.features, f"{name}.pt")) or name not in qs:
            continue
        scene = Scene(name, args.features, args.ckpt, args.masks, head=args.head, update=args.update)
        for q in qs[name]["questions"]:
            if q["question_type"] not in types:
                continue
            msgs = []
            got = parse_choices(answer(q["question"], scene, llm, [c["choice"] for c in q["choices"]], log=msgs),
                                len(q["choices"]))
            truth = [c["answer"] == "correct" for c in q["choices"]]
            ok = [g == t for g, t in zip(got, truth)]
            n_opt, ok_opt, n_q, ok_q = n_opt + len(ok), ok_opt + sum(ok), n_q + 1, ok_q + all(ok)
            t = by_type[q["question_type"]]
            t[0], t[1], t[2], t[3] = t[0] + len(ok), t[1] + sum(ok), t[2] + 1, t[3] + all(ok)
            if logf:
                logf.write(json.dumps({"video": name, "question_id": q["question_id"], "type": q["question_type"],
                                       "truth": truth,
                                       "got": got, "messages": msgs}) + "\n")
                logf.flush()
            if (args.limit and n_q >= args.limit) or cost() > args.max_usd:
                break
        if (args.limit and n_q >= args.limit) or cost() > args.max_usd:
            break
    label = "offline parser" if args.backend == "offline" else llm.model
    print(f"agent ({label}, {args.masks} masks): per-option {ok_opt / max(1, n_opt):.3f} "
          f"per-question {ok_q / max(1, n_q):.3f} ({n_q} questions)")
    if len(types) > 1:
        for t, (no, oo, nq, oq) in by_type.items():
            print(f"  {t:15s} per-option {oo / max(1, no):.3f} per-question {oq / max(1, nq):.3f} ({nq} questions)")
    if args.backend == "chat":
        print(f"tokens in {llm.tokens_in}, out {llm.tokens_out}, estimated ${cost():.4f} "
              f"(${args.price_in}/M in, ${args.price_out}/M out)" + ("  STOPPED at --max-usd" if cost() > args.max_usd else ""))
        print("first response usage:", json.dumps(llm.first_usage))
