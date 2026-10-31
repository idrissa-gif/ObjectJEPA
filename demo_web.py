"""Browser version of demo_live.py: pick or type a question, watch Muse Spark call the tools, see the forecast.

    python demo_web.py              # video_06926, Muse Spark, prints the URL to open
    python demo_web.py --offline    # rule-based stand-in, no API call

Same agent and tools as demo_live.py (agent.answer unchanged): each tool call and result is
streamed to the page (demo_web.html) as server-sent events. After the answer, the page plays the
matching forecast clip from visualize.py if one was rendered (<video>_whatif_remove<slot>_half.mp4,
<video>_next_half.mp4). The server listens on 127.0.0.1 only and every request needs the random
token in the printed URL. The Muse Spark backend reads META_API_KEY from the environment only.
"""
import argparse
import glob
import json
import os
import queue
import secrets
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent import ChatLLM, OfflineLLM, answer, parse_choices

TYPES = ("counterfactual", "predictive")  # the question types the tools answer


class Streamed:
    """Wraps an llm(messages, tools) callable and emits what goes in and out (like demo_live.Narrated)."""

    def __init__(self, llm, emit, pause):
        self.llm, self.emit, self.pause, self.seen, self.calls = llm, emit, pause, 0, []

    def __call__(self, messages, tools):
        for m in messages[self.seen:]:
            if m["role"] == "tool":
                self.emit({"type": "result", "name": m["name"], "result": json.loads(m["content"])})
                time.sleep(self.pause)
        self.emit({"type": "thinking"})
        t = time.time()
        out = self.llm(messages, tools)
        self.seen = len(messages)
        for c in out.get("tool_calls", []):
            self.calls.append((c["name"], c.get("arguments") or {}))
            self.emit({"type": "call", "name": c["name"], "arguments": c.get("arguments") or {},
                       "seconds": round(time.time() - t, 1)})
        return out


class Demo:
    def __init__(self, args):
        from agent_tools import Scene

        self.args, self.lock = args, threading.Lock()
        qs = next(v for v in json.load(open(args.questions)) if v["video_filename"] == f"{args.video}.mp4")
        self.questions = [q for q in qs["questions"] if q["question_type"] in TYPES]
        self.video_path = glob.glob(os.path.join(args.data, "clevrer", "train", "**", f"{args.video}.mp4"),
                                    recursive=True)[0]
        print(f"Loading {args.video} ({args.masks.upper()} masks, world model {os.path.basename(args.ckpt)})…")
        self.scene = Scene(args.video, args.features, args.ckpt, args.masks,
                           head=args.head or None, update=args.update or None)
        self.llm = OfflineLLM() if args.offline else ChatLLM(args.base_url, args.model)
        self.label = "offline parser" if args.offline else args.model

    def info(self):
        return {"video": self.args.video, "label": self.label, "questions": [
            {"id": q["question_id"], "type": q["question_type"], "question": q["question"],
             "choices": [c["choice"] for c in q.get("choices", [])]} for q in self.questions]}

    def clip(self, calls):
        """(file, (caption, legend)) of the rendered forecast clip for the last simulation the agent ran, or None."""
        for name, a in reversed(calls):
            if name == "predict_future":
                f, cap = f"{self.args.video}_next_half.mp4", ("Forecast after the video ends",
                                                              "the last frame freezes; dots = forecast paths, star = forecast collision")
            elif name == "what_if_remove" and len(a.get("objects", [])) == 1:
                s = int(a["objects"][0])
                f, cap = f"{self.args.video}_whatif_remove{s}_half.mp4", (f"Forecast without the {self.scene.name(s)}",
                                                                          "solid = real paths, dots = forecast, the removed object is a ghost")
            else:
                continue
            return (f, cap) if os.path.exists(os.path.join(self.args.clips, f)) else None
        return None

    def ask(self, emit, qid=None, text=None):
        if qid is not None:
            q = next(q for q in self.questions if q["question_id"] == qid)
            question, choices = q["question"], [c["choice"] for c in q.get("choices", [])] or None
            truth = [c["answer"] == "correct" for c in q.get("choices", [])]
        else:
            question, choices, truth = text, None, None
        emit({"type": "question", "question": question, "choices": choices})
        used = (getattr(self.llm, "tokens_in", 0), getattr(self.llm, "tokens_out", 0))
        llm = Streamed(self.llm, emit, self.args.pause)
        reply = answer(question, self.scene, llm, choices)
        ev = {"type": "answer", "label": self.label, "text": reply}
        if choices:
            ev |= {"answers": parse_choices(reply, len(choices)), "truth": truth}
        if not self.args.offline:
            t_in, t_out = self.llm.tokens_in - used[0], self.llm.tokens_out - used[1]
            ev |= {"tokens_in": t_in, "tokens_out": t_out, "usd": round((t_in * 1.25 + t_out * 4.25) / 1e6, 4)}
        emit(ev)
        clip = self.clip(llm.calls)
        if clip:
            emit({"type": "clip", "src": f"/media/clip/{clip[0]}", "caption": clip[1][0], "legend": clip[1][1],
                  "start": 11.5 if "_next_" in clip[0] else 0})  # next: skip the replay of the real video


def handler(demo, token, page):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path):
            size = os.path.getsize(path)
            start, end = 0, size - 1
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):  # browsers fetch .mp4 in ranges to seek
                a, b = rng[6:].split(",")[0].split("-", 1)
                start, end = (int(a), int(b) if b else size - 1) if a else (max(0, size - int(b)), size - 1)
                end = min(end, size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(end - start + 1))

        def stream(self, **kw):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            events = queue.Queue()

            def work():
                with demo.lock:  # one question at a time: the scene caches its rollouts
                    try:
                        demo.ask(events.put, **kw)
                    except Exception as e:
                        events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
                    events.put(None)

            threading.Thread(target=work, daemon=True).start()
            while (ev := events.get()) is not None:
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: {\"type\": \"done\"}\n\n")

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            params = dict(urllib.parse.parse_qsl(url.query))
            if not secrets.compare_digest(params.get("t", ""), token):
                return self.send(403, b"Open the URL printed by demo_web.py (it has the access token).", "text/plain")
            try:
                if url.path == "/":
                    self.send(200, page, "text/html; charset=utf-8")
                elif url.path == "/api/info":
                    self.send(200, json.dumps(demo.info()).encode(), "application/json")
                elif url.path == "/api/ask" and ("qid" in params or params.get("q", "").strip()):
                    self.stream(qid=int(params["qid"])) if "qid" in params else self.stream(text=params["q"].strip())
                elif url.path == "/media/video":
                    self.send_file(demo.video_path)
                elif url.path.startswith("/media/clip/"):
                    path = os.path.join(demo.args.clips, os.path.basename(url.path))
                    self.send_file(path) if path.endswith(".mp4") and os.path.exists(path) else self.send(404, b"", "text/plain")
                else:
                    self.send(404, b"", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return H


if __name__ == "__main__":
    D = os.path.expanduser("~/objectjepa_data")
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="video_06926")
    ap.add_argument("--data", default=D)
    ap.add_argument("--masks", choices=["clevrer", "sam"], default="sam")
    ap.add_argument("--features", default=os.path.join(D, "obj_features_sam"))
    ap.add_argument("--ckpt", default=os.path.join(D, "ckpt_obj_e60_w512d8"))
    ap.add_argument("--questions", default=os.path.join(D, "clevrer/questions/train.json"))
    ap.add_argument("--head", default=os.path.join(D, "ckpt_collision"), help="collision head ('' = distance rule)")
    ap.add_argument("--update", default=os.path.join(D, "ckpt_update"), help="learned what-if update ('' = delta rule)")
    ap.add_argument("--clips", default=os.path.join(D, "demos", "halfspeed"), help="forecast clips from visualize.py")
    ap.add_argument("--offline", action="store_true", help="rule-based parser, no API call")
    ap.add_argument("--base-url", default=os.environ.get("META_BASE_URL", "https://api.meta.ai/v1"))
    ap.add_argument("--model", default=os.environ.get("META_MODEL", "muse-spark-1.3"))
    ap.add_argument("--pause", type=float, default=0.6, help="seconds between streamed tool results")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    demo = Demo(args)
    token = secrets.token_urlsafe(9)
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_web.html"), "rb").read()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(demo, token, page))
    print(f"Ready. Open http://localhost:{args.port}/?t={token}")
    server.serve_forever()
