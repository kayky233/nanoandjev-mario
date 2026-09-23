# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.14.0", "transformers==4.57.6", "numpy==1.26.4"]
# ///
"""Serve cached Qwen weights with real, deterministic local generation.

Run with ``uv run --script serve_qwen.py``. Model weights must already exist;
this adapter never downloads them. Dependencies are separate from the game.
"""

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MODEL_ID = "Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
    / REVISION
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=11503, help="Local port (default: 11503)")
    parser.add_argument(
        "--model-path", type=Path, default=SNAPSHOT,
        help="Existing Qwen2.5-1.5B-Instruct directory (default: pinned Hugging Face cache)",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    snapshot = args.model_path.expanduser()
    if not snapshot.is_dir():
        parser.error("Cached model directory is missing; provide existing weights with --model-path")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {
                    "status": "ready", "model": MODEL_ID, "device": device,
                    "revision": REVISION if snapshot.resolve() == SNAPSHOT.resolve() else None,
                })
            elif self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [
                    {"id": MODEL_ID, "object": "model", "owned_by": "local"},
                ]})
            else:
                self.reply(404, {"error": {"type": "not_found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.reply(404, {"error": {"type": "not_found"}})
            started = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 131072:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object")
                if request.get("stream"):
                    raise ValueError("Streaming is not supported")
                if request.get("model", MODEL_ID) != MODEL_ID:
                    raise ValueError("Unknown model")
                messages = request["messages"]
                if not isinstance(messages, list) or not messages or any(
                    not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
                    or not isinstance(m.get("content"), str) for m in messages
                ):
                    raise ValueError("Invalid messages")
                max_tokens = int(request.get("max_tokens", 8))
                if not 1 <= max_tokens <= 128:
                    raise ValueError("max_tokens must be between 1 and 128")
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([prompt], return_tensors="pt")
                prompt_tokens = int(inputs.input_ids.shape[1])
                if prompt_tokens + max_tokens > 4096:
                    raise ValueError("Context exceeds 4096 tokens")
            except (ValueError, KeyError, TypeError) as exc:
                return self.reply(400, {"error": {"type": type(exc).__name__}})
            try:
                inputs = inputs.to(device)
                with torch.inference_mode():
                    output = model.generate(
                        **inputs, max_new_tokens=max_tokens, do_sample=False,
                        temperature=None, top_p=None, top_k=None, max_time=60,
                    )
                completion_tokens = int(output.shape[1] - prompt_tokens)
                content = tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)
                self.reply(200, {
                    "id": f"local-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": "length" if completion_tokens == max_tokens else "stop"}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                              "total_tokens": prompt_tokens + completion_tokens},
                })
                print(json.dumps({"event": "inference", "seconds": round(time.perf_counter() - started, 3),
                                  "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}), flush=True)
            except Exception as exc:
                self.reply(500, {"error": {"type": type(exc).__name__}})
                print(json.dumps({"event": "error", "type": type(exc).__name__}), flush=True)

    # Bind before importing/loading the model, so an occupied port fails cheaply.
    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        started = time.perf_counter()
        print(json.dumps({"event": "loading", "model": MODEL_ID, "device": device}), flush=True)
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
        ).to(device).eval()
        print(json.dumps({"event": "ready", "port": args.port,
                          "load_seconds": round(time.perf_counter() - started, 3)}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
