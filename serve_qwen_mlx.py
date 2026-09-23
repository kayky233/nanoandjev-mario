# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "transformers==5.17.0", "numpy==1.26.4"]
# ///
"""Serve existing Qwen weights with MLX FP16 and exact prefix KV reuse.

Run ``uv run --script serve_qwen_mlx.py`` on Apple Silicon. This adapter never
downloads model weights. Use ``--no-prefix-cache`` for a cache ablation using
the same weights, tokenizer, prompt and greedy decoding. The cache only reuses
identical input tokens; it does not reuse answers or select game actions.
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


def common_prefix_length(previous, current):
    # A final input token is always needed to compute the next-token logits.
    limit = min(len(previous), max(0, len(current) - 1))
    index = 0
    while index < limit and previous[index] == current[index]:
        index += 1
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=11504)
    parser.add_argument("--model-path", type=Path, default=SNAPSHOT)
    parser.add_argument("--no-prefix-cache", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    snapshot = args.model_path.expanduser()
    if not snapshot.is_dir():
        parser.error("Cached weights are missing; provide existing weights with --model-path")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    prefix_cache = None
    previous_tokens = []

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
                    "status": "ready", "model": MODEL_ID, "backend": "mlx",
                    "device": "gpu", "dtype": "float16", "prefix_cache": not args.no_prefix_cache,
                    "revision": REVISION if snapshot.resolve() == SNAPSHOT.resolve() else None,
                })
            elif self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [
                    {"id": MODEL_ID, "object": "model", "owned_by": "local"},
                ]})
            else:
                self.reply(404, {"error": {"type": "not_found"}})

        def do_POST(self):
            nonlocal prefix_cache, previous_tokens
            if self.path != "/v1/chat/completions":
                return self.reply(404, {"error": {"type": "not_found"}})
            started = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 131072:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict) or request.get("stream"):
                    raise ValueError("Expected a non-streaming JSON request")
                if request.get("model", MODEL_ID) != MODEL_ID:
                    raise ValueError("Unknown model")
                if request.get("temperature", 0) != 0:
                    raise ValueError("Only deterministic temperature=0 is supported")
                messages = request["messages"]
                if not isinstance(messages, list) or not messages or any(
                    not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
                    or not isinstance(m.get("content"), str) for m in messages
                ):
                    raise ValueError("Invalid messages")
                max_tokens = int(request.get("max_tokens", 8))
                if not 1 <= max_tokens <= 128:
                    raise ValueError("max_tokens must be between 1 and 128")
                tokens = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=False,
                )
                if not tokens or len(tokens) + max_tokens > 4096:
                    raise ValueError("Context exceeds 4096 tokens")
            except (ValueError, KeyError, TypeError) as exc:
                return self.reply(400, {"error": {"type": type(exc).__name__}})
            tokenize_seconds = time.perf_counter() - started
            try:
                cached = common_prefix_length(previous_tokens, tokens) if prefix_cache is not None else 0
                if args.no_prefix_cache or not cached:
                    prefix_cache = make_prompt_cache(model)
                    cached = 0
                else:
                    trim_prompt_cache(prefix_cache, prefix_cache[0].offset - cached)
                generation_started = time.perf_counter()
                parts = []
                for result in stream_generate(
                    model, tokenizer, tokens[cached:], max_tokens=max_tokens,
                    sampler=make_sampler(temp=0), prompt_cache=prefix_cache,
                ):
                    parts.append(result.text)
                inference_seconds = time.perf_counter() - generation_started
                # Keep input KV only. Generated answers must never enter a later
                # request's prefix or survive changes in the observed game state.
                if any(layer.offset < len(tokens) for layer in prefix_cache):
                    raise RuntimeError("Incomplete prompt KV cache")
                trim_prompt_cache(prefix_cache, prefix_cache[0].offset - len(tokens))
                if any(layer.offset != len(tokens) for layer in prefix_cache):
                    raise RuntimeError("Unexpected KV cache offset")
                previous_tokens = tokens
                completion_tokens = result.generation_tokens
                metrics = {
                    "backend": "mlx", "dtype": "float16",
                    "seconds": round(time.perf_counter() - started, 6),
                    "tokenize_seconds": round(tokenize_seconds, 6),
                    "inference_seconds": round(inference_seconds, 6),
                    "prefill_to_first_token_seconds": round(result.prompt_tokens / result.prompt_tps, 6),
                    "decode_seconds": round(completion_tokens / result.generation_tps, 6),
                    "prompt_tokens": len(tokens), "cached_tokens": cached,
                    "prefill_tokens": len(tokens) - cached, "completion_tokens": completion_tokens,
                    "peak_memory_gb": round(result.peak_memory, 4),
                }
                self.reply(200, {
                    "id": f"local-mlx-{time.time_ns()}", "object": "chat.completion",
                    "created": int(time.time()), "model": MODEL_ID,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                                 "finish_reason": result.finish_reason}],
                    "usage": {"prompt_tokens": len(tokens), "completion_tokens": completion_tokens,
                              "total_tokens": len(tokens) + completion_tokens,
                              "prompt_tokens_details": {"cached_tokens": cached}},
                    "timings": metrics,
                })
                # No messages, generated content or request headers enter logs.
                print(json.dumps({"event": "inference", **metrics}), flush=True)
            except Exception as exc:
                prefix_cache, previous_tokens = None, []
                self.reply(500, {"error": {"type": type(exc).__name__}})
                print(json.dumps({"event": "error", "type": type(exc).__name__}), flush=True)

    # Fail an occupied port before allocating model memory.
    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        if not mx.metal.is_available():
            parser.error("MLX GPU is unavailable; this adapter requires Apple Silicon")
        # The library otherwise restores its smaller default after every request.
        # Keep this process's resident-weight allowance stable across decisions.
        wired_limit_bytes = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit_bytes)
        started = time.perf_counter()
        print(json.dumps({"event": "loading", "model": MODEL_ID, "backend": "mlx", "dtype": "float16"}), flush=True)
        model, tokenizer = load(str(snapshot), lazy=True)
        model.set_dtype(mx.float16)
        mx.eval(model.parameters())
        mx.clear_cache()
        print(json.dumps({"event": "ready", "port": args.port,
                          "prefix_cache": not args.no_prefix_cache,
                          "wired_limit_gb": round(wired_limit_bytes / 1e9, 3),
                          "load_seconds": round(time.perf_counter() - started, 3)}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
