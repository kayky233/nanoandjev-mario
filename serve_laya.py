# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["laya[serve]==0.3.11", "torch==2.14.0", "transformers==4.57.6", "huggingface-hub==0.36.2"]
# ///
"""Serve a pinned, real Laya checkpoint using Laya's official Jev-compatible API.

Run ``uv run --script serve_laya.py``. Downloads only the selected checkpoint,
then keeps it resident at http://127.0.0.1:11505/v1/systemone. No generative
Qwen model, action rules, or cached decisions are used. Requests that would be
silently truncated by the upstream tokenizer are rejected before inference.
"""

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

MODEL_REPO = "convaiinnovations/laya"
REVISION = "aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8"
CHECKPOINTS = {"english": "", "typed-decisions": "typed-decisions"}


def token_budget(agent, state, questions):
    """Account for the pinned upstream build_sequence's three truncation points."""
    from laya.common import build_sequence, render_options, serialize_state

    tok = agent.tok
    maximum = agent.cfg.get("max_len", 512)
    head_maximum = agent.cfg.get("head_max_len", 192)
    state_count = len(tok(serialize_state(state).replace(tok.mask_token, " "),
                          add_special_tokens=False)["input_ids"])
    budgets = {}
    for qid, definition in questions.items():
        agent._check_question(qid, definition)
        question = agent._to_internal(definition)
        head = "%s question: %s" % (question["t"], str(question["ins"]).replace(tok.mask_token, " "))
        head_count = len(tok(head, add_special_tokens=False)["input_ids"])
        options = render_options(question)
        raw_options = [1 + len(tok(" " + option.replace(tok.mask_token, " "),
                                   add_special_tokens=False)["input_ids"]) for option in options]
        used_options = [min(49, count) for count in raw_options]
        instruction_budget = head_maximum - sum(used_options)
        if instruction_budget < 16:
            per_option = max(4, (head_maximum - 16) // max(1, len(used_options)))
            used_options = [min(per_option, count) for count in used_options]
            instruction_budget = head_maximum - sum(used_options)
        used_head = min(head_count, max(8, instruction_budget))
        state_room = max(0, maximum - (used_head + sum(used_options) + 4))
        used_state = min(state_count, state_room)
        seq, markers = build_sequence(tok, state, question, maximum, head_maximum,
                                      truncate_left=isinstance(state, list))
        truncated = (used_head < head_count or used_options != raw_options
                     or used_state < state_count or len(markers) != len(options))
        budgets[qid] = {
            "max_tokens": maximum,
            "header_max_tokens": head_maximum,
            "instruction_tokens": head_count,
            "instruction_tokens_kept": used_head,
            "option_tokens": raw_options,
            "option_tokens_kept": used_options,
            "state_tokens": state_count,
            "state_tokens_kept": used_state,
            "state_token_budget": state_room,
            "sequence_tokens": len(seq),
            "truncated": truncated,
        }
    return budgets


class FixedCheckpoint:
    """Small interface adapter accepted by the official laya.serve.create_app."""

    def __init__(self, agent, checkpoint, revision):
        self.agent = agent
        self.checkpoint = checkpoint
        self.revision = revision

    @property
    def loaded(self):
        return [self.checkpoint]

    def metadata(self):
        return {
            "model": MODEL_REPO,
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "laya_version": importlib.metadata.version("laya"),
            "device": str(self.agent.device),
            "dtype": str(self.agent.dtype),
            "max_tokens": self.agent.cfg.get("max_len", 512),
            "header_max_tokens": self.agent.cfg.get("head_max_len", 192),
        }

    def predict(self, state, questions, model=None):
        if model is not None and model != self.checkpoint:
            raise ValueError("this server is pinned to checkpoint %r" % self.checkpoint)
        started = time.perf_counter()
        budgets = token_budget(self.agent, state, questions)
        if any(item["truncated"] for item in budgets.values()):
            print(json.dumps({"event": "rejected_token_budget", "token_budget": budgets}), flush=True)
            raise ValueError("request exceeds Laya token budget: " + json.dumps(budgets))
        result = self.agent.predict(state, questions)
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        result["model"] = MODEL_REPO
        result["routing"] = {
            "model": self.checkpoint, "repo": MODEL_REPO, "revision": self.revision,
            "reason": "explicit fixed checkpoint; automatic routing disabled",
        }
        result["runtime"] = {**self.metadata(), "latency_ms": elapsed, "token_budget": budgets}
        print(json.dumps({"event": "inference", "latency_ms": elapsed,
                          "usage": result.get("usage"), "token_budget": budgets,
                          "device": str(self.agent.device)}), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=11505)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINTS), default="english")
    parser.add_argument("--model-path", type=Path, help="Existing checkpoint directory; no download")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.threads < 1:
        parser.error("--threads must be positive")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # CPU needs no GPU competition; users can explicitly select their own device.
    os.environ["LAYA_HOST"] = "127.0.0.1"
    os.environ["LAYA_THREADS"] = str(args.threads)
    prefix = CHECKPOINTS[args.checkpoint]
    revision = REVISION
    if args.model_path:
        model_path = args.model_path.expanduser()
        if not model_path.is_dir():
            parser.error("--model-path must contain an existing Laya checkpoint")
        revision = None  # A custom path is not evidence of the pinned Hub revision.
    else:
        from huggingface_hub import snapshot_download
        lead = prefix + "/" if prefix else ""
        snapshot = snapshot_download(MODEL_REPO, revision=REVISION, allow_patterns=[
            lead + name for name in ("rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*")
        ])
        model_path = Path(snapshot) / prefix
    if args.download_only:
        print(json.dumps({"model_path": str(model_path), "revision": revision,
                          "checkpoint": args.checkpoint}))
        return

    import laya
    import torch
    import uvicorn
    from laya.serve import create_app

    torch.set_num_threads(args.threads)
    agent = laya.load(str(model_path), device=args.device)
    adapter = FixedCheckpoint(agent, args.checkpoint, revision)
    os.environ["LAYA_DEVICE"] = str(agent.device)
    app = create_app(router=adapter)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": args.checkpoint, **adapter.metadata()}]}

    print(json.dumps({"event": "ready", **adapter.metadata(), "port": args.port}), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
