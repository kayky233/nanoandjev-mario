"""Run fresh model-only episodes and save requests, responses and source hashes.

Example: uv run python evaluate_models.py --bot local --attempts 3 --output output/local-eval
Credentials are read from the process environment or existing project configuration.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot", choices=("local", "jev"), required=True)
    parser.add_argument("--levels", default="1-1,1-2,1-3")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller", type=Path, default=Path(__file__).with_name("play_local.py"))
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    levels = args.levels.split(",")
    if any(level not in {f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)} for level in levels):
        parser.error("--levels must contain valid world-stage values")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty so independent experiments cannot be mixed")
    args.output.mkdir(parents=True, exist_ok=True)

    source = args.controller.resolve()
    spec = importlib.util.spec_from_file_location("evaluated_controller", source)
    player = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(player)
    player.load_env()
    started = time.perf_counter()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    report = {
        "bot": args.bot, "levels": levels, "attempts": args.attempts,
        "controller_sha256": source_hash,
        "model": os.environ.get("LOCAL_POLICY_MODEL", player.LOCAL_POLICY_MODEL) if args.bot == "local" else "jev-latest",
        "local_mode": os.environ.get("LOCAL_POLICY_MODE", player.LOCAL_POLICY_MODE) if args.bot == "local" else None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_only": True, "route": False, "guard": False, "rules_fallback": False,
        "trials": [],
    }
    if args.bot == "local" and report["local_mode"] == "laya":
        root = os.environ.get("LOCAL_POLICY_BASE_URL", player.LOCAL_POLICY_BASE_URL).rstrip("/")
        with player.httpx.Client(timeout=10) as metadata_client:
            response = metadata_client.get(root + "/models")
            response.raise_for_status()
            report["runtime"] = response.json()["data"]
        if not any(item.get("id") == report["model"] for item in report["runtime"]):
            raise ValueError("Configured Laya checkpoint does not match the running service")

    def save():
        report["wall_seconds"] = round(time.perf_counter() - started, 3)
        (args.output / "evaluation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    real_send = player.httpx.Client.send
    current = args.output

    def traced_send(client, request, *a, **kw):
        begin = time.perf_counter()
        row = {"request": json.loads(request.content)}
        try:
            response = real_send(client, request, *a, **kw)
            row["status_code"] = response.status_code
            try:
                row["response"] = response.json()
            except ValueError:
                row["response"] = {"error": "non-JSON response"}
            return response
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            raise
        finally:
            row["latency_seconds"] = round(time.perf_counter() - begin, 4)
            # Request/response JSON only; never persist authentication headers.
            with (current / "http.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    player.httpx.Client.send = traced_send
    save()
    try:
        for attempt in range(1, args.attempts + 1):
            for level in levels:
                current = args.output / f"attempt-{attempt}" / level
                current.mkdir(parents=True)
                player.RUNS = current
                begin = time.perf_counter()
                try:
                    result = player.run(args.bot, level, model_only=True)
                except Exception as exc:
                    path = current / "results.jsonl"
                    result = json.loads(path.read_text().splitlines()[-1]) if path.exists() else {}
                    result.update(level=level, status="error", error_type=type(exc).__name__)
                result.update(attempt=attempt, wall_seconds=round(time.perf_counter() - begin, 3))
                report["trials"].append(result)
                save()
                print("EVALUATION", json.dumps(result, ensure_ascii=False), flush=True)
                if result.get("status") == "error":
                    raise SystemExit(1)
    finally:
        player.httpx.Client.send = real_send
        report["source_unchanged"] = hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        save()


if __name__ == "__main__":
    main()
