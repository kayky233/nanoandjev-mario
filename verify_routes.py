"""Replay recovery routes offline and save reproducible acceptance evidence.

    uv run python verify_routes.py --levels 1-2,1-3 --repeats 2 --gif

No model, coach, web search, or tunnel is used. Nonzero exit means at least
one route failed or repeated runs disagreed on distance/frame count.
"""

import argparse
import contextlib
import io
import json
from importlib.metadata import version
from pathlib import Path

from PIL import Image, ImageDraw

import watch_local as viewer


def verify(levels, repeats, output, gif=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for level in levels:
        for attempt in range(1, repeats + 1):
            images = []
            last_image = None

            def capture(obs, frame):
                nonlocal last_image
                last_image = Image.fromarray(obs).copy()
                if frame % 6 == 0:
                    images.append(last_image)

            with contextlib.redirect_stdout(io.StringIO()):
                result = viewer.play_episode(
                    level, 1_000_000, "replay", attempt, model="replay",
                    verified_route=True, frame_sink=capture if gif and attempt == 1 else None,
                )
            runs.append(result)
            print(json.dumps({k: result[k] for k in (
                "level", "best_x", "flag", "frames", "calls", "route_complete",
            )} | {"attempt": attempt}), flush=True)
            if images:
                # Every sampled frame spans six real emulator steps (100ms).
                # Preserve the final partial interval and hold the result 1.5s.
                durations = [100] * len(images)
                if result["frames"] % 6:
                    images.append(last_image)
                    durations.append(round((result["frames"] % 6) * 1000 / 60))
                durations[-1] += 1500
                labeled = []
                for image in images:
                    canvas = Image.new("RGB", (256, 260), "#0e1116")
                    canvas.paste(image, (0, 20))
                    ImageDraw.Draw(canvas).text(
                        (5, 4), f"{level} VERIFIED ROUTE / NO MODEL", fill="#58a6ff",
                    )
                    labeled.append(canvas)
                labeled[0].save(
                    output / f"{level}-verified-route.gif", save_all=True,
                    append_images=labeled[1:], duration=durations, loop=0,
                )
    passed = all(r["flag"] and r["route_complete"] and r["calls"] == 0 for r in runs)
    for level in levels:
        outcomes = {(r["best_x"], r["frames"]) for r in runs if r["level"] == level}
        passed = passed and len(outcomes) == 1
    report = {
        "passed": passed,
        "execution": "offline verified route; no model decisions",
        "frame_basis": "actual env.step calls; 60 emulator frames per second",
        "versions": {name: version(name) for name in ("gym-super-mario-bros", "nes-py", "gym", "numpy")},
        "runs": runs,
    }
    path = output / "acceptance.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'}: {path}")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default=",".join(viewer.VERIFIED_ROUTES))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("output/verified-routes"))
    parser.add_argument("--gif", action="store_true", help="save a labeled 10 fps replay for each level")
    args = parser.parse_args()
    levels = args.levels.split(",")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not levels or any(level not in viewer.VERIFIED_ROUTES for level in levels):
        parser.error("available routes: " + ", ".join(viewer.VERIFIED_ROUTES))
    raise SystemExit(0 if verify(levels, args.repeats, args.output, args.gif) else 1)


if __name__ == "__main__":
    main()
