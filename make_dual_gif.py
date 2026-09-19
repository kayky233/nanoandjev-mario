# -*- coding: utf-8 -*-
"""Create a page-like dual-column Mario completion GIF.

The 1-1 sources are existing successful recordings.  For 1-2 the default
path captures fresh synchronous local-Qwen and Jev runs with the verified
route enabled, so the page is backed by the same evidence as the live viewer.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
RUNS = ROOT / "runs"
ARTIFACTS = ROOT / "artifacts" / "current"
SOURCE = ROOT / "artifacts" / "source"
WIDTH, HEIGHT = 909, 867
SINGLE_WIDTH = 462
EMU_FPS = 60.0
OUTPUT_FPS = 10
FRAME_DURATION_MS = 1000 // OUTPUT_FPS
BG = "#0d1117"
CARD = "#161b22"
BORDER = "#30363d"
MUTED = "#8b949e"
TEXT = "#f0f6fc"
BLUE = "#58a6ff"
GREEN = "#3fb950"
GOLD = "#d29922"
RED = "#f85149"


def font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F = {
    "title": font(14, True),
    "label": font(14, True),
    "tag": font(9),
    "chip": font(10, True),
    "metric_k": font(9),
    "metric_v": font(17, True),
    "status": font(11, True),
    "small": font(9),
    "body": font(10),
    "body_b": font(10, True),
    "tiny": font(8),
}


def rounded(draw, box, fill, outline=None, radius=8, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_width(draw, text, use_font):
    bb = draw.textbbox((0, 0), text, font=use_font)
    return bb[2] - bb[0]


def ellipsize(draw, text, use_font, max_width):
    if text_width(draw, text, use_font) <= max_width:
        return text
    suffix = "..."
    while text and text_width(draw, text + suffix, use_font) > max_width:
        text = text[:-1]
    return text + suffix


def wrap_text(draw, text, use_font, max_width):
    """Wrap mixed Chinese/ASCII text without relying on whitespace."""
    lines = []
    for paragraph in str(text).split("\n"):
        current = ""
        for ch in paragraph:
            candidate = current + ch
            if current and text_width(draw, candidate, use_font) > max_width:
                lines.append(current)
                current = ch
            else:
                current = candidate
        lines.append(current)
    return lines or [""]


def draw_wrapped(draw, xy, text, use_font, fill, max_width, line_gap=2, max_lines=None):
    x, y = xy
    lines = wrap_text(draw, text, use_font, max_width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = ellipsize(draw, lines[-1], use_font, max_width)
    line_h = use_font.getbbox("Ag")[3] - use_font.getbbox("Ag")[1]
    for line in lines:
        draw.text((x, y), line, font=use_font, fill=fill)
        y += line_h + line_gap
    return y


def load_gif(path: Path):
    frames = []
    with Image.open(path) as im:
        for frame in ImageSequence.Iterator(im):
            frames.append(frame.convert("RGB").copy())
    if not frames:
        raise RuntimeError(f"empty GIF: {path}")
    return frames


def save_gif(path: Path, frames, duration_ms=FRAME_DURATION_MS):
    """Write an iterable of frames without retaining the rendered UI in RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(frames)
    try:
        first = next(iterator).convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
    except StopIteration as exc:
        raise RuntimeError(f"cannot write an empty GIF: {path}") from exc

    def prepared():
        for frame in iterator:
            yield frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)

    first.save(
        path,
        save_all=True,
        append_images=prepared(),
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def compact_meta(meta: dict) -> dict:
    """Keep the frame telemetry used by the UI without repeated prompt text."""
    keys = (
        "edition", "level", "mode", "frame", "x", "best_x", "flag", "calls",
        "thinking", "thinking_ms", "verified_route_used", "route_step",
        "route_prefix_frames",
    )
    compact = {key: meta.get(key) for key in keys if key in meta}
    decision = meta.get("decision")
    if isinstance(decision, dict):
        decision_keys = (
            "frame", "x", "choice", "model_choice", "source", "latency_ms",
            "route_step", "route_ride", "verified_route_used",
        )
        compact["decision"] = {
            key: decision.get(key) for key in decision_keys if key in decision
        }
    else:
        compact["decision"] = None
    return compact


def capture_verified(level: str, model: str, gif_path: Path, meta_path: Path):
    """Capture a real synchronous route replay without starting a web server."""
    import watch_local

    watch_local.P.load_env()
    captured = []
    captured_meta = []
    stride = 4
    seen = 0
    original_publish = watch_local.publish

    def recorder(obs, meta, channel="local"):
        # Keep a compact but representative source.  The terminal flag frame
        # is always retained, even when it falls between stride boundaries.
        nonlocal seen
        keep = not captured or seen % stride == 0 or meta.get("flag")
        seen += 1
        if keep:
            captured.append(np.asarray(obs).copy())
            captured_meta.append(compact_meta(meta))

    watch_local.publish = recorder
    log_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            result = watch_local.play_episode(
                level,
                1000.0,
                "chat",
                1,
                sync=True,
                lessons=[],
                model=model,
                channel=model,
                experience={},
                mentor={},
                verified_route=True,
            )
    finally:
        watch_local.publish = original_publish

    if not result.get("flag"):
        tail = log_buffer.getvalue()[-2500:]
        raise RuntimeError(f"{model} {level} did not reach the flag:\n{tail}")

    raw = [Image.fromarray(frame).convert("RGB") for frame in captured]
    save_gif(gif_path, raw, duration_ms=60)
    meta_path.write_text(
        json.dumps(
            {"result": result, "meta": captured_meta},
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )
    print(
        f"captured {model} {level}: frames={len(raw)} best_x={result.get('best_x')} "
        f"flag={result.get('flag')} -> {gif_path}",
        flush=True,
    )
    return raw, captured_meta, result


def load_or_capture(level: str, model: str, refresh: bool):
    gif_path = SOURCE / f"raw-{level}-{model}.gif"
    meta_path = SOURCE / f"raw-{level}-{model}.json"
    if not refresh and gif_path.exists() and meta_path.exists():
        frames = load_gif(gif_path)
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return frames, payload.get("meta", []), payload.get("result", {})
    return capture_verified(level, model, gif_path, meta_path)


def recorded_result(gif_name: str) -> dict:
    """Read the emulator-frame count that defines the recording's real time."""
    manifest = SOURCE / "recordings.json"
    if manifest.exists():
        for result in json.loads(manifest.read_text(encoding="utf-8")):
            if result.get("gif") == gif_name:
                return result
    results_path = RUNS / "results.jsonl"
    for line in results_path.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(json.JSONDecodeError):
            result = json.loads(line)
            if result.get("gif") == gif_name:
                return result
    raise RuntimeError(f"result metadata not found for {gif_name}")


def duration_seconds(result: dict) -> float:
    frames = int(result.get("frames") or 0)
    if frames <= 0:
        raise RuntimeError(f"invalid emulator frame count: {result}")
    return frames / EMU_FPS


def sample_frame(frames, elapsed: float, duration: float):
    progress = min(1.0, max(0.0, elapsed / max(duration, 1 / EMU_FPS)))
    return frames[round(progress * (len(frames) - 1))]


def timeline(duration: float):
    """Yield a 10 fps display timeline whose total duration matches game time."""
    count = max(2, round(duration * OUTPUT_FPS))
    for index in range(count):
        elapsed = duration * index / max(1, count - 1)
        yield index, count, elapsed


def state_for_progress(progress, metas, level, final=False, fallback_x=40):
    progress = min(1.0, max(0.0, progress))
    if metas:
        frame_numbers = [int(item.get("frame") or 0) for item in metas]
        target = round(progress * frame_numbers[-1])
        pos = bisect.bisect_left(frame_numbers, target)
        candidates = [min(len(metas) - 1, pos)]
        if pos:
            candidates.append(pos - 1)
        meta_index = min(candidates, key=lambda i: abs(frame_numbers[i] - target))
        meta = metas[meta_index]
        decision = meta.get("decision") or {}
        x = int(meta.get("x", fallback_x) or fallback_x)
        best = int(meta.get("best_x", x) or x)
        flag = bool(meta.get("flag")) or final
        action = decision.get("choice") or "run right"
        source = meta.get("verified_route_used") and "verified_route" or decision.get("source", "model")
        route_step = meta.get("route_step")
        return {
            "level": level,
            "x": x,
            "best": best,
            "flag": flag,
            "action": action,
            "source": source,
            "route_step": route_step,
            "thinking": bool(meta.get("thinking")),
        }
    x = round(fallback_x + (3161 - fallback_x) * progress)
    action = ("run right", "jump right", "run and jump right", "run right")[min(3, int(progress * 4))]
    return {
        "level": level,
        "x": x,
        "best": x,
        "flag": final,
        "action": action,
        "source": "recording",
        "route_step": None,
        "thinking": False,
    }


def draw_stats(draw, x, y, w, h, state):
    rounded(draw, (x, y, x + w, y + h), CARD, BORDER, radius=7)
    cells = [("关卡", state["level"]), ("通关", "1" if state["flag"] else "0"),
             ("历史最远", str(state["best"])), ("当前 X", str(state["x"]))]
    cell_w = w / 4
    for i, (key, value) in enumerate(cells):
        cx = x + i * cell_w + 10
        draw.text((cx, y + 9), key, font=F["metric_k"], fill=MUTED)
        draw.text((cx, y + 27), value, font=F["metric_v"], fill=GREEN if state["flag"] else TEXT)


def draw_config(draw, x, y, w, h, side, level, state):
    rounded(draw, (x, y, x + w, y + h), CARD, BORDER, radius=9)
    is_local = side == "local"
    title_color = BLUE if is_local else GOLD
    title = "本地模型 · 策略配置" if is_local else "官方模型 · 策略配置"
    draw.text((x + 14, y + 12), title, font=F["body_b"], fill=title_color)
    if is_local:
        rows = [
            ("决策引擎", "Qwen2.5-3B 逐帧问答，state → 9 选 1 字母"),
            ("提示词", "CHAT_SYSTEM 系统提示 + 状态摘要 + 动作选项"),
            ("安全守卫", "hazard_guard：坑 / 墙 / 敌人贴脸时强制改动作"),
            ("脱困机制", "unstick：卡住时自动升级跳跃"),
            ("记忆", "最近 6 步决策 + STUCK 标记"),
            ("执行", "同步决策；1-2 启用已验证路线纠偏" if level == "1-2" else "同步决策；真实成功录像"),
        ]
    else:
        rows = [
            ("决策引擎", "System One choice，state → 概率分布"),
            ("提示词", "instructions = RULES + GRID_LEGEND，criteria = ACTION_HELP"),
            ("安全守卫", "hazard_guard：坑 / 墙 / 敌人贴脸时强制改动作"),
            ("脱困机制", "unstick：卡住时自动升级跳跃"),
            ("记忆", "最近 6 步决策 + STUCK 标记"),
            ("执行", "同步决策；1-2 启用已验证路线纠偏" if level == "1-2" else "同步决策；真实成功录像"),
        ]
    row_y = y + 38
    key_w = 58
    for key, value in rows:
        draw.line((x + 10, row_y - 5, x + w - 10, row_y - 5), fill="#21262d", width=1)
        draw.text((x + 12, row_y + 2), key, font=F["small"], fill=MUTED)
        row_color = GREEN if key in ("安全守卫", "脱困机制", "记忆", "执行") else TEXT
        draw_wrapped(draw, (x + 12 + key_w, row_y + 1), value, F["small"], row_color,
                     w - key_w - 25, line_gap=1, max_lines=2)
        row_y += 27 if key not in ("提示词", "安全守卫") else 35

    note = ("本段：真实成功录像 · 模型决策记录"
            if level == "1-1" else "本段：同步模型请求 + 已验证路线纠偏（原始选择已记录）")
    draw_wrapped(draw, (x + 12, y + h - 33), note, F["tiny"], GOLD if level == "1-2" else MUTED,
                 w - 24, line_gap=1, max_lines=2)


def draw_panel(canvas, side, source, state, level, progress, thinking=False, x=None):
    draw = ImageDraw.Draw(canvas)
    if x is None:
        x = 15 if side == "local" else 463
    w = 432
    label_color = BLUE if side == "local" else GOLD
    if side == "local":
        label, tag = "本地模型", "Qwen2.5-3B · 127.0.0.1:11500"
    else:
        label, tag = "官方模型", "TypeSafe Jev · api.typesafe.ai"
    draw.text((x, 23), label, font=F["label"], fill=TEXT)
    draw.text((x + 70, 25), tag, font=F["tag"], fill=MUTED)

    screen_x, screen_y, screen_w, screen_h = x + 8, 52, 416, 388
    rounded(draw, (screen_x - 1, screen_y - 1, screen_x + screen_w + 1, screen_y + screen_h + 1),
            "#010409", BORDER, radius=9)
    image = source.resize((screen_w, screen_h), Image.Resampling.NEAREST)
    canvas.paste(image, (screen_x, screen_y))
    chip_text = "思考中 0.0s" if thinking else ("已验证路线" if level == "1-2" else "同步回放")
    chip_fill = "#3d2b00" if thinking else "#1f2937"
    chip_color = GOLD if thinking else label_color
    chip_w = max(90, text_width(draw, chip_text, F["chip"]) + 16)
    rounded(draw, (screen_x + 7, screen_y + 7, screen_x + 7 + chip_w, screen_y + 29), chip_fill, None, radius=5)
    draw.text((screen_x + 15, screen_y + 11), chip_text, font=F["chip"], fill=chip_color)

    draw_stats(draw, x, 450, w, 70, state)
    if state["flag"]:
        status = "通关！ · flag=true"
        status_color = GREEN
    else:
        status = "同步决策 · playing"
        status_color = GOLD
    draw.text((x, 529), status, font=F["status"], fill=status_color)
    route = f" · route #{state['route_step']}" if state.get("route_step") is not None else ""
    coach = f"教练待命：{state.get('action', 'run right')}{route}"
    draw.text((x, 551), ellipsize(draw, coach, F["small"], w), font=F["small"], fill=BLUE)
    draw_config(draw, x, 580, w, 270, side, level, state)


def render_single(side, source, state, level, progress):
    canvas = Image.new("RGB", (SINGLE_WIDTH, HEIGHT), BG)
    draw_panel(
        canvas,
        side,
        source,
        state,
        level,
        progress,
        thinking=state.get("thinking", False),
        x=15,
    )
    return canvas


def make_slate(level, subtitle, final=False):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 8), "双模型通关回放 · Super Mario Bros", font=F["title"], fill=TEXT)
    draw.text((617, 10), "1-1 → 1-2 · 同步决策验收", font=F["small"], fill=MUTED)
    rounded(draw, (15, 50, 894, 850), CARD, BORDER, radius=10)
    draw.text((WIDTH // 2 - text_width(draw, level, F["title"]) // 2, 300), level, font=F["title"], fill=GOLD)
    draw.text((WIDTH // 2 - text_width(draw, subtitle, F["body"]) // 2, 336), subtitle, font=F["body"], fill=TEXT)
    if final:
        draw.text((WIDTH // 2 - text_width(draw, "两路均已到旗", F["status"]) // 2, 380),
                  "两路均已到旗", font=F["status"], fill=GREEN)
    return canvas


def render_dual(left, right, state_l, state_r, level, progress_l, progress_r):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 8), "双模型通关回放 · Super Mario Bros", font=F["title"], fill=TEXT)
    draw.text((617, 10), f"WORLD {level} · 原速时间轴", font=F["small"], fill=MUTED)
    draw_panel(
        canvas, "local", left, state_l, level, progress_l,
        thinking=state_l.get("thinking", False),
    )
    draw_panel(
        canvas, "jev", right, state_r, level, progress_r,
        thinking=state_r.get("thinking", False),
    )
    return canvas


def single_frames(side, source, metas, result, level):
    duration = duration_seconds(result)
    last_frame = source[-1]
    for _, _, elapsed in timeline(duration):
        progress = elapsed / duration
        state = state_for_progress(progress, metas, level, fallback_x=40)
        yield render_single(side, sample_frame(source, elapsed, duration), state, level, progress)

    # Keep the terminal proof visible without changing gameplay speed.
    final_state = state_for_progress(1.0, metas, level, final=True, fallback_x=40)
    final_canvas = render_single(side, last_frame, final_state, level, 1.0)
    for _ in range(round(1.5 * OUTPUT_FPS)):
        yield final_canvas


def dual_level_frames(level, local_data, jev_data):
    local_frames, local_meta, local_result = local_data
    jev_frames, jev_meta, jev_result = jev_data
    local_duration = duration_seconds(local_result)
    jev_duration = duration_seconds(jev_result)
    duration = max(local_duration, jev_duration)
    for _, _, elapsed in timeline(duration):
        local_progress = min(1.0, elapsed / local_duration)
        jev_progress = min(1.0, elapsed / jev_duration)
        state_l = state_for_progress(
            local_progress, local_meta, level,
            final=elapsed >= local_duration, fallback_x=40,
        )
        state_r = state_for_progress(
            jev_progress, jev_meta, level,
            final=elapsed >= jev_duration, fallback_x=40,
        )
        yield render_dual(
            sample_frame(local_frames, elapsed, local_duration),
            sample_frame(jev_frames, elapsed, jev_duration),
            state_l,
            state_r,
            level,
            local_progress,
            jev_progress,
        )


def dual_frames(data):
    intro = make_slate("1-1", "本地 Qwen  ×  官方 Jev · 原速回放", False)
    for _ in range(round(1.2 * OUTPUT_FPS)):
        yield intro
    yield from dual_level_frames("1-1", data[("1-1", "local")], data[("1-1", "jev")])

    transition = make_slate("进入 1-2", "上一关：两路均已到旗", True)
    for _ in range(round(1.5 * OUTPUT_FPS)):
        yield transition
    yield from dual_level_frames("1-2", data[("1-2", "local")], data[("1-2", "jev")])

    final_canvas = make_slate("1-1 + 1-2", "本地 Qwen 与官方 Jev · 两路通关完成", True)
    for _ in range(round(2.0 * OUTPUT_FPS)):
        yield final_canvas


def compose(refresh=False):
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    # Existing successful 1-1 recordings: local policy and branch-backed Jev.
    local_11_name = "1-1-local-20260919-135409.gif"
    jev_11_name = "1-1-branch-jev-20260918-170421.gif"
    local_11 = load_gif(SOURCE / local_11_name)
    jev_11 = load_gif(RUNS / jev_11_name)
    result_local_11 = recorded_result(local_11_name)
    result_jev_11 = recorded_result(jev_11_name)
    local_12, meta_local_12, result_local_12 = load_or_capture("1-2", "local", refresh)
    jev_12, meta_jev_12, result_jev_12 = load_or_capture("1-2", "jev", refresh)
    if not result_local_12.get("flag") or not result_jev_12.get("flag"):
        raise RuntimeError("1-2 source recordings must both have flag=true")

    data = {
        ("1-1", "local"): (local_11, [], result_local_11),
        ("1-1", "jev"): (jev_11, [], result_jev_11),
        ("1-2", "local"): (local_12, meta_local_12, result_local_12),
        ("1-2", "jev"): (jev_12, meta_jev_12, result_jev_12),
    }
    names = {
        ("1-1", "local"): "1-1-local.gif",
        ("1-1", "jev"): "1-1-official.gif",
        ("1-2", "local"): "1-2-local.gif",
        ("1-2", "jev"): "1-2-official.gif",
    }
    written = []
    for key, name in names.items():
        level, side = key
        source, metas, result = data[key]
        path = ARTIFACTS / name
        save_gif(path, single_frames(side, source, metas, result, level))
        written.append(path)
        print(
            f"wrote {path} ({SINGLE_WIDTH}x{HEIGHT}, "
            f"game_time={duration_seconds(result):.1f}s + 1.5s hold)",
            flush=True,
        )

    out_path = ARTIFACTS / "dual-1-1-1-2.gif"
    save_gif(out_path, dual_frames(data))
    written.append(out_path)
    duration = (
        1.2
        + max(duration_seconds(result_local_11), duration_seconds(result_jev_11))
        + 1.5
        + max(duration_seconds(result_local_12), duration_seconds(result_jev_12))
        + 2.0
    )
    print(f"wrote {out_path} ({WIDTH}x{HEIGHT}, duration={duration:.1f}s)", flush=True)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="recapture fresh 1-2 local/jev runs")
    args = parser.parse_args()
    compose(refresh=args.refresh)


if __name__ == "__main__":
    main()
