"""Jev plays Super Mario Bros from a text description of the emulator RAM.

    uv run python play.py --bot jev --level 1-1                # Jev picks every action
    uv run python play_local.py --bot local --level 1-1        # 本地模型决策（本项目新增）
    uv run python play.py --bot "run and jump right"           # scripted baseline, no API calls
    uv run python play.py --dump --level 2-1                   # print what Jev would see, no API calls
    uv run python play.py --inspect runs/<log>.jsonl [-n 3]    # last decisions: state, grid, probabilities
    uv run python play.py --bot "replay:runs/<log>.jsonl@18:jump right,run right"
                                                               # replay the first 18 logged choices, then hold the tail

Each run writes runs/<level>-<bot>-<stamp>.gif, a per-decision log next to it, and one line in runs/results.jsonl.

To improve play, edit the three blocks marked EDIT HERE: the actions Jev can choose, the rules it is given,
and features(), which turns the tile grid into the summary. Reproduce a death with --inspect, test a fix with
replay (no API calls), then run Jev again.
"""

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import contextlib
import io
import warnings

import httpx
import imageio.v2 as imageio
from mentor_search import prompt_block as mentor_search_prompt
from mario_env import make_mario_env

warnings.filterwarnings("ignore")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    # gym 0.26 prints a deprecation banner at import; nes-py needs this gym version.
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

HOLD = 6  # frames per decision on the ground; 60 fps
MAX_FRAMES = 6000
STALL_FRAMES = 360  # give up after 6 game-seconds without gaining distance
HOP_FRAMES = 8  # A held this long gives a ~2.4 tile hop
FULL_JUMP_FRAMES = 64  # a full-speed jump lasts about 50 frames
RUNS = Path(__file__).resolve().parent / "runs"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
USD_PER_TOKEN = 0.042 / 1e6
# 教练模型（强 LLM）：弱模型在同一场景反复失败时，调它来分析卡点、给出指导。
MENTOR_BASE_URL = os.environ.get("MENTOR_BASE_URL", "https://openav.icu/v1")
MENTOR_API_KEY = os.environ.get("MENTOR_API_KEY", "")
MENTOR_MODEL = os.environ.get("MENTOR_MODEL", "gpt-5.6-sol")

# ----------------------------------------------------------------------------- EDIT HERE: actions
# Joypad index in SIMPLE_MOVEMENT, or BACK_OFF for the harness-executed retreat.
BACK_OFF = -2
ACTIONS = {
    "stand": 0,
    "walk right": 1,
    "jump right": 2,
    "run right": 3,
    "run and jump right": 4,
    "jump in place": 5,
    "walk left": 6,
    "hop right": 2,  # same button as "jump right", released after HOP_FRAMES
    "back off for a run-up": BACK_OFF,
}
ACTION_HELP = {
    "stand": "wait in place; use when an enemy on a wall or pipe ahead has to move away first",
    "walk right": "move right slowly",
    "jump right": "full jump while moving right at walking speed; about 4 tiles high, 3 to 5 tiles far",
    "run right": "move right fast; does not jump; builds speed for a longer jump",
    "run and jump right": "full jump at running speed; up to 5 tiles high and 9 tiles far when already at full speed",
    "jump in place": "full jump straight up",
    "walk left": "back off to the left",
    "hop right": "short low jump to the right, about 2 tiles high; lands 1 to 4 tiles ahead depending on speed; "
    "use to land on top of an enemy 2 tiles ahead",
    "back off for a run-up": "walk left until the wall or gap ahead is 6 tiles away, so that 'run right' can "
    "build full speed before jumping; do not use with an enemy behind Mario",
}
JUMPS = {2, 4, 5}
RELEASE = {2: 1, 4: 3, 5: 0}  # same action without A

# ----------------------------------------------------------------------------- EDIT HERE: rules
RULES = (
    "You control Mario in Super Mario Bros. Pick the joypad action that moves right as far as possible "
    "without dying. A jump action is held until Mario lands, so one decision is one full jump. "
    "The 'summary' field describes what is ahead; the grid is the same information drawn out. "
    "Choose in this priority order: "
    "(1) Obey every WARNING in the summary as a hard constraint; never pick an option it rules out. "
    "(2) An enemy 1-3 tiles ahead at Mario's level must be jumped over; 'run right'/'walk right' never leave the ground. "
    "(3) If a full jump would land where an enemy will be, use 'hop right' or 'jump in place' instead. "
    "(4) A gap must be jumped 1 tile before its edge. "
    "(5) Otherwise move right as far and as fast as possible. "
    "Reach: a jump clears a wall only if its reach in tiles high is at least the wall height, and a gap "
    "only if its reach far exceeds the gap width plus 1. Reach grows with speed; 'run right' reaches "
    "full speed after about 3 decisions on clear ground. "
    "For a wall or gap you cannot clear from here, back off ONCE, then 'run right' to full speed, then "
    "'run and jump right' 2-3 tiles out; never back off twice in a row. "
    "A jump arc peaks halfway, so start a jump over a wall about half the far reach ahead. "
    "Against an enemy ahead, jump when it is 2-3 tiles away; a hop clears or stomps one enemy and lands sooner. "
    "Enemies appear at the right edge as Mario advances, so a jump that lands past what is visible lands blind. "
    "An enemy 1-2 tiles behind Mario will hit him within a second: jump immediately."
)
GRID_LEGEND = (
    "Text grid, 13 rows x 20 columns, each cell one 16px tile. Row 7 is Mario's row. "
    "M = Mario (column 5). # = solid ground, brick, block or pipe. . = empty air. Letters are enemies: "
    "G goomba, K koopa, S koopa shell, F flying koopa, P piranha plant, B buzzy beetle, H hammer brother. "
    "Mario walks right (toward higher columns). Falling into a column with no # below Mario is death. "
    "Touching an enemy from the side is death; landing on a goomba or koopa from above kills it. "
    "A stomped koopa leaves a shell that kills on touch and slides when kicked. A piranha plant cannot be "
    "stomped. A flying koopa bounces along the ground."
)

# Measured in this emulator: a full jump (A held until landing) at a given horizontal speed.
# (min speed byte, tiles high, tiles far). Height comes from the hold; distance from speed.
JUMP_TABLE = [(30, 5, 9), (15, 4, 5), (0, 4, 3)]
HOP_TABLE = [(30, 5), (15, 2), (0, 1)]  # (min speed byte, tiles far) for a hop; measured at 40, 28, 0


def jump_reach(v: int) -> tuple[int, int]:
    for min_v, high, far in JUMP_TABLE:
        if abs(v) >= min_v:
            return high, far
    return JUMP_TABLE[-1][1:]


def speed_word(v: int) -> str:
    # SMB horizontal speed byte 0x57, signed. Walking tops out near 28, running near 48.
    if v >= 32:
        return "running at full speed"
    if v >= 8:
        return "moving right slowly"
    if v <= -8:
        return "moving left"
    return "standing still"


# ----------------------------------------------------------------------------- EDIT HERE: state description
def features(g: str, v: int = 0, airborne: bool | None = None, visible: int = 15) -> dict:
    """Turn the grid into fields and a summary sentence. Mario is row 6, column 4; right is +column."""
    rows = g.splitlines()
    look = range(1, 9)
    # The grid is Mario-relative, so mid-jump his own row is air. Anchor on the ground under him.
    ground = next((r for r in range(7, 13) if rows[r][4] == "#"), 7)
    feet = ground - 1
    on_ground = (not airborne) if airborne is not None else ground == 7
    f = {"on_ground": on_ground, "wall_ahead": None, "gap_ahead": None, "enemy_ahead": None}
    for dx in look:
        if rows[feet][4 + dx] == "#":
            height = sum(1 for r in range(feet, -1, -1) if rows[r][4 + dx] == "#")
            f["wall_ahead"] = {"tiles": dx, "height": height}
            break

    def bottomless(col: int) -> bool:
        # No solid tile from the ground row to the bottom of the grid; a step down is not a gap.
        return all(rows[r][col] == "." for r in range(ground, 13))

    for dx in look:
        if bottomless(4 + dx):
            width = 0
            while 4 + dx + width < 20 and bottomless(4 + dx + width):
                width += 1
            f["gap_ahead"] = {"tiles": dx, "width": width}
            break
    # Enemies anywhere ahead, with height above Mario's feet (0 = same level).
    seen = []
    for dx in range(1, 16):
        for r in range(13):
            if rows[r][4 + dx] in ENEMY_LETTERS:
                seen.append({"tiles": dx, "up": feet - r, "kind": ENEMY_NAME[rows[r][4 + dx]]})
    f["enemies"] = seen
    enemies = [e["tiles"] for e in seen if -1 <= e["up"] <= 1]
    f["enemies_ahead"] = enemies
    f["enemy_ahead"] = {"tiles": enemies[0]} if enemies else None
    f["enemies_behind"] = [dx for dx in range(1, 5)
                           if any(rows[r][4 - dx] in ENEMY_LETTERS for r in (feet - 1, feet, ground))]
    behind = 0
    while behind < 4 and rows[feet][3 - behind] == "." and rows[ground][3 - behind] == "#":
        behind += 1
    f["clear_behind"] = behind

    parts, warns = [], []
    w, gp = f["wall_ahead"], f["gap_ahead"]
    parts.append(f"A solid wall {w['height']} tiles tall is {w['tiles']} tile(s) ahead." if w else "No wall ahead.")
    parts.append(f"There are {behind} tiles of clear ground behind Mario for a run-up.")
    parts.append(f"A gap {gp['width']} tiles wide is {gp['tiles']} tile(s) ahead." if gp else "Solid ground ahead.")
    if seen:
        def where(en: dict) -> str:
            if en["up"] > 1:
                return f"a {en['kind']} {en['tiles']} tiles ahead and {en['up']} tiles up (on top of something)"
            if en["up"] < -1:
                return f"a {en['kind']} {en['tiles']} tiles ahead and {-en['up']} tiles below"
            return f"a {en['kind']} {en['tiles']} tiles ahead at ground level"
        parts.append("Enemies: " + "; ".join(where(en) for en in seen[:4]) + ".")
    else:
        parts.append("No enemy ahead.")
    if f["enemies_behind"]:
        parts.append(f"An enemy is {f['enemies_behind'][0]} tile(s) behind Mario; walking left into it is death.")
    f["speed"] = speed_word(v)
    high, far = jump_reach(v)
    hop_far = HOP_TABLE[0][1] if abs(v) >= HOP_TABLE[0][0] else (HOP_TABLE[1][1] if abs(v) >= HOP_TABLE[1][0] else HOP_TABLE[2][1])
    f["hop_lands_tiles_ahead"] = hop_far
    f["visible_tiles_ahead"] = visible
    f["jump_reach_now"] = {"tiles_high": high, "tiles_far": far}
    f["jump_reach_at_full_speed"] = {"tiles_high": JUMP_TABLE[0][1], "tiles_far": JUMP_TABLE[0][2]}
    parts.append(("Mario is on the ground, " if f["on_ground"] else "Mario is in the air, ") + f["speed"] + ".")
    parts.append(f"A jump from this speed clears {high} tiles high and {far} tiles far; "
                 f"at full running speed it clears {JUMP_TABLE[0][1]} high and {JUMP_TABLE[0][2]} far.")
    if w:
        # The arc peaks halfway, so a wall must be about half the far reach away when the jump starts.
        if high >= w["height"]:  # verified by replay: a 4-tile standing jump lands on a 4-tall ledge
            parts.append(f"Start the jump when the wall is about {max(1, far // 2)} tiles ahead.")
        else:
            warns.append("A jump from this speed is not high enough for the wall ahead; more speed is needed.")
    # Headroom: blocks above the arc cut a jump short.
    headroom = 13
    for dx in range(0, min(far, 15) + 1):
        for r in range(feet - 1, -1, -1):
            if rows[r][4 + dx] == "#":
                headroom = min(headroom, feet - r - 1)
                break
    if headroom < high:
        eff_far = max(1, round(far * headroom / high))
        f["headroom"] = headroom
        parts.append(f"Blocks overhead {headroom + 1} tiles up cap the jump: it would land about {eff_far} tiles ahead instead of {far}.")
    else:
        eff_far = far
    # Keep the physics estimate available to the execution guard.  The raw
    # reach is optimistic under a low ceiling (the x≈1196/1339 failures had
    # raw far=5/9 but actually landed around 2/4 tiles).
    f["jump_lands_tiles_ahead"] = eff_far
    # Landing zone: enemies walk toward Mario about 2 tiles during a one-second jump.
    land = [en for en in seen if abs(en["up"]) <= 1 and eff_far - 4 <= en["tiles"] <= eff_far + 3]
    f["enemies_near_landing_spot"] = [en["tiles"] for en in land]
    parts.append(f"A full jump right now would land about {eff_far} tiles ahead; a hop about {hop_far}. "
                 f"The screen shows {visible} tiles ahead; enemies beyond that are unknown until Mario moves closer.")
    if land:
        warns.append(f"Do not take a full jump now: it lands about {eff_far} tiles ahead, where an enemy will be by then.")
    if eff_far >= visible:
        warns.append("A full jump now lands past what is visible, on unknown ground.")
    on_wall = [en for en in seen if w and en["tiles"] in (w["tiles"], w["tiles"] + 1) and en["up"] >= 1]
    if on_wall:
        clear_over = JUMP_TABLE[0][1] >= w["height"] + 2
        warns.append("An enemy is on top of the wall ahead: jumping onto it is death. "
                     + ("A full-speed jump started 4 tiles before the wall clears the wall and the enemy together."
                        if clear_over else "Back off, wait for it to move, then jump when the top is clear."))
    if enemies and enemies[0] <= 1:
        warns.append("An enemy is right in front of Mario: moving toward it is death. Jump in place if standing still; "
                     "momentum carries into a jump, so at speed a full jump forward is the only way over it.")
    if f["enemies_behind"] and f["enemies_behind"][0] <= 2:
        warns.append("The enemy behind reaches Mario in about a second: jump over it or away from it, but not into another enemy.")
    f["warnings"] = warns
    f["summary"] = (("WARNINGS: " + " ".join(warns) + " ") if warns else "") + " ".join(parts)
    return f


# ----------------------------------------------------------------------------- scripted twin
def policy(f: dict) -> str:
    """RULES as a deterministic if-chain over the same fields Jev gets. The 'rules' bot; no API calls."""
    v_full = f["speed"] == "running at full speed"
    high, far = f["jump_reach_now"]["tiles_high"], f["jump_reach_now"]["tiles_far"]
    w, gp = f["wall_ahead"], f["gap_ahead"]
    ahead = f["enemies_ahead"]
    landing_bad = bool(f["enemies_near_landing_spot"])
    forward_jump = "run and jump right" if v_full else "jump right"
    if ahead and ahead[0] <= 1:
        return "jump in place" if f["speed"] == "standing still" else forward_jump
    # Keep momentum when a landing warning is several tiles out.  A prior
    # fallback converted every 4-8 tile warning to an in-place jump, which
    # repeatedly met the approaching enemy around x≈670.  The point-blank
    # branch above remains responsible for vertical escapes.
    if ahead and 4 <= ahead[0] <= 8 and landing_bad:
        return forward_jump
    if f["enemies_behind"] and f["enemies_behind"][0] <= 2:
        return "jump in place" if landing_bad else forward_jump
    on_wall = [e for e in f["enemies"] if w and e["tiles"] in (w["tiles"], w["tiles"] + 1) and e["up"] >= 1]
    if on_wall:
        if JUMP_TABLE[0][1] >= w["height"] + 2:
            if v_full and 3 <= w["tiles"] <= 5:
                return "run and jump right"
            return "run right" if w["tiles"] >= 6 else "back off for a run-up"
        return "stand"
    # A raised Goomba/Koopa is an early warning for the low-ceiling section:
    # by the time it appears at Mario's level, the next six-frame decision is
    # already too late (the x≈1184 death).  Launch while it is still above the
    # route and the gap is at least four tiles away; the gap guard owns the
    # final approach once its edge is visible.
    upper = [e for e in f["enemies"] if e.get("up", 0) >= 2 and e.get("tiles", 99) <= 4]
    if upper and f.get("headroom", 13) <= 2 and (not gp or gp.get("tiles", 99) >= 4):
        return forward_jump
    if w and w["tiles"] <= 8:
        start = max(1, far // 2)
        if high < w["height"]:
            return "run right" if w["tiles"] >= 6 else "back off for a run-up"
        if start - 1 <= w["tiles"] <= start + 1:
            return "jump in place" if landing_bad else forward_jump
        if w["tiles"] > start + 1:
            return "run right"
        return "jump right"
    if gp and gp["tiles"] <= 8:
        if far > gp["width"] + 1:
            return forward_jump if gp["tiles"] <= 1 else "run right"
        return "run right" if gp["tiles"] >= 6 else "back off for a run-up"
    if ahead and 2 <= ahead[0] <= 3:
        blind = f["jump_reach_now"]["tiles_far"] >= f["visible_tiles_ahead"]
        return "hop right" if (landing_bad or blind) else forward_jump
    return "run right"


# ----------------------------------------------------------------------------- emulator and RAM
def load_env() -> None:
    p = Path(__file__).resolve().parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def nes(env):
    """Walk the wrapper chain to the nes_py env that owns the RAM."""
    e = env
    while not hasattr(e, "ram"):
        e = e.env
    return e


def tile(ram, x: int, y: int) -> int:
    # SMB keeps two 16x13 tile pages at 0x500, 208 bytes each; y offset 32 is the HUD.
    page = (x // 256) % 2
    sx, sy = (x % 256) // 16, (y - 32) // 16
    if sy < 0 or sy > 12:
        return 0
    return int(ram[0x500 + page * 208 + sy * 16 + sx])


# Enemy type byte 0x16+slot. Verified in-game: 6 goomba, 0 green koopa, 13 piranha plant, 14 paratroopa.
# The rest follow the standard SMB RAM map and are UNVERIFIED here.
ENEMY_LETTER = {0: "K", 1: "K", 2: "B", 3: "K", 4: "K", 5: "H", 6: "G", 13: "P", 14: "F", 15: "F", 16: "F", 9: "F"}
ENEMY_NAME = {"G": "goomba", "K": "koopa", "P": "piranha plant", "F": "flying koopa", "B": "buzzy beetle",
              "H": "hammer brother", "S": "koopa shell", "E": "enemy"}
ENEMY_LETTERS = set(ENEMY_NAME)


def grid(ram) -> tuple[str, int, int]:
    """13 rows x 20 columns around Mario: 6 rows above, 4 columns behind, 15 ahead. Letters mark enemy types."""
    mx = int(ram[0x6D]) * 256 + int(ram[0x86])
    my = int(ram[0x03B8]) + 16
    enemies = []
    for i in range(5):
        if ram[0x0F + i]:
            letter = ENEMY_LETTER.get(int(ram[0x16 + i]), "E")
            if letter == "K" and int(ram[0x1E + i]) in (2, 3):  # UNVERIFIED: stomped koopa state
                letter = "S"
            # +8 puts a ground enemy in Mario's row; verified against the first goomba.
            enemies.append((int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]), int(ram[0xCF + i]) + 8, letter))
    rows = []
    for dy in range(-6, 7):
        row = []
        for dx in range(-4, 16):
            x, y = mx + dx * 16, my + dy * 16
            ch = "#" if tile(ram, x, y) else "."
            for ex, ey, letter in enemies:
                if abs(ex - x) <= 8 and abs(ey - y) <= 8:
                    ch = letter
            if dx == 0 and dy == 0:
                ch = "M"
            row.append(ch)
        rows.append("".join(row))
    return "\n".join(rows), mx, my


def speed(ram) -> int:
    v = int(ram[0x57])
    return v - 256 if v > 127 else v


def airborne(ram) -> bool:
    return bool(ram[0x1D])  # 0 on the ground, 1 for the whole jump; verified against the y trace


def ask_mentor(death_info: dict, lessons) -> str | None:
    """用强 LLM 分析「反复失败的场景」，返回一句可执行的指导。

    弱模型（jev / 本地 3B）在同一场景签名失败多次后，把死亡现场 + 历史 + 已学教训
    交给教练模型，让它给出「下次这一步具体该怎么做」的指导。
    """
    # 动态读环境变量（.env 在 import 之后才 load_env，模块级常量会读到空）
    api_key = os.environ.get("MENTOR_API_KEY", "")
    base_url = os.environ.get("MENTOR_BASE_URL", "https://openav.icu/v1")
    model = os.environ.get("MENTOR_MODEL", "gpt-5.6-sol")
    if not api_key:
        return None
    def _line(s):
        if isinstance(s, dict):
            t = s["text"]
            x = s.get("x", s.get("last_x"))
            if x is None and s.get("positions"):
                x = s["positions"][-1]
            if x is not None:
                return f"[died at x≈{x}] {t}"
            return t
        return s

    failed_context = (
        f"The previous coach plan was '{death_info.get('failed_plan')}' and its first action "
        f"'{death_info.get('failed_first_action')}' failed {death_info.get('failed_plan_failures', 0)} times. "
        "You MUST choose a different first action and rethink the timing.\n"
        if death_info.get("failed_plan") else ""
    )
    mismatch_context = (
        f"The previous coach plan was '{death_info.get('mismatched_plan')}', but the execution trace shows "
        "that at least one requested phase was rewritten by the safety guard. Diagnose the rewrite and timing; "
        "do not treat the requested first action as tested or failed.\n"
        if death_info.get("mismatched_plan") else ""
    )
    web_context = mentor_search_prompt(death_info.get("web_search"))
    web_block = f"\n{web_context}\n" if web_context else ""
    trace_rows = death_info.get("mentor_trace") or []
    trace_block = ""
    if trace_rows:
        trace_block = (
            "MENTOR EXECUTION TRACE (requested -> actual; source=mentor_guard means the safety layer rewrote it):\n"
            + "\n".join(
                f"- phase={row.get('phase')} x={row.get('x')} "
                f"requested='{row.get('requested_action', row.get('action'))}' "
                f"actual='{row.get('action')}' source={row.get('source')}"
                for row in trace_rows[-8:]
            )
            + "\n\n"
        )
    prompt = (
        "You are a failure-analysis coach for a small AI that keeps dying at the same spot in Super Mario Bros. "
        "Infer the cause from the measured death scene, predicted landing, enemy motion, and recent decisions; "
        "do not blindly repeat the old policy or assume a fixed solution. Compare the failed action with at least "
        "one plausible alternative internally, then give ONE concrete short plan it can follow next time.\n\n"
        f"DEATH SCENE:\n{death_info.get('summary', '')}\n\n"
        f"GRID:\n{death_info.get('grid', '')}\n\n"
        f"RECENT DECISIONS:\n{death_info.get('history', '')}\n\n"
        + trace_block
        + f"LESSONS LEARNED (with where it died):\n" + ("\n".join(_line(s) for s in lessons) if lessons else "(none)") + "\n\n"
        + web_block
        + f"The last actual action was '{death_info.get('applied_action') or 'unknown'}' and it failed. "
        "First determine whether the requested coach action was actually executed or was rewritten by the safety guard; "
        "do not blame or ban a plan that never ran as requested. "
        + failed_context
        + mismatch_context
        + "The last actual action may still be useful at a different distance or after a preparation phase; "
        "only avoid it as the first action when the FAILED PLAN note explicitly requires that. "
        "Treat the WARNING text as a hard physical constraint: "
        "only wait if the current distance leaves a safe waiting window; if an enemy is already one tile ahead "
        "or right in front, waiting is unsafe and a vertical escape may be required. "
        "Reply in English using exactly two lines:\n"
        "REFLECTION: <one short causal finding from the death scene>\n"
        "PLAN: <first action>; THEN: <next action or none>.\n"
        "Use only these action names: stand, walk right, run right, jump right, run and jump right, "
        "jump in place, hop right, walk left, back off for a run-up. Add at most one short reason."
    )
    try:
        r = httpx.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 200, "temperature": 0.3},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"[mentor] 调用失败: {exc}", flush=True)
        return None


# 教练返回的是自由文本，而执行器只接受 ACTIONS 中的精确名称。保守地从
# 指导开头找一个动作，并跳过 "do not ..." 这类否定语句，避免把解释里的
# 反例误当成真正建议。别名只映射到确实存在的动作，无法识别就返回 None。
_MENTOR_ACTION_PHRASES = (
    ("back off for a run-up", "back off for a run-up"),
    ("run and jump right", "run and jump right"),
    ("jump in place", "jump in place"),
    ("jump right", "jump right"),
    ("hop right", "hop right"),
    ("walk right", "walk right"),
    ("run right", "run right"),
    ("walk left", "walk left"),
    ("run left", "walk left"),
    ("move left", "walk left"),
    ("back off", "back off for a run-up"),
    ("retreat", "back off for a run-up"),
    ("stand", "stand"),
    ("stand still", "stand"),
    ("wait", "stand"),
)


def mentor_text(mentor) -> str:
    """Return displayable guidance while accepting old string callers."""
    if isinstance(mentor, dict):
        return str(mentor.get("text") or "")
    return str(mentor or "")


def parse_mentor_action(mentor) -> str | None:
    """Map mentor free text to one legal action, or reject it safely."""
    if isinstance(mentor, dict):
        explicit = mentor.get("action")
        if explicit in ACTIONS:
            return explicit
        text = mentor.get("text", "")
    else:
        text = mentor or ""
    if not isinstance(text, str) or not text.strip():
        return None
    normalized = text.lower().replace("\u2013", "-").replace("\u2014", "-")
    occurrences = []
    for phrase, action in _MENTOR_ACTION_PHRASES:
        pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
        for hit in re.finditer(pattern, normalized):
            occurrences.append((hit.start(), hit.end(), -len(phrase), action))
    if not occurrences:
        return None
    # Mark the action phrase directly following a negation as non-executable.
    # This handles overlapping phrases ("run and jump right" / "jump right")
    # and long explanations such as "do not, under any circumstance, use ...".
    negative_spans = []
    for neg in re.finditer(r"\b(?:do not|don't|never|avoid|without)\b", normalized):
        boundary = re.search(r"[;.!?\n]", normalized[neg.end():])
        limit = neg.end() + (boundary.start() if boundary else len(normalized))
        candidates = [o for o in occurrences if neg.end() <= o[0] < limit]
        if candidates:
            first_start = min(o[0] for o in candidates)
            same_start = [o for o in candidates if o[0] == first_start]
            end = max(o[1] for o in same_start)
            negative_spans.append((neg.start(), end))

    def overlaps_negative(item):
        start, end = item[0], item[1]
        return any(start < stop and end > begin for begin, stop in negative_spans)

    # Earliest positive phrase wins; for overlapping positive phrases keep the
    # longest one so "run and jump right" is not shortened to "jump right".
    matches = sorted((o for o in occurrences if not overlaps_negative(o)),
                     key=lambda o: (o[0], o[2]))
    for item in matches:
        start, end, _length, action = item
        if any(start < other_end and end > other_start
               for other_start, other_end, _x, _a in matches
               if other_start < start):
            continue
        return action
    return None


def parse_mentor_plan(mentor, failed_action: str | None = None) -> dict | None:
    """Parse a one-step or wait-then-follow-up mentor plan.

    The returned object is deliberately small so the runtime can execute only
    legal actions: ``{action, follow_up, wait_decisions, text}``.
    """
    text = mentor_text(mentor)
    if not text:
        return None
    explicit_action = mentor.get("action") if isinstance(mentor, dict) else None
    explicit_follow = mentor.get("follow_up") if isinstance(mentor, dict) else None
    reflection = mentor.get("reflection") if isinstance(mentor, dict) else None
    # The coach may include a causal REFLECTION line before PLAN.  Restrict
    # free-text parsing to the PLAN line so action names mentioned as failed
    # alternatives in the reflection cannot become executable actions.
    plan_match = re.search(r"(?im)^\s*plan\s*:\s*(.+)$", text)
    plan_text = plan_match.group(1).strip() if plan_match else text
    if plan_match and reflection is None:
        reflection_match = re.search(r"(?im)^\s*reflection\s*:\s*(.+)$", text)
        if reflection_match:
            reflection = reflection_match.group(1).strip()
    if explicit_action in ACTIONS:
        first = explicit_action
    else:
        then = re.search(r"\b(?:then|after that|next)\s*[:,-]?", plan_text, re.IGNORECASE)
        first_text = plan_text[:then.start()] if then else plan_text
        first = parse_mentor_action(first_text)
        if re.search(r"\b(?:wait|pause|hold position|stay still|let .* pass)\b", first_text, re.IGNORECASE):
            first = "stand"
        explicit_follow = explicit_follow or (parse_mentor_action(plan_text[then.end():]) if then else None)
    if first not in ACTIONS:
        return None
    follow = explicit_follow if explicit_follow in ACTIONS else None
    if failed_action and first == failed_action:
        return None
    return {"text": text, "action": first, "follow_up": follow,
            "reflection": reflection,
            "wait_decisions": 1 if first == "stand" and follow else 0}


# ----------------------------------------------------------------------------- Jev
JEV_MEMORY_NOTE = (
    " The state may carry 'recent_decisions' (your last few choices with the x position after each, "
    "plus a STUCK line when a choice made no progress - never repeat a STUCK action) and "
    "'past_failures' (lessons from earlier deaths in previous runs; do not repeat those mistakes). "
    "Obey them when present; ignore them when absent."
)

JEV_DECISION_PROTOCOL = (
    " The JSON state is the complete measured game state. Read state.summary and its WARNINGS as hard "
    "physical constraints, then use the same action meanings listed in criteria. Decision priority is: "
    "(1) never violate a warning or choose a jump that the measured landing makes impossible; "
    "(2) if coach_plan is present, execute its first action exactly for this state; "
    "if it has a follow-up, expect that phase on the next decision and do not replace it with an old heuristic; "
    "(3) apply the normal enemy/gap/wall rules; (4) move right. "
    "A coach plan is advice from a separate failure-analysis LLM, not a reason to repeat an action marked failed. "
    "Return exactly one offered action choice."
)


def ask_jev(client: httpx.Client, state: dict, history=(), lessons=(), experience=None, mentor="") -> tuple[str, dict, int, float]:
    enriched = dict(state)
    if history:
        enriched["recent_decisions"] = _history_block(history)
    if lessons:
        enriched["past_failures"] = _rank_lessons(state, lessons)
    if experience:
        act = experience.get(scene_signature(state))
        if act:
            enriched["past_experience"] = (
                f"Last time in this exact situation you died; the correct action was '{act}'.")
    if mentor:
        enriched["coach_guidance"] = mentor_text(mentor)
        plan = parse_mentor_plan(mentor)
        if plan:
            enriched["coach_plan"] = {
                "first": plan.get("action"),
                "follow_up": plan.get("follow_up"),
                "wait_decisions": plan.get("wait_decisions", 0),
                "plan_id": mentor.get("plan_id") if isinstance(mentor, dict) else None,
                "origin": mentor.get("origin") if isinstance(mentor, dict) else "llm",
            }
    # Give System One the same compact, human-readable decision context that
    # the local chat prompt receives; the structured fields remain available
    # for its evaluator, while this removes an avoidable prompt-format gap.
    enriched["decision_context"] = (
        f"STATE SUMMARY:\n{state.get('summary', '')}\n"
        f"RECENT DECISIONS:\n{_history_block(history)}\n"
        f"PAST FAILURES:\n{json.dumps(enriched.get('past_failures', []), ensure_ascii=False)}\n"
        f"GRID:\n{state.get('grid', '')}"
    )
    body = {
        "state": enriched,
        "model": "jev-latest",
        "questions": {"action": {"type": "choice",
                                 "instructions": RULES + GRID_LEGEND + JEV_MEMORY_NOTE + JEV_DECISION_PROTOCOL,
                                 "criteria": ACTION_HELP}},
    }
    t0 = time.perf_counter()
    r = client.post(JEV_URL, headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"}, json=body)
    lat = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    a = d["answers"]["action"]
    return a["choice"], a["probabilities"], d["usage"]["input_tokens"], lat


# ----------------------------------------------------------------------------- 本地模型
LOCAL_POLICY_BASE_URL = os.environ.get("LOCAL_POLICY_BASE_URL", "http://127.0.0.1:11500/v1")
LOCAL_POLICY_API_KEY = os.environ.get("LOCAL_POLICY_API_KEY", "local")
LOCAL_POLICY_MODE = os.environ.get("LOCAL_POLICY_MODE", "chat")
LOCAL_TEMPERATURE = float(os.environ.get("LOCAL_POLICY_TEMPERATURE", "0.5"))
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 进提示词用的动作描述。**不要压缩** —— 实测把 ACTION_HELP 的长描述换成短短语后，
# 1-1 从 3161(通关) 掉到 722：那些描述承载的是"这个动作到底做什么"的语义
# （比如 hop right 是"低跳、约 2 格高、用来踩 2 格外的敌人"），删掉就没法选了。
# 省下的那点 token（约 130 个 ≈ 0.2 秒）换不来任何东西。
OPTION_HINTS = ACTION_HELP

CHAT_SYSTEM = """You play Super Mario Bros one decision at a time. You get the state and a multiple-choice question.
Answer with the letter of the best option. Reply with exactly one letter and nothing else.

How to choose, in order:
1. If the state begins with WARNINGS, obey every warning as a hard constraint. Never choose an
   option a warning rules out, even if it looks like progress.
2. An enemy 1 to 3 tiles ahead at Mario's level must be jumped over: choose a jump option.
   "run right" and "walk right" never leave the ground, so they walk straight into it.
3. If the state says a full forward jump would land where an enemy will be, choose "hop right" or
   "jump in place" instead of a full jump.
4. A gap must be jumped before its edge.
5. Otherwise move right as far and as fast as possible.
6. If the RECENT DECISIONS block says STUCK, do NOT choose that action again - pick a different one.

Example: the state says an enemy is 1 tile ahead at full speed, and warns that a full jump lands
about 4 tiles ahead where an enemy will be by then. A short hop clears the enemy in front without
landing on the next one, so the answer is the letter for "hop right".

Grid legend: M Mario, # solid ground/brick/block/pipe, . empty air,
G goomba, K koopa, S shell, F flying koopa, P piranha plant, B buzzy beetle, H hammer brother.
Row 7 is Mario's row; Mario walks toward higher columns."""


def _state_block(state: dict) -> str:
    """只给 summary + 上一动作 + 网格。

    RULES / GRID_LEGEND 是给 Jev 的 system-one 模型写的长说明，对 3B 本地模型是纯噪声，
    summary 里已经用自然语言复述了所有关键几何量，网格逐格可读，够用。
    """
    lines = [state["summary"], f"action just taken: {state.get('action_before')}", "grid:", state["grid"]]
    return "\n".join(lines)


def _letter_to_action(text: str, labels: str, keys: list) -> str | None:
    """严格解析：只认一个独立的候选字母，认不出就返回 None 交给兜底。"""
    if not text:
        return None
    stripped = text.strip()
    # 最短优先：整段回复就是一个字母，或 "E"、"E)"、"(E)"、"E." 这类。
    head = stripped[:8]
    for label in labels:
        if head.startswith(label) and not head[1:2].isalpha():
            return keys[labels.index(label)]
    # 退一步：在回复里找唯一出现的候选字母。
    found = [label for label in labels if label in stripped.upper()]
    return keys[labels.index(found[0])] if len(found) == 1 else None


def _history_block(history) -> str:
    """把最近几次决策连 x 的变化一起摊开。

    这一块是补上模型看不到的东西：它只看得到"当前这一帧"，不知道上一步自己选了什么、
    选完 x 有没有涨。实测它会在同一个位置连选 4 次 'hop right'、x 一点不动，白烧 140 帧，
    所以除了列出历史，还要显式标出 STUCK。
    """
    if not history:
        return "  (this is the first decision of the episode)"
    lines = []
    previous_x = None
    for choice, x in history:
        delta = "" if previous_x is None else f" ({x - previous_x:+d})"
        lines.append(f"  {choice} -> x={x}{delta}")
        previous_x = x
    # 尾部连续同一个动作、且每次记录到的 x 完全一样 => 卡住了
    repeat, stuck = 1, None
    for i in range(len(history) - 1, 0, -1):
        if history[i][0] == history[i - 1][0] and history[i][1] == history[i - 1][1]:
            repeat += 1
        else:
            break
    if repeat >= 3:
        stuck = history[-1][0]
        lines.append(f"  STUCK: '{stuck}' was chosen {repeat} times in a row and x did NOT increase. "
                     f"Do not pick '{stuck}' again.")
    return "\n".join(lines)


# 卡住时的升级链：同一个动作连续没让 x 前进，就按这个顺序换成"更强的跳跃"。
# 顺序按"跨度从小到大"，而不是按字母序 —— 先用代价最小的手段试着脱困。
ESCALATION = ("hop right", "jump right", "run and jump right", "jump in place",
              "back off for a run-up", "run right")


def unstick(name: str, log: list, x_now: int) -> str:
    """确定性兜底：模型在原地打转时，替它换一个还没试过、更强的动作。

    为什么不能靠提示词：RECENT DECISIONS 里已经显式写了 STUCK 并在 system prompt 里
    禁止重复该动作，但实测模型照样重复 —— 1-1 通关那局在 x=722 被标 STUCK 之后，
    又连选了 3 次 'hop right'，白白烧掉 150 帧。所以这一层必须执行前拦住。

    log 里每条记录的 x 是"那次决策时刻"的 x，x_now 是本次决策时刻的 x，
    因此"末尾连续若干条 x >= x_now"就等于"这些决策都没让 Mario 前进"。
    只在模型已经用行动证明推不动之后才介入；一旦有进展就完全不干预。
    """
    # 后退防护：back off 退完就该 run right 助跑，连续第二次后退是白烧帧。
    # 官方 Jev 实测在 x=433 附近连选 8 次 back off，x 一路退到 337。
    if name in ("back off for a run-up", "walk left") and log and \
            log[-1]["choice"] in ("back off for a run-up", "walk left"):
        return "run right"
    no_progress = 0
    for row in reversed(log):
        if row["x"] >= x_now:
            no_progress += 1
        else:
            break
    if no_progress < 2:
        return name
    tried = {row["choice"] for row in log[-no_progress:]}
    if name not in tried:
        # 模型自己换了一个没试过的动作（比如从 hop 升到 run and jump），先让它试
        return name
    for candidate in ESCALATION:
        if candidate not in tried:
            return candidate
    return name


def hazard_guard(name: str, state: dict, preserve_mentor_timing: bool = False) -> str:
    """把"物理上过不去"的动作换成过得去的那个。

    这些量 features() 全都已经算好了，模型却算不清 —— 实测它在 3 格宽的坑前选
    'hop right'（射程 2 格）直接掉进去，这是 1-1 上最确定的一次死法。
    只要"射程 <= 坑的起点 + 宽度"，落点就在坑里，这种选择不需要模型来判断。

    只在前方 1 格就是坑/墙时介入；障碍还远时不插手，避免破坏它的正常节奏。
    """
    reach = state.get("jump_reach_now") or {}
    far = reach.get("tiles_far", 0)
    high = reach.get("tiles_high", 0)
    effective_far = state.get("jump_lands_tiles_ahead", far)
    hop = state.get("hop_lands_tiles_ahead", 0)
    wall = state.get("wall_ahead")

    gap = state.get("gap_ahead")
    if gap:
        edge = gap["tiles"]
        need = edge + gap["width"]

        def falls_in_gap(reach):
            # The estimate is rounded to whole tiles. Treat a landing within
            # one tile of the far lip as a successful crossing; otherwise the
            # rounded value is overly pessimistic at the x≈1262 approach
            # (the real arc clears the last gap column). A landing deeper in
            # the gap is still a hard stop.
            return edge <= reach and reach + 1 < need

        # hop 落在坑里 = 必掉进去。
        # hop 只差 1 格贴在坑边时，只有"紧邻还有一级台阶"才算危险：台阶把起跳点顶死了，
        # hop 落地后恢复常规移动的那一格正好滑进坑。
        # 依据：1068 与 2404 两处的 gap 与射程数值完全相同(2/3、jump 5、hop 2)，
        # 唯一差别就是 2404 前方 1 格有 1 格高的台阶 —— 前者安全，后者死。
        # 判据必须带上 wall，否则会把 1068 这种本来能过的点也改掉（实测 2470 → 1946）。
        hop_doomed = falls_in_gap(hop) or (wall is not None and wall["tiles"] <= 1
                                           and hop >= edge - 1)
        if name == "hop right" and hop_doomed:
            name = "jump right" if far >= need else "run and jump right"
        # Re-check a hop that was upgraded above: under a low ceiling the
        # resulting full jump can still have a shortened landing and must not
        # be allowed to fall into the same gap.
        if name in ("jump right", "run and jump right") and falls_in_gap(effective_far):
            # A forward jump whose *actual* landing is at/near the gap cannot
            # be repaired by changing jump flavour.  Advance on the ground
            # while there is still a safe runway; the next decision sees a
            # smaller edge distance and can launch from the correct point.
            name = "run right" if edge > 1 else "run and jump right"
        elif name in ("walk right", "run right") and edge <= 2:
            # 不含跳跃的动作：走到坑边就直接掉下去
            if effective_far + 1 >= need:
                name = "jump right"
            elif edge > 1:
                # Low ceiling: one more short ground advance is safer than a
                # jump that lands before the far lip.  At edge==1 there is no
                # runway left, so use the strongest available jump.
                name = "run right"
            else:
                name = "run and jump right"

    if wall and wall["tiles"] <= 1:
        # hop 只有约 2 格高（见 ACTION_HELP），越不过更高的墙，只会撞上去
        if name == "hop right" and wall["height"] > 2:
            name = "jump right" if high >= wall["height"] else "run and jump right"
        elif name in ("stand", "walk right", "run right", "jump right"):
            # Ground actions cannot clear a wall one tile ahead.  If the
            # current jump is capped by a low ceiling, a walking-speed jump
            # is also a dead end: the 1-2 x≈365 sample had a 4-tile wall,
            # standing speed, and headroom=0, then repeated ``jump right``.
            # Keep the acceleration button so the next jump can clear it.
            headroom = state.get("headroom")
            capped = headroom is not None and headroom < high
            name = "run and jump right" if capped or high < wall["height"] else "jump right"

    ahead = state.get("enemies_ahead") or []
    # Raised enemies move into Mario's lane while the next decision is being
    # computed.  Start the full jump while they are still overhead; this is
    # deliberately limited to a clear route so wall/gap guards remain the
    # final authority near an actual landing hazard.
    upper = [e for e in (state.get("enemies") or [])
             if e.get("up", 0) >= 2 and e.get("tiles", 99) <= 4]
    if (upper and not ahead and state.get("headroom", 13) <= 2
            and (not gap or gap.get("tiles", 99) >= 4)
            and name in ("stand", "walk right", "run right")):
        name = "run and jump right" if state.get("speed") == "running at full speed" else "jump right"

    # ---- low-ceiling timing window ---------------------------------------
    # In 1-2 the enemies on the overhead platform descend into Mario's lane
    # while a 3-tile gap is still several tiles away.  Waiting until they show
    # up in ``enemies_ahead`` is too late: at x~1175 every jump collides at the
    # ceiling/enemy.  The safe window is the frame where an upper enemy is four
    # tiles ahead (headroom=2); a full-speed forward jump then clears both the
    # enemy sequence and the later gap.  This is geometry-based, so it also
    # works when the camera/world x differs between emulator runs.
    upper_enemies = [
        e for e in (state.get("enemies") or ())
        if e.get("tiles", 99) > 0 and e.get("up", -99) >= 3
    ]
    upper_distance = min((e["tiles"] for e in upper_enemies), default=99)
    if (
        state.get("headroom") == 2
        and upper_distance <= 9
        and name in ("stand", "walk right", "run right", "jump right", "run and jump right",
                     "hop right", "jump in place")
        and (not wall or wall.get("tiles", 99) > 3)
        and (not gap or gap.get("width") == 3 and gap.get("tiles", 99) >= 4)
    ):
        # The raised enemies descend into Mario's lane while a decision is in
        # flight.  At 5-9 tiles, jumping is early and lands in their path, so
        # keep running until the next observation; at <=4 tiles, launch the
        # full-speed arc.  This is the timing window that was missed at
        # x≈1194 when the model waited for a ground-level warning.
        if upper_distance > 4:
            name = "run right"
        elif state.get("speed") == "running at full speed":
            name = "run and jump right"
        else:
            name = "run right"

    # A second underground timing trap has a ground-level enemy cluster in
    # front of a raised piranha plant.  An in-place jump loses all horizontal
    # momentum and arrives at the later low wall with speed zero.  Launch a
    # forward arc while the cluster is still 6-10 tiles away; this keeps the
    # plant/wall section in the same jump and avoids the x≈1635 dead stop.
    raised_plants = [
        e for e in (state.get("enemies") or ())
        if e.get("kind") == "piranha plant" and e.get("up", -99) >= 2
        and e.get("tiles", 99) <= 14
    ]
    ground_cluster = [
        e for e in (state.get("enemies") or ())
        if -1 <= e.get("up", -99) <= 1 and e.get("tiles", 99) > 0
    ]
    nearest_ground = min((e["tiles"] for e in ground_cluster), default=99)
    if (raised_plants and 6 <= nearest_ground <= 10
            and not wall and not gap
            and name in ("stand", "walk right", "run right", "jump right", "jump in place", "hop right")):
        name = "run and jump right" if state.get("speed") == "running at full speed" else "run right"

    # ---- enemy 守卫：前方有敌人时，别用 hop / 走 / 站去撞 ----
    # 死亡帧(x=2749)证明：一路 hop 把速度耗在 walk 上，进头顶墙区后落点被压到 2 格，
    # 两个紧贴 goomba(1、2 格前) 就跳不过去。正确做法是"远处 run right 蓄全速、
    # 近处 run and jump right 全速越过"。这层必须在执行前拦住，否则模型会一直 hop。
    if ahead:
        d = ahead[0]
        landing_bad = bool(state.get("enemies_near_landing_spot"))
        headroom = state.get("headroom")
        if (d <= 1
                and name in ("jump right", "run and jump right", "hop right")
                and (landing_bad or headroom == 0)):
            # At one tile with a capped landing, a slow/standing Mario needs
            # the vertical escape.  At full speed, momentum is the only way
            # over the enemy: the x≈895 replay dies when this forward arc is
            # rewritten to a vertical jump.  Mentor follow-ups are handled by
            # the same rule through preserve_mentor_timing.
            if state.get("speed") == "running at full speed":
                pass
            elif preserve_mentor_timing and name in ("jump right", "run and jump right"):
                # A coach forward phase is only safe once the inserted
                # acceleration has reached full speed; keep building speed
                # instead of letting a low-speed plan collide point-blank.
                name = "run right"
            else:
                name = "jump in place"
        elif d <= 1 and name in ("stand", "walk right", "run right"):
            # Waiting or walking at point-blank range lets the enemy touch
            # Mario before the next six-frame decision boundary.  Use the
            # same vertical escape as the forward-jump case.
            name = "jump in place"
        elif (not preserve_mentor_timing and headroom == 0 and 2 <= d <= 3
              and landing_bad
              and name in ("jump right", "run and jump right", "hop right")):
            # A full jump is capped under this ceiling, but a short hop keeps
            # enough horizontal momentum to stomp the first enemy.  Rewriting
            # an explicit hop to a vertical jump stalls at x≈887-989; rewriting
            # it to a full jump hits the ceiling.  Only repair full-jump choices
            # here and preserve the model/policy's timing-sensitive hop.
            if name != "hop right":
                name = "jump in place"
        elif (not preserve_mentor_timing and 4 <= d <= 8 and landing_bad
              and name in ("jump right", "run and jump right")):
            # A landing warning is not sufficient reason to discard momentum.
            # The old rule changed every such jump to an in-place jump; in the
            # x≈670 scene that made Mario meet the approaching enemy with zero
            # separation.  Keep a forward arc and let the point-blank branch
            # above handle the genuinely unsafe one-tile case.
            name = "run and jump right" if state.get("speed") == "running at full speed" else "jump right"
        elif (not preserve_mentor_timing and headroom == 0 and 2 <= d <= 3
              and name in ("hop right", "stand", "walk right", "run right")):
            # At two or three tiles the low ceiling does not make a forward
            # jump impossible; it only shortens the landing.  Keep an explicit
            # ``jump in place`` untouched: at the x≈887-989 timing trap that
            # vertical escape is the measured safe choice, and rewriting it to
            # a forward arc makes Mario hit the low ceiling/enemy.  Ground
            # actions and hops still need a forward jump while there is
            # horizontal separation.
            name = "run and jump right" if state.get("speed") == "running at full speed" else "jump right"
        elif d <= 3 and name in ("hop right", "walk right", "run right", "stand"):
            name = "run and jump right"
        elif 4 <= d <= 6 and name == "hop right":
            name = "run right"
    return name


def diagnose_death(state: dict, last_choice: str, x_death: int, level: str | None = None) -> dict | None:
    """把死亡现场总结成一条结构化教训。

    返回 dict：{text, category, x, level, signature}，或 None（死因不明确）。

    - text：可执行的教训文本（去重键，不含位置 —— 同类死因同档生成同一条文本）
    - category：死因类别（enemy/gap/wall），用于分类展示和注入控量
    - x / level：失败位置（最远 x 与关卡），用于「记录在哪死的」与卡点判断
    - signature：场景签名（分桶），供经验库/教练机制复用
    """
    enemy = state.get("enemy_ahead")
    gap = state.get("gap_ahead")
    wall = state.get("wall_ahead")
    far = (state.get("jump_reach_now") or {}).get("tiles_far", 5)
    text = None
    category = None
    if enemy and enemy.get("tiles", 99) <= 8:
        category = "enemy"
        d = enemy["tiles"]
        if d <= 2:
            text = ("An enemy is 1-2 tiles ahead: 'hop right' only hops 2 tiles and lands on it. "
                    "Use 'run and jump right' to leap over.")
        elif d <= 4:
            text = ("An enemy is 3-4 tiles ahead and closing fast: build speed first. "
                    "'run right' to full speed, then 'run and jump right' to clear it.")
        elif d <= 8:
            repeated = f" Do not repeat '{last_choice}'." if last_choice else ""
            text = (f"An enemy is {d} tiles ahead. Move closer without a blind full jump, then "
                    f"use 'run and jump right' when it is 2-3 tiles away; if the warning says "
                    f"the forward landing is occupied, use 'jump in place' instead.{repeated}")
    elif gap and gap.get("tiles", 99) <= 5:
        category = "gap"
        w = gap["width"]
        text = (f"A gap {w} tiles wide is ahead: 'hop right' only reaches 2 tiles and falls in. "
                f"Use 'jump right' (or 'run and jump right' if wider than {far - 1}).")
    elif wall and wall.get("tiles", 99) <= 3:
        category = "wall"
        h = wall["height"]
        if h >= 4:
            text = ("A wall 4+ tiles tall is ahead: a walk-speed jump can't clear it. "
                    "Back off, 'run right' to full speed, then 'run and jump right'.")
        else:
            text = ("A wall is ahead: jump when close enough, using 'run and jump right' to be safe.")
    if text is None:
        return None
    return {"text": text, "category": category, "x": x_death, "level": level,
            "signature": scene_signature(state)}


def _rank_lessons(state: dict, lessons):
    """按「与当前场景的匹配度」给教训排序，命中的排最前并标 [APPLY NOW]。

    教训是去重后的通用文本，但当前 state 能告诉我们哪条「现在用得上」——
    前方就是 4 格高墙时，把「撞墙」教训提到最前并标注，让模型在正确时机看到
    正确经验，而不是让几条教训平铺在 prompt 里被习惯性忽略。
    """
    wall = state.get("wall_ahead")
    gap = state.get("gap_ahead")
    enemy = state.get("enemy_ahead")

    def _text(item) -> str:
        # 兼容两种形态：纯文本（注入路径）或结构化 dict（累积路径）
        return item["text"] if isinstance(item, dict) else item

    def score(item) -> int:
        t = _text(item)
        s = 0
        if wall and "wall" in t:
            s += 3 if wall.get("height", 0) >= 4 else 2
        if gap and "gap" in t:
            s += 3
        if enemy and "enemy" in t and enemy.get("tiles", 99) <= 8:
            s += 3
        return s

    out = []
    for item in sorted(lessons, key=score, reverse=True):
        t = _text(item)
        # 结构化教训带上失败位置，让模型知道「上次在这附近栽过」
        if isinstance(item, dict):
            x = item.get("x", item.get("last_x"))
            if x is None and item.get("positions"):
                x = item["positions"][-1]
            if x is not None:
                t = f"[x≈{x}] {t}"
        if score(item) >= 3:
            t = f"[APPLY NOW] {t}"
        out.append(t)
    return out


def scene_signature(state: dict) -> tuple:
    """把 state 压缩成可比较的场景签名（分桶）。

    经验库用它做「相似场景召回」：同样的签名 = 上次死过的同类场景。
    桶：敌人(0无/1贴脸/2近/3远) · 坑(0无/1窄/2宽) · 墙(0无/1矮/2高) · 速度(0停/1慢/2满)。
    """
    enemy = state.get("enemy_ahead")
    gap = state.get("gap_ahead")
    wall = state.get("wall_ahead")
    speed = state.get("speed", "")
    eb = 0 if not enemy else (1 if enemy.get("tiles", 99) <= 2 else (2 if enemy.get("tiles", 99) <= 4 else 3))
    gb = 0 if not gap else (1 if gap.get("width", 0) <= 3 else 2)
    wb = 0 if not wall else (1 if wall.get("height", 0) < 4 else 2)
    sb = 2 if speed == "running at full speed" else (1 if "right" in speed else 0)
    return (eb, gb, wb, sb)


def _experience_hint(state: dict, experience) -> str:
    """召回与当前场景签名匹配的经验（正确动作），生成一句提示。"""
    if not experience:
        return ""
    act = experience.get(scene_signature(state))
    if not act:
        return ""
    return (f"PAST EXPERIENCE: last time you were in this exact situation you died; "
            f"the correct action was '{act}'.\n\n")


def ask_local(client: httpx.Client, state: dict, history=(), lessons=(), experience=None, mentor="") -> tuple[str, dict | None, int, float, str]:
    """Same state, same question as ask_jev, but answered by the local model.

    Returns (action, probabilities_or_None, tokens, latency, source).
    source is "model" when the local model answered, "rules_fallback" when its reply
    could not be parsed as an offered option and the deterministic rules policy stood in.
    """
    keys = list(ACTION_HELP)
    labels = LETTERS[: len(keys)]
    options = "\n".join(f"  {label}) {key} = {OPTION_HINTS.get(key, ACTION_HELP[key])}"
                         for label, key in zip(labels, keys))
    lesson_block = ""
    if lessons:
        lesson_block = ("PAST FAILURES (the [APPLY NOW] one matches your current state, follow it; do NOT repeat these mistakes):\n"
                        + "\n".join(f"- {s}" for s in _rank_lessons(state, lessons)) + "\n\n")
    exp_block = _experience_hint(state, experience)
    mentor_block = ""
    if mentor:
        plan = parse_mentor_plan(mentor)
        if plan:
            coach_payload = {
                "first_action": plan.get("action"),
                "follow_up": plan.get("follow_up"),
                "wait_decisions": plan.get("wait_decisions", 0),
                "plan_id": mentor.get("plan_id") if isinstance(mentor, dict) else None,
                "origin": mentor.get("origin", "llm") if isinstance(mentor, dict) else "llm",
            }
            mentor_block = ("COACH PLAN (advisory from the failure-analysis LLM; execute the first_action "
                            "unless a measured physical WARNING makes it impossible):\n"
                            + json.dumps(coach_payload, ensure_ascii=False) + "\n"
                            + f"COACH REFLECTION:\n{plan.get('reflection') or mentor_text(mentor)}\n\n")
        else:
            mentor_block = f"COACH GUIDANCE (advisory; follow this when physically safe):\n{mentor_text(mentor)}\n\n"
    user = (
        f"RECENT DECISIONS (oldest first; x is where Mario stood at that decision):\n"
        f"{_history_block(history)}\n\n"
        f"{mentor_block}"
        f"{exp_block}"
        f"{lesson_block}"
        f"STATE:\n{_state_block(state)}\n\n"
        f"OPTIONS (answer with the LETTER):\n{options}\n\n"
        "Priority is WARNINGS/physical constraints, then the active COACH PLAN first_action, then normal rules. "
        "Never pick an option warnings rule out; a jump is required whenever an "
        "enemy is 1 to 3 tiles ahead; never repeat an action marked STUCK.\n"
        "Which letter is the best choice for action? Answer with one letter."
    )
    policy_key = os.environ.get("LOCAL_POLICY_API_KEY", LOCAL_POLICY_API_KEY)
    policy_mode = os.environ.get("LOCAL_POLICY_MODE", LOCAL_POLICY_MODE)
    policy_root = os.environ.get("LOCAL_POLICY_BASE_URL", LOCAL_POLICY_BASE_URL).rstrip("/")
    temperature = float(os.environ.get("LOCAL_POLICY_TEMPERATURE", str(LOCAL_TEMPERATURE)))
    headers = {"Authorization": f"Bearer {policy_key}"}
    t0 = time.perf_counter()

    if policy_mode == "score":
        # 候选打分：能拿到真实分布，且返回值必然落在候选集合内。
        body = {"items": [{"messages": [{"role": "system", "content": CHAT_SYSTEM},
                                        {"role": "user", "content": user}],
                           "candidates": [f" {label}" for label in labels]}]}
        r = client.post(policy_root + "/score", json=body, headers=headers, timeout=600)
        lat = time.perf_counter() - t0
        r.raise_for_status()
        item = r.json()["items"][0]
        top = max(item["scores"])
        weights = [math.exp((value - top) / temperature) for value in item["scores"]]
        total = sum(weights)
        probabilities = {key: weight / total for key, weight in zip(keys, weights)}
        best = item["best"].strip()
        if best not in labels:
            raise ValueError(f"本地模型返回 {best!r}，不在候选字母 {labels} 里")
        return keys[labels.index(best)], probabilities, 0, lat, "model"

    payload = {"model": "local",
               "messages": [{"role": "system", "content": CHAT_SYSTEM},
                            {"role": "user", "content": user}],
               "max_tokens": 8, "temperature": 0}
    r = client.post(policy_root + "/chat/completions", json=payload, headers=headers, timeout=600)
    lat = time.perf_counter() - t0
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    picked = _letter_to_action(text, labels, keys)
    if picked is None:
        return policy(state), None, 0, lat, "rules_fallback"
    return picked, None, 0, lat, "model"


# ----------------------------------------------------------------------------- run loop
class Replay:
    """replay:<log.jsonl>[@n]:<action,...> — replay the first n logged choices, then cycle through the tail."""

    def __init__(self, spec: str):
        _, path, tail = spec.split(":", 2)
        n = None
        if "@" in path:
            path, n = path.rsplit("@", 1)
        self.script = [json.loads(l)["choice"] for l in Path(path).read_text().splitlines()][: int(n) if n else None]
        self.tail = tail.split(",")
        self.i = 0

    def next(self) -> str:
        if self.script:
            return self.script.pop(0)
        name = self.tail[min(self.i, len(self.tail) - 1)]
        self.i += 1
        return name


def run(bot: str, level: str = "1-1", dump: bool = False) -> dict:
    warnings.simplefilter("ignore")  # gym's env checker re-enables a numpy deprecation warning
    env = JoypadSpace(make_mario_env(gym_super_mario_bros, level), SIMPLE_MOVEMENT)
    ram = nes(env).ram
    obs, _ = env.reset()
    # nes_py reuses one screen buffer, so every stored frame must be a copy.
    frames, tokens, lats, log = [obs.copy()], 0, [], []
    action, prev, hold_cap = 0, 0, FULL_JUMP_FRAMES
    frame, best, last_best, last_gain = 0, 0, 0, 0
    info = {"x_pos": 0, "flag_get": False}
    term = trunc = False
    deciding = bot in ("jev", "local", "rules") or bot.startswith("replay:") or dump
    replay = Replay(bot) if bot.startswith("replay:") else None
    client = httpx.Client(timeout=30)

    def step(a: int) -> bool:
        """Advance one frame; record every other frame; return True when the episode is over."""
        nonlocal obs, term, trunc, info, frame, best
        obs, _, term, trunc, info = env.step(a)
        frame += 1
        best = max(best, int(info["x_pos"]))
        if frame % 2 == 0:
            frames.append(obs.copy())
        return bool(term or trunc)

    while frame < MAX_FRAMES:
        if frame % HOLD == 0 and deciding:
            # Decide only on the ground: keep the current direction (without A) while airborne.
            fall = 0
            while airborne(ram) and fall < 120 and not step(RELEASE.get(action, action)):
                fall += 1
            if term or trunc:
                break
            frame += (-frame) % HOLD
        if frame % HOLD == 0:
            g, _, _ = grid(ram)
            feats = features(g, speed(ram), airborne=airborne(ram), visible=(256 - int(ram[0x03AD])) // 16)
            state = {"summary": feats.pop("summary"), **feats, "grid": g, "action_before": action}
            if dump:
                print(f"frame {frame} x={info['x_pos']}\n{state['summary']}\n{g}\n")
                name = "run and jump right" if (frame // 12) % 2 == 0 else "run right"
            elif bot == "jev":
                name, probs, tok, lat = ask_jev(client, state, [(r["choice"], r["x"]) for r in log[-6:]])
                guarded = hazard_guard(name, state)
                if guarded != name:
                    print("  [guard] " + name + " -> " + guarded + " (x=" + str(int(info["x_pos"])) + ")")
                    name = guarded
                forced = unstick(name, log, int(info["x_pos"]))
                if forced != name:
                    print("  [unstick] " + name + " -> " + forced + " (x=" + str(int(info["x_pos"])) + ")")
                    name = forced
                tokens += tok
                lats.append(lat)
                log.append({"frame": frame, "x": int(info["x_pos"]), "choice": name,
                            "p": round(probs.get(name, 0), 2) if probs else None,
                            "probs": {k: round(p, 2) for k, p in probs.items()} if probs else None,
                            "summary": state["summary"], "grid": g})
            elif bot == "local":
                name, probs, tok, lat, source = ask_local(client, state, [(r["choice"], r["x"]) for r in log[-6:]])
                guarded = hazard_guard(name, state)
                if guarded != name:
                    print("  [guard] " + name + " -> " + guarded + " (x=" + str(int(info["x_pos"])) + ")")
                    name, source = guarded, "guard"
                forced = unstick(name, log, int(info["x_pos"]))
                if forced != name:
                    print("  [unstick] " + name + " -> " + forced + " (x=" + str(int(info["x_pos"])) + ")")
                    name, source = forced, "unstick"
                lats.append(lat)
                log.append({"frame": frame, "x": int(info["x_pos"]), "choice": name, "p": round(probs[name], 2) if probs else None,
                            "probs": {k: round(p, 2) for k, p in probs.items()} if probs else None,
                            "local_source": source, "summary": state["summary"], "grid": g})
            elif replay:
                name = replay.next()
                log.append({"frame": frame, "x": int(info["x_pos"]), "choice": name, "summary": state["summary"], "grid": g})
            elif bot == "rules":
                name = policy({**feats, "summary": state["summary"]})
                log.append({"frame": frame, "x": int(info["x_pos"]), "choice": name, "summary": state["summary"], "grid": g})
            elif bot == "alternate":
                name = "run and jump right" if (frame // 12) % 2 == 0 else "run right"
            else:
                name = bot
            action = ACTIONS[name]
            hold_cap = HOP_FRAMES if name == "hop right" else FULL_JUMP_FRAMES

            if action == BACK_OFF:
                # Atomic retreat: actually brake and create a runway.  When no
                # wall/gap is currently visible, treating the missing obstacle
                # as distance=99 ends the retreat on its first frame and
                # destroys the coach's run-up plan.
                reversed_frames = 0
                for _ in range(60):
                    if step(6):
                        break
                    reversed_frames += 1
                    fe = features(grid(ram)[0], speed(ram))
                    obstacle = fe.get("wall_ahead") or fe.get("gap_ahead")
                    if obstacle and obstacle.get("tiles", 0) >= 6 and reversed_frames >= 18:
                        break
                    if (reversed_frames >= 12 and speed(ram) <= 0
                            and fe.get("clear_behind", 0) >= 3):
                        break
                    if reversed_frames >= 24 and speed(ram) <= 0:
                        break
                if term or trunc:
                    break
                frame += (-frame) % HOLD
                action = 0
            # The NES only jumps on an A press, not a hold: release A for one frame between two jumps.
            if action in JUMPS and prev in JUMPS and step(RELEASE[action]):
                break
            prev = action
            if action in JUMPS and bot != "alternate":
                # One decision is one jump: hold A until Mario lands, or hold_cap frames for a hop.
                for i in range(FULL_JUMP_FRAMES):
                    if step(action) or i >= hold_cap or (i > 4 and not airborne(ram)):
                        break
                if term or trunc:
                    break
                frame += (-frame) % HOLD
                action = RELEASE[action]
        if step(action):
            break
        if best > last_best:
            last_best, last_gain = best, frame
        if info["flag_get"] or frame - last_gain > STALL_FRAMES:
            break
    env.close()

    RUNS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{level}-{'replay' if replay else bot.replace(' ', '-')}"
    result = {
        "level": level, "bot": bot, "stamp": stamp, "best_x": best, "flag": bool(info["flag_get"]),
        "frames": frame, "api_calls": len(lats), "input_tokens": tokens,
        "cost_usd": round(tokens * USD_PER_TOKEN, 5),
        "latency_p50": round(sorted(lats)[len(lats) // 2], 3) if lats else None,
        "gif": None if dump else f"{tag}-{stamp}.gif",
    }
    if not dump:
        imageio.mimsave(RUNS / result["gif"], frames, duration=1 / 30, loop=0)
        with (RUNS / "results.jsonl").open("a") as fh:
            fh.write(json.dumps(result) + "\n")
        if log:
            (RUNS / f"{tag}-{stamp}.log.jsonl").write_text("\n".join(json.dumps(l) for l in log) + "\n")
    return result


def inspect(path: str, n: int) -> None:
    lines = [json.loads(l) for l in Path(path).read_text().splitlines()]
    print(f"{len(lines)} decisions:", [(l["x"], l["choice"]) for l in lines])
    for l in lines[-n:]:
        top = sorted(l.get("probs", {}).items(), key=lambda kv: -kv[1])[:4]
        print(f"\nframe {l['frame']} x={l['x']} -> {l['choice']} {l.get('p', '')}  {top}")
        print(l["summary"])
        print(l["grid"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", default="jev", help="jev | local | rules | alternate | <action name> | replay:<log.jsonl>[@n]:<action,...>")
    ap.add_argument("--level", default="1-1", help="world-stage, e.g. 2-1")
    ap.add_argument("--dump", action="store_true", help="print the state Jev would see, no API calls")
    ap.add_argument("--inspect", metavar="LOG", help="print the last decisions of a run log and exit")
    ap.add_argument("-n", type=int, default=3, help="decisions to show with --inspect")
    a = ap.parse_args()
    if a.inspect:
        inspect(a.inspect, a.n)
    else:
        load_env()
        print(json.dumps(run(a.bot, a.level, a.dump)))
