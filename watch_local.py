"""实时观看本地模型玩 Super Mario Bros：把模拟器画面和每次决策推到网页上。

    .venv-mario/Scripts/python.exe watch_local.py --level 1-1 --port 8123

浏览器打开 http://127.0.0.1:8123/，会看到：
  - 左：模拟器实时画面（MJPEG 流，原始 240x256，像素放大）
  - 右：当前关卡进度、这一步的决策、模型给出的概率分布（score 模式）

两个为了"可观看"的关键设计：

1. **默认同步决策**。模型回复期间只暂停对应模拟器，网页推流和状态卡仍继续响应并显示“思考中”；
   这样每个动作都对应当前状态，优先保证通关。显式传 `--async` 才会让模拟器不停步，结果可能来自旧状态，
   只适合延迟实验，不作为稳定通关路径。
2. **MJPEG 推流**。原版靠 JS 每 100ms 轮询换 `img.src`，显示上限被锁在 10fps；
   改成 `multipart/x-mixed-replace` 长连接，浏览器收到即渲染，没有轮询开销、没有帧率上限。

一局结束后停 3 秒自动开下一局，方便反复看。
"""
import argparse
import collections
import contextlib
import io
import json
import os
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import imageio.v2 as imageio  # noqa: F401  (play_local 需要)
from PIL import Image

warnings.filterwarnings("ignore")

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

import play_local as P
import mentor_search as MS

BOUNDARY = b"mario-frame"
CHANNELS = ("local", "jev")  # 双画面：本地模型 vs 官方模型
LATEST = {c: {"seq": 0, "jpeg": None, "meta": {"status": "starting"}} for c in CHANNELS}
LESSONS = {c: [] for c in CHANNELS}  # 每个模型全量累积的失败教训（去重后），供页面展开查看
LEVELS = tuple(f"{w}-{s}" for w in range(1, 9) for s in range(1, 5))  # 1-1 ... 8-4
STATS = {c: {"clears": 0, "level": LEVELS[0], "best_x": 0} for c in CHANNELS}  # 通关次数 + 当前关卡 + 历史最远
EXPERIENCE = {c: {} for c in CHANNELS}  # 仅保留外部验证过的经验；死亡现场不会直接写成“正确动作”
MENTOR_GUIDANCE = {c: {} for c in CHANNELS}  # (level, signature) -> {text, action, x, ...}
MENTOR_ATTEMPTS = {c: {} for c in CHANNELS}  # (level, signature) -> 最近一次请求失败计数
FAIL_COUNT = {c: {} for c in CHANNELS}  # (level, signature) -> 失败次数
MENTOR_PLAN_SEQ = {c: 0 for c in CHANNELS}  # stable ids for plan application/failure telemetry
# A verified route is enabled only for a later episode after repeated failures.
# Keep this separate from scene-level FAIL_COUNT: a route decision is an
# episode-level recovery policy, and must not be accidentally enabled halfway
# through a running episode.
VERIFIED_ROUTE_FAILURES = {c: {} for c in CHANNELS}  # channel -> level -> failed episodes
LOCK = threading.Lock()
STOP = threading.Event()
FRAME_TIMES = collections.deque(maxlen=400)  # 出帧时间戳，用来算真实帧率

# These routes were produced by branch.py's emulator-backed search and
# replayed twice to flag=true (1-2 x=3161; 1-3 x=2425). The normal model/mentor
# gets three attempts first. Route actions and model choices have separate
# provenance; --replay-only explicitly disables model requests altogether.
VERIFIED_ROUTES = {
    "1-2": (
        ("short run then jump", None),
        ("short run then jump", None),
        ("jump in place", None),
        ("run and jump right", None),
        ("walk left", "run then jump"),
        ("hop back onto the ledge behind", "run right"),
        ("short run then jump", None),
        ("run right", "walk left"),
        ("jump in place", "run right"),
        ("run then jump", None),
        ("jump right", None),
        ("run then jump", None),
        ("short run then jump", None),
        ("bounce on the spring behind", None),
        ("jump right", None),
        ("jump right", None),
        ("short run then jump", None),
        ("run then jump", None),
        ("hop right", None),
        ("run then jump", None),
        ("bounce on the spring behind", None),
        ("bounce on the spring behind", None),
        ("run and jump right", None),
    ),
    "1-3": (
        ("run and jump right", None),
        ("short run then jump", None),
        ("run and jump right", None),
        ("walk left", "run then jump"),
        ("jump in place", "jump right"),
        ("jump in place", "short run then jump"),
        ("hop right", None),
        ("run then jump", None),
        ("short run then jump", None),
        ("jump in place", None),
        ("hop back onto the ledge behind", "jump right"),
        ("jump right", None),
        ("jump right", None),
        ("jump in place", None),
        ("jump right", None),
        ("jump right", None),
        ("jump in place", None),
        ("hop back onto the ledge behind", "jump right"),
        ("hop back onto the ledge behind", "jump right"),
        ("run and jump right", None),
        ("short run then jump", None),
        ("jump in place", None),
        ("stand", "jump right"),
        ("run then jump", None),
        ("short run then jump", None),
        ("run then jump", None),
    ),
}

BRANCH_COMPOSITE = {
    "run then jump": (("run right", 24), ("run and jump right", 60)),
    "short run then jump": (("run right", 12), ("run and jump right", 60)),
}
BRANCH_RAW = {
    "bounce on the spring behind": ((6, 2), (5, 40), (0, 6), (2, 90)),
    "hop back onto the ledge behind": ((5, 8), (6, 30), (0, 10)),
}


# --------------------------------------------------------------------------- 画面分发
def publish(obs, meta, channel="local"):
    buf = io.BytesIO()
    Image.fromarray(obs).save(buf, format="JPEG", quality=80)
    now = time.perf_counter()
    with LOCK:
        FRAME_TIMES.append(now)
        recent = [t for t in FRAME_TIMES if now - t <= 2.0]
        meta = {**meta, "fps_actual": round(len(recent) / 2.0, 1) if len(recent) > 1 else 0.0}
        slot = LATEST[channel]
        # ``sync`` and ``channel`` are process-level facts initialized by
        # main(); keep them when a frame replaces the per-episode metadata.
        for key in ("sync", "channel", "enabled", "model_only", "model_name"):
            if key not in meta and key in slot["meta"]:
                meta[key] = slot["meta"][key]
        slot["seq"] += 1
        slot["jpeg"] = buf.getvalue()
        slot["meta"] = meta


def snapshot(channel="local"):
    with LOCK:
        slot = LATEST[channel]
        return slot["seq"], slot["jpeg"], dict(slot["meta"])


def _lesson_category(item) -> str:
    """按死因给教训分类，用于去重留重点（兼容文本与结构化 dict）。"""
    text = item["text"] if isinstance(item, dict) else item
    if "enemy" in text:
        return "撞敌人"
    if "gap" in text:
        return "掉坑"
    if "wall" in text:
        return "撞墙"
    return "其他"


def inject_lessons(lessons, per_category=3):
    """从全量教训里挑出注入 prompt 的重点。

    全保留（LESSONS 里不去丢），但注入 prompt 时按死因分类、每类只留最近
    per_category 条 —— 否则跑久了一类死因会堆几十条几乎相同的教训，
    撑爆 prompt、反而让模型退化。展示层不受这个限制，能看到全部。
    """
    buckets = {}
    for s in lessons:
        buckets.setdefault(_lesson_category(s), []).append(s)
    picked = []
    for cat in ("撞敌人", "掉坑", "撞墙", "其他"):
        picked.extend(buckets.get(cat, [])[-per_category:])
    out = []
    for s in picked:
        if isinstance(s, dict):
            t = s["text"]
            x = s.get("x", s.get("last_x"))
            if x is None and s.get("positions"):
                x = s["positions"][-1]
            if x is not None:
                t = f"[x≈{x}] {t}"
            out.append(t)
        else:
            out.append(s)
    return out


def _scene_key(level, signature):
    """Use a stable, hashable key without changing the public signature format."""
    return str(level), tuple(signature or ())


def _matching_mentor(guidance, level, state, x):
    """Return executable guidance for the current level and nearby hazard.

    The signature deliberately buckets enemy distance and speed.  Requiring an
    exact four-field match made a useful plan disappear while Mario was
    waiting for the same enemy to move from 2 tiles away to 1 tile away.  An
    exact match still wins; when only the distance/speed bucket changed, a
    nearby plan with the same wall/gap/enemy *presence* is safe to reuse.
    """
    if not guidance:
        return None
    # Accept a legacy single string while the process is being upgraded.
    if isinstance(guidance, str):
        plan = P.parse_mentor_plan(guidance)
        return {"text": guidance, **plan} if plan else None
    current_sig = tuple(P.scene_signature(state))
    current_x = int(x)

    def headroom_bucket(value):
        if value is None:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        return 0 if value <= 0 else (1 if value <= 2 else 2)

    current_headroom = headroom_bucket(state.get("headroom"))
    current_landing_bad = bool(state.get("enemies_near_landing_spot"))

    def compatible_shape(candidate_sig, distance):
        # A plan may follow an enemy as it moves one bucket closer, but must
        # stay in the same local scene.  The old bool-only match reused a plan
        # from (enemy=near, full-speed) at (enemy=point-blank, slow), which
        # repeatedly turned a fresh Jev decision into the same failed jump.
        candidate_sig = tuple(candidate_sig or ())
        if len(candidate_sig) < 4 or len(current_sig) < 4:
            return False
        if candidate_sig[1:3] != current_sig[1:3]:  # gap/wall geometry changed
            return False
        if abs(int(candidate_sig[0]) - int(current_sig[0])) > 1:
            return False
        if int(candidate_sig[3]) != int(current_sig[3]):  # speed phase changed
            return False
        return distance <= 64

    candidates = []
    values = guidance.values() if isinstance(guidance, dict) else guidance
    for item in values:
        if not isinstance(item, dict) or item.get("level") != level:
            continue
        item_sig = tuple(item.get("signature") or ())
        exact = item_sig == current_sig
        gx = item.get("x")
        distance = abs(current_x - int(gx)) if gx is not None else 10_000
        if not exact and not compatible_shape(item_sig, distance):
            continue
        # A coarse signature can occur in several places; keep the instruction near
        # the recorded death so a fix for x≈676 is not applied at the start of 1-2.
        if gx is not None and distance > 128:
            continue
        # New plans carry two extra physical qualifiers.  Legacy entries lack
        # them and remain usable, but a qualified plan must not jump from a
        # clear/high-ceiling state into a capped landing state.
        item_headroom = item.get("headroom_bucket")
        if item_headroom is not None and current_headroom is not None \
                and int(item_headroom) != int(current_headroom):
            continue
        if item.get("landing_bad") is not None \
                and bool(item.get("landing_bad")) != current_landing_bad:
            continue
        if P.parse_mentor_plan(item) is None:
            continue
        # Exact signatures are preferred, then nearest trigger position.  A
        # newer plan breaks a tie so a retired/obsolete entry cannot win.
        candidates.append((0 if exact else 1, distance,
                           -int(item.get("trigger_count", 0)), item))
    if not candidates:
        return None
    candidates.sort(key=lambda row: row[:3])
    return candidates[0][3]


def _mentor_plan_for_failure(guidance, level, death_info, key):
    """Find the plan that actually ran, even if the enemy moved signature.

    Scene signatures intentionally include nearby enemy distances.  A plan
    triggered at distance 2 can therefore die with a distance-1 signature.
    Looking up only ``guidance[key]`` made such plans appear to have zero
    failures and allowed the stale advice to live forever.
    """
    direct = guidance.get(key) if isinstance(guidance, dict) else None
    death_x = int(death_info.get("x", 0))
    best = None
    for plan_key, entry in (guidance.items() if isinstance(guidance, dict) else ()):
        if not isinstance(entry, dict) or entry.get("level") != level:
            continue
        plan_id = entry.get("plan_id")
        if plan_id is None:
            continue
        distances = []
        for row in death_info.get("mentor_trace") or ():
            if row.get("plan_id") != plan_id:
                continue
            try:
                distances.append(abs(death_x - int(row.get("x", death_x))))
            except (TypeError, ValueError):
                continue
        if not distances:
            continue
        distance = min(distances)
        if distance <= 128 and (best is None or distance < best[0]):
            best = (distance, plan_key, entry)
    if best:
        return best[1], best[2]
    # Preserve the old direct-key behavior for a plan that has not emitted a
    # trace yet; the caller will correctly leave its failure count unchanged.
    return (key, direct) if direct is not None else (None, None)


def _safety_mentor_plan(death_info, failed_action=None):
    """Return one emergency action when the mentor is unavailable.

    Multi-step timing plans deliberately belong to the LLM coach.  Keeping
    this fallback single-step makes ``origin=safety_fallback`` unambiguous and
    prevents a hard-coded wait-then-jump plan from masquerading as reflection.
    """
    enemy = death_info.get("enemy_ahead") or {}
    if (death_info.get("landing_bad") and enemy.get("tiles", 99) <= 8):
        return {
            "text": "Safety fallback: jump in place now; no multi-step coach plan was available.",
            "action": "jump in place",
            "follow_up": None,
            "wait_decisions": 0,
            "origin": "safety_fallback",
        }
    return None


# --------------------------------------------------------------------------- 异步决策
class Pipeline:
    """后台线程发请求，主循环永不阻塞。

    同一时刻只允许一个请求在飞（否则会排队，越等越旧），
    所以决策的自然频率就是 1/延迟；决策点比这更密时就沿用上一个动作。
    """

    def __init__(self, client, lessons=(), model="local", experience=None, mentor=""):
        self.client = client
        self.lessons = list(lessons)
        self.model = model
        self.experience = experience or {}
        self.mentor = mentor
        self.lock = threading.Lock()
        self.pending = None       # 已算好、还没被主循环取走的决策
        self.inflight = False
        self.started_at = None
        self.last_error = None

    def request(self, state, history, frame, x, mentor=None):
        with self.lock:
            if self.inflight:
                return False
            self.inflight = True
            self.started_at = time.perf_counter()
        selected_mentor = self.mentor if mentor is None else mentor
        threading.Thread(target=self._work,
                         args=(state, history, frame, x, selected_mentor), daemon=True).start()
        return True

    def _work(self, state, history, frame, x, mentor):
        result = None
        try:
            if self.model == "jev":
                name, probs, _tokens, latency = P.ask_jev(self.client, state, history, self.lessons, self.experience, mentor)
                source = "jev"
            else:
                name, probs, _tokens, latency, source = P.ask_local(self.client, state, history, self.lessons, self.experience, mentor)
            # The worker may be looking at an old frame.  Re-run hazard_guard and
            # unstick against the current state in the main loop instead of baking
            # a stale guard result into the queued decision.
            elapsed = time.perf_counter() - self.started_at
            result = {"name": name, "probs": probs, "source": source,
                      "latency_ms": round(latency * 1000), "frame": frame, "x": x,
                      "round_trip_ms": round(elapsed * 1000),
                      "mentor": bool(P.parse_mentor_plan(mentor))}
        except Exception as exc:  # 模型服务出问题不能让直播黑掉
            result = {"error": f"{type(exc).__name__}: {exc}", "frame": frame, "x": x}
        with self.lock:
            self.pending = result
            self.inflight = False
            if "error" in result:
                self.last_error = result["error"]

    def take(self):
        with self.lock:
            result, self.pending = self.pending, None
            return result

    def status(self):
        with self.lock:
            if self.inflight and self.started_at:
                return True, round((time.perf_counter() - self.started_at) * 1000)
            return False, None


# --------------------------------------------------------------------------- 网页
PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>双模型直播 · Super Mario Bros</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0e1116; color:#e6edf3; font:14px/1.5 "Segoe UI",system-ui,sans-serif; }
  .wrap { display:flex; gap:16px; padding:16px; align-items:flex-start; flex-wrap:wrap; }
  .side { flex:1; min-width:340px; max-width:620px; }
  .label { font-size:15px; font-weight:600; margin-bottom:8px; }
  .label .tag { display:block; font-size:11px; color:#8b949e; font-weight:400; }
  .screen { background:#000; border:1px solid #2a323d; border-radius:10px; padding:8px; position:relative; }
  img { display:block; width:100%; height:auto; image-rendering:pixelated; border-radius:6px; }
  .chip { position:absolute; top:14px; left:14px; background:rgba(13,17,23,.85); border:1px solid #2a323d;
          border-radius:999px; padding:3px 11px; font-size:12px; color:#d29922; opacity:0; transition:opacity .15s; }
  .chip.on { opacity:1; }
  .bar { display:flex; gap:14px; margin-top:10px; background:#161b22; border:1px solid #2a323d; border-radius:8px; padding:10px 12px; }
  .metric { flex:1; }
  .metric .k { color:#8b949e; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  .metric .v { font-size:20px; font-weight:600; font-variant-numeric:tabular-nums; margin-top:2px; }
  .status { font-size:12px; color:#d29922; margin-top:8px; }
  .mentor-status { color:#58a6ff; min-height:18px; }
  .flag { color:#238636; font-weight:700; }
  .configs { display:flex; gap:16px; padding:0 16px 20px; flex-wrap:wrap; }
  .cfg { flex:1; min-width:340px; max-width:620px; background:#161b22; border:1px solid #2a323d; border-radius:10px; padding:14px 16px; }
  .cfg-title { font-size:14px; font-weight:600; margin-bottom:10px; }
  .cfg.local .cfg-title { color:#58a6ff; }
  .cfg.jev .cfg-title { color:#d29922; }
  .cfg table { width:100%; border-collapse:collapse; font-size:12px; }
  .cfg td { padding:6px 8px; border-top:1px solid #21262d; vertical-align:top; line-height:1.4; }
  .cfg tr:first-child td { border-top:none; }
  .cfg td.k { color:#8b949e; width:78px; white-space:nowrap; }
  .cfg td.have { color:#238636; }
  .lessons-btn { cursor:pointer; color:#58a6ff; font-size:12px; background:none; border:none; padding:0; margin-left:4px; }
  .lessons { margin-top:8px; }
  .lessons ul { margin:0; padding-left:18px; font-size:11px; color:#adbac7; max-height:220px; overflow:auto; }
  .lessons li { margin:4px 0; }
  .lessons .empty { color:#6e7681; font-size:11px; }
  .lessons .lmeta { color:#8b949e; font-size:10px; }
  .lessons .lmeta .cat-enemy { color:#f85149; }
  .lessons .lmeta .cat-gap { color:#58a6ff; }
  .lessons .lmeta .cat-wall { color:#d29922; }
  .lessons .ltext { color:#adbac7; margin-top:2px; }
</style></head><body>
<div class="wrap">
  <div class="side">
    <div class="label" id="label-local">本地模型 <span class="tag">等待模型连接</span></div>
    <div class="screen">
      <img alt="本地模型画面" src="/stream.mjpg">
      <div class="chip" id="chip-local"></div>
    </div>
    <div class="bar">
      <div class="metric"><div class="k">关卡</div><div class="v" id="level-local">–</div></div>
      <div class="metric"><div class="k">通关</div><div class="v" id="clears-local">–</div></div>
      <div class="metric"><div class="k">历史最远</div><div class="v" id="best-local">–</div></div>
      <div class="metric"><div class="k">当前 x</div><div class="v" id="x-local">–</div></div>
    </div>
    <div class="status" id="status-local"></div>
    <div class="status" id="decision-local"></div>
    <div class="status mentor-status" id="mentor-local"></div>
  </div>
  <div class="side">
    <div class="label" id="label-jev">官方模型 <span class="tag">TypeSafe Jev · api.typesafe.ai</span></div>
    <div class="screen">
      <img alt="官方模型画面" src="/stream_jev.mjpg">
      <div class="chip" id="chip-jev"></div>
    </div>
    <div class="bar">
      <div class="metric"><div class="k">关卡</div><div class="v" id="level-jev">–</div></div>
      <div class="metric"><div class="k">通关</div><div class="v" id="clears-jev">–</div></div>
      <div class="metric"><div class="k">历史最远</div><div class="v" id="best-jev">–</div></div>
      <div class="metric"><div class="k">当前 x</div><div class="v" id="x-jev">–</div></div>
    </div>
    <div class="status" id="status-jev"></div>
    <div class="status" id="decision-jev"></div>
    <div class="status mentor-status" id="mentor-jev"></div>
  </div>
</div>
<div class="configs">
  <div class="cfg local">
    <div class="cfg-title">本地模型 · 策略配置</div>
    <table>
      <tr><td class="k">决策引擎</td><td>本地模型，结构化状态 → 9 选 1 字母</td></tr>
      <tr><td class="k">提示词</td><td>CHAT_SYSTEM 系统提示 + 状态摘要 + 动作选项</td></tr>
      <tr><td class="k">安全守卫</td><td class="have">hazard_guard：坑 / 墙 / 敌人贴脸时强制改动作</td></tr>
      <tr><td class="k">脱困机制</td><td class="have">unstick：卡住时自动升级跳跃</td></tr>
      <tr><td class="k">记忆</td><td class="have">最近 6 步决策 + STUCK 标记</td></tr>
      <tr><td class="k">失败教训</td><td class="have"><span id="lesson-count-local">0 条</span><button class="lessons-btn" onclick="toggleLessons('local')">展开▾</button></td></tr>
    </table>
    <div class="lessons" id="lessons-local" style="display:none"></div>
  </div>
  <div class="cfg jev">
    <div class="cfg-title">官方模型 · 策略配置</div>
    <table>
      <tr><td class="k">决策引擎</td><td>System One choice，state → 概率分布</td></tr>
      <tr><td class="k">提示词</td><td>instructions = RULES + GRID_LEGEND，criteria = ACTION_HELP</td></tr>
      <tr><td class="k">安全守卫</td><td class="have">hazard_guard：坑 / 墙 / 敌人贴脸时强制改动作</td></tr>
      <tr><td class="k">脱困机制</td><td class="have">unstick：卡住时自动升级跳跃</td></tr>
      <tr><td class="k">记忆</td><td class="have">最近 6 步决策 + STUCK 标记</td></tr>
      <tr><td class="k">失败教训</td><td class="have"><span id="lesson-count-jev">0 条</span><button class="lessons-btn" onclick="toggleLessons('jev')">展开▾</button></td></tr>
    </table>
    <div class="lessons" id="lessons-jev" style="display:none"></div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
function toggleLessons(c) {
  const box = $('lessons-' + c);
  box.style.display = box.style.display === 'none' ? 'block' : 'none';
}
async function tick() {
  let state;
  try { state = await (await fetch('/state', {cache:'no-store'})).json(); } catch (e) { return; }
  for (const [c, meta] of Object.entries(state)) {
    if (!meta) continue;
    $('label-' + c).closest('.side').hidden = meta.enabled === false;
    document.querySelector('.cfg.' + c).hidden = meta.enabled === false || meta.model_only;
    const label = $('label-' + c);
    label.textContent = (c === 'local' ? '本地模型 · ' : '官方模型 · ') + (meta.model_name || c);
    const mode = document.createElement('span');
    mode.className = 'tag';
    mode.textContent = meta.model_only
      ? '模型独立决策 · 无路线 / 守卫 / 教练 / 兜底'
      : '模型与执行策略组合';
    label.append(mode);
    if (meta.replay_only) {
      $('label-' + c).textContent = '离线验证路线回放 · ' + (c === 'local' ? 'A' : 'B') + '（未调用模型）';
      document.querySelector('.cfg.' + c).hidden = true;
    }
    const stt = meta.stats || {};
    $('best-' + c).textContent = Math.max(stt.best_x || 0, meta.best_x || 0);
    $('x-' + c).textContent = meta.x ?? '–';
    $('level-' + c).textContent = stt.level ?? '–';
    $('clears-' + c).textContent = stt.clears ?? '–';
    const st = $('status-' + c);
    const routeTag = meta.verified_route_used
      ? ' · 已验证路线纠偏' + (meta.route_step != null ? ' #' + meta.route_step : '') : '';
    if (meta.flag) st.innerHTML = '<span class="flag">通关！</span>' + routeTag;
    else st.textContent = (meta.status ?? '') + routeTag;
    if (meta.model_only) st.append(' · 第 ' + (meta.edition || 0) + ' 局 · ' + (meta.calls || 0) + ' 次请求 · ' + (meta.frame || 0) + ' 帧');
    const mentorBox = $('mentor-' + c);
    const mentors = meta.mentor || [];
    const decision = meta.decision || {};
    $('decision-' + c).textContent = '模型选择：' + (decision.model_choice || '–')
      + ' → 实际执行：' + (decision.choice || '–') + ' · 来源：' + (decision.source || '–')
      + (decision.latency_ms != null ? ' · ' + decision.latency_ms + ' ms' : '');
    if (decision.mentor_action) {
      mentorBox.textContent = '教练已应用：' + decision.mentor_action + '（' + (decision.source || '') + '）';
    } else if (mentors.length) {
      mentorBox.textContent = '教练待命：' + mentors.map(m => m.action).filter(Boolean).join('、');
    } else {
      mentorBox.textContent = '';
    }
    const chip = $('chip-' + c);
    if (meta.thinking) { chip.textContent = '思考中 ' + (meta.thinking_ms/1000).toFixed(1) + 's'; chip.classList.add('on'); }
    else chip.classList.remove('on');
    const ls = meta.lessons || [];
    $('lesson-count-' + c).textContent = ls.length + ' 条';
    $('lessons-' + c).innerHTML = ls.length
      ? '<ul>' + ls.map(s => {
          if (typeof s === 'object' && s !== null && s.text) {
            const catLabel = {enemy:'撞敌人', gap:'掉坑', wall:'撞墙'}[s.category] || s.category || '';
            const posList = (s.positions && s.positions.length) ? s.positions.join(', ') : (s.last_x != null ? String(s.last_x) : '');
            const cnt = s.count || 1;
            const meta = [];
            if (s.level) meta.push(s.level);
            if (catLabel) meta.push('<span class="cat-' + s.category + '">' + catLabel + '</span>');
            meta.push('×' + cnt + '次');
            if (posList) meta.push('x=' + posList);
            return '<li><span class="lmeta">' + meta.join(' · ') + '</span><div class="ltext">' + s.text + '</div></li>';
          }
          return '<li>' + s + '</li>';
        }).join('') + '</ul>'
      : '<div class="empty">还没有失败教训</div>';
  }
}
setInterval(tick, 250);
tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send(self, code, body, mime):
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            # Browser polling can be cancelled while a response is being sent.
            # It is not a server/game error and should not pollute the supervisor log.
            return

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/favicon.ico":
            return self.send(204, b"", "image/x-icon")

        if path in ("/stream.mjpg", "/stream_jev.mjpg"):
            channel = "jev" if path == "/stream_jev.mjpg" else "local"
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY.decode())
            self.send_header("Cache-Control", "no-store, no-transform")
            self.send_header("Connection", "close")
            self.end_headers()
            return self._stream(channel)

        if path == "/frame.jpg":
            _, jpeg, _ = snapshot("local")
            if jpeg is None:
                return self.send(503, b"", "image/jpeg")
            return self.send(200, jpeg, "image/jpeg")
        if path == "/state":
            with LOCK:
                body = {}
                for c in CHANNELS:
                    body[c] = dict(LATEST[c]["meta"])
                    body[c]["lessons"] = list(LESSONS[c])
                    body[c]["stats"] = dict(STATS[c])
                    body[c]["experience"] = len(EXPERIENCE[c])
                    body[c]["mentor"] = [dict(v) for v in MENTOR_GUIDANCE[c].values()]
                    body[c]["mentor_failures"] = {
                        str(k): v for k, v in FAIL_COUNT[c].items()
                    }
            return self.send(200, json.dumps(body, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        self.send(404, b"not found", "text/plain")

    def _stream(self, channel="local"):
        """长连接推流：只发最新帧，客户端慢就直接跳过中间帧。"""
        last = -1
        last_sent_at = 0.0
        try:
            while not STOP.is_set():
                seq, jpeg, _meta = snapshot(channel)
                now = time.monotonic()
                # A browser opening a finished episode still needs the next
                # multipart boundary to display its first JPEG. Repeat the
                # terminal image once a second even when no new frame arrives.
                if jpeg is None or (seq == last and now - last_sent_at < 1.0):
                    time.sleep(0.005)
                    continue
                last = seq
                last_sent_at = now
                header = (b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                          + str(len(jpeg)).encode() + b"\r\n\r\n")
                self.wfile.write(header + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return  # 观众关掉页面了，正常结束这个连接线程

    def log_message(self, fmt, *args):
        pass


def play_episode(level: str, fps: float, mode: str, edition: int, sync: bool = False, lessons=(), model="local", channel="local", experience=None, mentor=None, verified_route: bool = False, frame_sink=None, model_only: bool = False) -> dict:
    replay_only = model == "replay"
    if model_only:
        if not sync or verified_route or model not in ("local", "jev"):
            raise ValueError("model-only requires a synchronous model without a verified route")
        lessons, experience, mentor = (), None, None
    if fps <= 0:
        raise ValueError("fps must be positive")
    if verified_route and level not in VERIFIED_ROUTES:
        raise ValueError(f"No verified route for {level}")
    if replay_only:
        if not verified_route:
            raise ValueError("Offline replay requires a verified route")
        sync = True
    env = JoypadSpace(P.make_mario_env(gym_super_mario_bros, level), SIMPLE_MOVEMENT)
    ram = P.nes(env).ram
    obs, _ = env.reset()
    client = httpx.Client(timeout=600)
    pipeline = Pipeline(client, lessons, model, experience, mentor)

    action, prev, hold_cap = 0, 0, P.FULL_JUMP_FRAMES
    frame, best, last_best, last_gain = 0, 0, 0, 0
    emulator_frames = 0
    info = {"x_pos": 0, "flag_get": False}
    term = trunc = False
    calls, started, last_decision = 0, time.perf_counter(), None
    model_decision_sources = collections.Counter()
    model_error = None
    recent = []  # [(choice, x_at_decision)]，作为"最近决策"块喂给模型
    last_state = None  # 死亡前最后一次决策的 state，用来诊断死因（死亡帧里敌人/坑已消失）
    last_applied_state = None
    last_applied_action = None
    last_applied_frame = None
    mentor_used = set()  # 一局内同一指导只强制应用一次，避免原地重复触发
    mentor_pending = None  # wait-then-jump 计划的下一阶段
    mentor_trace = []  # phases actually executed from one mentor plan in this episode
    route_plan = tuple(VERIFIED_ROUTES.get(level, ())) if verified_route else ()
    route_index = 0
    route_trace = []
    route_used = bool(route_plan)
    route_prefix_frames = 0
    next_frame_at = time.perf_counter()

    def step(a: int) -> bool:
        nonlocal obs, term, trunc, info, frame, best, next_frame_at, emulator_frames
        obs, _, term, trunc, info = env.step(a)
        frame += 1
        emulator_frames += 1
        best = max(best, int(info["x_pos"]))
        thinking, thinking_ms = pipeline.status()
        publish(obs, {
            "edition": edition, "level": level, "mode": mode, "frame": emulator_frames, "fps": fps,
            "replay_only": replay_only, "model_only": model_only,
            "x": int(info["x_pos"]), "best_x": best, "flag": bool(info["flag_get"]),
            "calls": calls, "elapsed": round(time.perf_counter() - started, 1),
            "thinking": thinking, "thinking_ms": thinking_ms or 0,
            "status": "playing", "decision": last_decision,
            "model_decision_sources": dict(model_decision_sources),
            "verified_route_used": route_used,
            "route_step": max(0, route_index - 1) if route_used else None,
            "route_prefix_frames": route_prefix_frames if route_used else 0,
        }, channel)
        if frame_sink is not None:
            frame_sink(obs, emulator_frames)
        # 限速：模拟器本来远快于实时，不限速一局 12 秒就播完了。
        next_frame_at += 1.0 / fps
        delay = next_frame_at - time.perf_counter()
        if delay > 0:
            time.sleep(min(delay, 0.25))
        else:
            next_frame_at = time.perf_counter()
        return bool(term or trunc)

    def describe():
        """建一次 prompt 用的状态：网格 + features + 摘要。"""
        g, _, _ = P.grid(ram)
        feats = P.features(g, P.speed(ram), airborne=P.airborne(ram),
                           visible=(256 - int(ram[0x03AD])) // 16)
        return {"summary": feats.pop("summary"), **feats, "grid": g, "action_before": action}

    def route_over():
        return bool(term or trunc or info.get("flag_get"))

    def execute_route_move(name: str, budget: int = 60) -> bool:
        """Execute one branch.py move with the real ``step`` publisher.

        The verified route was found with branch.Sim.execute().  Reusing its
        exact release/hold/landing/settle semantics is important: translating
        a composite move into ordinary six-frame decisions changes the NES
        timing and can turn a previously safe jump into a collision.
        """
        if name in BRANCH_RAW:
            for joypad, count in BRANCH_RAW[name]:
                for _ in range(count):
                    if step(joypad):
                        return True
            return route_over()
        if name in BRANCH_COMPOSITE:
            for part, part_budget in BRANCH_COMPOSITE[name]:
                if execute_route_move(part, part_budget) or route_over():
                    return True
            return route_over()

        joypad = P.ACTIONS[name]
        used = 0
        if joypad in P.JUMPS:
            # nes-py needs one released frame before a fresh A press.
            if step(0):
                return True
            used += 1
            cap = P.HOP_FRAMES if name == "hop right" else P.FULL_JUMP_FRAMES
            for i in range(cap):
                used += 1
                if step(joypad) or (i > 4 and not P.airborne(ram)):
                    break
            follow = 0
        else:
            follow = 0 if name == "stand" else (6 if name == "walk left" else (3 if name == "run right" else joypad))
        while used < budget and not route_over():
            step(follow)
            used += 1
        # Match branch.py: include the landing in the outcome, then settle for
        # a few frames so contact deaths are visible before the next move.
        while P.airborne(ram) and used < budget + 90 and not route_over():
            step(follow)
            used += 1
        for _ in range(8):
            if route_over():
                break
            step(follow)
        return route_over()

    def choose_action(candidate, source, state_now, x_now, mentor_hit=None):
        """Apply the current-state rescue chain in a deterministic order."""
        nonlocal mentor_pending
        applied_mentor = None
        mentor_applied = False
        # A plan's second phase is tied to the original position, not to an
        # exact signature: the enemy is expected to move during the wait.
        if mentor_pending:
            pending = mentor_pending
            if (pending.get("level") == level and frame <= pending.get("expires_frame", -1)
                    and abs(x_now - int(pending.get("x", x_now))) <= 128):
                candidate = pending["action"]
                source = "mentor_followup"
                applied_mentor = dict(pending)
                mentor_applied = True
                mentor_pending = None
            elif frame > pending.get("expires_frame", -1):
                mentor_pending = None

        key = _scene_key(level, mentor_hit.get("signature")) if mentor_hit else None
        plan = P.parse_mentor_plan(mentor_hit) if mentor_hit and not mentor_applied else None
        mentor_action = plan.get("action") if plan else None
        if mentor_action and key not in mentor_used and not mentor_applied:
            candidate, source = mentor_action, "mentor"
            applied_mentor = dict(mentor_hit)
            applied_mentor.update(plan)
            applied_mentor["scene_key"] = key
            mentor_applied = True
            mentor_used.add(key)
        planned_mentor_action = (
            applied_mentor.get("action") if applied_mentor and applied_mentor.get("phase", "first") == "first"
            else None
        )
        if candidate not in P.ACTIONS:
            candidate = P.policy(state_now) or "run right"
            source = "fallback"
        # A deliberate wait is the mentor's answer to the timing problem at
        # x≈887-989: let the enemy advance for one decision, then jump.  The
        # generic enemy guard quite reasonably rewrites an ordinary model
        # ``stand`` into a forward jump, but that would erase this explicit
        # two-phase plan.  Only a plan that declares a follow-up may bypass
        # the guard for this one first phase; all normal/model ``stand``
        # choices still go through the safety chain below.
        mentor_wait = bool(
            mentor_applied
            and applied_mentor.get("phase", "first") == "first"
            and applied_mentor.get("wait_decisions", 0) > 0
            and candidate == "stand"
            and applied_mentor.get("follow_up") in P.ACTIONS
        )
        if mentor_wait:
            applied_mentor["micro_wait"] = True
        # The inserted acceleration phase is itself a deliberate part of the
        # coach plan.  Rewriting ``run right`` to a point-blank jump here
        # prevents the queued jump phase from ever being reached (and makes a
        # valid plan look like it failed).  The following jump still goes
        # through hazard_guard with the current state.
        mentor_prepare = bool(
            mentor_applied
            and applied_mentor.get("phase") == "follow_up_prepare"
            and candidate == "run right"
        )
        if not mentor_wait and not mentor_prepare:
            guarded = P.hazard_guard(
                candidate,
                state_now,
                preserve_mentor_timing=bool(
                    applied_mentor
                    and str(applied_mentor.get("phase", "")).startswith("follow_up")),
            )
            if guarded != candidate:
                candidate = guarded
                source = "mentor_guard" if mentor_applied else "guard"
        # A mentor's explicit retreat is intentional progress preparation; the
        # generic forward-stall detector must not immediately rewrite it.
        if mentor_wait or (mentor_applied and candidate in ("walk left", "back off for a run-up")):
            forced = candidate
        else:
            forced = P.unstick(candidate, [{"choice": c, "x": x} for c, x in recent], x_now)
        if forced != candidate:
            candidate = forced
            source = "mentor_unstick" if mentor_applied else "unstick"
        if (mentor_applied and applied_mentor
                and applied_mentor.get("phase", "first") == "first"
                and applied_mentor.get("follow_up")) \
                and (candidate == applied_mentor.get("action") or planned_mentor_action == applied_mentor.get("action")):
            follow_up = applied_mentor["follow_up"]
            # The coach often compresses the measured maneuver into
            # ``back off -> run and jump``.  A retreat only changes position;
            # it does not build running speed.  Insert the missing acceleration
            # phase while preserving the coach plan as the source of the
            # eventual jump.
            prepare = (candidate == "back off for a run-up"
                       and follow_up in ("jump right", "run and jump right"))
            mentor_pending = {
                "text": applied_mentor.get("text", ""),
                "action": "run right" if prepare else follow_up,
                "next_action": follow_up if prepare else None,
                "level": level,
                "signature": applied_mentor.get("signature"),
                "x": x_now,
                "plan_id": applied_mentor.get("plan_id"),
                "scene_key": applied_mentor.get("scene_key"),
                "expires_frame": frame + max(P.HOLD * 8, P.FULL_JUMP_FRAMES + P.HOLD * 2),
                "phase": "follow_up_prepare" if prepare else "follow_up",
            }
        elif (mentor_applied and applied_mentor
              and applied_mentor.get("phase") == "follow_up_prepare"
              and applied_mentor.get("next_action")
              and candidate == applied_mentor.get("action")):
            # Consume the inserted run-right phase, then expose the coach's
            # original jump as the next executable phase.
            mentor_pending = {
                **applied_mentor,
                "action": applied_mentor["next_action"],
                "next_action": None,
                "phase": "follow_up_2",
                "x": x_now,
                "expires_frame": frame + max(P.FULL_JUMP_FRAMES + P.HOLD * 2, P.HOLD * 8),
            }
        if mentor_applied:
            print(f"[{channel}] mentor apply phase={applied_mentor.get('phase', 'first') if applied_mentor else 'first'} "
                  f"action={candidate!r} source={source} x={x_now}", flush=True)
        return candidate, source, applied_mentor if mentor_applied else None

    def record_mentor_application(applied_mentor, actual_action, actual_source, x_now):
        """Persist lightweight evidence that a mentor phase really ran.

        The episode result uses the same evidence to distinguish a plan that
        was never executed from one that was executed and failed at the same
        scene.  That distinction is what lets the outer loop ask the coach to
        rethink a bad plan instead of silently reusing it forever.
        """
        if not applied_mentor:
            return
        phase = applied_mentor.get("phase", "first")
        plan_id = applied_mentor.get("plan_id")
        row = {"plan_id": plan_id, "phase": phase,
               "requested_action": applied_mentor.get("action"),
               "action": actual_action, "source": actual_source,
               "rewritten": actual_action != applied_mentor.get("action"),
               "x": int(x_now), "frame": emulator_frames,
               "scene_key": applied_mentor.get("scene_key")}
        mentor_trace.append(row)
        if plan_id is None:
            return
        with LOCK:
            for entry in MENTOR_GUIDANCE[channel].values():
                if entry.get("plan_id") != plan_id:
                    continue
                entry["applied_count"] = int(entry.get("applied_count", 0)) + 1
                entry["last_applied_x"] = int(x_now)
                entry["last_applied_phase"] = phase
                entry["last_applied_action"] = actual_action
                break

    # 第一帧就把请求发出去，别让 Mario 干站着等（同步基准模式不需要）
    if not sync:
        initial_state = describe()
        pipeline.request(initial_state, recent[-6:], 0, int(info["x_pos"]),
                         _matching_mentor(mentor, level, initial_state, int(info["x_pos"])))

    while frame < P.MAX_FRAMES and not STOP.is_set():
        if frame % P.HOLD == 0:
            # 沿用 play.py 的语义：只在落地时决策，空中保持方向不带 A。
            fall = 0
            # branch.py's verified route starts with the same short rightward
            # takeoff used by its search loop.  The ordinary watcher starts
            # neutral; use the branch-compatible input only for route episode
            # zero so the replayed timing/state is identical.
            initial_takeoff = P.ACTIONS["run right"] if route_used and route_index == 0 else P.RELEASE.get(action, action)
            while P.airborne(ram) and fall < 120:
                if route_used and route_index == 0:
                    route_prefix_frames += 1
                if step(initial_takeoff):
                    break
                fall += 1
            if term or trunc:
                break
            frame += (-frame) % P.HOLD

        if frame % P.HOLD == 0:
            state_now = describe()
            last_state = state_now  # 持续记下最新一次决策时的 state
            applied_mentor = None
            mentor_micro_wait = None
            model_choice = None
            source = "none"
            model_source = "none"
            probs = None
            latency = 0.0
            if sync:
                # 同步等决策：决策期间画面推流显示"思考中"，而不是完全冻死。
                publish(obs, {"edition": edition, "level": level, "mode": mode, "frame": emulator_frames, "fps": fps,
                              "replay_only": replay_only, "model_only": model_only,
                              "x": int(info["x_pos"]), "best_x": best, "flag": bool(info["flag_get"]),
                              "calls": calls, "elapsed": round(time.perf_counter() - started, 1),
                              "thinking": not replay_only, "thinking_ms": 0,
                              "status": "离线验证路线回放" if replay_only else "思考中...", "decision": last_decision,
                              "verified_route_used": route_used,
                              "route_step": max(0, route_index - 1) if route_used else None,
                              "route_prefix_frames": route_prefix_frames if route_used else 0}, channel)
                mentor_hit = _matching_mentor(mentor, level, state_now, int(info["x_pos"]))
                calls += int(not replay_only)
                try:
                    if replay_only:
                        name, probs, latency, source = "stand", None, 0.0, "offline_replay"
                    elif model == "jev":
                        name, probs, _tokens, latency = P.ask_jev(client, state_now, recent[-6:], lessons, experience, mentor_hit)
                        source = "jev"
                    else:
                        name, probs, _tokens, latency, source = P.ask_local(client, state_now, recent[-6:], lessons, experience, mentor_hit, strict=model_only)
                    if model_only and (source not in ("model", "jev") or name not in P.ACTIONS):
                        raise ValueError("model-only rejected non-model or invalid action")
                except Exception as exc:
                    if model_only:
                        # Keep credentials and provider response bodies out of the public viewer.
                        model_error = type(exc).__name__
                        model_decision_sources["error"] += 1
                        last_decision = {"frame": emulator_frames, "x": int(info["x_pos"]),
                                         "choice": None, "model_choice": None, "source": "error"}
                        break
                    # 官方/本地模型单次调用失败不能崩掉整局：退回确定性规则兜底，继续跑
                    name = P.policy(state_now) or "run right"
                    probs, latency, source = None, 0.0, "fallback"
                    print(f"[{channel}] 决策失败，规则兜底: {exc}", flush=True)
                model_choice = name if source in ("model", "jev") else None
                model_source = source
                if not route_used and not model_only:
                    name, source, applied_mentor = choose_action(
                        name, source, state_now, int(info["x_pos"]), mentor_hit)
                    record_mentor_application(applied_mentor, name, source, int(info["x_pos"]))
                mentor_micro_wait = applied_mentor if applied_mentor and applied_mentor.get("micro_wait") else None
                recent.append((name, int(info["x_pos"])))
                last_applied_state = state_now
                last_applied_action = name
                last_applied_frame = emulator_frames
                action = P.ACTIONS[name]
                hold_cap = P.HOP_FRAMES if name == "hop right" else P.FULL_JUMP_FRAMES
                last_decision = {
                    "frame": emulator_frames, "x": int(info["x_pos"]), "choice": name,
                    "model_choice": model_choice,
                    "source": source, "model_source": model_source, "latency_ms": round(latency * 1000),
                    "probs": {k: round(v, 4) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])}
                             if probs else None,
                    "mentor_action": applied_mentor.get("action") if applied_mentor else None,
                    "mentor_phase": applied_mentor.get("phase", "first") if applied_mentor else None,
                    "mentor_plan_id": applied_mentor.get("plan_id") if applied_mentor else None,
                    "summary": state_now["summary"],
                }
            else:
                # 实时模式：取走后台送来的决策；没有就继续用上一个动作 —— 主循环从不等待。
                ready = pipeline.take()
                current_x = int(info["x_pos"])
                mentor_hit = _matching_mentor(mentor, level, state_now, current_x)
                if ready and ready.get("name"):
                    ready_x = int(ready.get("x", current_x))
                    age = frame - int(ready.get("frame", frame))
                    # At 45 fps a 1.5 s result is already ~67 frames old; accepting
                    # it recreates the repeated-scene bug.  Keep only a short
                    # freshness window and let the current-state fallback decide.
                    stale = age > max(P.HOLD * 6, 36) or abs(current_x - ready_x) > 128
                    if stale:
                        # 旧结果已经不描述当前画面，宁可用当前状态规则决策，
                        # 也不要把过时动作继续灌进同一障碍。
                        name, source = P.policy(state_now) or "run right", "stale_fallback"
                        probs, latency = None, 0
                        print(f"[{channel}] 丢弃过期决策 age={age} x={ready_x}->{current_x}", flush=True)
                    else:
                        name, source = ready["name"], ready.get("source", "model")
                        probs, latency = ready.get("probs"), ready.get("latency_ms", 0)
                    model_choice = name if source in ("model", "jev") else None
                    model_source = source
                    if not route_used:
                        name, source, applied_mentor = choose_action(
                            name, source, state_now, current_x, mentor_hit)
                        record_mentor_application(applied_mentor, name, source, current_x)
                    mentor_micro_wait = applied_mentor if applied_mentor and applied_mentor.get("micro_wait") else None
                    calls += 1
                    recent.append((name, current_x))
                    last_applied_state = state_now
                    last_applied_action = name
                    last_applied_frame = emulator_frames
                    action = P.ACTIONS[name]
                    hold_cap = P.HOP_FRAMES if name == "hop right" else P.FULL_JUMP_FRAMES
                    last_decision = {
                        "frame": emulator_frames, "x": current_x, "choice": name,
                        "model_choice": model_choice,
                        "source": source, "latency_ms": latency,
                        "probs": {k: round(v, 4) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])}
                                 if probs else None,
                        "mentor_action": applied_mentor.get("action") if applied_mentor else None,
                        "mentor_phase": applied_mentor.get("phase", "first") if applied_mentor else None,
                        "mentor_plan_id": applied_mentor.get("plan_id") if applied_mentor else None,
                        "summary": state_now["summary"],
                    }
                elif ready and ready.get("error"):
                    # Pipeline 的错误不能只更新 UI，否则 action 会保持上一动作，
                    # 在坑/敌人前继续盲走。按当前 state 走同一规则链。
                    name, source = P.policy(state_now) or "run right", "fallback_error"
                    model_choice = name if source in ("model", "jev") else None
                    model_source = source
                    if not route_used:
                        name, source, applied_mentor = choose_action(
                            name, source, state_now, current_x, mentor_hit)
                        record_mentor_application(applied_mentor, name, source, current_x)
                    mentor_micro_wait = applied_mentor if applied_mentor and applied_mentor.get("micro_wait") else None
                    calls += 1
                    recent.append((name, current_x))
                    last_applied_state = state_now
                    last_applied_action = name
                    last_applied_frame = emulator_frames
                    action = P.ACTIONS[name]
                    hold_cap = P.HOP_FRAMES if name == "hop right" else P.FULL_JUMP_FRAMES
                    last_decision = {"frame": emulator_frames, "x": current_x, "choice": name,
                                     "model_choice": model_choice,
                                     "source": source, "latency_ms": 0, "probs": None,
                                     "mentor_action": applied_mentor.get("action") if applied_mentor else None,
                                      "mentor_phase": applied_mentor.get("phase", "first") if applied_mentor else None,
                                      "mentor_plan_id": applied_mentor.get("plan_id") if applied_mentor else None,
                                     "summary": f"模型请求失败，规则兜底：{ready['error']}"}
                    print(f"[{channel}] 异步决策失败，当前状态规则兜底: {ready['error']}", flush=True)
                # 立刻为下一步发请求（如果上一步还在飞就跳过，避免排队越等越旧）
                pipeline.request(state_now, recent[-6:], frame, current_x, mentor_hit)

            if model_source != "none":
                model_decision_sources[model_source] += 1

            # Once the episode has explicitly entered the verified recovery
            # route, the model is still queried and its raw answer is kept in
            # telemetry, but the emulator-backed route owns the actual move.
            # This branch must run before mentor micro-waits / ordinary action
            # handling, otherwise a queued coach phase could consume a frame
            # and change the route's physics timing.
            if route_used and route_index < len(route_plan):
                mentor_pending = None
                route_step = route_index
                route_name, route_ride = route_plan[route_index]
                route_index += 1
                x_before = int(info["x_pos"])
                route_entry = {
                    "route_step": route_step,
                    "requested_action": route_name,
                    "ride": route_ride,
                    "model_choice": model_choice,
                    "model_source": model_source,
                    "x_before": x_before,
                    "route_prefix_frames": route_prefix_frames if route_step == 0 else 0,
                    "actions": [],
                }
                route_actions = [route_name] + ([route_ride] if route_ride else [])
                for ride_index, route_action in enumerate(route_actions):
                    last_decision = {
                        "frame": emulator_frames,
                        "x": int(info["x_pos"]),
                        "choice": route_action,
                        "model_choice": model_choice,
                        "source": "verified_route",
                        "latency_ms": round(latency * 1000) if latency else 0,
                        "probs": {k: round(v, 4) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])}
                                 if probs else None,
                        "route_step": route_step,
                        "route_ride": bool(ride_index),
                        "verified_route_used": True,
                        "summary": state_now["summary"],
                    }
                    x_action_before = int(info["x_pos"])
                    last_applied_state = describe()
                    last_applied_action = route_action
                    last_applied_frame = emulator_frames
                    died_or_flagged = execute_route_move(route_action)
                    route_entry["actions"].append({
                        "action": route_action,
                        "x_before": x_action_before,
                        "x_after": int(info["x_pos"]),
                        "frame_after": emulator_frames,
                        "done": bool(died_or_flagged),
                    })
                    if died_or_flagged:
                        break
                route_entry["x_after"] = int(info["x_pos"])
                route_trace.append(route_entry)
                action, prev = 0, 0
                # Keep the next decision boundary aligned with the normal
                # six-frame loop without adding emulator frames.
                frame += (-frame) % P.HOLD
                if info.get("flag_get") or term or trunc or route_index >= len(route_plan):
                    break
                continue

            # A full HOLD-sized ``stand`` is too coarse when the enemy is one
            # or two tiles away: the collision can happen before the next
            # decision boundary.  Execute the mentor's explicit wait phase as
            # a short, observable micro-wait, then consume the queued
            # follow-up immediately.  This keeps the plan LLM-authored while
            # respecting the emulator's frame timing.
            if mentor_micro_wait and mentor_pending:
                initial_enemy = (state_now.get("enemy_ahead") or {}).get("tiles", 99)
                # If the enemy is already one tile away, any extra frame can
                # be the collision frame.  In that case the coach's "wait"
                # phase is already complete and we jump immediately.
                wait_frames = 0 if initial_enemy <= 1 else min(3, max(1, P.HOLD - 1))
                print(f"[{channel}] mentor micro-wait {wait_frames} frames before follow-up",
                      flush=True)
                for _ in range(wait_frames):
                    if step(P.ACTIONS["stand"]):
                        break
                    live_state = describe()
                    enemy = live_state.get("enemy_ahead") or {}
                    if enemy.get("tiles", 99) <= 1:
                        break
                if term or trunc:
                    break
                pending = mentor_pending
                mentor_pending = None
                requested_follow_action = pending["action"]
                follow_action = requested_follow_action
                follow_source = "mentor_followup"
                follow_x = int(info["x_pos"])
                follow_state = describe()
                # This is an explicit second phase from the coach.  Preserve
                # its timing-sensitive forward jump; the ordinary enemy guard
                # would rewrite it to ``jump in place`` under the low ceiling
                # at x≈887-989, which is the known repeated-death loop.  Gap
                # and wall impossibilities are still guarded by hazard_guard.
                guarded_follow = P.hazard_guard(
                    follow_action, follow_state, preserve_mentor_timing=True)
                if guarded_follow != follow_action:
                    follow_action = guarded_follow
                    follow_source = "mentor_guard"
                    print(f"[{channel}] mentor follow-up guard {requested_follow_action!r} -> "
                          f"{follow_action!r} x={follow_x}", flush=True)
                record_mentor_application(pending, follow_action, follow_source, follow_x)
                # A prepared coach maneuver may have one final jump phase
                # queued (back-off -> run-right -> run-and-jump).  Queue it
                # only after the acceleration action was actually executed.
                if pending.get("next_action"):
                    mentor_pending = {
                        **pending,
                        "action": pending["next_action"],
                        "next_action": None,
                        "phase": "follow_up_2",
                        "x": follow_x,
                        "expires_frame": frame + max(P.FULL_JUMP_FRAMES + P.HOLD * 2, P.HOLD * 8),
                    }
                calls += 1
                recent.append((follow_action, follow_x))
                last_state = follow_state
                last_applied_state = follow_state
                last_applied_action = follow_action
                last_applied_frame = emulator_frames
                action = P.ACTIONS[follow_action]
                hold_cap = P.HOP_FRAMES if follow_action == "hop right" else P.FULL_JUMP_FRAMES
                last_decision = {
                    "frame": emulator_frames, "x": follow_x, "choice": follow_action,
                    "model_choice": None,
                    "source": follow_source, "latency_ms": 0, "probs": None,
                    "mentor_action": requested_follow_action,
                    "mentor_phase": "follow_up",
                    "mentor_plan_id": pending.get("plan_id"),
                    "summary": follow_state["summary"],
                }
                print(f"[{channel}] mentor apply phase=follow_up action={follow_action!r} "
                      f"source={follow_source} x={follow_x}", flush=True)

            # A compressed coach plan ``back off -> run and jump`` gets an
            # explicit acceleration macro.  One normal HOLD (6 frames) is not
            # enough to rebuild full speed after a retreat, especially before
            # the low-ceiling gap at x≈1196.  The inserted pending phase is
            # marked prepare, so execute it for a measured 24 frames and then
            # let the queued jump phase be consumed on the next decision.
            if (applied_mentor and applied_mentor.get("phase") == "follow_up_prepare"
                    and action == P.ACTIONS["run right"]):
                for _ in range(24):
                    if step(action):
                        break
                if term or trunc:
                    break
                frame += (-frame) % P.HOLD
                action = 0
                prev = 0
                continue

            if action == P.BACK_OFF:
                # 原子动作：向左刹住并留出助跑距离。旧实现把“当前没有
                # 墙/坑”当成 tiles=99，第一帧就结束，教练的 back-off
                # 实际变成了原地等待，后续 run+jump 永远没有速度。
                # 只有真实检测到障碍且距离已经 >=6 时才提前结束；无障碍
                # 场景至少反向一小段，直到速度归零并有可用后方跑道。
                reversed_frames = 0
                for _ in range(60):
                    if step(6):
                        break
                    reversed_frames += 1
                    fe = P.features(P.grid(ram)[0], P.speed(ram))
                    obstacle = fe.get("wall_ahead") or fe.get("gap_ahead")
                    # Even when the obstacle is already six tiles away, keep
                    # reversing long enough to cancel rightward momentum and
                    # leave a real runway for the next run phase.
                    if obstacle and obstacle.get("tiles", 0) >= 6 and reversed_frames >= 18:
                        break
                    if (reversed_frames >= 12 and P.speed(ram) <= 0
                            and fe.get("clear_behind", 0) >= 3):
                        break
                    if reversed_frames >= 24 and P.speed(ram) <= 0:
                        break
                if term or trunc:
                    break
                frame += (-frame) % P.HOLD
                action = 0
            if action in P.JUMPS and prev in P.JUMPS and step(P.RELEASE[action]):
                break
            prev = action
            if action in P.JUMPS:
                # 一次决策 = 一次完整跳跃：按住 A 直到落地，或最多 hold_cap 帧。
                for i in range(P.FULL_JUMP_FRAMES):
                    if step(action) or i >= hold_cap or (i > 4 and not P.airborne(ram)):
                        break
                if term or trunc:
                    break
                frame += (-frame) % P.HOLD
                action = P.RELEASE[action]

        if step(action):
            break
        if best > last_best:
            last_best, last_gain = best, frame
        if info["flag_get"] or frame - last_gain > P.STALL_FRAMES:
            break

    lesson = None
    death_info = None
    if not info.get("flag_get") and not model_only:
        try:
            # 诊断必须尽量使用真正执行动作时的 state；异步结果可能来自更早的帧。
            death_state = last_applied_state or last_state or describe()
            applied_choice = last_applied_action or (last_decision or {}).get("choice")
            lesson = P.diagnose_death(death_state, applied_choice, best, level)
            # 死亡现场详情（供教练模型分析卡点）
            death_info = {
                "summary": death_state.get("summary", ""),
                "grid": death_state.get("grid", ""),
                "history": "\n".join(f"  {c} -> x={x}" for c, x in recent[-6:]),
                "signature": P.scene_signature(death_state),
                "x": best,
                "level": level,
                "applied_action": applied_choice,
                "applied_frame": last_applied_frame,
                "enemy_ahead": death_state.get("enemy_ahead"),
                "landing_bad": bool(death_state.get("enemies_near_landing_spot")),
                 "headroom": death_state.get("headroom"),
                 "mentor_trace": list(mentor_trace),
                 "verified_route_used": route_used,
                 "verified_route_step": max(0, route_index - 1) if route_used else None,
                 "route_prefix_frames": route_prefix_frames if route_used else 0,
                 "verified_route_trace": list(route_trace),
                 "mentor_plan_id": mentor_trace[-1].get("plan_id") if mentor_trace else None,
                "mentor_applied_x": mentor_trace[-1].get("x") if mentor_trace else None,
            }
        except Exception:
            lesson = None
    env.close()
    client.close()
    thinking, thinking_ms = pipeline.status()
    publish(obs, {"edition": edition, "level": level, "mode": mode, "frame": emulator_frames, "fps": fps,
                  "replay_only": replay_only, "model_only": model_only,
                  "x": int(info["x_pos"]), "best_x": best, "flag": bool(info["flag_get"]),
                  "calls": calls, "elapsed": round(time.perf_counter() - started, 1),
                  "thinking": thinking, "thinking_ms": thinking_ms or 0,
                  "status": f"模型错误，已停止：{model_error}" if model_error else ("离线回放结束" if replay_only else "本局结束"),
                  "error": model_error, "decision": last_decision,
                  "model_decision_sources": dict(model_decision_sources),
                  "verified_route_used": route_used,
                  "route_step": max(0, route_index - 1) if route_used else None,
                  "route_prefix_frames": route_prefix_frames if route_used else 0}, channel)
    result = {"level": level, "best_x": best, "flag": bool(info["flag_get"]),
              "frames": emulator_frames, "decision_clock": frame,
              "calls": calls, "mode": mode, "replay_only": replay_only, "lesson": lesson,
              "model_only": model_only, "error": model_error,
              "model_decision_sources": dict(model_decision_sources),
              "death_info": death_info, "mentor_trace": list(mentor_trace),
              "verified_route_used": route_used,
              "route_prefix_frames": route_prefix_frames if route_used else 0,
              "verified_route_steps": list(route_trace),
              "route_complete": bool(route_used and route_index >= len(route_plan) and info.get("flag_get")),}
    print(f"[run {edition}] {json.dumps(result, ensure_ascii=False)}", flush=True)
    return result


def main():
    P.load_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", default="1-1", choices=LEVELS)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--fps", type=float, default=45.0, help="画面播放帧率（模拟器不限速会看不清）")
    ap.add_argument("--mode", default=os.environ.get("LOCAL_POLICY_MODE", "chat"))
    ap.add_argument("--model", default="dual", choices=["local", "jev", "dual"],
                    help="模型来源：dual(本地+官方双画面，默认) / local(本地) / jev(官方)")
    ap.add_argument("--async", action="store_true", dest="async_mode",
                     help="实验性异步决策（不暂停模拟器；延迟会造成旧状态动作，通关不稳）")
    ap.add_argument("--verified-route", action="store_true",
                     help="直接启用已由模拟器分支搜索验证的恢复路线（用于验收；页面会明确标记）")
    ap.add_argument("--stay-on-level", action="store_true",
                     help="通关 --level 后停留在成功画面，不自动进入下一关（验收/观战用）")
    ap.add_argument("--replay-only", action="store_true",
                    help="离线回放已验证路线；双画面不调用模型、教练或搜索，结束后停留")
    ap.add_argument("--model-only", action="store_true",
                    help="仅同步模型决策；禁用路线、守卫、脱困、教练和规则兜底；模型错误时停止")
    a = ap.parse_args()
    if a.fps <= 0:
        ap.error("--fps must be positive")
    if a.model_only and (a.async_mode or a.verified_route or a.replay_only):
        ap.error("--model-only cannot be combined with --async, --verified-route or --replay-only")
    if (a.verified_route or a.replay_only) and a.level not in VERIFIED_ROUTES:
        ap.error("no verified route for " + a.level + "; available: " + ", ".join(VERIFIED_ROUTES))
    if a.replay_only:
        if a.async_mode:
            ap.error("--replay-only cannot be combined with --async")
        a.verified_route = True
        a.stay_on_level = True
    os.environ["LOCAL_POLICY_MODE"] = a.mode
    sync = not a.async_mode  # 默认同步：稳定通关优先，观看尽量流畅

    with LOCK:
        for c in CHANNELS:
            STATS[c]["level"] = a.level

    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    server.daemon_threads = True

    channels = CHANNELS if a.model == "dual" else (a.model,)

    def serve():
        with LOCK:
            for c in CHANNELS:
                LATEST[c]["meta"] = {"fps": a.fps, "mode": a.mode, "level": a.level, "sync": sync,
                                     "channel": c, "enabled": c in channels, "model_only": a.model_only,
                                     "model_name": "Jev (jev-latest)" if c == "jev" else P.LOCAL_POLICY_MODEL,
                                     "status": "起来了，等第一局", "thinking": False}
        server.serve_forever()

    threading.Thread(target=serve, daemon=True).start()
    print(json.dumps({"url": f"http://127.0.0.1:{a.port}/", "level": a.level, "mode": a.mode,
                      "fps": a.fps, "sync": sync, "stream": "/stream.mjpg"}, ensure_ascii=False), flush=True)

    def episode_loop(channel, model):
        edition = 0
        lessons = []  # 全量累积的失败教训（去重，不丢），供页面展开查看
        # Honour --level for targeted replay/verification; the public launcher
        # still passes 1-1, so normal runs keep the full progression.
        level_idx = LEVELS.index(a.level) if a.level in LEVELS else 0
        clears = 0     # 累计通关次数
        best_x = 0     # 当前关卡的历史最远（跨局累积；通关进新关后重置）
        while not STOP.is_set():
            edition += 1
            level = LEVELS[level_idx]
            with LOCK:
                route_failures = VERIFIED_ROUTE_FAILURES[channel].get(level, 0)
            route_enabled = bool(
                not a.model_only and VERIFIED_ROUTES.get(level)
                and (a.verified_route or route_failures >= 3)
            )
            r = play_episode(level, a.fps, a.mode, edition, sync=sync,
                             lessons=inject_lessons(lessons), model="replay" if a.replay_only else model, channel=channel,
                             experience=EXPERIENCE[channel], mentor=MENTOR_GUIDANCE[channel],
                             verified_route=route_enabled, model_only=a.model_only)
            best_x = max(best_x, r.get("best_x", 0))
            if a.model_only and not r.get("flag"):
                with LOCK:
                    STATS[channel] = {"clears": clears, "level": level, "best_x": best_x}
                if r.get("error"):
                    return
                time.sleep(3)
                continue
            if r.get("flag"):
                # 通关：进下一关（封顶），旧关教训清空，重新学新关
                clears += 1
                if a.stay_on_level:
                    with LOCK:
                        STATS[channel] = {"clears": clears, "level": level,
                                          "best_x": max(best_x, r.get("best_x", 0))}
                    print(f"[{channel}] 通关 {level}！按 --stay-on-level 停留在成功画面",
                          flush=True)
                    return
                level_idx = min(level_idx + 1, len(LEVELS) - 1)
                lessons = []
                best_x = 0  # 新关重新统计最远
                FAIL_COUNT[channel] = {}  # 新关重置失败计数
                MENTOR_ATTEMPTS[channel] = {}
                MENTOR_GUIDANCE[channel] = {}
                EXPERIENCE[channel].clear()
                with LOCK:
                    VERIFIED_ROUTE_FAILURES[channel].pop(level, None)
                with LOCK:
                    LESSONS[channel] = []
                print(f"[{channel}] 通关 {level}！累计 {clears} 次 → 进入 {LEVELS[level_idx]}", flush=True)
            else:
                if a.replay_only:
                    print(f"[{channel}] 离线回放未通关 {level}，停留以供检查", flush=True)
                    return
                if level in VERIFIED_ROUTES:
                    with LOCK:
                        VERIFIED_ROUTE_FAILURES[channel][level] = (
                            VERIFIED_ROUTE_FAILURES[channel].get(level, 0) + 1
                        )
                    print(
                        f"[{channel}] verified route failures level={level} -> "
                        f"{VERIFIED_ROUTE_FAILURES[channel][level]} "
                        f"(next episode {'enabled' if VERIFIED_ROUTE_FAILURES[channel][level] >= 3 else 'waiting'})",
                        flush=True,
                    )
                lesson = r["lesson"]  # 结构化 dict：{text, category, x, level, signature}
                if lesson:
                    text = lesson["text"]
                    existing = next((x for x in lessons if isinstance(x, dict) and x["text"] == text), None)
                    if existing:
                        # 同类死因：合并，累积失败位置与次数
                        if lesson["x"] not in existing["positions"]:
                            existing["positions"].append(lesson["x"])
                        existing["count"] += 1
                        existing["last_x"] = lesson["x"]
                    else:
                        lessons.append({
                            "text": text, "category": lesson.get("category"),
                            "positions": [lesson["x"]], "last_x": lesson["x"],
                            "level": lesson.get("level"), "signature": lesson.get("signature"),
                            "count": 1,
                        })
                    with LOCK:
                        LESSONS[channel] = list(lessons)

                # 失败计数必须基于 death_info，而不是 lesson；未知/远距离敌人
                # 也是真实的重复场景，不能因为结构化诊断为空而永远不升级。
                di = r.get("death_info")
                if di:
                    key = _scene_key(di.get("level", level), di.get("signature"))
                    retired_plan = None
                    with LOCK:
                        cnt = FAIL_COUNT[channel].get(key, 0) + 1
                        FAIL_COUNT[channel][key] = cnt
                        plan_key, current_plan = _mentor_plan_for_failure(
                            MENTOR_GUIDANCE[channel], di.get("level", level), di, key)
                        # A plan is considered failed only when it actually ran
                        # in this episode and the episode still died near the
                        # plan's application point.  A coarse scene signature
                        # can recur later in the level, so a far-away death
                        # must not retire a plan that already made progress.
                        if current_plan and current_plan.get("plan_id") is not None:
                            plan_id = current_plan.get("plan_id")
                            applied_rows = [
                                row for row in (di.get("mentor_trace") or [])
                                if row.get("plan_id") == plan_id
                            ]
                            if applied_rows:
                                applied_x = min(int(row.get("x", di.get("x", 0)))
                                              for row in applied_rows)
                                death_x = int(di.get("x", 0))
                                if death_x <= applied_x + 128:
                                    rewritten_rows = [row for row in applied_rows if row.get("rewritten")]
                                    if rewritten_rows:
                                        current_plan["execution_mismatch_count"] = int(
                                            current_plan.get("execution_mismatch_count", 0)) + 1
                                        current_plan["last_result"] = "execution_mismatch"
                                        current_plan["last_failed_x"] = death_x
                                        # The requested plan did not actually run.  Retire it
                                        # so the coach can inspect the trace, but do not ban its
                                        # first action as though the LLM itself chose the rewrite.
                                        retired_plan = dict(current_plan)
                                        MENTOR_GUIDANCE[channel].pop(plan_key, None)
                                        print(
                                            f"[{channel}] mentor retire plan={plan_id}; execution mismatch "
                                            f"near x≈{death_x}",
                                            flush=True,
                                        )
                                    else:
                                        current_plan["failed_count"] = int(current_plan.get("failed_count", 0)) + 1
                                        current_plan["last_result"] = "failed"
                                        current_plan["last_failed_x"] = death_x
                                        print(
                                            f"[{channel}] mentor plan={plan_id} failed "
                                            f"near x≈{death_x} ({current_plan['failed_count']}/2)",
                                            flush=True,
                                        )
                                        if current_plan["failed_count"] >= 2:
                                            retired_plan = dict(current_plan)
                                            # The plan may be stored under the
                                            # trigger signature, not the death
                                            # signature (enemy distances changed).
                                            MENTOR_GUIDANCE[channel].pop(plan_key, None)
                                            print(
                                                f"[{channel}] mentor retire plan={plan_id}; "
                                                "will ask for a different first action",
                                                flush=True,
                                            )
                        last_attempt = MENTOR_ATTEMPTS[channel].get(key, 0)
                        # First request after three failures; after a plan is
                        # retired, allow a fresh reflection after two more
                        # failures instead of locking the scene to stale advice.
                        retry_gap = 2 if retired_plan else 3
                        should_ask = (cnt >= 3 and key not in MENTOR_GUIDANCE[channel]
                                      and cnt >= max(3, last_attempt + retry_gap))
                        if should_ask:
                            MENTOR_ATTEMPTS[channel][key] = cnt
                    print(f"[{channel}] 重复场景失败计数 level={key[0]} sig={key[1]} -> {cnt}", flush=True)
                    if should_ask:
                        print(f"[{channel}] mentor trigger level={key[0]} sig={key[1]} count={cnt}", flush=True)
                        mentor_input = dict(di)
                        failed_action = None
                        if retired_plan:
                            if retired_plan.get("last_result") == "execution_mismatch":
                                mentor_input.update({
                                    "mismatched_plan": retired_plan.get("text", ""),
                                    "execution_mismatch_count": retired_plan.get("execution_mismatch_count", 0),
                                })
                            else:
                                mentor_input.update({
                                    "failed_plan": retired_plan.get("text", ""),
                                    "failed_first_action": retired_plan.get("action"),
                                    "failed_plan_failures": retired_plan.get("failed_count", 0),
                                })
                                failed_action = retired_plan.get("action")
                        web_search = MS.search_for_death(mentor_input)
                        if web_search and web_search.get("status") == "ok":
                            mentor_input["web_search"] = web_search
                            print(
                                f"[{channel}] mentor web search id={web_search.get('search_id')} "
                                f"refs={len(web_search.get('references') or [])} "
                                f"latency={web_search.get('latency_ms')}ms "
                                f"cache={web_search.get('cache')}",
                                flush=True,
                            )
                        elif web_search and web_search.get("status") != "disabled":
                            print(
                                f"[{channel}] mentor web search skipped "
                                f"status={web_search.get('status')} "
                                f"error={web_search.get('error', '')}",
                                flush=True,
                            )
                        guidance_text = P.ask_mentor(mentor_input, lessons)
                        plan = P.parse_mentor_plan(guidance_text, failed_action)
                        if plan is None:
                            plan = _safety_mentor_plan(mentor_input, failed_action)
                            if plan:
                                guidance_text = plan["text"]
                        if guidance_text and plan:
                            with LOCK:
                                MENTOR_PLAN_SEQ[channel] += 1
                                plan_id = MENTOR_PLAN_SEQ[channel]
                            entry = {
                                "text": plan.get("text", guidance_text),
                                "action": plan["action"],
                                "follow_up": plan.get("follow_up"),
                                "reflection": plan.get("reflection"),
                                "wait_decisions": plan.get("wait_decisions", 0),
                                "origin": plan.get("origin", "llm"),
                                "level": di.get("level", level),
                                "signature": tuple(di.get("signature") or ()),
                                "headroom_bucket": (
                                    0 if di.get("headroom") is not None and int(di.get("headroom")) <= 0
                                    else (1 if di.get("headroom") is not None and int(di.get("headroom")) <= 2 else 2)
                                ),
                                "landing_bad": bool(di.get("landing_bad")),
                                "x": di.get("x"),
                                "trigger_count": cnt,
                                "plan_id": plan_id,
                                "applied_count": 0,
                                "failed_count": 0,
                                "last_result": "pending",
                                "web_search": MS.public_metadata(web_search),
                            }
                            with LOCK:
                                MENTOR_GUIDANCE[channel][key] = entry
                            print(f"[{channel}] mentor parsed plan={entry['action']!r} -> {entry.get('follow_up')!r} "
                                  f"origin={entry['origin']} x≈{entry['x']} "
                                  f"reflection={entry.get('reflection')!r}", flush=True)
                        elif guidance_text:
                            print(f"[{channel}] mentor invalid guidance (ignored): {guidance_text[:100]}", flush=True)
                print(f"[{channel} lessons] {len(lessons)} 条", flush=True)
            with LOCK:
                STATS[channel] = {"clears": clears, "level": LEVELS[level_idx], "best_x": best_x}
            time.sleep(3)

    threads = [threading.Thread(target=episode_loop, args=(c, c), daemon=True) for c in channels]
    for t in threads:
        t.start()
    try:
        while not STOP.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        STOP.set()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
