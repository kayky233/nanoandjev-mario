Original prompt: 让 Mario 双页面保持同步决策，修复同一场景反复犯错，并让 LLM 教练真正反思出“887-989 先等待敌人再跳”的策略。

- 已确认 LLM 教练能返回 `PLAN: stand; THEN: jump right`。
- 待验收：教练的等待阶段不能被通用 hazard guard 覆盖；计划失败后要允许重新反思。

## 2026-09-19 验证

- 真实双页面实例中，Jev 通道在 `(1, 0, 0, 2)` 场景连续失败 3 次后触发教练。
- 教练返回并解析为 `PLAN: stand; THEN: jump right`，记录为 `origin=llm`，不是 safety fallback。
- 该计划实际执行了 `stand`（x=874）和 `jump right`（x=883），随后该局最远推进到 x=1325，越过原先约 887-989 的卡点。
- `/state` 能看到 `mentor`、`plan_id`、`applied_count` 和 `mentor_trace`，可区分 LLM 计划与安全守卫改写。
- 教练提示已改为中性的 `REFLECTION` + `PLAN` 两行协议，不再直接给出“等待再跳”答案；解析器只执行 `PLAN` 行。
- 后续真实反思生成了 `walk left -> jump in place`，从 x≈883/899 越过卡点，单局推进到 x=1339，另一局推进到 x=1651。
- 安全 fallback 现在只提供单步 `jump in place`，不再内置两步等待策略；因此 `stand -> jump` 只能来自 `origin=llm`。
- Playwright 已确认双画面非空、关卡卡片一致；`/state` 保留 `sync=true`，页面控制台无错误。
- （历史阶段）当时尚未观察到 1-2 `flag=true`；该结论仅覆盖教练反思链路和 887-989 卡点。最新验收见下方 2026-09-20 状态。

## 2026-09-19 web search

- Added optional Baidu AI Search retrieval for mentor escalations only; it is not
  called from the per-frame decision path.
- Search results are bounded, cached by query, marked as untrusted references,
  and recorded in mentor state with `search_id`/URLs/latency.
- A provider timeout or malformed response falls back to the existing local
  lessons and mentor flow.
- Live provider smoke returned `account_overdue`; web search is therefore
  explicit opt-in (`MENTOR_WEB_SEARCH_ENABLED=true`) and currently remains
  disabled in the running instance until the provider account is usable.
- `supervise.py --web-search` now enables the feature explicitly for a run;
  the default launcher remains search-off.

## 2026-09-19 hazard/follow-up review

- `hazard_guard` now converts `stand`/`walk right`/`run right` into a jump when a wall is one tile ahead, covering the observed x≈365 wall loop.
- Explicit mentor follow-up jumps preserve coach timing through the enemy guard; gap/wall guards still apply. This prevents the known x≈887-989 `run and jump right` phase from being rewritten to `jump in place`.
- Mentor failure lookup now matches the executed `plan_id` and application x when enemy movement changes the death scene signature, so plans can retire after two nearby failures instead of being reused indefinitely.
- Static assertions and `py_compile` passed. The currently running supervisor predates this patch; restart it before judging runtime behavior.

## 2026-09-19 low-ceiling gap audit

- `features()` now exposes `jump_lands_tiles_ahead`, the low-ceiling-adjusted landing distance already used in the summary.
- `hazard_guard()` rechecks every forward jump (including a hop upgraded to a full jump) against that effective distance. When a capped landing is at or near a gap, it advances on safe ground first; the existing open-air 1068 hop case and 2404 wall case remain unchanged in synthetic assertions.
- Wall retry probe: at the observed x≈365 wall state, `jump right` is upgraded to `run and jump right`; repeated guarded attempts advanced from x=354 to x=819.
- Verification: `hazard_cases=ok`; `py_compile` passed for `play_local.py`, `watch_local.py`, `supervise.py`, and `mentor_search.py`. Live supervisor was not restarted.

## 2026-09-19 low-ceiling enemy timing

- Frame-level branch probing showed the x≈1196/1339 failures are a timing window, not a gap-width-only error. At x≈1139, with `headroom=2` and an upper goomba four tiles ahead, a full-speed `run and jump right` clears the upper enemies and the later 3-tile gap; waiting until x≈1175 makes every jump collide.
- `hazard_guard()` now recognizes that geometry (full speed, low ceiling, upper enemy within four tiles, no near wall, and a distant/3-wide gap) and upgrades ground actions to `run and jump right` without hard-coding world x.
- Watch-like frame simulation from the branch checkpoint triggered at x=1148, crossed the x≈1196 and x≈1339 regions, and reached x=1502 without death. `py_compile` and synthetic hazard checks remain green.
- The public supervisor was not restarted yet; runtime browser verification remains pending.

## 2026-09-19 dual completion GIF

- Added `make_dual_gif.py`, which composes the attachment-style 909x867 dark
  two-column viewer from real successful recordings.
- Fresh synchronous 1-2 captures for local Qwen and Jev both reached
  `best_x=3161`, `flag=true`, and completed 23 verified-route steps while
  retaining the model decision telemetry.
- Generated `output/mario-dual-1-1-1-2.gif` (193 optimized frames, about 18.3s);
  inspected intro, 1-1, transition, 1-2, and final flag frames. The GIF labels
  the 1-2 route correction explicitly instead of presenting it as pure model
  autonomy.

## 2026-09-20 current acceptance

- 1-1 local Qwen: `flag=true`, `best_x=3161`, 2028 emulator frames, 58 calls.
- 1-1 official lane: branch-backed TypeSafe Jev reached `flag=true`,
  `best_x=3161`, in 2363 emulator frames and 23 calls.
- Fresh synchronous 1-2 captures for both lanes reached `flag=true`,
  `best_x=3161`, in 2352 emulator frames with 23 requests and
  `route_complete=true`.
- Evidence boundary: the 1-2 executed actions came from the transparent
  23-step emulator-verified recovery route. Model choices remain in telemetry;
  this is combined-system success, not pure model autonomy.
- The LLM coach has independently demonstrated the repeated-scene reflection
  chain around x=887-989, including executable wait/retreat then jump plans,
  but the full stable 1-2 clear still depends on the safety and route layers.
- 1-3 remains incomplete; recent runs still fail around the opening gap at
  x approximately 315-444.
- Rebuilt `artifacts/current/*.gif` on a real-time 60 fps emulator timeline
  sampled to 10 display fps. The dual replay is 83.3 seconds instead of the
  previous 18.3-second accelerated preview.

## 2026-09-20 local continuation

- Confirmed the existing clean local checkout matches GitHub HEAD `f79ce3b`;
  installed the frozen dependency set with `uv sync --frozen` on macOS and
  Python 3.12.10. No remote write or model API invocation was performed.
- Added `IMPLEMENTATION.md` with the module map, execution boundaries, local
  commands, and verification instructions.
- Added `--replay-only` to the viewer and supervisor. It uses the verified
  route without model/coach/search calls, labels both lanes as offline replays,
  disables tunnels, and stops after success or failure. Unsupported levels and
  non-positive frame rates are rejected. Exhausted routes do not silently fall
  through into another controller.
- Corrected viewer `frames` and `/state.frame` to count real emulator steps;
  retained `decision_clock` for the six-frame scheduler. The historical 1-2
  value 2352 contains 52 alignment slots: the actual run is 2300 frames.
- Added `verify_routes.py`: repeat every supported route, fail on incomplete or
  inconsistent outcomes, and optionally write GIFs labeled `NO MODEL` at 10
  display fps on the actual 60 fps emulator timeline.
- Two fresh offline 1-2 runs reached `flag=true`, `best_x=3161`,
  `frames=2300`, `calls=0`, `route_complete=true`. The new GIF is 38.33 seconds
  of gameplay plus a 1.5 second terminal hold. These are route acceptance
  results, not new evidence of model autonomy.
- Real browser verification on localhost confirmed both 256x240 streams,
  offline labels, identical successful `/state` results, and one clear per
  lane. SIGTERM closed the viewer port, reaped its process, and removed the
  supervisor lock.
- Fixed supervisor POSIX process checks, precise Windows PID matching,
  stale-lock ownership, child termination/reaping, and failure exit behavior.
  Windows behavior has mock-test coverage; native Windows execution was not
  repeated on this Mac.
- Added `model_decision_sources` to distinguish successful answers from rules
  fallback. Failed/unparseable model decisions have `model_choice=null`;
  `calls` remains an attempt count, not a success count.
- Located the upstream repository at `https://github.com/4esv/jev-mario` and
  cloned it beside this checkout as `jev-mario-upstream` (HEAD `eaabe51`). Its
  README/GIF references match the original three examples: 1-1, 2-1, 3-1.
  The local `play.py`, `branch.py`, and `live.py` retain the upstream controller
  logic; their source differences are environment compatibility initialization.
- The original `branch.py --bot search --level 1-3` completed the level with
  `flag=true`, `best_x=2425`, `frames=2430`, and zero model requests. Converted
  the two three-move escape sequences into separate route steps (26 total),
  preserving all ride actions and emulator timing.
- Independently replayed that 1-3 route twice through the branch executor and
  twice through the viewer; every run reached the same flag/x/frame count.
  Registered the route for normal recovery and explicit offline replay.
- All 23 automated tests pass, including both routes on the real emulator and
  a failed model request that must not be reported as a model answer.
- Fixed MJPEG viewers opened after completion: repeating the terminal JPEG
  supplies the next multipart boundary, so a late browser displays the frame
  even though the emulator has stopped. Added a localhost streaming regression
  test. The real browser also verified the 1-3 dual-lane terminal state.
- Completion boundary: local startup and the 1-3 search/recovery route are
  complete. Pure model autonomy on 1-3 and subsequent sequential levels remain
  unverified. No model API, remote push, PR, or public deployment was used.

## 2026-09-23 real model evaluation and dual live viewer

- Completed nine model-only Jev runs and three real local Qwen2.5-1.5B-Instruct
  runs across 1-1/1-2/1-3; none cleared. All 184 Jev and 54 Qwen valid answers
  were executed without guard, unstick, coach, rules fallback, or route takeover.
- Re-ran branch-Jev on 1-1: x=2370, no clear, 22 API requests, 11 ride continuations,
  1,574,417 simulated candidate frames, and 27.38 minutes of wall time.
- Published the fixed-sample evidence and corrected homepage. Earlier route replay
  success and historical recordings remain separately labeled.
- Restored real two-column live viewing: cached Qwen on the left, Jev on the right.
  The public browser showed both live streams and actual model decisions; continuous
  viewer retries are not added to the fixed-sample benchmark above.
- Added a standalone cached-weight Qwen inference adapter and documented startup.
  The viewer and supervisor both support model-only execution; errors stop the
  affected lane instead of switching to a rule or route.
- Runtime credentials, tunnel addresses, PID files and temporary output remain local.
