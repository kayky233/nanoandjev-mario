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
