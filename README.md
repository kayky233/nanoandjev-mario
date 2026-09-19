# jev-mario

[TypeSafe Jev](https://typesafe.ai) plays Super Mario Bros from the emulator's RAM. Three harnesses, each with a deterministic control.

| 1-1 | 2-1 | 3-1 |
|---|---|---|
| ![1-1](runs/1-1-branch-jev-20260918-170421.gif) | ![2-1](runs/2-1-branch-jev-20260918-155948.gif) | ![3-1](runs/3-1-branch-jev-20260918-144804.gif) |

## Results

`x` is distance reached; the flag is at about x = 3160. One life per run.

**Branching** (`branch.py`). Every option is played in the emulator for one second plus two follow-up moves, from a snapshot. Jev chooses among the measured outcomes. `--bot search` chooses by (alive, not a dead end, distance) with no API call.

| level | Jev | search |
|---|---|---|
| 1-1 | flag | 2370 |
| 2-1 | flag ¹ | 2066 |
| 3-1 | flag ¹ | 2786 |

¹ On earlier builds of `branch.py` (runs `2-1-branch-jev-20260918-155948`, `3-1-branch-jev-20260918-144804`). The current build reaches 2066 on 2-1 and 2770 on 3-1; each stall is a spot where no move in the option set survives three seconds, and the stalls differ by build because the option set does.

25–35 calls and under $0.002 per level; 15–20 minutes of emulation per level.

**Direct control** (`play.py`). One action every 6 frames from a text summary; the emulator pauses during the call. `--bot rules` is the same rules as an if-chain.

| level | Jev, 3 runs | rules | hold "run and jump" |
|---|---|---|---|
| 1-1 | 686, 686, 686 | 1129 | 1526 |
| 2-1 | 473, 473, 473 | 741 | 476 |
| 3-1 | 607, 754, 841 | 608 | 775 |
| 4-1 | 339, 339, 1305 | 1827 | |
| 5-1 | 305, 439, 439 | 271 | |

**Live speed** (`live.py`). JSON state, a Choice plus a Noul plus a Score combined in code, no pause: the emulator advances by the measured latency while the request is in flight. The rules twin gets the same 0.2 s delay.

| level | Jev, 3 runs | rules |
|---|---|---|
| 1-1 | 315, 315, 315 | 315 |
| 2-1 | 451, 453, 457 | 471 |
| 3-1 | 408, 423, 596 | 515 |

## Run

```bash
cp .env.example .env        # TYPESAFE_API_KEY
uv sync
uv run python branch.py --bot jev --level 1-1
uv run python branch.py --bot jev --levels 1-1,2-1,3-1 --attempts 2
uv run python play.py --bot jev --level 1-1
uv run python live.py --bot jev --level 1-1
```

Each run writes `runs/<level>-<mode>-<bot>-<stamp>.gif`, a per-decision log beside it, and one line to `runs/results.jsonl`.

## Dual-model viewer and recovery system

`watch_local.py` runs a side-by-side viewer with synchronous decisions by
default. The local lane sends the measured state, recent decisions, lessons,
coach plan, and nine legal choices to Qwen2.5-3B at
`http://127.0.0.1:11500/v1`. The official lane sends the same decision context
plus the structured state, `RULES`, `GRID_LEGEND`, and `ACTION_HELP` to
TypeSafe Jev.

Repeated scenes are reflected on by a separate LLM coach between episodes. A
coach response is parsed as `REFLECTION` plus a legal `PLAN`; the executed
phases and any safety rewrite remain visible in `mentor_trace`.
`hazard_guard` and `unstick` enforce measured pit, wall, ceiling, and enemy
constraints. `--async` remains an explicit latency experiment because a stale
1.5-second answer can miss a jump window.

After three failed 1-2 episodes, or immediately with `--verified-route`, an
emulator-verified recovery route owns the executed actions. Model requests
continue and their raw choices are recorded beside the route action.

| level | lane | result | execution boundary |
|---|---|---|---|
| 1-1 | local Qwen | `flag=true`, x=3161, 2028 frames, 58 calls | direct local run with guards |
| 1-1 | official Jev | `flag=true`, x=3161, 2363 frames, 23 calls | branch-backed Jev |
| 1-2 | local lane | `flag=true`, x=3161, 2352 frames, 23 calls | 23-step verified route; Qwen choices retained |
| 1-2 | official lane | `flag=true`, x=3161, 2352 frames, 23 calls | 23-step verified route; Jev choices retained |

**The 1-2 rows are successful combined-system acceptance results, not proof
that either model autonomously cleared the level.** The current scope is 1-1
and 1-2; 1-3 is not complete.

Run the local viewer without a public tunnel:

```powershell
uv run python supervise.py --level 1-1 --tunnel none
```

Use `--model dual --verified-route --stay-on-level` for a deterministic 1-2
acceptance replay. For ngrok, put `ngrok` on `PATH` (or set `NGROK_BIN`) and
optionally pass `--domain`; the viewer keeps its proxy environment so remote
model calls are not forced into fallback mode.

Five labeled, real-time completion GIFs are in
[`artifacts/current`](artifacts/current/README.md). Regenerate them with
`python make_dual_gif.py`; the display is sampled to 10 fps while duration is
derived from the emulator's 60 fps frame count.

## Mentor web search

The repeated-failure coach can optionally retrieve current walkthrough material
after a scene fails three times. It runs between episodes, never in the
per-frame decision path, and treats web text as untrusted advisory context;
emulator warnings and the hazard guard always win.

Set `BAIDU_AI_SEARCH_API_KEY` in the process environment (do not commit it),
then start the viewer with `supervise.py --web-search`. The default launcher
keeps web search off. Provider errors, timeouts, or an overdue account leave
the normal reflection path running and are recorded in `/state`.

## Notes

- `play.py` holds the shared parts: RAM to tile grid (`grid`), grid to fields and summary (`features`), the rules policy (`policy`), jump physics measured in this emulator (`JUMP_TABLE`, `HOP_TABLE`), and the three blocks marked `EDIT HERE`.
- `play.py --inspect <log> -n 3` prints the last decisions with state, grid and probabilities. `play.py --bot "replay:<log>@N:<action,...>"` replays N logged choices then holds the tail, without API calls.
- Snapshot/restore is nes-py's `_backup`/`_restore`; it has one slot, so continuations are replayed from the root. Restore was verified exact over 240 frames.
- `mario_env.py` normalizes the reset/step API across the pinned 7.x Gym stack and the newer 9.x Gymnasium stack.
- RAM: x = `0x6D:0x86`, y = `0x3B8`, horizontal speed = `0x57`, airborne = `0x1D`, screen x = `0x3AD`, tiles at `0x500` (two 16×13 pages), enemy slots `0x0F`–`0x13` with type at `0x16+i`. Verified types: 6 goomba, 0 green koopa, 13 piranha plant, 14 flying koopa.
- Frame-exact moves (`RAW` in `branch.py`) were found by probe and fail if shifted by one frame.
- Emulator: `gym-super-mario-bros` with `nes-py`, pinned to `numpy<2` and `gym==0.26`.
