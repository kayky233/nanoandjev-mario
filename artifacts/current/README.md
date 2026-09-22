# 验证录像与历史录像

## 2026-09-20 本地离线验收

以下两段 GIF 为真实模拟器的验证路线回放，均标有 `VERIFIED ROUTE / NO MODEL`，未调用模型、教练或搜索服务。

| 文件 | 游戏时长 | 重复验收结果 |
| --- | ---: | --- |
| [1-2-verified-route.gif](1-2-verified-route.gif) | 38.33 秒 | 2/2，`flag=true`，x=3161，2300 实帧，23 步路线，0 次模型请求 |
| [1-3-verified-route.gif](1-3-verified-route.gif) | 40.50 秒 | 2/2，`flag=true`，x=2425，2430 实帧，26 步路线，0 次模型请求 |

每段 GIF 展示第一次回放；两次结果见[验收 JSON](offline-acceptance.json)。画面按 60 fps 模拟器时间线抽样为 10 fps，结尾另停留 1.5 秒，因此 GIF 总时长分别为 39.83 和 42.00 秒。

1-3 路线由 `branch.py --bot search` 得到；两关均通过本地扩展 viewer 回放。这是搜索/路线执行证据，不是模型自主通关证据。生成它们的离线扩展代码与测试尚未同步主分支，见[首页版本说明](../../README.md#验收范围与版本)。

## 历史模型与组合系统录像

以下五个文件随原仓库提供，保留原样。1-2 的旧计数 2352 包含调度对齐，实际为 2300 模拟器帧；下面时长保留历史录像的原始口径。

These GIFs use a 10 fps display timeline mapped to the emulator's 60 fps frame
count. Gameplay therefore keeps its recorded speed; frames are sampled for
size, not time-compressed. Single-lane GIFs hold the terminal frame for 1.5
seconds after the measured run.

| file | lane | gameplay time | result |
|---|---|---:|---|
| `1-1-local.gif` | local Qwen2.5-3B | 33.80 s | `flag=true`, `best_x=3161` |
| `1-1-official.gif` | branch-backed TypeSafe Jev | 39.38 s | `flag=true`, `best_x=3161` |
| `1-2-local.gif` | local lane + verified route | 39.20 s | `flag=true`, `best_x=3161` |
| `1-2-official.gif` | official lane + verified route | 39.20 s | `flag=true`, `best_x=3161` |
| `dual-1-1-1-2.gif` | side-by-side acceptance replay | 83.30 s total | both lanes reach both flags |

The 1-2 GIFs are evidence of the combined system. Model requests and raw
choices are retained as telemetry, while the emulator-verified 23-step route
owns the executed actions. They are not evidence that either model clears 1-2
autonomously.

Regenerate all five files with:

```powershell
uv run python make_dual_gif.py
```

Use `--refresh` only when both model endpoints are available and fresh 1-2
source captures are intended.
