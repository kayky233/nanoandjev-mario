# 录像索引与模型证据边界

本目录保留历史展示和执行器回归材料。**这里的通关画面不能单独证明 Jev 或本地模型自主通关。** 模型能力评估需要逐决策记录，确认模型有效回答、实际执行动作，以及是否发生规则、搜索或路线接管；新的评估结果见[项目首页](../../README.md)。

## 1-1：历史成功记录，动作来源尚不完整

[历史汇总](../source/recordings.json)记录本地策略为 `flag=true`、x=3161、2028 帧、58 次调用；branch-Jev 为 `flag=true`、x=3161、2363 帧、23 次调用。这些是仓库附带的历史记录，不是本次重新运行模型所得的结果。

对应逐决策日志未随仓库保留，因此不能核实本地模型有效回答数、guard/unstick 改写数，也不能核实 branch-Jev 的自动 escape 搜索和 ride 续招次数。模型请求数不等于实际执行动作全部由模型决定；branch-Jev 本身使用模拟器试演候选动作。

| 原始录像 | 来源 |
| --- | --- |
| [1-1-local-20260919-135409.gif](../source/1-1-local-20260919-135409.gif) | 历史本地策略录像，展示说明标为 Qwen2.5-3B，汇总未保存模型版本 |
| [1-1-branch-jev-20260918-170421.gif](../../runs/1-1-branch-jev-20260918-170421.gif) | 历史 branch-Jev 录像，另有[结果汇总](../../runs/results.jsonl) |

**合成展示的面板不能作为逐决策证据。** [make_dual_gif.py](../../make_dual_gif.py) 给两段 1-1 录像传入空的状态记录列表；面板的 x 按播放进度插值，动作文字按预设阶段显示。`1-1-local.gif`、`1-1-official.gif` 及双路合成中的 1-1 面板不是实际模型响应或逐帧遥测。游戏画面来自上述历史录像。

## 1-2：路线实际执行，模型回答仅作记录

两路历史记录均为 `verified_route_used=true`、`route_complete=true`，23 步已验证路线负责执行动作。两路都有通关画面，但不能计入模型通关结果。

| 记录 | 模型回答来源 | 实际动作来源 |
| --- | --- | --- |
| [本地 lane 原始 JSON](../source/raw-1-2-local.json) · [录像](../source/raw-1-2-local.gif) | 23/23 步标为 `model` | 全部为 `verified_route`；模型选择未控制执行 |
| [Jev lane 原始 JSON](../source/raw-1-2-jev.json) · [录像](../source/raw-1-2-jev.gif) | **23/23 步标为 `fallback`** | 全部为 `verified_route`；没有有效 Jev 决策证据 |

旧记录的 2352 帧包含调度对齐；实际执行为 2300 模拟器帧。不能通过比较这两路的通关时间来判断模型速度或能力，因为二者执行同一条路线。

## 历史合成文件索引

以下文件保留用于追溯展示来源，不作为模型验收材料，也不据此推断模型连续完成 1-1、1-2。

| 文件 | 内容与限制 |
| --- | --- |
| [1-1-local.gif](1-1-local.gif) | 历史本地策略游戏画面；位置与动作面板为插值/预设 |
| [1-1-official.gif](1-1-official.gif) | 历史 branch-Jev 游戏画面；位置与动作面板为插值/预设 |
| [1-2-local.gif](1-2-local.gif) | 本地 lane 请求记录 + 已验证路线执行 |
| [1-2-official.gif](1-2-official.gif) | Jev lane 的 fallback 记录 + 已验证路线执行 |
| [dual-1-1-1-2.gif](dual-1-1-1-2.gif) | 上述历史素材拼接；不是一次连续模型运行 |

## 离线路线回放：仅用于执行器回归

2026-09-20 的两组回放关闭模型，检查路线时序和执行器是否可重复工作，不计入 Jev 或本地模型能力评估。

| 文件 | 回归结果 |
| --- | --- |
| [1-2-verified-route.gif](1-2-verified-route.gif) | 2/2，`flag=true`，x=3161，2300 实帧，23 步路线，0 次模型请求 |
| [1-3-verified-route.gif](1-3-verified-route.gif) | 2/2，`flag=true`，x=2425，2430 实帧，26 步路线，0 次模型请求 |

重复记录见[离线回归 JSON](offline-acceptance.json)。1-2 使用已有路线，1-3 路线由 `branch.py --bot search` 产生。GIF 标有 `VERIFIED ROUTE / NO MODEL`，各展示一次回放，按模拟器时间线抽样为 10 fps，结尾停留 1.5 秒。产生这些回归材料的本地扩展代码与测试未包含在此前的首页文档发布中。
