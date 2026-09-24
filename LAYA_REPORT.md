# Laya 接入与实测：2026-09-24

当前默认组合是 **左栏本地 Laya、右栏官方 Jev**。已替换原本地 Qwen/NanoJev 调用；Qwen 的适配器和历史实验仍保留用于复现。Laya 直接运行官方权重、输出九个动作的 typed-choice 概率，不经过聊天模型生成字母，也没有规则代打。

## 实际运行配置

- Laya `0.3.11`，官方 `convaiinnovations/laya` 的 English checkpoint，421M ModernBERT-large。
- 固定 revision：`aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8`。权重约 804 MiB，SHA-256：`891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`。
- 本机 Apple M5 / 16 GiB，PyTorch `2.14.0`、Transformers `4.57.6`，实际设备 MPS、float32；未回退 CPU。
- `serve_laya.py` 复用官方 HTTP 服务，绑定 `127.0.0.1:11505`；策略请求发往 `/v1/systemone`，不使用 `/chat/completions`。
- 固定单个 checkpoint，不启用自动模型路由。响应记录实际 checkpoint、revision、设备、usage 和输入预算；客户端拒绝 checkpoint 错配。

项目依据：[官方代码与接口](https://github.com/NandhaKishorM/laya)、[固定版本模型](https://huggingface.co/convaiinnovations/laya/tree/aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8)。

## 九局真实对局

每关从初始状态独立运行三次，关闭 guard、自动脱困、教练、规则兜底和路线。以下全部为真实模型返回后原样执行。

| 关卡 | 三轮最远 x | 每轮有效请求数 | 三轮单局延迟 p50 | 通关 |
| --- | --- | ---: | --- | ---: |
| 1-1 | 1802 / 1802 / 1802 | 14 | 236 / 239 / 246 ms | 0/3 |
| 1-2 | 893 / 893 / 893 | 15 | 274 / 286 / 316 ms | 0/3 |
| 1-3 | 305 / 305 / 305 | 2 | 791 / 815 / 807 ms | 0/3 |

合计 **93 次有效决策，原始选择等于实际动作，0 次输入截断**。完整游戏请求的总体 HTTP p50 **257 ms**、p95 **790 ms**，包含本地 HTTP、预算检查、模型推理及冷形状开销。不能用这批不同状态的请求与昨日 Qwen 的隔离推理测试直接计算严格加速倍数。

必须保留的负结果：**93 次全部选择 `run and jump right`（跑跳）**。相较昨日 Qwen guided 的 x=1411 / 658 / 410，Laya 在前两关走得更远，第三关退步；仍没有通关，也没有表现出足够的动作变化。重复三次证明当前轨迹可重复，不等于三个独立多样样本，更不能据此宣称已具备更好的通用 agent 能力。

另一个 121-token 短请求探针：第一次约 3.247 秒，随后五次 HTTP p50 约 64.69 ms。这与视频描述的 45 ms 不是同一实验；首页和报告采用完整游戏请求结果，不用短探针替代实际工作负载。没有远端按 token 收费，不等于本地计算成本为零；本地 `cost_usd=null`，实际 token 数仍累计。

## 输入预算与能力归因

English checkpoint 的总上下文为 512 tokens，header 预算为 192 tokens。上游会裁剪过长的 instructions、criteria 或 state；直接搬用 Jev 长提示可能让模型看不到关键状态。

本实现采用明确的 Laya 请求格式：短规则、保留九个动作语义的短 criteria，以及地形、敌人、速度、墙/坑、可见范围、跳跃估计和最近三步动作与位置。保留全部可见地形段；去掉由可见范围推导的重复边界和重复 joypad 编号，不截取部分地形来凑预算。Jev 的输入和执行器不变，因此这不是只更换权重的严格同输入实验。

服务逐项核对官方编码预算，任何裁剪均返回 422，不执行该次推理。156 个历史状态预检全部通过；真实 93 次请求最多使用 500 tokens，没有截断。未来出现更复杂输入时可能超预算并停止，该行为比悄悄丢弃状态更可审计。`typed-decisions` 的 1024-token checkpoint 可显式另行实验，本轮未使用。

[结果汇总](artifacts/laya-eval-20260924/summary.json)与[原始证据压缩包](artifacts/laya-eval-20260924/evidence.tar.gz)包含逐次 HTTP、动作日志、usage、预算、源码快照、模型身份、短请求探针及三类超预算拒绝测试，不含认证头或凭据。游戏 GIF 保留于本机 `output/laya-20260924/evaluation/`。评测后的控制器只补充了错误 checkpoint 和非字符串 choice 的拒绝检查，未改变这批有效请求的输入或动作执行。

## 启动与复现

```bash
uv sync --frozen
uv run --script serve_laya.py --port 11505 --checkpoint english --device mps
```

首次启动下载指定 revision 的必要权重与 tokenizer；模型依赖与游戏环境分离。另一个终端配置：

```bash
export LOCAL_POLICY_BASE_URL=http://127.0.0.1:11505/v1
export LOCAL_POLICY_MODEL=english
export LOCAL_POLICY_MODE=laya
export NO_PROXY=127.0.0.1,localhost
# TYPESAFE_API_KEY 由自己的凭据管理方式注入，用于右栏 Jev。
uv run python watch_local.py --model dual --model-only --stay-on-level --level 1-1
```

打开 <http://127.0.0.1:8123/>。`supervise.py` 不再用默认 chat 覆盖 `.env`，同样可以启动 Laya 双栏；页面模型名在加载配置后读取。

```bash
uv run python evaluate_models.py --bot local --attempts 3 --output output/new-laya-eval
```

输出目录必须为空。评测会先核对服务实际 checkpoint，再记录模型版本、请求响应、实际动作和真实 token 用量。旧 Qwen 需显式选择 `LOCAL_POLICY_MODE=chat` 并指向对应推理服务。

## 行内离线部署的变化

准备固定 revision 的完整 checkpoint 目录，包含 `rl_agent_config.json`、`model.safetensors`、`encoder/` 和 `tokenizer/`，并保留版本及权重摘要。按行内目标操作系统准备 Python 依赖包；已安装依赖后，可以完全使用本地模型路径：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python serve_laya.py --model-path /srv/models/laya/english --checkpoint english --device cpu
```

`--model-path` 不下载权重，响应的 revision 记为未知，部署者应通过保存的权重摘要确认版本。这里仅实测了 macOS MPS；行内 CPU、NVIDIA 或其他加速器的性能与兼容性需在目标机器另行验收，不能套用本机延迟。

后续若评估浏览器动作或代码检视，应建立真实业务的候选动作和失败样本集。当前 Mario 上的单一动作现象说明，接入一个更小、更快的 typed-choice 模型，并不能自动获得更好的决策能力；应以动作正确率、任务完成率及误报率验收。
