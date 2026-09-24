# Jev 模型使用与本地复现能力报告

日期：2026-09-20；模型能力实测更新：2026-09-23

2026-09-24 更新：已将默认本地通道换为官方 Laya 英文权重，右栏保留 Jev。新增九局、93 次真实模型决策，没有输入截断或规则接管，但仍未通关且全部选择跑跳。以下保留前期研究与历史基线，Laya 当前实现、局限、实测及行内部署变化见[Laya 接入报告](LAYA_REPORT.md)。

后续模型优化已另完成 33 局、475 次真实决策：本机相同输入下热请求延迟 p50 从 1.344 秒降到 0.953 秒，Qwen 的 1-1 从 x=722 提升到三轮均 x=1411，但 1-3 退步，Jev 与 Qwen 仍无通关。本报告以下保留首轮基线与业务分析；新增候选、负结果、缓存验证及完整证据见[模型优化报告](MODEL_OPTIMIZATION.md)。这些结果不能换算为浏览器或代码检视业务的已实现收益。

## 结论先行

**2026-09-23 已用真实 Jev API 完成关闭路线、guard、自动脱困和规则兜底的 9 局测试，1-1、1-2、1-3 各为 0/3 通关。184 次有效模型回答全部原样执行。** 这衡量当前状态表示与语义动作接口下的控制效果，不能外推为 Jev 在所有控制方式下的能力上限。本机 Gemma4:e2b 的 Ollama 推理服务在 Metal 初始化阶段失败；随后通过独立 PyTorch/MPS 运行时，以已缓存 Qwen2.5-1.5B-Instruct 完成三关各一局，均未通关，54 次真实模型选择全部执行。此前 1-2/1-3 的离线路线成绩退出模型能力验收，仅保留为执行器回归。

这个项目已经把 Jev 的使用方式拆成了三种控制实验：当前状态直接决策、模拟器分支试演后决策、以及把网络延迟放进闭环的实时决策。三者都使用 NES 模拟器 RAM 转换出的结构化状态，模型输出的是动作选择或结构化判断，不是直接看游戏截图。

最有效的原始方案是 `branch.py`：程序先在模拟器里试跑动作和后续动作，把“会不会死、能走多远、后续有多少路能活”压缩成候选结果，再让 Jev 在这些实测结果中选择。这个设计把困难的物理预测交给模拟器，把 Jev 放在更适合它的比较和取舍位置。历史上 1-1、2-1、3-1 都有 Jev 分支通关记录，但 2-1、3-1 的成功录像来自较早版本，当前版本不能稳定复现。

本地复现已经分成两条边界：

- **协议复现**：`play_local.py` / `watch_local.py` 可把同一份状态、动作集合、历史、教训和教练计划发给本地 OpenAI 兼容模型。仓库不包含 Qwen 权重；现已提供 `serve_qwen.py`，加载本地缓存或指定目录中的权重，默认监听 `http://127.0.0.1:11503/v1`。控制器通过 `LOCAL_POLICY_BASE_URL` 连接该服务。
- **行为复现**：`branch.py --bot search`、规则策略和已验证路线不需要模型服务。当前 1-2 和 1-3 都能重复到旗杆，但这证明的是模拟器搜索/路线层，而不是本地模型自主通关。

当前需要验证的工程方案是：模型负责在结构化候选中做决策，环境或工具负责验证后果，安全层负责拦截明显危险；模型评测必须关闭预写路线并单独统计规则改写。对我们的 agent 开发和基础设施，真正可迁移的价值不是“让 Mario 玩得更好”，而是形成一套可审计的决策执行闭环：先把状态压缩成可比较的候选，再让模型选择，再验证、过期检查、回退并记录真实执行结果。

新增参考项目 Laya 提供了另一条本地化路径：把快速的 `choice/score/noul` 判别、语言路由和风险预检放到内网进程，把 Jev 或大模型保留给候选生成、深层分析和不确定场景。它可以降低远程调用和等待，但不能替代 Mario 已验证的 evaluator、guard、executor 和 provenance。

## 1. Jev 的三种使用方式

### 1.1 直接控制：当前状态到一个动作

`play.py` 每 6 个模拟器帧在落地边界做一次决策。`grid()` 从 RAM 读取 Mario 位置、速度、跳跃状态、地形和敌人，`features()` 把网格整理成摘要、墙、坑、敌人、跳跃距离和警告。`ask_jev()` 将这些字段、规则说明和动作字典发给 `jev-latest`，请求一个 `choice` 和概率分布，随后执行该动作。

核心请求形状如下，字段名与 [play.py](play.py#L354-L367) 一致：

```json
{
  "state": {
    "summary": "...",
    "grid": "13 rows x 20 columns ...",
    "wall_ahead": {"height": 3, "tiles": 4},
    "gap_ahead": null,
    "enemy_ahead": null,
    "jump_reach_now": {"height": 4, "far": 3}
  },
  "model": "jev-latest",
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "RULES + GRID_LEGEND",
      "criteria": {"run right": "...", "jump right": "..."}
    }
  }
}
```

认证从环境变量 `TYPESAFE_API_KEY` 读取，请求发往 `https://api.typesafe.ai/v1/systemone`。模型只看到结构化文字；RAM 地址、动作含义和跳跃物理由代码和提示词提供。

### 1.2 分支试演：先问模拟器，再问 Jev

`branch.py` 是原仓库最有价值的实验设计。每次决策会：

1. 在快照上执行候选动作约 60 帧；跳跃会执行到落地，并额外稳定约 8 帧。
2. 对候选动作之后的 6×6 个后续动作组合继续试演。
3. 对每个候选计算死亡、是否到旗、位移、是否落地、存活后续数和最佳后续路径。
4. 把这些实测结果转成文字，例如“存活、前进若干格、36 条后续中有多少条存活、最佳后续是什么”。
5. 让 Jev 只在这些已测量的结果中选择；恢复快照后，按同一动作时序执行。

关键实现见 [branch.py](branch.py#L25-L157) 和 [branch.py](branch.py#L213-L295)。如果所有候选都没有正收益，`escape()` 会在有限动作集上做三步搜索。`--bot search` 使用相同模拟结果，但用 `(到旗、存活、非死路、距离、后续数)` 的确定性排序替代 Jev，适合离线验证。

这个方案的含义是：Jev 没有被要求从零学习 Mario 物理，而是在模拟器已经回答“这样做会发生什么”之后做选择。它更接近一个结构化决策器，而不是端到端视觉控制器。

### 1.3 实时控制：把网络延迟当作环境的一部分

`live.py` 同时请求三个判断：动作 `choice`、是否现在开始或继续跳跃的 `noul`、以及下一秒危险度 `score`。代码将三者合并；高置信度的跳跃判断可以覆盖普通动作选择。请求飞行期间模拟器继续运行，保持上一动作，等响应回来再应用，因此模型面对的是带延迟的闭环，而不是暂停的环境。

实时状态额外包含预计延迟帧数、每次决策盲走距离、预计敌人接触时间和“本次必须起跳”的警告。见 [live.py](live.py#L45-L117) 和 [live.py](live.py#L141-L181)。这个设计更接近真实 agent，但也更依赖低延迟和准确的前向估计。

## 2. 原仓库实验结果

`runs/results.jsonl` 保存了原仓库的逐次结果。下面的数字是历史记录，不能理解为每次运行都稳定复现的保证。

| 控制方式 | 关卡 | 代表结果 | Jev 请求 | 中位延迟 | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| 直接 `play.py` | 1-1 | x=686，未到旗 | 8 | 约 0.24 s | 每 6 帧决策，暂停模拟器等待请求 |
| 直接 `play.py` | 2-1 | x=473，未到旗 | 5 | 约 0.23 s | 物理窗口很快错过 |
| 直接 `play.py` | 3-1 | 最好 x=841，未到旗 | 9–15 | 约 0.20–0.24 s | 三次结果波动 |
| `live.py` | 1-1 | x=315，未到旗 | 8–10 | 约 0.18–0.22 s | 模拟器不停，延迟直接造成状态过期 |
| `live.py` | 2-1 | x=457，未到旗 | 14–19 | 约 0.16–0.20 s | 与规则延迟双胞胎接近 |
| `live.py` | 3-1 | 最好 x=596，未到旗 | 12–20 | 约 0.17–0.21 s | 延迟和跳跃窗口叠加 |
| `branch.py --bot jev` 历史记录 | 1-1 | x=3161，通旗 | 23–30 | 约 0.37–0.40 s | 录像对应 `20260918-170421` 等记录 |
| `branch.py --bot jev` 历史记录 | 2-1 | x=3193，通旗 | 29 | 约 0.40 s | 成功录像来自早期动作集 |
| `branch.py --bot jev` 历史记录 | 3-1 | x=3193，通旗 | 25 | 约 0.40 s | 成功录像来自早期动作集 |

旗杆大约在 x=3160。原 README 同时记录了当前构建的失败或停滞：2-1 约 x=2066，3-1 约 x=2770。差异来自动作集合和试演结果的变化，说明这套系统对候选动作、恢复动作和微妙帧时序很敏感。

直接控制和实时控制的结果说明，模型调用次数少并不等于系统效率高：一次错误决策可能直接消耗一条命，而 0.2 秒级延迟足以让跳跃窗口失效。分支试演牺牲了时间，把大量计算换成了更可靠的局部前瞻。

按 `runs/results.jsonl` 的全量历史记录聚合，直接 Jev 的 15 次运行全部未到旗，平均每局 8.13 次请求、12,003 个输入 token、约 $0.000504、p50 延迟约 0.228 秒；实时 Jev 的 9 次运行也全部未到旗，平均每局 13.33 次请求、15,904 个输入 token、约 $0.000667、p50 延迟约 0.194 秒。分支 Jev 的混合历史记录为 9 次中 5 次到旗，平均每局 26.2 次请求、45,031 个输入 token、约 $0.001892、p50 延迟约 0.391 秒，但其中 2-1、3-1 的成功来自旧动作集，不能作为当前版本成功率。

这里的成本只是仓库 `cost_usd` 的记录口径：按 `$0.042 / 1M` 乘以 API 返回的输入 token 估算，没有包含输出 token、本地推理、模拟器计算和墙钟时间。分支模式真正的主要成本是 15–20 分钟的模拟器试演，而不是 API 账单。

## 3. 当前本地复现实现

### 3.1 本地模型通道

`play_local.py` 复用了同一套 RAM 网格和动作字典，支持两种本地接口：

- `/v1/chat/completions`：模型从动作字母中选择一个动作。
- `/v1/score`：对所有合法字母打分，再转成候选动作概率。

本地状态还会带上最近决策、已验证经验、过去失败教训和教练计划。`watch_local.py` 默认同步决策，模型返回期间暂停对应模拟器；`--async` 才会让模拟器继续前进，因此同步模式是当前稳定实验路径。异步模式会丢弃过期结果，并检查结果对应的帧和 x 位置，避免把旧状态动作灌进当前障碍。

复现入口按证据强度区分：

```bash
# 实测模型独立选择动作；不启用 guard/unstick/route，错误直接记录并退出
uv run python play_local.py --bot jev --model-only --level 1-1
uv run python play_local.py --bot local --model-only --level 1-1

# 无模型，复现搜索路线和观战执行器
uv run python supervise.py --level 1-3 --replay-only --tunnel none

# 接本地 OpenAI 兼容服务；实际动作仍会经过 guard/unstick
uv run python supervise.py --level 1-1 --model local --tunnel none
uv run python play_local.py --bot local --level 1-1

# 接官方 Jev；需要 TYPESAFE_API_KEY，会产生远程请求和费用
uv run python supervise.py --level 1-1 --model jev --tunnel none
uv run python branch.py --bot jev --level 1-1
```

viewer 的普通模式可能退回规则策略；`play_local.py --model-only` 对服务错误或非完整单字母回答直接报错，保留错误结果与已有日志，不执行规则兜底。`model_choice` 必须与 `executed_choice` 对照，不能只看最终 `flag`。

### 3.2 安全、教练和恢复层

本地复现并不是把 Jev API 换成本地地址这么简单。当前执行链分为：

1. 模型或规则提出候选动作。
2. `hazard_guard()` 检查坑、墙、低天花板、敌人和落点风险，必要时改成跳跃或提前跑动。
3. `unstick()` 处理同一位置重复动作。
4. 连续失败的场景交给教练模型，解析成 `REFLECTION` 和合法 `PLAN`，可执行等待、后撤和后续跳跃。
5. 已登记的验证路线在三局失败后或显式开启时接管动作；模型原始选择仍保留在 telemetry，但不再决定物理输入。

因此，本地 lane 的“通关”必须拆解成 `model_choice`、`source`、`mentor_trace`、`verified_route_used` 和 `model_decision_sources`。当前代码已经把模型回答、规则兜底、离线回放分开记录；`calls` 是请求尝试数，不代表成功回答数。

### 3.3 当前模型能力实测

2026-09-23 使用同一版 `play_local.py --model-only`，没有修改提示词，首轮后补测两轮，保留全部结果。后两轮并发运行，因此延迟只描述本次系统表现，不是隔离推理性能基准。

| 关卡 | Jev 通关 | 三次最远 x | 有效回答数合计 |
| --- | --- | --- | ---: |
| 1-1 | 0/3 | 1435 / 679 / 2471 | 95 |
| 1-2 | 0/3 | 198 / 655 / 960 | 69 |
| 1-3 | 0/3 | 284 / 315 / 315 | 20 |

184 次有效回答都直接执行，guard 和 unstick 改写均为 0。每局 API 延迟 p50 为 0.319–0.603 秒。实验暂停模拟器等待模型；模型选择九种语义动作，执行器负责按键持续、跳跃到落地及后退。样本量有限，当前结果说明这套接口尚未通过自主通关验收，不能断言更换状态表示、动作时序或候选试演后仍无法通关。

首轮逐帧诊断揭示三个接口问题：

- **1-1：决策窗口被跳过。** 最后请求在 f624/x1357，摘要距坑 2 格，模型选跑，提示却要求距坑 1 格才跳。六帧后 x1375 已离地，运行器等待落地直到死亡，没有再次询问模型。应向模型暴露像素距离、速度及“下一次观察前是否还可起跳”。
- **1-2：候选描述未覆盖连续敌人。** 最后选择低跳，踩扁首只 Goomba 后与紧邻的第二只接触死亡。低顶与落点警告主要描述全跳，未充分描述低跳和踩敌反弹后果。
- **1-3：摘要把观测范围当作安全结论。** 特征仅扫描前 8 格，grid 的断层在第 9 格，摘要仍为 `Solid ground ahead`。模型过早全跳，下降时撞到更远平台侧面，跳跃期间没有重新决策。

诊断回放只用于解释已发生的失败，不新增模型样本。这些结果要求同时评估模型、状态抽取与执行接口。对浏览器 agent，下一次观察前的状态有效期和长操作中断点应纳入契约；对代码检视，“扫描范围内未发现”不能概括为“没有风险”。尚未取得真实业务的效率收益数据。

本地服务初始三次请求均 HTTP 500，零有效回答；随后 4096 上下文 alias 和 `num_gpu=0` alias 的最小请求也均失败。观测根因为 Apple M5 / Ollama 0.24.0 的 Metal `bfloat/half` 编译类型不匹配，不应写成模型游戏失败或未经证实的内存不足。需要通过目标硬件上的推理健康检查后，才能公平比较本地模型与 Jev。

本次还重新运行原始在线分支 Jev 的 1-1：未通关，x=2370，22 次模型请求，实际执行 2544 帧，候选试演 1,574,417 帧，墙钟 1642.901 秒（27.38 分钟），API p50 0.624 秒。没有预写路线或 escape 接管，但有 11 次自动 ride 续招。它的动作集包含额外复合动作，因此不能作为只改变“有无试演”的配对实验。历史 branch-Jev 成功仍是历史材料，这次没有复现通关。

额外模拟量约为实际执行帧的 619 倍。对 agent 基础设施，这是重要的成本约束：候选验证本身可能成为主要瓶颈，不能只优化模型延迟。浏览器应优先验证 DOM 条件、动作可用性和后置断言，代码检视应先做增量静态检查，再对少量高价值候选运行测试；不能直接照搬 Mario 的全量分支展开并假定会加速业务。

本地 Qwen 的真实基线如下：

| 模型 | 关卡 | 通关 | 最远 x | 模型回答 | 实际帧 | 推理 p50 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B-Instruct | 1-1 | 0/1 | 722 | 26 | 753 | 0.826 s |
| 同上 | 1-2 | 0/1 | 656 | 16 | 468 | 0.822 s |
| 同上 | 1-3 | 0/1 | 618 | 12 | 359 | 0.802 s |

权重 revision 为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，运行时 PyTorch 2.14.0 / Transformers 4.57.6，MPS FP16，复用本机缓存，未下载权重。通过 localhost adapter 将项目原提示交给真实模型生成，并非模拟响应。54 次输出全部为 `H / hop right`，已收到 `STUCK` 提示仍不改变选择，说明在此配置下缺少有效的状态适应。历史展示标为 Qwen2.5-3B，与本次 1.5B 模型不能混用。模型加载约 9.83 秒，三局连同加载共约 76.47 秒。

**本轮不支持“本地模型必然更快”的判断。** 本地 Qwen 的约 0.8 秒推理 p50 高于这次 Jev 的约 0.3–0.6 秒 API p50，且两者提示编码、输入 token 数和样本量不同，不能视为公平的纯模型速度排名。本地化在行内的可验证收益首先是数据驻留、版本可控、推理链路可审计；响应速度和有效任务成本仍要在业务硬件上测量。

完整实测产物见发布工作区 `artifacts/model-eval-20260923/` 的汇总、HTTP 请求响应、模型版本标识、逐决策日志和录像；公开后从项目首页访问。HTTP 证据不含认证头。运行结果中的 `api_calls` 当前计有效回答，HTTP 500 等失败请求另行统计。

#### 执行器回归与历史材料

| 结果 | 方法 | 重复证据 | 模型请求 |
| --- | --- | ---: | ---: |
| 1-2，x=3161，2300 实际模拟器帧 | 23 步验证路线 | 2/2 | 0 |
| 1-3，x=2425，2430 实际模拟器帧 | 原版 `branch.py --bot search` 找路线，再由 viewer 回放 | 2/2 | 0 |
| 1-1，x=3161 | 历史本地 Qwen 与 branch-Jev 记录 | 缺少对应逐决策归因，未作为当前验收 | 有请求汇总 |

离线回放使用 `--replay-only`，会关闭公网隧道、跳过模型/教练/搜索请求，并在页面标记“未调用模型”。验收命令是：

```bash
uv sync --frozen
uv run python verify_routes.py --repeats 2 --gif
uv run python -m unittest discover -s tests -v
```

当前报告对应的本地验收记录在 [artifacts/current/offline-acceptance.json](artifacts/current/offline-acceptance.json)。1-2、1-3 的路线成功证明了模拟器搜索和精确时序可以被复现；它们不证明本地 Qwen 或 Jev 在没有路线接管时能自主完成关卡。

已有的 1-2 原始录制也要按来源阅读：`artifacts/source/raw-1-2-local.json` 和 `raw-1-2-jev.json` 的路线动作都由 verified route 执行，Jev 的 23 步均标为 fallback，本地的 23 步虽标为 model，也均未控制实际执行。因此它们是双 lane 系统验收材料，不是两个模型各自独立清关的材料。

要复现“模型协议”而不是仅复现路线，需要额外的服务：官方通道配置 `TYPESAFE_API_KEY`，本地通道启动一个兼容 `/v1/chat/completions` 或 `/v1/score` 的服务，并配置 `.env` 中的 `LOCAL_POLICY_BASE_URL`。仓库没有 Jev 权重，也没有本地 Qwen 权重，因此 `uv sync --frozen` 只能准备模拟器和客户端依赖，不能离线启动 Jev 本体。

### 3.4 本地模型拿到内网后的部署方案

本轮 M5/Ollama 故障说明：模型权重和 OpenAI 兼容接口可用，不等于目标硬件能完成推理。进入行内部署前，先固定硬件、操作系统、驱动/加速库、推理服务和权重摘要，依次测模型加载、单次真实生成、业务长度状态提示及持续请求。健康检查应包含真实推理；只检查进程存活或 `/models` 列表会漏掉这次的故障。macOS 开发机问题与 Linux/NVIDIA 行内服务器的可用性应分别验证。

如果把 Qwen 或其他策略模型拿到公司内网，推荐把它作为一个独立的 OpenAI 兼容推理服务，Mario agent 只访问内网的 `/v1` 地址：

```text
play_local.py / watch_local.py
        │  chat 请求（state → 9 个字母）
        ▼
内网 policy gateway（鉴权、限流、日志、模型路由）
        ▼
Ollama / LM Studio / vLLM / llama.cpp server
        ▼
固定版本的 Qwen2.5-3B-Instruct 或同等策略模型
```

本仓库不包含模型权重，`uv sync --frozen` 不会下载权重。新增的 `serve_qwen.py` 是本地实验适配器，读取已准备好的 Qwen 权重并执行真实生成；双栏命令见 [README](README.md#双栏实时观战)。权重应从公司批准的模型仓库取得，记录版本或 SHA，按许可证使用；生产环境不要在运行时从任意公网地址拉取模型。生产遥测应记录模型版本、延迟、状态版本和是否 fallback，但日志要过滤用户数据和密钥；当前代码已记录来源和延迟，贯穿全链路的固定模型版本、状态 hash、request id 仍是待补的基础设施字段。

部署 manifest 还应固定模型 revision、tokenizer、推理 runtime 版本以及 CUDA/Metal 驱动版本。只固定 `served name` 不足以保证决策一致；量化、采样默认值或 tokenizer 变化都可能改变字母选择。

当前客户端支持的配置如下。`LOCAL_POLICY_MODEL` 是本次补上的配置项，避免服务必须把真实模型名伪装成 `local`：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `LOCAL_POLICY_BASE_URL` | `http://127.0.0.1:11500/v1` | OpenAI 兼容服务根地址 |
| `LOCAL_POLICY_API_KEY` | `local` | 发给内网网关的 Bearer token；不要提交真实密钥 |
| `LOCAL_POLICY_MODEL` | `local` | chat 请求中的模型名，必须与服务端的 served name/模型 ID 匹配；当前 score 请求不携带此字段 |
| `LOCAL_POLICY_MODE` | `chat` | `chat` 使用 `/v1/chat/completions`；`score` 使用 `/v1/score` |
| `LOCAL_POLICY_TIMEOUT` | `30` 秒 | 本地策略单次请求超时；超时后记录错误并走规则兜底 |
| `LOCAL_POLICY_TEMPERATURE` | `0.5` | 仅影响 score 模式将分数转概率的温度；chat 请求固定为 0 |

部署容量可以先按下面的粗略档位规划；量化方式、上下文长度、并发数和运行时会改变实际占用，最终以压测数据为准：

| 策略模型 | 开发机建议 | 共享服务建议 | 用途 |
| --- | --- | --- | --- |
| Qwen2.5-3B，4-bit | 8 GB 可启动，16 GB 更稳 | 6–8 GB VRAM 或等量统一内存 | 默认逐步策略、浏览器可逆动作 |
| 7B，4-bit | 16 GB 内存起 | 10–12 GB VRAM 起 | 复杂页面、代码 review 候选排序 |
| 14B，4-bit | 32 GB 内存起 | 20–24 GB VRAM 起 | 教练、低频深度分析，不放逐帧路径 |

这些是容量规划量级，不是成功或延迟保证。Mario 的 6 帧决策窗口尤其不适合等待大模型；实际业务应把快模型用于常规动作，把深模型或分支试演留给高风险候选。

#### 开发机：macOS / Apple Silicon

Ollama 是最短的本地复现路径，适合单人开发和提示词调试。示例把模型复制成客户端使用的稳定别名：

```bash
brew install ollama
ollama serve
ollama pull qwen2.5:3b
ollama cp qwen2.5:3b jev-local

cp .env.example .env
# .env
LOCAL_POLICY_BASE_URL=http://127.0.0.1:11434/v1
LOCAL_POLICY_API_KEY=ollama
LOCAL_POLICY_MODEL=jev-local
LOCAL_POLICY_MODE=chat
LOCAL_POLICY_TIMEOUT=30

uv run python play_local.py --bot local --level 1-1
```

LM Studio 也可以使用同一客户端：在 Local Server 中加载 Qwen2.5-3B-Instruct，监听 `127.0.0.1:1234`，把 `LOCAL_POLICY_BASE_URL` 改成 `http://127.0.0.1:1234/v1`，`LOCAL_POLICY_MODEL` 填 UI 显示的模型 ID。开发机建议使用 4-bit 量化并把上下文控制在 4K–8K；16 GB 内存通常比 8 GB 更适合同时运行模拟器、浏览器和模型服务。实际延迟以本机 p50/p95 测量为准，不能用 Mario 历史的远程延迟直接推断。

#### GPU 服务器：Linux / NVIDIA

团队共享服务优先用 vLLM，并把模型名固定为 `jev-local`。推理服务应运行在单独的 Python/容器环境，不要强行塞进 Mario 项目的 `uv` 环境：

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --served-model-name jev-local \
  --host 127.0.0.1 --port 8000 \
  --dtype auto --max-model-len 4096

# 同机 gateway/agent 的 .env；跨机器时让 gateway 代理这个 localhost 服务，
# 不要直接把 vLLM 端口暴露到公网。
LOCAL_POLICY_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_POLICY_API_KEY=<由内网网关注入>
LOCAL_POLICY_MODEL=jev-local
LOCAL_POLICY_MODE=chat
LOCAL_POLICY_TIMEOUT=30
```

上面的 vLLM 命令加载的是模型仓库提供的原始权重，显存应按 BF16/FP16 重新估算；若要使用容量表中的 4-bit 档位，应换成经过批准的 AWQ/GPTQ 等量化权重，并按该 runtime 的量化参数启动。3B 模型可作为低成本策略模型；如果升级到 7B/14B，应重新测量首 token、完整响应和并发队列，而不是只看显存能否装下。Mario 的同步决策会等待响应，浏览器和代码 review 则应分别设置交互延迟预算，并在超时后走规则或人工确认路径。当前客户端是单一请求超时，后续可按连接、读取和队列等待拆分更细的预算。

#### CPU 或边缘节点：llama.cpp

没有 NVIDIA GPU 时，可以使用 GGUF 量化模型和 `llama-server`。不同版本对模型别名参数名称可能略有差异，关键是让服务暴露 OpenAI 兼容的 `/v1/chat/completions`，并把实际模型 ID 填回 `LOCAL_POLICY_MODEL`：

```bash
llama-server \
  -m /models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 4096 --alias jev-local

LOCAL_POLICY_BASE_URL=http://127.0.0.1:8080/v1
LOCAL_POLICY_MODEL=jev-local
LOCAL_POLICY_MODE=chat
```

CPU 模式适合离线回放、低并发工具调用和开发验证，不适合把每个高频浏览器事件都同步阻塞在模型上。应先做批量/事件合并，或只在高风险候选出现时调用模型。

#### 内网生产：网关、鉴权和灰度

内网服务前建议放一个 OpenAI 兼容 gateway，统一做 API key、TLS、租户限流、模型路由、超时、重试预算和审计。模型服务只绑定 localhost 或受控内网网卡，不直接暴露公网；Docker 部署应固定镜像 digest，并把模型目录只读挂载。上线顺序建议是：

1. **shadow**：只记录模型选择，不执行动作，和规则策略对比合法率、延迟和选择差异。
2. **canary**：只放给可回滚的浏览器导航、筛选和只读代码 review，保留 `rules_fallback`、guard 和人工接管。
3. **扩容**：确认 p95 延迟、队列长度、GPU/内存占用和错误率后，再增加并发或更大模型。

浏览器提交、代码写入、生产变更、付款、删除和发送消息仍必须经过动作 guard、freshness 检查、幂等/回滚策略和人工审批。模型部署到内网只解决数据边界和推理可用性，不会自动获得这些安全能力。

`.env.example` 中 `LOCAL_POLICY_API_KEY=local` 只是开发占位值；当前客户端不会因为这个默认值自动拒绝启动，生产 gateway 必须拒绝默认 key、要求真实凭据并限制来源。另一个独立数据出口是 `MENTOR_BASE_URL`：教练和可选网页检索可能把失败现场、网格和历史发送到外部服务。若要求全链路内网，应关闭 mentor/web search，或把它们改成经过脱敏和审批的内网 endpoint。

#### `/score` 的兼容边界和上线检查

`score` 模式不是标准 OpenAI API；本仓库期待响应包含 `items[0].scores` 和 `items[0].best`，并据此计算九个动作的概率。Ollama、LM Studio、vLLM 和大多数 llama.cpp 服务只保证 `/v1/chat/completions`，因此默认使用 `LOCAL_POLICY_MODE=chat`。服务不支持 `/v1/score` 时不要伪造分数，也不要把单个 chat 字母重复当成概率分布。客户端当前不使用 streaming，且严格读取 `choices[0].message.content`，所以服务必须支持非流式 chat、`max_tokens` 和 `temperature` 字段。

上线前至少执行一次健康检查和一次真实客户端请求：

```bash
set -a; source .env; set +a
curl -fsS "$LOCAL_POLICY_BASE_URL/models" \
  -H "Authorization: Bearer $LOCAL_POLICY_API_KEY"
curl -fsS "$LOCAL_POLICY_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $LOCAL_POLICY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"jev-local","messages":[{"role":"user","content":"Reply with A only."}],"max_tokens":8,"temperature":0}'
uv run python play_local.py --bot local --level 1-1
```

验收不能只看 `flag=true`。必须同时检查 `model_decision_sources` 是否出现 `model`、fallback 比例、请求 p50/p95、首 token/完整响应延迟、队列长度、GPU/内存占用、guard 改写次数、过期动作丢弃数，以及最终实际执行动作。服务不可用或输出无法解析时，系统会回到规则策略；这应被记录为可观测的降级，而不是算作本地模型成功。

## 4. Laya 参考项目分析

本次将 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) 克隆到同级 `laya-reference` 目录，分析基于 commit `d113dca2512fb3eaca313534bc54c7162d87c1d4`（v0.3.4）。项目代码采用 Apache-2.0 许可证；模型权重、训练数据和 Hugging Face model card 需要单独核验许可，不能由 GitHub 代码许可证推断可商用。Laya 仓库不包含模型权重；权重默认从 Hugging Face 下载，也支持本地目录加载。

### 4.1 它解决的问题

Laya 是一个本地、非自回归的 typed decision engine。它把 `text / email / ticket / JSON state` 与一组结构化问题一起编码，在一次 encoder forward 中同时回答多个问题，输出三种结果：

| 类型 | 输出 | 适合的 agent 用途 |
| --- | --- | --- |
| `choice` | 选项、各选项概率、confidence | 路由、意图、下一步工具类别 |
| `score` | 有序等级的期望分数、分布、confidence | 紧急度、风险、影响范围 |
| `noul` | `P(true)`、confidence | 是否需要人工、是否敏感、是否注入攻击 |

它不生成自然语言，因此没有“生成一段话再解析字母”的额外失败面。源码中的 `build_sequence()` 把问题类型、说明、选项和状态编码到固定序列，`Agent.system_one()` 把多个问题 batch 后一次前向推理，再按温度参数输出概率。这和当前 `play_local.py` 的 chat 模式有本质差异：当前本地 Mario lane 仍要让生成式模型返回一个字母；Laya 可以直接返回结构化 choice。

需要避免一个命名误解：模型内部虽然有 `act_head`，运行时返回的 `action` 目前只有 `act_probability`，并没有浏览器点击、代码修改或 joypad 动作。typed-decisions 微调路径也没有把它训练成可执行动作头。因此 Laya 的 `choice/score/noul` 是判别结果和风险信号，不是执行授权；真正的动作仍由上层 Action Registry、guard、审批和 executor 产生。

项目提供三个 checkpoint：英文 `ModernBERT-large`（约 421M 参数，512 context）、多语言 `mmBERT-base`（约 322M，1,024 context）和面向 typed-decisions 工作流微调的英文 checkpoint（约 421M，1,024 context）。`Router` 先做 Unicode script 和少量 Latin stopword 语言检测，再选择 checkpoint；显式 `model`、`task`、`lang` 优先于自动路由，typed workflow 自动识别默认关闭。

### 4.2 对本地部署的价值

Laya 是可嵌入 Python 进程的模型运行时，而不是现成的 HTTP 服务。核心依赖是 `torch`、`transformers`、`safetensors`、`huggingface_hub` 和 `numpy`；`laya.load()` 支持 `device="cuda"`、`device="mps"`、CPU，以及显存不足时回退 CPU。生产进程应在启动时预加载模型：

```python
from laya import Router

router = Router(preload=True, device="cuda")
result = router.predict(state, questions)
```

内网落地时应先把权重从批准的模型仓库预置到本地目录，再用 `Router(models=...)` 指向这些目录；不要让业务请求触发公网下载。示意流程如下：

```bash
python -m venv /srv/laya-venv
/srv/laya-venv/bin/pip install -e /srv/src/laya-reference
# 由模型仓库管理员将三个 checkpoint 放到：
# /srv/models/laya /srv/models/laya-multilingual /srv/models/laya-typed-decisions
```

```python
from laya import Router

router = Router(
    models={
        "english": "/srv/models/laya",
        "multilingual": "/srv/models/laya-multilingual",
        "typed-decisions": "/srv/models/laya-typed-decisions",
    },
    preload=True,
    device="cuda",
)
```

Laya 本身没有 OpenAI-compatible HTTP server；若要接入现有 policy gateway，需要自行封装一个只接收 `state + questions`、返回 typed JSON 的 sidecar，并将模型加载、鉴权、超时、并发队列和版本信息放在服务层。一次 forward 是同一 state 下多个问题并行，并不等于跨请求自动 batching；高并发仍需 gateway 自己做 micro-batching 或队列。

`Router(max_loaded=1)` 默认使用 LRU，只保留一个 checkpoint；中英文流量交替时会触发重新加载。项目 README 报告 CPU reload 中位数约 7.4 秒、T4 约 10.3 秒，而预加载后 T4 单题约 32.8 ms、CPU 约 193–464 ms。内网服务至少要固定模型 revision、tokenizer、推理 runtime 和 CUDA/Metal 版本，并在 gateway 或 sidecar 外面补鉴权、限流、健康检查和审计。若已有进程加载了模型，可以用 `Router.attach()` 避免重复占用显存。

接入时要统一 `RouteDecision` 的遥测 schema：普通路由返回的 `repo` 是字符串，而 typed workflow 自动路由分支可能保留 `(repo, subfolder)` 元组。适配层应把它规范成稳定的 `model`、`repo`、`model_version`、`reason` 和 `detection` 字段，避免下游日志/指标出现两种类型。

Laya 的 benchmark 声称 T4 单题 32.8 ms、多题批处理最高 103–332 questions/sec；README 同时明确 Jev 的 236–276 ms 数据来自第三方发布结果，样本、提示和运行环境不同。本报告把这些作为部署方向参考，不把它们当成我们业务的 SLA，也没有把 Laya 权重下载到当前工作区运行真实模型。

工程治理也值得借鉴：仓库的 Security workflow 包含 gitleaks、pip-audit、CodeQL，并检查模型加载路径不使用 pickle/`torch.load`/动态 shell 执行。对我们的 agent gateway，这对应模型供应链、依赖漏洞和工具执行边界三类基础门禁。

本次对源码自带的 `tests/test_router.py`（98 项）和 `tests/test_criteria.py`（34 项）做了本地运行，均通过；`tests/test_local_e2e.py` 需要 `~/laya_models` 中的三套真实权重，因此本次没有宣称端到端模型结果。

### 4.3 与 Jev 和当前 Mario 栈的关系

| 层次 | Laya | Jev / 当前 Mario | 迁移判断 |
| --- | --- | --- | --- |
| 决策表示 | 本地 encoder 直接输出 `choice/score/noul` | Jev API 返回 typed choice/score/noul；当前本地 chat 还要解析字母 | Laya 可替换低延迟的本地 policy/guard 层 |
| 多问题处理 | 一次 forward 并行回答多个问题 | 请求也能带多个 typed questions，但本地 lane 目前逐步取一个动作 | Laya 适合一次判断风险、紧急度、路由和人工确认 |
| 候选后果 | 不执行工具，也不试演环境 | `branch.py` 用快照试演候选，路线层可执行复合动作 | Laya 不能替代 evaluator、snapshot、route 或 executor |
| 语言路由 | script/语言检测后选模型，并记录 reason | 当前 Mario 没有同等的本地模型路由 | 可作为多语言 agent 的前置路由器 |
| 置信度 | choice 概率、score 分布、noul 概率 | Jev 概率来自 API；当前 chat 模式无原生概率 | 仍必须用 calibration、guard 和人工门，不可只看 confidence |
| 部署边界 | 本地权重、进程内推理、可 MPS/CUDA/CPU | Jev 依赖远程 API；本地 Mario 依赖外部兼容服务 | Laya 适合内网 fast path，Jev/大模型做深层候选分析 |

Laya README 报告面向 typed-decisions 微调后的 checkpoint 在其数据上为 0.766，并引用 Jev 0.727；它还报告 77 类 Banking77 为 0.425，而引用的 Jev 为 0.870，原因是选项共享固定 token budget。这些数字来自不同样本、提示和运行环境，不能作为同条件优劣结论。其零样本 typed-decisions 基础模型约 0.34–0.36，低于多数类基线 0.461；0.766 主要来自领域微调。英文模型在非英文脚本上可能高置信错误（README 以 Khmer 为例），所以必须先路由，不能把 confidence threshold 当成语言识别器。

这给当前项目三个明确启发：

1. **把 Laya 放在快速判别层。** 浏览器状态可以同时问 `next_action`、`risk`、`needs_confirmation`、`state_fresh`；代码 review 可以同时问 `severity`、`needs_review`、`security_impact`、`next_probe`。Laya 只产生结构化建议，后续仍由 guard、候选验证和 executor 决定是否执行。
2. **用 Laya 做模型路由。** `router_questions()` 已提供 difficulty、domain、needs_tools、is_sensitive，可先判断请求是否需要浏览器/代码工具、是否进入 Jev/大模型深思路径，再把普通任务交给更快的本地模型。这样能减少每个简单动作都调用远程模型。
3. **把 domain 微调变成可测量资产。** 现有 notebook 用 RLCD（log、spherical、ranked probability scoring）和温度校准训练 typed decisions。我们可以用历史浏览器轨迹、review 结论、guard 审批和实际后果生成内部 decision corpus，按 held-out 场景评估 accuracy、Brier/ECE、option-order 稳定性和 fallback，而不是只看最终任务成功率。

### 4.4 适合的业务接入方式

推荐的组合架构是：

```text
State Envelope（DOM/AX、diff、测试、版本、freshness）
        ↓
Laya Router / typed pre-check（快、本地、可批量）
        ↓
Jev / 大模型 / branch evaluator 生成或比较候选
        ↓
Guard + 权限 + dry-run / worktree / clone tab 验证
        ↓
Executor → Provenance / outcome / recovery
```

浏览器自动化优先用 Laya 做页面语言路由、危险动作预判、表单提交前检查和“是否需要确认”；付款、删除、发送消息等不可逆动作仍要人工批准。代码检视可用 Laya 快速给 diff 分级、选择下一类探针，再让 CodeGraph、测试和临时 worktree 提供事实证据。对用户体验，Laya 可以低延迟回答“是否过期”“是否需要重读页面”“是否升级人工”，减少等待远程模型的次数。

不建议直接把 Laya 的高置信度当作执行许可：项目自己的 benchmark 说明基础模型出厂概率可能过度自信，`score` 是较弱的 primitive，高基数 `choice` 需要提高 `head_max_len` 或拆成 coarse-to-fine。生产部署要保留低置信度回退、状态新鲜度检查、风险 guard 和审计 provenance。

### 4.5 纳入当前路线图

- **P0：** 定义统一 `DecisionBroker`，同时适配 `rules`、Laya、本地 chat、Jev；统一输出 `choice/probabilities/confidence/source/model_version`。
- **P1：** 在浏览器只读/可撤销流程和代码 review 只读探针中，用 Laya 做 shadow 路由与风险预检，比较远程调用数、延迟、过期动作和人工接管。
- **P2：** 用内部轨迹微调 typed-decisions checkpoint；对超过 20 个选项的任务使用层级分类或分桶，避免固定 head budget 造成退化。
- **P3：** 将 Laya 的快速 pre-check 与 Jev 的候选后果比较组合起来：常规低风险动作走本地 fast path，高风险或不确定场景进入 branch/深模型，并用同一套 `GuardResult`、`ExecutionResult`、`TraceEvent` 评估。

## 5. 能力归因：哪些是 Jev，哪些是系统

| 能力 | 主要来源 | 评价 |
| --- | --- | --- |
| 读取画面中的地形/敌人 | RAM 地址、tile parser、enemy map | 代码确定，不是 Jev 的视觉能力 |
| 跳跃距离和碰撞后果 | NES 模拟器实际执行、快照恢复 | 代码确定，分支模式可直接测量 |
| 当前候选的比较和取舍 | Jev `choice`；分支模式中尤其明显 | 是 Jev 最清楚的贡献点 |
| 长时序规划 | 分支后续搜索、escape、验证路线 | 主要是搜索/工程层，模型只参与部分选择 |
| 对危险的快速拦截 | `hazard_guard`、`unstick` | 工程安全层，降低模型失误代价 |
| 根据失败形成新策略 | mentor 反思、lessons、experience | 由外部教练和跨局状态管理提供 |
| 真实延迟下的控制 | `live.py` 延迟模拟、同步/异步策略 | 取决于系统设计和服务延迟，不是单独模型能力 |

还有一个容易被忽略的契约差异：模型通常只能从 9 个简单动作中选，而分支路线还可以执行 `run then jump`、`short run then jump`、弹簧反弹和 frame-exact 的后撤跳跃等复合动作。这些宏动作的表达能力高于模型表面的动作空间，所以路线成功与模型选择之间天然存在信息鸿沟。若要认真复现 Jev，应把复合宏纳入统一的候选 API，或者让模型直接选择分支试演结果，而不是由路线层私下补出模型没有机会选择的动作。

这说明“本地复现 Jev 能力”有两个不同目标：

1. **复现协议能力**：让本地模型在相同状态和动作契约上给出类似的动作选择，并比较它与 Jev 的选择、概率、延迟和失败模式。
2. **复现任务结果**：让整个 agent 到达旗杆。这个目标可以用搜索、守卫和路线达成，但不能直接作为模型能力指标。

当前系统已经较好地完成了第二个目标的工程闭环，第一目标仍缺少严格的逐决策对齐实验。

本地模型难以直接复现 Jev 的原因有三层：第一，权重和服务行为不可得；第二，原始 Jev 的优势主要出现在 branch 的候选结果比较，而普通本地 lane 默认面对当前状态；第三，最终成功记录里混入了守卫、教练和验证路线。要证明“复现了 Jev”，必须固定候选状态集，逐条比较模型原始 choice，再单独报告守卫和路线后的最终成绩。

可以把当前能力分成六级：

| 等级 | 能力 | 当前状态 |
| --- | --- | --- |
| L0 | RAM、网格和物理特征抽取 | 已稳定复现 |
| L1 | 外部 Jev choice API 接入 | 已接通；直接/实时模式尚未自主通关 |
| L2 | 模拟器分支前瞻 + Jev 选择 | 历史 1-1、2-1、3-1 有成功；动作集变化后当前版本不稳定 |
| L3 | 守卫、脱困、教练和跨局教训 | 已实现，能减少重复卡死，但会遮蔽模型原始能力 |
| L4 | 搜索轨迹编译成验证路线 | 1-2、1-3 已重复稳定回放 |
| L5 | 纯模型长程通关和跨关泛化 | 尚未验证 |

## 6. 对 agent 产品和基础设施的实际收益

Jev 最适合带来的业务收益，不是替代所有自由文本推理，而是提高“下一步工具动作”的可靠性。一个 agent 面对浏览器、代码仓库或运维系统时，通常也在做同样的循环：观察当前状态，产生几个合法动作，判断每个动作的后果，选择一个并等待系统反馈。Mario 实验把这个循环做成了可测量的最小闭环。

| 业务场景 | Jev 风格能力的直接收益 | 需要补充的基础设施 | 首要指标 | 优先级 |
| --- | --- | --- | --- | --- |
| 浏览器自动化 | 从“盲点一个按钮”变成“在合法候选中选择”；减少过期页面上的点击、错误表单提交和重复重试 | DOM/AX 状态摘要、候选动作注册、页面版本/元素新鲜度、可逆操作回滚、提交前 guard | 任务完成率、无效点击率、重试次数、过期动作丢弃率、每任务模型调用数 | P0 |
| 代码检视与 PR review | 先对 diff、调用链、测试和配置变更做候选影响分析，再选择下一处最值得检查的文件或命令；减少无效上下文阅读 | CodeGraph/索引状态、只读探针、风险评分、测试结果快照、行级 provenance | 高风险缺陷召回率、误报率、首个有效发现时间、review token、漏检率 | P0 |
| 工具调用与工作流编排 | 对多个 API/CLI 步骤做前置条件和后果比较，失败时从安全分支恢复，而不是重复同一个调用 | typed action schema、dry-run、幂等键、补偿动作、审批门、重试预算 | 一次成功率、不可逆失败率、自动恢复率、人工接管率 | P0 |
| 用户体验 | 在模型思考或工具执行期间保持状态新鲜，给出“正在验证/已回退/需要确认”的真实反馈；减少界面卡住和结果误报 | progress events、决策来源、fallback 状态、延迟预算、可解释的确认卡片 | 首次响应时间、过期结果率、用户重试率、人工取消率、满意度 | P1 |
| 运维和 GUI 操作 | 把 runbook 步骤先在沙箱或 dry-run 中试演，遇到风险时选择回退路径；降低误操作 | 沙箱快照、权限 guard、影响面估计、审批和审计日志 | 变更失败率、回滚成功率、MTTR、未授权动作数 | P1 |
| 长任务个人助理 | 把一次失败归因成场景技能，后续相似任务复用等待、重试或替代路径 | 场景签名、经验库、技能版本、反事实记录、过期时间 | 相似场景复用率、重复失败率、任务总时长 | P1 |

### 6.1 浏览器自动化：收益最大的是“新鲜度 + 后果验证”

浏览器 agent 最常见的失败不是完全不理解页面，而是动作基于旧页面：点击后页面重绘、弹窗出现、列表排序变化、登录状态过期，模型仍执行上一轮的坐标或元素引用。Jev 风格的闭环可以把每个浏览器动作包装成候选对象：

```json
{
  "action": "click",
  "target": "submit-order",
  "preconditions": ["button.enabled", "cart.total == expected"],
  "effects_to_check": ["url changes", "order id appears"],
  "risk": "irreversible",
  "state_version": "dom-hash"
}
```

模型只负责在当前版本的候选动作中选择；执行器在点击前检查 `state_version`，点击后用 DOM/网络/可访问性状态验证预期效果。页面已变化时，旧动作直接丢弃并重新生成候选。这个机制能直接减少过期点击和重复提交，且比继续堆提示词更容易审计。

浏览器场景不一定有像 NES 那样的完整快照，所以不能照搬“每个动作都真实试跑”。优先使用只读探针、页面预览、请求 dry-run 和可撤销操作；付款、删除、发送消息等不可逆动作必须经过风险 guard 和用户确认。Jev 的价值在候选比较和状态新鲜度控制，回滚和权限边界仍由执行器负责。

浏览器是最值得先做的业务试点：登录后导航、筛选、分页和表单预填都频繁、失败率高，而且通常可以重试或撤销。付款、删除、发帖等不可逆动作则只做预览和人工确认。浏览器 what-if 不应假设能真实回滚外部副作用，应优先使用 clone tab、网络拦截、只读探针和提交前校验。

### 6.2 代码检视：收益是缩小搜索空间和提高发现顺序

代码 review 可以把 diff 后的工作表示成候选检查动作：查看受影响调用者、读取配置来源、运行窄测试、检查迁移兼容性、比较错误处理路径。每个动作带有预期信息增益、耗时、是否只读和风险。模型选择下一步，CodeGraph、静态分析和测试执行器提供实际结果，再把结果写回状态。

这比让模型一次性阅读整个仓库更适合基础设施：状态摘要可以包含变更文件、符号影响、测试失败和未索引标记；候选动作可以被限制在当前 diff 的影响面；每条发现都能关联到命令输出、文件行和提交版本。最终收益不是“模型看得更多”，而是更快到达高风险路径，并降低漏掉关键调用链的概率。

代码检视的安全 guard 也比游戏更清晰：默认只读，禁止自动 push、merge、部署和修改凭据；测试或构建可以在隔离工作树执行；任何写操作都把审批作为候选动作的前置条件。这样可以把 Jev 风格的选择能力接入现有 CodeGraph、测试和 review 面板，而不扩大模型权限。

代码 review 的候选后果可以在临时 worktree 中验证：读取调用者、运行窄测试、编译、lint、依赖扫描和比较候选补丁。Jev 负责在这些证据中排序下一步，而不是把“看起来可能有问题”当成事实。首要指标是有效评论率、漏检率、首个有效发现时间、reviewer 分钟和额外测试成本。

### 6.3 用户体验：把“模型能力”转成可感知的可靠反馈

用户最在意的通常不是模型选了什么，而是 agent 是否在做有进展的事情、是否误用了旧状态、失败后有没有恢复。当前 Mario viewer 里 `thinking`、`source`、`mentor_trace`、`verified_route_used` 和终态帧的处理，正好对应产品层需要的事件：

- `observed`：agent 看到的状态版本和摘要；
- `proposed`：模型选择的候选动作和置信度；
- `validated`：guard 或 dry-run 对动作的验证结果；
- `executed`：实际执行的动作和耗时；
- `rejected`：因状态过期、风险或权限被拒绝；
- `recovered`：回退、重试或替代计划的结果。

有了这些事件，界面可以显示“正在重新读取页面”“已丢弃过期操作”“需要你确认发送”，而不是显示一个无意义的转圈。这样既能降低用户对 agent 误操作的焦虑，也能让支持团队定位问题。这里的收益来自 provenance 和状态机设计，不能简单归功于某个模型。

### 6.4 对 agent 基础设施的具体抽象

建议把 Mario 代码里的隐含机制抽成四个通用组件：

1. **State Envelope**：统一保存状态摘要、版本、时间戳、来源、未知字段和最近动作；浏览器用 DOM/AX hash，代码用 git/tree/index revision，运维用资源版本。
2. **Action Registry**：每个工具动作声明参数 schema、前置条件、预期后果、风险等级、是否可逆、最大耗时和所需权限。模型只能选择 registry 中的动作或宏。
3. **Outcome Evaluator**：优先使用 dry-run、只读探针、测试、模拟器或沙箱试演，输出成功、失败、风险、信息增益和可用后续动作。模型选择的是可验证结果，而不是凭空预测世界。
4. **Execution Shield + Provenance**：执行前做状态新鲜度、权限、风险和幂等检查；执行后记录模型原始意图、guard 改写、实际动作、输出、回退原因和最终结果。

这四个组件比“把 Jev API 直接接到更多工具”更值得优先建设。它们可以让不同模型共享同一个可靠执行面，也能让规则、Jev、本地模型和人工选择在同一个评测协议下比较。

建议采用分级自治：L0 只读建议，L1 执行可逆动作并经过 guard，L2 在快照/沙箱里试演后自动提交，L3 不可逆动作必须人工批准。每个试点都要做四组消融：规则、单次模型、Jev 加候选验证、完整 guard/记忆闭环。这样才能知道收益来自模型、候选后果生成还是执行保护。

### 6.5 哪些场景不适合优先使用

没有明确候选动作、没有可验证后果、状态变化极慢或主要价值在开放式写作时，Jev 风格的收益会很小。它也不应直接控制付款、删除、部署等不可逆操作；这些场景应先做风险分类、dry-run 和用户审批。对浏览器和代码 review，先把动作空间和证据链做好，再比较 Jev 与其他模型，才不会把基础设施缺陷误判成模型差异。

业务 ROI 可以按单任务估算：

```text
(任务动作减少 × 单次工具成本)
+ (失败重试减少 × 单次工具成本)
+ (人工接管时间减少 × 人工时薪)
- 候选试演成本
- 新增延迟成本
```

Mario 的通关数字只能证明控制栈可运行，不能填入这些业务变量。浏览器应做任务成功率、过期动作率、重复点击和人工接管的 A/B；代码 review 应做有效评论率、漏检率和 reviewer 时间的 A/B；运维应做 MTTR、变更失败率和回滚成功率；用户体验应做完成率、纠正/撤销率和 p95 延迟。

落地顺序建议是：P0 统一状态、候选、结果、guard、freshness 和 provenance schema；P1 做浏览器可逆流程与只读代码 review；P2 在 clone tab、临时 worktree 和 staging 中引入候选试演及模型路由；P3 再把已验证轨迹编译成跨场景技能，接入低风险运维和 CI 恢复。

## 7. 对 agent 能力的启发

### 7.1 结构化世界模型比原始观察更重要

RAM 到网格、特征、警告、跳跃可达范围和接触时间的转换，把一个高频控制问题压缩成了少量可比较的语义变量。对 agent 来说，好的状态抽象往往比更长的自然语言提示更有价值。

### 7.2 让环境做可验证的预测，让模型做选择

分支模式的提升来自“实测候选结果 + 模型选择”，而不是单纯增加提示词。只要环境足够快、可回滚，agent 可以把模型从低可靠的物理模拟器变成高层决策器。

### 7.3 延迟是能力的一部分

实时实验说明，同一个模型在暂停环境时可以思考，在不停顿环境中却可能因答案过期而失败。agent 系统必须把延迟、动作保持时间、状态新鲜度和危险窗口作为显式状态，而不是把网络调用当成零成本函数。

### 7.4 安全层和模型能力要分别测量

守卫可以显著提升通关率，但也可能遮蔽模型的错误。当前 `model_decision_sources`、`verified_route_used` 和 `mentor_trace` 的记录方向是正确的；后续应在报告中同时给出“模型原始选择成功率”和“最终系统成功率”。

### 7.5 失败记录要变成可检索的经验，而不是一串日志

场景签名、动作、x 位置、死因、教练反思和后续结果已经形成了经验数据的雏形。真正的收益来自按障碍类型、速度、天花板、落点和时序聚类，复用“这一类场景的安全动作”，而不是按绝对 x 坐标硬编码。

## 8. 后续收益方向

以下按预期收益与投入比排序。

### A. 先建立逐决策离线基准（最高优先级）

保留 Jev 的原始请求响应，固定模拟器版本和随机状态，记录每个决策的状态哈希、候选结果、Jev 选择、本地选择、守卫改写、最终动作、延迟和结果。对同一批状态做本地模型离线重放，计算：

- 动作一致率和合法动作率；
- 在候选结果排序上的一致率；
- 遇到死亡候选时的拒绝率；
- 规则/守卫改写前后的差异；
- 单步选择正确但三步后果错误的比例。

这会直接回答“本地模型是否复现了 Jev 的决策习惯”，成本远低于反复跑完整关卡。

### B. 把分支搜索变成可调预算的模型工具

当前 11 个候选 × 36 个后续组合的计算量很大。可以按风险分配预算：安全直路只做一层，检测到坑、低天花板或敌人时加深到两三层；对相似状态缓存结果；先用规则筛掉明显死亡动作，再把剩余 3–5 个候选交给 Jev。这样能保留前瞻收益，降低每关 15–20 分钟的搜索时间。

### C. 学习动作结果模型，减少真实模拟器试演

把每次 `(state, action) -> (alive, dx, landing, hazard)` 存成数据，训练一个轻量结果预测器，只在不确定或高风险场景回到 NES 真试演。它不需要替代 Jev，而是替代大量重复模拟器调用。收益指标是“相同通关率下的模拟器帧数和墙钟时间”。

### D. 让路线从坐标脚本升级为障碍技能

目前验证路线能解决 1-2、1-3，但路线是动作时序，泛化有限。将路线片段抽象为 `长坑 + 低天花板 + 当前速度`、`敌人在上方平台`、`需要后撤腾挪` 等技能，并用局部状态触发，可以把一次搜索结果迁移到相同几何结构，而不必绑定绝对 x。

### E. 建立本地模型蒸馏/微调数据集

上游 Jev 的 choice、候选结果和最终成败可以作为教师数据；同时要保留教师没有选择的反事实候选，避免模型只学动作频率。先做监督式动作选择，再加入“选择后续存活率”的排序损失，最后用失败场景做难例回放。验收应包含未见过的关卡和不同起始快照。

### F. 将教练反思变成可验证的计划搜索

教练目前输出可执行计划，但计划仍依赖自然语言解析。下一步可让教练输出结构化的有限状态计划：触发条件、等待帧数、动作、终止条件和失败回退；执行器逐帧验证计划，失败后自动比较新旧计划。这样可以降低“反思说得对但时序执行错”的风险。

### G. 再做真实延迟和多模型路由

当 A–F 有基线后，再比较 Jev、本地模型、规则和混合策略的延迟/成本/成功率。可以让快模型负责常规直路，Jev 或深搜索只在高风险场景介入。没有逐决策基准前直接换模型，无法知道收益来自模型还是守卫/路线。

## 9. 建议的下一阶段验收表

| 阶段 | 交付 | 成功标准 |
| --- | --- | --- |
| 1 | 决策数据集与状态哈希 | 可在无网络条件下重放同一批状态，结果逐条可对齐 |
| 2 | Jev vs 本地模型离线比较器 | 输出动作一致率、合法率、危险候选拒绝率和延迟 |
| 3 | 风险自适应分支预算 | 先建立无预写路线的分支基线，再比较相同场景的成功率、最远距离和搜索墙钟时间 |
| 4 | 修正状态摘要和动作窗口，扩展无路线接管实验 | 以本次 Jev 0/9、Qwen 0/3 为基线，分别报告纯模型和带守卫结果；路线回归另列 |
| 5 | 技能化经验库 | 相似障碍跨关卡复用，不能只靠绝对 x 命中 |
| 6 | 延迟/成本路由 | 在同等成功率下减少 Jev 请求或总体等待时间 |

## 10. 限制与证据声明

- 本地仓库没有 Jev 权重，不能离线运行 Jev 本体；`TYPESAFE_API_KEY` 也没有提交。
- 历史 Jev 成绩来自仓库现有 `runs/results.jsonl` 和 README；2026-09-23 已重新调用远程 Jev API，完成 9 局纯模型、1 局带守卫及 1 局在线分支实验，并保留请求、响应和动作证据。
- 前期 1-2、1-3 的离线回归来自模拟器搜索/验证路线，`calls=0`。本次纯模型实验另行统计：Jev 0/9 通关，Qwen2.5-1.5B-Instruct 0/3 通关；分支 Jev 的 1-1 实验也未通关。
- 1-3 的搜索路线曾重复两次并接入双画面；本次已测试 1-3 纯模型执行但未成功，1-4 及后续连续推进仍未验证。失败结果受状态摘要、动作接口及模型选择共同影响，不能单独归因于模型。
- 进程看护、离线回放、路线和流媒体测试已通过；Jev 服务和本地 Qwen 推理已实际调用。Windows 原机执行、行内目标硬件部署及生产业务收益仍未验证。
