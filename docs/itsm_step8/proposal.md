# 第 8 步执行提案：E0-base dev 诊断与 GRPO train 组内方差诊断（待验收，未执行）

日期：2026-11-01。本文件只是提案：没有调用 LLM，没有启动 GPU，没有生成 SFT 轨迹，没有进入第 8 步执行或 S1。

依据：
- `docs/playbook2.md` 第 8 步："在固定 dev 上记录未 SFT 基座的有界基线，另在已批准的 GRPO train 上记录同题多采样与真实 G 组诊断……不调用 held-out；不把 train 诊断叫 dev 成绩。预先确定 SFT 和 RL 的选 checkpoint 规则、诊断后的调整权限与计算预算。"
- 协议 §12（G = 4、重试与丢组）、§13.1（哈希种子）、§14.3（D7）、预算 `configs/itsm/budget_v1.yaml`、第 7B 步结果（`docs/itsm_step7b/README.md`）。

**7B 的结果不作为能力基线**：7B 的用户是不告知目标的脚本化替身，4 条 episode 的 reward = 0 只说明工程链路通了，与 E0-base 的能力无关，下面的估算也不使用它们的成功率。7B 只为本提案提供了时延、长度和显存的工程数据。

## 1. 两项诊断分开进行、分开报告

| | A. E0-base dev 诊断 | B. GRPO train 组内方差诊断 |
|---|---|---|
| 目的 | 未 SFT 基座在 dev 上的有界基线（playbook：E0-base） | 在 GRPO 可训练题上，同题按 G 组采样时，有多少组的 reward 不全相同（有效组比例） |
| 任务 | dev 全部 13 题（`itsm_split_v3`） | GRPO 可训练题全部 26 题（`itsm_usage_v1` 的 `grpo_trainable`） |
| 不包括 | held-out 12 题；train 题 | R1 待定 5 题、SFT-only 6 题、dev、held-out |
| 每题采样 | n = 4 | 1 组，G = 4（与协议 §12 相同） |
| episode 数 | 52 | 104 |
| 主指标 | 逐题成功次数 c/4（fuzzy DB）；另列 LLM 文本比较的 DB 结果和 NL 断言结果 | 有效组比例；全 0 组、全 1 组的数量；逐题成功次数 |
| 评分 | fuzzy DB（D1b）为主；同一个最终 DB 再做 D7 的 LLM 文本比较，并评 NL 断言（D7） | 只用训练 reward（fuzzy DB），不调用裁判 |
| 报告名称 | "E0-base dev 诊断" | "GRPO train 诊断"，**不称为 dev 成绩，也不称为能力基线** |

统计口径（沿用之前的更正）：
- 每题报告 c/n；
- n ≥ k 时可以给出 pass@k 的估计 1 − C(n−c,k)/C(n,k)，并注明 n、k、温度和模拟器的随机性；
- 有效组比例用实测值，混合组概率公式 1 − p^G − (1 − p)^G 只作解释；
- 不给独立二项标准误。dev 13 题里有 12 题属于同一个 workflow，题数和家族都很少，结果只作诊断，不作结论。
- 同题的多条轨迹不增加独立任务数。

## 2. 采样设置

| 项 | 值 | 说明 |
|---|---|---|
| 策略 | Qwen3-8B @ b968826d 初始权重，bf16；策略版本 = `global_step 0` | 与 7B 相同的权重和哈希核对 |
| 温度 / top_p / top_k | **1.0 / 1 / −1**（建议） | verl rollout 的默认值；7B 只在 1.0 下验证了 rollout logprob 与重算一致。airline 用过的 0.8 未验证，如果你想用，要先补一次 logprob 对照 |
| `repetition_penalty` | 1.0 | 与 verl 相同 |
| A 和 B 的温度 | 相同 | 让 B 的组内方差与 A 的分布可比；如果你希望 A 另做温度 0 的贪心基线，可以加 13 条，单列 |
| `enable_thinking` | False | 预算 v1 |
| 引擎 seed | 42；`PYTHONHASHSEED=0`（协议 §13.1） | 有并发时采样不可复现；固定哈希种子只固定列表观察的顺序 |
| 记录 | 每条轨迹的完整 token、mask、采样 logprob、用户回复、工具调用和观察、终止原因、reward 分项、费用 | 存入 `output/`，摘要和可读版本放入 docs |

## 3. 长度候选与截断规则

- **`max_model_len` = 40,960**（候选；7B 只做了工程验证）。同时统计每条 episode 的总长度：超过 32,768 的 episode 有多少，在 32,768 下它们会在哪一回合被 `response_length` 截断。这样跑一次就能得到两种长度下截断率的实际证据，不需要跑两遍。是否把 40,960 定下来，等你看了这份证据再决定。
- 其余沿用预算 v1，不改：
  - `max_prompt_length` 28,672，`max_response_length` 12,288，`max_action_tokens` 512，`max_steps` 40；
  - 工具结果完整进入上下文，不裁剪；
  - 下一条观察加一个动作放不下时，`truncate("response_length")`；
  - 非法动作（含 `max_action_tokens` 触顶）`truncate("invalid_action")`，outcome = 0（D2）；
  - 步数用尽按协议 §7 处理。
- **这些都算有效样本**：reward 按规则给（截断和非法动作为 0），不当作基础设施故障，也不重试。
- **基础设施故障**（`user_sim`、`timeout`、`worker_crashed`、`sampler_contract`、裁判故障等）按协议 §12：首次之外最多再重试 3 次；评分级故障只重做评分，用同一条轨迹和同一个最终 DB；用尽后整组丢弃，不用 reward = 0 补位。B 中丢弃的组超过 25%（D6）就停止并报告。
- 单列报告：截断率和终止原因分布、`max_action_tokens` 触顶次数、业务错误（`domain_error`、`unknown_tool`）的次数。

## 4. 用户模拟器与裁判：正式诊断之前的最小复核

先单独跑一小批付费请求（独立预算 id `step8_review`），结果交你验收后，再开始 A/B。

| 对象 | 内容 | 请求数 |
|---|---|---|
| D5 模拟器 `deepseek/deepseek-flash` | 用 3 道 **GRPO train** 题的场景（不用 dev、held-out），首个用户回合，同一输入各重复 3 次。核对：返回的模型 ID；`thinking: disabled` 是否生效（没有 `reasoning_content`，推理 token 为 0）；`finish_reason`；输出长度；3 次回复是否完全一致（记录一致率，不假设确定） | 9 |
| D7 裁判 `deepseek/deepseek-v4-pro` | 2 组固定的文本对（取自 train 题的 gold 与改写）和 1 组 NL 断言，各重复 2 次。核对：没有 `reasoning_content`；`finish_reason == stop`；判定 JSON 能解析；重复判定是否一致 | 6 |
| 合计 | 上限 20 次请求、1 美元（含重试），预留式控费，与 A/B 分开记账 | ≤ 20 |

**需要你决定一项 D5 参数**：现在的 D5 没有 `max_tokens`，所以模拟器的输出没有上界，按"请求前预留"的规则无法给出可靠的预留。建议加上 `max_tokens = 1024`：正常的用户回复约 50–200 token；输出到 1024 被截断时（`finish_reason == length`），按 `user_sim` 基础设施故障处理。这会改变 D5 的调用参数，需要你批准，并在复核中验证。

Micu Teacher 那批遗留的待复核项（B2/B3 的参数值）属于 S1 的准备工作，用的是 Micu 的预算，不放进第 8 步。

## 5. 付费请求与金额上限

价格依据与 7A 相同（DeepSeek 官方价格页，高峰价、不命中缓存）：
- flash：输入 0.30、输出 1.20 美元/百万 token；
- v4-pro：输入 1.32、输出 3.96 美元/百万 token。

预留方式与 7A 相同：每个请求发出前，用请求体字节数 + 64 作为输入上界，加上输出上限，写入预留；结算时用实际 usage；实际费用超过预留就停止。

| 项 | 次数（估算） | 估算费用 | 预留上界下的费用 |
|---|---|---|---|
| 模拟器（A + B，156 条） | 约 6 次/条 × 156 ≈ 940 次；每次约 4k 输入、150 输出 | ≈ 1.3 美元 | 每条 15 次、8k 输入、1,024 输出 → ≈ 8.5 美元 |
| 裁判（只用于 A 的 52 条） | 约 6 次/条 × 52 ≈ 310 次；每次约 3k 输入、300 输出 | ≈ 1.6 美元 | 每条 12 次，按 `max_tokens = 4096`（D7）全额预留 → ≈ 16.9 美元 |
| 复核（§4） | ≤ 20 次 | ≈ 0.3 美元 | 单独上限 1 美元 |

- 这些估算没有实测依据：每条 episode 的用户回合数、裁判调用次数，都要等复核和开头几组结果出来才知道。
- **建议上限：A + B 合计 12 美元**（含全部重试），复核另计 1 美元；都与 S1 的 45 美元（Micu）分开。账本用新的预算 id 和新的账本文件，单一写入者。
- 预留可能超过上限时，在**组的边界**停止：已经开始的组如果因此不完整，整组丢弃并记录，不会从中截出部分样本，然后停止并报告。顺序固定为先 A 后 B，任务按 ID 排序，事先确定，不按结果调整。
- **DeepSeek 余额只有 ¥5.92（约 0.8 美元），不够**。开始前需要你充值，至少覆盖上限（约 ¥100）。账本记账只是估算，实际扣费以 DeepSeek 账单为准，两者分开写。

## 6. GPU 时长上限

| 阶段 | 估算 |
|---|---|
| 开机、环境检查、依赖推送后的 preflight、确认权重仍在磁盘上并核对哈希（不需要重新下载） | ≈ 5–10 分钟 |
| vLLM 初始化（`max_model_len` 40,960，复用 7B 的编译缓存） | ≈ 2–4 分钟 |
| A + B 共 156 条 episode，8 条并发 | 7B 单并发时平均约 22 秒/条（用户回复是瞬时的）；加上真实模拟器约 6 × 2 秒，约 35 秒/条；8 条并发约 12–20 分钟。裁判评分在 worker 内进行，另加约 5 分钟 |
| 回传、核对、关机 | ≈ 3 分钟 |
| **合计** | **约 0.5–0.8 GPU·h** |

- **建议上限：2 GPU·h**。计时方式与 7B 相同：从开机请求到确认关机的挂钟时间，开机、空闲、重试都计入；每个阶段开始前都核对预算。只用 `exp-itsm` 和它现有的磁盘、单卡。
- 显存：沿用 7B 的 `gpu_memory_utilization = 0.45`，与训练时 rollout 的配置相同。所有 episode 共享约 28k token 的相同前缀（policy 和工具 schema），前缀缓存命中后，8 条并发应该能放进 175,760 token 的 KV 池。这一点要实测（记录 vLLM 的 preemption 次数）；如果 preemption 太多，就把并发降到 4，不改显存配置。
- 并发只影响吞吐和时延，不改变采样语义。

## 7. 单进程适配器与正式 verl 链路的边界

第 8 步**只采样、不训练**。建议继续用 7B 的单进程链路：`VllmTokenServer` + `ItsmAgentLoop` + `ItsmEnvWorker`，再加一个并发运行器。

| | 单进程适配器（第 8 步） | 正式 verl 链路（第 9 步验证） |
|---|---|---|
| 引擎构造与参数 | 按 verl 方式构造（`build_cli_args_from_config` → `AsyncLLM`），参数相同 | Ray actor 中的 `vLLMHttpServer` |
| `generate` 的语义 | 复刻：`max_tokens` 限制、`logprobs = 0`、逐 token 取 logprob、`stop_reason` | 原生实现 |
| agent loop、环境、评分 | 同一份 `ItsmAgentLoop` / `ItsmEnvWorker` 代码 | 同一份代码，但由 `AgentLoopManager` 调度 |
| 没有覆盖的部分 | Ray、`AgentLoopManager` 的调度和负载均衡、`generate_sequences` 批处理、数据集 pipeline、reward manager、权重同步（这里用 `load_format = auto` 直接读磁盘权重）、FSDP actor 的 `compute_log_prob`、flash_attn | 全部要在第 9 步验证 |

- 结论边界：第 8 步的数字代表"冻结的基座策略，在与 verl 相同的采样语义下"的表现。它**不能证明** verl 训练链路正确。第 9 步要在 verl 链路上对 2 道题各跑 1 组（G = 4）作冒烟测试，核对字段、mask、logprob 和 reward，但**不要求**与第 8 步的结果逐条一致。
- 第 8 步的 rollout logprob 照常记录，但不再做重算对照：那是第 9 步在真实 actor 配置下的任务。

## 8. 部署顺序与运行时依赖（本轮已修好的部分）

开机后、启动诊断之前，按以下顺序执行：

1. `finalize_runs(experiment_id="exp-itsm")`：把 7B 遗留的 5 条运行记录收尾（见 §9.1）。
2. 本机执行 `python scripts/itsm/runtime_deps.py push --ssh-alias autodl-exp-itsm-fc77b761 --remote-root /workspace`：按 `configs/itsm/runtime_dependencies.json` 把运行时依赖推送到原路径，并在远端核对 SHA256。目前只有 `docs/itsm_stepS0/usage_audit.json` 一项，它的哈希与冻结的 `itsm_usage_v1.json` 中记录的一致。
3. 远端启动脚本先在 Python 启动前导出 `PYTHONHASHSEED=0`，再运行 `runtime_deps.py preflight`：核对依赖、冻结数据、用途校验，以及 grpo/dev 的取题门槛。任何一项缺失或被篡改，都会在采样之前以非零码退出。
4. 然后才启动 vLLM 和诊断。

本轮的离线验证：
- 在一个干净的部署目录里，模拟代码同步之后远端会有的文件（不含 `/docs`）。推送之前 preflight 失败；推送之后成功（grpo 26 题、dev 13 题，加载的是部署目录里的代码）；把文件改动 1 个字节，或者删掉，preflight 都失败；本地文件被改过时，`push` 拒绝推送。
- rollout 在依赖有问题时，于建 env 和采样之前失败。
- `candidate_screening.json` 不在部署目录中，以上检查也照常通过，确认它不是运行时依赖。

测试：`tests/unit/test_itsm_runtime_deps.py`、`tests/unit/test_itsm_hash_seed.py`。

## 9. projtool 遗留的 running 记录与关机定时器

### 9.1 遗留的 5 条 running 记录（已核实，未处理）

| run_id | 远端退出码（`.projtool-exit-code`） | 实际情况 |
|---|---|---|
| `exp-itsm/2026-11-01T03:54:38.416643Z` | 1 | flash_attn 检查导致环境报告失败（脚本 bug） |
| `exp-itsm/2026-11-01T03:55:39.890636Z` | 1 | 清单文件在 `docs/`，没有同步到远端 |
| `exp-itsm/2026-11-01T03:57:04.884339Z` | 1 | Xet 下载返回 HTTP 401，下载阶段失败 |
| `exp-itsm/2026-11-01T04:03:58.749199Z` | 1 | 下载和核对完成；复建长度输入时缺 `usage_audit.json` |
| `exp-itsm/2026-11-01T04:44:15.822424Z` | 1 | 7B 的全部工作都已完成，只有最后统计缓存大小的 `du` 返回了 1 |

（`04:00:33` 那一条已经用 `cancel_run` 收尾，记为 killed。）

- 5 条都已有退出哨兵（本地镜像中可见），tmux 会话也都已不存在。
- **为什么这次没有处理**：
  - `finalize_runs` 要通过 SSH 探测哨兵；实例已关机，5 条都返回 "SSH transport failure"，没有关闭任何一条。
  - 这次调用还有一个副作用：它在本地 docs 仓库提交了一行 journal（`5c826de`），记录的是另一条早已结束的运行 `exp-itsm/2026-10-30T00:39:44.435155Z`（failed），与这 5 条无关，我没有改动它。
  - `finalize_run` 只有在绑定已释放（`binding_state != bound`）时才走离线收尾路径；释放会删掉数据盘上 16 GB 的权重，不可取。
  - `reconcile_manifests` 只处理"state 中已结束、本地 manifest 仍写 running"的情况，不适用于这 5 条。
  - 所以，在"不开机"的前提下，没有安全的受支持接口。我没有修改 projtool 的内部状态文件。
- **方案**：下次因第 8 步开机后，第一件事是运行 `finalize_runs(experiment_id="exp-itsm")`。它会读取哨兵，按"退出码非 0 记为 failed"的规则确定性地关闭这 5 条。在此之前，这 5 条会让 `instance_down` 的占用检查拒绝关机；那时要先确认没有 tmux 会话和 GPU 进程，再传 `force=true`（7B 就是这么做的）。
- 04:44:15 那条会被记为 failed，但 7B 的工作其实都已完成；这一点写在 7B 报告里，不靠改记录来表达。

### 9.2 旧关机定时器（7B 结论已更正）

- **事实**：projtool 的定时器到期时，会检查有没有 `proj-*` tmux 会话，有就跳过（`projtool/mcp/tools/training.py` 中的 `_auto_shutdown_watchdog`）。7B 中 03:59:01 的那次关机，是第 1 次运行留下的定时器到期：当时 3 次运行都已退出，所以是按设计关机，**没有误伤正在运行的任务**。7B 报告里原先的说法已更正。
- **实际存在的风险**：
  1. 旧定时器在新运行开始之后仍然有效。一旦新运行也结束了，旧定时器就可能在人工排查、上传文件的空档里关机。7B 就是这样被打断的。
  2. 我在 7B 里用的办法是"运行开始时建 `/root/no_autoshutdown`，结束时删掉"。如果运行被 `cancel_run` 或 SIGKILL 杀掉，删除那一步不会执行，标记就会遗留下来，**之后所有定时器都会失效，空闲计费失去保护**。
- **建议（修 projtool，本仓库不改，交给 projtool 维护）**：
  1. **代次令牌**：`run_script` 每次启动都写入 `/root/.projtool-launch-generation`（单调递增）。每个定时器在设定时记下当时的代次，到期时发现已经有更新的代次就跳过，由最新一次运行的定时器负责。最后结束的那次运行总会设定自己的定时器，所以超时关机的保护还在，只是旧定时器不会抢在它前面。
  2. **有时限的租约**代替永久标记：人工排查时写 `/root/no_autoshutdown_until`，内容是截止时间（例如 30 分钟后），定时器只在截止时间之前跳过。即使忘了删除或进程被杀，到期后保护会自动恢复。
  3. 在 projtool 修好之前，本仓库的做法：启动脚本不再"开始时建、结束时删"标记；需要人工排查时，只用带截止时间的租约，由我在排查结束时删除；排查期间实例的空闲计费照常计入 GPU 时长。

## 10. 执行前还需要离线完成的工作（不需要 GPU，也不调用 LLM）

1. 并发运行器：用 asyncio 和信号量调度多条 episode，每条一个 `ItsmEnvWorker`；按 §3 实现重试和丢组；计费与账本对接；用假引擎和假 litellm 测试。
2. dev 评分：在 worker 子进程内、最终 DB 销毁之前，按 D7 完成 LLM 文本比较和 NL 断言评分。现在的 env 只给出 fuzzy DB，`nl_status` 一律是 `not_evaluated`。裁判故障按评分级故障处理，用同一个最终 DB 重做，不重跑 episode。用假 litellm 测试参数、提示词 sha256 和故障路径。
3. D5 加 `max_tokens`（需要你批准，§4）；模拟器调用接入预留式账本（每次调用前预留）。
4. 第 8 步的启动脚本：导出 `PYTHONHASHSEED=0` → preflight → vLLM → A → B；每个阶段计时并核对预算；失败也要回传结果。
5. 汇总脚本：生成逐题 c/n、有效组比例、截断、长度分布（包括超过 32,768 的比例）、费用（账本估算与实际扣费分开写）。

## 11. 按 playbook 需事先固定的规则（提案）

- **选 checkpoint 的规则**：不用 dev 分数选，也不用 held-out。SFT（S4）用事先登记的 epoch 数，取最后一个 checkpoint；GRPO（第 9 步以后）用事先登记的步数，取最后一步。dev 只用来发现崩溃（例如格式错误率骤升），一旦发现，停下报告，不据此挑选 checkpoint。这与 `budget_v1.yaml` 的 `checkpoint_selection: pre_registered_steps_and_rule_frozen_by_step8` 一致；具体的 epoch 数和步数在 S2 和第 9 步的方案中报批。
- **诊断之后的调整权限**：A/B 的结果只用来提出调整建议，例如 B 的有效组比例过低时，提议改 G、温度或任务集。不自动改任务、改 reward、调温度或增加 SFT 量。

## 12. 需要你决定的事项

1. 两项诊断的规模：A 每题 n = 4（52 条），B 每题 1 组、G = 4（104 条）；
2. 温度 1.0（A 是否另加 13 条贪心）；
3. `max_model_len` 40,960 作为本次运行的设置，同时统计 32,768 下的截断证据；
4. D5 是否加 `max_tokens = 1024`；
5. 付费上限：复核 1 美元，A + B 12 美元；DeepSeek 充值；
6. GPU 上限 2 GPU·h，8 条并发；
7. §9.2 的 projtool 修复建议，是否交给 projtool 维护；
8. §11 的选 checkpoint 规则和调整权限。

## 13. 不做 / 停止条件

- **不做**：held-out、SFT/Teacher 轨迹、权重更新、修改冻结数据、修改 Bench2 工具/policy/任务/reward、启动 S1。
- **立即停止并报告**：
  - 复核中模型 ID 不一致、思考没有关闭、裁判输出无法解析；
  - 预留超过上限、实际费用超过预留；
  - GPU 预算不够完成下一阶段；
  - B 中丢弃的组超过 25%；
  - preflight 失败，或子进程的哈希种子没有固定。
