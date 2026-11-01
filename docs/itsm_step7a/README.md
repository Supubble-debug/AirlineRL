# 第 7A 步：离线 loop 验证与 API 兼容探测（待验收；付费探测未执行）

范围：只做第 7A 步。没有运行 held-out，没有生成 SFT 轨迹，没有更新权重，没有启动 GPU。40,960 仍是待验证的候选值。

**结论**：
- 离线部分完成：冻结记录、`ItsmAgentLoop` 契约验证、预留式控费账本、探测脚本。
- **云端付费探测没有执行**，按你的停止条件报告，原因见 §3：
  - 能用的云端实例只有 GPU 实例；
  - 当前环境里没有 Micu 的 key 和 base URL；
  - Micu 的价格无法在请求前核实。
- 第 7A 步的花费为 **0 美元**，付费请求 0 次。

| 产物 | 位置 |
|---|---|
| 冻结记录 | `src/agent_rl/data/itsm_frozen.py`（提交号和两个文件的完整 SHA256）；进度文件 |
| ITSM agent loop | `src/agent_rl/rollout/itsm_agent_loop.py` |
| loop 契约测试 | `tests/unit/test_itsm_agent_loop.py`（23 项，确定性采样替身） |
| 预留式控费账本 | `src/agent_rl/utils/spend_ledger.py`（第 7A 步和 S1 共用） |
| 探测脚本 | `scripts/itsm/probe_api_7a.py`（只能在云端运行；默认 dry run） |
| 账本与探测测试 | `tests/unit/test_itsm_api_probe.py`（11 项，模拟 HTTP，不联网） |

## 1. 冻结记录

| 文件 | 完整 SHA256 |
|---|---|
| `configs/data/itsm_split_v3.json` | `9326167dfee696a809f685f178284e23e6e8f3402559d4b47650c4fcf85b5bc0` |
| `configs/data/itsm_usage_v1.json` | `87f15baf96a5dbeec6a8c07ca0952edafd86667d18dfa01d63f066d35934fba7` |

冻结依据是提交 `70e6a3b451149bb24ae0eba18fc544705d4c323a`（2026-10-31）中的文件内容。

- **冻结文件本身不改**：`itsm_usage_v1.json` 里的 `status` 仍写着 "candidate for freeze acceptance"。我没有为改这个字去重写文件，冻结记录写在 `itsm_frozen.py` 和进度文件里，测试也断言这个字段保持原样。
- **加载时核对哈希**：`itsm_task_sets` 加载 v3 或用途登记时，会先核对两个文件的 SHA256 与冻结值一致，不一致就报错。
- **以后的变化另建版本**：用途资格有变，新建用途版本；轨迹审核另存文件，在文件里引用上面两个冻结哈希，不写回冻结文件。

## 2. ItsmAgentLoop 离线验证

`ItsmAgentLoop` 继承 verl 的 `AgentLoopBase`，直接使用 verl 自带的 `apply_chat_template`、`turn_separator` 和 `server_manager.generate`。

测试的设置：
- 分词器用固定版本的 Qwen3-8B；
- 环境用真实的 `ItsmEnv`、真实 Bench2 工具，用户是脚本化替身；
- **只有采样服务**换成确定性替身：它按脚本逐回合返回 token，并附上人为构造的 logprob。

**重要**：这些 logprob 是测试构造出来的。测试只能证明 loop 逐位置原样复制了采样方给的值，没有自己补值；**不能当作真实采样 logprob 的验收**，那一项留到第 7B 步。

| 契约 | 验证内容 | 结果 |
|---|---|---|
| 输入渲染 | `prompt_ids` 与直接调用 Qwen3 模板（policy + 公开工具 schema + 首个观察，`add_generation_prompt=True`、`enable_thinking=False`）逐 token 相同，结尾是 `<think>\n\n</think>\n\n`；没有打开 `enable_thinking=False` 时拒绝启动 | 通过 |
| 信息边界 | 解码后的 prompt + response 中不含任务描述、模拟器指引、NL 断言、task_id 或 "EnterpriseOps" 字样 | 通过 |
| 单动作 | 文字加工具调用、两个工具调用、工具调用 JSON 错误、工具调用块没闭合、空输出、达到 `max_action_tokens` 还没输出结束符，都以 `invalid_action` 结束，outcome = 0（D2） | 通过（6 种） |
| 修正 verl 解析器的两处行为 | hermes 解析器会**静默丢掉**格式错误的工具调用块（"文字 + 坏工具调用"会被当成普通消息），TauAgentLoop 在有工具调用时会**丢掉**旁边的文字。ITSM loop 自己数工具调用块、检查残留文字，这两种情况都判为非法 | 通过 |
| 增量观察 | 每个环境回合 = `turn_separator` + 只含一条 user 消息的模板渲染；工具结果原文完整出现，不压缩、不裁剪；token 数与第 6 步预算测量用的计数方法逐回合一致 | 通过 |
| token / span / mask | 策略回合的 token 与采样方返回的完全相同，mask = 1；环境回合的 mask = 0；两类区间互不重叠，合起来正好覆盖整个 response；每次 `generate` 收到的输入都是 `prompt_ids` 加上此前的 response | 通过 |
| logprob 字段 | mask = 1 的位置等于采样方给的值，来源记为 `sampler`；mask = 0 的位置是 0.0 占位，来源记为 `placeholder_masked`。采样方**不返回** logprob，或条数和 token 数对不上时，抛出 `SamplerContractError`，**不补 0** | 通过 |
| 不轮换上下文（D3） | 放不下下一条观察加一个动作时，以 `response_length` 结束；全程只有一个 request_id，prompt 不重写 | 通过 |
| 完整 episode | 第 1 题（887f5580）：9 个策略回合（4 个查询调用、说明拒绝关联已退役 CI 的计划消息、2 个写入调用、回顾消息、done），以 `agent_done` 结束，DB = 1 | 通过 |
| 按用途限制取题 | `grpo` loop 拒绝 SFT-only（b6ef60ca、78a5）、R1 pending（single_005）、dev 和 held-out 的题；`dev` loop 拒绝 train 题；held-out 只有 `final_evaluation` 用途能取 | 通过 |

**尚未覆盖、留到第 7B 步**：
- 真实 vLLM 采样能否返回逐 token 的 logprob；
- verl 训练实际用的是 rollout 返回的 logprob（bypass 模式，`old_log_probs = rollout_log_probs`），还是由训练引擎重算（仓库里两种配置都有：`grpo_sequence.yaml` 和 `grpo_sequence_bypass.yaml`）；
- 真实模型输出的工具调用格式；
- 32,768 / 40,960 长度下的推理和显存。

## 3. API 兼容探测：没有执行，原因如下

### 3.1 停止原因（对照你的停止条件）

1. **没有不启动 GPU 的云端执行环境**：本项目唯一的云端实例 `exp-itsm`（pro-7933a95ce90d，RTX PRO 6000）处于关机状态。projtool 的 `instance_up` 只会以 GPU 实例开机，不支持 AutoDL 的"无卡模式"，所以要在云端发请求，就得启动 GPU，与\"不启动 GPU\"冲突。我没有开机。
2. **没有凭据**：本地和项目配置里都没有 `MICU_API_KEY` 或 Micu 的 base URL；云端是否已经配好，要开机才能看到。我没有向任何地方写入、读取或打印密钥。
3. **价格无法在请求前核实**：
   - `www.micuapi.ai` 是 New API 面板，首页是前端渲染的，看不到内容；`/api/pricing` 返回 401，要登录才能看价格。
   - 公开资料只有第三方站点列出的 OpenAI 官方标价（`gpt-6.1-sol`：输入 2 美元、输出 10 美元/百万 token），**不是 Micu 的实际售价**，不能用来做预留。
   - 按你的规则，计费核实之前不能发出付费请求。

### 3.2 已准备好的探测（可以随时在云端运行）

`scripts/itsm/probe_api_7a.py` 的设计：

| 编号 | 是否收费 | 内容 | 核对什么 |
|---|---|---|---|
| F1 | 免费 | `GET {base}/models` | 账号下是否列出指定的模型 ID。没有列出，就不发任何付费请求 |
| B1 | 收费 | Responses API，纯文本，`max_output_tokens=256` | 返回的 `model` 字段、usage、推理 token 数，输出是否被上限截住 |
| B2 | 收费 | Responses API，一个 Bench2 函数工具，`tool_choice=required`，`parallel_tool_calls=false` | 是否恰好一个 function_call，参数是否是 JSON 对象 |
| B3 | 收费 | Chat Completions API，同一个工具 | 另一条工具调用路径能否用 |
| B4 | 收费 | 用户模拟器 `deepseek/deepseek-v4-flash`（D5，经 litellm） | 返回的模型标识、`finish_reason` |
| B5 | 收费 | 裁判 `deepseek/deepseek-v4-pro`，用 D7 参数（`extra_body` 关闭思考） | 服务端是否接受 disabled、有没有 `reasoning_content`、`finish_reason == "stop"`、模型 ID 是否一致 |

**控费方式**：
- 预算 1 美元、最多 6 次收费请求，**重试也计入**；不自动重试；与 S1 的 45 美元分开记账（账本 id `step7a_api_probe`）。
- 运行前先算出 5 个探测各自的最坏费用并加总，超过预算就不开始。
- 每个请求发出之前先写一条预留，写不进去就不发；拿不到 usage 的请求按预留全额计费。
- **输入 token 上界**：请求体的 UTF-8 字节数加 64。字节级 BPE 分词器（OpenAI o200k、DeepSeek）产生的 token 数不会超过字节数，所以不需要知道对方的分词器。

**立即停止的情况**：模型没有列出或无法确认；返回的模型标识与请求的不一致；输出超过上限（说明上限没有生效）；实际费用超过预留；预算用尽。

**不照搬 DeepSeek 参数**：Teacher 只发上面写明的参数，不传温度、不传思考开关；接口接受哪些参数，以实测的报错和返回为准。

**密钥**：从环境变量读取；结果和账本里都会把密钥替换掉；有测试专门检查服务端把 key 回显在错误信息里的情况。

**dry run**：不加 `--confirm-billing` 时，只做 F1 并给出预留计划，不会发出付费请求。

### 3.3 解除阻塞，需要你决定

| 项 | 选项 |
|---|---|
| 云端执行环境 | ① 你在 AutoDL 控制台把 `exp-itsm` 以**无卡模式**开机，告诉我后我用 `remote_exec` 运行（不算启动 GPU，费用很低）；② 明确允许以 GPU 模式开机、只用于发 HTTP 请求，跑完立即关机；③ 改为在本机运行（与你\"在云端核对\"的要求不同，需要你明确同意） |
| 凭据 | 在云端设置环境变量 `MICU_BASE_URL`、`MICU_API_KEY`、`DEEPSEEK_API_KEY`（例如写入只有 root 可读的 `/root/.itsm_api_env`）。不要贴到对话里 |
| Micu 价格 | 提供 Micu 控制台里 GPT-6.1 Sol 的价格依据（截图或价格页导出均可）：输入单价、输出单价（是否包含推理 token）、有无按次费用、币种与汇率、分组倍率。我把它写进 `PRICES.json` 的 `source` 字段，作为预留的依据 |
| 模型 ID | 如果 F1 列出的 ID 不是 `gpt-6.1-sol` 这样一眼能对上的名字，要由你确认用哪个，我不自行选择 |

**参考，不作为依据**：如果 Micu 按 OpenAI 标价（2 / 10 美元）收费，代入 S0 §6.1 的公式（`k = 1.3`、`N_in = 40,960`、`O_max = 512`），单次请求预留约 0.112 美元，超过"12 条轨迹都能预留"所需的 0.0728 美元。那样 45 美元只能预留约 7 条轨迹；推理 token 上限越大，能预留的条数越少。实际售价要等核实后再算。

## 4. 第 7B 步方案（提案，未执行）

前提是第 7A 步验收通过，并且 §3.3 的探测补跑完、结果也验收通过。

1. **云端准备**：
   - 系统盘只有 30 GB，Qwen3-8B 权重约 16 GB，空间不够。`instance_up` 只有在新建实例时才能扩盘，所以需要你决定：新建一台扩盘约 100 GB 的实例，或者另找存储。
   - 环境沿用第 1 步建好的 `itsm-train`。
2. **长度与显存**（GPU，上限 ≤ 3 GPU·h，S0 §6.2）：用 vLLM 加载 Qwen3-8B，分别以 `max_model_len` = 32,768 和 40,960 推理一条真实长度的 prompt（约 28k token）；记录显存、生成是否正常、长 prompt 的预填充耗时。40,960 只有在这一步实测通过后，才能从"候选"改为"已验证"，并且需要你验收。
3. **真实采样 logprob**：通过 verl 的 rollout server 让 `ItsmAgentLoop` 跑少量 episode，只用 train 中 GRPO 可训练的题和 dev 题，各 ≤ 2 道题、每题 1 条，**不更新权重**。核对四件事：
   - vLLM 返回的逐 token logprob 和 token 一一对应；
   - 同一批 token 用训练引擎重算的 logprob 与之相差多少（统计误差分布）；
   - 决定训练用 bypass 还是重算模式，写进协议；
   - 真实模型输出的工具调用格式能被 `parse_action` 正确解析，统计非法动作的比例和原因。
4. **用户模拟器的付费调用**：第 3 步会真实调用 `deepseek-v4-flash`，要在 7B 开始前按 §3 的同一套预留方式报批费用，预计每条 episode 不超过 40 次调用。
5. **不做**：held-out、SFT 轨迹、权重更新、Teacher（GPT-6.1 Sol）轨迹。Teacher 的轨迹生成属于 S1。

## 5. 运行记录

运行（在 `code/main` 下，日志在 `/home/sar/work/EnterGym/output/itsm-step7a/`）：v3、用途登记（哈希种子 1、7）、S0 审计、held-out 候选、6D-2 批次、v1、v2 的 `--check` 均为 up to date；两个冻结文件的 SHA256 与冻结值一致；`pytest tests/unit tests/remote -q`：773 passed（`pytest_full.log`），跑完没有残留子进程。
