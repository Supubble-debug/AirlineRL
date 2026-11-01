# 第 7A 步：离线 loop 验证与 API 兼容探测（待验收；7A-R1 离线补修已完成）

## 最终状态（2026-10-31，7A-R1 之后）

范围：只做第 7A 步和你要求的 7A-R1 离线补修。没有运行 held-out，没有生成 SFT 轨迹，没有更新权重；40,960 仍是待验证的候选值。

- **离线部分**：冻结记录、`ItsmAgentLoop` 契约验证、预留式控费账本、探测脚本都已完成。7A-R1 补了五处：
  1. loop 改为在已验收的进程 worker 中运行，补了取消和超时的集成测试；
  2. logprob 出现 NaN/Inf、缺失或条数不符时，按基础设施故障处理；
  3. 探测脚本的通过条件改严，失败一律停止并以非零码退出；
  4. 账本只允许一个写入者；
  5. D5 显式关闭思考。
- **API 探测**：在第 7A 步的预算内完成，共 **6 次付费请求，账本记账 0.028066 美元**（上限为 6 次、1 美元）。这 6 次已经全部用完，7A-R1 没有再发请求。
  - 账本记账是按你给的价格估算的，**不是**实际扣费。实际扣费无法核对，原因见 §3.7。
- **GPU**：第 7A 步为了发 HTTP 请求，以 GPU 模式开过两次 `exp-itsm`（你已批准），每次跑完都立即关机，没有在 GPU 上运行任何程序。7A-R1 没有开机。
- **已记录、尚未完成的**：
  - 真实采样 logprob、真实长度推理和显存，留到第 7B 步；
  - D5 关闭思考之后的真实参数复核，留到另一批 API 验证（需要另行批准预算）；
  - Teacher 输出上限**尚未触顶验证**（§3.7）。

下面的 §3.1–§3.6 是第 7A 步当时的过程记录，按时间保留，**其中的"未执行""0 美元"等说法只代表当时的状态**，以本节为准。

| 产物 | 位置 |
|---|---|
| 冻结记录 | `src/agent_rl/data/itsm_frozen.py`（提交号和两个文件的完整 SHA256）；进度文件 |
| ITSM agent loop | `src/agent_rl/rollout/itsm_agent_loop.py`（每个 episode 在 `ItsmEnvWorker` 子进程中运行） |
| loop 契约测试 | `tests/unit/test_itsm_agent_loop.py`（32 项：确定性采样替身，其中 3 项走真实 worker 子进程） |
| 预留式控费账本 | `src/agent_rl/utils/spend_ledger.py`（第 7A 步和 S1 共用；单一写入者） |
| 探测脚本 | `scripts/itsm/probe_api_7a.py`（只能在云端运行；默认 dry run） |
| 账本与探测测试 | `tests/unit/test_itsm_api_probe.py`（42 项，模拟 HTTP，不联网） |

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
- 环境用真实的 `ItsmEnv`、真实 Bench2 工具，用户是脚本化替身。契约测试为了速度在本进程里运行环境；取消、超时和一条完整 episode 的测试走默认路径，也就是真实的 `ItsmEnvWorker` 子进程（7A-R1）；
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
| logprob 字段 | mask = 1 的位置等于采样方给的值，来源记为 `sampler`；mask = 0 的位置是 0.0 占位，来源记为 `placeholder_masked`。采样方没有返回 token、**不返回** logprob、条数和 token 数对不上，或者出现 NaN、+Inf、−Inf 时，都抛出 `ItsmInfrastructureError(stage="sampler_contract")`，按基础设施故障处理：不产生 reward，**不补 0**（7A-R1，6 种情况） | 通过 |
| 不轮换上下文（D3） | 放不下下一条观察加一个动作时，以 `response_length` 结束；全程只有一个 request_id，prompt 不重写 | 通过 |
| 完整 episode | 第 1 题（887f5580）：9 个策略回合（4 个查询调用、说明拒绝关联已退役 CI 的计划消息、2 个写入调用、回顾消息、done），以 `agent_done` 结束，DB = 1 | 通过 |
| 进程 worker：取消（7A-R1） | 慢工具（`update_configuration_item`，在子进程中先睡 5 秒，再写标记文件，然后才执行）执行到一半时取消 rollout 任务。结果：任务以 `CancelledError` 结束，没有 `AgentLoopOutput`，也就没有 reward；返回时子进程已被杀掉并回收（`failure.stage = cancelled`，`execution_stopped = true`）；等过 5 秒后标记文件仍不存在，说明工具没有在后台完成延迟写入；`multiprocessing.active_children()` 为空，没有残留进程；之后再调 `score()` 会被拒绝。取消的是子进程本身，不只是等待它的线程 | 通过 |
| 进程 worker：超时（7A-R1） | 同一个慢工具，`call_timeout_s = 2`：抛出 `ItsmInfrastructureError(stage="timeout")`，没有 reward；子进程已被杀掉回收，标记文件不存在，没有残留进程 | 通过 |
| 进程 worker：完整 episode（7A-R1） | 默认路径（子进程）跑完第 1 题的 9 个回合：以 `agent_done` 结束，reward = 1，结束后子进程已退出 | 通过 |
| D5 参数（7A-R1） | 用假的 `litellm` 模块（不联网）截下 Bench2 用户模拟器的调用参数：loop 交给 worker 的配置，最终以 `deepseek/deepseek-flash`、温度 0、`extra_body={"thinking": {"type": "disabled"}}`、超时 60 秒传给 `litellm.completion` | 通过 |
| 按用途限制取题 | `grpo` loop 拒绝 SFT-only（b6ef60ca、78a5）、R1 pending（single_005）、dev 和 held-out 的题；`dev` loop 拒绝 train 题；held-out 只有 `final_evaluation` 用途能取 | 通过 |

**尚未覆盖、留到第 7B 步**：
- 真实 vLLM 采样能否返回逐 token 的 logprob；
- verl 训练实际用的是 rollout 返回的 logprob（bypass 模式，`old_log_probs = rollout_log_probs`），还是由训练引擎重算（仓库里两种配置都有：`grpo_sequence.yaml` 和 `grpo_sequence_bypass.yaml`）；
- 真实模型输出的工具调用格式；
- 32,768 / 40,960 长度下的推理和显存。

## 3. API 兼容探测（过程记录，按时间顺序）

§3.1–§3.3 是第 7A 步首次提交时（2026-10-31，付费探测之前）的记录，当时探测还没有执行；§3.4–§3.6 是之后的免费核对和付费探测；§3.7 是 7A-R1 的补修与决定。

### 3.1 首次提交时的停止原因（2026-10-31，后来已由 §3.4–§3.6 解决）

1. **没有不启动 GPU 的云端执行环境**：本项目唯一的云端实例 `exp-itsm`（pro-7933a95ce90d，RTX PRO 6000）处于关机状态。projtool 的 `instance_up` 只会以 GPU 实例开机，不支持 AutoDL 的"无卡模式"，所以要在云端发请求，就得启动 GPU，与\"不启动 GPU\"冲突。我没有开机。
2. **没有凭据**：本地和项目配置里都没有 `MICU_API_KEY` 或 Micu 的 base URL；云端是否已经配好，要开机才能看到。我没有向任何地方写入、读取或打印密钥。
3. **价格无法在请求前核实**：
   - `www.micuapi.ai` 是 New API 面板，首页是前端渲染的，看不到内容；`/api/pricing` 返回 401，要登录才能看价格。
   - 公开资料只有第三方站点列出的 OpenAI 官方标价（`gpt-6.1-sol`：输入 2 美元、输出 10 美元/百万 token），**不是 Micu 的实际售价**，不能用来做预留。
   - 按你的规则，计费核实之前不能发出付费请求。

### 3.2 首次提交时准备的探测设计（2026-10-31；通过条件已在 7A-R1 中改严，见 §3.7）

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

### 3.3 首次提交时请你决定的事（2026-10-31；你的决定见 §3.4–§3.6）

| 项 | 选项 |
|---|---|
| 云端执行环境 | ① 你在 AutoDL 控制台把 `exp-itsm` 以**无卡模式**开机，告诉我后我用 `remote_exec` 运行（不算启动 GPU，费用很低）；② 明确允许以 GPU 模式开机、只用于发 HTTP 请求，跑完立即关机；③ 改为在本机运行（与你\"在云端核对\"的要求不同，需要你明确同意） |
| 凭据 | 在云端设置环境变量 `MICU_BASE_URL`、`MICU_API_KEY`、`DEEPSEEK_API_KEY`（例如写入只有 root 可读的 `/root/.itsm_api_env`）。不要贴到对话里 |
| Micu 价格 | 提供 Micu 控制台里 GPT-6.1 Sol 的价格依据（截图或价格页导出均可）：输入单价、输出单价（是否包含推理 token）、有无按次费用、币种与汇率、分组倍率。我把它写进 `PRICES.json` 的 `source` 字段，作为预留的依据 |
| 模型 ID | 如果 F1 列出的 ID 不是 `gpt-6.1-sol` 这样一眼能对上的名字，要由你确认用哪个，我不自行选择 |

### 3.4 免费核对结果（2026-10-31，本机，只发 GET，未扣费）

你的决定：免费查询在本机跑；付费探测允许以 GPU 模式开机，跑完立即关机；价格以价格接口为准，查不到就停，等你提供截图。

凭据来自 `code/main/micu.md`（一个 API key 加网址 `https://www.micuapi.ai`）。该文件没有被 git 跟踪，已加入本地的 `.git/info/exclude`，权限改为 600；key 没有出现在任何输出或日志里。原始返回（key 已替换掉）存放在 `/home/sar/work/EnterGym/output/itsm-step7a/free_checks/`，不入 git。

| 查询 | 结果 |
|---|---|
| `GET /api/status`（公开） | 站点为 New API 面板（\"米醋API\"），额度显示单位为 CNY，`quota_per_unit=500000`；**不含**按模型的价格或倍率 |
| `GET /v1/models`（用 key） | 列出 `gpt-6.1-sol`，与请求的 ID **完全一致**；同时列出 `gpt-6-sol`、`gpt-5.6-sol`、`gpt-6-astra`、`codex-auto-review`。`gpt-6.1-sol` 的 `supported_endpoint_types` 只有 `openai`，是否支持 Responses API 仍要靠 B1/B2 实测 |
| `GET /api/pricing`（用 key） | **拒绝**：`Unauthorized, invalid access token`。这个接口要的是账号的系统访问令牌，和 API key 不是同一个凭据 |
| `GET /v1/dashboard/billing/subscription`、`/usage`（用 key） | 成功。该 key 的额度上限远高于第 7A 步的 1 美元预算，**不能**拿它当控费手段，只能靠我们自己的预留账本。usage 总量可以在探测前后各读一次，独立核对实际扣费 |

**结论**：模型 ID 已确认；**价格仍无法核实**，按你的规则停下。没有开机，没有发出付费请求，花费 0 美元。

**继续需要以下二者之一**：
- Micu 控制台中 `gpt-6.1-sol` 的价格信息（截图或数字），以及 `deepseek-v4-flash`、`deepseek-v4-pro` 的 DeepSeek 官方价格：
  - 输入单价、输出单价，推理 token 是否按输出计费；
  - 有无按次附加费；
  - 分组倍率，以及 CNY 与 USD 怎么换算；
- 或者账号的系统访问令牌，用来查询价格接口。同样写进本地文件，不要贴到对话里。

拿到价格后：写入 `PRICES.json`（标明来源），先本地 dry run 核对预留合计不超过 1 美元，再以 GPU 模式开机，在云端运行 `probe_api_7a.py --confirm-billing`，跑完立即关机，最后用 usage 前后读数核对账本。

### 3.5 付费探测结果（2026-10-31，云端，已按停止条件停止）

**你的决定**（原话）：\"改用5.6 sol影响大吗，输入5人民币/M，输出30人民币/M\"、\"long context模式价格要乘以2.跑吧\"。据此：
- Teacher 改为 `gpt-5.6-sol`（D-S0-5 更新）；
- 预留按乘 2 后的价格（输入 ¥10/M、输出 ¥60/M），按 6.5 人民币兑 1 美元的保守汇率折算，并假定推理 token 按输出计费；
- 价格文件存在 `output/itsm-step7a/run/prices.json`，不入 git。

**DeepSeek 价格依据**：DeepSeek 官方价格页（2026-10-31），取高峰价、按不命中缓存计：
- v4-pro：输入 $1.32/M，输出 $3.96/M；
- flash：输入 $0.30/M，输出 $1.20/M。

**发现 1（影响 D5）**：官方价格页写明 `deepseek-v4-flash` **已下线**。用这个旧名发出的请求仍会被接受，但实际由 `DeepSeek-V4.1-Flash-1017` 提供服务并计费。D5 固定的用户模拟器，已经在服务商那边被替换了。
- 为了让模拟器的模型不一致不连带影响裁判核对，我把裁判探测（B5）排到模拟器（B4）之前。
- 但云端和本机都**没有 `DEEPSEEK_API_KEY`**，B4、B5 都没有运行（记为\"not run: no DeepSeek credential\"），没有发出请求。

**执行过程**：
- 以 GPU 模式开机（你已批准），只用来发 HTTP 请求，跑完立即关机，平台确认已为 shutdown。
- 凭据通过 ssh 标准输入写入云端 `/root/.itsm_api_env`（600 权限，只有 root 可读）；结束后删除了凭据文件和临时目录。
- 凭据文件已从代码目录移到 `/home/sar/work/EnterGym/.secrets/micu.md`。原位置在代码同步范围内，会被复制到云端 `/workspace`。
- 本机和云端的 dry run 都只调用了免费的模型列表，`gpt-5.6-sol` 在列表中，计划预留合计 0.009856 美元。

**付费请求**：只发出 **1 次**（B1：Responses 纯文本）。
- 计算出的实际费用 0.006805 美元，超过了这次请求的预留 0.002629 美元，触发\"实际超过预留即停\"。B2、B3 没有发出。

| 项 | 结果 |
|---|---|
| HTTP | 200，Responses API 可用 |
| usage | input 4,393（其中 `request_fields.instructions` 占 **4,380**，我们发出的消息只有 13）；output 5；reasoning 0 |
| 隐藏指令 | **发现 2**：每个请求都被注入了约 4,380 token 的 `instructions`，我们请求里没有这一项。这些指令来自 Micu 或它的上游，内容我们看不到 |
| 返回的模型 ID | 没有保存：停止发生在结算环节，旧代码在记录结果之前就抛出了异常。已修正为先记录再停止，并补了测试；没有为了补这一项再发付费请求 |
| 账本记账 | 0.006805 美元（约 ¥0.044，按乘 2 价格计） |
| 账号用量的实际增量 | 1.1058（面板单位\"分\"），约 **¥0.011**，**低于账本记账**，说明账本偏保守（这条核对后来作废：账号用量混有他人用量，见 §3.6、§3.7）。这个数是按乘 1 价格算出的 ¥0.022 的一半左右，原因可能是分组倍率或缓存折扣，从用量读数无法区分；同一账号在这段时间里的其他用量也排除不了 |

**发现 2 的影响**：
1. **预留公式**：\"请求字节数 ≥ 计费 token 数\"这个上界对 Micu 不成立。今后每次请求的预留，要加上隐藏指令的开销。按实测，至少要加 4,400 token，并留余量（例如 8,192），而且要先实测不同端点、带工具调用时的开销。对 S1 的单次请求（约 28k–41k 输入）来说，这项只多出约 10%–30%，不会改变 \"45 美元内能否预留 12 条\" 的结论。
2. **Teacher 的输入边界**（S0 §4.2 规定 Teacher 只看到本项目的 policy、工具 schema 和对话）：现在每个请求都多了一段我们看不到的系统指令，这一条不能满足。在 S1 之前需要你决定：接受并记录这项偏差；要求 Micu 提供不注入指令的通道；改走 Chat Completions，看是否同样被注入（B3 还没测）；或者换用其他供应商。

**本次花费**：
- API：1 次付费请求，账本 0.006805 美元，账号实际约 ¥0.011；第 7A 步预算 1 美元，剩余 0.993 美元，可用请求还剩 5 次。
- GPU：exp-itsm 从开机到关机不到约 15 分钟（未精确计时），只用于发请求，没有在 GPU 上运行任何程序。

**还没完成的核对**：
- B2 Responses 工具调用、B3 Chat 工具调用；
- Teacher 实际返回的模型 ID；
- B4 模拟器（需要 DeepSeek key，并且要你先决定 D5 怎么处理）；
- B5 裁判（需要 DeepSeek key）。

### 3.6 补完的付费探测（2026-10-31，云端）

**你的决定**（原话）：\"那就deepseek吗，api在deepseek.md\"。据此：
- D5 模拟器继续用 DeepSeek，改为现行名称 `deepseek/deepseek-flash`（DeepSeek-V4.1-Flash）；
- ITSM 的代码默认值、`budget_v1.yaml` 和协议 D5 都已同步。airline/tau2 的配置仍写着旧名，那是原 airline 协议的一部分，没有改动，但同样受到下线影响。

**凭据**：`deepseek.md` 已从代码目录移到 `/home/sar/work/EnterGym/.secrets/`（600 权限）。上传到云端时用的是只有 root 可读的文件，跑完已删除，实例已关机，平台确认为 shutdown。

**免费核对**（本机）：
- DeepSeek 的 `/models` 列出 `deepseek-flash`（DeepSeek-V4.1-Flash）和 `deepseek-v4-pro`；
- 账户余额 **¥5.92**，探测前后没有变化：两次调用约 0.00004 美元，低于余额的显示精度。

**付费探测结果**：沿用累计账本，**总计 6 次请求、0.028066 美元**，没有超过第 7A 步的 6 次、1 美元上限，没有触发任何停止条件。Teacher 的预留另加了 8,192 token 的隐藏指令余量。

| 探测 | 返回的模型 | 结果 |
|---|---|---|
| B1 Responses 纯文本 | `gpt-5.6-sol` | 正常；input 4,393（隐藏指令 4,380），output 5，推理 0；本次计费 0.006805 美元 |
| B2 Responses 工具调用 | `gpt-5.6-sol` | **恰好一个 function_call**，工具名正确，参数是 JSON 对象；input 4,540，output 24，推理 0 |
| B3 Chat 工具调用 | `gpt-5.6-sol` | 恰好一个 tool_call，`finish_reason=tool_calls`；prompt 4,540，与 B2 相同，说明 **Chat 接口同样被注入了隐藏指令** |
| B5 裁判 `deepseek-v4-pro`（D7 参数） | `deepseek-v4-pro` | 服务端接受关闭思考，没有 `reasoning_content`，`finish_reason=stop`，满足 D7 第 7/8 步核对的全部条件 |
| B4 模拟器 `deepseek-flash` | `deepseek-flash` | `finish_reason=stop`；**默认开启思考**，响应带 `reasoning_content` |

**结论与需要决定的事**：
1. **Teacher 接口兼容**：`gpt-5.6-sol` 的 Responses 和 Chat 两条路径都支持单个工具调用，返回的模型 ID 与请求一致，简单请求没有产生推理 token。（7A-R1 更正：原来写的"输出上限有效"不成立。实际输出只有 5 和 24 token，远低于 256 的上限，所以上限是否被执行**尚未触顶验证**，见 §3.7。）
2. **隐藏指令**：两条路径都有约 4,380 token 的隐藏指令，改走 Chat 接口避不开。按此前的说明，它不会进入学生的训练输入；在 S1 中需要把它记为偏离 S0 §4.2 的偏差，每条轨迹都记录返回的模型 ID 和输入 token 数，并把 S1 预留的输入上界加上 8,192 token。是否接受这项偏差，待你决定。
3. **模拟器的思考模式**：deepseek-flash 默认开启思考。开启时每个用户回合会多出推理 token，带来额外的费用和延迟，也可能让回复更不稳定。建议与裁判一样，显式传 `extra_body={\"thinking\": {\"type\": \"disabled\"}}` 关闭，具体由你决定；选定后写入 D5，并在第 7B 步复核。
4. **DeepSeek 余额只有 ¥5.92**：够第 7B 步少量采样用，但第 8 步基线和 GRPO 训练之前必须充值。
5. **Micu 账号用量**：探测期间 Micu 的用量增加了约 ¥0.51，其中混有同一账号上其他人的用量，**无法**用来核对我们的扣费。第 3.5 节里\"实际约 ¥0.011\"那条核对也因此作废。费用以本地账本为准；账本按乘 2 价格和保守汇率记账，偏保守。

**参考，不作为依据**：如果 Micu 按 OpenAI 标价（2 / 10 美元）收费，代入 S0 §6.1 的公式（`k = 1.3`、`N_in = 40,960`、`O_max = 512`），单次请求预留约 0.112 美元，超过"12 条轨迹都能预留"所需的 0.0728 美元。那样 45 美元只能预留约 7 条轨迹；推理 token 上限越大，能预留的条数越少。实际售价要等核实后再算。

### 3.7 7A-R1 的补修与记录的决定（2026-10-31，离线，未发请求）

你的决定（原话节选）："接受 Micu 未知额外输入作为 S1 试点的记录偏差；不把观察到的 token 开销当固定上界。用户提供的价格可用于估算，账本费用与实际扣费分开写。输出上限标为尚未触顶验证。"

**探测的通过条件（脚本已改，用模拟 HTTP 测试，没有新增付费请求）**

现在每一项检查都是必需的，下面任何一种情况都会立即停止，写出报告，并以非零码退出。只有全部探测都跑完、所有必需检查都通过时，才以 0 退出。

| 情况 | 例子（都有对应测试） |
|---|---|
| 非 200 | 500、503、429 |
| 返回体不是 JSON 对象 | text/html |
| 缺失结果 | 没有 usage；Responses 没有 output；Chat 的 choices 为空；DeepSeek 没有 choices 或 usage |
| 错误工具 | 调用了 `list_incidents` / `get_incident` |
| 参数不符 | 参数必须**正好**是 `{"number": "INC0000007"}`：号码不同、多出参数、空对象、不是 JSON，都算失败 |
| 调用个数不对 | 两个 function_call；Chat 只回了文字 |
| 其他必需检查为 false | Chat 的 `finish_reason` 不是 `tool_calls`；裁判或模拟器的 `finish_reason` 不是 `stop`、内容为空、带 `reasoning_content` 或推理 token |
| 原有的停止条件 | 模型没有列出或对不上、输出超过上限、实际费用超过预留、预算用尽 |
| 意外异常 | 解析时出现任何异常 |

另外两处改动：
- 报告里现在会记录返回的工具名和参数，便于事后复核；
- 模拟器探测（B4）会发送 `thinking: disabled`。

**用新条件回看第 7A 步已发出的 6 次请求**（不补发请求，只说明哪些能确认、哪些不能）：

| 探测 | 按新条件 |
|---|---|
| B1 | 通过（有文本输出） |
| B2 | 个数、工具名、参数是 JSON 对象，都已确认。**参数值没有确认**：当时的报告丢弃了原始返回，只保存了检查结果 |
| B3 | 个数和 `finish_reason=tool_calls` 已确认。**工具名和参数值没有确认**，原因同上 |
| B5 裁判 | 不带 `reasoning_content`、`finish_reason=stop`，已确认；回复内容当时没有保存 |
| B4 模拟器 | **按新条件不通过**：带 `reasoning_content`，推理 token 有 10 个。这是因为当时请求没有关闭思考，也正是 D5 现在改为显式关闭的原因 |

这些没确认的项，以及 D5 关闭思考之后的真实行为，都要在另一批 API 验证中复核。第 7A 步的 6 次额度已经用完，账本会拒绝第 7 次请求（有测试）。那一批要另行批准，并使用新的预算 id 和新的账本。

**账本的单一所有者和并发边界**
- 一个账本文件只能有一个写入者。打开账本时，会对 `<账本>.lock` 加独占、非阻塞的 `flock`，一直持有到 `close()` 或进程退出。
- 同一路径上的第二个 `SpendLedger`（同一进程或同一主机上的另一个进程）会直接被拒绝（`LedgerBusy`），不会读进过时的快照后多预留。
- 前一个写入者关闭后，新的写入者会重新读取文件，看到所有已有的预留。
- 同一个写入者内部，用线程锁让预留和结算依次进行。
- 回归测试：同一进程里两个实例、两个进程，各想预留 0.75 美元、总上限 1 美元——第二个要么被拒绝打开，要么重读后因超出上限被拒绝，总计只预留了 1 次。
- 锁只在本机有效：账本只能在一台机器上写，同步到别处的副本只当记录看；只读汇总用 `SpendLedger.read_summary`。

**D5 关闭思考**
- 用户模拟器保持 `deepseek/deepseek-flash`，温度 0，显式传 `extra_body={"thinking": {"type": "disabled"}}`，与 D7 裁判走同一条路径。
- 已同步到 ITSM 代码默认值、loop 设置、`budget_v1.yaml`（`user_llm_thinking: disabled`）和协议 D5。
- 本步只做了离线验证（假的 litellm 截下参数）。服务端是否真的关闭了思考，留到另一批 API 验证。

**Micu 的隐藏指令：作为 S1 试点的记录偏差接受**
- 这是对 S0 §4.2 输入边界的偏离：Micu 或它的上游会在请求中加入我们看不到的指令，已作为 S1 试点的**记录偏差**接受。
- 第 7A 步观察到的约 4,380 token **只是两次观察值，不是固定上界**，内容和长度都可能变。所以：
  - S1 每条轨迹都要记录返回的模型 ID、计费的输入 token 数，以及它减去请求体上界后的差额（也就是观察到的额外开销）；
  - 预留仍按"请求体上界 + 余量"计算，但不能把 8,192 当作保证。任何一次请求的实际费用超过预留，就立即停止（账本已经这样做）；
  - 由于额外开销没有上界，最后一次请求可能超出它的预留。在 S1 的方案里，我会提议在总上限下面留一段缓冲来吸收这部分超出，由你决定。

**费用：账本记账和实际扣费分开写**

| 项 | 数值 | 说明 |
|---|---|---|
| 账本记账（估算） | **0.028066 美元**，6 次 | Teacher 按你给的价格（输入 ¥5/M、输出 ¥30/M，长上下文模式乘 2）估算，推理 token 按输出计，按 6.5 人民币兑 1 美元换算；DeepSeek 按官方价格页的高峰价、不命中缓存计 |
| Micu 实际扣费 | **无法核对** | 同一账号上还有别人的用量，面板读数无法拆分；§3.5 里"实际约 ¥0.011"那条核对作废 |
| DeepSeek 实际扣费 | **无法核对** | 余额 ¥5.92，探测前后显示不变，低于显示精度 |

账本按保守价格估算，只用来控费，不等于实际账单。

**输出上限：尚未触顶验证**
- Teacher 的上限是 256，实际输出只有 5 和 24 token；裁判和模拟器的上限是 32，实际只有 1 和 12 token。没有一次请求接近上限，所以：
  - 服务端会不会在上限处截断，没有验证；
  - 推理 token 是否计入上限，也没有验证。
- 脚本在输出超过上限时会停止，但这只是事后发现。S1 的费用预留不能假定上限一定生效，要靠"实际超过预留就停止"作为兜底。

## 4. 第 7B 步方案（修订于 7A-R1，提案，未执行）

前提：第 7A 步（含 7A-R1）验收通过，并且你另行批准第 7B 步。**不自动扩盘，不新租实例**；空间不够时停下来报告，由你决定。

**1. 存储方案（只用 `exp-itsm` 现有的磁盘）**

已知情况（第 1 步时记录）：
- 系统盘 30 GB，没有独立数据盘；装完环境后已用 12 GB（其中 `itsm-train` 环境 9.9 GB），剩余约 19 GB；
- 共享的 `/root/autodl-fs` 只剩约 4.5 GB，大部分被其他项目占用，不碰。

需要的空间：
- Qwen3-8B 固定版本 `b968826d` 的 safetensors 约 16 GB，以下载时列出的文件大小为准；
- 第 7B 步不保存 checkpoint，输出只有 JSON 日志和指标，在 MB 量级。

做法：
- 开机后先运行 `df -h` 和 `du`，确认剩余空间。只有剩余空间不少于"权重大小 + 2 GB"时才下载。
- 下载时用 `local_dir` 直接写到 `configs/runtime` 里已有的路径 `/root/autodl-tmp/models/Qwen3-8B`，不在 HF 缓存里再存一份。vLLM 和 torch 的编译缓存放在系统盘上，并记录它们的大小。
- 空间不够时，可以清理本项目自己的 `conda`/`pip` 下载缓存（不动环境本身），这一步要你先同意。清理后还不够，就停下报告。
- 这个方案解决不了第 9 步保存 checkpoint 的空间问题，那要单独决定。

**2. 实际长度测试**（GPU，第 7B 步总上限 3 GPU·h，S0 §6.2）

输入只用 train 和 dev 题的真实状态，不用 held-out；渲染方式与 `ItsmAgentLoop` 相同（同一个模板、同样的工具 schema、`enable_thinking=False`）。三档长度：

| 档 | 内容 | 长度 |
|---|---|---|
| a | 最长的真实初始 prompt（第 6 步实测）加一条简短的首个用户回合 | 约 28.1k |
| b | a 加上第 6 步 gold 回放中真实的环境观察和 gold 动作，在回合边界截断 | 不超过 32,768 − 512 |
| c | 继续用同一题的真实工具调用和观察往后接 | 约 40,448 |

c 档的内容是真实的，但它不是一条自然的 episode：gold 回放最长的 response 只有 5,764 token，自然 episode 达不到这个长度。c 档只用来测显存，以及生成是否正常。

测法：
- `max_model_len` = 32,768 时跑 a、b 两档；= 40,960 时跑 a、b、c 三档。
- 单并发，温度 0，每次最多生成 512 token。
- 记录：能否正常启动；vLLM 报告的 KV cache 容量；峰值显存；预填充耗时；输出 token 数；输出能否被 `parse_action` 解析；每个 token 是否都有一个有限的 logprob。
- **两个设置之间的一致性**：同一个 a、b 输入，在两种 `max_model_len` 下生成的 token 和 logprob 应当一致。
- **32k 以后的质量**：用 teacher forcing 计算 b、c 两档末尾那段真实 gold 动作的平均 logprob，并排列出来。差多少才算 40,960 不可用，我不自己定阈值，报告数字后由你判断。40,960 只有在你验收这些结果之后，才从"候选"改为"已验证"。

**3. 真实采样 logprob**
- 通过 verl 的 rollout server 让 `ItsmAgentLoop` 跑少量 episode：GRPO 可训练题和 dev 题各不超过 2 道，每题 1 条，**不更新权重**。
- 用户回合用 gold 回放的脚本化用户，**不调用付费 API**。
- 核对四件事：
  - logprob 与 token 一一对应，并且都是有限值；
  - 训练引擎重算的 logprob 与之相差多少（统计误差分布）；
  - 训练用 bypass 还是重算模式，定下来写进协议；
  - 真实模型输出的工具调用能否被 `parse_action` 正确解析，统计非法动作的比例和原因。

**4. 另一批 API 验证（与第 7B 步分开报批）**
- 内容：D5 关闭思考后的真实复核，以及 §3.7 里没确认的 B2/B3 参数值。用新的预算 id 和新的账本。
- DeepSeek 余额只有 ¥5.92，第 8 步之前必须充值。

**5. 不做**：held-out、SFT 轨迹、权重更新、Teacher 轨迹（属于 S1）。

## 5. 运行记录

运行（在 `code/main` 下，日志在 `/home/sar/work/EnterGym/output/itsm-step7a/`）：v3、用途登记（哈希种子 1、7）、S0 审计、held-out 候选、6D-2 批次、v1、v2 的 `--check` 均为 up to date；两个冻结文件的 SHA256 与冻结值一致；`pytest tests/unit tests/remote -q`：773 passed（`pytest_full.log`），跑完没有残留子进程。

**7A-R1 重跑（2026-10-31，本机，离线）**：
- 两个冻结文件的 SHA256 与冻结值一致，`verify_frozen()` 通过；
- `test_itsm_agent_loop.py` 32 项、`test_itsm_api_probe.py` 42 项全部通过；
- 全量 `pytest tests/unit tests/remote -q`（不设 API key 环境变量、`HF_HUB_OFFLINE=1`）：**813 passed**，退出码 0（`pytest_full_r1.log`），跑完没有残留子进程。
  - 第一次全量运行中，`test_itsm_env.py` 有 1 项失败：它断言的是旧的 D5 参数（没有 `extra_body`）。按 D5 的新决定更新这项断言后重跑，全部通过；第一次运行的日志是 `pytest_full_r1fix.log`。
- 没有调用 LLM，没有启动 GPU，没有新增付费请求。

