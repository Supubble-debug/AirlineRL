# 第 6 步：数据划分与预算冻结（待验收）

本目录与下列文件一起构成第 6 步的产物。经用户验收后，它们才成为冻结协议；之后若要调整，需要说明原因并重新验收。

| 产物 | 位置 | 说明 |
|---|---|---|
| 划分 manifest | `configs/data/itsm_split_v1.json` | 任务 ID、家族、业务类型、tenancy、数据校验值、划分方法、各划分分布 |
| 家族与划分代码 | `src/agent_rl/data/itsm_split.py` | 家族规则、划分枚举、校验、held-out 访问保护 |
| 生成脚本 | `scripts/data/build_itsm_split.py` | 写出 manifest 和审计结果；`--check` 校验已提交文件可复现 |
| 家族审计原始结果 | `docs/itsm_step6/family_audit.json` | 每条连边的依据、点名近重复对、规则未覆盖的相似信号、dev/held-out 与 train 的残余重叠 |
| token 测量脚本 | `scripts/data/measure_itsm_tokens.py` | CPU，固定版本 tokenizer，只读 train/dev |
| token 测量原始结果 | `docs/itsm_step6/token_measurements.json` | 逐题、逐步数据和汇总（含本机计时，重跑时计时会变） |
| 预算配置 | `configs/itsm/budget_v1.yaml` | 训练与评测预算 |
| 测试 | `tests/unit/test_itsm_split.py`、`tests/unit/test_itsm_budget.py` | 划分可复现、互斥、覆盖、家族不跨划分、近重复同划分；预算与测量一致 |

## 1. 任务家族

### 1.1 规则

家族不是按任务 ID 手工划定的，而是按规则算出来的。

- **连边条件**（满足其一即连边）：
  - 两题 gold 动作所用写工具集合的 Jaccard 相似度 ≥ 0.6；
  - 两题都使用同一种"新建对象"的写工具（`create_problem`、`add_new_user`、`add_new_sla_definition` 等，完整列表见 manifest 的 `creation_tools`）。
- **家族**：上述连边图的连通分量。
- **人工部分**：只有家族名称和业务类型是人工标注的，按每个家族的第一个任务 ID 作为键。如果数据变动导致家族构成改变，代码会直接报错，不会悄悄沿用旧标注。

### 1.2 结果：11 个家族

| 划分 | 家族 | 业务类型 | 题数 | MSP / single | 任务 |
|---|---|---|---|---|---|
| train | change_problem_advance_with_kb | problem_change_lifecycle | 17 | 15 / 2 | msp_005–018, msp_031, single_011, single_015 |
| train | onboarding_and_catalog_admin | user_and_catalog_admin | 11 | 4 / 7 | msp_001–004, single_001–007 |
| train | ci_maintenance | cmdb | 1 | 0 / 1 | single_018 |
| dev | incident_hold_and_notify | incident_lifecycle | 12 | 7 / 5 | msp_019, 021, 022, 024, 025, 026, 029, single_008, 010, 014, 019, 020 |
| dev | incident_reassign_with_ci | incident_lifecycle | 1 | 0 / 1 | single_009 |
| heldout | problem_from_incident_and_change_mapping | problem_change_lifecycle | 5 | 2 / 3 | msp_027, msp_028, single_017, single_021, single_022 |
| heldout | incident_handoff_with_child | incident_lifecycle | 2 | 0 / 2 | single_012, single_013 |
| heldout | change_with_ci_and_kb_links | problem_change_lifecycle | 1 | 1 / 0 | msp_030 |
| heldout | incident_resolve | incident_lifecycle | 1 | 1 / 0 | msp_023 |
| heldout | knowledge_update_and_notify | knowledge | 1 | 0 / 1 | single_016 |
| heldout | service_retirement | service_catalog | 1 | 1 / 0 | msp_020 |

### 1.3 点名的近重复

| 对 | 写工具 Jaccard | 描述 Jaccard | 结论 |
|---|---|---|---|
| MSP-021 / 022 | 1.00 | 0.75 | 同一事件单 INC_054、同一 SLA、同一收件人，只是 persona 不同；同一家族，都在 dev |
| MSP-025 / 026 | 1.00 | 0.25 | 同一事件单 INC_006、同一来电人；同一家族，都在 dev |
| MSP-027 / 028 | 1.00 | 0.39 | 同一事件单 INC_020，都是从事件新建问题单；同一家族，都在 held-out |
| MSP-021 / single_019 | 1.00 | 0.08 | **跨 tenancy**：动作序列完全相同（挂起、暂停 SLA、发通知）；同一家族 |
| MSP-027 / single_017 | 0.25 | — | **跨 tenancy**：都从事件新建问题单，共用 `create_problem` 规则；同一家族 |

"挂起等待来电人、暂停 SLA、通知"这个工作流横跨两种 tenancy，共 12 题，所以整体放在同一个划分里。

### 1.4 规则没覆盖到的相似信号

完整列表见 `family_audit.json` 的 `unlinked_similarity_signals`。

- **任务间共享 ID 不等于共享对象。** delta 和新建工具会复用下一个顺序编号，例如 single_007、015、022 里的 CHG_004 是三张不同的、由 delta 新建的变更单。审计里另列了 `same_seed_entities`，只保留两边都指向同一条种子原有记录的 ID。
- **唯一同时涉及 train 和 held-out 的一对是 single_015（train）与 single_022（held-out）。** 两者没有共同的种子实体，但工作流部分相同：都是授权变更、把问题推进到根因、建立映射。写工具 Jaccard 为 0.5，低于阈值。
- **dev 与 held-out 之间有两对共享同一条种子事件单**：single_009 / 017，single_020 / 021。它们都不在训练集里，不构成训练泄漏。

### 1.5 残余重叠

`family_audit.json` 的 `residual_to_train` 记录了每个 dev/held-out 任务与训练集的最大写工具重叠。`update_incident`、`send_notification` 这类子操作在各划分之间普遍存在，这是 ITSM 任务的共性，按家族划分无法消除。held-out 里与训练集最接近的是 msp_030、single_017、single_021、single_022，写工具 Jaccard 为 0.5。

## 2. 划分

### 2.1 方法

- 对 11 个家族到三个划分的全部分配（3¹¹ 种）逐一枚举，保留满足约束的 381 种。
- **约束**：
  - 规模：train 26–34，dev 9–14，held-out 10–14；
  - dev 和 held-out 中每种 tenancy 至少 2 题；
  - dev 至少 2 个家族；
  - held-out 至少 5 个家族、至少 3 种业务类型，且任一家族不超过 held-out 的一半。
- **评分**：Σ|规模 − 目标| + 10 × Σ（dev、held-out 的 MSP 占比与总体 31/53 之差的绝对值）。目标规模为 train 29、dev 12、held-out 12。
- **并列处理**：最低分有 6 种并列，用 `random.Random(20261030)` 在排好序的并列方案中选定一种。

### 2.2 结果

| 划分 | 题数 | MSP / single | 家族数 | 业务类型 | 写入的对象 | gold 动作总数 |
|---|---|---|---|---|---|---|
| train | 29 | 19 / 10 | 3 | problem_change 17、user_and_catalog_admin 11、cmdb 1 | knowledge 17, change 16, problem 16, user 16, sla 9, cmdb 8, incident 5, location 5, notification 2, template 2, service 1 | 268 |
| dev | 13 | 7 / 6 | 2 | incident_lifecycle 13 | incident 13, notification 11, sla 9, cmdb 1, user 1 | 78 |
| held-out | 11 | 5 / 6 | 6 | problem_change 6、incident_lifecycle 3、knowledge 1、service_catalog 1 | incident 5, knowledge 5, change 4, notification 4, problem 4, cmdb 1, service 1 | 82 |

三份划分互斥，合起来恰好是 53 题，同一家族不跨划分。这几点都由测试验证。

### 2.3 需要你决定的取舍

53 题里有 40 题集中在三个家族（17、12、11 题）。按家族整体划分，无法同时做到 held-out 又大又多样、又让三个主要工作流都留在训练集。

| 方案 | train | dev | held-out | 问题 |
|---|---|---|---|---|
| **v1（本次提交）** | 29（2 个大家族 + 1） | 13（"挂起并通知"12 + 1） | 11，6 个家族，5 种业务类型 | 训练集没有"挂起并通知"工作流；dev 实际上只反映这一个工作流，用 dev 选 checkpoint 会偏向它 |
| 三个大家族都放训练集 | 40 | 6，5 个家族 | 7，3 个家族，其中一个占 71% | held-out 太小，只有约 3 个独立工作流，按任务汇总后不确定性很大 |
| 只按规模和 tenancy 打分（最初的版本） | 31 | 9 | 13，其中 12 题是"挂起并通知" | held-out 基本只有一个没见过的工作流 |

选 v1 的理由：最终结论依赖 held-out，任务数和工作流种类都要尽量多；dev 只用于调试和选 checkpoint。如果你更看重训练覆盖面或 dev 的多样性，改 `itsm_split.py` 里的约束常量后重新生成即可，整个过程可复现。

### 2.4 held-out 使用规则

- held-out 只用于最终冻结评测，不用于调试、调参或选 checkpoint。
- `split_task_ids("heldout")` 必须显式传入 `final_evaluation=True`，否则抛出 `PermissionError`。
- 本步的 token 测量只读取了 train 和 dev。

## 3. token 测量

### 3.1 测量设置

- **tokenizer**：`Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`。只下载了 tokenizer 和配置文件，没有下载权重；各文件的 sha256 记在结果里。
- **chat template**：sha256 `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`。
- **其他设置**：`enable_thinking=False`；transformers 5.21.0；`max_position_embeddings=40960`，不做 rope scaling。
- **prompt 构造**与现有 `TauAgentLoop` 一致：
  - 初始 prompt = `build_action_messages(policy, reset 观察)`，加上 94 个工具 schema 作为 `tools=`，再加生成提示；
  - 每个动作之后追加其生成 token；
  - 每条观察追加一个换行分隔，再加上单独渲染的一个 user 回合（用 `OBSERVATION_PROMPT` 包装，带生成提示）。
  - 注意：不能把整段对话重新渲染一遍来计数，因为 Qwen3 模板在重新渲染时会删掉历史回合里空的 `<think>` 块，算出的长度与 rollout 实际拼接的序列不一致。
- **重要限制**：
  - 数据来自 train/dev 共 42 题的脚本化 gold 回放，"生成 token"是 gold 工具调用经模板渲染后的长度，**不是实际策略输出的分布**；
  - 用户回复是固定的占位文本，没有调用用户模拟器；
  - 计时在本机 WSL2 的 CPU 上测得。

### 3.2 结果

| 项 | 数值 |
|---|---|
| policy 文本 | 7,794 token（32,016 字符） |
| 94 个工具 schema 块 | 20,051 token（最大的单个 schema：`list_incidents` 652） |
| system 消息（含工具） | 28,037 token |
| 初始 prompt | 28,104 token；各任务都一样，因为公开信息不随任务变化 |
| 每个 gold 工具调用的生成长度 | 中位数 32，p90 63，最大 127（`send_notification`） |
| 每条工具观察的回合长度 | 中位数 224，p90 360，最大 2,736（原始内容最大 2,690） |
| gold 回放的响应部分（生成 + 环境） | 中位数 2,513，p90 3,241，最大 5,764 |
| 完整序列 | 中位数 30,617，最大 33,868，已超过原生 32,768 |
| gold 回放的 step_count / agent_turns | 中位数 10，最大 18 |
| 不带过滤的 list/find 调用最坏情况 | `list_users` 10,371，`list_incidents` 9,656，`find_service_offerings` 8,014 |
| 本机耗时 | 单次工具调用 ≤ 0.9 ms；建环境 + reset ≤ 112 ms；fuzzy DB 评分 ≤ 26 ms |

## 4. 初版预算（`configs/itsm/budget_v1.yaml`）

| 项 | 值 | 依据 |
|---|---|---|
| 上下文 `max_model_len` | 40,960 | 等于模型的 `max_position_embeddings`。gold 回放的完整序列已有 33,868，超过原生 32,768 |
| `max_prompt_length` | 28,672 | 初始 prompt 28,104，余量 568 留给更长的真实首句 |
| `max_response_length`（整段响应） | 12,288 | 40,960 − 28,672；是 gold 回放最大值 5,764 的 2.1 倍 |
| `max_action_tokens`（单次生成） | 512 | 最长 gold 调用 127 的 4 倍，为发给用户的消息留余量（这部分没有测）。没有照搬 airline 的 256 |
| `max_steps`（底层步数） | 40 | gold 工具调用最多 16，加约 8 次真实用户回合，再乘约 1.7 留出出错重试的余量；按中位数单步成本（39 + 224 token）计算，约等于响应预算 |
| agent 决策轮数 | 不单独设上限 | 每个非终止动作都会消耗一步，所以 agent_turns ≤ max_steps 自然成立 |
| 单条观察上限 `max_observation_tokens` | 4,096（建议） | 大于 gold 最大观察 2,690；不设上限的话，一次不带过滤的 list 调用（约 1 万 token）就会耗尽响应预算。**截断机制尚未实现，留到第 7 步** |
| worker：用户模拟器 / 单次调用 / episode 超时 | 60 / 90 / 1800 秒 | 环境和评分都在毫秒级，主要耗时在用户模拟器的 HTTP 调用，这部分**待第 7 步实测** |
| 评分重试 | 3 次（共 4 次） | 协议 §12 |

训练（E1–E5）与评测（E0–E5）使用同一套 episode 预算，测试会检查两边一致。这样训练 reward 和评测 DB 分数的含义相同。

## 5. 未验证的边界

- **实际策略输出长度**：真实的消息长度、单次生成被截断的比例、最终使用的响应长度分布，都要在第 7 步用真实采样核实。
- **用户回合**：真实用户回复的长度和轮数没有测过，`max_steps=40` 里约 8 次用户回合是假设值。
- **时间预算**：用户模拟器 HTTP 的耗时、调用失败率、episode 总耗时都没有测，worker 超时取值待第 7 步验证。
- **观察截断**：`max_observation_tokens` 还只是建议值，截断方式和截断比例要在第 7 步实现和统计。
- **训练资源**：约 28k 的初始 prompt 加约 12k 的响应，每个样本接近 40k token。G=4 时的显存需求和单卡能否放下，要在第 9 步的 smoke 中实测，本步没有启动 GPU。
- **超出原生窗口**：完整序列超过 Qwen3 原生的 32,768，用的是 config 声明的 40,960 位置。这一段的模型质量本步没有验证。
- **缩短 prompt 的办法**：工具 schema 占 20k token，精简 schema 或按需提供工具都会改变公开接口，本步没有做，需要单独批准。
- **残余重叠**：single_015 与 single_022 的工作流部分重叠（见 1.4），以及子操作层面的普遍重叠（见 1.5）。
