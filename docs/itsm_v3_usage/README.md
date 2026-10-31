# v3 manifest 与用途登记 `itsm_usage_v1`（冻结待验收）

依据：2026-10-31 你的决定：
- S0-R1 验收通过；
- 五道派生题暂停 GRPO；
- 78a5 保留为 SFT-only 来源；
- 6D-3 验收通过，54f35603 纳入 held-out；
- 生成 v3 manifest 与用途登记，提交冻结验收。

本步只生成文件和离线验证：v1/v2 没有改动；没有开始第 7 步或 S1；没有调用 LLM，没有启动 GPU，没有训练。

| 产物 | 位置 |
|---|---|
| v3 manifest | `configs/data/itsm_split_v3.json`（sha256 `9326167d…`） |
| 用途登记 | `configs/data/itsm_usage_v1.json`（sha256 `87f15baf…`） |
| 构建与校验 | `src/agent_rl/data/itsm_split_v3.py`、`src/agent_rl/data/itsm_usage.py`；`scripts/data/build_itsm_split_v3.py [--check]`、`scripts/data/build_itsm_usage.py [--check]` |
| 数据入口 | `src/agent_rl/data/itsm_task_sets.py`：新增 `itsm_split_v3`、`training_task_ids(purpose=..., usage_version=...)`、`load_usage(...)` |
| 测试 | `tests/unit/test_itsm_split_v3.py`（17 项） |

## 1. v3 manifest

| 划分 | v2 | v3 | 新增 |
|---|---|---|---|
| train（训练来源并集） | 32 | **37** | 6D-2 获批的 4 题：`99ba2325_887f5580`、`01395ded_eff85dae`、`46ca2862_56a4f834`、`99ba2325_7915481a`；SFT-only 来源 `78a5b71e_a4fb07a9` |
| dev | 13 | **13** | 无，列表与 v2 完全相同 |
| held-out | 11 | **12** | `99ba2325_54f35603` |

- **不移动原有任务**：v2 的 56 题全部保持原划分，校验会逐题检查，也会对整个文件做重建比对。
- **v3 的 train 不是 GRPO 集合**：它只表示"可以作为训练来源"。每题能用于 SFT 还是 GRPO，只记在用途登记里（§2）。manifest 的 `usage` 字段也写明了这一点。
- **逐题哈希**：每题记录两个哈希。
  - `content_sha256`：只覆盖决定环境和评分的字段，口径与 v2 相同；
  - `source_sha256`：覆盖该题在来源文件中的完整条目，包括来源信息、上游验证器和反例。
  - 派生题另外记录上游 task_id、上游 prompt 和种子的 sha256，以及来源文件，并写明 `equivalent_to_upstream_evaluation: false`。
- **固定的输入**：`data` 段记录 Bench2 数据、试点文件、6D-2 批次文件、held-out 候选文件，以及 v1、v2 manifest 的 sha256。其中任何一项变化，v3 都会校验失败。

### 1.1 家族（规则同 v1/v2，在 62 题上重算）

v2 有 11 个家族，v3 有 15 个：
- **成员不变的 10 个**：沿用 v2 的名称。
- **合并家族扩展**：`46ca2862_56a4f834`（新建 offering）和 `99ba2325_7915481a`（新建 KB）并入原合并家族，改名为 `onboarding_catalog_change_problem_kb_v3`，共 32 题，全部在 train。
- **新增 4 个单题家族**：
  - `ci_retire_and_affected_link`（887f5580，train）
  - `notification_queue_repair`（eff85dae，train）
  - `change_post_implementation_review`（78a5，train）
  - `service_hold_instead_of_retirement`（54f35603，held-out）

| 划分 | 家族数 | 构成 |
|---|---|---|
| train | 6 | 合并家族 32 题，另外 5 个家族各 1 题 |
| dev | 2 | `incident_hold_and_notify` 12 题、`incident_reassign_with_ci` 1 题 |
| held-out | 7 | `problem_from_incident_and_change_mapping` 5 题、`incident_handoff_with_child` 2 题，另外 5 个家族各 1 题 |

没有任何家族跨越两个划分。

### 1.2 来源、实体与近重复隔离（重算）

训练侧 37 题，评测侧 25 题（dev 13 + held-out 12）：

| 检查 | 结果 |
|---|---|
| 没有家族跨划分 | 通过 |
| 涉及派生题的训练 × 评测题对：共享写工具 ≤ 1，共享同种子 gold 实体 < 2 | 通过（0 对超出） |
| 训练与评测之间没有近重复（同种子、写工具集合相同、共享 ≥ 2 个 gold 实体） | 通过 |
| 同一上游题不同时出现在训练和评测两侧 | 通过 |

Bench2 原 train 题与评测题之间，有 118 对共享 ≥ 2 个写工具。这是 v1 就写明的"子工作流跨划分重新组合"，只计数，不阻断，与 S0 审计的数字一致。

## 2. 用途登记 `itsm_usage_v1`

| 池 | 题数 | 内容 |
|---|---|---|
| SFT 来源 | **37** | v3 train 全部，来源已获准 |
| GRPO 候选 | **31** | 未暂停 GRPO 的 train 题 |
| ├ GRPO 可训练 | **26** | 只有这个池能被 GRPO 训练入口选中 |
| └ R1 pending | **5** | `single_005/006/007/011/015`：第 9 步前补完 R1，**在此之前不能进入 GRPO 训练** |
| SFT-only | **6** | 五道暂停 GRPO 的派生题（`b6ef60ca`、`52c8290e`、`1d219d93`、`eff85dae`、`7915481a`），以及 `78a5`（`never_grpo`） |
| dev / held-out | 13 / 12 | 都不是训练来源 |

可训练的 26 题分属 3 个家族：合并家族 24 题，`887f5580`、`single_018` 各 1 题。

**"来源获准"和"轨迹经审核接纳"分开登记**：
- 每题的 `source_approval` 记录 SFT、GRPO 两种用途是否获准；GRPO 一栏写出具体状态（approved、pending_r1_before_step_9、suspended_reward_blind_spot 或 never_grpo）。
- 接纳轨迹记在独立的 `trajectory_acceptance` 段，目前为空（尚未生成或审核任何轨迹），并列出将来逐条要记录的字段：轨迹 id、来源哈希、Teacher 供应商与返回的模型 ID、采样参数、最终 DB 哈希、硬性检查结果、审读结论、审读人和日期。
- 来源获准**不等于**接纳了任何轨迹。

**与 S0 的一致性**：用途登记引用已验收的 S0 审计文件（记录其 sha256），并逐项校验 SFT-only、R1 pending、GRPO 候选、SFT 池与 S0 结论一致。任何一项不一致，构建或校验都会失败。

## 3. 数据入口按用途区分

| 调用 | 行为 |
|---|---|
| `task_ids("train", split_version="itsm_split_v3")` | **拒绝**（PermissionError）：v3 train 混有 SFT-only 和 R1 pending 的题，不能直接当作训练集 |
| `training_task_ids(purpose="grpo", usage_version="itsm_usage_v1")` | 只返回 GRPO 可训练的 26 题；不含 SFT-only、暂停或 R1 pending 的题，也不含 dev 或 held-out |
| `training_task_ids(purpose="sft", usage_version="itsm_usage_v1")` | 返回 37 个 SFT 来源 |
| 未知的 `purpose` 或 `usage_version` | 拒绝；两个参数都必填，没有默认值 |
| `task_ids("heldout", split_version="itsm_split_v3")` | 拒绝，除非传 `final_evaluation=True`（与 v1/v2 相同） |
| `resolve_task(..., split_version="itsm_split_v3")`、`ItsmEnvConfig(split_version="itsm_split_v3")` | 62 题都能解析，每道派生题都按 v3 的 `content_sha256` 核对 |

v1/v2 的入口行为不变。GRPO 训练入口（第 9 步）尚未实现；到时只能通过 `training_task_ids(purpose="grpo", ...)` 取题，这一点会写进第 9 步的验收条目。

## 4. 解析与过滤验证（`tests/unit/test_itsm_split_v3.py`，17 项）

- v1、v2 文件的 sha256 不变。
- v3 规模为 37/13/12；dev 与 v2 相同；held-out 等于 v2 加 54f35603；train 等于 v2 加 5 题；v2 的每一题划分不变。
- 家族数（6/2/7）、隔离检查、每题两个哈希、派生题的来源记录都已核对；提交的 manifest 和用途登记都能重建出完全相同的内容。
- 篡改会被拒绝：把 78a5 挪到 dev、改 content 或 source 哈希、改规模、把 R1 pending 的题加进 GRPO 可训练池、把 held-out 题加进 SFT 池，校验都会失败。
- 入口：v3 train 被拒绝；GRPO 只得到 26 题，与 SFT-only、pending、dev、held-out 都不相交；SFT 得到 37 题；held-out 需要 `final_evaluation=True`；未知用途被拒绝。
- 62 题全部可以解析；78a5、54f35603、eff85dae 经 v3 入口解析后，gold 回放 DB = 1。

## 5. Teacher 与 S1 预算（更新 S0 文档）

详见 `docs/itsm_stepS0/README.md` §4.2 和 §6.1。
- **Teacher 候选**：GPT-6.1 Sol，供应商 Micu（https://www.micuapi.ai）。接口兼容性、实际模型 ID、参数、版本、价格都留到第 7 步验证；不照搬 DeepSeek 的参数（撤回"关闭思考""温度 0/0.7"）；运行中返回的模型标识与确认值不一致就停，不静默换模型。
- **裁判关系**：评测裁判仍是 deepseek-v4-pro。Teacher 与裁判不再同源，但不能因此假定裁判中立。主指标仍用 fuzzy DB，LLM 裁判单列，裁判偏差另行测量。
- **S1 总上限 45 美元**，包含全部重试和收费请求。DeepSeek 口径下算出的 0.072、3.35 美元作废，改为公式：
  - 单请求预留 `R_T = k·N_in·P_in + O_max·P_out + F`（`O_max` 含推理 token，`k` 暂定 1.3，第 7 步校准）；
  - 单轨迹预留 `R_traj = 45·R_T + 0.097`；
  - 总预留中保留 4.5 美元余量，可开始的轨迹数 `n = min(12, ⌊40.5/R_traj⌋)`。
  - 12 条轨迹都能预留的条件是 `R_T ≤ 0.0728` 美元。实际价格要到第 7 步核实后才能代入，本步没有填写任何价格数字。

## 6. 运行记录

在 `code/main` 下运行，日志在 `/home/sar/work/EnterGym/output/itsm-v3-usage/`：`build_itsm_split_v3.py --check`、`build_itsm_usage.py --check` 在哈希种子 1、7、13 下都是 up to date；S0 审计、held-out 候选、6D-2 批次、试点、v1、v2 的 `--check` 均为 up to date；`pytest tests/unit tests/remote -q`：738 passed（`pytest_full.log`），跑完没有残留子进程。另有 2 条旧断言按当前状态更新：`test_itsm_eops_heldout_v3` 原先断言 v3 文件不存在，改为断言 54f35603 在 v3 held-out、隔离题不在任何划分；`test_itsm_task_sets` 中"未知版本"的示例从 itsm_split_v3 改为 itsm_split_v4。

## 7. 需要你验收

- v3 manifest 与 `itsm_usage_v1` 是否冻结。冻结后，任何改动都要新建版本并说明原因。
- GPU 时长上限（S0 §6.2）仍待批准。
