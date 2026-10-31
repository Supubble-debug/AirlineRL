# 第 6B 步：EnterpriseOps-Gym ITSM 迁移试点（逐题验收，待确认）

范围：只处理 6A 推荐的 8 题，不扩展到其余候选。**3 题通过验证，5 题隔离**。试点题单独存放，不进 Bench2 `tasks.json`，不进 v1 划分；dev 和 held-out 不变。纳入训练集 v2 要等你确认。没有启动 GPU，没有加载权重，没有调用 LLM。

**不声称与 EnterpriseOps-Gym 原评测等价**。差异见 §3。

| 产物 | 位置 |
|---|---|
| 试点任务文件 | `configs/data/itsm_eops_pilot_v1.json`（3 个 Bench2 Task + 来源 + 上游验证器副本 + 转换说明 + 信息边界 + 反例；5 题隔离记录） |
| 生成脚本 | `scripts/data/build_itsm_eops_pilot.py`：转换内容手写在脚本里；上游验证器和校验值从参考目录读取；校验"Bench2 种子 + delta = 上游初始状态"；`--check` 可复现 |
| 验证与测量脚本 | `scripts/data/validate_itsm_eops_pilot.py` |
| 验证与测量原始结果 | `docs/itsm_step6b/pilot_validation.json` |
| 代码 | `src/agent_rl/data/itsm_eops_pilot.py`（Bench2 DB 导出到 SQLite 后跑上游验证器；跨来源家族与重叠复查）；`ItsmEnv` 增加可选参数 `task=`，原有行为不变 |
| 测试 | `tests/unit/test_itsm_eops_pilot.py` |

## 1. 判定原则

1. **Agent 只看 Bench2 的 `policy.md`**：这份 policy 对所有任务相同，上游每题自己的系统提示不提供给 agent。所以 gold 必须遵守 Bench2 policy 中**创建事件单时的 REQUIRED 连带操作**：生成 SLA、搜索知识并关联匹配文章。这些操作 Bench2 现有 53 题从未覆盖，上游验证器也不检查。
2. **用户可控的字段**可以由多轮对话中的用户明确给出：资产名、事件类别、是否发通知，等等。前提是不与上游 prompt 或验证器冲突，而且上游验证器本来就不约束这些字段。每处补充都记在转换说明里。
3. **REQUIRED 连带操作的结果不唯一**时（policy 要求必须做，但做法有多种合理选择），用户信息无法化解，因为 REQUIRED 操作不接受用户的选择。**上游 prompt 与验证器互相矛盾、或上游期望状态与 Bench2 的 REQUIRED 规则冲突**时，也同样处理。这三类一律**隔离**，不改上游的验证标准，也不靠挑选解释来凑题数。

## 2. 逐题结果

### 2.1 通过（3 题）

所有题共同的情况：
- **来源**：EnterpriseOps-Gym commit `271f2c3`，HF revision `c8e538ea`，oracle 模式，Apache-2.0。每题的上游 prompt、系统提示和种子 sha256 都记在试点文件的 `source` 里。
- **组织范围**：不做组织切分（`org_ids: null`），所以新建记录的顺序 ID 与上游验证器写死的一致（`CI_005`、`INC_024`、`LOC_024`）。
- **任务时间**：`current_time` 取上游 task_id 里的日期，时刻定为 09:00。

| 项 | `itsmbench_eops_46ca2862_b6ef60ca` | `itsmbench_eops_66392a82_52c8290e` | `itsmbench_eops_1d219d93_d1ba6c2d` |
|---|---|---|---|
| 上游 task_id | `task_20260113_182602_664_46ca2862_b6ef60ca` | `task_20251218_124419_383_66392a82_52c8290e` | `task_20251223_101136_983_1d219d93_d1ba6c2d` |
| 内容 | 登记 Benjamin 的新笔记本，并为该笔记本开单（软件安装） | 修正视频会议终端的成本，并开高优先级事件单 | 新建 Pune HQ：建地点、迁移在库资产、登记 2 个 CI、提升经理角色、写知识文章 |
| 上游种子 → Bench2 | `db_1765301900121` → `single_tenant_db.json`，无 delta | `db_1765984530018` → `single_tenant_db.json`，delta：新建上游的 `CI_005`（Boardroom Video Hub） | `db_1765893060767` → `msp_db.json`，delta：删除 `CRM_017`、`PRB_011`，`CHG_017.status` 改回 `scheduled` |
| 种子 + delta 与上游初始状态的差异 | 0 条（只有 `MEMBER_`→`MEM_` 的 ID 改名，内容相同） | 0 条（同左） | 0 条 |
| tenancy | `single_tenant` | `single_tenant` | `msp_provider_internal`（MSP 世界，但涉及的记录全在提供方 ORG_001） |
| 为 Bench2 补充的信息 | 资产名 `Benjamin Chen Laptop`；类别 `software`；拒绝"给来电人发报告"这项建议 | 类别 `hardware`；拒绝发报告的建议 | `Tech Park, Building 4` 填在 street 字段；Elizabeth 的地点不变；知识文章 owner 为 Karen |
| policy 带来的 gold 新增 | `channel=chat`（默认值）；`SLA_004`（ORG_001 唯一的 moderate SLA）；知识搜索无匹配，不关联 | `channel=chat`；`SLA_001`（唯一的 critical SLA）；知识搜索无匹配 | 无事件单，所以没有 SLA 或知识连带操作 |
| gold 动作 | 8 个 | 8 个 | 10 个 |
| Bench2 DB（fuzzy） | **1** | **1** | **1** |
| 上游验证器（去重后），gold 回放后 / 不做任何操作 | 2/2 通过 / 0/2 | 2/2 / 0/2 | 6/6 / 0/6 |
| 实测：agent 轮数 / 底层步数 | 11 / 11 | 11 / 11 | 13 / 13 |
| 实测：初始 prompt / 响应 / 完整序列 token | 28,131 / 2,379 / 30,510 | 28,132 / 2,242 / 30,374 | 28,130 / 2,926 / 31,056 |
| 实测：单次生成最大值 / 单条观察最大值 | 160 / 394 | 147 / 332 | 252 / 280 |
| 实测耗时（本机 CPU） | 建环境 + reset 8.6 ms；工具调用合计 2.6 ms；回放 + 评分 7.7 ms | 7.8 / 1.2 / 5.3 ms | 11.6 / 2.1 / 20.6 ms |

**实测说明**：
- 测量用的是脚本化 episode：gold 工具调用，加 1 条模板化的计划消息和 1 条回顾消息，用户回复是固定文本，没有调用用户模拟器。它**不代表实际策略的输出分布**。
- 单次生成最大值来自模板化的计划消息，低于 `max_action_tokens=512`。
- 完整序列 30.4k–31.1k，低于 40,960 候选值，也低于原生 32,768。

### 2.2 反例（每题 3–5 个，全部判 Bench2 DB=0）

"上游全部通过"表示同一份最终 DB 在上游验证器下是否全部通过。

| 题 | 反例 | 预期失败原因 | Bench2 mismatch | 上游全部通过？ |
|---|---|---|---|---|
| b6ef60ca | `wrong_priority`（high） | 优先级应按 medium × medium 取 moderate | `incident/INC_024 priority` | 否（Report incident） |
| b6ef60ca | `sla_not_spawned` | Bench2 policy 要求必须生成 SLA | `incident_sla: missing` | **是**（上游不检查 SLA） |
| b6ef60ca | `laptop_registered_in_london` | 笔记本在纽约办公室使用 | `configuration_item/CI_005 location_id` | 否 |
| b6ef60ca | `unconfirmed_report_notification` | 操作员已拒绝发报告 | `notification: unexpected` | **是**（上游不惩罚多余写入） |
| 52c8290e | `cost_not_corrected` | 漏了成本修正 | `configuration_item/CI_005 cost` | 否 |
| 52c8290e | `operator_as_caller` | 来电人应为 Elena | `incident/INC_024 caller_id` | 否 |
| 52c8290e | `sla_for_wrong_priority` | SLA 应与 critical 对应 | `incident_sla/TSLA_011 sla_def_id` | **是** |
| 1d219d93 | `dell_left_in_bangalore` | 两个新资产都应在 Pune HQ | `configuration_item/CI_102 location_id` | 否 |
| 1d219d93 | `role_not_changed` | Elizabeth 应升为经理 | `users/USER_042 role` | 否 |
| 1d219d93 | `in_stock_item_not_moved` | `CI_035` 在库，应迁移 | `configuration_item/CI_035 location_id` | 否 |
| 1d219d93 | `in_use_item_also_moved` | 只迁移在库资产 | `configuration_item/CI_023 location_id` | **是** |
| 1d219d93 | `elizabeth_relocated` | 操作员说只改角色 | `users/USER_042 location_id` | **是** |

12 个反例中，7 个是上游验证器也能识别的错误，5 个只有 Bench2 识别（见 §3）。

### 2.3 用户可见信息与评分私有信息的边界

所有题相同，试点文件的 `information_boundary` 中有记录。

- **策略可见**：Bench2 `policy.md` 和 94 个工具 schema（各题相同）、固定问候语、用户消息、真实工具结果。
- **私有**：
  - 任务描述、模拟器指引、`known_info`（用户说出前）；
  - gold、NL 断言；
  - delta、种子名、`current_time`；
  - 上游 task_id、prompt、系统提示和验证器；
  - 反例。
- **测试验证**：每题用 `ItsmEnv(task=…)` reset 后，观察和公开信息中都不含上述私有文本。

### 2.4 隔离（5 题）

| 上游题 | 原因 | 证据 |
|---|---|---|
| `97d7df7d_a52988e5` | REQUIRED 连带操作结果不唯一 | 建单时必须搜索知识并关联"匹配"的文章。对邮件服务变慢这个问题，`KB_002 Email Client Configuration Guide` 关联与否都说得通，全库评分下两种合理行为必有一种得 0 |
| `66392a82_74b17170` | 上游 prompt 与验证器矛盾 | prompt 写 "$64,00"，验证器要求 6400；采用 6400 等于用验证器去改写 prompt |
| `e6427839_ca9119ec` | 上游期望状态与 REQUIRED 规则冲突 | 验证器要求解散 `GROUP_001`，但它仍有 4 个活动事件单（INC_003/004/014/016），§12.3 要求在改派之前阻止解散；上游没有要求改派 |
| `46ca2862_796e7685` | REQUIRED 连带操作结果不唯一 | MSP 世界里 ORG_001 的 moderate SLA 有两个（Response `SLA_005` 和 Resolution `SLA_006`），种子中已有事件单的用法也不统一；知识搜索还可能匹配 `KB_006` |
| `66392a82_fd509f97` | REQUIRED 操作无法完成，且 SLA 不唯一 | 停用账号时必须删除 `user_role` 行 `USER_011:ROLE_002:ORG_001`，Bench2 没有这个工具；low 优先级 SLA 也有两个候选；另有 4 项 PROPOSED 连带操作 |

测试会在 Bench2 数据上重新核对这些证据。

## 3. 与 EnterpriseOps-Gym 原评测的差异（不等价）

1. **多轮改写**：上游是一条完整指令；这里由用户在对话中逐步提供信息，再由模拟器指引给出对 PROPOSED/ASK 的回答。结果取决于模拟器的行为，第 7 步才首次真实调用。
2. **policy 不同**：上游每题有自己的系统提示（共 9 种）；这里统一使用 Bench2 `policy.md`。gold 因此多了 SLA 生成、默认渠道等连带操作。
3. **评分更严格**：
   - 上游只检查被点名的行，`LIKE`、`IS NOT NULL` 这类条件都算通过；
   - Bench2 比对全库，任何未要求的写入或漏掉的 REQUIRED 连带操作都判 0。例如 §2.2 中 5 个"上游全部通过"的反例。
   - 自由文本字段（short_description、message、body）在 Bench2 中用 fuzzy 比对 gold 文本，上游用 `LIKE` 关键词匹配，两种标准不同。
4. **新增内容**：NL 断言是新写的（训练 reward 不使用，见 D1）；资产名、类别等补充信息上游没有。
5. **结果解读**：通过 Bench2 评分和上游验证器两道检查，只说明 gold 与两套标准一致，**不说明**模型在这些任务上的得分能与 EnterpriseOps-Gym 公布的结果比较。

## 4. 合并试点后的跨来源家族与重叠复查

在 53 题与 3 道试点题的合集上，重新运行第 6 步的家族规则：写工具 Jaccard ≥ 0.6，或共享新建对象工具。

| 检查 | 结果 |
|---|---|
| 试点题所在的家族只含 train 任务 | 通过 |
| 没有家族把 v1 不同划分的任务连起来 | 通过 |
| dev、held-out 家族成员不变 | 通过 |
| 与 dev/held-out 任务共享 ≥ 2 个同种子实体 | 无 |
| 与任何 dev/held-out 任务共享 ≥ 2 个写工具 | 无（最大为 0） |

需要你知道的情况：
- **`1d219d93` 合并了两个 v1 训练家族**：它同时新建 CI 和知识文章，因此把训练集里的 `onboarding_and_catalog_admin` 与 `change_problem_advance_with_kb` 连成一个分量。`b6ef60ca` 通过 `register_configuration_item` 也进入这个分量。两边都在训练集，不影响 dev/held-out，但 v2 的训练家族统计会变化。
- **`52c8290e` 自成新家族**：它的写工具是 `create_incident` 和 `link_new_incident_sla`，Bench2 现有题没有相近的组合。
- **与 dev/held-out 共享单个实体**：有，但都是人物或 SLA 这类公共记录，例如 USER_005 出现在 dev 的 single_008/019，USER_006 出现在 held-out 的 single_022，SLA_001 出现在 dev 的 single_010。都低于 2 个的阈值，详见 `pilot_validation.json`。

## 5. 需要你确认的事

- 是否把这 3 题纳入 **训练集 v2**：train 从 29 题变为 32 题，dev 13、held-out 11 不变。确认后我会生成 `itsm_split_v2` 并重新校验。
- 隔离的 5 题保持隔离。如果你要放行其中某题（例如接受 6400 的读法，或者为 MSP 指定 SLA 规则），需要明确给出规则，我再单独处理。
- 本轮不扩大规模。
