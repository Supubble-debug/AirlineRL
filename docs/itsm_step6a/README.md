# 第 6A 步：EnterpriseOps-Gym ITSM 扩展可行性审计（待验收）

本步只做审计和试点推荐：没有批量导入，没有修改业务工具、现有评分或 v1 划分，没有启动 GPU、加载权重或调用 LLM。上游数据只放在独立参考目录里读取，不进入 Bench2 或项目任务集。

| 产物 | 位置 |
|---|---|
| 审计脚本 | `scripts/data/audit_eopsgym_itsm.py`（只解析 parquet/JSON；SQL 种子和验证器在内存 SQLite 中执行，禁止 ATTACH） |
| 审计原始结果 | `docs/itsm_step6a/eopsgym_itsm_audit.json`（103 题逐题结果、9 个种子对照、试点清单与自动检查；上游 prompt 只存 200 字摘录和 sha256） |
| 测试 | `tests/unit/test_itsm_eopsgym_audit.py`（只读已提交的审计结果） |
| 参考数据（不入 git） | `/home/sar/work/EnterGym/reference/enterpriseops-gym/`，来源记录 `SOURCE.json` |

## 1. 来源与版本

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/ServiceNow/EnterpriseOps-Gym`，commit `271f2c357f763376997dfd16807fcde2474ae41b`，Apache-2.0 |
| 数据集 | HF `ServiceNow-AI/EnterpriseOps-Gym`，revision `c8e538eae8a6205294f0a86675fefdc1fac408f6`，未设访问限制，Apache-2.0 |
| 下载内容 | 4 个 ITSM parquet（`oracle`、`plus_5_tools`、`plus_10_tools`、`plus_15_tools`）；从 `gym_dbs.zip`（sha256 `d947543d…`）只解出 9 个 ITSM SQL 种子，逐个成员做路径检查 |
| 校验值 | 全部写在审计结果的 `provenance` 中（4 个 parquet、zip、9 个种子） |
| 与 Bench2 的关系 | `Bench2/data/itsmbench/README.md` 写明：Bench2 的种子数据和动作空间改编自 EnterpriseOps-Gym，任务是原创的。Bench2 的 conformance 测试使用的 canonical seed 就是上游的 `db_1766420030407` |

## 2. 去重：工具集模式与独立任务

- **4 种模式是同一套任务**：task_id 完全相同，prompt、种子、用户和验证器集合也相同。区别只有两点：
  - `selected_tools` 中加入了干扰工具：分别多 5、7–10、13–15 个，oracle 工具始终包含在内；
  - 12 处验证器的**顺序**不同，但作为多重集是相同的。
- **独立任务数是 103，不是 412。** 103 题之间没有内容完全相同的、没有 prompt 完全相同的，也没有验证器集合完全相同的。共有 9 套不同的系统提示。
- 本审计只用 oracle 模式的工具集作为每题的必需工具。干扰工具在 Bench2 中也全部存在。

## 3. 对照 Bench2：工具、参数、表结构、初始状态

### 3.1 工具与参数

- **工具**：103 题共用到 64 个 oracle 工具，**全部存在于 Bench2**；干扰工具也全部存在。
- **参数**：上游 MCP 的工具 schema 不在公开仓库中（在 Docker 镜像里），所以参数不能逐一比对。替代检查是：试点题的验证器要求的每个字段，都确认 Bench2 对应工具有参数可以写出来（例如 `create_incident.parent_incident`、渠道枚举 `virtual-agent`、`update_user_group`）。全部 103 题的参数级比对留待 6B 实际回放时进行。
- **新增覆盖**：有 5 个写工具，Bench2 现有 53 题从未使用：

| 写工具 | 上游使用题数 |
|---|---|
| `create_incident` | 39 |
| `create_change` | 3 |
| `update_user_group` | 3 |
| `remove_group_membership` | 1 |
| `update_notification` | 1 |

### 3.2 表结构

- 上游 SQL 种子**只有 INSERT 语句**，表定义在闭源的服务镜像里。审计时用"INSERT 中出现的列 ∪ Bench2 模型字段"建表。
- 9 个种子都没有 Bench2 缺少的表或列。
- 反过来，Bench2 的 `incident` 有 9 个字段、`incident_knowledge` 有 1 个字段在 MSP 种子的 INSERT 中没有出现，按 NULL 处理。

### 3.3 初始状态（时间戳不参与比较）

| 上游种子 | 使用题数 | 对应 Bench2 种子 | 记录相同比例 | 与 Bench2 的差异 |
|---|---|---|---|---|
| `db_1765301900121` | 40 | `single_tenant_db.json` | 100%（含改名） | 只有 `user_group_member` 的 ID 前缀不同：上游 `MEMBER_00x`，Bench2 `MEM_00x`，内容相同 |
| `db_1765984530018` | 1 | `single_tenant_db.json` | 99.6% | 另多一条 `configuration_item/CI_005` |
| `db_1765893060767`（MSP 主种子） | 51 | `msp_db.json` | 99.9% | Bench2 多出 `problem/PRB_011`、`change_request_mapping/CRM_017`；`change/CHG_017` 字段不同 |
| `db_1766420030407` | 1 | `msp_db.json` | 100% | 无（即 Bench2 conformance 测试用的 canonical seed） |
| 其余 5 个 MSP 变体 | 1–4 | `msp_db.json` | ≥ 99.6% | 在主种子差异之外，各自另有 1–3 条记录不同（详见 JSON） |

**结论**：

- 单租户主种子上的 40 题可以直接使用 Bench2 种子。
- 62 题需要一个小的 `initial_state_delta` 来还原上游初始状态。主要是 MSP 主种子：删除 `PRB_011`、`CRM_017`，并还原 `CHG_017`。

## 4. 验证器审计（464 个，全部是 `database_state`，比较方式全部是 `equals`）

| 检查 | 结果 |
|---|---|
| 上游评分语义 | 一题的全部验证器都通过才算成功（二值），另报告验证器级通过率。只检查被点名的行，**不惩罚多余写入**；Bench2 则是全库比对，多余写入会判 0 |
| 不可满足：同一记录要求不同值 | 2 题。`391f1eff_ad65c160` 同一事件单既要求 `channel="chat"` 又要求 `"phone"`；`e6427839_6413728a` 的 `priority` 同时要求 `critical`/`high`，通知 `type` 同时要求 `other`/`reminder`。看起来是多轮标注的验证器被拼接在了一起 |
| 不可执行 | 1 题（`e6427839_47076c15`）：读取不存在的列 `incident_sla.applies_to_priority` |
| 重复验证器 | 15 题含有完全重复的验证器（上面两题重复最严重） |
| 写死新建记录的 ID | 40 题的验证器写死了"任务新建记录的下一个顺序 ID"（如 `INC_024`、`CRM_017`）。其中 2 题在 Bench2 种子中会直接撞上已有记录（`CRM_017`、`PRB_011`），必须先用 delta 还原上游种子 |
| 初始状态就已通过 | 6 题有验证器在不做任何操作时就通过。有的是前置条件检查，有的是"不得建立链接"这类否定约束，对 no-op 没有区分力 |
| 宽松谓词 | 38 题使用 `LIKE`（63 处）、`IS NOT NULL`（16 处）、`!=`（4 处），另有 1 处 `x = 'a' OR x IS NOT NULL` 实际恒为真。注意 SQLite 的 `LIKE` 对 ASCII 字母不区分大小写 |

这些验证器问题**不影响** Bench2 的 reward。Bench2 仍然用 gold 回放加全库比对来评分；上游验证器在 6B 中只用作 gold 正确性的第二道独立检查（见 §7）。

## 5. 跨来源：与现有 53 题的近重复和共享工作流

- **两边是同一个世界**：上游与 Bench2 共用种子，也共用 persona（Karen Watkins、Marcus Thompson、Priya Sharma 等），所以不能用 task_id 或人名判断是否重复。
- **文本相似度不能作为依据**：上游 prompt 很短，Bench2 是长段的第二人称场景描述，103 题与 Bench2 的最大文本 Jaccard 都不超过 0.2，说明不了任何问题。所以采用三类信号，并且**不以低于阈值作为没有重复的证明**（与第 6 步口径一致）：
  1. 第 6 步的家族规则：写工具 Jaccard ≥ 0.6，或共享新建对象工具；
  2. 同一种子上的同一实体：去掉 delta 新建和顺序 ID 撞号的情况；
  3. 训练侧额外门槛：与任何 dev/held-out 任务共享的写工具不超过 1 个。
- **家族连边结果**：33 题不与任何 v1 家族连边；25 题只连 train；21 题连 dev；17 题连 held-out；7 题同时连多个划分。
- **实体重叠**：3 题与 dev/held-out 任务共享 ≥ 2 个同种子实体。
  - `97d7df7d_7196ef19` 与 held-out 的 single_012 共享 `GROUP_001`、`INC_003`、`INC_014`。写工具 Jaccard 只有 0.2，家族规则没有连上，但放进训练集就会让 held-out 用到的记录出现在训练中。
- **近重复**：1 题。`e6427839_03a04c19` 与 dev 的 single_008 / single_019 操作的是同一事件单 `INC_002`、同一 SLA `TSLA_002`，工作流也相同（挂起、提醒）。

## 6. 分类

| 类别 | 题数 | 判据 |
|---|---|---|
| 可直接适配 | 10 | 没有阻断项，也没有小幅适配项（9 题在单租户种子，1 题在 canonical MSP 种子） |
| 需要小幅适配 | 44 | 只需机械性工作：种子 delta、重新确定写死的 ID、删除重复验证器 |
| 暂不适合 | 49 | 有任一阻断项，见下 |

- **所有题都需要的转换**（不计入分类）：上游是单轮指令，Bench2 是多轮对话。每题都要写 scenario、persona、known_info 和 simulator guidance，把指令里的信息改为由用户逐步提供；再写 gold actions、NL 断言和 `current_time`。
- **"暂不适合"的阻断原因**（一题可以有多个）：

| 原因 | 题数 |
|---|---|
| 连到 dev/held-out 家族 | 45 |
| 同时连多个划分的家族 | 7 |
| 与 dev/held-out 共享实体 | 3 |
| 验证器冲突 | 2 |
| 验证器不可执行 | 1 |
| 近重复 | 1 |

- 其中 **46 题被阻断仅仅是因为 v1 划分**，这些题的工具、种子和验证器都可用。只有调整 dev/held-out（v2 划分）后才能用，本步不建议这样做。

分类规则写在审计结果的 `classification_rules` 里，按规则自动得出，不是逐题手工判断。

## 7. 试点推荐（8 题，都只进训练侧）

**入选条件**（脚本自动检查，不满足就报错）：

- 不属于"暂不适合"；
- 不连 dev/held-out 家族；
- 与任何 dev/held-out 任务共享的写工具不超过 1 个；
- 不与 dev/held-out 任务共享实体。

在此之上，按新增覆盖、两种种子、两条适配路径（直接 / 需要 delta）、不同长度来选。

**预计长度**用第 6 步实测的平均单次工具调用成本估算：生成约 39 token，观察约 246 token，合计约 285 token/次，再加 2–4 次用户回合。这是估计值，不是实测。初始 prompt 都是 28,104。

| # | 上游题（后缀） | 种子 | 新增价值 | 预计调用次数 / 响应 token | 所需改动 |
|---|---|---|---|---|---|
| 1 | `97d7df7d_a52988e5` | 单租户，无 delta | `create_incident`：根据虚拟助手渠道的报告建单，要识别服务、设备和指派人，再发报告通知。Bench2 没有"建单"工作流 | 约 7 / 约 2.3k | 只需对话化；2 个验证器里有 `LIKE` |
| 2 | `46ca2862_b6ef60ca` | 单租户，无 delta | 先登记资产（CI），再为该资产建单。验证器写死 `CI_005`、`INC_024`，正好用来检验 Bench2 的顺序 ID 与上游一致 | 约 6 / 约 2.0k | 对话化；确认 ID 与上游一致 |
| 3 | `66392a82_74b17170` | 单租户，无 delta | 新站点：建 location、登记 2 个 CI、把其中一个改为维修。CMDB 生命周期在训练集中只有 1 题 | 约 7 / 约 2.3k | 对话化。**prompt 写成 "$64,00"，验证器是 6400**，known_info 要写成明确的 6400 |
| 4 | `e6427839_ca9119ec` | 单租户，无 delta | 组织调整：新建组、迁移成员、停用旧组、指定经理。用到 `update_user_group`（Bench2 未覆盖），并且需要判断"排除 IT Support Team" | 约 8 / 约 2.7k | 对话化；"任意可用经理"在 ORG_001 中只有 `USER_002`，已确认唯一 |
| 5 | `66392a82_52c8290e` | 单租户变体，delta 1 条 | 改正资产成本并建单；用来检验"单租户种子 + delta"这条路径 | 约 5 / 约 1.7k | delta：新建 `CI_005`（视频会议终端）；对话化 |
| 6 | `46ca2862_796e7685` | MSP 主种子，delta 3 条 | 带条件的多步操作：先确认 CI 负责人是否激活，没有就激活（`USER_001` 在种子中确实未激活），再改 CI 成本，最后建**子事件单**（`parent_incident`） | 约 7 / 约 2.3k；如果模型调用无过滤的 `list_incidents`，单次观察约 9.7k，可能触发 `response_length` 结束 | delta：删除 `PRB_011`、`CRM_017`，还原 `CHG_017` |
| 7 | `66392a82_fd509f97` | MSP 主种子，delta 3 条 | 离职回收：停用账户、回收资产、建单、关联受影响 CI、通知办公室经理。5 个写工具，跨用户、CMDB、事件、通知 | 约 11 / 约 3.5k | 同上 delta；通知正文是完全匹配，与 Bench2 自由文本 fuzzy 的差异需记录 |
| 8 | `1d219d93_d1ba6c2d` | MSP 主种子，delta 3 条 | 新办公点：先判断是否存在再建 location，跨地点迁移在库资产，登记 2 个 CI，提升经理角色，写知识文章。最长的组合工作流 | 约 11 / 约 3.6k | 同上 delta；连到两个 train 家族（变更与问题推进、入职与目录管理），全部在训练侧 |

**评分验证方法**（6B 中每题都做，不改动现有评分）：

1. 写 Bench2 Task：
   - scenario、persona、known_info，把指令改为由用户逐步提供；
   - `current_time`；
   - 需要时写 `initial_state_delta`，以还原上游种子。
2. 写 gold actions，用现有严格回放在任务时钟下回放，Bench2 DB 评分应为 1。
3. **第二道独立检查**：把 gold 回放后的最终 DB 导入内存 SQLite（与本审计用同一套加载器），执行该题全部上游验证器，要求全部通过。不通过时，先查我们写的 gold 是否有误，或上游验证器本身是否有问题，不能改验证器去凑通过。
4. 每题设计 2–3 个反例，例如漏关联、优先级写错、多发一条通知。要求 Bench2 DB 为 0，并说明上游验证器是否也能识别（多余写入这一类上游识别不了，是已知差异）。
5. 用第 6 步的家族规则重新计算，确认这些题都归入训练侧，dev/held-out 不变。

## 8. 6B 前需要你决定的事

- **训练集变化**：试点题加入训练会产生 `itsm_split_v2`，dev/held-out 不变，train 从 29 题增加到最多 37 题。需要你确认这种扩展方式，以及新任务 ID 的命名（建议 `itsmbench_eops_<上游后缀>`）。
- **任务来源和授权**：上游是 Apache-2.0。转换后的任务需要在 NOTICE 和任务元数据中注明出处与上游 task_id。
- **评分差异的定位**：上游只检查被点名的行，Bench2 全库比对。同一条 gold 在两边都通过，但模型的多余写入只有 Bench2 会扣分。报告中要分开说明。

## 9. 未验证的边界

- **工具参数**：上游工具 schema 不公开，参数没有逐一比对；试点题已按验证器字段做了检查，其余题留到 6B 回放时发现。
- **gold 回放**：没有做任何 gold 回放；上游也不提供 gold 动作。试点题的可解性、预计调用次数和 token 数都是估计值。
- **约束解析的局限**：冲突与谓词检查基于对等值条件的正则解析，没有覆盖 `IN`、子查询等复杂约束。可执行性和"初始状态已通过"是在内存 SQLite 中实际执行得到的。
- **上游与 Bench2 的行为差异**：上游 MCP 的工具行为（自动 SLA、时间戳、ID 生成）与 Bench2 实现是否完全一致，只能靠 Bench2 自带的 conformance 测试（需要 Docker oracle，本环境没有运行）。
