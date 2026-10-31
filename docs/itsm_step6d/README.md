# 第 6D-1 步：EnterpriseOps-Gym ITSM 下一批候选筛选（待验收）

本步只做候选筛选和规模方案：没有转换任何题，没有写 gold，没有修改 policy、评分、v1/v2 划分或已隔离的 5 题；没有启动 GPU、加载权重或调用 LLM。重划分只写成提案（`v3_proposal.md`），不生成 manifest。

| 产物 | 位置 |
|---|---|
| 筛选脚本 | `scripts/data/screen_itsm_eops_candidates.py`（`--check` 可复现；只读已提交的 6A 审计、v2 划分、Bench2 种子；分词器只用于测长度） |
| 筛选结果 | `docs/itsm_step6d/candidate_screening.json`（95 题逐题的自动门槛、policy 标记、人工结论；入选批次逐题方案；v3 可用量） |
| 重划分提案 | `docs/itsm_step6d/v3_proposal.md` |
| 测试 | `tests/unit/test_itsm_eops_screening.py`（13 项，只读筛选结果） |

## 1. 方法

**候选范围**：6A 审计的 103 题去掉 6B 的 8 题（3 题已入 v2，5 题隔离且不重新考虑），共 95 题。

**三层判断，在结果文件中分开记录**：

1. **自动门槛**（按 v2 重算，规则与 6A 相同）：
   - 不连 dev/held-out 家族；
   - 与任何 dev/held-out 任务共享的写工具 ≤ 1 个；
   - 与 dev/held-out 任务共享的同种子实体 < 2 个；
   - 不是近重复，验证器没有冲突或不可执行。
2. **自动 policy 标记**（从验证器和种子数据推出，偏保守）：
   - 建事件单时，按事件单所在组织和优先级统计 SLA 定义：唯一 / 没有 / 不唯一；
   - 验证器要求的优先级是否与 §3.1 矩阵矛盾；
   - 涉及 `user_role` 的 REQUIRED 操作（没有工具能做）；
   - 解散组前的检查。
3. **人工审读**：通过门槛的 34 题逐题读完整的上游 prompt 和验证器，并在 Bench2 种子上核对数据，每题给出结论和理由。

**两条路线**：

| 路线 | ID | 验收标准 |
|---|---|---|
| 忠实迁移 | `itsmbench_eops_<后缀>` | 沿用 6B 标准：Bench2 DB=1；去重后的上游验证器在 gold 上全部通过、在初始状态下不全通过；"切分后的种子 + delta"等于上游种子的同一切分 |
| Bench2 改编 | `itsmbench_eopsadapt_<后缀>` | 以明确的用户需求、Bench2 policy 和真实工具能力为准，重新设计 scenario、known_info、gold、NL 断言和评分预期。每项改动都记录原因；原题和上游验证器保留，只用于溯源，不要求通过；**不声称与 EnterpriseOps-Gym 原评测等价** |

两条路线都**不放行**以下三类：REQUIRED 连带操作结果不唯一；REQUIRED 步骤缺少工具；请求违反强制 policy。

### 1.1 本轮新确认的 policy 事实

这些事实决定了哪些题能选：

- **MSP 种子的 SLA**：ORG_001 的每个优先级都有 response 和 resolution 两条 SLA，其他 19 个组织没有任何 SLA 定义。种子中已有 SLA 行的事件单里，18 道只有 resolution，3 道只有 response，2 道两条都有。上游 `391f1eff_8cb006f5` 要求两条都关联。所以 MSP 上建事件单时，REQUIRED 的 SLA 生成**不唯一**（见 D6D-1）。
- **单租户种子的 SLA**：ORG_001 的 critical、high、moderate 各只有一条，是唯一的；low 有两条；planning 一条都没有。
- **知识库**：单租户 ORG_001 只有 3 篇文章（密码重置、邮件客户端、开机故障）。ORG_002 有两篇邮件相关文章，所以邮件类故障在 ORG_002 很容易产生"匹配不唯一"。
- **`user_role`**：Bench2 没有任何工具能写 `user_role`。Bench2 原题里有 9 道建了新用户，gold 都不含 `user_role` 行；6B 却把"停用账号时必须删除 `user_role`"当作隔离理由之一。两种处理口径不一致（见 D6D-2）。本轮不选建用户或停用账号的题。
- **工具不检查 policy**：`update_change` 不检查状态流转，`link_affected_ci_to_incident` 不拒绝已退役的 CI，`update_problem` 和 `update_incident` 不自动推导优先级。这些判断都要靠 agent 自己做。
- **切分与新建 ID**：Bench2 新建记录的 ID 是"当前集合里最大序号 + 1"。按组织切分可能改变新建 ID，所以只有上游验证器不写死新建 ID 的题，才能在切分后仍走忠实迁移。本批 8 道忠实迁移题都满足这一点。

## 2. 筛选结果总览

| 阶段 | 题数 | 说明 |
|---|---|---|
| 候选 | 95 | |
| 未过自动门槛 | 61 | 多数是连 dev/held-out 家族（45）或与 dev/held-out 共享 ≥ 2 个写工具（52 次命中，与前者重叠）；另有 3 题共享实体，2 题验证器冲突，1 题验证器不可执行，1 题近重复 |
| 通过门槛并人工审读 | 34 | 入选 10、备选 2、排除 12、暂缓 4、等 D6D-1 决定 6 |

### 2.1 入选批次（10 题：8 道忠实迁移 + 2 道改编）

**长度**：
- 估算响应 = 预计调用次数 × 232。232 是 v2 的 45 个 gold 回放每步响应 token 的中位数，所以这里只是估计。
- 初始 prompt 都是 28,104。
- "最长未过滤读取"是在该题建议的切分范围内实测的无参只读调用结果，测量时已加上 MSP 主种子的 delta。

**适配成本**：
- S：≤ 6 次调用，没有分支；
- M：7–10 次调用，或者有一项需要模拟器回应的确认或澄清，或者是改编题；
- L：> 10 次调用，或者有多个连带操作。

| # | 新 ID | 路线 | 工作流 | 范围 | 计划写工具 | 预计调用 / 响应 / 完整序列 | 最长未过滤读取 | 成本 | 风险 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `itsmbench_eops_99ba2325_887f5580` | 忠实 | 退役备用 CI；把在用打印机作为受影响 CI 关联到事件单；拒绝关联已退役的 CI | MSP ORG_001 | `update_configuration_item`、`link_affected_ci_to_incident` | 6 / ~1.4k / ~29.5k | 9,924 | S | 低 |
| 2 | `itsmbench_eops_c5ca08eb_67009578` | 忠实 | 问题影响升级，按矩阵重算问题和关联事件单的优先级 | MSP ORG_001+006+013 | `update_problem`、`update_incident` | 5 / ~1.2k / ~29.3k | 10,660 | S | 低 |
| 3 | `itsmbench_eops_01395ded_eff85dae` | 忠实 | 修复卡住的通知队列，按角色发状态通知 | MSP ORG_001 | `update_notification`（**现有 56 题未用过**）、`send_notification` | 9 / ~2.1k / ~30.2k | 9,924 | M | 低 |
| 4 | `itsmbench_eops_78a5b71e_a4fb07a9` | 忠实 | 已完成的变更经实施进入实施后评审，并记录额外的受影响 CI | MSP ORG_001 | `update_change` | 5 / ~1.2k / ~29.3k | 9,924 | S | 低到中 |
| 5 | `itsmbench_eops_66392a82_514cea3c` | 忠实 | 前线组重组：新建枢纽组、激活经理、迁移全部成员 | MSP ORG_001 | `add_new_user_group`、`update_user_details`、`add_new_group_member`、`remove_group_membership`（**未用过**） | 15 / ~3.5k / ~31.6k | 9,924 | M | 低到中 |
| 6 | `itsmbench_eops_46ca2862_56a4f834` | 忠实 | 扩充服务的 offering 组合，并把事件模板改指向新 offering | MSP ORG_001 | `register_new_service_offering`、`update_incident_template` | 8 / ~1.9k / ~30.0k | 9,924 | M | 低到中 |
| 7 | `itsmbench_eops_99ba2325_7915481a` | 忠实 | 服务恢复文档：发布知识文章，更正被替换 CI 的序列号，恢复服务状态 | MSP ORG_001 | `create_knowledge_article`、`update_configuration_item`、`update_service` | 7 / ~1.6k / ~29.7k | 9,924 | M | 低到中 |
| 8 | `itsmbench_eops_97d7df7d_8fd400d1` | 忠实 | 按日期找出某组的积压单，提醒各自的负责人 | 单租户 ORG_001 | `send_notification` | 6 / ~1.4k / ~29.5k | 4,167 | S | 低 |
| 9 | `itsmbench_eopsadapt_0be70e35_6bb33392` | **改编** | 邮件报来的硬件故障建单、关联 SLA、发报告通知（PROPOSED） | 单租户 ORG_001 | `create_incident`、`link_new_incident_sla`、`send_notification` | 8 / ~1.9k / ~30.0k | 4,167 | M | 中 |
| 10 | `itsmbench_eopsadapt_66392a82_a172d90d` | **改编** | 审查一个客户的密码重置单，为缺少链接的单关联已有文章（applied） | MSP ORG_001+016 | `link_knowledge_to_incident` | 5 / ~1.2k / ~29.3k | 10,232 | S | 低 |

**覆盖面**：
- 10 题按家族规则两两不连边。入选理由是新增工作流，不是换个实体重复一遍。
- 新覆盖两个写工具：`update_notification`、`remove_group_membership`。
- 新增 policy 判断点：
  - §5.1 不得关联已退役的 CI；
  - §4.2 变更流转；
  - §8.1 激活非活动经理；
  - §5.2 offering 继承父服务属性；
  - §7 `kb_use` 的 applied / resolution 区别；
  - 优先级重算后的 SLA 处理，以及"没有 SLA 定义"时什么都不用做。
- 3 题落入 v2 的训练合并家族（5、6、7，因为用到新建对象工具），其余 7 题各自构成新家族。全部只在训练侧，不连 dev/held-out。

**分布**：MSP 8 题，单租户 2 题。单租户偏少有三个原因：其余单租户候选要么是已验收试点的近重复（201、245、690），要么知识匹配不唯一（225、314），要么属于建用户类（161）。

### 2.2 逐题要点

完整内容见 `candidate_screening.json` 的 `batch`，包括 policy、歧义处理、转换说明、反例和验收条目。这里列出关键判断。

1. **99ba2325_887f5580**
   - §5.1 规定已退役的 CI 不得关联到事件单，所以 gold 只关联 CI_096，agent 要说明拒绝关联 CI_041 的原因。上游第 3 个验证器正好要求这条关联不存在。
   - CI_041 没有被活动事件单引用，退役时不触发 PROPOSED 连带操作。
   - in_stock → retired 跳过了中间状态；§5.1 允许按采购路径跳步。
2. **c5ca08eb_67009578**
   - 工具不自动推导优先级，gold 要显式写入：问题 high×high=critical，事件单 high×medium=high。
   - 优先级变化本应触发 §6 的 SLA 处理，但 INC_061 没有 SLA 行，ORG_013 也没有 SLA 定义，所以不需要任何 SLA 操作。这一点写进 gold 说明。
   - **范围不规范**：PRB_009 在 ORG_006，INC_061 在 ORG_013，用"provider + 一个客户"的组合会丢掉其中一条，所以范围用了 3 个组织。
3. **01395ded_eff85dae**
   - 通知正文按 §1.2 ASK 由操作员决定，所以由操作员给出，文本含 "status"、"progress"，满足上游的 LIKE 条件。
   - 收件人和通知类型都由数据决定：来电人收 update，负责人和 GROUP_008 的经理收 reminder。
   - NOTIF_003 是 INC_001 唯一一条 queued 通知。
4. **78a5b71e_a4fb07a9**
   - CHG_001 当前是 scheduled，gold 依次改为 implement、review。
   - Bench2 的变更只有一个 `configuration_item` 字段，没法关联第二个 CI，所以由操作员给出写进 short_description 的确切文字（满足上游 LIKE）。不声称已经关联了 CI。
   - 变更模型没有计划窗口字段；操作员说明实施是在批准的窗口内完成的，任务时间也设在这个叙述之内。
5. **66392a82_514cea3c**
   - Leah 当前未激活，按 §8.1 要先激活再设为经理。
   - **范围取舍**：如果把客户组织 ORG_005 也纳入范围，组里两道活动单会触发 PROPOSED 改派，gold 要多一个 `update_incident`，这样就和 dev 题 `itsmbench_msp_019` 共享 2 个写工具，不满足门槛。所以改用 provider 内部范围；上游要求不变。
6. **46ca2862_56a4f834**
   - 新 offering 按 §5.2 REQUIRED 继承父服务的 critical / production，和请求一致。
   - 负责人取父服务负责人 USER_004；short_description 由操作员给出。
   - TMPL_004 的 service 本来就是对的，只把 offering 改成新的 Mobile Email Access。
7. **99ba2325_7915481a**
   - 模拟器做两处澄清：在现有的 CI_001 上改序列号，不新建 CI；服务已经恢复，可以改为 operational。两处都不与上游验证器矛盾。
   - Joanne 是经理，可以直接发布文章（external）。
8. **97d7df7d_8fd400d1**
   - 数据给出唯一的结果集：INC_003 和 INC_004。
   - 模拟器说明"当作最高优先级处理"不是要改优先级字段，改了还会触发 §6 的 SLA 切换。
   - 消息模板由操作员给出，agent 按每道单填入摘要和创建日期。
9. **改编 0be70e35_6bb33392**，4 项改动：
   - urgency 从 high 改为 medium，原因是上游要求 priority=high，与 high×high 的矩阵结果 critical 矛盾；改后保留原意中的 high 优先级，以及 ORG_001 唯一的 high SLA（SLA_002）；
   - channel 从 phone 改为 email，原因是 prompt 写的是邮件来报；
   - 通知类型从 alert 改为 report，并在 agent 提议、操作员确认后才发，对应 §3.1 的 PROPOSED；
   - 场景写明笔记本能正常开机、是负载下关机，让 REQUIRED 的知识搜索明确没有匹配。
   - 预期改编后上游的事件单验证器和通知验证器不通过，SLA 验证器通过。
10. **改编 66392a82_a172d90d**，3 项改动：
    - 范围从 10 个客户组织缩到 ORG_016，原因是 MSP 惯例是"provider + 一个客户"，而不切分时一次 `list_incidents` 约 30k token；
    - `kb_use` 从 resolution 改为 applied，原因是按 §7，resolution 只用于为该单专门撰写的文章，KB_001 是已有文章；
    - 删去"如果文章有帮助"这个条件，因为从记录里判断不了，改由操作员说明。

### 2.3 每题的验收方法（6D-2 中执行）

1. 写 Bench2 Task：
   - 多轮 scenario、persona、known_info；
   - 模拟器指引，包括对 PROPOSED、ASK 和澄清的回答；
   - `current_time`、`org_ids`，需要时加 `initial_state_delta`；
   - 使用新 ID。
2. 用严格回放在任务时钟下回放 gold：Bench2 DB（fuzzy）= 1，重复回放后最终 DB 哈希相同。
3. 按路线检查上游验证器：
   - **忠实**：去重后的上游验证器在 gold 最终 DB 上全部通过、在初始状态下不全通过；切分后的种子 + delta 等于上游种子的同一切分；
   - **改编**：逐个验证器记录"预期通过 / 不通过"及其对应的改动，实际结果要与记录一致，不声称等价。
4. 每题 3–4 个反例（`negatives` 已列出）：Bench2 DB = 0，mismatch 与预期一致，同时记录上游验证器的判定。
5. 信息边界：reset 观察和公开信息中不含 task_id、上游信息、gold、改动说明等私有文本。
6. 家族复查：在 v2 + 本批上重算，本批每题所在家族只含 train 任务，dev/held-out 不变。
7. 测量脚本化 gold episode 的 token 数（prompt、响应、完整序列）。

任何一题不满足以上条件，就隔离并写明原因，不调整标准去凑数。

### 2.4 未入选的 24 题

| 结论 | 题（后缀） | 原因 |
|---|---|---|
| 备选 2 | `391f1eff_27934d6b` | 为已解决的事件单补写知识文章，与 §7 REQUIRED 一致；但上游写死了 KB_026/IKB_046，要先确认切分是否改变 ID；与训练合并家族中已有的写文章工作流重叠 |
| | `97d7df7d_be7218cc` | 与入选的 8 号是同一工作流，留作备选，避免换实体重复 |
| 等 D6D-1 决定 6 | `c5ca08eb_3ce3c52f`、`391f1eff_09a746e7`、`a8eea1c0_7e9733a5`、`46ca2862_c5083638`、`a8eea1c0_7a0619e9`、`d4a463c3_3f20bdca` | MSP 上建事件单（来电人在 ORG_001），SLA 有两条候选；其中 `3ce3c52f`（不关联已退役的 CI）和 `3f20bdca`（不关联未发布的文章）都考 policy 拒绝 |
| 排除 12 | 4 题：`391f1eff_30369342`、`391f1eff_8cb006f5`、`a8eea1c0_0d6acfd0`、`e6427839_91142068` | 事件单所在组织或优先级没有 SLA 定义，REQUIRED 的 SLA 生成做不到；按 6B 对"REQUIRED 步骤无法完成"的处理口径排除 |
| | `99ba2325_51726f9c`、`1d219d93_76cd5b4d`、`764e5ead_f8619b02` | 已验收试点（b6ef60ca、1d219d93）的工作流近重复 |
| | `a8eea1c0_73a73ce7` | 与入选的 1 号写工具组合和模式都相同 |
| | `66392a82_1c69521d`、`66392a82_c1a599bb`、`764e5ead_20429ef8` | 建用户类：`user_role` 没有工具（D6D-2）；训练集里已经有这类工作流 |
| | `c5ca08eb_db61cf35` | 唯一的验证器在新单**不**沿用参考单的服务时才通过，与 prompt 矛盾 |
| 暂缓 4 | `764e5ead_48850a98` | "确认在用且合法的 CI"没有界定范围，需要整题重写 |
| | `accab84d_d7ec710e`、`391f1eff_c23d76df` | ORG_002 的知识匹配不唯一（KB_005 与邮件服务器故障）；前者还有"建单时直接分派"是否触发 §3.2 工作备注的问题 |
| | `99ba2325_f931fd42` | 题目引用单租户的记录，却映射到 MSP 种子，而 MSP 种子里没有 Elena；要先查清种子映射 |

## 3. 需要你决定的事

| 编号 | 问题 | 选项（第一项为默认） | 影响 |
|---|---|---|---|
| D6D-1 | MSP 建事件单时，REQUIRED"生成与优先级匹配的 incident_sla 行"应生成哪些行？ | ① 继续排除 MSP 建单题；② 每条匹配的有效定义各生成一行（response + resolution，与上游 391f1eff_8cb006f5 一致）；③ 只生成 resolution（种子中的多数做法） | ② 或 ③ 能解锁 6 道训练候选。这是对 policy 的固定解释，不改 policy 文本，写入协议的方式与 D8 相同 |
| D6D-2 | 建用户类派生题是否沿用 Bench2 原题的口径（gold 不含 `user_role`）？ | ① 否；② 是，逐题记录这个缺口 | 能解锁 3 题，但都与训练集已有工作流接近，价值有限 |

此外，本批的切分范围有两个和惯例不同的地方，请一并确认：
- **2 号**：用了 3 个组织，因为 PRB_009 和 INC_061 在两个不同的客户组织。
- **5 号**：没有用"provider + 客户"，而是 provider 内部范围，原因见 §2.2。

## 4. 后续数据规模方案

### 4.1 训练集：工程目标 50–70 题

| 来源 | 新增 | 累计 | 前提 |
|---|---|---|---|
| 现有 v2 train | — | 32 | |
| 本批（6D-2） | ≤ 10 | 42 | 逐题验收通过 |
| 备选 2 题 | ≤ 2 | 44 | 备选 1 号需先确认切分后的 ID |
| D6D-1 选 ② 或 ③ | ≤ 6 | 50 | 你的决定 |
| 放宽训练侧门槛：与 dev/held-out 共享写工具从 ≤ 1 放宽到 ≤ 2，仍要求不连家族、不共享实体 | ≤ 5 | 55 | 规则变更，需要你决定。会增加跨划分的子工作流重叠（与 v1 定位"工作流重新组合的泛化"一致，但 held-out 结论要相应说明）。自动统计符合条件的有 7 道，其中 2 道同时是 held-out 候选，只能二选一 |
| 暂缓 4 题改编；D6D-2 选 ② | ≤ 7 | ~60 | 逐题改编；建用户类价值有限 |
| 新写 Bench2 原生任务（不来自上游） | 按需 | 60–70 | 需要另立一步 |

**结论**：
- 只靠上游题库，现实上限约 50–55 题，前提是 D6D-1 采用固定解释，并且放宽一次门槛。
- 要达到 60–70，需要新写任务。
- 其余 45 道上游题连着 dev/held-out 家族，放进训练集会污染评测，只能通过 v3 进入 dev/held-out，不能进 train。

### 4.2 held-out：能否增加独立任务

能增加，但**只能扩充已有的 held-out 家族**，上游题库里没有能构成新 held-out 家族的题。

- **自动统计**：把本批算进 train 之后，满足 held-out 门槛、没有自动 policy 风险的有 8 道。门槛是：不连 train/dev 家族，与任何 train 任务共享写工具 ≤ 1、共享实体 < 2，不是近重复，验证器可用。这 8 道分成两组：
  - **服务目录组 4 题**：`0be70e35_0916fba8`、`99ba2325_54f35603`、`99ba2325_ae6b0778`、`c5ca08eb_3c4271f0`，并入 `service_retirement`（目前只有 1 题）；
  - **问题创建组 4 题**：`66392a82_ccb66a9a`、`78a5b71e_cab5cc9f`、`97d7df7d_ece487bd`、`99ba2325_dd71c332`，并入 `problem_from_incident_and_change_mapping`（目前 5 题）。其中 `ece487bd` 同时连两个 held-out 家族，加入后会把它们合并。
- **规模**：held-out 最多从 11 题增加到 19 题。按任务计的标准误（p = 0.5 时）从约 0.15 降到约 0.11。家族数不变，多跑几个评测 seed 也不等于增加独立任务。
- **dev**：只连 dev 家族的 18 题全部属于 `incident_hold_and_notify`，加进去会让 dev 更单一，不建议。
- **新的独立 held-out 家族**：需要新写任务（或专门为新工作流设计的改编题），并且要在任何模型结果出来之前按预先定好的规则选入。

具体做法见 `v3_proposal.md`。v1、v2 保持不变。

## 5. 未验证的边界

- **自动标记只是提示**：policy 标记基于对验证器等值条件的正则解析和种子统计。知识匹配是否唯一、PROPOSED 和 ASK 应该怎么回应，都靠人工判断，6D-2 回放时才真正验证。
- **gold 写工具是计划值**：本批的家族复查用的是计划写工具，6D-2 写出 gold 后要按实际 gold 再算一次。
- **长度是估算**：估算响应 = 调用次数 × v2 的每步中位数。未过滤读取是在切分范围内实测的，但还没有逐题的 gold episode。
- **切分与上游一致性**：MSP 题按组织切分后，"切分种子 + delta = 上游同一切分"还没有逐题验证。2 号题用的是上游的种子变体（delta 4 条），6D-2 要单独核对。
- **v3 的 held-out 候选**只过了自动门槛，还没有人工审读。
