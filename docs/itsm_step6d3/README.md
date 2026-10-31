# 第 6D-3 步：v3 held-out 候选审读与迁移验证

状态：2026-10-31 验收通过，`99ba2325_54f35603` 纳入 v3 held-out（`docs/itsm_v3_usage/README.md`）。

范围：只处理 `docs/itsm_step6d/v3_proposal.md` 列出的 8 道候选。不另找题替补，不放宽门槛，不生成正式 v3 manifest，不运行模型，不测量 held-out 候选的 token 长度，也不用它们调试预算或模型配置。

## 1. 候选清单与准入规则（转换之前固定）

本节在转换任何一题之前写定并提交，之后不改。

### 1.1 候选（8 题，固定）

| 组 | 上游后缀 | 上游种子 → Bench2 种子 | 上游操作员 |
|---|---|---|---|
| 服务目录 | `0be70e35_0916fba8` | `db_1765301900121` → 单租户 | Raj Patel |
| 服务目录（优先留给 held-out） | `99ba2325_54f35603` | `db_1765893060767` → MSP | Karen Watkins |
| 服务目录（优先留给 held-out） | `99ba2325_ae6b0778` | `db_1766091624245` → MSP | Karen Watkins |
| 服务目录 | `c5ca08eb_3c4271f0` | `db_1765949061488` → MSP | Karen Watkins |
| 问题创建 | `66392a82_ccb66a9a` | `db_1765301900121` → 单租户 | Marcus Thompson |
| 问题创建 | `78a5b71e_cab5cc9f` | `db_1765301900121` → 单租户 | Raj Patel |
| 问题创建 | `97d7df7d_ece487bd` | `db_1765301900121` → 单租户 | Marcus Thompson |
| 问题创建 | `99ba2325_dd71c332` | `db_1765893060767` → MSP | Karen Watkins |

### 1.2 对照集合

v2 的 56 题，加上 6D-2 获批的 4 道训练题（`itsmbench_eops_99ba2325_887f5580`、`itsmbench_eops_01395ded_eff85dae`、`itsmbench_eops_46ca2862_56a4f834`、`itsmbench_eops_99ba2325_7915481a`）。即 train 36、dev 13、held-out 11。旧的筛选结论只作参考，以下检查全部按候选题**实际写出的 gold** 重新计算。

### 1.3 准入条件（全部满足才通过）

1. **家族**：用第 6 步的规则（写工具 Jaccard ≥ 0.6 或共享新建对象工具）在对照集合加候选上重算。候选所在的连通分量里不得有 train 或 dev 成员；可以有 held-out 成员和其他候选。分量合并了原有的 held-out 家族时照常准入，但要记录。
2. **写工具**：与任何 train **或 dev** 任务共享的写工具 ≤ 1。（比 v3 提案多对照了 dev，因为 dev 用于调试。）
3. **同种子实体**：按 gold 参数中的记录 ID 计（不含 delta 新建的记录），与任何 train 或 dev 任务共享 < 2。
4. **近重复**：与对照集合中任何任务（含 held-out），或与另一道候选，满足三条——同一种子、gold 写工具集合完全相同、共享 ≥ 2 个 gold 实体——的，视为近重复，隔离。近重复只增加相关的观察，不增加独立信息。
5. **只走忠实迁移路线**。held-out 本轮不收改编题。标准同 6B/6D-2：
   - gold Bench2 DB（fuzzy）= 1，重复回放后最终 DB 哈希相同；
   - 去重后的上游验证器在 gold 上全部通过；在初始状态下不全通过，初始就通过的必须逐个登记并写明理由；
   - Bench2 种子 + delta = 上游初始状态（在不切分的 DB 上比较，忽略时间戳）；
   - 信息边界：reset 观察和公开信息中不含任何私有文本；
   - 脚本化 episode（gold 工具调用、固定用户回复）以 `agent_done` 结束且 DB 匹配。**不做 token 测量。**
6. **核心要求反例**：每题在验证之前先列出核心要求，每项至少配一个针对性反例，反例必须得到 Bench2 DB = 0。核心要求落在自由文本上时，必须有\"其余内容都正确，只遗漏或写错关键事实\"的反例；如果它得到 DB = 1，说明这项核心要求在训练和评测奖励中没有被区分，该题隔离。D8、fuzzy 阈值（0.5，按 gold 实词召回率计）和评分都不改。
7. **policy**：以下情况一律隔离（6B 口径）：REQUIRED 连带操作结果不唯一；REQUIRED 步骤没有工具可做；上游期望状态与强制 policy 冲突；prompt 与验证器矛盾（包括有歧义的数字写法）；请求中的关键对象有多个候选，而 prompt 没有给出区分依据。
8. **切分范围**：任务的判断依赖跨组织记录时，范围必须包含这些记录，并检查判断用到的量在切分前后相同。验证器写死了新建记录 ID 的题，不切分。

不通过的候选隔离并记录理由和可复核的证据。

## 2. 结果：1 题通过，7 题隔离

| # | 候选 | 结论 | 原因（一句话） |
|---|---|---|---|
| 1 | `0be70e35_0916fba8` | 隔离（转换后） | 核心自由文本要求\"注明服务级、反复出现、正在调查\"不被评分区分 |
| 2 | `99ba2325_54f35603` | **通过** | 全部检查通过 |
| 3 | `99ba2325_ae6b0778` | 隔离（转换前） | prompt 自相矛盾，通知该挂在哪张事件单上不唯一 |
| 4 | `c5ca08eb_3c4271f0` | 隔离（转换前） | 上游要求关闭两张 new 状态的事件单：prompt 没要求，也违反 §3.3 |
| 5 | `66392a82_ccb66a9a` | 隔离（转换后） | 关键决策\"选哪篇文章\"只体现在通知正文里，写错也得 DB = 1 |
| 6 | `78a5b71e_cab5cc9f` | 隔离（转换前） | prompt 与验证器的逐字消息不一致；验证器要求状态 sent |
| 7 | `97d7df7d_ece487bd` | 隔离（转换后） | 两项自由文本核心要求不被区分；与 train 共享 2 个实体 |
| 8 | `99ba2325_dd71c332` | 隔离（转换前） | `$9,50` 有歧义，只有验证器给出 950 |

两道优先留给 held-out 的服务目录题：`54f35603` 通过，`ae6b0778` 隔离。按规定没有另找题替补。

## 3. 通过的题：`itsmbench_eops_99ba2325_54f35603`

- **内容**：计划退役 SAP ERP。先检查仍在进行的事件单，多于 3 张就不退役，改为把服务和它的 3 个 offering 都设为 non-operational。
- **范围**：MSP 主种子加 6B 同款 delta，范围是 `ORG_001 + ORG_007 + ORG_016`。正在进行的事件单有 4 张：INC_001（进行中），INC_044、INC_069、INC_082（挂起）。如果只看 provider 的 ORG_001，只能看到 2 张，判断会变成\"可以退役\"。验证脚本确认切分范围内和不切分时都是 4。\"进行中\"按 policy 的 active 口径计（不是 resolved/closed），挂起也算，模拟器被问到时也这样回答。
- **gold**：8 个动作；种子加 delta 与上游的差异为 0；上游验证器在 gold 上 4/4 通过，初始状态 0/4。
- **核心要求与反例**（5 个反例全部 DB = 0，上游验证器也都识别）：

| 核心要求 | 反例 |
|---|---|
| 多于 3 张时不退役 | `retired_instead`（服务和 offering 都改为 retired）、`on_hold_not_counted`（不把挂起单算进去，结果退役了服务） |
| 服务设为 non-operational | `service_not_updated` |
| 每个 offering 同状态 | `one_offering_missed`、`offerings_wrong_status` |

  这道题不涉及自由文本。
- **其他检查**：重复回放的最终 DB 相同；信息边界没有泄漏；脚本化 episode 以 `agent_done` 结束且 DB 匹配。
- **policy**：non-operational 不触发任何连带操作；没有退役，所以 §5.2 的退役连带操作也不适用。

## 4. 隔离的 7 题

### 4.1 转换后隔离（gold DB = 1，验证失败）

关键事实反例的 fuzzy 召回率由 Bench2 的 `text_overlap` 计算，阈值是 0.5。

| # | 失败的核心要求 | 反例：其余内容正确，只改关键事实 | 召回率 | Bench2 DB |
|---|---|---|---|---|
| 1 | 工作备注写明服务级、反复出现、正在调查 | 改为\"isolated and one-off\" | 0.90 | 1 |
| | | 删掉\"正在调查\"那一句 | 0.70 | 1 |
| 5 | 通知点名找到的文章 | 点名 `Desktop Boot Failure Troubleshooting` | 0.64 | 1 |
| | 问题陈述 \"Outlook connectivity failure\" | 改为 \"Outlook sync failure\" | 0.67 | 1 |
| 7 | 服务描述是主要的认证入口 | 改为\"次要的报表入口\" | 0.80 | 1 |
| | 问题陈述\"认证完全阻断\" | 改为\"认证间歇性变慢\" | 0.86 | 1 |

- 这 3 题的结构化核心要求（状态、优先级、选哪张事件单、选哪个 CI、负责人、收件人等），反例都得到 DB = 0。
- 第 7 题还违反了实体门槛：gold 与 train 题 `itsmbench_single_003` 共享 LOC_003、USER_002，与 `itsmbench_single_005` 共享 LOC_001、USER_002。另外它会把两个 held-out 家族合并。
- 三题的转换都保存在候选文件的 `isolated[].attempted_conversion` 中，由测试重新计算。

### 4.2 转换前隔离

证据中引用的上游 prompt 和验证器原文，构建时都会对照参考目录检查；种子事实由测试重新核对。

| # | 证据 |
|---|---|
| 3 | prompt 先要求把服务改为 repair in progress，又要求\"从服务角度视为已解决……即使部分事件单细节仍未补齐\"。验证器选了前一种理解；采用它等于用验证器改写 prompt，而带着缺失细节去解决事件单违反 §3.3。另外，判断\"多于 3 张\"需要所有组织的数据（共 9 张进行中），其中高优先级的有 INC_026、INC_068 两张，验证器选了 INC_026，但没有给出依据。\"提及高优先级\"本身也是自由文本 |
| 4 | prompt 只说服务退役、offering 不再支持；验证器另外要求关闭 INC_015、INC_064。这两张都是 new 状态，new → closed 不是合法流转；手动关闭只限 Manager，而 Karen 是 admin；其余 7 张进行中的单没有任何处理依据 |
| 6 | prompt 原文是 \"customer-facing systems\"，验证器逐字比对 \"customer facing systems\"（与 6B 的 `$64,00` 属同一类）；验证器还要求 status = 'sent'，而 `send_notification` 生成的是 queued，写成 sent 等于捏造投递结果（§1.3） |
| 8 | `$9,50` 可以读作 9.50 或 950，验证器取 950（种子值是 800），与 6B 隔离的 `66392a82_74b17170` 属同一类 |

## 5. 家族与重叠审计

对照集合：v2 的 56 题加 4 道获批训练题，即 train 36、dev 13、held-out 11。全部结果在 `heldout_validation.json`。

- **只含通过题（即 v3 会实际包含的）**：`54f35603` 的写工具是 `update_service`、`update_service_offering`。
  - 它与任何已有任务都不连边，所以单独构成一个新的 held-out 家族。它与 held-out 的 `service_retirement` 只有 Jaccard 0.5，没有共享新建对象工具。
  - 与 train 共享的写工具最多 1 个（与 `itsmbench_eops_99ba2325_7915481a` 共享 `update_service`）；与 dev 共享 0 个。
  - gold 实体是 SVC_017 和 SVCOFF_021/022/038，与任何任务都不共享 2 个以上，也没有近重复。
- **含全部 4 道转换题（审计用）**：
  - 第 7 题（新建服务、新建问题单）会把 `service_retirement` 和 `problem_from_incident_and_change_mapping` 连成一个分量，第 1、2、5 题也经由它并入。这说明候选之间、候选与 held-out 之间的工作流相关性很高；隔离第 7 题后，这次合并不再发生。
  - 第 1、5 题与 train/dev 共享写工具都不超过 1 个，也没有实体重叠。它们只因自由文本被隔离。

## 6. v3 预期规模

| 划分 | v2 | v3（若你验收本步） | 说明 |
|---|---|---|---|
| train | 32 | 36 | 6D-2 获批的 4 题 |
| dev | 13 | 13 | 不变 |
| held-out | 11（6 个家族） | 12（7 个家族） | 新增 `54f35603`，单独成族 |

**统计口径**：held-out 只多了 1 题，按任务计的统计功效基本不变。新增的这个家族只有 1 道题，不能单独支撑\"跨家族泛化\"的结论。实验对比应在同一批任务上做配对，按任务或家族做聚类重抽样，并逐题报告结果。v3 manifest 尚未生成，冻结要等你验收。

## 7. 需要你知道的事

- **自由文本是系统性的盲区**：按现行评分，自由文本字段只要覆盖 gold 一半的实词就算匹配。只有一个关键事实错误时召回率通常在 0.6–0.9，所以凡是核心要求落在自由文本上的题，都无法满足本步的反例标准。这次 7 道隔离题里有 3 道是这个原因，另有 2 道（第 3、8 题）也有同类问题。
- **已获批训练题的同类风险（推断，未实测）**：6D-2 获批的第 3 题（通知正文）和第 7 题（知识文章正文）当时没有做\"只改关键事实\"的反例。按同一阈值推断，大概率也区分不了。它们的结构化部分已经验证过（状态、收件人、类型、序列号等）。是否补测，等你决定。
- **规则差异**：本步规则比 v3 提案更严：共享写工具和实体也对照 dev，实体按 gold 参数计，增加了关键事实反例。规则在转换前写定并提交（`3c7e2b7`），没有在看到结果后修改。

## 8. 运行记录

在 `code/main` 下运行，日志在 `/home/sar/work/EnterGym/output/itsm-step6d3/`：

- `python scripts/data/build_itsm_eops_heldout_v3.py --ref-dir /home/sar/work/EnterGym/reference/enterpriseops-gym --check`：up to date（哈希种子 1、7 下相同）；
- `python scripts/data/validate_itsm_eops_heldout_v3.py`：退出码 0，输出在不同哈希种子下逐字节相同；不加载分词器，不测量 token；
- `pytest tests/unit/test_itsm_eops_heldout_v3.py`：17 passed；`pytest tests/unit tests/remote -q`：712 passed，跑完没有残留子进程。

| 产物 | 位置 |
|---|---|
| 候选文件 | `configs/data/itsm_eops_heldout_v3_candidates.json`（1 道通过题；7 条隔离记录，其中 3 条附带已完成的转换） |
| 构建 / 验证脚本 | `scripts/data/build_itsm_eops_heldout_v3.py`、`scripts/data/validate_itsm_eops_heldout_v3.py` |
| 准入检查 | `src/agent_rl/data/itsm_eops_heldout_v3.py`（`heldout_gate_check`） |
| 验证结果 | `docs/itsm_step6d3/heldout_validation.json` |
| 测试 | `tests/unit/test_itsm_eops_heldout_v3.py` |

没有生成 v3 manifest，没有开始第 7 步，没有启动 GPU，没有调用 LLM，没有做 SFT 或训练。
