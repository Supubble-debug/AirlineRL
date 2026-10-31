# 第 6C 步：数据与预算整合（待验收）

范围：只把 6B 获批的 3 题并入训练集 v2，其余 5 题继续隔离。歧义解释、SLA 规则和政策要求都没有改，规模也没有扩大。没有启动 GPU，没有加载权重，没有调用真实 LLM。

| 产物 | 位置 |
|---|---|
| v2 manifest | `configs/data/itsm_split_v2.json`（v1 `configs/data/itsm_split_v1.json` 保留，sha256 `77cc5519…`，未改动） |
| 生成与校验 | `src/agent_rl/data/itsm_split_v2.py`，`scripts/data/build_itsm_split_v2.py [--check]` |
| 显式任务解析入口 | `src/agent_rl/data/itsm_task_sets.py`；`ItsmEnvConfig.split_version` |
| v2 token 测量 | `docs/itsm_step6c/token_measurements_v2.json`（45 个 train/dev 任务） |
| 预算依据 | `configs/itsm/budget_v1.yaml` 新增 `basis` 段，**数值未改** |
| 测试 | `tests/unit/test_itsm_split_v2.py`、`test_itsm_task_sets.py`、`test_itsm_budget_v2.py`；`test_itsm_split.py` 增加反例 |

## 1. v2 manifest

- **规模**：train 32 / dev 13 / held-out 11，共 56 题。dev 和 held-out 列表与 v1 逐项相同（包括顺序）。53 道 Bench2 题都保留 v1 的划分，划分**没有重新优化**。
- **原 53 题定义不变**：
  - Bench2 数据文件的校验值与 v1 记录一致（`data.bench2 == v1.data`）；
  - 每题记录 `content_sha256`，即任务定义字段 `id, scenario, evaluation_criteria, org_ids, org_id, seed_db, initial_state_delta, current_time, tenancy` 的规范 JSON 的 sha256。测试会用 `tasks.json` 重新计算比对。
- **新增 3 题**，`source = enterpriseops_gym_derived`，各自带 `derived_from` 字段：

| task_id | 上游 task_id | 家族 | 业务类型 | tenancy | content_sha256 |
|---|---|---|---|---|---|
| `itsmbench_eops_1d219d93_d1ba6c2d` | `task_20251223_101136_983_1d219d93_d1ba6c2d` | 合并家族 | cmdb | msp_provider_internal | `36108549851b8f2fffd35303d03e5e7b9612659dd1948c8377fd8055e3be2704` |
| `itsmbench_eops_46ca2862_b6ef60ca` | `task_20260113_182602_664_46ca2862_b6ef60ca` | 合并家族 | incident_lifecycle | single_tenant | `d4001e9d5128a7a72d505938af1f4648b0f5dff1a6d8203eb50d0ba2e738677c` |
| `itsmbench_eops_66392a82_52c8290e` | `task_20251218_124419_383_66392a82_52c8290e` | `incident_intake_with_ci_update`（新） | incident_lifecycle | single_tenant | `b03debc7b6c4cf27875bb06bc083c43a33a151942b07904b6795e071a6985866` |

- **来源与派生标注**：`derived_from` 记录以下内容：
  - 数据集 EnterpriseOps-Gym；repo commit `271f2c3`；HF revision `c8e538ea`；Apache-2.0 及署名；oracle 模式；
  - 上游 prompt 与种子的 sha256；试点文件版本与路径；
  - `multi_turn_rewrite: true`、`org_slicing: none`；
  - 逐题 `conversion_notes`（多轮改写和补充信息，原文照搬自试点文件）；
  - `scoring_differences`（不等价说明，详见 6B README §3）、`equivalent_to_upstream_evaluation: false`；
  - 你的批准原话与日期。
- **版本固定**：manifest 的 `data` 段同时固定 Bench2 文件校验值、试点文件 sha256（`7aa17c58…`）和 v1 manifest sha256。试点文件只改了 `status` 一行，现在写明"3 题已纳入 v2，5 题继续隔离"。

### 1.1 家族重算（56 题，规则与 v1 相同）

规则不变：写工具 Jaccard ≥ 0.6，或共享新建对象工具，就连成一个家族；家族取连通分量。命名规则如下：
- 成员与 v1 **完全相同**的家族，沿用 v1 名称；
- 其他家族（合并、扩展或新建）一律使用 v2 新标签。v2 标签若与 v1 名称重名，构建直接报错。

| 变化 | 内容 |
|---|---|
| 不变（9 个） | `change_with_ci_and_kb_links`、`ci_maintenance`、`incident_handoff_with_child`、`incident_hold_and_notify`、`incident_reassign_with_ci`、`incident_resolve`、`knowledge_update_and_notify`、`problem_from_incident_and_change_mapping`、`service_retirement` |
| **合并** | v1 train 的 `onboarding_and_catalog_admin`（11）与 `change_problem_advance_with_kb`（17），被 `1d219d93`（同时新建 CI 和知识文章）连成一个分量；`b6ef60ca` 通过 `register_configuration_item` 也并入。新名 `onboarding_catalog_and_change_problem_kb_merged`，共 **30 题**。两个 v1 名称在 v2 中不再出现 |
| **新增** | `incident_intake_with_ci_update`：只有 `52c8290e`（更新 CI → 建事件单 → 关联 SLA）。它与 `b6ef60ca` 的写工具 Jaccard 为 2/4 = 0.5，`create_incident` 也不在新建对象工具清单里，因此两者不相连 |
| 数量 | v1 11 个家族 → v2 11 个（−2 合并 +1 合并后 +1 新增）。train 家族：v1 为 3 个（11/17/1），v2 为 3 个（**30**/1/1） |

需要你知道：**train 里有 30/32 题属于同一个家族**。这是连通分量的传递效果（1 道题把两个 v1 家族连在一起），不表示这 30 题彼此相似。家族只用来保证划分不跨家族，dev/held-out 不受影响。如果后续要按家族统计 train 指标，应改用 v1 家族名加 `derived` 标签，不要直接使用这个合并分量。

业务类型改为**按题记录**：原 53 题沿用 v1 的类型；新 3 题按主要请求人工标注；家族只列出成员的类型集合。原因是合并家族横跨 4 种类型，无法用单一类型表示。

### 1.2 一致性校验与反例

`validate_split_structure`（v1、v2 共用，v1 也一并收紧）对以下三处**分别**检查：
- **三个划分列表**：无重复、两两不相交、恰好覆盖全部题目；
- **逐题映射**：ID 集合正确，每题的 `split` 与它所在的列表一致；
- **家族**：每题恰好属于一个家族，与 `family` 字段一致；家族不跨划分，并与声明的 `split` 一致。

v2 另外检查：
- 规模 32/13/11；dev/held-out 与 v1 相同；53 题不改划分；
- 派生题恰好是批准的 3 题，都在 train，且有标注；
- 数据与试点文件的校验值、每题内容校验值一致；
- 最后与重新构建的结果整体比对。

反例（测试中全部被拒，并核对报错原因）：

| 类别 | 反例 |
|---|---|
| 划分列表 | 只在两个列表间互换 ID（映射不动）、列表内重复、跨列表重复、漏 ID、插入隔离题 ID |
| 逐题映射 | 映射里 split 被改、映射缺题 |
| 家族 | 家族成员重复、一题属两家族、成员缺失、`family` 字段错误、家族 split 错误、沿用失效的 v1 家族名 |
| 规模与版本 | 派生题被一致地移到 dev（触发规模检查）、声明规模错误、dev 顺序变动、派生题被标为 bench2 |
| 校验值与其他 | 内容校验值、试点文件校验值、分布统计被改、版本号错误 |

v1 的反例也补充了"只互换列表""列表内重复""family 字段错误"三项。

## 2. 显式任务解析入口

- `itsm_task_sets.load_task_set(*, split_version)`、`resolve_task(task_id, *, split_version)`、`task_ids(split, *, split_version, final_evaluation=False)`。`split_version` 是**必填关键字**，没有默认值，不认识的版本直接报错。held-out 仍然需要 `final_evaluation=True`。
- `itsm_split_v1` 包含 53 道 Bench2 题。`itsm_split_v2` 再加入 3 道派生题，每题加载时都会用 v2 manifest 的内容校验值核对。
- `ItsmEnvConfig.split_version`：
  - 默认 `None`：环境保持原有查找方式，只从 Bench2 `tasks.json` 取题（53 题），派生题 ID 会报 unknown，**不会静默切换版本**；
  - 设为 `itsm_split_v1` 时，派生题同样报错；
  - 设为 `itsm_split_v2` 时，按 v2 解析；
  - 与 `task=` 同时传入时报错。
- **worker**：`ItsmEnvConfig` 本身可以 pickle，子进程按 config 自己解析任务，不经管道传递 Task 对象。
- 测试结果：
  - 3 道派生题在**进程内 env** 和 **worker 子进程**中都能加载，跑完 gold 后 DB=1；
  - worker 不带版本时，启动阶段就拒绝派生题；
  - reset 和每步观察中都不含以下私有信息：task_id、上游 task_id、commit、任务描述、模拟器指引、NL 断言、转换说明、上游验证器 SQL，以及 "EnterpriseOps" 字样。解析得到的只是 Bench2 `Task`；manifest 和试点文件中的来源字段不会进入环境。

## 3. token 测量与预算依据

`measure_itsm_tokens.py` 增加了必填参数 `--split-version`，每个版本有各自的默认输出路径，v1 文件不会被覆盖。改动后用 v1 重跑一遍：static、summary、每个 episode（不含耗时）、未过滤读取等结果与已提交的 v1 文件**完全一致**，说明重构没有改变 v1 的任何数字。

### 3.1 gold 路径（45 题，脚本化 gold 回放）

| 指标 | v1（42 题）最大 | v2（45 题）最大 | 3 道派生题 |
|---|---|---|---|
| 初始 prompt | 28,104 | 28,104 | 28,104 |
| 响应 | 5,764 | 5,764 | 2,010 / 2,139 / 2,586 |
| 完整序列 | 33,868 | 33,868 | 30,114 / 30,243 / 30,690 |
| 单次工具调用生成 | 127 | 127 | ≤ 82 |
| 单条工具观察（原始） | 2,690 | 2,690 | ≤ 348 |
| 底层步数 | 18 | 18 | 10 / 10 / 12 |

gold 路径的最大值全部不变（6B 实测的 30.4k–31.1k 包含模板化的计划和回顾消息，这里与 v1 一样不包含）。

### 3.2 未过滤只读工具（不带参数调用，按各题自己的加载方式）

派生题**不做组织切分**，加载整个种子；Bench2 原题都按组织切分。变化集中在 MSP 派生题 `1d219d93`，它加载了全部 20 个组织：

| 工具 | v1 最大 | v2 最大（`1d219d93`） |
|---|---|---|
| `list_incidents` / `get_incidents_assigned_to` | 9,656 | **30,041** |
| `list_users` | 10,371 | **15,137** |
| `find_configuration_items` | 6,876 | 12,001 |
| `find_incident_slas` | 2,265 | 4,190 |
| `find_notifications` | 1,394 | 3,688 |

另外两道单租户派生题，单题最大为 7,415（`list_incidents`）。v1 中按组织切分的单租户题最大为 4,177。

**预算影响**：
- v1 的所有未过滤读取都能放进 `max_response_length=12,288`；
- v2 中有 3 个工具在 `1d219d93` 上单次调用就超过 12,288，而且 `list_incidents` 的 30,041 也超过 40,960 − 28,104 = 12,856 的上下文余量；
- `find_configuration_items` 的 12,001 只有在它是第一个动作时才放得下。

按现行规则（完整观察，不轮换上下文），这类调用会以 `truncate("response_length")` 结束 episode，trace 保留完整观察。这 3 个工具的 gold 都不需要无参调用：gold 用的是 `get_user_using_name` 和按地点过滤的 `find_configuration_items`。

### 3.3 预算提案（待你验收，未自行宣布已验证）

`budget_v1.yaml` 的**数值不变**，只新增 `basis` 段：
- 记录 v1 和 v2 的测量依据；
- `itsm_split_v2.status = pending_step6c_acceptance`、`values_changed: false`；
- 列出超过 12,288 的 3 项未过滤读取。`test_itsm_budget_v2.py` 会核对这份清单与测量文件一致。

40,960 仍然只是候选值，原有的第 7 步和第 9 步验证要求不变。

关于 3.2，有以下选项，需要你决定：

| 选项 | 内容 | 影响 |
|---|---|---|
| **A（建议）** | 保持现状：完整观察，超长时以 response_length 结束 | 只影响 1/32 道 train 题上的宽泛读取；规则与 v1 一致。第 7 步需按题统计 response_length 截断率，`1d219d93` 单独列出 |
| B | 对 `1d219d93` 做组织切分 | 会改变新建记录的顺序 ID（`LOC_024`、`CI_101/102` 与上游验证器写死的 ID 不一致），相当于改动 6B 已验收的题目定义，需要重新验收 |
| C | 提高 `max_model_len` | 需要 RoPE/YaRN 扩展，与"不加 RoPE"的现行约束冲突 |
| D | 从 v2 移除 `1d219d93` | 与你的批准不符，仅列出备查 |

## 4. 运行记录（`code/main`，日志 `/home/sar/work/EnterGym/output/itsm-step6c/`）

- `build_itsm_split.py --check`：v1 split manifest and family audit are up to date
- `build_itsm_split_v2.py --check`：up to date
- `build_itsm_eops_pilot.py --ref-dir … --check`：up to date
- `measure_itsm_tokens.py --split-version itsm_split_v1 --out …/measure_itsm_split_v1.json`：与已提交的 v1 文件比对一致
- `measure_itsm_tokens.py --split-version itsm_split_v2`：退出码 0，45 个 episode 的 db_match 全部为 True
- 全量回归：655 passed（`pytest_full.log`）
