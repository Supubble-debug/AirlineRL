# 第 6D-2 步：7 题迁移验证（逐题结果，待验收）

范围：只处理 2026-10-30 批准的 6D-1 第 1–4、6–8 题，全部走忠实迁移路线。第 5、9、10 题暂缓，没有处理。本步没有修改 policy、评分、v1/v2 划分或预算数值，没有制定 MSP SLA 规则，没有放宽门槛，没有启动 GPU，没有调用 LLM，也没有把任何题并入划分。

**结果：4 题通过（第 1、3、6、7 题），3 题隔离（第 2、4、8 题）。** 不声称与 EnterpriseOps-Gym 原评测等价。

| 产物 | 位置 |
|---|---|
| 批次文件 | `configs/data/itsm_eops_batch2_v1.json`：4 个通过的 Bench2 Task，附来源、上游验证器副本、转换说明、信息边界、反例；3 条隔离记录，其中两题附有已完成的转换（`attempted_conversion`），便于复核 |
| 构建脚本 | `scripts/data/build_itsm_eops_batch2.py`：转换内容手写在脚本里；`--check` 可复现，需要参考目录 |
| 验证与测量 | `scripts/data/validate_itsm_eops_batch2.py`，结果写入 `docs/itsm_step6d2/batch_validation.json` |
| 代码 | `src/agent_rl/data/itsm_eops_batch2.py`：批次路径，以及按 v2 做的家族和重叠复查 |
| 测试 | `tests/unit/test_itsm_eops_batch2.py`（27 项，只读批次文件） |

## 1. 验收标准

沿用 6D-1 §2.3 定下的标准，没有放宽。每题都要满足以下全部条件：

1. 用严格回放在任务时钟下回放 gold，Bench2 DB（fuzzy）= 1；第二次回放的最终 DB 哈希相同。
2. 去重后的上游验证器在 gold 上全部通过；在初始状态下不全通过，而且初始状态下就通过的，必须恰好是 `initial_pass_allowed` 里逐个登记、写明理由的那几个（"不得做某事"的约束或本来就成立的前提）。
3. 每个反例：Bench2 DB = 0，出现预期的 mismatch；上游验证器的判定与记录一致。
4. 信息边界：reset 观察和公开信息里不含任务描述、模拟器指引、NL 断言、task_id、上游 task_id 和验证器 SQL、转换说明，也不含 "EnterpriseOps" 字样。
5. 脚本化 episode 以 `agent_done` 结束，且 DB 匹配。
6. "Bench2 种子 + delta = 上游初始状态"：在不切分的 DB 上比较，忽略时间戳。切分是只依赖内容的确定性函数，输入相同，切分结果就相同；这 4 题的上游验证器都不写死新建记录的 ID。
7. 家族复查：在 v2 的 56 题加本批上重算。本批所在家族只含 train 成员；与任何 dev/held-out 任务共享写工具 ≤ 1、按 **gold 参数**计的同种子实体 < 2；dev/held-out 家族不变。

## 2. 通过的 4 题

都按组织切分，范围是 provider 内部的 `["ORG_001"]`，种子是 MSP 主种子加 6B 同款 delta（删除 CRM_017、PRB_011，CHG_017 改回 scheduled）。

| 项 | 第 1 题 `itsmbench_eops_99ba2325_887f5580` | 第 3 题 `itsmbench_eops_01395ded_eff85dae` | 第 6 题 `itsmbench_eops_46ca2862_56a4f834` | 第 7 题 `itsmbench_eops_99ba2325_7915481a` |
|---|---|---|---|---|
| 内容 | 退役 CI_041；把 CI_096 作为受影响 CI 关联到 INC_026；**拒绝**关联已退役的 CI_041 | 把 INC_001 唯一一条 queued 通知改为 failed；给来电人发 update，给负责人和组经理发 reminder | 在 Enterprise Email 下新建 3 个 offering；把 TMPL_004 的 offering 改为新建的 Mobile Email Access | 发布外部可见的知识文章；把 CI_001 的序列号改为新硬件的序列号；SVC_007 恢复为 operational |
| 操作员 | Karen（admin） | Karen | Karen | Joanne（manager） |
| gold 动作 | 6 | 11 | 8 | 6 |
| 种子 + delta 与上游的差异 | 0 | 0 | 0 | 0 |
| 上游验证器：gold 后 / 初始状态 | 3/3 / 1/3（"不得关联 CI_041"是已登记的约束） | 4/4 / 0/4 | 5/5 / 1/5（"模板 service 为 Enterprise Email"是已登记的前提） | 3/3 / 0/3 |
| 反例（全部 DB=0），其中上游验证器全部通过的个数 | 4 个，0 个 | 4 个，1 个 | 4 个，1 个 | 4 个，0 个 |
| 脚本化 episode：轮数 / 步数 | 9 / 9 | 14 / 14 | 11 / 11 | 9 / 9 |
| token：初始 prompt / 响应 / 完整序列 | 28,133 / 1,682 / 29,815 | 28,126 / 3,602 / 31,728 | 28,122 / 4,918 / **33,040** | 28,145 / 2,435 / 30,580 |
| 单次生成最大 / 单条观察最大 | 53 / 347 | 300 / 360 | 221 / 2,720 | 362 / 447 |
| 所在家族（v2 + 本批） | 新家族（单独成族） | 新家族（单独成族） | 训练合并家族（与第 7 题一起） | 训练合并家族 |

**说明**：

- **policy 判断点**：
  - 第 1 题：§5.1 规定已退役的 CI 不得关联到活动事件单，所以 agent 要说明拒绝的原因。
  - 第 3 题：通知文本属于操作员决定的事（§1.2 ASK），由操作员给出。来电人账号是停用状态，但通知收件人不属于 §8.1 说的"活跃参与者"，模拟器指引写明照常发送。
  - 第 6 题：§5.2 规定 offering 继承父服务的 critical / production。
  - 第 7 题：§7 规定经理可以直接发布文章，不必走草稿审核。
- **转换补充**：
  - 第 6 题：操作员提供 3 个 short_description。
  - 第 7 题：操作员澄清两点：只改现有 CI 的序列号，不新建 CI；服务已经恢复。另外，上游正文里的字面 `\n` 改为真正的换行。
- **反例中上游无法识别的情况**：
  - 第 3 题：把已送达的 NOTIF_002 也改成 failed；
  - 第 6 题：把 offering 挂到错误的父服务下。
  这两种错误上游验证器都发现不了，只有 Bench2 的全库比对能识别。
- **长度**：
  - 第 6 题的完整序列 33,040 超过原生上下文 32,768，主要因为 `find_service_offerings` 的观察有 2,720 token。它仍低于候选值 40,960；v1 的最大值是 33,868，本来就超过原生上下文，所以结论与现有预算一致，40,960 仍然待验证。
  - 4 题的单次生成都不超过 362，低于 `max_action_tokens=512`。
  - 脚本化 episode 的工具调用全部来自 gold，计划和回顾消息用模板生成，用户回复是固定文本，所以这些数字不代表实际策略的输出分布。

## 3. 隔离的 3 题

| 题 | 原因 | 证据（由验证脚本或测试重新计算） |
|---|---|---|
| 第 2 题 `c5ca08eb_67009578` | REQUIRED 连带操作结果不唯一 | 把 PRB_009 的影响改为 high，会使关联的 INC_061（ORG_013）优先级从 moderate 升为 high。§6 要求"优先级变化时重置、替换或取消 incident_sla 行"，但 INC_061 没有 SLA 行。种子里客户事件单也使用 provider 的 SLA 定义（例如 ORG_013 的 INC_049 挂着 SLA_002），所以有两种合理做法：什么都不做；或者按新优先级补建一行，而 high 又有 SLA_003、SLA_004 两条候选。要放行需要 MSP SLA 规则，本轮你明确不定。**6D-1 中"ORG_013 没有 SLA 定义，所以不用做 SLA 操作"的说法是错的，已更正**（6D-1 README §7） |
| 第 4 题 `78a5b71e_a4fb07a9` | 核心要求不被 Bench2 评分识别 | 反例 `server_named_in_description_only`（状态改为 review，short_description 不改，把服务器名写进 description）的 Bench2 DB = **1**。description 是 gold 没有改动的自由文本字段，按 D8 不参与比对；没改的 short_description 和 gold 文本（只在原文后加了一句）做 fuzzy（按 token 召回）比对，仍然算匹配。所以"记录 SRV-LNX-4421"这一核心要求不受训练 reward 约束。我**没有**在看到失败之后改写操作员文本去迎合评分；如果要重新设计，需要你决定 |
| 第 8 题 `97d7df7d_8fd400d1` | 与 held-out 共享 2 个同种子实体 | gold 用到的 GROUP_001 和 INC_003，也出现在 held-out 题 `itsmbench_single_012`（`incident_handoff_with_child`）的 gold 里。6A 门槛是"与 dev/held-out 共享实体 < 2"，因为放进训练集就会让 held-out 用到的记录出现在训练中。6D-1 筛选只看了验证器里出现的实体，所以漏掉了；6D-2 按 gold 参数复查，和 6B 的做法相同 |

第 4、8 题的转换和反例都已经完成，保存在批次文件的 `isolated[].attempted_conversion` 中。两题的 gold 都是 DB=1，第 8 题的 4 个反例也都是 DB=0。隔离只出于上表所列的原因，没有其他问题。

## 4. 对 6D-1 的更正

详见 `docs/itsm_step6d/README.md` §7，以下更正都已反映在重新生成的 `candidate_screening.json` 中：

1. **MSP 的 SLA 定义对所有组织通用**：有 3 道 MSP 建单题从"SLA 做不到"改为"SLA 不唯一"。按 D6D-1 ①，结论仍是排除。
2. **实体门槛**：6D-1 的自动门槛只看验证器里出现的实体，6D-2 改为按 gold 参数复查。
3. **筛选脚本的不确定性**：验证器取值互相冲突时，原脚本取到哪个值取决于 Python 的字符串哈希种子，所以每次运行可能不同。现已改为排序取值并保留全部取值，在 3 个不同的哈希种子下 `--check` 都能复现。受影响的只有"验证器优先级与矩阵矛盾"的计数（2 → 1），入选批次、各题人工结论和 v3 可用量都没有变化。

## 5. 对规模的影响

- 本批通过 4 题。如果你验收通过并决定纳入，train 会从 32 增加到 36。
- 加上暂缓的第 5、9、10 题和备选 2 题，在现行规则下 train 的近期上限约 41。
- 6D-1 估计"只靠上游题库约 50–55"，那需要 D6D-1 和放宽门槛，这两项你本轮都没有采纳。所以达到 50 以上，需要新写任务，或者重新评估那些规则。
- held-out：v3 的服务目录组和问题创建组候选不受本步影响，验证完成后再冻结。

## 6. 运行记录

在 `code/main` 下运行，日志在 `/home/sar/work/EnterGym/output/itsm-step6d2/`：

- `python scripts/data/build_itsm_eops_batch2.py --ref-dir /home/sar/work/EnterGym/reference/enterpriseops-gym --check`：up to date（哈希种子 1、7 下相同）；
- `python scripts/data/validate_itsm_eops_batch2.py --tokenizer-dir ~/.cache/entergym/tokenizers/Qwen3-8B-b968826d`：4 题全部检查通过，家族复查 5/5 通过；
- `python scripts/data/screen_itsm_eops_candidates.py --tokenizer-dir … --check`：up to date（哈希种子 1、2、3 下相同）；
- `pytest tests/unit tests/remote -q`：695 passed，跑完没有残留子进程。

## 7. 需要你决定

- **是否验收 4 道通过题**：第 1、3、6、7 题，以及是否纳入训练集。纳入会产生新的划分版本，按你的要求放在 v3 中一起处理，v1/v2 不改。
- **第 4 题**：是否重新设计。比如由操作员给出一段与原文差异很大的完整 short_description，使 fuzzy 比对能区分"写了服务器名"和"没写"。这属于看到验证结果之后的改写，需要你明确授权。
- **第 2、8 题**：在现行规则下保持隔离。
