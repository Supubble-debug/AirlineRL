# 第 7B 步：真实 GPU 推理、长度/显存与采样 logprob 验证（待验收）

范围：只做第 7B 步。没有更新权重，没有修改冻结数据，没有运行 held-out，没有生成 SFT/Teacher 轨迹，没有调用外部 LLM，没有新增付费 API 请求；单卡，只用 `exp-itsm` 现有的磁盘，没有扩盘、没有新租实例、没有删除其他项目的文件。结束后已确认关机。

## 结论

1. **资源**：累计 **约 3,517 秒 ≈ 0.98 GPU·h**（上限 3 GPU·h），从开机请求算到平台确认关机，开机、下载、编译、空闲和重试都计入。
   - 其中下载权重约 36 分钟；GPU 上真正做推理和重算的阶段合计约 7 分钟；其余是开机、失败重试和空闲。
   - 共开机 2 次，两次结束时都已确认关机（第 1 次是 3 次启动都失败后，由自动关机安全网按设计关掉的，见 §7.3；原文写成"误关"，2026-11-01 更正）。
2. **长度与显存**：
   - 32,768 和 40,960 两种 `max_model_len` 先后运行，都能正常启动。KV cache 都是 175,760 token（24.14 GiB），分别能容纳 5.36 条和 4.29 条满长序列，峰值显存约 46.0 / 47.6 GB（显存利用率按 0.45 设置）。
   - 约 28.1k、30.8k、40.4k 三档全部正常生成：以结束符结束，每个 token 恰好一个有限的 logprob。第 7A-R3 定下的 3 条工程判据全部满足。
   - **40,960 仍是候选**："工程上可运行"不等于自然长任务能力，也不等于长上下文质量已验证。约 40.4k 那一档是拼接出来的，只作工程压测。
3. **跨设置一致性（只作诊断）**：a 档在两种设置下逐 token 相同；b 档在第 77 个 token（共 89 个）分叉。分叉位置上，40,960 设置给两个候选的 logprob 完全相同（−0.7359），是一个精确平局。分叉之前的 |Δlogprob| 最大 0.10。
4. **真实 episode**：GRPO 可训练题 2 道、dev 2 道，每题 1 条，共 72 次采样。
   - 0 个非法动作，0 次截断，0 次 `max_action_tokens` 触顶；最长序列 36,083 token。
   - 4 条都以脚本化用户的 STOP 结束（`user_stop`），reward 都是 0：用户没有告诉目标，本来就不预期成功。
5. **logprob 对照**：
   - 72 次采样共 4,203 个策略 token，每个都恰好有一个有限 logprob。
   - 用同一权重、同一序列，在训练端重算（只统计 mask = 1 的位置）：|Δ| 中位数 0、均值 0.0056、p99 0.169、最大 0.748；1.8% 的 token 超过 0.1，没有超过 1.0 的。
   - 每条 episode 的累计差在 −0.08 到 +2.10 nat 之间。
   - **建议用重算模式**（`bypass_mode: false`，即仓库默认的 `grpo_sequence.yaml`），并在第 9 步用真实的 actor 配置复测，理由见 §5。
6. **发现**（详见 §7）：
   - Bench2 的 org 切片会让列表类工具返回结果的顺序随 Python 哈希种子变化；
   - 磁盘实际是 128 GB，与第 1 步记录的 30 GB 不同；
   - projtool 的旧关机定时器在我还在排查时关了机：当时已经没有运行中的任务，属于按设计关机，但打断了人工排查（原文写成"没有识别出正在运行的任务"，2026-11-01 更正）；
   - 运行时代码会读 `docs/` 下的文件，而 `docs/` 不同步到远端；
   - 4 条 episode 的第一个动作都是没有先查询的写调用；只有 msp_013 那一条执行成功，另外 3 条返回业务错误（`domain_error`）。（2026-11-01 更正：原文写成"写入成功了"，与轨迹不符。）

## 1. 资源记录

GPU·h = 从开机请求到平台确认关机的挂钟时间 × 1 张卡，按本机时钟计。远端时钟比本机快几十秒，所以远端日志里的 `elapsed_since_boot_s` 略偏大，偏保守。

| 时段（UTC，2026-11-01） | 秒 | 内容 |
|---|---|---|
| 第 1 次开机 03:53:50 → 约 03:58:40 | 300（保守计） | 3 次启动都失败了：① 环境报告本该"预期失败"的 flash_attn 检查让整个阶段返回失败（脚本 bug）；② 运行时要读的清单文件当时放在 `docs/` 下，而 `docs/` 不同步到远端（之后移到了 `configs/itsm/step7b/`）；③ 走代理用 HF 下载，Xet 的 CAS 返回 HTTP 401，下载阶段失败，运行以退出码 1 结束（远端 03:58:26）。随后第 ① 次运行留下的自动关机定时器在远端 03:59:01 到期，此时已经没有运行中的任务，按设计关机（§7.3）。（2026-11-01 更正：原文写成"30 秒内没有进展"，并说运行还在跑。） |
| 第 2 次开机 03:59:46 → 04:04:47 | 约 301 | 开机、环境和空间检查；HF 下载约 3–4 MB/s，太慢，取消（`cancel_run`） |
| 04:04:54 → 04:41:19 | 2,185 | 改从 hf-mirror 按固定 revision 并行下载 16.4 GB；最后一个分片单连接只有约 3 MB/s |
| 04:41:19 → 04:41:57 | 38 | 核对通过；复建长度输入失败：`docs/itsm_stepS0/usage_audit.json` 不在远端 |
| 04:41:57 → 04:45:03 | 约 186 | 空闲：上传这两个已提交的 docs 文件（SHA256 一致），提交报告草稿后重新启动 |
| 04:45:03 → 04:45:36 | 33 | 环境、空间（下载跳过）、再次核对、复建长度输入（与提交的 meta 一致） |
| 04:45:36 → 04:49:09 | 213 | `max_model_len` 32,768：引擎初始化 193 秒（其中图捕获 62 秒），a、b 两档 |
| 04:49:09 → 04:52:19 | 190 | `max_model_len` 40,960：引擎初始化 74 秒（复用了编译缓存），a、b、c 三档，4 条 episode 约 90 秒 |
| 04:52:19 → 04:52:54 | 35 | HF 重算（加载 4.9 秒） |
| 04:52:54 → 04:53:23 | 约 29 | 输出回传、24 个文件的哈希比对、关机（平台确认 shutdown） |
| **合计** | **约 3,517 ≈ 0.98 GPU·h** | 第 2 次开机共 3,217 秒 |

**磁盘**：
- 开机时 `/` 共 128 GB，剩余 125.2 GB；结束时已用 27 GB（新增：模型 16 GB、vLLM 编译缓存 137 MB、`/tmp` 54 MB）；
- 权重只存一份，在 `/root/autodl-tmp/models/Qwen3-8B`，HF 缓存只有 128 KB；
- 第 1 次开机中断的下载，在模型目录的 `.cache/` 里留下约 0.3 GB 的残片，属于本项目，没有清理；
- 共享的 `/root/autodl-fs` 只写了本次运行的输出（1.8 MB）。

**最后一个阶段"失败"的说明**：结束时统计缓存大小的 `du`，因为 `~/.cache/torch` 不存在而返回 1，这一阶段因此记为 failed。统计到的数字是完整的，前面所有阶段都已完成。

**原始记录**：`artifacts/final_run/phases.jsonl`、`gpu_*.csv`（每 0.5 秒采一次显存）、`*.log`、`run_manifest.yaml`；下载那次运行的记录在 `artifacts/download_run/`。

## 2. 长度与显存

两种设置先后运行，其他条件都相同：同一引擎参数（附录 B），不加 RoPE/YaRN，不强制超限；单并发；每一档测之前清空前缀缓存，所以首个输出时间是冷启动的预填充时间。

| 设置 | 引擎初始化 | KV cache | 满长并发 | 峰值显存（nvidia-smi） |
|---|---|---|---|---|
| 32,768 | 193 秒（图捕获 62 秒） | 175,760 token / 24.14 GiB | 5.36 条 | 46,035 MiB |
| 40,960 | 74 秒（编译缓存命中） | 175,760 token / 24.14 GiB | 4.29 条 | 47,577 MiB |

权重占 15.27 GiB；整卡 97,887 MiB；HF 重算阶段峰值 21,705 MiB（torch 统计的已分配峰值 19.3 GiB）。显存基本由 `gpu_memory_utilization = 0.45` 决定，两种设置的 KV 池一样大，只是能容纳的满长条数不同。

| 档 | 输入 token | 设置 | 输出 token | 结束 | 首个输出（秒） | 总耗时（秒） | logprob | 输出 |
|---|---|---|---|---|---|---|---|---|
| a 约 28.1k（真实初始 prompt） | 28,104 | 32k / 40k | 23 / 23 | 结束符 | 1.93 / 1.93 | 2.23 / 2.23 | 23 个，全部有限 | 一句询问细节的消息 |
| b 约 30.8k（gold 回放，在回合边界截断） | 30,836 | 32k / 40k | 89 / 89 | 结束符 | 2.17 / 2.18 | 3.45 / 3.46 | 89 个，全部有限 | `send_notification` 工具调用 |
| c 约 40.4k（**拼接，仅作工程压测**） | 40,402 | 40k | 29 | 结束符 | 3.24 | 3.66 | 29 个，全部有限 | `find_incident_knowledge_links` 工具调用 |

- 三档的输出都能被 `parse_action` 正确解析：a 档是一条消息，b、c 两档是工具调用。
- **为什么 b 档不是"接近 32k"的 32,256**：在 msp_013 上，30,836 是不超过 32,256 的最后一个自然回合边界，再接下一回合（`retrieve_knowledge_articles` 的观察）就超了。这里没有为了凑长度去拼接。
- **c 档**：在前 9 个 gold 动作之后，接了 4 个真实的只读调用及其观察（`list_incidents` 6,385、`list_problems` 1,375、`find_locations` 764、`find_sla_definitions` 974 token）。内容是真实的，但这不是一条自然的 episode。
- **工程判据**（7A-R3 固定）：① 能启动，KV 至少能放 1 条满长序列；② 不报错、不 OOM，正常结束；③ 每个 token 恰好一个有限的 logprob。两种设置下三条都满足。
- **结论**：40,960 只能记为"工程上可运行"，**仍是候选**。本步没有任何证据说明 32k 以后的生成质量，也不能说明自然长任务的能力。

## 3. 跨设置一致性（诊断，不作门槛）

| 档 | 两种设置是否逐 token 相同 | 第一个分叉位置 | 分叉前的 \|Δlogprob\| | 分叉位置上两个候选的 logprob |
|---|---|---|---|---|
| a | 相同（23/23） | — | 均值 0.0071，中位数约 1e-7，最大 0.072 | — |
| b | 不同（各 89 个） | 第 77 个 | 均值 0.0057，中位数 0，最大 0.102 | 32k：自己选的 −0.612，另一个 −0.862；40k：两者都是 −0.7359（精确平局，取 argmax 时选了另一个） |

两种设置下的 KV 布局、分块预填充的切分方式都可能不同，bf16 的数值误差会让接近平局的位置翻转。分叉之后两边的前缀不同，所以没有再逐位置相减。原样报告，不作解释。

## 4. 真实 episode

任务事先定好：GRPO 可训练题选长度测试用的 msp_013 和 7A 离线测试用的 887f5580；dev 选按 ID 排序的第一道 msp 题和第一道 single 题。每题 1 条，温度 1.0。轨迹全文见 `trajectories.md`，完整的 token、mask 和 logprob 在 `artifacts/final_run/len40k/episodes.json`。

| 用途 · 题 | 策略回合 | 结束 | reward | prompt / 策略 / response token | 最长一个动作 | 非法动作 | 截断 |
|---|---|---|---|---|---|---|---|
| grpo · msp_013 | 12 | user_stop | 0 | 28,166 / 956 / 2,478 | 269 | 0 | 0 |
| grpo · eops_99ba2325_887f5580 | 32 | user_stop | 0 | 28,238 / 1,879 / 7,845 | 123 | 0 | 0 |
| dev · msp_019 | 8 | user_stop | 0 | 28,176 / 475 / 1,270 | 163 | 0 | 0 |
| dev · single_008 | 20 | user_stop | 0 | 28,219 / 893 / 4,315 | 87 | 0 | 0 |

- **采样**：72 次请求全部以 stop 结束，都没有触到 `max_tokens`。首个输出时间中位数 0.072 秒、最大 0.38 秒（前缀缓存开着）；单次请求最长 3.97 秒；解码速度中位数约 70 token/s。最长序列 36,083 token，最长 response 7,845 token，都没有达到预算上限（40,960 / 12,288）。
- **动作格式**：所有工具调用都是 `<tool_call>{json}</tool_call><|im_end|>` 的格式，没有一处文字与工具调用混在一起，也没有出现多个调用或坏 JSON。
- **行为观察**（只记录，不作能力判断）：
  - 4 条 episode 的第一个动作都是**写调用**，而且之前没有任何查询。结果（按轨迹中的观察，2026-11-01 更正）：

    | 题 | 第一个动作 | 参数中的 ID | 结果 |
    |---|---|---|---|
    | msp_013 | `link_knowledge_to_incident` | `INC_001`、`KB_001` | **成功**（这两个 ID 恰好存在于该题的 DB 中，写入了 `IKB_042`） |
    | eops_887f5580 | `update_incident` | CI 写成序列号 `SPR-AGT-01` | 业务错误：`Configuration item with ID 'SPR-AGT-01' not found` |
    | msp_019 | `update_incident` | `INC-000008`（编号当成了 ID） | 业务错误：`Incident with ID 'INC-000008' not found` |
    | single_008 | `update_incident` | `INC0000002`（编号当成了 ID） | 业务错误：`Incident with ID 'INC0000002' not found` |

    原文写的"参数里的 ID 是编造的，写入却执行成功了"不准确：只有 1 条成功，另外 3 条被工具以业务错误拒绝。
  - 之后的写调用有成有败：例如 msp_013 的第一次 `create_incident` 返回业务错误，第二次成功；887f5580 调用了不存在的工具 `close_incident`（`unknown_tool` 错误，按环境规则作为观察返回，不算非法动作）。
  - 只记录现象：不知道目标时，初始策略常常不查询就直接写；写调用用了猜测的 ID 时，结果取决于这个 ID 是否恰好存在。本步不据此作能力判断，不改 Bench2 工具、policy、任务或 reward。
- **没有触发外部 LLM**：用户是脚本化替身，`litellm` 已替换成一调用就报错的替身，没有任何报错记录。

## 5. logprob 对照与模式建议

**策略与概率定义**：
- 策略：Qwen3-8B @ b968826d 的初始权重（没有更新，global step 0），bf16；
- rollout 端：vLLM 0.24.0 的 `processed_logprobs`。温度 1.0、top_p 1、top_k −1 时，就是 log softmax(logits) 中采样 token 的值；
- 训练端：HF transformers（sdpa）对**同一条 token 序列**（同一前缀、同一目标 token）计算 log softmax(logits / 1.0)，用的是 verl 的 `logprobs_from_logits`；
- Δ = log p_train − log p_rollout，**只统计 mask = 1 的位置**；环境回合的 0.0 占位不参与；
- rollout 与重算用的是同一条序列，所以不存在"分叉后对不上位置"的问题。

| 题 | 策略 token | \|Δ\| 均值 | p90 | p99 | 最大 | \|Δ\|>0.1 | 累计差 Σ_rollout → Σ_train（Σ Δ，nat） | token 比率 exp(Δ) 范围 |
|---|---|---|---|---|---|---|---|---|
| msp_013 | 956 | 0.0071 | 0.0010 | 0.214 | 0.616 | 2.4% | −32.82 → −32.27（+0.55） | 0.54–1.72 |
| eops_887f5580 | 1,879 | 0.0047 | 0.0001 | 0.159 | 0.625 | 1.5% | −62.30 → −60.20（+2.10） | 0.58–1.87 |
| msp_019 | 475 | 0.0077 | 0.0038 | 0.224 | 0.748 | 2.3% | −34.28 → −34.25（+0.03） | 0.47–1.33 |
| single_008 | 893 | 0.0050 | 0.0009 | 0.150 | 0.437 | 1.6% | −39.31 → −39.40（−0.08） | 0.65–1.49 |
| **合计** | **4,203** | **0.0056** | 0.0007 | 0.169 | 0.748 | 1.8%（>0.5：0.14%，>1.0：0） | 平均有符号差 +0.0006 | — |

逐回合的累计差见 `summary.json`（`logprobs.episodes[].per_turn`）。

另外：verl 的 bf16 logprob 函数与 fp32 log-softmax 之间最多相差 0.026。也就是说，训练端自己的数值路径也带有这个量级的误差。

**建议：训练用重算模式**（`bypass_mode: false`，仓库默认 `configs/algorithm/grpo_sequence.yaml`），不用 bypass（`grpo_sequence_bypass.yaml`）。理由：
1. 本仓库的损失是 token 级的 PPO 比率，裁剪 0.2。如果用 bypass，`old_log_prob` 直接取 rollout 的值：在第 0 步、策略还没有更新时，就已经有 36/4,203（0.86%）的 token 比率落在 [0.8, 1.2] 之外，这些 token 会被裁剪，误差纯粹来自两个引擎的差异。
2. 序列累计差随长度增长（1,879 个 token 的 episode 达到 +2.1 nat），以后如果改用序列级比率，偏差会被放大。
3. 重算的代价是每批多一次前向。本步 4 条约 3 万 token 的序列，加上 2 条 gold 诊断，共用了约 13 秒（不含加载）。
4. 无论用哪种模式，每步都应记录 rollout 与 `old_log_prob` 的差（例如 \|Δ\| 的 p99、裁剪比例），作为基础设施监控。

**这个建议的限制**：训练端用的是 HF sdpa，不是 verl 的 FSDP actor；verl 默认用 flash_attention_2，但这里没有安装 flash_attn；温度只测了 1.0。第 9 步必须用实际的 actor 配置复测一次，再定下来。

## 6. gold 动作的 teacher-forcing logprob（只作描述）

b、c 两档末尾的目标都是同一个 gold 动作 `retrieve_knowledge_articles`（22 token）：

| 档 | 输入 token | 22 个 token 的 logprob 合计 | 平均 | 主要贡献 |
|---|---|---|---|---|
| b | 30,836 | −39.66 | −1.80 | 第 7、16、20 个 token：−20.6、−6.5、−12.5 |
| c（拼接） | 40,402 | −16.67 | −0.76 | 同样三个位置：−13.6、−1.3、−1.3 |

c 档在拼接部分里看到了 `list_*` 的结果，这让同一个动作的概率上升，主要差别就集中在上面这三个 token。按你的要求，**这里的平均差值不用来单独判断长上下文质量**：两档前文的内容不同，只有一个目标、22 个 token；c 档本身也是拼接的。

## 7. 发现

1. **Bench2 列表结果的顺序依赖哈希种子**：`slice_db_to_orgs`（`enterprise_worlds/domains/itsm/data_model.py`）用 `{rid: rows[rid] for rid in keep[coll]}` 重建切片后的数据库，遍历的是 `set`，所以在 org 切片的任务上，不带过滤条件的 `list_*` 结果顺序（只是顺序，不是内容）会随 `PYTHONHASHSEED` 变化，每个 worker 子进程都可能不同。这会影响训练观察的可复现性，也会让同组 episode 看到不同顺序的观察。本步通过固定 `PYTHONHASHSEED=0` 绕开了，**没有修改环境**。是否修 Bench2（例如按原表顺序重建）、是否把固定哈希种子写进协议，需要你决定。
2. **磁盘与第 1 步记录不符**：`/` 是 128 GB（剩余 125 GB），不是第 1 步记录的 30 GB 系统盘（剩余约 19 GB）。原因没有核实（可能平台调整过）。第 9 步的存储方案应以实测为准，并在开机后先测一次。
3. **旧关机定时器在人工排查期间到期**（2026-11-01 更正）：第 1 次运行失败后，它的 3 分钟定时器（03:56:01 设定）在 03:59:01 执行了关机。原文说"当时第 3 次运行还在跑，检查没有生效"，**这是错的**：第 3 次运行已在 03:58:26 因 Xet 下载报 401 而退出（退出码 1，见其 `.projtool-exit-code` 和 `.projtool-autoshutdown`）。projtool 的定时器到期时会检查有没有 `proj-*` tmux 会话，有就跳过；当时一个都没有，所以按设计关机，并没有误伤正在运行的任务。真实的问题有两点：一是旧定时器在新运行开始后仍然有效，可能打断人工排查；二是我后来"运行开始时建标记、结束时删标记"的做法，在运行被 `cancel_run` 杀掉时会把标记遗留下来，安全网就失效了。改进方案见 `docs/itsm_step8/proposal.md` §9。
4. **运行时读 `docs/`**：用途登记的校验会读 `docs/itsm_stepS0/usage_audit.json`（`itsm_usage.py`），`docs/itsm_step6d/candidate_screening.json` 只被离线的 S0 审计脚本读取，不属于运行时依赖（2026-11-01 核实：在没有这个文件的干净部署目录中，取题和用途校验都能通过）；但按 projtool 的约定，`docs/` 不同步到远端。本步手动上传了这两个已提交的文件（SHA256 一致）。第 9 步之前，应当把运行时依赖的文件移到同步范围内，或改为在远端也能取到，否则云端训练会在第一次取题时失败。（2026-11-01 已修：`configs/itsm/runtime_dependencies.json` 显式列出运行时依赖，`scripts/itsm/runtime_deps.py push` 负责同步并校验，取题前会核对哈希，见 `docs/itsm_step8/proposal.md` §8。）
5. **下载**：经 AutoDL 代理访问 huggingface.co，Xet 的 CAS 返回 HTTP 401，下载直接失败（原文写成"30 秒没有进展"，2026-11-01 更正）；改用普通 HTTP 后只有约 3–4 MB/s。hf-mirror 按 revision 下载约 13–16 MB/s，单连接约 3 MB/s。权重的来源可以换，因为每个文件都要按 HF 的 SHA256 核对。
6. **不查询就直接写**（§4）：4 条的首动作都是写调用，1 条因猜测的 ID 恰好存在而成功，3 条返回业务错误。这是初始策略和环境的现状，不是本步引入的。（2026-11-01 更正，原文为"策略编造 ID 也能写入成功"。）
7. **nvidia-smi 的利用率读数不可靠**：空闲时（显存 3 MiB）也显示 100%，所以报告里没有用利用率，只用显存。

## 8. 与方案的偏差

1. **没有走 verl 的 Ray rollout server**：用单进程适配器复刻了它的引擎构造和 `generate` 语义（附录 B 列出了没有复刻的部分）。verl 完整的 AgentLoopManager/Ray 链路，要到第 9 步才验证。
2. **训练端是 HF sdpa**，不是 verl 的 FSDP actor（见 §5 的限制）。
3. **两个 docs 文件是手动上传的**（§7.4），**`PYTHONHASHSEED=0`**（§7.1）。
4. **权重文件来自 hf-mirror**（固定 revision 的 URL），不是 huggingface.co；全部按 HF 的哈希核对通过。
5. **脚本化用户的设计**（附录 B）由我定，用户没有告诉目标，所以这些 episode 只用于工程验证。
6. **b 档是 30,836 token**，比"接近 32k"短一些（原因见 §2）。

## 9. 未验证项

- verl 完整训练链路（Ray、AgentLoopManager、FSDP actor 的 `compute_log_prob`、权重同步、flash_attn 与 sdpa 的差异）；
- 温度 1.0 以外的采样（例如 airline 用过的 0.8），以及温度不为 1 时 processed logprob 与重算是否一致；
- 并发采样（本步都是单并发）；同组多条采样；训练时与 actor 同卡的显存；
- 32k 以后的生成质量，以及自然长任务的能力；
- 真实用户模拟器（D5）的行为、采样时延、关闭思考后的真实参数（属于另一批 API 验证）；
- 输出触到 `max_action_tokens`（512）时 `parse_action` 的行为：本步没有出现过。

## 附录 A. 固定版本与可追溯性

| 项 | 值 |
|---|---|
| 数据 | split v3（SHA256 `9326167d…5bc0`）、用途登记 `itsm_usage_v1`（`87f15baf…a7`），冻结文件未改动；只用 train 中 GRPO 可训练题和 dev 题，没有读 held-out |
| 模型 | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`，bf16，没有更新权重（策略版本 = 初始权重，global step 0） |
| 权重核对 | 5 个 safetensors 分片按 HF 记录的 SHA256 逐个核对，其余文件按 git blob SHA1 核对；小文件（tokenizer、config 等）还要与本机固定的分词器文件一致；chat template SHA256 `a55ee1b1…74d8`。清单：`configs/itsm/step7b/model_files.json` |
| 分词器与模板 | 模型目录自带的分词器（与本机固定版本逐文件一致），`enable_thinking=False` |
| 代码 | 运行时代码与本地提交逐文件一致（`run_script` 的 code provenance：319 个文件，0 处不同） |
| 环境 | python 3.12.16、vllm 0.24.0、torch 2.11.0+cu130、transformers 5.21.0、flashinfer-python 0.6.12、verl 0.9.0.dev0、huggingface_hub 2.3.0；**未安装 flash_attn**；GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition（96 GB），单卡 |
| 随机性 | vLLM 引擎 seed 42（verl：replica_rank 0 + rollout.seed 42）；`PYTHONHASHSEED=0`（原因见"发现"） |

## 附录 B. 配置

**vLLM 引擎**：参数与 verl `vLLMHttpServer` 的构造方式相同，取 verl rollout 默认值，再按本仓库 `configs/runtime/pro6000_single_gpu.yaml` 覆盖。构造路径也相同：`build_cli_args_from_config` → `AsyncEngineArgs.from_cli_args` → `create_engine_config` → `AsyncLLM.from_vllm_config`。

| 参数 | 值 | 来源 |
|---|---|---|
| dtype | bfloat16 | verl 默认 |
| gpu_memory_utilization | 0.45 | `pro6000_single_gpu.yaml`（训练时与 actor 同卡） |
| max_num_batched_tokens / max_num_seqs | 32768 / 24 | `pro6000_single_gpu.yaml` |
| enable_chunked_prefill / enable_prefix_caching | True / True | verl 默认 |
| logprobs_mode | processed_logprobs | verl 默认 |
| compilation_config | `{"cudagraph_mode": "FULL_AND_PIECEWISE"}` | verl |
| enforce_eager | False | verl 默认 |
| load_format | auto（从磁盘读真实权重） | **与 verl 不同**：verl 用 dummy，再从 actor 同步权重 |
| max_model_len | 32768 或 40960 | 本步测试的两种设置；不加 RoPE/YaRN 设置，不强制超限 |

**没有照搬 verl 的部分**：Ray、`worker_extension_cls`、权重同步、`monkey_patch_model`（vocab 大小补丁）、LoRA、PD 分离、prometheus。

**请求**：与 verl `generate` 相同。
- `max_tokens` 限制在 `[1, max_model_len − prompt 长度]` 之内；
- 请求 logprob 时设 `logprobs = 0`，只返回采样 token 自己的 logprob；
- `repetition_penalty` 1.0，`ignore_eos` False；
- 逐 token 取 `outputs[0].logprobs[i][token_ids[i]].logprob`；
- finish reason 为 stop/length 时，记 `stop_reason = completed`。

适配器：`src/agent_rl/rollout/vllm_token_server.py`。

**采样参数**：
- 长度测试：贪心（温度 0），每次最多生成 512 token，另外返回 top-5 logprob（只用于诊断）；
- 真实 episode：verl rollout 默认值，温度 1.0、top_p 1、top_k −1、`repetition_penalty` 1.0、`logprobs=True`，与 verl agent loop 传给服务端的参数一致。ITSM 的策略采样温度在协议里还没有定，本步只测了 1.0。

**episode**：
- loop：`ItsmAgentLoop`，每个 episode 在 `ItsmEnvWorker` 子进程中运行；预算 v1（max_steps 40、max_action_tokens 512、response_length 12288、prompt_length 28672），文本比较用 fuzzy（D1b）；
- 用户：`PersonaScriptedUser`（`persona_scripted_user_v1`）。只说 persona 自己会透露的信息：姓名、邮箱、角色、`known_info` 里的编号；不说模拟器才有的 `task_description`，也就不告诉策略任务目标；之后依次给 3 条固定的跟进回复，再回复带 STOP 的结束语。因此不要求、也不预期任务成功。
- 外部 LLM：父进程和每个 worker 子进程里，`litellm` 都被替换成一调用就报错的替身（`forbid_external_llm`），API key 环境变量也已清除。

**训练端重算**：
- 模型：HF transformers 加载同一份权重，bf16，`attn_implementation="sdpa"`；
- 计算方式：对与 rollout **完全相同的 token 序列**（prompt + response）做一次前向；在 mask = 1 的位置，取 logits ÷ 温度，用 verl 自己的 `verl.utils.torch_functional.logprobs_from_logits` 计算（没有 flash_attn，所以走它的 torch 分支），另外用 fp32 log-softmax 算一遍作对照；
- 没有运行 verl 的 FSDP actor。verl 默认的注意力实现是 flash_attention_2，但环境里没有安装 flash_attn；它和 sdpa 的差异没有验证。

## 附录 C. 完整命令

本机（CPU）准备与核对：

```bash
# 长度输入（固定 PYTHONHASHSEED=0；与 configs/itsm/step7b/length_inputs_meta.json 逐 token 一致）
python scripts/itsm/build_7b_length_inputs.py --tokenizer-dir ~/.cache/entergym/tokenizers/Qwen3-8B-b968826d \
    --out output/itsm-step7b/local/length_inputs.json --meta-out configs/itsm/step7b/length_inputs_meta.json
# 离线测试（适配器、脚本化用户、LLM 替身、episode 运行器、重算对齐、长度输入）
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY HF_HUB_OFFLINE=1 python -m pytest tests/unit/test_itsm_step7b.py -q
```

云端（`exp-itsm`，经 projtool `run_script` 启动，自动关机安全网 3 分钟）：

```bash
bash scripts/itsm/run_7b.sh; rc=$?; rm -f /root/no_autoshutdown; exit $rc
# 环境变量：BOOT_EPOCH=<本次开机请求的 Unix 时间>  PRIOR_GPU_S=<此前各次开机已用的秒数>
```

`run_7b.sh` 依次执行（每段都记录 UTC 起止时间；每段开始前核对预算）：

```bash
python scripts/itsm/model_files_7b.py check-space --dir /root/autodl-tmp/models/Qwen3-8B      # 剩余空间 ≥ 缺失文件 + 2 GiB
# download：curl 从 https://hf-mirror.com/Qwen/Qwen3-8B/resolve/<revision>/<file> 并行下载、可续传，停滞 180 秒即终止
python scripts/itsm/model_files_7b.py verify --dir /root/autodl-tmp/models/Qwen3-8B
python scripts/itsm/build_7b_length_inputs.py --tokenizer-dir $MODEL --out $OUT/length_inputs.json --check configs/itsm/step7b/length_inputs_meta.json
python scripts/itsm/run_7b_gpu.py length --model $MODEL --max-model-len 32768 --tiers a,b --inputs $OUT/length_inputs.json --out $OUT/len32k
python scripts/itsm/run_7b_gpu.py length --model $MODEL --max-model-len 40960 --tiers a,b,c --episodes --inputs $OUT/length_inputs.json --out $OUT/len40k
python scripts/itsm/run_7b_gpu.py recompute --model $MODEL --trajectories $OUT/len40k/episodes.json --length-inputs $OUT/length_inputs.json --out $OUT/recompute
```

本机汇总：

```bash
python scripts/itsm/analyze_7b.py --run-dir output/exp-itsm/<run>/step7b --out docs/itsm_step7b/summary.json
```

## 附录 D. 离线验证

- 开机前与收尾时各跑一次全量离线回归：`pytest tests/unit tests/remote -q`（不设 API key 环境变量，`HF_HUB_OFFLINE=1`），两次都是 **833 passed**，退出码 0，跑完没有残留子进程。新增的 8 项是 `tests/unit/test_itsm_step7b.py`。
- 两个冻结文件的 SHA256 不变。
- 这些离线测试只验证代码路径（适配器与 verl 语义是否一致、重算时位置是否对齐等），不能证明真实 vLLM 的行为；真实行为以本报告 §2–§5 的 GPU 结果为准。
