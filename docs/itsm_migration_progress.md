# ITSM 迁移进度

规则来源：`docs/itsm_migration_agent_playbook.md`（原路线）；2026-10-31 起以 `docs/playbook2.md`（SFT → GRPO 新路线）为准，历史记录保留原文。“测试已通过”与“用户已批准”分开记录；用户批准一栏只记录用户原话，不由 agent 填写结论。

## 步骤状态

| 步骤 | 状态 | 用户批准记录 |
|---|---|---|
| 0：云端只读审计 | 用户认可 | 2026-10-28 用户原话：“我目前认为可以验收了，你可以额外检查一遍，没问题就执行第一步”。agent 补查第 0 步无问题后进入第 1 步。 |
| 1：独立环境与现有测试基线 | 用户认可；GPU 训练环境于 2026-10-29 补建完成（见“第 1 步补充：GPU 训练环境”），补充部分待验收 | 2026-10-28 用户原话：“未提交文件是什么，你认为是否需要接受？没问题就接受，开始第二步。实例等gpu充足的时候再创建”。 |
| 2：接口与奖励契约 | 用户认可；D1–D6 已定，D5 改为 `deepseek/deepseek-v4-flash` | 2026-10-28 用户原话：“认可，除了D5我认为可以用deepseek-flash。至于风险，到了那一步再说。没问题就第三步”。 |
| 3：离线环境与评分回归 | 用户认可（第 1 次修订后）；D7、D8 已定 | 2026-10-28 用户要求修订（该消息不是验收通过）。2026-10-29 用户原话：“根据本次修订报告，第 3 步验收通过。请记录以下决定：D8：保持 Bench2 现有自由文本比较规则……D7：评测时 DB 文本裁判与 NL 裁判均采用 deepseek/deepseek-v4-pro，温度 0……完成上述文档收尾后，仅执行第 4 步”。决定全文写入协议 §12、§14.2–§14.4。 |
| 4：最小单 episode adapter | 用户认可（含第 1、2 次补修，提交 `223a3ac`） | 2026-10-30 用户原话：“第 4 步及相关补修验收通过，包含提交 223a3ac。记录本条批准，现在仅执行第 5 步：生命周期与隔离。” |
| 5：生命周期与隔离 | 用户认可（含第 1 次补修，提交 `b067d90`） | 2026-10-30 用户原话：“第 5 步及补修 b067d90 验收通过。记录本条批准，现在仅执行第 6 步：数据划分与预算冻结。” |
| 6：数据划分与预算冻结 | **收尾待验收**（划分已选 v1） | 未批准。2026-10-30 用户原话：“第 6 步划分选择 v1：train 29 / dev 13 / held-out 11。暂不最终验收，不进入第 7 步。仅完成以下收尾”（实验定位、checkpoint 规则、无损紧凑序列化对照、40,960 待验证、不裁剪观察）。 |
| 6A：EnterpriseOps-Gym ITSM 扩展可行性审计 | **待验收** | 未批准。2026-10-30 用户原话：“暂不进入第 7 步，也不修改现有 v1 划分。现在仅执行新增的第 6A 步：EnterpriseOps-Gym ITSM 扩展可行性审计。……只提交审计报告和试点清单，等待我验收后再执行第 6B 步。” 第 6 步仍为收尾待验收。 |
| 6B：EnterpriseOps-Gym ITSM 迁移试点 | 用户认可（3 题通过、5 题继续隔离） | 2026-10-30 用户原话：“第 6B 步验收通过。批准将以下三题纳入训练集 v2：itsmbench_eops_46ca2862_b6ef60ca、itsmbench_eops_66392a82_52c8290e、itsmbench_eops_1d219d93_d1ba6c2d。其余五题继续隔离……现在仅执行第 6C 步：数据与预算整合。” |
| 6C：数据与预算整合 | 用户认可；预算选项 A（完整观察，超长时以 response_length 结束）；40,960 仍待验证 | 2026-10-30 先有用户原话：“第6C步验收结果保留，但暂不开始完整第7步。”随后用户原话：“第6C已验收通过，预算选项为A，40,960仍待验证。” |
| 6D-1：EnterpriseOps-Gym 下一批候选筛选与规模方案 | 用户部分批准：第 1–4、6–8 题进入 6D-2 验证；第 5、9、10 题暂缓；D6D-1、D6D-2 均选 ①；门槛不放宽 | 2026-10-30 用户原话：“本轮仅批准第1–4、6–8题进入6D-2迁移验证，不保证最终全部纳入。第5、9、10题暂缓，不改政策、不制定MSP SLA全局规则，不沿用省略user_role的口径，不放宽共享写工具门槛。……提交6D-1报告及本条决定后，仅完成上述7题的6D-2验证，按既定逐题标准提交结果，失败则隔离。不为凑数放宽标准，不自动扩大规模或启动训练。”全文见 `docs/itsm_step6d/README.md` §6。 |
| 6D-2：7 题迁移验证 | 用户认可：第 1、3、6、7 题作为 v3 新增训练题（train 目标 32 → 36）；第 2、4、8 题继续隔离；不授权重新设计第 4 题 | 2026-10-30 用户原话：“第 6D-2 验收通过。一、数据决定 1. 批准第 1、3、6、7 题作为 v3 新增训练题，训练集目标规模由 32 增至 36。2. v1/v2 不改；本次批准不等于 v3 最终验收或冻结。3. 第 2、4、8 题继续隔离。4. 暂不授权重新设计第 4 题，不修改 D8、fuzzy 阈值、policy 或评分。5. 第 5、9、10 题及其他备选题继续暂缓，不扩大训练候选范围。” |
| 6D-3：v3 held-out 候选审读与迁移验证 | 用户认可：1 题通过（`99ba2325_54f35603`，纳入 held-out），7 题隔离 | 2026-10-30 用户原话（同一条消息）：“二、只执行第 6D-3：v3 held-out 候选审读与迁移验证。范围限定为 docs/itsm_step6d/v3_proposal.md 已列出的 8 道候选。两道服务目录题继续优先保留给 held-out，不转入 train。……不通过的候选隔离并记录理由，不另找题替补，不放宽门槛。不运行模型，不用 held-out 调试长度、预算或模型配置。……本步先不生成正式 v3 manifest，不开始第 7 步，不启动 GPU、不调用真实 LLM、不做 SFT 或训练。完成后停止，等待我检验确认。” 2026-10-31 用户原话：“目前已完成6d-3”，未出现“验收通过”，状态按实际仍记为待验收。 2026-10-31 用户原话：“6D-3 验收通过，54f35603 纳入 held-out。” |
| S0：联合数据用途与 SFT 轨迹试点设计（新路线，仅离线） | 用户认可（含 S0-R1）；D-S0-1/2/3/4/5/6 已定，D-S0-8 的 API 上限已定（45 美元），GPU 上限待批 | 未批准。2026-10-31 用户原话：“研究主线改为 SFT → GRPO，执行仍逐步验收。本次只执行 S0：联合数据用途与 SFT 轨迹试点设计，不开始采样或训练。……将之前建议的 6D-3-R1 检查纳入本步，避免另开重复流程。……仅修改设计与必要离线审计，不联网调用 LLM、不启动 GPU、不训练、不自动进入第 7 步或 S1。完成后停止，等待我验收。” 随后：“请阅读code/main/docs/playbook2.md，这是新路线，之前的code/main/docs/itsm_migration_agent_playbook.md是原路线”。 2026-10-31 用户原话：“S0 暂不最终验收，只执行 S0-R1 补修。按本次逐项决定更新提案……1. 新增 b6ef60ca 反例……将该题改为暂停 GRPO、保留 SFT 来源。2. 修正核心要求的判定逻辑：不能用一个结构化反例 DB=0，覆盖另一个已确认的核心错误反例 DB=1……3. ……原有 5 题登记为待补 R1，第 9 步前完成。78a5 接纳为 SFT-only 来源；54f35603 保留给 held-out。4. 修正 Teacher/裁判偏差表述，补充实际模型版本核对方案。API 上限须包含重试及所有收费请求，并在发起请求前预留费用……只修订 S0、运行受影响的离线验证并提交验收。本次不生成正式 v3，不进入第 7 步或 S1，不调用 LLM、不启动 GPU、不训练。完成后停止。” 2026-10-31 用户原话：“S0-R1 验收通过。明确确认本次回复中的各项决定：五道派生题暂停 GRPO；78a5 保留为 SFT-only 来源”。 |
| v3 manifest 与用途登记 `itsm_usage_v1` 冻结 | **用户认可（已冻结）**：提交 `70e6a3b451149bb24ae0eba18fc544705d4c323a`；`itsm_split_v3.json` SHA256 `9326167dfee696a809f685f178284e23e6e8f3402559d4b47650c4fcf85b5bc0`；`itsm_usage_v1.json` SHA256 `87f15baf96a5dbeec6a8c07ca0952edafd86667d18dfa01d63f066d35934fba7` | 未批准。2026-10-31 用户原话：“S0-R1 验收通过。明确确认本次回复中的各项决定：五道派生题暂停 GRPO；78a5 保留为 SFT-only 来源；6D-3 验收通过，54f35603 纳入 held-out。现在只执行：生成 v3 manifest 与用途登记，提交冻结验收。……4. Teacher 候选改为 GPT-6.1 Sol，供应商为 Micu：https://www.micuapi.ai 接口兼容性、实际模型 ID、参数、版本和价格留到第 7 步验证；不静默替换模型，不照搬 DeepSeek 参数。5. S1 总费用上限为 45 美元。换用 GPT 后重新核算单请求、单轨迹和总预留；本步只更新预算说明，不调用模型。……不开始第 7 步或 S1，不调用 LLM、不启动 GPU、不训练。完成后停止，等待我验收。” 2026-10-31 用户原话：“v3 manifest 与 itsm_usage_v1 冻结验收通过。冻结依据为提交 70e6a3b 的文件内容及完整 SHA256。” |
| 7A：离线 loop 验证与 API 兼容探测 | **用户认可（2026-11-01，随 7A-R3 验收收尾）**：离线部分、付费探测（6 次、账本记账 0.028066 美元）、7A-R1/R2/R3 补修 | 未批准。2026-10-31 用户原话：“v3 manifest 与 itsm_usage_v1 冻结验收通过。冻结依据为提交 70e6a3b 的文件内容及完整 SHA256。现在只执行第 7A 步：离线 loop 验证与 API 兼容探测。1. 将批准原话、提交和两个文件的完整哈希写入进度。不为修改 candidate/frozen 状态而重写冻结文件。后续用途资格变化另建版本；轨迹审核另存文件并引用冻结哈希。2. 用确定性采样替身验证 ItsmAgentLoop 的输入渲染、单动作转换、增量观察、token/span/mask 和 logprob 字段契约。mock 验证不能当作真实采样 logprob 验收。3. 在云端核对 Micu 的 GPT-6.1 Sol：端点、Responses 与工具调用兼容性、实际模型 ID、参数、输出/推理 token 上限、usage 和价格。同时核对模拟器与裁判的实际模型标识。不照搬 DeepSeek 参数，不静默换模型，不输出密钥。4. 先核实计费并完成请求前预留，再做最小付费探测：最多 6 次收费请求，总费用不超过 1 美元，包含所有重试。此预算属于第 7A 步，与 S1 的 45 美元分开记账。无法可靠约束费用、接口不兼容或版本无法确认时停止并报告。5. 不运行 held-out，不生成 SFT 轨迹，不更新权重。不启动 GPU；40,960 继续保留为待验证候选。提交离线验证、实际请求结果、费用及第 7B 步方案后停止，等待我验收。不得自动进入 7B、第 8 步或 S1。” |
| 7A-R1：7A 离线补修 | **主要补修用户认可**（2026-10-31），7A 暂不整体验收：worker 接入与取消/超时测试、非有限 logprob、探测通过条件、账本单一所有者、D5 关闭思考、偏差与费用记录、README 与 7B 方案均已完成；没有调用 LLM，没有启动 GPU，没有新增付费请求 | 未批准。2026-10-31 用户原话：“第 7A 步暂不整体验收，只执行 7A-R1 离线补修。<br>1. ItsmAgentLoop 接入已验收的进程 worker，补取消/超时集成测试：慢工具执行中取消后，确认子进程退出、工具不再完成延迟写入、无有效 reward、无残留进程。不能只取消等待线程。<br>2. 采样契约补充非有限 logprob 检查。NaN/Inf、缺失或数量不匹配均按基础设施故障处理。<br>3. 修复 API 探测的通过条件：非 200、缺失结果、错误工具、参数不符、必需 checks=false 都必须停止并非零退出。用模拟 HTTP 测试，不新增付费请求。<br>4. 明确账本的单一所有者和并发边界。同一路径被第二个写入者使用时，要么拒绝，要么正确锁定并重读；补“两实例各预留 0.75、总上限 1”的回归测试。<br>5. D5 明确关闭思考，保持 deepseek-flash；本步只改配置和离线验证，真实参数复核留到另批的 API 验证。<br>6. 接受 Micu 未知额外输入作为 S1 试点的记录偏差；不把观察到的 token 开销当固定上界。用户提供的价格可用于估算，账本费用与实际扣费分开写。输出上限标为尚未触顶验证。<br>7. 重写 README 开头的最终状态，消除“0 次请求、未开 GPU”与后续结果的冲突；历史过程保留并注明时间。修订 7B 的实际长度测试和存储方案，不自动扩盘或新租实例。<br>重跑受影响验证，提交 7A-R1 后停止。本次不调用 LLM、不启动 GPU，不突破已用尽的 6 次探测额度，不进入 7B 或 S1。” |
| 7A-R2：worker 启动期间的取消清理 | **启动取消修复用户认可**（2026-10-31），7A 整体验收仍待完成：先取得句柄、再启动的有界生命周期，启动中取消的回归测试，7B 长度判据与资源计时修订；没有调用 LLM，没有启动 GPU，没有新增付费请求 | 未批准。2026-10-31 用户原话：“7A-R1 的主要补修认可，暂不整体验收。<br>现在只执行 7A-R2：修复 worker 启动期间的取消清理。<br><br>1. 将 worker 创建、取得句柄和清理纳入完整生命周期管理。<br>   启动过程中被取消后，后台构造若仍会返回 worker，<br>   必须取得并回收它，不能遗失结果。<br>   启动等待和清理均须有界；仅加 shield 或取消 Future 不算完成。<br><br>2. 新增独立回归：<br>   worker 子进程已经创建、但构造尚未返回时取消 rollout；<br>   确认没有 AgentLoopOutput/reward，最终子进程已回收，<br>   无残留进程，重复 close 安全。<br>   原有慢工具取消、超时及完整 episode 测试继续通过。<br><br>3. 按本次意见更新 7B 的长度判据与资源计时说明。<br>   40,960 仍是候选，不提前标为长上下文质量已验证。<br><br>重跑受影响验证，提交 7A-R2 后停止。<br>不新增付费请求、不调用 LLM、不启动 GPU，<br>不进入 7B 或 S1。” |
| 7A-R3：清理期间取消与回收失败的传播 | **用户认可**（2026-11-01）：清理期间第一次取消会在清理完成后传播；回收失败或清理超时报 `not_reclaimed` 基础设施错误；传播规则已写明；7B 跨设置一致性降为诊断项；没有调用 LLM，没有启动 GPU，没有新增付费请求 | 未批准。2026-10-31 用户原话：“7A-R2 的启动取消修复认可；7A 整体验收仍待完成。<br>现在只执行 7A-R3，范围限定如下：<br><br>1. 修复首次取消发生在正常清理期间的路径：<br>   清理继续完成，但取消必须随后往上传播，<br>   不得正常返回 AgentLoopOutput/reward。<br><br>2. 修复回收失败或清理超时的路径：<br>   必须报告明确的基础设施错误及 not_reclaimed 信息，<br>   不得只记日志后返回有效样本，也不能用 reward=0 替代。<br>   明确取消、原始异常与清理失败的传播规则，保留诊断信息。<br><br>3. 新增针对性回归：<br>   - 正常 episode 结束，进入清理后首次取消；<br>   - 故障注入使回收结果为 False；<br>   - 清理超时。<br>   确认这些路径都没有有效输出/reward。<br>   保留原有启动取消、二次取消、慢工具取消及正常完成测试。<br><br>4. 修改 7B 判据：<br>   跨配置贪心输出一致性仅作诊断，不作为通过硬门槛；<br>   记录分歧及概率误差，不用 b/c 平均 gold logprob 差值<br>   单独判断长上下文质量。<br><br>只修上述分支和文档，重跑受影响回归后提交验收。<br>不重做启动方案，不调用 LLM、不启动 GPU、<br>不新增付费请求，不进入 7B 或 S1。完成后停止。” |
| 7B：真实 GPU 推理、长度/显存与采样 logprob | **用户认可**（2026-11-01，工程验证）：约 0.98 GPU·h；32,768/40,960 三档长度全部满足工程判据（40,960 仍为候选，c 档为拼接压测）；4 条真实 episode 无非法动作、无截断；4,203 个策略 token 的 logprob 都有限，训练端重算 \|Δ\| 中位数 0、p99 0.169、最大 0.748，建议重算模式；已关机 | 2026-11-01 用户原话：“7A-R3 验收通过，第 7A 步收尾。<br>现在仅执行第 7B 步：真实 GPU 推理、长度/显存和采样 logprob 验证。<br><br>1. 固定版本<br>   使用冻结的 split v3、usage v1 和 Qwen3-8B 固定 revision。<br>   保持权重、dtype、tokenizer、模板及其他条件可追溯。<br>   不更新权重，不修改冻结数据。<br><br>2. 资源边界<br>   仅使用 exp-itsm 现有磁盘、单卡，累计上限 3 GPU·h。<br>   从开机请求到确认关机全部计时，包含下载、编译、空闲和重试。<br>   先核实实际空间与文件大小，避免重复缓存权重。<br>   空间不足或预算不足就停止报告，不自动扩盘、新租实例或删除其他项目文件。<br>   结束、失败或停止时都确认关机。<br><br>3. 长度与显存<br>   按已批准方案测试约 28k、接近 32k、约 40.4k 三档。<br>   两种 max_model_len 顺序运行，记录实际输入/输出长度、<br>   KV 容量、峰值显存、耗时和终止原因。<br>   不擅自增加 RoPE/YaRN 或强制超限配置。<br><br>4. 长度结论<br>   拼接的 40.4k 输入仅作工程压测，单独标注。<br>   40,960 即使可运行，也继续保留为候选，<br>   不称为自然长任务能力或长上下文质量已验证。<br>   跨配置输出一致性只作诊断，不作为硬门槛。<br><br>5. 真实 episode<br>   GRPO 可训练题、dev 各最多 2 道，每题 1 条。<br>   用户使用脚本化替身，禁止触发外部 LLM 调用。<br>   不运行 held-out，不生成 SFT/Teacher 轨迹。<br>   记录实际输出、非法动作和截断，不要求任务必须成功。<br><br>6. logprob 对照<br>   核对每个采样 token 都有一个有限 logprob，记录策略版本及概率定义。<br>   与训练引擎重算时，必须使用同一权重、前缀和目标 token；<br>   仅对 mask=1 的位置统计误差，环境零占位不参与。<br>   自由生成出现分歧后，不直接把不同 token/前缀的概率逐位置相减。<br>   报告误差分布和序列累计差，并提出 bypass/重算模式建议。<br><br>提交完整命令、配置、轨迹、资源记录、失败和未验证项后停止。<br>不进入第 8 步或 S1，不开展新的付费 API 验证。” |
| 7B 收尾与第 8 步提案（离线） | **用户认可**（2026-11-01）：行为结论已更正（4 条首动作都是写调用，只有 msp_013 成功）；`PYTHONHASHSEED=0` 写入协议 §13.1，在启动入口 Python 启动之前设置，worker 继承情况已用独立进程验证；运行时依赖清单与 push/preflight 完成，干净部署目录中验证通过；5 条遗留 running 记录已核实，因为没有不开机的安全接口而未处理，给出了方案；第 8 步提案已提交；没有调用 LLM，没有启动 GPU | 2026-11-01 用户原话：“7B 工程验证验收通过。现在仅做离线收尾与第8步执行提案。<br><br>1. 更正 README、轨迹说明及进度中的行为结论：<br>   4条首动作均为写调用，只有msp_013成功，另外3条是业务错误。<br>   不改Bench2工具、policy、任务、reward或冻结文件。<br><br>2. 将PYTHONHASHSEED=0写入协议及相关启动入口，在Python启动前设置。<br>   用新启动的独立进程验证worker实际继承，并检查同题列表观察顺序一致。<br>   不因此宣称模型输出完全确定。<br><br>3. 修复部署依赖：<br>   建立运行依赖的显式同步清单，自动同步并校验当前必需的审计JSON，<br>   保持原路径及冻结哈希，不通过删除校验或手动补文件绕过问题。<br>   在干净的部署目录模拟远端文件集合，验证取题和用途校验能成功；<br>   缺失或篡改依赖时必须在采样前失败。<br><br>4. 核实本次5条projtool遗留running记录，用受支持接口收尾。<br>   若无安全接口，报告处理方案，不直接修改内部状态文件。<br>   提出防止旧关机定时器误伤新运行的方案，保留有效的超时关机保护。<br>   不开机验证。<br><br>5. 提交第8步提案：<br>   分列E0-base的dev诊断与GRPO train组内方差诊断；<br>   写明任务范围、每题采样数、G、温度、长度候选及截断规则、<br>   用户模拟器和裁判最小复核、付费请求与金额上限、GPU时长上限。<br>   明确本次单进程适配器与正式verl链路的边界。<br>   不使用held-out，不把7B的脚本化用户结果作为能力基线。<br><br>提交必要回归结果和提案后停止，等我验收。<br>本轮不调用LLM、不启动GPU、不生成SFT轨迹、不进入第8步执行或S1。” |
| 第 8 步离线实现 | **暂不验收**（2026-11-01，先做 R1）：并发运行器、worker 内辅助评测、统一账本（组级预留和单一管理者）、DeepSeek 直连客户端（每次请求预留一次、不经过 SDK 重试）、D5 `max_tokens=1024`、启动和汇总脚本、32k 反事实分析都已完成；proposal 已按确认的参数修订；没有调用 LLM，没有开机，预算不构成授权 | 2026-11-01 用户原话：“本轮离线收尾验收通过。现在仅执行第8步的离线实现，不调用LLM、不开机、不进入S1。<br><br>按本次确认的规模和参数更新proposal：<br>A=dev 13×4，B=GRPO可训练26×G4；策略温度1.0；<br>本次诊断上下文40960仍为候选；D5显式max_tokens=1024；<br>后续复核预算≤20次且≤1美元，正式诊断≤12美元，GPU≤2 GPU·h。<br>这些预算当前不构成执行付费请求或开机的授权。<br><br>完成并发运行器、worker内辅助评测、统一账本、启动与汇总脚本。<br><br>必须落实：<br>1. 32k分析标为沿40960轨迹的反事实预算分析，不冒充实际运行结果。<br>2. 整组开始前预留，逐请求检查；所有可能收费请求及重试计入。<br>   不够就不开新组；不承诺预算内一定完成全部156条。<br>3. 由单一管理者统一处理并发worker的预留和结算。<br>   测试并发抢额度、取消、超时、usage缺失、重启恢复及预算耗尽。<br>   检查SDK内部自动重试，不能有账本之外的收费请求。<br>4. fuzzy结果与LLM DB/NL结果分别保存。<br>   辅助裁判故障只重评原轨迹和原最终DB，不重跑策略；<br>   有效负判不重试，未评估不记为0。<br>5. D5返回finish_reason=length时按user_sim基础设施故障处理；<br>   不使用被截断的回复。保留其费用。<br>6. 从低并发开始，按预先写定的工程规则升至最多8；<br>   OOM、取消和任务失败均须回收worker并保留诊断。<br>7. 不修改Bench2、policy、reward、冻结数据或用途范围。<br><br>提交必要离线回归结果、修订后的费用预留计算和完整命令后停止。<br>我验收离线实现后，再单独批准小额API复核；<br>API复核验收后，再决定正式GPU诊断。” |
| 第 8 步离线实现 R1：审查修复 | **前四项用户认可**（2026-11-01）；费用方案已定为①：复核必需检查失败即停止并非零退出；辅助评测失败不再触发重采样（记为 not_evaluated）；`_unqueued` 重复递减已修；模型不符、实际费用超过预留会 halt 统一管理者；完整组硬上限已计算（12 美元连一个完整组都放不下，方案待决定）；没有调用 LLM，没有开机 | 2026-11-01 用户原话：“第8步离线实现暂不验收。仅修复本次审查指出的问题，不调用LLM、<br>不开机、不发付费请求、不进入S1。<br><br>1. API复核：必需检查失败立即停止，保存失败原因并非零退出。<br>   覆盖模型不符、思考未关闭、协议错误及固定案例判定错误；<br>   重复回复不一致仍只作诊断。<br><br>2. 分离业务结果与辅助评测：<br>   fuzzy完成后保留轨迹和结果；辅助评分预算拒绝、超时、<br>   worker崩溃或评测异常不得触发策略/模拟器重采样。<br>   未评估不记0；清理失败仍遵守原有基础设施故障规则。<br><br>3. 修复排队取消时_unqueued重复递减。<br>   测试组内失败后取消排队成员，下一组仍能继续；<br>   同时检查计数、账本和worker回收。<br><br>4. 模型标识不符、实际费用超过预留设为不可重试故障。<br>   统一管理者停止新预留、新请求和新组，保留在途费用记录。<br>   测试并发情况下故障发生后的请求停止行为。<br><br>5. 重新计算完整组的费用硬上限，包含允许的重试。<br>   不能以估算12次调用替代硬上限。<br>   如果12美元不足，提交具体计算和方案，不自动加预算或改任务规则。<br><br>先用新增测试复现修补前的问题，再修复并运行相关回归。<br>提交结果后停止，等我验收；小额API复核尚未获批。” |
| 第 8 步小额 API 复核 | **已获批，未执行**：云端唯一的实例 exp-itsm 处于关机状态，AutoDL API 不支持无卡模式开机，projtool 只能以 GPU 模式开机，所以按第 6 条停下报告，没有开机，没有发出请求。复核报告字段的准备、汇总中顶层 not_evaluated 漏计的修复已完成（离线） | 2026-11-01 用户原话：“第8步 R1 前四项修复验收通过。费用方案明确选择①：<br>维持正式诊断12美元上限，采用组级计划预留＋逐请求预留，<br>接受预算耗尽时存在未完成组。不加请求大小限制，不改重试次数。<br><br>现在仅批准小额 API 复核：<br>1. 使用已提交的固定输入和重复次数，计划13次请求；<br>   本轮累计最多20次可能收费请求、最多1美元，先到即止。<br>   不为用满20次追加案例，不自动重试，不换账本绕过上限。<br>2. 仅复核D5模拟器和D7两种裁判，不调用Teacher或Micu。<br>3. D5保持温度0、关闭思考、max_tokens=1024；<br>   D7保持已固定的模型、提示词及参数。<br>4. 必需检查失败立即停止，保存原因，非零退出；<br>   重复回复不一致只作诊断。<br>5. 保存实际HTTP请求数、返回模型、finish_reason、usage、<br>   思考相关字段、逐项检查、账本预留及结算。<br>   输出很短不能证明1024触顶行为已验证，未验证项如实列出。<br>6. 使用云端可用的CPU环境，不启动GPU。<br>   若必须启动GPU才能执行，停下报告，不自行开机。<br>7. 复核完成后提交报告并停止，等我验收。<br>   不运行A/B正式诊断，不进入S1，不训练。<br><br>将方案①决定落档。辅助评测顶层not_evaluated在汇总中的漏计，<br>须在正式GPU诊断前修好并验证。” |
| 7 及以后（新路线顺序 S0 → 7 → 8 → S1 → … → S5 → 9 → … → 15） | 未开始 | — |

## 运行环境

工作区：`/home/sar/work/EnterGym/code/main`（WSL2，用户指定的工作区；训练在 AutoDL 上运行）。

| 项目 | 值 |
|---|---|
| 代码版本 | `main` @ `db9ad11`；`tau2-bench` @ `1901a30`；`verl` @ `0a30f52`（均为固定提交） |
| 工作区改动 | 第 1 步时为 `.gitignore`（projtool 脚手架段，非业务改动）和本文件；之后各步改动均已提交，见各步证据 |
| Bench2 来源 | 仓库内目录，导入自 Enterprise-Worlds `37175af` |
| Bench2 数据 sha256 前 16 位 | tasks.json `e82c54a25edf334e`；msp_db.json `96f1c9ae06c0fbbf`；single_tenant_db.json `660e23dc68305af2`；policy.md `6d029256b224722e`；fk_spec.json `54fa139096687e1a` |
| 任务构成 | 53 题，ID 不重复；31 个 `provider_client`（msp）+ 22 个 `single_tenant`；含 `itsmbench_msp_021`、`itsmbench_single_017` |
| 离线环境 | `~/.venvs/entergym-itsm-offline`，uv 管理的 CPython 3.12.14；系统 Python 3.14 未改动 |
| 关键版本 | torch 2.14.1+cpu、ray 2.59.0、transformers 5.21.0、verl 0.9.0.dev、tensordict 0.10.0、pydantic 2.14.1、gymnasium 1.4.0、enterprise-worlds 0.1.0 |
| 未安装（本机离线环境） | vLLM、CUDA 版 torch；这两项已装在 GPU 训练环境中，见下文 |

## 第 1 步证据

产物目录：`/home/sar/work/EnterGym/output/itsm-step1/`（不入 git）。

| 产物 | 内容 |
|---|---|
| `requirements-freeze.txt` | 154 个包的实际版本；`uv pip check` 显示无冲突 |
| `imports.txt` | 核心模块导入结果 |
| `pytest_airlinerl.log` | `pytest tests/unit tests/remote -q -rs`：**113 passed, 0 skipped**，退出码 0 |
| `pytest_bench2.log` | `cd Bench2 && pytest tests -q -rs`：**242 passed, 39 skipped**，退出码 0；39 个 skip 全部是 `test_itsm_conformance.py`，原因是 reference MCP oracle（Docker）不可达 |

安装方式：沿用 `scripts/setup/install_runtime.sh` 的依赖列表。与原脚本的差异如下：

- 没有 conda，改用 `uv venv --python 3.12`。
- torch 取自 CPU 源，并且没有安装 vLLM。
- 额外安装了 `-e ./Bench2[gym,dev]`。
- 额外安装了 `cachetools`：verl 以 `--no-deps` 安装，`cachetools` 原本由 vLLM 间接带入；缺少它时 `tests/remote/test_tau_agent_loop_reliability.py` 会被 skip。

仓库脚本未修改。

测试时取消了 `OPENAI_API_KEY`/`DEEPSEEK_API_KEY`，本机也没有 `.env`，因此测试中没有调用付费模型。

## 既有失败与跳过清单

- AirLineRL：无失败、无跳过。
- Bench2：conformance 共 39 项，需要 Docker 版参考 oracle，本环境未执行。
- 告警：tau2 voice 模块使用了已弃用的 `audioop`，不影响测试结果。

## 第 1 步补充：GPU 训练环境（2026-10-29）

用户指令原话：“现在autodl中显卡充裕，先完成之前暂时搁置的任务”。这是第 1 步中延后的部分，不是新步骤的批准。

- **实例：** `exp-itsm`，ssh 别名 `autodl-exp-itsm-fc77b761`，`instance_uuid` 为 `pro-7933a95ce90d`。1 张 RTX PRO 6000 Blackwell Server Edition，97887 MiB，驱动 595.71.05（CUDA 13.2），计算能力 sm_120。
- **镜像：** `base-image-l2t43iu6uk`，Ubuntu 20.04，glibc 2.31。
  - projtool 默认镜像 `base-image-12be412037` 是 Ubuntu 18.04、glibc 2.27，不满足 torch/vLLM wheel 的 manylinux_2_28 要求。
  - 用默认镜像开出的第一台实例 `pro-7933394d0ec8` 上面没有任何东西，已 `release`。
  - 为了让 `run_script` 能运行，在实例上用 apt 安装了 `tmux`。
- **环境：** `/root/autodl-tmp/conda-envs/itsm-train`，conda 创建的 Python 3.12，占 9.9 GB。
  - 安装方式：原 `scripts/setup/install_runtime.sh`，只通过 `ENV_PREFIX` 改了路径；之后额外安装 `-e ./Bench2[gym,dev]`。
  - 包索引：阿里云 PyPI 镜像。仓库脚本没有修改。
- **关键版本：** torch 2.11.0+cu130、vllm 0.24.0、flashinfer-python 0.6.12、verl 0.9.0.dev、ray 2.59.0、transformers 5.21.0、tensordict 0.10.0、numpy 2.3.5、litellm 1.82.6、pydantic 2.14.1、gymnasium 1.4.0。共 262 个包，`uv pip check` 显示全部兼容。
- **运行记录：** run_id `exp-itsm/2026-10-30T00:40:42.739244Z`。
  - 代码对照：251 个文件与本地 `2e0e57a` 一致。
  - 本地镜像目录：`output/exp-itsm/run-20261030-004042-1c7a83/`，内含 `train.log`、`versions.txt`、`requirements-freeze.txt`、`pytest_airlinerl.log`、`pytest_bench2.log`。
  - 更早的一次运行 `…00:40:18…` 因脚本前置探测中 `| head` 与 pipefail 冲突，以 SIGPIPE（退出码 141）退出，已标记为 failed。
- **验证结果：**
  - `install_runtime.sh` 自带的导入检查和 `assert torch.cuda.is_available()` 通过。
  - bf16 GPU 矩阵乘法正常。
  - AirLineRL `tests/unit tests/remote`：**458 passed，0 skipped**，退出码 0。
  - Bench2：242 passed，39 skipped（Docker oracle，与本机相同），退出码 0。
  - 没有启动模型或训练，也没有调用付费 LLM。
- **与原脚本或预期的差异：**
  - 未安装 `flash_attn`，原脚本本来也不安装。verl 训练是否需要它，留到第 9 步核实。

## 阻塞与资源约束

1. **系统盘只有 30 GB，没有独立数据盘。** 安装完后已用 12 GB，剩余 19 GB。Qwen3-8B 权重约 16 GB，加上 checkpoint，在第 7 步之前需要扩盘或另找存储位置。`instance_up` 不提供扩盘参数。
2. **共享 `/root/autodl-fs` 只剩约 4.5 GB**（已用 98%，其中 `runs/` 占 156 GB，属于其他项目）。`[paths].outputs_remote` 就在这里，后续保存 checkpoint 时空间不够。其他项目的文件不由本项目处理。
3. ~~GitHub 推送凭据未配置~~（2026-10-28 两次推送失败；已解决，`origin/main` 已包含至 `3a63136` 的提交）。

## 第 2 步证据

- 产物：`docs/itsm_protocol.md`。只有协议设计，没有实现 adapter，也没有修改代码。
- 依据：静态阅读 Bench2 的 `gym_agent`、`orchestrator`、`environment`、`toolkit`、`evaluator*`、`text_match_strategy`、`clock`、`run`、`user_simulator`，以及 AirLineRL 的 `tau_env`、`tau_agent_loop`、`schemas`、`process_reward`、`credit_encoding`、`verl_algorithms`。
- 实测：对 `itsmbench_single_017` 做离线、只读的真实工具探测，包括正常调用、`ItsmError`、未知工具、错误参数名四种情况，没有调用 LLM。结果记在协议 §11.0。
- 未验证：协议中所有 "须验证" 项，分别留给第 3、4、5、6、7、9 步。

## 第 3 步证据（第 1 次修订后）

- 改动（已提交于 `2e0e57a`，第 3 步已验收）：
  - `src/agent_rl/envs/itsm_scoring.py`：任务时钟、严格 gold 回放、DB 评分，以及 `llm` 裁判原始响应的协议校验。
  - `tests/unit/test_itsm_scoring.py`：345 项测试。
  - `docs/itsm_protocol.md`：§8 裁判判定边界、§12 重试上限、§14.1 模型角色，以及待定项 D7、D8。
  - Bench2 源码、数据和 airline 代码均未修改。
- 产物目录：`/home/sar/work/EnterGym/output/itsm-step3/`（不入 git）。
  - `pytest_itsm_scoring_r1.log`：345 passed，退出码 0。
  - `pytest_airlinerl_full_r1.log`：`tests/unit tests/remote` 共 458 passed（原 113 项 + 新增 345 项），退出码 0。
  - 第 1 次提交时的日志 `pytest_itsm_scoring.log` 和 `pytest_airlinerl_full.log` 保留作对照。
  - Bench2 自带测试没有重跑：本次修订未改动 Bench2，第 1 次提交时为 242 passed、39 skipped。
- 测试性质：
  - 真实工具离线回放：53 题 gold、时钟、MSP-021 和 Single-017 的状态核验及正反例、DB 比对路径与 Bench2 `compare_dbs` 的对照。
  - mock：裁判响应解析，以及 `llm` 路径端到端测试，都替换了 `generate`，没有联网。
  - 未调用任何真实 LLM。
- 关键结论：
  - 53 题 gold 回放在任务时钟下全部 DB=1（exact 与 fuzzy 均如此），重复回放结果一致。
  - 34 题在错误时钟下 DB=0；另外 19 题的 `current_time` 本身就等于 `DEFAULT_NOW`。
  - 原方案"错写通知状态"对应真实字段 `notification.status`，取值为 queued、sent、delivered、opened、failed；gold 未传该参数，工具默认写入 `queued`。
  - “可选字段为空”拆成两项（随第 3 步验收后的文档收尾）：
    1. gold 状态验证 `test_single_017_gold_state_leaves_optional_problem_fields_empty`：gold 回放后 problem 的 7 个可选字段为空。只说明 gold 本身。
    2. 评分约束范围：`test_single_017_scoring_scope_constrains_structured_optional_field`（5 个结构字段被设值时 DB=0）；`test_single_017_scoring_scope_excludes_unchanged_freetext_field`（2 个自由文本字段被设值时 exact/fuzzy/llm 下 DB 仍为 1，`llm` 下这两个字段不送判）。
    - 结论：DB=1 **不能**保证所有可选字段为空。规则适用于所有 gold 未改变的自由文本字段（D8，协议 §14.4），本轮不加严格字段检查。

### 第 3 步验收后的文档收尾（2026-10-29）

- 协议 §12：重试口径为首次之外额外重试 3 次（总计最多 4 次）；评分级重试使用同一条已保存轨迹和同一最终 DB；有效负判不重试；整组丢弃需记录原因和数量。只定协议，未实现调度。
- 协议 §14.2：Bench2 来源 commit、树哈希及 12 个相关源码的 sha256；测试 `test_bench2_sources_match_pinned_checksums` 固定这些值。
- 协议 §14.3：D7 裁判参数（模型、温度 0、思考模式关闭、`max_tokens=4096`、两段提示词 sha256）；测试 `test_judge_prompts_match_pinned_versions` 固定提示词。API 连通性未验证。
- 协议 §14.4：D8 规则说明。
- 回归：`output/itsm-step3/pytest_itsm_scoring_r2.log`，349 passed，退出码 0。

## 第 4 步证据

- 改动：
  - 新增 `src/agent_rl/envs/itsm_env.py`：`ItsmEnv`（一个实例只跑一个 episode，单线程同步驱动，不起后台线程）、`ItsmAction` 与 `to_itsm_action`（模型输出到动作的转换）、`ItsmEnvConfig`、`ItsmToolCallRecord`、`ItsmRewardInfo`、`export_trace()`。
  - 新增 `tests/unit/test_itsm_env.py`：38 项。
  - Bench2、airline 代码和评分模块均未修改；没有接入训练器。
- 产物目录：`/home/sar/work/EnterGym/output/itsm-step4/`（不入 git）。
  - `pytest_itsm_env.log`：38 passed，退出码 0。
  - `pytest_airlinerl_full.log`：`tests/unit tests/remote` 共 500 passed（113 + 349 + 38），退出码 0。
- 测试性质：
  - 真实 Bench2 工具、种子、任务和 DB 评分。
  - 策略为脚本代理，用户为确定性替身 `ScriptedUser`（固定回复列表）。
  - 一项测试用 monkeypatch 让工具抛出非 `ItsmError` 异常。
  - 未调用任何 LLM；默认的 Bench2 `UserSimulator` 只检查了构造参数，没有发出调用。
- 覆盖的协议条目：
  - 动作与 schema（§2、§3）：文本 / 单个工具调用 / `done` 三种动作；拒绝"文本加工具调用"、并行调用、带参数的 `done`、含 `###STOP###` 的文本和空输出。公开工具为 Bench2 工具加 `done`。
  - 公开信息（§3）：只有 `policy` 和 `tools`，并且在 53 题之间完全相同（sha256 只有一种取值），因此不携带任何任务专属信息。任务描述、模拟器指引、NL 断言和 task_id 都没有出现。
    - 注意：`known_info` 中的通用值（例如枚举值 `agent`、组名、示例单号）会在 policy 或工具 schema 中以子串形式出现，这是碰撞，不是泄漏；任务不变性才是判据。
  - 增量观察（§4）：reset 返回问候语和用户首句；工具调用只返回 `tool[<call_id>] ok|error(<class>): …` 一行；消息动作只返回 `user: …`；结束时返回空串。各段内容不重复出现。
  - 调用 ID（§5）：最初为 `{episode_id}/c{seq:03d}`，第 1 次补修改为与任务无关的 `call_{seq:03d}`（见下文补修证据）；每个调用恰好对应一条 ToolMessage；ID 在 episode 内唯一，且与记录一致；模型自带的 ID 存为 `model_call_id`。
  - 错误分类（§8）：`domain_error`（保留 `code` 和 `field`）、`unknown_tool`、`bad_arguments`（在调用前检查，观察中不含 Python 类名，且该调用不会执行）。出错后 episode 继续，可以恢复。最初还有 `tool_exception_other`（可恢复），第 1 次补修改为基础设施错误 `tool_internal`。
  - 用户模拟器异常、空回复、首句就 STOP，按基础设施错误处理，不产生 reward。
  - 时钟（§6）：每一步都在独立线程中执行，msp_021（对时钟敏感）仍得到 DB=1，且执行线程的时钟在结束后恢复。
  - 奖励（§9）：`done` 或用户 STOP 时只评一次 DB；`nl_status=not_evaluated`，`joint_reward=None`，`training_reward_source=db_only`。走错路径时得到有效的 DB=0。
  - 与 Bench2 `Orchestrator.run` 对照：MSP-021 和 Single-017 各一条脚本（含业务错误和未知工具调用），消息序列、用户收到的消息、最终 DB 哈希完全相同。`bad_arguments` 的文本有意与 Bench2 不同，因此不在对照范围内。
- 不在本步范围（第 5 步）：`max_steps` 与 token 截断、超时、取消、重试调度、进程隔离、并行隔离，以及评分失败后的重新评分接口。

### 第 4 步第 1 次补修（2026-10-30）

- 改动：
  - `src/agent_rl/envs/itsm_env.py`：
    - 公开调用 ID 改为 `call_{seq:03d}`，不由 `episode_id`/`task_id` 派生；
    - 参数在签名绑定之外，再按工具类型注解做 pydantic 严格校验，不通过记为 `bad_arguments`；
    - 校验通过后工具抛出的非 `ItsmError` 异常改为基础设施错误 `tool_internal`：不向模型返回结果，记录 `result_received=False`，env 作废，`export_trace()` 增加 `failure` 字段；
    - reset 遇到 `TaskDefinitionError` 原样抛出，不包装成 `env_build`。
  - `src/agent_rl/envs/itsm_scoring.py`：评分时构造 gold、baseline 环境失败，归为 `ItsmInfrastructureError(stage="score_env_build", kind="gold"|"baseline")`，原始异常保留为 `__cause__`；`TaskDefinitionError` 不包装。
  - `docs/itsm_protocol.md`：§3 泄漏检查改为按信息来源判定；§5 公开调用 ID；§8 删除 `tool_exception_other`，新增 `tool_internal`、`score_env_build` 及 `TaskDefinitionError` 不重试的说明；§11 样例 ID；§12.6 删除"优势置零即丢弃"，要求真正移除或用有效性掩码覆盖全部损失项与统计分母（第 9 步验证）；D5、§12.2、§14.3 修正"温度 0 即确定"的表述。
  - 测试：`test_itsm_env.py` 43 项（+5：类型错误、428 个 gold 调用通过严格校验、全部工具参数有校验器、`tool_internal` 后不能得到 reward、模型可见内容不含任务/episode 元数据、reset 时 `TaskDefinitionError` 不包装；删除旧的 `tool_exception_other` 测试）；`test_itsm_scoring.py` 352 项（+3：gold、baseline 两处构造故障注入，评分时 `TaskDefinitionError` 不包装）。
- 产物：`/home/sar/work/EnterGym/output/itsm-step4-r1/`
  - `pytest_itsm.log`：两套 ITSM 测试 395 passed，退出码 0。
  - `pytest_airlinerl_full.log`：`tests/unit tests/remote` 共 508 passed（113 + 352 + 43），退出码 0。
- 未做：评分失败后的 env 状态与重新评分接口（第 5 步）；无效组排除的实现与梯度验证（第 9 步）。没有重建环境，没有启动 GPU，没有调用 LLM。

### 第 4 步第 2 次补修（2026-10-30）

- 问题：`tool_internal` 记录同时有 `result_received=False` 和非空 `error`，`to_tool_call_record()` 会触发公共 `ToolCallRecord` 的校验异常（`a tool error requires result_received=True`）。
- 改动（`src/agent_rl/envs/itsm_env.py`）：
  - `tool_internal` 记录改为 `result_received=False`、`result=None`、`error=None`，不伪造工具返回；公共 schema 未改。
  - 故障详情只放在 episode 的 `failure`：`{"stage": "tool_internal", "detail": "<异常类型>: <信息>", "call_id": "call_NNN"}`，新增只读属性 `ItsmEnv.failure`，`export_trace()["failure"]` 与之相同。原始异常仍是 `ItsmInfrastructureError.__cause__`。
  - `ItsmToolCallRecord` 增加与公共记录相同的约束（有 `error` 必须 `result_received=True`），防止再出现不可转换的记录。
- 测试（`tests/unit/test_itsm_env.py`，45 项，+2）：
  - `test_internal_tool_exception_is_infrastructure_and_yields_no_reward`：故障记录可转换，转换后 `result_received=False`、`error=None`、`succeeded=False`；`failure` 含 stage、detail、call_id；异常链保留；`reward_info` 为空；之后的 step 和 `done` 都被拒绝。
  - `test_tool_record_rejects_error_without_received_result`：新约束。
  - `test_normal_and_business_error_records_convert_unchanged`：正常结果与已返回的业务错误，转换后各字段与原记录一致。
- 产物：`/home/sar/work/EnterGym/output/itsm-step4-r2/`
  - `pytest_itsm_env.log`：45 passed，退出码 0。
  - `pytest_airlinerl_full.log`：`tests/unit tests/remote` 共 510 passed（113 + 352 + 45），退出码 0。
- 仍未做：评分失败状态与重新评分接口（第 5 步）；无效组排除与梯度检查（第 9 步）。

## 第 5 步证据

- 改动：
  - `src/agent_rl/envs/itsm_env.py`：
    - 状态机 `new / active / interaction_ended / scoring_retry_pending / scored / failed`，外加 `closed` 标志（协议 §7.1）；
    - 新增 `score()`：基于结束时保存的轨迹和最终 DB 重新评分，评分前校验最终 DB 哈希；有效结果缓存复用；基础设施错误最多共 4 次；
    - 预算：`max_steps` 必填，`step_count` 与 `agent_turns` 分开计数，`truncate("response_length" | "invalid_action")`；`invalid_action` 的 `training_reward=0`（D2）；
    - 默认用户模拟器带 `timeout=60s`。
  - 新增 `src/agent_rl/envs/itsm_worker.py`：`ItsmEnvWorker` 在 spawn 子进程中运行一个 episode，支持单次调用超时、episode 截止时间、跨线程 `cancel()`；超时、取消、崩溃时 kill 并 join 子进程，确认退出后才记 `execution_stopped=True`。
  - 新增 `tests/unit/itsm_test_support.py`（确定性用户替身、子进程内可用的慢工具和崩溃工具注入、种子文件校验）、`tests/unit/test_itsm_lifecycle.py`（32 项）、`tests/unit/test_itsm_worker.py`（7 项）；`tests/unit/test_itsm_env.py` 改为显式传入测试预算 `max_steps=40`，用户替身移入辅助模块。
  - `docs/itsm_protocol.md`：§2 接口、§7 计数与预算、§7.1 生命周期、§8.1 超时与取消、§13 隔离验证结果。
  - Bench2、airline 代码和评分模块未修改。
- 产物目录：`/home/sar/work/EnterGym/output/itsm-step5/`（不入 git）。
  - `pytest_itsm_lifecycle.log`：32 passed，退出码 0。
  - `pytest_itsm_worker.log`：7 passed，退出码 0，约 41 秒（每个子进程启动约 3 秒，超时与取消测试各等待慢工具 5 秒以确认其未完成）。
  - `pytest_airlinerl_full.log`：`tests/unit tests/remote` 共 549 passed（113 + 352 + 45 + 32 + 7），退出码 0；测试结束后没有残留子进程。
- 测试性质：真实 Bench2 工具、种子和 DB 评分；确定性用户替身；故障均为受控注入（monkeypatch 让环境构造失败、工具抛异常或变慢；`llm` 裁判为 mock；子进程内通过 initializer 让工具变慢或 `os._exit`）。没有启动 GPU，没有调用真实 LLM，没有实现分组调度。
- 关键结论：
  - 评分失败后业务动作被拒绝；重新评分不调用工具和用户、结束消息只有一条、轨迹和最终 DB 不变；DB=0 与 DB=1 都只评一次并复用；重试共 4 次后转 `failed`；裁判协议错误重评时两次请求完全相同。
  - 生命周期矩阵中每个状态下被拒绝的调用都有测试；`close` 幂等；六条路径（正常、业务错误、工具内部异常、评分失败、预算截断、用户故障）结束后时钟都恢复。
  - `max_steps` = 1、2、3、4、20 与 Bench2 `Orchestrator` 逐项一致；预算处的写调用不执行、DB 不变。
  - 进程内线程超时不能停止执行（反例测试）；worker 超时和取消后子进程已退出，慢工具的完成标记在其原定完成时间之后仍不存在。
  - 并行：线程内 4 个 episode（两个不同时钟任务）和同任务 4 路分组、两个 worker 进程，均互不影响；种子文件前后不变。
- 未解决边界：
  - 进程内 `ItsmEnv` 本身不能超时，需要可中断执行时必须用 `ItsmEnvWorker`；worker 被杀后最终 DB 随之消失，只能重跑 episode，不能重新评分。
  - worker 的复用、进程池、并发数与启动开销（约 3 秒/个）留给第 7/9 步的 rollout 设计。
  - 用户模拟器的 60 秒 HTTP 超时只在代码路径上接入，没有调用真实 API 验证。
  - `kill_grace_s` 内未能回收子进程的情况（`not_reclaimed`）只有代码路径，没有构造出该故障。
  - 正式 `max_steps` 和 token 预算留到第 6 步；无效组排除与梯度检查留到第 9 步。

### 第 5 步第 1 次补修（2026-10-30）

- 问题（已复现）：子进程空闲时退出，下一次调用的 `conn.send()` 抛出未分类的 `BrokenPipeError`，worker 仍为 `active`、`failure=None`。用修补前的代码运行新测试：`exited_before_send` 失败于 `BrokenPipeError`（`output/itsm-step5-r1/pytest_before_fix.log`）。
- 改动（`src/agent_rl/envs/itsm_worker.py`）：发送阶段的 `OSError`/`EOFError` 直接判为 `ItsmInfrastructureError(stage="worker_crashed")`，走与超时、崩溃相同的回收路径（kill、join、记录 `exitcode`，确认退出后才记 `execution_stopped=True`）；原始错误保存在 `failure["cause"]` 和异常的 `__cause__`。没有用发送前的 `is_alive()` 检查替代：发送先于退出完成的情况由原有的接收路径判定为同一 stage。
- 测试（`tests/unit/test_itsm_worker.py`，9 项，+2）：`test_child_exit_while_idle_is_a_crash_on_the_next_call[exited_before_send | racing_exit]`。reset 后 SIGKILL 子进程，前者等子进程确认退出后再 step，后者立即 step；均验证 stage 与 kind、`failed` 状态、`exitcode=-9`、`execution_stopped=True`、子进程已回收、后续 step/score/truncate 被拒绝、轨迹不可导出（无 reward）、`close` 幂等；前者另验证原始 `OSError` 保留。
- 产物：`/home/sar/work/EnterGym/output/itsm-step5-r1/`
  - `pytest_itsm_worker.log`：9 passed，退出码 0；
  - `pytest_idle_exit_repeats.log`：新测试重复 3 次，均 2 passed；
  - `pytest_airlinerl_full.log`：`tests/unit tests/remote` 共 551 passed，退出码 0；无残留子进程。

## 第 6 步证据

- 产物清单与全部依据：`docs/itsm_step6/README.md`。
- 改动：
  - 新增 `src/agent_rl/data/itsm_split.py`、`scripts/data/build_itsm_split.py`、`scripts/data/measure_itsm_tokens.py`；
  - 新增 `configs/data/itsm_split_v1.json`、`configs/itsm/budget_v1.yaml`；
  - 新增 `docs/itsm_step6/`（README、`family_audit.json`、`token_measurements.json`）；
  - 新增 `tests/unit/test_itsm_split.py`（16 项）、`tests/unit/test_itsm_budget.py`（5 项）；
  - 协议 §7 改为指向预算文件。
  - Bench2、环境、评分和训练代码均未修改。
- 划分：11 个家族（写工具 Jaccard ≥ 0.6 或共享新建对象工具，取连通分量），train 29 / dev 13 / held-out 11，按家族整体划分，种子 20261030。互斥、覆盖、家族不跨划分、MSP-021/022、025/026、027/028 及跨 tenancy 近重复同处一个划分，均有测试。
- **待你决定**：三个大家族占 40/53 题，存在取舍（README §2.3）。v1 选择让 held-out 更大更多样，代价是训练集没有"挂起并通知"工作流，dev 主要反映该工作流。
- token 测量：tokenizer `Qwen/Qwen3-8B@b968826d…`，chat template sha256 `a55ee1b1…`，`enable_thinking=False`；只用 train/dev 的 gold 回放，不代表实际策略输出。初始 prompt 28,104，gold 回放响应最大 5,764，完整序列最大 33,868。
- 运行（在 `code/main` 下，日志在 `/home/sar/work/EnterGym/output/itsm-step6/`）：
  - `python scripts/data/build_itsm_split.py --check`：up to date；
  - `python scripts/data/measure_itsm_tokens.py --tokenizer-dir ~/.cache/entergym/tokenizers/Qwen3-8B-b968826d`：退出码 0（`measure_tokens.log`），HF 离线模式运行；
  - `pytest tests/unit/test_itsm_split.py tests/unit/test_itsm_budget.py -v`：21 passed（`pytest_itsm_step6.log`）；
  - `pytest tests/unit tests/remote -q`：572 passed（`pytest_airlinerl_full.log`）。
- 没有启动 GPU，没有加载模型权重，没有调用真实 LLM，没有接入第 7 步采样或第 9 步调度。

### 第 6 步收尾（2026-10-30）

- 划分：用户选定 v1，划分结果未变（train 29 / dev 13 / held-out 11）。
- 实验定位：跨家族与工作流组合泛化。manifest 新增 `positioning`；审计新增 `cross_split_shared_subworkflows`（68 对 train/held-out 共享 ≥ 2 个写工具）。single_015/022 保留跨划分，记录共享子流程 `update_change → update_problem → map_change_request` 及各自不同的后续组合；不以 Jaccard 低于阈值作为无近重复的证明。
- checkpoint：不用 dev 选最佳 checkpoint，held-out 不参与；训练步数与规则事先确定，最迟第 8 步冻结（预算文件 `checkpoint_selection`、manifest `usage`）。
- 无损紧凑序列化（离线）：新增 `src/agent_rl/prompts/tool_serialization.py`、`scripts/data/measure_itsm_schema_serialization.py`、`docs/itsm_step6/schema_serialization.json`。三个变体解析后与原 94 个 schema 完全相等，工具块以外 prompt 不变。`json_compact` 省 3,772 token（初始 prompt 28,104 → 24,332），`yaml_block` 省 2,127；不足以解决长度问题，未采用。
- 上下文：40,960 记为待验证候选（原生 32,768），不加 RoPE 扩展；第 7 步单并发有限生成验证长度与推理显存，第 9 步验证训练显存。
- 观察规则：删除 `max_observation_tokens: 4096`，改为 `observation_rule: full_tool_results_no_truncation`；放不下时以 `response_length` 结束并保留完整诊断轨迹。`token_measurements.json` 重跑后写明同一规则，token 数字与上次完全一致，仅计时变化。
- 测试：`test_itsm_split.py` 17 项、`test_itsm_budget.py` 8 项、新增 `test_itsm_tool_serialization.py` 5 项。
- 产物：`/home/sar/work/EnterGym/output/itsm-step6-r1/`（`measure_schema_serialization.log`、`measure_tokens.log`、`pytest_itsm_step6.log`、`pytest_airlinerl_full.log`）。

## 第 6A 步证据

- 全部结论与试点清单：`docs/itsm_step6a/README.md`；原始结果：`docs/itsm_step6a/eopsgym_itsm_audit.json`。
- 参考数据：`/home/sar/work/EnterGym/reference/enterpriseops-gym/`（不入 git，`SOURCE.json` 记录来源）。仓库 commit `271f2c3`，HF revision `c8e538ea`，均为 Apache-2.0。下载 4 个 ITSM parquet，从 `gym_dbs.zip` 只解出 9 个 ITSM SQL 种子。没有覆盖 Bench2，没有并入项目任务集。
- 改动：新增 `scripts/data/audit_eopsgym_itsm.py`、`tests/unit/test_itsm_eopsgym_audit.py`（6 项）、`docs/itsm_step6a/`。业务工具、评分、v1 划分均未修改。
- 主要结论：
  - 4 种工具集模式是同一套 103 题；
  - 64 个 oracle 工具全部存在于 Bench2；
  - 9 个种子都对应 Bench2 现有种子（单租户 100%，MSP ≥ 99.6%），62 题需要小 delta；
  - 464 个验证器中有 2 题冲突不可满足、1 题不可执行、15 题有重复、40 题写死新建 ID；
  - 分类：可直接适配 10、需小幅适配 44、暂不适合 49（其中 46 题仅因 v1 划分关系）；
  - 推荐 8 题训练侧试点。
- 运行（`code/main`，日志 `/home/sar/work/EnterGym/output/itsm-step6a/`）：
  - `python scripts/data/audit_eopsgym_itsm.py --ref-dir …/reference/enterpriseops-gym`：退出码 0；
  - `pytest tests/unit/test_itsm_eopsgym_audit.py`：见 `pytest_eopsgym_audit.log`；
  - 全量回归：见 `pytest_airlinerl_full.log`。
- 没有 gold 回放，没有启动 GPU，没有加载权重，没有调用 LLM。

## 第 6B 步证据

- 逐题验收：`docs/itsm_step6b/README.md`；原始结果：`docs/itsm_step6b/pilot_validation.json`；试点文件：`configs/data/itsm_eops_pilot_v1.json`（不进 Bench2 `tasks.json`，不进 v1 划分）。
- 改动：
  - 新增 `src/agent_rl/data/itsm_eops_pilot.py`、`scripts/data/build_itsm_eops_pilot.py`、`scripts/data/validate_itsm_eops_pilot.py`、`tests/unit/test_itsm_eops_pilot.py`；
  - `ItsmEnv` 增加可选参数 `task=`（id 必须与 config 一致；不传时行为不变）。
  - Bench2、评分模块、v1 划分均未修改。
- 结果：
  - 通过 3 题：`b6ef60ca`、`52c8290e`、`1d219d93`。Bench2 DB=1；上游验证器在 gold 上全部通过、在初始状态下全部不通过；12 个反例全部 DB=0（7 个上游也能识别，5 个只有 Bench2 能识别）。
  - 隔离 5 题，原因：REQUIRED 连带操作结果不唯一、prompt 与验证器矛盾、与 REQUIRED 规则冲突、REQUIRED 操作无对应工具。
- 实测（脚本化 episode）：11–13 步，响应 2.2k–2.9k，完整序列 30.4k–31.1k token。
- 合并后复查：试点题只连到 train 家族，dev/held-out 不变；`1d219d93` 把两个 v1 训练家族连成一个分量。
- 运行：`build_itsm_eops_pilot.py --check` 结果为 up to date；`validate_itsm_eops_pilot.py` 退出码 0；测试日志见 `/home/sar/work/EnterGym/output/itsm-step6b/`。
- 没有启动 GPU，没有调用 LLM，没有扩大规模；纳入训练集 v2 待确认。

## 第 6C 步证据

- 说明：`docs/itsm_step6c/README.md`；v2 manifest：`configs/data/itsm_split_v2.json`（train 32 / dev 13 / held-out 11）；v1 manifest 保留未改（sha256 `77cc5519…`）。
- 改动：
  - 新增 `src/agent_rl/data/itsm_split_v2.py`、`src/agent_rl/data/itsm_task_sets.py`、`scripts/data/build_itsm_split_v2.py`；
  - `itsm_split.py`：抽出 `connected_components`，新增共用的 `validate_split_structure`（v1 校验也一并收紧，v1 文件不变）；
  - `ItsmEnvConfig.split_version`（默认 `None`，行为不变）；worker 子进程按 config 解析任务；
  - `measure_itsm_tokens.py` 增加必填参数 `--split-version`，并按题记录未过滤读取上界；
  - `budget_v1.yaml` 新增 `basis` 段，数值未改；试点文件 `status` 一行更新。
  - Bench2、评分、业务工具、隔离的 5 题均未改动。
- 结果：
  - 家族重算后，两个 v1 train 家族合并（30 题），另新增 1 个家族，其余 9 个家族不变；
  - gold 路径的 token 最大值全部不变；
  - `1d219d93` 加载整个 MSP 种子，3 个未过滤读取单次超过 `max_response_length`（最大 30,041）。预算处理方案待验收（建议保持现状）。
- 测试：新增 47 项；全量回归 655 passed。日志见 `/home/sar/work/EnterGym/output/itsm-step6c/`。
- 没有启动 GPU，没有加载权重，没有调用真实 LLM，没有进入第 7 步。

## 第 6D-1 步证据

- 产物与全部依据：`docs/itsm_step6d/README.md`；重划分提案 `docs/itsm_step6d/v3_proposal.md`（只是提案，未生成 manifest）。
- 改动：新增 `scripts/data/screen_itsm_eops_candidates.py`、`docs/itsm_step6d/candidate_screening.json`、`tests/unit/test_itsm_eops_screening.py`（13 项）。Bench2、policy、评分、环境、v1/v2 manifest、试点文件均未修改；已隔离的 5 题不在候选范围内。
- 结果：95 个候选中 34 题通过训练侧门槛（按 v2 重算），逐题人工审读后：入选 10（8 道忠实迁移、2 道 Bench2 改编，均只进训练侧）、备选 2、排除 12、暂缓 4、等待 D6D-1 决定 6。入选 10 题两两不连家族，新覆盖 `update_notification`、`remove_group_membership`。
- 待决定：D6D-1（MSP 建事件单的 SLA 生成规则）、D6D-2（建用户类派生题是否沿用 Bench2 原题不写 `user_role` 的口径），以及 2 号、5 号题的非惯例切分范围。
- 规模：只靠上游题库，train 现实上限约 50–55（需 D6D-1 与一次门槛放宽）；60–70 需新写任务。held-out 最多 11 → 19，只能扩充已有 held-out 家族。
- 运行（在 `code/main` 下，日志在 `/home/sar/work/EnterGym/output/itsm-step6d/`）：
  - `python scripts/data/screen_itsm_eops_candidates.py --tokenizer-dir ~/.cache/entergym/tokenizers/Qwen3-8B-b968826d --check`：up to date（`screen_check.log`）；
  - `pytest tests/unit/test_itsm_eops_screening.py`：13 passed；`pytest tests/unit tests/remote -q`：668 passed（`pytest_full.log`）；v1、v2 的 `--check` 均为 up to date。
- 没有转换任何题，没有启动 GPU，没有加载权重，没有调用 LLM，没有进入第 7 步或 6D-2。

### 第 6D-1 步之后的用户决定（2026-10-30）

- 原话全文与逐项决定见 `docs/itsm_step6d/README.md` §6。要点：7 题进入 6D-2；第 5、9、10 题暂缓；不制定 MSP SLA 全局规则；不沿用省略 `user_role` 的口径；不放宽共享写工具门槛；继续准备 v3，两道服务目录题优先作为 held-out 候选，验证完成后再冻结，v1/v2 不改；当前以纯 GRPO 为首选，SFT 不在本轮授权范围；第 6C 步验收通过，预算选项 A，40,960 仍待验证。
- 预算文件 `configs/itsm/budget_v1.yaml` 中 `basis.itsm_split_v2.status` 改为 `accepted_step6c_option_a`，数值不变；`test_itsm_budget_v2.py` 同步。
- 更正：上一轮对话中说 AirlineRL 正式训练“约 4 个优化步”，有误。正式配置为每步 30 题 × G=4 = 120 条轨迹，目标 75 个 trainer 全局步（`run_formal_phase.sh`、`docs/experiment_plan.md`）；之前误用了性能测试配置 `pro6000_g4_tokens32k.yaml` 推算。

## 第 6D-2 步证据

- 产物与逐题结果：`docs/itsm_step6d2/README.md`；批次文件 `configs/data/itsm_eops_batch2_v1.json`；验证结果 `docs/itsm_step6d2/batch_validation.json`。
- 改动：
  - 新增 `src/agent_rl/data/itsm_eops_batch2.py`、`scripts/data/build_itsm_eops_batch2.py`、`scripts/data/validate_itsm_eops_batch2.py`、`tests/unit/test_itsm_eops_batch2.py`（27 项）。
  - 6D-1 的更正：修改 `scripts/data/screen_itsm_eops_candidates.py`（MSP 的 SLA 定义对所有组织通用；实体门槛的局限写进规则；冲突取值改为确定），重新生成 `candidate_screening.json`，并在 `docs/itsm_step6d/README.md` 增加 §7。入选批次、人工结论和 v3 可用量都没有变化。
  - Bench2、policy、评分、环境、v1/v2 manifest、预算数值、试点文件均未修改。
- 结果：
  - **通过 4 题**：第 1、3、6、7 题。这 4 题满足：gold DB=1 且重复回放结果一致；上游验证器在 gold 上全部通过，在初始状态下只有已登记的约束或前提通过；16 个反例全部 DB=0，mismatch 符合预期；信息边界没有泄漏；脚本化 episode 以 agent_done 结束；种子 + delta 与上游的差异为 0；按 v2 的家族复查 5/5 通过。
  - **隔离 3 题**：
    - 第 2 题：REQUIRED 的 SLA 连带操作结果不唯一。6D-1 的相关说法有误，已更正。
    - 第 4 题：核心要求"记录 SRV-LNX-4421"在 Bench2 评分中体现不出来，反例的 DB = 1。
    - 第 8 题：与 held-out 题 `itsmbench_single_012` 共享 GROUP_001 和 INC_003。
  - 隔离证据由验证脚本和测试重新计算。
- 运行（在 `code/main` 下，日志在 `/home/sar/work/EnterGym/output/itsm-step6d2/`）：
  - 批次文件、试点文件、v1、v2、筛选结果的 `--check` 均为 up to date；
  - `pytest tests/unit tests/remote -q`：695 passed（`pytest_full.log`），跑完没有残留子进程。
- 没有把任何题并入划分，没有启动 GPU，没有加载权重，没有调用 LLM，没有处理暂缓的第 5、9、10 题，也没有开始 v3 的审读。

## 第 6D-3 步证据

- 准入规则：转换之前写定并推送（`3c7e2b7`，`docs/itsm_step6d3/README.md` §1），之后没有改动。规则比 v3 提案更严：共享写工具和实体也对照 dev；实体按 gold 参数计；核心自由文本要求必须有"只改关键事实"的反例。
- 产物与逐题结论：`docs/itsm_step6d3/README.md`；候选文件 `configs/data/itsm_eops_heldout_v3_candidates.json`；验证结果 `docs/itsm_step6d3/heldout_validation.json`。
- 改动：新增 `src/agent_rl/data/itsm_eops_heldout_v3.py`、`scripts/data/build_itsm_eops_heldout_v3.py`、`scripts/data/validate_itsm_eops_heldout_v3.py`、`tests/unit/test_itsm_eops_heldout_v3.py`（17 项）。Bench2、policy、D8、fuzzy 阈值、评分、v1/v2 manifest、预算、已有批次文件均未修改。
- 结果：
  - **通过 1 题**：`99ba2325_54f35603`（SAP ERP：不退役，设为 non-operational）。全部检查通过：5 个核心反例 DB=0；判断量在切分范围内与不切分时都是 4；单独成为一个新的 held-out 家族。
  - **转换后隔离 3 题**：第 1、5、7 题，核心自由文本要求的关键事实反例得到 DB=1（召回率 0.64–0.90，阈值 0.5）；第 7 题另与 train 共享 2 个 gold 实体。
  - **转换前隔离 4 题**：
    - 第 3 题：prompt 自相矛盾，事件单不唯一；
    - 第 4 题：上游要求关闭 new 状态的事件单，违反 §3.3，也不是 prompt 的要求；
    - 第 6 题：逐字消息与验证器不一致，验证器要求 sent 状态；
    - 第 8 题：`$9,50` 有歧义。
- v3 预期规模（如果你验收本步）：train 36、dev 13、held-out 12（7 个家族）。没有生成 v3 manifest。
- 统计表述已更正（6D-1 README §4.2、`v3_proposal.md`）：撤回独立二项标准误和"小于标准误就无法区分"的说法。
- 运行（在 `code/main` 下，日志在 `/home/sar/work/EnterGym/output/itsm-step6d3/`）：
  - 候选文件、6D-2 批次、试点文件、v1、v2、筛选结果的 `--check` 均为 up to date；
  - `pytest tests/unit tests/remote -q`：712 passed（`pytest_full.log`），跑完没有残留子进程。
- 没有测量 held-out 候选的 token，没有运行模型，没有启动 GPU，没有调用 LLM，没有做 SFT 或训练，没有开始第 7 步。

## 第 S0 步证据（新路线 SFT → GRPO）

- 设计与结论：`docs/itsm_stepS0/README.md`；离线审计结果：`docs/itsm_stepS0/usage_audit.json`。
- 改动：新增 `src/agent_rl/data/itsm_s0_usage.py`、`scripts/data/build_itsm_s0_usage_audit.py`、`tests/unit/test_itsm_s0_usage.py`（7 项）；提交用户提供的新路线文件 `docs/playbook2.md`（未改内容）。Bench2、policy、D8、fuzzy 阈值、评分、gold、用户文字、v1/v2 manifest、试点与批次文件、预算数值均未修改；没有生成 v3 manifest 或用途版本。
- 6D-3-R1 并入本步：7 道派生训练题共补 12 个 S0 反例（10 个关键事实反例全部得 DB=1，召回率 0.50–0.98；2 个结构化反例得 DB=0），既有 28 个验证反例重跑仍为 DB=0。
- 结果（均为提案，待验收）：
  - GRPO 建议暂停 4 题（`52c8290e`、`1d219d93`、`eff85dae`、`7915481a`），其余 3 道派生题保留；7 题都可作 SFT 来源；
  - 用途登记候选：both 32、SFT-only 4、SFT-only 候选 1（`78a5b71e_a4fb07a9`）、dev 13、held-out 11（6D-3 候选 `54f35603` 保留为评测题）；
  - SFT∪GRPO 对照 dev/held-out 的联合隔离：5 项检查全部通过；Bench2 原题有 118 对跨划分共享写工具的子工作流（v1 已说明），没有实体重叠 ≥ 2，也没有近重复；
  - Bench2 原 29 道 train 题中有 5 题的 gold 写了自由文本，本步没有做关键事实反例（待决定）；
  - 试点：P1–P5 必选，P6 可选，每题至多 2 条，总上限 10 条（含 P6 为 12 条）；API 上限 35 美元，预期 1–3 美元；GPU 时长只提方案。
- 运行（在 `code/main` 下，日志在 `/home/sar/work/EnterGym/output/itsm-stepS0/`）：`build_itsm_s0_usage_audit.py --check` 在哈希种子 1、7 下都是 up to date；`pytest tests/unit/test_itsm_s0_usage.py`：7 passed；全量回归见 `pytest_full.log`。
- 没有调用 LLM，没有加载分词器或模型，没有启动 GPU，没有采样或训练，没有进入第 7 步或 S1。

### 第 S0-R1 补修证据（2026-10-31）

- 改动：`src/agent_rl/data/itsm_s0_usage.py`、`tests/unit/test_itsm_s0_usage.py`（7 → 9 项）、重新生成的 `docs/itsm_stepS0/usage_audit.json`（版本 `itsm_usage_s0_candidate_v2`）、`docs/itsm_stepS0/README.md`（新增"S0-R1 补修"节，修订 §1–§4、§6.1、§8、§9）。评分、阈值、D8、gold、policy、用户措辞、v1/v2 manifest 和各批次文件均未修改。
- 结果：
  - `b6ef60ca` 新增反例"Software removal"（category 仍为 software）：真实工具调用成功，fuzzy 召回率 0.50，DB = 1。该题改为暂停 GRPO、保留 SFT。
  - 判定规则改为一项核心要求（一个事实）的每个关键反例都须 DB = 0；`b6ef60ca` 的主题拆成"软件而非硬件""安装而非卸载"两个事实，两项都未被区分。新规则下其余 6 道派生题结论不变。
  - 用途候选：GRPO 31（其中 single_005/006/007/011/015 待补 R1，第 9 步前完成）；SFT 37（SFT-only 6：`b6ef60ca`、`52c8290e`、`1d219d93`、`eff85dae`、`7915481a`、`78a5b71e`）；dev 13；held-out 11 + 保留 `54f35603`。GRPO 池 3 个家族（29/1/1），SFT 池 6 个家族（32/1/1/1/1/1）。联合隔离 5 项检查重新计算，全部通过。
  - Teacher/裁判同源偏差的表述已修正（不再声称 E1–E5 彼此比较不受影响），补充实际模型版本核对方案；API 上限改为 45 美元，计入全部重试和收费请求，按轨迹和按请求预留费用，停止条件不依赖账单。
- 运行（在 `code/main` 下，日志在 `/home/sar/work/EnterGym/output/itsm-stepS0-r1/`）：`build_itsm_s0_usage_audit.py --check` 在哈希种子 1、7 下都是 up to date；held-out 候选、6D-2 批次、试点、v1、v2 的 `--check` 均为 up to date；`pytest tests/unit tests/remote -q`：721 passed（`pytest_full.log`）。
- 没有生成 v3，没有进入第 7 步或 S1，没有调用 LLM，没有启动 GPU，没有训练。

## v3 manifest 与用途登记证据（2026-10-31）

- 产物与变更说明：`docs/itsm_v3_usage/README.md`；`configs/data/itsm_split_v3.json`；`configs/data/itsm_usage_v1.json`。
- 改动：新增 `src/agent_rl/data/itsm_split_v3.py`、`src/agent_rl/data/itsm_usage.py`、`scripts/data/build_itsm_split_v3.py`、`scripts/data/build_itsm_usage.py`、`tests/unit/test_itsm_split_v3.py`（17 项）；`src/agent_rl/data/itsm_task_sets.py` 增加 v3 与按用途选题（`training_task_ids`），并拒绝直接列出 v3 train；`docs/itsm_stepS0/README.md` 更新 Teacher（GPT-6.1 Sol / Micu）、§6.1 预算公式与 §8 决定；`docs/itsm_step6d/v3_proposal.md`、`docs/itsm_step6d3/README.md` 更新状态。v1/v2 manifest、Bench2、评分、阈值、policy、gold 与各批次文件均未修改。
- 结果：
  - v3：train 37 / dev 13 / held-out 12，15 个家族（train 6、dev 2、held-out 7）；家族、实体、近重复与上游来源隔离检查全部通过；每题记录 content 与 source 两个哈希。
  - 用途：SFT 来源 37，GRPO 候选 31（可训练 26、R1 pending 5），SFT-only 6；来源获准与轨迹接纳分开登记，轨迹接纳目前为空。
  - 入口：v3 train 必须通过 `training_task_ids` 选取；`purpose="grpo"` 只返回 26 题，不含 SFT-only、pending、dev、held-out；held-out 仍需 `final_evaluation=True`。
  - S1 预算：总上限 45 美元；GPT-6.1 Sol 价格未知，改为公式核算，第 7 步核实价格后代入。
- 运行：在 `code/main` 下运行，日志在 `/home/sar/work/EnterGym/output/itsm-v3-usage/`：`build_itsm_split_v3.py --check`、`build_itsm_usage.py --check` 在哈希种子 1、7、13 下都是 up to date；S0 审计、held-out 候选、6D-2 批次、试点、v1、v2 的 `--check` 均为 up to date；`pytest tests/unit tests/remote -q`：738 passed（`pytest_full.log`），跑完没有残留子进程。另有 2 条旧断言按当前状态更新：`test_itsm_eops_heldout_v3` 原先断言 v3 文件不存在，改为断言 54f35603 在 v3 held-out、隔离题不在任何划分；`test_itsm_task_sets` 中"未知版本"的示例从 itsm_split_v3 改为 itsm_split_v4。
- 没有开始第 7 步或 S1，没有调用 LLM，没有启动 GPU，没有训练。

## 冻结记录：v3 manifest 与 `itsm_usage_v1`（2026-10-31）

- 用户原话：“v3 manifest 与 itsm_usage_v1 冻结验收通过。冻结依据为提交 70e6a3b 的文件内容及完整 SHA256。”
- 提交：`70e6a3b451149bb24ae0eba18fc544705d4c323a`（2026-10-31 08:34:14 -0400）。
- `configs/data/itsm_split_v3.json`：SHA256 `9326167dfee696a809f685f178284e23e6e8f3402559d4b47650c4fcf85b5bc0`。
- `configs/data/itsm_usage_v1.json`：SHA256 `87f15baf96a5dbeec6a8c07ca0952edafd86667d18dfa01d63f066d35934fba7`。
- 冻结文件本身不改，`itsm_usage_v1.json` 中的 `status` 字段保持原文。冻结记录写在本文件和 `src/agent_rl/data/itsm_frozen.py`；加载 v3 或用途登记时会先核对上面两个 SHA256。
- 以后用途资格有变化，新建用途版本；轨迹审核另存文件，并引用上面两个冻结哈希。

## 第 7A 步证据

- 报告：`docs/itsm_step7a/README.md`。
- 改动：
  - 新增 `src/agent_rl/rollout/itsm_agent_loop.py`、`src/agent_rl/data/itsm_frozen.py`、`src/agent_rl/utils/spend_ledger.py`、`scripts/itsm/probe_api_7a.py`；
  - 新增测试 `tests/unit/test_itsm_agent_loop.py`（23 项）、`tests/unit/test_itsm_api_probe.py`（11 项）；`tests/unit/test_itsm_split_v3.py` 增加冻结哈希测试（+1）；
  - `src/agent_rl/data/itsm_task_sets.py` 加载 v3 或用途登记时核对冻结哈希。
  - 冻结文件、v1/v2、Bench2、评分、policy 和预算数值均未修改。
- 离线验证：确定性采样替身加真实 Qwen3 分词器、真实 ItsmEnv。验证了输入渲染、单动作（6 种非法输出）、增量观察（与第 6 步计数一致）、token/span/mask、logprob 来源（缺失或条数不对时报错，不补 0）、不轮换上下文、按用途限制取题。**这些都不是真实采样 logprob 的验收**。
- API 探测：**没有发出任何付费请求，花费 0 美元**（第 7A 步预算 1 美元 / 6 次，与 S1 分开记账）。停止原因：
  1. 唯一的云端实例需要以 GPU 模式开机，与"不启动 GPU"冲突；
  2. 没有 Micu 的 key 和 base URL；
  3. Micu 的价格页要登录才能看，公开资料只有第三方列出的 OpenAI 标价，不能作为预留依据。
  - 探测脚本和账本已经写好，并用模拟 HTTP 做了离线测试。
- 运行（在 `code/main` 下，日志在 `/home/sar/work/EnterGym/output/itsm-step7a/`）：v3、用途登记（哈希种子 1、7）、S0 审计、held-out 候选、6D-2 批次、v1、v2 的 `--check` 均为 up to date；两个冻结文件的 SHA256 与冻结值一致；`pytest tests/unit tests/remote -q`：773 passed（`pytest_full.log`），跑完没有残留子进程。
- 没有运行 held-out，没有生成 SFT 轨迹，没有更新权重，没有启动 GPU，没有进入第 7B 步、第 8 步或 S1。

### 第 7A 步补充：免费核对（2026-10-31）

- 用户决定（AskUserQuestion）：免费查询在本机跑；付费探测允许以 GPU 模式开机、跑完立即关机；价格以价格接口为准，查不到就停，等用户提供截图。
- 凭据文件 `code/main/micu.md` 没有被 git 跟踪，已加入 `.git/info/exclude`，权限 600；key 没有出现在任何输出里。
- 结果：`/v1/models` 列出 `gpt-6.1-sol`（与请求完全一致）；`/api/pricing` 拒绝 API key（要系统访问令牌）；`/api/status` 不含按模型的价格；key 的额度上限远高于预算，不能用来控费。
- **价格仍未核实，按用户规则停止**：没有开机，没有付费请求，花费 0 美元。详见 `docs/itsm_step7a/README.md` §3.4。

### 第 7A 步补充：付费探测（2026-10-31）

- 用户原话：\"改用5.6 sol影响大吗，输入5人民币/M，输出30人民币/M\"、\"long context模式价格要乘以2.跑吧\"。据此 Teacher 改为 `gpt-5.6-sol`，预留按乘 2 后的价格、6.5 人民币兑 1 美元的保守汇率计算。
- 凭据文件已从 `code/main/micu.md` 移到 `/home/sar/work/EnterGym/.secrets/micu.md`（700/600 权限），原位置在代码同步范围内。
- 云端：exp-itsm 以 GPU 模式开机（用户已批准），只用来发请求；结束后删除了云端的凭据文件，并已关机（平台确认 shutdown）。
- 结果：
  - 只发出 1 次付费请求（B1，Responses 纯文本，HTTP 200）。实际计费 input 4,393 token，其中 4,380 是 Micu 或其上游**注入的隐藏指令**，超过了这次的预留，按停止条件停下；B2、B3 没有发出。
  - 账本记账 0.006805 美元；账号实际用量增加约 ¥0.011，低于账本记账。
  - DeepSeek 官方价格页显示 `deepseek-v4-flash` 已下线，旧名请求实际由 `DeepSeek-V4.1-Flash-1017` 提供服务（影响 D5）。
  - 云端和本机都没有 DeepSeek key，B4、B5 没有运行。
- 待用户决定：隐藏指令与 S0 §4.2 输入边界的冲突；D5 模拟器模型；DeepSeek key；剩余探测（预算还剩 0.993 美元、5 次请求）。详见 `docs/itsm_step7a/README.md` §3.5。

### 第 7A 步补充：D5 改为 deepseek-flash，补完付费探测（2026-10-31）

- 用户原话：\"那就deepseek吗，api在deepseek.md\"。D5 改为 `deepseek/deepseek-flash`（DeepSeek-V4.1-Flash），不用 Micu GPT；ITSM 代码默认值、`budget_v1.yaml` 和协议 D5 已同步，airline/tau2 的配置没有改动。
- 凭据 `deepseek.md` 已移到 `.secrets/`（600）。云端用的是只有 root 可读的临时文件，跑完已删除，实例已关机（平台确认 shutdown）。
- 付费探测（累计账本）：总计 6 次请求、0.028066 美元，在 6 次、1 美元上限之内，没有触发停止条件。
  - Teacher `gpt-5.6-sol`：Responses 和 Chat 两条路径都能正确返回单个工具调用，返回的模型 ID 一致，推理 token 为 0；**两条路径都被注入了约 4,380 token 的隐藏指令**。
  - 裁判 `deepseek-v4-pro`：接受 D7 参数，没有 `reasoning_content`，`finish_reason=stop`。
  - 模拟器 `deepseek-flash`：返回模型一致，但默认开启思考。
- 待用户决定：是否接受隐藏指令这项偏差；模拟器是否关闭思考；DeepSeek 余额（¥5.92）需在第 8 步前充值。Micu 账号用量里混有他人用量，不能用来核对扣费，费用以本地账本为准。详见 `docs/itsm_step7a/README.md` §3.6。

## 第 7A-R1 步证据（2026-10-31，离线）

- 报告：`docs/itsm_step7a/README.md` 开头的"最终状态"、§2（新增 4 行）、§3.7、§4（7B 方案修订）。
- 改动：
  1. `itsm_agent_loop.py`：每个 episode 在 `ItsmEnvWorker` 子进程中运行（`call_timeout_s` 90、`episode_timeout_s` 1800，取自 budget v1）。rollout 任务被取消时先 `cancel()`（杀掉并回收子进程），再让取消继续传播，`finally` 中 `close()`。新增测试：慢工具执行中取消、调用超时、默认路径跑完一条 episode。
  2. `check_sampler_output`：没有 token、logprob 缺失、条数不符、NaN/±Inf，都抛出 `ItsmInfrastructureError(stage="sampler_contract")`，不产生 reward；`SamplerContractError` 已删除。
  3. `probe_api_7a.py`：所有检查都是必需的；非 200、返回体不是 JSON、缺失结果或 usage、工具错误、参数不正好是 `{"number": "INC0000007"}`、其他检查为 false、意外异常，都会停止并以非零码退出；报告里记录工具名和参数；B4 发送 `thinking: disabled`。全部用模拟 HTTP 测试（包括 `main()` 的退出码），没有新增付费请求。用新条件回看已发出的 6 次请求：B2/B3 的参数值和 B3 的工具名当时没有保存，无法确认；B4 当时没有关闭思考，按新条件不通过。
  4. `spend_ledger.py`：对 `<账本>.lock` 加独占非阻塞 `flock`，第二个写入者直接被拒绝（`LedgerBusy`）；新写入者重读文件；增加 `close()`/上下文管理器/`read_summary`。回归测试：同一进程两个实例、两个进程，各预留 0.75、上限 1。
  5. D5：`DEFAULT_USER_LLM_ARGS` 改为温度 0 + `extra_body={"thinking": {"type": "disabled"}}`；`budget_v1.yaml` 的 train/eval 增加 `user_llm_thinking: disabled`；协议 D5 已同步。离线测试用假的 litellm 截下参数核对。
  6. 文档：S0 §4.2 和 D-S0-5 记录隐藏指令为 S1 试点的记录偏差；账本记账与实际扣费分开写；输出上限标为尚未触顶验证。
  7. README 开头改为最终状态，§3.1–§3.6 作为带时间的过程记录保留；7B 方案修订（真实长度三档测试、只用现有磁盘的存储方案、不自动扩盘、不新租实例）。
- 冻结文件、v1/v2、Bench2、评分和预算数值均未修改。
- 没有调用 LLM，没有启动 GPU，没有新增付费请求，没有进入 7B 或 S1。

## 第 7A-R2 步证据（2026-10-31，离线）

- 问题：原来 worker 在线程里构造，构造函数里会启动子进程并等待它就绪。rollout 在这段时间被取消时，后台线程仍会构造完成，并把带着活子进程的 worker 返回给一个已经没人接收的 Future，这个 worker 就丢了。
- 改动：
  - `itsm_worker.py`：新增 `start=False` 和 `start()`。拿到句柄时还没有子进程；在 `cancel()` 或 `close()` 之后调用 `start()` 会被拒绝；创建子进程与 `_stop_process` 在同一把锁下互斥；未启动的句柄可以安全关闭；默认 `start=True`，保持第 5 步的行为。
  - `itsm_agent_loop.py`：先在事件循环上拿到句柄（不起进程），再在线程里启动，等待上限为 `startup_timeout_s + kill_grace_s + 5` 秒。无论正常结束、出错、启动超时还是被取消，都在单独的 task 里执行同一套有界清理：取消、有界等待启动线程返回、关闭（杀掉并回收）。第二次取消打断不了清理；清理没能在上限内确认子进程退出，就记日志 `not_reclaimed`。只有取消时才直接杀子进程；其他错误走正常关闭，保留 trace 以便诊断。
  - 测试：启动中取消（子进程已存活、还没报回 pid，另有清理中再取消一次的变体）；后台启动返回了就绪 worker 后才处理取消；启动失败；3 项未启动句柄的 worker 测试。原有的慢工具取消、超时、完整 episode 测试照常通过。
  - 变异检查：把清理改成"不取消、不等启动线程"，4 项取消测试都失败。
  - 新增的生命周期测试单独重复跑 5 次，均通过。
- 文档：README 最终状态与 §2（新增 4 行）；§4 7B 方案改为事先固定的工程判据（启动、KV cache 至少容纳 1 条、无报错、logprob 有限、两种长度设置下贪心输出逐 token 一致）。满足判据的 40,960 也只记为"工程上可运行"，仍是候选，不标为长上下文质量已验证。资源计时按开机请求到确认关机的挂钟时间计，第 7A 步两次开机共约 4 分 54 秒（约 0.082 GPU·h，按工具时间戳，没有和账单核对）。
- 全量 `pytest tests/unit tests/remote -q`：820 passed，退出码 0，没有残留子进程；冻结文件哈希不变。
- 没有调用 LLM，没有启动 GPU，没有新增付费请求，没有进入 7B 或 S1。

## 第 7A-R3 步证据（2026-10-31，离线）

- 问题：
  - 第一次取消如果发生在正常清理期间，会被 `_reclaim` 吞掉，`run()` 照常返回 `AgentLoopOutput`，样本带着 reward；
  - 回收失败或清理超时只记日志，样本照常返回。
- 改动（只改 `itsm_agent_loop.py` 的清理与传播部分，启动方案不动）：
  - `_reclaim` 现在返回"清理期间是否收到过取消"和"失败原因"。清理期间第一次收到取消时，会立即调用 `env.cancel()` 停掉子进程，但清理照样做完。
  - `run()` 在清理结束后按顺序决定传播什么，取第一个成立的：
    1. 清理失败 → `ItsmInfrastructureError(stage="cleanup", kind="not_reclaimed")`，`__cause__` 是原始异常或取消；`detail` 和 `.cleanup` 里有 episode id、pid、失败原因、`cancel_requested`、原始错误、worker 的失败记录；
    2. 收到过取消 → `CancelledError`；
    3. episode 本身报错 → 原样抛出；
    4. 以上都没有 → 返回输出。
  - 新增设置 `cleanup_margin_s`（默认 5 秒，与原来的常数相同）和 `cleanup_bound_s()`，这样测试可以用较短的上限。
- 新增测试 5 项：
  - 正常结束后，清理期间第一次取消；
  - 回收结果改报为 False，分正常结束、episode 中途取消、原始异常三种来源；
  - 清理超时：到达上限就报错，不会等挂起结束才报。
  - 这些路径都没有输出，也就没有 reward。
  - 变异检查：把传播改回 7A-R2 的写法，这 5 项都失败。
- 原有测试：启动取消、清理期间再取消一次、后台启动返回就绪 worker、启动失败、慢工具取消、超时、完整 episode 都照常通过。生命周期和清理相关的 11 项单独重复跑 5 次，均通过。
- 7B 判据：工程判据减为 3 条（启动并能容纳 1 条该长度的序列、不报错、logprob 有限）。跨设置贪心输出一致性降为诊断项，记录第一个分歧位置、logprob 差和分歧位置的候选概率。b、c 两档平均 gold logprob 的差值不单独用来判断长上下文质量。40,960 仍是候选。
- 全量 `pytest tests/unit tests/remote -q`：**825 passed**，退出码 0（`output/itsm-step7a/pytest_full_r3.log`），没有残留子进程；两个冻结文件的 SHA256 不变。
- 没有调用 LLM，没有启动 GPU，没有新增付费请求，没有进入 7B 或 S1。

## 第 7B 步证据（2026-11-01）

- 报告：`docs/itsm_step7b/README.md`；轨迹：`docs/itsm_step7b/trajectories.md`；汇总：`docs/itsm_step7b/summary.json`；原始产物：`docs/itsm_step7b/artifacts/`（远端运行 `exp-itsm/2026-11-01T04:44:15.822424Z`，下载那次是 `exp-itsm/2026-11-01T04:03:58.749199Z`）。
- 代码（提交 `789d097` 起）：`vllm_token_server.py`（按 verl 方式构造引擎、复刻 `generate`）、`itsm_scripted_user.py`（persona 脚本化用户和 LLM 替身）、`build_7b_length_inputs.py`、`model_files_7b.py`、`run_7b.sh`、`run_7b_gpu.py`、`analyze_7b.py`；测试 `tests/unit/test_itsm_step7b.py`（8 项）。没有修改 loop、环境、评分、冻结数据或预算数值。
- 固定版本：Qwen3-8B @ b968826d，12 个文件按 HF 的 SHA256 或 git blob SHA1 核对通过，chat template 哈希一致；长度输入在远端用模型自带的分词器复建，与提交的 meta 逐 token 一致；远端代码与本地提交一致（319 个文件）。
- 资源：开机 2 次，约 3,517 秒 ≈ 0.98 GPU·h（上限 3），两次结束时都已确认关机。下载约 36 分钟（hf-mirror）；第 1 次开机在 3 次启动都失败后，由旧的自动关机定时器按设计关机，当时没有运行中的任务（2026-11-01 更正：原写"被 projtool 自动关机误关"，§7.3）。
- 结果：见报告的"结论"一节。
- 发现：Bench2 `slice_db_to_orgs` 的顺序依赖哈希种子（本步固定为 `PYTHONHASHSEED=0`，没有修改环境）；磁盘实际 128 GB；旧关机定时器在人工排查期间到期（不是误伤运行中的任务，2026-11-01 更正）；运行时读 `docs/` 而 `docs/` 不同步（手动上传了两个已提交的文件，哈希一致）；4 条首动作都是写调用，只有 msp_013 成功、另外 3 条返回业务错误（2026-11-01 更正：原写"策略编造 ID 也能写入成功"）。
- 没有更新权重，没有运行 held-out，没有生成 SFT/Teacher 轨迹，没有调用外部 LLM，没有新增付费请求，没有进入第 8 步或 S1。

## 第 7B 步收尾与第 8 步提案证据（2026-11-01，离线）

- **行为结论更正**：根据 `episodes.json` 中的观察核实，4 条首动作都是写调用：msp_013 的 `link_knowledge_to_incident` 成功；887f5580、msp_019、single_008 的 `update_incident` 都返回 `domain_error`。`docs/itsm_step7b/README.md` §4、§7.6 和结论一节、`trajectories.md` 开头的说明、上面 7B 证据中的相应说法都已更正，并标明更正日期。
- **同时发现并更正 7B 的另外两处错误**：
  - 第 3 次运行不是"下载 30 秒没有进展"，而是 Xet 的 CAS 返回 HTTP 401，运行以退出码 1 结束；
  - 03:59:01 的关机是旧定时器到期，当时已经没有运行中的任务。projtool 的检查（有 `proj-*` tmux 会话就跳过）没有失效。
  - 原先说的"projtool 自动关机误伤正在运行的任务"不成立，相关段落已更正。
- **`PYTHONHASHSEED=0`**：
  - 写入协议 §13.1；
  - 在 Python 启动之前设置：启动器 `scripts/itsm/itsm_python.sh` 和 `run_7b.sh` 都导出这个变量；
  - `run_7b_gpu.py` 没有固定种子时直接报错；`build_7b_length_inputs.py` 没有固定种子时会先重新启动自己；
  - `ItsmEnvWorker` 子进程在就绪消息中报告自己的哈希标志；`ItsmAgentLoop` 默认拒绝没有固定种子的子进程（在启动阶段失败，此时还没有采样）。
  - 测试 `tests/unit/test_itsm_hash_seed.py`：
    - 调用方环境里没有这个变量，经启动器新起两个独立进程：子进程都报告 `hash_randomization=0`，msp_013 的 `list_incidents`、`list_users` 观察逐字节一致；
    - 反例：种子 1 和种子 2 的进程，列表顺序不同；
    - 没有固定种子的子进程被拒绝。
  - 范围说明：这只固定列表观察的顺序，不代表模型输出是确定的。
- **运行时依赖**：
  - 清单 `configs/itsm/runtime_dependencies.json` 列出 `docs/itsm_stepS0/usage_audit.json`，路径不变，SHA256 与冻结的 `itsm_usage_v1.json` 中记录的一致；
  - 校验模块 `agent_rl.data.itsm_runtime_deps`：`_s0_audit` 在读取前核对哈希；`itsm_task_sets._usage` 加载用途登记前核对全部依赖；
  - 工具 `scripts/itsm/runtime_deps.py`：
    - `push`（ssh 或本地目录）：推送后在目的地逐个核对哈希；本地文件被改过时拒绝推送；
    - `preflight`：核对依赖、做用途校验和取题；`run_7b.sh` 已加入这一阶段；
    - `simulate-remote`：构建代码同步之后远端会有的文件集合。
  - 测试 `tests/unit/test_itsm_runtime_deps.py`：干净部署目录在推送前失败、推送后成功、文件被改动或删除时失败；rollout 在依赖有问题时，于建 env 和采样之前失败。
  - `candidate_screening.json` 经核实只被离线脚本读取，不是运行时依赖。
  - 没有删除任何校验，也没有手动补文件。
  - 没有在真实远端运行 ssh 模式的 `push`（本轮不开机）。
- **projtool 遗留的 5 条记录**：
  - 5 条都有退出哨兵（退出码均为 1），tmux 会话都已不存在；
  - `finalize_runs` 需要 SSH，实例关机时无法关闭任何一条；`finalize_run` 的离线路径只适用于已释放的绑定，而释放会删掉数据盘；
  - 因此没有修改内部状态文件。方案是下次开机后先运行 `finalize_runs`。
  - 附带说明：这次 `finalize_runs` 调用在本地 docs 仓库提交了一行 journal（`5c826de`），记录的是 2026-10-30 一条早已结束的运行。
  - 旧定时器的修复建议（代次令牌和有时限的租约）见 `docs/itsm_step8/proposal.md` §9。
- **第 8 步提案**：`docs/itsm_step8/proposal.md`（A：dev 13 × 4；B：GRPO 可训练 26 × G = 4；温度 1.0；`max_model_len` 40,960，并统计 32,768 下的截断；先做最小复核，≤ 20 次请求、≤ 1 美元；A + B 上限 12 美元；GPU 上限 2 GPU·h；说明了单进程适配器与 verl 链路的边界）。
- **回归**：`pytest tests/unit tests/remote -q` 结果为 **841 passed**，退出码 0（`output/itsm-step7b/local/pytest_full_8prep.log`），新增 8 项；冻结文件哈希不变。
- 没有调用 LLM，没有启动 GPU，没有生成 SFT 轨迹，没有进入第 8 步执行或 S1。

## 第 8 步离线实现证据（2026-11-01）

- 方案与预留计算：`docs/itsm_step8/proposal.md`；参数 `configs/itsm/step8/plan.json`；价格 `configs/itsm/step8/prices.json`；复核输入 `configs/itsm/step8/review_inputs.json`（手写的固定文本对和 NL 样例）。
- 代码：`spend_ledger.py`（组级预留、追加、关闭和恢复）、`budget_manager.py`（单一管理者和 IPC）、`llm/deepseek_chat.py`、`envs/itsm_ledgered_user.py`、`eval/itsm_aux_eval.py`、`eval/itsm_length_counterfactual.py`、`rollout/itsm_diagnostic_runner.py`、`rollout/itsm_step8.py`；`ItsmEnv.auxiliary_inputs`、`ItsmEnvWorker.aux_evaluate`、`ItsmAgentLoopSettings.auxiliary_evaluation`（默认关闭）。脚本：`run_8.sh`、`run_8_diagnostics.py`、`summarize_8.py`、`review_8_api.py`。
- D5 改为 `max_tokens=1024`：`DEFAULT_USER_LLM_ARGS`、`budget_v1.yaml` 的 `user_llm_max_tokens` 和协议 D5 已同步。D5/D7 的付费请求改走直连客户端，请求体与 litellm 1.82.6 一致，已写入协议 §14.3。
- 新增测试（41 项）：`test_itsm_step8_budget.py`（17）、`test_itsm_step8_eval.py`（12）、`test_itsm_step8_runner.py`（8，真实 worker 子进程 + 本机 fake DeepSeek 服务端 + 采样替身）、`test_itsm_step8_plan.py`（4）。
- 回归：`pytest tests/unit tests/remote -q`：**882 passed**，退出码 0（`output/itsm-step7b/local/pytest_full_8impl2.log`），没有残留子进程；冻结文件哈希不变。上一次全量运行中，litellm 等价测试因为别的测试先把 litellm 换成了替身而失败（测试隔离问题，产品代码不受影响）；修正测试后重跑通过。
- 没有调用 LLM，没有启动 GPU，没有发出付费请求，没有修改 Bench2、policy、reward、冻结数据或用途范围，没有进入 S1。

## 第 8 步离线实现 R1 证据（2026-11-01）

- 说明：`docs/itsm_step8/proposal.md` §11。
- 复现与修复：先写 `tests/unit/test_itsm_step8_r1.py`（16 项），修补前失败 15 项（`output/itsm-step7b/local/r1_before_fix.log`：复核失败后继续发请求；辅助评测超时导致重采样 4 次；预算拒绝或评测器异常把样本判为 budget 或 error；排队计数错误导致下一组永远开不了；模型不符被重试，halt 不存在）。修复后 16 项全部通过（`r1_after_fix.log`）。修复后又改了两处测试写法：评测器工厂改用 partial，避免被当成绑定方法；并发 halt 场景改为第一个请求就模型不符，消除时序依赖。
- 代码：
  - `review_8_api.py`：`ReviewStop`、`exit_code`；
  - `itsm_agent_loop.py`：辅助评测失败记为 `not_evaluated`，新增 `classify_aux_failure`；
  - `itsm_diagnostic_runner.py`：修复排队计数，处理 halt 状态，辅助评测的预算问题只停止开新组；
  - `spend_ledger.py`：`halt` 事件、`RunHalted`，实际费用超过预留时自动 halt；
  - `budget_manager.py`：`halt`、`halt_reason`，子进程可调用 `halt`；
  - `deepseek_chat.py`：模型不符时先结算，再让管理者 halt；
  - `itsm_aux_eval.py`：模型不符不重试；
  - `itsm_step8.py`：`hard_cap_table`、`max_system_prompt_bytes`，以及 `settings_overrides`。
- 完整组硬上限（按规则计数：每次尝试最多 40 次模拟器调用、每条样本最多 4 次尝试、每条有效样本最多 8 次裁判调用、每组 4 条样本）：
  - 现有规则下 A 组 7,374 美元、B 组 6,045 美元；
  - 按假设的请求大小规则（64/64/256 KiB，目前没有实施）A 组 20.82 美元、B 组 13.38 美元；
  - 两种情况下 12 美元都放不下一个完整组。三个方案见 proposal §11，没有自动加预算，也没有改规则。
- 回归：`pytest tests/unit tests/remote -q`：**899 passed**，退出码 0（`output/itsm-step7b/local/pytest_full_8r1.log`），没有残留子进程；冻结文件哈希不变。
- 没有调用 LLM，没有开机，没有发出付费请求，没有进入 S1；小额 API 复核尚未获批。

## 第 8 步小额 API 复核：执行前状态（2026-11-01）

- 费用方案已定为①（proposal §12）：12 美元上限，组级计划预留加逐请求预留，接受预算耗尽时有未完成的组，不加请求大小限制，不改重试次数。
- 复核**没有执行**：云端没有可用的 CPU 环境（AutoDL 开机 API 不支持无卡模式，`exp-itsm` 处于关机状态）。按第 6 条停下报告，没有开机，没有发出请求。执行方式待你选择（proposal §12）。
- 离线完成：
  - 汇总中顶层 `not_evaluated` 漏计的修复：新测试修复前失败、修复后通过；
  - 复核报告字段：HTTP 请求数、逐请求记录（含 usage 和思考相关字段）、逐项检查、账本事件、未验证项，由新测试固定；
  - `DeepSeekChat` 的请求记录增加 usage。
- 回归：`pytest tests/unit tests/remote -q`：**901 passed**，退出码 0（`output/itsm-step7b/local/pytest_full_8rev.log`），没有残留子进程。

