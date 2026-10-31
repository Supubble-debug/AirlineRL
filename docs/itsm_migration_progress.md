# ITSM 迁移进度

规则来源：`docs/itsm_migration_agent_playbook.md`。“测试已通过”与“用户已批准”分开记录；用户批准一栏只记录用户原话，不由 agent 填写结论。

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
| 6C：数据与预算整合 | **待验收**（测试通过 ≠ 验收） | 未批准。2026-10-30 用户原话：“现在仅执行第 6C 步：数据与预算整合。……完成后提交v2、测量结果与验证证据，等待验收，不进入第7步。” 2026-10-30 用户原话：“第6C步验收结果保留，但暂不开始完整第7步。”（原话未写“验收通过”；6C README §3.3 的预算选项 A–D 也未明确选择，状态按原话记录，不由 agent 推断。） |
| 6D-1：EnterpriseOps-Gym 下一批候选筛选与规模方案 | **待验收**（测试通过 ≠ 验收） | 未批准。2026-10-30 用户原话：“新增第6D步：继续扩充正式实验数据。本轮先只做6D-1候选筛选，不直接迁移全部任务。……提交候选清单和规模方案后停止，等待验收。不自动改政策，不放行已隔离题，不启动正式评测或训练。”同条消息允许无法原生迁移的题作为“Bench2 改编任务候选”处理（新标识、记录每项改动原因、保留原题与验证器溯源、不宣称等价），并说明“具体怎么做交给你把关”。 |
| 7 及以后 | 未开始 | — |

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

