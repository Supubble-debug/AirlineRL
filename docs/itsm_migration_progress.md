# ITSM 迁移进度

规则来源：`docs/itsm_migration_agent_playbook.md`。“测试已通过”与“用户已批准”分开记录；用户批准一栏只记录用户原话，不由 agent 填写结论。

## 步骤状态

| 步骤 | 状态 | 用户批准记录 |
|---|---|---|
| 0：云端只读审计 | 用户认可 | 2026-10-28 用户原话：“我目前认为可以验收了，你可以额外检查一遍，没问题就执行第一步”。agent 补查第 0 步无问题后进入第 1 步。 |
| 1：独立环境与现有测试基线 | 用户认可；GPU 训练环境于 2026-10-29 补建完成（见“第 1 步补充：GPU 训练环境”），补充部分待验收 | 2026-10-28 用户原话：“未提交文件是什么，你认为是否需要接受？没问题就接受，开始第二步。实例等gpu充足的时候再创建”。 |
| 2：接口与奖励契约 | 用户认可；D1–D6 已定，D5 改为 `deepseek/deepseek-v4-flash` | 2026-10-28 用户原话：“认可，除了D5我认为可以用deepseek-flash。至于风险，到了那一步再说。没问题就第三步”。 |
| 3：离线环境与评分回归 | 用户认可（第 1 次修订后）；D7、D8 已定 | 2026-10-28 用户要求修订（该消息不是验收通过）。2026-10-29 用户原话：“根据本次修订报告，第 3 步验收通过。请记录以下决定：D8：保持 Bench2 现有自由文本比较规则……D7：评测时 DB 文本裁判与 NL 裁判均采用 deepseek/deepseek-v4-pro，温度 0……完成上述文档收尾后，仅执行第 4 步”。决定全文写入协议 §12、§14.2–§14.4。 |
| 4：最小单 episode adapter | **待验收** | 未批准 |
| 5 及以后 | 未开始 | — |

## 运行环境

工作区：`/home/sar/work/EnterGym/code/main`（WSL2，用户指定的工作区；训练在 AutoDL 上运行）。

| 项目 | 值 |
|---|---|
| 代码版本 | `main` @ `db9ad11`；`tau2-bench` @ `1901a30`；`verl` @ `0a30f52`（均为固定提交） |
| 工作区改动 | `.gitignore`（projtool 脚手架段，非业务改动）；本文件 |
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
3. **GitHub 推送凭据未配置**（2026-10-28 两次推送均失败）。本地 `main` 领先 `origin/main` 3 个提交。

## 第 2 步证据

- 产物：`docs/itsm_protocol.md`。只有协议设计，没有实现 adapter，也没有修改代码。
- 依据：静态阅读 Bench2 的 `gym_agent`、`orchestrator`、`environment`、`toolkit`、`evaluator*`、`text_match_strategy`、`clock`、`run`、`user_simulator`，以及 AirLineRL 的 `tau_env`、`tau_agent_loop`、`schemas`、`process_reward`、`credit_encoding`、`verl_algorithms`。
- 实测：对 `itsmbench_single_017` 做离线、只读的真实工具探测，包括正常调用、`ItsmError`、未知工具、错误参数名四种情况，没有调用 LLM。结果记在协议 §11.0。
- 未验证：协议中所有 "须验证" 项，分别留给第 3、4、5、6、7、9 步。

## 第 3 步证据（第 1 次修订后）

- 改动（全部未提交，等待验收）：
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
  - 调用 ID（§5）：`{episode_id}/c{seq:03d}`；每个调用恰好对应一条 ToolMessage；ID 唯一，且与记录一致；模型自带的 ID 存为 `model_call_id`。
  - 错误分类（§8）：`domain_error`（保留 `code` 和 `field`）、`unknown_tool`、`bad_arguments`（在调用前用签名绑定检查，观察中不含 Python 类名，且该调用不会执行）、`tool_exception_other`。出错后 episode 继续，可以恢复。
  - 用户模拟器异常、空回复、首句就 STOP，按基础设施错误处理，不产生 reward。
  - 时钟（§6）：每一步都在独立线程中执行，msp_021（对时钟敏感）仍得到 DB=1，且执行线程的时钟在结束后恢复。
  - 奖励（§9）：`done` 或用户 STOP 时只评一次 DB；`nl_status=not_evaluated`，`joint_reward=None`，`training_reward_source=db_only`。走错路径时得到有效的 DB=0。
  - 与 Bench2 `Orchestrator.run` 对照：MSP-021 和 Single-017 各一条脚本（含业务错误和未知工具调用），消息序列、用户收到的消息、最终 DB 哈希完全相同。`bad_arguments` 的文本有意与 Bench2 不同，因此不在对照范围内。
- 不在本步范围（第 5 步）：`max_steps` 与 token 截断、超时、取消、重试调度、进程隔离、并行隔离，以及评分失败后的重新评分接口。

