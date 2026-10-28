# ITSM 迁移进度

规则来源：`docs/itsm_migration_agent_playbook.md`。“测试已通过”与“用户已批准”分开记录；用户批准一栏只记录用户原话，不由 agent 填写结论。

## 步骤状态

| 步骤 | 状态 | 用户批准记录 |
|---|---|---|
| 0：云端只读审计 | 用户认可 | 2026-10-28 用户原话：“我目前认为可以验收了，你可以额外检查一遍，没问题就执行第一步”。agent 补查第 0 步无问题后进入第 1 步。 |
| 1：独立环境与现有测试基线 | 用户认可；GPU 训练环境延后（见阻塞） | 2026-10-28 用户原话：“未提交文件是什么，你认为是否需要接受？没问题就接受，开始第二步。实例等gpu充足的时候再创建”。 |
| 2：接口与奖励契约 | 用户认可；D1–D6 已定，D5 改为 `deepseek/deepseek-v4-flash` | 2026-10-28 用户原话：“认可，除了D5我认为可以用deepseek-flash。至于风险，到了那一步再说。没问题就第三步”。 |
| 3：离线环境与评分回归 | 进行中 | 未批准 |
| 4 及以后 | 未开始 | — |

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
| 未安装 | vLLM、CUDA 版 torch（属于 GPU 训练环境，见阻塞） |

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

## 阻塞

1. **GPU 训练环境未建立。** 2026-10-28 AutoDL 的 pro6000-p 无库存。改为扩大候选重试后，平台分配到一台 H800（`pro-7931feabab3c`）。用户随即表示暂不租实例，该实例已 `release`（`acted: true`），列表核实已消失。该实例存在了几分钟，可能产生了少量计费。因此 CUDA、驱动、vLLM 0.24.0 与 GPU 的兼容性都未验证。
2. 在 GPU 实例上，tests/remote 中依赖 GPU 的路径尚未执行。本机 113 项是在 CPU 版 torch 下通过的。

## 第 2 步证据

- 产物：`docs/itsm_protocol.md`。只有协议设计，没有实现 adapter，也没有修改代码。
- 依据：静态阅读 Bench2 的 `gym_agent`、`orchestrator`、`environment`、`toolkit`、`evaluator*`、`text_match_strategy`、`clock`、`run`、`user_simulator`，以及 AirLineRL 的 `tau_env`、`tau_agent_loop`、`schemas`、`process_reward`、`credit_encoding`、`verl_algorithms`。
- 实测：对 `itsmbench_single_017` 做离线、只读的真实工具探测，包括正常调用、`ItsmError`、未知工具、错误参数名四种情况，没有调用 LLM。结果记在协议 §11.0。
- 未验证：协议中所有 "须验证" 项，分别留给第 3、4、5、6、7、9 步。
