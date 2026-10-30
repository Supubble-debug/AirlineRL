# ITSM episode 协议与奖励契约（第 2 步设计稿）

状态：2026-10-28 用户验收通过（决策见 §14）。本文只定义协议，不包含实现；文中的类名和字段名是后续步骤（3–9）的实现目标。

依据：
- 代码版本 `main` @ `3ea815a`，`Bench2/` 导入自 Enterprise-Worlds `37175af`。
- 第 2 步对源码的静态阅读，以及一次离线工具探测（真实 ITSM 工具，不调用 LLM，见 §11.0）。
- 文中所有 "Bench2 现状" 的结论都给出了源码位置；凡是写着 "须验证" 的，由对应步骤用测试证明。

## 0. 设计总原则

1. **一个 `ItsmEnv` 实例只对应一个 episode。** `reset()` 只能调用一次，开下一个 episode 必须新建实例。不提供 "复用实例再 reset" 的路径，从结构上排除旧 episode 残留。
2. **不使用 Bench2 的后台线程 gym。** `AgentGymEnv` 在后台线程里跑 `Orchestrator`，靠 `Event` 阻塞同步，会带来三个问题：
   - `close`/`reset` 只 `join(timeout=1.0)`（`gym_agent.py:240,273`），不能证明线程已经退出；
   - 线程局部时钟没有设置（见 §6）；
   - 异常被吞成 `run_result=None`，最终变成 reward=0（`gym_agent.py:223-227,277-278`）。

   ITSM adapter 改为**由调用方驱动的单线程状态机**：每次 `step()` 同步完成 "执行工具" 或 "调用用户模拟器"，内部不起任何线程。语义上逐条对齐 `Orchestrator.run()`，第 4 步需要用对照测试证明两者一致。
3. **复用 Bench2 的真实组件：** `get_environment`、`ItsmTools`、`Environment.make_tool_call`、`UserSimulator`、`compare_dbs`、`judge_free_text`、`evaluate_nl_assertions`、`set_now`/`reset_now`。只重写编排和错误分类，不改业务工具、数据和评分逻辑。
4. **基础设施故障不进入奖励。** 模型行为错误留在轨迹中参与训练；基础设施故障抛出 `ItsmInfrastructureError`，该 episode 不产生 reward。

## 1. 需由协议修正的 Bench2 现状

| # | 现状 | 位置 | 协议处理 |
|---|---|---|---|
| F1 | 所有工具调用的 ID 固定为 `"gym_call"`；JSON 动作里自带的 `id` 被丢弃 | `gym_agent.py:66-67,96-97,104-105` | 由 adapter 生成全局唯一 ID（§5） |
| F2 | 每步观察都是 `render_trajectory` 渲染的整段历史 | `gym_agent.py:245,261` | 改为增量观察（§4） |
| F3 | `truncated` 恒为 False；`max_steps` 截断时不区分原因 | `gym_agent.py:265`；`orchestrator.py` 主循环 | 显式给出终止原因（§7） |
| F4 | gym 路径不调用 `set_now`，时钟停在 `2024-06-01 00:00:00`；例如 `single_017` 的任务时间是 `09:00:00` | `clock.py`（thread-local）；只有 `run.py:145` 调用了 `set_now` | 每次执行工具或评分时，在执行线程内设置时钟（§6） |
| F5 | gym 默认 `skip_nl_assertions=True` | `gym_agent.py:281` | NL 状态显式记为 `not_evaluated`，不当作通过（§9） |
| F6 | 文本比较默认 `llm`；没配裁判模型时只打一条 warning，退回 `fuzzy` | `text_match_strategy.py` `effective_strategy()` | 请求策略与实际策略不一致时直接报错（§9） |
| F7 | 两个裁判的输出无法解析时都判为 "不通过" | `evaluator_nl._parse_judge_response`；`text_match_strategy._parse_judge_response` | 识别为裁判故障，归入基础设施（§8） |
| F8 | gold action 回放失败只记 warning，评分照常继续 | `evaluator_env._build_gold_env` | adapter 自行回放，失败即报错（§9） |
| F9 | 工具异常一律变成 `"Error: ..."`，`ItsmError.code/field` 丢失；参数名错误时还会泄露 Python 类名 | `environment.Environment.get_response` | adapter 先校验参数签名再分类（§8） |
| F10 | `litellm.completion` 没有设置超时和重试 | `utils/llm_utils.generate` | 由 adapter 传入超时参数，失败归入基础设施（§8） |
| F11 | AirLineRL 在采样服务没返回 logprob 时填 0.0 | `tau_agent_loop.py:299-303` | ITSM loop 禁止零填充（§10） |
| F12 | AirLineRL 上下文轮换会清空此前的 token、mask、logprob 和 span | `tau_agent_loop.py:486-502` | 首版禁用轮换（§10，决策 D3） |
| F13 | AirLineRL 基础设施重试用尽后直接抛异常，没有分组处理策略 | `tau_agent_loop.py:146-160` | 固定分组失败策略（§12） |

## 2. 接口

```python
@dataclass(frozen=True)
class ItsmEnvConfig:
    task_id: str
    max_steps: int                     # 与 Orchestrator 的计数口径相同（§7）；预算在第 6 步测定
    user_llm: str                      # 决策 D5：deepseek/deepseek-v4-flash
    user_llm_args: dict
    user_llm_timeout_s: float
    text_match: TextMatchConfig        # 显式指定，不使用默认值（§9）
    nl_mode: Literal["skip", "evaluate"]
    nl_llm: str | None
    nl_llm_args: dict | None
    expose_current_time: bool = False  # 决策 D4

class ItsmEnv:
    def reset(self) -> ItsmReset: ...                  # 只能调用一次
    def step(self, action: ItsmAction) -> ItsmTransition: ...
    def truncate(self, reason: str) -> ItsmTransition: ...  # 由外部预算（token）触发截断
    def close(self) -> None: ...                       # 幂等

@dataclass(frozen=True)
class ItsmReset:
    observation: str            # 固定问候语 + 用户首句，见 §4
    public_info: dict           # 只包含 §3 中的公开字段
    episode_id: str

@dataclass(frozen=True)
class ItsmTransition:
    observation: str            # 增量观察，只含本步新增内容
    terminated: bool
    truncated: bool
    termination_reason: str | None   # §7 枚举
    tool_record: ToolCallRecord | None   # 复用 agent_rl.data.schemas.ToolCallRecord
    reward_info: ItsmRewardInfo | None   # 仅在 episode 结束时非空（§9）
    step_count: int
```

动作 `ItsmAction` 只有三种，每轮只能选一种。这与 AirLineRL 的 `to_tau_action` 和 Bench2 `LLMAgent` 的 "一轮只做一件事" 一致：

| 种类 | 来源 | 说明 |
|---|---|---|
| `message(text)` | 模型输出纯文本 | 发送给用户 |
| `tool_call(name, arguments, model_call_id)` | 模型输出**恰好一个**工具调用 | 首版不支持并行调用；出现多个调用视为非法动作 |
| `done()` | 模型调用 `done` 工具 | 结束 episode |

把模型输出解析成上述动作的工作由 rollout loop 完成（沿用 `to_tau_action` 的规则）。解析失败时，loop 调用 `env.truncate("invalid_action")`，见决策 D2。

## 3. 公开字段与私有字段

| 字段 | 可见性 | 说明 |
|---|---|---|
| `policy.md`（约 32 KB） | 公开 | 进入系统提示 |
| 95 个 ITSM 工具 schema + `done` | 公开 | 由 `env.get_tool_schemas()` 生成，加上 `DONE_TOOL_SCHEMA` |
| 固定问候语、用户在对话中说的话 | 公开 | 通过观察获得 |
| 工具返回内容（含错误信息） | 公开 | 通过观察获得 |
| `task_id` | 仅日志与分组 | 不进入模型输入 |
| `scenario.task_description` / `simulator_guidance` | **私有** | 只给用户模拟器 |
| `persona.known_info` / `identity` / `personality` | **私有** | 只给用户模拟器；`identity.user_id` 由 env 内部用作调用者身份 |
| `evaluation_criteria.actions`（gold） | **私有** | 只用于评分，以及第 3 步明确标注的离线回放 |
| `evaluation_criteria.nl_assertions` | **私有** | 只用于评分 |
| `org_ids` / `seed_db` / `initial_state_delta` / `tenancy` | **私有** | 只用于构造 env |
| `current_time` | 私有（默认） | Bench2 自带的 `LLMAgent` 不向模型展示时间；是否公开见决策 D4 |

**泄漏检查按信息来源判定（第 4 步补修修订）：** 判据是"每段模型输入从哪里来"，不是"某个字符串有没有出现"。
- 允许进入模型输入的来源只有：
  - 公开 policy 和工具 schema（53 题完全相同，不随任务变化）；
  - 固定问候语；
  - 用户模拟器产生的消息；
  - 真实工具执行返回的结果和错误信息；
  - 策略自己生成的内容；
  - 由 adapter 生成、不含任务信息的公开调用 ID（§5）。
- 禁止由 adapter 或 rollout loop 直接放进模型输入的私有元数据：`task_id`、`episode_id`、组 ID、`sample_index`、`seed_db`、`org_ids`、`tenancy`、`initial_state_delta`、gold actions、NL 断言、任务描述、模拟器指引、`known_info`、`current_time`（D4）。
- 同一个值经由合法来源出现不算泄漏，例如：用户说出单号，工具结果里出现组名，`known_info` 的枚举值恰好写在 policy 或工具 schema 里。所以不能用"`known_info` 的值在用户说出前不得出现"这种子串规则来判定：它会把 schema 里的通用值误判为泄漏，又查不出经由 ID 或格式字段夹带的元数据。
- 验证方式：
  - 公开信息在全部任务间相同（第 4 步已测）；
  - 调用方故意传入含 `task_id` 的 `episode_id` 时，`task_id`、`episode_id`、`seed_db` 不出现在任何观察、公开信息或轨迹消息中（第 4 步已测）；
  - 第 7 步对实际渲染出的 prompt 逐段标注来源，并重复上述元数据检查。

## 4. 观察格式（增量）

每次 `step()` 只返回本步新增的内容：

| 本步动作 | 返回的 observation |
|---|---|
| `tool_call` | `tool[<call_id>] ok: <结果>` 或 `tool[<call_id>] error(<error_class>): <信息>` |
| `message` | `user: <用户回复>` |
| `done` / 截断 | `""`（已终止） |

`reset()` 的观察是两行：固定问候语 `assistant: Hi! How can I help you today?`（`orchestrator.DEFAULT_FIRST_AGENT_MESSAGE`）和用户首句。问候语不是策略采样产生的，它进入 prompt 时 mask=0（§10）。

轨迹的完整渲染（给 NL 裁判用）由 env 内部维护的 `list[Message]` 生成，与 Bench2 的 `render_trajectory` 格式一致，不依赖模型侧观察的拼接方式。

**验收（第 4 步）：** 把各步观察拼接起来，应当等于完整轨迹去掉 assistant 部分的结果；同一内容不得重复追加。

## 5. 工具调用 ID

- `episode_id` 只用于日志、分组和追踪，例如 `"{task_id}:{group_uid}:{sample_index}:{uuid8}"`，**不得出现在模型输入中**。
- 模型可见的调用 ID 是 `call_id = "call_{seq:03d}"`，`seq` 在 episode 内从 1 递增。它不由 `episode_id` 或 `task_id` 派生，所以调用方传入的 `episode_id` 无论包含什么，都不会经由调用 ID 泄漏（第 4 步补修：原先的 `{episode_id}/c{seq}` 会把 `task_id` 带进工具观察）。
- `call_id` 只在 episode 内唯一；跨 episode 的全局主键是 `(episode_id, call_id)`，由导出的轨迹提供。
- 模型自带的 ID（例如 hermes 解析得到的 ID）另存为 `model_call_id`，只用于对照，不作主键。
- 一一对应规则：
  - 每个 `tool_call` 动作恰好生成一个 `ToolCallRecord(call_id=...)`，最多对应一个 `ToolMessage(id=call_id)`。
  - `result_received=False` 只在两种情况下出现：§7 的 "截断时有待执行调用"，以及 §8 的 `tool_internal` 基础设施故障（该 episode 作废，不产生 reward）。
  - 不存在没有对应调用的结果，也不存在重复 ID。

## 6. 任务时钟

- 执行工具、gold 回放、构造 baseline、比较 DB，**都在同一个上下文管理器里运行**：

  ```python
  with task_clock(task.current_time):   # set_now；退出时恢复调用前的值（不是 DEFAULT_NOW）
      ...
  ```

- 必须在**实际执行的线程内**进入该上下文。rollout loop 用 `run_in_executor` 调用 `env.step`，线程池每次可能分配不同的线程。因此每次 `step()` 和每次评分都要重新进入，不能只在 `reset()` 里设一次。
- 进入后断言 `get_now() == canonical(task.current_time)`。退出后，该线程的时钟恢复为进入前的值，不会泄漏到下一个任务。
- 所有 53 题都带有 `current_time`（共 23 个不同值）。缺失时报 `TaskDefinitionError`，不回退到 `DEFAULT_NOW`。
- **验收（第 3 / 5 步）：**
  - 对新建的 incident 或 SLA，`created_on` 等于任务时间；
  - 两个时间不同的 episode 并发运行时，各自写入的时间戳互不串扰。

## 7. 步数、终止与截断

`step_count` 的口径与 `Orchestrator.run()` 一致：每次 "用户回复" 和每次 "工具执行" 各计 1 步，`reset()` 产生的用户首句算第 1 步。

| 情况 | 行为 | terminated | truncated | termination_reason |
|---|---|---|---|---|
| 模型调用 `done()` | 结束，进入评分 | True | False | `agent_done` |
| 用户模拟器输出 `###STOP###` | 结束，进入评分 | True | False | `user_stop` |
| 收到 `tool_call` 时 `step_count >= max_steps` | **不执行**该调用，记 `result_received=False`，进入评分 | False | True | `max_steps` |
| 收到 `message` 时 `step_count >= max_steps` | 不再调用用户模拟器，进入评分 | False | True | `max_steps` |
| loop 发现剩余 token 预算不足 → `truncate("response_length")` | 进入评分 | False | True | `response_length` |
| 模型输出无法解析 → `truncate("invalid_action")` | 进入评分，outcome 规则见决策 D2 | True | False | `invalid_action` |
| 基础设施故障或超时 | 抛出 `ItsmInfrastructureError`，**不评分** | — | — | （记在异常的 `stage` 字段） |
| `reset()` 时用户首句就是 STOP | 用户模拟器故障，按基础设施处理 | — | — | `stage=user_sim_empty_episode` |

前两种 "不执行或不调用" 的行为，与 `Orchestrator` 在 `steps >= max_steps` 时跳出循环的效果相同，第 4 步要做对照测试。

`max_steps` 的取值不在本步决定。参考值：Bench2 `run_task` 默认 12，gym 默认 100，AirLineRL airline 是 64；gold action 数为 2–16，中位数 8。具体预算由第 6 步测定。

## 8. 错误分类

**模型侧错误**：写入轨迹，episode 继续；作为过程证据，可供 E3 使用。

| error_class | 判定方式（adapter 在调用 `make_tool_call` 之前和之中判断） | 返回给模型的观察 |
|---|---|---|
| `unknown_tool` | 工具名不在 `ItsmTools._tool_names` 中 | `Error: Tool '<name>' not found.` |
| `bad_arguments` | `inspect.signature(tool).bind(**args)` 失败（缺少参数或参数名不对），或参数值不满足工具类型注解的严格校验（pydantic strict；公开 schema 由同一组注解生成） | 规范化信息，**不含** Python 类名（修正 F9）；该调用不执行 |
| `domain_error` | 工具抛出 `ItsmError` | `Error: <message>`，并在记录里保留 `code`、`field` |

模型侧错误只有以上三类。参数通过签名和类型校验之后，工具再抛出任何非 `ItsmError` 异常，都视为**环境缺陷**，归入下表的 `tool_internal`（第 4 步补修：原先的 `tool_exception_other` 把这种情况当作可恢复错误返回给模型，episode 仍能得到有效 reward，已删除）。严格类型校验已对 53 题全部 428 个 gold 调用验证不会误拒。

**基础设施错误**：抛出 `ItsmInfrastructureError(stage, task_id, episode_id)`，不产生 reward。

| stage | 触发条件 |
|---|---|
| `env_build` | episode 开始时读取种子、应用 delta、切分 org 或校验完整性失败 |
| `tool_internal` | 参数已通过校验，工具仍抛出非 `ItsmError` 异常。不向模型返回结果，记录 `result_received=False`，env 作废，不产生 reward；按 episode 级故障重试（§12） |
| `score_env_build` | 评分时构造 gold 环境（`kind=gold`）或 baseline 环境（`kind=baseline`）失败；原始异常保留为 `__cause__`。属评分级故障，按 §12 用同一最终状态重新评分 |
| `user_sim` | 用户模拟器 LLM 异常、超时或返回空内容 |
| `gold_replay` | 任一 gold action 回放失败（修正 F8） |
| `text_judge_call` | 自由文本裁判调用本身抛异常或超时 |
| `text_judge_protocol` | 裁判原始响应不符合判定协议，`kind` 为以下之一：`unparseable`（无法解析或缺 `results` 列表）、`invalid_item`、`invalid_index`（缺失、非整数或越界）、`duplicate_index`、`invalid_equivalent`（缺失或非布尔）、`missing_index`（有 pending 对没有判定） |
| `nl_judge_call` / `nl_judge_protocol` | NL 裁判沿用同一原则，在接入 NL 评估时实现（第 8 步） |
| `clock` | 时钟断言失败 |
| `timeout` | 单步或整个 episode 的墙钟超时 |
| `cancelled` | 外部取消 |

**不属于基础设施错误：** `TaskDefinitionError`（任务缺少 `current_time`、`seed_db` 或 gold actions 等必需字段）表示任务本身有误，重试不会改变结果，所以在 `env_build`、`score_env_build` 等位置都原样抛出，不包装成可重试的基础设施错误。

**裁判判定的边界**（第 3 步修订，`agent_rl.envs.itsm_scoring.parse_text_judge_response`）：
- 判定在原始响应的解析环节完成，不使用 Bench2 解析后的 `FreeTextVerdict`（它会把各种异常都默认成 `equivalent=False`），也不依据 reasoning 文本推断。
- 一个结果项只要 `index` 是范围内且只出现一次的整数、`equivalent` 是布尔值，就是有效判定。`true` 和 `false` 都有效，没有 reasoning 也不影响判定结果。
- 有效的 `false` 是正常的业务负判，计入 DB=0，不触发重试。

工具在进程内同步执行，无法从外部强杀。如果出现 `timeout`，该 env 标记为 poisoned，不再接受任何调用。是否需要进程隔离，由第 5 步的测试结果决定。

## 9. 奖励契约

```python
@dataclass(frozen=True)
class ItsmRewardInfo:
    db_evaluated: bool
    db_match: bool | None
    db_reward: float | None             # 1.0 / 0.0
    db_mismatches: list[str]            # 截取前 20 条
    text_match_requested: str           # exact | fuzzy | llm
    text_match_effective: str           # 必须等于 requested，否则在评分前报错
    text_judge_model: str | None
    nl_status: Literal["not_evaluated", "evaluated"]
    nl_reward: float | None             # not_evaluated 时为 None
    joint_reward: float | None          # 仅当 db 与 nl 都评估过时才有值
    training_reward: float              # 实际送给训练器的标量（决策 D1）
    training_reward_source: str         # 例如 "db_only"
```

规则：
1. **DB 评分。** 在任务时钟上下文中完成以下流程：
   - 新建 env，逐条回放 gold，任何一条失败都报错；
   - 新建一份 baseline；
   - 调用 `compare_dbs(gold_db, final_db, baseline_db, text_match)`，复用 Bench2 自身的比较和自由文本逻辑。

   不调用 `calculate_db_reward`，因为它会吞掉 gold 回放失败。第 3 步需证明：对全部 53 题，这条路径与 `calculate_db_reward` 在回放成功时结论一致。
2. **文本比较策略必须显式指定。** 构造时如果 `effective_strategy() != strategy`，抛出 `ValueError`（修正 F6）。
3. **NL 状态。** NL 不评估时记 `not_evaluated`，`nl_reward=None`，`joint_reward=None`。任何报表中不得把 DB-only 的结果称为 "联合成功率"。
4. **哪些情况评分。** 正常终止和截断（`agent_done`、`user_stop`、`max_steps`、`response_length`）都对最终 DB 评分，与 Bench2 `run_task` 一致，后者不论是否 `stopped` 都评分。截断信息另行记录。
5. **`invalid_action`。** outcome 的取法见决策 D2。
6. **基础设施故障没有 `ItsmRewardInfo`。** 不允许用 0 填充。
7. **评分只在 episode 结束时做一次。** 每步的 `environment_reward` 恒为 0。

## 10. Token 记录与上下文

- **loss mask：** 策略采样得到的 token 记为 1；系统提示、工具 schema、固定问候语、用户消息、工具结果和 chat template 分隔符都记为 0。
- **logprob：**
  - 采样服务必须返回逐 token 的 logprob。ITSM loop 拿不到时报错终止，不填 0.0（修正 F11）。
  - mask=0 的位置写 0.0 作占位，并记录 `logprob_source="placeholder_masked"`，训练器不得使用这些值。
  - 是否改为由训练器在当前策略版本下重算，第 7 步需核实 verl 实际读取的是哪一份。
- **turn span：** 每个策略回合记录 `(turn_index, start, end)`，与 `TurnRecord.token_trace` 一一对应。
- **上下文策略（决策 D3）：** 首版建议**禁用轮换**。token 预算不够时调用 `truncate("response_length")`，episode 以截断结束，保留的策略 token 占比恒为 1。由第 6 步测定的预算保证大多数 episode 不触发截断。截断率作为指标上报。

## 11. 数据样例

### 11.0 真实工具探测（本步实测）

- 命令：用独立环境的 Python，对 `itsmbench_single_017` 构造 env，直接调用 `Environment.get_response`。没有调用任何 LLM。
- 前提：任务时钟没有设置，但这些调用都是只读的，不受时钟影响。

| 调用 | Bench2 返回 |
|---|---|
| `find_incident_by_number(number="INC0000007")` | `error=False`，`{"incident_id":"INC_020","number":"INC0000007","short_description":"Development server access denied",...}` |
| `find_incident_by_number(number="INC9999999")` | `error=True`，`Error: Incident with number 'INC9999999' not found` |
| `find_incident_by_numbr()` | `error=True`，`Error: Tool 'find_incident_by_numbr' not found.` |
| `find_incident_by_number(num="x")` | `error=True`，`Error: IncidentToolsMixin.find_incident_by_number() got an unexpected keyword argument 'num'` |

以下样例中的 `episode_id` 简写为 `E` = `itsmbench_single_017:g7:0:1a2b3c4d`；`E` 只出现在调用方记录里，模型可见的调用 ID 是与之无关的 `call_NNN`（§5）。

11.1–11.6 是**设计样例**，表示协议要求的输入和输出，不是运行结果：
- 11.1 和 11.2 的工具返回文本取自 11.0 的实测；
- 11.3 和 11.5 中的步数、记录 ID、mismatch 文本都是示意值，未经运行验证。

### 11.1 正常工具调用

```json
{"action": {"kind": "tool_call", "name": "find_incident_by_number",
            "arguments": {"number": "INC0000007"}, "model_call_id": "call_0"},
 "transition": {"observation": "tool[call_001] ok: {\"incident_id\":\"INC_020\",\"number\":\"INC0000007\",...}",
                "terminated": false, "truncated": false, "termination_reason": null, "step_count": 2,
                "tool_record": {"call_id": "call_001", "name": "find_incident_by_number",
                                "arguments": {"number": "INC0000007"}, "result_received": true,
                                "error": null, "error_class": null}}}
```
预期：DB 不变（只读工具）；episode 继续；`tool_record` 和 ToolMessage 一一对应。

### 11.2 工具报错（模型侧，可恢复）

```json
{"action": {"kind": "tool_call", "name": "find_incident_by_number", "arguments": {"number": "INC9999999"}},
 "transition": {"observation": "tool[call_002] error(domain_error): Incident with number 'INC9999999' not found",
                "terminated": false, "truncated": false, "step_count": 3,
                "tool_record": {"call_id": "call_002", "result_received": true,
                                "error": "Incident with number 'INC9999999' not found",
                                "error_class": "domain_error", "error_code": "VALIDATION_ERROR",
                                "error_field": "number"}}}
```
预期：episode 继续，模型可以重试。下一次同名调用成功时，算一次 `error_recovery` 证据，供 E3 使用。参数名错误（`num=`）记为 `bad_arguments`，观察中不出现 `IncidentToolsMixin`。

### 11.3 模型结束（done）

```json
{"action": {"kind": "done"},
 "transition": {"observation": "", "terminated": true, "truncated": false,
                "termination_reason": "agent_done", "step_count": 14,
                "reward_info": {"db_evaluated": true, "db_match": true, "db_reward": 1.0,
                                "text_match_requested": "fuzzy", "text_match_effective": "fuzzy",
                                "nl_status": "not_evaluated", "nl_reward": null, "joint_reward": null,
                                "training_reward": 1.0, "training_reward_source": "db_only"}}}
```
预期：评分在任务时钟 `2024-06-01 09:00:00` 下进行；`joint_reward` 为 null，不能报为联合成功。

### 11.4 超时（基础设施）

```json
{"action": {"kind": "message", "text": "Could you confirm the CI serial number?"},
 "raises": {"type": "ItsmInfrastructureError", "stage": "user_sim",
            "task_id": "itsmbench_single_017", "episode_id": "E",
            "cause": "litellm Timeout after 60.0s"}}
```
预期：不产生 transition 和 reward；env 进入 poisoned 状态，`close()` 后丢弃。按 §12 处理：同一样本在同一策略版本下重试，重试用尽则整组丢弃。不得记为 reward=0。

### 11.5 截断（步数用尽，有待执行的调用）

```json
{"state": {"step_count": 12, "max_steps": 12},
 "action": {"kind": "tool_call", "name": "update_problem", "arguments": {"problem_id": "PRB_004", "state": "resolved"}},
 "transition": {"observation": "", "terminated": false, "truncated": true,
                "termination_reason": "max_steps", "step_count": 12,
                "tool_record": {"call_id": "call_011", "name": "update_problem", "result_received": false},
                "reward_info": {"db_evaluated": true, "db_match": false, "db_reward": 0.0,
                                "db_mismatches": ["problem/PRB_004 state: 'resolved' != 'in_progress'"],
                                "nl_status": "not_evaluated", "training_reward": 0.0}}}
```
预期：这次调用**没有执行**（DB 不含该修改）；`missing_result` 和 `abnormal_truncation` 证据可供 E3 使用。由 token 预算引起的截断，结构相同，只是 `termination_reason="response_length"`，`tool_record` 为空。

### 11.6 用户结束（补充）

模型发送 `message` 后，用户回复 `###STOP###`：`terminated=True`，`termination_reason="user_stop"`，进入评分。与 `agent_done` 分开统计。

## 12. 分组（G=4）与失败策略

1. 同一组的 4 个样本必须满足以下条件，并在 episode 记录中写明：
   - 相同 `task_id`；
   - 相同策略版本，记录 `policy_version` 或 `global_step`；
   - 各自用 `get_environment` 新建 DB。
2. 基础设施故障的重试（2026-10-29 用户明确口径）：**首次调用之外最多额外重试 3 次，即总计最多 4 次**。注意这与 AirLineRL 的 `max_episode_attempts=3`（总计 3 次）不同，ITSM 的设置项命名为 `max_infra_retries=3`，以免混淆。
   - **episode 级故障**（`env_build`、`user_sim`、`timeout` 等，episode 本身没有跑完）：在同一 rollout 步内用相同任务、相同策略版本、新建的 DB 重跑该样本。
   - **评分级故障**（`text_judge_call`、`text_judge_protocol`、`nl_judge_*`，episode 已跑完）：只重做评分调用，使用**同一条已保存的轨迹和同一个最终 DB 状态**，不重跑 episode，也不重新采样策略。
   - 只有基础设施错误会触发重试。有效判定（包括没有理由的负判）一律不重试。
   - 上限固定，不能一直重试到得到通过结果。温度 0 降低了采样随机性，但同样的裁判协议错误仍可能反复出现；用尽 4 次后按第 3 条处理。
3. 重试用尽后，**整组丢弃**。不用 reward=0 补位，不从别的任务借样本，不跨策略版本凑组。每次丢弃都要记录：`task_id`、组 ID、策略版本、触发丢弃的样本、最终的 `stage`/`kind`、已用尝试次数；并按 `stage`/`kind` 汇总每步的丢弃数量。
4. 一个批次中被丢弃的组超过阈值（建议 25%，决策 D6）时，该训练步中止并报告，不做参数更新。
5. 每步上报以下指标：
   - 有效组比例，即组内 reward 不全相同的组所占比例；
   - 全 0 组和全 1 组的数量；
   - 丢弃的组数及原因（按 `stage` 统计）；
   - 截断率与终止原因分布。
6. **无效组必须真正排除（第 9 步实现并验证）。** AirLineRL 现在的做法是重试用尽就抛异常（F13）。可接受的做法只有两种：
   - 把无效组从批次中真正移除，不进入任何前向、loss 或统计；
   - 或者保留张量形状，但用**有效性掩码**覆盖全部损失项（策略梯度、KL、熵及其他正则项）**以及所有统计分母**（token-mean / seq-mean 的分母、reward 和有效组比例等指标的分母）。

   **只把优势置零不等于丢弃样本。** 优势为 0 只消掉了策略梯度项；KL、熵等其他项仍会对这些样本产生梯度，它们的 token 也仍计入 loss 归一化分母和统计分母，会改变有效样本的权重和报告的指标。第 9 步需用梯度检查证明：无效组对参数梯度的贡献为 0，并且与"直接移除该组"的 loss 和梯度在数值容限内一致。

## 13. 数据库隔离

- 每个 episode、每次 gold 回放、每个 baseline 都调用 `get_environment(...)`，从磁盘重新加载 JSON，得到彼此独立的 pydantic 对象。
- 总是显式传入 `seed_db`（53 题都有），不依赖 `EW_ITSM_DB` 环境变量。
- 除时钟（thread-local，§6）外，没有发现模块级的可变共享状态。
- **须验证（第 5 步）：**
  - 并行跑 4 个 episode，各写各的 DB，结束后彼此的 hash 不受影响；
  - 种子文件的 hash 在运行前后保持不变。

## 14. 已定决策

2026-10-28 用户认可 D1–D6，其中 D5 改为 deepseek flash；2026-10-29 用户决定 D7、D8（随第 3 步验收）。下表 "决定" 列即为固定规则；修改需用户重新批准。

| # | 决策 | 决定 | 理由 |
|---|---|---|---|
| D1 | 训练 reward 的来源 | `db_only`；NL 只在 E0 和最终评测时评估 | NL 裁判每个 episode 都要调用付费 LLM，且结果不确定；DB 评分可复现。报告中要分开写 DB、NL 和联合指标 |
| D1b | 训练时的文本比较策略 | `fuzzy`（确定、免费）；评测时再并行报告 `llm` | 若选 `llm`，需固定裁判模型和温度，裁判故障按基础设施处理 |
| D2 | `invalid_action` 的 outcome | 沿用 AirLineRL：outcome 记 0，episode 终止 | 与 airline 实验协议一致，便于对照；另一种做法是 "返回错误观察让模型重试" |
| D3 | 上下文轮换 | 首版禁用，预算不够就以 `response_length` 截断 | 避免 "只训练最后一段" 带来的信用丢失；第 6 步测出预算后再复核 |
| D4 | 是否向模型公开 `current_time` | 不公开（与 Bench2 `LLMAgent` 一致） | 公开会改变任务难度。如果 policy 中有依赖当前时间的规则，第 6 步检查后再议 |
| D5 | 用户模拟器的模型和温度 | `deepseek/deepseek-v4-flash`，温度 0（用户指定 flash，不沿用 airline 的 pro） | Bench2 默认是 `gpt-4o-mini`、温度 0.7。温度 0 能减少用户模拟器带来的组内差异，但**不等于完全确定**：服务端批处理、浮点非确定性、模型版本更新等都可能让同一输入得到不同输出，所以组内差异不能全部归因于策略。第 7/8 步用同一输入重复调用，实测并记录用户回复的一致率。litellm 已登记该 ID；首次真实调用（第 7/8 步）时核实可用性 |
| D6 | 丢弃组超过多少时中止训练步 | 25% | 防止有效样本过少时仍做更新 |
| D7 | 评测用的 DB 文本裁判与 NL 裁判 | 两者都用 `deepseek/deepseek-v4-pro`，温度 0；调用参数见 §14.3 | 与 airline 的 `evaluator_llm` 一致，并与用户模拟器（flash）使用不同模型 |
| D8 | gold 未改变的自由文本字段 | 保持 Bench2 规则，见 §14.4；本轮不在训练 reward 或 benchmark 分数中增加严格字段检查 | 修改评分规则会使结果与 Bench2 不可比 |

### 14.1 模型角色（第 3 步修订时明确）

| 角色 | 模型 | 何时调用 | 状态 |
|---|---|---|---|
| 策略模型（被训练） | verl 当前策略权重，由 verl rollout 的 vLLM 采样；基座为 `configs/model/qwen3_8b_lora.yaml` 中的 `Qwen/Qwen3-8B` | 每个策略回合 | 不使用任何外部 API；logprob 必须来自该采样服务（§10） |
| 用户模拟器 | `deepseek/deepseek-v4-flash`，温度 0（D5） | 训练、E0 和最终评测中的每个用户回合 | **API 连通性未验证**，首次真实调用在第 7 步 |
| DB 自由文本裁判 | 训练时不使用（D1b 定为 `fuzzy`）；评测时并行报告 `llm` 策略，用 `deepseek/deepseek-v4-pro`（D7） | 只在评测阶段 | **API 连通性未验证**，第 7/8 步有界验证 |
| NL 断言裁判 | 训练时不使用（D1 定为 `db_only`）；E0 和最终评测时使用 `deepseek/deepseek-v4-pro`（D7） | 只在评测阶段 | **API 连通性未验证**，第 7/8 步有界验证 |

`deepseek/deepseek-v4-flash` 只用于用户模拟器；两个裁判用 `deepseek/deepseek-v4-pro`。用户模拟器的思考模式和输出预算在第 7 步首次真实调用前另行固定。

### 14.2 评分适配层对应的 Bench2 版本

- `Bench2/` 导入自 Enterprise-Worlds `37175af590d3b9c7d2288501b604020e547e3783`，在 AirLineRL 提交 `db9ad11` 中引入；此后到 `2e0e57a` 没有改动。`Bench2` 的 git 树哈希为 `e87531fe4fd7ed3d664b4f8e48ef9e6c848867ab`。
- `agent_rl.envs.itsm_scoring` 和 `itsm_env` 调用了 Bench2 的若干私有函数（`evaluator_env._compare_record`、`evaluator_env._dump`、`text_match_strategy._extract_json`、`environment._stringify`），所以下列源码的 sha256 由测试 `test_bench2_sources_match_pinned_checksums` 固定；任何一个变化都会让测试失败，需重新核对适配层：

| 文件（`Bench2/src/enterprise_worlds/`） | sha256 |
|---|---|
| `evaluator/evaluator_env.py` | `cd8b458c8f928b443a71c8dc7dbeb03c1a3461242725ce441bfda93826c2e95b` |
| `evaluator/text_match_strategy.py` | `6bdd229561c4c38c39bd3276fa626667931343365fd66cf08736b53036bc4b20` |
| `evaluator/evaluator_nl.py` | `a7c6f23b0a0f2ad735f6e54e7cf6c65e879fc4550fece6c587ec507c93e91b51` |
| `evaluator/evaluator.py` | `db5f87b82a3ded5df1fbd2c90f0faf200f4ed3b0b7ada60000085dd94ac60be4` |
| `utils/clock.py` | `2ca1286d862c9886bace4f4ecd835a23f9d166373c177a917372e0149a203088` |
| `utils/text_match.py` | `d4790b0113c6ec06de0b24caa8949c3a97bf3d57df04ae88d47cb8b5b942a27c` |
| `environment/environment.py` | `0f39c84f0c271d88129d773d4ae54778786fdf0c647bd3128130384d9eff67a9` |
| `environment/toolkit.py` | `67a943876e6cd3de49c375cfcbd9db978395194f1217b2a502a96bfe50972a78` |
| `domains/itsm/environment.py` | `4783009792689f3dc6e7064fb55d5a87c15ffdac7737eda23ad9d46518ca6da3` |
| `domains/itsm/data_model.py` | `0ad66f253e4ea52cc213667f0a4d892e53ebf122aeb923d3e82ab83161910ab5` |
| `data_model/message.py` | `618acdbb19dcb947b2fba5e734530523c577f483e746939dbcbbfd417662040b` |
| `orchestrator/orchestrator.py` | `731831c31506c1a1676a8fd1db842fd0c95c5f5489cef5f3ada95d1adbf7767c` |

### 14.3 评测裁判的调用参数（D7）

| 项 | DB 自由文本裁判 | NL 断言裁判 |
|---|---|---|
| 模型 | `deepseek/deepseek-v4-pro` | `deepseek/deepseek-v4-pro` |
| 温度 | 0 | 0 |
| 思考模式 | **关闭**，显式传 `extra_body={"thinking": {"type": "disabled"}}` | 同左 |
| 输出预算 | `max_tokens=4096`（即 Bench2 `DEFAULT_LLM_NL_JUDGE_ARGS`） | 同左 |
| 提示词版本 | Bench2 `text_match_strategy.JUDGE_SYSTEM_PROMPT`，sha256 `2e2350908ca2b54271cc96dab31c414a14413427af177a2fbcbcd1df272761d4`；用户消息格式与 `judge_free_text` 相同（已有对照测试） | Bench2 `evaluator_nl.JUDGE_SYSTEM_PROMPT`，sha256 `295597be7c47347429e8adf6f236e62eae335d7ba587cdc221e866f4dac7927f`；用户消息格式与 `evaluate_nl_assertions` 相同 |

- **为什么要显式关闭思考：** litellm 1.82.6 的 DeepSeek 适配只会透传 `thinking={"type": "enabled"}`，传 `{"type": "disabled"}` 会被静默丢弃（`litellm/llms/deepseek/chat/transformation.py` 的 `map_openai_params`），那样实际模式就取决于服务端默认值。所以用 `extra_body` 直接传。选择关闭思考，是为了让 4096 的输出预算只用于判定 JSON；如需开启思考，要重新批准，并重新确定输出预算。airline 的 `evaluator_llm` 没有设置思考模式，这是 ITSM 与 airline 的一处差异。
- **第 7/8 步的有界验证**（仅几次调用）：确认服务端接受 `disabled`；响应中没有 `reasoning_content`；`finish_reason == "stop"`；模型 ID 与请求一致。
- **失败处理：** 调用失败、参数被拒绝或 `finish_reason == "length"`，都按基础设施错误处理（`text_judge_call`/`text_judge_protocol`、`nl_judge_*`），并按 §12 重试。**不得静默换模型、去掉思考参数或退回 `fuzzy`。**
- 实际发出的模型、参数和提示词 sha256 要写进每次评测的运行记录。
- 温度 0 不保证同一输入得到完全相同的判定。评测报告不得假设裁判完全确定；第 8 步对少量固定输入重复调用，记录判定一致率。

### 14.4 gold 未改变的自由文本字段（D8）

- **规则：** 对 `ItsmDB.freetext_fields()` 列出的字段（notification、incident、problem、change、knowledge、service、service_offering、user_group 的说明类字段），如果 gold 值等于 baseline（gold 没有改变它），Bench2 的 `_compare_record` 在选择比较策略之前就跳过该字段。因此这一规则对 `exact`、`fuzzy`、`llm` 三种策略都成立，不限于 problem 的 `workaround`、`fix_notes`；`llm` 策略下这类字段也不会送去裁判。
- `llm` 策略的另一面：gold **改变过**的自由文本字段，即使预测值与 gold 完全相同，也会送去裁判（`_compare_record` 在 `llm` 分支不先做相等比较）。例如 Single-017 gold 回放在 `llm` 下送判 problem 的 `problem_statement`、`short_description`、`worknotes` 三项。这影响评测成本和裁判方差，不影响 exact/fuzzy。
- **因此：DB=1 不能保证所有可选字段为空或未改动。** DB=1 只保证结构字段与 gold 一致，以及 gold 改变过的自由文本字段按所选策略匹配。
- 第 3 步把 "可选字段为空" 拆成两项分别验证（见 `docs/itsm_migration_progress.md` 第 3 步证据）：
  1. **gold 状态：** gold 回放后，Single-017 的 problem 中 7 个应为空的可选字段确实为空。这一项说明的是 gold 本身，不是评分能力。
  2. **评分约束范围：** 5 个结构可选字段被设值时 DB=0；2 个自由文本可选字段被设值时 DB 仍为 1（exact、fuzzy、llm 三种策略都已测试）。
- 本轮不在训练 reward 或 benchmark 分数中增加严格字段检查。误差分析和 NL 断言可以把这类改动单列出来，但不改变 DB 分数。

## 15. 对后续步骤的约束（本步不实现）

- 第 3 步：gold 回放走 §9 规则 1 的路径，固定时钟，对比 `calculate_db_reward`；MSP-021 和 Single-017 各做正例和反例。
- 第 4 步：实现 `ItsmEnv`；与 `Orchestrator` 做对照测试；验收 §4 的增量观察、§5 的 ID 一一对应、§8 的错误分类。
- 第 5 步：§6 时钟串扰、§13 隔离、§8 超时与 poisoned 处理、进程隔离方案。
- 第 6 步：`max_steps` 和 token 预算；D3、D4 复核。
- 第 7 步：§10 logprob 来源与 §3 泄漏检查。
- 第 9 步：§12 丢弃整组的机制。
