# 第 7B 步：真实 GPU 推理、长度/显存与采样 logprob 验证（待验收）

<!-- RESULTS -->

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
