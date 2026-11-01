#!/usr/bin/env bash
# Step 7B pipeline on exp-itsm (launched with projtool run_script; one run, all phases).
#
#   BOOT_EPOCH=<unix time of the power-on request> bash scripts/itsm/run_7b.sh
#
# Phases (each logged to $OUT/phases.jsonl with UTC start/end and status):
#   env        versions, GPU, disk usage                      (no download)
#   space      free space >= missing model bytes + 2 GiB, else stop (exit 4)
#   download   snapshot_download(local_dir) of the pinned revision, only if files are missing
#   verify     SHA256 / git blob SHA1 of every file, chat template hash
#   inputs     rebuild the length inputs with the model tokenizer; must match the committed meta
#   len32k     vLLM max_model_len 32768: tiers a, b
#   len40k     vLLM max_model_len 40960: tiers a, b, c, then the real episodes on the same engine
#   recompute  HF transformers recompute of the episode tokens + gold-target diagnostics
#   caches     sizes of the caches written during the run
# Budget: before each phase, elapsed since BOOT_EPOCH + the phase's conservative estimate must
# stay below 3 GPU-h minus a 10-minute shutdown margin, else stop (exit 5). Any failing phase
# stops the pipeline. Never deletes files outside this run's own directories.
set -uo pipefail

P=/root/autodl-tmp/conda-envs/itsm-train/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-8B
OUT=${OUTPUT_ROOT:?OUTPUT_ROOT must be set}/step7b
BUDGET_S=$((3 * 3600 - 600))
: "${BOOT_EPOCH:?BOOT_EPOCH must be set}"
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
unset OPENAI_API_KEY DEEPSEEK_API_KEY MICU_API_KEY ANTHROPIC_API_KEY
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
# Bench2 slice_db_to_orgs iterates a set: fix the hash seed so list-result order is reproducible
# (inherited by every ItsmEnvWorker child). Recorded as a 7B finding, not an environment fix.
export PYTHONHASHSEED=0

log_phase() { # phase status start end [detail]
  printf '{"phase":"%s","status":"%s","start_utc":"%s","end_utc":"%s","elapsed_since_boot_s":%s,"detail":%s}\n' \
    "$1" "$2" "$3" "$4" "$(( $(date +%s) - BOOT_EPOCH ))" "${5:-null}" >> "$OUT/phases.jsonl"
}
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { # code reason
  log_phase stop "exit_$1" "$(utc)" "$(utc)" "\"$2\""
  echo "[7b] STOP ($1): $2"
  exit "$1"
}
gpu_sampler_start() { nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -lms 500 > "$OUT/gpu_$1.csv" 2>/dev/null & SAMPLER=$!; }
gpu_sampler_stop() { kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null; }
phase() { # name estimate_s command...   (ALLOW_RC="2 3": record these exit codes and continue)
  local name=$1 est=$2; shift 2
  local elapsed=$(( $(date +%s) - BOOT_EPOCH ))
  if (( elapsed + est > BUDGET_S )); then stop 5 "budget: ${elapsed}s used + ${est}s estimate for $name > ${BUDGET_S}s"; fi
  local start; start=$(utc)
  echo "[7b] phase $name start $start (elapsed ${elapsed}s)"
  gpu_sampler_start "$name"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  gpu_sampler_stop
  tail -n 25 "$OUT/$name.log"
  if (( rc != 0 )) && [[ " ${ALLOW_RC:-} " == *" $rc "* ]]; then
    log_phase "$name" "completed_with_failures_rc_$rc" "$start" "$(utc)"; return 0
  fi
  if (( rc != 0 )); then log_phase "$name" "failed_rc_$rc" "$start" "$(utc)"; stop "$rc" "phase $name failed (see $name.log)"; fi
  log_phase "$name" ok "$start" "$(utc)"
}

env_report() {
  date -u; nvidia-smi; df -h / /root/autodl-tmp /root/autodl-fs; du -sh /root/autodl-tmp/* 2>/dev/null
  ls -la "$MODEL" 2>/dev/null || echo "model dir absent"
  git -C . rev-parse HEAD 2>/dev/null; git -C . status --short 2>/dev/null | head
  $P -c "import vllm, torch, transformers, sys; print('python', sys.version.split()[0]); print('vllm', vllm.__version__); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__); print('gpu', torch.cuda.get_device_name(0))"
  $P -c "import importlib.metadata as m; [print(p, m.version(p)) for p in ('flashinfer-python','verl','huggingface_hub','safetensors')]" 2>&1
  $P -c "import flash_attn" 2>&1 | tail -1 || true  # expected to fail: flash_attn is not installed (step 1)
  return 0
}

download() {
  if [ -f /etc/network_turbo ]; then source /etc/network_turbo >/dev/null 2>&1; echo "network_turbo sourced"; fi
  local rev; rev=$($P -c "import json;print(json.load(open('docs/itsm_step7b/model_files.json'))['revision'])")
  local code; code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' "https://huggingface.co/api/models/Qwen/Qwen3-8B/revision/$rev")
  echo "huggingface.co -> $code"
  if [ "$code" != 200 ]; then
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    export HF_ENDPOINT=https://hf-mirror.com
    code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' "$HF_ENDPOINT/api/models/Qwen/Qwen3-8B/revision/$rev")
    echo "hf-mirror.com -> $code"
  fi
  [ "$code" = 200 ] || { echo "no reachable HF endpoint"; return 1; }
  echo "endpoint ${HF_ENDPOINT:-https://huggingface.co}"
  $P scripts/itsm/model_files_7b.py download --dir "$MODEL"
}

caches() {
  df -h / /root/autodl-tmp
  du -sh "$MODEL" ~/.cache/vllm ~/.cache/flashinfer ~/.triton ~/.cache/huggingface ~/.cache/torch /tmp 2>/dev/null
}

echo "[7b] start $(utc), boot epoch $BOOT_EPOCH"
phase env 120 env_report
$P scripts/itsm/model_files_7b.py check-space --dir "$MODEL" > "$OUT/space.json"
rc=$?
cat "$OUT/space.json"
log_phase space "rc_$rc" "$(utc)" "$(utc)" "$(cat "$OUT/space.json")"
(( rc == 4 )) && stop 4 "not enough disk space for the model (see space.json)"
(( rc != 0 )) && stop "$rc" "space check failed"
if grep -q '"missing_bytes": 0' "$OUT/space.json"; then
  log_phase download skipped_present "$(utc)" "$(utc)"
else
  phase download 3600 download
fi
# From here on: local files only (no hub access, no proxy).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
phase verify 600 $P scripts/itsm/model_files_7b.py verify --dir "$MODEL"
phase inputs 600 $P scripts/itsm/build_7b_length_inputs.py --tokenizer-dir "$MODEL" --out "$OUT/length_inputs.json" --check docs/itsm_step7b/length_inputs_meta.json
ALLOW_RC="2" phase len32k 1200 $P scripts/itsm/run_7b_gpu.py length --model "$MODEL" --max-model-len 32768 --tiers a,b --inputs "$OUT/length_inputs.json" --out "$OUT/len32k"
ALLOW_RC="2 3" phase len40k 3000 $P scripts/itsm/run_7b_gpu.py length --model "$MODEL" --max-model-len 40960 --tiers a,b,c --episodes --inputs "$OUT/length_inputs.json" --out "$OUT/len40k"
phase recompute 1200 $P scripts/itsm/run_7b_gpu.py recompute --model "$MODEL" --trajectories "$OUT/len40k/episodes.json" --length-inputs "$OUT/length_inputs.json" --out "$OUT/recompute"
phase caches 60 caches
log_phase done ok "$(utc)" "$(utc)"
echo "[7b] done $(utc)"
