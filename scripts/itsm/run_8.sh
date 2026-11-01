#!/usr/bin/env bash
# Step 8 diagnostics on exp-itsm (launched with projtool run_script). NOT approved to run yet:
# the API review and the GPU diagnostic each need a separate approval.
#
#   BOOT_EPOCH=<unix time of the power-on request> PRIOR_GPU_S=<earlier boots> bash scripts/itsm/run_8.sh [A,B]
#
# Before launching (local): python scripts/itsm/runtime_deps.py push --ssh-alias <alias> --remote-root /workspace
# and write DEEPSEEK_API_KEY into /root/.itsm_api_env (root only, 600) over ssh stdin.
#
# Phases (UTC start/end in $OUT/phases.jsonl): env, preflight (runtime deps + usage + task gates),
# verify (model files vs pinned hashes, no download), plan (reservations), diagnostics, summary.
# Budget guard: before each phase, elapsed GPU time + the phase estimate must stay below the cap
# minus a 10-minute shutdown margin; the diagnostics phase enforces its own soft/hard deadlines.
set -uo pipefail

P=/root/autodl-tmp/conda-envs/itsm-train/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-8B
OUT=${OUTPUT_ROOT:?OUTPUT_ROOT must be set}/step8
PARTS=${1:-A,B}
: "${BOOT_EPOCH:?BOOT_EPOCH must be set}"
PRIOR_GPU_S=${PRIOR_GPU_S:-0}
BUDGET_S=$((2 * 3600 - 600))
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
# Protocol §13.1: before any Python starts; inherited by every ItsmEnvWorker child.
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
unset OPENAI_API_KEY MICU_API_KEY ANTHROPIC_API_KEY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
if [ -f /root/.itsm_api_env ]; then
  set -a; source /root/.itsm_api_env; set +a
  rm -f /root/.itsm_api_env   # the key lives only in this process tree from here on
fi
unset MICU_API_KEY MICU_BASE_URL

log_phase() { printf '{"phase":"%s","status":"%s","start_utc":"%s","end_utc":"%s","elapsed_gpu_s":%s}\n' \
  "$1" "$2" "$3" "$4" "$(( $(date +%s) - BOOT_EPOCH + PRIOR_GPU_S ))" >> "$OUT/phases.jsonl"; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { log_phase stop "exit_$1" "$(utc)" "$(utc)"; echo "[8] STOP ($1): $2"; exit "$1"; }
gpu_sampler_start() { nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits -lms 1000 > "$OUT/gpu_$1.csv" 2>/dev/null & SAMPLER=$!; }
gpu_sampler_stop() { kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null; }
phase() { # name estimate_s command...   (ALLOW_RC="3": a stopped-but-recorded run, continue to the summary)
  local name=$1 est=$2; shift 2
  local elapsed=$(( $(date +%s) - BOOT_EPOCH + PRIOR_GPU_S ))
  (( elapsed + est > BUDGET_S )) && stop 5 "GPU budget: ${elapsed}s used + ${est}s for $name > ${BUDGET_S}s"
  local start; start=$(utc)
  gpu_sampler_start "$name"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  gpu_sampler_stop
  tail -n 20 "$OUT/$name.log"
  if (( rc != 0 )) && [[ " ${ALLOW_RC:-} " == *" $rc "* ]]; then log_phase "$name" "stopped_rc_$rc" "$start" "$(utc)"; return 0; fi
  (( rc != 0 )) && { log_phase "$name" "failed_rc_$rc" "$start" "$(utc)"; stop "$rc" "phase $name failed (see $name.log)"; }
  log_phase "$name" ok "$start" "$(utc)"
}

env_report() { date -u; nvidia-smi; df -h /; git -C . rev-parse HEAD 2>/dev/null; $P -c "import vllm, torch; print(vllm.__version__, torch.__version__)"; return 0; }

phase env 120 env_report
phase preflight 180 $P scripts/itsm/runtime_deps.py preflight
phase hashseed 180 $P scripts/itsm/check_hash_seed.py --require-fixed
phase verify 600 $P scripts/itsm/model_files_7b.py verify --dir "$MODEL"
phase plan 60 $P scripts/itsm/run_8_diagnostics.py plan
[ -n "${DEEPSEEK_API_KEY:-}" ] || stop 6 "DEEPSEEK_API_KEY not provided"
ALLOW_RC="3" phase diagnostics 4200 $P scripts/itsm/run_8_diagnostics.py run --model "$MODEL" --out "$OUT/run" --parts "$PARTS" --confirm-billing
phase summary 300 $P scripts/itsm/summarize_8.py --run-dir "$OUT/run" --tokenizer-dir "$MODEL" --out "$OUT/summary.json"
log_phase done ok "$(utc)" "$(utc)"
