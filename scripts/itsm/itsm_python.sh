#!/usr/bin/env bash
# Launch Python for any ITSM run with the protocol hash seed (docs/itsm_protocol.md §13.1).
# PYTHONHASHSEED is read once at interpreter start, so it must be set here, before Python
# starts; ItsmEnvWorker children inherit it. Usage: scripts/itsm/itsm_python.sh SCRIPT.py ARGS...
# Interpreter: $ITSM_PYTHON, else the cloud env, else python3.
set -euo pipefail
export PYTHONHASHSEED=0
if [[ -n "${ITSM_PYTHON:-}" ]]; then PY=$ITSM_PYTHON
elif [[ -x /root/autodl-tmp/conda-envs/itsm-train/bin/python ]]; then PY=/root/autodl-tmp/conda-envs/itsm-train/bin/python
else PY=python3; fi
exec "$PY" "$@"
