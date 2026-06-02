#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FOREGROUND=false
PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --foreground)
            FOREGROUND=true
            shift
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

PATH_TO_VENV="${PROJECT_ROOT}/.venv"
if [[ ! -f "${PATH_TO_VENV}/bin/activate" ]]; then
    PATH_TO_VENV="${PROJECT_ROOT}/.venv312"
fi

if [[ ! -f "${PATH_TO_VENV}/bin/activate" ]]; then
    echo "Virtual environment not found at: ${PROJECT_ROOT}/.venv or ${PROJECT_ROOT}/.venv312" >&2
    echo "Create one of these venvs before running." >&2
    exit 1
fi

source "${PATH_TO_VENV}/bin/activate"
RUNNER="${PROJECT_ROOT}/rnn_shakespeare_train_runner.py"

if [[ "${FOREGROUND}" == true ]]; then
    exec python "${RUNNER}" "${PYTHON_ARGS[@]}"
fi

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/rnn_shakespeare_train_$(date +%Y%m%d_%H%M%S).log"

nohup python "${RUNNER}" "${PYTHON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Training started in background (nohup)."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
