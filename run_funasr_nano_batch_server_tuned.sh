#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ENV_PREFIX="$(cd "${PROJECT_ROOT}/../.." && pwd)/conda-envs/funasr_vllm"

python_has_runtime() {
    [ -x "$1" ] && "$1" -c 'import funasr, numpy, soundfile, torch, transformers' \
        >/dev/null 2>&1
}

if [ -n "${FUNASR_PYTHON:-}" ]; then
    PYTHON_BIN="${FUNASR_PYTHON}"
elif python_has_runtime "${CONDA_PREFIX:-}/bin/python"; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif python_has_runtime "${DEFAULT_ENV_PREFIX}/bin/python"; then
    PYTHON_BIN="${DEFAULT_ENV_PREFIX}/bin/python"
else
    echo "[ERROR] No usable FunASR Python found. Activate funasr_vllm or set FUNASR_PYTHON."
    exit 1
fi
if ! python_has_runtime "${PYTHON_BIN}"; then
    echo "[ERROR] Missing FunASR runtime dependencies in ${PYTHON_BIN}."
    exit 1
fi
RUNTIME_PREFIX="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
export CONDA_PREFIX="${RUNTIME_PREFIX}"

GPU="${1:-0}"
SOCKET_PATH="${2:-${PROJECT_ROOT}/output/funasr_nano_tuned.sock}"
DTYPE="${3:-fp32}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/pipeline:${PYTHONPATH:-}"
for nvidia_root in "${RUNTIME_PREFIX}"/lib/python3.*/site-packages/nvidia; do
    if [ -d "${nvidia_root}" ]; then
        for lib_dir in "${nvidia_root}"/*/lib; do
            [ -d "${lib_dir}" ] && LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
        done
    fi
done
export LD_LIBRARY_PATH

exec "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/asr/funasr_nano_batch_server_tuned.py" \
    --model-dir "${PROJECT_ROOT}/models/Fun-ASR-Nano-2512" \
    --socket "${SOCKET_PATH}" \
    --device cuda:0 \
    --dtype "${DTYPE}"
