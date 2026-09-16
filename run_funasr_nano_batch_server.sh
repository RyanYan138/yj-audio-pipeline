#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "[ERROR] 请先激活 funasr 环境"
    exit 1
fi

GPU="${1:-0}"
SOCKET_PATH="${2:-${PROJECT_ROOT}/output/funasr_nano_batch.sock}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/pipeline:${PYTHONPATH:-}"
for nvidia_root in "${CONDA_PREFIX}"/lib/python3.*/site-packages/nvidia; do
    if [ -d "${nvidia_root}" ]; then
        for lib_dir in "${nvidia_root}"/*/lib; do
            [ -d "${lib_dir}" ] && LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
        done
    fi
done
export LD_LIBRARY_PATH

exec "${CONDA_PREFIX}/bin/python" \
    "${PROJECT_ROOT}/asr/funasr_nano_batch_server.py" \
    --model-dir "${PROJECT_ROOT}/models/Fun-ASR-Nano-2512" \
    --socket "${SOCKET_PATH}" \
    --device cuda:0
