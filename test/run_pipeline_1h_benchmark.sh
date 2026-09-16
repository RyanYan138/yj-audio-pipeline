#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "[ERROR] 请先激活 funasr 环境"
    exit 1
fi

GPU="${1:-0}"
OUT_DIR="${2:-${PROJECT_ROOT}/output/bucket_benchmark_1h}"
SOURCE_TAR="${PROJECT_ROOT}/test/audio_200.tar"
BENCH_DIR="${PROJECT_ROOT}/test/benchmark_1h"
mkdir -p "${BENCH_DIR}" "${OUT_DIR}"
for index in 0 1 2 3; do
    ln -f "${SOURCE_TAR}" "${BENCH_DIR}/audio_200_${index}.tar"
done

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

SERVER_ARGS=()
if [ -n "${ASR_SERVER_SOCKET:-}" ]; then
    SERVER_ARGS+=(--asr_server_socket "${ASR_SERVER_SOCKET}")
fi

"${CONDA_PREFIX}/bin/python" \
    "${PROJECT_ROOT}/pipeline/tar_pipeline_nodnsmos_bucket_batch_ckpt.py" \
    --tar_paths \
        "${BENCH_DIR}/audio_200_0.tar" \
        "${BENCH_DIR}/audio_200_1.tar" \
        "${BENCH_DIR}/audio_200_2.tar" \
        "${BENCH_DIR}/audio_200_3.tar" \
    --out_json "${OUT_DIR}/labels.json" \
    --metrics_json "${OUT_DIR}/metrics.json" \
    --funasr_model_dir "${PROJECT_ROOT}/models/Fun-ASR-Nano-2512" \
    --lid_model_dir "${PROJECT_ROOT}/models/faster-whisper-tiny" \
    --fireredvad_model "${PROJECT_ROOT}/models/FireRedVAD" \
    --fireredvad_root "${PROJECT_ROOT}/FireRedVAD" \
    --batch_size 16 \
    --lid_batch_size 32 \
    --lid_sample_seconds 12 \
    --bucket_edges 1.5 2 3 4 5 6 7 8 10 12 14 16 20 24 30 \
    --bucket_lookahead_batches 4 \
    --vad_device cpu \
    --min_dur 1.0 \
    --max_dur 30.0 \
    --target_langs en \
    --min_lang_prob 0.75 \
    --force_language_from_lid \
    ${SERVER_ARGS[@]+"${SERVER_ARGS[@]}"}
