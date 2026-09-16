#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:-3}"
OUT_DIR="${2:-${PROJECT_ROOT}/output/nano_auto_lang_perf_20260817/fp32}"
SOCKET_PATH="${3:-/tmp/funasr_nano_gpu3_perf.sock}"
VAD_WORKERS="${4:-1}"
LID_MODE="${5:-metadata}"
VAD_THREADS="${6:-8}"
ASR_DTYPE="${ASR_DTYPE:-fp32}"
CONFIGS="${CONFIGS:-b16_ts b32_ts b64_ts b96_ts b128_ts b64_no_ts b96_no_ts}"

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "[ERROR] Activate the FunASR conda environment first."
    exit 1
fi

SOURCE_TAR="${PROJECT_ROOT}/test/audio_200.tar"
BENCH_DIR="${PROJECT_ROOT}/test/benchmark_1h"
mkdir -p "${BENCH_DIR}" "${OUT_DIR}"
for index in 0 1 2 3; do
    if [ ! -e "${BENCH_DIR}/audio_200_${index}.tar" ]; then
        ln "${SOURCE_TAR}" "${BENCH_DIR}/audio_200_${index}.tar"
    fi
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

run_config() {
    local name="$1"
    local batch_size="$2"
    local timestamps="$3"
    local config_dir="${OUT_DIR}/${name}"
    local timestamp_args=()
    if [ "${timestamps}" = "off" ]; then
        timestamp_args+=(--no_timestamps)
    fi
    mkdir -p "${config_dir}"
    "${CONDA_PREFIX}/bin/python" \
        "${PROJECT_ROOT}/pipeline/tar_pipeline_nodnsmos_nano_auto_lang_ckpt.py" \
        --tar_paths \
            "${BENCH_DIR}/audio_200_0.tar" \
            "${BENCH_DIR}/audio_200_1.tar" \
            "${BENCH_DIR}/audio_200_2.tar" \
            "${BENCH_DIR}/audio_200_3.tar" \
        --out_json "${config_dir}/labels.json" \
        --metrics_json "${config_dir}/metrics.json" \
        --funasr_model_dir "${PROJECT_ROOT}/models/Fun-ASR-Nano-2512" \
        --lid_model_dir "${PROJECT_ROOT}/models/faster-whisper-tiny" \
        --fireredvad_model "${PROJECT_ROOT}/models/FireRedVAD" \
        --fireredvad_root "${PROJECT_ROOT}/FireRedVAD" \
        --batch_size "${batch_size}" \
        --lid_batch_size 32 \
        --lid_mode "${LID_MODE}" \
        --lid_sample_seconds 12 \
        --bucket_edges 1.5 2 3 4 5 6 7 8 10 12 14 16 20 24 30 \
        --bucket_lookahead_batches 1 \
        --asr_prefetch_batches 64 \
        --vad_device cpu \
        --vad_workers "${VAD_WORKERS}" \
        --vad_threads "${VAD_THREADS}" \
        --min_dur 1.0 \
        --max_dur 30.0 \
        --post_keep_langs zh zh-en en other \
        --asr_dtype "${ASR_DTYPE}" \
        --asr_server_socket "${SOCKET_PATH}" \
        ${timestamp_args[@]+"${timestamp_args[@]}"} \
        > "${config_dir}/run.log" 2>&1
}

for config in ${CONFIGS}; do
    case "${config}" in
        b16_ts) run_config "${config}" 16 on ;;
        b32_ts) run_config "${config}" 32 on ;;
        b64_ts) run_config "${config}" 64 on ;;
        b96_ts) run_config "${config}" 96 on ;;
        b128_ts) run_config "${config}" 128 on ;;
        b64_no_ts) run_config "${config}" 64 off ;;
        b96_no_ts) run_config "${config}" 96 off ;;
        b128_no_ts) run_config "${config}" 128 off ;;
        b160_no_ts) run_config "${config}" 160 off ;;
        b192_no_ts) run_config "${config}" 192 off ;;
        b224_no_ts) run_config "${config}" 224 off ;;
        *) echo "[ERROR] Unknown CONFIGS entry: ${config}"; exit 2 ;;
    esac
done

date -Iseconds > "${OUT_DIR}/DONE"
