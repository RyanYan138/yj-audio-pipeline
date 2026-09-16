#!/usr/bin/env bash
#
# Usage:
#   bash run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh INPUT.tar OUTPUT.json GPU [ASR_BATCH] [LID_BATCH]
#
# Environment:
#   ASR_DTYPE=fp32|bf16
#   POST_KEEP_LANGS="zh zh-en"
#   NO_TIMESTAMPS=1
#   ASR_SERVER_SOCKET=/tmp/funasr_nano_gpu3.sock
#   VAD_WORKERS=1..8
#   VAD_THREADS=1..8 (per worker)
#   LID_MODE=metadata|off
#   RESUME=1
#   FUNASR_PYTHON=/path/to/funasr_vllm/bin/python

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ENV_PREFIX="$(cd "${PROJECT_ROOT}/../.." && pwd)/conda-envs/funasr_vllm"
LID_MODE="${LID_MODE:-off}"

python_has_runtime() {
    if [ "${LID_MODE}" = "metadata" ]; then
        [ -x "$1" ] && "$1" -c \
            'import ctranslate2, faster_whisper, funasr, numpy, soundfile, torch, transformers' \
            >/dev/null 2>&1
    else
        [ -x "$1" ] && "$1" -c \
            'import funasr, numpy, soundfile, torch, transformers' >/dev/null 2>&1
    fi
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

TAR_PATH="${1:-${PROJECT_ROOT}/test/test_4.tar}"
OUT_JSON="${2:-${PROJECT_ROOT}/output/nano_auto_lang/labels.json}"
GPU="${3:-0}"
ASR_BATCH="${4:-96}"
LID_BATCH="${5:-32}"
ASR_DTYPE="${ASR_DTYPE:-fp32}"
POST_KEEP_LANGS="${POST_KEEP_LANGS:-zh zh-en}"
VAD_WORKERS="${VAD_WORKERS:-4}"
VAD_THREADS="${VAD_THREADS:-8}"

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

LID_MODEL="${PROJECT_ROOT}/models/faster-whisper-tiny"
if [ "${LID_MODE}" = "metadata" ] && [ ! -f "${LID_MODEL}/model.bin" ]; then
    echo "[ERROR] Missing tiny LID model: ${LID_MODEL}/model.bin"
    exit 1
fi

mkdir -p "$(dirname "${OUT_JSON}")"
read -r -a POST_LANG_ARGS <<< "${POST_KEEP_LANGS}"

EXTRA_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then
    EXTRA_ARGS+=(--resume)
fi
if [ "${NO_TIMESTAMPS:-0}" = "1" ]; then
    EXTRA_ARGS+=(--no_timestamps)
fi
if [ -n "${ASR_SERVER_SOCKET:-}" ]; then
    EXTRA_ARGS+=(--asr_server_socket "${ASR_SERVER_SOCKET}")
fi

echo "[$(date '+%F %T')] === Nano auto-language pipeline (no vLLM / no pre-LID filter) ==="
echo "[$(date '+%F %T')] INPUT: ${TAR_PATH} GPU: ${GPU} ASR_BATCH: ${ASR_BATCH} LID_BATCH: ${LID_BATCH}"
echo "[$(date '+%F %T')] ASR_DTYPE: ${ASR_DTYPE} POST_KEEP_LANGS: ${POST_KEEP_LANGS}"
echo "[$(date '+%F %T')] VAD_WORKERS: ${VAD_WORKERS}"
echo "[$(date '+%F %T')] VAD_THREADS: ${VAD_THREADS}"
echo "[$(date '+%F %T')] LID_MODE: ${LID_MODE}"
echo "[$(date '+%F %T')] OUTPUT: ${OUT_JSON}"

"${PYTHON_BIN}" \
    "${PROJECT_ROOT}/pipeline/tar_pipeline_nodnsmos_nano_auto_lang_ckpt.py" \
    --tar_paths "${TAR_PATH}" \
    --out_json "${OUT_JSON}" \
    --funasr_model_dir "${PROJECT_ROOT}/models/Fun-ASR-Nano-2512" \
    --lid_model_dir "${LID_MODEL}" \
    --fireredvad_model "${PROJECT_ROOT}/models/FireRedVAD" \
    --fireredvad_root "${PROJECT_ROOT}/FireRedVAD" \
    --batch_size "${ASR_BATCH}" \
    --lid_batch_size "${LID_BATCH}" \
    --lid_mode "${LID_MODE}" \
    --lid_sample_seconds 12 \
    --bucket_edges 1.5 2 3 4 5 6 7 8 10 12 14 16 20 24 30 \
    --bucket_lookahead_batches 2 \
    --asr_prefetch_batches 64 \
    --vad_device cpu \
    --vad_workers "${VAD_WORKERS}" \
    --vad_threads "${VAD_THREADS}" \
    --min_dur 1.0 \
    --max_dur 30.0 \
    --asr_dtype "${ASR_DTYPE}" \
    --post_keep_langs "${POST_LANG_ARGS[@]}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "[$(date '+%F %T')] Complete: ${OUT_JSON}"
