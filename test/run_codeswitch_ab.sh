#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:-3}"
TAR_PATH="${2:-${PROJECT_ROOT}/data/codeswitch/benchmark_450/audio.tar}"
REFERENCES="${3:-${PROJECT_ROOT}/data/codeswitch/benchmark_450/references.jsonl}"
OUT_DIR="${4:-${PROJECT_ROOT}/output/codeswitch_ab_20260817}"
SOCKET_PATH="${ASR_SERVER_SOCKET:-/tmp/funasr_nano_gpu3_auto_ab.sock}"

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "[ERROR] Activate the FunASR conda environment first."
    exit 1
fi

mkdir -p "${OUT_DIR}/baseline" "${OUT_DIR}/auto"
export ASR_SERVER_SOCKET="${SOCKET_PATH}"

bash "${PROJECT_ROOT}/run_tar_pipeline_nodnsmos_bucket_batch_ckpt.sh" \
    "${TAR_PATH}" "${OUT_DIR}/baseline/labels.json" \
    "${GPU}" 16 zh 32 0.90 \
    > "${OUT_DIR}/baseline.log" 2>&1

LID_MODE=metadata VAD_WORKERS=1 VAD_THREADS=8 \
bash "${PROJECT_ROOT}/run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh" \
    "${TAR_PATH}" "${OUT_DIR}/auto/labels.json" \
    "${GPU}" 16 32 \
    > "${OUT_DIR}/auto.log" 2>&1

"${CONDA_PREFIX}/bin/python" "${PROJECT_ROOT}/test/evaluate_codeswitch_ab.py" \
    --references "${REFERENCES}" \
    --baseline-labels "${OUT_DIR}/baseline/labels.json" \
    --auto-labels "${OUT_DIR}/auto/labels.all.json" \
    --baseline-metrics "${OUT_DIR}/baseline/metrics.json" \
    --auto-metrics "${OUT_DIR}/auto/metrics.json" \
    --output-json "${OUT_DIR}/codeswitch_ab_report.json" \
    --output-md "${OUT_DIR}/codeswitch_ab_report.md" \
    > "${OUT_DIR}/evaluate.log" 2>&1

date -Iseconds > "${OUT_DIR}/DONE"
