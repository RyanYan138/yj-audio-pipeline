#!/usr/bin/env bash
#
# 用法:
#   bash run_tar_pipeline_nodnsmos_bucket_batch_ckpt.sh INPUT.tar OUTPUT.json GPU
# 可选位置参数: ASR_BATCH TARGET_LANG LID_BATCH MIN_LANG_PROB

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "[ERROR] 请先 conda activate <环境>"; exit 1
fi
# python 通过 CONDA_PREFIX 动态解析

TAR_PATH="${1:-${PROJECT_ROOT}/test/audio_200.tar}"
OUT_JSON="${2:-${PROJECT_ROOT}/output/ad200_tar_pipeline_bucket_batch/labels.json}"
GPU="${3:-0}"
ASR_BATCH="${4:-8}"
TARGET_LANG="${5:-en}"
LID_BATCH="${6:-32}"
MIN_LANG_PROB="${7:-0.90}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/pipeline:${PYTHONPATH:-}"

for _NV in "${CONDA_PREFIX}"/lib/python3.*/site-packages/nvidia; do
  if [ -d "$_NV" ]; then
    for _d in "$_NV"/*/lib; do
      [ -d "$_d" ] && LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"
    done
  fi
done
export LD_LIBRARY_PATH

mkdir -p "$(dirname "${OUT_JSON}")"

LID_MODEL="${PROJECT_ROOT}/models/faster-whisper-tiny"
if [ ! -f "${LID_MODEL}/model.bin" ]; then
    echo "[ERROR] 缺少 tiny LID 模型: ${LID_MODEL}/model.bin"
    exit 1
fi

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

echo "[$(date '+%F %T')] === bucket batch pipeline (无 vLLM / 无 DNSMOS) ==="
echo "[$(date '+%F %T')] INPUT: ${TAR_PATH}  GPU: ${GPU}  ASR_BATCH: ${ASR_BATCH}  LID_BATCH: ${LID_BATCH}"
echo "[$(date '+%F %T')] LID: ${TARGET_LANG}  MIN_LANG_PROB: ${MIN_LANG_PROB}"
echo "[$(date '+%F %T')] OUTPUT: ${OUT_JSON}"

"${CONDA_PREFIX}/bin/python" "${PROJECT_ROOT}/pipeline/tar_pipeline_nodnsmos_bucket_batch_ckpt.py" \
    --tar_paths         "${TAR_PATH}" \
    --out_json          "${OUT_JSON}" \
    --funasr_model_dir  "${PROJECT_ROOT}/models/Fun-ASR-Nano-2512" \
    --lid_model_dir     "${LID_MODEL}" \
    --fireredvad_model  "${PROJECT_ROOT}/models/FireRedVAD" \
    --fireredvad_root   "${PROJECT_ROOT}/FireRedVAD" \
    --batch_size        "${ASR_BATCH}" \
    --lid_batch_size    "${LID_BATCH}" \
    --lid_sample_seconds 12 \
    --bucket_edges      1.5 2 3 4 5 6 7 8 10 12 14 16 20 24 30 \
    --bucket_lookahead_batches 4 \
    --vad_device        cpu \
    --min_dur           1.0 \
    --max_dur           30.0 \
    --target_langs      "${TARGET_LANG}" \
    --min_lang_prob     "${MIN_LANG_PROB}" \
    --force_language_from_lid \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "[$(date '+%F %T')] 完成！输出: ${OUT_JSON}"
