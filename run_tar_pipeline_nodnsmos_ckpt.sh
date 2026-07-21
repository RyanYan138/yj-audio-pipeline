#!/usr/bin/env bash
#
# 示例:
#   bash run_tar_pipeline_nodnsmos_ckpt.sh /data/my.tar /data/out.json 2

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "[ERROR] 请先 conda activate <环境>"; exit 1
fi
# python 通过 CONDA_PREFIX 动态解析

TAR_PATH="${1:-${PROJECT_ROOT}/test/test_4.tar}"
OUT_JSON="${2:-${PROJECT_ROOT}/output/tar_pipeline_nodnsmos_ckpt/labels.json}"
GPU="${3:-0}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PROJECT_ROOT}/pipeline:${PYTHONPATH:-}"

_NV="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia"
if [ -d "$_NV" ]; then
  for _d in "$_NV"/*/lib; do
    [ -d "$_d" ] && LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"
  done
  export LD_LIBRARY_PATH
fi

mkdir -p "$(dirname "${OUT_JSON}")"

echo "[$(date '+%F %T')] === tar pipeline nodnsmos ckpt (无DNSMOS) 开始 ==="
echo "[$(date '+%F %T')] INPUT: ${TAR_PATH}  GPU: ${GPU}"
echo "[$(date '+%F %T')] OUTPUT: ${OUT_JSON}"

"${CONDA_PREFIX}/bin/python" "${PROJECT_ROOT}/pipeline/tar_pipeline_nodnsmos_ckpt.py" \
    --tar_paths         "${TAR_PATH}" \
    --out_json          "${OUT_JSON}" \
    --funasr_model_dir  "${PROJECT_ROOT}/models/Fun-ASR-Nano-2512" \
    --lid_model_dir     "${PROJECT_ROOT}/models/faster-whisper-large-v3" \
    --fireredvad_model  "${PROJECT_ROOT}/models/FireRedVAD" \
    --fireredvad_root   "${PROJECT_ROOT}/FireRedVAD" \
    --gpu               "${GPU}" \
    --batch_size        16 \
    --min_dur           1.0 \
    --max_dur           30.0 \
    --target_langs      zh \
    --min_lang_prob     0.90 \

echo "[$(date '+%F %T')] 完成！输出: ${OUT_JSON}"
