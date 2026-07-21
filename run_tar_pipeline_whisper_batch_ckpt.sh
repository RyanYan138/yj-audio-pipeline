#!/usr/bin/env bash
# =============================================================================
# run_tar_pipeline_whisper_batch_ckpt.sh
# FireRedVAD + DNSMOS + LID(batch) + ASR(batch) 五阶段全流水线
#
# 用法:
#   bash run_tar_pipeline_whisper_batch_ckpt.sh [TAR] [OUT_JSON] [GPU] [ASR_BATCH] [LID_BATCH]
#
# 参数说明:
#   GPU         GPU 编号（默认 0）
#   ASR_BATCH   ASR batch size（默认 16，RTX 4090 建议 32~64，V100 建议 8~16）
#   LID_BATCH   LID batch size（默认 16，可与 ASR_BATCH 一致）
#
# 前提:
#   conda activate faster_whisper
# =============================================================================
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "[ERROR] 请先 conda activate faster_whisper"; exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${PROJECT_ROOT}/pipeline"

WHISPER_MODEL="${PROJECT_ROOT}/models/faster-whisper-large-v3"
FIREREDVAD_MODEL="${PROJECT_ROOT}/models/FireRedVAD"
FIREREDVAD_ROOT="${PROJECT_ROOT}/FireRedVAD"
DNSMOS_DIR="${PROJECT_ROOT}/dns_mos"

# ===== 参数（可命令行传入） =====
TAR_PATH="${1:-${PROJECT_ROOT}/test/audio_200.tar}"
OUT_JSON="${2:-${PROJECT_ROOT}/output/tar_pipeline_whisper_batch/labels2.json}"
GPU="${3:-0}"
ASR_BATCH="${4:-32}"   # ASR batch size（4090 建议 32，V100 建议 8~16）
LID_BATCH="${5:-16}"   # LID batch size（可与 ASR_BATCH/2 对齐）
# ================================

# LD_LIBRARY_PATH 动态补全
for _NV in "${CONDA_PREFIX}"/lib/python3.*/site-packages/nvidia; do
  [ -d "$_NV" ] || continue
  for _d in "$_NV"/*/lib; do
    [ -d "$_d" ] && LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"
  done
done
for _CT in "${CONDA_PREFIX}"/lib/python3.*/site-packages/ctranslate2.libs; do
  [ -d "$_CT" ] && LD_LIBRARY_PATH="${_CT}:${LD_LIBRARY_PATH:-}"
done
export LD_LIBRARY_PATH
unset _NV _d _CT

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"

mkdir -p "$(dirname "${OUT_JSON}")"

echo "[$(date '+%F %T')] === tar pipeline whisper batch 开始 ==="
echo "[$(date '+%F %T')] INPUT: ${TAR_PATH}  GPU: ${GPU}"
echo "[$(date '+%F %T')] ASR_BATCH: ${ASR_BATCH}  LID_BATCH: ${LID_BATCH}"
echo "[$(date '+%F %T')] OUTPUT: ${OUT_JSON}"

"${CONDA_PREFIX}/bin/python" "${PIPELINE_DIR}/tar_pipeline_whisper_batch_ckpt.py" \
    --tar_paths         "${TAR_PATH}" \
    --out_json          "${OUT_JSON}" \
    --whisper_model_dir "${WHISPER_MODEL}" \
    --fireredvad_model  "${FIREREDVAD_MODEL}" \
    --fireredvad_root   "${FIREREDVAD_ROOT}" \
    --dnsmos_dir        "${DNSMOS_DIR}" \
    --gpu               "${GPU}" \
    --batch_size        "${ASR_BATCH}" \
    --lid_batch_size    "${LID_BATCH}" \
    --min_dur           1.0 \
    --max_dur           30.0 \
    --min_mos_ovr       2.0 \
    --min_mos_sig       2.0 \
    --min_mos_bak       3.5 \
    --target_langs      en zh \
    --min_lang_prob     0.90 \
    --resume

echo "[$(date '+%F %T')] 完成！输出: ${OUT_JSON}"
