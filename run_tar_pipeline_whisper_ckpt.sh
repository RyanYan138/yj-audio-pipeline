#!/usr/bin/env bash
# =============================================================================
# run_tar_pipeline_whisper_ckpt.sh — FireRedVAD + DNSMOS + faster-whisper 流水线
#                                    （断点续跑，路径动态解析，可跨集群直接用）
#
# 用法:
#   bash run_tar_pipeline_whisper_ckpt.sh [TAR_PATH] [OUT_JSON] [GPU]
#
# 前提:
#   conda activate <含 faster-whisper/onnxruntime/fireredvad 的环境>
#   （106集群验证环境: funasr_vllm / yj_pipeline）
# =============================================================================
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "[ERROR] 请先 conda activate <环境>"; exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${PROJECT_ROOT}/pipeline"

WHISPER_MODEL="${PROJECT_ROOT}/models/faster-whisper-large-v3"
FIREREDVAD_MODEL="${PROJECT_ROOT}/models/FireRedVAD"
FIREREDVAD_ROOT="${PROJECT_ROOT}/FireRedVAD"
DNSMOS_DIR="${PROJECT_ROOT}/dns_mos"

# ===== 按需修改（或命令行传参） =====
TAR_PATH="${1:-${PROJECT_ROOT}/test/test_4.tar}"
OUT_JSON="${2:-${PROJECT_ROOT}/output/tar_pipeline_whisper_ckpt/labels.json}"
GPU="${3:-0}"
# =====================================

# LD_LIBRARY_PATH：从 CONDA_PREFIX 自动补全 nvidia/ctranslate2 库路径
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

echo "[$(date '+%F %T')] === tar pipeline whisper ckpt 开始 ==="
echo "[$(date '+%F %T')] INPUT: ${TAR_PATH}  GPU: ${GPU}"
echo "[$(date '+%F %T')] OUTPUT: ${OUT_JSON}"

"${CONDA_PREFIX}/bin/python" "${PIPELINE_DIR}/tar_pipeline_whisper_ckpt.py" \
    --tar_paths         "${TAR_PATH}" \
    --out_json          "${OUT_JSON}" \
    --whisper_model_dir "${WHISPER_MODEL}" \
    --fireredvad_model  "${FIREREDVAD_MODEL}" \
    --fireredvad_root   "${FIREREDVAD_ROOT}" \
    --dnsmos_dir        "${DNSMOS_DIR}" \
    --gpu               "${GPU}" \
    --batch_size        32 \
    --min_dur           1.0 \
    --max_dur           30.0 \
    --min_mos_ovr       2.0 \
    --min_mos_sig       2.0 \
    --min_mos_bak       3.5 \
    --target_langs      en zh \
    --min_lang_prob     0.90 \
    --resume

echo "[$(date '+%F %T')] 完成！输出: ${OUT_JSON}"
