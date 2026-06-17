#!/usr/bin/env bash
# =============================================================================
# run_fireredvad_funasr_vllm.sh — FireRedVAD + FunASR Nano vLLM 流水线（散装音频输入）
#
# 用法:
#   bash run_fireredvad_funasr_vllm.sh [INPUT_ROOT] [OUTPUT_ROOT] [GPU]
#
# 参数:
#   INPUT_ROOT   输入音频目录（默认: <项目目录>/test/audio）
#   OUTPUT_ROOT  输出目录（默认: <项目目录>/output/fireredvad_funasr_vllm）
#                输出结构: OUTPUT_ROOT/final_dataset/labels.json
#                          OUTPUT_ROOT/final_dataset/audio/
#   GPU          GPU 编号（默认: 2）
#
# 示例:
#   bash run_fireredvad_funasr_vllm.sh                          # 用默认参数跑测试
#   bash run_fireredvad_funasr_vllm.sh /data/wavs               # 指定输入目录
#   bash run_fireredvad_funasr_vllm.sh /data/wavs /data/output  # 指定输入+输出
#   bash run_fireredvad_funasr_vllm.sh /data/wavs /data/output 0  # 指定 GPU 0
#
# 前提:
#   conda activate funasr_vllm
# =============================================================================
set -euo pipefail

# 使用当前激活的 conda 环境（conda activate funasr_vllm 后生效）
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "[ERROR] 请先 conda activate funasr_vllm"; exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 模型路径（全部相对 PROJECT_ROOT）
FUNASR_MODEL="${PROJECT_ROOT}/models/Fun-ASR-Nano-2512"
LID_MODEL="${PROJECT_ROOT}/models/faster-whisper-large-v3"
FIREREDVAD_MODEL="${PROJECT_ROOT}/models/FireRedVAD"
FIREREDVAD_ROOT="${PROJECT_ROOT}/FireRedVAD"
DNSMOS_DIR="${PROJECT_ROOT}/dns_mos"

# ===== 按需修改（或命令行传参） =====
INPUT_ROOT="${1:-${PROJECT_ROOT}/test/audio}"                           # 第1个参数：输入音频目录
OUTPUT_ROOT="${2:-${PROJECT_ROOT}/output/fireredvad_funasr_vllm}"       # 第2个参数：输出目录
GPU="${3:-2}"                                                            # 第3个参数：GPU 编号
# =====================================

OUT_JSON="${OUTPUT_ROOT}/final_dataset/labels.json"
AUDIO_DIR="${OUTPUT_ROOT}/final_dataset/audio"

log() { echo "[$(date '+%F %T')] $*"; }

# LD_LIBRARY_PATH：从 CONDA_PREFIX 自动补全 nvidia 库路径
_NV="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia"
if [ -d "$_NV" ]; then
  for _d in "$_NV"/*/lib; do
    [ -d "$_d" ] && LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"
  done
  export LD_LIBRARY_PATH
fi
unset _NV _d

export MODELSCOPE_CACHE="${PROJECT_ROOT}/models"
export CUDA_VISIBLE_DEVICES="${GPU}"

# 清理旧输出
[[ -d "$OUTPUT_ROOT" ]] && rm -rf "$OUTPUT_ROOT"
mkdir -m 777 -p "$AUDIO_DIR"

log "=== FireRedVAD + FunASR Nano vLLM (流水线版) ==="
log "INPUT: ${INPUT_ROOT}  OUTPUT: ${OUTPUT_ROOT}  GPU: ${GPU}"

"${CONDA_PREFIX}/bin/python" "${PROJECT_ROOT}/pipeline/streaming_pipeline.py" \
    --input_root        "${INPUT_ROOT}" \
    --out_json          "${OUT_JSON}" \
    --audio_dir         "${AUDIO_DIR}" \
    --vad_model         fireredvad \
    --asr_model         funasr_vllm \
    --funasr_model_dir  "${FUNASR_MODEL}" \
    --lid_model_dir     "${LID_MODEL}" \
    --fireredvad_model  "${FIREREDVAD_MODEL}" \
    --fireredvad_root   "${FIREREDVAD_ROOT}" \
    --dnsmos_dir        "${DNSMOS_DIR}" \
    --gpus              "${GPU}" \
    --vllm_batch_size   64 \
    --min_dur           1.0 \
    --max_dur           30.0 \
    --min_mos_ovr       2.0 \
    --min_mos_sig       2.0 \
    --min_mos_bak       3.5 \
    --target_langs      en zh \
    --min_lang_prob     0.90 \
    --queue_maxsize     200

chmod -R 777 "$OUTPUT_ROOT"
log "完成！labels: ${OUT_JSON}"
