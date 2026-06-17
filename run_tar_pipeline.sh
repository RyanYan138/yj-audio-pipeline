#!/usr/bin/env bash
# =============================================================================
# run_tar_pipeline.sh — FireRedVAD + FunASR Nano vLLM 流水线（tar包输入）
#
# 用法:
#   bash run_tar_pipeline.sh [TAR_PATH] [OUT_JSON] [GPU]
#
# 参数:
#   TAR_PATH   输入 tar 包路径（默认: <项目目录>/test/test_4.tar）
#   OUT_JSON   输出 labels.json 路径（默认: <项目目录>/output/tar_pipeline/labels.json）
#   GPU        GPU 编号（默认: 2）
#
# 示例:
#   bash run_tar_pipeline.sh                              # 用默认参数跑测试
#   bash run_tar_pipeline.sh /data/my.tar                 # 指定输入
#   bash run_tar_pipeline.sh /data/my.tar /data/out.json  # 指定输入+输出
#   bash run_tar_pipeline.sh /data/my.tar /data/out.json 0  # 指定 GPU 0
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
PIPELINE_DIR="${PROJECT_ROOT}/pipeline"

# 模型路径（全部相对 PROJECT_ROOT）
FUNASR_MODEL="${PROJECT_ROOT}/models/Fun-ASR-Nano-2512"
LID_MODEL="${PROJECT_ROOT}/models/faster-whisper-large-v3"
FIREREDVAD_MODEL="${PROJECT_ROOT}/models/FireRedVAD"
FIREREDVAD_ROOT="${PROJECT_ROOT}/FireRedVAD"
DNSMOS_DIR="${PROJECT_ROOT}/dns_mos"

# ===== 按需修改（或命令行传参） =====
TAR_PATH="${1:-${PROJECT_ROOT}/test/test_4.tar}"          # 第1个参数：输入 tar 包
OUT_JSON="${2:-${PROJECT_ROOT}/output/tar_pipeline/labels.json}"  # 第2个参数：输出路径
GPU="${3:-2}"                                              # 第3个参数：GPU 编号
# =====================================

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
export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"

mkdir -m 777 -p "$(dirname "${OUT_JSON}")"

echo "[$(date '+%F %T')] === tar pipeline 开始 ==="
echo "[$(date '+%F %T')] INPUT: ${TAR_PATH}"
echo "[$(date '+%F %T')] OUTPUT: ${OUT_JSON}  GPU: ${GPU}"

"${CONDA_PREFIX}/bin/python" "${PIPELINE_DIR}/tar_pipeline.py" \
    --tar_paths         "${TAR_PATH}" \
    --out_json          "${OUT_JSON}" \
    --funasr_model_dir  "${FUNASR_MODEL}" \
    --lid_model_dir     "${LID_MODEL}" \
    --fireredvad_model  "${FIREREDVAD_MODEL}" \
    --fireredvad_root   "${FIREREDVAD_ROOT}" \
    --dnsmos_dir        "${DNSMOS_DIR}" \
    --gpu               "${GPU}" \
    --batch_size        64 \
    --min_dur           1.0 \
    --max_dur           30.0 \
    --min_mos_ovr       2.0 \
    --min_mos_sig       2.0 \
    --min_mos_bak       3.5 \
    --target_langs      en zh \
    --min_lang_prob     0.90

chmod -R 777 "$(dirname "${OUT_JSON}")"
echo "[$(date '+%F %T')] 完成！输出: ${OUT_JSON}"
