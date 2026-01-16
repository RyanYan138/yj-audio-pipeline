#!/usr/bin/env bash
set -euo pipefail

############################
# 你只需要改这里（必填/常改）
############################

# 项目根目录
export PIPE_ROOT="/CDShare3/Huawei_Encoder_Proj/codes/jiahao/Yj_Pipeline"

# 这次 run 的输出根目录（强烈建议每次 run 一个新目录）
# 例如：export RUN_ID="$(date +%Y%m%d_%H%M%S)_devclean"
export RUN_ID="devclean_run1"
export OUT_DIR="${PIPE_ROOT}/runs/${RUN_ID}"

# 输入数据（根据你的 vad_pipeline.py 需要什么来填）
# 如果 vad_pipeline.py 本身写死了输入，这里可以先不填
export INPUT_ROOT="/CDShare3/Huawei_Encoder_Proj/datas/LibriSpeech/dev-other"

# Whisper 模型目录
export WHISPER_MODEL_DIR="/Work21/2025/yanjiahao/modelscope_cache/models/AI-ModelScope/whisper-large-v3"

# GPU 配置
export GPUS=(0 1)          # DNSMOS / Whisper 用
export BATCH_SIZE=16       # Whisper batch
export MAX_FILES=""        # 调试：比如 500；全量就留空 ""

# 语言（已筛成 en 就固定 en）
export WHISPER_LANG="en"

############################
# 环境/缓存（强烈建议统一）
############################
export HF_HOME="/Work21/2025/yanjiahao/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${HF_HOME}"
export HOME="${HF_HOME}/home"
mkdir -p "${HF_HOME}" "${HOME}"

############################
# conda（按你真实路径改）
############################
export CONDA_SOURCE="/Work21/2025/yanjiahao/miniconda3/etc/profile.d/conda.sh"

# 各步骤 env（你按实际 env 名称/路径填）
export ENV_VAD="/Work21/2025/yanjiahao/conda-envs/vad_env"
export ENV_SPK="/Work21/2025/yanjiahao/conda-envs/spk_consistency"
export ENV_LANG="/Work21/2025/yanjiahao/conda-envs/lang_id"
export ENV_ASR="/Work21/2025/yanjiahao/conda-envs/asr_whisper"

############################
# 各 step 的输入/输出路径约定
############################
export STEP01_VAD_DIR="${OUT_DIR}/01_vad"
export STEP02_DNSMOS_DIR="${OUT_DIR}/02_dnsmos"
export STEP03_DNSFILT_DIR="${OUT_DIR}/03_dnsmos_filter"
export STEP04_SPK_DIR="${OUT_DIR}/04_spk"
export STEP05_LANG_DIR="${OUT_DIR}/05_lang"
export STEP06_ASR_DIR="${OUT_DIR}/06_asr"
export LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${STEP01_VAD_DIR}" "${STEP02_DNSMOS_DIR}" "${STEP03_DNSFILT_DIR}" \
         "${STEP04_SPK_DIR}" "${STEP05_LANG_DIR}" "${STEP06_ASR_DIR}" "${LOG_DIR}"

# 关键中间产物文件名（后续脚本都用这些变量）
export VAD_JSON="${STEP01_VAD_DIR}/segments.json"

export DNSMOS_ALL_TSV="${STEP02_DNSMOS_DIR}/dns_from_vad_all.json_order.tsv"

export DNSMOS_FILTERED_TSV="${STEP03_DNSFILT_DIR}/dns_filtered.tsv"

export SPK_FILTERED_TSV="${STEP04_SPK_DIR}/spk_filtered.tsv"

export LANG_FILTERED_TSV="${STEP05_LANG_DIR}/lang_filtered.tsv"
export LANG_SEG_JSON="${STEP05_LANG_DIR}/lang_filtered_segments.json"

export WHISPER_PREFIX="${STEP06_ASR_DIR}/whisper_lv3"
export WHISPER_ALL_JSON="${WHISPER_PREFIX}_all.json"

############################
# 工具函数（脚本里直接用）
############################
activate_env () {
  local env_path="$1"
  # shellcheck disable=SC1090
  source "${CONDA_SOURCE}"
  conda activate "${env_path}"
}

# 把 GPUS=(0 1 3) 转成 "0 1 3"
gpu_list_str () {
  echo "${GPUS[*]}"
}
