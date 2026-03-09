#!/usr/bin/env bash
set -euo pipefail

############################################
# YJ 音频清洗流程
#
# 使用的镜像：
#   1) yj-pipeline-runtime:with-spk   -> VAD / 合并 / 过滤 / LID / Whisper
#   2) dnsmos_gpu:cuda118            -> DNSMOS 打分
#
# 流程：
#   1) VAD 切段
#   2) DNSMOS 打分 + 合并
#   3) DNSMOS 过滤
#   4) LID 过滤 + TSV 转 segments.json
#   5) Whisper 多卡转写 + 合并
############################################

############################
# 1）用户主要配置区（优先改这里）
############################

# 是否强制重跑：1=不管输出是否存在都重跑；0=有输出就跳过
FORCE=1

# 项目根目录（代码目录）
PROJECT_ROOT="/CDShare3/Huawei_Encoder_Proj/codes/jiahao/Yj_Pipeline"

# 需要挂载进容器的大目录，通常尽量往大了挂
MOUNT_ROOT="/CDShare3"

# 本次实验/输出名称
DATASET_NAME="t3"

# 输入音频根目录（可以是一个目录）
INPUT_ROOT="/CDShare3/Huawei_Encoder_Proj/datas/LibriSpeech/dev-clean/84"

# 主流程镜像：用于 VAD / 合并 / 过滤 / LID / Whisper
RUNTIME_IMAGE="yj-pipeline-runtime:with-spk"

# DNSMOS 专用镜像
DNSMOS_DOCKER_IMAGE="dnsmos_gpu:cuda118"

# 主流程用到的 GPU（VAD 本身通常不需要 GPU；Whisper 用这里）
PIPELINE_GPUS=(0)

# DNSMOS 打分用到的 GPU
DNSMOS_GPUS=(0)

# LID 单独绑定的 GPU
LID_GPU=0

# 宿主机上的缓存/模型目录（会原路径挂载进容器）
HF_HOME_HOST="/Work21/2025/yanjiahao/hf_cache"
TORCH_HOME_HOST="/Work21/2025/yanjiahao/torch_cache"
MODELSCOPE_CACHE_HOST="/Work21/2025/yanjiahao/modelscope_cache"
WHISPER_MODEL_DIR_HOST="/Work21/2025/yanjiahao/modelscope_cache/models/AI-ModelScope/whisper-large-v3"

# 各步骤开关：1=执行，0=跳过
DO_VAD=1
DO_DNSMOS=1
DO_DNSMOS_FILTER=1
DO_LID=1
DO_WHISPER=1

############################
# 2）进阶配置区（一般不用改）
############################

# 输出总目录
OUTPUT_ROOT="${PROJECT_ROOT}/output/${DATASET_NAME}"

# ===== VAD =====
VAD_PIPELINE_PY="${PROJECT_ROOT}/vad/vad_pipeline.py"
VAD_OUT_JSON="${OUTPUT_ROOT}/vad_output/${DATASET_NAME}_silero_vad_segments_mp_Ordered.json"
VAD_MIN_DUR="5.0"
VAD_NUM_WORKERS="32"
VAD_MAX_FILES="400"

# ===== DNSMOS =====
DNSMOS_DIR="${PROJECT_ROOT}/dns_mos"
DNSMOS_SAVE_HOME="${OUTPUT_ROOT}/dnsmos_output"
DNSMOS_INPUT_LENGTH="9"
DNSMOS_MIN_DUR="1.0"
DNSMOS_MERGED_TSV="${DNSMOS_SAVE_HOME}/dns_from_vad_all.json_order.tsv"

# ===== DNSMOS 过滤 =====
DNSMOS_FILTERED_TSV="${DNSMOS_SAVE_HOME}/dns_filtered_emilia.tsv"
DNSMOS_FILTER_MIN_DUR="3.0"
DNSMOS_FILTER_MAX_DUR="30.0"
DNSMOS_MIN_MOS_OVR="2.25"
DNSMOS_MIN_MOS_SIG="2.2"
DNSMOS_MIN_MOS_BAK="3.8"
DNSMOS_KEEP_QUANTILE=""   # 留空表示不用 quantile

# ===== 语言识别 LID =====
LANG_DIR="${PROJECT_ROOT}/language_filter"
LANG_OUT_SCORES="${DNSMOS_SAVE_HOME}/lang_scores_all.tsv"
LANG_OUT_FILTERED="${DNSMOS_SAVE_HOME}/lang_filtered.tsv"
LANG_CACHE_DB="${DNSMOS_SAVE_HOME}/lang_cache.sqlite"
LANG_MODEL_ID="Systran/faster-whisper-large-v3"
LANG_DEVICE="cuda"
LANG_COMPUTE_TYPE="float16"
LANG_TARGET_LANGS=("en" "zh")
LANG_MIN_PROB="0.90"
TSV2SEG_PY="${LANG_DIR}/tsv_to_segments_json.py"
LANG_SEG_JSON="${DNSMOS_SAVE_HOME}/lang_filtered_segments.json"

# ===== Whisper 转写 =====
ASR_DIR="${PROJECT_ROOT}/asr"
WHISPER_OUT_PREFIX="${OUTPUT_ROOT}/whisper_lv3_output/${DATASET_NAME}"
WHISPER_BATCH_SIZE="16"
WHISPER_MAX_FILES=""

############################
# 3）通用函数
############################

# 打印带时间戳的日志
log() {
  echo "[$(date '+%F %T')] $*"
}

# 检查文件是否存在
need_file() {
  local f="$1"
  [[ -f "$f" ]] || { echo "ERROR: 文件不存在: $f" >&2; exit 1; }
}

# 如果输出已存在且 FORCE=0，则跳过
maybe_skip() {
  local out="$1"
  local name="$2"
  if [[ "$FORCE" -eq 0 && -e "$out" ]]; then
    log "[SKIP] ${name}: ${out}"
    return 0
  fi
  return 1
}

# 创建运行所需目录
ensure_dirs() {
  mkdir -p \
    "${OUTPUT_ROOT}/vad_output" \
    "${OUTPUT_ROOT}/whisper_lv3_output" \
    "${DNSMOS_SAVE_HOME}" \
    "${HF_HOME_HOST}" \
    "${TORCH_HOME_HOST}" \
    "${MODELSCOPE_CACHE_HOST}"
}

# 运行主流程镜像
# 参数：
#   1) GPU 规格，例如 "" / "device=0" / "all"
#   2) 要执行的 bash 命令
docker_run_runtime() {
  local gpu_spec="$1"
  local cmd="$2"

  if [[ -n "${gpu_spec}" ]]; then
    docker run --rm \
      --gpus "${gpu_spec}" \
      --ipc=host \
      -u "$(id -u):$(id -g)" \
      -v "${MOUNT_ROOT}:${MOUNT_ROOT}" \
      -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
      -v "${HF_HOME_HOST}:${HF_HOME_HOST}" \
      -v "${TORCH_HOME_HOST}:${TORCH_HOME_HOST}" \
      -v "${MODELSCOPE_CACHE_HOST}:${MODELSCOPE_CACHE_HOST}" \
      -w "${PROJECT_ROOT}" \
      -e HF_HOME="${HF_HOME_HOST}" \
      -e TORCH_HOME="${TORCH_HOME_HOST}" \
      -e TRANSFORMERS_CACHE="${HF_HOME_HOST}" \
      -e HUGGINGFACE_HUB_CACHE="${HF_HOME_HOST}/hub" \
      -e MODELSCOPE_CACHE="${MODELSCOPE_CACHE_HOST}" \
      "${RUNTIME_IMAGE}" \
      bash -lc "${cmd}"
  else
    docker run --rm \
      --ipc=host \
      -u "$(id -u):$(id -g)" \
      -v "${MOUNT_ROOT}:${MOUNT_ROOT}" \
      -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
      -v "${HF_HOME_HOST}:${HF_HOME_HOST}" \
      -v "${TORCH_HOME_HOST}:${TORCH_HOME_HOST}" \
      -v "${MODELSCOPE_CACHE_HOST}:${MODELSCOPE_CACHE_HOST}" \
      -w "${PROJECT_ROOT}" \
      -e HF_HOME="${HF_HOME_HOST}" \
      -e TORCH_HOME="${TORCH_HOME_HOST}" \
      -e TRANSFORMERS_CACHE="${HF_HOME_HOST}" \
      -e HUGGINGFACE_HUB_CACHE="${HF_HOME_HOST}/hub" \
      -e MODELSCOPE_CACHE="${MODELSCOPE_CACHE_HOST}" \
      "${RUNTIME_IMAGE}" \
      bash -lc "${cmd}"
  fi
}

# 运行 DNSMOS 镜像
# 参数：
#   1) 要执行的 bash 命令
docker_run_dnsmos() {
  local cmd="$1"

  docker run --rm \
    --gpus all \
    --ipc=host \
    -u "$(id -u):$(id -g)" \
    -v "${MOUNT_ROOT}:${MOUNT_ROOT}" \
    -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -w "${DNSMOS_DIR}" \
    "${DNSMOS_DOCKER_IMAGE}" \
    bash -lc "${cmd}"
}

ensure_dirs

############################
# 4）Step 1：VAD 切段
############################

if [[ "$DO_VAD" -eq 1 ]]; then
  need_file "${VAD_PIPELINE_PY}"

  if ! maybe_skip "${VAD_OUT_JSON}" "VAD"; then
    log "[RUN] VAD -> ${VAD_OUT_JSON}"

    VAD_CMD="python3 '${VAD_PIPELINE_PY}' \
      --input_root '${INPUT_ROOT}' \
      --out_json '${VAD_OUT_JSON}' \
      --min_dur '${VAD_MIN_DUR}' \
      --num_workers '${VAD_NUM_WORKERS}'"

    if [[ -n "${VAD_MAX_FILES}" ]]; then
      VAD_CMD="${VAD_CMD} --max_files '${VAD_MAX_FILES}'"
    fi

    docker_run_runtime "" "${VAD_CMD}"
  fi
fi

############################
# 5）Step 2：DNSMOS 打分 + 合并
############################

if [[ "$DO_DNSMOS" -eq 1 ]]; then
  need_file "${VAD_OUT_JSON}"

  if ! maybe_skip "${DNSMOS_MERGED_TSV}" "DNSMOS+MERGE"; then
    local_num_gpus=${#DNSMOS_GPUS[@]}
    GPUS_STR=$(IFS=,; echo "${DNSMOS_GPUS[*]}")

    # 如果 shard 文件已经齐全，则不重新打分，只做 merge
    shopt -s nullglob
    SHARDS=( "${DNSMOS_SAVE_HOME}/dns_from_vad_"*"_${local_num_gpus}.tsv" )
    shopt -u nullglob

    if [[ ${#SHARDS[@]} -ne ${local_num_gpus} ]]; then
      log "[RUN] DNSMOS 打分..."

      DNSMOS_CMD="
        set -euo pipefail
        export VAD_JSON='${VAD_OUT_JSON}'
        export SAVE_HOME='${DNSMOS_SAVE_HOME}'
        export GPUS_STR='${GPUS_STR}'
        export NUM_GPUS='${local_num_gpus}'
        IFS=',' read -r -a GPUS <<< \"\$GPUS_STR\"

        for IDX in \"\${!GPUS[@]}\"; do
          GPU=\${GPUS[\$IDX]}
          CUDA_VISIBLE_DEVICES=\$GPU python3 eval_dns_from_vad_json.py \
            --vad-json \"\$VAD_JSON\" \
            --np \"\$NUM_GPUS\" --idx \"\$IDX\" \
            --input-length '${DNSMOS_INPUT_LENGTH}' \
            --min-dur '${DNSMOS_MIN_DUR}' \
            --save-home \"\$SAVE_HOME\" &
        done

        wait
      "

      docker_run_dnsmos "${DNSMOS_CMD}"
    else
      log "[SKIP] DNSMOS 打分：已有 shard 文件"
    fi

    log "[RUN] Merge DNSMOS shards -> ${DNSMOS_MERGED_TSV}"
    docker_run_runtime "" \
      "python3 '${DNSMOS_DIR}/merge_dns_by_json_order.py' \
        --vad-json '${VAD_OUT_JSON}' \
        --shard-dir '${DNSMOS_SAVE_HOME}' \
        --output '${DNSMOS_MERGED_TSV}'"
  fi
fi

############################
# 6）Step 3：DNSMOS 过滤
############################

if [[ "$DO_DNSMOS_FILTER" -eq 1 ]]; then
  need_file "${DNSMOS_MERGED_TSV}"

  if ! maybe_skip "${DNSMOS_FILTERED_TSV}" "DNSMOS_FILTER"; then
    log "[RUN] DNSMOS 过滤 -> ${DNSMOS_FILTERED_TSV}"

    FILTER_CMD="python3 '${DNSMOS_DIR}/filter_dnsmos_tsv.py' \
      --in_tsv '${DNSMOS_MERGED_TSV}' \
      --out_tsv '${DNSMOS_FILTERED_TSV}' \
      --min_dur '${DNSMOS_FILTER_MIN_DUR}' \
      --max_dur '${DNSMOS_FILTER_MAX_DUR}' \
      --min_mos_ovr '${DNSMOS_MIN_MOS_OVR}' \
      --min_mos_sig '${DNSMOS_MIN_MOS_SIG}' \
      --min_mos_bak '${DNSMOS_MIN_MOS_BAK}'"

    if [[ -n "${DNSMOS_KEEP_QUANTILE}" ]]; then
      FILTER_CMD="${FILTER_CMD} --keep_quantile '${DNSMOS_KEEP_QUANTILE}'"
    fi

    docker_run_runtime "" "${FILTER_CMD}"
  fi
fi

############################
# 7）Step 4：LID 过滤 + TSV 转 JSON
############################

if [[ "$DO_LID" -eq 1 ]]; then
  need_file "${DNSMOS_FILTERED_TSV}"

  if ! maybe_skip "${LANG_SEG_JSON}" "LANG_ID + TSV2SEG"; then
    log "[RUN] LangID -> ${LANG_OUT_FILTERED}"

    # 这里显式使用塞进镜像的 spk_consistency 环境
    docker_run_runtime "device=${LID_GPU}" \
      "/opt/envs/spk_consistency/bin/python '${LANG_DIR}/lang_id_filter.py' \
        --in_tsv '${DNSMOS_FILTERED_TSV}' \
        --out_scores '${LANG_OUT_SCORES}' \
        --out_filtered '${LANG_OUT_FILTERED}' \
        --cache_db '${LANG_CACHE_DB}' \
        --model_id '${LANG_MODEL_ID}' \
        --device '${LANG_DEVICE}' \
        --compute_type '${LANG_COMPUTE_TYPE}' \
        --target_langs ${LANG_TARGET_LANGS[*]} \
        --min_lang_prob '${LANG_MIN_PROB}'"

    log "[RUN] TSV -> segments json -> ${LANG_SEG_JSON}"
    docker_run_runtime "" \
      "python3 '${TSV2SEG_PY}' \
        --in_tsv '${LANG_OUT_FILTERED}' \
        --out_json '${LANG_SEG_JSON}'"
  fi
fi

############################
# 8）Step 5：Whisper 多卡转写
############################

if [[ "$DO_WHISPER" -eq 1 ]]; then
  need_file "${LANG_SEG_JSON}"
  FINAL_WHISPER_JSON="${WHISPER_OUT_PREFIX}_all.json"

  if ! maybe_skip "${FINAL_WHISPER_JSON}" "WHISPER"; then
    log "[RUN] Whisper 多卡转写 -> ${FINAL_WHISPER_JSON}"

    NUM_SHARDS=${#PIPELINE_GPUS[@]}
    [[ "$NUM_SHARDS" -gt 0 ]] || { echo "ERROR: PIPELINE_GPUS 不能为空" >&2; exit 1; }

    EXTRA_MAX=""
    if [[ -n "${WHISPER_MAX_FILES}" ]]; then
      EXTRA_MAX="--max_files '${WHISPER_MAX_FILES}'"
    fi

    for IDX in "${!PIPELINE_GPUS[@]}"; do
      GPU=${PIPELINE_GPUS[$IDX]}
      OUT_SHARD="${WHISPER_OUT_PREFIX}_shard${IDX}.json"

      docker_run_runtime "device=${GPU}" \
        "python3 '${ASR_DIR}/whisper_ms_from_segments.py' \
          --seg_json '${LANG_SEG_JSON}' \
          --out_json '${OUT_SHARD}' \
          --model_dir '${WHISPER_MODEL_DIR_HOST}' \
          --device 'cuda:0' \
          --batch_size '${WHISPER_BATCH_SIZE}' \
          --num_shards '${NUM_SHARDS}' \
          --shard_id '${IDX}' \
          ${EXTRA_MAX}" &
    done

    wait

    docker_run_runtime "" \
      "python3 '${ASR_DIR}/merge_whisper_shards.py' \
        --seg_json '${LANG_SEG_JSON}' \
        --inputs ${WHISPER_OUT_PREFIX}_shard*.json \
        --out '${FINAL_WHISPER_JSON}'"
  fi
fi

############################
# 9）结束信息
############################

log "流程结束。"
log "  VAD_OUT_JSON:        ${VAD_OUT_JSON}"
log "  DNSMOS_MERGED_TSV:   ${DNSMOS_MERGED_TSV}"
log "  DNSMOS_FILTERED_TSV: ${DNSMOS_FILTERED_TSV}"
log "  LANG_OUT_FILTERED:   ${LANG_OUT_FILTERED}"
log "  LANG_SEG_JSON:       ${LANG_SEG_JSON}"
log "  WHISPER_FINAL_JSON:  ${WHISPER_OUT_PREFIX}_all.json"