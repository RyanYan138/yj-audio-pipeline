#!/usr/bin/env bash
set -euo pipefail

############################################
# One-shot pipeline runner
#   1) VAD (silero) -> segments json
#   2) DNSMOS (docker multi-GPU) -> per-shard TSV -> merge
#   3) filter_dnsmos_tsv.py -> dns_filtered.tsv
#   4) speaker_consistency_filter.py -> spk_filtered.tsv
#   5) lang_id_filter.py -> lang_filtered.tsv -> segments json
#   6) Whisper multi-GPU -> shard json -> merge
#
# 变量统一放在最上面；默认按你给的脚本逻辑拼起来。
############################################

############### 0) 只改这里（配置区） ###############

# 如果输出已存在，是否强制重跑（1=重跑 / 0=自动跳过）
FORCE=0
# ========== 基础路径 ==========
NOW_PROJECT="/CDShare3/Huawei_Encoder_Proj"
PROJECT_ROOT="/CDShare3/Huawei_Encoder_Proj/codes/jiahao/Yj_Pipeline"
CONDA_SOURCE="/Work21/2025/yanjiahao/miniconda3/etc/profile.d/conda.sh"

# ========== VAD ==========
CONDA_ENV_VAD="/Work21/2025/yanjiahao/conda-envs/Huawei_Encoder_Vad"
VAD_PIPELINE_PY="${PROJECT_ROOT}/vad/vad_pipeline.py"
VAD_INPUT_ROOT="/CDShare3/Huawei_Encoder_Proj/datas/LibriSpeech/test-other"
VAD_OUT_JSON="${PROJECT_ROOT}/output/librispeech_silero_vad_segments_mp_Ordered.json"
VAD_MIN_DUR="0.75"
VAD_NUM_WORKERS="32"
VAD_MAX_FILES="100"   # 空=全量；例如 1000

# ========== DNSMOS ==========
DNSMOS_DIR="${PROJECT_ROOT}/dns_mos"
DNSMOS_DOCKER_IMAGE="dnsmos_gpu:cuda118"
DNSMOS_SAVE_HOME="${DNSMOS_DIR}/output"
DNSMOS_INPUT_LENGTH="9"
DNSMOS_MIN_DUR="1.0"
DNSMOS_GPUS=(3 5)
DNSMOS_MERGED_TSV="${DNSMOS_SAVE_HOME}/dns_from_vad_all.json_order.tsv"

# ========== DNSMOS 过滤 ==========
DNSMOS_FILTERED_TSV="${DNSMOS_SAVE_HOME}/dns_filtered_q70.tsv"
DNSMOS_FILTER_MIN_DUR="2.0"
DNSMOS_FILTER_MAX_DUR="20.0"
DNSMOS_KEEP_QUANTILE="0.7"   # 保留 top 70% by mos_ovr；不想用分位数可自己改脚本参数

# ========== Speaker 一致性 ==========

CONDA_ENV_SPK="/Work21/2025/yanjiahao/conda-envs/spk_consistency"

HF_HOME="/Work21/2025/yanjiahao/hf_cache"
TORCH_HOME="/Work21/2025/yanjiahao/torch_cache"

SPEAKER_DIR="${PROJECT_ROOT}/spk"
SPEAKER_IN_TSV="${DNSMOS_FILTERED_TSV}"
SPEAKER_OUT_TSV="${DNSMOS_SAVE_HOME}/spk_filtered.tsv"
SPEAKER_CACHE_DB="${DNSMOS_SAVE_HOME}/spk_emb_cache.sqlite"
SPEAKER_ADD_SCORES_TSV="${DNSMOS_SAVE_HOME}/spk_scores_all.tsv"
SPEAKER_DEVICE="cuda:0"
SPEAKER_MIN_DUR="1.0"
SPEAKER_MIN_SIM_SPK="0.60"
SPEAKER_MIN_SIM_FILE="0.65"

# ========== Language ID 过滤 ==========
CONDA_ENV_LID="/Work21/2025/yanjiahao/conda-envs/spk_consistency"  # env
LANG_DIR="${PROJECT_ROOT}/language_filter"
LANG_IN_TSV="${SPEAKER_OUT_TSV}"
LANG_OUT_SCORES="${DNSMOS_SAVE_HOME}/lang_scores_all.tsv"
LANG_OUT_FILTERED="${DNSMOS_SAVE_HOME}/lang_filtered.tsv"
LANG_CACHE_DB="${DNSMOS_SAVE_HOME}/lang_cache.sqlite"
LANG_MODEL_ID="Systran/faster-whisper-large-v3"
LANG_DEVICE="cuda"
LANG_COMPUTE_TYPE="float16"
LANG_TARGET_LANG="en"       # 例如 en / zh / ja ...
LANG_MIN_PROB="0.90"

# 把 lang_filtered.tsv 变成 whisper 用的 segments json
TSV2SEG_PY="${LANG_DIR}/tsv_to_segments_json.py"
LANG_SEG_JSON="${DNSMOS_SAVE_HOME}/lang_filtered_segments.json"

# ========== Whisper 多卡 ==========
CONDA_ENV_ASR="/Work21/2025/yanjiahao/conda-envs/asr_whisper"
ASR_DIR="${PROJECT_ROOT}/asr"
WHISPER_SEG_JSON="${LANG_SEG_JSON}"
WHISPER_OUT_PREFIX="${PROJECT_ROOT}/output/whisper_lv3"
WHISPER_GPUS=(3 5)
WHISPER_BATCH_SIZE="16"
WHISPER_MAX_FILES=""   # 空=全量；例如 1000

# ========== 执行开关（想跳过某步就改成 0） ==========
DO_VAD=1
DO_DNSMOS=1
DO_DNSMOS_FILTER=1
DO_SPEAKER=1
DO_LID=1
DO_WHISPER=1



########################################################
# 1) 通用工具函数（下面一般不用改）
########################################################

log() { echo "[$(date '+%F %T')] $*"; }

need_file() {
  local f="$1"
  [[ -f "$f" ]] || { echo "ERROR: file not found: $f" >&2; exit 1; }
}

maybe_skip() {
  # usage: maybe_skip <output_path> <step_name>
  local out="$1"; local name="$2"
  if [[ "$FORCE" -eq 0 && -e "$out" ]]; then
    log "[SKIP] ${name} (exists): ${out}"
    return 0
  fi
  return 1
}

ensure_dirs() {
  mkdir -p "${DNSMOS_SAVE_HOME}" "${ASR_DIR}/whisperoutput" "${HF_HOME}" "${TORCH_HOME}"
}

conda_activate() {
  # usage: conda_activate <env_path_or_name>
  local env="$1"
  # shellcheck disable=SC1090
  source "${CONDA_SOURCE}"
  conda activate "${env}"
}

########################################################
# 2) 开始跑
########################################################
# ---------- Step 1: VAD ----------
if [[ "$DO_VAD" -eq 1 ]]; then
  need_file "${VAD_PIPELINE_PY}"

  if ! maybe_skip "${VAD_OUT_JSON}" "VAD"; then
    log "[RUN] VAD -> ${VAD_OUT_JSON}"

    conda_activate "${CONDA_ENV_VAD}"

    EXTRA_MAX=""
    if [[ -n "${VAD_MAX_FILES}" ]]; then
      EXTRA_MAX="--max_files ${VAD_MAX_FILES}"
    fi

    python3 "${VAD_PIPELINE_PY}" \
      --input_root "${VAD_INPUT_ROOT}" \
      --out_json "${VAD_OUT_JSON}" \
      --min_dur "${VAD_MIN_DUR}" \
      --num_workers "${VAD_NUM_WORKERS}" \
      ${EXTRA_MAX}
  fi
else
  log "[SKIP] VAD (DO_VAD=0)"
fi


# ---------- Step 2: DNSMOS (docker multi-GPU) + merge ----------
if [[ "$DO_DNSMOS" -eq 1 ]]; then
  need_file "${VAD_OUT_JSON}"
  if ! maybe_skip "${DNSMOS_MERGED_TSV}" "DNSMOS+MERGE"; then
    log "[RUN] DNSMOS (docker) -> shards -> merge -> ${DNSMOS_MERGED_TSV}"

    USER_ID=$(id -u)
    GROUP_ID=$(id -g)
    NUM_GPUS=${#DNSMOS_GPUS[@]}
    GPUS_STR=$(IFS=,; echo "${DNSMOS_GPUS[*]}")

    log "[HOST] VAD_JSON=${VAD_OUT_JSON}"
    log "[HOST] SAVE_HOME=${DNSMOS_SAVE_HOME}"
    log "[HOST] GPUS=${GPUS_STR} (NUM_GPUS=${NUM_GPUS})"

    docker run --rm --gpus all \
      --user "${USER_ID}:${GROUP_ID}" \
      -v "${NOW_PROJECT}:${NOW_PROJECT}" \
      -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
      -w "${DNSMOS_DIR}" \
      -e VAD_JSON="${VAD_OUT_JSON}" \
      -e SAVE_HOME="${DNSMOS_SAVE_HOME}" \
      -e GPUS_STR="${GPUS_STR}" \
      -e NUM_GPUS="${NUM_GPUS}" \
      "${DNSMOS_DOCKER_IMAGE}" \
      bash -lc '
        set -euo pipefail
        echo "==== [DOCKER] Running DNSMOS with ${NUM_GPUS} GPUs ===="
        trap "echo [DOCKER] Caught CTRL+C, killing children...; kill 0; exit 1" SIGINT SIGTERM

        IFS="," read -r -a GPUS <<< "$GPUS_STR"

        echo "[DOCKER] GPUs: ${GPUS[*]}, np=${NUM_GPUS}"
        echo "[DOCKER] VAD_JSON = $VAD_JSON"
        echo "[DOCKER] SAVE_HOME = $SAVE_HOME"

        for IDX in "${!GPUS[@]}"; do
          GPU=${GPUS[$IDX]}
          echo "[DOCKER]  -> Launch shard idx=${IDX} on GPU=${GPU}"
          CUDA_VISIBLE_DEVICES=$GPU python3 eval_dns_from_vad_json.py \
            --vad-json "$VAD_JSON" \
            --np "$NUM_GPUS" --idx "$IDX" \
            --input-length "'"${DNSMOS_INPUT_LENGTH}"'" \
            --min-dur "'"${DNSMOS_MIN_DUR}"'" \
            --save-home "$SAVE_HOME" &
        done

        wait
        echo "==== [DOCKER] All shards finished ===="
        ls -lh "$SAVE_HOME" || true
      '

    log "[RUN] Merge DNSMOS shards -> ${DNSMOS_MERGED_TSV}"
    pushd "${DNSMOS_DIR}" >/dev/null
    python3 merge_dns_by_json_order.py \
      --vad-json "${VAD_OUT_JSON}" \
      --shard-dir "${DNSMOS_SAVE_HOME}" \
      --output "${DNSMOS_MERGED_TSV}"
    popd >/dev/null
  fi
else
  log "[SKIP] DNSMOS (DO_DNSMOS=0)"
fi

# ---------- Step 3: filter_dnsmos_tsv.py ----------
if [[ "$DO_DNSMOS_FILTER" -eq 1 ]]; then
  need_file "${DNSMOS_MERGED_TSV}"
  if ! maybe_skip "${DNSMOS_FILTERED_TSV}" "DNSMOS_FILTER"; then
    log "[RUN] Filter DNSMOS -> ${DNSMOS_FILTERED_TSV}"
    pushd "${DNSMOS_DIR}" >/dev/null
    python3 filter_dnsmos_tsv.py \
      --in_tsv "${DNSMOS_MERGED_TSV}" \
      --out_tsv "${DNSMOS_FILTERED_TSV}" \
      --min_dur "${DNSMOS_FILTER_MIN_DUR}" --max_dur "${DNSMOS_FILTER_MAX_DUR}" \
      --keep_quantile "${DNSMOS_KEEP_QUANTILE}"
    popd >/dev/null
  fi
else
  log "[SKIP] DNSMOS_FILTER (DO_DNSMOS_FILTER=0)"
fi

# ---------- Step 4: Speaker consistency ----------
if [[ "$DO_SPEAKER" -eq 1 ]]; then
  need_file "${SPEAKER_IN_TSV}"
  if ! maybe_skip "${SPEAKER_OUT_TSV}" "SPEAKER_FILTER"; then
    log "[RUN] Speaker consistency -> ${SPEAKER_OUT_TSV}"

    export HF_HOME="${HF_HOME}"
    export TORCH_HOME="${TORCH_HOME}"
    export TRANSFORMERS_CACHE="${HF_HOME}"

    conda_activate "${CONDA_ENV_SPK}"
    pushd "${SPEAKER_DIR}" >/dev/null

    python3 speaker_consistency_filter.py \
      --in_tsv  "${SPEAKER_IN_TSV}" \
      --out_tsv "${SPEAKER_OUT_TSV}" \
      --cache_db "${SPEAKER_CACHE_DB}" \
      --device "${SPEAKER_DEVICE}" \
      --min_dur "${SPEAKER_MIN_DUR}" \
      --min_sim_spk "${SPEAKER_MIN_SIM_SPK}" \
      --min_sim_file "${SPEAKER_MIN_SIM_FILE}" \
      --add_scores_tsv "${SPEAKER_ADD_SCORES_TSV}"

    popd >/dev/null
  fi
else
  log "[SKIP] SPEAKER (DO_SPEAKER=0)"
fi

# ---------- Step 5: Language ID filter + tsv->segments.json ----------
if [[ "$DO_LID" -eq 1 ]]; then
  need_file "${LANG_IN_TSV}"
  if ! maybe_skip "${LANG_SEG_JSON}" "LANG_ID + TSV2SEG"; then
    log "[RUN] LangID -> ${LANG_OUT_FILTERED}  and segments -> ${LANG_SEG_JSON}"

    export HF_HOME="${HF_HOME}"
    export TRANSFORMERS_CACHE="${HF_HOME}"
    export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

    conda_activate "${CONDA_ENV_LID}"

    pushd "${LANG_DIR}" >/dev/null
    python3 lang_id_filter.py \
      --in_tsv "${LANG_IN_TSV}" \
      --out_scores "${LANG_OUT_SCORES}" \
      --out_filtered "${LANG_OUT_FILTERED}" \
      --cache_db "${LANG_CACHE_DB}" \
      --model_id "${LANG_MODEL_ID}" \
      --device "${LANG_DEVICE}" \
      --compute_type "${LANG_COMPUTE_TYPE}" \
      --target_lang "${LANG_TARGET_LANG}" \
      --min_lang_prob "${LANG_MIN_PROB}"
    popd >/dev/null

    log "[RUN] TSV -> segments json: ${LANG_SEG_JSON}"
    python3 "${TSV2SEG_PY}" \
      --in_tsv "${LANG_OUT_FILTERED}" \
      --out_json "${LANG_SEG_JSON}"
  fi
else
  log "[SKIP] LID (DO_LID=0)"
fi

# ---------- Step 6: Whisper multi-GPU ----------
if [[ "$DO_WHISPER" -eq 1 ]]; then
  need_file "${WHISPER_SEG_JSON}"

  FINAL_WHISPER_JSON="${WHISPER_OUT_PREFIX}_all.json"
  if ! maybe_skip "${FINAL_WHISPER_JSON}" "WHISPER"; then
    log "[RUN] Whisper multi-GPU -> ${FINAL_WHISPER_JSON}"

    conda_activate "${CONDA_ENV_ASR}"
    pushd "${ASR_DIR}" >/dev/null

    # Ctrl+C 时杀掉 shard 子进程
    cleanup() {
      echo -e "\n[INTERRUPT] Ctrl+C received, killing all shard jobs..." >&2
      jobs -pr | xargs -r kill -TERM
      sleep 1
      jobs -pr | xargs -r kill -KILL || true
      exit 130
    }
    trap cleanup INT TERM

    NUM_SHARDS=${#WHISPER_GPUS[@]}
    if [[ "$NUM_SHARDS" -eq 0 ]]; then
      echo "ERROR: WHISPER_GPUS 未配置" >&2; exit 1
    fi

    EXTRA_MAX=""
    if [[ -n "${WHISPER_MAX_FILES}" ]]; then
      EXTRA_MAX="--max_files ${WHISPER_MAX_FILES}"
    fi

    log "Use GPUs: ${WHISPER_GPUS[*]}, num_shards=${NUM_SHARDS}, max_files=${WHISPER_MAX_FILES:-ALL}"

    for IDX in "${!WHISPER_GPUS[@]}"; do
      GPU=${WHISPER_GPUS[$IDX]}
      SHARD_ID=$IDX
      OUT_SHARD="${WHISPER_OUT_PREFIX}_shard${SHARD_ID}.json"

      log "[LAUNCH] GPU${GPU} -> shard ${SHARD_ID}, out=${OUT_SHARD}"

      CUDA_VISIBLE_DEVICES=${GPU} \
      python3 whisper_ms_from_segments.py \
        --seg_json "${WHISPER_SEG_JSON}" \
        --out_json "${OUT_SHARD}" \
        --batch_size "${WHISPER_BATCH_SIZE}" \
        --num_shards "${NUM_SHARDS}" --shard_id "${SHARD_ID}" \
        ${EXTRA_MAX} &
    done

    wait
    log "All shards finished, start merging..."

    python3 merge_whisper_shards.py \
      --seg_json "${WHISPER_SEG_JSON}" \
      --inputs ${WHISPER_OUT_PREFIX}_shard*.json \
      --out "${FINAL_WHISPER_JSON}"

    popd >/dev/null
    log "[DONE] Whisper merged: ${FINAL_WHISPER_JSON}"
  fi
else
  log "[SKIP] WHISPER (DO_WHISPER=0)"
fi

log "Pipeline finished. Key outputs:"
log "  VAD_OUT_JSON:          ${VAD_OUT_JSON}"
log "  DNSMOS_MERGED_TSV:     ${DNSMOS_MERGED_TSV}"
log "  DNSMOS_FILTERED_TSV:   ${DNSMOS_FILTERED_TSV}"
log "  SPEAKER_OUT_TSV:       ${SPEAKER_OUT_TSV}"
log "  LANG_OUT_FILTERED:     ${LANG_OUT_FILTERED}"
log "  LANG_SEG_JSON:         ${LANG_SEG_JSON}"
log "  WHISPER_FINAL_JSON:    ${WHISPER_OUT_PREFIX}_all.json"
