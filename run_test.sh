#!/usr/bin/env bash
set -euo pipefail

############################################
# YJ Pipeline 环境测试脚本（自包含版）
#
# 所有路径在 PROJECT_ROOT 内，无需外部依赖
# 模型自动下载（小模型，快）：
#   - LID:    faster-whisper small (~500MB)
#   - Whisper: faster-whisper small (~500MB)
#   - DNSMOS: ONNX 模型已在 dns_mos/ 目录
#
# 使用：
#   conda activate vllm-qwen  (或你的环境)
#   bash run_test.sh
############################################

# PROJECT_ROOT 自动定位到脚本所在目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 所有输出放在项目内
OUTPUT_ROOT="${PROJECT_ROOT}/test/output"
INPUT_ROOT="${PROJECT_ROOT}/test/audio"

# 小模型：自动下载到项目内的 model_cache
MODEL_CACHE="${PROJECT_ROOT}/test/model_cache"
WHISPER_MODEL="small"        # 自动下载 ~500MB
LID_MODEL="small"            # 自动下载 ~500MB

DATASET_NAME="pipeline_test"
FORCE=1
DO_PREPROCESS=0              # 测试音频已是 16k，跳过
DO_VAD=1
DO_DNSMOS=1
DO_DNSMOS_FILTER=1
DO_LID=1
DO_WHISPER=1
DO_EXPORT_FINAL=1
DO_CLEAN_INTERMEDIATE=0      # 测试时保留中间产物方便排查

PIPELINE_GPUS=(0)
DNSMOS_GPUS=(0)
LID_GPU=0

LANG_TARGET_LANGS=("en" "zh")

############################
# 进阶配置（一般不用改）
############################
VAD_PIPELINE_PY="${PROJECT_ROOT}/vad/vad_pipeline.py"
VAD_OUT_JSON="${OUTPUT_ROOT}/vad_output/${DATASET_NAME}_vad.json"
VAD_MIN_DUR="1.0"
VAD_MAX_DUR="30"
VAD_NUM_WORKERS="4"
VAD_MAX_FILES=""

DNSMOS_DIR="${PROJECT_ROOT}/dns_mos"
DNSMOS_SAVE_HOME="${OUTPUT_ROOT}/dnsmos_output"
DNSMOS_MERGED_DIR="${DNSMOS_SAVE_HOME}/merged"
DNSMOS_MERGED_TSV="${DNSMOS_MERGED_DIR}/dns_merged.tsv"
DNSMOS_INPUT_LENGTH="9"
DNSMOS_MIN_DUR="1.0"

DNSMOS_FILTERED_TSV="${DNSMOS_SAVE_HOME}/dns_filtered.tsv"
DNSMOS_FILTER_MIN_DUR="1.0"
DNSMOS_FILTER_MAX_DUR="100.0"
DNSMOS_MIN_MOS_OVR="1.0"
DNSMOS_MIN_MOS_SIG="1.0"
DNSMOS_MIN_MOS_BAK="1.0"
DNSMOS_KEEP_QUANTILE=""

LANG_DIR="${PROJECT_ROOT}/language_filter"
LANG_OUT_SCORES="${DNSMOS_SAVE_HOME}/lang_scores.tsv"
LANG_OUT_FILTERED="${DNSMOS_SAVE_HOME}/lang_filtered.tsv"
LANG_CACHE_DB="${DNSMOS_SAVE_HOME}/lang_cache.sqlite"
LANG_MIN_PROB="0.50"
TSV2SEG_PY="${LANG_DIR}/tsv_to_segments_json.py"
LANG_SEG_JSON="${DNSMOS_SAVE_HOME}/lang_filtered_segments.json"

ASR_DIR="${PROJECT_ROOT}/asr"
WHISPER_OUT_PREFIX="${OUTPUT_ROOT}/whisper_output/${DATASET_NAME}"
WHISPER_FINAL_JSON="${WHISPER_OUT_PREFIX}_all.json"
WHISPER_BATCH_SIZE="4"
WHISPER_MAX_FILES=""

FINAL_DATASET_ROOT="${OUTPUT_ROOT}/final_dataset"
FINAL_AUDIO_DIR="${FINAL_DATASET_ROOT}/audio"
FINAL_LABEL_JSON="${FINAL_DATASET_ROOT}/labels.json"
FINAL_EXPORT_REQUIRE_TEXT=1
FINAL_EXPORT_FORMAT="wav"
PREPROCESS_TARGET_SR="16000"
PREPROCESS_CHANNELS="1"

############################
# 通用函数
############################
log() { echo "[$(date '+%F %T')] $*"; }
need_file() { [[ -f "$1" ]] || { echo "ERROR: 不存在: $1" >&2; exit 1; }; }
maybe_skip() {
  [[ "$FORCE" -eq 0 && -e "$1" ]] && { log "[SKIP] $2"; return 0; }
  return 1
}
json_item_count() {
  python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$1"
}

run_cmd() {
  local gpu_spec="$1" cmd="$2"
  if [[ -z "${gpu_spec}" || "${gpu_spec}" == "all" ]]; then
    bash -c "${cmd}"
  else
    CUDA_VISIBLE_DEVICES="${gpu_spec#device=}" bash -c "${cmd}"
  fi
}

mkdir -p "${OUTPUT_ROOT}/vad_output" "${OUTPUT_ROOT}/whisper_output" \
         "${DNSMOS_SAVE_HOME}" "${DNSMOS_MERGED_DIR}" \
         "${FINAL_DATASET_ROOT}" "${FINAL_AUDIO_DIR}" "${MODEL_CACHE}"

[[ "$FORCE" -eq 1 ]] && rm -rf "${OUTPUT_ROOT}" && \
  mkdir -p "${OUTPUT_ROOT}/vad_output" "${OUTPUT_ROOT}/whisper_output" \
           "${DNSMOS_SAVE_HOME}" "${DNSMOS_MERGED_DIR}" \
           "${FINAL_DATASET_ROOT}" "${FINAL_AUDIO_DIR}" "${MODEL_CACHE}"

log "PROJECT_ROOT: ${PROJECT_ROOT}"
log "INPUT_ROOT:   ${INPUT_ROOT}"
log "OUTPUT_ROOT:  ${OUTPUT_ROOT}"

############################
# Step 1: VAD
############################
if [[ "$DO_VAD" -eq 1 ]]; then
  need_file "${VAD_PIPELINE_PY}"
  if ! maybe_skip "${VAD_OUT_JSON}" "VAD"; then
    log "[RUN] VAD"
    VAD_CMD="python3 '${VAD_PIPELINE_PY}' \
      --input_root '${INPUT_ROOT}' \
      --out_json '${VAD_OUT_JSON}' \
      --min_dur '${VAD_MIN_DUR}' \
      --max_dur '${VAD_MAX_DUR}' \
      --num_workers '${VAD_NUM_WORKERS}'"
    run_cmd "" "${VAD_CMD}"
  fi
fi
need_file "${VAD_OUT_JSON}"
log "[INFO] VAD 条目数: $(json_item_count "${VAD_OUT_JSON}")"

############################
# Step 2: DNSMOS
############################
if [[ "$DO_DNSMOS" -eq 1 ]]; then
  if ! maybe_skip "${DNSMOS_MERGED_TSV}" "DNSMOS"; then
    local_num_gpus=${#DNSMOS_GPUS[@]}
    GPUS_STR=$(IFS=,; echo "${DNSMOS_GPUS[*]}")
    log "[RUN] DNSMOS 打分"
    DNSMOS_CMD="
      set -euo pipefail
      IFS=',' read -r -a GPUS <<< '${GPUS_STR}'
      for IDX in \"\${!GPUS[@]}\"; do
        GPU=\${GPUS[\$IDX]}
        CUDA_VISIBLE_DEVICES=\$GPU python3 '${DNSMOS_DIR}/eval_dns_from_vad_json.py' \
          --vad-json '${VAD_OUT_JSON}' \
          --np '${local_num_gpus}' --idx \"\$IDX\" \
          --input-length '${DNSMOS_INPUT_LENGTH}' \
          --min-dur '${DNSMOS_MIN_DUR}' \
          --save-home '${DNSMOS_SAVE_HOME}' &
      done
      wait
    "
    run_cmd "all" "${DNSMOS_CMD}"

    log "[RUN] Merge DNSMOS shards"
    run_cmd "" "python3 '${DNSMOS_DIR}/merge_dns_by_json_order.py' \
      --vad-json '${VAD_OUT_JSON}' \
      --shard-dir '${DNSMOS_SAVE_HOME}' \
      --output '${DNSMOS_MERGED_TSV}'"
  fi
fi

############################
# Step 3: DNSMOS 过滤
############################
if [[ "$DO_DNSMOS_FILTER" -eq 1 ]]; then
  need_file "${DNSMOS_MERGED_TSV}"
  if ! maybe_skip "${DNSMOS_FILTERED_TSV}" "DNSMOS_FILTER"; then
    log "[RUN] DNSMOS 过滤"
    run_cmd "" "python3 '${DNSMOS_DIR}/filter_dnsmos_tsv.py' \
      --in_tsv '${DNSMOS_MERGED_TSV}' \
      --out_tsv '${DNSMOS_FILTERED_TSV}' \
      --min_dur '${DNSMOS_FILTER_MIN_DUR}' \
      --max_dur '${DNSMOS_FILTER_MAX_DUR}' \
      --min_mos_ovr '${DNSMOS_MIN_MOS_OVR}' \
      --min_mos_sig '${DNSMOS_MIN_MOS_SIG}' \
      --min_mos_bak '${DNSMOS_MIN_MOS_BAK}'"
  fi
fi

############################
# Step 4: LID
############################
if [[ "$DO_LID" -eq 1 ]]; then
  need_file "${DNSMOS_FILTERED_TSV}"
  if ! maybe_skip "${LANG_SEG_JSON}" "LID"; then
    log "[RUN] LangID（模型: ${LID_MODEL}，首次运行自动下载）"
    run_cmd "device=${LID_GPU}" \
      "python3 '${LANG_DIR}/lang_id_filter.py' \
        --in_tsv '${DNSMOS_FILTERED_TSV}' \
        --out_scores '${LANG_OUT_SCORES}' \
        --out_filtered '${LANG_OUT_FILTERED}' \
        --cache_db '${LANG_CACHE_DB}' \
        --model_id '${LID_MODEL}' \
        --device cuda \
        --compute_type float16 \
        --target_langs ${LANG_TARGET_LANGS[*]} \
        --min_lang_prob '${LANG_MIN_PROB}'"

    run_cmd "" "python3 '${TSV2SEG_PY}' \
      --in_tsv '${LANG_OUT_FILTERED}' \
      --out_json '${LANG_SEG_JSON}'"
  fi
fi

############################
# Step 5: Whisper
############################
if [[ "$DO_WHISPER" -eq 1 ]]; then
  need_file "${LANG_SEG_JSON}"
  if ! maybe_skip "${WHISPER_FINAL_JSON}" "WHISPER"; then
    log "[RUN] Whisper（模型: ${WHISPER_MODEL}，首次运行自动下载）"
    NUM_SHARDS=${#PIPELINE_GPUS[@]}
    for IDX in "${!PIPELINE_GPUS[@]}"; do
      GPU=${PIPELINE_GPUS[$IDX]}
      OUT_SHARD="${WHISPER_OUT_PREFIX}_shard${IDX}.json"
      run_cmd "device=${GPU}" \
        "python3 '${ASR_DIR}/whisper_ms_from_segments.py' \
          --seg_json '${LANG_SEG_JSON}' \
          --out_json '${OUT_SHARD}' \
          --model_dir '${WHISPER_MODEL}' \
          --device cuda:0 \
          --batch_size '${WHISPER_BATCH_SIZE}' \
          --num_shards '${NUM_SHARDS}' \
          --shard_id '${IDX}'" &
    done
    wait

    run_cmd "" "python3 '${ASR_DIR}/merge_whisper_shards.py' \
      --seg_json '${LANG_SEG_JSON}' \
      --inputs ${WHISPER_OUT_PREFIX}_shard*.json \
      --out '${WHISPER_FINAL_JSON}'"
  fi
fi

############################
# Step 6: 导出
############################
if [[ "$DO_EXPORT_FINAL" -eq 1 ]]; then
  need_file "${WHISPER_FINAL_JSON}"
  if ! maybe_skip "${FINAL_LABEL_JSON}" "EXPORT"; then
    log "[RUN] 导出最终数据集"
    python3 - << PYEOF
import json, os, re, subprocess, sys
from pathlib import Path

items = json.load(open("${WHISPER_FINAL_JSON}"))
Path("${FINAL_AUDIO_DIR}").mkdir(parents=True, exist_ok=True)
labels = []
for idx, item in enumerate(items):
    src = item.get("path") or item.get("source_path")
    text = (item.get("text") or "").strip()
    if not text or not src or not os.path.isfile(src): continue
    try:
        s, e = float(item["start_sec"]), float(item["end_sec"])
    except: continue
    if e <= s: continue
    uid = f"{idx:06d}_{re.sub(r'[^0-9A-Za-z._-]+','_', item.get('audio_id','x'))}"
    out = f"${FINAL_AUDIO_DIR}/{uid}.wav"
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y",
        "-ss",str(s),"-to",str(e),"-i",src,"-ac","1","-ar","16000","-c:a","pcm_s16le",out], check=True)
    labels.append({**item, "uid": uid, "path": out})
json.dump(labels, open("${FINAL_LABEL_JSON}","w"), ensure_ascii=False, indent=2)
print(f"[EXPORT] {len(labels)} 条")
PYEOF
  fi
fi

log "=============================="
log "测试完成！"
log "  输出目录: ${OUTPUT_ROOT}"
log "  标注文件: ${FINAL_LABEL_JSON}"
log "=============================="
