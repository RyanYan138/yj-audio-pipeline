#!/usr/bin/env bash
set -euo pipefail

# 主流程用 emilia_pipe_clean，FunASR Nano ASR 步骤用 funasr_nano（Python 3.10）

############################################
# FireRedVAD + FunASR Nano
# VAD: 小红书 FireRedVAD
# ASR: FunASR Nano
############################################

FORCE=1
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=python3
PYTHON_FUNASR=python3
# FireRedVAD 用 funasr_nano (Python 3.10)，kaldi_native_fbank 兼容性好
PYTHON_VAD=python3

DATASET_NAME="test_run_fireredvad_funasr"
OUTPUT_ROOT="${PROJECT_ROOT}/output/${DATASET_NAME}"
INPUT_ROOT="${PROJECT_ROOT}/test/audio"

PIPELINE_GPUS=(0 1)
DNSMOS_GPUS=(0 1)
LID_GPU=0
LANG_TARGET_LANGS=("en" "zh")

# 模型路径
FIREREDVAD_MODEL="${PROJECT_ROOT}/FireRedVAD/pretrained_models/xukaituo/FireRedVAD/VAD"
FIREREDVAD_ROOT="${PROJECT_ROOT}/FireRedVAD"
FUNASR_MODEL_DIR="${PROJECT_ROOT}/models/FunASR-Nano"
LID_MODEL_DIR="${PROJECT_ROOT}/models/faster-whisper-large-v3"

# FireRedVAD 参数
VAD_USE_GPU=1
VAD_MIN_DUR="1.0"
VAD_MAX_DUR="30.0"

# FunASR Nano 参数
FUNASR_BATCH_SIZE=16
FUNASR_LANGUAGE="auto"

# 步骤开关
DO_VAD=1
DO_DNSMOS=1
DO_DNSMOS_FILTER=1
DO_LID=1
DO_ASR=1
DO_EXPORT_FINAL=1
DO_CLEAN_INTERMEDIATE=0

PREPROCESS_TARGET_SR="16000"
PREPROCESS_CHANNELS="1"

# 路径
VAD_PIPELINE_PY="${PROJECT_ROOT}/vad/fireredvad_pipeline.py"
VAD_OUT_JSON="${OUTPUT_ROOT}/vad_output/${DATASET_NAME}_fireredvad_segments.json"

DNSMOS_DIR="${PROJECT_ROOT}/dns_mos"
DNSMOS_SAVE_HOME="${OUTPUT_ROOT}/dnsmos_output"
DNSMOS_MERGED_DIR="${DNSMOS_SAVE_HOME}/merged"
DNSMOS_MERGED_TSV="${DNSMOS_MERGED_DIR}/dns_merged.tsv"
DNSMOS_INPUT_LENGTH="9"
DNSMOS_MIN_DUR="1.0"
DNSMOS_FILTERED_TSV="${DNSMOS_SAVE_HOME}/dns_filtered.tsv"
DNSMOS_FILTER_MIN_DUR="1.0"
DNSMOS_FILTER_MAX_DUR="100.0"
DNSMOS_MIN_MOS_OVR="2.0"
DNSMOS_MIN_MOS_SIG="2.0"
DNSMOS_MIN_MOS_BAK="3.5"
DNSMOS_KEEP_QUANTILE=""

LANG_DIR="${PROJECT_ROOT}/language_filter"
LANG_OUT_SCORES="${DNSMOS_SAVE_HOME}/lang_scores.tsv"
LANG_OUT_FILTERED="${DNSMOS_SAVE_HOME}/lang_filtered.tsv"
LANG_CACHE_DB="${DNSMOS_SAVE_HOME}/lang_cache.sqlite"
LANG_MIN_PROB="0.90"
TSV2SEG_PY="${LANG_DIR}/tsv_to_segments_json.py"
LANG_SEG_JSON="${DNSMOS_SAVE_HOME}/lang_filtered_segments.json"

ASR_DIR="${PROJECT_ROOT}/asr"
ASR_OUT_PREFIX="${OUTPUT_ROOT}/funasr_output/${DATASET_NAME}"
ASR_FINAL_JSON="${ASR_OUT_PREFIX}_all.json"

FINAL_DATASET_ROOT="${OUTPUT_ROOT}/final_dataset"
FINAL_AUDIO_DIR="${FINAL_DATASET_ROOT}/audio"
FINAL_LABEL_JSON="${FINAL_DATASET_ROOT}/labels.json"
FINAL_EXPORT_REQUIRE_TEXT=1
FINAL_EXPORT_FORMAT="wav"

############################
log() { echo "[$(date '+%F %T')] $*"; }
need_file() { [[ -f "$1" ]] || { echo "ERROR: 不存在: $1" >&2; exit 1; }; }
maybe_skip() {
  [[ "$FORCE" -eq 0 && -e "$1" ]] && { log "[SKIP] $2"; return 0; }
  return 1
}
json_count() { $PYTHON -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$1"; }
run_cmd() {
  local gpu_spec="$1" cmd="$2"
  if [[ -z "${gpu_spec}" || "${gpu_spec}" == "all" ]]; then
    bash -c "${cmd}"
  else
    CUDA_VISIBLE_DEVICES="${gpu_spec#device=}" bash -c "${cmd}"
  fi
}

[[ "$FORCE" -eq 1 ]] && rm -rf "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/vad_output" "${OUTPUT_ROOT}/funasr_output" \
         "${DNSMOS_SAVE_HOME}" "${DNSMOS_MERGED_DIR}" \
         "${FINAL_DATASET_ROOT}" "${FINAL_AUDIO_DIR}"

log "=== FireRedVAD + FunASR Nano ==="
log "INPUT: ${INPUT_ROOT}"
log "OUTPUT: ${OUTPUT_ROOT}"

# Step 1: FireRedVAD
if [[ "$DO_VAD" -eq 1 ]]; then
  need_file "${VAD_PIPELINE_PY}"
  if ! maybe_skip "${VAD_OUT_JSON}" "FireRedVAD"; then
    log "[RUN] FireRedVAD -> ${VAD_OUT_JSON}"
    CUDA_VISIBLE_DEVICES="${PIPELINE_GPUS[0]}" $PYTHON_VAD "${VAD_PIPELINE_PY}" \
      --input_root "${INPUT_ROOT}" \
      --out_json "${VAD_OUT_JSON}" \
      --model_dir "${FIREREDVAD_MODEL}" \
      --fireredvad_root "${FIREREDVAD_ROOT}" \
      --min_dur "${VAD_MIN_DUR}" \
      --max_dur "${VAD_MAX_DUR}" \
      --use_gpu "${VAD_USE_GPU}"
  fi
fi
need_file "${VAD_OUT_JSON}"
log "[INFO] VAD 条目数: $(json_count "${VAD_OUT_JSON}")"

# Step 2: DNSMOS
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
        CUDA_VISIBLE_DEVICES=\$GPU ${PYTHON} '${DNSMOS_DIR}/eval_dns_from_vad_json.py' \
          --vad-json '${VAD_OUT_JSON}' \
          --np '${local_num_gpus}' --idx \"\$IDX\" \
          --input-length '${DNSMOS_INPUT_LENGTH}' \
          --min-dur '${DNSMOS_MIN_DUR}' \
          --save-home '${DNSMOS_SAVE_HOME}' &
      done; wait"
    run_cmd "all" "${DNSMOS_CMD}"
    log "[RUN] Merge DNSMOS"
    run_cmd "" "${PYTHON} '${DNSMOS_DIR}/merge_dns_by_json_order.py' \
      --vad-json '${VAD_OUT_JSON}' \
      --shard-dir '${DNSMOS_SAVE_HOME}' \
      --output '${DNSMOS_MERGED_TSV}'"
  fi
fi

# Step 3: DNSMOS 过滤
if [[ "$DO_DNSMOS_FILTER" -eq 1 ]]; then
  need_file "${DNSMOS_MERGED_TSV}"
  if ! maybe_skip "${DNSMOS_FILTERED_TSV}" "DNSMOS_FILTER"; then
    log "[RUN] DNSMOS 过滤"
    FILTER_CMD="${PYTHON} '${DNSMOS_DIR}/filter_dnsmos_tsv.py' \
      --in_tsv '${DNSMOS_MERGED_TSV}' \
      --out_tsv '${DNSMOS_FILTERED_TSV}' \
      --min_dur '${DNSMOS_FILTER_MIN_DUR}' \
      --max_dur '${DNSMOS_FILTER_MAX_DUR}' \
      --min_mos_ovr '${DNSMOS_MIN_MOS_OVR}' \
      --min_mos_sig '${DNSMOS_MIN_MOS_SIG}' \
      --min_mos_bak '${DNSMOS_MIN_MOS_BAK}'"
    [[ -n "${DNSMOS_KEEP_QUANTILE}" ]] && FILTER_CMD="${FILTER_CMD} --keep_quantile '${DNSMOS_KEEP_QUANTILE}'"
    run_cmd "" "${FILTER_CMD}"
  fi
fi

# Step 4: LID
if [[ "$DO_LID" -eq 1 ]]; then
  need_file "${DNSMOS_FILTERED_TSV}"
  if ! maybe_skip "${LANG_SEG_JSON}" "LID"; then
    log "[RUN] LangID"
    run_cmd "device=${LID_GPU}" \
      "${PYTHON} '${LANG_DIR}/lang_id_filter.py' \
        --in_tsv '${DNSMOS_FILTERED_TSV}' \
        --out_scores '${LANG_OUT_SCORES}' \
        --out_filtered '${LANG_OUT_FILTERED}' \
        --cache_db '${LANG_CACHE_DB}' \
        --model_id '${LID_MODEL_DIR}' \
        --device cuda \
        --compute_type float16 \
        --target_langs ${LANG_TARGET_LANGS[*]} \
        --min_lang_prob '${LANG_MIN_PROB}'"
    run_cmd "" "${PYTHON} '${TSV2SEG_PY}' \
      --in_tsv '${LANG_OUT_FILTERED}' \
      --out_json '${LANG_SEG_JSON}'"
  fi
fi

# Step 5: FunASR Nano 多卡转写
if [[ "$DO_ASR" -eq 1 ]]; then
  need_file "${LANG_SEG_JSON}"
  if ! maybe_skip "${ASR_FINAL_JSON}" "FunASR-Nano"; then
    log "[RUN] FunASR Nano 多卡转写"
    NUM_SHARDS=${#PIPELINE_GPUS[@]}
    for IDX in "${!PIPELINE_GPUS[@]}"; do
      GPU=${PIPELINE_GPUS[$IDX]}
      OUT_SHARD="${ASR_OUT_PREFIX}_shard${IDX}.json"
      run_cmd "device=${GPU}" \
        "${PYTHON_FUNASR} '${ASR_DIR}/funasr_nano_from_segments.py' \
          --seg_json '${LANG_SEG_JSON}' \
          --out_json '${OUT_SHARD}' \
          --model_dir '${FUNASR_MODEL_DIR}' \
          --device cuda:0 \
          --batch_size '${FUNASR_BATCH_SIZE}' \
          --language '${FUNASR_LANGUAGE}' \
          --num_shards '${NUM_SHARDS}' \
          --shard_id '${IDX}'" &
    done
    wait
    # 合并 shards（复用 whisper merge 脚本，接口相同）
    run_cmd "" "${PYTHON} '${ASR_DIR}/merge_whisper_shards.py' \
      --seg_json '${LANG_SEG_JSON}' \
      --inputs ${ASR_OUT_PREFIX}_shard*.json \
      --out '${ASR_FINAL_JSON}'"
  fi
fi

# Step 6: 导出
if [[ "$DO_EXPORT_FINAL" -eq 1 ]]; then
  need_file "${ASR_FINAL_JSON}"
  if ! maybe_skip "${FINAL_LABEL_JSON}" "EXPORT"; then
    log "[RUN] 导出最终数据集"
    $PYTHON - << PYEOF
import json, os, re, subprocess, sys
from pathlib import Path
items = json.load(open("${ASR_FINAL_JSON}"))
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
        "-ss",str(s),"-t",str(round(e-s,3)),"-i",src,"-ac","1","-ar","16000","-c:a","pcm_s16le",out], check=True)
    labels.append({**item, "uid": uid, "path": out})
json.dump(labels, open("${FINAL_LABEL_JSON}","w"), ensure_ascii=False, indent=2)
print(f"[EXPORT] {len(labels)} 条")
PYEOF
  fi
fi

log "=============================="
log "完成！输出: ${OUTPUT_ROOT}"
log "  labels: ${FINAL_LABEL_JSON}"
log "=============================="
