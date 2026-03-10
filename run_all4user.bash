#!/usr/bin/env bash
set -euo pipefail

############################################
# YJ 音频清洗流程（Huawei 交付版）
#
# 特点：
#   1) 仅依赖两个 Docker 镜像
#   2) 不依赖 HF / Torch / ModelScope cache 变量
#   3) 直接填写本地模型目录
#   4) 支持 Step 0 统一预处理（采样率 / 声道 / 格式）
#   5) 最终导出切片音频 + 标注 JSON
#   6) 可自动清理中间产物，只保留最终数据集
############################################

############################
# 1）用户主要配置区（优先改这里）
############################

# 是否强制重跑：1=重跑并清理本次输出；0=有输出就跳过
FORCE=1

# 项目根目录（代码目录）
PROJECT_ROOT="/Work21/2025/yanjiahao/YJ-audio-pipeline/yj-audio-pipeline"

# 需要挂载进容器的两个大目录
WORK_MOUNT_ROOT="/Work21"
DATA_MOUNT_ROOT="/CDShare3"

# 本次实验/输出名称
DATASET_NAME="4hw_wild_datas1_v1"

# 输出总目录
OUTPUT_ROOT="/CDShare3/Huawei_Encoder_Proj/datas/YJoutput/${DATASET_NAME}"
# 例如你也可以改成：
# OUTPUT_ROOT="/CDShare3/Huawei_Encoder_Proj/datas/YJoutput/${DATASET_NAME}"

# 输入音频根目录（可以是目录，也可以是单文件）
INPUT_ROOT="/Work21/2026/liangjintao/WavCrawler/wav_segments"

# Step 0：是否先统一预处理（1=开，0=关）
DO_PREPROCESS=1

# 统一预处理参数
PREPROCESS_TARGET_SR="16000"
PREPROCESS_CHANNELS="1"      # 1=单声道, 2=双声道
PREPROCESS_AUDIO_CODEC="pcm_s16le"
PREPROCESS_EXT="wav"

# 主流程镜像
RUNTIME_IMAGE="yj-pipeline-runtime:with-spk"

# DNSMOS 专用镜像
DNSMOS_DOCKER_IMAGE="dnsmos_gpu:cuda118"

# 主流程用到的 GPU（Whisper 用这里）
PIPELINE_GPUS=(0 1)

# DNSMOS 打分用到的 GPU
DNSMOS_GPUS=(0 1)

# LID 单独绑定的 GPU
LID_GPU=0
#lang
LANG_TARGET_LANGS=("en" "zh")


# ===== 直接填写模型目录（不依赖 cache 变量）=====
# Whisper 用的本地模型目录（transformers / modelscope 版本）
WHISPER_MODEL_DIR_HOST="/Work21/2025/yanjiahao/modelscope_cache/models/AI-ModelScope/whisper-large-v3"

# LID 用的本地 faster-whisper 模型目录
# 建议这里直接填“模型目录”，不要填 hf_cache 根目录
LID_MODEL_DIR_HOST="/Work21/2025/yanjiahao/hf_cache/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"

# 各步骤开关：1=执行，0=跳过
DO_VAD=1
DO_DNSMOS=1
DO_DNSMOS_FILTER=1
DO_LID=1
DO_WHISPER=1
DO_EXPORT_FINAL=1

# 是否在最终导出后自动清理中间产物（1=清理，0=保留）
DO_CLEAN_INTERMEDIATE=1

############################
# 2）进阶配置区（一般不用改）
############################

# Step 0：预处理输出目录（临时目录）
PREPROCESS_OUT_ROOT="${OUTPUT_ROOT}/_tmp_16k_audio"

# ===== VAD =====
VAD_PIPELINE_PY="${PROJECT_ROOT}/vad/vad_pipeline.py"
VAD_OUT_JSON="${OUTPUT_ROOT}/vad_output/${DATASET_NAME}_silero_vad_segments_mp_Ordered.json"
VAD_MIN_DUR="5.0"
VAD_NUM_WORKERS="32"
VAD_MAX_FILES=""

# ===== DNSMOS =====
DNSMOS_DIR="${PROJECT_ROOT}/dns_mos"
DNSMOS_SAVE_HOME="${OUTPUT_ROOT}/dnsmos_output"
DNSMOS_INPUT_LENGTH="9"
DNSMOS_MIN_DUR="1.0"

# 把 merged 输出放单独子目录，避免被 shard glob 误读
DNSMOS_MERGED_DIR="${DNSMOS_SAVE_HOME}/merged"
DNSMOS_MERGED_TSV="${DNSMOS_MERGED_DIR}/dns_from_vad_all.json_order.tsv"

# ===== DNSMOS 过滤 =====
DNSMOS_FILTERED_TSV="${DNSMOS_SAVE_HOME}/dns_filtered_emilia.tsv"
DNSMOS_FILTER_MIN_DUR="3.0"
DNSMOS_FILTER_MAX_DUR="100.0"
DNSMOS_MIN_MOS_OVR="2.0"
DNSMOS_MIN_MOS_SIG="2.0"
DNSMOS_MIN_MOS_BAK="3.5"
DNSMOS_KEEP_QUANTILE=""

# ===== 语言识别 LID =====
LANG_DIR="${PROJECT_ROOT}/language_filter"
LANG_OUT_SCORES="${DNSMOS_SAVE_HOME}/lang_scores_all.tsv"
LANG_OUT_FILTERED="${DNSMOS_SAVE_HOME}/lang_filtered.tsv"
LANG_CACHE_DB="${DNSMOS_SAVE_HOME}/lang_cache.sqlite"   # 这是结果缓存，不是模型 cache
LANG_MODEL_ID="${LID_MODEL_DIR_HOST}"
LANG_DEVICE="cuda"
LANG_COMPUTE_TYPE="float16"

LANG_MIN_PROB="0.90"
TSV2SEG_PY="${LANG_DIR}/tsv_to_segments_json.py"
LANG_SEG_JSON="${DNSMOS_SAVE_HOME}/lang_filtered_segments.json"

# ===== Whisper 转写 =====
ASR_DIR="${PROJECT_ROOT}/asr"
WHISPER_OUT_PREFIX="${OUTPUT_ROOT}/whisper_lv3_output/${DATASET_NAME}"
WHISPER_FINAL_JSON="${WHISPER_OUT_PREFIX}_all.json"
WHISPER_BATCH_SIZE="8"
WHISPER_MAX_FILES=""

# ===== 最终导出（切片音频 + 标注 JSON） =====
FINAL_DATASET_ROOT="${OUTPUT_ROOT}/final_dataset"
FINAL_AUDIO_DIR="${FINAL_DATASET_ROOT}/audio"
FINAL_LABEL_JSON="${FINAL_DATASET_ROOT}/labels.json"
FINAL_EXPORT_REQUIRE_TEXT=1   # 1=只导出有 text 的片段，0=全部导出
FINAL_EXPORT_FORMAT="wav"

############################
# 3）通用函数
############################

log() {
  echo "[$(date '+%F %T')] $*"
}

need_file() {
  local f="$1"
  [[ -f "$f" ]] || { echo "ERROR: 文件不存在: $f" >&2; exit 1; }
}

need_path() {
  local p="$1"
  [[ -e "$p" ]] || { echo "ERROR: 路径不存在: $p" >&2; exit 1; }
}

maybe_skip() {
  local out="$1"
  local name="$2"
  if [[ "$FORCE" -eq 0 && -e "$out" ]]; then
    log "[SKIP] ${name}: ${out}"
    return 0
  fi
  return 1
}

count_audio_files() {
  local root="$1"
  python3 - "$root" <<'PY'
import os
import sys
root = sys.argv[1]
exts = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".mp4", ".mkv"}
cnt = 0
if os.path.isfile(root):
    cnt = 1 if os.path.splitext(root)[1].lower() in exts else 0
elif os.path.isdir(root):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in exts:
                cnt += 1
print(cnt)
PY
}

json_item_count() {
  local json_path="$1"
  python3 - "$json_path" <<'PY'
import json
import sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    x = json.load(f)
print(len(x))
PY
}

ensure_dirs() {
  mkdir -p \
    "${OUTPUT_ROOT}" \
    "${OUTPUT_ROOT}/vad_output" \
    "${OUTPUT_ROOT}/whisper_lv3_output" \
    "${DNSMOS_SAVE_HOME}" \
    "${DNSMOS_MERGED_DIR}" \
    "${PREPROCESS_OUT_ROOT}" \
    "${FINAL_DATASET_ROOT}" \
    "${FINAL_AUDIO_DIR}"
}

unlock_outputs() {
  chmod -R a+rwX "${OUTPUT_ROOT}" 2>/dev/null || true
}

prepare_fresh_run() {
  if [[ "${FORCE}" -eq 1 ]]; then
    log "[RUN] FORCE=1，清理本次输出目录中的旧产物"
    rm -rf \
      "${PREPROCESS_OUT_ROOT}" \
      "${OUTPUT_ROOT}/vad_output" \
      "${DNSMOS_SAVE_HOME}" \
      "${OUTPUT_ROOT}/whisper_lv3_output" \
      "${FINAL_DATASET_ROOT}" \
      "${WHISPER_FINAL_JSON}" \
      "${OUTPUT_ROOT}/_tmp_preprocess_audio.py" \
      "${OUTPUT_ROOT}/_tmp_export_final_dataset.py" 2>/dev/null || true
  fi
}

# 运行主流程镜像
docker_run_runtime() {
  local gpu_spec="$1"
  local cmd="$2"

  if [[ -n "${gpu_spec}" ]]; then
    docker run --rm \
      --gpus "${gpu_spec}" \
      --ipc=host \
      -u "$(id -u):$(id -g)" \
      -v "${WORK_MOUNT_ROOT}:${WORK_MOUNT_ROOT}" \
      -v "${DATA_MOUNT_ROOT}:${DATA_MOUNT_ROOT}" \
      -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
      -v "${WHISPER_MODEL_DIR_HOST}:${WHISPER_MODEL_DIR_HOST}" \
      -v "${LID_MODEL_DIR_HOST}:${LID_MODEL_DIR_HOST}" \
      -w "${PROJECT_ROOT}" \
      "${RUNTIME_IMAGE}" \
      bash -lc "${cmd}"
  else
    docker run --rm \
      --ipc=host \
      -u "$(id -u):$(id -g)" \
      -v "${WORK_MOUNT_ROOT}:${WORK_MOUNT_ROOT}" \
      -v "${DATA_MOUNT_ROOT}:${DATA_MOUNT_ROOT}" \
      -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
      -v "${WHISPER_MODEL_DIR_HOST}:${WHISPER_MODEL_DIR_HOST}" \
      -v "${LID_MODEL_DIR_HOST}:${LID_MODEL_DIR_HOST}" \
      -w "${PROJECT_ROOT}" \
      "${RUNTIME_IMAGE}" \
      bash -lc "${cmd}"
  fi
}

# 运行 DNSMOS 镜像
docker_run_dnsmos() {
  local cmd="$1"

  docker run --rm \
    --gpus all \
    --ipc=host \
    -u "$(id -u):$(id -g)" \
    -v "${WORK_MOUNT_ROOT}:${WORK_MOUNT_ROOT}" \
    -v "${DATA_MOUNT_ROOT}:${DATA_MOUNT_ROOT}" \
    -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -w "${DNSMOS_DIR}" \
    "${DNSMOS_DOCKER_IMAGE}" \
    bash -lc "${cmd}"
}

########################################################
# Step 0：统一预处理（采样率 / 声道 / 格式）
########################################################

ACTIVE_INPUT_ROOT="${INPUT_ROOT}"

preprocess_input_audio() {
  local src="${INPUT_ROOT%/}"
  local helper_py="${OUTPUT_ROOT}/_tmp_preprocess_audio.py"

  [[ "${src}" = /* ]] || { echo "ERROR: INPUT_ROOT 必须是绝对路径，当前值为: ${src}" >&2; exit 1; }
  need_path "${src}"

  cat > "${helper_py}" <<'PY'
import os
import sys
import subprocess
from pathlib import Path

src = sys.argv[1]
dst_root = sys.argv[2]
target_sr = sys.argv[3]
channels = sys.argv[4]
codec = sys.argv[5]
ext = sys.argv[6]
overwrite = sys.argv[7]

exts = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".mp4", ".mkv"}
Path(dst_root).mkdir(parents=True, exist_ok=True)

def ffmpeg_one(inp, outp):
    if os.path.exists(outp) and overwrite == "-n":
        return False
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        overwrite,
        "-i", inp,
        "-vn",
        "-ac", channels,
        "-ar", target_sr,
        "-c:a", codec,
        outp,
    ]
    subprocess.run(cmd, check=True)
    return True

seen = 0
created = 0

if os.path.isfile(src):
    stem = os.path.splitext(os.path.basename(src))[0]
    outp = os.path.join(dst_root, f"{stem}.{ext}")
    seen = 1
    if ffmpeg_one(src, outp):
        created += 1

elif os.path.isdir(src):
    for dp, _, fns in os.walk(src):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() not in exts:
                continue
            inp = os.path.join(dp, fn)
            rel = os.path.relpath(inp, src)
            rel_noext = os.path.splitext(rel)[0]
            outp = os.path.join(dst_root, f"{rel_noext}.{ext}")
            seen += 1
            if ffmpeg_one(inp, outp):
                created += 1
else:
    print(f"ERROR: INPUT_ROOT 不存在: {src}", file=sys.stderr)
    sys.exit(2)

print(f"[PREPROCESS] seen={seen}, created={created}, dst={dst_root}")
if seen == 0:
    sys.exit(3)
PY

  chmod a+rx "${helper_py}"

  local ff_overwrite="-n"
  [[ "${FORCE}" -eq 1 ]] && ff_overwrite="-y"

  if [[ -f "${src}" ]]; then
    log "[RUN] Preprocess single file -> ${PREPROCESS_OUT_ROOT}"
  else
    log "[RUN] Preprocess directory -> ${PREPROCESS_OUT_ROOT}"
  fi

  docker_run_runtime "" \
    "python3 '${helper_py}' \
      '${src}' \
      '${PREPROCESS_OUT_ROOT}' \
      '${PREPROCESS_TARGET_SR}' \
      '${PREPROCESS_CHANNELS}' \
      '${PREPROCESS_AUDIO_CODEC}' \
      '${PREPROCESS_EXT}' \
      '${ff_overwrite}'"

  rm -f "${helper_py}"
  ACTIVE_INPUT_ROOT="${PREPROCESS_OUT_ROOT}"
}

########################################################
# Step 6：根据最终 JSON 导出切片音频 + 标注 JSON
########################################################

export_final_dataset() {
  local in_json="$1"
  local helper_py="${OUTPUT_ROOT}/_tmp_export_final_dataset.py"

  cat > "${helper_py}" <<'PY'
import json
import os
import re
import subprocess
import sys
from pathlib import Path

in_json = sys.argv[1]
audio_dir = sys.argv[2]
label_json = sys.argv[3]
audio_fmt = sys.argv[4]
target_sr = sys.argv[5]
channels = sys.argv[6]
require_text = int(sys.argv[7])

Path(audio_dir).mkdir(parents=True, exist_ok=True)

with open(in_json, "r", encoding="utf-8") as f:
    items = json.load(f)

def safe_name(x):
    x = str(x)
    x = re.sub(r"[^0-9A-Za-z._-]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "na"

labels = []
exported = 0
skipped = 0

for idx, item in enumerate(items):
    src = item.get("path") or item.get("source_path")
    text = (item.get("text") or "").strip()

    if require_text == 1 and not text:
        skipped += 1
        continue

    if not src or not os.path.isfile(src):
        skipped += 1
        continue

    try:
        start_sec = float(item.get("start_sec", 0.0))
        end_sec = float(item.get("end_sec", 0.0))
    except Exception:
        skipped += 1
        continue

    if end_sec <= start_sec:
        skipped += 1
        continue

    audio_id = safe_name(item.get("audio_id", "audio"))
    seg_id = safe_name(item.get("seg_id", idx))
    uid = f"{idx:08d}_{audio_id}_{seg_id}"
    out_path = os.path.join(audio_dir, f"{uid}.{audio_fmt}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", src,
        "-vn",
        "-ac", str(channels),
        "-ar", str(target_sr),
    ]

    if audio_fmt.lower() == "wav":
        ffmpeg_cmd += ["-c:a", "pcm_s16le"]

    ffmpeg_cmd += [out_path]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except subprocess.CalledProcessError:
        skipped += 1
        continue

    label = {
        "uid": uid,
        "audio_id": item.get("audio_id"),
        "seg_id": item.get("seg_id"),
        "path": out_path,
        "text": text,
        "source_path": src,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": float(item.get("duration_sec", end_sec - start_sec)),
        "lang": item.get("lang"),
        "lang_prob": item.get("lang_prob"),
    }

    for k in [
        "mos_sig", "mos_bak", "mos_ovr",
        "spk_id", "sim_spk", "sim_file",
        "error"
    ]:
        if k in item:
            label[k] = item.get(k)

    labels.append(label)
    exported += 1

with open(label_json, "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)

print(f"[EXPORT] exported={exported}, skipped={skipped}, labels={label_json}")
PY

  chmod a+rx "${helper_py}"

  log "[RUN] Export final dataset -> ${FINAL_AUDIO_DIR} + ${FINAL_LABEL_JSON}"
  docker_run_runtime "" \
    "python3 '${helper_py}' \
      '${in_json}' \
      '${FINAL_AUDIO_DIR}' \
      '${FINAL_LABEL_JSON}' \
      '${FINAL_EXPORT_FORMAT}' \
      '${PREPROCESS_TARGET_SR}' \
      '${PREPROCESS_CHANNELS}' \
      '${FINAL_EXPORT_REQUIRE_TEXT}'"

  rm -f "${helper_py}"
}

cleanup_intermediate_artifacts() {
  log "[RUN] Cleaning intermediate artifacts ..."
  rm -rf \
    "${PREPROCESS_OUT_ROOT}" \
    "${OUTPUT_ROOT}/vad_output" \
    "${DNSMOS_SAVE_HOME}" \
    "${OUTPUT_ROOT}/whisper_lv3_output" \
    "${WHISPER_FINAL_JSON}" \
    "${OUTPUT_ROOT}/_tmp_preprocess_audio.py" \
    "${OUTPUT_ROOT}/_tmp_export_final_dataset.py" 2>/dev/null || true
}

prepare_fresh_run
ensure_dirs

############################
# 4）Step 0：统一预处理
############################

if [[ "${DO_PREPROCESS}" -eq 1 ]]; then
  preprocess_input_audio
  ACTIVE_INPUT_ROOT="${PREPROCESS_OUT_ROOT}"
else
  ACTIVE_INPUT_ROOT="${INPUT_ROOT}"
fi

local_audio_count="$(count_audio_files "${ACTIVE_INPUT_ROOT}")"
log "[INFO] 当前进入 VAD 的音频数: ${local_audio_count}"
[[ "${local_audio_count}" -gt 0 ]] || { echo "ERROR: 进入 VAD 的音频数为 0，请检查 INPUT_ROOT 或预处理步骤" >&2; exit 1; }

log "[INFO] VAD 输入路径: ${ACTIVE_INPUT_ROOT}"

############################
# 5）Step 1：VAD 切段
############################

if [[ "$DO_VAD" -eq 1 ]]; then
  need_file "${VAD_PIPELINE_PY}"

  if ! maybe_skip "${VAD_OUT_JSON}" "VAD"; then
    log "[RUN] VAD -> ${VAD_OUT_JSON}"

    VAD_CMD="python3 '${VAD_PIPELINE_PY}' \
      --input_root '${ACTIVE_INPUT_ROOT}' \
      --out_json '${VAD_OUT_JSON}' \
      --min_dur '${VAD_MIN_DUR}' \
      --num_workers '${VAD_NUM_WORKERS}'"

    if [[ -n "${VAD_MAX_FILES}" ]]; then
      VAD_CMD="${VAD_CMD} --max_files '${VAD_MAX_FILES}'"
    fi

    docker_run_runtime "" "${VAD_CMD}"
  fi
fi

need_file "${VAD_OUT_JSON}"
local_vad_items="$(json_item_count "${VAD_OUT_JSON}")"
log "[INFO] VAD 输出条目数: ${local_vad_items}"
[[ "${local_vad_items}" -gt 0 ]] || { echo "ERROR: VAD 输出为 0，停止后续流程" >&2; exit 1; }

############################
# 6）Step 2：DNSMOS 打分 + 合并
############################

if [[ "$DO_DNSMOS" -eq 1 ]]; then
  if ! maybe_skip "${DNSMOS_MERGED_TSV}" "DNSMOS+MERGE"; then
    local_num_gpus=${#DNSMOS_GPUS[@]}
    GPUS_STR=$(IFS=,; echo "${DNSMOS_GPUS[*]}")

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
# 7）Step 3：DNSMOS 过滤
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
# 8）Step 4：LID 过滤 + TSV 转 JSON
############################

if [[ "$DO_LID" -eq 1 ]]; then
  need_file "${DNSMOS_FILTERED_TSV}"

  if ! maybe_skip "${LANG_SEG_JSON}" "LANG_ID + TSV2SEG"; then
    log "[RUN] LangID -> ${LANG_OUT_FILTERED}"

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
# 9）Step 5：Whisper 多卡转写
############################

if [[ "$DO_WHISPER" -eq 1 ]]; then
  need_file "${LANG_SEG_JSON}"

  if ! maybe_skip "${WHISPER_FINAL_JSON}" "WHISPER"; then
    log "[RUN] Whisper 多卡转写 -> ${WHISPER_FINAL_JSON}"

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
        --out '${WHISPER_FINAL_JSON}'"
  fi
fi

############################
# 10）Step 6：导出最终数据集（切片音频 + 标注 JSON）
############################

if [[ "${DO_EXPORT_FINAL}" -eq 1 ]]; then
  need_file "${WHISPER_FINAL_JSON}"

  if ! maybe_skip "${FINAL_LABEL_JSON}" "EXPORT_FINAL_DATASET"; then
    export_final_dataset "${WHISPER_FINAL_JSON}"
  fi
fi

############################
# 11）可选：清理中间产物
############################

if [[ "${DO_CLEAN_INTERMEDIATE}" -eq 1 ]]; then
  cleanup_intermediate_artifacts
fi

############################
# 12）结束信息
############################

unlock_outputs

log "流程结束。"
log "  FINAL_AUDIO_DIR:  ${FINAL_AUDIO_DIR}"
log "  FINAL_LABEL_JSON: ${FINAL_LABEL_JSON}"