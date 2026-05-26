#!/usr/bin/env bash
set -euo pipefail


PYTHON=python3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_ROOT="${PROJECT_ROOT}/test/audio"
OUTPUT_ROOT="${PROJECT_ROOT}/output/test_run_silero_whisper_fast"
GPUS="1,2"

FUNASR_MODEL="${PROJECT_ROOT}/models/FunASR-Nano"
LID_MODEL="${PROJECT_ROOT}/models/faster-whisper-large-v3"
FIREREDVAD_MODEL="${PROJECT_ROOT}/FireRedVAD/pretrained_models/xukaituo/FireRedVAD/VAD"
FIREREDVAD_ROOT="${PROJECT_ROOT}/FireRedVAD"
DNSMOS_DIR="${PROJECT_ROOT}/dns_mos"

OUT_JSON="${OUTPUT_ROOT}/final_dataset/labels.json"
AUDIO_DIR="${OUTPUT_ROOT}/final_dataset/audio"

log() { echo "[$(date '+%F %T')] $*"; }
[[ -d "$OUTPUT_ROOT" ]] && rm -rf "$OUTPUT_ROOT"
mkdir -p "$AUDIO_DIR"

log "=== Silero + faster-whisper (流水线版) ==="
log "INPUT: ${INPUT_ROOT}  OUTPUT: ${OUTPUT_ROOT}"

CUDA_VISIBLE_DEVICES="${GPUS}" $PYTHON "${PROJECT_ROOT}/pipeline/streaming_pipeline.py" \
    --input_root        "${INPUT_ROOT}" \
    --out_json          "${OUT_JSON}" \
    --audio_dir         "${AUDIO_DIR}" \
    --vad_model         silero \
    --asr_model         whisper \
    --whisper_model_dir "${PROJECT_ROOT}/models/faster-whisper-large-v3" \
    --lid_model_dir     "${LID_MODEL}" \
    --fireredvad_model  "${FIREREDVAD_MODEL}" \
    --fireredvad_root   "${FIREREDVAD_ROOT}" \
    --dnsmos_dir        "${DNSMOS_DIR}" \
    --gpus              "${GPUS}" \
    --min_dur       1.0 \
    --max_dur       30.0 \
    --min_mos_ovr   2.0 \
    --min_mos_sig   2.0 \
    --min_mos_bak   3.5 \
    --target_langs  en zh \
    --min_lang_prob 0.90 \
    --queue_maxsize 200

log "完成！labels: ${OUT_JSON}"
