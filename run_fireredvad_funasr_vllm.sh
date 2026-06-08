#!/usr/bin/env bash
set -euo pipefail

PYTHON="/opt/conda/envs/funasr_vllm/bin/python"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_ROOT="${PROJECT_ROOT}/test/audio"
OUTPUT_ROOT="${PROJECT_ROOT}/output/test_run_fireredvad_funasr_vllm"
GPU="2"

FUNASR_MODEL="/Work21/2025/yanjiahao/modelscope_cache/models/FunAudioLLM/Fun-ASR-Nano-2512"
LID_MODEL="/Work21/2025/yanjiahao/hf_cache/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
FIREREDVAD_MODEL="${PROJECT_ROOT}/FireRedVAD/pretrained_models/xukaituo/FireRedVAD/VAD"
FIREREDVAD_ROOT="${PROJECT_ROOT}/FireRedVAD"
DNSMOS_DIR="${PROJECT_ROOT}/dns_mos"

OUT_JSON="${OUTPUT_ROOT}/final_dataset/labels.json"
AUDIO_DIR="${OUTPUT_ROOT}/final_dataset/audio"

log() { echo "[$(date '+%F %T')] $*"; }
[[ -d "$OUTPUT_ROOT" ]] && rm -rf "$OUTPUT_ROOT"
mkdir -p "$AUDIO_DIR"

export LD_LIBRARY_PATH="/opt/conda/envs/funasr_vllm/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export MODELSCOPE_CACHE="/Work21/2025/yanjiahao/modelscope_cache"
export CUDA_VISIBLE_DEVICES="${GPU}"

log "=== FireRedVAD + FunASR Nano vLLM (流水线版) ==="
log "INPUT: ${INPUT_ROOT}  OUTPUT: ${OUTPUT_ROOT}  GPU: ${GPU}"

$PYTHON "${PROJECT_ROOT}/pipeline/streaming_pipeline.py" \
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
    --min_dur       1.0 \
    --max_dur       30.0 \
    --min_mos_ovr   2.0 \
    --min_mos_sig   2.0 \
    --min_mos_bak   3.5 \
    --target_langs  en zh \
    --min_lang_prob 0.90 \
    --queue_maxsize 200

log "完成！labels: ${OUT_JSON}"
