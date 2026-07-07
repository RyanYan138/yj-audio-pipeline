#!/usr/bin/env bash
# 用法：bash run_wilddata_no_dnsmos.sh [GPU]
GPU=${1:-1}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES=${GPU}
export MODELSCOPE_CACHE="${PROJECT_ROOT}/models"

# 自动设 LD_LIBRARY_PATH
_NV="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia"
if [ -d "$_NV" ]; then
  for _d in "$_NV"/*/lib; do [ -d "$_d" ] && LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"; done
  export LD_LIBRARY_PATH
fi

"${CONDA_PREFIX}/bin/python" "${PROJECT_ROOT}/pipeline/streaming_pipeline_nodnsmos.py" \
    --input_root        /Work21/2026/liangjintao/WavCrawler/wav_segments \
    --out_json          "${PROJECT_ROOT}/output/wilddata_no_dnsmos/labels.json" \
    --audio_dir         "${PROJECT_ROOT}/output/wilddata_no_dnsmos/audio" \
    --vad_model         fireredvad \
    --asr_model         funasr_vllm \
    --funasr_model_dir  "${PROJECT_ROOT}/models/Fun-ASR-Nano-2512" \
    --lid_model_dir     "${PROJECT_ROOT}/models/faster-whisper-large-v3" \
    --fireredvad_model  "${PROJECT_ROOT}/models/FireRedVAD" \
    --fireredvad_root   "${PROJECT_ROOT}/FireRedVAD" \
    --gpus              "${GPU}" \
    --vllm_batch_size   64 \
    --min_dur           1.0 \
    --max_dur           30.0 \
    --target_langs      zh \
    --min_lang_prob     0.90 \
    --queue_maxsize     200
