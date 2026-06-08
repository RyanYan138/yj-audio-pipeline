#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Work21/2025/yanjiahao/YJ-audio-pipeline/yj-audio-pipeline"
PIPELINE_DIR="${PROJECT_ROOT}/pipeline"

FUNASR_MODEL="/Work21/2025/yanjiahao/modelscope_cache/models/FunAudioLLM/Fun-ASR-Nano-2512"
LID_MODEL="/Work21/2025/yanjiahao/hf_cache/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
FIREREDVAD_MODEL="${PROJECT_ROOT}/FireRedVAD/pretrained_models/xukaituo/FireRedVAD/VAD"
FIREREDVAD_ROOT="${PROJECT_ROOT}/FireRedVAD"
DNSMOS_DIR="${PROJECT_ROOT}/dns_mos"

TAR_PATH="/Work21/2025/yanjiahao/test_4.tar"
OUT_JSON="/Work21/2025/yanjiahao/test_tar_pipeline_output/labels.json"
GPU="2"

export LD_LIBRARY_PATH="/opt/conda/envs/funasr_vllm/lib/python3.12/site-packages/nvidia/cusparselt/lib:/opt/conda/envs/funasr_vllm/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/conda/envs/funasr_vllm/lib/python3.12/site-packages/ctranslate2.libs:${LD_LIBRARY_PATH:-}"
export MODELSCOPE_CACHE="/Work21/2025/yanjiahao/modelscope_cache"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"

mkdir -p "$(dirname ${OUT_JSON})"

/opt/conda/envs/funasr_vllm/bin/python "${PIPELINE_DIR}/tar_pipeline.py" \
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

echo "完成！输出: ${OUT_JSON}"
