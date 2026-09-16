#!/usr/bin/env bash
set -euo pipefail

ROOT=/Work21/2025/yanjiahao/YJ-audio-pipeline/yj-audio-pipeline
PYTHON=/Work21/2025/yanjiahao/conda-envs/funasr_vllm/bin/python
GPU=${GPU:-2}
DTYPE=${DTYPE:-fp32}
OUTPUT=${SWEEP_OUTPUT:-${ROOT}/output/4090d_batch_sweep/sweep.json}
BATCH_SIZES=${BATCH_SIZES:-"16 32 64 96 128 160 192 224 256"}

mkdir -p "$(dirname "${OUTPUT}")"
export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${ROOT}"
exec "${PYTHON}" test/benchmark_nano_batch_sweep.py \
  --labels output/bucket_benchmark_1h_final/labels.json \
  --model-dir models/Fun-ASR-Nano-2512 \
  --output "${OUTPUT}" \
  --repeat 1 \
  --dtype "${DTYPE}" \
  --batch-sizes ${BATCH_SIZES}
