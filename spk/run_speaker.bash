#!/usr/bin/env bash
set -euo pipefail

# 1) 激活环境（按你实际 conda 路径改一下）
source /Work21/2025/yanjiahao/miniconda3/etc/profile.d/conda.sh
conda activate spk_consistency

cd /CDShare3/Huawei_Encoder_Proj/codes/jiahao/Speaker
# 2) 缓存目录
export HF_HOME=/Work21/2025/yanjiahao/hf_cache
export TORCH_HOME=/Work21/2025/yanjiahao/torch_cache
export TRANSFORMERS_CACHE="$HF_HOME"

mkdir -p "$HF_HOME" "$TORCH_HOME"

python speaker_consistency_filter.py \
  --in_tsv  /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/dns_filtered_q70.tsv \
  --out_tsv /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/spk_filtered.tsv \
  --cache_db /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/spk_emb_cache.sqlite \
  --device cuda:0 \
  --min_dur 1.0 \
  --min_sim_spk 0.60 \
  --min_sim_file 0.65 \
  --add_scores_tsv /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/spk_scores_all.tsv
