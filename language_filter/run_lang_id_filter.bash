#!/usr/bin/env bash
set -euo pipefail

# ========= 1) conda env =========
source /Work21/2025/yanjiahao/miniconda3/etc/profile.d/conda.sh
conda activate /Work21/2025/yanjiahao/conda-envs/spk_consistency

cd /CDShare3/Huawei_Encoder_Proj/codes/jiahao/language_filter
# ========= 2) HF cache (avoid your broken .locks dir) =========
export HF_HOME=/Work21/2025/yanjiahao/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

# ========= 5) run LID =========
python lang_id_filter.py \
  --in_tsv  /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/spk_filtered.tsv \
  --out_scores   /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/lang_scores_all.tsv \
  --out_filtered /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/lang_filtered.tsv \
  --cache_db /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/lang_cache.sqlite \
  --model_id Systran/faster-whisper-large-v3 \
  --device cuda \
  --compute_type float16 \
  --target_lang en \
  --min_lang_prob 0.90


python /CDShare3/Huawei_Encoder_Proj/codes/jiahao/ASR/tsv_to_segments_json.py \
  --in_tsv /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/lang_filtered.tsv \
  --out_json /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/lang_filtered_segments.json
