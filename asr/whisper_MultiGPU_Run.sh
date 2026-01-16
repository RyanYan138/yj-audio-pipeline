#!/bin/bash
set -e

########## 可配参数 ##########
#conda环境配置
CONDA_SOURCE="/Work21/2025/yanjiahao/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_PATH="/Work21/2025/yanjiahao/conda-envs/asr_whisper"

#当前文件夹位置，后面要CD
PATH_NOW="/CDShare3/Huawei_Encoder_Proj/codes/jiahao/ASR"

# 输入：VAD 切好的 segments 索引
SEG_JSON="/CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/lang_filtered_segments.json"

# 输出前缀：最终会生成 ${OUT_PREFIX}_shard*.json 和 ${OUT_PREFIX}_all.json
OUT_PREFIX="/CDShare3/Huawei_Encoder_Proj/codes/jiahao/ASR/whisperoutput/whisper_lv3"

# 使用哪些 GPU（想用 这3 张：就写 (0 1 3)，自动按数量分 shard）
GPUS=(1 2)

# batch size
BATCH_SIZE=16

# 限制最大处理条数（调试用）
# 为空表示全量；想只跑 1000 条就改成：MAX_FILES=1000
MAX_FILES=500

########## 环境 ##########

# 外面已经在 asr_whisper 可以注释掉
source "$CONDA_SOURCE"
conda activate "$CONDA_ENV_PATH"

cd "$PATH_NOW"

# Ctrl+C 时杀掉本脚本启动的所有后台进程
cleanup() {
  echo -e "\n[INTERRUPT] Ctrl+C received, killing all shard jobs..."
  jobs -pr | xargs -r kill -TERM
  sleep 1
  jobs -pr | xargs -r kill -KILL || true
  exit 130
}
trap cleanup INT TERM

########## 启动多卡 shard 推理 ##########

NUM_SHARDS=${#GPUS[@]}
if [[ "$NUM_SHARDS" -eq 0 ]]; then
  echo "ERROR: GPUS 未配置"; exit 1
fi

EXTRA_MAX=""
if [[ -n "$MAX_FILES" ]]; then
  EXTRA_MAX="--max_files $MAX_FILES"
fi

echo "Use GPUs: ${GPUS[*]}, num_shards=${NUM_SHARDS}, max_files=${MAX_FILES:-ALL}"

for IDX in "${!GPUS[@]}"; do
  GPU=${GPUS[$IDX]}
  SHARD_ID=$IDX

  OUT_SHARD="${OUT_PREFIX}_shard${SHARD_ID}.json"

  echo "[LAUNCH] GPU${GPU} -> shard ${SHARD_ID}, out=${OUT_SHARD}"

  CUDA_VISIBLE_DEVICES=${GPU} \
  python whisper_ms_from_segments.py \
    --seg_json "${SEG_JSON}" \
    --out_json "${OUT_SHARD}" \
    --batch_size "${BATCH_SIZE}" \
    --num_shards "${NUM_SHARDS}" --shard_id "${SHARD_ID}" \
    ${EXTRA_MAX} &
done

wait
echo "All shards finished, start merging..."

########## 合并为一个文件（按原 seg_json 顺序排序） ##########
#注意，需要py文件放在一个文件夹内
python merge_whisper_shards.py \
  --seg_json "${SEG_JSON}" \
  --inputs ${OUT_PREFIX}_shard*.json \
  --out "${OUT_PREFIX}_all.json"

echo "Done. Final merged: ${OUT_PREFIX}_all.json"
