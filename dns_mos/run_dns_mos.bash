#!/bin/bash

USER_ID=$(id -u)
GROUP_ID=$(id -g)   # ← 这里改成 -g，不要 -gd

# ========= 配置区：只改这里 =========
VAD_JSON=/CDShare3/Huawei_Encoder_Proj/codes/jiahao/vad/librispeech_silero_vad_segments_mp1.json
SAVE_HOME=/CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech
GPUS=(0 1 2 3)   # 想用哪些 GPU 就写哪些，比如 (0 1) 或 (0 1 2 3)
# ===================================

NUM_GPUS=${#GPUS[@]}
GPUS_STR=$(IFS=,; echo "${GPUS[*]}")   # 把数组变成 0,1,2,3 这种字符串

echo "[HOST] VAD_JSON = ${VAD_JSON}"
echo "[HOST] SAVE_HOME = ${SAVE_HOME}"
echo "[HOST] Using GPUs: ${GPUS_STR}, NUM_GPUS=${NUM_GPUS}"

docker run --rm --gpus all \
  --user ${USER_ID}:${GROUP_ID} \
  -v /CDShare3/Huawei_Encoder_Proj:/CDShare3/Huawei_Encoder_Proj \
  -w /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos \
  -e VAD_JSON="${VAD_JSON}" \
  -e SAVE_HOME="${SAVE_HOME}" \
  -e GPUS_STR="${GPUS_STR}" \
  -e NUM_GPUS="${NUM_GPUS}" \
  dnsmos_gpu:cuda118 \
  bash -c '
    echo "==== [DOCKER] Running DNSMOS with ${NUM_GPUS} GPUs ===="

    trap "echo [DOCKER] Caught CTRL+C, killing children...; kill 0; exit 1" SIGINT SIGTERM

    # 从环境变量还原 GPU 数组
    IFS="," read -r -a GPUS <<< "$GPUS_STR"

    echo "[DOCKER] GPUs: ${GPUS[*]}, np=${NUM_GPUS}"
    echo "[DOCKER] VAD_JSON = $VAD_JSON"
    echo "[DOCKER] SAVE_HOME = $SAVE_HOME"

    for IDX in "${!GPUS[@]}"; do
      GPU=${GPUS[$IDX]}
      echo "[DOCKER]  -> Launch shard idx=${IDX} on GPU=${GPU}"

      CUDA_VISIBLE_DEVICES=$GPU python3 eval_dns_from_vad_json.py \
        --vad-json "$VAD_JSON" \
        --np "$NUM_GPUS" --idx "$IDX" \
        --input-length 9 \
        --min-dur 1.0 \
        --save-home "$SAVE_HOME" &
    done

    wait
    echo "==== [DOCKER] All shards finished ===="

    echo "[DOCKER] Files in SAVE_HOME:"
    ls -lh "$SAVE_HOME" || echo "[DOCKER] SAVE_HOME not found"
  '

# ========= 新增：合并部分 =========

# 1) 检查 docker 是否成功
DOCKER_STATUS=$?
echo "[HOST] Docker finished with status: ${DOCKER_STATUS}"

if [ ${DOCKER_STATUS} -ne 0 ]; then
  echo "[HOST] Docker DNSMOS FAILED, skip merging."
  exit 1
fi

# 2) 检查 shard 文件是否存在
echo "[HOST] Checking shard files in ${SAVE_HOME} ..."
shopt -s nullglob
shards=( "${SAVE_HOME}"/dns_from_vad_*_*.tsv )
shopt -u nullglob

if [ ${#shards[@]} -eq 0 ]; then
  echo "[HOST] ERROR: No shard files found matching: ${SAVE_HOME}/dns_from_vad_*_*.tsv"
  exit 1
fi

echo "[HOST] Found shard files:"
printf '  %s\n' "${shards[@]}"

# 3) 调用 merge 脚本，按 JSON 顺序 merge
cd /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos

python3 merge_dns_by_json_order.py \
  --vad-json "${VAD_JSON}" \
  --shard-dir "${SAVE_HOME}" \
  --output "${SAVE_HOME}/dns_from_vad_all.json_order.tsv"

echo "[HOST] Merge done. Output: ${SAVE_HOME}/dns_from_vad_all.json_order.tsv"
