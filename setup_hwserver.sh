#!/usr/bin/env bash
# 华为服务器环境配置脚本
# 使用方式：bash setup_hwserver.sh
set -e

echo "===== Step 1: 创建 conda 环境 ====="
conda create -n yj_pipeline python=3.10 -y
conda activate yj_pipeline

echo "===== Step 2: 安装系统依赖（需要 sudo，没有就跳过手动装） ====="
sudo apt-get install -y ffmpeg sox libsndfile1 2>/dev/null || \
    echo "[WARN] sudo 失败，请手动确认 ffmpeg/sox 已安装"

echo "===== Step 3: 升级 pip ====="
pip install --upgrade pip setuptools wheel

echo "===== Step 4: 安装 torch cu118 ====="
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

echo "===== Step 5: 安装 pipeline 依赖 ====="
pip install \
    numpy==1.26.4 \
    scipy==1.15.3 \
    soundfile==0.13.1 \
    silero-vad==6.2.0 \
    faster-whisper==1.2.1 \
    ctranslate2==4.6.2 \
    transformers==4.57.1 \
    modelscope==1.31.0 \
    huggingface-hub==0.36.0 \
    sentencepiece==0.2.1 \
    safetensors==0.6.2 \
    av==16.1.0 \
    coloredlogs==15.0.1 \
    tqdm==4.66.5 \
    requests==2.32.3

echo "===== Step 6: 锁定 onnxruntime-gpu（CUDA 12.x 用 1.20.2） ====="
pip install --force-reinstall onnxruntime-gpu==1.20.2 numpy==1.26.4

echo "===== Step 7: 验证环境 ====="
python - << 'PY'
import torch
print(f"torch:       {torch.__version__}")
print(f"cuda_avail:  {torch.cuda.is_available()}")
print(f"gpu_count:   {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"gpu_name:    {torch.cuda.get_device_name(0)}")

import onnxruntime as ort
print(f"onnxrt:      {ort.__version__}  device: {ort.get_device()}")

import faster_whisper, silero_vad, transformers
print(f"faster-whisper: {faster_whisper.__version__}")
print(f"transformers:   {transformers.__version__}")
print("===== ALL OK =====")
PY

echo ""
echo "环境配置完成！运行 pipeline："
echo "  conda activate yj_pipeline"
echo "  bash run_all4user_nodocker.sh"
