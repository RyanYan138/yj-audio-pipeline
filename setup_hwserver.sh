#!/usr/bin/env bash
# 华为云 Notebook 环境配置脚本
# 目标：在现有 vllm-qwen 环境里补装 pipeline 缺失依赖
# 使用：conda activate vllm-qwen && bash setup_hwserver.sh
#
# 已有（不动）：torch==2.9.0+cu128 / numpy==2.2.6 / soundfile / silero-vad / av / transformers
# 需要补装：faster-whisper / ctranslate2 / modelscope / onnxruntime-gpu（替换 CPU 版）
set -e

echo "===== 当前环境检查 ====="
python -c "import torch; print('torch:', torch.__version__, '  cuda:', torch.cuda.is_available())"

echo "===== 补装 pipeline 缺失依赖 ====="
pip install \
    faster-whisper==1.2.1 \
    ctranslate2==4.6.2 \
    modelscope==1.31.0 \
    coloredlogs==15.0.1

echo "===== 替换 onnxruntime 为 GPU 版（CUDA 12.x + numpy 2.x 兼容）====="
# onnxruntime-gpu 1.20.2：支持 CUDA 12.x / cuDNN 9.x / numpy 2.x
pip install --force-reinstall onnxruntime-gpu==1.20.2

echo "===== 验证 ====="
python - << 'PY'
import torch
print(f"torch:          {torch.__version__}  cuda={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}")

import onnxruntime as ort
print(f"onnxruntime:    {ort.__version__}  device={ort.get_device()}")
print(f"ort providers:  {ort.get_available_providers()}")

import faster_whisper, ctranslate2, silero_vad, transformers, modelscope
print(f"faster-whisper: {faster_whisper.__version__}")
print(f"ctranslate2:    {ctranslate2.__version__}")
print(f"transformers:   {transformers.__version__}")
print(f"modelscope:     {modelscope.__version__}")
import soundfile, av, scipy, numpy
print(f"numpy:          {numpy.__version__}")
print(f"scipy:          {scipy.__version__}")
print(f"soundfile:      {soundfile.__version__}")
print("===== ALL OK =====")
PY

echo ""
echo "运行 pipeline："
echo "  conda activate vllm-qwen"
echo "  bash run_all4user_nodocker.sh"
