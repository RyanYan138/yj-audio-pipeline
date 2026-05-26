#!/usr/bin/env bash
# Docker 容器入口脚本
# 激活 yj_pipeline conda 环境，加载 cuDNN 路径，然后透传执行命令
#
# 用法：
#   docker run ... yj-pipeline:cu122-py312 bash docker_entry.sh bash run_xxx.sh
#   docker run ... yj-pipeline:cu122-py312 bash docker_entry.sh python3 script.py

set -euo pipefail

# 激活 conda 环境
source /opt/conda/etc/profile.d/conda.sh
conda activate yj_pipeline

# cuDNN/cuBLAS（从 pip 包加载，容器内无系统级 cuDNN）
_PY_SITE=/opt/conda/envs/yj_pipeline/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="${_PY_SITE}/cudnn/lib:${_PY_SITE}/cublas/lib:${LD_LIBRARY_PATH:-}"

# 确认环境
echo "[docker_entry] Python: $(python3 --version 2>&1)"
echo "[docker_entry] torch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"

# 执行传入的命令
exec "$@"
