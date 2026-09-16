# Models

模型文件不纳入 Git。当前推荐的 Nano 自动语种 pipeline 需要下面前三项：

```text
models/
├── Fun-ASR-Nano-2512/         # 推荐：Fun-ASR-Nano 权重
├── FireRedVAD/                # 推荐：FireRedVAD 检测模型
├── faster-whisper-tiny/       # 可选：Whisper Tiny，仅 LID_MODE=metadata 使用
├── faster-whisper-large-v3/   # 旧 Whisper/faster-whisper 对照方案
├── whisper-large-v3/          # 旧 HuggingFace Whisper 方案
└── whisper-large-v3-hf/       # HF batch 对照方案
```

## 推荐方案模型

```bash
# Fun-ASR-Nano：目录名必须与 launcher 约定一致
modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 \
  --local_dir models/Fun-ASR-Nano-2512

# Whisper Tiny：只有 LID_MODE=metadata 才需要
huggingface-cli download Systran/faster-whisper-tiny \
  --local-dir models/faster-whisper-tiny
```

FireRedVAD 权重放到 `models/FireRedVAD/`，并在仓库根目录准备 FireRedVAD 源码目录。具体版本应与集群已验证环境保持一致。

## 历史对照方案模型

```bash
# faster-whisper（旧 LID / ASR 对照）
huggingface-cli download Systran/faster-whisper-large-v3 \
  --local-dir models/faster-whisper-large-v3

# HuggingFace Whisper（可选）
modelscope download --model AI-ModelScope/whisper-large-v3 \
  --local_dir models/whisper-large-v3
```

详见仓库根目录的 [`README.md`](../README.md)。
