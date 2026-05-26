# Models

将模型放到对应子目录，clone 后直接可用：

```
models/
├── whisper-large-v3/          # Whisper ASR（HuggingFace 格式）
├── faster-whisper-large-v3/   # faster-whisper ASR + LID（CTranslate2 格式）
└── FunASR-Nano/               # FunASR Nano ASR
```

## 下载方式

```bash
# faster-whisper（LID 和 ASR 都用这个，推荐先下这个）
huggingface-cli download Systran/faster-whisper-large-v3 \
    --local-dir models/faster-whisper-large-v3

# FunASR Nano
modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --local_dir models/FunASR-Nano

# Whisper HuggingFace 版（可选，只有 run_all4user_nodocker.sh 用）
modelscope download --model AI-ModelScope/whisper-large-v3 \
    --local_dir models/whisper-large-v3
```
