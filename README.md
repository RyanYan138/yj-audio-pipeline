# YJ Audio Pipeline

音频数据清洗 + 转写流水线。输入一批音频（散装或 tar 包），输出带时间戳的转写结果 JSON。

流程：**VAD 切段 → DNSMOS 音质过滤 → 语种识别（LID）→ FunASR Nano vLLM 转写**

支持中文 / 英文，带字级时间戳。

---

## 第一步：clone 代码

```bash
git clone git@github.com:RyanYan138/yj-audio-pipeline.git
cd yj-audio-pipeline
```

---

## 第二步：把模型放进来

clone 下来之后 `models/` 目录是空的，需要手动把模型放进去。

**目录结构长这样：**

```
yj-audio-pipeline/
└── models/
    ├── Fun-ASR-Nano-2512/        ← FunASR 转写模型（3.5G）
    ├── faster-whisper-large-v3/  ← 语种识别模型（2.9G）
    └── FireRedVAD/               ← VAD 模型权重，放两个文件
        ├── model.pth.tar
        └── cmvn.ark
```

**下载方式：**

```bash
# FunASR Nano（转写用）
modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --local_dir models/Fun-ASR-Nano-2512

# faster-whisper-large-v3（语种识别用）
huggingface-cli download Systran/faster-whisper-large-v3 \
    --local-dir models/faster-whisper-large-v3

# FireRedVAD（VAD 用，只需要两个权重文件）
# 从 https://huggingface.co/xukaituo/FireRedVAD 下载：
#   pretrained_models/FireRedVAD/VAD/model.pth.tar
#   pretrained_models/FireRedVAD/VAD/cmvn.ark
# 放到 models/FireRedVAD/ 下
```

> `dns_mos/` 里的 DNSMOS 模型已经在仓库里，不用额外下载。

---

## 第三步：准备环境

### 方式 A：用 Docker（推荐，game 节点）

直接用现有镜像，不用装任何东西：

```bash
# 在 game 节点上运行
docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=2 \
  -v /你的/yj-audio-pipeline:/workspace \
  -v /你的/数据目录:/你的/数据目录 \
  yj-pipeline:funasr-vllm \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate funasr_vllm && bash /workspace/run_tar_pipeline.sh /workspace/test/test_4.tar /workspace/output/test/labels.json 2"
```

### 方式 B：裸机 conda

```bash
conda activate funasr_vllm
# 然后直接跑脚本
```

---

## 第四步：跑起来

### 输入是 tar 包（推荐）

```bash
# 用法：bash run_tar_pipeline.sh [输入tar] [输出json] [GPU编号]

# 最简单（用默认测试数据）
bash run_tar_pipeline.sh

# 指定输入输出
bash run_tar_pipeline.sh /data/my_audio.tar /data/output/labels.json 0

# 用 GPU 1
bash run_tar_pipeline.sh /data/my_audio.tar /data/output/labels.json 1
```

### 输入是散装音频目录

```bash
# 用法：bash run_fireredvad_funasr_vllm.sh [输入目录] [输出目录] [GPU编号]

bash run_fireredvad_funasr_vllm.sh /data/wavs/ /data/output/ 0
```

---

## 输出格式

输出是一个 `labels.json`，每条记录长这样：

```json
{
  "tar_path": "/data/my_audio.tar",
  "wav_uuid": "xxxx",
  "seg_start": 0.28,
  "seg_end": 10.12,
  "duration": 9.84,
  "text_lang": "zh",
  "transcribe": {
    "funasr-nano": "下一步将进一步完善城投债券发行制度和防范风险机制。"
  },
  "timestamp": {
    "funasr-nano": [["下", 0.3, 0.36], ["一", 0.54, 0.6], ...]
  }
}
```

---

## 常见问题

**Q：跑脚本报 `[ERROR] 请先 conda activate funasr_vllm`**
A：先 `conda activate funasr_vllm` 再跑。

**Q：报 `No such file or directory` 找不到模型**
A：检查 `models/` 下面的目录名有没有对上，参考第二步的目录结构。

**Q：报 `No CUDA GPUs are available`**
A：docker 启动时没加 `--gpus all`，或者 GPU 编号不对。
