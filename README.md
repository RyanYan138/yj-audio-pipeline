# YJ Audio Pipeline

面向 TAR 音频数据清洗的 Fun-ASR-Nano Batch 推理 Pipeline。当前推荐方案不依赖 vLLM，使用 FP32，可部署在不支持 BF16 的 V100 上；它以 FireRedVAD 切分音频、按时长分桶进行手工 Batch ASR，并在 Nano 转写后做文本级语种筛选。

## 当前推荐方案

请使用自动语种 Pipeline：

```bash
conda activate /Work21/2025/yanjiahao/conda-envs/funasr_vllm
cd /Work21/2025/yanjiahao/YJ-audio-pipeline/yj-audio-pipeline

bash run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh \
  /path/to/input.tar \
  /path/to/output/labels.json \
  0 \
  96 \
  32
```

五个位置参数依次为：

| 位置 | 参数 | 示例 | 含义 |
|---:|---|---|---|
| 1 | `INPUT.tar` | `/path/to/input.tar` | 输入 TAR 音频包 |
| 2 | `OUTPUT.json` | `/path/to/output/labels.json` | 最终筛选后的结果 |
| 3 | `GPU` | `0` | 运行 Pipeline 的可见 GPU 编号 |
| 4 | `ASR_BATCH` | `96` | Fun-ASR-Nano 单个 Batch 的最大片段数 |
| 5 | `LID_BATCH` | `32` | 仅 `LID_MODE=metadata` 时使用的 Whisper Tiny Batch |

默认配置为 FP32、`ASR_BATCH=96`、4 路 CPU VAD、每路 8 个算子线程、`LID_MODE=off`、保留时间戳。对于 V100，建议从 `ASR_BATCH=32` 或 `64` 开始，再按显存逐步上调。

## 最新方案的数据流

```text
TAR 解码 -> FireRedVAD -> 按时长分桶 -> Fun-ASR-Nano language=None
        -> 根据转写文本标记 zh / zh-en / en / other -> 后置筛选
```

这个方案不会在 ASR 前因为 Whisper Tiny 的语种置信度删除音频，也不会把 Whisper Tiny 的单语种预测强制传给 Nano。它会产生两份结果：

- `labels.all.json`：Nano 对全部合法 VAD 片段的完整转写，适合审计和重新设定筛选策略。
- `labels.json`：由 `POST_KEEP_LANGS` 从完整结果筛选得到的最终结果，默认保留 `zh zh-en`。

`zh-en` 不是 Nano 官方返回的语种标签：它是 Pipeline 根据 Nano 转写文本中同时出现中文字符和英文单词而生成的文本脚本标签。

## 常用配置

### 保留中文和中英混合，关闭诊断 LID

这是默认生产配置。Whisper Tiny 不加载、不推理；所有 VAD 片段都先进入 Nano。

```bash
LID_MODE=off \
POST_KEEP_LANGS="zh zh-en" \
bash run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh \
  /path/to/input.tar /path/to/output/labels.json 0 96 32
```

### 保留所有转写结果

```bash
LID_MODE=off \
POST_KEEP_LANGS="zh zh-en en other" \
bash run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh \
  /path/to/input.tar /path/to/output/labels.json 0 96 32
```

也可以始终直接读取同目录的 `labels.all.json`，其中不做后置筛选。

### 记录 LID 审计元数据，但不让它筛数据

```bash
LID_MODE=metadata \
POST_KEEP_LANGS="zh zh-en" \
bash run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh \
  /path/to/input.tar /path/to/output/labels.json 0 96 32
```

`LID_MODE=metadata` 表示 Whisper Tiny 对每条源音频最长的 VAD 片段做一次整段级语种预测，并在结果中写入 `lid_lang` 和 `lid_prob`。它仅用于审计；不会删除片段，也不会控制 Nano 的语种提示。`LID_MODE=off` 则完全跳过该模型。

### 无时间戳高吞吐模式

当下游不需要 CTC 词级时间戳时：

```bash
NO_TIMESTAMPS=1 \
LID_MODE=off \
POST_KEEP_LANGS="zh zh-en" \
bash run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh \
  /path/to/input.tar /path/to/output/labels.json 0 160 32
```

### 断点续跑

```bash
RESUME=1 \
bash run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh \
  /path/to/input.tar /path/to/output/labels.json 0 96 32
```

## 多次任务使用常驻 Nano Server

对于多次小到中等规模任务，可先启动常驻 server，让 Nano 权重持续驻留在显存中，避免每次约 55 秒的模型加载。server 和 Pipeline 必须在同一个计算节点，但可以通过各自的 `CUDA_VISIBLE_DEVICES` 使用不同 GPU。

终端 A：

```bash
conda activate /Work21/2025/yanjiahao/conda-envs/funasr_vllm
cd /Work21/2025/yanjiahao/YJ-audio-pipeline/yj-audio-pipeline

bash run_funasr_nano_batch_server_tuned.sh \
  0 /tmp/funasr_nano_gpu0.sock fp32
```

终端 B：

```bash
ASR_SERVER_SOCKET=/tmp/funasr_nano_gpu0.sock \
LID_MODE=off \
bash run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh \
  /path/to/input.tar /path/to/output/labels.json 0 96 32
```

没有任务运行时请停止 server，避免长期占用 GPU。大规模连续任务可以不启常驻 server，因为一次性加载成本会被大量音频摊薄。

## 环境变量

| 变量 | 默认值 | 作用 |
|---|---|---|
| `ASR_DTYPE` | `fp32` | `fp32` 或 `bf16`；V100 使用 `fp32` |
| `POST_KEEP_LANGS` | `zh zh-en` | ASR 后文本语种白名单 |
| `NO_TIMESTAMPS` | `0` | 设为 `1` 时关闭 CTC 时间戳 |
| `LID_MODE` | `off` | `off` 或 `metadata` |
| `VAD_WORKERS` | `4` | 并行 FireRedVAD 进程数，建议不超过 8 |
| `VAD_THREADS` | `8` | 每个 VAD 进程的算子线程上限 |
| `RESUME` | `0` | 设为 `1` 时从已完成记录继续 |
| `ASR_SERVER_SOCKET` | 空 | 常驻 Nano server 的 Unix socket 路径 |
| `FUNASR_PYTHON` | 自动发现 | 显式指定包含 FunASR 依赖的 Python |

## 启动脚本说明

| 脚本 | 状态 | 用途 |
|---|---|---|
| `run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh` | 推荐 | 最新自动语种 Pipeline。Nano 先转写，文本后筛选。 |
| `run_funasr_nano_batch_server_tuned.sh` | 推荐 | 最新 FP32/BF16 可选的常驻 Nano Batch server。 |
| `run_tar_pipeline_nodnsmos_bucket_batch_ckpt.sh` | 历史对照 | 旧流水线：Whisper Tiny 在 ASR 前按目标语种和阈值硬筛选。 |
| `run_tar_pipeline_nodnsmos_cascade_batch_ckpt.sh` | 历史对照 | 旧方案的严格级联版：VAD、LID 完整结束后再做全局分桶 ASR。 |
| `run_funasr_nano_batch_server.sh` | 兼容旧版 | 早期常驻 server，不建议新任务使用。 |
| `run_4090d_batch_sweep.sh` | 性能实验 | 4090D 上的 Batch 配置扫描。 |
| `test/run_nano_auto_lang_perf_sweep.sh` | 性能实验 | 最新方案的端到端与纯 ASR 性能扫描。 |
| `test/run_codeswitch_ab.sh` | 质量实验 | ASCEND 上的旧/新 code-switch A/B 对照。 |

以下脚本属于仓库保留的早期方案；它们使用不同 ASR 后端或带 DNSMOS 的质量筛选，不应与推荐方案混用。

| 脚本 | 适用场景 |
|---|---|
| `run_tar_pipeline.sh` | 原始 FireRedVAD + Fun-ASR Nano vLLM TAR 流水线。需要 vLLM，V100 不适用。 |
| `run_tar_pipeline_ckpt.sh` | 原始带 DNSMOS 的 checkpoint 流水线，ASR 前按 LID 阈值筛选。 |
| `run_tar_pipeline_nodnsmos_ckpt.sh` | 原始无 DNSMOS checkpoint 流水线，仍有 ASR 前 LID 筛选。 |
| `run_tar_pipeline_hf_batch_ckpt.sh` | HuggingFace Whisper 的 batch ASR 对照方案。 |
| `run_tar_pipeline_whisper_batch_ckpt.sh` | faster-whisper batch ASR 对照方案。 |
| `run_all4user_nodocker.sh` | 更早的目录输入、全流程配置式脚本；需要在脚本顶部修改配置。 |
| `run_fireredvad_funasr*.sh`、`run_fireredvad_whisper.sh`、`run_silero_whisper_fast.sh` | 单阶段/旧基准脚本，供历史复现与排障。 |

## 目录与依赖

模型和数据不会上传到 GitHub，需要自行准备：

```text
models/Fun-ASR-Nano-2512/     Fun-ASR-Nano 权重
models/FireRedVAD/            FireRedVAD 模型文件
models/faster-whisper-tiny/   Whisper Tiny LID 模型，仅 metadata 模式需要
FireRedVAD/                   FireRedVAD 源码目录
```

Python 环境至少需要：`torch`、`funasr`、`transformers`、`numpy`、`soundfile`。如需 `LID_MODE=metadata`，还需要 `ctranslate2` 和 `faster-whisper`。模型目录、下载方式及旧方案额外依赖见 [`models/README.md`](models/README.md)。

本项目的主要验证脚本位于 `test/`。V100 使用前请先在目标节点以 FP32、Batch 32 或 64 做小规模显存与吞吐探测，再逐步调大 Batch。

## 已验证的结论

在单张 RTX 4090D、FP32、常驻模型、约 1 小时输入的实验中：

| 配置 | 端到端速度 | 说明 |
|---|---:|---|
| Batch 96，保留时间戳，4 路 VAD，LID off | 118.57X | 推荐的带时间戳生产配置 |
| Batch 160，无时间戳，4 路 VAD，LID off | 158.98X | 无时间戳时吞吐与 Padding 的平衡点 |
| Batch 224，无时间戳，4 路 VAD，LID off | 160.76X | 测得峰值，Padding 效率更低 |

质量验证中，ASCEND 官方 `mixed` 子集的留存率从旧方案的 50.67% 提升到新方案的 100%。实验详情见 `docs/reports/2026-08-17-nano-auto-language-results.md`。

## 注意事项

- `LID_MODE=metadata` 给的是整段音频级 LID 元数据，不是逐词或逐字符 LID。
- `text_lang=zh-en` 是根据 Nano 转写文本自动生成的脚本级标签，不是 FunASR 官方逐词语种标注。
- `labels.all.json` 是审计和重新筛选的事实来源；不要仅保留 `labels.json` 后就删除它。
- V100 不支持 BF16，必须设定 `ASR_DTYPE=fp32`。
- 4090D 的 Batch 上限不能直接迁移到 V100；请从 Batch 32/64 做实际显存和吞吐测试。
