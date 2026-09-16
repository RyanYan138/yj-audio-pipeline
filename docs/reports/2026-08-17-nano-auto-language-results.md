# Fun-ASR-Nano 自动语种与 Pipeline 优化实验报告

日期：2026-08-17

硬件：gpu07，单张 RTX 4090D（GPU 3）

推理后端：手工 Batch Fun-ASR-Nano，非 vLLM
主要精度：FP32（兼容 V100）

## 1. 汇报结论

1. 新增独立 pipeline，取消 ASR 前的语种概率门控，不再把 Whisper Tiny 的语种结果强制传给 Nano。所有合法 VAD 片段先由 Nano 以 `language=None` 转写，再根据转写文本标注 `zh`、`en`、`zh-en` 或 `other` 并做后筛选。
2. 在 ASCEND 中英混合测试集上，旧方案只保留 50.67% 的 mixed 语句；新方案保留 100%，额外救回 74 条 mixed 语句。总体额外救回 271/450 条。
3. 在旧、新方案都保留的相同样本上，新方案总体 MER 为 6.94%，优于旧方案的 7.67%；mixed 子集为 7.75% 对 8.67%。旧方案表面的全量 MER 更低，是因为它提前删掉了大量困难样本，不能直接视为更准。
4. 单卡 FP32、常驻模型、约 1 小时输入的端到端速度：
   - 保留时间戳：118.57X，处理 1 小时约 30.4 秒。
   - 不保留时间戳：160.76X，处理 1 小时约 22.4 秒。
   - 所有正式性能组均为 816 个 ASR 片段、0 个失败片段。
5. 4090D BF16 的无时间戳峰值为 164.41X，只比 FP32 的 160.76X 高约 2.3%；带时间戳 BF16 反而略慢。因此 V100 部署使用 FP32 是合理选择。

## 2. 数据集选择

主实验使用 [ASCEND](https://huggingface.co/datasets/CAiRE/ASCEND)。官方数据卡给出 10.62 小时、12,314 条自发中英对话，含 `zh`、`en` 和 `mixed` 标签，许可证为 CC-BY-SA-4.0。

从 ASCEND test parquet 按固定随机种子 `20260817` 平衡抽取：

| 分组 | 条数 | 音频时长 |
|---|---:|---:|
| 中文 `zh` | 150 | 349.872 秒 |
| 英文 `en` | 150 | 430.079 秒 |
| 句内混合 `mixed` | 150 | 682.843 秒 |
| 合计 | 450 | 1462.794 秒 |

备选数据集调查：

- [CS-Dialogue](https://huggingface.co/datasets/BAAI/CS-Dialogue)：104.02 小时、25.4 GB、CC-BY-NC-SA-4.0。数据仓库的短音频拆成多个大压缩包，今天没有为一次验证搬运完整数据。
- [SEAME / LDC2015S04](https://catalog.ldc.upenn.edu/LDC2015S04)：约 192 小时，但需要 LDC 授权。
- [TALCS](https://www.isca-archive.org/interspeech_2022/li22j_interspeech.html)：约 587 小时且公开，适合后续大规模复验，但本次不需要下载这么大的语料。

## 3. 新旧流程

旧方案：

```text
TAR -> FireRedVAD -> Whisper Tiny LID
    -> 仅保留目标语种且概率 >= 0.90
    -> 按预测语种分桶
    -> Nano 强制使用 LID 语种
```

新方案：

```text
TAR -> 4 路 CPU FireRedVAD
    -> 可选 Whisper Tiny LID（只做审计元数据，默认关闭）
    -> 仅按时长分桶
    -> Nano(language=None)
    -> 根据转写文本判断 zh / en / zh-en / other
    -> ASR 后筛选
```

新方案同时写两个文件：

- `labels.all.json`：保留 Nano 的全部原始预测，用于审计和调整筛选策略。
- `labels.json`：按 `POST_KEEP_LANGS` 进行 ASR 后筛选，默认保留 `zh zh-en`。

## 4. Code-switch A/B 结果

质量 A/B 固定为同一张卡、FP32、Batch 16、保留时间戳、同一 VAD 和同一模型。旧方案使用 `target=zh, min_prob=0.90`；新方案不做前置拒绝，也不强制 Nano 语种。

| 分组 | 输入 | 旧方案留存 | 新方案留存 | 新方案救回 | 旧方案 MER | 新方案 MER | 共同样本旧/新 MER | 新方案英文词召回 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 中文 | 150 | 61.33% | 96.67% | 53 | 6.25% | 7.11% | 6.25% / 5.78% | - |
| 中英混合 | 150 | 50.67% | 100.00% | 74 | 8.67% | 11.61% | 8.67% / 7.75% | 78.42% |
| 英文 | 150 | 0.00% | 96.00% | 144 | - | 21.91% | - | 81.11% |
| 总体 | 450 | 37.33% | 97.56% | 271 | 7.67% | 12.41% | 7.67% / 6.94% | 80.29% |

新方案没有达到 100% 总体留存的 11 条由 VAD/最短 1 秒约束造成，不是语种门控。旧方案的全量 MER 只在其保留的简单子集上计算，存在明显选择偏差；“共同样本 MER”才是公平对比。

代表性救回样本：

- `太expensive`：LID=`en`, prob=0.9023，Nano 完整保留中英混合。
- `因为如果你让AI去drive a car`：LID=`zh`, prob=0.4771，Nano 精确转写。
- `因为他们如果要sell这个technology...take the responsibility...`：LID=`ko`, prob=0.5986，说明单标签 LID 会误判，但 Nano 仍能保留句内切换。

关闭诊断 LID 后，451 个 VAD 片段仍全部输出。与保留 LID 元数据的一次独立运行相比有 14/451 条文本出现轻微生成差异；重新评分后总体 MER 为 12.47%（原 12.41%），英文词召回为 80.08%（原 80.29%），共同样本 MER 均为 6.94%，未发现系统性 code-switch 退化。

## 5. 端到端优化结果

性能基准输入为 3623.712 秒，VAD 后送入 ASR 2481.680 秒，共 816 段。模型通过 Unix socket 常驻，表中端到端时间包含 TAR 解码、VAD、可选 LID、分桶、ASR 和输出，不包含一次性模型加载约 55 秒。

| 配置 | 时间戳 | LID | VAD 进程 | 端到端 | 纯 ASR | Padding 有效率 | 总耗时 |
|---|---|---|---:|---:|---:|---:|---:|
| FP32 Batch 16 | 有 | metadata | 1 | 60.02X | 47.39X | 88.27% | 60.37 秒 |
| FP32 Batch 32 | 有 | metadata | 1 | 74.96X | 72.22X | 87.31% | 48.34 秒 |
| FP32 Batch 96 | 有 | metadata | 1 | 84.11X | 123.78X | 81.30% | 43.08 秒 |
| FP32 Batch 96 | 有 | off | 1 | 92.16X | 122.11X | 81.30% | 39.32 秒 |
| FP32 Batch 96 | 有 | off | 2 | 107.15X | 117.03X | 81.30% | 33.82 秒 |
| **FP32 Batch 96** | **有** | **off** | **4** | **118.57X** | **116.76X** | **81.30%** | **30.56 秒** |
| FP32 Batch 96 | 无 | metadata | 1 | 99.75X | 190.26X | 81.30% | 36.33 秒 |
| FP32 Batch 96 | 无 | off | 4 | 153.80X | 176.15X | 81.30% | 23.56 秒 |
| FP32 Batch 160 | 无 | off | 4 | 158.98X | 206.60X | 79.65% | 22.79 秒 |
| **FP32 Batch 224** | **无** | **off** | **4** | **160.76X** | **247.66X** | **71.72%** | **22.54 秒** |
| BF16 Batch 96 | 有 | off | 4 | 114.80X | 112.84X | 81.30% | 31.57 秒 |
| BF16 Batch 224 | 无 | off | 4 | 164.41X | 261.33X | 71.72% | 22.04 秒 |

短数据会因为 Batch 填不满而更慢：最终默认配置处理 905.9 秒输入的烟测为 11.53 秒，即 78.58X；204 个 ASR 片段、0 个失败，padding 有效率 71.0%。因此生产吞吐应以大数据连续运行测量，不能把大样本峰值直接套到很小的 TAR。

## 6. 瓶颈变化

1. Batch 16 时主要瓶颈仍是 Qwen 自回归生成；第一组约 1 小时实验中，ASR 计算 53.30 秒，其中生成 34.70 秒、CTC 时间戳 7.86 秒、Encoder 6.11 秒、准备 3.96 秒。
2. Batch 提高到 96 后，纯 ASR 已超过 120X，单路 CPU VAD 和 TAR 解码变成上限。
3. 将 FireRedVAD 扩为 4 个独立 CPU 进程后，VAD 阶段从约 26 秒降到 14–17 秒，并和 ASR 并行重叠。
4. Whisper Tiny LID 的 30 秒固定特征导致约 10.33% 的特征 padding 有效率，但实际 GPU 推理只有约 3.3–3.7 秒。它既不参与新方案决策，默认关闭可以省掉特征提取、GPU 显存和一次模型初始化；需要审计时再打开。
5. 关闭 CTC 时间戳后，Batch 96、4 路 VAD 从 118.57X 提高到 153.80X。是否关闭必须由下游是否需要词级时间戳决定。
6. Batch 224 虽达到最高 160.76X，但 padding 只有 71.72%，相对 Batch 160 的提升不足 2X。Batch 96 是保留时间戳的稳妥配置；Batch 160 是无时间戳时更平衡的配置。
7. 常驻 server 现在会在客户端断开时调用 `torch.cuda.empty_cache()`，只释放未使用缓存，不卸载 Nano 权重，避免大 Batch 后占满显存导致下一个 Whisper LID 回退到 CPU。

## 7. 使用方式

### 大数据单次运行，不使用常驻 server

```bash
conda activate /Work21/2025/yanjiahao/conda-envs/funasr_vllm
cd /Work21/2025/yanjiahao/YJ-audio-pipeline/yj-audio-pipeline

bash run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh \
  /path/input.tar /path/output/labels.json 3
```

新 launcher 默认：FP32、Batch 96、4 路 CPU VAD、每个 VAD 进程最多 8 个 PyTorch/OMP 算子线程、`LID_MODE=off`、保留时间戳。gpu07 有 40 个 CPU 核，因此该组实测允许最多约 32 个 VAD 算子线程并行；这里的 4 个 VAD 工作进程仍低于集群规定的 8 个 worker 上限。换到 CPU 核数较少的节点时可通过 `VAD_THREADS=2` 或 `VAD_THREADS=4` 下调，避免过度抢占。

两个新 launcher 会先验证当前 conda Python 是否包含完整 FunASR 依赖；若 SSH/tmux 继承的是 base 环境，会自动回退到同一工作目录下的 `conda-envs/funasr_vllm`。也可以用 `FUNASR_PYTHON=/path/to/python` 明确指定，避免“`CONDA_PREFIX` 有值但环境不对”的隐蔽失败。

### 多次任务复用常驻 server

先在 gpu07 的一个终端启动：

```bash
conda activate /Work21/2025/yanjiahao/conda-envs/funasr_vllm
cd /Work21/2025/yanjiahao/YJ-audio-pipeline/yj-audio-pipeline
bash run_funasr_nano_batch_server_tuned.sh \
  3 /tmp/funasr_nano_gpu3.sock fp32
```

再在同一计算节点的另一个终端运行：

```bash
ASR_SERVER_SOCKET=/tmp/funasr_nano_gpu3.sock \
bash run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh \
  /path/input.tar /path/output/labels.json 3
```

无时间戳高吞吐模式：

```bash
NO_TIMESTAMPS=1 ASR_SERVER_SOCKET=/tmp/funasr_nano_gpu3.sock \
bash run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh \
  /path/input.tar /path/output/labels.json 3 160 32
```

重新开启 Whisper Tiny 诊断元数据：

```bash
LID_MODE=metadata VAD_WORKERS=4 \
bash run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh \
  /path/input.tar /path/output/labels.json 3 96 32
```

限制 VAD 的每进程算子线程数：

```bash
VAD_WORKERS=4 VAD_THREADS=4 \
bash run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh \
  /path/input.tar /path/output/labels.json 3 96 32
```

V100 不支持 BF16，应保持 `ASR_DTYPE=fp32`。4090D 的 Batch 上限和速度不能直接套到 V100；V100 建议先从 Batch 32/64 起测，显存不足时现有代码会自动拆半重试。

## 8. 产物与复现

主要代码：

- `pipeline/tar_pipeline_nodnsmos_nano_auto_lang_ckpt.py`
- `run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh`
- `asr/funasr_nano_batch_server_tuned.py`
- `test/prepare_codeswitch_benchmark.py`
- `test/evaluate_codeswitch_ab.py`
- `test/run_codeswitch_ab.sh`
- `test/run_nano_auto_lang_perf_sweep.sh`

可靠性收尾包括：最终 JSON 原子替换、损坏 JSON 时从 JSONL checkpoint 恢复、Batch 返回数量校验并自动拆分重试、VAD/LID 失败计数、server dtype/模型目录校验，以及拒绝覆盖仍在使用的 Unix socket。

本地报告与指标：

- `output/codeswitch_ab_20260817/codeswitch_ab_report.json`
- `output/codeswitch_ab_20260817/codeswitch_ab_lid_off_report.json`
- `output/nano_auto_lang_perf_20260817/**/metrics.json`

集群原 pipeline 和原 launcher 的 SHA-256 在实验前后完全一致：

```text
7721c329de4504b20bc6b39a4e0c9c16628b1fef92b6a4d7a8421b9ac40e85d3  pipeline/tar_pipeline_nodnsmos_bucket_batch_ckpt.py
16c0270b454fcda2e65ce2fda0890d4300d65acd4915d50e5bc986c6a92077b3  run_tar_pipeline_nodnsmos_bucket_batch_ckpt.sh
```

实验结束后 gpu07 无残留 tmux 任务，GPU 3 显存占用为 0 MiB。

## 9. 限制

- 质量实验目前只完成 ASCEND test 的平衡子集；CS-Dialogue、TALCS 等跨域复验尚未执行。
- 性能基准通过同一 TAR 的四个硬链接组成约 1 小时输入，文件系统已预热，更接近稳定计算吞吐，不代表冷盘首次读取速度。
- 性能数字使用常驻模型，不包含约 55 秒模型加载；单次小任务应报告冷启动时间。
- 这里的脚本级 `zh/en/zh-en` 判断用于后筛选和审计，不等同于经过训练的逐词语种标注器。
