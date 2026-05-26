#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 加速版 Whisper 转写
用 vLLM 替换 HuggingFace pipeline，利用 continuous batching 提升吞吐量

vLLM >= 0.6.0 支持 Whisper（openai/whisper-large-v3 格式）
输入/输出接口与 whisper_ms_from_segments.py 完全对齐

用法：
  python3 whisper_vllm_from_segments.py \
    --seg_json /path/to/segments.json \
    --out_json /path/to/output.json \
    --model_dir /path/to/whisper-large-v3 \
    --device cuda:0 \
    --tensor_parallel_size 2 \
    [--num_shards 2 --shard_id 0]

依赖：pip install vllm>=0.6.0
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seg_json", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--model_dir", default=
        "/Work21/2025/yanjiahao/modelscope_cache/models/AI-ModelScope/whisper-large-v3")
    p.add_argument("--device", default=None,
                   help="cuda:0 / cpu；默认自动")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="vLLM tensor 并行数（多卡时设为 GPU 数）")
    p.add_argument("--max_num_seqs", type=int, default=32,
                   help="vLLM 最大并发序列数（continuous batching）")
    p.add_argument("--language", default=None,
                   help="zh / en / None=自动检测")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--max_files", type=int, default=None)
    return p.parse_args()


def load_segments(seg_json: str, num_shards: int, shard_id: int,
                  max_files: Optional[int]) -> List[dict]:
    with open(seg_json, encoding="utf-8") as f:
        data = json.load(f)
    if max_files:
        data = data[:max_files]
    flat = []
    for item in data:
        for i, seg in enumerate(item.get("segments", [])):
            flat.append({
                "seg_id": f"{item['audio_id']}_seg{i:03d}",
                "audio_id": item["audio_id"],
                "path": item["path"],
                "start_sec": float(seg["start_sec"]),
                "end_sec": float(seg["end_sec"]),
                "duration_sec": float(seg["duration_sec"]),
                **{k: item.get(k) or seg.get(k)
                   for k in ["lang", "lang_prob", "mos_sig", "mos_bak", "mos_ovr",
                              "spk_id", "sim_spk", "sim_file"]
                   if item.get(k) is not None or seg.get(k) is not None},
            })
    shard = [s for j, s in enumerate(flat) if j % num_shards == shard_id]
    logger.info(f"Loaded {len(flat)} segments, shard {shard_id}/{num_shards}: {len(shard)}")
    return shard


def read_segment(path: str, start_sec: float, end_sec: float) -> np.ndarray:
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        f.seek(int(start_sec * sr))
        audio = f.read(int((end_sec - start_sec) * sr), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        import torchaudio
        a = torch.from_numpy(audio).unsqueeze(0)
        audio = torchaudio.functional.resample(a, sr, SAMPLE_RATE).squeeze(0).numpy()
    return audio


def main():
    args = parse_args()

    try:
        from vllm import LLM, SamplingParams
        from vllm.assets.audio import AudioAsset
    except ImportError:
        logger.error("vLLM 未安装，请先 pip install vllm>=0.6.0")
        sys.exit(1)

    segments = load_segments(args.seg_json, args.num_shards, args.shard_id, args.max_files)
    if not segments:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump([], f)
        return

    # 初始化 vLLM（Whisper encoder-decoder）
    logger.info(f"Loading vLLM Whisper from {args.model_dir} "
                f"tp={args.tensor_parallel_size} max_seqs={args.max_num_seqs}")
    llm = LLM(
        model=args.model_dir,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="float16",
        trust_remote_code=True,
        # 音频模态输入
        limit_mm_per_prompt={"audio": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=448,
    )

    results = []
    tmp_dir = tempfile.mkdtemp(prefix="vllm_whisper_")

    # 批量推理 — vLLM continuous batching，一次性提交所有请求
    inputs = []
    valid_segs = []
    for seg in segments:
        try:
            audio = read_segment(seg["path"], seg["start_sec"], seg["end_sec"])
            tmp_path = os.path.join(tmp_dir, f"{seg['seg_id']}.wav")
            sf.write(tmp_path, audio, SAMPLE_RATE)
            prompt = "<|startoftranscript|>"
            if args.language:
                prompt += f"<|{args.language}|>"
            else:
                prompt += "<|notimestamps|>"

            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"audio": tmp_path},
            })
            valid_segs.append(seg)
        except Exception as ex:
            logger.warning(f"[vLLM] load failed {seg['seg_id']}: {ex}")
            results.append({**seg, "text": ""})

    logger.info(f"Submitting {len(inputs)} segments to vLLM...")
    outputs = llm.generate(inputs, sampling_params)

    for seg, out in zip(valid_segs, outputs):
        text = out.outputs[0].text.strip() if out.outputs else ""
        results.append({**seg, "text": text})

    # 清理临时文件
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 按原始顺序排序输出
    seg_order = {s["seg_id"]: i for i, s in enumerate(segments)}
    results.sort(key=lambda x: seg_order.get(x.get("seg_id", ""), 99999))

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Done. Wrote {len(results)} segments to {args.out_json}")


if __name__ == "__main__":
    main()
