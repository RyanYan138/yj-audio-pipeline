#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunASR Nano vLLM ASR 转写 — 接口与 funasr_nano_from_segments.py 对齐
使用 AutoModelVLLM，批量并发推理，RTFx 约 340（相比普通模式 21）

用法：
  python3 funasr_nano_vllm_from_segments.py \
    --seg_json   /path/to/segments.json \
    --out_json   /path/to/output.json \
    --model_dir  /Work21/2025/yanjiahao/modelscope_cache/models/FunAudioLLM/Fun-ASR-Nano-2512 \
    --device     0 \
    --batch_size 64 \
    [--num_shards 2 --shard_id 0] \
    [--language zh]
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

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
    p.add_argument("--seg_json", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--model_dir", type=str,
        default="/Work21/2025/yanjiahao/modelscope_cache/models/FunAudioLLM/Fun-ASR-Nano-2512")
    p.add_argument("--device", type=str, default="0",
        help="GPU index，例如 0 / 1；vllm 用 CUDA_VISIBLE_DEVICES 控制")
    p.add_argument("--batch_size", type=int, default=64,
        help="一次送给 vllm 的音频条数，越大吞吐越高")
    p.add_argument("--language", type=str, default="auto",
        help="zh / en / auto；默认 auto")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--max_files", type=int, default=None)
    return p.parse_args()


def load_segments(seg_json: str, num_shards: int, shard_id: int, max_files=None):
    with open(seg_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_files:
        data = data[:max_files]
    flat = []
    for item in data:
        audio_id = item["audio_id"]
        path = item["path"]
        for i, seg in enumerate(item.get("segments", [])):
            flat.append({
                "seg_id": f"{audio_id}_seg{i:03d}",
                "audio_id": audio_id,
                "path": path,
                "start_sec": float(seg["start_sec"]),
                "end_sec": float(seg["end_sec"]),
                "duration_sec": float(seg["duration_sec"]),
            })
    shard = [s for j, s in enumerate(flat) if j % num_shards == shard_id]
    logger.info(f"Loaded {len(flat)} segments total, this shard: {len(shard)}")
    return shard


def read_segment(path: str, start_sec: float, end_sec: float, tmp_dir: str) -> str:
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        start_frame = int(start_sec * sr)
        end_frame = int(end_sec * sr)
        f.seek(start_frame)
        audio = f.read(end_frame - start_frame, dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        import torchaudio
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio_t = torchaudio.functional.resample(audio_t, sr, SAMPLE_RATE)
        audio = audio_t.squeeze(0).numpy()
    tmp_path = os.path.join(tmp_dir, f"{os.getpid()}_{hash(path)}_{start_sec:.3f}.wav")
    sf.write(tmp_path, audio, SAMPLE_RATE)
    return tmp_path


def main():
    args = parse_args()

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # LD_LIBRARY_PATH for cusparseLt（容器内必须）
    cusparselt_path = "/opt/conda/envs/funasr_vllm/lib/python3.12/site-packages/nvidia/cusparselt/lib"
    if os.path.exists(cusparselt_path):
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{cusparselt_path}:{existing}" if existing else cusparselt_path

    logger.info(f"GPU: {args.device}  model: {args.model_dir}")
    logger.info(f"Shard: {args.shard_id}/{args.num_shards}  batch_size: {args.batch_size}")

    from funasr.auto.auto_model_vllm import AutoModelVLLM
    model = AutoModelVLLM(
        model=args.model_dir,
        tensor_parallel_size=1,
    )

    segments = load_segments(args.seg_json, args.num_shards, args.shard_id, args.max_files)
    if not segments:
        logger.warning("No segments to process, writing empty output")
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump([], f)
        return

    tmp_dir = tempfile.mkdtemp(prefix="funasr_vllm_seg_")
    results_map = {}  # seg_id -> result dict

    # 准备临时音频文件
    tmp_paths = {}
    for seg in segments:
        try:
            tmp_paths[seg["seg_id"]] = read_segment(
                seg["path"], seg["start_sec"], seg["end_sec"], tmp_dir
            )
        except Exception as e:
            logger.warning(f"[SKIP read] {seg['seg_id']}: {e}")
            tmp_paths[seg["seg_id"]] = None

    # 分批调用 vllm（批量并发，吞吐大幅提升）
    valid_segs = [s for s in segments if tmp_paths.get(s["seg_id"])]
    failed_segs = [s for s in segments if not tmp_paths.get(s["seg_id"])]

    for seg in failed_segs:
        results_map[seg["seg_id"]] = {**seg, "text": ""}

    total = len(valid_segs)
    for batch_start in range(0, total, args.batch_size):
        batch = valid_segs[batch_start: batch_start + args.batch_size]
        batch_paths = [tmp_paths[s["seg_id"]] for s in batch]

        try:
            res_list = model.generate(batch_paths)
            for seg, res in zip(batch, res_list):
                if isinstance(res, dict):
                    text = res.get("text", "").strip()
                else:
                    text = str(res).strip()
                results_map[seg["seg_id"]] = {**seg, "text": text}
        except Exception as e:
            logger.warning(f"[BATCH FAIL] batch {batch_start}: {e}")
            for seg in batch:
                results_map[seg["seg_id"]] = {**seg, "text": ""}

        # 清理临时文件
        for p in batch_paths:
            try:
                os.remove(p)
            except Exception:
                pass

        done = min(batch_start + args.batch_size, total)
        logger.info(f"  {done}/{total} segments done")

    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass

    # 按原始顺序输出
    results = [results_map[s["seg_id"]] for s in segments]

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Done. Wrote {len(results)} segments to {args.out_json}")


if __name__ == "__main__":
    main()
