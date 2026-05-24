#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunASR Nano ASR 转写 — 接口与 whisper_ms_from_segments.py 完全对齐

用法：
  python3 funasr_nano_from_segments.py \
    --seg_json   /path/to/segments.json \
    --out_json   /path/to/output.json \
    --model_dir  /Work21/2025/yanjiahao/modelscope_cache/models/FunAudioLLM/Fun-ASR-Nano-2512 \
    --device     cuda:0 \
    --batch_size 16 \
    [--num_shards 2 --shard_id 0] \
    [--language zh]
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import os
import tempfile
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
    p.add_argument("--device", type=str, default=None,
        help="cuda:0 / cuda:1 / cpu；默认自动选第一块可用 GPU")
    p.add_argument("--batch_size", type=int, default=16)
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
    """切片音频并写入临时文件，返回临时文件路径（FunASR Nano 只接受文件路径输入）"""
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

    # 设备
    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}  model: {args.model_dir}")
    logger.info(f"Shard: {args.shard_id}/{args.num_shards}")

    # 加载 FunASR Nano
    from funasr import AutoModel
    model = AutoModel(
        model=args.model_dir,
        device=device,
        disable_update=True,
    )

    segments = load_segments(args.seg_json, args.num_shards, args.shard_id, args.max_files)
    if not segments:
        logger.warning("No segments to process, writing empty output")
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump([], f)
        return

    results = []
    # FunASR Nano 只接受文件路径，逐条推理（不支持 batch 也不支持 numpy array）
    lang_arg = {} if args.language == "auto" else {"language": args.language}
    tmp_dir = tempfile.mkdtemp(prefix="funasr_seg_")

    for i, seg in enumerate(segments):
        tmp_path = None
        try:
            tmp_path = read_segment(seg["path"], seg["start_sec"], seg["end_sec"], tmp_dir)
        except Exception as e:
            logger.warning(f"[SKIP] {seg['seg_id']}: {e}")
            results.append({**seg, "text": ""})
            continue

        try:
            res = model.generate(input=tmp_path, **lang_arg)
            if isinstance(res, list) and len(res) > 0:
                r = res[0]
                text = r.get("text", "").strip() if isinstance(r, dict) else str(r).strip()
            else:
                text = ""
        except Exception as e:
            logger.warning(f"[ASR FAIL] {seg['seg_id']}: {e}")
            text = ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        results.append({**seg, "text": text})

        if (i + 1) % 20 == 0:
            logger.info(f"  {i+1}/{len(segments)}")

    # 清理临时目录
    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Done. Wrote {len(results)} segments to {args.out_json}")


if __name__ == "__main__":
    main()
