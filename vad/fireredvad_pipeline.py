#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FireRedVAD pipeline — 输出格式与 silero vad_pipeline.py 完全一致

用法：
  python3 fireredvad_pipeline.py \
    --input_root /path/to/audio \
    --out_json   /path/to/output.json \
    --model_dir  /Work21/2025/yanjiahao/FireRedVAD/pretrained_models/xukaituo/FireRedVAD/VAD \
    --fireredvad_root /Work21/2025/yanjiahao/FireRedVAD \
    --min_dur 1.0 \
    --max_dur 30.0 \
    [--use_gpu 1] \
    [--num_workers 8] \
    [--max_files N]
"""

import argparse
import json
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_vad = None
_vad_model_dir = None
_use_gpu = False


def init_worker(model_dir: str, use_gpu: bool, fireredvad_root: str):
    global _vad, _vad_model_dir, _use_gpu
    if fireredvad_root not in sys.path:
        sys.path.insert(0, fireredvad_root)
    from fireredvad.vad import FireRedVad, FireRedVadConfig
    _vad_model_dir = model_dir
    _use_gpu = use_gpu
    cfg = FireRedVadConfig(
        use_gpu=use_gpu,
        smooth_window_size=5,
        speech_threshold=0.5,
        min_speech_frame=20,
        max_speech_frame=2000,
        min_silence_frame=10,
        merge_silence_frame=50,
        extend_speech_frame=5,
    )
    _vad = FireRedVad.from_pretrained(model_dir, cfg)


def collect_audio_files(root: Path) -> List[Path]:
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
    files = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    return files


def make_audio_id(root: Path, audio_path: Path) -> str:
    try:
        rel = audio_path.relative_to(root)
        return str(rel.with_suffix("")).replace(os.sep, "/")
    except ValueError:
        return audio_path.stem


def process_one(args: Tuple) -> Optional[Dict[str, Any]]:
    root_str, audio_path_str, min_dur, max_dur = args
    global _vad
    audio_path = Path(audio_path_str)
    root = Path(root_str)
    audio_id = make_audio_id(root, audio_path)

    try:
        result, _ = _vad.detect(str(audio_path))
        timestamps = result.get("timestamps", [])  # list of [start_sec, end_sec]
        total_dur = result.get("dur", 0.0)

        segments = []
        for seg in timestamps:
            s, e = float(seg[0]), float(seg[1])
            dur = e - s
            if dur < min_dur or dur > max_dur:
                continue
            segments.append({
                "start_sec": round(s, 4),
                "end_sec": round(e, 4),
                "duration_sec": round(dur, 4),
            })

        return {
            "audio_id": audio_id,
            "path": str(audio_path),
            "num_segments": len(segments),
            "segments": segments,
        }
    except Exception as e:
        logger.warning(f"[SKIP] {audio_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--model_dir",
        default="/Work21/2025/yanjiahao/FireRedVAD/pretrained_models/xukaituo/FireRedVAD/VAD")
    parser.add_argument("--fireredvad_root",
        default="/Work21/2025/yanjiahao/FireRedVAD")
    parser.add_argument("--min_dur", type=float, default=1.0)
    parser.add_argument("--max_dur", type=float, default=30.0)
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    if not input_root.exists():
        logger.error(f"input_root 不存在: {input_root}")
        sys.exit(1)

    if input_root.is_file():
        files = [input_root]
    else:
        files = collect_audio_files(input_root)

    if args.max_files:
        files = files[:args.max_files]

    logger.info(f"[FireRedVAD] Found {len(files)} audio files under {input_root}")
    logger.info(f"[FireRedVAD] min_dur={args.min_dur}s  max_dur={args.max_dur}s  workers={args.num_workers}")

    task_args = [(str(input_root), str(f), args.min_dur, args.max_dur) for f in files]

    results = []
    # FireRedVAD 不是多进程安全的（模型初始化重），用单进程
    init_worker(args.model_dir, bool(args.use_gpu), args.fireredvad_root)
    t0 = time.time()
    for i, ta in enumerate(task_args):
        r = process_one(ta)
        if r is not None:
            results.append(r)
        if (i + 1) % 50 == 0:
            logger.info(f"  processed {i+1}/{len(task_args)}")
    logger.info(f"[FireRedVAD] Done in {time.time()-t0:.1f}s. Wrote {len(results)} items")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"[FireRedVAD] Saved -> {out_path}")


if __name__ == "__main__":
    main()
