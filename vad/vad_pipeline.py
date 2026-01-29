#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Silero VAD -> segments json (ordered)

与 run_all.bash 配套：
  python3 vad_pipeline.py \
    --input_root ... \
    --out_json ... \
    --min_dur ... \
    --num_workers ... \
    [--max_files N]

约定：
- 输出是一个 list，顺序严格按照 input_root 下的音频文件路径排序后的顺序
- 每条包含：audio_id, path, num_segments, segments[]
- segments 每段包含：start_sec, end_sec, duration_sec
"""

import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Iterable, List, Dict, Any, Optional, Tuple

import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

SAMPLE_RATE = 16000
_model = None  # 每个子进程各自的模型实例


def init_worker():
    """子进程初始化：加载一次模型 + 限制 torch 线程，避免 CPU 打满"""
    global _model
    _model = load_silero_vad()
    torch.set_num_threads(1)


def collect_audio_files(root: Path) -> Iterable[Path]:
    """递归收集音频文件"""
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def make_audio_id(root: Path, audio_path: Path) -> str:
    """
    用相对路径生成 audio_id，保证稳定且不含扩展名
    e.g. root/a/b/xxx.wav -> a_b_xxx
    """
    rel = audio_path.relative_to(root)
    return rel.with_suffix("").as_posix().replace("/", "_")


def process_one(args: Tuple[str, str, float]) -> Dict[str, Any]:
    """处理单个音频：跑 VAD -> segments"""
    root_str, audio_path_str, min_dur = args
    root = Path(root_str)
    audio_path = Path(audio_path_str)
    audio_id = make_audio_id(root, audio_path)

    try:
        wav = read_audio(str(audio_path))
        timestamps = get_speech_timestamps(
            wav, _model, sampling_rate=SAMPLE_RATE, return_seconds=True
        )
    except Exception as e:
        return {
            "audio_id": audio_id,
            "path": str(audio_path),
            "num_segments": 0,
            "segments": [],
            "error": str(e),
        }

    segments = []
    for t in timestamps:
        start_sec = float(t["start"])
        end_sec = float(t["end"])
        dur = end_sec - start_sec
        if dur < min_dur:
            continue
        segments.append(
            {
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(dur, 3),
            }
        )

    return {
        "audio_id": audio_id,
        "path": str(audio_path),
        "num_segments": len(segments),
        "segments": segments,
    }


def build_vad_index_mp(
    input_root: str,
    out_json: str,
    min_dur: float,
    num_workers: int,
    max_files: int,
):
    input_root_p = Path(input_root)
    if not input_root_p.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")

    # 1) 收集 + 排序，保证顺序稳定
    audio_files = sorted(collect_audio_files(input_root_p), key=lambda p: str(p))

    # 2) max_files 约定：0 或负数 => 不限制
    if max_files and max_files > 0:
        audio_files = audio_files[:max_files]

    total = len(audio_files)
    limit_info = f" (limited to {max_files})" if (max_files and max_files > 0) else ""
    print(f"[VAD] Found {total} audio files under {input_root_p}{limit_info}")

    # 3) worker 数：<=0 则自动
    if num_workers <= 0:
        num_workers = max(1, min(32, cpu_count() or 4))
    print(f"[VAD] Using {num_workers} workers, min_dur={min_dur}")

    tasks = [(str(input_root_p), str(p), float(min_dur)) for p in audio_files]

    all_items: List[Optional[Dict[str, Any]]] = [None] * total

    # 4) 有序 imap，结果顺序与 tasks 一致
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        for idx, item in enumerate(pool.imap(process_one, tasks, chunksize=8)):
            all_items[idx] = item
            if (idx + 1) % 1000 == 0:
                print(f"[VAD] Processed {idx + 1}/{total}")

    # 防御：理论不会 None
    out_list: List[Dict[str, Any]] = [x if x is not None else {} for x in all_items]

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)

    print(f"[VAD] Done. Wrote {len(out_list)} items -> {out_path}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True, help="root dir containing audio files")
    ap.add_argument("--out_json", required=True, help="output segments json path")
    ap.add_argument("--min_dur", type=float, default=0.75, help="drop segments shorter than this (sec)")
    ap.add_argument("--num_workers", type=int, default=32, help="mp workers; <=0 means auto")
    ap.add_argument("--max_files", type=int, default=0, help="0 means all files; otherwise process first N files")
    args = ap.parse_args()

    build_vad_index_mp(
        input_root=args.input_root,
        out_json=args.out_json,
        min_dur=args.min_dur,
        num_workers=args.num_workers,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
