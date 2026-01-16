import os
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count

import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# ====================== 配置区：只改这里 ======================

###################环境Huawei_Encoder_Vad
# 要扫描的音频根目录
INPUT_ROOT = "/CDShare3/Huawei_Encoder_Proj/datas/LibriSpeech"

# VAD 结果 JSON 输出路径
OUT_JSON = "/CDShare3/Huawei_Encoder_Proj/codes/jiahao/vad/vadout/librispeech_silero_vad_segments_mp_Ordered.json"

# 丢掉时长小于这个值的段（秒）
MIN_DUR = 0.75

# 多进程 worker 数（None 表示自动根据 CPU 数取一个比较稳的值）
NUM_WORKERS = 32

# 最多处理多少个音频文件；设为 None 表示全部文件，调试时可以设 1000
MAX_FILES = 10

# ============================================================

SAMPLE_RATE = 16000
_model = None  # 每个进程自己的模型实例


def init_worker():
    """子进程初始化时加载一次模型（避免重复下载 & pickle 问题）"""
    global _model
    _model = load_silero_vad()
    torch.set_num_threads(1)


def collect_audio_files(root: Path):
    exts = {".wav", ".flac",".mp3"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            yield p


def make_audio_id(root: Path, audio_path: Path) -> str:
    rel = audio_path.relative_to(root)
    return rel.with_suffix("").as_posix().replace("/", "_")


def process_one(args):
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
    min_dur: float = 0.0,
    num_workers: int | None = None,
    max_files: int | None = None,
):
    input_root = Path(input_root)

    # 1) 收集并排序所有音频文件，保证顺序稳定
    audio_files = sorted(collect_audio_files(input_root), key=lambda p: str(p))

    if max_files is not None:
        audio_files = audio_files[:max_files]

    total = len(audio_files)
    print(f"Found {total} audio files under {input_root}"
          f"{' (limited by MAX_FILES)' if max_files is not None else ''}")

    # 2) 决定 worker 数
    if num_workers is None:
        num_workers = max(1, min(8, (cpu_count() or 4)))
    print(f"Using {num_workers} workers")

    tasks = [
        (str(input_root), str(p), min_dur)
        for p in audio_files
    ]

    all_items = [None] * total  # 为了按 index 写入，顺序 = audio_files 顺序

    # 3) 有序 imap，保证结果顺序与 tasks 一致
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        for idx, item in enumerate(pool.imap(process_one, tasks)):
            all_items[idx] = item
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{total} files")

    # 4) 输出 JSON
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    print(f"Done. Indexed {len(all_items)} files into {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", default=INPUT_ROOT)
    ap.add_argument("--out_json", default=OUT_JSON)
    ap.add_argument("--min_dur", type=float, default=MIN_DUR)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--max_files", type=int, default=MAX_FILES)
    args = ap.parse_args()

    build_vad_index_mp(
        input_root=args.input_root,
        out_json=args.out_json,
        min_dur=args.min_dur,
        num_workers=args.num_workers,
        max_files=args.max_files,
    )

