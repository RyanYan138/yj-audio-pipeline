#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faster-whisper 版 ASR 转写（CTranslate2 后端）
接口与 whisper_ms_from_segments.py 完全一致
无 HuggingFace numpy 类型兼容问题
"""
import argparse, json, logging, os
from pathlib import Path
from typing import Optional, List
import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
SAMPLE_RATE = 16000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seg_json", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--model_dir", default=
        "/Work21/2025/yanjiahao/hf_cache/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--language", default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--max_files", type=int, default=None)
    return p.parse_args()


def load_segments(seg_json, num_shards, shard_id, max_files):
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
                **{k: item.get(k) for k in
                   ["lang","lang_prob","mos_sig","mos_bak","mos_ovr","spk_id","sim_spk","sim_file"]
                   if item.get(k) is not None},
            })
    shard = [s for j, s in enumerate(flat) if j % num_shards == shard_id]
    logger.info(f"Loaded {len(flat)} segments, shard {shard_id}/{num_shards}: {len(shard)}")
    return shard


def read_segment(path, start_sec, end_sec):
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        f.seek(int(start_sec * sr))
        audio = f.read(int((end_sec - start_sec) * sr), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        import torchaudio
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio).unsqueeze(0), sr, SAMPLE_RATE
        ).squeeze(0).numpy()
    return audio


def main():
    args = parse_args()
    raw_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # faster-whisper 不支持 cuda:N 格式，拆成 device + device_index
    if ":" in raw_device:
        device, device_index = raw_device.split(":")
        device_index = int(device_index)
    else:
        device, device_index = raw_device, 0
    logger.info(f"Device: {device}[{device_index}]  model: {args.model_dir}")
    logger.info(f"Shard: {args.shard_id}/{args.num_shards}")

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model_dir, device=device, device_index=device_index,
                         compute_type="float16")

    segments = load_segments(args.seg_json, args.num_shards, args.shard_id, args.max_files)
    if not segments:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump([], f)
        return

    results = []
    for i, seg in enumerate(segments):
        try:
            audio = read_segment(seg["path"], seg["start_sec"], seg["end_sec"])
            segs_out, _ = model.transcribe(
                audio,
                beam_size=5,
                language=args.language,
            )
            text = " ".join(s.text for s in segs_out).strip()
        except Exception as ex:
            logger.warning(f"[SKIP] {seg['seg_id']}: {ex}")
            text = ""
        results.append({**seg, "text": text})
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(segments)}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Done. Wrote {len(results)} segments to {args.out_json}")


if __name__ == "__main__":
    main()
