#!/usr/bin/env python3
"""Benchmark only Fun-ASR-Nano on segments produced by the cleaning pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from asr.funasr_nano_batch import FunASRNanoBatch
from pipeline.tar_pipeline_nodnsmos_bucket_batch_ckpt import (
    LANGUAGE_NAMES,
    DurationBucketBatcher,
    load_audio_seg,
    read_audio_from_tar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-timestamps", action="store_true")
    return parser.parse_args()


def sync_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    records = json.load(open(args.labels, encoding="utf-8"))

    preload_started = time.perf_counter()
    audios = {}
    audio_seconds = 0.0
    for record in records:
        source = read_audio_from_tar(
            record["tar_path"], record["wav_offset"], record["wav_size"]
        )
        audio = load_audio_seg(source, record["seg_start"], record["seg_end"])
        record_key = (record["wav_uuid"], float(record["seg_start"]))
        audios[record_key] = audio
        audio_seconds += len(audio) / 16000.0
    preload_seconds = time.perf_counter() - preload_started

    load_started = time.perf_counter()
    engine = FunASRNanoBatch(
        model_dir=args.model_dir,
        device=args.device,
        dtype="fp32",
    )
    sync_cuda(args.device)
    model_load_seconds = time.perf_counter() - load_started

    bucket_edges = [1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 30]
    batches = list(
        DurationBucketBatcher(args.batch_size, bucket_edges).batches(records)
    )

    totals = {
        "prepare_seconds": 0.0,
        "encode_seconds": 0.0,
        "generate_seconds": 0.0,
        "ctc_seconds": 0.0,
    }
    text_characters = 0
    sync_cuda(args.device)
    infer_started = time.perf_counter()
    for batch_index, batch in enumerate(batches):
        batch_audios = [
            audios[(record["wav_uuid"], float(record["seg_start"]))]
            for record in batch
        ]
        languages = {record.get("text_lang") for record in batch}
        language = None
        if len(languages) == 1:
            language = LANGUAGE_NAMES.get(next(iter(languages)))
        results = engine.transcribe_batch(
            batch_audios,
            keys=[
                f"{record['wav_uuid']}:{record['seg_start']}:{batch_index}"
                for record in batch
            ],
            language=language,
            return_timestamps=not args.no_timestamps,
        )
        text_characters += sum(len(result.get("text", "")) for result in results)
        for key in totals:
            totals[key] += float(engine.last_stats.get(key, 0.0))
    sync_cuda(args.device)
    infer_seconds = time.perf_counter() - infer_started

    result = {
        "segments": len(records),
        "batches": len(batches),
        "batch_size": args.batch_size,
        "timestamps": not args.no_timestamps,
        "audio_seconds": audio_seconds,
        "preload_seconds": preload_seconds,
        "model_load_seconds": model_load_seconds,
        "pure_asr_seconds": infer_seconds,
        "pure_asr_x": audio_seconds / infer_seconds,
        "cold_model_plus_asr_seconds": model_load_seconds + infer_seconds,
        "cold_model_plus_asr_x": audio_seconds / (model_load_seconds + infer_seconds),
        "text_characters": text_characters,
        **totals,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
