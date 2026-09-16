#!/usr/bin/env python3
"""Sweep Fun-ASR-Nano batch sizes with one resident model."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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


BUCKET_EDGES = [1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--no-timestamps", action="store_true")
    return parser.parse_args()


def sync_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def save_result(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    base_records = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    if not base_records:
        raise RuntimeError("labels file is empty")

    preload_started = time.perf_counter()
    audios = {}
    base_audio_seconds = 0.0
    for record in base_records:
        source = read_audio_from_tar(
            record["tar_path"], record["wav_offset"], record["wav_size"]
        )
        audio = load_audio_seg(source, record["seg_start"], record["seg_end"])
        audio_key = (record["wav_uuid"], float(record["seg_start"]))
        audios[audio_key] = audio
        base_audio_seconds += len(audio) / 16000.0
    preload_seconds = time.perf_counter() - preload_started

    records: List[dict] = []
    for repeat_index in range(args.repeat):
        for record in base_records:
            repeated = dict(record)
            repeated["_repeat_index"] = repeat_index
            records.append(repeated)
    audio_seconds = base_audio_seconds * args.repeat

    load_started = time.perf_counter()
    engine = FunASRNanoBatch(
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
    )
    sync_cuda(args.device)
    model_load_seconds = time.perf_counter() - load_started

    # Remove one-time CUDA initialization from every measured configuration.
    warmup_batch = next(
        iter(DurationBucketBatcher(4, BUCKET_EDGES).batches(records))
    )
    warmup_audios = [
        audios[(record["wav_uuid"], float(record["seg_start"]))]
        for record in warmup_batch
    ]
    warmup_language = LANGUAGE_NAMES.get(warmup_batch[0].get("text_lang"))
    engine.transcribe_batch(
        warmup_audios,
        language=warmup_language,
        return_timestamps=not args.no_timestamps,
    )
    sync_cuda(args.device)

    output_path = Path(args.output)
    report: Dict[str, Any] = {
        "device": args.device,
        "dtype": args.dtype,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "base_segments": len(base_records),
        "repeat": args.repeat,
        "segments": len(records),
        "audio_seconds": audio_seconds,
        "timestamps": not args.no_timestamps,
        "preload_seconds": preload_seconds,
        "model_load_seconds": model_load_seconds,
        "results": [],
    }
    save_result(output_path, report)

    for batch_size in args.batch_sizes:
        batches = list(
            DurationBucketBatcher(batch_size, BUCKET_EDGES).batches(records)
        )
        real_audio_seconds = 0.0
        padded_audio_seconds = 0.0
        for batch in batches:
            durations = [
                len(audios[(record["wav_uuid"], float(record["seg_start"]))])
                / 16000.0
                for record in batch
            ]
            real_audio_seconds += sum(durations)
            padded_audio_seconds += max(durations) * len(durations)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        totals = {
            "prepare_seconds": 0.0,
            "encode_seconds": 0.0,
            "generate_seconds": 0.0,
            "ctc_seconds": 0.0,
        }
        completed_segments = 0
        text_characters = 0
        reference_matches = 0
        text_previews = []
        sync_cuda(args.device)
        infer_started = time.perf_counter()
        try:
            for batch_index, batch in enumerate(batches):
                batch_audios = [
                    audios[(record["wav_uuid"], float(record["seg_start"]))]
                    for record in batch
                ]
                languages = {record.get("text_lang") for record in batch}
                language = (
                    LANGUAGE_NAMES.get(next(iter(languages)))
                    if len(languages) == 1
                    else None
                )
                results = engine.transcribe_batch(
                    batch_audios,
                    keys=[
                        f"{record['wav_uuid']}:{record['seg_start']}:"
                        f"{record['_repeat_index']}:{batch_index}"
                        for record in batch
                    ],
                    language=language,
                    return_timestamps=not args.no_timestamps,
                )
                completed_segments += len(results)
                for record, item in zip(batch, results):
                    text = item.get("text", "")
                    reference = record.get("transcribe")
                    if isinstance(reference, dict):
                        reference = reference.get("funasr-nano")
                    text_characters += len(text)
                    reference_matches += int(reference is not None and text == reference)
                    if len(text_previews) < 5:
                        text_previews.append(
                            {
                                "key": f"{record['wav_uuid']}:{record['seg_start']}",
                                "reference": reference,
                                "text": text,
                            }
                        )
                for key in totals:
                    totals[key] += float(engine.last_stats.get(key, 0.0))
                del results
            sync_cuda(args.device)
        except RuntimeError as exc:
            try:
                sync_cuda(args.device)
            except RuntimeError:
                pass
            failed = {
                "batch_size": batch_size,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "completed_segments": completed_segments,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
            }
            report["results"].append(failed)
            save_result(output_path, report)
            print(json.dumps(failed, ensure_ascii=False), flush=True)
            break

        infer_seconds = time.perf_counter() - infer_started
        result = {
            "batch_size": batch_size,
            "status": "ok",
            "batches": len(batches),
            "segments": completed_segments,
            "audio_seconds": real_audio_seconds,
            "pure_asr_seconds": infer_seconds,
            "pure_asr_x": real_audio_seconds / infer_seconds,
            "padding_efficiency": (
                real_audio_seconds / padded_audio_seconds
                if padded_audio_seconds
                else 1.0
            ),
            "text_characters": text_characters,
            "reference_exact_matches": reference_matches,
            "reference_exact_match_rate": reference_matches / completed_segments,
            "text_previews": text_previews,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
            **totals,
        }
        report["results"].append(result)
        save_result(output_path, report)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
