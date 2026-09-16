#!/usr/bin/env python3
"""Benchmark fixed-30s and dynamic-length CTranslate2 Whisper LID batches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ctranslate2
import numpy as np
from faster_whisper.feature_extractor import FeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.tar_reader import iter_tar_wavs
from pipeline.tar_pipeline_nodnsmos_bucket_batch_ckpt import (
    BatchedWhisperFeatureExtractor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--sample-seconds", type=float, default=12.0)
    return parser.parse_args()


def run_mode(model, extractor, audios, batch_size, dynamic):
    feature_seconds = 0.0
    inference_seconds = 0.0
    detected = []
    frame_counts = []
    for begin in range(0, len(audios), batch_size):
        batch = audios[begin : begin + batch_size]
        feature_started = time.perf_counter()
        if dynamic:
            max_samples = max(len(audio) for audio in batch)
            waveforms = [
                np.pad(audio, (0, max_samples - len(audio))) for audio in batch
            ]
            features = [extractor(audio, padding=False) for audio in waveforms]
        else:
            features = [
                extractor(audio, padding=True)[:, : extractor.nb_max_frames]
                for audio in batch
            ]
        batch_array = np.stack(features).astype(np.float32, copy=False)
        feature_seconds += time.perf_counter() - feature_started
        frame_counts.append(int(batch_array.shape[-1]))

        inference_started = time.perf_counter()
        outputs = model.detect_language(
            ctranslate2.StorageView.from_array(batch_array)
        )
        inference_seconds += time.perf_counter() - inference_started
        detected.extend(
            (candidates[0][0].strip("<>|"), float(candidates[0][1]))
            for candidates in outputs
        )
    return {
        "feature_seconds": feature_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": feature_seconds + inference_seconds,
        "frame_counts": frame_counts,
        "detected": detected,
    }


def main() -> None:
    args = parse_args()
    max_samples = int(args.sample_seconds * 16000)
    audios = []
    input_seconds = 0.0
    for entry in iter_tar_wavs(args.tar):
        audio = np.ascontiguousarray(entry.audio[:max_samples], dtype=np.float32)
        if audio.size:
            audios.append(audio)
            input_seconds += len(audio) / 16000
        if len(audios) >= args.limit:
            break

    model = ctranslate2.models.Whisper(
        args.model_dir,
        device="cuda",
        device_index=0,
        compute_type="float16",
        inter_threads=1,
        intra_threads=2,
    )
    extractor = FeatureExtractor(feature_size=80)
    fixed = run_mode(model, extractor, audios, args.batch_size, dynamic=False)
    dynamic = run_mode(model, extractor, audios, args.batch_size, dynamic=True)
    vectorized_extractor = BatchedWhisperFeatureExtractor(feature_size=80)
    vectorized_started = time.perf_counter()
    vectorized_detected = []
    vectorized_feature_seconds = 0.0
    vectorized_inference_seconds = 0.0
    for begin in range(0, len(audios), args.batch_size):
        batch = audios[begin : begin + args.batch_size]
        started = time.perf_counter()
        features = vectorized_extractor(batch)
        vectorized_feature_seconds += time.perf_counter() - started
        started = time.perf_counter()
        outputs = model.detect_language(ctranslate2.StorageView.from_array(features))
        vectorized_inference_seconds += time.perf_counter() - started
        vectorized_detected.extend(
            (candidates[0][0].strip("<>|"), float(candidates[0][1]))
            for candidates in outputs
        )
    vectorized_seconds = time.perf_counter() - vectorized_started
    agreement = sum(
        left[0] == right[0]
        for left, right in zip(fixed["detected"], dynamic["detected"])
    ) / len(audios)

    print(
        json.dumps(
            {
                "items": len(audios),
                "input_seconds": input_seconds,
                "agreement": agreement,
                "fixed": fixed,
                "dynamic": dynamic,
                "vectorized": {
                    "feature_seconds": vectorized_feature_seconds,
                    "inference_seconds": vectorized_inference_seconds,
                    "total_seconds": vectorized_seconds,
                    "detected": vectorized_detected,
                },
                "fixed_vectorized_agreement": sum(
                    left[0] == right[0]
                    for left, right in zip(fixed["detected"], vectorized_detected)
                )
                / len(audios),
                "fixed_vectorized_max_probability_delta": max(
                    abs(left[1] - right[1])
                    for left, right in zip(fixed["detected"], vectorized_detected)
                ),
                "dynamic_speedup": fixed["total_seconds"]
                / dynamic["total_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
