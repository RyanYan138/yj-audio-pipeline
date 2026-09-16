#!/usr/bin/env python3
"""Live correctness check for the standalone Fun-ASR-Nano batch engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from asr.funasr_nano_batch import FunASRNanoBatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("audio", nargs="+", help="At least two audio paths")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.audio) < 2:
        raise SystemExit("Provide at least two audio files")

    inputs = args.audio[:2]
    keys = [Path(path).stem for path in inputs]
    engine = FunASRNanoBatch(
        model_dir=args.model_dir,
        device=args.device,
        dtype="fp32",
        max_new_tokens=128,
    )

    batched = engine.transcribe_batch(
        inputs,
        keys=keys,
        return_timestamps=True,
    )
    batch_stats = dict(engine.last_stats)
    sequential = [
        engine.transcribe_batch(
            [audio],
            keys=[key],
            return_timestamps=False,
        )[0]
        for audio, key in zip(inputs, keys)
    ]

    batch_texts = [item["text"] for item in batched]
    sequential_texts = [item["text"] for item in sequential]
    if batch_texts != sequential_texts:
        raise AssertionError(
            f"Batch/sequential mismatch: {batch_texts!r} != {sequential_texts!r}"
        )
    if not all(item.get("timestamps") for item in batched):
        raise AssertionError("At least one batched result has no LLM timestamps")
    if not all(item.get("ctc_timestamps") for item in batched):
        raise AssertionError("At least one batched result has no CTC timestamps")

    print(
        json.dumps(
            {
                "status": "ok",
                "batch_stats": batch_stats,
                "texts": batch_texts,
                "timestamp_counts": [
                    len(item["timestamps"]) for item in batched
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
