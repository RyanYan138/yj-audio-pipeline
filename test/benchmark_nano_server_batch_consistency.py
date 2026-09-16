#!/usr/bin/env python3
"""Compare persistent Nano server batch texts with singleton decoding."""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.tar_pipeline_nodnsmos_bucket_batch_ckpt import read_audio_from_tar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--limit", type=int, default=24)
    return parser.parse_args()


def transcribe(client, records, audios):
    client.send(
        {
            "command": "transcribe",
            "audios": audios,
            "keys": [record["wav_uuid"] for record in records],
            "language": "英文",
            "return_timestamps": False,
        }
    )
    response = client.recv()
    if not response.get("ok"):
        raise RuntimeError(response.get("error"))
    return [item["text"] for item in response["results"]]


def main() -> None:
    args = parse_args()
    records = json.load(open(args.labels, encoding="utf-8"))[: args.limit]
    audios = []
    for record in records:
        source = read_audio_from_tar(
            record["tar_path"], record["wav_offset"], record["wav_size"]
        )
        start = int(record["seg_start"] * 16000)
        end = int(record["seg_end"] * 16000)
        audios.append(source[start:end])

    client = Client(args.socket, family="AF_UNIX")
    started = time.perf_counter()
    singleton = [
        transcribe(client, [record], [audio])[0]
        for record, audio in zip(records, audios)
    ]
    singleton_seconds = time.perf_counter() - started
    comparisons = {}
    for batch_size in (8, 12, 16):
        started = time.perf_counter()
        batched = []
        for begin in range(0, len(records), batch_size):
            batched.extend(
                transcribe(
                    client,
                    records[begin : begin + batch_size],
                    audios[begin : begin + batch_size],
                )
            )
        elapsed = time.perf_counter() - started
        mismatches = [
            {
                "index": index,
                "singleton": expected,
                "batch": actual,
            }
            for index, (expected, actual) in enumerate(zip(singleton, batched))
            if expected != actual
        ]
        comparisons[str(batch_size)] = {
            "seconds": elapsed,
            "speedup": singleton_seconds / elapsed,
            "exact_agreement": 1.0 - len(mismatches) / len(records),
            "mismatches": mismatches,
        }
    client.close()
    print(
        json.dumps(
            {
                "items": len(records),
                "singleton_seconds": singleton_seconds,
                "comparisons": comparisons,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
