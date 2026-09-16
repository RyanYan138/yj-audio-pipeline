#!/usr/bin/env python3
"""Strict stage-by-stage variant of the optimized audio cleaning pipeline.

Stages are separated by barriers:

    tar decode + FireRedVAD -> batched tiny Whisper LID -> bucketed Nano ASR

Unlike the streaming pipeline, the next model stage starts only after the
previous stage has completely finished. Intermediate audio stays in memory,
so this mode is intended for one TAR or another bounded input shard at a time.
"""

from __future__ import annotations

import bisect
import json
import multiprocessing as mp
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from pipeline.tar_pipeline_nodnsmos_bucket_batch_ckpt import (
    LOGGER,
    SENTINEL,
    _collect_stage_stats,
    funasr_bucket_worker,
    lid_worker_batched,
    parse_args,
    producer,
    vad_worker,
)


def _drain_until_sentinel(
    output_q: mp.Queue, processes: Sequence[mp.Process]
) -> List[dict]:
    items: List[dict] = []
    while True:
        try:
            item = output_q.get(timeout=0.5)
        except queue.Empty:
            failed = [
                process
                for process in processes
                if process.exitcode is not None and process.exitcode != 0
            ]
            if failed:
                details = ", ".join(
                    f"{process.name}={process.exitcode}" for process in failed
                )
                raise RuntimeError(f"Cascade stage failed: {details}")
            if all(process.exitcode is not None for process in processes):
                try:
                    item = output_q.get(timeout=2)
                except queue.Empty as exc:
                    raise RuntimeError("Cascade stage exited without a sentinel") from exc
            else:
                continue
        if item is SENTINEL:
            return items
        items.append(item)


def _join_success(processes: Iterable[mp.Process]) -> None:
    for process in processes:
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise RuntimeError(f"Cascade stage timed out: {process.name}")
        if process.exitcode != 0:
            raise RuntimeError(
                f"Cascade process {process.name} exited with {process.exitcode}"
            )


def _feed_queue(input_q: mp.Queue, items: Sequence[dict]) -> None:
    for item in items:
        input_q.put(item)
    input_q.put(SENTINEL)


def _run_decode_vad(args: Any, context: mp.context.BaseContext):
    input_q = context.Queue(maxsize=args.queue_maxsize)
    output_q = context.Queue(maxsize=args.queue_maxsize)
    stats_q = context.Queue()
    processes = [
        context.Process(
            target=vad_worker,
            args=(
                input_q,
                output_q,
                stats_q,
                args.min_dur,
                args.max_dur,
                args.fireredvad_model,
                args.fireredvad_root,
                args.vad_device,
            ),
            name="CascadeVAD",
        ),
        context.Process(
            target=producer,
            args=(args.tar_paths, input_q, stats_q),
            name="CascadeProducer",
        ),
    ]
    started = time.perf_counter()
    try:
        for process in processes:
            process.start()
        items = _drain_until_sentinel(output_q, processes)
        _join_success(processes)
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    return items, _collect_stage_stats(stats_q, expected=2), time.perf_counter() - started


def _run_lid(args: Any, context: mp.context.BaseContext, vad_items: Sequence[dict]):
    input_q = context.Queue(maxsize=args.queue_maxsize)
    output_q = context.Queue(maxsize=args.queue_maxsize)
    stats_q = context.Queue()
    process = context.Process(
        target=lid_worker_batched,
        args=(
            input_q,
            output_q,
            stats_q,
            args.lid_model_dir,
            args.target_langs,
            args.min_lang_prob,
            args.lid_batch_size,
            args.lid_sample_seconds,
        ),
        name="CascadeLID",
    )
    started = time.perf_counter()
    feeder = threading.Thread(
        target=_feed_queue,
        args=(input_q, vad_items),
        name="CascadeLIDFeeder",
        daemon=True,
    )
    try:
        process.start()
        feeder.start()
        segments = _drain_until_sentinel(output_q, [process])
        feeder.join(timeout=60)
        if feeder.is_alive():
            raise RuntimeError("Cascade LID feeder did not finish")
        _join_success([process])
    except BaseException:
        if process.is_alive():
            process.terminate()
        raise
    stats = _collect_stage_stats(stats_q, expected=1)
    return segments, stats, time.perf_counter() - started


def _duration_sort_key(segment: dict, edges: Sequence[float]):
    duration = float(segment["seg_end"] - segment["seg_start"])
    return (
        str(segment.get("text_lang", "unknown")),
        bisect.bisect_left(edges, duration),
        duration,
    )


def _run_asr(args: Any, context: mp.context.BaseContext, segments: Sequence[dict]):
    input_q = context.Queue(maxsize=args.queue_maxsize)
    stats_q = context.Queue()
    out_dir = str(Path(args.out_json).parent)
    checkpoint_path = os.path.join(out_dir, "cascade_asr_ckpt.jsonl")
    process = context.Process(
        target=funasr_bucket_worker,
        args=(
            input_q,
            stats_q,
            args.funasr_model_dir,
            args.batch_size,
            args.bucket_edges,
            args.out_json,
            checkpoint_path,
            args.audio_dir,
            not args.no_timestamps,
            args.force_language_from_lid,
            args.resume,
            args.asr_prefetch_batches,
            args.bucket_lookahead_batches,
            args.asr_server_socket,
        ),
        name="CascadeASR",
    )
    ordered = sorted(segments, key=lambda item: _duration_sort_key(item, args.bucket_edges))
    feeder = threading.Thread(
        target=_feed_queue,
        args=(input_q, ordered),
        name="CascadeASRFeeder",
        daemon=True,
    )
    started = time.perf_counter()
    try:
        process.start()
        feeder.start()
        while process.is_alive():
            process.join(timeout=0.5)
        feeder.join(timeout=60)
        if feeder.is_alive():
            raise RuntimeError("Cascade ASR feeder did not finish")
        _join_success([process])
    except BaseException:
        if process.is_alive():
            process.terminate()
        raise
    stats = _collect_stage_stats(stats_q, expected=1)
    return stats, time.perf_counter() - started


def _checkpoint_metadata(path: Path, items: Sequence[dict], audio_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            record = {key: value for key, value in item.items() if key != audio_key}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_json or out_dir / "metrics.json")
    context = mp.get_context("spawn")
    pipeline_started = time.perf_counter()

    LOGGER.info("=== Cascade stage 1/3: tar decode + FireRedVAD ===")
    vad_items, decode_vad_stats, decode_vad_seconds = _run_decode_vad(args, context)
    _checkpoint_metadata(out_dir / "cascade_vad_ckpt.jsonl", vad_items, "audio")
    LOGGER.info(
        "Cascade stage 1 done: %d source items, %.2fs",
        len(vad_items),
        decode_vad_seconds,
    )

    LOGGER.info("=== Cascade stage 2/3: batched tiny Whisper LID ===")
    segments, lid_stats, lid_seconds = _run_lid(args, context, vad_items)
    vad_items.clear()
    _checkpoint_metadata(out_dir / "cascade_lid_ckpt.jsonl", segments, "segment_audio")
    LOGGER.info(
        "Cascade stage 2 done: %d accepted segments, %.2fs",
        len(segments),
        lid_seconds,
    )

    LOGGER.info("=== Cascade stage 3/3: global duration buckets + Nano ASR ===")
    asr_stats_map, asr_seconds = _run_asr(args, context, segments)
    segments.clear()
    total_seconds = time.perf_counter() - pipeline_started

    stage_stats: Dict[str, dict] = {
        **decode_vad_stats,
        **lid_stats,
        **asr_stats_map,
    }
    missing = {"producer", "vad", "lid", "asr"} - set(stage_stats)
    if missing:
        raise RuntimeError(f"Missing cascade metrics: {sorted(missing)}")
    input_audio_seconds = stage_stats["producer"]["input_audio_seconds"]
    asr_stats = stage_stats["asr"]
    accepted_audio_seconds = asr_stats["asr_audio_seconds"]
    asr_compute_seconds = asr_stats["asr_batch_compute_seconds"]
    metrics: Dict[str, Any] = {
        "mode": "strict_cascade",
        "input_audio_seconds": input_audio_seconds,
        "accepted_audio_seconds": accepted_audio_seconds,
        "total_seconds": total_seconds,
        "end_to_end_x": input_audio_seconds / total_seconds,
        "accepted_audio_x": accepted_audio_seconds / total_seconds,
        "asr_compute_x": accepted_audio_seconds / asr_compute_seconds,
        "stage_wall_seconds": {
            "decode_vad": decode_vad_seconds,
            "lid": lid_seconds,
            "asr": asr_seconds,
        },
        "stage_stats": stage_stats,
        **{key: value for key, value in asr_stats.items() if key != "stage"},
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    LOGGER.info(
        "Cascade complete: %.2fs total, %.2fX end-to-end, output=%s",
        total_seconds,
        metrics["end_to_end_x"],
        args.out_json,
    )


if __name__ == "__main__":
    main()
