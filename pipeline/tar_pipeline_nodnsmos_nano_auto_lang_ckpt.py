#!/usr/bin/env python3
"""Auto-language tar audio cleaning pipeline for non-vLLM inference.

Pipeline:
    tar decode -> FireRedVAD -> batched tiny Whisper LID metadata
               -> duration buckets -> Fun-ASR-Nano auto language -> post filter

The LID result never rejects a segment and is never passed as a language prompt
to Nano. Language filtering happens only after transcription.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import sys
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s %(message)s",
)
LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SENTINEL = None
LANGUAGE_NAMES = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
}
EN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def setup_runtime_libraries() -> None:
    """Expose CUDA libraries from the active environment before CT2 import."""
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        return
    candidates: List[str] = []
    candidates.extend(
        glob.glob(f"{conda_prefix}/lib/python3.*/site-packages/nvidia/*/lib")
    )
    candidates.extend(
        glob.glob(f"{conda_prefix}/lib/python3.*/site-packages/ctranslate2.libs")
    )
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [path for path in candidates if os.path.isdir(path)]
    if existing:
        paths.append(existing)
    if paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths)


def load_audio_seg(audio: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
    start = max(0, int(start_sec * SAMPLE_RATE))
    end = min(len(audio), int(end_sec * SAMPLE_RATE))
    return np.ascontiguousarray(audio[start:end], dtype=np.float32)


def classify_transcript_language(text: str) -> Dict[str, Any]:
    """Classify transcript scripts without making a pre-ASR decision."""
    zh_chars = sum("\u4e00" <= char <= "\u9fff" for char in text)
    en_tokens = EN_TOKEN_RE.findall(text)
    if zh_chars and en_tokens:
        language = "zh-en"
    elif zh_chars:
        language = "zh"
    elif en_tokens:
        language = "en"
    else:
        language = "other"
    return {
        "language": language,
        "zh_chars": zh_chars,
        "en_tokens": en_tokens,
    }


def post_filter_records(
    records: Sequence[dict], keep_langs: Sequence[str]
) -> List[dict]:
    keep = set(keep_langs)
    return [record for record in records if record.get("text_lang") in keep]


def atomic_write_json(path: str, value: Any) -> None:
    """Write a complete JSON document before atomically publishing it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_resume_records(paths: Sequence[str]) -> List[dict]:
    """Load valid final/checkpoint records and de-duplicate segment UUIDs."""
    records: List[dict] = []
    seen = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            if path.endswith(".jsonl"):
                with open(path, encoding="utf-8") as handle:
                    candidates = []
                    for line in handle:
                        try:
                            candidates.append(json.loads(line))
                        except (json.JSONDecodeError, TypeError):
                            continue
            else:
                with open(path, encoding="utf-8") as handle:
                    candidates = json.load(handle)
                if not isinstance(candidates, list):
                    raise ValueError("result JSON must contain a list")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("[Resume] ignoring unreadable %s: %s", path, exc)
            continue
        for record in candidates:
            segment_uuid = record.get("wav_uuid") if isinstance(record, dict) else None
            if segment_uuid and segment_uuid not in seen:
                seen.add(segment_uuid)
                records.append(record)
    return records


def validate_batch_result_count(results: Sequence[dict], expected: int) -> None:
    if len(results) != expected:
        raise RuntimeError(
            f"Nano returned {len(results)} results for {expected} inputs"
        )


def validate_server_dtype(response: dict, expected_dtype: str) -> str:
    actual_dtype = response.get("dtype")
    if not actual_dtype:
        raise RuntimeError("Nano server ping is missing dtype metadata")
    actual_dtype = str(actual_dtype)
    if actual_dtype != expected_dtype:
        raise RuntimeError(
            f"Nano server dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
        )
    return actual_dtype


def validate_server_model(response: dict, expected_model_dir: str) -> str:
    actual_model_dir = response.get("model_dir")
    if not actual_model_dir:
        raise RuntimeError("Nano server ping is missing model_dir metadata")
    actual_model_dir = os.path.realpath(str(actual_model_dir))
    expected_model_dir = os.path.realpath(expected_model_dir)
    if actual_model_dir != expected_model_dir:
        raise RuntimeError(
            "Nano server model mismatch: "
            f"expected {expected_model_dir}, got {actual_model_dir}"
        )
    return actual_model_dir


def detect_lid_batch_or_unknown(
    feature_extractor,
    model,
    storage_view_type,
    audios: Sequence[np.ndarray],
) -> Tuple[List[List[Tuple[str, float]]], float, float, Optional[Exception]]:
    """Run one LID batch, preserving its cardinality on every failure."""
    unknown = [[("<|unknown|>", 0.0)] for _ in audios]
    feature_started = time.perf_counter()
    try:
        batch_array = feature_extractor(audios)
    except Exception as exc:
        return unknown, time.perf_counter() - feature_started, 0.0, exc
    feature_seconds = time.perf_counter() - feature_started

    infer_started = time.perf_counter()
    try:
        results = model.detect_language(storage_view_type.from_array(batch_array))
        if len(results) != len(audios):
            raise RuntimeError(
                f"LID returned {len(results)} results for {len(audios)} inputs"
            )
    except Exception as exc:
        return (
            unknown,
            feature_seconds,
            time.perf_counter() - infer_started,
            exc,
        )
    return (
        list(results),
        feature_seconds,
        time.perf_counter() - infer_started,
        None,
    )


def annotate_lid_segments(item: dict, candidates: Sequence[Tuple[str, float]]) -> List[dict]:
    """Attach LID metadata and forward every VAD segment."""
    if candidates:
        language, probability = candidates[0]
        language = str(language).strip("<>|") or "unknown"
        probability = float(probability)
    else:
        language, probability = "unknown", 0.0
    base = {
        key: value for key, value in item.items() if key not in {"audio", "segments"}
    }
    records = []
    for segment in item["segments"]:
        records.append(
            {
                **base,
                **segment,
                "segment_audio": load_audio_seg(
                    item["audio"], segment["seg_start"], segment["seg_end"]
                ),
                "lid_lang": language,
                "lid_prob": round(probability, 4),
            }
        )
    return records


def emit_lid_records(
    item: dict,
    candidates: Sequence[Tuple[str, float]],
    collect_q: mp.Queue,
) -> int:
    records = annotate_lid_segments(item, candidates)
    for record in records:
        collect_q.put(record)
    return len(records)


def read_audio_from_tar(tar_path: str, wav_offset: int, wav_size: int) -> np.ndarray:
    from pipeline.tar_reader import _decode_audio

    with open(tar_path, "rb") as handle:
        handle.seek(wav_offset)
        payload = handle.read(wav_size)
    audio, _ = _decode_audio(payload)
    return audio


def producer(
    tar_paths: Sequence[str],
    vad_q: mp.Queue,
    stats_q: mp.Queue,
    end_signals: int = 1,
) -> None:
    from pipeline.tar_reader import iter_tar_wavs

    total_wavs = 0
    total_seconds = 0.0
    started = time.perf_counter()
    for tar_path in tar_paths:
        LOGGER.info("[Producer] reading %s", tar_path)
        for entry in iter_tar_wavs(tar_path):
            vad_q.put(
                {
                    "tar_path": entry.tar_path,
                    "wav_uuid": entry.wav_uuid,
                    "wav_offset": entry.wav_offset,
                    "wav_size": entry.wav_size,
                    "audio": entry.audio,
                    "source_duration": entry.duration,
                    "source_num_sample": entry.num_sample,
                }
            )
            total_wavs += 1
            total_seconds += entry.duration
    for _ in range(end_signals):
        vad_q.put(SENTINEL)
    stats_q.put(
        {
            "stage": "producer",
            "source_wavs": total_wavs,
            "input_audio_seconds": total_seconds,
            "seconds": time.perf_counter() - started,
        }
    )
    LOGGER.info("[Producer] done: %d wavs, %.1fs audio", total_wavs, total_seconds)


def vad_worker(
    vad_q: mp.Queue,
    lid_q: mp.Queue,
    stats_q: mp.Queue,
    min_dur: float,
    max_dur: float,
    model_dir: str,
    fireredvad_root: str,
    vad_device: str,
    worker_index: int = 0,
    torch_threads: int = 8,
) -> None:
    sys.path.insert(0, fireredvad_root)
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    import torch
    from fireredvad.vad import FireRedVad, FireRedVadConfig

    torch.set_num_threads(max(1, torch_threads))
    use_gpu = vad_device == "cuda" or (
        vad_device == "auto" and torch.cuda.is_available()
    )
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("VAD CUDA requested but CUDA is unavailable")
    config = FireRedVadConfig(
        use_gpu=use_gpu,
        speech_threshold=0.5,
        min_speech_frame=20,
        max_speech_frame=2000,
        min_silence_frame=10,
        merge_silence_frame=50,
        extend_speech_frame=5,
    )
    model = FireRedVad.from_pretrained(model_dir, config)
    LOGGER.info("[VAD] ready, device=%s", "cuda" if config.use_gpu else "cpu")

    source_wavs = 0
    emitted_segments = 0
    failed_wavs = 0
    started = time.perf_counter()
    with torch.inference_mode():
        while True:
            item = vad_q.get()
            if item is SENTINEL:
                lid_q.put(SENTINEL)
                break
            source_wavs += 1
            try:
                audio = item["audio"]
                pcm16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
                result, _ = model.detect(pcm16)
                segments = []
                for start_sec, end_sec in result.get("timestamps", []):
                    start_sec = float(start_sec)
                    end_sec = float(end_sec)
                    duration = end_sec - start_sec
                    if duration < min_dur or (max_dur > 0 and duration > max_dur):
                        continue
                    segments.append(
                        {
                            "seg_start": round(start_sec, 4),
                            "seg_end": round(end_sec, 4),
                        }
                    )
                if segments:
                    lid_q.put({**item, "segments": segments})
                    emitted_segments += len(segments)
            except Exception as exc:
                failed_wavs += 1
                LOGGER.warning("[VAD] %s failed: %s", item["wav_uuid"][:8], exc)

    stats_q.put(
        {
            "stage": "vad",
            "worker_index": worker_index,
            "source_wavs": source_wavs,
            "vad_segments": emitted_segments,
            "vad_failed_wavs": failed_wavs,
            "seconds": time.perf_counter() - started,
        }
    )
    LOGGER.info("[VAD] done: %d wavs -> %d segments", source_wavs, emitted_segments)


def _load_lid_feature_size(model_dir: str) -> int:
    config_path = os.path.join(model_dir, "preprocessor_config.json")
    if not os.path.exists(config_path):
        return 80
    with open(config_path, encoding="utf-8") as handle:
        return int(json.load(handle).get("feature_size", 80))


class BatchedWhisperFeatureExtractor:
    """Vectorized CPU log-mel extraction matching Whisper's fixed 30s input."""

    def __init__(self, feature_size: int, threads: int = 8) -> None:
        import torch
        from faster_whisper.feature_extractor import FeatureExtractor

        torch.set_num_threads(min(8, max(1, threads)))
        reference = FeatureExtractor(feature_size=feature_size)
        self.torch = torch
        self.n_samples = reference.n_samples
        self.window = torch.hann_window(reference.n_fft, periodic=True)
        self.mel_filters = torch.from_numpy(reference.mel_filters)
        self.n_fft = reference.n_fft
        self.hop_length = reference.hop_length

    def __call__(self, audios: Sequence[np.ndarray]) -> np.ndarray:
        torch = self.torch
        waveforms = np.zeros((len(audios), self.n_samples), dtype=np.float32)
        for index, audio in enumerate(audios):
            length = min(len(audio), self.n_samples)
            waveforms[index, :length] = audio[:length]
        waveform_tensor = torch.from_numpy(waveforms)
        stft = torch.stft(
            waveform_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        magnitudes = stft[:, :, :-1].abs().square()
        mel_spec = torch.matmul(self.mel_filters, magnitudes)
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        floor = log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0
        log_spec = torch.maximum(log_spec, floor)
        log_spec = (log_spec + 4.0) / 4.0
        return np.ascontiguousarray(log_spec.numpy(), dtype=np.float32)


def lid_worker_annotate_batched(
    lid_q: mp.Queue,
    collect_q: mp.Queue,
    stats_q: mp.Queue,
    model_dir: str,
    batch_size: int,
    sample_seconds: float,
    upstream_sentinels: int = 1,
) -> None:
    setup_runtime_libraries()
    import ctranslate2

    feature_size = _load_lid_feature_size(model_dir)
    feature_extractor = BatchedWhisperFeatureExtractor(feature_size=feature_size)

    try:
        model = ctranslate2.models.Whisper(
            model_dir,
            device="cuda",
            device_index=0,
            compute_type="float16",
            inter_threads=1,
            intra_threads=2,
        )
        lid_device = "cuda/float16"
    except Exception as exc:
        LOGGER.warning("[LID] CUDA init failed (%s), using CPU int8", exc)
        model = ctranslate2.models.Whisper(
            model_dir,
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=8,
        )
        lid_device = "cpu/int8"
    LOGGER.info(
        "[LID] ready: %s, feature=%d, batch=%d", lid_device, feature_size, batch_size
    )

    pending_buckets: Dict[int, List[Tuple[dict, np.ndarray]]] = {}
    source_wavs = 0
    annotated_wavs = 0
    annotated_segments = 0
    unknown_wavs = 0
    language_counts: Dict[str, int] = {}
    lid_batches = 0
    feature_seconds = 0.0
    infer_seconds = 0.0
    real_audio_seconds = 0.0
    padded_audio_seconds = 0.0
    started = time.perf_counter()

    def flush_batch(batch: List[Tuple[dict, np.ndarray]]) -> None:
        nonlocal annotated_wavs, annotated_segments, unknown_wavs, lid_batches
        nonlocal feature_seconds, infer_seconds
        nonlocal real_audio_seconds, padded_audio_seconds
        if not batch:
            return
        pending_items = [item for item, _ in batch]
        pending_audio = [audio for _, audio in batch]
        real_audio_seconds += sum(len(audio) for audio in pending_audio) / SAMPLE_RATE
        padded_audio_seconds += 30.0 * len(pending_audio)
        raw_results, feature_elapsed, infer_elapsed, error = (
            detect_lid_batch_or_unknown(
                feature_extractor,
                model,
                ctranslate2.StorageView,
                pending_audio,
            )
        )
        feature_seconds += feature_elapsed
        infer_seconds += infer_elapsed
        if error is not None:
            LOGGER.warning("[LID] batch failed: %s", error)
        lid_batches += 1

        for item, candidates in zip(pending_items, raw_results):
            records = annotate_lid_segments(item, candidates)
            language = records[0]["lid_lang"] if records else "unknown"
            language_counts[language] = language_counts.get(language, 0) + 1
            unknown_wavs += int(language == "unknown")
            annotated_wavs += 1
            for record in records:
                collect_q.put(record)
                annotated_segments += 1

    completed_upstreams = 0
    while True:
        item = lid_q.get()
        if item is SENTINEL:
            completed_upstreams += 1
            if completed_upstreams < upstream_sentinels:
                continue
            for bucket_id in sorted(pending_buckets):
                bucket = pending_buckets[bucket_id]
                for begin in range(0, len(bucket), batch_size):
                    flush_batch(bucket[begin : begin + batch_size])
            collect_q.put(SENTINEL)
            break
        source_wavs += 1
        try:
            longest = max(
                item["segments"],
                key=lambda segment: segment["seg_end"] - segment["seg_start"],
            )
            end_sec = min(longest["seg_end"], longest["seg_start"] + sample_seconds)
            audio = load_audio_seg(item["audio"], longest["seg_start"], end_sec)
            bucket_id = 0
            bucket = pending_buckets.setdefault(bucket_id, [])
            bucket.append((item, audio))
            if len(bucket) >= batch_size:
                flush_batch(bucket[:batch_size])
                del bucket[:batch_size]
        except Exception as exc:
            LOGGER.warning("[LID] %s failed: %s", item["wav_uuid"][:8], exc)
            emitted = emit_lid_records(item, [], collect_q)
            annotated_wavs += 1
            annotated_segments += emitted
            unknown_wavs += 1
            language_counts["unknown"] = language_counts.get("unknown", 0) + 1

    stats_q.put(
        {
            "stage": "lid",
            "lid_mode": "metadata",
            "lid_source_wavs": source_wavs,
            "lid_annotated_wavs": annotated_wavs,
            "annotated_segments": annotated_segments,
            "lid_unknown_wavs": unknown_wavs,
            "lid_language_counts": language_counts,
            "lid_batches": lid_batches,
            "lid_feature_seconds": feature_seconds,
            "lid_infer_seconds": infer_seconds,
            "lid_padding_efficiency": (
                real_audio_seconds / padded_audio_seconds
                if padded_audio_seconds
                else 1.0
            ),
            "seconds": time.perf_counter() - started,
        }
    )
    LOGGER.info(
        "[LID] done: annotated %d/%d wavs, %d segments, %d batches, inference %.3fs",
        annotated_wavs,
        source_wavs,
        annotated_segments,
        lid_batches,
        infer_seconds,
    )


def lid_passthrough_worker(
    lid_q: mp.Queue,
    collect_q: mp.Queue,
    stats_q: mp.Queue,
    upstream_sentinels: int = 1,
) -> None:
    """Forward all VAD segments when diagnostic Whisper LID is disabled."""
    source_wavs = 0
    annotated_segments = 0
    completed_upstreams = 0
    started = time.perf_counter()
    while completed_upstreams < upstream_sentinels:
        item = lid_q.get()
        if item is SENTINEL:
            completed_upstreams += 1
            continue
        source_wavs += 1
        records = annotate_lid_segments(item, [])
        for record in records:
            collect_q.put(record)
            annotated_segments += 1
    collect_q.put(SENTINEL)
    stats_q.put(
        {
            "stage": "lid",
            "lid_mode": "off",
            "lid_source_wavs": source_wavs,
            "lid_annotated_wavs": source_wavs,
            "annotated_segments": annotated_segments,
            "lid_unknown_wavs": source_wavs,
            "lid_language_counts": {"unknown": source_wavs},
            "lid_batches": 0,
            "lid_feature_seconds": 0.0,
            "lid_infer_seconds": 0.0,
            "lid_padding_efficiency": 1.0,
            "seconds": time.perf_counter() - started,
        }
    )
    LOGGER.info(
        "[LID] disabled: forwarded %d wavs, %d segments",
        source_wavs,
        annotated_segments,
    )


class DurationBucketBatcher:
    """Create full batches from similar durations and merge only final tails."""

    def __init__(
        self, batch_size: int, bucket_edges: Sequence[float]
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if list(bucket_edges) != sorted(bucket_edges):
            raise ValueError("bucket_edges must be sorted")
        self.batch_size = batch_size
        self.bucket_edges = list(bucket_edges)

    @staticmethod
    def duration(segment: dict) -> float:
        return float(segment["seg_end"] - segment["seg_start"])

    def batches(self, segments: Sequence[dict]) -> Iterable[List[dict]]:
        buckets: Dict[Tuple[str, int], List[dict]] = {}
        for segment in segments:
            bucket_id = bisect.bisect_left(
                self.bucket_edges, self.duration(segment)
            )
            key = (segment.get("text_lang", "unknown"), bucket_id)
            buckets.setdefault(key, []).append(segment)

        tails: Dict[str, List[dict]] = {}
        for key in sorted(buckets):
            bucket = sorted(buckets[key], key=self.duration)
            full_count = len(bucket) // self.batch_size * self.batch_size
            for begin in range(0, full_count, self.batch_size):
                yield bucket[begin : begin + self.batch_size]
            language = key[0]
            tails.setdefault(language, []).extend(bucket[full_count:])

        for language in sorted(tails):
            language_tail = sorted(tails[language], key=self.duration)
            for begin in range(0, len(language_tail), self.batch_size):
                yield language_tail[begin : begin + self.batch_size]


class StreamingDurationBuckets:
    """Emit a batch as soon as one language-duration bucket becomes full."""

    def __init__(
        self,
        batch_size: int,
        bucket_edges: Sequence[float],
        lookahead_batches: int = 1,
    ) -> None:
        self.batch_size = batch_size
        self.bucket_edges = list(bucket_edges)
        self.lookahead_batches = max(1, lookahead_batches)
        self.buckets: Dict[Tuple[str, int], List[dict]] = {}

    @staticmethod
    def duration(segment: dict) -> float:
        return float(segment["seg_end"] - segment["seg_start"])

    def add(self, segment: dict) -> Optional[List[dict]]:
        bucket_id = bisect.bisect_left(
            self.bucket_edges, self.duration(segment)
        )
        key = (segment.get("text_lang", "unknown"), bucket_id)
        bucket = self.buckets.setdefault(key, [])
        bucket.append(segment)
        if len(bucket) < self.batch_size * self.lookahead_batches:
            return None
        bucket.sort(key=self.duration)
        batch = bucket[: self.batch_size]
        del bucket[: self.batch_size]
        return batch

    def flush(self) -> Iterable[List[dict]]:
        tails = [segment for bucket in self.buckets.values() for segment in bucket]
        self.buckets.clear()
        yield from DurationBucketBatcher(
            self.batch_size, self.bucket_edges
        ).batches(tails)


class AutoLanguageStreamingDurationBuckets(StreamingDurationBuckets):
    """Duration buckets that never split batches by a LID prediction."""

    def __init__(
        self,
        batch_size: int,
        bucket_edges: Sequence[float],
        lookahead_batches: int = 1,
    ) -> None:
        super().__init__(batch_size, bucket_edges, lookahead_batches)
        self.buckets: Dict[int, List[dict]] = {}

    def add(self, segment: dict) -> Optional[List[dict]]:
        bucket_id = bisect.bisect_left(
            self.bucket_edges, self.duration(segment)
        )
        bucket = self.buckets.setdefault(bucket_id, [])
        bucket.append(segment)
        if len(bucket) < self.batch_size * self.lookahead_batches:
            return None
        bucket.sort(key=self.duration)
        batch = bucket[: self.batch_size]
        del bucket[: self.batch_size]
        return batch

    def flush(self) -> Iterable[List[dict]]:
        tails = sorted(
            (segment for bucket in self.buckets.values() for segment in bucket),
            key=self.duration,
        )
        self.buckets.clear()
        for begin in range(0, len(tails), self.batch_size):
            yield tails[begin : begin + self.batch_size]


class SourceAudioCache:
    """Small LRU used only when phase-2 resume must reconstruct segment audio."""

    def __init__(self, max_items: int) -> None:
        self.max_items = max(1, max_items)
        self.cache: OrderedDict[Tuple[str, int, int], np.ndarray] = OrderedDict()

    def segment(self, item: dict) -> np.ndarray:
        if isinstance(item.get("segment_audio"), np.ndarray):
            return item["segment_audio"]
        key = (item["tar_path"], item["wav_offset"], item["wav_size"])
        audio = self.cache.pop(key, None)
        if audio is None:
            audio = read_audio_from_tar(*key)
        self.cache[key] = audio
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return load_audio_seg(audio, item["seg_start"], item["seg_end"])


def _timestamp_records(result: dict) -> List[list]:
    timestamps = []
    for timestamp in result.get("timestamps", []):
        token = str(timestamp.get("token", "")).strip()
        if token:
            timestamps.append(
                [
                    token,
                    round(float(timestamp["start_time"]), 3),
                    round(float(timestamp["end_time"]), 3),
                ]
            )
    return timestamps


def annotate_transcription_record(
    segment: dict,
    result: dict,
    segment_uuid: str,
    kept_wav: Optional[str],
) -> dict:
    text = result.get("text", "").strip()
    evidence = classify_transcript_language(text)
    duration = round(segment["seg_end"] - segment["seg_start"], 4)
    return {
        "tar_path": segment["tar_path"],
        "wav": kept_wav,
        "wav_uuid": segment_uuid,
        "source_wav_uuid": segment["wav_uuid"],
        "wav_offset": segment["wav_offset"],
        "wav_size": segment["wav_size"],
        "seg_start": segment["seg_start"],
        "seg_end": segment["seg_end"],
        "duration": duration,
        "num_sample": int(duration * SAMPLE_RATE),
        "lid_lang": segment.get("lid_lang", "unknown"),
        "lid_prob": segment.get("lid_prob", 0.0),
        "text_lang": evidence["language"],
        "text_lang_evidence": {
            "zh_chars": evidence["zh_chars"],
            "en_tokens": evidence["en_tokens"],
        },
        "transcribe": {"funasr-nano": text},
        "timestamp": {"funasr-nano": _timestamp_records(result)},
    }


def resolve_all_out_json(out_json: str, all_out_json: Optional[str]) -> str:
    if all_out_json:
        if os.path.abspath(all_out_json) == os.path.abspath(out_json):
            raise ValueError("--all_out_json must differ from --out_json")
        return all_out_json
    output = Path(out_json)
    return str(output.with_name(f"{output.stem}.all{output.suffix or '.json'}"))


def write_post_filtered_output(
    all_out_json: str,
    out_json: str,
    keep_langs: Sequence[str],
) -> Dict[str, Any]:
    with open(all_out_json, encoding="utf-8") as handle:
        all_records = json.load(handle)
    kept_records = post_filter_records(all_records, keep_langs)
    atomic_write_json(out_json, kept_records)

    language_counts: Dict[str, int] = {}
    for record in all_records:
        language = record.get("text_lang", "other")
        language_counts[language] = language_counts.get(language, 0) + 1
    return {
        "all_asr_segments": len(all_records),
        "post_kept_segments": len(kept_records),
        "post_keep_rate": len(kept_records) / len(all_records) if all_records else 0.0,
        "post_kept_audio_seconds": sum(
            float(record.get("duration", 0.0)) for record in kept_records
        ),
        "transcript_language_counts": language_counts,
        "post_keep_langs": list(keep_langs),
    }


def _funasr_bucket_phase2_offline(
    segments: Sequence[dict],
    model_dir: str,
    batch_size: int,
    bucket_edges: Sequence[float],
    out_json: str,
    phase2_ckpt_path: Optional[str],
    audio_dir: Optional[str],
    resume_audio_cache: int,
    return_timestamps: bool,
) -> Dict[str, Any]:
    import soundfile as sf
    import torch

    from asr.funasr_nano_batch import FunASRNanoBatch
    from pipeline.tar_reader import make_seg_uuid

    done_uuids = set()
    existing_results = []
    if phase2_ckpt_path and os.path.exists(phase2_ckpt_path):
        with open(phase2_ckpt_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    existing_results.append(record)
                    done_uuids.add(record["wav_uuid"])
                except Exception:
                    continue

    remaining = [
        segment
        for segment in segments
        if make_seg_uuid(segment["wav_uuid"], segment["seg_start"])
        not in done_uuids
    ]
    LOGGER.info(
        "[ASR] %d segments remaining, %d resumed", len(remaining), len(done_uuids)
    )

    load_started = time.perf_counter()
    engine = FunASRNanoBatch(
        model_dir=model_dir,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype="fp32",
    )
    model_load_seconds = time.perf_counter() - load_started
    LOGGER.info("[ASR] model loaded in %.2fs", model_load_seconds)

    if audio_dir:
        Path(audio_dir).mkdir(parents=True, exist_ok=True)
    ckpt_handle = None
    if phase2_ckpt_path:
        Path(phase2_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        ckpt_handle = open(phase2_ckpt_path, "a", encoding="utf-8")

    loader = SourceAudioCache(resume_audio_cache)
    batcher = DurationBucketBatcher(batch_size, bucket_edges)
    batches = list(batcher.batches(remaining))
    new_results: List[dict] = []
    infer_started = time.perf_counter()
    batch_infer_seconds = 0.0
    total_real_audio_seconds = 0.0
    total_padded_audio_seconds = 0.0
    failed_segments = 0

    def process_batch(batch: List[dict]) -> List[dict]:
        nonlocal batch_infer_seconds, total_real_audio_seconds
        nonlocal total_padded_audio_seconds, failed_segments
        if not batch:
            return []
        valid: List[dict] = []
        audios: List[np.ndarray] = []
        for segment in batch:
            try:
                audio = loader.segment(segment)
                if audio.size == 0:
                    raise ValueError("empty segment")
                audios.append(audio)
                valid.append(segment)
            except Exception as exc:
                failed_segments += 1
                LOGGER.warning("[ASR] %s load failed: %s", segment["wav_uuid"][:8], exc)
        if not valid:
            return []

        keys = [make_seg_uuid(item["wav_uuid"], item["seg_start"]) for item in valid]
        batch_started = time.perf_counter()
        try:
            results = engine.transcribe_batch(
                audios,
                keys=keys,
                language=None,
                return_timestamps=return_timestamps,
            )
            validate_batch_result_count(results, len(valid))
            batch_infer_seconds += time.perf_counter() - batch_started
        except RuntimeError as exc:
            if len(valid) > 1:
                LOGGER.warning(
                    "[ASR] batch of %d failed (%s); retrying as halves", len(valid), exc
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                midpoint = len(valid) // 2
                return process_batch(valid[:midpoint]) + process_batch(valid[midpoint:])
            failed_segments += 1
            LOGGER.warning("[ASR] single segment failed: %s", exc)
            return []

        durations = [len(audio) / SAMPLE_RATE for audio in audios]
        total_real_audio_seconds += sum(durations)
        total_padded_audio_seconds += max(durations) * len(durations)

        records = []
        for segment, audio, result, segment_uuid in zip(valid, audios, results, keys):
            kept_wav = None
            if audio_dir:
                kept_wav = os.path.join(
                    audio_dir,
                    f"{segment['wav_uuid'][:16]}_{segment['seg_start']:.3f}.wav",
                )
                sf.write(kept_wav, audio, SAMPLE_RATE)
            records.append(
                annotate_transcription_record(
                    segment=segment,
                    result=result,
                    segment_uuid=segment_uuid,
                    kept_wav=kept_wav,
                )
            )
        return records

    try:
        for batch_index, batch in enumerate(batches, start=1):
            records = process_batch(batch)
            if ckpt_handle:
                for record in records:
                    ckpt_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                ckpt_handle.flush()
            new_results.extend(records)
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(batches):
                LOGGER.info(
                    "[ASR] batch %d/%d, records=%d", batch_index, len(batches), len(new_results)
                )
    finally:
        if ckpt_handle:
            ckpt_handle.close()

    inference_wall_seconds = time.perf_counter() - infer_started
    all_results = existing_results + new_results
    atomic_write_json(out_json, all_results)

    if phase2_ckpt_path and os.path.exists(phase2_ckpt_path):
        os.remove(phase2_ckpt_path)

    padding_efficiency = (
        total_real_audio_seconds / total_padded_audio_seconds
        if total_padded_audio_seconds
        else 1.0
    )
    LOGGER.info(
        "[ASR] done: %d records, %.1f%% padding efficiency, model %.2fs, inference %.2fs",
        len(all_results),
        padding_efficiency * 100,
        model_load_seconds,
        inference_wall_seconds,
    )
    return {
        "asr_model_load_seconds": model_load_seconds,
        "asr_inference_wall_seconds": inference_wall_seconds,
        "asr_batch_compute_seconds": batch_infer_seconds,
        "asr_batches": len(batches),
        "asr_segments": len(new_results),
        "asr_failed_segments": failed_segments,
        "asr_audio_seconds": total_real_audio_seconds,
        "padding_efficiency": padding_efficiency,
    }


def funasr_bucket_worker(
    segment_q: mp.Queue,
    stats_q: mp.Queue,
    model_dir: str,
    batch_size: int,
    bucket_edges: Sequence[float],
    out_json: str,
    checkpoint_path: str,
    audio_dir: Optional[str],
    return_timestamps: bool,
    asr_dtype: str,
    resume: bool,
    prefetch_batches: int,
    bucket_lookahead_batches: int,
    asr_server_socket: Optional[str],
) -> None:
    """Load Nano concurrently, then consume and infer full streaming buckets."""
    import soundfile as sf

    from pipeline.tar_reader import make_seg_uuid

    worker_started = time.perf_counter()
    existing_results = (
        load_resume_records([out_json, checkpoint_path]) if resume else []
    )
    done_uuids = {
        record["wav_uuid"] for record in existing_results if record.get("wav_uuid")
    }
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_mode = "a" if resume else "w"
    checkpoint_handle = open(checkpoint_path, checkpoint_mode, encoding="utf-8")
    if audio_dir:
        Path(audio_dir).mkdir(parents=True, exist_ok=True)

    engine_holder: Dict[str, Any] = {}
    engine_ready = threading.Event()

    def load_engine() -> None:
        load_started = time.perf_counter()
        try:
            if asr_server_socket:
                from multiprocessing.connection import Client

                deadline = time.monotonic() + 30
                while True:
                    try:
                        client = Client(asr_server_socket, family="AF_UNIX")
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.1)
                client.send({"command": "ping"})
                response = client.recv()
                if not response.get("ok"):
                    raise RuntimeError(f"Nano server ping failed: {response}")
                engine_holder["asr_dtype"] = validate_server_dtype(
                    response, asr_dtype
                )
                validate_server_model(response, model_dir)
                engine_holder["client"] = client
                engine_holder["backend"] = "server"
            else:
                import torch
                from asr.funasr_nano_batch import FunASRNanoBatch

                engine_holder["torch"] = torch
                engine_holder["engine"] = FunASRNanoBatch(
                    model_dir=model_dir,
                    device="cuda:0" if torch.cuda.is_available() else "cpu",
                    dtype=asr_dtype,
                )
                engine_holder["backend"] = "local"
                engine_holder["asr_dtype"] = asr_dtype
        except BaseException as exc:
            engine_holder["error"] = exc
        finally:
            engine_holder["load_seconds"] = time.perf_counter() - load_started
            engine_ready.set()

    load_thread = threading.Thread(target=load_engine, name="NanoLoader", daemon=True)
    load_thread.start()

    buckets = AutoLanguageStreamingDurationBuckets(
        batch_size, bucket_edges, lookahead_batches=bucket_lookahead_batches
    )
    ready_batches = deque()
    new_results: List[dict] = []
    batch_compute_seconds = 0.0
    total_real_audio_seconds = 0.0
    total_padded_audio_seconds = 0.0
    failed_segments = 0
    successful_batches = 0
    prepare_seconds = 0.0
    encode_seconds = 0.0
    generate_seconds = 0.0
    ctc_seconds = 0.0
    peak_allocated_gib = 0.0
    peak_reserved_gib = 0.0
    received_segments = 0
    skipped_segments = 0
    first_batch_started_at: Optional[float] = None

    def process_batch(batch: List[dict]) -> List[dict]:
        nonlocal batch_compute_seconds, total_real_audio_seconds
        nonlocal total_padded_audio_seconds, failed_segments
        nonlocal successful_batches, first_batch_started_at
        nonlocal prepare_seconds, encode_seconds, generate_seconds, ctc_seconds
        nonlocal peak_allocated_gib, peak_reserved_gib
        if not batch:
            return []
        valid: List[dict] = []
        audios: List[np.ndarray] = []
        for segment in batch:
            audio = segment.get("segment_audio")
            if not isinstance(audio, np.ndarray) or audio.size == 0:
                failed_segments += 1
                LOGGER.warning("[ASR] %s has no segment audio", segment["wav_uuid"][:8])
                continue
            valid.append(segment)
            audios.append(audio)
        if not valid:
            return []

        keys = [make_seg_uuid(item["wav_uuid"], item["seg_start"]) for item in valid]
        if first_batch_started_at is None:
            first_batch_started_at = time.perf_counter()
        batch_started = time.perf_counter()
        try:
            if engine_holder["backend"] == "server":
                client = engine_holder["client"]
                client.send(
                    {
                        "command": "transcribe",
                        "audios": audios,
                        "keys": keys,
                        "language": None,
                        "return_timestamps": return_timestamps,
                    }
                )
                response = client.recv()
                if not response.get("ok"):
                    raise RuntimeError(response.get("error", "Nano server failed"))
                results = response["results"]
                batch_stats = response.get("stats", {})
            else:
                engine = engine_holder["engine"]
                results = engine.transcribe_batch(
                    audios,
                    keys=keys,
                    language=None,
                    return_timestamps=return_timestamps,
                )
                batch_stats = engine.last_stats
            validate_batch_result_count(results, len(valid))
            batch_compute_seconds += time.perf_counter() - batch_started
            prepare_seconds += batch_stats.get("prepare_seconds", 0.0)
            encode_seconds += batch_stats.get("encode_seconds", 0.0)
            generate_seconds += batch_stats.get("generate_seconds", 0.0)
            ctc_seconds += batch_stats.get("ctc_seconds", 0.0)
            peak_allocated_gib = max(
                peak_allocated_gib, batch_stats.get("peak_allocated_gib", 0.0)
            )
            peak_reserved_gib = max(
                peak_reserved_gib, batch_stats.get("peak_reserved_gib", 0.0)
            )
        except RuntimeError as exc:
            batch_compute_seconds += time.perf_counter() - batch_started
            if len(valid) > 1:
                LOGGER.warning(
                    "[ASR] batch of %d failed (%s); retrying as halves",
                    len(valid),
                    exc,
                )
                torch_module = engine_holder.get("torch")
                if torch_module is not None and torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
                midpoint = len(valid) // 2
                return process_batch(valid[:midpoint]) + process_batch(valid[midpoint:])
            failed_segments += 1
            LOGGER.warning("[ASR] single segment failed: %s", exc)
            return []

        successful_batches += 1
        durations = [len(audio) / SAMPLE_RATE for audio in audios]
        total_real_audio_seconds += sum(durations)
        total_padded_audio_seconds += max(durations) * len(durations)
        records = []
        for segment, audio, result, segment_uuid in zip(valid, audios, results, keys):
            kept_wav = None
            if audio_dir:
                kept_wav = os.path.join(
                    audio_dir,
                    f"{segment['wav_uuid'][:16]}_{segment['seg_start']:.3f}.wav",
                )
                sf.write(kept_wav, audio, SAMPLE_RATE)
            records.append(
                annotate_transcription_record(
                    segment=segment,
                    result=result,
                    segment_uuid=segment_uuid,
                    kept_wav=kept_wav,
                )
            )
        return records

    def persist(records: Sequence[dict]) -> None:
        if not records:
            return
        for record in records:
            checkpoint_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint_handle.flush()
        new_results.extend(records)

    saw_sentinel = False
    flushed_tails = False
    try:
        while not saw_sentinel or ready_batches:
            if engine_ready.is_set() and "error" in engine_holder:
                raise RuntimeError("Fun-ASR-Nano model load failed") from engine_holder["error"]

            if engine_ready.is_set() and ready_batches:
                persist(process_batch(ready_batches.popleft()))
                continue

            can_prefetch = len(ready_batches) < max(1, prefetch_batches)
            if not saw_sentinel and can_prefetch:
                try:
                    segment = segment_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if segment is SENTINEL:
                    saw_sentinel = True
                else:
                    received_segments += 1
                    segment_uuid = make_seg_uuid(
                        segment["wav_uuid"], segment["seg_start"]
                    )
                    if segment_uuid in done_uuids:
                        skipped_segments += 1
                    else:
                        batch = buckets.add(segment)
                        if batch:
                            ready_batches.append(batch)

            if saw_sentinel and not flushed_tails:
                ready_batches.extend(buckets.flush())
                flushed_tails = True

            if not engine_ready.is_set() and (
                saw_sentinel or len(ready_batches) >= max(1, prefetch_batches)
            ):
                engine_ready.wait(timeout=0.1)

        load_thread.join()
    finally:
        checkpoint_handle.close()
        client = engine_holder.get("client")
        if client is not None:
            client.close()

    all_results = existing_results + new_results
    atomic_write_json(out_json, all_results)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    padding_efficiency = (
        total_real_audio_seconds / total_padded_audio_seconds
        if total_padded_audio_seconds
        else 1.0
    )
    torch_module = engine_holder.get("torch")
    if torch_module is not None and torch_module.cuda.is_available():
        peak_allocated_gib = max(
            peak_allocated_gib,
            torch_module.cuda.max_memory_allocated() / (1024 ** 3),
        )
        peak_reserved_gib = max(
            peak_reserved_gib,
            torch_module.cuda.max_memory_reserved() / (1024 ** 3),
        )
    resumed_audio_seconds = sum(
        float(record.get("duration", 0.0)) for record in existing_results
    )
    total_audio_seconds = resumed_audio_seconds + total_real_audio_seconds
    stats = {
        "stage": "asr",
        "asr_backend": engine_holder.get("backend", "unknown"),
        "asr_dtype": engine_holder.get("asr_dtype", "unknown"),
        "asr_model_load_seconds": engine_holder.get("load_seconds", 0.0),
        "asr_stream_seconds": time.perf_counter() - worker_started,
        "asr_first_batch_delay_seconds": (
            first_batch_started_at - worker_started
            if first_batch_started_at is not None
            else None
        ),
        "asr_batch_compute_seconds": batch_compute_seconds,
        "asr_prepare_seconds": prepare_seconds,
        "asr_encode_seconds": encode_seconds,
        "asr_generate_seconds": generate_seconds,
        "asr_ctc_seconds": ctc_seconds,
        "asr_batches": successful_batches,
        "asr_received_segments": received_segments,
        "asr_skipped_segments": skipped_segments,
        "asr_segments": len(all_results),
        "asr_new_segments": len(new_results),
        "asr_resumed_segments": len(existing_results),
        "asr_failed_segments": failed_segments,
        "asr_audio_seconds": total_audio_seconds,
        "asr_new_audio_seconds": total_real_audio_seconds,
        "asr_resumed_audio_seconds": resumed_audio_seconds,
        "padding_efficiency": padding_efficiency,
        "asr_peak_allocated_gib": peak_allocated_gib,
        "asr_peak_reserved_gib": peak_reserved_gib,
    }
    stats_q.put(stats)
    LOGGER.info(
        "[ASR] done: %d records, %.1f%% padding efficiency, load %.2fs, compute %.2fs",
        len(all_results),
        padding_efficiency * 100,
        stats["asr_model_load_seconds"],
        batch_compute_seconds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_paths", nargs="+", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--metrics_json", default=None)
    parser.add_argument("--funasr_model_dir", required=True)
    parser.add_argument("--lid_model_dir", default=None)
    parser.add_argument("--fireredvad_model", required=True)
    parser.add_argument("--fireredvad_root", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lid_batch_size", type=int, default=32)
    parser.add_argument("--lid_sample_seconds", type=float, default=12.0)
    parser.add_argument("--lid_mode", choices=["metadata", "off"], default="metadata")
    parser.add_argument("--bucket_edges", nargs="+", type=float,
                        default=[2, 4, 6, 8, 12, 16, 24, 30])
    parser.add_argument("--min_dur", type=float, default=1.0)
    parser.add_argument("--max_dur", type=float, default=30.0)
    parser.add_argument("--post_keep_langs", nargs="+", default=["zh", "zh-en"])
    parser.add_argument("--all_out_json", default=None)
    parser.add_argument("--asr_dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--audio_dir", default=None)
    parser.add_argument("--queue_maxsize", type=int, default=32)
    parser.add_argument("--asr_prefetch_batches", type=int, default=64)
    parser.add_argument("--bucket_lookahead_batches", type=int, default=4)
    parser.add_argument("--vad_device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--vad_workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--vad_threads", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--asr_server_socket", default=None)
    parser.add_argument("--resume_audio_cache", type=int, default=16)
    parser.add_argument("--checkpoint_flush_interval", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_timestamps", action="store_true")
    return parser.parse_args()


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.lid_mode == "metadata" and not args.lid_model_dir:
        raise ValueError("--lid_model_dir is required when --lid_mode=metadata")


def _collect_stage_stats(stats_q: mp.Queue, expected: int = 4) -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    for _ in range(expected):
        try:
            item = stats_q.get(timeout=5)
        except queue.Empty:
            break
        stage = item["stage"]
        if stage == "vad" and stage in stats:
            combined = stats[stage]
            combined["workers"] += 1
            combined["source_wavs"] += item.get("source_wavs", 0)
            combined["vad_segments"] += item.get("vad_segments", 0)
            combined["vad_failed_wavs"] = combined.get(
                "vad_failed_wavs", 0
            ) + item.get("vad_failed_wavs", 0)
            combined["seconds"] = max(
                combined.get("seconds", 0.0), item.get("seconds", 0.0)
            )
        else:
            stats[stage] = dict(item)
            if stage == "vad":
                stats[stage]["workers"] = 1
    return stats


def _main_offline() -> None:
    args = parse_args()
    validate_runtime_args(args)
    out_dir = str(Path(args.out_json).parent)
    phase1_ckpt = os.path.join(out_dir, "phase1_ckpt.jsonl")
    phase2_ckpt = os.path.join(out_dir, "phase2_ckpt.jsonl")
    metrics_path = args.metrics_json or os.path.join(out_dir, "metrics.json")
    pipeline_started = time.perf_counter()
    phase1_started = pipeline_started
    segments: List[dict] = []
    stage_stats: Dict[str, dict] = {}

    if args.resume and os.path.exists(phase1_ckpt):
        LOGGER.info("[Resume] loading %s", phase1_ckpt)
        with open(phase1_ckpt, encoding="utf-8") as handle:
            for line in handle:
                try:
                    segment = json.loads(line)
                    segment["segment_audio"] = None
                    segments.append(segment)
                except Exception:
                    continue
        LOGGER.info("[Resume] loaded %d phase-1 segments", len(segments))
    else:
        context = mp.get_context("spawn")
        vad_q = context.Queue(maxsize=args.queue_maxsize)
        lid_q = context.Queue(maxsize=args.queue_maxsize)
        collect_q = context.Queue(maxsize=args.queue_maxsize)
        stats_q = context.Queue()
        processes = [
            context.Process(
                target=producer,
                args=(args.tar_paths, vad_q, stats_q),
                name="Producer",
            ),
            context.Process(
                target=vad_worker,
                args=(
                    vad_q,
                    lid_q,
                    stats_q,
                    args.min_dur,
                    args.max_dur,
                    args.fireredvad_model,
                    args.fireredvad_root,
                    args.vad_device,
                ),
                name="VAD",
            ),
            context.Process(
                target=lid_worker_annotate_batched,
                args=(
                    lid_q,
                    collect_q,
                    stats_q,
                    args.lid_model_dir,
                    args.lid_batch_size,
                    args.lid_sample_seconds,
                ),
                name="LID",
            ),
        ]
        LOGGER.info("=== Phase 1: in-memory VAD + batched tiny LID ===")
        for process in processes:
            process.start()

        Path(phase1_ckpt).parent.mkdir(parents=True, exist_ok=True)
        pending_flush = 0
        with open(phase1_ckpt, "w", encoding="utf-8") as handle:
            while True:
                item = collect_q.get()
                if item is SENTINEL:
                    break
                segments.append(item)
                metadata = {
                    key: value for key, value in item.items() if key != "segment_audio"
                }
                handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                pending_flush += 1
                if pending_flush >= args.checkpoint_flush_interval:
                    handle.flush()
                    pending_flush = 0

        for process in processes:
            process.join(timeout=60)
            if process.is_alive():
                LOGGER.warning("Terminating stuck process %s", process.name)
                process.terminate()
                process.join(timeout=5)
            if process.exitcode not in (0, None):
                raise RuntimeError(f"{process.name} exited with code {process.exitcode}")
        stage_stats = _collect_stage_stats(stats_q)

    phase1_seconds = time.perf_counter() - phase1_started
    accepted_audio_seconds = sum(
        item["seg_end"] - item["seg_start"] for item in segments
    )
    LOGGER.info(
        "Phase 1 done: %d segments, %.1fs retained audio, %.2fs wall",
        len(segments),
        accepted_audio_seconds,
        phase1_seconds,
    )

    LOGGER.info("=== Phase 2: duration-bucketed Fun-ASR-Nano batch ===")
    phase2_stats = _funasr_bucket_phase2_offline(
        segments=segments,
        model_dir=args.funasr_model_dir,
        batch_size=args.batch_size,
        bucket_edges=args.bucket_edges,
        out_json=args.out_json,
        phase2_ckpt_path=phase2_ckpt,
        audio_dir=args.audio_dir,
        resume_audio_cache=args.resume_audio_cache,
        return_timestamps=not args.no_timestamps,
    )

    total_seconds = time.perf_counter() - pipeline_started
    input_audio_seconds = stage_stats.get("producer", {}).get(
        "input_audio_seconds", 0.0
    )
    metrics: Dict[str, Any] = {
        "input_audio_seconds": input_audio_seconds,
        "accepted_audio_seconds": accepted_audio_seconds,
        "phase1_seconds": phase1_seconds,
        "total_seconds": total_seconds,
        "end_to_end_x": input_audio_seconds / total_seconds if input_audio_seconds else None,
        "accepted_audio_x": (
            accepted_audio_seconds / total_seconds if accepted_audio_seconds else None
        ),
        "stage_stats": stage_stats,
        **phase2_stats,
    }
    atomic_write_json(metrics_path, metrics)
    LOGGER.info(
        "Complete: %.2fs total, end-to-end %.2fX, output=%s, metrics=%s",
        total_seconds,
        metrics["end_to_end_x"] or 0.0,
        args.out_json,
        metrics_path,
    )


def main() -> None:
    args = parse_args()
    validate_runtime_args(args)
    out_dir = str(Path(args.out_json).parent)
    all_out_json = resolve_all_out_json(args.out_json, args.all_out_json)
    checkpoint_path = os.path.join(out_dir, "nano_auto_phase2_ckpt.jsonl")
    metrics_path = args.metrics_json or os.path.join(out_dir, "metrics.json")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")
    vad_q = context.Queue(maxsize=args.queue_maxsize)
    lid_q = context.Queue(maxsize=args.queue_maxsize)
    asr_q = context.Queue(maxsize=args.queue_maxsize)
    stats_q = context.Queue()
    if args.lid_mode == "metadata":
        lid_process = context.Process(
            target=lid_worker_annotate_batched,
            args=(
                lid_q,
                asr_q,
                stats_q,
                args.lid_model_dir,
                args.lid_batch_size,
                args.lid_sample_seconds,
                args.vad_workers,
            ),
            name="BatchLID",
        )
    else:
        lid_process = context.Process(
            target=lid_passthrough_worker,
            args=(lid_q, asr_q, stats_q, args.vad_workers),
            name="LIDPassthrough",
        )
    processes = [
        context.Process(
            target=funasr_bucket_worker,
            args=(
                asr_q,
                stats_q,
                args.funasr_model_dir,
                args.batch_size,
                args.bucket_edges,
                all_out_json,
                checkpoint_path,
                args.audio_dir,
                not args.no_timestamps,
                args.asr_dtype,
                args.resume,
                args.asr_prefetch_batches,
                args.bucket_lookahead_batches,
                args.asr_server_socket,
            ),
            name="BucketASR",
        ),
        lid_process,
    ]
    processes.extend(
        context.Process(
            target=vad_worker,
            args=(
                vad_q,
                lid_q,
                stats_q,
                args.min_dur,
                args.max_dur,
                args.fireredvad_model,
                args.fireredvad_root,
                args.vad_device,
                worker_index,
                args.vad_threads,
            ),
            name=f"FireRedVAD-{worker_index}",
        )
        for worker_index in range(args.vad_workers)
    )
    processes.append(
        context.Process(
            target=producer,
            args=(args.tar_paths, vad_q, stats_q, args.vad_workers),
            name="Producer",
        )
    )

    pipeline_started = time.perf_counter()
    LOGGER.info(
        "=== Streaming pipeline: VAD -> LID metadata -> Nano auto language -> post filter ==="
    )
    try:
        for process in processes:
            process.start()
        while True:
            failed = [
                process
                for process in processes
                if process.exitcode is not None and process.exitcode != 0
            ]
            if failed:
                details = ", ".join(
                    f"{process.name}={process.exitcode}" for process in failed
                )
                raise RuntimeError(f"Pipeline process failed: {details}")
            if all(process.exitcode is not None for process in processes):
                break
            time.sleep(0.25)
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in processes:
            process.join(timeout=10)

    total_seconds = time.perf_counter() - pipeline_started
    stage_stats = _collect_stage_stats(stats_q, expected=len(processes))
    missing = {"producer", "vad", "lid", "asr"} - set(stage_stats)
    if missing:
        raise RuntimeError(f"Missing stage metrics: {sorted(missing)}")

    input_audio_seconds = stage_stats["producer"]["input_audio_seconds"]
    asr_stats = stage_stats["asr"]
    accepted_audio_seconds = asr_stats["asr_audio_seconds"]
    asr_compute_seconds = asr_stats["asr_batch_compute_seconds"]
    post_filter_stats = write_post_filtered_output(
        all_out_json, args.out_json, args.post_keep_langs
    )
    metrics: Dict[str, Any] = {
        "language_policy": "nano_auto_then_post_filter",
        "nano_language_prompt": None,
        "all_out_json": all_out_json,
        "input_audio_seconds": input_audio_seconds,
        "accepted_audio_seconds": accepted_audio_seconds,
        "total_seconds": total_seconds,
        "end_to_end_x": (
            input_audio_seconds / total_seconds if input_audio_seconds else None
        ),
        "accepted_audio_x": (
            accepted_audio_seconds / total_seconds if accepted_audio_seconds else None
        ),
        "asr_compute_x": (
            accepted_audio_seconds / asr_compute_seconds
            if asr_compute_seconds
            else None
        ),
        "input_to_asr_compute_x": (
            input_audio_seconds / asr_compute_seconds
            if input_audio_seconds and asr_compute_seconds
            else None
        ),
        "stage_stats": stage_stats,
        **post_filter_stats,
        **{key: value for key, value in asr_stats.items() if key != "stage"},
    }
    atomic_write_json(metrics_path, metrics)
    LOGGER.info(
        "Complete: %.2fs total, end-to-end %.2fX, ASR compute %.2fX, output=%s",
        total_seconds,
        metrics["end_to_end_x"] or 0.0,
        metrics["asr_compute_x"] or 0.0,
        args.out_json,
    )


if __name__ == "__main__":
    main()
