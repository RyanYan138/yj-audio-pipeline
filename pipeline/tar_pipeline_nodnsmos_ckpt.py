#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tar包音频 pipeline — 支持客户格式输出 + 断点续跑（无DNSMOS，106集群版）
输入: 一个或多个本地 .tar 包
输出: labels.json

流水线: Producer → VAD → LID → FunASR vLLM

断点续跑说明:
  Phase1 ckpt: {out_json_dir}/phase1_ckpt.jsonl  (每条segment落盘, 不含audio)
  Phase2 ckpt: {out_json_dir}/phase2_ckpt.jsonl  (每batch结果追加落盘)
  加 --resume 参数后:
    - 若 phase1_ckpt.jsonl 存在 → 跳过Phase1直接加载
    - 若 phase2_ckpt.jsonl 存在 → 跳过已完成segment

用法:
  python3 tar_pipeline_nodnsmos_ckpt.py \
    --tar_paths /data/shards_000.tar \
    --out_json  /data/labels.json \
    --funasr_model_dir /path/to/Fun-ASR-Nano-2512 \
    --lid_model_dir    /path/to/faster-whisper-large-v3 \
    --fireredvad_model /path/to/FireRedVAD/VAD \
    --fireredvad_root  /path/to/FireRedVAD \
    --gpu 0 --resume
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SENTINEL = None


def load_audio_seg(audio: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
    s = int(start_sec * SAMPLE_RATE)
    e = int(end_sec * SAMPLE_RATE)
    return audio[s:e]


def read_audio_from_tar(tar_path: str, wav_offset: int, wav_size: int) -> np.ndarray:
    """从tar文件按offset重新读取音频，用于Phase2 resume时重建audio数组"""
    from tar_reader import _decode_audio
    with open(tar_path, "rb") as f:
        f.seek(wav_offset)
        data = f.read(wav_size)
    audio, _ = _decode_audio(data)
    return audio


def producer(tar_paths: List[str], vad_q: mp.Queue):
    from tar_reader import iter_tar_wavs
    total = 0
    for tar_path in tar_paths:
        logger.info(f"[Producer] reading {tar_path}")
        for entry in iter_tar_wavs(tar_path):
            vad_q.put({
                "tar_path":   entry.tar_path,
                "wav_uuid":   entry.wav_uuid,
                "wav_offset": entry.wav_offset,
                "wav_size":   entry.wav_size,
                "audio":      entry.audio,
                "duration":   entry.duration,
                "num_sample": entry.num_sample,
            })
            total += 1
    vad_q.put(SENTINEL)
    logger.info(f"[Producer] done, {total} wavs")


def vad_worker(vad_q: mp.Queue, lid_q: mp.Queue,
               min_dur: float, max_dur: float,
               model_dir: str, fireredvad_root: str):
    sys.path.insert(0, fireredvad_root)
    from fireredvad.vad import FireRedVad, FireRedVadConfig
    import tempfile, soundfile as sf

    cfg = FireRedVadConfig(use_gpu=True, speech_threshold=0.5,
                           min_speech_frame=20, max_speech_frame=2000,
                           min_silence_frame=10, merge_silence_frame=50,
                           extend_speech_frame=5)
    model = FireRedVad.from_pretrained(model_dir, cfg)
    logger.info("[VAD] ready")

    while True:
        item = vad_q.get()
        if item is SENTINEL:
            lid_q.put(SENTINEL)
            break
        audio = item["audio"]
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, SAMPLE_RATE)
                tmp_path = tmp.name
            result, _ = model.detect(tmp_path)
            os.unlink(tmp_path)

            for ts in result.get("timestamps", []):
                s, e = float(ts[0]), float(ts[1])
                dur = e - s
                if dur < min_dur or (max_dur > 0 and dur > max_dur):
                    continue
                lid_q.put({**item, "seg_start": round(s, 4), "seg_end": round(e, 4)})
        except Exception as ex:
            logger.warning(f"[VAD] {item['wav_uuid'][:8]} failed: {ex}")


def lid_worker_collect(lid_q: mp.Queue, out_q: mp.Queue,
                       model_dir: str,
                       target_langs: List[str], min_lang_prob: float):
    from faster_whisper import WhisperModel
    for cudnn_path in [
        "/opt/conda/envs/funasr_vllm/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "/opt/conda/envs/funasr_vllm/lib/python3.12/site-packages/ctranslate2.libs",
    ]:
        if os.path.exists(cudnn_path):
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{cudnn_path}:{existing}" if existing else cudnn_path
    try:
        model = WhisperModel(model_dir, device="cuda", compute_type="float16")
        logger.info("[LID] ready (cuda/float16)")
    except Exception as e:
        logger.warning(f"[LID] cuda failed ({e}), fallback cpu")
        model = WhisperModel(model_dir, device="cpu", compute_type="int8")
        logger.info("[LID] ready (cpu/int8)")
    lang_cache: Dict[str, Tuple[str, float]] = {}

    while True:
        item = lid_q.get()
        if item is SENTINEL:
            out_q.put(SENTINEL)
            break
        uuid = item["wav_uuid"]
        try:
            if uuid not in lang_cache:
                audio = load_audio_seg(item["audio"], item["seg_start"],
                                       min(item["seg_end"], item["seg_start"] + 12.0))
                _, info = model.transcribe(audio, task="transcribe")
                lang_cache[uuid] = (info.language, info.language_probability)
            lang, prob = lang_cache[uuid]
            if target_langs and lang not in target_langs:
                continue
            if prob < min_lang_prob:
                continue
            out_q.put({**item, "text_lang": lang, "lang_prob": round(prob, 4)})
        except Exception as ex:
            logger.warning(f"[LID] {uuid[:8]} failed: {ex}")


def funasr_vllm_phase2(segments: list, model_dir: str, gpu_id: int,
                       batch_size: int, out_json: str, audio_dir=None,
                       gpu_memory_utilization: float = 0.9,
                       phase2_ckpt_path: str = None):
    from tar_reader import make_seg_uuid

    cusparselt = "/opt/conda/envs/funasr_vllm/lib/python3.12/site-packages/nvidia/cusparselt/lib"
    if os.path.exists(cusparselt):
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{cusparselt}:{existing}" if existing else cusparselt
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from funasr.auto.auto_model_vllm import AutoModelVLLM

    done_uuids: set = set()
    existing_results: list = []
    if phase2_ckpt_path and os.path.exists(phase2_ckpt_path):
        with open(phase2_ckpt_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    existing_results.append(r)
                    done_uuids.add(r["wav_uuid"])
                except Exception:
                    pass
        logger.info(f"[vLLM] Resume: {len(done_uuids)} segments already done from ckpt")

    remaining = [seg for seg in segments
                 if make_seg_uuid(seg["wav_uuid"], seg["seg_start"]) not in done_uuids]
    logger.info(f"[vLLM] {len(remaining)} segments to process "
                f"({len(segments) - len(remaining)} skipped by resume)")

    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
    _n_gpu = len(_cvd.split(",")) if _cvd else 1
    model = AutoModelVLLM(model=model_dir, tensor_parallel_size=_n_gpu,
                          gpu_memory_utilization=gpu_memory_utilization)
    logger.info("[vLLM] model loaded")

    tmp_dir = tempfile.mkdtemp(prefix="tar_pipe_vllm_")
    ckpt_f = None
    if phase2_ckpt_path:
        Path(phase2_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        ckpt_f = open(phase2_ckpt_path, "a", encoding="utf-8")

    new_results = []
    try:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start: batch_start + batch_size]
            tmp_paths = []
            valid = []
            for seg in batch:
                try:
                    if seg.get("audio") is None:
                        seg["audio"] = read_audio_from_tar(
                            seg["tar_path"], seg["wav_offset"], seg["wav_size"])
                    audio = load_audio_seg(seg["audio"], seg["seg_start"], seg["seg_end"])
                    tmp_path = os.path.join(
                        tmp_dir, f"{seg['wav_uuid'][:16]}_{seg['seg_start']:.3f}.wav")
                    sf.write(tmp_path, audio, SAMPLE_RATE)
                    if audio_dir:
                        import shutil as _sh, pathlib as _pl
                        _pl.Path(audio_dir).mkdir(parents=True, exist_ok=True)
                        kept = os.path.join(audio_dir, os.path.basename(tmp_path))
                        _sh.copy2(tmp_path, kept)
                        seg["_kept_wav"] = kept
                    tmp_paths.append(tmp_path)
                    valid.append(seg)
                except Exception as e:
                    logger.warning(f"[vLLM] skip seg {seg['wav_uuid'][:8]}: {e}")

            if not valid:
                continue
            try:
                res_list = model.generate(tmp_paths)
            except Exception as e:
                logger.warning(f"[vLLM] batch {batch_start} failed: {e}")
                for p in tmp_paths:
                    try: os.remove(p)
                    except: pass
                continue

            batch_records = []
            for seg, res, tmp_path in zip(valid, res_list, tmp_paths):
                try: os.remove(tmp_path)
                except: pass
                text = ""
                timestamps = []
                if isinstance(res, dict):
                    text = res.get("text", "").strip()
                    for t in res.get("timestamps", []):
                        tok = t.get("token", "").strip()
                        if tok:
                            timestamps.append([tok, round(t["start_time"], 3),
                                               round(t["end_time"], 3)])
                else:
                    text = str(res).strip()
                seg_dur = round(seg["seg_end"] - seg["seg_start"], 4)
                seg_uuid = make_seg_uuid(seg["wav_uuid"], seg["seg_start"])
                record = {
                    "tar_path":   seg["tar_path"],
                    "wav":        seg.get("_kept_wav"),
                    "wav_uuid":   seg_uuid,
                    "wav_offset": seg["wav_offset"],
                    "wav_size":   seg["wav_size"],
                    "seg_start":  seg["seg_start"],
                    "seg_end":    seg["seg_end"],
                    "duration":   seg_dur,
                    "num_sample": int(seg_dur * SAMPLE_RATE),
                    "text_lang":  seg["text_lang"],
                    "transcribe": {"funasr-nano": text},
                    "timestamp":  {"funasr-nano": timestamps},
                }
                batch_records.append(record)

            if ckpt_f:
                for r in batch_records:
                    ckpt_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                ckpt_f.flush()
            new_results.extend(batch_records)
            done = min(batch_start + batch_size, len(remaining))
            logger.info(f"[vLLM] {done}/{len(remaining)} done (this run)")

    finally:
        if ckpt_f:
            ckpt_f.close()
        try:
            import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
        except: pass

    all_results = existing_results + new_results
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"[vLLM] wrote {len(all_results)} records to {out_json}")

    if phase2_ckpt_path and os.path.exists(phase2_ckpt_path):
        try:
            os.remove(phase2_ckpt_path)
            logger.info(f"[vLLM] removed phase2 ckpt {phase2_ckpt_path}")
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tar_paths", nargs="+", required=True)
    p.add_argument("--out_json",  required=True)
    p.add_argument("--funasr_model_dir", required=True)
    p.add_argument("--lid_model_dir",    required=True)
    p.add_argument("--fireredvad_model", required=True)
    p.add_argument("--fireredvad_root",  required=True)
    p.add_argument("--gpu",        type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--min_dur",    type=float, default=1.0)
    p.add_argument("--max_dur",    type=float, default=30.0)
    p.add_argument("--target_langs", nargs="*", default=["en", "zh"])
    p.add_argument("--min_lang_prob", type=float, default=0.90)
    p.add_argument("--audio_dir",  default=None)
    p.add_argument("--queue_maxsize", type=int, default=200)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--resume", action="store_true",
                   help="断点续跑: 从phase1/phase2 ckpt文件恢复")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = str(Path(args.out_json).parent)
    p1_ckpt = os.path.join(out_dir, "phase1_ckpt.jsonl")
    p2_ckpt = os.path.join(out_dir, "phase2_ckpt.jsonl")

    t0 = time.time()
    segments = []

    # ── Phase 1 ─────────────────────────────────
    if args.resume and os.path.exists(p1_ckpt):
        logger.info(f"[Resume] Loading Phase1 from {p1_ckpt}")
        with open(p1_ckpt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    seg = json.loads(line)
                    seg["audio"] = None
                    segments.append(seg)
                except Exception:
                    pass
        logger.info(f"[Resume] Loaded {len(segments)} segments, skipping Phase1")
    else:
        ctx = mp.get_context("spawn")
        vad_q     = ctx.Queue(maxsize=args.queue_maxsize)
        lid_q     = ctx.Queue(maxsize=args.queue_maxsize)
        collect_q = ctx.Queue(maxsize=args.queue_maxsize)

        prod_p = ctx.Process(target=producer,
                             args=(args.tar_paths, vad_q), name="Producer")
        vad_p  = ctx.Process(target=vad_worker,
                             args=(vad_q, lid_q, args.min_dur, args.max_dur,
                                   args.fireredvad_model, args.fireredvad_root), name="VAD")
        lid_p  = ctx.Process(target=lid_worker_collect,
                             args=(lid_q, collect_q, args.lid_model_dir,
                                   args.target_langs, args.min_lang_prob), name="LID")

        logger.info("=== Phase 1: VAD / LID ===")
        for p in [prod_p, vad_p, lid_p]:
            p.start()

        Path(p1_ckpt).parent.mkdir(parents=True, exist_ok=True)
        with open(p1_ckpt, "w", encoding="utf-8") as ckpt_f:
            while True:
                item = collect_q.get()
                if item is SENTINEL:
                    break
                segments.append(item)
                meta = {k: v for k, v in item.items() if k != "audio"}
                ckpt_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                ckpt_f.flush()

        for p in [prod_p, vad_p, lid_p]:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate(); p.join(timeout=5)

        logger.info(f"Phase 1 done: {len(segments)} segments in {time.time()-t0:.1f}s")

    # ── Phase 2 ─────────────────────────────────
    logger.info("=== Phase 2: FunASR Nano vLLM ===")
    funasr_vllm_phase2(
        segments=segments,
        model_dir=args.funasr_model_dir,
        gpu_id=args.gpu,
        batch_size=args.batch_size,
        out_json=args.out_json,
        audio_dir=args.audio_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        phase2_ckpt_path=p2_ckpt,
    )
    logger.info(f"完成！总耗时 {time.time()-t0:.1f}s  输出: {args.out_json}")


if __name__ == "__main__":
    main()
