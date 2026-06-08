#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tar包音频 pipeline — 支持客户格式输出
输入: 一个或多个本地 .tar 包
输出: labels.json，每条记录包含：
  tar_path / wav_uuid / wav_offset / wav_size /
  seg_start / seg_end / duration / num_sample /
  text_lang / transcribe / timestamp

流水线结构（funasr_vllm模式）:
  TarReader(主进程) → [vad_q] → VAD进程
                    → [dnsmos_q] → DNSMOS进程
                    → [lid_q] → LID进程
                    → collect → Phase2: FunASR vLLM(主进程)

用法:
  python3 tar_pipeline.py \
    --tar_paths /data/shards_000.tar /data/shards_001.tar \
    --out_json  /data/labels.json \
    --funasr_model_dir /path/to/Fun-ASR-Nano-2512 \
    --lid_model_dir    /path/to/faster-whisper-large-v3 \
    --fireredvad_model /path/to/FireRedVAD/VAD \
    --fireredvad_root  /path/to/FireRedVAD \
    --dnsmos_dir       /path/to/dns_mos \
    --gpu 2
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


# ─────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────

def load_audio_seg(audio: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
    s = int(start_sec * SAMPLE_RATE)
    e = int(end_sec * SAMPLE_RATE)
    return audio[s:e]


# ─────────────────────────────────────────────
# Stage 1: Producer — 读tar包，放入vad队列
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Stage 2: VAD Worker (FireRedVAD)
# ─────────────────────────────────────────────

def vad_worker(vad_q: mp.Queue, dnsmos_q: mp.Queue,
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
            dnsmos_q.put(SENTINEL)
            break
        audio = item["audio"]
        try:
            # FireRedVAD需要文件路径，写临时文件
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
                dnsmos_q.put({**item, "seg_start": round(s, 4), "seg_end": round(e, 4)})
        except Exception as ex:
            logger.warning(f"[VAD] {item['wav_uuid'][:8]} failed: {ex}")


# ─────────────────────────────────────────────
# Stage 3: DNSMOS Worker
# ─────────────────────────────────────────────

def dnsmos_worker(dnsmos_q: mp.Queue, lid_q: mp.Queue,
                  dnsmos_dir: str, gpu_id: int,
                  input_length: int,
                  min_mos_ovr: float, min_mos_sig: float, min_mos_bak: float):
    import onnxruntime as ort
    from scipy.signal import stft
    import numpy.polynomial.polynomial as poly

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_sig = ort.InferenceSession(os.path.join(dnsmos_dir, "sig.onnx"), providers=providers)
    sess_bak = ort.InferenceSession(os.path.join(dnsmos_dir, "bak_ovr.onnx"), providers=providers)
    COEFS_SIG = np.array([9.651228e-01, 6.592638e-01, 7.572373e-02])
    COEFS_BAK = np.array([-3.733460e+00, 2.700114e+00, -1.721333e-01])
    COEFS_OVR = np.array([8.924547e-01, 6.609982e-01, 7.600270e-02])
    logger.info(f"[DNSMOS gpu={gpu_id}] ready")

    def score(audio: np.ndarray) -> Tuple[float, float, float]:
        hop = SAMPLE_RATE
        target = input_length * SAMPLE_RATE
        if len(audio) < target:
            audio = np.tile(audio, int(np.ceil(target / len(audio))))
        num_hops = max(1, int(np.floor((len(audio) - target) / hop)) + 1)
        sig_list, bak_list, ovr_list = [], [], []
        for i in range(num_hops):
            chunk = audio[i * hop: i * hop + target]
            if len(chunk) < target:
                chunk = np.pad(chunk, (0, target - len(chunk)))
            cp = np.pad(chunk, (160, 160))
            _, _, Zxx = stft(cp, fs=SAMPLE_RATE, nperseg=320, noverlap=160,
                             nfft=320, boundary=None, padded=False)
            lps = np.log10(np.maximum(np.abs(Zxx) ** 2, 1e-12)).T
            feat = lps[np.newaxis, :, :].astype(np.float32)
            raw_sig = float(np.array(sess_sig.run(None, {sess_sig.get_inputs()[0].name: feat})[0]).ravel()[0])
            bv = np.array(sess_bak.run(None, {sess_bak.get_inputs()[0].name: feat})[0]).ravel()
            sig_list.append(float(poly.polyval(raw_sig, COEFS_SIG)))
            bak_list.append(float(poly.polyval(float(bv[1]), COEFS_BAK)))
            ovr_list.append(float(poly.polyval(float(bv[2]), COEFS_OVR)))
        return (round(float(np.mean(sig_list)), 3),
                round(float(np.mean(bak_list)), 3),
                round(float(np.mean(ovr_list)), 3))

    while True:
        item = dnsmos_q.get()
        if item is SENTINEL:
            lid_q.put(SENTINEL)
            break
        try:
            audio = load_audio_seg(item["audio"], item["seg_start"], item["seg_end"])
            mos_sig, mos_bak, mos_ovr = score(audio)
            if mos_ovr < min_mos_ovr or mos_sig < min_mos_sig or mos_bak < min_mos_bak:
                continue
            lid_q.put({**item, "mos_sig": mos_sig, "mos_bak": mos_bak, "mos_ovr": mos_ovr})
        except Exception as ex:
            logger.warning(f"[DNSMOS] {item.get('wav_uuid','?')[:8]} failed: {ex}")


# ─────────────────────────────────────────────
# Stage 4: LID Worker
# ─────────────────────────────────────────────

def lid_worker_collect(lid_q: mp.Queue, out_q: mp.Queue,
                       model_dir: str,
                       target_langs: List[str], min_lang_prob: float):
    from faster_whisper import WhisperModel
    # 优先用GPU，失败回退CPU
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


# ─────────────────────────────────────────────
# Stage 5: FunASR vLLM Phase2 (主进程)
# ─────────────────────────────────────────────

def funasr_vllm_phase2(segments: list, model_dir: str, gpu_id: int,
                       batch_size: int, out_json: str):
    from tar_reader import make_seg_uuid

    cusparselt = "/opt/conda/envs/funasr_vllm/lib/python3.12/site-packages/nvidia/cusparselt/lib"
    if os.path.exists(cusparselt):
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{cusparselt}:{existing}" if existing else cusparselt
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MODELSCOPE_CACHE"] = "/Work21/2025/yanjiahao/modelscope_cache"

    from funasr.auto.auto_model_vllm import AutoModelVLLM
    import tempfile, soundfile as sf

    model = AutoModelVLLM(model=model_dir, tensor_parallel_size=1)
    logger.info(f"[vLLM] model loaded, {len(segments)} segments to process")

    tmp_dir = tempfile.mkdtemp(prefix="tar_pipe_vllm_")
    results = []

    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start: batch_start + batch_size]

        # 写临时wav文件
        tmp_paths = []
        valid = []
        for seg in batch:
            try:
                audio = load_audio_seg(seg["audio"], seg["seg_start"], seg["seg_end"])
                tmp_path = os.path.join(tmp_dir, f"{seg['wav_uuid'][:16]}_{seg['seg_start']:.3f}.wav")
                sf.write(tmp_path, audio, SAMPLE_RATE)
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

        for seg, res, tmp_path in zip(valid, res_list, tmp_paths):
            try: os.remove(tmp_path)
            except: pass

            text = ""
            timestamps = []
            if isinstance(res, dict):
                text = res.get("text", "").strip()
                # 提取token级时间戳，转成[字, start, end]格式
                for t in res.get("timestamps", []):
                    tok = t.get("token", "").strip()
                    if tok:
                        timestamps.append([tok, round(t["start_time"], 3),
                                           round(t["end_time"], 3)])
            else:
                text = str(res).strip()

            seg_dur = round(seg["seg_end"] - seg["seg_start"], 4)
            seg_uuid = make_seg_uuid(seg["wav_uuid"], seg["seg_start"])

            results.append({
                "tar_path":   seg["tar_path"],
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
            })

        done = min(batch_start + batch_size, len(segments))
        logger.info(f"[vLLM] {done}/{len(segments)} done")

    try:
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
    except: pass

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"[vLLM] wrote {len(results)} records to {out_json}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tar_paths", nargs="+", required=True, help="输入tar包路径，可多个")
    p.add_argument("--out_json",  required=True)
    p.add_argument("--funasr_model_dir", required=True)
    p.add_argument("--lid_model_dir",    required=True)
    p.add_argument("--fireredvad_model", required=True)
    p.add_argument("--fireredvad_root",  required=True)
    p.add_argument("--dnsmos_dir",       required=True)
    p.add_argument("--gpu",        type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--min_dur",    type=float, default=1.0)
    p.add_argument("--max_dur",    type=float, default=30.0)
    p.add_argument("--min_mos_ovr", type=float, default=2.0)
    p.add_argument("--min_mos_sig", type=float, default=2.0)
    p.add_argument("--min_mos_bak", type=float, default=3.5)
    p.add_argument("--target_langs", nargs="*", default=["en", "zh"])
    p.add_argument("--min_lang_prob", type=float, default=0.90)
    p.add_argument("--queue_maxsize", type=int, default=200)
    p.add_argument("--dnsmos_input_length", type=int, default=9)
    return p.parse_args()


def main():
    args = parse_args()
    ctx = mp.get_context("spawn")

    vad_q    = ctx.Queue(maxsize=args.queue_maxsize)
    dnsmos_q = ctx.Queue(maxsize=args.queue_maxsize)
    lid_q    = ctx.Queue(maxsize=args.queue_maxsize)
    collect_q = ctx.Queue(maxsize=args.queue_maxsize)

    # 启动各stage进程
    prod_p = ctx.Process(target=producer,
                         args=(args.tar_paths, vad_q), name="Producer")
    vad_p = ctx.Process(target=vad_worker,
                        args=(vad_q, dnsmos_q, args.min_dur, args.max_dur,
                              args.fireredvad_model, args.fireredvad_root), name="VAD")
    dns_p = ctx.Process(target=dnsmos_worker,
                        args=(dnsmos_q, lid_q, args.dnsmos_dir, args.gpu,
                              args.dnsmos_input_length,
                              args.min_mos_ovr, args.min_mos_sig, args.min_mos_bak), name="DNSMOS")
    lid_p = ctx.Process(target=lid_worker_collect,
                        args=(lid_q, collect_q, args.lid_model_dir,
                              args.target_langs, args.min_lang_prob), name="LID")

    processes = [prod_p, vad_p, dns_p, lid_p]

    logger.info("=== Phase 1: VAD / DNSMOS / LID ===")
    t0 = time.time()
    for p in processes:
        p.start()

    # 收集Phase1结果
    segments = []
    while True:
        item = collect_q.get()
        if item is SENTINEL:
            break
        segments.append(item)

    # 等所有子进程退出
    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    logger.info(f"Phase 1 done: {len(segments)} segments in {time.time()-t0:.1f}s")

    # Phase 2: vLLM推理（主进程，无子进程干扰）
    logger.info("=== Phase 2: FunASR Nano vLLM ===")
    funasr_vllm_phase2(
        segments=segments,
        model_dir=args.funasr_model_dir,
        gpu_id=args.gpu,
        batch_size=args.batch_size,
        out_json=args.out_json,
    )

    logger.info(f"完成！总耗时 {time.time()-t0:.1f}s  输出: {args.out_json}")


if __name__ == "__main__":
    main()
