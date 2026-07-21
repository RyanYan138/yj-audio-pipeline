#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tar包音频 pipeline — transformers WhisperPipeline batch ASR + 断点续跑
架构：Producer → VAD → DNSMOS → LID → ASR(transformers batch) → 写结果

核心区别：Phase 2 用 transformers AutoModelForSpeechSeq2Seq + pipeline
  - 真正的 GPU batch 推理（所有样本同时在 GPU 上计算）
  - 支持 chunk_length_s=30 处理长音频
  - 显存利用率比 ctranslate2 逐条推理高很多

用法:
  conda activate faster_whisper
  bash run_tar_pipeline_hf_batch_ckpt.sh /data/input.tar /data/labels.json 0 8
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
WHISPER_MODEL_KEY = "whisper-large-v3"


def load_audio_seg(audio: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
    s = int(start_sec * SAMPLE_RATE)
    e = int(end_sec * SAMPLE_RATE)
    return audio[s:e]


def read_audio_from_tar(tar_path: str, wav_offset: int, wav_size: int) -> np.ndarray:
    from tar_reader import _decode_audio
    with open(tar_path, "rb") as f:
        f.seek(wav_offset)
        data = f.read(wav_size)
    audio, _ = _decode_audio(data)
    return audio


def _setup_ld_library_path():
    import glob as _glob
    _conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if _conda_prefix:
        for p in (
            _glob.glob(f"{_conda_prefix}/lib/python3.*/site-packages/nvidia/cublas/lib")
            + _glob.glob(f"{_conda_prefix}/lib/python3.*/site-packages/nvidia/cudnn/lib")
            + _glob.glob(f"{_conda_prefix}/lib/python3.*/site-packages/ctranslate2.libs")
        ):
            if os.path.exists(p):
                existing = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = f"{p}:{existing}" if existing else p


# ─────────────────────────────────────────────
# Stage 1: Producer
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
    try:
        import onnxruntime as ort
        from scipy.signal import stft
        import numpy.polynomial.polynomial as poly
    except Exception as e:
        logger.error(f"[DNSMOS] 初始化失败: {e}，透传所有片段")
        while True:
            item = dnsmos_q.get()
            lid_q.put(item)
            if item is SENTINEL:
                break
        return

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
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
            raw_sig = float(np.array(sess_sig.run(
                None, {sess_sig.get_inputs()[0].name: feat})[0]).ravel()[0])
            bv = np.array(sess_bak.run(
                None, {sess_bak.get_inputs()[0].name: feat})[0]).ravel()
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
# Stage 4: LID Worker (ctranslate2 batch)
# ─────────────────────────────────────────────

def lid_worker_collect(lid_q: mp.Queue, asr_q: mp.Queue,
                       model_dir: str,
                       target_langs: List[str], min_lang_prob: float,
                       lid_batch_size: int = 16):
    _setup_ld_library_path()
    import ctranslate2
    import queue as _queue
    from faster_whisper.feature_extractor import FeatureExtractor

    try:
        ct2_model = ctranslate2.models.Whisper(
            model_dir, device="cuda", device_index=0,
            compute_type="float16", inter_threads=1)
        logger.info("[LID] ctranslate2 Whisper ready (cuda/float16)")
    except Exception as e:
        logger.warning(f"[LID] cuda failed ({e}), fallback cpu/int8")
        ct2_model = ctranslate2.models.Whisper(
            model_dir, device="cpu", compute_type="int8", inter_threads=2)
        logger.info("[LID] ctranslate2 Whisper ready (cpu/int8)")

    feat_extractor = FeatureExtractor(feature_size=128)
    nb_max_frames = feat_extractor.nb_max_frames
    lang_cache: Dict[str, Tuple[str, float]] = {}
    buf_items = []
    buf_feats = []

    def detect_batch(items, feats):
        try:
            batch_arr = np.stack(feats, axis=0).astype(np.float32)
            storage = ctranslate2.StorageView.from_array(batch_arr)
            results = ct2_model.detect_language(storage)
            out = []
            for res in results:
                lang, prob = res[0]
                lang = lang.strip("<>|")
                out.append((lang, float(prob)))
            return out
        except Exception as e:
            logger.warning(f"[LID] batch detect failed: {e}")
            return [("unknown", 0.0)] * len(items)

    def flush_lid_buf():
        if not buf_items:
            return
        langs = detect_batch(buf_items, buf_feats)
        for item, (lang, prob) in zip(buf_items, langs):
            lang_cache[item["wav_uuid"]] = (lang, prob)
            if target_langs and lang not in target_langs:
                continue
            if prob < min_lang_prob:
                continue
            asr_q.put({**item, "text_lang": lang, "lang_prob": round(prob, 4)})
        buf_items.clear()
        buf_feats.clear()

    TIMEOUT = 1.0
    while True:
        try:
            item = lid_q.get(timeout=TIMEOUT)
        except _queue.Empty:
            flush_lid_buf()
            continue

        if item is SENTINEL:
            flush_lid_buf()
            asr_q.put(SENTINEL)
            break

        uuid = item["wav_uuid"]
        try:
            if uuid not in lang_cache:
                audio = load_audio_seg(item["audio"], item["seg_start"],
                                       min(item["seg_end"], item["seg_start"] + 12.0))
                feat = feat_extractor(audio, padding=True)
                feat = feat[:, :nb_max_frames]
                if feat.shape[1] < nb_max_frames:
                    feat = np.pad(feat, ((0, 0), (0, nb_max_frames - feat.shape[1])))
                buf_items.append(item)
                buf_feats.append(feat)
                if len(buf_items) >= lid_batch_size:
                    flush_lid_buf()
            else:
                lang, prob = lang_cache[uuid]
                if target_langs and lang not in target_langs:
                    continue
                if prob < min_lang_prob:
                    continue
                asr_q.put({**item, "text_lang": lang, "lang_prob": round(prob, 4)})
        except Exception as ex:
            logger.warning(f"[LID] {uuid[:8]} failed: {ex}")


# ─────────────────────────────────────────────
# Stage 5: ASR Worker — transformers pipeline batch
# ─────────────────────────────────────────────

def asr_worker_collect(asr_q: mp.Queue, out_q: mp.Queue,
                       model_dir: str, gpu_id: int,
                       batch_size: int,
                       force_language: Optional[str],
                       audio_dir: Optional[str],
                       ckpt_path: Optional[str]):
    """
    transformers AutoModelForSpeechSeq2Seq + pipeline batch 推理
    真正的 GPU batch：batch_size 条同时在 GPU 上并行计算
    """
    _setup_ld_library_path()
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import soundfile as sf
    import queue as _queue

    # CUDA_VISIBLE_DEVICES 已将指定 GPU 映射为 device 0，始终用 cuda:0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if "cuda" in device else torch.float32

    logger.info(f"[ASR] loading transformers Whisper from {model_dir}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir, torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_dir)

    generate_kwargs = {"task": "transcribe"}
    if force_language:
        generate_kwargs["language"] = force_language

    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        # chunk_length_s 不设置：我们的 segments 已经是 VAD 切好的短片段（1-30s）
        # 设置 chunk_length_s=30 会把所有 segment pad 到 30s 导致空输出
        batch_size=batch_size,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs=generate_kwargs,
    )
    logger.info(f"[ASR] transformers pipeline ready  device={device}  batch={batch_size}")

    from tar_reader import make_seg_uuid

    if audio_dir:
        Path(audio_dir).mkdir(parents=True, exist_ok=True)

    ckpt_f = None
    if ckpt_path:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        ckpt_f = open(ckpt_path, "a", encoding="utf-8")

    buf: List[dict] = []
    ASR_TIMEOUT = 2.0
    total_done = 0

    def flush(batch):
        if not batch:
            return []
        audios = []
        valid = []
        for seg in batch:
            try:
                if seg.get("audio") is None:
                    seg["audio"] = read_audio_from_tar(
                        seg["tar_path"], seg["wav_offset"], seg["wav_size"])
                audio = load_audio_seg(seg["audio"], seg["seg_start"], seg["seg_end"])
                if audio_dir:
                    kept = os.path.join(
                        audio_dir, f"{seg['wav_uuid'][:16]}_{seg['seg_start']:.3f}.wav")
                    sf.write(kept, audio, SAMPLE_RATE)
                    seg["_kept_wav"] = kept
                audios.append({"array": audio, "sampling_rate": SAMPLE_RATE})
                valid.append(seg)
            except Exception as e:
                logger.warning(f"[ASR] skip {seg['wav_uuid'][:8]}: {e}")

        if not valid:
            return []

        try:
            lang = force_language or valid[0].get("text_lang") or None
            gkw = {"task": "transcribe"}
            if lang:
                gkw["language"] = lang
            outs = asr_pipe(audios, generate_kwargs=gkw)
            if isinstance(outs, dict):
                outs = [outs]
        except Exception as e:
            logger.warning(f"[ASR] batch failed: {e}")
            outs = [{"text": ""}] * len(valid)

        records = []
        for seg, out in zip(valid, outs):
            text = (out.get("text") or "").strip()
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
                "text_lang":  seg.get("text_lang", ""),
                "transcribe": {WHISPER_MODEL_KEY: text},
                "timestamp":  {WHISPER_MODEL_KEY: []},
            }
            records.append(record)

        if ckpt_f:
            for r in records:
                ckpt_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            ckpt_f.flush()

        for r in records:
            out_q.put(r)

        return records

    while True:
        try:
            item = asr_q.get(timeout=ASR_TIMEOUT)
        except _queue.Empty:
            if buf:
                recs = flush(buf)
                total_done += len(recs)
                logger.info(f"[ASR] {total_done} done (timeout flush {len(recs)})")
                buf = []
            continue

        if item is SENTINEL:
            if buf:
                recs = flush(buf)
                total_done += len(recs)
            break
        buf.append(item)
        if len(buf) >= batch_size:
            recs = flush(buf)
            total_done += len(recs)
            logger.info(f"[ASR] {total_done} done")
            buf = []

    if ckpt_f:
        ckpt_f.close()
    out_q.put(SENTINEL)
    logger.info(f"[ASR] worker done, total {total_done}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tar_paths", nargs="+", required=True)
    p.add_argument("--out_json",  required=True)
    p.add_argument("--whisper_model_dir", required=True)
    p.add_argument("--lid_model_dir", default=None,
                   help="LID 专用 ct2 模型（默认用 faster-whisper-large-v3）")
    p.add_argument("--fireredvad_model", required=True)
    p.add_argument("--fireredvad_root",  required=True)
    p.add_argument("--dnsmos_dir",       required=True)
    p.add_argument("--gpu",        type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8,
                   help="transformers pipeline batch size（RTX4090建议8~16，V100建议4~8）")
    p.add_argument("--lid_batch_size", type=int, default=16)
    p.add_argument("--language",   default=None)
    p.add_argument("--min_dur",    type=float, default=1.0)
    p.add_argument("--max_dur",    type=float, default=30.0)
    p.add_argument("--min_mos_ovr", type=float, default=2.0)
    p.add_argument("--min_mos_sig", type=float, default=2.0)
    p.add_argument("--min_mos_bak", type=float, default=3.5)
    p.add_argument("--target_langs", nargs="*", default=["en", "zh"])
    p.add_argument("--min_lang_prob", type=float, default=0.90)
    p.add_argument("--audio_dir",  default=None)
    p.add_argument("--queue_maxsize",       type=int, default=200)
    p.add_argument("--dnsmos_input_length", type=int, default=9)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    lid_model = args.lid_model_dir or args.whisper_model_dir
    out_dir   = str(Path(args.out_json).parent)
    ckpt_path = os.path.join(out_dir, "asr_ckpt.jsonl")

    t0 = time.time()
    done_uuids: set = set()
    existing_results: list = []
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
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
        logger.info(f"[Resume] loaded {len(done_uuids)} done segments")

    ctx = mp.get_context("spawn")
    vad_q  = ctx.Queue(maxsize=args.queue_maxsize)
    dns_q  = ctx.Queue(maxsize=args.queue_maxsize)
    lid_q  = ctx.Queue(maxsize=args.queue_maxsize)
    asr_q  = ctx.Queue(maxsize=args.queue_maxsize)
    out_q  = ctx.Queue(maxsize=args.queue_maxsize)

    prod_p = ctx.Process(target=producer,
                         args=(args.tar_paths, vad_q), name="Producer")
    vad_p  = ctx.Process(target=vad_worker,
                         args=(vad_q, dns_q, args.min_dur, args.max_dur,
                               args.fireredvad_model, args.fireredvad_root), name="VAD")
    dns_p  = ctx.Process(target=dnsmos_worker,
                         args=(dns_q, lid_q, args.dnsmos_dir, args.gpu,
                               args.dnsmos_input_length,
                               args.min_mos_ovr, args.min_mos_sig, args.min_mos_bak),
                         name="DNSMOS")
    lid_p  = ctx.Process(target=lid_worker_collect,
                         args=(lid_q, asr_q, lid_model,
                               args.target_langs, args.min_lang_prob,
                               args.lid_batch_size), name="LID")
    asr_p  = ctx.Process(target=asr_worker_collect,
                         args=(asr_q, out_q, args.whisper_model_dir, args.gpu,
                               args.batch_size, args.language,
                               args.audio_dir, ckpt_path), name="ASR")

    import threading
    def _dns_watchdog():
        dns_p.join()
        if dns_p.exitcode != 0:
            logger.warning(f"[DNSMOS] 异常退出(exitcode={dns_p.exitcode})，透传 dnsmos_q → lid_q")
            while True:
                item = dns_q.get()
                lid_q.put(item)
                if item is SENTINEL:
                    break
    _wdog = threading.Thread(target=_dns_watchdog, daemon=True)

    logger.info("=== 五阶段流水线（transformers batch ASR）启动 ===")
    for p in [prod_p, vad_p, dns_p, lid_p, asr_p]:
        p.start()
    _wdog.start()

    new_results = []
    while True:
        item = out_q.get()
        if item is SENTINEL:
            break
        if item["wav_uuid"] in done_uuids:
            continue
        new_results.append(item)

    for p in [prod_p, vad_p, dns_p, lid_p, asr_p]:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    all_results = existing_results + new_results
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"[Main] wrote {len(all_results)} records to {args.out_json}")

    if os.path.exists(ckpt_path) and len(all_results) > 0:
        try:
            os.remove(ckpt_path)
        except Exception:
            pass

    logger.info(f"完成！总耗时 {time.time()-t0:.1f}s  输出: {args.out_json}")


if __name__ == "__main__":
    main()
