#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流水线化 audio pipeline — Queue + 多进程
优化：每个 audio VAD 完成后立即进入 DNSMOS，不等全量完成

设计：
  Producer → [VAD Queue] → VAD Workers → [DNSMOS Queue] → DNSMOS Worker
           → [LID Queue] → LID Worker → [ASR Queue] → ASR Workers
           → [Result Queue] → Consumer

用法：
  python3 streaming_pipeline.py \
    --input_root /path/to/audio \
    --out_json   /path/to/labels.json \
    --vad_model  silero|fireredvad \
    --asr_model  whisper|funasr|funasr_vllm \
    --whisper_model_dir /path/to/whisper-large-v3 \
    --funasr_model_dir  /path/to/Fun-ASR-Nano-2512 \
    --lid_model_dir     /path/to/faster-whisper-large-v3 \
    --dnsmos_dir        /path/to/dns_mos \
    --gpus 0,1,2 \
    [--min_dur 1.0] [--max_dur 30.0] \
    [--min_mos_ovr 2.0] [--target_langs en zh] \
    [--fireredvad_root /path/to/FireRedVAD]

注意：funasr_vllm 模式下 ASR 在主进程运行（vllm 不支持子进程），
      Consumer 也合并到主进程，--gpus 第一个 GPU 给 vllm 用。
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path
from queue import Empty
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SENTINEL = None  # 毒丸信号，关闭队列
DNSMOS_PYTHON = "/Work21/2025/yanjiahao/conda-envs/dnsmos/bin/python"
FUNASR_PYTHON = "/Work21/2025/yanjiahao/conda-envs/funasr_nano/bin/python"


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def collect_audio_files(root: Path, max_files: Optional[int] = None) -> List[Path]:
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
    files = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    return files[:max_files] if max_files else files


def make_audio_id(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root).with_suffix("")).replace(os.sep, "/")
    except ValueError:
        return path.stem


def load_audio(path: str, start_sec: float = 0.0, end_sec: float = -1.0) -> np.ndarray:
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        s = int(start_sec * sr)
        e = int(end_sec * sr) if end_sec > 0 else -1
        f.seek(s)
        n = (e - s) if e > 0 else -1
        audio = f.read(n if n > 0 else -1, dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        import torchaudio, torch
        a = torch.from_numpy(audio).unsqueeze(0)
        audio = torchaudio.functional.resample(a, sr, SAMPLE_RATE).squeeze(0).numpy()
    return audio


# ─────────────────────────────────────────────
# Stage 1: Producer — 读音频文件放入 VAD 队列
# ─────────────────────────────────────────────

def producer(audio_files: List[Tuple[str, str]], vad_q: mp.Queue):
    """将 (audio_id, path) 放入 VAD 队列"""
    for audio_id, path in audio_files:
        vad_q.put({"audio_id": audio_id, "path": path})
        logger.debug(f"[Producer] queued {audio_id}")
    vad_q.put(SENTINEL)
    logger.info(f"[Producer] done, {len(audio_files)} files")


# ─────────────────────────────────────────────
# Stage 2: VAD Worker
# ─────────────────────────────────────────────

def vad_worker_silero(vad_q: mp.Queue, dnsmos_q: mp.Queue,
                      min_dur: float, max_dur: float):
    """Silero VAD worker"""
    import torch
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    torch.set_num_threads(2)
    model = load_silero_vad()
    logger.info("[VAD-Silero] ready")

    while True:
        item = vad_q.get()
        if item is SENTINEL:
            dnsmos_q.put(SENTINEL)
            break
        audio_id, path = item["audio_id"], item["path"]
        try:
            wav = read_audio(path, sampling_rate=SAMPLE_RATE)
            tss = get_speech_timestamps(wav, model, sampling_rate=SAMPLE_RATE,
                                        min_speech_duration_ms=int(min_dur * 1000))
            segments = []
            for ts in tss:
                s = ts["start"] / SAMPLE_RATE
                e = ts["end"] / SAMPLE_RATE
                dur = e - s
                if dur < min_dur or (max_dur > 0 and dur > max_dur):
                    continue
                segments.append({"start_sec": round(s, 4), "end_sec": round(e, 4),
                                  "duration_sec": round(dur, 4)})
            for seg in segments:
                dnsmos_q.put({**item, **seg})
            logger.debug(f"[VAD] {audio_id}: {len(segments)} segs")
        except Exception as ex:
            logger.warning(f"[VAD] {audio_id} failed: {ex}")


def vad_worker_firered(vad_q: mp.Queue, dnsmos_q: mp.Queue,
                       min_dur: float, max_dur: float,
                       model_dir: str, fireredvad_root: str):
    """FireRedVAD worker"""
    sys.path.insert(0, fireredvad_root)
    from fireredvad.vad import FireRedVad, FireRedVadConfig
    cfg = FireRedVadConfig(use_gpu=True, speech_threshold=0.5,
                           min_speech_frame=20, max_speech_frame=2000,
                           min_silence_frame=10, merge_silence_frame=50,
                           extend_speech_frame=5)
    model = FireRedVad.from_pretrained(model_dir, cfg)
    logger.info("[VAD-FireRed] ready")

    while True:
        item = vad_q.get()
        if item is SENTINEL:
            dnsmos_q.put(SENTINEL)
            break
        audio_id, path = item["audio_id"], item["path"]
        try:
            result, _ = model.detect(str(path))
            for ts in result.get("timestamps", []):
                s, e = float(ts[0]), float(ts[1])
                dur = e - s
                if dur < min_dur or (max_dur > 0 and dur > max_dur):
                    continue
                dnsmos_q.put({**item,
                               "start_sec": round(s, 4),
                               "end_sec": round(e, 4),
                               "duration_sec": round(dur, 4)})
        except Exception as ex:
            logger.warning(f"[VAD-FireRed] {audio_id} failed: {ex}")


# ─────────────────────────────────────────────
# Stage 3: DNSMOS Worker
# ─────────────────────────────────────────────

def dnsmos_worker(dnsmos_q: mp.Queue, lid_q: mp.Queue,
                  dnsmos_dir: str, gpu_id: int,
                  input_length: int, min_mos_ovr: float,
                  min_mos_sig: float, min_mos_bak: float):
    """DNSMOS 打分 + 过滤 worker — 模型常驻，每条音频直接推理"""
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

    logger.info(f"[DNSMOS gpu={gpu_id}] ready, providers={sess_sig.get_providers()}")

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
            audio = load_audio(item["path"], item["start_sec"], item["end_sec"])
            mos_sig, mos_bak, mos_ovr = score(audio)
            if mos_ovr < min_mos_ovr or mos_sig < min_mos_sig or mos_bak < min_mos_bak:
                continue
            lid_q.put({**item, "mos_sig": mos_sig, "mos_bak": mos_bak, "mos_ovr": mos_ovr})
        except Exception as ex:
            logger.warning(f"[DNSMOS] {item.get('audio_id')} failed: {ex}")


def lid_worker(lid_q: mp.Queue, asr_q: mp.Queue,
               model_dir: str, gpu_id: int,
               target_langs: List[str], min_lang_prob: float):
    """语言识别 + 过滤 worker（基于 faster-whisper）"""
    from faster_whisper import WhisperModel
    # ctranslate2 cudnn9 依赖在 funasr_vllm 镜像里不满足，固定用 cpu/int8
    model = WhisperModel(model_dir, device="cpu", compute_type="int8")
    logger.info("[LID] ready (cpu/int8)")

    lang_cache: Dict[str, Tuple[str, float]] = {}

    while True:
        item = lid_q.get()
        if item is SENTINEL:
            asr_q.put(SENTINEL)
            break
        audio_id = item["audio_id"]
        try:
            if audio_id not in lang_cache:
                audio = load_audio(item["path"], item["start_sec"],
                                   min(item["end_sec"], item["start_sec"] + 12.0))
                _, info = model.transcribe(audio, task="transcribe")
                lang = info.language
                prob = info.language_probability
                lang_cache[audio_id] = (lang, prob)
            else:
                lang, prob = lang_cache[audio_id]

            if target_langs and lang not in target_langs:
                continue
            if prob < min_lang_prob:
                continue
            asr_q.put({**item, "lang": lang, "lang_prob": round(prob, 4)})
        except Exception as ex:
            logger.warning(f"[LID] {audio_id} failed: {ex}")


# ─────────────────────────────────────────────
# Stage 5a: Whisper ASR Worker
# ─────────────────────────────────────────────

def whisper_asr_worker(asr_q: mp.Queue, result_q: mp.Queue,
                       model_dir: str, gpu_id: int,
                       batch_size: int = 8):
    """Whisper 转写 worker，用 faster-whisper（CTranslate2 后端，无 numpy 类型问题）"""
    from faster_whisper import WhisperModel

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        model = WhisperModel(model_dir, device="cuda", compute_type="float16")
        logger.info(f"[Whisper gpu={gpu_id}] ready (cuda)")
    except Exception as e:
        logger.warning(f"[Whisper] cuda init failed ({e}), fallback to cpu")
        model = WhisperModel(model_dir, device="cpu", compute_type="int8")
        logger.info(f"[Whisper] ready (cpu/int8)")

    while True:
        try:
            item = asr_q.get(timeout=2.0)
        except Empty:
            continue
        if item is SENTINEL:
            result_q.put(SENTINEL)
            break
        try:
            audio = load_audio(item["path"], item["start_sec"], item["end_sec"])
            segs, info = model.transcribe(audio, beam_size=5)
            text = " ".join(s.text for s in segs).strip()
        except Exception as ex:
            logger.warning(f"[Whisper] {item.get('audio_id')} failed: {ex}")
            text = ""
        result_q.put({**item, "text": text, "lang": item.get("lang"), "lang_prob": item.get("lang_prob")})


def funasr_asr_worker(asr_q: mp.Queue, result_q: mp.Queue,
                      model_dir: str, gpu_id: int):
    """FunASR Nano 常驻 worker — 模型只加载一次，循环处理队列"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from funasr import AutoModel
    model = AutoModel(model=model_dir, device="cuda:0", disable_update=True)
    tmp_dir = tempfile.mkdtemp(prefix="funasr_stream_")
    logger.info(f"[FunASR gpu={gpu_id}] ready")

    while True:
        item = asr_q.get()
        if item is SENTINEL:
            result_q.put(SENTINEL)
            break
        tmp_path = None
        try:
            audio = load_audio(item["path"], item["start_sec"], item["end_sec"])
            tmp_path = os.path.join(tmp_dir, f"{os.getpid()}_{time.time_ns()}.wav")
            sf.write(tmp_path, audio, SAMPLE_RATE)
            res = model.generate(input=tmp_path)
            text = res[0].get("text", "").strip() if res else ""
        except Exception as ex:
            logger.warning(f"[FunASR] {item.get('audio_id')} failed: {ex}")
            text = ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        result_q.put({**item, "text": text})

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# Stage 5c: FunASR Nano vLLM — 主进程内运行
# 两阶段方案：
#   Phase 1 — 多进程跑 VAD+DNSMOS+LID，结果收集到 shared list
#   Phase 2 — 所有子进程退出后，主进程独占 GPU 跑 vllm
# ─────────────────────────────────────────────

def lid_worker_collect(lid_q: mp.Queue, out_q: mp.Queue,
                       model_dir: str, gpu_id: int,
                       target_langs: List[str], min_lang_prob: float):
    """LID worker，通过的 segment 放进 out_q（供主进程收集），最后发 SENTINEL"""
    from faster_whisper import WhisperModel
    # ctranslate2 cudnn9 依赖在 funasr_vllm 镜像里不满足，固定用 cpu/int8
    model = WhisperModel(model_dir, device="cpu", compute_type="int8")
    logger.info("[LID] ready (cpu/int8)")

    lang_cache: Dict[str, Tuple[str, float]] = {}

    while True:
        item = lid_q.get()
        if item is SENTINEL:
            out_q.put(SENTINEL)
            break
        audio_id = item["audio_id"]
        try:
            if audio_id not in lang_cache:
                audio = load_audio(item["path"], item["start_sec"],
                                   min(item["end_sec"], item["start_sec"] + 12.0))
                _, info = model.transcribe(audio, task="transcribe")
                lang = info.language
                prob = info.language_probability
                lang_cache[audio_id] = (lang, prob)
            else:
                lang, prob = lang_cache[audio_id]

            if target_langs and lang not in target_langs:
                continue
            if prob < min_lang_prob:
                continue
            out_q.put({**item, "lang": lang, "lang_prob": round(prob, 4)})
        except Exception as ex:
            logger.warning(f"[LID] {audio_id} failed: {ex}")


def funasr_vllm_phase2(segments: List[dict],
                       model_dir: str,
                       gpu_id: int,
                       batch_size: int,
                       out_json: str,
                       audio_dir: str):
    """Phase 2: 所有子进程已退出，主进程独占 GPU 跑 vllm 批量 ASR"""
    import re
    import subprocess

    cusparselt_path = "/opt/conda/envs/funasr_vllm/lib/python3.12/site-packages/nvidia/cusparselt/lib"
    if os.path.exists(cusparselt_path):
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{cusparselt_path}:{existing}" if existing else cusparselt_path

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MODELSCOPE_CACHE"] = "/Work21/2025/yanjiahao/modelscope_cache"

    logger.info(f"[Phase2-vLLM] {len(segments)} segments to transcribe, gpu={gpu_id}")

    from funasr.auto.auto_model_vllm import AutoModelVLLM
    model = AutoModelVLLM(model=model_dir, tensor_parallel_size=1)
    logger.info(f"[Phase2-vLLM] model ready, batch_size={batch_size}")

    tmp_dir = tempfile.mkdtemp(prefix="funasr_vllm_p2_")
    Path(audio_dir).mkdir(parents=True, exist_ok=True)
    labels = []

    def write_segment_wav(item, text):
        src = item.get("path")
        if not src or not os.path.isfile(src):
            return
        s, e = float(item["start_sec"]), float(item["end_sec"])
        uid = f"{len(labels):08d}_{re.sub(r'[^0-9A-Za-z._-]+', '_', item.get('audio_id', 'x'))}"
        out_wav = os.path.join(audio_dir, f"{uid}.wav")
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-ss", str(s), "-t", str(round(e - s, 3)),
                 "-i", src, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav],
                check=True,
            )
        except Exception as ex:
            logger.warning(f"[Export] ffmpeg failed {uid}: {ex}")
            return
        label = {k: item[k] for k in
                 ["audio_id", "path", "start_sec", "end_sec", "duration_sec",
                  "lang", "lang_prob", "mos_sig", "mos_bak", "mos_ovr"]
                 if k in item}
        label["uid"] = uid
        label["path"] = out_wav
        label["text"] = text
        labels.append(label)

    total = len(segments)
    for batch_start in range(0, total, batch_size):
        batch = segments[batch_start: batch_start + batch_size]
        tmp_paths = []
        valid_idx = []
        for i, it in enumerate(batch):
            try:
                audio = load_audio(it["path"], it["start_sec"], it["end_sec"])
                tp = os.path.join(tmp_dir, f"{batch_start+i}.wav")
                sf.write(tp, audio, SAMPLE_RATE)
                tmp_paths.append(tp)
                valid_idx.append(i)
            except Exception as ex:
                logger.warning(f"[Phase2] prep failed {it.get('audio_id')}: {ex}")
                tmp_paths.append(None)

        valid_paths = [p for p in tmp_paths if p]
        valid_items = [batch[i] for i in valid_idx]

        if valid_paths:
            try:
                results = model.generate(valid_paths)
                for it, res in zip(valid_items, results):
                    text = (res.get("text", "") if isinstance(res, dict) else str(res)).strip()
                    if text:
                        write_segment_wav(it, text)
            except Exception as ex:
                logger.warning(f"[Phase2] vllm batch failed: {ex}")

        for p in valid_paths:
            try:
                os.remove(p)
            except Exception:
                pass

        done = min(batch_start + batch_size, total)
        logger.info(f"[Phase2] {done}/{total} processed, {len(labels)} exported")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    logger.info(f"[Phase2] done. exported={len(labels)} -> {out_json}")


def export_results(result_q: mp.Queue, out_json: str,
                   audio_dir: str, n_sentinels: int,
                   target_sr: int = 16000):
    """收集结果、切片音频、写 labels.json"""
    import re
    import subprocess
    labels = []
    received = 0
    sentinel_count = 0
    Path(audio_dir).mkdir(parents=True, exist_ok=True)

    while sentinel_count < n_sentinels:
        try:
            item = result_q.get(timeout=5)
        except Empty:
            continue
        if item is SENTINEL:
            sentinel_count += 1
            continue
        received += 1
        text = (item.get("text") or "").strip()
        if not text:
            continue
        src = item.get("path")
        if not src or not os.path.isfile(src):
            continue
        s, e = float(item["start_sec"]), float(item["end_sec"])
        uid = f"{len(labels):08d}_{re.sub(r'[^0-9A-Za-z._-]+','_', item.get('audio_id','x'))}"
        out_wav = os.path.join(audio_dir, f"{uid}.wav")
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-ss", str(s), "-t", str(round(e - s, 3)),
                 "-i", src, "-ac", "1", "-ar", str(target_sr), "-c:a", "pcm_s16le", out_wav],
                check=True,
            )
        except Exception as ex:
            logger.warning(f"[Export] ffmpeg failed {uid}: {ex}")
            continue
        label = {k: item[k] for k in
                 ["audio_id", "seg_id", "path", "start_sec", "end_sec", "duration_sec",
                  "text", "lang", "lang_prob", "mos_sig", "mos_bak", "mos_ovr"]
                 if k in item}
        label["uid"] = uid
        label["path"] = out_wav
        labels.append(label)
        if received % 100 == 0:
            logger.info(f"[Export] {received} processed, {len(labels)} exported")

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    logger.info(f"[Export] done. exported={len(labels)} -> {out_json}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--audio_dir", default=None,
                   help="切片音频输出目录，默认 out_json 同级的 audio/")
    p.add_argument("--vad_model", default="fireredvad", choices=["silero", "fireredvad"])
    p.add_argument("--asr_model", default="funasr", choices=["whisper", "funasr", "funasr_vllm"])
    p.add_argument("--whisper_model_dir",
        default="/Work21/2025/yanjiahao/modelscope_cache/models/AI-ModelScope/whisper-large-v3")
    p.add_argument("--funasr_model_dir",
        default="/Work21/2025/yanjiahao/modelscope_cache/models/FunAudioLLM/Fun-ASR-Nano-2512")
    p.add_argument("--lid_model_dir",
        default="/Work21/2025/yanjiahao/hf_cache/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478")
    p.add_argument("--dnsmos_dir",
        default="/Work21/2025/yanjiahao/YJ-audio-pipeline/yj-audio-pipeline/dns_mos")
    p.add_argument("--fireredvad_model",
        default="/Work21/2025/yanjiahao/FireRedVAD/pretrained_models/xukaituo/FireRedVAD/VAD")
    p.add_argument("--fireredvad_root",
        default="/Work21/2025/yanjiahao/FireRedVAD")
    p.add_argument("--gpus", default="0,1", help="可用 GPU 列表，逗号分隔")
    p.add_argument("--min_dur", type=float, default=1.0)
    p.add_argument("--max_dur", type=float, default=30.0)
    p.add_argument("--min_mos_ovr", type=float, default=2.0)
    p.add_argument("--min_mos_sig", type=float, default=2.0)
    p.add_argument("--min_mos_bak", type=float, default=3.5)
    p.add_argument("--target_langs", nargs="*", default=["en", "zh"])
    p.add_argument("--min_lang_prob", type=float, default=0.90)
    p.add_argument("--whisper_batch_size", type=int, default=8)
    p.add_argument("--vllm_batch_size", type=int, default=64,
                   help="funasr_vllm 模式每批送给 vllm 的音频数")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--queue_maxsize", type=int, default=200)
    args = p.parse_args()

    input_root = Path(args.input_root)
    audio_files = collect_audio_files(input_root, args.max_files)
    logger.info(f"Found {len(audio_files)} audio files")

    audio_dir = args.audio_dir or str(Path(args.out_json).parent / "audio")
    gpus = [int(g) for g in args.gpus.split(",")]

    ctx = mp.get_context("spawn")
    Q = lambda: ctx.Queue(maxsize=args.queue_maxsize)
    vad_q = Q()
    dnsmos_q = Q()
    lid_q = Q()
    asr_q = Q()
    result_q = Q()

    file_list = [(make_audio_id(input_root, f), str(f)) for f in audio_files]

    processes = []

    # Producer
    prod = ctx.Process(target=producer, args=(file_list, vad_q), name="Producer")
    processes.append(prod)

    # VAD
    if args.vad_model == "fireredvad":
        vad_p = ctx.Process(
            target=vad_worker_firered,
            args=(vad_q, dnsmos_q, args.min_dur, args.max_dur,
                  args.fireredvad_model, args.fireredvad_root),
            name="VAD-FireRed",
        )
    else:
        vad_p = ctx.Process(
            target=vad_worker_silero,
            args=(vad_q, dnsmos_q, args.min_dur, args.max_dur),
            name="VAD-Silero",
        )
    processes.append(vad_p)

    # DNSMOS
    dnsmos_p = ctx.Process(
        target=dnsmos_worker,
        args=(dnsmos_q, lid_q, args.dnsmos_dir, gpus[0],
              9, args.min_mos_ovr, args.min_mos_sig, args.min_mos_bak),
        name="DNSMOS",
    )
    processes.append(dnsmos_p)

    if args.asr_model == "funasr_vllm":
        # ── 两阶段方案 ─────────────────────────────────────────────────
        # Phase 1: 多进程跑 VAD+DNSMOS+LID，通过的 segment 收集到 collect_q
        # Phase 2: 所有子进程退出后，主进程独占 GPU 跑 vllm
        # ─────────────────────────────────────────────────────────────
        collect_q = ctx.Queue()   # LID 把通过的 segment 写这里
        lid_p = ctx.Process(
            target=lid_worker_collect,
            args=(lid_q, collect_q, args.lid_model_dir, gpus[0],
                  args.target_langs, args.min_lang_prob),
            name="LID",
        )
        processes.append(lid_p)

        t0 = time.time()
        for proc in processes:
            proc.start()
            logger.info(f"Started {proc.name} pid={proc.pid}")

        # Phase 1: 主进程从 collect_q 收集，等 SENTINEL
        segments_phase1 = []
        logger.info("[Phase1] collecting segments from LID...")
        while True:
            item = collect_q.get()
            if item is SENTINEL:
                break
            segments_phase1.append(item)

        logger.info(f"[Phase1] collected {len(segments_phase1)} segments, waiting for all workers to exit...")
        for proc in processes:
            proc.join(timeout=60)
            if proc.is_alive():
                logger.warning(f"{proc.name} still alive, terminating")
                proc.terminate()
                proc.join(timeout=5)

        elapsed_p1 = time.time() - t0
        logger.info(f"[Phase1] done in {elapsed_p1:.1f}s, starting vllm Phase 2...")

        # Phase 2: 全部子进程已退出，主进程安全初始化 vllm
        funasr_vllm_phase2(
            segments=segments_phase1,
            model_dir=args.funasr_model_dir,
            gpu_id=gpus[0],
            batch_size=args.vllm_batch_size,
            out_json=args.out_json,
            audio_dir=audio_dir,
        )

    else:
        # ── 普通模式 ───────────────────────────────────────────────────
        lid_p = ctx.Process(
            target=lid_worker,
            args=(lid_q, asr_q, args.lid_model_dir, gpus[0],
                  args.target_langs, args.min_lang_prob),
            name="LID",
        )
        processes.append(lid_p)

        t0 = time.time()
        for proc in processes:
            proc.start()
            logger.info(f"Started {proc.name} pid={proc.pid}")

        if args.asr_model == "whisper":
            asr_p = ctx.Process(
                target=whisper_asr_worker,
                args=(asr_q, result_q, args.whisper_model_dir, gpus[0],
                      args.whisper_batch_size),
                name="ASR-Whisper",
            )
        else:
            asr_p = ctx.Process(
                target=funasr_asr_worker,
                args=(asr_q, result_q, args.funasr_model_dir, gpus[0]),
                name="ASR-FunASR",
            )
        processes.append(asr_p)
        asr_p.start()
        logger.info(f"Started {asr_p.name} pid={asr_p.pid}")
        export_results(result_q, args.out_json, audio_dir, n_sentinels=1)

        for proc in processes:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()

    logger.info(f"Pipeline done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
