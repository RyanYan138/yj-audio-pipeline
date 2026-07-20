#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tar包音频 pipeline — faster-whisper ASR + 断点续跑（106集群版）
架构与 tar_pipeline_ckpt.py 完全一致（Producer-Consumer 五阶段流水线），
Phase2 由 FunASR vLLM 替换为 faster-whisper。

关键差异：
  - Phase2 直接接受 numpy float32 数组，无需写临时 wav 文件
  - 输出 record 的 transcribe/timestamp key 为 "whisper-large-v3"
  - 新增 --language 参数；不指定时自动使用 LID 阶段检测到的 text_lang
  - --whisper_model_dir 同时用于 LID 和 ASR；可用 --lid_model_dir 单独指定 LID 模型
  - batch_size 仅控制 ckpt 刷盘粒度，faster-whisper 本身逐条推理

断点续跑说明:
  Phase1 ckpt: {out_json_dir}/phase1_ckpt.jsonl  (每条segment落盘，不含audio)
  Phase2 ckpt: {out_json_dir}/phase2_ckpt.jsonl  (每batch结果追加落盘)
  加 --resume 参数后:
    - 若 phase1_ckpt.jsonl 存在 → 跳过Phase1直接加载
    - 若 phase2_ckpt.jsonl 存在 → 跳过已完成segment

用法:
  python3 tar_pipeline_whisper_ckpt.py \
    --tar_paths /data/shards_000.tar \
    --out_json  /data/labels.json \
    --whisper_model_dir /Work21/2025/yanjiahao/YJ-audio-pipeline/yj-audio-pipeline/models/faster-whisper-large-v3 \
    --fireredvad_model  /Work21/2025/yanjiahao/YJ-audio-pipeline/yj-audio-pipeline/models/FireRedVAD/VAD \
    --fireredvad_root   /path/to/FireRedVAD \
    --dnsmos_dir        /path/to/dns_mos \
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
WHISPER_MODEL_KEY = "whisper-large-v3"


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_audio_seg(audio: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
    s = int(start_sec * SAMPLE_RATE)
    e = int(end_sec * SAMPLE_RATE)
    return audio[s:e]


def read_audio_from_tar(tar_path: str, wav_offset: int, wav_size: int) -> np.ndarray:
    """从 tar 文件按 offset 重新读取音频，用于 Phase2 resume 时重建 audio 数组"""
    from tar_reader import _decode_audio
    with open(tar_path, "rb") as f:
        f.seek(wav_offset)
        data = f.read(wav_size)
    audio, _ = _decode_audio(data)
    return audio


def _setup_ld_library_path():
    """从 CONDA_PREFIX 动态解析 nvidia/ctranslate2 库路径（兼容任意集群/python版本）"""
    import glob as _glob
    _conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if _conda_prefix:
        for p in (
            _glob.glob(f"{_conda_prefix}/lib/python3.*/site-packages/nvidia/cudnn/lib")
            + _glob.glob(f"{_conda_prefix}/lib/python3.*/site-packages/ctranslate2.libs")
        ):
            if os.path.exists(p):
                existing = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = f"{p}:{existing}" if existing else p


def _load_whisper_model(model_dir: str, device_str: str = "cuda"):
    """
    加载 faster-whisper WhisperModel。
    device_str 支持 "cuda"、"cuda:0"、"cpu" 等格式；
    faster-whisper 不接受 "cuda:N"，需拆分为 device + device_index。
    """
    _setup_ld_library_path()
    from faster_whisper import WhisperModel

    if ":" in device_str:
        device, device_index = device_str.split(":", 1)
        device_index = int(device_index)
    else:
        device, device_index = device_str, 0

    try:
        model = WhisperModel(model_dir, device=device, device_index=device_index,
                             compute_type="float16")
        logger.info(f"[Whisper] ready  device={device}[{device_index}]  compute=float16")
    except Exception as e:
        logger.warning(f"[Whisper] cuda 失败 ({e})，fallback cpu/int8")
        model = WhisperModel(model_dir, device="cpu", device_index=0, compute_type="int8")
        logger.info("[Whisper] ready  device=cpu  compute=int8")
    return model


# ─────────────────────────────────────────────
# Stage 1: Producer — 读 tar 包，放入 vad 队列
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
        logger.error(f"[DNSMOS] 初始化失败: {e}，跳过 DNSMOS，直接透传所有片段")
        # 崩溃时必须把 dnsmos_q 排空并把 SENTINEL 传给 lid_q，否则流水线死锁
        while True:
            item = dnsmos_q.get()
            if item is SENTINEL:
                lid_q.put(SENTINEL)
                break
            lid_q.put(item)  # 透传，不做 DNSMOS 过滤
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
# Stage 4: LID Worker（faster-whisper，只检测语言）
# ─────────────────────────────────────────────

def lid_worker_collect(lid_q: mp.Queue, out_q: mp.Queue,
                       model_dir: str,
                       target_langs: List[str], min_lang_prob: float):
    _setup_ld_library_path()
    from faster_whisper import WhisperModel

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
                # 最多取前12秒做语种检测
                audio = load_audio_seg(item["audio"], item["seg_start"],
                                       min(item["seg_end"], item["seg_start"] + 12.0))
                # transcribe() 返回 (generator, info)；info 含 language/language_probability
                # 丢弃 generator（_）只取 info，无需消费 segments
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
# Stage 5: faster-whisper Phase2（主进程）
# ─────────────────────────────────────────────

def whisper_phase2(segments: list, model_dir: str, gpu_id: int,
                   batch_size: int, out_json: str,
                   audio_dir: Optional[str] = None,
                   force_language: Optional[str] = None,
                   phase2_ckpt_path: Optional[str] = None):
    """
    faster-whisper ASR 阶段。

    与原 funasr_vllm_phase2 的关键差异：
      1. 直接传入 numpy float32 数组，无需写临时 wav 文件
      2. model.transcribe() 返回惰性生成器，必须 list() 强制求值
      3. word_timestamps=True 获取词级时间戳
      4. force_language=None 时，自动使用 seg["text_lang"]（LID结果）
      5. 输出 record 中 key 为 WHISPER_MODEL_KEY（"whisper-large-v3"）
    """
    from tar_reader import make_seg_uuid
    import soundfile as sf

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # CUDA_VISIBLE_DEVICES 已将指定 GPU 映射为 device 0，始终用 cuda:0
    model = _load_whisper_model(model_dir, device_str="cuda:0")

    # ── Resume：加载已完成结果 ──────────────────
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
                    done_uuids.add(r["wav_uuid"])  # r["wav_uuid"] 已是 seg_uuid
                except Exception:
                    pass
        logger.info(f"[Whisper] Resume: {len(done_uuids)} segments already done from ckpt")

    remaining = [seg for seg in segments
                 if make_seg_uuid(seg["wav_uuid"], seg["seg_start"]) not in done_uuids]
    logger.info(f"[Whisper] {len(remaining)} segments to process "
                f"({len(segments) - len(remaining)} skipped by resume)")

    if audio_dir:
        Path(audio_dir).mkdir(parents=True, exist_ok=True)

    ckpt_f = None
    if phase2_ckpt_path:
        Path(phase2_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        ckpt_f = open(phase2_ckpt_path, "a", encoding="utf-8")

    new_results = []
    try:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start: batch_start + batch_size]
            batch_records = []

            for seg in batch:
                text = ""
                timestamps = []
                try:
                    # Phase2 resume 时 audio 为 None，需从 tar 重读
                    if seg.get("audio") is None:
                        seg["audio"] = read_audio_from_tar(
                            seg["tar_path"], seg["wav_offset"], seg["wav_size"])

                    audio = load_audio_seg(seg["audio"], seg["seg_start"], seg["seg_end"])

                    # 可选：保留音频副本
                    if audio_dir:
                        kept_path = os.path.join(
                            audio_dir,
                            f"{seg['wav_uuid'][:16]}_{seg['seg_start']:.3f}.wav")
                        sf.write(kept_path, audio, SAMPLE_RATE)
                        seg["_kept_wav"] = kept_path

                    # 使用 LID 检测到的语言（除非 CLI 强制指定）
                    lang = force_language or seg.get("text_lang") or None

                    # ── 关键：直接传 numpy array，无需临时 wav 文件 ──
                    # transcribe() 返回 (segments_generator, TranscriptionInfo)
                    # segments_generator 是惰性的，必须 list() 强制求值才会真正推理
                    segs_gen, _ = model.transcribe(
                        audio,            # float32 numpy array，16 kHz，单声道
                        beam_size=5,
                        language=lang,
                        word_timestamps=True,   # 获取词级时间戳
                    )
                    segs_list = list(segs_gen)  # 强制求值

                    text = " ".join(s.text for s in segs_list).strip()

                    # 词级时间戳：[[word, start_sec, end_sec], ...]
                    for s in segs_list:
                        if s.words:
                            for w in s.words:
                                tok = w.word.strip()
                                if tok:
                                    timestamps.append(
                                        [tok, round(w.start, 3), round(w.end, 3)])

                except Exception as e:
                    logger.warning(f"[Whisper] skip seg {seg['wav_uuid'][:8]}: {e}")

                seg_dur = round(seg["seg_end"] - seg["seg_start"], 4)
                seg_uuid = make_seg_uuid(seg["wav_uuid"], seg["seg_start"])
                record = {
                    "tar_path":   seg["tar_path"],
                    "wav":        seg.get("_kept_wav"),      # None 若未保留
                    "wav_uuid":   seg_uuid,
                    "wav_offset": seg["wav_offset"],
                    "wav_size":   seg["wav_size"],
                    "seg_start":  seg["seg_start"],
                    "seg_end":    seg["seg_end"],
                    "duration":   seg_dur,
                    "num_sample": int(seg_dur * SAMPLE_RATE),
                    "text_lang":  seg["text_lang"],
                    "transcribe": {WHISPER_MODEL_KEY: text},
                    "timestamp":  {WHISPER_MODEL_KEY: timestamps},
                }
                batch_records.append(record)

            # ── ckpt 刷盘 ──
            if ckpt_f:
                for r in batch_records:
                    ckpt_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                ckpt_f.flush()
            new_results.extend(batch_records)
            done_cnt = min(batch_start + batch_size, len(remaining))
            logger.info(f"[Whisper] {done_cnt}/{len(remaining)} done (this run)")

    finally:
        if ckpt_f:
            ckpt_f.close()

    all_results = existing_results + new_results
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"[Whisper] wrote {len(all_results)} records to {out_json}")

    # 只有真的有结果才删ckpt，避免误删断点
    if phase2_ckpt_path and os.path.exists(phase2_ckpt_path) and len(all_results) > 0:
        try:
            os.remove(phase2_ckpt_path)
            logger.info(f"[Whisper] removed phase2 ckpt {phase2_ckpt_path}")
        except Exception:
            pass


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Tar包音频 pipeline（faster-whisper ASR + 断点续跑）")
    p.add_argument("--tar_paths", nargs="+", required=True)
    p.add_argument("--out_json",  required=True)
    # ── 模型路径 ──
    p.add_argument("--whisper_model_dir", required=True,
                   help="faster-whisper 模型目录（同时用于 LID 和 ASR）")
    p.add_argument("--lid_model_dir", default=None,
                   help="LID 专用模型目录；不指定则复用 whisper_model_dir")
    p.add_argument("--fireredvad_model", required=True)
    p.add_argument("--fireredvad_root",  required=True)
    p.add_argument("--dnsmos_dir",       required=True)
    # ── 运行参数 ──
    p.add_argument("--gpu",        type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32,
                   help="每批 segment 数（影响 ckpt 刷盘粒度，不影响 GPU 利用率）")
    p.add_argument("--language",   default=None,
                   help="强制 ASR 语言（zh/en/...）；None=使用 LID 结果")
    # ── 过滤阈值 ──
    p.add_argument("--min_dur",    type=float, default=1.0)
    p.add_argument("--max_dur",    type=float, default=30.0)
    p.add_argument("--min_mos_ovr", type=float, default=2.0)
    p.add_argument("--min_mos_sig", type=float, default=2.0)
    p.add_argument("--min_mos_bak", type=float, default=3.5)
    p.add_argument("--target_langs", nargs="*", default=["en", "zh"])
    p.add_argument("--min_lang_prob", type=float, default=0.90)
    # ── 输出 ──
    p.add_argument("--audio_dir",  default=None,
                   help="若指定，保留每条 segment 的 wav 副本到该目录")
    p.add_argument("--queue_maxsize",       type=int, default=200)
    p.add_argument("--dnsmos_input_length", type=int, default=9)
    p.add_argument("--resume", action="store_true",
                   help="断点续跑：从 phase1/phase2 ckpt 文件恢复")
    return p.parse_args()


def main():
    args = parse_args()
    lid_model = args.lid_model_dir or args.whisper_model_dir
    out_dir   = str(Path(args.out_json).parent)
    p1_ckpt   = os.path.join(out_dir, "phase1_ckpt.jsonl")
    p2_ckpt   = os.path.join(out_dir, "phase2_ckpt.jsonl")

    t0 = time.time()
    segments = []

    # ── Phase 1: VAD / DNSMOS / LID ─────────────
    if args.resume and os.path.exists(p1_ckpt):
        logger.info(f"[Resume] Loading Phase1 from {p1_ckpt}")
        with open(p1_ckpt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    seg = json.loads(line)
                    seg["audio"] = None   # Phase2 会从 tar 重读
                    segments.append(seg)
                except Exception:
                    pass
        logger.info(f"[Resume] Loaded {len(segments)} segments, skipping Phase1")
    else:
        ctx = mp.get_context("spawn")
        vad_q     = ctx.Queue(maxsize=args.queue_maxsize)
        dnsmos_q  = ctx.Queue(maxsize=args.queue_maxsize)
        lid_q     = ctx.Queue(maxsize=args.queue_maxsize)
        collect_q = ctx.Queue(maxsize=args.queue_maxsize)

        prod_p = ctx.Process(target=producer,
                             args=(args.tar_paths, vad_q), name="Producer")
        vad_p  = ctx.Process(target=vad_worker,
                             args=(vad_q, dnsmos_q, args.min_dur, args.max_dur,
                                   args.fireredvad_model, args.fireredvad_root), name="VAD")
        dns_p  = ctx.Process(target=dnsmos_worker,
                             args=(dnsmos_q, lid_q, args.dnsmos_dir, args.gpu,
                                   args.dnsmos_input_length,
                                   args.min_mos_ovr, args.min_mos_sig, args.min_mos_bak),
                             name="DNSMOS")
        lid_p  = ctx.Process(target=lid_worker_collect,
                             args=(lid_q, collect_q, lid_model,
                                   args.target_langs, args.min_lang_prob), name="LID")

        logger.info("=== Phase 1: VAD / DNSMOS / LID ===")
        for p in [prod_p, vad_p, dns_p, lid_p]:
            p.start()

        # Watchdog: 若 dns_p 异常退出，把 dnsmos_q 剩余数据透传给 lid_q
        # 注意：不能立刻塞 SENTINEL，因为 VAD 可能还在往 dnsmos_q 里放数据
        import threading
        def _dns_watchdog():
            dns_p.join()  # 等 dns_p 结束（正常或崩溃）
            if dns_p.exitcode != 0:
                logger.warning(f"[DNSMOS] 进程异常退出(exitcode={dns_p.exitcode})，"
                                "启动透传模式：dnsmos_q → lid_q（跳过音质过滤）")
                # 把 dnsmos_q 的剩余数据全部透传给 lid_q，保持流水线畅通
                while True:
                    item = dnsmos_q.get()
                    lid_q.put(item)  # SENTINEL 也会被转发，从而关闭 LID
                    if item is SENTINEL:
                        break
        _wdog = threading.Thread(target=_dns_watchdog, daemon=True)
        _wdog.start()

        Path(p1_ckpt).parent.mkdir(parents=True, exist_ok=True)
        with open(p1_ckpt, "w", encoding="utf-8") as ckpt_f:
            while True:
                item = collect_q.get()
                if item is SENTINEL:
                    break
                segments.append(item)
                # phase1 ckpt 不保存 audio（numpy array 无法 JSON 序列化）
                meta = {k: v for k, v in item.items() if k != "audio"}
                ckpt_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                ckpt_f.flush()

        for p in [prod_p, vad_p, dns_p, lid_p]:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate(); p.join(timeout=5)

        logger.info(f"Phase 1 done: {len(segments)} segments in {time.time()-t0:.1f}s")

    # ── Phase 2: faster-whisper ASR ─────────────
    logger.info("=== Phase 2: faster-whisper ASR ===")
    whisper_phase2(
        segments=segments,
        model_dir=args.whisper_model_dir,
        gpu_id=args.gpu,
        batch_size=args.batch_size,
        out_json=args.out_json,
        audio_dir=args.audio_dir,
        force_language=args.language,
        phase2_ckpt_path=p2_ckpt,
    )
    logger.info(f"完成！总耗时 {time.time()-t0:.1f}s  输出: {args.out_json}")


if __name__ == "__main__":
    main()
