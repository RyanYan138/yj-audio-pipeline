#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tar包音频读取工具
支持本地tar和S3 tar（流式读取）
记录每条wav的offset/size，用于输出JSON索引
"""

import hashlib
import io
import os
import tarfile
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000


@dataclass
class TarWavEntry:
    tar_path: str
    wav_uuid: str
    wav_offset: int
    wav_size: int
    audio: np.ndarray  # 16kHz float32
    sample_rate: int
    duration: float
    num_sample: int


def _make_uuid(tar_path: str, member_name: str) -> str:
    """根据tar路径+文件名生成确定性uuid"""
    raw = f"{tar_path}::{member_name}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """从wav字节解码音频，返回(audio_float32, sample_rate)"""
    with sf.SoundFile(io.BytesIO(data)) as f:
        sr = f.samplerate
        audio = f.read(dtype="float32", always_2d=False)
    # 合并多声道为单声道，防止 faster-whisper MemoryError
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != SAMPLE_RATE:
        import torch, torchaudio
        t = torch.from_numpy(audio).unsqueeze(0)
        audio = torchaudio.functional.resample(t, sr, SAMPLE_RATE).squeeze(0).numpy()
        sr = SAMPLE_RATE
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    return audio, sr


def iter_tar_wavs(tar_path: str) -> Iterator[TarWavEntry]:
    """
    遍历tar包里所有wav/flac文件，yield TarWavEntry
    tar_path: 本地路径，如 /data/shards_000.tar
    """
    audio_exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            ext = os.path.splitext(member.name)[1].lower()
            if ext not in audio_exts:
                continue
            if member.size == 0:
                continue

            wav_offset = member.offset_data
            wav_size = member.size
            wav_uuid = _make_uuid(tar_path, member.name)

            f = tf.extractfile(member)
            if f is None:
                continue
            data = f.read()

            try:
                audio, sr = _decode_audio(data)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"[TarReader] skip {member.name}: {e}"
                )
                continue

            duration = len(audio) / SAMPLE_RATE
            num_sample = len(audio)

            yield TarWavEntry(
                tar_path=tar_path,
                wav_uuid=wav_uuid,
                wav_offset=wav_offset,
                wav_size=wav_size,
                audio=audio,
                sample_rate=sr,
                duration=duration,
                num_sample=num_sample,
            )


def make_seg_uuid(orig_uuid: str, start_sec: float) -> str:
    """根据原始uuid+起始时间生成切片uuid，保证唯一且可复现"""
    raw = f"{orig_uuid}::{start_sec:.4f}"
    return hashlib.sha256(raw.encode()).hexdigest()
