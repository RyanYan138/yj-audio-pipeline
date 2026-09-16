#!/usr/bin/env python3
"""Compare FireRedVAD timestamps from a WAV path and in-memory PCM."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "FireRedVAD"))

from fireredvad.vad import FireRedVad, FireRedVadConfig
from pipeline.tar_reader import iter_tar_wavs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", required=True)
    parser.add_argument("--model-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entry = next(iter_tar_wavs(args.tar))
    config = FireRedVadConfig(use_gpu=torch.cuda.is_available())
    model = FireRedVad.from_pretrained(args.model_dir, config)

    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        sf.write(handle.name, entry.audio, 16000)
        path_result, _ = model.detect(handle.name)

    pcm16 = np.clip(entry.audio * 32768.0, -32768, 32767).astype(np.int16)
    memory_result, _ = model.detect(pcm16)
    path_timestamps = path_result.get("timestamps", [])
    memory_timestamps = memory_result.get("timestamps", [])
    if path_timestamps != memory_timestamps:
        raise AssertionError(
            f"VAD timestamp mismatch: {path_timestamps!r} != {memory_timestamps!r}"
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "wav_uuid": entry.wav_uuid,
                "timestamps": memory_timestamps,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
