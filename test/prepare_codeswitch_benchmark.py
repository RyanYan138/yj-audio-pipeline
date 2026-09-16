#!/usr/bin/env python3
"""Build a balanced TAR benchmark directly from ASCEND Parquet shards."""

from __future__ import annotations

import argparse
import io
import json
import random
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence


GROUPS = ("zh", "en", "mixed")


def select_balanced_rows(
    rows: Sequence[dict],
    per_group: int,
    seed: int,
    min_dur: float,
    max_dur: float,
) -> List[dict]:
    rng = random.Random(seed)
    selected = []
    for group in GROUPS:
        candidates = [
            row
            for row in rows
            if row.get("language") == group
            and min_dur <= float(row.get("duration", 0.0)) <= max_dur
        ]
        if len(candidates) < per_group:
            raise ValueError(
                f"ASCEND group {group!r} has {len(candidates)} eligible rows, "
                f"fewer than requested {per_group}"
            )
        selected.extend(rng.sample(candidates, per_group))
    rng.shuffle(selected)
    return selected


def load_parquet_rows(paths: Sequence[str]) -> List[dict]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read ASCEND Parquet files"
        ) from exc
    columns = [
        "id",
        "audio",
        "transcription",
        "duration",
        "language",
        "original_speaker_id",
        "session_id",
        "topic",
    ]
    rows = []
    for path in paths:
        rows.extend(parquet.read_table(path, columns=columns).to_pylist())
    return rows


def _member_name(row: dict, index: int) -> str:
    audio_path = row.get("audio", {}).get("path") or f"{row['id']}.wav"
    base_name = Path(audio_path).name
    return f"{row['language']}/{index:04d}_{row['id']}_{base_name}"


def write_benchmark(
    rows: Sequence[dict],
    output_dir: str,
    source_shards: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tar_path = (output / "audio.tar").resolve()
    names: List[str] = []
    with tarfile.open(tar_path, "w", format=tarfile.GNU_FORMAT) as archive:
        for index, row in enumerate(rows):
            payload = row.get("audio", {}).get("bytes")
            if not payload:
                raise ValueError(f"ASCEND row {row.get('id')} has no audio bytes")
            name = _member_name(row, index)
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
            names.append(name)

    with tarfile.open(tar_path, "r") as archive:
        offsets = {member.name: member.offset_data for member in archive.getmembers()}

    reference_path = output / "references.jsonl"
    references = []
    with open(reference_path, "w", encoding="utf-8") as handle:
        for row, name in zip(rows, names):
            reference = {
                "id": str(row["id"]),
                "group": row["language"],
                "text": row["transcription"],
                "duration": float(row["duration"]),
                "member_name": name,
                "tar_path": str(tar_path),
                "wav_offset": offsets[name],
                "speaker_id": row.get("original_speaker_id"),
                "session_id": row.get("session_id"),
                "topic": row.get("topic"),
            }
            references.append(reference)
            handle.write(json.dumps(reference, ensure_ascii=False) + "\n")

    group_counts = Counter(row["language"] for row in rows)
    group_seconds: Dict[str, float] = {}
    for row in rows:
        group = row["language"]
        group_seconds[group] = group_seconds.get(group, 0.0) + float(row["duration"])
    manifest = {
        "dataset": "CAiRE/ASCEND",
        "dataset_url": "https://huggingface.co/datasets/CAiRE/ASCEND",
        "license": "CC-BY-SA-4.0",
        "source_shards": [str(Path(path).resolve()) for path in source_shards],
        "seed": seed,
        "utterances": len(rows),
        "audio_seconds": sum(float(row["duration"]) for row in rows),
        "group_counts": dict(group_counts),
        "group_audio_seconds": group_seconds,
        "tar_path": str(tar_path),
        "references": str(reference_path.resolve()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-group", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--min-dur", type=float, default=1.0)
    parser.add_argument("--max-dur", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_parquet_rows(args.parquet)
    selected = select_balanced_rows(
        rows,
        per_group=args.per_group,
        seed=args.seed,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
    )
    manifest = write_benchmark(selected, args.output_dir, args.parquet, args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
