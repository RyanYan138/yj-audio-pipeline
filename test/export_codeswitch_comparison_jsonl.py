#!/usr/bin/env python3
"""Export aligned ASCEND baseline/auto-language comparison JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


EN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
OUTPUT_FILES = {
    "all": "comparison_all_450.jsonl",
    "both_retained": "both_retained_168.jsonl",
    "zh_false_rejected": "zh_false_rejected_53.jsonl",
    "mixed_all": "mixed_all_150.jsonl",
}


def _prediction_text(record: dict) -> str:
    return record.get("transcribe", {}).get("funasr-nano", "").strip()


def _group_predictions(records: Iterable[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for record in records:
        grouped.setdefault(int(record["wav_offset"]), []).append(record)
    for segments in grouped.values():
        segments.sort(key=lambda item: float(item.get("seg_start", 0.0)))
    return grouped


def _segments_view(segments: Sequence[dict]) -> List[dict]:
    return [
        {
            "start_seconds": segment.get("seg_start"),
            "end_seconds": segment.get("seg_end"),
            "duration_seconds": segment.get("duration"),
            "text": _prediction_text(segment),
        }
        for segment in segments
    ]


def _joined_text(segments: Sequence[dict]) -> str:
    return " ".join(
        text for text in (_prediction_text(segment) for segment in segments) if text
    )


def _baseline_view(segments: Sequence[dict]) -> dict:
    if not segments:
        return {
            "retained": False,
            "text": None,
            "segment_count": 0,
            "lid_gate": None,
            "segments": [],
        }
    first = segments[0]
    return {
        "retained": True,
        "text": _joined_text(segments),
        "segment_count": len(segments),
        "lid_gate": {
            "language": first.get("text_lang"),
            "probability": first.get("lang_prob"),
        },
        "segments": _segments_view(segments),
    }


def _aggregate_transcript_language(segments: Sequence[dict]) -> Dict[str, Any]:
    text = _joined_text(segments)
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
        "zh_character_count": zh_chars,
        "english_tokens": en_tokens,
    }


def _auto_view(segments: Sequence[dict]) -> dict:
    if not segments:
        return {
            "retained": False,
            "text": None,
            "segment_count": 0,
            "diagnostic_lid": None,
            "transcript_language": None,
            "segments": [],
        }
    first = segments[0]
    return {
        "retained": True,
        "text": _joined_text(segments),
        "segment_count": len(segments),
        "diagnostic_lid": {
            "language": first.get("lid_lang"),
            "probability": first.get("lid_prob"),
        },
        "transcript_language": _aggregate_transcript_language(segments),
        "segments": _segments_view(segments),
    }


def _old_gate_replay(auto: dict) -> dict:
    lid = auto.get("diagnostic_lid")
    if not lid:
        return {
            "target_language": "zh",
            "minimum_probability": 0.9,
            "evaluable": False,
            "would_pass": False,
            "rejection_reasons": ["no_vad_or_lid_result"],
        }
    language_matches = lid.get("language") == "zh"
    probability = lid.get("probability")
    probability_passes = probability is not None and float(probability) >= 0.9
    reasons = []
    if not language_matches:
        reasons.append("predicted_language_is_not_zh")
    if not probability_passes:
        reasons.append("lid_probability_below_0.90")
    return {
        "target_language": "zh",
        "minimum_probability": 0.9,
        "evaluable": True,
        "language_matches": language_matches,
        "probability_passes": probability_passes,
        "would_pass": language_matches and probability_passes,
        "rejection_reasons": reasons,
    }


def build_comparison_records(
    references: Sequence[dict], baseline_records: Sequence[dict], auto_records: Sequence[dict]
) -> List[dict]:
    baseline_by_offset = _group_predictions(baseline_records)
    auto_by_offset = _group_predictions(auto_records)
    seen_offsets = set()
    output = []

    for reference in references:
        wav_offset = int(reference["wav_offset"])
        if wav_offset in seen_offsets:
            raise ValueError(f"Duplicate reference wav_offset: {wav_offset}")
        seen_offsets.add(wav_offset)

        baseline = _baseline_view(baseline_by_offset.get(wav_offset, []))
        auto = _auto_view(auto_by_offset.get(wav_offset, []))
        if baseline["retained"] and auto["retained"]:
            status = "both_retained"
        elif baseline["retained"]:
            status = "baseline_only"
        elif auto["retained"]:
            status = "auto_only"
        else:
            status = "neither_retained"

        official_group = reference.get("group", "unknown")
        zh_false_rejected = (
            official_group == "zh"
            and not baseline["retained"]
            and auto["retained"]
        )
        mixed_outcome = None
        if official_group == "mixed":
            mixed_outcome = {
                "both_retained": "both_retained",
                "auto_only": "old_false_rejected",
                "baseline_only": "new_missing",
                "neither_retained": "neither_retained",
            }[status]

        views = []
        if status == "both_retained":
            views.append("both_retained")
        if zh_false_rejected:
            views.append("zh_false_rejected")
        if official_group == "mixed":
            views.append("mixed_all")

        output.append(
            {
                "id": str(reference.get("id")),
                "official_group": official_group,
                "reference_text": reference.get("text"),
                "duration_seconds": reference.get("duration"),
                "member_name": reference.get("member_name"),
                "tar_path": reference.get("tar_path"),
                "wav_offset": wav_offset,
                "speaker_id": reference.get("speaker_id"),
                "session_id": reference.get("session_id"),
                "topic": reference.get("topic"),
                "baseline": baseline,
                "auto": auto,
                "comparison": {
                    "status": status,
                    "both_retained": status == "both_retained",
                    "auto_rescued": status == "auto_only",
                    "zh_false_rejected": zh_false_rejected,
                    "mixed_outcome": mixed_outcome,
                    "old_lid_gate_replay": _old_gate_replay(auto),
                    "views": views,
                },
            }
        )
    return output


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_exports(records: Sequence[dict], output_dir: str) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    both = [row for row in records if row["comparison"]["both_retained"]]
    zh_false = [row for row in records if row["comparison"]["zh_false_rejected"]]
    mixed = [row for row in records if row["official_group"] == "mixed"]

    exports = {
        "all": list(records),
        "both_retained": both,
        "zh_false_rejected": zh_false,
        "mixed_all": mixed,
    }
    for name, rows in exports.items():
        _write_jsonl(output / OUTPUT_FILES[name], rows)

    status_counts = Counter(row["comparison"]["status"] for row in records)
    mixed_outcomes = Counter(row["comparison"]["mixed_outcome"] for row in mixed)
    summary = {
        "counts": {
            "all": len(records),
            "both_retained": len(both),
            "zh_false_rejected": len(zh_false),
            "mixed_all": len(mixed),
        },
        "official_group_counts": dict(
            sorted(Counter(row["official_group"] for row in records).items())
        ),
        "retention_status_counts": dict(sorted(status_counts.items())),
        "mixed_outcome_counts": dict(sorted(mixed_outcomes.items())),
        "files": OUTPUT_FILES,
        "field_notes": {
            "official_group": "ASCEND official zh/en/mixed label.",
            "baseline": "Old pipeline output after Whisper Tiny zh>=0.90 gate.",
            "auto": "Nano language=None output; diagnostic_lid never filters ASR.",
            "auto.transcript_language": "Script label derived from Nano transcript characters.",
            "comparison.old_lid_gate_replay": "Replay of the old target=zh, probability>=0.90 decision.",
            "comparison.views": "Convenience slice membership for this record.",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_json(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True)
    parser.add_argument("--baseline-labels", required=True)
    parser.add_argument("--auto-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_comparison_records(
        _load_jsonl(args.references),
        _load_json(args.baseline_labels),
        _load_json(args.auto_labels),
    )
    summary = write_exports(records, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
