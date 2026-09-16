#!/usr/bin/env python3
"""Compare pre-LID filtering with Nano auto-language transcription."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?")
EN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def mixed_tokens(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def english_tokens(text: str) -> List[str]:
    return [token.lower() for token in EN_RE.findall(text)]


def levenshtein_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + int(ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def english_token_recall(reference: str, hypothesis: str) -> Optional[float]:
    ref_counts = Counter(english_tokens(reference))
    if not ref_counts:
        return None
    hyp_counts = Counter(english_tokens(hypothesis))
    matched = sum(min(count, hyp_counts[token]) for token, count in ref_counts.items())
    return matched / sum(ref_counts.values())


def _prediction_index(records: Iterable[dict]) -> Dict[int, dict]:
    grouped: Dict[int, List[dict]] = {}
    for record in records:
        grouped.setdefault(int(record["wav_offset"]), []).append(record)
    indexed = {}
    for wav_offset, segments in grouped.items():
        segments.sort(key=lambda item: float(item.get("seg_start", 0.0)))
        texts = []
        for segment in segments:
            text = segment.get("transcribe", {}).get("funasr-nano", "").strip()
            if text:
                texts.append(text)
        indexed[wav_offset] = {
            "text": " ".join(texts),
            "segments": segments,
            "lid_lang": segments[0].get("lid_lang", segments[0].get("text_lang")),
            "lid_prob": segments[0].get("lid_prob", segments[0].get("lang_prob")),
        }
    return indexed


def _group_metrics(
    references: Sequence[dict],
    baseline_index: Dict[int, dict],
    auto_index: Dict[int, dict],
) -> Dict[str, Any]:
    baseline_edits = baseline_ref_tokens = 0
    auto_edits = auto_ref_tokens = 0
    baseline_paired_edits = auto_paired_edits = paired_ref_tokens = 0
    paired_utterances = 0
    auto_en_matched = auto_en_total = 0
    baseline_retained = auto_retained = rescued = 0
    low_confidence_auto = 0

    for reference in references:
        offset = int(reference["wav_offset"])
        ref_text = reference["text"]
        ref_tokens = mixed_tokens(ref_text)
        baseline = baseline_index.get(offset)
        auto = auto_index.get(offset)
        if baseline:
            baseline_retained += 1
            baseline_edits += levenshtein_distance(
                ref_tokens, mixed_tokens(baseline["text"])
            )
            baseline_ref_tokens += len(ref_tokens)
        if auto:
            auto_retained += 1
            auto_edits += levenshtein_distance(ref_tokens, mixed_tokens(auto["text"]))
            auto_ref_tokens += len(ref_tokens)
            ref_en = Counter(english_tokens(ref_text))
            hyp_en = Counter(english_tokens(auto["text"]))
            auto_en_matched += sum(
                min(count, hyp_en[token]) for token, count in ref_en.items()
            )
            auto_en_total += sum(ref_en.values())
            probability = auto.get("lid_prob")
            low_confidence_auto += int(
                probability is not None and float(probability) < 0.9
            )
        if baseline and auto:
            paired_utterances += 1
            paired_ref_tokens += len(ref_tokens)
            baseline_paired_edits += levenshtein_distance(
                ref_tokens, mixed_tokens(baseline["text"])
            )
            auto_paired_edits += levenshtein_distance(
                ref_tokens, mixed_tokens(auto["text"])
            )
        rescued += int(baseline is None and auto is not None)

    total = len(references)
    return {
        "input_utterances": total,
        "input_audio_seconds": sum(float(row.get("duration", 0.0)) for row in references),
        "baseline_retained": baseline_retained,
        "baseline_retention_rate": baseline_retained / total if total else 0.0,
        "auto_retained": auto_retained,
        "auto_retention_rate": auto_retained / total if total else 0.0,
        "rescued_after_lid_gate": rescued,
        "auto_low_lid_confidence": low_confidence_auto,
        "baseline_mer": (
            baseline_edits / baseline_ref_tokens if baseline_ref_tokens else None
        ),
        "auto_mer": auto_edits / auto_ref_tokens if auto_ref_tokens else None,
        "paired_utterances": paired_utterances,
        "baseline_paired_mer": (
            baseline_paired_edits / paired_ref_tokens if paired_ref_tokens else None
        ),
        "auto_paired_mer": (
            auto_paired_edits / paired_ref_tokens if paired_ref_tokens else None
        ),
        "auto_english_token_recall": (
            auto_en_matched / auto_en_total if auto_en_total else None
        ),
    }


def evaluate_rows(
    references: Sequence[dict], baseline: Sequence[dict], auto: Sequence[dict]
) -> Dict[str, Any]:
    baseline_index = _prediction_index(baseline)
    auto_index = _prediction_index(auto)
    groups: Dict[str, List[dict]] = {}
    for reference in references:
        groups.setdefault(reference.get("group", "unknown"), []).append(reference)

    rescued_examples = []
    for reference in references:
        offset = int(reference["wav_offset"])
        if offset not in baseline_index and offset in auto_index:
            prediction = auto_index[offset]
            rescued_examples.append(
                {
                    "id": reference.get("id"),
                    "group": reference.get("group"),
                    "reference": reference.get("text"),
                    "auto_text": prediction["text"],
                    "lid_lang": prediction.get("lid_lang"),
                    "lid_prob": prediction.get("lid_prob"),
                }
            )

    return {
        "groups": {
            group: _group_metrics(rows, baseline_index, auto_index)
            for group, rows in sorted(groups.items())
        },
        "overall": _group_metrics(references, baseline_index, auto_index),
        "rescued_examples": rescued_examples[:30],
    }


def _load_references(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_json(path: Optional[str]) -> Any:
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _format_rate(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Mandarin-English Code-Switch A/B Report",
        "",
        "| Group | Input | Baseline kept | Auto kept | Rescued | Baseline MER | Auto MER | Paired B/A MER | Auto EN recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, metrics in report["groups"].items():
        lines.append(
            f"| {group} | {metrics['input_utterances']} | "
            f"{_format_rate(metrics['baseline_retention_rate'])} | "
            f"{_format_rate(metrics['auto_retention_rate'])} | "
            f"{metrics['rescued_after_lid_gate']} | "
            f"{_format_rate(metrics['baseline_mer'])} | "
            f"{_format_rate(metrics['auto_mer'])} | "
            f"{_format_rate(metrics['baseline_paired_mer'])} / "
            f"{_format_rate(metrics['auto_paired_mer'])} | "
            f"{_format_rate(metrics['auto_english_token_recall'])} |"
        )
    lines.extend(["", "## Rescued Mixed Examples", ""])
    examples = [
        item for item in report["rescued_examples"] if item.get("group") == "mixed"
    ]
    if not examples:
        lines.append("No mixed utterance was rescued in this sample.")
    for item in examples[:10]:
        lines.extend(
            [
                f"- `{item['id']}` LID={item.get('lid_lang')} "
                f"({item.get('lid_prob')})",
                f"  Reference: {item.get('reference', '')}",
                f"  Nano auto: {item.get('auto_text', '')}",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True)
    parser.add_argument("--baseline-labels", required=True)
    parser.add_argument("--auto-labels", required=True)
    parser.add_argument("--baseline-metrics")
    parser.add_argument("--auto-metrics")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_rows(
        _load_references(args.references),
        _load_json(args.baseline_labels),
        _load_json(args.auto_labels),
    )
    report["pipeline_metrics"] = {
        "baseline": _load_json(args.baseline_metrics),
        "auto": _load_json(args.auto_metrics),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
