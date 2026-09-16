#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_codeswitch_comparison_jsonl import build_comparison_records, write_exports


class CodeSwitchComparisonExportTest(unittest.TestCase):
    def setUp(self):
        self.references = [
            {
                "id": "zh-kept",
                "group": "zh",
                "text": "今天开会",
                "duration": 2.0,
                "wav_offset": 100,
                "member_name": "zh/zh-kept.wav",
            },
            {
                "id": "zh-rejected",
                "group": "zh",
                "text": "那你觉得",
                "duration": 1.4,
                "wav_offset": 200,
                "member_name": "zh/zh-rejected.wav",
            },
            {
                "id": "mixed-kept",
                "group": "mixed",
                "text": "我 love AI",
                "duration": 3.0,
                "wav_offset": 300,
                "member_name": "mixed/mixed-kept.wav",
            },
            {
                "id": "mixed-rejected",
                "group": "mixed",
                "text": "用 GPU 训练",
                "duration": 2.5,
                "wav_offset": 400,
                "member_name": "mixed/mixed-rejected.wav",
            },
        ]
        self.baseline = [
            {
                "wav_offset": 100,
                "seg_start": 0.0,
                "seg_end": 2.0,
                "text_lang": "zh",
                "lang_prob": 0.98,
                "transcribe": {"funasr-nano": "今天 开会"},
            },
            {
                "wav_offset": 300,
                "seg_start": 1.0,
                "seg_end": 3.0,
                "text_lang": "zh",
                "lang_prob": 0.95,
                "transcribe": {"funasr-nano": "AI"},
            },
            {
                "wav_offset": 300,
                "seg_start": 0.0,
                "seg_end": 1.0,
                "text_lang": "zh",
                "lang_prob": 0.95,
                "transcribe": {"funasr-nano": "我 love"},
            },
        ]
        self.auto = [
            {
                "wav_offset": 100,
                "seg_start": 0.0,
                "seg_end": 2.0,
                "lid_lang": "zh",
                "lid_prob": 0.98,
                "text_lang": "zh",
                "text_lang_evidence": {"zh_chars": 4, "en_tokens": []},
                "transcribe": {"funasr-nano": "今天开会"},
            },
            {
                "wav_offset": 200,
                "seg_start": 0.0,
                "seg_end": 1.4,
                "lid_lang": "zh",
                "lid_prob": 0.40,
                "text_lang": "zh",
                "text_lang_evidence": {"zh_chars": 4, "en_tokens": []},
                "transcribe": {"funasr-nano": "那你觉得"},
            },
            {
                "wav_offset": 300,
                "seg_start": 0.0,
                "seg_end": 3.0,
                "lid_lang": "zh",
                "lid_prob": 0.95,
                "text_lang": "zh-en",
                "text_lang_evidence": {"zh_chars": 1, "en_tokens": ["love", "AI"]},
                "transcribe": {"funasr-nano": "我 love AI"},
            },
            {
                "wav_offset": 400,
                "seg_start": 0.0,
                "seg_end": 2.5,
                "lid_lang": "en",
                "lid_prob": 0.72,
                "text_lang": "zh-en",
                "text_lang_evidence": {"zh_chars": 3, "en_tokens": ["GPU"]},
                "transcribe": {"funasr-nano": "用 GPU 训练"},
            },
        ]

    def test_builds_statuses_and_orders_segments(self):
        records = build_comparison_records(
            self.references, self.baseline, self.auto
        )
        by_id = {record["id"]: record for record in records}

        self.assertEqual(by_id["zh-kept"]["comparison"]["status"], "both_retained")
        self.assertTrue(
            by_id["zh-rejected"]["comparison"]["zh_false_rejected"]
        )
        self.assertEqual(
            by_id["mixed-rejected"]["comparison"]["mixed_outcome"],
            "old_false_rejected",
        )
        self.assertEqual(by_id["mixed-kept"]["baseline"]["text"], "我 love AI")
        self.assertEqual(
            by_id["zh-rejected"]["auto"]["diagnostic_lid"],
            {"language": "zh", "probability": 0.4},
        )

    def test_writes_master_slices_and_summary(self):
        records = build_comparison_records(
            self.references, self.baseline, self.auto
        )
        with tempfile.TemporaryDirectory() as output_dir:
            summary = write_exports(records, output_dir)
            output = Path(output_dir)

            self.assertEqual(summary["counts"]["all"], 4)
            self.assertEqual(summary["counts"]["both_retained"], 2)
            self.assertEqual(summary["counts"]["zh_false_rejected"], 1)
            self.assertEqual(summary["counts"]["mixed_all"], 2)
            self.assertEqual(
                len((output / "comparison_all_450.jsonl").read_text().splitlines()),
                4,
            )
            self.assertTrue((output / "both_retained_168.jsonl").is_file())
            self.assertTrue((output / "zh_false_rejected_53.jsonl").is_file())
            self.assertTrue((output / "mixed_all_150.jsonl").is_file())
            saved_summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(saved_summary["counts"], summary["counts"])


if __name__ == "__main__":
    unittest.main()
