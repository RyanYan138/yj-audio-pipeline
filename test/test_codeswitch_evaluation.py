#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_codeswitch_ab import (
    english_token_recall,
    evaluate_rows,
    levenshtein_distance,
    mixed_tokens,
)


class CodeSwitchEvaluationTest(unittest.TestCase):
    def test_mixed_tokens_use_chinese_characters_and_english_words(self):
        self.assertEqual(
            mixed_tokens("我们 Love AI, 很 useful!"),
            ["我", "们", "love", "ai", "很", "useful"],
        )

    def test_levenshtein_distance_counts_token_edits(self):
        self.assertEqual(
            levenshtein_distance(["我", "love", "ai"], ["我", "like", "ai"]),
            1,
        )

    def test_english_token_recall_uses_reference_occurrences(self):
        self.assertEqual(
            english_token_recall("我爱 machine learning machine", "我爱 machine"),
            1 / 3,
        )

    def test_evaluation_reports_mixed_audio_rescued_after_lid_gate(self):
        references = [
            {
                "id": "mixed-1",
                "wav_offset": 512,
                "group": "mixed",
                "text": "我们 love AI",
                "duration": 2.0,
            },
            {
                "id": "zh-1",
                "wav_offset": 1024,
                "group": "zh",
                "text": "今天开会",
                "duration": 1.5,
            },
        ]
        baseline = [
            {
                "wav_offset": 1024,
                "seg_start": 0.0,
                "transcribe": {"funasr-nano": "今天开会"},
            }
        ]
        auto = [
            {
                "wav_offset": 512,
                "seg_start": 0.0,
                "lid_lang": "en",
                "lid_prob": 0.72,
                "transcribe": {"funasr-nano": "我们 love AI"},
            },
            {
                "wav_offset": 1024,
                "seg_start": 0.0,
                "lid_lang": "zh",
                "lid_prob": 0.99,
                "transcribe": {"funasr-nano": "今天开会"},
            },
        ]

        report = evaluate_rows(references, baseline, auto)

        mixed = report["groups"]["mixed"]
        self.assertEqual(mixed["input_utterances"], 1)
        self.assertEqual(mixed["baseline_retained"], 0)
        self.assertEqual(mixed["auto_retained"], 1)
        self.assertEqual(mixed["rescued_after_lid_gate"], 1)
        self.assertEqual(mixed["auto_mer"], 0.0)
        self.assertEqual(mixed["auto_english_token_recall"], 1.0)

    def test_paired_mer_uses_only_utterances_retained_by_both_paths(self):
        references = [
            {"id": "kept", "wav_offset": 1, "group": "mixed", "text": "我 love AI"},
            {"id": "rescued", "wav_offset": 2, "group": "mixed", "text": "用 GPU"},
        ]
        baseline = [
            {"wav_offset": 1, "seg_start": 0.0, "transcribe": {"funasr-nano": "我 love AI"}}
        ]
        auto = [
            {"wav_offset": 1, "seg_start": 0.0, "transcribe": {"funasr-nano": "我 love AI"}},
            {"wav_offset": 2, "seg_start": 0.0, "transcribe": {"funasr-nano": "完全错误"}},
        ]

        mixed = evaluate_rows(references, baseline, auto)["groups"]["mixed"]

        self.assertEqual(mixed["paired_utterances"], 1)
        self.assertEqual(mixed["baseline_paired_mer"], 0.0)
        self.assertEqual(mixed["auto_paired_mer"], 0.0)


if __name__ == "__main__":
    unittest.main()
