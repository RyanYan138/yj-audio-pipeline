#!/usr/bin/env python3

import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.tar_pipeline_nodnsmos_nano_auto_lang_ckpt import (
    annotate_transcription_record,
    parse_args,
    resolve_all_out_json,
    validate_runtime_args,
    write_post_filtered_output,
)


class NanoAutoLanguageCliTest(unittest.TestCase):
    def test_completion_log_does_not_assume_a_cold_model(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "pipeline/tar_pipeline_nodnsmos_nano_auto_lang_ckpt.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("cold end-to-end", source)

    def test_complete_output_is_separate_from_filtered_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "labels.json"
            all_out_json = resolve_all_out_json(str(out_json), None)
            Path(all_out_json).write_text(
                json.dumps(
                    [
                        {"text_lang": "zh", "duration": 1.0},
                        {"text_lang": "zh-en", "duration": 2.0},
                        {"text_lang": "en", "duration": 3.0},
                    ]
                ),
                encoding="utf-8",
            )

            summary = write_post_filtered_output(
                all_out_json, str(out_json), ["zh", "zh-en"]
            )

            self.assertEqual(Path(all_out_json).name, "labels.all.json")
            self.assertEqual(len(json.loads(out_json.read_text())), 2)
            self.assertEqual(summary["all_asr_segments"], 3)
            self.assertEqual(summary["post_kept_segments"], 2)
            self.assertEqual(summary["post_kept_audio_seconds"], 3.0)

    def test_cli_defaults_keep_chinese_and_mixed_after_asr(self):
        argv = [
            "pipeline.py",
            "--tar_paths",
            "input.tar",
            "--out_json",
            "labels.json",
            "--funasr_model_dir",
            "nano",
            "--lid_model_dir",
            "lid",
            "--fireredvad_model",
            "vad",
            "--fireredvad_root",
            "vad-root",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.post_keep_langs, ["zh", "zh-en"])
        self.assertEqual(args.asr_dtype, "fp32")
        self.assertEqual(args.vad_workers, 1)
        self.assertEqual(args.lid_mode, "metadata")
        self.assertEqual(args.all_out_json, None)
        self.assertFalse(hasattr(args, "target_langs"))
        self.assertFalse(hasattr(args, "min_lang_prob"))
        self.assertFalse(hasattr(args, "force_language_from_lid"))

    def test_cli_accepts_parallel_vad_workers(self):
        argv = [
            "pipeline.py",
            "--tar_paths",
            "input.tar",
            "--out_json",
            "labels.json",
            "--funasr_model_dir",
            "nano",
            "--fireredvad_model",
            "vad",
            "--fireredvad_root",
            "vad-root",
            "--vad_workers",
            "4",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.vad_workers, 4)

    def test_cli_can_disable_diagnostic_lid(self):
        argv = [
            "pipeline.py",
            "--tar_paths",
            "input.tar",
            "--out_json",
            "labels.json",
            "--funasr_model_dir",
            "nano",
            "--fireredvad_model",
            "vad",
            "--fireredvad_root",
            "vad-root",
            "--lid_mode",
            "off",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.lid_mode, "off")
        self.assertIsNone(args.lid_model_dir)

    def test_metadata_lid_requires_a_model_directory(self):
        argv = [
            "pipeline.py",
            "--tar_paths",
            "input.tar",
            "--out_json",
            "labels.json",
            "--funasr_model_dir",
            "nano",
            "--fireredvad_model",
            "vad",
            "--fireredvad_root",
            "vad-root",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        with self.assertRaisesRegex(ValueError, "--lid_model_dir"):
            validate_runtime_args(args)

    def test_output_record_keeps_lid_and_classifies_transcript(self):
        segment = {
            "tar_path": "input.tar",
            "wav_uuid": "source-uuid",
            "wav_offset": 512,
            "wav_size": 1024,
            "seg_start": 0.2,
            "seg_end": 2.2,
            "lid_lang": "zh",
            "lid_prob": 0.62,
        }

        record = annotate_transcription_record(
            segment=segment,
            result={"text": "这个 feature 很重要", "timestamps": []},
            segment_uuid="segment-uuid",
            kept_wav=None,
        )

        self.assertEqual(record["lid_lang"], "zh")
        self.assertEqual(record["lid_prob"], 0.62)
        self.assertEqual(record["text_lang"], "zh-en")
        self.assertEqual(record["transcribe"]["funasr-nano"], "这个 feature 很重要")
        self.assertEqual(record["text_lang_evidence"]["en_tokens"], ["feature"])


if __name__ == "__main__":
    unittest.main()
