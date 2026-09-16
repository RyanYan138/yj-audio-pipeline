#!/usr/bin/env python3

import unittest
import queue
import json
import tempfile
from pathlib import Path

import numpy as np

from pipeline.tar_pipeline_nodnsmos_nano_auto_lang_ckpt import (
    AutoLanguageStreamingDurationBuckets,
    annotate_lid_segments,
    classify_transcript_language,
    _collect_stage_stats,
    atomic_write_json,
    detect_lid_batch_or_unknown,
    emit_lid_records,
    lid_passthrough_worker,
    load_resume_records,
    post_filter_records,
    validate_batch_result_count,
    validate_server_dtype,
    validate_server_model,
)


class NanoAutoLanguagePolicyTest(unittest.TestCase):
    def test_lid_annotation_never_rejects_low_confidence_english(self):
        item = {
            "audio": np.ones(32000, dtype=np.float32),
            "segments": [{"seg_start": 0.0, "seg_end": 1.0}],
            "wav_uuid": "u",
        }

        records = annotate_lid_segments(item, [("<|en|>", 0.31)])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["lid_lang"], "en")
        self.assertEqual(records[0]["lid_prob"], 0.31)

    def test_lid_annotation_marks_unknown_without_rejecting(self):
        item = {
            "audio": np.ones(16000, dtype=np.float32),
            "segments": [{"seg_start": 0.0, "seg_end": 1.0}],
            "wav_uuid": "u",
        }

        records = annotate_lid_segments(item, [])

        self.assertEqual(records[0]["lid_lang"], "unknown")
        self.assertEqual(records[0]["lid_prob"], 0.0)

    def test_transcript_language_detects_intra_sentence_code_switch(self):
        result = classify_transcript_language("我们下午有一个 project meeting")

        self.assertEqual(result["language"], "zh-en")
        self.assertGreater(result["zh_chars"], 0)
        self.assertEqual(result["en_tokens"], ["project", "meeting"])

    def test_transcript_language_distinguishes_monolingual_text(self):
        self.assertEqual(classify_transcript_language("今天下午开会")["language"], "zh")
        self.assertEqual(classify_transcript_language("project meeting")["language"], "en")
        self.assertEqual(classify_transcript_language("123 ...")["language"], "other")

    def test_auto_bucket_mixes_lid_predictions(self):
        buckets = AutoLanguageStreamingDurationBuckets(2, [2, 4], 1)
        self.assertIsNone(
            buckets.add(
                {"seg_start": 0.0, "seg_end": 1.0, "lid_lang": "zh"}
            )
        )

        batch = buckets.add(
            {"seg_start": 0.0, "seg_end": 1.1, "lid_lang": "en"}
        )

        self.assertEqual(len(batch), 2)
        self.assertEqual({item["lid_lang"] for item in batch}, {"zh", "en"})

    def test_post_filter_defaults_to_chinese_and_mixed(self):
        records = [
            {"text_lang": "zh"},
            {"text_lang": "zh-en"},
            {"text_lang": "en"},
        ]

        kept = post_filter_records(records, ["zh", "zh-en"])

        self.assertEqual(kept, records[:2])

    def test_stage_stats_merge_parallel_vad_workers(self):
        stats_queue = queue.Queue()
        stats_queue.put(
            {
                "stage": "vad",
                "worker_index": 0,
                "source_wavs": 40,
                "vad_segments": 42,
                "seconds": 5.0,
            }
        )
        stats_queue.put(
            {
                "stage": "vad",
                "worker_index": 1,
                "source_wavs": 60,
                "vad_segments": 63,
                "seconds": 4.5,
            }
        )

        stats = _collect_stage_stats(stats_queue, expected=2)

        self.assertEqual(stats["vad"]["workers"], 2)
        self.assertEqual(stats["vad"]["source_wavs"], 100)
        self.assertEqual(stats["vad"]["vad_segments"], 105)
        self.assertEqual(stats["vad"]["seconds"], 5.0)

    def test_stage_stats_sum_parallel_vad_failures(self):
        stats_queue = queue.Queue()
        for worker_index, failures in enumerate((2, 3)):
            stats_queue.put(
                {
                    "stage": "vad",
                    "worker_index": worker_index,
                    "source_wavs": 5,
                    "vad_segments": 5,
                    "vad_failed_wavs": failures,
                    "seconds": 1.0,
                }
            )

        stats = _collect_stage_stats(stats_queue, expected=2)

        self.assertEqual(stats["vad"]["vad_failed_wavs"], 5)

    def test_lid_off_passthrough_waits_for_all_vad_workers(self):
        input_queue = queue.Queue()
        output_queue = queue.Queue()
        stats_queue = queue.Queue()
        input_queue.put(
            {
                "audio": np.ones(32000, dtype=np.float32),
                "segments": [{"seg_start": 0.0, "seg_end": 1.0}],
                "wav_uuid": "u",
            }
        )
        input_queue.put(None)
        input_queue.put(None)

        lid_passthrough_worker(
            input_queue, output_queue, stats_queue, upstream_sentinels=2
        )

        record = output_queue.get_nowait()
        self.assertEqual(record["lid_lang"], "unknown")
        self.assertIsNone(output_queue.get_nowait())
        self.assertEqual(stats_queue.get_nowait()["lid_mode"], "off")

    def test_lid_fallback_emits_unknown_records(self):
        output_queue = queue.Queue()
        item = {
            "audio": np.ones(32000, dtype=np.float32),
            "segments": [{"seg_start": 0.0, "seg_end": 1.0}],
            "wav_uuid": "u",
        }

        emitted = emit_lid_records(item, [], output_queue)

        self.assertEqual(emitted, 1)
        self.assertEqual(output_queue.get_nowait()["lid_lang"], "unknown")

    def test_lid_batch_feature_failure_marks_every_item_unknown(self):
        class BrokenFeatureExtractor:
            def __call__(self, audios):
                raise RuntimeError("feature failed")

        results, _, _, error = detect_lid_batch_or_unknown(
            BrokenFeatureExtractor(), object(), object(), [np.ones(4), np.ones(8)]
        )

        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(results, [[("<|unknown|>", 0.0)]] * 2)

    def test_lid_batch_result_count_mismatch_marks_every_item_unknown(self):
        class FeatureExtractor:
            def __call__(self, audios):
                return np.ones((len(audios), 2, 2), dtype=np.float32)

        class StorageView:
            @staticmethod
            def from_array(value):
                return value

        class Model:
            def detect_language(self, features):
                return [[("<|zh|>", 0.9)]]

        results, _, _, error = detect_lid_batch_or_unknown(
            FeatureExtractor(), Model(), StorageView, [np.ones(4), np.ones(8)]
        )

        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(results, [[("<|unknown|>", 0.0)]] * 2)

    def test_corrupt_final_json_falls_back_to_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            final_path = Path(temp_dir) / "labels.json"
            checkpoint_path = Path(temp_dir) / "checkpoint.jsonl"
            final_path.write_text('[{"wav_uuid": "broken"}', encoding="utf-8")
            checkpoint_path.write_text(
                json.dumps({"wav_uuid": "from-checkpoint", "duration": 1.0})
                + "\n",
                encoding="utf-8",
            )

            records = load_resume_records([str(final_path), str(checkpoint_path)])

            self.assertEqual([row["wav_uuid"] for row in records], ["from-checkpoint"])

    def test_atomic_json_write_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.json"

            atomic_write_json(str(path), [{"wav_uuid": "u"}])

            self.assertEqual(json.loads(path.read_text()), [{"wav_uuid": "u"}])
            self.assertEqual(list(Path(temp_dir).glob("*.tmp.*")), [])

    def test_batch_result_count_mismatch_is_retryable_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "returned 1 results for 2 inputs"):
            validate_batch_result_count([{}], expected=2)

    def test_server_dtype_and_model_metadata_are_required(self):
        self.assertEqual(validate_server_dtype({"dtype": "fp32"}, "fp32"), "fp32")
        with self.assertRaisesRegex(RuntimeError, "missing dtype"):
            validate_server_dtype({}, "fp32")
        with self.assertRaisesRegex(RuntimeError, "dtype mismatch"):
            validate_server_dtype({"dtype": "bf16"}, "fp32")
        with self.assertRaisesRegex(RuntimeError, "missing model_dir"):
            validate_server_model({}, "/models/nano")
        self.assertEqual(
            validate_server_model(
                {"model_dir": "/models/nano"}, "/models/nano"
            ),
            "/models/nano",
        )


if __name__ == "__main__":
    unittest.main()
