#!/usr/bin/env python3

import sys
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_codeswitch_benchmark import select_balanced_rows


class CodeSwitchPreparationTest(unittest.TestCase):
    def test_balanced_selection_is_deterministic_and_duration_bounded(self):
        rows = [
            {"id": "zh-short", "language": "zh", "duration": 0.5},
            {"id": "zh-1", "language": "zh", "duration": 2.0},
            {"id": "zh-2", "language": "zh", "duration": 3.0},
            {"id": "en-1", "language": "en", "duration": 2.0},
            {"id": "en-2", "language": "en", "duration": 3.0},
            {"id": "mix-1", "language": "mixed", "duration": 2.0},
            {"id": "mix-2", "language": "mixed", "duration": 3.0},
        ]

        first = select_balanced_rows(rows, per_group=1, seed=17, min_dur=1.0, max_dur=30.0)
        second = select_balanced_rows(rows, per_group=1, seed=17, min_dur=1.0, max_dur=30.0)

        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(Counter(row["language"] for row in first), {"zh": 1, "en": 1, "mixed": 1})
        self.assertNotIn("zh-short", {row["id"] for row in first})


if __name__ == "__main__":
    unittest.main()
