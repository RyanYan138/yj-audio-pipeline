#!/usr/bin/env python3

import sys
import tempfile
import time
import unittest
from pathlib import Path
from multiprocessing.connection import Listener
from unittest.mock import MagicMock, patch

from asr.funasr_nano_batch_server_tuned import (
    ensure_socket_available,
    parse_args,
    pin_owned_socket,
    release_cuda_cache,
    unlink_owned_socket,
)


class TunedServerCliTest(unittest.TestCase):
    def test_server_accepts_bf16_without_changing_default(self):
        with patch.object(
            sys,
            "argv",
            [
                "server.py",
                "--model-dir",
                "nano",
                "--socket",
                "/tmp/nano.sock",
                "--dtype",
                "bf16",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.dtype, "bf16")
        self.assertEqual(args.device, "cuda:0")

    def test_server_releases_cached_cuda_memory_between_clients(self):
        torch_module = MagicMock()
        torch_module.cuda.is_available.return_value = True

        release_cuda_cache(torch_module)

        torch_module.cuda.empty_cache.assert_called_once_with()

    def test_server_refuses_to_replace_an_active_socket(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "nano.sock")
            listener = Listener(path, family="AF_UNIX")
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    ensure_socket_available(Path(path))
            finally:
                listener.close()
            self.assertLess(time.monotonic() - started, 1.0)

    def test_server_cleanup_does_not_unlink_replaced_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nano.sock"
            path.write_text("old")
            owner_link = pin_owned_socket(path)
            path.unlink()
            path.write_text("replacement")

            unlink_owned_socket(path, owner_link)

            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(), "replacement")
            self.assertFalse(owner_link.exists())

    def test_server_cleanup_unlinks_its_own_pinned_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nano.sock"
            path.write_text("owned")
            owner_link = pin_owned_socket(path)

            unlink_owned_socket(path, owner_link)

            self.assertFalse(path.exists())
            self.assertFalse(owner_link.exists())


if __name__ == "__main__":
    unittest.main()
