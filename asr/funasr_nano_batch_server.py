#!/usr/bin/env python3
"""Persistent local Unix-socket server for manual Fun-ASR-Nano batches."""

from __future__ import annotations

import argparse
import logging
import os
from multiprocessing.connection import Listener
from pathlib import Path


LOGGER = logging.getLogger("funasr_nano_batch_server")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    from asr.funasr_nano_batch import FunASRNanoBatch
    import torch

    engine = FunASRNanoBatch(
        model_dir=args.model_dir,
        device=args.device,
        dtype="fp32",
        max_new_tokens=args.max_new_tokens,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    listener = Listener(str(socket_path), family="AF_UNIX")
    os.chmod(socket_path, 0o600)
    LOGGER.info("READY socket=%s pid=%d", socket_path, os.getpid())
    shutting_down = False
    try:
        while not shutting_down:
            connection = listener.accept()
            LOGGER.info("Client connected")
            try:
                while True:
                    try:
                        request = connection.recv()
                    except EOFError:
                        break
                    command = request.get("command")
                    if command == "ping":
                        connection.send({"ok": True, "status": "ready"})
                        continue
                    if command == "shutdown":
                        connection.send({"ok": True})
                        shutting_down = True
                        break
                    if command != "transcribe":
                        connection.send(
                            {"ok": False, "error": f"Unknown command: {command!r}"}
                        )
                        continue
                    try:
                        results = engine.transcribe_batch(
                            request["audios"],
                            keys=request["keys"],
                            language=request.get("language"),
                            return_timestamps=request.get("return_timestamps", True),
                        )
                        stats = dict(engine.last_stats)
                        if torch.cuda.is_available():
                            stats["peak_allocated_gib"] = (
                                torch.cuda.max_memory_allocated() / (1024 ** 3)
                            )
                            stats["peak_reserved_gib"] = (
                                torch.cuda.max_memory_reserved() / (1024 ** 3)
                            )
                        connection.send(
                            {"ok": True, "results": results, "stats": stats}
                        )
                    except Exception as exc:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        LOGGER.exception("Batch inference failed")
                        connection.send(
                            {
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
            finally:
                connection.close()
                LOGGER.info("Client disconnected")
    finally:
        listener.close()
        if socket_path.exists():
            socket_path.unlink()


if __name__ == "__main__":
    main()
