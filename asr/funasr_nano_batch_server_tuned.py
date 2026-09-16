#!/usr/bin/env python3
"""Persistent Nano batch server with an explicit FP32/BF16 choice."""

from __future__ import annotations

import argparse
import errno
import logging
import os
import socket
import uuid
from multiprocessing.connection import Listener
from pathlib import Path


LOGGER = logging.getLogger("funasr_nano_batch_server")


def release_cuda_cache(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def pin_owned_socket(path: Path):
    """Pin the bound inode so pathname replacement cannot reuse its identity."""
    owner_link = path.with_name(
        f".{path.name}.owner.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        os.link(path, owner_link)
    except OSError as exc:
        LOGGER.warning("Could not pin socket ownership for %s: %s", path, exc)
        return None
    return owner_link


def unlink_owned_socket(path: Path, owner_link: Path | None) -> None:
    if owner_link is None:
        return
    try:
        try:
            owns_path = path.samefile(owner_link)
        except FileNotFoundError:
            owns_path = False
        if owns_path:
            path.unlink()
    finally:
        owner_link.unlink(missing_ok=True)


def ensure_socket_available(path: Path, timeout_seconds: float = 1.0) -> None:
    if not path.exists():
        return
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(timeout_seconds)
    try:
        probe.connect(str(path))
    except socket.timeout as exc:
        raise RuntimeError(f"Nano server socket is active or busy at {path}") from exc
    except OSError as exc:
        if exc.errno in {errno.ECONNREFUSED, errno.ENOENT}:
            path.unlink(missing_ok=True)
            return
        raise RuntimeError(f"Cannot verify existing socket {path}: {exc}") from exc
    finally:
        probe.close()
    raise RuntimeError(f"Nano server is already active at {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_socket_available(socket_path)

    from asr.funasr_nano_batch import FunASRNanoBatch
    import torch

    engine = FunASRNanoBatch(
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    listener = Listener(str(socket_path), family="AF_UNIX")
    os.chmod(socket_path, 0o600)
    owner_link = pin_owned_socket(socket_path)
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
                        connection.send(
                            {
                                "ok": True,
                                "status": "ready",
                                "dtype": args.dtype,
                                "model_dir": os.path.realpath(args.model_dir),
                            }
                        )
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
                release_cuda_cache(torch)
                LOGGER.info("Client disconnected")
    finally:
        listener.close()
        unlink_owned_socket(socket_path, owner_link)


if __name__ == "__main__":
    main()
