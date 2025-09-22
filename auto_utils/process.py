"""Utilities for launching subprocesses with robust logging."""

from __future__ import annotations

import logging
import signal
import subprocess
import threading
from typing import Iterable, List, Optional, Sequence, Union

_STREAM_CHUNK_SIZE = 4096


def _log_line(logger: Optional[logging.Logger], level: int, prefix: str, text: str) -> None:
    """Send a single line to the provided logger, if any."""
    if logger is None:
        return
    message = text.rstrip("\r")
    if prefix:
        logger.log(level, f"[{prefix}] {message}")
    else:
        logger.log(level, message)


def _drain_stream(
    stream,
    logger: Optional[logging.Logger],
    level: int,
    prefix: str,
) -> None:
    """Continuously read from *stream* and forward its content to *logger*.

    The function reads the stream in chunks to avoid blocking when the child
    process writes large amounts of data without flushing newlines.
    """
    buffer = ""
    try:
        while True:
            chunk = stream.read(_STREAM_CHUNK_SIZE)
            if not chunk:
                break
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="replace")
            buffer += chunk
            while True:
                newline_index = buffer.find("\n")
                if newline_index == -1:
                    break
                line, buffer = buffer[:newline_index], buffer[newline_index + 1 :]
                _log_line(logger, level, prefix, line)
            if len(buffer) > _STREAM_CHUNK_SIZE:
                _log_line(logger, level, prefix, buffer)
                buffer = ""
        if buffer:
            _log_line(logger, level, prefix, buffer)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _start_stream_thread(
    stream,
    logger: Optional[logging.Logger],
    level: int,
    prefix: str,
) -> Optional[threading.Thread]:
    if stream is None:
        return None
    thread = threading.Thread(
        target=_drain_stream,
        args=(stream, logger, level, prefix),
        name=f"{prefix}-logger",
        daemon=True,
    )
    thread.start()
    return thread


class ManagedProcess:
    """Wraps :class:`subprocess.Popen` ensuring stdout/stderr pipes are drained."""

    def __init__(self, popen: subprocess.Popen, drain_threads: Iterable[Optional[threading.Thread]]):
        self._popen = popen
        self._drain_threads: List[threading.Thread] = [t for t in drain_threads if t is not None]
        self._joined = False

    def __getattr__(self, item):
        return getattr(self._popen, item)

    def poll(self) -> Optional[int]:
        return self._popen.poll()

    def wait(self, timeout: Optional[float] = None) -> int:
        try:
            return self._popen.wait(timeout=timeout)
        finally:
            if self._popen.poll() is not None:
                self._join_threads()

    def _join_threads(self) -> None:
        if self._joined:
            return
        for thread in self._drain_threads:
            thread.join()
        self._joined = True

    def terminate(self) -> None:
        self._popen.terminate()

    def kill(self) -> None:
        self._popen.kill()

    def send_signal(self, sig: Union[int, signal.Signals]) -> None:
        self._popen.send_signal(sig)

    @property
    def pid(self) -> int:
        return self._popen.pid

    @property
    def returncode(self) -> Optional[int]:
        return self._popen.returncode


def launch_process(
    command: Union[Sequence[str], str],
    *,
    logger: Optional[logging.Logger] = None,
    name: Optional[str] = None,
    inherit_streams: bool = False,
    stdout_level: int = logging.INFO,
    stderr_level: int = logging.ERROR,
    **popen_kwargs,
) -> ManagedProcess:
    """Launch *command* and ensure stdout/stderr pipes never block the child.

    When *inherit_streams* is ``False`` (default), stdout and stderr are piped
    and drained using background threads. Each captured line is logged using the
    provided *logger*. If *inherit_streams* is ``True`` the child inherits the
    parent's file descriptors.
    """
    if "stdout" in popen_kwargs or "stderr" in popen_kwargs:
        raise ValueError("launch_process controls stdout/stderr; do not override them")

    if inherit_streams:
        popen = subprocess.Popen(command, **popen_kwargs)
        return ManagedProcess(popen, [])

    popen = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )

    process_name = name or (command if isinstance(command, str) else " ".join(map(str, command)))
    stdout_thread = _start_stream_thread(popen.stdout, logger, stdout_level, f"{process_name} stdout")
    stderr_thread = _start_stream_thread(popen.stderr, logger, stderr_level, f"{process_name} stderr")

    return ManagedProcess(popen, [stdout_thread, stderr_thread])
