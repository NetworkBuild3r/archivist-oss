"""Ultra-lightweight ``stdio`` MCP JSON-RPC shim (INIT-024/SPEC-007).

Imports **strictly Python stdlib** — ZERO ML/ONNX/MCP/numpy/sqlite imports —
so warm connect to a running ``archivistd`` Unix domain socket completes in
``<15ms`` and ZERO stray bytes can ever be emitted to ``stdout``.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import BinaryIO

from archivist.app.socket_security import (
    MAX_FRAME_BYTES,
    default_socket_path,
    open_private_log_fd,
    validate_socket_parent,
    verify_peer_uid,
)


def _connect_unix(sock_path: Path, timeout_s: float = 1.0) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout_s)
        sock.connect(str(sock_path))
        # Verify the remote daemon's kernel peer UID matches our own UID (prevents local MITM).
        verify_peer_uid(sock)
        sock.settimeout(None)
        return sock
    except Exception:
        sock.close()
        raise


def _read_bounded_line(stream: BinaryIO, max_bytes: int = MAX_FRAME_BYTES) -> bytes:
    """Read one newline-delimited frame capped at *max_bytes* (prevents OOM)."""
    line = stream.readline(max_bytes + 1)
    if len(line) > max_bytes and not line.endswith(b"\n"):
        raise ValueError(f"Incoming stdio JSON-RPC frame exceeds {max_bytes} bytes")
    if len(line) > max_bytes:
        raise ValueError(f"Incoming stdio JSON-RPC frame exceeds {max_bytes} bytes")
    return line


def ensure_daemon_connected(
    socket_path: Path | None = None,
    *,
    db_path: Path | None = None,
    embed_url: str = "",
    offline_embed: bool = False,
    startup_timeout_s: float = 15.0,
) -> socket.socket:
    """Connect to running ``archivistd``, auto-spawning it detached on demand."""
    sock_path = socket_path or default_socket_path()

    # Fast warm path (<1ms): connect directly to already-running daemon.
    try:
        return _connect_unix(sock_path, timeout_s=0.5)
    except OSError:
        pass

    # Auto-spawn archivistd detached with stdin/stdout decoupled so shim stdout stays 100% pure.
    validate_socket_parent(sock_path.parent)
    log_path = sock_path.parent / "archivistd.log"
    cmd = [
        sys.executable,
        "-m",
        "archivist.app.daemon",
        "--socket",
        str(sock_path),
    ]
    if db_path is not None:
        cmd.extend(["--db", str(db_path)])
    if embed_url:
        cmd.extend(["--embed-url", embed_url])
    if offline_embed:
        cmd.append("--offline-embed")

    log_fd = open_private_log_fd(log_path)
    try:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log_fd,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(log_fd)

    deadline = time.monotonic() + startup_timeout_s
    delay = 0.01
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return _connect_unix(sock_path, timeout_s=0.5)
        except PermissionError:
            raise
        except OSError as exc:
            last_err = exc
            time.sleep(delay)
            delay = min(delay * 1.5, 0.1)

    raise RuntimeError(
        f"Timed out waiting for archivistd at {sock_path} after {startup_timeout_s:.1f}s: {last_err}"
    )


def run_stdio_bridge(
    socket_path: Path | None = None,
    *,
    db_path: Path | None = None,
    embed_url: str = "",
    offline_embed: bool = False,
) -> int:
    """Bridge ``sys.stdin.buffer`` <-> ``archivistd`` Unix socket <-> ``sys.stdout.buffer``."""
    sock = ensure_daemon_connected(
        socket_path=socket_path,
        db_path=db_path,
        embed_url=embed_url,
        offline_embed=offline_embed,
    )
    stdin_buf = sys.stdin.buffer
    stdout_buf = sys.stdout.buffer
    done = threading.Event()
    frame_error: list[BaseException] = []

    def _stdin_to_sock() -> None:
        try:
            while not done.is_set():
                line = _read_bounded_line(stdin_buf, MAX_FRAME_BYTES)
                if not line:
                    break
                sock.sendall(line)
        except ValueError as exc:
            frame_error.append(exc)
            sys.stderr.write(f"[archivist.stdio] {exc}\n")
            sys.stderr.flush()
        except OSError:
            pass
        finally:
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def _sock_to_stdout() -> None:
        try:
            while True:
                chunk = sock.recv(65_536)
                if not chunk:
                    break
                stdout_buf.write(chunk)
                stdout_buf.flush()
        except OSError:
            pass
        finally:
            done.set()

    t_in = threading.Thread(target=_stdin_to_sock, name="archivist-stdin-to-sock", daemon=True)
    t_out = threading.Thread(target=_sock_to_stdout, name="archivist-sock-to-stdout", daemon=True)
    t_in.start()
    t_out.start()

    # Wait until remote closes (which happens right after stdin EOF propagates).
    t_out.join()
    t_in.join(timeout=1.0)
    try:
        sock.close()
    except OSError:
        pass
    return 1 if frame_error else 0


def main(argv: list[str] | None = None) -> int:
    """Minimal CLI entrypoint for ``python -m archivist.app.stdio_shim``."""
    args = list(sys.argv[1:] if argv is None else argv)
    sock_path: Path | None = None
    db_path: Path | None = None
    embed_url = os.environ.get("ARCHIVIST_EMBED_URL", "")
    offline = os.environ.get("ARCHIVIST_LOCAL_EMBED_OFFLINE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--socket" and i + 1 < len(args):
            sock_path = Path(args[i + 1])
            i += 2
        elif tok == "--db" and i + 1 < len(args):
            db_path = Path(args[i + 1])
            i += 2
        elif tok == "--embed-url" and i + 1 < len(args):
            embed_url = args[i + 1]
            i += 2
        elif tok == "--offline-embed":
            offline = True
            i += 1
        else:
            i += 1

    return run_stdio_bridge(
        socket_path=sock_path,
        db_path=db_path,
        embed_url=embed_url,
        offline_embed=offline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
