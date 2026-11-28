"""Shared local ``archivistd`` Unix domain socket daemon (INIT-024/SPEC-007).

Zero-port local architecture (``ARCHIVIST_PROFILE=local``):
- Listens strictly on a ``0600`` Unix domain socket (never binds TCP ports).
- Verifies peer process UID via kernel credentials (``SO_PEERCRED`` on Linux,
  ``LOCAL_PEERCRED`` on macOS) and constructs an immutable :class:`RequestContext`
  (``local:<uid>``), rejecting mismatched UIDs immediately.
- Hosts ONE shared :class:`LocalMemoryStore` + ONNX runtime across concurrent
  IDE windows (Cursor, Claude Code, Codex, OpenClaw) — zero ``SQLITE_BUSY``
  contention and zero duplicated ONNX model RAM.
- Enforces idle-TTL ONNX model unloading (default 15 min) and graceful idle
  daemon shutdown (default 60 min).
- Writes all logs strictly to ``sys.stderr`` (never ``sys.stdout``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path

import anyio
import mcp.types as types
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.message import SessionMessage

from archivist.app.handlers.scope_resolver import reset_request_context, set_request_context
from archivist.app.handlers.v3_coding import set_local_store
from archivist.app.mcp_server import create_mcp_server
from archivist.app.socket_security import (
    MAX_FRAME_BYTES,
    PeerCredentials,
    default_socket_path,
    read_peer_credentials,
    safe_remove_stale_socket,
    validate_socket_parent,
    verify_peer_uid,
)
from archivist.features.local_onnx import (
    DEFAULT_IDLE_UNLOAD_SECONDS,
    DEFAULT_LOCAL_EMBED_DIM,
    DEFAULT_LOCAL_EMBED_MODEL,
    LocalOnnxConfig,
    LocalOnnxRuntime,
)
from archivist.security.trust_tiers import RequestContext
from archivist.storage.local_store import LocalMemoryStore, default_local_db_path

logger = logging.getLogger("archivist.daemon")

DEFAULT_IDLE_DAEMON_TTL_SECONDS = 3600.0  # 60 minutes

__all__ = [
    "DEFAULT_IDLE_DAEMON_TTL_SECONDS",
    "ArchivistDaemon",
    "PeerCredentials",
    "default_socket_path",
    "read_peer_credentials",
    "verify_peer_uid",
]


def _is_socket_live(sock_path: Path) -> bool:
    """Return True if an existing Unix socket at *sock_path* accepts connections."""
    if not sock_path.exists():
        return False
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        probe.connect(str(sock_path))
        return True
    except OSError:
        return False
    finally:
        probe.close()


class ArchivistDaemon:
    """Local ``archivistd`` Unix domain socket server hosting MCP over JSON-RPC."""

    def __init__(
        self,
        *,
        socket_path: Path | None = None,
        db_path: Path | None = None,
        vector_backend: str = "sqlite-vec",
        embed_model: str = DEFAULT_LOCAL_EMBED_MODEL,
        vector_dim: int = DEFAULT_LOCAL_EMBED_DIM,
        embed_url: str = "",
        idle_model_ttl_seconds: float = DEFAULT_IDLE_UNLOAD_SECONDS,
        idle_daemon_ttl_seconds: float = DEFAULT_IDLE_DAEMON_TTL_SECONDS,
        offline_deterministic: bool = False,
        expected_uid: int | None = None,
    ) -> None:
        self.socket_path = socket_path or default_socket_path()
        self.db_path = db_path or default_local_db_path()
        self.idle_model_ttl_seconds = float(idle_model_ttl_seconds)
        self.idle_daemon_ttl_seconds = float(idle_daemon_ttl_seconds)
        self.expected_uid = expected_uid
        self._server: asyncio.AbstractServer | None = None
        self._active_connections = 0
        self._last_activity = time.monotonic()
        self._stop_event = asyncio.Event()
        self._watchdog_task: asyncio.Task[None] | None = None

        onnx_cfg = LocalOnnxConfig(
            embed_model=embed_model,
            vector_dim=vector_dim,
            embed_url=embed_url,
            idle_unload_seconds=self.idle_model_ttl_seconds,
            offline_deterministic=offline_deterministic,
        )
        self.store = LocalMemoryStore(
            db_path=self.db_path,
            onnx_runtime=LocalOnnxRuntime(onnx_cfg),
            vector_backend=vector_backend,
            embed_model=embed_model,
            vector_dim=vector_dim,
            offline_deterministic=offline_deterministic,
        )
        set_local_store(self.store)

    def _touch(self) -> None:
        self._last_activity = time.monotonic()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        raw_sock = writer.get_extra_info("socket")
        if raw_sock is None:
            writer.close()
            return
        try:
            creds = verify_peer_uid(raw_sock, expected_uid=self.expected_uid)
        except PermissionError as exc:
            logger.warning("%s", exc)
            writer.close()
            await writer.wait_closed()
            return

        ctx = RequestContext.for_local_uid(
            creds.uid,
            source_client="archivistd-stdio",
            channel="mcp",
            actor_type="verified_local",
        )
        token = set_request_context(ctx)
        self._active_connections += 1
        self._touch()

        read_writer: MemoryObjectSendStream[SessionMessage | Exception]
        read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
        write_stream: MemoryObjectSendStream[SessionMessage]
        write_reader: MemoryObjectReceiveStream[SessionMessage]

        read_writer, read_stream = anyio.create_memory_object_stream(16)
        write_stream, write_reader = anyio.create_memory_object_stream(16)

        mcp_srv = create_mcp_server("archivist")
        init_opts = mcp_srv.create_initialization_options()

        async def _socket_to_mcp() -> None:
            try:
                async with read_writer:
                    while True:
                        raw_line = await reader.readline()
                        if not raw_line:
                            break
                        self._touch()
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            msg = types.JSONRPCMessage.model_validate_json(line)
                            await read_writer.send(SessionMessage(msg))
                        except Exception as exc:
                            await read_writer.send(exc)
            except (ConnectionResetError, BrokenPipeError, anyio.ClosedResourceError):
                pass

        async def _mcp_to_socket() -> None:
            try:
                async with write_reader:
                    async for session_msg in write_reader:
                        self._touch()
                        encoded = (
                            session_msg.message.model_dump_json(by_alias=True, exclude_none=True)
                            + "\n"
                        ).encode("utf-8")
                        writer.write(encoded)
                        await writer.drain()
            except (ConnectionResetError, BrokenPipeError, anyio.ClosedResourceError):
                pass

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(_socket_to_mcp)
                tg.start_soon(_mcp_to_socket)
                try:
                    await mcp_srv.run(read_stream, write_stream, init_opts)
                finally:
                    # Close write_stream gracefully so _mcp_to_socket flushes all
                    # queued JSON-RPC responses before the connection closes.
                    await write_stream.aclose()
        except Exception as exc:
            logger.debug("Client session closed: %s", exc)
        finally:
            self._active_connections = max(0, self._active_connections - 1)
            self._touch()
            reset_request_context(token)
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    async def _idle_watchdog(self) -> None:
        check_interval = min(
            5.0,
            max(
                0.05,
                min(
                    self.idle_model_ttl_seconds if self.idle_model_ttl_seconds > 0 else 5.0,
                    self.idle_daemon_ttl_seconds if self.idle_daemon_ttl_seconds > 0 else 5.0,
                )
                / 4.0,
            ),
        )
        while not self._stop_event.is_set():
            await asyncio.sleep(check_interval)
            now = time.monotonic()
            self.store.onnx.maybe_unload_idle(now=now)
            if (
                self.idle_daemon_ttl_seconds > 0
                and self._active_connections == 0
                and (now - self._last_activity) >= self.idle_daemon_ttl_seconds
            ):
                logger.info(
                    "archivistd idle for %.1fs; initiating graceful shutdown.",
                    now - self._last_activity,
                )
                self._stop_event.set()
                break

    async def start(self) -> None:
        """Bind the ``0600`` Unix domain socket and start serving."""
        validate_socket_parent(self.socket_path.parent)
        safe_remove_stale_socket(self.socket_path, is_live=_is_socket_live)

        # Restrict umask around bind() to eliminate any brief permission race window.
        prev_umask = os.umask(0o177) if os.name == "posix" else None
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_client,
                path=str(self.socket_path),
                limit=MAX_FRAME_BYTES,
            )
        finally:
            if prev_umask is not None:
                os.umask(prev_umask)

        if os.name == "posix" and self.socket_path.exists():
            os.chmod(self.socket_path, 0o600)

        self._watchdog_task = asyncio.create_task(self._idle_watchdog())
        logger.info("archivistd listening on unix://%s (mode 0600)", self.socket_path)

    async def stop(self) -> None:
        """Shut down listener, cancel watchdog, close store, and remove socket file."""
        self._stop_event.set()
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        self.store.close()
        set_local_store(None)
        self.socket_path.unlink(missing_ok=True)

    async def serve_forever(self) -> None:
        """Run daemon until stopped by signal or idle TTL."""
        await self.start()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop_event.set)
            except (NotImplementedError, RuntimeError):
                pass
        try:
            await self._stop_event.wait()
        finally:
            await self.stop()


def _configure_stderr_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m archivist.app.daemon``."""
    _configure_stderr_logging()
    parser = argparse.ArgumentParser(prog="archivistd", description="Archivist local MCP daemon")
    parser.add_argument("--socket", type=Path, default=None, help="Unix domain socket path")
    parser.add_argument("--db", type=Path, default=None, help="SQLite memory.db path")
    parser.add_argument(
        "--vector-backend",
        default=os.environ.get("ARCHIVIST_LOCAL_VECTOR_BACKEND", "sqlite-vec"),
        choices=["sqlite-vec", "usearch"],
    )
    parser.add_argument(
        "--embed-url",
        default=os.environ.get("ARCHIVIST_EMBED_URL", ""),
        help="Optional OpenAI-compatible embeddings endpoint URL",
    )
    parser.add_argument(
        "--idle-model-ttl",
        type=float,
        default=DEFAULT_IDLE_UNLOAD_SECONDS,
        help="Idle seconds before unloading ONNX model sessions (default: 900)",
    )
    parser.add_argument(
        "--idle-daemon-ttl",
        type=float,
        default=DEFAULT_IDLE_DAEMON_TTL_SECONDS,
        help="Idle seconds with 0 clients before graceful exit (default: 3600)",
    )
    parser.add_argument(
        "--offline-embed",
        action="store_true",
        default=os.environ.get("ARCHIVIST_LOCAL_EMBED_OFFLINE", "").strip().lower()
        in ("1", "true", "yes"),
        help="Use deterministic offline local embedder (no downloads)",
    )
    args = parser.parse_args(argv)

    daemon = ArchivistDaemon(
        socket_path=args.socket,
        db_path=args.db,
        vector_backend=args.vector_backend,
        embed_url=args.embed_url,
        idle_model_ttl_seconds=args.idle_model_ttl,
        idle_daemon_ttl_seconds=args.idle_daemon_ttl,
        offline_deterministic=args.offline_embed,
    )
    asyncio.run(daemon.serve_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
