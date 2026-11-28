"""Integration tests for SPEC-007 stdio shim & shared local archivistd daemon."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from archivist.app.daemon import ArchivistDaemon, read_peer_credentials, verify_peer_uid

pytestmark = [pytest.mark.integration, pytest.mark.mcp]


async def _run_stdio_client(
    sock_path: Path,
    client_idx: int,
    scope: str,
) -> tuple[float, dict, dict, bytes]:
    """Spawn one stdio_shim subprocess, save + retrieve memory, return ping_ms and payloads."""
    env = dict(os.environ, PYTHONPATH=str(Path.cwd() / "src"))
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "archivist.app.stdio_shim",
        "--socket",
        str(sock_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None

    async def _rpc(msg: dict) -> dict | None:
        proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        await proc.stdin.drain()
        if "id" not in msg:
            return None
        raw = await asyncio.wait_for(proc.stdout.readline(), timeout=10.0)
        assert raw, "Unexpected EOF on stdio_shim stdout"
        return json.loads(raw.decode("utf-8"))

    init_resp = await _rpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": f"ide-client-{client_idx}", "version": "1.0"},
            },
        }
    )
    assert init_resp is not None and "result" in init_resp
    await _rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})

    t_ping = time.perf_counter()
    ping_resp = await _rpc({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    ping_ms = (time.perf_counter() - t_ping) * 1000.0
    assert ping_resp is not None and "result" in ping_resp

    save_resp = await _rpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "memory_save",
                "arguments": {
                    "content": f"Concurrent client {client_idx} architecture note alpha_{client_idx}",
                    "scope": scope,
                },
            },
        }
    )
    assert save_resp is not None
    save_payload = json.loads(save_resp["result"]["content"][0]["text"])

    ctx_resp = await _rpc(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "memory_context",
                "arguments": {
                    "query": f"architecture note alpha_{client_idx}",
                    "scope": scope,
                },
            },
        }
    )
    assert ctx_resp is not None
    ctx_payload = json.loads(ctx_resp["result"]["content"][0]["text"])

    proc.stdin.close()
    trailing_stdout = await proc.stdout.read()
    await proc.wait()
    assert proc.returncode == 0
    return ping_ms, save_payload, ctx_payload, trailing_stdout


class TestArchivistdAndStdioShim:
    def test_stdio_shim_imports_zero_ml_modules(self):
        """stdio_shim must import strictly stdlib (zero onnxruntime/numpy/fastembed/qdrant)."""
        code = (
            "import sys, archivist.app.stdio_shim; "
            "heavy = [m for m in ('onnxruntime', 'numpy', 'fastembed', 'qdrant_client', 'mcp') "
            "if m in sys.modules]; "
            "assert not heavy, f'Unexpected heavy imports: {heavy}'"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=dict(os.environ, PYTHONPATH=str(Path.cwd() / "src")),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == ""

    def test_peer_credentials_verified_and_mismatched_uid_rejected(self):
        s1, s2 = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            creds = read_peer_credentials(s1)
            expected_uid = os.getuid() if hasattr(os, "getuid") else 0
            assert creds.uid == expected_uid
            verified = verify_peer_uid(s1, expected_uid=expected_uid)
            assert verified.uid == expected_uid
            with pytest.raises(PermissionError, match="Rejected Unix socket peer"):
                verify_peer_uid(s1, expected_uid=expected_uid + 99999)
        finally:
            s1.close()
            s2.close()

    @pytest.mark.asyncio
    async def test_four_concurrent_stdio_clients_and_0600_zero_tcp(self, tmp_path: Path):
        sock_path = tmp_path / "archivistd.sock"
        db_path = tmp_path / "memory.db"
        daemon = ArchivistDaemon(
            socket_path=sock_path,
            db_path=db_path,
            offline_deterministic=True,
        )
        await daemon.start()
        try:
            # 1. Socket mode must be 0600 and server sockets must be AF_UNIX only (0 TCP ports)
            if os.name == "posix":
                assert stat.S_IMODE(os.stat(sock_path).st_mode) == 0o600
            assert daemon._server is not None
            for srv_sock in daemon._server.sockets:
                assert srv_sock.family == socket.AF_UNIX

            # 2. Spawn 4 concurrent stdio_shim clients against one shared archivistd
            scope = "repo:github.com/org/concurrent-test"
            results = await asyncio.gather(
                *[_run_stdio_client(sock_path, i, scope) for i in range(4)]
            )

            ping_latencies: list[float] = []
            for idx, (ping_ms, save_p, ctx_p, trailing) in enumerate(results):
                ping_latencies.append(ping_ms)
                assert trailing == b"", f"Non-JSON-RPC trailing bytes on stdout: {trailing!r}"
                assert save_p["memory_id"].startswith("mem_")
                assert save_p["enrichment_status"] == "synced"
                assert "<archivist_memory " in ctx_p["context"]
                assert f"alpha_{idx}" in ctx_p["context"]

            p95_ping = float(np.percentile(ping_latencies, 95))
            assert p95_ping < 25.0, f"Warm stdio ping p95={p95_ping:.2f}ms exceeded 25ms"
        finally:
            await daemon.stop()
            assert not sock_path.exists()

    @pytest.mark.asyncio
    async def test_idle_daemon_graceful_shutdown(self, tmp_path: Path):
        sock_path = tmp_path / "idle.sock"
        db_path = tmp_path / "idle.db"
        daemon = ArchivistDaemon(
            socket_path=sock_path,
            db_path=db_path,
            idle_model_ttl_seconds=0.1,
            idle_daemon_ttl_seconds=0.25,
            offline_deterministic=True,
        )
        await asyncio.wait_for(daemon.serve_forever(), timeout=3.0)
        assert not sock_path.exists()
