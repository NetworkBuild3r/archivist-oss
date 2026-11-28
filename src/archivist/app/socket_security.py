"""Stdlib-only Unix socket & peer-credential security helpers (INIT-024/SPEC-007).

This module imports **strictly Python stdlib** so ``stdio_shim.py``, ``daemon.py``,
and ``cli/init_cmd.py`` can share socket path resolution, ``lstat``/``S_ISSOCK``
guards, ``0600`` private log creation, and kernel peer-UID verification without
pulling in ML, MCP, or legacy storage dependencies.
"""

from __future__ import annotations

import os
import socket
import stat
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

MAX_FRAME_BYTES = 4 * 1024 * 1024  # 4 MiB hard cap on NDJSON frames


def default_socket_path() -> Path:
    """Return default Unix domain socket path without importing heavy modules."""
    override = os.environ.get("ARCHIVIST_SOCKET_PATH", "").strip()
    if override:
        return Path(override)
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if xdg_runtime:
        base = Path(xdg_runtime)
        if base.exists() and os.access(base, os.W_OK):
            return base / "archivist" / "archivistd.sock"
    xdg_data = os.environ.get("XDG_DATA_HOME", "").strip()
    data_base = Path(xdg_data) if xdg_data else (Path.home() / ".local" / "share")
    return data_base / "archivist" / "archivistd.sock"


@dataclass(frozen=True)
class PeerCredentials:
    """Kernel-verified peer process identity from a Unix domain socket."""

    pid: int | None
    uid: int
    gid: int | None


def read_peer_credentials(sock: socket.socket) -> PeerCredentials:
    """Extract peer ``(pid, uid, gid)`` via ``SO_PEERCRED`` (Linux) or ``LOCAL_PEERCRED`` (macOS).

    Fails closed with :class:`PermissionError` on platforms without kernel
    peer-credential support (never synthesizes the local UID as a fallback).
    """
    if sys.platform.startswith("linux"):
        so_peercred = getattr(socket, "SO_PEERCRED", 17)
        raw = sock.getsockopt(socket.SOL_SOCKET, so_peercred, struct.calcsize("3i"))
        pid, uid, gid = struct.unpack("3i", raw)
        return PeerCredentials(pid=int(pid), uid=int(uid), gid=int(gid))

    if sys.platform == "darwin":
        # macOS: LOCAL_PEERCRED returns struct xucred { u_int cr_version; uid_t cr_uid; ... }
        sol_local = 0
        local_peercred = 0x001
        raw = sock.getsockopt(sol_local, local_peercred, struct.calcsize("IIh16I"))
        _ver, uid, _ngroups = struct.unpack("IIh", raw[:10])
        return PeerCredentials(pid=None, uid=int(uid), gid=None)

    raise PermissionError(
        f"Kernel Unix socket peer credential verification is unsupported on platform "
        f"{sys.platform!r}; refusing fail-open connection."
    )


def verify_peer_uid(sock: socket.socket, *, expected_uid: int | None = None) -> PeerCredentials:
    """Verify peer UID matches *expected_uid* (default: current process UID)."""
    if expected_uid is None:
        if not hasattr(os, "getuid"):
            raise PermissionError("Cannot determine local process UID for peer verification.")
        target_uid = os.getuid()
    else:
        target_uid = expected_uid

    creds = read_peer_credentials(sock)
    if creds.uid != target_uid:
        raise PermissionError(
            f"Rejected Unix socket peer with uid={creds.uid} (expected uid={target_uid})"
        )
    return creds


def validate_socket_parent(parent: Path) -> None:
    """Ensure *parent* is a real, non-symlink, user-owned ``0700`` directory."""
    try:
        st = os.lstat(parent)
    except FileNotFoundError:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        st = os.lstat(parent)

    if stat.S_ISLNK(st.st_mode):
        raise PermissionError(f"Refusing symlink socket parent directory: {parent}")
    if not stat.S_ISDIR(st.st_mode):
        raise PermissionError(f"Socket parent path is not a directory: {parent}")

    if os.name == "posix":
        my_uid = os.getuid()
        if st.st_uid != my_uid:
            raise PermissionError(
                f"Refusing socket parent directory {parent} owned by uid={st.st_uid} "
                f"(expected uid={my_uid})"
            )
        if not (st.st_mode & stat.S_ISVTX):
            os.chmod(parent, 0o700)


def safe_remove_stale_socket(
    socket_path: Path,
    *,
    is_live: Callable[[Path], bool],
) -> None:
    """Safely remove a stale Unix socket without ever unlinking symlinks or regular files."""
    try:
        st = os.lstat(socket_path)
    except FileNotFoundError:
        return

    if stat.S_ISLNK(st.st_mode):
        raise PermissionError(f"Refusing to remove symlink at socket path: {socket_path}")
    if not stat.S_ISSOCK(st.st_mode):
        raise PermissionError(
            f"Refusing to unlink non-socket path at socket location: {socket_path}"
        )

    if is_live(socket_path):
        raise RuntimeError(f"archivistd already listening at {socket_path}")

    socket_path.unlink(missing_ok=True)


def open_private_log_fd(log_path: Path) -> int:
    """Open *log_path* strictly as a non-symlink ``0600`` regular file for appending."""
    validate_socket_parent(log_path.parent)
    try:
        st = os.lstat(log_path)
    except FileNotFoundError:
        st = None

    if st is not None:
        if stat.S_ISLNK(st.st_mode):
            raise PermissionError(f"Refusing symlink daemon log path: {log_path}")
        if not stat.S_ISREG(st.st_mode):
            raise PermissionError(f"Daemon log path is not a regular file: {log_path}")

    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(log_path), flags, 0o600)
    try:
        if os.name == "posix":
            os.fchmod(fd, 0o600)
    except Exception:
        os.close(fd)
        raise
    return fd
