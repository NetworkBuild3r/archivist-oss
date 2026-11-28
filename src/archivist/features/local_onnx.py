"""Local ONNX INT8 embedder & bounded-window cross-encoder reranker (INIT-024/SPEC-006).

Design highlights:
- Uses XDG cache directory (``$XDG_CACHE_HOME/archivist/models`` or
  ``~/.cache/archivist/models``).
- Verifies SHA-256 checksums before loading ONNX model files and NEVER swallows
  :class:`ModelIntegrityError` (fail-closed on tamper).
- Emits ALL download/load progress strictly to ``sys.stderr`` (NEVER ``stdout``)
  so MCP ``stdio`` framing is never polluted.
- Supports idle-TTL model unloading (default 15 min) to free RAM in ``archivistd``.
- Wave 1 ablation finding honored: cross-encoder reranking operates strictly on
  bounded L1/chunk windows (``<= max_rerank_chars`` / top-K) rather than
  unchunked full sessions so CPU p50 stays fast.
- Supports 1-flag ``--embed-url`` cloud override and an honest deterministic
  hash-projection fallback (reported under ``archivist/deterministic-hash-*`` so
  ``archivist_meta`` never mislabels hash vectors as ``BAAI/bge-small-en-v1.5``).
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import os
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger("archivist.local_onnx")

DEFAULT_LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LOCAL_EMBED_DIM = 384
DEFAULT_LOCAL_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
DETERMINISTIC_FALLBACK_MODEL_PREFIX = "archivist/deterministic-hash"
DEFAULT_IDLE_UNLOAD_SECONDS = 900.0  # 15 minutes

# Wave 1 finding: cap reranker input passages to bounded L1/chunk windows so
# CPU latency stays comfortably within budget (never pass unchunked multi-turn logs).
DEFAULT_RERANK_MAX_CHARS = 1024
DEFAULT_RERANK_MAX_CANDIDATES = 12

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]{2,}")


class ModelIntegrityError(RuntimeError):
    """Raised when an ONNX model file fails SHA-256 verification."""


def get_model_cache_dir() -> Path:
    """Return the XDG model cache directory (``~/.cache/archivist/models``)."""
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else (Path.home() / ".cache")
    cache_dir = base / "archivist" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def compute_file_sha256(path: Path) -> str:
    """Compute lowercase hex SHA-256 digest of *path* in bounded 64KB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(65_536)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_file_sha256(path: Path, expected_sha256: str) -> None:
    """Verify *path* matches *expected_sha256* or raise :class:`ModelIntegrityError`."""
    expected = (expected_sha256 or "").strip().lower()
    if not expected:
        return
    if not path.exists():
        raise ModelIntegrityError(f"ONNX model artifact missing: {path}")
    actual = compute_file_sha256(path)
    if actual != expected:
        raise ModelIntegrityError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}. "
            "Refusing to load potentially corrupted or tampered model artifact."
        )


def _find_onnx_artifact(cache_dir: Path, explicit_filename: str, model_name: str) -> Path | None:
    """Locate the concrete ``.onnx`` artifact for *model_name* under *cache_dir*."""
    direct = cache_dir / explicit_filename
    if direct.is_file():
        return direct
    if not cache_dir.exists():
        return None
    slug_parts = [p.lower() for p in re.split(r"[/\\]+", model_name) if p]
    candidates = sorted(cache_dir.rglob("*.onnx"))
    for cand in candidates:
        lower = str(cand).lower()
        if any(part in lower for part in slug_parts):
            return cand
    return candidates[0] if len(candidates) == 1 else None


def verify_cached_models(
    cache_dir: Path,
    *,
    embed_model: str = DEFAULT_LOCAL_EMBED_MODEL,
    rerank_model: str = DEFAULT_LOCAL_RERANK_MODEL,
    expected_embed_sha256: str = "",
    expected_rerank_sha256: str = "",
) -> dict[str, Any]:
    """Inspect cached ONNX artifacts under *cache_dir* and verify configured SHA-256 digests."""
    embed_path = _find_onnx_artifact(cache_dir, "embed.onnx", embed_model)
    rerank_path = _find_onnx_artifact(cache_dir, "rerank.onnx", rerank_model)

    embed_sha = compute_file_sha256(embed_path) if embed_path is not None else None
    rerank_sha = compute_file_sha256(rerank_path) if rerank_path is not None else None

    errors: list[str] = []
    if expected_embed_sha256:
        try:
            verify_file_sha256(embed_path or (cache_dir / "embed.onnx"), expected_embed_sha256)
        except ModelIntegrityError as exc:
            errors.append(str(exc))
    if expected_rerank_sha256:
        try:
            verify_file_sha256(rerank_path or (cache_dir / "rerank.onnx"), expected_rerank_sha256)
        except ModelIntegrityError as exc:
            errors.append(str(exc))

    return {
        "ok": len(errors) == 0,
        "cache_dir": str(cache_dir),
        "embed_artifact": str(embed_path) if embed_path is not None else None,
        "embed_sha256": embed_sha,
        "rerank_artifact": str(rerank_path) if rerank_path is not None else None,
        "rerank_sha256": rerank_sha,
        "errors": errors,
    }


def _deterministic_hash_embed(text: str, dim: int = DEFAULT_LOCAL_EMBED_DIM) -> list[float]:
    """Fast, zero-network deterministic feature-hashing unit vector.

    Preserves lexical and character 3-gram overlap so cosine similarity behaves
    meaningfully in completely offline environments and tests without downloading
    external model weights.
    """
    vec = np.zeros(dim, dtype=np.float32)
    cleaned = (text or "").strip().lower()
    if not cleaned:
        vec[0] = 1.0
        return vec.tolist()

    tokens = _TOKEN_RE.findall(cleaned)
    features: list[tuple[str, float]] = [(tok, 2.0) for tok in tokens]
    # Add character 3-grams for morphological similarity.
    for tok in tokens[:128]:
        if len(tok) >= 4:
            for i in range(len(tok) - 2):
                features.append((f"3g:{tok[i : i + 3]}", 0.5))

    if not features:
        features = [(cleaned[:64], 1.0)]

    for feat, weight in features:
        h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] & 1) == 0 else -1.0
        vec[idx] += sign * weight

    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    else:
        vec[0] = 1.0
    return vec.tolist()


def _log_stderr(msg: str) -> None:
    """Write progress strictly to ``sys.stderr`` — NEVER ``sys.stdout``."""
    sys.stderr.write(f"[archivist.onnx] {msg}\n")
    sys.stderr.flush()


@dataclass
class LocalOnnxConfig:
    """Configuration for :class:`LocalOnnxRuntime`."""

    embed_model: str = DEFAULT_LOCAL_EMBED_MODEL
    vector_dim: int = DEFAULT_LOCAL_EMBED_DIM
    rerank_model: str = DEFAULT_LOCAL_RERANK_MODEL
    rerank_enabled: bool = False
    rerank_max_chars: int = DEFAULT_RERANK_MAX_CHARS
    rerank_max_candidates: int = DEFAULT_RERANK_MAX_CANDIDATES
    embed_url: str = ""
    embed_api_key: str = ""
    cache_dir: Path | None = None
    expected_embed_sha256: str = ""
    expected_rerank_sha256: str = ""
    idle_unload_seconds: float = DEFAULT_IDLE_UNLOAD_SECONDS
    offline_deterministic: bool = False


class LocalOnnxRuntime:
    """Local ONNX INT8 embedder & bounded-window reranker with idle unload."""

    def __init__(self, config: LocalOnnxConfig | None = None) -> None:
        self.config = config or LocalOnnxConfig()
        self.cache_dir = self.config.cache_dir or get_model_cache_dir()
        self._embedder: Any = None
        self._reranker: Any = None
        self._last_used_monotonic: float = time.monotonic()
        self._fell_back_to_hash: bool = False

    def _is_offline_mode(self) -> bool:
        return self.config.offline_deterministic or os.environ.get(
            "ARCHIVIST_LOCAL_EMBED_OFFLINE", ""
        ).strip().lower() in ("1", "true", "yes")

    @property
    def is_deterministic_fallback(self) -> bool:
        """Return True when deterministic hash embeddings are active instead of ONNX/remote."""
        if self.config.embed_url:
            return False
        if self._is_offline_mode() or self._fell_back_to_hash:
            return True
        return importlib.util.find_spec("fastembed") is None

    @property
    def active_embed_model_id(self) -> str:
        """Honest model identifier persisted in ``archivist_meta``.

        Never mislabels deterministic hash fallback vectors as ``BAAI/bge-small-en-v1.5``.
        """
        if self.config.embed_url:
            return self.config.embed_model
        if self.is_deterministic_fallback:
            if self.config.embed_model == DEFAULT_LOCAL_EMBED_MODEL:
                return f"{DETERMINISTIC_FALLBACK_MODEL_PREFIX}-{self.config.vector_dim}"
            return f"{DETERMINISTIC_FALLBACK_MODEL_PREFIX}:{self.config.embed_model}"
        return self.config.embed_model

    @property
    def embed_model(self) -> str:
        return self.active_embed_model_id

    @property
    def vector_dim(self) -> int:
        return self.config.vector_dim

    @property
    def is_loaded(self) -> bool:
        return self._embedder is not None or self._reranker is not None

    def _touch(self) -> None:
        self._last_used_monotonic = time.monotonic()

    def unload(self) -> None:
        """Explicitly release loaded ONNX sessions to reclaim RAM."""
        if self._embedder is not None or self._reranker is not None:
            _log_stderr("Unloading idle ONNX runtime sessions to reclaim memory.")
        self._embedder = None
        self._reranker = None

    def maybe_unload_idle(self, *, now: float | None = None) -> bool:
        """Unload ONNX models if idle longer than ``idle_unload_seconds``."""
        if not self.is_loaded or self.config.idle_unload_seconds <= 0:
            return False
        current = time.monotonic() if now is None else now
        if (current - self._last_used_monotonic) >= self.config.idle_unload_seconds:
            self.unload()
            return True
        return False

    def _verify_embed_artifact(self, *, require_present: bool) -> None:
        if not self.config.expected_embed_sha256:
            return
        artifact = _find_onnx_artifact(self.cache_dir, "embed.onnx", self.config.embed_model)
        if artifact is not None:
            verify_file_sha256(artifact, self.config.expected_embed_sha256)
        elif require_present:
            verify_file_sha256(self.cache_dir / "embed.onnx", self.config.expected_embed_sha256)

    def _verify_rerank_artifact(self, *, require_present: bool) -> None:
        if not self.config.expected_rerank_sha256:
            return
        artifact = _find_onnx_artifact(self.cache_dir, "rerank.onnx", self.config.rerank_model)
        if artifact is not None:
            verify_file_sha256(artifact, self.config.expected_rerank_sha256)
        elif require_present:
            verify_file_sha256(self.cache_dir / "rerank.onnx", self.config.expected_rerank_sha256)

    def _load_onnx_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder

        # Pre-check any existing cached ONNX artifact before loading (fail-closed on tamper).
        self._verify_embed_artifact(require_present=False)

        _log_stderr(
            f"Loading local ONNX embedder '{self.config.embed_model}' (cache: {self.cache_dir})"
        )
        fastembed_mod = importlib.import_module("fastembed")
        text_embedding_cls = fastembed_mod.TextEmbedding

        # FastEmbed writes progress bars via tqdm; ensure cache_dir stays in XDG path.
        embedder = text_embedding_cls(
            model_name=self.config.embed_model,
            cache_dir=str(self.cache_dir),
        )
        # Verify the actual downloaded/resolved ONNX model file(s) before caching session.
        try:
            self._verify_embed_artifact(require_present=True)
        except ModelIntegrityError:
            self._embedder = None
            raise

        self._embedder = embedder
        return self._embedder

    def _load_onnx_reranker(self) -> Any:
        if self._reranker is not None:
            return self._reranker

        self._verify_rerank_artifact(require_present=False)

        _log_stderr(
            f"Loading local ONNX reranker '{self.config.rerank_model}' (cache: {self.cache_dir})"
        )
        cross_enc_mod = importlib.import_module("fastembed.rerank.cross_encoder")
        text_cross_encoder_cls = cross_enc_mod.TextCrossEncoder

        reranker = text_cross_encoder_cls(
            model_name=self.config.rerank_model,
            cache_dir=str(self.cache_dir),
        )
        try:
            self._verify_rerank_artifact(require_present=True)
        except ModelIntegrityError:
            self._reranker = None
            raise

        self._reranker = reranker
        return self._reranker

    def _embed_via_url(self, texts: Sequence[str]) -> list[list[float]]:
        base = self.config.embed_url.rstrip("/")
        url = base if base.endswith("/embeddings") else f"{base}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.config.embed_api_key:
            headers["Authorization"] = f"Bearer {self.config.embed_api_key}"
        resp = httpx.post(
            url,
            headers=headers,
            json={"model": self.config.embed_model, "input": list(texts)},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        data.sort(key=lambda item: int(item.get("index", 0)))
        return [list(map(float, item["embedding"])) for item in data]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed *texts* into ``vector_dim``-dimensional float32 unit vectors."""
        if not texts:
            return []
        self._touch()

        if self.config.embed_url:
            return self._embed_via_url(texts)

        if self._is_offline_mode():
            return [_deterministic_hash_embed(t, self.config.vector_dim) for t in texts]

        # Always verify expected SHA-256 even if fastembed is missing/fails later.
        self._verify_embed_artifact(require_present=False)

        if importlib.util.find_spec("fastembed") is None:
            self._fell_back_to_hash = True
            return [_deterministic_hash_embed(t, self.config.vector_dim) for t in texts]

        try:
            embedder = self._load_onnx_embedder()
            raw_vecs = list(embedder.embed(list(texts)))
            out: list[list[float]] = []
            for v in raw_vecs:
                arr = np.asarray(v, dtype=np.float32).reshape(-1)
                if arr.shape[0] > self.config.vector_dim:
                    # Matryoshka truncation + re-normalization.
                    arr = arr[: self.config.vector_dim]
                    norm = float(np.linalg.norm(arr))
                    if norm > 0.0:
                        arr = arr / norm
                out.append(arr.tolist())
            return out
        except ModelIntegrityError:
            # Never fall back on tampered or corrupted ONNX artifacts.
            raise
        except Exception as exc:
            self._fell_back_to_hash = True
            _log_stderr(
                f"ONNX embedder unavailable ({exc}); falling back to deterministic local embedder."
            )
            return [_deterministic_hash_embed(t, self.config.vector_dim) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        max_chars: int | None = None,
        max_candidates: int | None = None,
    ) -> list[float]:
        """Score *passages* against *query* using bounded L1 windows (Wave 1 finding).

        Truncates each passage to ``max_chars`` (default 1024 chars ≈ 256 tokens)
        and limits cross-encoder inference to ``max_candidates`` (default 12) so
        CPU latency stays fast and never blows up on unchunked long transcripts.
        """
        if not passages:
            return []
        self._touch()

        char_cap = self.config.rerank_max_chars if max_chars is None else max(64, int(max_chars))
        cand_cap = (
            self.config.rerank_max_candidates
            if max_candidates is None
            else max(1, int(max_candidates))
        )
        bounded = [p[:char_cap] for p in passages[:cand_cap]]

        if not self.config.rerank_enabled or self._is_offline_mode():
            # Fast lexical overlap fallback when reranker is off or offline.
            q_terms = set(_TOKEN_RE.findall((query or "").lower()))
            scores: list[float] = []
            for p in passages:
                if not q_terms:
                    scores.append(0.0)
                    continue
                p_terms = set(_TOKEN_RE.findall((p[:char_cap]).lower()))
                scores.append(len(q_terms & p_terms) / max(1, len(q_terms)))
            return scores

        self._verify_rerank_artifact(require_present=False)

        try:
            reranker = self._load_onnx_reranker()
            raw_scores = [float(s) for s in reranker.rerank(query, bounded)]
            if len(passages) > len(raw_scores):
                tail_floor = (min(raw_scores) - 1.0) if raw_scores else 0.0
                raw_scores.extend([tail_floor] * (len(passages) - len(raw_scores)))
            return raw_scores
        except ModelIntegrityError:
            # Never fall back on tampered or corrupted ONNX artifacts.
            raise
        except Exception as exc:
            _log_stderr(f"ONNX reranker unavailable ({exc}); returning neutral scores.")
            return [0.0] * len(passages)
