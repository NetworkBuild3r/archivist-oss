"""Pure text chunking helpers — flat and hierarchical parent/child splits.

Used by indexer and tests; keeps logic in one place to avoid drift.
"""

import hashlib
import re

NEEDLE_PATTERNS = [
    re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?"),      # IP / CIDR
    re.compile(r"(?:[\d*]+\s){4}[\d*]+"),                                    # cron expression
    re.compile(r"[\d,]+\s*(?:MiB|GiB|KiB|TiB|ms|KB|MB|GB|TB|PB)\b"),       # numeric with units
    re.compile(r"[A-Z]{2,}-\d{4,}"),                                         # employee / ticket IDs
    re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"),                        # datetime stamps
    re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I),  # UUID
    re.compile(r"[A-Z_]{2,}=\S+"),                                           # key=value pairs
    re.compile(r"(?:(?<=\s)|(?<=^)):\d{2,5}\b", re.MULTILINE),              # port numbers (require preceding whitespace)
]

_NEEDLE_MICRO_SIZE = 200
_NEEDLE_MICRO_CONTEXT = 80


def _extract_needle_micro_chunks(text: str) -> list[str]:
    """Find high-specificity tokens and build micro-chunks centred on them."""
    spans: list[tuple[int, int]] = []
    for pat in NEEDLE_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    if not spans:
        return []

    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + _NEEDLE_MICRO_CONTEXT:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    micros: list[str] = []
    for s, e in merged:
        lo = max(0, s - _NEEDLE_MICRO_CONTEXT)
        hi = min(len(text), e + _NEEDLE_MICRO_CONTEXT)
        if hi - lo > _NEEDLE_MICRO_SIZE:
            mid = (s + e) // 2
            lo = max(0, mid - _NEEDLE_MICRO_SIZE // 2)
            hi = lo + _NEEDLE_MICRO_SIZE
        snippet = text[lo:hi].strip()
        if len(snippet) >= 10:
            micros.append(snippet)

    return micros


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks by paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > size and current:
            chunks.append(current.strip())
            words = current.split()
            overlap_words = words[-overlap // 4:] if len(words) > overlap // 4 else []
            current = " ".join(overlap_words) + "\n\n" + para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text.strip()] if text.strip() else []


def chunk_text_hierarchical(
    text: str,
    filepath: str,
    parent_size: int = 2000,
    parent_overlap: int = 200,
    child_size: int = 500,
    child_overlap: int = 100,
) -> list[dict]:
    """Split text into hierarchical parent + child chunks.

    Parent IDs include filepath and parent index so IDs are stable and unique
    across files even when parent text matches another document.
    """
    parent_chunks = chunk_text(text, size=parent_size, overlap=parent_overlap)
    result: list[dict] = []

    for pi, parent in enumerate(parent_chunks):
        h = hashlib.md5(f"{filepath}\0{pi}\0{parent}".encode()).hexdigest()
        parent_id = f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
        result.append({
            "id": parent_id,
            "parent_id": None,
            "content": parent,
            "is_parent": True,
        })

        child_chunks = chunk_text(parent, size=child_size, overlap=child_overlap)
        for ci, child in enumerate(child_chunks):
            ch = hashlib.md5(f"{filepath}\0{pi}\0{ci}\0{child}".encode()).hexdigest()
            child_id = f"{ch[:8]}-{ch[8:12]}-{ch[12:16]}-{ch[16:20]}-{ch[20:32]}"
            result.append({
                "id": child_id,
                "parent_id": parent_id,
                "content": child,
                "is_parent": False,
            })

        micro_chunks = _extract_needle_micro_chunks(parent)
        for mi, micro in enumerate(micro_chunks):
            mh = hashlib.md5(f"{filepath}\0{pi}\0needle\0{mi}\0{micro}".encode()).hexdigest()
            micro_id = f"{mh[:8]}-{mh[8:12]}-{mh[12:16]}-{mh[16:20]}-{mh[20:32]}"
            result.append({
                "id": micro_id,
                "parent_id": parent_id,
                "content": micro,
                "is_parent": False,
            })

    return result
