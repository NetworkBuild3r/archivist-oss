"""Pure text chunking helpers — flat and hierarchical parent/child splits.

Used by indexer and tests; keeps logic in one place to avoid drift.

Chunking strategies
-------------------
chunk_text(text, size, overlap)
    Paragraph-accumulation splitter.  Splits on double-newlines and
    accumulates paragraphs until the size limit is reached.  Used for the
    child pass and as a fallback for headingless documents.

chunk_text_semantic(text, size, overlap)
    Markdown-aware, meaning-boundary splitter.  Splits first on heading
    boundaries (## …), then on horizontal rules, and only falls back to
    paragraph accumulation inside oversized sections.  Fenced code blocks
    are never broken.  Short documents (len(text) <= size) are returned as
    a single chunk without any splitting.

chunk_text_hierarchical(text, filepath, …)
    Two-level split: parent pass uses chunk_text_semantic by default
    (controlled by CHUNKING_STRATEGY config), child pass uses chunk_text.
    Also produces needle micro-chunks via _extract_needle_micro_chunks.
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


_HEADING_RE = re.compile(r"^(#{1,6}\s.+)$", re.MULTILINE)
_FENCE_RE = re.compile(r"^```", re.MULTILINE)
_HR_RE = re.compile(r"^(?:---+|\*\*\*+|___+)\s*$", re.MULTILINE)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (heading, body) section pairs.

    Returns a list of (heading, body) tuples.  Content before the first
    heading is yielded with an empty heading string.  If no headings exist,
    returns a single ('', text) pair.
    """
    parts = _HEADING_RE.split(text)
    # parts alternates between pre-heading content and heading lines when
    # headings exist: [pre, h1, body1, h2, body2, ...]
    if len(parts) == 1:
        # No headings found
        return [("", text.strip())]

    sections: list[tuple[str, str]] = []
    pre = parts[0].strip()
    if pre:
        sections.append(("", pre))

    # Walk pairs of (heading, body) that follow
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((heading, body))

    return sections


def _split_preserving_fences(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-split text while keeping fenced code blocks intact.

    Lines inside a ``` fence are accumulated verbatim regardless of size.
    Outside fences, the same paragraph-accumulation logic as chunk_text is
    used.
    """
    lines = text.split("\n")
    blocks: list[str] = []
    current_block: list[str] = []
    in_fence = False

    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        current_block.append(line)
        # Flush on blank line outside a fence
        if line == "" and not in_fence:
            block_text = "\n".join(current_block).strip()
            if block_text:
                blocks.append(block_text)
            current_block = []

    remaining = "\n".join(current_block).strip()
    if remaining:
        blocks.append(remaining)

    # Now accumulate blocks into size-respecting chunks
    chunks: list[str] = []
    current = ""

    for block in blocks:
        if current and len(current) + len(block) + 2 > size:
            chunks.append(current.strip())
            words = current.split()
            overlap_words = words[-overlap // 4:] if len(words) > overlap // 4 else []
            current = " ".join(overlap_words) + "\n\n" + block if overlap_words else block
        else:
            current = current + "\n\n" + block if current else block

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text.strip()] if text.strip() else []


def chunk_text_semantic(text: str, size: int = 2000, overlap: int = 200) -> list[str]:
    """Split text at meaning boundaries, respecting markdown structure.

    Splitting priority:
    1. Heading boundaries (## …) — sections are never merged across headings.
    2. Horizontal rules (--- / *** / ___).
    3. Fenced code blocks — kept intact even if they exceed *size*.
    4. Blank-line-separated paragraphs (same as chunk_text fallback).

    Short-document fast path: if len(text) <= size the text is returned as a
    single chunk with no splitting, guaranteeing zero behaviour change for
    short atomic documents.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= size:
        return [stripped]

    sections = _split_into_sections(stripped)
    chunks: list[str] = []

    for heading, body in sections:
        # Combine heading and body for size calculation
        section_text = (f"{heading}\n\n{body}".strip()) if heading else body
        if not section_text:
            continue

        if len(section_text) <= size:
            chunks.append(section_text)
        else:
            # Section exceeds size — sub-split at paragraph/fence boundaries
            # Prepend heading to every sub-chunk so context is preserved
            sub_chunks = _split_preserving_fences(body, size, overlap)
            for i, sub in enumerate(sub_chunks):
                if heading and i == 0:
                    chunks.append(f"{heading}\n\n{sub}")
                elif heading:
                    # Continuation sub-chunks carry the heading as a breadcrumb
                    chunks.append(f"{heading} (cont.)\n\n{sub}")
                else:
                    chunks.append(sub)

    return chunks if chunks else [stripped]


def chunk_text_hierarchical(
    text: str,
    filepath: str,
    parent_size: int = 2000,
    parent_overlap: int = 200,
    child_size: int = 500,
    child_overlap: int = 100,
    strategy: str = "semantic",
) -> list[dict]:
    """Split text into hierarchical parent + child chunks.

    Parent IDs include filepath and parent index so IDs are stable and unique
    across files even when parent text matches another document.

    strategy="semantic" (default): parent pass uses chunk_text_semantic —
        heading-aligned, code-block-preserving splits.
    strategy="fixed": parent pass uses chunk_text — legacy paragraph
        accumulation, useful for regression testing.
    """
    if strategy == "semantic":
        parent_chunks = chunk_text_semantic(text, size=parent_size, overlap=parent_overlap)
    else:
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
        from config import MAX_MICRO_CHUNKS_PER_MEMORY
        micro_chunks = micro_chunks[:MAX_MICRO_CHUNKS_PER_MEMORY]
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
