"""Pure text chunking helpers — flat and hierarchical parent/child splits.

Used by indexer and tests; keeps logic in one place to avoid drift.
"""

import hashlib


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

    return result
