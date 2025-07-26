from __future__ import annotations

from typing import List, Sequence, TypeVar

T = TypeVar("T")


def chunk_list(items: Sequence[T], size: int) -> List[List[T]]:
    """Split ``items`` into chunks of length ``size``.

    Parameters
    ----------
    items:
        Sequence of items to chunk.
    size:
        Maximum chunk size.

    Returns
    -------
    list of list
        Items split into consecutive chunks of ``size`` or less.
    """
    if size <= 0:
        raise ValueError("size must be positive")

    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def chunk_texts(texts: Sequence[str], size: int) -> List[List[str]]:
    """Alias for :func:`chunk_list` specialized for strings."""
    return chunk_list(list(texts), size)
