"""Batching utilities for similarity requests under Pulse API limits."""

from typing import Any, Dict, List, Tuple
import numpy as np

# Maximum total items per similarity request
MAX_ITEMS = 10_000
# For self-similarity, chunk size is half of MAX_ITEMS
HALF_CHUNK = MAX_ITEMS // 2


def _make_self_chunks(items: List[str]) -> List[List[str]]:
    """Split a single list into chunks sized for self-similarity."""
    N = len(items)
    if N <= MAX_ITEMS:
        return [items]
    C = HALF_CHUNK
    return [items[i : i + C] for i in range(0, N, C)]


def _make_cross_bodies(
    set_a: List[str], set_b: List[str], flatten: bool
) -> List[Dict[str, Any]]:
    """Determine request bodies for cross-similarity with batching."""
    A, B = len(set_a), len(set_b)
    # If combined size fits
    if A + B <= MAX_ITEMS:
        return [{"set_a": set_a, "set_b": set_b, "flatten": flatten}]

    # Keep the smaller set intact if possible
    if A <= B < MAX_ITEMS:
        chunk_size = MAX_ITEMS - A
        chunks_b = [set_b[i : i + chunk_size] for i in range(0, B, chunk_size)]
        return [{"set_a": set_a, "set_b": b, "flatten": flatten} for b in chunks_b]
    if B <= A < MAX_ITEMS:
        chunk_size = MAX_ITEMS - B
        chunks_a = [set_a[i : i + chunk_size] for i in range(0, A, chunk_size)]
        return [{"set_a": a, "set_b": set_b, "flatten": flatten} for a in chunks_a]

    # Otherwise, chunk both sets into halves
    chunks_a = [set_a[i : i + HALF_CHUNK] for i in range(0, A, HALF_CHUNK)]
    chunks_b = [set_b[j : j + HALF_CHUNK] for j in range(0, B, HALF_CHUNK)]
    bodies: List[Dict[str, Any]] = []
    for a in chunks_a:
        for b in chunks_b:
            bodies.append({"set_a": a, "set_b": b, "flatten": flatten})
    return bodies


def _stitch_results(
    results: List[Any],
    bodies: List[Dict[str, Any]],
    full_a: List[str],
    full_b: List[str],
) -> Any:
    """Stitch block results back into a full similarity matrix."""
    A, B = len(full_a), len(full_b)
    matrix = np.zeros((A, B), dtype=float)

    if full_a is full_b:
        # Self-similarity stitching
        chunks = _make_self_chunks(full_a)
        offsets = [0]
        for c in chunks:
            offsets.append(offsets[-1] + len(c))
        coords: List[Tuple[int, int]] = []
        for i in range(len(chunks)):
            for j in range(i, len(chunks)):
                coords.append((i, j))
        for res, (i, j) in zip(results, coords):
            block = res.matrix
            r0, r1 = offsets[i], offsets[i + 1]
            c0, c1 = offsets[j], offsets[j + 1]
            matrix[r0:r1, c0:c1] = block
            if i != j:
                matrix[c0:c1, r0:r1] = block.T
    else:
        # Cross-similarity stitching
        if bodies:
            # Stitch based on submitted bodies
            # Determine unique chunks of set_a and set_b in order
            chunks_a: List[List[str]] = []
            chunks_b: List[List[str]] = []
            for body in bodies:
                a = body["set_a"]
                if not any(a is existing or a == existing for existing in chunks_a):
                    chunks_a.append(a)
                b = body["set_b"]
                if not any(b is existing or b == existing for existing in chunks_b):
                    chunks_b.append(b)

            # Compute offsets for rows and columns
            offsets_a = [0]
            for a in chunks_a:
                offsets_a.append(offsets_a[-1] + len(a))
            offsets_b = [0]
            for b in chunks_b:
                offsets_b.append(offsets_b[-1] + len(b))

            # Stitch each block into the full matrix
            for idx, body in enumerate(bodies):
                a = body["set_a"]
                b = body["set_b"]
                # find chunk indices
                i = next(
                    i for i, chunk in enumerate(chunks_a) if chunk is a or chunk == a
                )
                j = next(
                    j for j, chunk in enumerate(chunks_b) if chunk is b or chunk == b
                )
                res = results[idx]
                block = res["matrix"]
                r0, r1 = offsets_a[i], offsets_a[i + 1]
                c0, c1 = offsets_b[j], offsets_b[j + 1]
                matrix[r0:r1, c0:c1] = block
        else:
            # Fallback to original splitting logic when no bodies provided
            chunks_a = (
                _make_self_chunks(full_a) if len(full_a) > MAX_ITEMS else [full_a]
            )
            chunks_b = (
                _make_self_chunks(full_b) if len(full_b) > MAX_ITEMS else [full_b]
            )
            offsets_a = [0]
            for c in chunks_a:
                offsets_a.append(offsets_a[-1] + len(c))
            offsets_b = [0]
            for c in chunks_b:
                offsets_b.append(offsets_b[-1] + len(c))
            idx = 0
            for i, a in enumerate(chunks_a):
                for j, b in enumerate(chunks_b):
                    res = results[idx]
                    block = res.matrix
                    r0, r1 = offsets_a[i], offsets_a[i + 1]
                    c0, c1 = offsets_b[j], offsets_b[j + 1]
                    matrix[r0:r1, c0:c1] = block
                    idx += 1
    return matrix
