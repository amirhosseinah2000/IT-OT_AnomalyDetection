"""Small shared primitives used by protocol decoder modules only."""

from __future__ import annotations

import math
from collections import Counter


def entropy(value: bytes | str) -> float:
    """Calculate a bounded Shannon entropy without an external dependency."""
    if not value:
        return 0.0
    items = value.encode("utf-8", errors="ignore") if isinstance(value, str) else value
    total = len(items)
    return round(
        -sum((count / total) * math.log2(count / total) for count in Counter(items).values()), 4
    )


def safe_text(value: bytes, limit: int = 2048) -> str:
    """Decode a payload defensively and bound retained text."""
    return value[:limit].decode("latin-1", errors="replace")
