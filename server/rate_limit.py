from __future__ import annotations

from collections import deque
import time
from typing import Deque, Dict, Tuple


class RateLimiter:
    def __init__(self):
        self._buckets: Dict[Tuple[str, str], Deque[float]] = {}

    def allow(self, scope: str, owner_id: str, limit: int, window_seconds: int) -> bool:
        key = (scope, owner_id)
        now = time.time()
        bucket = self._buckets.setdefault(key, deque())
        cutoff = now - window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True
