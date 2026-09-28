"""Smart decision caching — avoid redundant predictions.

Caches predict() results by state+questions hash with configurable TTL.
Useful in servers and pipelines where the same (or very similar) decisions
repeat within a short window.

    from laya import Router
    from laya.cache import CachedRouter

    router = Router(preload=True)
    cached = CachedRouter(router, ttl=300, max_size=10_000)

    # First call: runs the model (~33 ms)
    r1 = cached.predict(state, questions)

    # Second call with same inputs: returns from cache (<0.1 ms)
    r2 = cached.predict(state, questions)

    print(cached.stats)  # CacheStats(hits=1, misses=1, hit_rate=0.5, ...)
    cached.clear()
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


def _stable_hash(state: Any, questions: dict, model: str | None = None) -> str:
    """Produce a deterministic hash for a (state, questions, model) triple."""
    payload = json.dumps(
        {"s": state, "q": questions, "m": model},
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


@dataclass
class CacheStats:
    """Live statistics for the decision cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    def __repr__(self) -> str:
        return (
            "CacheStats(hits=%d, misses=%d, hit_rate=%.2f, "
            "evictions=%d, expirations=%d, size=%d/%d)"
            % (self.hits, self.misses, self.hit_rate,
               self.evictions, self.expirations, self.size, self.max_size)
        )


@dataclass
class _CacheEntry:
    result: dict[str, Any]
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0


class DecisionCache:
    """LRU cache for decision results with TTL-based expiration.

    Thread-safe. Entries are evicted when the cache exceeds max_size (LRU) or
    when their TTL expires (checked lazily on access).

    Parameters
    ----------
    ttl : float
        Time-to-live in seconds for each cache entry (default 300 = 5 minutes).
    max_size : int
        Maximum number of cached results (default 10000).
    """

    def __init__(self, *, ttl: float = 300.0, max_size: int = 10_000):
        self._ttl = ttl
        self._max_size = max_size
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats(max_size=max_size)

    def get(self, key: str) -> dict[str, Any] | None:
        """Look up a cached result by key. Returns None on miss or expiry."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            now = time.monotonic()
            if now - entry.created_at > self._ttl:
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                self._stats.size = len(self._cache)
                return None

            self._cache.move_to_end(key)
            entry.access_count += 1
            entry.last_accessed = now
            self._stats.hits += 1
            result = dict(entry.result)
            result["_cache"] = {"hit": True, "key": key, "age_ms": (now - entry.created_at) * 1000}
            return result

    def put(self, key: str, result: dict[str, Any]) -> None:
        """Store a result in the cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = _CacheEntry(
                    result=result, created_at=time.monotonic(), last_accessed=time.monotonic(),
                )
            else:
                while len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                    self._stats.evictions += 1
                now = time.monotonic()
                self._cache[key] = _CacheEntry(
                    result=result, created_at=now, last_accessed=now,
                )
            self._stats.size = len(self._cache)

    def clear(self) -> int:
        """Remove all entries. Returns the number of entries cleared."""
        with self._lock:
            n = len(self._cache)
            self._cache.clear()
            self._stats.size = 0
            return n

    def evict_expired(self) -> int:
        """Proactively remove all expired entries. Returns the count removed."""
        now = time.monotonic()
        removed = 0
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items()
                if now - v.created_at > self._ttl
            ]
            for k in expired_keys:
                del self._cache[k]
                removed += 1
            self._stats.expirations += removed
            self._stats.size = len(self._cache)
        return removed

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            self._stats.size = len(self._cache)
        return self._stats

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            return time.monotonic() - entry.created_at <= self._ttl


class CachedRouter:
    """A drop-in wrapper around Router (or Agent) that caches predict() results.

    Parameters
    ----------
    runner : Router or Agent
        The underlying Laya Router or Agent.
    ttl : float
        Cache TTL in seconds (default 300).
    max_size : int
        Maximum cache entries (default 10000).
    """

    def __init__(self, runner, *, ttl: float = 300.0, max_size: int = 10_000):
        self._runner = runner
        self._cache = DecisionCache(ttl=ttl, max_size=max_size)

    def predict(self, state: Any, questions: dict[str, Any], **kwargs) -> dict[str, Any]:
        """predict() with caching. Same signature as Router.predict().

        Cached results include a '_cache' key with hit/miss metadata.
        """
        model = kwargs.get("model")
        key = _stable_hash(state, questions, model)

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = self._runner.predict(state, questions, **kwargs)
        self._cache.put(key, result)
        result["_cache"] = {"hit": False, "key": key}
        return result

    def predict_batch(self, requests: list[dict], **kwargs) -> list[dict]:
        """predict_batch() with per-request caching.

        Each request in the batch is checked against the cache individually.
        Only uncached requests are sent to the model.
        """
        results: list[dict | None] = [None] * len(requests)
        uncached_indices = []
        uncached_requests = []

        for i, req in enumerate(requests):
            s = req.get("state", req) if isinstance(req, dict) else req
            q = req.get("questions", {}) if isinstance(req, dict) else {}
            m = req.get("model") if isinstance(req, dict) else None
            key = _stable_hash(s, q, m)

            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_requests.append(req)

        if uncached_requests:
            batch_results = self._runner.predict_batch(uncached_requests, **kwargs)
            for idx, br in zip(uncached_indices, batch_results):
                req = requests[idx]
                s = req.get("state", req) if isinstance(req, dict) else req
                q = req.get("questions", {}) if isinstance(req, dict) else {}
                m = req.get("model") if isinstance(req, dict) else None
                key = _stable_hash(s, q, m)
                self._cache.put(key, br)
                br["_cache"] = {"hit": False, "key": key}
                results[idx] = br

        return results  # type: ignore[return-value]

    @property
    def stats(self) -> CacheStats:
        return self._cache.stats

    def clear(self) -> int:
        return self._cache.clear()

    def __getattr__(self, name: str):
        return getattr(self._runner, name)
