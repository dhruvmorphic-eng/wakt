"""Tests for laya.cache — decision caching with TTL and LRU."""

import time
from laya.cache import DecisionCache, CachedRouter, CacheStats, _stable_hash


class _MockRouter:
    def __init__(self):
        self._call_count = 0

    def predict(self, state, questions, **kwargs):
        self._call_count += 1
        answers = {}
        for qid, qdef in questions.items():
            qtype = qdef.get("type", "choice")
            if qtype == "choice":
                labels = list((qdef.get("criteria") or {}).keys())
                answers[qid] = {"choice": labels[0] if labels else "a", "confidence": 0.9, "probabilities": {}}
            elif qtype == "score":
                answers[qid] = {"score": 1.0, "confidence": 0.8}
            elif qtype == "noul":
                answers[qid] = {"noul": 0.85, "confidence": 0.9}
        return {"answers": answers, "routing": {"model": "english"}}

    def predict_batch(self, requests, **kwargs):
        return [self.predict(
            r.get("state", r) if isinstance(r, dict) else r,
            r.get("questions", {}) if isinstance(r, dict) else {},
            **kwargs,
        ) for r in requests]


QUESTIONS = {
    "dept": {
        "type": "choice",
        "instructions": "Which?",
        "criteria": {"a": "x", "b": "y"},
    },
}


def test_stable_hash_deterministic():
    h1 = _stable_hash("hello", QUESTIONS, "english")
    h2 = _stable_hash("hello", QUESTIONS, "english")
    assert h1 == h2


def test_stable_hash_different_inputs():
    h1 = _stable_hash("hello", QUESTIONS, "english")
    h2 = _stable_hash("world", QUESTIONS, "english")
    assert h1 != h2


def test_cache_put_get():
    cache = DecisionCache(ttl=60, max_size=100)
    cache.put("key1", {"answers": {"a": 1}})
    result = cache.get("key1")
    assert result is not None
    assert result["answers"]["a"] == 1
    assert result["_cache"]["hit"] is True


def test_cache_miss():
    cache = DecisionCache(ttl=60, max_size=100)
    result = cache.get("nonexistent")
    assert result is None


def test_cache_ttl_expiry():
    cache = DecisionCache(ttl=0.01, max_size=100)
    cache.put("key1", {"answers": {}})
    time.sleep(0.02)
    result = cache.get("key1")
    assert result is None
    assert cache.stats.expirations >= 1


def test_cache_lru_eviction():
    cache = DecisionCache(ttl=60, max_size=2)
    cache.put("a", {"x": 1})
    cache.put("b", {"x": 2})
    cache.put("c", {"x": 3})
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None
    assert cache.stats.evictions >= 1


def test_cache_clear():
    cache = DecisionCache(ttl=60, max_size=100)
    cache.put("a", {})
    cache.put("b", {})
    n = cache.clear()
    assert n == 2
    assert len(cache) == 0


def test_cache_contains():
    cache = DecisionCache(ttl=60, max_size=100)
    cache.put("key1", {})
    assert "key1" in cache
    assert "key2" not in cache


def test_cache_evict_expired():
    cache = DecisionCache(ttl=0.01, max_size=100)
    cache.put("a", {})
    cache.put("b", {})
    time.sleep(0.02)
    removed = cache.evict_expired()
    assert removed == 2
    assert len(cache) == 0


def test_cache_stats():
    cache = DecisionCache(ttl=60, max_size=100)
    cache.put("a", {})
    cache.get("a")
    cache.get("miss")
    stats = cache.stats
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.hit_rate == 0.5
    assert stats.total_requests == 2


def test_cached_router_predict():
    router = _MockRouter()
    cached = CachedRouter(router, ttl=60, max_size=100)

    r1 = cached.predict("test", QUESTIONS)
    assert router._call_count == 1
    assert r1["_cache"]["hit"] is False

    r2 = cached.predict("test", QUESTIONS)
    assert router._call_count == 1
    assert r2["_cache"]["hit"] is True


def test_cached_router_different_inputs():
    router = _MockRouter()
    cached = CachedRouter(router, ttl=60, max_size=100)

    cached.predict("input1", QUESTIONS)
    cached.predict("input2", QUESTIONS)
    assert router._call_count == 2


def test_cached_router_stats():
    router = _MockRouter()
    cached = CachedRouter(router, ttl=60, max_size=100)

    cached.predict("test", QUESTIONS)
    cached.predict("test", QUESTIONS)
    assert cached.stats.hits == 1
    assert cached.stats.misses == 1


def test_cached_router_clear():
    router = _MockRouter()
    cached = CachedRouter(router, ttl=60, max_size=100)

    cached.predict("test", QUESTIONS)
    cached.clear()
    cached.predict("test", QUESTIONS)
    assert router._call_count == 2


def test_cached_router_batch():
    router = _MockRouter()
    cached = CachedRouter(router, ttl=60, max_size=100)

    cached.predict("state1", QUESTIONS)

    batch = [
        {"state": "state1", "questions": QUESTIONS},
        {"state": "state2", "questions": QUESTIONS},
    ]
    results = cached.predict_batch(batch)
    assert len(results) == 2
    assert results[0]["_cache"]["hit"] is True
    assert results[1]["_cache"]["hit"] is False


def test_cached_router_passthrough():
    router = _MockRouter()
    router.custom_attr = "hello"
    cached = CachedRouter(router, ttl=60, max_size=100)
    assert cached.custom_attr == "hello"


if __name__ == "__main__":
    test_stable_hash_deterministic()
    test_stable_hash_different_inputs()
    test_cache_put_get()
    test_cache_miss()
    test_cache_ttl_expiry()
    test_cache_lru_eviction()
    test_cache_clear()
    test_cache_contains()
    test_cache_evict_expired()
    test_cache_stats()
    test_cached_router_predict()
    test_cached_router_different_inputs()
    test_cached_router_stats()
    test_cached_router_clear()
    test_cached_router_batch()
    test_cached_router_passthrough()
    print("All cache tests passed.")
