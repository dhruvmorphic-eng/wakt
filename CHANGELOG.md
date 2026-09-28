# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-09-28

### Added
- **Decision Pipelines** (`laya.pipeline`) — chain multiple decisions with conditional branching, state transforms, dry-run previews, max-steps safety limits, and on-complete callbacks
- **Decision Explainability** (`laya.explain`) — human-readable explanations with per-question reasoning, confidence levels, margin analysis, entropy, and flag detection for close calls and low confidence
- **Smart Decision Caching** (`laya.cache`) — thread-safe LRU cache with TTL expiration, batch-aware caching, and live statistics; `CachedRouter` is a drop-in wrapper for Router
- **A/B Testing Framework** (`laya.testing`) — compare checkpoints, question schemas, or thresholds across arms with agreement rates, per-question breakdowns, disagreement reports, and JSON export
- Test suites for all four new modules (`tests/test_pipeline.py`, `tests/test_explain.py`, `tests/test_cache.py`, `tests/test_testing.py`)
- `wakt` and `wakt-serve` CLI entry points alongside the original `laya` commands
- NOTICE file with proper Apache 2.0 attribution to upstream Laya project
- CHANGELOG.md for tracking changes
- `py.typed` marker for PEP 561 type checker support
- Beginner-friendly README with clear examples and integration guides
- SECURITY.md with vulnerability reporting guidelines

### Changed
- Rebranded project metadata to Wakt (package still imports as `laya` for compatibility)
- Improved CONTRIBUTING.md with clearer setup instructions
- Updated project URLs to point to this repository
- Bumped version to 0.4.0

### Upstream
- Based on [Laya v0.3.21](https://github.com/NandhaKishorM/laya) by Convai Innovations
