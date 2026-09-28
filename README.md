<p align="center">
  <img src="assets/wakt-logo.png" alt="Wakt" width="180" />
  <h1 align="center">Wakt</h1>
  <p align="center"><strong>Instant decisions. Any language. One forward pass.</strong></p>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-convaiinnovations%2Flaya-blue)](https://huggingface.co/convaiinnovations/laya)

</div>

---

**Wakt** is a non-autoregressive System 1 decision engine that makes typed decisions — choice, score, and yes/no — over any text in **a single forward pass (~33ms)**, across **100+ languages**.

No text generation. No parsing. No hallucination. Just fast, calibrated, structured decisions.

> Built on top of [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations — see [NOTICE](NOTICE) for attribution.

---

## Why Wakt?

| Problem | How Wakt Solves It |
|---|---|
| LLM calls are slow and expensive for simple routing | Single forward pass in ~33ms on GPU |
| LLM outputs need parsing and can hallucinate | Returns structured data with calibrated probabilities |
| Most classifiers only work in English | Supports 100+ languages out of the box |
| Confidence scores are usually meaningless | Trained with RLCD for statistically calibrated confidence |

## Quick Start

### Install

```bash
pip install laya
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install laya
```

> **Python 3.10+** required. Optional extras: `laya[serve]` (HTTP server), `laya[mcp]` (MCP server), `laya[langchain]` (LangChain/LangGraph), `laya[onnx]` (ONNX Runtime).

### Your First Decision

```python
from laya import Router

router = Router()

state = "Hi, we were billed twice for March. Please refund the duplicate today or we cancel."

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "soon", "blocking"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
}

result = router.predict(state, questions)

print(result["answers"]["department"]["choice"])   # billing
print(result["answers"]["urgency"]["score"])        # 1.84 / 2.0
print(result["answers"]["churn_risk"]["noul"])      # 0.892 (89.2% yes)
```

### Works in Any Language

```python
# Hindi
result = router.predict(
    "मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।",
    {"department": questions["department"]},
)
print(result["answers"]["department"]["choice"])  # billing

# Spanish
result = router.predict(
    "La aplicación se cierra cada vez que abro la configuración.",
    {"department": questions["department"]},
)
print(result["answers"]["department"]["choice"])  # technical
```

### From the Command Line

```bash
laya "My payment failed twice" --preset triage       # built-in preset
laya "Refactor this service" --predict                # full answers
laya --batch tickets.txt --predict --json             # batch mode
```

---

## Three Decision Primitives

| Primitive | What It Returns | Use Cases |
|---|---|---|
| **choice** | Top label + probabilities per option + confidence | Department routing, intent classification, topic categorization |
| **score** | Expected level on an ordinal rubric + distribution | Urgency rating, frustration level, harm severity |
| **noul** | Calibrated probability P(yes) from 0.0 to 1.0 | Spam detection, churn risk, jailbreak detection |

## Three Checkpoints + Smart Router

| Checkpoint | Encoder | Params | Context | Best For |
|---|---|---|---|---|
| `laya` | ModernBERT-large | 421M | 512 | English |
| `laya-multilingual` | mmBERT-base | 322M | 1024 (up to 8,192) | 100+ languages, 2x faster |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 | Fine-tuned typed workflows |

The **Router** auto-detects the input language/script and dispatches to the optimal checkpoint — no configuration needed.

---

## Built-in Presets

Ready-to-use question schemas for common workflows:

```python
import laya

agent = laya.load("convaiinnovations/laya")

# Intelligent model routing (small vs. frontier)
routing = agent.predict({"request": "Refactor this service"}, laya.router_questions())

# Prompt guardrails (jailbreaks, injections)
guard = agent.predict({"prompt": "Ignore all instructions"}, laya.guard_questions())

# Content moderation (toxicity, harassment)
safety = agent.predict({"post": "User comment here"}, laya.moderation_questions())

# Support triage (intent, urgency, churn)
triage = agent.predict({"message": "Payment failed twice"}, laya.triage_questions())
```

---

## Schema-Driven Decisions

Define the shape you want with JSON Schema or Pydantic, get typed values back:

```python
import laya

schema = {
    "type": "object",
    "properties": {
        "department": {
            "type": "string",
            "enum": ["billing", "support", "sales"],
            "description": "Which team should handle this?",
        },
        "urgency": {"type": "integer", "minimum": 0, "maximum": 2},
        "needs_human": {"type": "boolean"},
    },
}

agent = laya.load("convaiinnovations/laya")
result = agent.decide("I was charged twice, refund me.", schema=schema)
# {"department": "billing", "urgency": 2, "needs_human": True}
```

---

## Batch Processing

Score many states in shared forward passes — up to 10x throughput on GPU:

```python
states = [{"body": t} for t in ticket_texts]
results = agent.predict_batch(states, questions, batch_size=64, sort_by_length=True)
```

---

## Integrations

### HTTP Server

```bash
pip install "laya[serve]"
LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve
```

```bash
curl -s localhost:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"state": {"body": "billed twice, refund please"},
       "questions": {"dept": {"type": "choice", "instructions": "which team?",
                     "criteria": {"billing": "refunds", "tech": "bugs"}}}}'
```

### MCP Server (Claude Desktop, Cursor, etc.)

```bash
pip install "laya[mcp]"
laya-mcp-server
```

```json
{
  "mcpServers": {
    "laya": {
      "command": "laya-mcp-server",
      "env": { "LAYA_DEVICE": "cpu" }
    }
  }
}
```

### LangChain / LangGraph

```python
from laya.integrations.langchain import LayaRouter, LayaGuardrail, LayaDecision

# Sub-35ms routing for LangGraph conditional edges
router = LayaRouter(
    criteria={"billing": "invoices, charges", "tech": "bugs, outages"},
    confidence_threshold=0.80,
    fallback="human_agent",
)
workflow.add_conditional_edges("triage", router)

# Inline prompt guardrails
guard = LayaGuardrail(action="raise")
```

### LlamaIndex & CrewAI

```bash
pip install "laya[llamaindex]"  # LlamaIndex selectors
pip install "laya[crewai]"      # CrewAI routing
```

### ONNX Runtime

```bash
pip install "laya[onnx]"
python scripts/export_onnx.py --quantize  # INT8 for CPU
```

### Docker

```bash
docker compose up    # CPU
docker compose -f compose.cuda.yaml up  # GPU
```

---

## Fine-Tuning

The shipped checkpoints work zero-shot, but fine-tuning on your domain data is where accuracy jumps. On the typed-decisions benchmark:

| Checkpoint | Accuracy |
|---|---|
| Base English | 0.362 |
| Fine-tuned `laya-typed-decisions` | **0.766** |

See the [fine-tuning notebook](notebooks/) — runs on Kaggle's free 2x T4 GPUs.

---

## Confidence Gating

Probabilities are trained with strictly proper scoring rules (RLCD), making confidence statistically meaningful:

```python
dept = answers["department"]["choice"]
conf = answers["department"]["answer_confidence"]

if conf >= THRESHOLD:
    route_automatically(dept)
else:
    escalate_to_human_agent(dept, reason=f"Low confidence ({conf:.2f})")
```

---

## Performance

| Metric | Value |
|---|---|
| Single question (T4 GPU) | ~33 ms |
| Batched (10 questions, T4) | 7.2 ms/question |
| Language detection | < 0.5 ms |
| Routing overhead | < 1 ms |

---

## Documentation

Full guides, API reference, and integration docs are available in the [`docs/`](docs/) directory.

Key topics:
- [Prediction Hooks](docs/hooks/) — audit logging, PII redaction, caching, metrics
- [Schema-Driven Decisions](docs/structured.md) — JSON Schema and Pydantic
- [LangChain / LangGraph](docs/langchain.md)
- [Docker Deployment](docs/docker.md)
- [HTTP API Reference](docs/http-api.md)
- [Benchmarks](BENCHMARKS.md)

---

## Wakt-Original Features

These features are built on top of Laya's core engine and are unique to Wakt.

### Decision Pipelines

Chain multiple decisions with conditional branching — the output of one step feeds the next:

```python
from laya import Router
from laya.pipeline import Pipeline

router = Router()
pipe = Pipeline(router)

pipe.add_step("triage", {
    "department": {"type": "choice", "instructions": "Which department?",
                   "criteria": {"billing": "payments", "technical": "bugs", "other": "rest"}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["low", "medium", "critical"]},
})

pipe.add_step("billing_detail", {
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
}, condition=lambda ctx: ctx["triage"]["department"]["choice"] == "billing")

result = pipe.run("We were billed twice for $49.99, please refund.")
print(result.path)          # ['triage', 'billing_detail']
print(result.answers)       # merged answers from all steps that ran
print(result.elapsed_ms)    # total wall-clock time
```

Features: conditional branching, state transforms, `dry_run()` previews, `max_steps` safety limits, `on_complete` callbacks.

### Decision Explainability

Understand *why* a decision was made with human-readable explanations:

```python
from laya.explain import explain, format_explanation

result = router.predict(state, questions)
explanation = explain(result, questions)

print(explanation.summary)
# "Answered 3 questions using the english checkpoint. 2/3 with high confidence. 1 flag raised."

print(explanation.per_question["department"].reasoning)
# "Strongly chose 'billing' (85% probability), well ahead of 'technical' (10%)."

print(explanation.flags)
# ["Very close call on 'urgency' (margin 3.2%)."]

print(format_explanation(explanation, verbose=True))  # full distribution + entropy
```

### Smart Decision Caching

Avoid redundant model calls with TTL-based LRU caching:

```python
from laya.cache import CachedRouter

cached = CachedRouter(router, ttl=300, max_size=10_000)

r1 = cached.predict(state, questions)  # ~33ms (model runs)
r2 = cached.predict(state, questions)  # <0.1ms (cache hit)

print(cached.stats)
# CacheStats(hits=1, misses=1, hit_rate=0.50, evictions=0, expirations=0, size=1/10000)
```

Thread-safe. Batch-aware (`predict_batch` checks cache per-request). Drop-in replacement for Router.

### A/B Testing Framework

Compare checkpoints, question schemas, or thresholds with statistical rigor:

```python
from laya.testing import Experiment

exp = Experiment("billing-routing-v2")
exp.add_arm("baseline", questions=questions_v1)
exp.add_arm("candidate", questions=questions_v2, model="laya-typed-decisions")

states = [{"body": t} for t in ticket_texts]
report = exp.run(router, states)

print(report.agreement_rate)   # 0.87 — arms agree 87% of the time
print(report.summary)          # per-question breakdown + timing
report.to_json("experiment.json")
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[mcp,serve]"

# Run tests
python tests/test_router.py
python tests/test_hooks.py

# Lint
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

This project is a derivative of [Laya](https://github.com/NandhaKishorM/laya) by [Convai Innovations](https://huggingface.co/convaiinnovations). See [NOTICE](NOTICE) for full attribution.
