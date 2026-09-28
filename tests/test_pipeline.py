"""Tests for laya.pipeline — decision chaining and conditional branching."""

from laya.pipeline import Pipeline, PipelineResult, StepResult


class _MockRouter:
    """Fake router that returns deterministic answers without loading models."""

    def __init__(self, answers_by_step=None):
        self._answers = answers_by_step or {}
        self._call_count = 0

    def predict(self, state, questions, **kwargs):
        self._call_count += 1
        answers = {}
        for qid, qdef in questions.items():
            qtype = qdef.get("type", "choice")
            if qid in self._answers:
                answers[qid] = self._answers[qid]
            elif qtype == "choice":
                labels = list((qdef.get("criteria") or {}).keys())
                answers[qid] = {"choice": labels[0] if labels else "a", "confidence": 0.9, "probabilities": {}}
            elif qtype == "score":
                answers[qid] = {"score": 1.0, "confidence": 0.8}
            elif qtype == "noul":
                answers[qid] = {"noul": 0.85, "confidence": 0.9}
        return {"answers": answers, "routing": {"model": "english"}}


TRIAGE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department?",
        "criteria": {"billing": "payments", "technical": "bugs", "other": "rest"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["low", "medium", "critical"],
    },
}

BILLING_QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
}

TECH_QUESTIONS = {
    "severity": {
        "type": "score",
        "instructions": "How severe?",
        "criteria": ["cosmetic", "degraded", "down"],
    },
}


def test_basic_pipeline():
    router = _MockRouter()
    pipe = Pipeline(router)
    pipe.add_step("triage", TRIAGE_QUESTIONS)

    result = pipe.run("test input")
    assert isinstance(result, PipelineResult)
    assert result.path == ["triage"]
    assert "department" in result.answers
    assert "urgency" in result.answers
    assert result.elapsed_ms > 0
    assert len(result.steps) == 1
    assert not result.steps[0].skipped


def test_conditional_branching():
    router = _MockRouter(answers_by_step={
        "department": {"choice": "billing", "confidence": 0.95, "probabilities": {}},
    })
    pipe = Pipeline(router)
    pipe.add_step("triage", TRIAGE_QUESTIONS)
    pipe.add_step("billing_detail", BILLING_QUESTIONS,
                  condition=lambda ctx: ctx.get("triage", {}).get("department", {}).get("choice") == "billing")
    pipe.add_step("tech_detail", TECH_QUESTIONS,
                  condition=lambda ctx: ctx.get("triage", {}).get("department", {}).get("choice") == "technical")

    result = pipe.run("billed twice, refund please")
    assert "triage" in result.path
    assert "billing_detail" in result.path
    assert "tech_detail" not in result.path
    assert "refund" in result.answers
    assert result.skipped_steps == ["tech_detail"]


def test_skipped_step_has_reason():
    router = _MockRouter()
    pipe = Pipeline(router)
    pipe.add_step("always", TRIAGE_QUESTIONS)
    pipe.add_step("never", BILLING_QUESTIONS, condition=lambda ctx: False)

    result = pipe.run("test")
    never_step = result.get_step("never")
    assert never_step is not None
    assert never_step.skipped
    assert never_step.skip_reason == "condition returned False"


def test_duplicate_step_name_raises():
    router = _MockRouter()
    pipe = Pipeline(router)
    pipe.add_step("triage", TRIAGE_QUESTIONS)
    try:
        pipe.add_step("triage", BILLING_QUESTIONS)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "duplicate" in str(e).lower()


def test_empty_questions_raises():
    router = _MockRouter()
    pipe = Pipeline(router)
    try:
        pipe.add_step("empty", {})
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "no questions" in str(e).lower()


def test_transform_state():
    router = _MockRouter()
    pipe = Pipeline(router)
    pipe.add_step("step1", TRIAGE_QUESTIONS)
    pipe.add_step("step2", BILLING_QUESTIONS,
                  transform_state=lambda state, answers: {"enriched": True, "body": state})

    result = pipe.run("test")
    assert len(result.path) == 2


def test_max_steps_limit():
    router = _MockRouter()
    pipe = Pipeline(router, max_steps=1)
    pipe.add_step("step1", TRIAGE_QUESTIONS)
    pipe.add_step("step2", BILLING_QUESTIONS)

    result = pipe.run("test")
    assert result.path == ["step1"]
    assert result.get_step("step2").skipped
    assert "max_steps" in result.get_step("step2").skip_reason


def test_dry_run():
    router = _MockRouter()
    pipe = Pipeline(router)
    pipe.add_step("always", TRIAGE_QUESTIONS)
    pipe.add_step("maybe", BILLING_QUESTIONS, condition=lambda ctx: False)

    preview = pipe.dry_run("test")
    assert len(preview) == 2
    assert preview[0]["would_run"] is True
    assert preview[1]["would_run"] is False


def test_repr():
    router = _MockRouter()
    pipe = Pipeline(router)
    pipe.add_step("a", TRIAGE_QUESTIONS)
    pipe.add_step("b", BILLING_QUESTIONS)
    assert repr(pipe) == "Pipeline(a -> b)"


def test_len():
    router = _MockRouter()
    pipe = Pipeline(router)
    assert len(pipe) == 0
    pipe.add_step("a", TRIAGE_QUESTIONS)
    assert len(pipe) == 1


def test_on_complete_callback():
    called = []
    router = _MockRouter()
    pipe = Pipeline(router)
    pipe.add_step("step1", TRIAGE_QUESTIONS,
                  on_complete=lambda sr, acc: called.append(sr.name))

    pipe.run("test")
    assert called == ["step1"]


def test_chaining_add_step():
    router = _MockRouter()
    pipe = Pipeline(router)
    result = pipe.add_step("a", TRIAGE_QUESTIONS).add_step("b", BILLING_QUESTIONS)
    assert result is pipe
    assert len(pipe) == 2


if __name__ == "__main__":
    test_basic_pipeline()
    test_conditional_branching()
    test_skipped_step_has_reason()
    test_duplicate_step_name_raises()
    test_empty_questions_raises()
    test_transform_state()
    test_max_steps_limit()
    test_dry_run()
    test_repr()
    test_len()
    test_on_complete_callback()
    test_chaining_add_step()
    print("All pipeline tests passed.")
