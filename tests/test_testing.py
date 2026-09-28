"""Tests for laya.testing — A/B experiment framework."""

import json
import os
import tempfile
from laya.testing import Experiment, ExperimentReport


class _MockRouter:
    def __init__(self, answers_map=None):
        self._answers_map = answers_map or {}
        self._call_count = 0

    def predict(self, state, questions, **kwargs):
        self._call_count += 1
        answers = {}
        for qid, qdef in questions.items():
            qtype = qdef.get("type", "choice")
            if qid in self._answers_map:
                answers[qid] = self._answers_map[qid]
            elif qtype == "choice":
                labels = list((qdef.get("criteria") or {}).keys())
                answers[qid] = {"choice": labels[0] if labels else "a", "confidence": 0.9, "probabilities": {}}
            elif qtype == "score":
                answers[qid] = {"score": 1.0, "confidence": 0.8}
            elif qtype == "noul":
                answers[qid] = {"noul": 0.85, "confidence": 0.9}
        return {"answers": answers, "routing": {"model": kwargs.get("model", "english")}}


QUESTIONS_V1 = {
    "dept": {
        "type": "choice",
        "instructions": "Which department?",
        "criteria": {"billing": "payments", "technical": "bugs"},
    },
}

QUESTIONS_V2 = {
    "dept": {
        "type": "choice",
        "instructions": "Which department handles this?",
        "criteria": {"billing": "payments", "technical": "bugs", "other": "rest"},
    },
}


def test_basic_experiment():
    router = _MockRouter()
    exp = Experiment("test-exp")
    exp.add_arm("baseline", questions=QUESTIONS_V1)
    exp.add_arm("candidate", questions=QUESTIONS_V2)

    states = ["ticket 1", "ticket 2", "ticket 3"]
    report = exp.run(router, states)

    assert isinstance(report, ExperimentReport)
    assert report.name == "test-exp"
    assert report.arms == ["baseline", "candidate"]
    assert report.n_states == 3
    assert report.elapsed_ms > 0
    assert 0.0 <= report.agreement_rate <= 1.0


def test_full_agreement():
    router = _MockRouter()
    exp = Experiment("agree")
    exp.add_arm("a", questions=QUESTIONS_V1)
    exp.add_arm("b", questions=QUESTIONS_V1)

    report = exp.run(router, ["test"])
    assert report.agreement_rate == 1.0
    assert len(report.disagreements) == 0


def test_disagreement_detected():
    class _SplitRouter:
        def __init__(self):
            self._call_count = 0

        def predict(self, state, questions, **kwargs):
            self._call_count += 1
            answers = {}
            for qid, qdef in questions.items():
                labels = list((qdef.get("criteria") or {}).keys())
                if self._call_count % 2 == 1:
                    answers[qid] = {"choice": labels[0], "confidence": 0.9, "probabilities": {}}
                else:
                    answers[qid] = {"choice": labels[-1], "confidence": 0.9, "probabilities": {}}
            return {"answers": answers, "routing": {"model": "english"}}

    router = _SplitRouter()
    exp = Experiment("disagree")
    exp.add_arm("a", questions=QUESTIONS_V1)
    exp.add_arm("b", questions=QUESTIONS_V1)

    report = exp.run(router, ["test"])
    assert report.agreement_rate < 1.0
    assert len(report.disagreements) > 0


def test_per_question_breakdown():
    router = _MockRouter()
    exp = Experiment("breakdown")
    exp.add_arm("a", questions=QUESTIONS_V1)
    exp.add_arm("b", questions=QUESTIONS_V1)

    report = exp.run(router, ["test1", "test2"])
    assert "dept" in report.per_question
    qc = report.per_question["dept"]
    assert qc.question_type == "choice"
    assert 0.0 <= qc.agreement_rate <= 1.0


def test_timing():
    router = _MockRouter()
    exp = Experiment("timing")
    exp.add_arm("a", questions=QUESTIONS_V1)
    exp.add_arm("b", questions=QUESTIONS_V1)

    report = exp.run(router, ["test"])
    assert "a" in report.timing
    assert "b" in report.timing
    assert report.timing["a"] >= 0
    assert report.timing["b"] >= 0


def test_summary_string():
    router = _MockRouter()
    exp = Experiment("summary")
    exp.add_arm("a", questions=QUESTIONS_V1)
    exp.add_arm("b", questions=QUESTIONS_V1)

    report = exp.run(router, ["test"])
    summary = report.summary
    assert "summary" in summary.lower() or "Experiment" in summary
    assert "a" in summary
    assert "b" in summary


def test_to_json():
    router = _MockRouter()
    exp = Experiment("json-test")
    exp.add_arm("a", questions=QUESTIONS_V1)
    exp.add_arm("b", questions=QUESTIONS_V1)

    report = exp.run(router, ["test"])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name

    try:
        report.to_json(path)
        with open(path) as f:
            data = json.load(f)
        assert data["name"] == "json-test"
        assert data["arms"] == ["a", "b"]
        assert data["n_states"] == 1
        assert "agreement_rate" in data
    finally:
        os.unlink(path)


def test_duplicate_arm_raises():
    exp = Experiment("dup")
    exp.add_arm("a", questions=QUESTIONS_V1)
    try:
        exp.add_arm("a", questions=QUESTIONS_V2)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "duplicate" in str(e).lower()


def test_less_than_two_arms_raises():
    router = _MockRouter()
    exp = Experiment("one-arm")
    exp.add_arm("only", questions=QUESTIONS_V1)
    try:
        exp.run(router, ["test"])
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "at least 2" in str(e)


def test_model_override_per_arm():
    router = _MockRouter()
    exp = Experiment("model-test")
    exp.add_arm("english", questions=QUESTIONS_V1, model="english")
    exp.add_arm("multilingual", questions=QUESTIONS_V1, model="multilingual")

    report = exp.run(router, ["test"])
    assert report.n_states == 1


def test_chaining():
    exp = Experiment("chain")
    result = exp.add_arm("a", questions=QUESTIONS_V1).add_arm("b", questions=QUESTIONS_V2)
    assert result is exp


def test_repr():
    exp = Experiment("my-exp")
    exp.add_arm("a", questions=QUESTIONS_V1)
    exp.add_arm("b", questions=QUESTIONS_V2)
    r = repr(exp)
    assert "my-exp" in r
    assert "a" in r
    assert "b" in r


if __name__ == "__main__":
    test_basic_experiment()
    test_full_agreement()
    test_disagreement_detected()
    test_per_question_breakdown()
    test_timing()
    test_summary_string()
    test_to_json()
    test_duplicate_arm_raises()
    test_less_than_two_arms_raises()
    test_model_override_per_arm()
    test_chaining()
    test_repr()
    print("All testing tests passed.")
