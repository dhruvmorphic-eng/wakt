"""Tests for laya.explain — decision explainability."""

from laya.explain import explain, format_explanation, Explanation, QuestionExplanation


CHOICE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department?",
        "criteria": {"billing": "payments", "technical": "bugs", "other": "rest"},
    },
}

SCORE_QUESTIONS = {
    "urgency": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["low", "medium", "critical"],
    },
}

NOUL_QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
}


def test_explain_choice_strong():
    result = {
        "answers": {
            "department": {
                "choice": "billing",
                "confidence": 0.95,
                "probabilities": {"billing": 0.85, "technical": 0.10, "other": 0.05},
            },
        },
        "routing": {"model": "english"},
    }
    exp = explain(result, CHOICE_QUESTIONS)
    assert isinstance(exp, Explanation)
    assert "department" in exp.per_question
    qe = exp.per_question["department"]
    assert qe.answer == "billing"
    assert qe.margin == 0.75
    assert "Strongly" in qe.reasoning
    assert qe.confidence_level == "very high"
    assert exp.model_used == "english"
    assert len(exp.flags) == 0


def test_explain_choice_close_call():
    result = {
        "answers": {
            "department": {
                "choice": "billing",
                "confidence": 0.4,
                "probabilities": {"billing": 0.35, "technical": 0.33, "other": 0.32},
            },
        },
        "routing": {"model": "english"},
    }
    exp = explain(result, CHOICE_QUESTIONS)
    qe = exp.per_question["department"]
    assert qe.margin < 0.10
    assert "Close call" in qe.reasoning


def test_explain_score():
    result = {
        "answers": {
            "urgency": {
                "score": 2.0,
                "confidence": 0.8,
                "probabilities": {"0": 0.05, "1": 0.10, "2": 0.85},
            },
        },
        "routing": {"model": "english"},
    }
    exp = explain(result, SCORE_QUESTIONS)
    qe = exp.per_question["urgency"]
    assert qe.question_type == "score"
    assert qe.answer == 2.0


def test_explain_noul_yes():
    result = {
        "answers": {"refund": {"noul": 0.92, "confidence": 0.88}},
        "routing": {"model": "english"},
    }
    exp = explain(result, NOUL_QUESTIONS)
    qe = exp.per_question["refund"]
    assert "Very likely yes" in qe.reasoning


def test_explain_noul_uncertain():
    result = {
        "answers": {"refund": {"noul": 0.48, "confidence": 0.5}},
        "routing": {"model": "english"},
    }
    exp = explain(result, NOUL_QUESTIONS)
    qe = exp.per_question["refund"]
    assert "Uncertain" in qe.reasoning


def test_explain_noul_no():
    result = {
        "answers": {"refund": {"noul": 0.08, "confidence": 0.9}},
        "routing": {"model": "english"},
    }
    exp = explain(result, NOUL_QUESTIONS)
    qe = exp.per_question["refund"]
    assert "Very likely no" in qe.reasoning


def test_format_explanation_verbose():
    result = {
        "answers": {
            "department": {
                "choice": "billing",
                "confidence": 0.95,
                "probabilities": {"billing": 0.85, "technical": 0.10, "other": 0.05},
            },
        },
        "routing": {"model": "english"},
    }
    exp = explain(result, CHOICE_QUESTIONS)
    output = format_explanation(exp, verbose=True)
    assert "distribution" in output
    assert "entropy" in output


def test_overall_confidence():
    result = {
        "answers": {
            "department": {
                "choice": "billing",
                "confidence": 0.90,
                "probabilities": {"billing": 0.85, "technical": 0.10, "other": 0.05},
            },
        },
        "routing": {"model": "english"},
    }
    exp = explain(result, CHOICE_QUESTIONS)
    assert 0.0 <= exp.overall_confidence <= 1.0


def test_low_confidence_flag():
    result = {
        "answers": {
            "department": {
                "choice": "billing",
                "confidence": 0.15,
                "probabilities": {"billing": 0.35, "technical": 0.33, "other": 0.32},
            },
        },
        "routing": {"model": "english"},
    }
    exp = explain(result, CHOICE_QUESTIONS)
    assert any("low confidence" in f.lower() or "Low confidence" in f for f in exp.flags)


def test_alternatives():
    result = {
        "answers": {
            "department": {
                "choice": "billing",
                "confidence": 0.9,
                "probabilities": {"billing": 0.7, "technical": 0.2, "other": 0.1},
            },
        },
        "routing": {"model": "english"},
    }
    exp = explain(result, CHOICE_QUESTIONS)
    qe = exp.per_question["department"]
    assert len(qe.alternatives) >= 1
    assert qe.alternatives[0]["label"] == "technical"


if __name__ == "__main__":
    test_explain_choice_strong()
    test_explain_choice_close_call()
    test_explain_score()
    test_explain_noul_yes()
    test_explain_noul_uncertain()
    test_explain_noul_no()
    test_format_explanation_verbose()
    test_overall_confidence()
    test_low_confidence_flag()
    test_alternatives()
    print("All explain tests passed.")
