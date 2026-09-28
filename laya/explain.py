"""Decision explainability — understand why a decision was made.

Provides human-readable explanations and structured breakdowns of any
predict() result, including confidence analysis and comparative reasoning.

    from laya import Router
    from laya.explain import explain, format_explanation

    router = Router()
    result = router.predict(state, questions)

    explanation = explain(result, questions)
    print(explanation.summary)
    print(explanation.per_question["department"].reasoning)
    print(format_explanation(explanation))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionExplanation:
    """Explanation for a single question's answer."""
    question_id: str
    question_type: str
    answer: Any
    confidence: float
    answer_confidence: float | None
    reasoning: str
    distribution: dict[str, float] | None = None
    margin: float | None = None
    entropy: float | None = None
    confidence_level: str = "unknown"
    alternatives: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Explanation:
    """Full explanation of a prediction result."""
    summary: str
    per_question: dict[str, QuestionExplanation]
    routing: dict[str, Any] | None = None
    model_used: str | None = None
    overall_confidence: float = 0.0
    flags: list[str] = field(default_factory=list)


def _entropy(probs: list[float]) -> float:
    """Shannon entropy of a probability distribution."""
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h


def _max_entropy(n: int) -> float:
    if n <= 1:
        return 0.0
    return math.log2(n)


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.90:
        return "very high"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.50:
        return "moderate"
    if confidence >= 0.25:
        return "low"
    return "very low"


def _explain_choice(qid: str, answer: dict[str, Any], question: dict[str, Any]) -> QuestionExplanation:
    choice = answer.get("choice", "?")
    probs = answer.get("probabilities", {})
    confidence = answer.get("confidence", 0.0)
    answer_conf = answer.get("answer_confidence")

    sorted_options = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    margin = 0.0
    if len(sorted_options) >= 2:
        margin = sorted_options[0][1] - sorted_options[1][1]

    prob_values = list(probs.values()) if probs else [1.0]
    ent = _entropy(prob_values)
    max_ent = _max_entropy(len(prob_values))

    alternatives = []
    for label, prob in sorted_options[1:3]:
        desc = (question.get("criteria") or {}).get(label, "")
        alternatives.append({"label": label, "probability": prob, "description": desc})

    if margin > 0.5:
        reasoning = "Strongly chose '%s' (%.0f%% probability), well ahead of the next option '%s' (%.0f%%)." % (
            choice, sorted_options[0][1] * 100,
            sorted_options[1][0] if len(sorted_options) > 1 else "none",
            sorted_options[1][1] * 100 if len(sorted_options) > 1 else 0,
        )
    elif margin > 0.15:
        reasoning = "Chose '%s' (%.0f%%) over '%s' (%.0f%%) with a clear but not overwhelming margin." % (
            choice, sorted_options[0][1] * 100,
            sorted_options[1][0] if len(sorted_options) > 1 else "none",
            sorted_options[1][1] * 100 if len(sorted_options) > 1 else 0,
        )
    else:
        reasoning = "Close call: '%s' (%.0f%%) barely edges out '%s' (%.0f%%). Consider reviewing manually." % (
            choice, sorted_options[0][1] * 100,
            sorted_options[1][0] if len(sorted_options) > 1 else "none",
            sorted_options[1][1] * 100 if len(sorted_options) > 1 else 0,
        )

    conf = answer_conf if answer_conf is not None else confidence
    return QuestionExplanation(
        question_id=qid,
        question_type="choice",
        answer=choice,
        confidence=confidence,
        answer_confidence=answer_conf,
        reasoning=reasoning,
        distribution=probs,
        margin=margin,
        entropy=ent,
        confidence_level=_confidence_level(conf),
        alternatives=alternatives,
    )


def _explain_score(qid: str, answer: dict[str, Any], question: dict[str, Any]) -> QuestionExplanation:
    score = answer.get("score", 0.0)
    legend = answer.get("legend", {})
    probs = answer.get("probabilities", {})
    confidence = answer.get("confidence", 0.0)
    answer_conf = answer.get("answer_confidence")

    criteria = question.get("criteria", [])
    max_score = max(1.0, float(len(criteria) - 1)) if criteria else 1.0
    normalized = score / max_score if max_score > 0 else 0.0

    if normalized >= 0.75:
        level_desc = "high"
    elif normalized >= 0.4:
        level_desc = "moderate"
    else:
        level_desc = "low"

    top_label = criteria[round(score)] if criteria and round(score) < len(criteria) else "level %s" % round(score)
    reasoning = "Scored %.2f / %.1f (%s). Most likely level: '%s'." % (score, max_score, level_desc, top_label)

    prob_values = list(probs.values()) if probs else [1.0]
    ent = _entropy(prob_values)

    conf = answer_conf if answer_conf is not None else confidence
    return QuestionExplanation(
        question_id=qid,
        question_type="score",
        answer=score,
        confidence=confidence,
        answer_confidence=answer_conf,
        reasoning=reasoning,
        distribution=probs,
        entropy=ent,
        confidence_level=_confidence_level(conf),
    )


def _explain_noul(qid: str, answer: dict[str, Any], question: dict[str, Any]) -> QuestionExplanation:
    prob = answer.get("noul", 0.5)
    confidence = answer.get("confidence", 0.0)
    answer_conf = answer.get("answer_confidence")

    if prob >= 0.85:
        reasoning = "Very likely yes (%.0f%% probability)." % (prob * 100)
    elif prob >= 0.65:
        reasoning = "Likely yes (%.0f%%), but some uncertainty remains." % (prob * 100)
    elif prob >= 0.35:
        reasoning = "Uncertain — %.0f%% probability. Near the decision boundary, manual review recommended." % (prob * 100)
    elif prob >= 0.15:
        reasoning = "Likely no (%.0f%% probability of yes)." % (prob * 100)
    else:
        reasoning = "Very likely no (only %.0f%% probability of yes)." % (prob * 100)

    ent = _entropy([1 - prob, prob]) if 0 < prob < 1 else 0.0

    conf = answer_conf if answer_conf is not None else confidence
    return QuestionExplanation(
        question_id=qid,
        question_type="noul",
        answer=prob,
        confidence=confidence,
        answer_confidence=answer_conf,
        reasoning=reasoning,
        distribution={"false": 1 - prob, "true": prob},
        entropy=ent,
        confidence_level=_confidence_level(conf),
    )


_EXPLAINERS = {
    "choice": _explain_choice,
    "score": _explain_score,
    "noul": _explain_noul,
}


def explain(result: dict[str, Any], questions: dict[str, Any]) -> Explanation:
    """Generate a structured explanation of a prediction result.

    Parameters
    ----------
    result : dict
        The output from predict() or Router.predict().
    questions : dict
        The question schema that was passed to predict().

    Returns
    -------
    Explanation
        Human-readable explanations with per-question breakdowns.
    """
    answers = result.get("answers", {})
    routing = result.get("routing")
    per_question: dict[str, QuestionExplanation] = {}
    confidences = []
    flags: list[str] = []

    for qid, qdef in questions.items():
        qtype = qdef.get("type", "choice")
        ans = answers.get(qid, {})

        explainer = _EXPLAINERS.get(qtype)
        if explainer is None:
            per_question[qid] = QuestionExplanation(
                question_id=qid, question_type=qtype,
                answer=ans, confidence=0.0, answer_confidence=None,
                reasoning="Unknown question type: %s" % qtype,
            )
            continue

        qe = explainer(qid, ans, qdef)
        per_question[qid] = qe

        conf = qe.answer_confidence if qe.answer_confidence is not None else qe.confidence
        confidences.append(conf)

        if qe.confidence_level in ("low", "very low"):
            flags.append("Low confidence on '%s' — consider manual review." % qid)

        if qe.margin is not None and qe.margin < 0.10 and qtype == "choice":
            flags.append("Very close call on '%s' (margin %.1f%%)." % (qid, qe.margin * 100))

    overall = sum(confidences) / len(confidences) if confidences else 0.0
    model_used = routing.get("model") if routing else None

    n_questions = len(per_question)
    n_high = sum(1 for q in per_question.values() if q.confidence_level in ("high", "very high"))

    summary = "Answered %d question%s" % (n_questions, "s" if n_questions != 1 else "")
    if model_used:
        summary += " using the %s checkpoint" % model_used
    summary += ". %d/%d with high confidence." % (n_high, n_questions)
    if flags:
        summary += " %d flag%s raised." % (len(flags), "s" if len(flags) != 1 else "")

    return Explanation(
        summary=summary,
        per_question=per_question,
        routing=routing,
        model_used=model_used,
        overall_confidence=overall,
        flags=flags,
    )


def format_explanation(exp: Explanation, *, verbose: bool = False) -> str:
    """Format an Explanation as a human-readable string.

    Parameters
    ----------
    exp : Explanation
        The explanation to format.
    verbose : bool
        If True, include probability distributions and entropy.
    """
    lines = [exp.summary, ""]

    for qid, qe in exp.per_question.items():
        lines.append("  %s [%s]: %s" % (qid, qe.question_type, qe.reasoning))

        if verbose and qe.distribution:
            dist_str = ", ".join("%s: %.1f%%" % (k, v * 100) for k, v in sorted(qe.distribution.items(), key=lambda x: -x[1]))
            lines.append("    distribution: %s" % dist_str)
            if qe.entropy is not None:
                lines.append("    entropy: %.3f bits" % qe.entropy)

        if qe.alternatives:
            alt_str = ", ".join("'%s' (%.0f%%)" % (a["label"], a["probability"] * 100) for a in qe.alternatives)
            lines.append("    alternatives: %s" % alt_str)

    if exp.flags:
        lines.append("")
        lines.append("Flags:")
        for f in exp.flags:
            lines.append("  - %s" % f)

    return "\n".join(lines)
