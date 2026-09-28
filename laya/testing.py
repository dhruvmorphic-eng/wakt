"""A/B testing framework for comparing decision configurations.

Compare checkpoints, question schemas, thresholds, or any predict()
parameters against each other with statistical rigor.

    from laya import Router
    from laya.testing import Experiment

    router = Router(preload=True)

    exp = Experiment("billing-routing-v2")
    exp.add_arm("baseline", questions=questions_v1)
    exp.add_arm("candidate", questions=questions_v2)

    states = [{"body": t} for t in ticket_texts]
    report = exp.run(router, states)

    print(report.summary)
    print(report.agreement_rate)      # how often the arms agree
    print(report.per_question)        # per-question breakdown
    report.to_json("experiment.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArmConfig:
    """Configuration for one arm of an experiment."""
    name: str
    questions: dict[str, Any]
    model: str | None = None
    predict_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionComparison:
    """Comparison of one question across all arms."""
    question_id: str
    question_type: str
    agreement_rate: float
    per_arm: dict[str, dict[str, Any]]
    disagreements: list[dict[str, Any]]


@dataclass
class StateComparison:
    """Comparison of all arms on a single state."""
    index: int
    state: Any
    per_arm: dict[str, dict[str, Any]]
    agrees: bool
    disagreed_questions: list[str]


@dataclass
class ExperimentReport:
    """Full report from running an experiment."""
    name: str
    arms: list[str]
    n_states: int
    agreement_rate: float
    per_question: dict[str, QuestionComparison]
    per_state: list[StateComparison]
    timing: dict[str, float]
    elapsed_ms: float

    @property
    def summary(self) -> str:
        lines = [
            "Experiment: %s" % self.name,
            "Arms: %s" % ", ".join(self.arms),
            "States: %d" % self.n_states,
            "Overall agreement: %.1f%%" % (self.agreement_rate * 100),
            "",
        ]
        for qid, qc in self.per_question.items():
            lines.append("  %s [%s]: %.1f%% agreement, %d disagreements" % (
                qid, qc.question_type, qc.agreement_rate * 100, len(qc.disagreements),
            ))
        lines.append("")
        for arm, ms in self.timing.items():
            lines.append("  %s: %.1f ms total (%.2f ms/state)" % (arm, ms, ms / max(1, self.n_states)))
        return "\n".join(lines)

    @property
    def disagreements(self) -> list[StateComparison]:
        return [s for s in self.per_state if not s.agrees]

    def to_json(self, path: str) -> None:
        """Write the report as JSON."""
        data = {
            "name": self.name,
            "arms": self.arms,
            "n_states": self.n_states,
            "agreement_rate": self.agreement_rate,
            "timing": self.timing,
            "elapsed_ms": self.elapsed_ms,
            "per_question": {},
            "disagreements": [],
        }
        for qid, qc in self.per_question.items():
            data["per_question"][qid] = {
                "type": qc.question_type,
                "agreement_rate": qc.agreement_rate,
                "n_disagreements": len(qc.disagreements),
            }
        for sc in self.per_state:
            if not sc.agrees:
                data["disagreements"].append({
                    "index": sc.index,
                    "state": str(sc.state)[:200],
                    "disagreed_questions": sc.disagreed_questions,
                    "per_arm": {name: {k: str(v)[:100] for k, v in arm.items()} for name, arm in sc.per_arm.items()},
                })

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


def _extract_answer_value(answer: dict, qtype: str) -> Any:
    if qtype == "choice":
        return answer.get("choice")
    elif qtype == "score":
        raw = answer.get("score", 0)
        return round(raw) if isinstance(raw, float) else raw
    elif qtype == "noul":
        prob = answer.get("noul", 0.5)
        return prob >= 0.5
    return None


class Experiment:
    """Compare two or more decision configurations on the same inputs.

    Parameters
    ----------
    name : str
        A descriptive name for this experiment.
    """

    def __init__(self, name: str):
        self.name = name
        self._arms: list[ArmConfig] = []
        self._arm_names: set[str] = set()

    def add_arm(
        self,
        name: str,
        questions: dict[str, Any],
        *,
        model: str | None = None,
        **predict_kwargs,
    ) -> "Experiment":
        """Add an experimental arm.

        Parameters
        ----------
        name : str
            Unique name for this arm (e.g. "baseline", "candidate").
        questions : dict
            The question schema for this arm.
        model : str, optional
            Force a specific checkpoint for this arm.
        **predict_kwargs
            Extra arguments passed to predict().
        """
        if name in self._arm_names:
            raise ValueError("duplicate arm name: %r" % name)
        self._arm_names.add(name)
        self._arms.append(ArmConfig(name=name, questions=questions, model=model, predict_kwargs=predict_kwargs))
        return self

    def run(self, runner, states: list[Any], **kwargs) -> ExperimentReport:
        """Run the experiment across all arms and states.

        Parameters
        ----------
        runner : Router or Agent
            The Laya Router or Agent to use.
        states : list
            List of states to evaluate.
        **kwargs
            Extra arguments forwarded to predict().

        Returns
        -------
        ExperimentReport
        """
        if len(self._arms) < 2:
            raise ValueError("need at least 2 arms to compare, got %d" % len(self._arms))

        t0 = time.perf_counter()
        arm_results: dict[str, list[dict]] = {}
        timing: dict[str, float] = {}

        for arm in self._arms:
            ta = time.perf_counter()
            results = []
            merged_kwargs = dict(kwargs)
            merged_kwargs.update(arm.predict_kwargs)
            if arm.model:
                merged_kwargs["model"] = arm.model

            for state in states:
                r = runner.predict(state, arm.questions, **merged_kwargs)
                results.append(r)
            arm_results[arm.name] = results
            timing[arm.name] = (time.perf_counter() - ta) * 1000

        all_qids: dict[str, str] = {}
        for arm in self._arms:
            for qid, qdef in arm.questions.items():
                if qid not in all_qids:
                    all_qids[qid] = qdef.get("type", "choice")

        per_state: list[StateComparison] = []
        q_agreements: dict[str, int] = {qid: 0 for qid in all_qids}
        q_total: dict[str, int] = {qid: 0 for qid in all_qids}
        q_disagreements: dict[str, list[dict]] = {qid: [] for qid in all_qids}

        for i, state in enumerate(states):
            arm_answers = {}
            disagreed = []

            for arm in self._arms:
                answers = arm_results[arm.name][i].get("answers", {})
                arm_answers[arm.name] = answers

            for qid, qtype in all_qids.items():
                values = {}
                for arm in self._arms:
                    ans = arm_answers.get(arm.name, {}).get(qid, {})
                    if ans:
                        values[arm.name] = _extract_answer_value(ans, qtype)

                if len(values) >= 2:
                    q_total[qid] += 1
                    unique_values = set(values.values())
                    if len(unique_values) == 1:
                        q_agreements[qid] += 1
                    else:
                        disagreed.append(qid)
                        q_disagreements[qid].append({
                            "state_index": i,
                            "values": dict(values),
                        })

            per_state.append(StateComparison(
                index=i,
                state=state,
                per_arm=arm_answers,
                agrees=len(disagreed) == 0,
                disagreed_questions=disagreed,
            ))

        per_question = {}
        for qid, qtype in all_qids.items():
            total = q_total[qid]
            agreed = q_agreements[qid]
            per_arm_summary = {}
            for arm in self._arms:
                arm_vals = [
                    _extract_answer_value(arm_results[arm.name][i].get("answers", {}).get(qid, {}), qtype)
                    for i in range(len(states))
                ]
                arm_vals = [v for v in arm_vals if v is not None]
                if qtype == "choice" and arm_vals:
                    from collections import Counter
                    counts = Counter(arm_vals)
                    per_arm_summary[arm.name] = {"distribution": dict(counts), "n": len(arm_vals)}
                elif qtype == "noul" and arm_vals:
                    yes_count = sum(1 for v in arm_vals if v)
                    per_arm_summary[arm.name] = {"yes_rate": yes_count / len(arm_vals), "n": len(arm_vals)}
                elif qtype == "score" and arm_vals:
                    avg = sum(arm_vals) / len(arm_vals)
                    per_arm_summary[arm.name] = {"mean": avg, "n": len(arm_vals)}

            per_question[qid] = QuestionComparison(
                question_id=qid,
                question_type=qtype,
                agreement_rate=agreed / total if total > 0 else 1.0,
                per_arm=per_arm_summary,
                disagreements=q_disagreements[qid],
            )

        n_agree = sum(1 for s in per_state if s.agrees)
        overall = n_agree / len(states) if states else 1.0

        return ExperimentReport(
            name=self.name,
            arms=[a.name for a in self._arms],
            n_states=len(states),
            agreement_rate=overall,
            per_question=per_question,
            per_state=per_state,
            timing=timing,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    def __repr__(self) -> str:
        return "Experiment(%r, arms=[%s])" % (self.name, ", ".join(a.name for a in self._arms))
