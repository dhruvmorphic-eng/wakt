"""Decision pipelines — chain, branch, and compose multiple decisions.

Laya answers one set of questions at a time. A Pipeline lets you chain decisions
together: the output of one step feeds the next, with conditional branching so
different inputs take different paths.

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
        "amount_mentioned": {"type": "noul", "instructions": "Is a dollar amount mentioned?"},
    }, condition=lambda ctx: ctx["triage"]["department"]["choice"] == "billing")

    pipe.add_step("tech_detail", {
        "severity": {"type": "score", "instructions": "How severe is the technical issue?",
                     "criteria": ["cosmetic", "degraded", "down"]},
    }, condition=lambda ctx: ctx["triage"]["department"]["choice"] == "technical")

    result = pipe.run("We were billed twice for $49.99, please refund.")
    print(result.answers)       # merged answers from all steps that ran
    print(result.path)          # ['triage', 'billing_detail']
    print(result.elapsed_ms)    # total wall-clock time
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StepResult:
    """Result from a single pipeline step."""
    name: str
    answers: dict[str, Any]
    routing: dict[str, Any] | None = None
    elapsed_ms: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class PipelineResult:
    """Aggregated result from running a full pipeline."""
    answers: dict[str, Any]
    path: list[str]
    steps: list[StepResult]
    elapsed_ms: float = 0.0
    state: Any = None

    @property
    def skipped_steps(self) -> list[str]:
        return [s.name for s in self.steps if s.skipped]

    def get_step(self, name: str) -> StepResult | None:
        for s in self.steps:
            if s.name == name:
                return s
        return None


@dataclass
class _Step:
    name: str
    questions: dict[str, Any]
    condition: Callable[[dict], bool] | None = None
    transform_state: Callable[[Any, dict], Any] | None = None
    model: str | None = None
    on_complete: Callable[[StepResult, dict], None] | None = None


class Pipeline:
    """Compose multiple decision steps into a pipeline with conditional branching.

    Parameters
    ----------
    runner : Router or Agent
        The Laya Router or Agent instance that runs predictions.
    default_model : str, optional
        Default model override for all steps (can be overridden per step).
    max_steps : int
        Safety limit on the number of steps that can execute (default 50).
    """

    def __init__(self, runner, *, default_model: str | None = None, max_steps: int = 50):
        self._runner = runner
        self._default_model = default_model
        self._max_steps = max_steps
        self._steps: list[_Step] = []
        self._step_names: set[str] = set()

    def add_step(
        self,
        name: str,
        questions: dict[str, Any],
        *,
        condition: Callable[[dict], bool] | None = None,
        transform_state: Callable[[Any, dict], Any] | None = None,
        model: str | None = None,
        on_complete: Callable[[StepResult, dict], None] | None = None,
    ) -> "Pipeline":
        """Add a decision step to the pipeline.

        Parameters
        ----------
        name : str
            Unique name for this step.
        questions : dict
            Question schema passed to predict().
        condition : callable, optional
            A function receiving the accumulated answers dict. The step runs
            only when this returns True. If None, the step always runs.
        transform_state : callable, optional
            A function (state, answers) -> new_state that transforms the input
            state before this step runs. Useful for enriching context.
        model : str, optional
            Force a specific checkpoint for this step.
        on_complete : callable, optional
            Callback invoked after this step completes, receiving (StepResult, answers).
        """
        if name in self._step_names:
            raise ValueError("duplicate step name: %r" % name)
        if not questions:
            raise ValueError("step %r has no questions" % name)
        self._step_names.add(name)
        self._steps.append(_Step(
            name=name,
            questions=questions,
            condition=condition,
            transform_state=transform_state,
            model=model,
            on_complete=on_complete,
        ))
        return self

    def run(self, state: Any, **predict_kwargs) -> PipelineResult:
        """Execute the pipeline on the given state.

        Each step whose condition passes (or has no condition) runs in order.
        Answers accumulate across steps and are available to later conditions.

        Parameters
        ----------
        state : str or dict
            The input state passed to predict().
        **predict_kwargs
            Extra keyword arguments forwarded to every predict() call
            (e.g. max_len, hooks).

        Returns
        -------
        PipelineResult
            Aggregated answers, the execution path, and per-step details.
        """
        t0 = time.perf_counter()
        accumulated: dict[str, dict[str, Any]] = {}
        path: list[str] = []
        step_results: list[StepResult] = []
        current_state = state
        executed = 0

        for step in self._steps:
            if executed >= self._max_steps:
                step_results.append(StepResult(
                    name=step.name, answers={}, skipped=True,
                    skip_reason="max_steps (%d) reached" % self._max_steps,
                ))
                continue

            if step.condition is not None:
                try:
                    should_run = step.condition(accumulated)
                except Exception as exc:
                    step_results.append(StepResult(
                        name=step.name, answers={}, skipped=True,
                        skip_reason="condition raised: %s" % exc,
                    ))
                    continue
                if not should_run:
                    step_results.append(StepResult(
                        name=step.name, answers={}, skipped=True,
                        skip_reason="condition returned False",
                    ))
                    continue

            if step.transform_state is not None:
                current_state = step.transform_state(current_state, accumulated)

            kwargs = dict(predict_kwargs)
            model = step.model or self._default_model
            if model is not None:
                kwargs["model"] = model

            ts = time.perf_counter()
            result = self._runner.predict(current_state, step.questions, **kwargs)
            te = time.perf_counter()

            answers = result.get("answers", {})
            sr = StepResult(
                name=step.name,
                answers=answers,
                routing=result.get("routing"),
                elapsed_ms=(te - ts) * 1000,
            )

            accumulated[step.name] = answers
            path.append(step.name)
            step_results.append(sr)
            executed += 1

            if step.on_complete is not None:
                step.on_complete(sr, accumulated)

        merged: dict[str, Any] = {}
        for name in path:
            merged.update(accumulated[name])

        return PipelineResult(
            answers=merged,
            path=path,
            steps=step_results,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            state=current_state,
        )

    def dry_run(self, state: Any) -> list[dict[str, Any]]:
        """Preview which steps would run without executing any predictions.

        Returns a list of dicts with 'name', 'would_run', and 'reason' for each step.
        Conditions that depend on previous answers are evaluated with empty dicts.
        """
        preview = []
        dummy: dict[str, dict] = {}
        for step in self._steps:
            if step.condition is None:
                preview.append({"name": step.name, "would_run": True, "reason": "no condition"})
                dummy[step.name] = {}
            else:
                try:
                    result = step.condition(dummy)
                    preview.append({
                        "name": step.name,
                        "would_run": result,
                        "reason": "condition returned %s (evaluated with empty prior answers)" % result,
                    })
                    if result:
                        dummy[step.name] = {}
                except Exception as exc:
                    preview.append({
                        "name": step.name,
                        "would_run": "unknown",
                        "reason": "condition raised: %s" % exc,
                    })
        return preview

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        names = [s.name for s in self._steps]
        return "Pipeline(%s)" % " -> ".join(names)
