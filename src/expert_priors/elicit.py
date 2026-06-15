"""Prior elicitation — turn a dataset + context into a :class:`PriorBundle`.

High-level entry point :func:`elicit_priors` plus the :class:`PriorElicitor`
class that holds provider/config state and runs the sampling loop.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .bundle import PriorBundle
from .context import FeatureContext
from .prompts import build_system_prompt, build_user_prompt
from .providers import LLMClient
from .summaries import feature_summary_block

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

_DEFAULT_MODEL = "claude-opus-4-8"


class PriorElicitor:
    """Elicit coefficient priors from an LLM for one task/model configuration.

    Parameters
    ----------
    model:
        LLM id. Defaults to the latest Claude model.
    task:
        ``"regression"`` or ``"classification"``.
    n_samples:
        Number of independent prior draws to request (each a full coefficient vector).
    temperature:
        Sampling temperature for the draws (thesis default 0.8).
    system_prompt, prompt:
        Optional overrides. If ``prompt`` is given it fully replaces the generated
        user prompt (the context's scale block is still appended unless you also
        pass ``feature_summaries=""``); use it to bring your own wording.
    max_workers:
        Parallel in-flight requests.
    retries:
        Per-sample retry attempts on transient failures.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        task: str = "regression",
        *,
        n_samples: int = 50,
        temperature: float = 0.8,
        system_prompt: str | None = None,
        prompt: str | None = None,
        max_workers: int = 8,
        retries: int = 2,
        client: LLMClient | None = None,
    ) -> None:
        self.model = model
        self.task = task
        self.n_samples = n_samples
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.max_workers = max_workers
        self.retries = retries
        self.client = client or LLMClient(model)

    def render_prompts(self, context: FeatureContext, feature_summaries: str = "") -> tuple[str, str]:
        """Return ``(system_prompt, user_prompt)`` for this configuration."""
        system = self.system_prompt or build_system_prompt(self.task, sample_mode=self.n_samples > 1)
        if self.prompt is not None:
            user = self.prompt
        else:
            user = build_user_prompt(context, feature_summaries, self.task)
        return system, user

    def _one_call(self, system: str, user: str, feature_names) -> dict[str, float]:
        last_error: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                return self.client.complete_coefficients(
                    system, user, feature_names, temperature=self.temperature
                )
            except Exception as exc:  # transient API/parse errors -> retry
                last_error = exc
        raise RuntimeError(f"elicitation failed after {self.retries + 1} attempts: {last_error}")

    def elicit(self, context: FeatureContext, feature_summaries: str = "") -> PriorBundle:
        """Run the sampling loop and return a :class:`PriorBundle`."""
        feature_names = context.feature_names
        system, user = self.render_prompts(context, feature_summaries)

        rows: list[list[float]] = []
        if self.n_samples == 1 or self.max_workers <= 1:
            for _ in range(self.n_samples):
                coeffs = self._one_call(system, user, feature_names)
                rows.append([coeffs[f] for f in feature_names])
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [
                    pool.submit(self._one_call, system, user, feature_names)
                    for _ in range(self.n_samples)
                ]
                for fut in as_completed(futures):
                    coeffs = fut.result()
                    rows.append([coeffs[f] for f in feature_names])

        return PriorBundle(
            feature_names=list(feature_names),
            samples=np.asarray(rows, dtype=float),
            task=self.task,
            model=self.model,
            metadata={
                "temperature": self.temperature,
                "n_samples_requested": self.n_samples,
                "system_prompt": system,
                "user_prompt": user,
            },
        )


def elicit_priors(
    X: "pd.DataFrame",
    y=None,
    *,
    task: str = "regression",
    model: str = _DEFAULT_MODEL,
    n_samples: int = 50,
    temperature: float = 0.8,
    context: FeatureContext | None = None,
    auto_context: bool = False,
    prompt: str | None = None,
    cache: str | Path | None = None,
    max_workers: int = 8,
) -> PriorBundle:
    """Elicit coefficient priors for a dataset.

    Parameters
    ----------
    X:
        Feature dataframe (predictors only).
    y:
        Optional target — only used to phrase context toward the outcome.
    task:
        ``"regression"`` or ``"classification"``.
    model:
        LLM id. Defaults to the latest Claude model.
    n_samples, temperature:
        Number of prior draws and sampling temperature.
    context:
        Explicit :class:`FeatureContext`. If omitted and ``auto_context`` is
        True, one is generated with the LLM; otherwise a blank context is used.
    auto_context:
        Let the LLM draft the feature context from the data (always review it).
    prompt:
        Fully custom user prompt; bypasses context wording.
    cache:
        If given, load the bundle from this path when it exists, else save to it
        after eliciting — so reruns and demos need no API call.
    max_workers:
        Parallel in-flight requests during sampling.

    Returns
    -------
    PriorBundle
    """
    if cache is not None and Path(cache).exists():
        return PriorBundle.load(cache)

    target_name = getattr(y, "name", None)
    if context is None:
        if auto_context:
            from .context import generate_feature_context

            context = generate_feature_context(X, target=target_name, model=model)
        else:
            context = FeatureContext.blank(list(X.columns), target=target_name)

    summaries = feature_summary_block(X, context.feature_names) if hasattr(X, "columns") else ""

    elicitor = PriorElicitor(
        model=model,
        task=task,
        n_samples=n_samples,
        temperature=temperature,
        prompt=prompt,
        max_workers=max_workers,
    )
    bundle = elicitor.elicit(context, feature_summaries=summaries)

    if cache is not None:
        bundle.save(cache)
    return bundle
