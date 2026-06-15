"""Prior elicitation — turn a dataset + context into a :class:`PriorBundle`.

High-level entry point :func:`elicit_priors` plus the :class:`PriorElicitor`
class that holds provider/config state for repeated calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .bundle import PriorBundle
from .context import FeatureContext

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
        user prompt (context is then ignored); use this to bring your own prompt.
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
    ) -> None:
        self.model = model
        self.task = task
        self.n_samples = n_samples
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.prompt = prompt

    def build_prompt(self, context: FeatureContext) -> str:
        """Render the user prompt from a context (or return the custom override)."""
        if self.prompt is not None:
            return self.prompt
        # TODO(port): lift the production prompt templates (feature_context variant)
        # from priors/build_priors_v2.py so wording matches the thesis.
        raise NotImplementedError("default prompt template not yet ported — pass prompt=... for now")

    def elicit(self, context: FeatureContext) -> PriorBundle:
        """Run the elicitation and return a :class:`PriorBundle`."""
        raise NotImplementedError("elicitation call loop not yet ported — see TODO in elicit.py")


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
        Fully custom user prompt; bypasses context entirely.
    cache:
        If given, load the bundle from this path when it exists, else save to it
        after eliciting — so reruns and demos need no API key.

    Returns
    -------
    PriorBundle
    """
    if cache is not None and Path(cache).exists():
        return PriorBundle.load(cache)

    if prompt is None:
        if context is None:
            if auto_context:
                from .context import generate_feature_context

                context = generate_feature_context(X, target=getattr(y, "name", None), model=model)
            else:
                context = FeatureContext.blank(list(X.columns), target=getattr(y, "name", None))

    elicitor = PriorElicitor(
        model=model, task=task, n_samples=n_samples, temperature=temperature, prompt=prompt
    )
    bundle = elicitor.elicit(context if prompt is None else FeatureContext.blank(list(X.columns)))

    if cache is not None:
        bundle.save(cache)
    return bundle
