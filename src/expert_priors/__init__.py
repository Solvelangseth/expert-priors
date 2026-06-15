"""expert_priors — LLM-elicited expert priors for regularized statistical models.

Point it at a dataset + target, let an LLM elicit coefficient priors (optionally
writing the feature context for you), then fit scikit-learn-style prior-informed
models that improve out-of-domain prediction.

Quickstart
----------
>>> import expert_priors as ep
>>> priors = ep.elicit_priors(X, y, task="regression",
...                           model="claude-opus-4-8", auto_context=True)
>>> model = ep.PriorMeanRidge(priors).fit(X_train, y_train)
>>> preds = model.predict(X_test)

This package is the user-friendly library distilled from the master's thesis
"LLM-Elicited Priors for Regularized Statistical Models". The full, frozen
reproducibility pipeline lives in the companion thesis repository.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .benchmark import ComparisonReport, compare
from .bundle import PriorBundle
from .context import FeatureContext, generate_feature_context
from .elicit import PriorElicitor, elicit_priors
from .models import (
    CorrelatedPriorLogistic,
    CorrelatedPriorRidge,
    IndependentPriorLogistic,
    IndependentPriorRidge,
    PriorMeanLogistic,
    PriorMeanRidge,
)

__all__ = [
    "__version__",
    # data containers
    "PriorBundle",
    "FeatureContext",
    # elicitation
    "elicit_priors",
    "PriorElicitor",
    "generate_feature_context",
    # models — regression
    "PriorMeanRidge",
    "CorrelatedPriorRidge",
    "IndependentPriorRidge",
    # models — classification
    "PriorMeanLogistic",
    "CorrelatedPriorLogistic",
    "IndependentPriorLogistic",
    # benchmarking
    "compare",
    "ComparisonReport",
]
