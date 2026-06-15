"""Prior-informed model families, as scikit-learn-compatible estimators.

Each estimator takes a :class:`~expert_priors.bundle.PriorBundle` and adds a
prior-informed penalty on top of an L2 baseline:

* ``PriorMean*``     — L2 penalty toward the LLM prior mean mu; scalar lambda.
* ``Correlated*``    — quadratic penalty with full precision P = Sigma^-1 from the
  LLM sample covariance.
* ``Independent*``   — diagonal precision P_jj = lambda / (var_j + eps).

Regression variants wrap ridge; classification variants wrap L2 logistic. Priors
are stored in raw feature space and transformed to standardized space at fit time.

These are thin, documented wrappers around the validated implementations in the
thesis ``llm_statistics`` package; porting those fit/predict bodies is the
"port the model families" build step.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .bundle import PriorBundle

AlphaSpec = float | Literal["cv"]


class _PriorModelBase(BaseEstimator):
    """Shared construction/validation for prior-informed estimators.

    Notes
    -----
    Keyword-only constructor args are kept as plain attributes (no validation in
    ``__init__``) to satisfy scikit-learn's estimator contract / ``clone``.
    """

    def __init__(self, priors: PriorBundle | None = None, *, alpha: AlphaSpec = "cv") -> None:
        self.priors = priors
        self.alpha = alpha

    def _check_priors(self, feature_names) -> PriorBundle:
        if self.priors is None:
            raise ValueError("a PriorBundle must be provided")
        return self.priors.align_to(feature_names)

    def fit(self, X, y):  # noqa: D401
        raise NotImplementedError(
            f"{type(self).__name__}.fit not yet ported — see TODO in models.py "
            "(port from llm_statistics)."
        )

    def predict(self, X):
        raise NotImplementedError(f"{type(self).__name__}.predict not yet ported")


# --- regression -----------------------------------------------------------
class PriorMeanRidge(_PriorModelBase, RegressorMixin):
    """Ridge with the L2 penalty centred on the LLM prior mean."""


class CorrelatedPriorRidge(_PriorModelBase, RegressorMixin):
    """Ridge with a full-precision (Sigma^-1) prior penalty."""


class IndependentPriorRidge(_PriorModelBase, RegressorMixin):
    """Ridge with a diagonal per-feature prior precision."""


# --- classification -------------------------------------------------------
class PriorMeanLogistic(_PriorModelBase, ClassifierMixin):
    """L2 logistic regression with the penalty centred on the LLM prior mean."""

    def predict_proba(self, X):
        raise NotImplementedError(f"{type(self).__name__}.predict_proba not yet ported")


class CorrelatedPriorLogistic(_PriorModelBase, ClassifierMixin):
    """L2 logistic regression with a full-precision prior penalty."""

    def predict_proba(self, X):
        raise NotImplementedError(f"{type(self).__name__}.predict_proba not yet ported")


class IndependentPriorLogistic(_PriorModelBase, ClassifierMixin):
    """L2 logistic regression with a diagonal per-feature prior precision."""

    def predict_proba(self, X):
        raise NotImplementedError(f"{type(self).__name__}.predict_proba not yet ported")


_REGRESSION_FAMILIES = {
    "prior_mean": PriorMeanRidge,
    "correlated": CorrelatedPriorRidge,
    "independent": IndependentPriorRidge,
}
_CLASSIFICATION_FAMILIES = {
    "prior_mean": PriorMeanLogistic,
    "correlated": CorrelatedPriorLogistic,
    "independent": IndependentPriorLogistic,
}


def family(name: str, task: str):
    """Look up an estimator class by short family name and task."""
    table = _REGRESSION_FAMILIES if task == "regression" else _CLASSIFICATION_FAMILIES
    if name not in table:
        raise KeyError(f"unknown family {name!r}; choose from {sorted(table)}")
    return table[name]
