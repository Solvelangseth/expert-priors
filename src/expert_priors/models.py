"""Prior-informed model families, as scikit-learn-compatible estimators.

Each estimator takes a :class:`~expert_priors.bundle.PriorBundle` (coefficient
priors in **raw** feature units) and adds a prior-informed penalty on top of an
L2 baseline. All three families share one parameterization::

    penalty(beta) = 0.5 * alpha * (beta - mu)^T P0 (beta - mu)

where the base precision ``P0`` is what distinguishes them:

* ``PriorMean*``   — ``P0 = I``                       (shrink toward the prior mean mu)
* ``Correlated*``  — ``P0 = Sigma^-1``                (full LLM sample covariance)
* ``Independent*`` — ``P0 = diag(1 / (var_j + eps))`` (per-feature only)

Because ``alpha`` multiplies the same outer ``0.5 * (...)`` for every family, the
selected ``alpha`` is directly comparable across families (matching the thesis).

Standardization is handled internally: you pass raw ``X`` and a raw-space prior;
the estimator standardizes ``X``, scales the prior into standardized space, fits
there, then exposes raw-space ``coef_`` / ``intercept_`` so ``predict`` works on
raw ``X`` like any sklearn model.

Regression families solve the penalized normal equations in closed form;
classification families maximize the penalized binomial log-likelihood with
L-BFGS. Ported from the validated thesis ``llm_statistics`` implementations.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from .bundle import PriorBundle

AlphaSpec = float | Literal["cv"]

_DEFAULT_ALPHA_GRID = np.logspace(-4, 6, 51)
_LBFGS_MAXITER_PRIMARY = 5000
_LBFGS_MAXITER_RETRY = 20000


# =============================================================================
# Base
# =============================================================================
class _PriorModelBase(BaseEstimator):
    """Shared standardization, prior derivation, and alpha-CV machinery.

    Subclasses set ``_family`` (``"prior_mean"`` / ``"correlated"`` /
    ``"independent"``) and ``_task`` (``"regression"`` / ``"classification"``).
    Constructor args are stored unvalidated as plain attributes to honour the
    scikit-learn estimator / ``clone`` contract.
    """

    _family: str = ""
    _task: str = ""

    def __init__(
        self,
        priors: PriorBundle | None = None,
        *,
        alpha: AlphaSpec = "cv",
        fit_intercept: bool = True,
        standardize: bool = True,
        eps: float = 1e-8,
        cov_ridge: float = 1e-6,
        cv_folds: int = 5,
        alpha_grid=None,
        random_state: int | None = 42,
    ) -> None:
        self.priors = priors
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.eps = eps
        self.cov_ridge = cov_ridge
        self.cv_folds = cv_folds
        self.alpha_grid = alpha_grid
        self.random_state = random_state

    # -- data prep ---------------------------------------------------------
    def _feature_names(self, X):
        return list(X.columns) if hasattr(X, "columns") else None

    def _as_array(self, X):
        return X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)

    def _aligned_bundle(self, feature_names, n_features) -> PriorBundle:
        if self.priors is None:
            raise ValueError("a PriorBundle must be provided via priors=")
        bundle = self.priors
        if feature_names is not None:
            bundle = bundle.align_to(feature_names)
        elif bundle.n_features != n_features:
            raise ValueError(
                f"prior has {bundle.n_features} features but X has {n_features}; "
                "pass X as a DataFrame so columns can be matched by name."
            )
        return bundle

    # -- standardized prior quantities ------------------------------------
    def _prior_in_std_space(self, bundle: PriorBundle, scale: np.ndarray):
        """Scale raw-space samples to standardized space and derive mu and P0."""
        samples_std = bundle.samples * scale  # beta_std = beta_raw * scale
        mu = samples_std.mean(axis=0)
        if self._family == "prior_mean":
            P0 = np.eye(mu.shape[0])
        elif self._family == "independent":
            var = samples_std.var(axis=0, ddof=1) if samples_std.shape[0] > 1 else np.zeros_like(mu)
            P0 = np.diag(1.0 / (var + self.eps))
        elif self._family == "correlated":
            cov = np.cov(samples_std, rowvar=False) + self.cov_ridge * np.eye(mu.shape[0])
            P0 = np.linalg.pinv(cov)
        else:  # pragma: no cover - guarded by subclassing
            raise ValueError(f"unknown family {self._family!r}")
        return mu, P0

    # -- fit ---------------------------------------------------------------
    def fit(self, X, y):
        feature_names = self._feature_names(X)
        Xa = self._as_array(X)
        y = np.asarray(y).ravel()
        if Xa.ndim != 2:
            raise ValueError("X must be 2D")
        if Xa.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        n, p = Xa.shape

        bundle = self._aligned_bundle(feature_names, p)
        self.feature_names_in_ = bundle.feature_names

        # standardization (population std, like StandardScaler)
        self.center_ = Xa.mean(axis=0) if self.standardize else np.zeros(p)
        if self.standardize:
            scale = Xa.std(axis=0, ddof=0)
            scale = np.where(scale < 1e-12, 1.0, scale)
        else:
            scale = np.ones(p)
        self.scale_ = scale
        Xs = (Xa - self.center_) / self.scale_

        mu, P0 = self._prior_in_std_space(bundle, self.scale_)
        self.mu_std_ = mu
        self.P0_ = P0

        yfit = self._prepare_target(y)

        # choose alpha
        if isinstance(self.alpha, str) and self.alpha == "cv":
            self.alpha_ = self._select_alpha_cv(Xs, yfit, mu, P0)
        else:
            self.alpha_ = float(self.alpha)

        beta_std, b_std = self._solve(Xs, yfit, mu, P0, self.alpha_)
        self._store_raw_coefficients(beta_std, b_std)
        return self

    def _store_raw_coefficients(self, beta_std, b_std):
        # z = (X - center)/scale @ beta_std + b_std
        #   = X @ (beta_std/scale) + (b_std - sum(beta_std*center/scale))
        self.coef_std_ = beta_std
        self.coef_ = beta_std / self.scale_
        self.intercept_ = float(b_std - np.sum(beta_std * self.center_ / self.scale_))

    def _linear(self, X):
        Xa = self._as_array(X)
        if Xa.ndim == 1:
            Xa = Xa.reshape(1, -1)
        return Xa @ self.coef_ + self.intercept_

    # -- hooks implemented by task mixins ---------------------------------
    def _prepare_target(self, y):
        raise NotImplementedError

    def _solve(self, Xs, y, mu, P0, alpha):
        raise NotImplementedError

    def _cv_splitter(self):
        raise NotImplementedError

    def _cv_score(self, y_true, y_pred):
        raise NotImplementedError

    def _select_alpha_cv(self, Xs, y, mu, P0):
        grid = self.alpha_grid if self.alpha_grid is not None else _DEFAULT_ALPHA_GRID
        splitter = self._cv_splitter()
        best_alpha, best_score = None, -np.inf
        for alpha in grid:
            fold_scores = []
            for tr, va in splitter.split(Xs, y):
                beta, b = self._solve(Xs[tr], y[tr], mu, P0, float(alpha))
                pred = Xs[va] @ beta + b
                fold_scores.append(self._cv_score(y[va], pred))
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:  # strict-best mean CV
                best_score, best_alpha = mean_score, float(alpha)
        return best_alpha


# =============================================================================
# Regression
# =============================================================================
class _PriorRidge(_PriorModelBase, RegressorMixin):
    _task = "regression"

    def _prepare_target(self, y):
        return y.astype(float)

    def _solve(self, Xs, y, mu, P0, alpha):
        n, p = Xs.shape
        half = 0.5 * alpha
        if self.fit_intercept:
            Xw = np.column_stack([np.ones(n), Xs])
            P_ext = np.zeros((p + 1, p + 1))
            P_ext[1:, 1:] = P0
            mu_ext = np.concatenate([[0.0], mu])
        else:
            Xw, P_ext, mu_ext = Xs, P0, mu
        A = Xw.T @ Xw + half * P_ext
        b = Xw.T @ y + half * (P_ext @ mu_ext)
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ b
        if self.fit_intercept:
            return beta[1:], float(beta[0])
        return beta, 0.0

    def _cv_splitter(self):
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

    def _cv_score(self, y_true, y_pred):
        return -float(np.mean((y_true - y_pred) ** 2))  # maximize -> minimize MSE

    def predict(self, X):
        return self._linear(X)


class PriorMeanRidge(_PriorRidge):
    """Ridge with the L2 penalty centred on the LLM prior mean (``P0 = I``)."""

    _family = "prior_mean"


class CorrelatedPriorRidge(_PriorRidge):
    """Ridge with a full-precision (``Sigma^-1``) prior penalty."""

    _family = "correlated"


class IndependentPriorRidge(_PriorRidge):
    """Ridge with a diagonal per-feature prior precision."""

    _family = "independent"


# =============================================================================
# Classification
# =============================================================================
class _PriorLogistic(_PriorModelBase, ClassifierMixin):
    _task = "classification"

    def _prepare_target(self, y):
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError("classification target must be binary")
        self.classes_ = classes
        return (y == classes[1]).astype(float)

    def _solve(self, Xs, y, mu, P0, alpha):
        n, p = Xs.shape
        if self.fit_intercept:
            Xw = np.column_stack([np.ones(n), Xs])
            d = p + 1
            P_ext = np.zeros((d, d))
            P_ext[1:, 1:] = P0
            mu_ext = np.concatenate([[0.0], mu])
        else:
            Xw, d, P_ext, mu_ext = Xs, p, P0, mu

        def objective(beta):
            z = np.clip(Xw @ beta, -500, 500)
            loglik = np.sum(y * z - np.logaddexp(0, z))
            diff = beta - mu_ext
            return -loglik + 0.5 * alpha * (diff @ P_ext @ diff)

        def gradient(beta):
            z = np.clip(Xw @ beta, -500, 500)
            p_hat = 1.0 / (1.0 + np.exp(-z))
            return -Xw.T @ (y - p_hat) + alpha * (P_ext @ (beta - mu_ext))

        res = minimize(objective, np.zeros(d), jac=gradient, method="L-BFGS-B",
                       options={"maxiter": _LBFGS_MAXITER_PRIMARY})
        if not res.success:
            retry = minimize(objective, res.x, jac=gradient, method="L-BFGS-B",
                             options={"maxiter": _LBFGS_MAXITER_RETRY})
            if retry.success or retry.nit > res.nit:
                res = retry
        self.optimizer_success_ = bool(res.success)
        beta = res.x
        if self.fit_intercept:
            return beta[1:], float(beta[0])
        return beta, 0.0

    def _cv_splitter(self):
        return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

    def _cv_score(self, y_true, z_pred):
        proba = 1.0 / (1.0 + np.exp(-np.clip(z_pred, -500, 500)))
        if np.unique(y_true).shape[0] < 2:
            return 0.5
        return float(roc_auc_score(y_true, proba))

    def decision_function(self, X):
        return self._linear(X)

    def predict_proba(self, X):
        z = np.clip(self._linear(X), -500, 500)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return self.classes_[(p1 > 0.5).astype(int)]


class PriorMeanLogistic(_PriorLogistic):
    """L2 logistic regression with the penalty centred on the prior mean (``P0 = I``)."""

    _family = "prior_mean"


class CorrelatedPriorLogistic(_PriorLogistic):
    """L2 logistic regression with a full-precision prior penalty."""

    _family = "correlated"


class IndependentPriorLogistic(_PriorLogistic):
    """L2 logistic regression with a diagonal per-feature prior precision."""

    _family = "independent"


# =============================================================================
# Lookup
# =============================================================================
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
