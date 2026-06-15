"""Tests for the prior-informed model families.

Covers the limiting behaviours that pin down the math (alpha->0 recovers OLS;
alpha->inf shrinks to the prior mean), a hand-computed closed-form cross-check,
DataFrame column alignment, sklearn-clone compatibility, and the classification
path.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from expert_priors.bundle import PriorBundle
from expert_priors.models import (
    CorrelatedPriorRidge,
    IndependentPriorLogistic,
    IndependentPriorRidge,
    PriorMeanLogistic,
    PriorMeanRidge,
    family,
)


def _reg_data(seed=0, n=200, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta_true = np.array([2.0, -1.0, 0.5])[:p]
    y = X @ beta_true + 0.1 * rng.normal(size=n)
    cols = [f"x{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y"), beta_true


def _prior(cols, mean, spread=0.3, n_samples=40, seed=1, task="regression"):
    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=mean, scale=spread, size=(n_samples, len(cols)))
    return PriorBundle(list(cols), samples, task=task, model="test")


def test_high_alpha_shrinks_to_raw_prior_mean():
    X, y, _ = _reg_data()
    prior_mean = np.array([0.7, -0.2, 0.1])
    priors = _prior(X.columns, prior_mean)
    model = PriorMeanRidge(priors, alpha=1e9).fit(X, y)
    # beta_std -> mu_std, so raw coef_ -> raw prior sample mean
    np.testing.assert_allclose(model.coef_, priors.mean(), rtol=1e-3, atol=1e-3)


def test_low_alpha_recovers_ols():
    X, y, _ = _reg_data()
    priors = _prior(X.columns, np.zeros(3))
    model = PriorMeanRidge(priors, alpha=1e-10).fit(X, y)

    Xd = np.column_stack([np.ones(len(X)), X.to_numpy()])
    ols, *_ = np.linalg.lstsq(Xd, y.to_numpy(), rcond=None)
    np.testing.assert_allclose(model.intercept_, ols[0], atol=1e-4)
    np.testing.assert_allclose(model.coef_, ols[1:], atol=1e-4)


def test_prior_mean_ridge_matches_closed_form():
    """Cross-check the std-space solve against an independent hand computation."""
    X, y, _ = _reg_data(seed=3, n=60, p=2)
    priors = _prior(X.columns, np.array([1.0, -0.5]), seed=4)
    alpha = 3.7
    model = PriorMeanRidge(priors, alpha=alpha, fit_intercept=True).fit(X, y)

    Xa = X.to_numpy()
    center = Xa.mean(axis=0)
    scale = Xa.std(axis=0, ddof=0)
    Xs = (Xa - center) / scale
    mu = (priors.samples * scale).mean(axis=0)
    n, p = Xs.shape
    Xw = np.column_stack([np.ones(n), Xs])
    P_ext = np.zeros((p + 1, p + 1))
    P_ext[1:, 1:] = np.eye(p)
    mu_ext = np.concatenate([[0.0], mu])
    A = Xw.T @ Xw + 0.5 * alpha * P_ext
    b = Xw.T @ y.to_numpy() + 0.5 * alpha * (P_ext @ mu_ext)
    beta = np.linalg.solve(A, b)
    expected_coef = beta[1:] / scale
    expected_intercept = beta[0] - np.sum(beta[1:] * center / scale)
    np.testing.assert_allclose(model.coef_, expected_coef, atol=1e-8)
    np.testing.assert_allclose(model.intercept_, expected_intercept, atol=1e-8)


def test_dataframe_alignment_is_order_invariant():
    X, y, _ = _reg_data(seed=5)
    priors = _prior(X.columns, np.array([0.3, 0.1, -0.4]), seed=6)
    # shuffle the prior's feature order; alignment should make predictions identical
    shuffled = PriorBundle(
        [priors.feature_names[i] for i in [2, 0, 1]],
        priors.samples[:, [2, 0, 1]],
        task="regression",
    )
    p1 = PriorMeanRidge(priors, alpha=5.0).fit(X, y).predict(X)
    p2 = PriorMeanRidge(shuffled, alpha=5.0).fit(X, y).predict(X)
    np.testing.assert_allclose(p1, p2, atol=1e-10)


def test_correlated_and_independent_fit_finite():
    X, y, _ = _reg_data(seed=7)
    priors = _prior(X.columns, np.array([0.5, -0.5, 0.2]), seed=8)
    for Est in (CorrelatedPriorRidge, IndependentPriorRidge):
        m = Est(priors, alpha=2.0).fit(X, y)
        assert np.all(np.isfinite(m.coef_))
        assert np.isfinite(m.intercept_)
        assert m.predict(X).shape == (len(X),)


def test_alpha_cv_selects_from_grid():
    X, y, _ = _reg_data(seed=9)
    priors = _prior(X.columns, np.zeros(3))
    grid = np.array([1e-3, 1.0, 1e3])
    model = PriorMeanRidge(priors, alpha="cv", alpha_grid=grid).fit(X, y)
    assert model.alpha_ in set(grid.tolist())


def test_sklearn_clone_roundtrip():
    priors = _prior(["a", "b"], np.zeros(2))
    est = PriorMeanRidge(priors, alpha=2.0, fit_intercept=False)
    cloned = clone(est)
    assert cloned.get_params()["alpha"] == 2.0
    assert cloned.get_params()["fit_intercept"] is False


def test_family_lookup():
    assert family("prior_mean", "regression") is PriorMeanRidge
    assert family("correlated", "classification").__name__ == "CorrelatedPriorLogistic"
    with pytest.raises(KeyError):
        family("nope", "regression")


# --- classification -------------------------------------------------------
def _clf_data(seed=0, n=300, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([1.5, -1.0, 0.5])[:p]
    z = X @ beta
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
    cols = [f"x{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def test_logistic_proba_and_labels():
    X, y = _clf_data()
    priors = _prior(X.columns, np.zeros(3), task="classification")
    model = PriorMeanLogistic(priors, alpha=1.0).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    labels = model.predict(X)
    assert set(np.unique(labels)).issubset(set(model.classes_.tolist()))


def test_logistic_high_alpha_shrinks_to_prior_mean():
    X, y = _clf_data(seed=2)
    prior_mean = np.array([0.4, -0.3, 0.2])
    priors = _prior(X.columns, prior_mean, task="classification")
    model = IndependentPriorLogistic(priors, alpha=1e7).fit(X, y)
    np.testing.assert_allclose(model.coef_, priors.mean(), rtol=2e-2, atol=2e-2)


def test_logistic_alpha_cv_runs():
    X, y = _clf_data(seed=3)
    priors = _prior(X.columns, np.zeros(3), task="classification")
    model = PriorMeanLogistic(priors, alpha="cv", alpha_grid=np.array([0.1, 10.0])).fit(X, y)
    assert model.alpha_ in {0.1, 10.0}
    assert 0.0 <= model.score(X, y) <= 1.0
