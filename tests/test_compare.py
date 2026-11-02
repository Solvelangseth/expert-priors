"""Tests for the compare() benchmark."""

import numpy as np
import pandas as pd
import pytest

import expert_priors as ep
from expert_priors.benchmark import compare
from expert_priors.bundle import PriorBundle


def _reg(seed=0, n=240, p=3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    beta = np.array([2.0, -1.0, 0.5])[:p]
    y = X.to_numpy() @ beta + 0.1 * rng.normal(size=n)
    return X, pd.Series(y, name="y"), beta


def _prior_near(beta, cols, spread=0.2, seed=1, task="regression"):
    rng = np.random.default_rng(seed)
    return PriorBundle(list(cols), rng.normal(beta, spread, size=(40, len(cols))), task=task)


def test_compare_structure_id_only():
    X, y, beta = _reg()
    priors = _prior_near(beta, X.columns)
    report = compare(X, y, priors, alpha=1.0)
    df = report.summary()
    assert set(df["split"]) == {"ID"}
    assert set(df["model"]) == {"Ridge", "Prior-Mean", "Correlated", "Independent"}
    assert "mse" in df.columns and "r2" in df.columns


def test_compare_with_ood_mask():
    X, y, beta = _reg(seed=2)
    priors = _prior_near(beta, X.columns, seed=3)
    mask = X["x0"] > 0.8  # carve out an OOD region
    report = compare(X, y, priors, ood_mask=mask.to_numpy(), alpha=1.0)
    df = report.summary()
    assert set(df["split"]) == {"ID", "OOD"}
    # every model evaluated on both splits
    assert (df.groupby("model").size() == 2).all()


def test_baseline_matches_standardized_ridge():
    """The zero-prior baseline must equal a directly-fit standardized ridge."""
    from expert_priors.models import PriorMeanRidge

    X, y, beta = _reg(seed=4)
    priors = _prior_near(beta, X.columns)
    report = compare(X, y, priors, families=["prior_mean"], alpha=2.0, test_size=0.25)

    # reproduce the baseline fit independently with an explicit zero bundle
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.25, random_state=42)
    zero = PriorBundle(list(X.columns), np.zeros((2, 3)))
    base = PriorMeanRidge(zero, alpha=2.0).fit(X.iloc[tr], y.to_numpy()[tr])
    expected_mse = float(np.mean((y.to_numpy()[te] - base.predict(X.iloc[te])) ** 2))
    got = report.summary().query("model == 'Ridge' and split == 'ID'")["mse"].iloc[0]
    assert got == pytest.approx(expected_mse, rel=1e-9)


def test_lift_sign_and_columns():
    X, y, beta = _reg(seed=5)
    priors = _prior_near(beta, X.columns)
    report = compare(X, y, priors, alpha=1.0)
    lift = report.lift()
    assert {"split", "model", "lift", "lift_pct"}.issubset(lift.columns)
    # lift is signed so positive => better than baseline; recompute for one row
    summ = report.summary().set_index(["model", "split"])
    row = lift.iloc[0]
    base_mse = summ.loc[("Ridge", row["split"]), "mse"]
    model_mse = summ.loc[(row["model"], row["split"]), "mse"]
    assert row["lift"] == pytest.approx(-(model_mse - base_mse))


def test_compare_classification_runs():
    rng = np.random.default_rng(6)
    n, p = 300, 3
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    beta = np.array([1.5, -1.0, 0.5])
    pr = 1 / (1 + np.exp(-(X.to_numpy() @ beta)))
    y = pd.Series((rng.uniform(size=n) < pr).astype(int), name="y")
    priors = _prior_near(beta, X.columns, task="classification")
    report = compare(X, y, priors, ood_mask=(X["x0"] > 0.7).to_numpy(),
                     alpha="cv", task="classification")
    df = report.summary()
    assert "auc" in df.columns and "brier" in df.columns
    assert df["auc"].between(0, 1).all()
    assert "L2 Logistic" in set(df["model"])


def test_compare_via_package_namespace():
    X, y, beta = _reg(seed=7)
    priors = _prior_near(beta, X.columns)
    report = ep.compare(X, y, priors, families=["prior_mean"], alpha=1.0)
    assert not report.summary().empty


def test_compare_warns_once_for_flat_prior():
    import warnings

    import expert_priors as ep

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 2)), columns=["a", "b"])
    y = X["a"] - X["b"] + rng.normal(scale=0.1, size=120)
    samples = rng.normal(size=(8, 2))
    samples[:, 0] = 1.0
    priors = ep.PriorBundle(["a", "b"], samples)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ep.compare(X, y, priors, alpha=1.0)
    msgs = [str(w.message) for w in caught if "no spread" in str(w.message)]
    assert len(msgs) == 1 and "Correlated and Independent" in msgs[0]
