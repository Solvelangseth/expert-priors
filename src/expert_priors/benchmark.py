"""compare() — one-call benchmark of a baseline vs prior-informed families.

Fits the L2 baseline (Ridge for regression, L2 Logistic for classification) and
each requested prior family, then reports in-domain (ID) and out-of-domain (OOD)
metrics so the prior's OOD lift is visible at a glance.

The baseline is the prior-mean estimator with a **zero** prior — i.e. exactly
standardized Ridge / L2 logistic — so it shares the prior families'
standardization and alpha grid and every "Delta vs baseline" is apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .bundle import PriorBundle
from .models import family

_DEFAULT_FAMILIES = ("prior_mean", "correlated", "independent")
_FAMILY_LABELS = {"prior_mean": "Prior-Mean", "correlated": "Correlated", "independent": "Independent"}
_BASELINE_LABEL = {"regression": "Ridge", "classification": "L2 Logistic"}
_PRIMARY_METRIC = {"regression": "mse", "classification": "auc"}


@dataclass
class ComparisonReport:
    """Result of :func:`compare` — tidy metrics plus convenience renderers."""

    rows: list[dict]
    task: str
    baseline_label: str

    def summary(self):
        """Return the metrics as a pandas DataFrame (one row per model/split)."""
        import pandas as pd

        return pd.DataFrame(self.rows)

    def lift(self):
        """Δ on the primary metric vs the baseline, per split (signed so >0 = better)."""
        import pandas as pd

        metric = _PRIMARY_METRIC[self.task]
        sign = -1.0 if self.task == "regression" else 1.0  # lower MSE is better
        df = pd.DataFrame(self.rows)
        out = []
        for split in df["split"].unique():
            sub = df[df["split"] == split].set_index("model")
            base = sub.loc[self.baseline_label, metric]
            for model, val in sub[metric].items():
                if model == self.baseline_label:
                    continue
                out.append({
                    "split": split, "model": model, "metric": metric,
                    "baseline": base, "value": val,
                    "lift": sign * (val - base),
                    "lift_pct": sign * (val - base) / abs(base) * 100 if base else float("nan"),
                })
        return pd.DataFrame(out)

    def plot(self, ax=None):
        """Grouped bar chart of the primary metric per model and split."""
        import matplotlib.pyplot as plt
        import pandas as pd

        metric = _PRIMARY_METRIC[self.task]
        df = pd.DataFrame(self.rows)
        pivot = df.pivot(index="model", columns="split", values=metric)
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.5))
        pivot.plot.bar(ax=ax)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by model and split ({self.task})")
        ax.tick_params(axis="x", rotation=30)
        return ax


# --- metrics --------------------------------------------------------------
def _regression_metrics(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classification_metrics(y_true, proba) -> dict:
    proba = np.clip(proba, 1e-7, 1 - 1e-7)
    pred = (proba > 0.5).astype(int)
    auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "logloss": float(log_loss(y_true, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def _predict_for_eval(model, X, task):
    if task == "regression":
        return model.predict(X)
    return model.predict_proba(X)[:, 1]


def compare(
    X,
    y,
    priors: PriorBundle,
    *,
    families: Sequence[str] = _DEFAULT_FAMILIES,
    ood_mask=None,
    task: str | None = None,
    alpha: float | str = "cv",
    test_size: float = 0.25,
    random_state: int = 42,
) -> ComparisonReport:
    """Benchmark baseline vs prior families on ID and (optional) OOD splits.

    Parameters
    ----------
    X, y:
        Full design matrix (DataFrame recommended) and target.
    priors:
        The :class:`PriorBundle` feeding the prior-informed families.
    families:
        Which prior families to include (defaults to all three).
    ood_mask:
        Boolean mask selecting the out-of-domain rows. Models train on a split of
        the in-domain rows and are evaluated on the held-out ID rows and the OOD
        rows. If omitted, only a held-out in-domain split is reported.
    task:
        Override task inference; defaults to ``priors.task``.
    alpha:
        Passed to every estimator (``"cv"`` selects per model).
    test_size, random_state:
        In-domain train/test split controls.

    Returns
    -------
    ComparisonReport
    """
    task = task or priors.task
    baseline_label = _BASELINE_LABEL[task]
    import pandas as pd

    if not hasattr(X, "iloc"):
        X = pd.DataFrame(np.asarray(X), columns=priors.feature_names)
    y = np.asarray(y).ravel()

    n = len(X)
    mask = np.zeros(n, dtype=bool) if ood_mask is None else np.asarray(ood_mask, dtype=bool)
    id_idx = np.where(~mask)[0]
    ood_idx = np.where(mask)[0]

    stratify = y[id_idx] if task == "classification" else None
    tr, te = train_test_split(id_idx, test_size=test_size, random_state=random_state, stratify=stratify)

    X_tr, y_tr = X.iloc[tr], y[tr]
    eval_splits = [("ID", te)]
    if ood_idx.size > 0:
        eval_splits.append(("OOD", ood_idx))

    # baseline = prior-mean family with a zero prior (== Ridge / L2 logistic)
    zero_bundle = PriorBundle(
        list(priors.feature_names), np.zeros((2, priors.n_features)), task=task
    )
    models = {baseline_label: family("prior_mean", task)(zero_bundle, alpha=alpha)}
    for fam in families:
        models[_FAMILY_LABELS.get(fam, fam)] = family(fam, task)(priors, alpha=alpha)

    metric_fn = _regression_metrics if task == "regression" else _classification_metrics
    rows: list[dict] = []
    for label, model in models.items():
        model.fit(X_tr, y_tr)
        for split_name, idx in eval_splits:
            preds = _predict_for_eval(model, X.iloc[idx], task)
            rows.append({
                "model": label, "split": split_name, "n": int(idx.size),
                "alpha": float(getattr(model, "alpha_", np.nan)),
                **metric_fn(y[idx], preds),
            })

    return ComparisonReport(rows=rows, task=task, baseline_label=baseline_label)
