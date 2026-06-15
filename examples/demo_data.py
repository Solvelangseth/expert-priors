"""Self-contained demo dataset for the quickstart (no files, no downloads).

A small synthetic housing dataset with a deliberate **domain shift**: the
in-domain (ID) homes are nearly new (age 0-4 years), so the depreciation slope
is barely identifiable from the training data. The out-of-domain (OOD) homes are
old (age 6-40 years), where that slope matters a lot. A data-only baseline shrinks
the age effect toward zero and extrapolates badly to old homes; an LLM prior that
knows "older houses are worth less" generalizes — which is the whole point of the
library.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from expert_priors.context import FeatureContext

FEATURES = ["sqft", "bedrooms", "age_years", "garage"]
# True data-generating coefficients (USD per unit).
_TRUE_BETA = {"sqft": 150.0, "bedrooms": 15000.0, "age_years": -1500.0, "garage": 20000.0}
_INTERCEPT = 30000.0


def make_housing(seed: int = 42):
    """Return ``(X, y, ood_mask)`` for the demo (ID = new homes, OOD = old homes)."""
    rng = np.random.default_rng(seed)

    def _block(n, age_low, age_high):
        return pd.DataFrame({
            "sqft": rng.uniform(800, 2500, n).round(0),
            "bedrooms": rng.integers(1, 6, n).astype(float),
            "age_years": rng.uniform(age_low, age_high, n).round(1),
            "garage": (rng.uniform(size=n) < 0.6).astype(float),
        })

    id_df = _block(150, 0.0, 4.0)    # nearly-new homes: age barely varies
    ood_df = _block(60, 6.0, 40.0)   # old homes: age is decisive
    X = pd.concat([id_df, ood_df], ignore_index=True)

    noise = rng.normal(0, 8000, len(X))
    y = pd.Series(
        _INTERCEPT + sum(_TRUE_BETA[f] * X[f] for f in FEATURES) + noise,
        name="price_usd",
    )
    ood_mask = np.r_[np.zeros(len(id_df), bool), np.ones(len(ood_df), bool)]
    return X, y, ood_mask


def housing_context() -> FeatureContext:
    """Hand-written context for the demo (so the notebook needs no API call)."""
    return FeatureContext.from_dict({
        "target": "price_usd",
        "target_description": "sale price of a single-family home in US dollars",
        "dataset_description": "Single-family home sales with basic property attributes.",
        "features": [
            {"name": "sqft", "description": "interior living area", "unit": "square feet"},
            {"name": "bedrooms", "description": "number of bedrooms", "unit": "count"},
            {"name": "age_years", "description": "age of the home", "unit": "years"},
            {"name": "garage", "description": "has a garage",
             "category_labels": {"0": "no garage", "1": "has garage"}},
        ],
    })
