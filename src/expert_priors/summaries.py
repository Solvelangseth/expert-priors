"""Feature-scale summaries computed from the actual design matrix.

These per-feature one-liners (binary prevalence, or median/IQR/central-95%) are
injected into the elicitation prompt so the LLM can reason about coefficient
*magnitude* in raw units, not just sign. Ported from the thesis prompt builder.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def feature_summary_line(series: pd.Series) -> str:
    """One-line scale summary for a single feature."""
    s = pd.to_numeric(series, errors="coerce")
    n_total = int(s.shape[0])
    s_non_null = s.dropna()
    missing_rate = float((n_total - s_non_null.shape[0]) / n_total) if n_total else 1.0
    missing_suffix = f", missing={missing_rate:.3f}" if missing_rate > 0 else ""

    if s_non_null.empty:
        return "summary unavailable"

    arr = s_non_null.to_numpy(dtype=float)
    if set(np.unique(arr)).issubset({0, 1}):
        return f"binary 0/1 indicator; prevalence={float(arr.mean()):.3f}{missing_suffix}"

    q05, q25, q50, q75, q95 = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95])
    return (
        "continuous/raw numeric; "
        f"median={q50:.3g}, IQR=[{q25:.3g}, {q75:.3g}], "
        f"central95%≈[{q05:.3g}, {q95:.3g}]{missing_suffix}"
    )


def feature_summary_block(X: pd.DataFrame, feature_names: Sequence[str] | None = None) -> str:
    """Render ``- name: summary`` lines for the given features."""
    names = list(feature_names) if feature_names is not None else list(X.columns)
    return "\n".join(f"- {name}: {feature_summary_line(X[name])}" for name in names)
