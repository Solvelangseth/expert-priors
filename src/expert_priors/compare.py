"""compare() — one-call benchmark of a baseline vs prior-informed families.

Fits the L2 baseline (Ridge for regression, L2 Logistic for classification) and
each requested prior family, then reports in-domain and out-of-domain metrics so
the prior's OOD lift is visible at a glance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .bundle import PriorBundle

_DEFAULT_FAMILIES = ("prior_mean", "correlated", "independent")


@dataclass
class ComparisonReport:
    """Result of :func:`compare` — tidy metrics plus convenience renderers."""

    rows: list[dict]
    task: str

    def summary(self):
        """Return the metrics as a pandas DataFrame (one row per model/split)."""
        import pandas as pd

        return pd.DataFrame(self.rows)

    def plot(self):
        """Plot OOD lift vs the baseline. TODO(port): figure code from new_results/."""
        raise NotImplementedError("plotting not yet ported — see TODO in compare.py")


def compare(
    X,
    y,
    priors: PriorBundle,
    *,
    families: Sequence[str] = _DEFAULT_FAMILIES,
    ood_mask=None,
    task: str | None = None,
) -> ComparisonReport:
    """Benchmark baseline vs prior families on ID and (optional) OOD splits.

    Parameters
    ----------
    X, y:
        Full design matrix and target.
    priors:
        The :class:`PriorBundle` to feed the prior-informed families.
    families:
        Which prior families to include (defaults to all three).
    ood_mask:
        Boolean mask selecting the out-of-domain rows. If omitted, only an
        in-domain CV comparison is reported.
    task:
        Override task inference; defaults to ``priors.task``.

    Returns
    -------
    ComparisonReport

    Notes
    -----
    TODO(port): wire to the locked CV + bootstrap evaluation in the thesis
    ``pipeline`` package so numbers match the published results.
    """
    raise NotImplementedError("compare() evaluation loop not yet ported — see TODO in compare.py")
