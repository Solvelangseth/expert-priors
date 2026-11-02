"""Render the README figure from the cached demo priors (no API key).

Regularization paths on the housing demo: as the penalty grows, Ridge pulls every
coefficient toward zero, while the prior-informed model pulls it toward the
LLM-elicited prior. Cross-validation picks the point on each path.

    python examples/generate_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import expert_priors as ep
from demo_data import make_housing
from expert_priors.bundle import PriorBundle

OUT = Path(__file__).parent / "figures" / "regularization_paths.png"

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SURFACE = "#fcfcfb"
PRIOR = "#2a78d6"     # categorical slot 1
BASELINE = "#8a8984"  # neutral: the data-only reference

TITLES = {
    "sqft": "Living area (USD / sq ft)",
    "bedrooms": "Bedrooms (USD / room)",
    "age_years": "Age (USD / year)",
    "garage": "Garage (USD)",
}


def paths(estimator_cls, priors, X, y, grid):
    coefs = np.array([estimator_cls(priors, alpha=a).fit(X, y).coef_ for a in grid])
    chosen = estimator_cls(priors, alpha="cv").fit(X, y)
    return coefs, chosen.alpha_, chosen.coef_


def main() -> None:
    X, y, ood_mask = make_housing()
    X_id, y_id = X[~ood_mask], y[~ood_mask]  # fit on in-domain homes only
    priors = PriorBundle.load(Path(__file__).parent / "priors" / "housing_prior.json")
    zero = PriorBundle(priors.feature_names, np.zeros((2, priors.n_features)))

    grid = np.logspace(-2, 5, 60)
    ridge, ridge_a, ridge_c = paths(ep.PriorMeanRidge, zero, X_id, y_id, grid)
    prior, prior_a, prior_c = paths(ep.PriorMeanRidge, priors, X_id, y_id, grid)
    mu, sd = priors.mean(), priors.samples.std(axis=0, ddof=1)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2})
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), facecolor=SURFACE)

    for j, (ax, name) in enumerate(zip(axes, priors.feature_names)):
        ax.set_facecolor(SURFACE)
        ax.axhspan(mu[j] - sd[j], mu[j] + sd[j], color=PRIOR, alpha=0.10, lw=0)
        ax.axhline(mu[j], color=PRIOR, lw=1, ls=(0, (3, 3)), alpha=0.7)
        ax.axhline(0, color=GRID, lw=1)
        ax.plot(grid, ridge[:, j], color=BASELINE, lw=2, label="Ridge → shrinks to 0")
        ax.plot(grid, prior[:, j], color=PRIOR, lw=2, label="Prior-Mean → shrinks to LLM prior")
        ax.plot([ridge_a], [ridge_c[j]], "o", ms=7, color=BASELINE, mec=SURFACE, mew=2)
        ax.plot([prior_a], [prior_c[j]], "o", ms=7, color=PRIOR, mec=SURFACE, mew=2)
        ax.set_xscale("log")
        ax.set_title(TITLES.get(name, name), color=INK, fontsize=9.5, loc="left")
        ax.tick_params(length=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("penalty strength α", color=MUTED)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=PRIOR, alpha=0.15, lw=0))
    labels.append("LLM prior (mean ± 1 sd)")
    handles.append(plt.Line2D([], [], marker="o", ls="", color=INK_2, ms=6))
    labels.append("chosen by cross-validation")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=9, labelcolor=INK_2)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
