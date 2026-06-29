"""Render the README results figure from the cached demo priors (no API key).

Shows that the LLM prior leaves in-domain accuracy unchanged but improves
out-of-domain error, using the housing demo with its built-in domain shift.

    python examples/generate_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import expert_priors as ep
from expert_priors.bundle import PriorBundle
from demo_data import make_housing

OUT = Path(__file__).parent / "figures" / "ood_lift.png"


def main() -> None:
    X, y, ood_mask = make_housing()
    priors = PriorBundle.load(Path(__file__).parent / "priors" / "housing_prior.json")
    report = ep.compare(X, y, priors, ood_mask=ood_mask, alpha="cv")
    df = report.summary()

    baseline = report.baseline_label  # "Ridge"
    prior = "Prior-Mean"
    splits = ["ID", "OOD"]
    labels = ["In-domain", "Out-of-domain"]

    def metric(model, split, col):
        return float(df.query("model == @model and split == @split")[col].iloc[0])

    base_vals = [metric(baseline, s, "rmse") for s in splits]
    prior_vals = [metric(prior, s, "rmse") for s in splits]

    x = np.arange(len(splits))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    b1 = ax.bar(x - w / 2, base_vals, w, label=f"{baseline} (data only)", color="#9aa5b1")
    b2 = ax.bar(x + w / 2, prior_vals, w, label=f"{prior} (LLM prior)", color="#2f6fed")

    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE (USD) — lower is better")
    ax.set_title("LLM priors help where the data runs out (housing demo)")
    ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.0f", padding=2, fontsize=9)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # annotate the OOD improvement (both metrics, so the number is unambiguous)
    rmse_lift = (base_vals[1] - prior_vals[1]) / base_vals[1] * 100
    base_mse, prior_mse = metric(baseline, "OOD", "mse"), metric(prior, "OOD", "mse")
    mse_lift = (base_mse - prior_mse) / base_mse * 100
    ax.annotate(
        f"-{rmse_lift:.1f}% RMSE\n(-{mse_lift:.1f}% MSE)\nout-of-domain",
        xy=(1 + w / 2, prior_vals[1]),
        xytext=(1.05, prior_vals[1] + 0.16 * base_vals[1]),
        fontsize=9, color="#2f6fed", fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color="#2f6fed"),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}  (OOD improvement {rmse_lift:.1f}% RMSE / {mse_lift:.1f}% MSE)")


if __name__ == "__main__":
    main()
