"""Elicit the demo priors once and cache them for the quickstart notebook.

Run this if you want to regenerate ``examples/priors/housing_prior.json`` (needs
an API key in your environment / .env). The committed cache lets the notebook run
with no key at all.

    python examples/build_demo_priors.py --model claude-opus-4-8 --samples 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import expert_priors as ep
from demo_data import housing_context, make_housing

CACHE = Path(__file__).parent / "priors" / "housing_prior.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--out", default=str(CACHE))
    args = parser.parse_args()

    X, y, _ = make_housing()
    priors = ep.elicit_priors(
        X.iloc[:150], y.iloc[:150],     # elicit from the in-domain block only
        task="regression",
        model=args.model,
        n_samples=args.samples,
        context=housing_context(),
        cache=None,                      # force a fresh elicitation
    )
    priors.save(args.out)
    print(f"Saved {priors.n_samples} samples -> {args.out}")
    print("Prior mean (USD per unit):")
    for name, val in zip(priors.feature_names, priors.mean().round(1)):
        print(f"  {name:>10}: {val:>10}")


if __name__ == "__main__":
    main()
