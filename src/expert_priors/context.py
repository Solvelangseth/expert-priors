"""FeatureContext — the per-feature domain description that grounds elicitation.

In the thesis pipeline this context (human-readable description, units, and
category labels for each feature) was hand-curated per dataset. Here it is a
first-class object you can:

  * generate automatically with an LLM (:func:`generate_feature_context`),
  * inspect and edit, or
  * build by hand / load from a dict.

You can also bypass it entirely and pass a fully custom prompt to
:func:`expert_priors.elicit_priors`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass
class FeatureDescription:
    """Domain context for a single feature."""

    name: str
    description: str = ""
    unit: str | None = None
    # For categorical features: maps raw code -> human label, e.g. {"1": "male"}.
    category_labels: dict[str, str] = field(default_factory=dict)

    def to_prompt_line(self) -> str:
        bits = [f"- {self.name}: {self.description}".rstrip(": ").rstrip()]
        if self.unit:
            bits.append(f"(unit: {self.unit})")
        if self.category_labels:
            labels = ", ".join(f"{k}={v}" for k, v in self.category_labels.items())
            bits.append(f"[categories: {labels}]")
        return " ".join(bits)


@dataclass
class FeatureContext:
    """Ordered collection of per-feature descriptions for one dataset/target."""

    features: list[FeatureDescription]
    target: str | None = None
    target_description: str = ""
    dataset_description: str = ""

    # -- access ------------------------------------------------------------
    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def __getitem__(self, name: str) -> FeatureDescription:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(name)

    def edit(self, name: str, **changes: Any) -> "FeatureContext":
        """Update fields of one feature description in place; returns self for chaining."""
        f = self[name]
        for key, value in changes.items():
            if not hasattr(f, key):
                raise AttributeError(f"FeatureDescription has no field {key!r}")
            setattr(f, key, value)
        return self

    # -- prompt rendering --------------------------------------------------
    def to_prompt_block(self) -> str:
        """Render the context as the feature block injected into the user prompt."""
        lines = []
        if self.dataset_description:
            lines.append(self.dataset_description.strip())
            lines.append("")
        if self.target:
            tgt = f"Target: {self.target}"
            if self.target_description:
                tgt += f" — {self.target_description}"
            lines.append(tgt)
            lines.append("")
        lines.append("Features:")
        lines.extend(f.to_prompt_line() for f in self.features)
        return "\n".join(lines)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureContext":
        feats = [
            f if isinstance(f, FeatureDescription) else FeatureDescription(**f)
            for f in data["features"]
        ]
        return cls(
            features=feats,
            target=data.get("target"),
            target_description=data.get("target_description", ""),
            dataset_description=data.get("dataset_description", ""),
        )

    @classmethod
    def blank(cls, feature_names: Sequence[str], target: str | None = None) -> "FeatureContext":
        """A skeleton context with empty descriptions — fill in by hand."""
        return cls(features=[FeatureDescription(name=n) for n in feature_names], target=target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "target_description": self.target_description,
            "dataset_description": self.dataset_description,
            "features": [vars(f) for f in self.features],
        }


def generate_feature_context(
    X: "pd.DataFrame",
    target: str | None = None,
    *,
    model: str = "claude-opus-4-8",
    n_sample_rows: int = 5,
    data_dictionary: str | None = None,
) -> FeatureContext:
    """Use an LLM to draft a :class:`FeatureContext` from a dataframe.

    The model is shown the column names, dtypes, and a few sample rows (plus an
    optional free-text data dictionary) and asked to write a one-line domain
    description, unit, and category labels for each feature. The result is a
    fully editable :class:`FeatureContext` — always review it before eliciting.

    Parameters
    ----------
    X:
        Feature dataframe (predictors only; drop the target/OOD-splitter first).
    target:
        Optional target-variable name, used to phrase descriptions toward it.
    model:
        LLM used to draft the context. Defaults to the latest Claude model.
    n_sample_rows:
        How many example rows to show the model.
    data_dictionary:
        Optional free-text notes / codebook to ground the descriptions.

    Returns
    -------
    FeatureContext

    Notes
    -----
    TODO(port): implemented in the auto-context step. Will call the provider
    layer in :mod:`expert_priors.providers` with a structured-output schema and
    map the response onto :class:`FeatureDescription` objects.
    """
    raise NotImplementedError(
        "generate_feature_context is part of the auto-context build step (not yet ported). "
        "For now, build context with FeatureContext.from_dict(...) or FeatureContext.blank(...)."
    )
