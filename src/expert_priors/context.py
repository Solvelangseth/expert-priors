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


_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "dataset_description": {"type": "string"},
        "target_description": {"type": "string"},
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "unit": {"type": "string"},
                    "category_labels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "label": {"type": "string"},
                            },
                            "required": ["code", "label"],
                        },
                    },
                },
                "required": ["name", "description"],
            },
        },
    },
    "required": ["features"],
}


def _build_context_prompt(
    X: "pd.DataFrame",
    target: str | None,
    n_sample_rows: int,
    data_dictionary: str | None,
) -> tuple[str, str]:
    cols = list(X.columns)
    dtypes = "\n".join(f"- {c}: {X[c].dtype}" for c in cols)
    sample = X.head(n_sample_rows).to_dict(orient="records")
    sample_lines = "\n".join(str(row) for row in sample)

    system = (
        "You are a data documentation assistant. Given a tabular dataset, you write "
        "concise, accurate domain descriptions for each feature so a downstream model "
        "can reason about it. Be factual; do not invent units you cannot infer."
    )
    parts = [
        "Describe each feature of this dataset.",
        f"\nColumns and dtypes:\n{dtypes}",
        f"\nFirst {min(n_sample_rows, len(X))} rows:\n{sample_lines}",
    ]
    if target:
        parts.append(f"\nThe prediction target is '{target}'. Phrase descriptions with that outcome in mind.")
    if data_dictionary:
        parts.append(f"\nAnalyst-provided data dictionary / notes:\n{data_dictionary}")
    parts.append(
        "\nReturn a JSON object with keys 'dataset_description' (one sentence), "
        "'target_description' (one sentence, or empty), and 'features' (an array). "
        "Each feature has 'name' (exactly matching a column), 'description' "
        "(one concise line), optional 'unit', and optional 'category_labels' as a "
        "list of {code, label} for categorical/coded columns. Include every column."
    )
    return system, "\n".join(parts)


def generate_feature_context(
    X: "pd.DataFrame",
    target: str | None = None,
    *,
    model: str = "claude-opus-5-5",
    n_sample_rows: int = 5,
    data_dictionary: str | None = None,
    temperature: float = 0.4,
    client=None,
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
        LLM used to draft the context. Defaults to Claude Opus 5.5.
    n_sample_rows:
        How many example rows to show the model.
    data_dictionary:
        Optional free-text notes / codebook to ground the descriptions.
    temperature:
        Sampling temperature (low by default — this is a description task).
    client:
        Optional pre-built provider client (mainly for testing / reuse).

    Returns
    -------
    FeatureContext
        Always in the original ``X`` column order, with every column present
        (columns the model omitted get a blank description to fill in).
    """
    if client is None:
        from .providers import LLMClient

        client = LLMClient(model)

    system, user = _build_context_prompt(X, target, n_sample_rows, data_dictionary)
    payload = client.complete_json(system, user, schema=_CONTEXT_SCHEMA, temperature=temperature)

    drafted = {}
    for item in payload.get("features", []):
        if not isinstance(item, dict) or "name" not in item:
            continue
        labels = {}
        for pair in item.get("category_labels", []) or []:
            if isinstance(pair, dict) and "code" in pair and "label" in pair:
                labels[str(pair["code"])] = str(pair["label"])
        unit = item.get("unit") or None
        drafted[str(item["name"])] = FeatureDescription(
            name=str(item["name"]),
            description=str(item.get("description", "")),
            unit=unit,
            category_labels=labels,
        )

    # Preserve X column order; fill any omitted columns with a blank entry.
    features = [drafted.get(col, FeatureDescription(name=col)) for col in X.columns]
    return FeatureContext(
        features=features,
        target=target,
        target_description=str(payload.get("target_description", "")),
        dataset_description=str(payload.get("dataset_description", "")),
    )
