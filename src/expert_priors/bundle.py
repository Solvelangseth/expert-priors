"""PriorBundle — the container for LLM-elicited coefficient priors.

A bundle holds the raw per-sample coefficient vectors returned by an LLM (in raw
feature units) plus the schema needed to align them to a design matrix. Summary
objects used by the model families (mean vector, covariance, diagonal variance)
are derived from the samples on demand, so a single bundle drives every prior
family consistently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class PriorBundle:
    """A set of LLM-elicited coefficient priors in raw feature space.

    Parameters
    ----------
    feature_names:
        Ordered feature schema. Defines the column order of ``samples`` and the
        contract every consuming model must match.
    samples:
        Array of shape ``(n_samples, n_features)`` — one elicited coefficient
        vector per LLM draw, in raw feature units.
    task:
        ``"regression"`` or ``"classification"``.
    model:
        Identifier of the LLM that produced the samples (e.g. ``"claude-opus-4-8"``).
    metadata:
        Free-form provenance (prompt variant, temperature, timestamp, context used).
    """

    feature_names: list[str]
    samples: np.ndarray
    task: str = "regression"
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.feature_names = list(self.feature_names)
        self.samples = np.atleast_2d(np.asarray(self.samples, dtype=float))
        if self.samples.shape[1] != len(self.feature_names):
            raise ValueError(
                f"samples has {self.samples.shape[1]} columns but "
                f"{len(self.feature_names)} feature_names were given."
            )
        if self.task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got {self.task!r}")

    # -- derived summaries -------------------------------------------------
    @property
    def n_samples(self) -> int:
        return self.samples.shape[0]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def mean(self) -> np.ndarray:
        """Prior mean coefficient vector (sample mean)."""
        return self.samples.mean(axis=0)

    def cov(self) -> np.ndarray:
        """Sample covariance of the elicited coefficients (n_features x n_features)."""
        if self.n_samples < 2:
            return np.zeros((self.n_features, self.n_features))
        return np.cov(self.samples, rowvar=False)

    def var(self) -> np.ndarray:
        """Per-feature variance of the elicited coefficients."""
        return self.samples.var(axis=0, ddof=1) if self.n_samples > 1 else np.zeros(self.n_features)

    def align_to(self, feature_names: Sequence[str]) -> "PriorBundle":
        """Return a bundle reordered/subset to match a design-matrix schema.

        Raises if any requested feature is missing from the bundle.
        """
        idx = []
        for name in feature_names:
            if name not in self.feature_names:
                raise KeyError(f"feature {name!r} not present in prior bundle")
            idx.append(self.feature_names.index(name))
        return PriorBundle(
            feature_names=list(feature_names),
            samples=self.samples[:, idx],
            task=self.task,
            model=self.model,
            metadata={**self.metadata, "aligned": True},
        )

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "samples": self.samples.tolist(),
            "task": self.task,
            "model": self.model,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        """Write the bundle to JSON. Returns the path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PriorBundle":
        return cls(
            feature_names=list(data["feature_names"]),
            samples=np.asarray(data["samples"], dtype=float),
            task=data.get("task", "regression"),
            model=data.get("model"),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PriorBundle":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PriorBundle(task={self.task!r}, model={self.model!r}, "
            f"n_samples={self.n_samples}, n_features={self.n_features})"
        )
