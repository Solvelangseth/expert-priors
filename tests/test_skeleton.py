"""Tests for the parts of the scaffold that are fully implemented.

The elicitation / model-fit / compare bodies are intentionally stubbed (ported in
later build steps); these tests cover the data containers and public API surface.
"""

import json

import numpy as np
import pandas as pd
import pytest

import expert_priors as ep
from expert_priors.bundle import PriorBundle
from expert_priors.context import FeatureContext
from expert_priors.providers import detect_provider


def test_version_and_public_api():
    assert isinstance(ep.__version__, str)
    for name in ["elicit_priors", "PriorBundle", "FeatureContext", "PriorMeanRidge", "compare"]:
        assert hasattr(ep, name)


def test_prior_bundle_summaries_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(50, 3))
    bundle = PriorBundle(["a", "b", "c"], samples, task="regression", model="claude-opus-4-8")

    assert bundle.n_samples == 50
    assert bundle.n_features == 3
    assert bundle.mean().shape == (3,)
    assert bundle.cov().shape == (3, 3)
    assert bundle.var().shape == (3,)

    path = bundle.save(tmp_path / "p.json")
    again = PriorBundle.load(path)
    assert again.feature_names == bundle.feature_names
    np.testing.assert_allclose(again.samples, bundle.samples)


def test_prior_bundle_align_reorders_and_subsets():
    samples = np.arange(6, dtype=float).reshape(2, 3)
    bundle = PriorBundle(["a", "b", "c"], samples)
    aligned = bundle.align_to(["c", "a"])
    assert aligned.feature_names == ["c", "a"]
    np.testing.assert_allclose(aligned.samples, samples[:, [2, 0]])
    with pytest.raises(KeyError):
        bundle.align_to(["a", "missing"])


def test_prior_bundle_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        PriorBundle(["a", "b"], np.zeros((4, 3)))


def test_feature_context_prompt_block_and_edit():
    ctx = FeatureContext.from_dict(
        {
            "target": "income",
            "dataset_description": "Census adults.",
            "features": [
                {"name": "age", "description": "age in years", "unit": "years"},
                {"name": "sex", "description": "sex", "category_labels": {"0": "female", "1": "male"}},
            ],
        }
    )
    block = ctx.to_prompt_block()
    assert "age in years" in block
    assert "0=female" in block
    ctx.edit("age", description="respondent age")
    assert "respondent age" in ctx.to_prompt_block()
    assert json.loads(json.dumps(ctx.to_dict()))["target"] == "income"


def test_feature_context_blank_from_dataframe():
    X = pd.DataFrame({"x1": [1, 2], "x2": [3, 4]})
    ctx = FeatureContext.blank(list(X.columns), target="y")
    assert ctx.feature_names == ["x1", "x2"]


def test_detect_provider():
    assert detect_provider("claude-opus-4-8") == "anthropic"
    assert detect_provider("gpt-5.1") == "openai"
    assert detect_provider("gemini-2.5-pro") == "google"
    with pytest.raises(ValueError):
        detect_provider("mystery-model")


def test_elicit_uses_cache_when_present(tmp_path):
    cache = tmp_path / "cached.json"
    PriorBundle(["a", "b"], np.ones((3, 2)), model="claude-opus-4-8").save(cache)
    X = pd.DataFrame({"a": [1.0], "b": [2.0]})
    bundle = ep.elicit_priors(X, task="regression", cache=cache)  # no API call: cache hit
    assert bundle.n_features == 2
    assert bundle.model == "claude-opus-4-8"
