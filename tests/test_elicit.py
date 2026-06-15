"""Tests for elicitation, the provider parsing layer, prompts, and summaries.

No network: a FakeClient stands in for the provider SDK so the sampling loop,
prompt rendering, and bundle assembly are exercised deterministically.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from expert_priors.context import FeatureContext
from expert_priors.elicit import PriorElicitor, elicit_priors
from expert_priors.prompts import build_system_prompt, build_user_prompt
from expert_priors.providers import detect_provider, parse_coefficients
from expert_priors.summaries import feature_summary_block


class FakeClient:
    """Returns deterministic per-call coefficients and records prompts seen."""

    def __init__(self):
        self.calls = []
        self._counter = itertools.count()

    def complete_coefficients(self, system, user, feature_names, *, temperature=0.8):
        i = next(self._counter)
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return {f: float(i + j) for j, f in enumerate(feature_names)}


# --- provider parsing -----------------------------------------------------
def test_parse_plain_json():
    out = parse_coefficients('{"a": 1.0, "b": -2.5}', ["a", "b"])
    assert out == {"a": 1.0, "b": -2.5}


def test_parse_fenced_and_wrapped():
    assert parse_coefficients('```json\n{"a": 1, "b": 2}\n```', ["a", "b"]) == {"a": 1.0, "b": 2.0}
    wrapped = '{"reasoning": "...", "coefficients": {"a": 3, "b": 4}}'
    assert parse_coefficients(wrapped, ["a", "b"]) == {"a": 3.0, "b": 4.0}


def test_parse_embedded_in_prose():
    text = "Sure! Here you go: {\"a\": 0.1, \"b\": 0.2} hope that helps"
    assert parse_coefficients(text, ["a", "b"]) == {"a": 0.1, "b": 0.2}


def test_parse_rejects_missing_keys_and_nonfinite():
    with pytest.raises(ValueError):
        parse_coefficients('{"a": 1}', ["a", "b"])
    with pytest.raises(ValueError):
        parse_coefficients('{"a": 1, "b": "Infinity"}', ["a", "b"])
    with pytest.raises(ValueError):
        parse_coefficients("", ["a"])


def test_detect_provider_default_claude():
    assert detect_provider("claude-opus-4-8") == "anthropic"
    assert detect_provider("gpt-5.1") == "openai"
    assert detect_provider("gemini-2.5-pro") == "google"


# --- summaries & prompts --------------------------------------------------
def test_feature_summary_block():
    X = pd.DataFrame({"flag": [0, 1, 1, 1], "age": [20, 30, 40, 80]})
    block = feature_summary_block(X)
    assert "flag: binary 0/1 indicator; prevalence=0.750" in block
    assert "age: continuous/raw numeric" in block


def test_prompt_builders_mention_task_and_features():
    sys_reg = build_system_prompt("regression")
    sys_clf = build_system_prompt("classification")
    assert "regression model" in sys_reg
    assert "log-odds" in sys_clf
    ctx = FeatureContext.from_dict(
        {"target": "y", "features": [{"name": "age", "description": "age in years"}]}
    )
    user = build_user_prompt(ctx, "- age: continuous", "regression")
    assert "age in years" in user
    assert "Target outcome: y" in user
    assert "- age: continuous" in user


# --- elicitation loop -----------------------------------------------------
def _ctx():
    return FeatureContext.blank(["a", "b", "c"], target="y")


def test_elicitor_collects_samples_in_order():
    elic = PriorElicitor("claude-opus-4-8", "regression", n_samples=4,
                         max_workers=1, client=FakeClient())
    bundle = elic.elicit(_ctx(), feature_summaries="- a: x")
    assert bundle.feature_names == ["a", "b", "c"]
    assert bundle.samples.shape == (4, 3)
    assert bundle.model == "claude-opus-4-8"
    assert bundle.metadata["temperature"] == 0.8
    # FakeClient sample i = [i, i+1, i+2]; sequential -> first row [0,1,2]
    np.testing.assert_allclose(bundle.samples[0], [0, 1, 2])


def test_elicitor_parallel_shape():
    elic = PriorElicitor("claude-opus-4-8", "regression", n_samples=10,
                         max_workers=4, client=FakeClient())
    bundle = elic.elicit(_ctx())
    assert bundle.samples.shape == (10, 3)


def test_custom_prompt_overrides_user_text():
    fake = FakeClient()
    elic = PriorElicitor("claude-opus-4-8", n_samples=1, max_workers=1,
                         prompt="MY CUSTOM PROMPT", client=fake)
    elic.elicit(_ctx(), feature_summaries="ignored")
    assert fake.calls[0]["user"] == "MY CUSTOM PROMPT"


def test_elicit_priors_cache_roundtrip(tmp_path, monkeypatch):
    # Force elicitation through a fake client by patching the default client.
    import expert_priors.elicit as elicit_mod

    fake = FakeClient()

    class _Elic(PriorElicitor):
        def __init__(self, *a, **k):
            k["client"] = fake
            super().__init__(*a, **k)

    monkeypatch.setattr(elicit_mod, "PriorElicitor", _Elic)
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    cache = tmp_path / "b.json"
    first = elicit_priors(X, task="regression", n_samples=3, cache=cache, max_workers=1)
    assert cache.exists()
    assert first.samples.shape == (3, 2)
    # second call hits the cache (no new client calls)
    n_before = len(fake.calls)
    second = elicit_priors(X, task="regression", n_samples=3, cache=cache)
    assert len(fake.calls) == n_before
    np.testing.assert_allclose(first.samples, second.samples)
