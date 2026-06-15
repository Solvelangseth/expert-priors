"""Tests for generate_feature_context and the generic JSON parsing path."""

import pandas as pd
import pytest

from expert_priors.context import generate_feature_context
from expert_priors.providers import parse_json_object


class FakeJsonClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete_json(self, system, user, *, schema=None, temperature=0.4, max_tokens=4000):
        self.calls.append({"system": system, "user": user, "schema": schema})
        return self.payload


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('here: {"a": 1, "b": 2} done') == {"a": 1, "b": 2}
    with pytest.raises(ValueError):
        parse_json_object("")


def test_generate_feature_context_maps_payload():
    X = pd.DataFrame({"age": [20, 30], "sex": [0, 1], "income": [100.0, 200.0]})
    payload = {
        "dataset_description": "Census records.",
        "target_description": "whether income exceeds 50k",
        "features": [
            {"name": "age", "description": "age in years", "unit": "years"},
            {"name": "sex", "description": "biological sex",
             "category_labels": [{"code": "0", "label": "female"}, {"code": "1", "label": "male"}]},
            {"name": "income", "description": "annual income", "unit": "USD"},
        ],
    }
    client = FakeJsonClient(payload)
    ctx = generate_feature_context(X, target="high_income", client=client)

    assert ctx.dataset_description == "Census records."
    assert ctx.target == "high_income"
    assert ctx.feature_names == ["age", "sex", "income"]  # X column order
    assert ctx["age"].unit == "years"
    assert ctx["sex"].category_labels == {"0": "female", "1": "male"}
    # the prompt actually showed the model the columns and dtypes
    assert "age" in client.calls[0]["user"]
    assert client.calls[0]["schema"] is not None


def test_generate_feature_context_fills_omitted_columns():
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    payload = {"features": [{"name": "a", "description": "feature a"}]}  # b, c omitted
    ctx = generate_feature_context(X, client=FakeJsonClient(payload))
    assert ctx.feature_names == ["a", "b", "c"]  # all present, original order
    assert ctx["a"].description == "feature a"
    assert ctx["b"].description == ""  # blank placeholder to fill in


def test_generate_feature_context_ignores_malformed_feature_entries():
    X = pd.DataFrame({"a": [1, 2]})
    payload = {"features": ["not a dict", {"no_name": "x"}, {"name": "a", "description": "ok"}]}
    ctx = generate_feature_context(X, client=FakeJsonClient(payload))
    assert ctx["a"].description == "ok"
