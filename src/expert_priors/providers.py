"""Provider abstraction for LLM calls (OpenAI / Anthropic / Google).

Ported from the thesis ``llm_prior_elicitor_v2`` multi-provider layer. Default
provider is the latest Claude model; OpenAI and Gemini are supported for the
cross-provider comparison the thesis runs. Each provider SDK is an optional
dependency — import errors are deferred until a provider is actually used.
"""

from __future__ import annotations

from typing import Any, Sequence


def detect_provider(model_name: str) -> str:
    """Infer the provider from a model id.

    >>> detect_provider("claude-opus-4-8")
    'anthropic'
    >>> detect_provider("gpt-5.1")
    'openai'
    >>> detect_provider("gemini-2.5-pro")
    'google'
    """
    m = model_name.lower()
    if m.startswith(("claude", "anthropic")):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "openai")):
        return "openai"
    if m.startswith(("gemini", "google")):
        return "google"
    raise ValueError(f"Could not infer provider from model id {model_name!r}; pass provider= explicitly.")


class LLMClient:
    """Thin wrapper that routes structured-output calls to the right SDK.

    TODO(port): lift the provider-specific call/parse logic out of the thesis
    ``llm_prior_elicitor_v2.LLMPriorElicitor`` (the ``_call_openai`` /
    ``_call_anthropic`` / ``_call_google`` methods and the JSON-schema
    constraining). Keep the structured-output schema so coefficient dicts come
    back reliably.
    """

    def __init__(self, model: str, provider: str | None = None) -> None:
        self.model = model
        self.provider = provider or detect_provider(model)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_keys: Sequence[str],
        *,
        temperature: float = 0.8,
    ) -> dict[str, Any]:
        """Return a JSON object constrained to ``schema_keys``."""
        raise NotImplementedError("provider call layer not yet ported — see TODO in providers.py")
