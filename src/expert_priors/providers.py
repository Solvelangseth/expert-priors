"""Provider abstraction for LLM calls (Anthropic / OpenAI / Google).

Ported from the thesis ``llm_prior_elicitor_v2`` multi-provider layer, trimmed to
the non-reasoning structured-output path the library needs. Each call asks the
model for a strict JSON object of ``{feature: coefficient}`` and parses/validates
it back into a Python dict.

The default model is the latest Claude (Anthropic tool-use path). Each provider
SDK is an optional dependency — its import is deferred until the provider is used.
"""

from __future__ import annotations

import json
import math
import re
from typing import Sequence


_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    """Best-effort load of a local .env so API keys are picked up automatically."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


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
    raise ValueError(
        f"Could not infer provider from model id {model_name!r}; pass provider= explicitly."
    )


# --- JSON parsing helpers (provider-agnostic) -----------------------------
def _extract_first_braced_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, ignoring braces in strings."""
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
    return None


def _coerce_coefficients(payload: object, feature_names: Sequence[str]) -> dict[str, float]:
    """Validate a parsed object against the feature schema and return floats."""
    if isinstance(payload, dict):
        for key in ("coefficients", "targets"):
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise ValueError("parsed payload is not a JSON object")

    feature_set = set(feature_names)
    missing = [f for f in feature_names if f not in payload]
    extra = [k for k in payload if k not in feature_set]
    if missing or extra:
        raise ValueError(f"keys mismatch: missing={missing[:10]} extra={extra[:10]}")

    out: dict[str, float] = {}
    for feature in feature_names:
        value = float(payload[feature])
        if not math.isfinite(value):
            raise ValueError(f"non-finite value for feature {feature!r}")
        out[feature] = value
    return out


def parse_json_object(text: str | None) -> dict:
    """Parse the first valid JSON object from raw model text (fences/prose tolerated)."""
    if not text or not str(text).strip():
        raise ValueError("empty response")
    raw = str(text).strip()
    candidates = [raw]
    candidates += [c.strip() for c in re.findall(r"```json\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)]
    candidates += [c.strip() for c in re.findall(r"```\s*(.*?)```", raw, re.DOTALL)]
    inline = _extract_first_braced_object(raw)
    if inline:
        candidates.append(inline.strip())
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception as exc:
            last_error = exc
    raise ValueError(f"unable to parse JSON object ({last_error})")


def parse_coefficients(text: str | None, feature_names: Sequence[str]) -> dict[str, float]:
    """Parse coefficients from raw model text, tolerating fences / prose."""
    if not text or not str(text).strip():
        raise ValueError("empty response")
    raw = str(text).strip()

    candidates = [raw]
    candidates += [c.strip() for c in re.findall(r"```json\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)]
    candidates += [c.strip() for c in re.findall(r"```\s*(.*?)```", raw, re.DOTALL)]
    inline = _extract_first_braced_object(raw)
    if inline:
        candidates.append(inline.strip())

    seen: set[str] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return _coerce_coefficients(json.loads(candidate), feature_names)
        except Exception as exc:  # keep trying other candidates
            last_error = exc
    raise ValueError(f"unable to parse coefficients JSON ({last_error})")


# --- schema builders ------------------------------------------------------
def _coefficient_schema(feature_names: Sequence[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {f: {"type": "number"} for f in feature_names},
        "required": list(feature_names),
    }


def _schema_constrained_prompt(user_prompt: str, feature_names: Sequence[str]) -> str:
    return (
        user_prompt.rstrip()
        + "\n\nOutput format (strict):\n"
        + "- Return only a JSON object. No markdown, code fences, or commentary.\n"
        + "- Include exactly these keys (no more, no fewer): "
        + json.dumps(list(feature_names), ensure_ascii=False)
        + "\n- Every value must be a finite numeric coefficient in raw-feature units.\n"
    )


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "temperature" in msg and any(
        s in msg for s in ("unsupported parameter", "unsupported value", "does not support", "only the default")
    )


# --- client ---------------------------------------------------------------
class LLMClient:
    """Routes a structured coefficient request to the right provider SDK."""

    def __init__(
        self,
        model: str,
        provider: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.provider = provider or detect_provider(model)
        self.api_key = api_key
        self.timeout = timeout
        _load_dotenv_once()

    def complete_coefficients(
        self,
        system_prompt: str,
        user_prompt: str,
        feature_names: Sequence[str],
        *,
        temperature: float = 0.8,
    ) -> dict[str, float]:
        """Request one coefficient vector and return it as ``{feature: value}``."""
        caller = {
            "anthropic": self._call_anthropic,
            "openai": self._call_openai,
            "google": self._call_google,
        }.get(self.provider)
        if caller is None:
            raise ValueError(f"unsupported provider: {self.provider}")
        try:
            text = caller(system_prompt, user_prompt, feature_names, temperature)
        except Exception as exc:
            raise RuntimeError(f"[{self.provider}] call failed: {exc}") from exc
        return parse_coefficients(text, feature_names)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        schema: dict | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4000,
    ) -> dict:
        """Request a free-form JSON object (used by the auto-context generator)."""
        try:
            if self.provider == "anthropic":
                obj = self._json_anthropic(system_prompt, user_prompt, schema, temperature, max_tokens)
            elif self.provider == "openai":
                obj = self._json_openai(system_prompt, user_prompt, temperature, max_tokens)
            elif self.provider == "google":
                obj = self._json_google(system_prompt, user_prompt, schema, temperature)
            else:
                raise ValueError(f"unsupported provider: {self.provider}")
        except Exception as exc:
            raise RuntimeError(f"[{self.provider}] json call failed: {exc}") from exc
        return obj

    def _json_anthropic(self, system_prompt, user_prompt, schema, temperature, max_tokens) -> dict:
        import os

        import anthropic

        key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("missing ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=key)
        if schema is not None:
            tool_name = "emit"
            resp = client.messages.create(
                model=self.model, system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens, temperature=float(temperature),
                tools=[{"name": tool_name, "description": "Emit the requested JSON object.",
                        "input_schema": schema}],
                tool_choice={"type": "tool", "name": tool_name},
            )
            for block in getattr(resp, "content", []) or []:
                btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                payload = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
                if btype == "tool_use" and isinstance(payload, dict):
                    return payload
        resp = client.messages.create(
            model=self.model, system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens, temperature=float(temperature),
        )
        text = "\n".join(getattr(b, "text", "") or "" for b in getattr(resp, "content", []) or [])
        return parse_json_object(text)

    def _json_openai(self, system_prompt, user_prompt, temperature, max_tokens) -> dict:
        import os

        from openai import OpenAI

        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("missing OPENAI_API_KEY")
        client = OpenAI(api_key=key, timeout=self.timeout, max_retries=0)
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_tokens,
            "temperature": float(temperature),
        }
        try:
            resp = client.chat.completions.create(**params)
        except Exception as exc:
            if not _is_unsupported_temperature_error(exc):
                raise
            params.pop("temperature", None)
            resp = client.chat.completions.create(**params)
        return parse_json_object(resp.choices[0].message.content or "")

    def _json_google(self, system_prompt, user_prompt, schema, temperature) -> dict:
        import os

        import google.generativeai as genai

        key = self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("missing GOOGLE_API_KEY or GEMINI_API_KEY")
        genai.configure(api_key=key)
        model = genai.GenerativeModel(self.model, system_instruction=system_prompt)
        config = {"temperature": float(temperature), "response_mime_type": "application/json"}
        if schema is not None:
            config["response_schema"] = schema
        try:
            resp = model.generate_content(user_prompt, generation_config=config)
        except Exception:
            resp = model.generate_content(
                user_prompt, generation_config={"temperature": float(temperature)}
            )
        return parse_json_object(str(getattr(resp, "text", "") or ""))

    # -- provider implementations -----------------------------------------
    def _call_anthropic(self, system_prompt, user_prompt, feature_names, temperature) -> str:
        import os

        import anthropic

        key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("missing ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=key)
        tool_name = "prior_coefficients"
        prompt = _schema_constrained_prompt(user_prompt, feature_names)
        try:
            resp = client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=float(temperature),
                tools=[{
                    "name": tool_name,
                    "description": "Return prior coefficients as a strict JSON object.",
                    "input_schema": _coefficient_schema(feature_names),
                }],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except Exception:
            resp = client.messages.create(
                model=self.model, system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200, temperature=float(temperature),
            )
        for block in getattr(resp, "content", []) or []:
            btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            payload = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
            if btype == "tool_use" and isinstance(payload, dict):
                return json.dumps(payload)
        return "\n".join(
            getattr(b, "text", "") or "" for b in getattr(resp, "content", []) or []
        )

    def _call_openai(self, system_prompt, user_prompt, feature_names, temperature) -> str:
        import os

        from openai import OpenAI

        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("missing OPENAI_API_KEY")
        client = OpenAI(api_key=key, timeout=self.timeout, max_retries=0)
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _schema_constrained_prompt(user_prompt, feature_names)},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 1600,
            "temperature": float(temperature),
        }
        try:
            resp = client.chat.completions.create(**params)
        except Exception as exc:
            if not _is_unsupported_temperature_error(exc):
                raise
            params.pop("temperature", None)
            resp = client.chat.completions.create(**params)
        return resp.choices[0].message.content or ""

    def _call_google(self, system_prompt, user_prompt, feature_names, temperature) -> str:
        import os

        import google.generativeai as genai

        key = self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("missing GOOGLE_API_KEY or GEMINI_API_KEY")
        genai.configure(api_key=key)
        model = genai.GenerativeModel(self.model, system_instruction=system_prompt)
        prompt = _schema_constrained_prompt(user_prompt, feature_names)
        schema = {
            "type": "object",
            "properties": {f: {"type": "number"} for f in feature_names},
            "required": list(feature_names),
        }
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": float(temperature),
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )
        except Exception:
            resp = model.generate_content(prompt, generation_config={"temperature": float(temperature)})
        return str(getattr(resp, "text", "") or "").strip()
