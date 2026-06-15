"""Default elicitation prompt builder.

Renders the system + user prompts for the ``feature_context`` style used in the
thesis: ground the model in raw-feature units, give it the feature meanings and
scale summaries, and ask for one plausible coefficient draw. You can bypass this
entirely by passing a custom ``prompt=`` to :func:`expert_priors.elicit_priors`.
"""

from __future__ import annotations

from .context import FeatureContext

_TASK_NOUN = {"regression": "regression model", "classification": "logistic (log-odds) model"}


def build_system_prompt(task: str, *, sample_mode: bool = True) -> str:
    noun = _TASK_NOUN.get(task, "regression model")
    rules = [
        f"You are helping construct informative priors for a {noun}.",
        "Use the feature definitions and feature-scale summaries to reason about BOTH sign and magnitude.",
        "All coefficients must be given in the original raw-feature units used by the model matrix.",
        "Do not standardize, rescale, or reinterpret the coefficients as standardized effects.",
        "Do not include an intercept.",
        "Output must be only a JSON object matching the provided schema.",
        "Do not include explanations, markdown, code fences, or extra keys.",
        "Prefer interpretable, conservative, domain-plausible coefficients.",
        "Avoid extreme magnitudes unless clearly justified by the variable scale and domain mechanism.",
    ]
    if sample_mode:
        rules += [
            "Return one plausible coefficient vector draw from your belief distribution.",
            "Do not collapse uncertainty into a single average if multiple plausible settings exist.",
            "Keep the draw realistic and domain-grounded rather than intentionally noisy.",
        ]
    else:
        rules += [
            "Return your best central estimate of the coefficient vector.",
            "Prefer stable, representative values over edge-case values.",
        ]
    return "\n".join(rules) + "\n"


def build_user_prompt(context: FeatureContext, feature_summaries: str, task: str) -> str:
    outcome = "Provide prior coefficients for the model described below."
    if context.target:
        outcome = f"Target outcome: {context.target}"
        if context.target_description:
            outcome += f" — {context.target_description}"

    sections = ["Task context", outcome, ""]
    if context.dataset_description:
        sections += ["Domain context:", context.dataset_description, ""]
    sections += [
        "Feature meanings:",
        "\n".join(f.to_prompt_line() for f in context.features),
        "",
        "Feature scale summaries from the actual design matrix:",
        feature_summaries,
        "",
        "Important modeling reminders:",
        "- Binary indicators use 0/1 coding, so their coefficient is the change from turning the indicator on.",
        "- Continuous-feature coefficients should be scaled to the feature summaries shown above.",
        "- When multiple variables capture overlapping structure, distribute signal conservatively.",
        "- The output schema already fixes the keys. Choose only the numeric values.",
    ]
    return "\n".join(sections).strip() + "\n"
