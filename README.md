# expert-priors

**LLM-elicited expert priors for regularized statistical models — better out-of-domain prediction with scalable domain knowledge.**

Point `expert-priors` at a dataset and a target. An LLM elicits coefficient
priors (and can write the feature descriptions for you), and you get
scikit-learn-style prior-informed models that hold up better when the test data
drifts away from the training domain.

> Distilled into a clean, installable library from the master's thesis
> *“LLM-Elicited Priors for Regularized Statistical Models.”* The full frozen
> reproducibility pipeline (6 datasets × 12 LLM cells, locked CV + bootstrap)
> lives in the companion thesis repo — linked below.

> ⚠️ **Status: early scaffold (v0.1.0).** The public API and data containers are
> in place and tested; elicitation, model fitting, and benchmarking are being
> ported from the thesis code and currently raise `NotImplementedError`. See the
> roadmap below.

## Install

```bash
pip install -e ".[all]"        # all providers + plotting
# or pick what you need: ".[anthropic]", ".[openai]", ".[google]", ".[plot]"
```

## Quickstart (target API)

```python
import expert_priors as ep

# 1. Elicit priors. auto_context=True lets the LLM draft the feature
#    descriptions instead of you hand-writing them. Or pass prompt=... for
#    a fully custom prompt, or context=FeatureContext(...) for your own.
priors = ep.elicit_priors(
    X, y,
    task="regression",            # or "classification"
    model="claude-opus-4-8",      # latest Claude by default
    n_samples=50,
    auto_context=True,
    cache="priors/mydata.json",   # reruns/demos need no API key
)

# 2. Fit a prior-informed model — sklearn-compatible (fit/predict/get_params)
model = ep.PriorMeanRidge(priors, alpha="cv").fit(X_train, y_train)
preds = model.predict(X_test)

# 3. Or benchmark baseline vs every prior family on ID + OOD splits
report = ep.compare(X, y, priors, ood_mask=mask)
report.summary()   # tidy DataFrame
report.plot()      # OOD-lift figure
```

## Prior families

| Family | Penalty |
|---|---|
| `PriorMean{Ridge,Logistic}` | L2 toward the LLM prior mean μ |
| `Correlated{Ridge,Logistic}` | full precision `P = Σ⁻¹` from the LLM sample covariance |
| `Independent{Ridge,Logistic}` | diagonal precision `P_jj = λ / (var_j + ε)` |

Baselines for "OOD lift" comparisons: **Ridge** (regression) and **L2 Logistic**
(classification).

## Roadmap

- [x] Package scaffold, public API, `PriorBundle` + `FeatureContext` (tested)
- [x] Prior-informed model families as sklearn estimators (numerically matched to the thesis classes)
- [x] Elicitation + multi-provider call layer (Anthropic/OpenAI/Google), parallel sampling, disk caching
- [x] Auto feature-context generator (`auto_context=True` / `generate_feature_context`)
- [x] `compare()` benchmark (ID/OOD metrics, lift-vs-baseline, plotting)
- [ ] Example notebook with cached demo priors (no API key needed)

## Companion thesis repo

Full methodology, locked experiments, and reproducibility bundles: **llm-statistics-clean**.

## License

MIT
