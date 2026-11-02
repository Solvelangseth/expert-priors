# Examples

**[`quickstart.ipynb`](quickstart.ipynb)** — the 60-second tour, runnable with
**no API key**. Raw housing data → cached LLM priors → a prior-informed model →
an ID/OOD benchmark showing the out-of-domain lift. Outputs are pre-rendered so
it reads on GitHub without running.

## Files

- `demo_data.py` — the synthetic housing dataset (with a built-in domain shift) and its feature context.
- `priors/housing_prior.json` — priors elicited once with `claude-opus-4-8` and cached, so the notebook needs no API call.
- `build_demo_priors.py` — regenerate the cached priors yourself (needs an API key):

  ```bash
  python examples/build_demo_priors.py --model claude-opus-4-8 --samples 20
  ```
- `generate_figure.py` — render the README figure from the cached priors (no API key):

  ```bash
  python examples/generate_figure.py   # writes figures/regularization_paths.png
  ```
- `figures/regularization_paths.png` — the regularization-path figure shown in the top-level README.
