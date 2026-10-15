# Probabilistic Forecasting: a notebook course (monthly data)

**Repository:** https://github.com/abhijeetkumar1/beyond-point-forecasts

Fifteen self-contained Jupyter notebooks. Each one explains the **theory** of one type of probabilistic forecast
(with formulas), then implements it from scratch on **monthly** data and evaluates it with proper scoring rules.
The notebooks are saved *with outputs*, so you can read them without running anything.

| # | Notebook | Covers |
|---|----------|--------|
| 00 | `00_foundations` | point vs probabilistic, aleatoric/epistemic, calibration & sharpness, PIT, proper scoring rules, losses that elicit mean/median/quantiles, CRPS, rolling-origin backtests |
| 01 | `01_full_predictive_distribution` | SARIMA → log-normal predictive density, σ_h growth via ψ-weights, deriving any quantity from F, CRPS / log score (with Jacobian) / PIT |
| 02 | `02_quantile_forecasts` | pinball loss derivation, quantile regression & heteroscedasticity, parametric vs empirical quantiles, wQL, quantile calibration, crossing & rearrangement |
| 03 | `03_prediction_intervals` | PI vs CI, analytic / bootstrap / empirical-error / naive intervals, coverage by horizon, Winkler score, pointwise vs path coverage |
| 04 | `04_conformal_prediction` | split conformal & the (n+1) correction, per-horizon conformal for time series, ACI under regime change, CQR |
| 05 | `05_ensembles_and_sample_paths` | simulation, parameter-uncertainty and multi-model ensembles, path functionals (annual totals, peaks), linear pool vs quantile averaging, rank histograms, spread–skill, foundation models |
| 06 | `06_event_probabilities` | P(event) from distributions vs classifiers, Brier / log loss / Murphy decomposition, reliability diagrams, AUC caveat, isotonic recalibration, cost–loss value curves |
| 07 | `07_categorical_and_binned` | tercile outlooks, RPS vs Brier, multinomial logit, histogram (binned density) forecasts, randomised PIT |
| 08 | `08_distribution_families_and_counts` | choosing a family, Student-t tails, intermittent demand (ADI/CV²), Poisson vs Negative Binomial vs Gaussian, discrete CRPS & randomised PIT |
| 09 | `09_multivariate_joint` | Sklar / copulas, Gaussian copula, Schaake shuffle, energy & variogram scores, why marginal scores miss dependence |
| 10 | `10_hierarchical_reconciliation` | summing matrix, bottom-up / OLS / WLS / MinT-shrink, probabilistic reconciliation of joint samples |
| 11 | `11_bayesian_forecasting` | posterior predictive, epistemic vs aleatoric decomposition, conjugate regression (and prior traps), AR posterior simulation, Gaussian process in NumPy, PyMC sketch |
| 12 | `12_evaluation_and_decisions` | full backtest scoreboard, skill scores, Diebold–Mariano, newsvendor / critical-ratio decisions, choosing the forecast type |
| 13 | `13_temporal_convolutional_networks` | causal & dilated convolutions, receptive field, residual blocks, global models on a 300-SKU panel, non-crossing multi-horizon quantile head, integer quantiles for counts, conformal calibration (needs PyTorch) |
| 14 | `14_deepar` | autoregressive LSTM + NegBin / Gaussian likelihood, scaling, teacher forcing, ancestral sampling of paths, P(zero sales), annual totals from paths (needs PyTorch) |

Suggested order: 00 → 01 → 02 → 03 → 04 → 05, then any of 06–11, then 12. 13 → 14 are the deep-learning extension.

## Data (all monthly, generated or embedded by `pf_utils.py`, so no downloads are needed)
* **AirPassengers** 1949–1960 (embedded): trend + multiplicative seasonality.
* **Synthetic retail hierarchy** 2015–2024: Total → North/South → 4 stores, with common + regional AR(1) shocks.
* **Synthetic SKU demand** 2012–2024: lumpy, over-dispersed count data (Gamma–Poisson).
* **Synthetic SKU panel** 2015–2024 (nb 13–14): 300 related count series with four seasonal families, scales from <1 to
  hundreds of units, latent demand shocks, and staggered product launches. The true rates are returned too, so the
  notebooks can show an oracle benchmark.

`pf_utils.py` also holds every scoring rule (pinball, wQL, CRPS, interval score, Brier, RPS, energy, variogram, PIT)
and the **Plotly** helpers: one shared template (a colour-blind-safe palette in a fixed order, so the same method or
series keeps the same colour) plus fan charts, PIT histograms, reliability diagrams, step histograms and sample-path plots.
All charts are interactive: hover a fan chart to read every interval for that month.

## Setup
Only numpy, pandas, scipy, plotly (+ nbformat, which Plotly needs to display figures in notebooks) and statsmodels are
needed for notebooks 00–12. No scikit-learn. Notebooks 13–14 also need **PyTorch** (`pip install torch`; the CPU build is
enough, and each runs in 1–2 minutes). The saved outputs are Plotly figures, so open the notebooks in
JupyterLab, Jupyter Notebook or VS Code to see them (GitHub's notebook preview does not render Plotly).

```bash
git clone https://github.com/abhijeetkumar1/beyond-point-forecasts.git
cd beyond-point-forecasts
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/jupyter lab
```

To export a chart as PNG/SVG, `pip install kaleido` and call `fig.write_image("chart.png")`.

Optional extras mentioned in the notebooks: `chronos-forecasting` (nb 05), `pymc` (nb 11),
`hierarchicalforecast` (nb 10), `lightgbm` (nb 02).
