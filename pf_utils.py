"""Shared helpers for the probabilistic-forecasting notebooks.

Contents
--------
* Datasets (all MONTHLY):
    - load_air_passengers()      classic Box-Jenkins airline data, 1949-1960
    - make_retail_hierarchy()    synthetic 4-store / 2-region retail sales, 2015-2024
    - make_sku_demand()          synthetic intermittent count demand for one SKU
    - make_sku_panel()           300 related SKU count series with launches (for global / deep-learning models)
* Design-matrix helpers (trend + month dummies)
* Panel baselines (seasonal naive, local NegBin GLM) and panel quantile scores
* Scoring rules: pinball, WQL, CRPS (Gaussian + sample), log score, interval/Winkler,
  Brier, log loss, RPS, energy score, variogram score, PIT
* Plotly helpers: one shared template + fan charts, PIT histograms, reliability diagrams,
  step histograms, spaghetti paths and reference lines

Every metric here is NEGATIVELY oriented (lower = better) unless stated otherwise.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats

# ---------------------------------------------------------------------------
# Style: a validated, colour-blind-safe palette (fixed slot order, never cycled)
# ---------------------------------------------------------------------------

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE = {100: "#cde2fb", 200: "#9ec5f4", 300: "#6da7ec", 400: "#3987e5", 450: "#2a78d6", 550: "#1c5cab", 650: "#104281"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
       "axis": "#c3c2b7", "surface": "#fcfcfb", "neutral": "#c3c2b7"}
HISTORY, ACTUAL = INK["secondary"], INK["primary"]     # roles, not series: never collide with SERIES colours
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def set_style(renderer: str = "notebook_connected+plotly_mimetype"):
    """Register the shared Plotly template and display options. Call once, after the imports."""
    # statsmodels re-enables its ConvergenceWarning on import, so silence warnings here
    warnings.filterwarnings("ignore")
    axis = dict(showgrid=True, gridcolor=INK["grid"], gridwidth=1, showline=True, linecolor=INK["axis"],
                ticks="outside", tickcolor=INK["axis"], zeroline=False,
                tickfont=dict(color=INK["secondary"]), title=dict(font=dict(color=INK["secondary"])))
    pio.templates["pf"] = go.layout.Template(layout=go.Layout(
        font=dict(family=FONT, size=12, color=INK["primary"]),
        title=dict(font=dict(size=15, color=INK["primary"]), x=0.01, xanchor="left"),
        paper_bgcolor=INK["surface"], plot_bgcolor=INK["surface"], colorway=SERIES,
        xaxis=axis, yaxis=axis,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK["secondary"]), tracegroupgap=2),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=INK["axis"], font=dict(family=FONT, color=INK["primary"])),
        hovermode="closest", bargap=0.08, barcornerradius=4,
        margin=dict(l=60, r=20, t=70, b=50), height=420,
    ))
    pio.templates.default = "pf"
    pio.renderers.default = renderer
    pd.set_option("display.precision", 3)
    pd.set_option("display.width", 120)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_AIR = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]


def load_air_passengers() -> pd.Series:
    """Monthly international airline passengers (thousands), Jan-1949 .. Dec-1960."""
    idx = pd.date_range("1949-01-01", periods=len(_AIR), freq="MS")
    return pd.Series(np.array(_AIR, dtype=float), index=idx, name="passengers")


HIER_NAMES = ["Total", "North", "South", "North_A", "North_B", "South_A", "South_B"]
BOTTOM_NAMES = HIER_NAMES[3:]


def summing_matrix() -> np.ndarray:
    """S maps the 4 bottom series to all 7 series:  y_all = S @ y_bottom."""
    return np.array(
        [
            [1, 1, 1, 1],  # Total
            [1, 1, 0, 0],  # North
            [0, 0, 1, 1],  # South
            [1, 0, 0, 0],  # North_A
            [0, 1, 0, 0],  # North_B
            [0, 0, 1, 0],  # South_A
            [0, 0, 0, 1],  # South_B
        ],
        dtype=float,
    )


def _ar1(n, phi, sigma, rng):
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for t in range(n):
        x[t] = (phi * x[t - 1] if t else 0.0) + e[t]
    return x


def make_retail_hierarchy(n_months: int = 120, start: str = "2015-01-01", seed: int = 42) -> pd.DataFrame:
    """Synthetic monthly retail sales for a Total -> Region -> Store hierarchy.

    Each store = (level + trend) x seasonality x exp(shock). Shocks are the sum of a
    COMMON factor, a REGION factor and an idiosyncratic part, all AR(1). So the
    stores are positively correlated, more strongly inside a region, which is
    exactly what the multivariate / hierarchical notebooks need.

    Returns a DataFrame with all 7 columns (HIER_NAMES), coherent by construction.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_months, freq="MS")
    t = np.arange(n_months)
    m = idx.month.to_numpy()

    common = _ar1(n_months, 0.5, 0.060, rng)
    region = {"North": _ar1(n_months, 0.3, 0.040, rng), "South": _ar1(n_months, 0.3, 0.040, rng)}
    # level, trend/month, seasonal amplitude, phase (months), idiosyncratic sd
    spec = {
        "North_A": (500, 2.0, 0.20, 0, 0.035),
        "North_B": (300, 1.2, 0.15, 1, 0.040),
        "South_A": (400, 1.5, 0.25, 2, 0.035),
        "South_B": (250, 1.0, 0.30, 3, 0.045),
    }
    out = {}
    for name, (lvl, tr, amp, ph, sd) in spec.items():
        seas = 1 + amp * np.sin(2 * np.pi * (m - 1 - ph) / 12) + 0.15 * (m == 12)
        shock = common + region[name.split("_")[0]] + _ar1(n_months, 0.2, sd, rng)
        out[name] = (lvl + tr * t) * seas * np.exp(shock)
    bottom = pd.DataFrame(out, index=idx).round(1)
    all_series = pd.DataFrame(bottom.to_numpy() @ summing_matrix().T, index=idx, columns=HIER_NAMES)
    return all_series.round(1)


def load_retail_total() -> pd.Series:
    """The top level of the retail hierarchy as a single monthly series."""
    return make_retail_hierarchy()["Total"].rename("sales")


def make_sku_demand(n_months: int = 156, start: str = "2012-01-01", seed: int = 7) -> pd.Series:
    """Intermittent, over-dispersed monthly COUNT demand (units) for one slow-moving SKU.

    y ~ NegBin(mean = mu_t, alpha = 1.0) generated as a Gamma-Poisson mixture.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_months, freq="MS")
    t = np.arange(n_months)
    m = idx.month.to_numpy()
    mu = np.exp(0.3 + 0.004 * t + 0.6 * np.sin(2 * np.pi * (m - 1) / 12))
    alpha = 1.0
    lam = rng.gamma(shape=1 / alpha, scale=alpha * mu)
    return pd.Series(rng.poisson(lam).astype(float), index=idx, name="units")


PANEL_PROFILES = {   # log-scale seasonal shapes of the four product families (Jan..Dec), demeaned below
    "holiday": [-0.3, -0.4, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1, 0.0, 0.2, 0.6, 1.0],
    "summer": [-0.5, -0.4, -0.2, 0.1, 0.4, 0.7, 0.8, 0.6, 0.1, -0.3, -0.5, -0.6],
    "school": [-0.2, -0.2, -0.2, -0.2, -0.2, -0.1, 0.1, 0.9, 0.7, -0.1, -0.2, -0.3],
    "spring": [-0.3, 0.2, 0.7, 0.6, 0.2, -0.1, -0.2, -0.3, -0.2, -0.2, -0.2, -0.3],
}


def make_sku_panel(n_series: int = 300, n_months: int = 120, start: str = "2015-01-01", seed: int = 2024):
    """A PANEL of related monthly count series: many SKUs that share seasonal shapes but differ in scale and age.

    For each SKU i (family f_i, launch month L_i):
        log mu_it = log(level_i) + trend_i * t + amp_i * profile_f(month) + x_it + log ramp_it
        x_it      = 0.7 x_i,t-1 + N(0, 0.15^2)            (persistent latent demand shocks)
        ramp_it   = 1 - exp(-(t - L_i + 1) / 4) for t >= L_i, demand is 0 before launch
        y_it      ~ NegBin(mean = mu_it, alpha_i)          (Gamma-Poisson)
    * level_i is log-normal (from ~0.5 to a few hundred units/month): intermittent slow movers + smooth fast movers
    * 4 seasonal families with NON-sinusoidal shapes (holiday spike, summer, back-to-school, spring)
    * ~40% of SKUs launch during 2016-2022, so they have short histories at the test origins

    Returns (y, mu, info): y and mu are (n_months x n_series) DataFrames, info holds per-SKU parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_months, freq="MS")
    t = np.arange(n_months)[:, None]
    month = idx.month.to_numpy() - 1
    fams = list(PANEL_PROFILES)
    prof = np.array([np.subtract(v, np.mean(v)) for v in PANEL_PROFILES.values()])      # (4, 12)
    family = rng.integers(0, len(fams), n_series)
    level = np.exp(rng.normal(np.log(8.0), 1.3, n_series))
    trend = rng.normal(0.0, 0.004, n_series)
    amp = rng.uniform(0.4, 1.1, n_series)
    shape = amp[:, None] * prof[family] + rng.normal(0, 0.05, (n_series, 12))          # (N, 12)
    alpha = rng.uniform(0.05, 0.6, n_series)
    launch = np.where(rng.uniform(size=n_series) < 0.4, rng.integers(12, 85, n_series), 0)
    x = np.zeros((n_months, n_series))
    eps = rng.normal(0, 0.15, (n_months, n_series))
    x[0] = eps[0] / np.sqrt(1 - 0.7**2)
    for k in range(1, n_months):
        x[k] = 0.7 * x[k - 1] + eps[k]
    ramp = np.where(t >= launch, 1 - np.exp(-(t - launch + 1) / 4.0), 0.0)
    mu = level * np.exp(trend * (t - n_months / 2) + shape[:, month].T + x) * ramp
    lam = rng.gamma(shape=1 / alpha, scale=alpha * np.maximum(mu, 1e-12))
    cols = [f"SKU_{i:03d}" for i in range(n_series)]
    y = pd.DataFrame(rng.poisson(lam).astype(float), index=idx, columns=cols)
    info = pd.DataFrame({"family": np.array(fams)[family], "level": level, "trend": trend, "amp": amp, "alpha": alpha,
                         "launch": idx[launch]}, index=cols)
    return y, pd.DataFrame(mu, index=idx, columns=cols), info


def seasonal_naive_quantiles(y: np.ndarray, origin: int, H: int, taus) -> np.ndarray:
    """Seasonal-naive point forecast y_{t-12} + empirical quantiles of past seasonal-naive errors, clipped at 0.

    y: (T, N) panel; uses rows < origin only.  Returns (N, H, len(taus)).
    """
    hist = y[:origin]
    err = hist[12:] - hist[:-12]                                  # (origin-12, N)
    eq = np.quantile(err, taus, axis=0).T                         # (N, K)
    base = np.stack([hist[origin - 12 + (h % 12)] for h in range(H)], axis=1)   # (N, H)
    return np.maximum(base[:, :, None] + eq[:, None, :], 0.0)


def local_negbin_quantiles(y: np.ndarray, origin: int, H: int, taus) -> np.ndarray:
    """A LOCAL model per series: Poisson GLM (log link, trend + annual sine/cosine) for the mean, NegBin
    dispersion alpha by moments.  Fitted from each series' first sale up to `origin` (rows < origin only).  Returns (N, H, len(taus)) NegBin quantiles."""
    T, N = y.shape
    t = np.arange(origin + H)
    X = np.column_stack([np.ones_like(t), (t - origin) / 12, np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12)]).astype(float)
    out = np.empty((N, H, len(taus)))
    for i in range(N):
        s0 = int(np.argmax(y[:origin, i] > 0))                    # fit from the first sale (launch) onwards
        yi, Xi = y[s0:origin, i], X[s0:origin]
        beta = np.zeros(4); beta[0] = np.log(yi.mean() + 0.1)
        for _ in range(50):                                       # IRLS for the Poisson GLM
            mu = np.exp(Xi @ beta)
            z = Xi @ beta + (yi - mu) / mu
            new = np.linalg.solve(Xi.T @ (mu[:, None] * Xi) + 1e-3 * np.eye(4), Xi.T @ (mu * z))
            if np.max(np.abs(new - beta)) < 1e-8:
                beta = new
                break
            beta = new
        mu = np.exp(Xi @ beta)
        a = max(np.mean(((yi - mu) ** 2 - mu) / mu**2), 1e-4)
        mf = np.exp(X[origin:origin + H] @ beta)
        n = 1 / a
        out[i] = stats.nbinom.ppf(np.asarray(taus)[None, :], n, (n / (n + mf))[:, None])
    return out


def panel_quantile_scores(y: np.ndarray, Q: np.ndarray, taus, scale: np.ndarray) -> dict:
    """Scores for a panel of multi-horizon quantile forecasts.

    y: (N, H) actuals, Q: (N, H, K) quantiles at levels `taus`, scale: (N,) per-series scale (e.g. mean of history).
    * wQL  : volume-weighted  2*sum(pinball)/sum|y|, averaged over tau (GluonTS mean_wQuantileLoss)
    * sQL  : series-weighted  mean over series of 2*mean(pinball)/scale_i, averaged over tau (every SKU counts equally)
    * cov80 / cov90 : empirical coverage of the central 80% / 90% intervals (needs those levels in `taus`)
    """
    taus = np.round(np.asarray(taus, float), 6)
    pl = np.stack([pinball_loss(y, Q[..., k], tau) for k, tau in enumerate(taus)], axis=-1)   # (N, H, K)
    out = {"wQL": float(np.mean(2 * pl.sum(axis=(0, 1)) / np.abs(y).sum())),
           "sQL": float(np.mean(2 * pl.mean(axis=1) / scale[:, None]))}
    for lo, hi, name in [(0.1, 0.9, "cov80"), (0.05, 0.95, "cov90")]:
        if lo in taus and hi in taus:
            a, b = Q[..., list(taus).index(lo)], Q[..., list(taus).index(hi)]
            out[name] = float(np.mean((y >= a) & (y <= b)))
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def seasonal_design(index: pd.DatetimeIndex, origin: str | pd.Timestamp = "1949-01-01", const: bool = True) -> pd.DataFrame:
    """Linear trend (months since `origin`) + 11 month dummies (January = baseline)."""
    origin = pd.Timestamp(origin)
    t = (index.year - origin.year) * 12 + (index.month - origin.month)
    X = pd.DataFrame({"trend": np.asarray(t, dtype=float)}, index=index)
    for k in range(2, 13):
        X[f"m{k:02d}"] = (index.month == k).astype(float)
    if const:
        X.insert(0, "const", 1.0)
    return X


# ---------------------------------------------------------------------------
# Scoring rules (lower is better)
# ---------------------------------------------------------------------------


def pinball_loss(y, q, tau):
    """Quantile (pinball) loss, elementwise.  rho_tau(u) = max(tau*u, (tau-1)*u), u = y - q."""
    u = np.asarray(y, float) - np.asarray(q, float)
    return np.maximum(tau * u, (tau - 1) * u)


def weighted_quantile_loss(y, qdict: dict) -> float:
    """Scaled pinball loss averaged over quantile levels (the GluonTS / M5-style 'mean wQL').

    wQL_tau = 2 * sum_t rho_tau(y_t - q_t) / sum_t |y_t|
    """
    y = np.asarray(y, float)
    vals = [2 * pinball_loss(y, q, tau).sum() / np.abs(y).sum() for tau, q in qdict.items()]
    return float(np.mean(vals))


def crps_gaussian(y, mu, sigma):
    """Closed-form CRPS of N(mu, sigma^2) at observation y."""
    y, mu, sigma = (np.asarray(a, float) for a in (y, mu, sigma))
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_sample(y, samples):
    """CRPS estimated from samples:  E|X - y| - 0.5 E|X - X'|.

    samples: shape (n_samples, ...) ; y: shape (...)   ->   returns shape (...)
    Uses the O(n log n) sorted-sample identity for E|X - X'|.
    """
    x = np.sort(np.asarray(samples, float), axis=0)
    y = np.asarray(y, float)
    n = x.shape[0]
    term1 = np.mean(np.abs(x - y), axis=0)
    i = np.arange(1, n + 1).reshape((-1,) + (1,) * (x.ndim - 1))
    term2 = np.sum((2 * i - n - 1) * x, axis=0) / n**2
    return term1 - term2


def crps_quantiles(y, qdict: dict):
    """Approximate CRPS from a set of quantiles: CRPS ~= 2 * mean_tau pinball_tau.

    Exact when the quantile grid becomes dense and uniform on (0, 1).
    """
    return 2 * np.mean([pinball_loss(y, q, tau) for tau, q in qdict.items()], axis=0)


def log_score_gaussian(y, mu, sigma):
    """Negative log predictive density (a.k.a. NLL / ignorance, in nats)."""
    return -stats.norm.logpdf(y, mu, sigma)


def interval_score(y, lo, hi, alpha):
    """Gneiting-Raftery interval score (Winkler score) for a central (1-alpha) interval."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def brier_score(outcome, p):
    return np.mean((np.asarray(p, float) - np.asarray(outcome, float)) ** 2)


def log_loss(outcome, p, eps=1e-12):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    o = np.asarray(outcome, float)
    return np.mean(-(o * np.log(p) + (1 - o) * np.log(1 - p)))


def brier_decomposition(outcome, p, n_bins=10) -> dict:
    """Murphy (1973) decomposition: BS ~= reliability - resolution + uncertainty."""
    o = np.asarray(outcome, float)
    p = np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    obar = o.mean()
    rel = res = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            nk = mask.sum()
            rel += nk * (p[mask].mean() - o[mask].mean()) ** 2
            res += nk * (o[mask].mean() - obar) ** 2
    n = len(o)
    return {"reliability": rel / n, "resolution": res / n, "uncertainty": obar * (1 - obar), "brier": brier_score(o, p)}


def rps(y_cat, probs):
    """Ranked Probability Score for ordinal categories.

    y_cat: int array (N,) with values 0..K-1 ; probs: (N, K).  Returns per-forecast RPS
    (un-normalised: sum over the K-1 cumulative thresholds).
    """
    probs = np.asarray(probs, float)
    K = probs.shape[1]
    obs = np.eye(K)[np.asarray(y_cat, int)]
    cf, co = np.cumsum(probs, axis=1), np.cumsum(obs, axis=1)
    return np.sum((cf - co)[:, :-1] ** 2, axis=1)


def energy_score(y, samples):
    """Multivariate generalisation of CRPS:  E||X - y|| - 0.5 E||X - X'||.

    y: (d,) ; samples: (n, d)
    """
    X = np.asarray(samples, float)
    y = np.asarray(y, float)
    n = X.shape[0]
    t1 = np.mean(np.linalg.norm(X - y, axis=1))
    t2 = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2).sum() / (n * n)
    return t1 - 0.5 * t2


def variogram_score(y, samples, p: float = 0.5):
    """Scheuerer & Hamill (2015) variogram score of order p (sensitive to dependence)."""
    X = np.asarray(samples, float)
    y = np.asarray(y, float)
    obs = np.abs(y[:, None] - y[None, :]) ** p
    pred = np.mean(np.abs(X[:, :, None] - X[:, None, :]) ** p, axis=0)
    iu = np.triu_indices(len(y), k=1)
    return float(np.sum((obs - pred)[iu] ** 2))


def pit_from_samples(y, samples):
    """Probability Integral Transform u = F(y), estimated from samples (n, ...)."""
    return np.mean(np.asarray(samples) <= np.asarray(y), axis=0)


def randomized_pit_discrete(y, cdf, rng=None):
    """Randomised PIT for discrete forecasts: u = F(y-1) + V * (F(y) - F(y-1)), V ~ U(0,1)."""
    rng = np.random.default_rng(0) if rng is None else rng
    y = np.asarray(y)
    lo, hi = cdf(y - 1), cdf(y)
    return lo + rng.uniform(size=y.shape) * (hi - lo)


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def subplots(rows=1, cols=1, titles=None, height=None, width=None, shared_xaxes=False, shared_yaxes=False,
             title=None, **kw) -> go.Figure:
    """make_subplots with the house style: subplot titles in primary ink, sensible default height."""
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, shared_xaxes=shared_xaxes,
                        shared_yaxes=shared_yaxes, horizontal_spacing=kw.pop("horizontal_spacing", 0.06),
                        vertical_spacing=kw.pop("vertical_spacing", 0.14), **kw)
    fig.update_annotations(font=dict(size=13, color=INK["primary"]))
    fig.update_layout(height=height or (360 * rows + 40), width=width, title=title)
    return fig


def _rc(row, col):
    return dict(row=row, col=col) if row is not None else {}


def add_fan(fig, index, qdict: dict, history: pd.Series | None = None, actual=None, color=SERIES[0],
            label="forecast", n_history=48, row=None, col=None, showlegend=True, fmt=",.4~g"):
    """Fan chart from a {tau: values} dict: shaded central intervals + median, with a tooltip that
    lists every interval for the hovered month."""
    q = {round(float(k), 6): np.asarray(v, float) for k, v in qdict.items()}
    x = list(index)
    rc = _rc(row, col)
    if history is not None:
        h = history.iloc[-n_history:]
        fig.add_trace(go.Scatter(x=h.index, y=h.values, mode="lines", name="history", legendgroup="history",
                                 line=dict(color=HISTORY, width=1.5), showlegend=showlegend,
                                 hovertemplate=f"history: %{{y:{fmt}}}<extra></extra>"), **rc)
    pairs = sorted((lo, round(1 - lo, 6)) for lo in q if lo < 0.5 and round(1 - lo, 6) in q)
    for k, (lo, hi) in enumerate(pairs):                       # outer (lightest) first
        pct = int(round((hi - lo) * 100))
        fig.add_trace(go.Scatter(x=x + x[::-1], y=np.r_[q[hi], q[lo][::-1]], fill="toself", mode="lines",
                                 line=dict(width=0), fillcolor=rgba(color, 0.16 + 0.16 * k), hoverinfo="skip",
                                 name=f"{pct}% interval", legendgroup=f"{label}{pct}", showlegend=showlegend), **rc)
    if 0.5 in q:
        cd = np.column_stack([q[t] for p in pairs[::-1] for t in p]) if pairs else None
        tip = f"{label} median: %{{y:{fmt}}}" + "".join(
            f"<br>{int(round((hi - lo) * 100))}%: %{{customdata[{2*i}]:{fmt}}} – %{{customdata[{2*i+1}]:{fmt}}}"
            for i, (lo, hi) in enumerate(pairs[::-1]))
        fig.add_trace(go.Scatter(x=x, y=q[0.5], mode="lines", name=f"{label} median", legendgroup=f"{label}med",
                                 line=dict(color=color, width=2), customdata=cd, showlegend=showlegend,
                                 hovertemplate=tip + "<extra></extra>"), **rc)
    if actual is not None:
        fig.add_trace(go.Scatter(x=x, y=np.asarray(actual, float), mode="markers+lines", name="actual",
                                 legendgroup="actual", showlegend=showlegend,
                                 line=dict(color=ACTUAL, width=1), marker=dict(color=ACTUAL, size=8,
                                 line=dict(color=INK["surface"], width=2)),
                                 hovertemplate=f"actual: %{{y:{fmt}}}<extra></extra>"), **rc)
    fig.update_layout(hovermode="x unified")
    return fig


def add_pit(fig, pit, bins=10, row=None, col=None, color=SERIES[0], name="PIT", showlegend=False):
    """PIT histogram (density scale) with the uniform reference line at 1."""
    pit = np.asarray(pit).ravel()
    dens, edges = np.histogram(pit, bins=bins, range=(0, 1), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(x=centers, y=dens, width=np.diff(edges) * 0.92, marker_color=color, name=name,
                         showlegend=showlegend, customdata=np.c_[edges[:-1], edges[1:]],
                         hovertemplate="u in [%{customdata[0]:.2f}, %{customdata[1]:.2f})<br>density %{y:.2f}<extra></extra>"),
                  **_rc(row, col))
    fig.add_hline(y=1, line=dict(color=INK["primary"], dash="dash", width=1), **_rc(row, col))
    fig.update_xaxes(range=[0, 1], **_rc(row, col))
    return fig


def reliability_table(outcome, p, n_bins=10) -> pd.DataFrame:
    o = np.asarray(outcome, float)
    p = np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    df = pd.DataFrame({"bin": bins, "p": p, "o": o})
    return df.groupby("bin").agg(mean_forecast=("p", "mean"), observed_freq=("o", "mean"), n=("o", "size"))


def add_reliability(fig, outcome, p, n_bins=10, name="forecast", color=SERIES[0], row=None, col=None,
                    diagonal=True, showlegend=True):
    """Reliability (calibration) curve; hover shows how many forecasts fell in each bin."""
    tab = reliability_table(outcome, p, n_bins)
    rc = _rc(row, col)
    if diagonal:
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                                 legendgroup="diag", showlegend=showlegend, hoverinfo="skip",
                                 line=dict(color=INK["muted"], dash="dash", width=1)), **rc)
    fig.add_trace(go.Scatter(x=tab.mean_forecast, y=tab.observed_freq, mode="lines+markers", name=name,
                             legendgroup=name, showlegend=showlegend, customdata=tab.n,
                             line=dict(color=color, width=2), marker=dict(size=8, color=color),
                             hovertemplate=f"{name}<br>forecast %{{x:.2f}} → observed %{{y:.2f}}"
                                           "<br>n = %{customdata}<extra></extra>"), **rc)
    fig.update_xaxes(range=[0, 1], title_text="forecast probability", **rc)
    fig.update_yaxes(range=[0, 1], title_text="observed frequency", **rc)
    return fig


def add_step_hist(fig, data, bins, name, color, row=None, col=None, density=True, range_=None, showlegend=True,
                  dash=None):
    """Outline ('step') histogram, so several distributions can be overlaid without hiding each other."""
    dens, edges = np.histogram(np.asarray(data).ravel(), bins=bins, range=range_, density=density)
    xs = np.r_[edges[0], np.repeat(edges[1:-1], 2), edges[-1]]
    ys = np.repeat(dens, 2)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name, legendgroup=name, showlegend=showlegend,
                             line=dict(color=color, width=2, dash=dash),
                             hovertemplate=f"{name}<br>x %{{x:.4~g}}<br>{'density' if density else 'count'} %{{y:.3~g}}<extra></extra>"),
                  **_rc(row, col))
    return fig


def add_paths(fig, index, paths, color=SERIES[0], name="sample paths", n=40, row=None, col=None,
              showlegend=True, opacity=0.35):
    """Spaghetti plot: the first n sample paths drawn as ONE trace (None-separated) to keep figures light."""
    x, y = [], []
    for p in np.asarray(paths)[:n]:
        x += list(index) + [None]
        y += list(p) + [None]
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, legendgroup=name, showlegend=showlegend,
                             line=dict(color=color, width=1), opacity=opacity, hoverinfo="skip"),
                  **_rc(row, col))
    return fig


def add_vline(fig, x, y0, y1, name, color=INK["primary"], dash="dash", row=None, col=None, showlegend=True, width=1.5):
    """A labelled vertical reference line drawn as a trace, so it appears in the legend and the tooltip."""
    fig.add_trace(go.Scatter(x=[x, x], y=[y0, y1], mode="lines", name=name, legendgroup=name,
                             showlegend=showlegend, line=dict(color=color, dash=dash, width=width),
                             hovertemplate=f"{name}<extra></extra>"), **_rc(row, col))
    return fig


def add_band(fig, index, lo, hi, name, color=SERIES[0], alpha=0.25, row=None, col=None, showlegend=True):
    """A shaded [lo, hi] band (no hover of its own: put the numbers in a companion line's tooltip)."""
    x = list(index)
    fig.add_trace(go.Scatter(x=x + x[::-1], y=np.r_[np.asarray(hi, float), np.asarray(lo, float)[::-1]], fill="toself",
                             fillcolor=rgba(color, alpha), line=dict(width=0), hoverinfo="skip", name=name,
                             legendgroup=name, showlegend=showlegend), **_rc(row, col))
    return fig


def add_lines_by(fig, frame: pd.DataFrame, colors: dict, row=None, col=None, showlegend=True, fmt=".3f", markers=True):
    """One line per column of `frame` (index = x), coloured by a fixed {column: colour} map."""
    for name in frame.columns:
        fig.add_trace(go.Scatter(x=frame.index, y=frame[name], mode="lines+markers" if markers else "lines", name=str(name),
                                 legendgroup=str(name), showlegend=showlegend,
                                 line=dict(color=colors[name], width=2), marker=dict(size=8),
                                 hovertemplate=f"{name}: %{{y:{fmt}}}<extra></extra>"), **_rc(row, col))
    return fig
