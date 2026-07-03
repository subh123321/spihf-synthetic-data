"""
visualization_module.py
=======================
Comparative visualisation of Original vs Synthetic SPIHF (Single Point
Incremental Hole Flanging) datasets.

This module produces five publication-quality figure files using
**matplotlib only** (no seaborn styles):

  1. distribution_comparison.png   – Histograms, KDE, boxplots, violins
  2. correlation_heatmap.png       – Side-by-side Pearson heatmaps
  3. pca_comparison.png            – 2-D PCA projections
  4. tsne_comparison.png           – 2-D t-SNE projections
  5. engineering_relationships.png – Six domain-specific scatter plots

Optionally generates:
  6. umap_comparison.png           – 2-D UMAP projections (if umap-learn
                                     is installed)

All colour palettes are derived programmatically — no hardcoded hex
values — ensuring consistency and accessibility.

Author : Visualisation Module (auto-generated)
Seed   : np.random.seed(42)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ──────────────────────────── Global seed ────────────────────────────
np.random.seed(42)

# ──────────────────────────── Optional UMAP ──────────────────────────
try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ──────────────────────────── Colour Palette ─────────────────────────
# Derived from matplotlib's default colour cycle — no hardcoding.
_PROP_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def _real_color(alpha: float = 1.0) -> Tuple[float, ...]:
    """Return RGBA for the 'Real' dataset from the active prop cycle."""
    rgba = matplotlib.colors.to_rgba(_PROP_CYCLE[0], alpha)
    return rgba

def _synth_color(alpha: float = 1.0) -> Tuple[float, ...]:
    """Return RGBA for the 'Synthetic' dataset from the active prop cycle."""
    rgba = matplotlib.colors.to_rgba(_PROP_CYCLE[1], alpha)
    return rgba

def _accent_color(idx: int = 2, alpha: float = 1.0) -> Tuple[float, ...]:
    """Return an accent colour from the prop cycle."""
    rgba = matplotlib.colors.to_rgba(_PROP_CYCLE[idx % len(_PROP_CYCLE)], alpha)
    return rgba

# ──────────────────────────── Constants ──────────────────────────────
NUMERIC_FEATURES: List[str] = [
    "Thickness (mm)",
    "Precut dimensions (diameter/side length) mm",
    "Total Strain/Elongation (%)",
    "UTS (MPa)",
    "YS (MPa)",
    "Strength Coefficient (k in MPa)",
    "Strain hardening coefficient (n)",
    "Anisotropic (R Value)",
    "Feed rate (mm/min)",
    "Tool speed (rpm)",
    "Step depth (mm)",
    "No of stages",
    "HER",
    "Flange Height (mm)",
    "Roughness (um)",
    "Minimum thickness (after final stage, mm)",
    "Final angle after the final stage (degrees)",
]

# Short labels for axis readability
_SHORT_LABELS: Dict[str, str] = {
    "Thickness (mm)": "Thickness",
    "Precut dimensions (diameter/side length) mm": "Precut dim.",
    "Total Strain/Elongation (%)": "Elongation %",
    "UTS (MPa)": "UTS",
    "YS (MPa)": "YS",
    "Strength Coefficient (k in MPa)": "Strength k",
    "Strain hardening coefficient (n)": "n (hardening)",
    "Anisotropic (R Value)": "R-value",
    "Feed rate (mm/min)": "Feed rate",
    "Tool speed (rpm)": "Tool speed",
    "Step depth (mm)": "Step depth",
    "No of stages": "Stages",
    "HER": "HER",
    "Flange Height (mm)": "Flange Height",
    "Roughness (um)": "Roughness",
    "Minimum thickness (after final stage, mm)": "Min thickness",
    "Final angle after the final stage (degrees)": "Final angle",
}

# Engineering relationship pairs:  (x_col, y_col, x_label, y_label)
ENGINEERING_PAIRS: List[Tuple[str, str, str, str]] = [
    ("HER", "Flange Height (mm)",
     "Hole Expansion Ratio (HER)", "Flange Height (mm)"),
    ("Step depth (mm)", "Roughness (um)",
     "Step Depth (mm)", "Surface Roughness (um)"),
    ("No of stages", "HER",
     "Number of Stages", "HER"),
    ("Thickness (mm)", "Minimum thickness (after final stage, mm)",
     "Initial Thickness (mm)", "Minimum Thickness (mm)"),
    ("Total Strain/Elongation (%)", "HER",
     "Total Elongation (%)", "HER"),
    ("Anisotropic (R Value)", "HER",
     "Lankford R-value", "HER"),
]

# Column-name mapping (raw CSV -> canonical)
_RAW_TO_CANONICAL: Dict[str, str] = {
    "Precut dimensions (diameter/ side length) mm":
        "Precut dimensions (diameter/side length) mm",
    "Precut Shape (circle/ square/etc)":
        "Precut Shape (circle/square/etc)",
    "Total Strain/ Elongation (%)":
        "Total Strain/Elongation (%)",
    "Step depth(mm)": "Step depth (mm)",
    "Final angle after the final stage, degrees":
        "Final angle after the final stage (degrees)",
}

# ──────────────────────────── Global rcParams ────────────────────────
# A clean, modern look without seaborn — purely matplotlib rcParams.
_RC_OVERRIDES: Dict[str, Any] = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "legend.framealpha": 0.85,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}


def _apply_rc() -> None:
    """Apply custom rcParams for consistent look across all figures."""
    for k, v in _RC_OVERRIDES.items():
        matplotlib.rcParams[k] = v


# ════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════
def _harmonise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw CSV columns to canonical names and coerce numerics.

    Handles the inconsistent column naming in the raw SPIHF_Data.csv
    (spaces, slashes, embedded units) so both datasets share identical
    column names before plotting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (raw or synthetic).

    Returns
    -------
    pd.DataFrame
        DataFrame with canonical column names.
    """
    df = df.copy()
    df.rename(columns=_RAW_TO_CANONICAL, inplace=True)

    # Handle roughness column name (mu symbol encoding)
    for col in list(df.columns):
        if "Roughness" in col and col not in NUMERIC_FEATURES:
            df.rename(columns={col: "Roughness (um)"}, inplace=True)
            break

    # Strip embedded unit strings from numeric cells
    unit_suffixes = [
        " mm/min", " rpm clockwise", " rpm", " mm/cycle",
        " mm", " um", "\u00b0", "\u00b0 ",
    ]
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip()
        for suffix in unit_suffixes:
            s = s.str.replace(suffix, "", regex=False)
        s = s.str.replace("\u00b0", "", regex=False)
        s = s.str.replace("<", "", regex=False)
        s = s.str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(s, errors="coerce")

    # Forward-fill Material
    if "Material" in df.columns:
        df["Material"] = df["Material"].replace(r"^\s*$", np.nan, regex=True)
        df["Material"] = df["Material"].ffill()

    return df


def _short(col: str) -> str:
    """Return a short, axis-friendly label for a column name."""
    return _SHORT_LABELS.get(col, col[:20])


def _safe_kde(data: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Compute a KDE, returning zeros if the data is degenerate."""
    clean = data[np.isfinite(data)]
    if len(clean) < 3 or np.std(clean) < 1e-12:
        return np.zeros_like(x_grid)
    try:
        kde = gaussian_kde(clean, bw_method="scott")
        return kde(x_grid)
    except Exception:
        return np.zeros_like(x_grid)


def _prepare_numeric(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Return the list of plottable numeric columns and clean subsets."""
    avail = [c for c in NUMERIC_FEATURES
             if c in real.columns and c in synth.columns]
    # Keep only columns with >= 5 non-NaN values in both datasets
    keep = []
    for c in avail:
        if real[c].dropna().shape[0] >= 5 and synth[c].dropna().shape[0] >= 5:
            keep.append(c)
    return keep, real, synth


# ════════════════════════════════════════════════════════════════════
#  1.  DISTRIBUTION PLOTS (histograms + KDE)
# ════════════════════════════════════════════════════════════════════
def plot_distributions(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    n_bins: int = 35,
) -> matplotlib.figure.Figure:
    """Plot overlaid histograms and KDE curves for every numeric feature.

    For each feature, a single axes shows:
      - A semi-transparent histogram for the real data.
      - A semi-transparent histogram for the synthetic data.
      - KDE smoothed density curves for both.

    This allows immediate visual assessment of whether the synthetic
    generator has preserved unimodal/multimodal shapes, support
    boundaries, and peak locations.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.

    Engineering note
    ----------------
    Features like *Final angle* and *Is lubricant used?* are highly
    peaked (most values at 90 deg or 0/1).  The KDE may appear over-
    smoothed for these — the histogram gives the true picture.
    """
    cols, real, synth = _prepare_numeric(real, synth)
    n = len(cols)
    ncols_grid = 3
    nrows_grid = int(np.ceil(n / ncols_grid))

    fig, axes = plt.subplots(nrows_grid, ncols_grid,
                             figsize=(5.5 * ncols_grid, 4.0 * nrows_grid))
    axes = np.atleast_2d(axes)

    rc = _real_color(0.45)
    sc = _synth_color(0.45)
    rc_line = _real_color(1.0)
    sc_line = _synth_color(1.0)

    for i, col in enumerate(cols):
        row_idx, col_idx = divmod(i, ncols_grid)
        ax = axes[row_idx, col_idx]

        rv = real[col].dropna().values.astype(float)
        sv = synth[col].dropna().values.astype(float)

        lo = min(rv.min(), sv.min())
        hi = max(rv.max(), sv.max())
        margin = 0.05 * (hi - lo + 1e-9)
        bins = np.linspace(lo - margin, hi + margin, n_bins + 1)
        x_grid = np.linspace(lo - margin, hi + margin, 300)

        ax.hist(rv, bins=bins, density=True, color=rc,
                edgecolor=rc_line, linewidth=0.4, label="Real", zorder=2)
        ax.hist(sv, bins=bins, density=True, color=sc,
                edgecolor=sc_line, linewidth=0.4, label="Synthetic", zorder=2)

        kde_r = _safe_kde(rv, x_grid)
        kde_s = _safe_kde(sv, x_grid)
        ax.plot(x_grid, kde_r, color=rc_line, linewidth=1.6, zorder=3)
        ax.plot(x_grid, kde_s, color=sc_line, linewidth=1.6,
                linestyle="--", zorder=3)

        ax.set_title(_short(col))
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", frameon=True)

    # Turn off unused axes
    for j in range(n, nrows_grid * ncols_grid):
        row_idx, col_idx = divmod(j, ncols_grid)
        axes[row_idx, col_idx].set_visible(False)

    fig.suptitle("Distribution Comparison: Real vs Synthetic SPIHF Data",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════
#  2.  BOXPLOTS & VIOLIN PLOTS
# ════════════════════════════════════════════════════════════════════
def plot_boxplots(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Side-by-side boxplots and violin plots for each numeric feature.

    Each feature gets two rows: the top row shows paired box plots, the
    bottom row shows paired violin plots.  This simultaneously reveals
    the median, IQR, outlier envelope (box) and the full density shape
    (violin).

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.

    Engineering note
    ----------------
    Violin width directly encodes local density, making it easier to
    spot if the synthetic data has collapsed a bimodal distribution
    into a unimodal one (a common failure mode of Gaussian noise
    augmentation on skewed manufacturing data).
    """
    cols, real, synth = _prepare_numeric(real, synth)
    n = len(cols)
    ncols_grid = 3
    nrows_grid = int(np.ceil(n / ncols_grid))

    fig, axes = plt.subplots(nrows_grid, ncols_grid,
                             figsize=(5.5 * ncols_grid, 4.5 * nrows_grid))
    axes = np.atleast_2d(axes)

    rc_face = _real_color(0.6)
    sc_face = _synth_color(0.6)
    rc_edge = _real_color(1.0)
    sc_edge = _synth_color(1.0)

    for i, col in enumerate(cols):
        row_idx, col_idx = divmod(i, ncols_grid)
        ax = axes[row_idx, col_idx]

        rv = real[col].dropna().values.astype(float)
        sv = synth[col].dropna().values.astype(float)

        # Boxplots
        bp = ax.boxplot(
            [rv, sv],
            positions=[1, 2],
            widths=0.35,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
            medianprops=dict(color="black", linewidth=1.5),
        )
        bp["boxes"][0].set_facecolor(rc_face)
        bp["boxes"][0].set_edgecolor(rc_edge)
        bp["boxes"][1].set_facecolor(sc_face)
        bp["boxes"][1].set_edgecolor(sc_edge)

        # Violin overlays
        if len(rv) >= 5 and np.std(rv) > 1e-12:
            vp_r = ax.violinplot([rv], positions=[1], widths=0.55,
                                 showmeans=False, showmedians=False,
                                 showextrema=False)
            for body in vp_r["bodies"]:
                body.set_facecolor(rc_face)
                body.set_edgecolor(rc_edge)
                body.set_alpha(0.25)

        if len(sv) >= 5 and np.std(sv) > 1e-12:
            vp_s = ax.violinplot([sv], positions=[2], widths=0.55,
                                 showmeans=False, showmedians=False,
                                 showextrema=False)
            for body in vp_s["bodies"]:
                body.set_facecolor(sc_face)
                body.set_edgecolor(sc_edge)
                body.set_alpha(0.25)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Real", "Synthetic"])
        ax.set_title(_short(col))

    # Turn off unused axes
    for j in range(n, nrows_grid * ncols_grid):
        row_idx, col_idx = divmod(j, ncols_grid)
        axes[row_idx, col_idx].set_visible(False)

    fig.suptitle("Box + Violin Comparison: Real vs Synthetic SPIHF Data",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════
#  3.  CORRELATION HEATMAPS
# ════════════════════════════════════════════════════════════════════
def plot_correlation_heatmaps(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Side-by-side Pearson correlation heatmaps plus a difference map.

    Three panels are shown:
      - Left:   Real data correlation matrix.
      - Centre: Synthetic data correlation matrix.
      - Right:  Element-wise difference (Real - Synthetic).

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.

    Engineering note
    ----------------
    The difference panel highlights where the synthetic data has gained
    or lost correlations.  Blue = synthetic overestimates (more positive
    than real), red = synthetic underestimates.  Ideal = all white.
    """
    cols, _, _ = _prepare_numeric(real, synth)
    short_labels = [_short(c) for c in cols]

    r_corr = real[cols].corr().values
    s_corr = synth[cols].corr().values
    diff = r_corr - s_corr

    # Build diverging colormaps from the prop cycle
    cmap_main = plt.cm.RdBu_r
    cmap_diff = plt.cm.PiYG

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))

    def _draw_heatmap(ax: plt.Axes, data: np.ndarray, title: str,
                      cmap: Any, vmin: float, vmax: float) -> None:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="equal", interpolation="nearest")
        ax.set_xticks(range(len(short_labels)))
        ax.set_yticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=55, ha="right", fontsize=7)
        ax.set_yticklabels(short_labels, fontsize=7)
        ax.set_title(title, fontweight="bold")
        # Annotate cells
        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                val = data[ii, jj]
                if np.isnan(val):
                    continue
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color=text_color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

    _draw_heatmap(ax1, r_corr, "Real Data Correlations",
                  cmap_main, -1, 1)
    _draw_heatmap(ax2, s_corr, "Synthetic Data Correlations",
                  cmap_main, -1, 1)

    max_diff = max(np.nanmax(np.abs(diff)), 0.1)
    _draw_heatmap(ax3, diff, "Difference (Real - Synthetic)",
                  cmap_diff, -max_diff, max_diff)

    fig.suptitle("Correlation Matrix Comparison", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════
#  4.  PCA ANALYSIS
# ════════════════════════════════════════════════════════════════════
def perform_pca_analysis(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Project both datasets into 2-D PCA space for visual comparison.

    The PCA is fitted on the *combined* dataset so that both point
    clouds share the same principal axes.  Four panels are shown:

      - Top-left:     Real samples in PC1-PC2.
      - Top-right:    Synthetic samples in PC1-PC2.
      - Bottom-left:  Overlay of both.
      - Bottom-right: Explained variance (scree plot).

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.

    Engineering note
    ----------------
    Good overlap in PCA space means the synthetic data occupies the
    same region of the principal feature manifold.  Gaps indicate
    under-represented process regimes; outlier clusters indicate
    physics-constraint violations that pushed samples outside the
    real data envelope.
    """
    cols, _, _ = _prepare_numeric(real, synth)

    r_clean = real[cols].dropna()
    s_clean = synth[cols].dropna()

    combined = pd.concat([r_clean, s_clean], axis=0, ignore_index=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined.values)

    n_r = len(r_clean)

    pca = PCA(n_components=min(10, X_scaled.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    X_r = X_pca[:n_r]
    X_s = X_pca[n_r:]

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    rc = _real_color(0.55)
    sc = _synth_color(0.55)
    rc_edge = _real_color(1.0)
    sc_edge = _synth_color(1.0)

    ev = pca.explained_variance_ratio_ * 100.0
    xl = f"PC1 ({ev[0]:.1f}%)"
    yl = f"PC2 ({ev[1]:.1f}%)"

    # Real only
    axes[0, 0].scatter(X_r[:, 0], X_r[:, 1], c=[rc], edgecolors=[rc_edge],
                       s=30, linewidths=0.4, alpha=0.7, zorder=2)
    axes[0, 0].set_title("Real Data")
    axes[0, 0].set_xlabel(xl)
    axes[0, 0].set_ylabel(yl)

    # Synthetic only
    axes[0, 1].scatter(X_s[:, 0], X_s[:, 1], c=[sc], edgecolors=[sc_edge],
                       s=20, linewidths=0.3, alpha=0.5, zorder=2)
    axes[0, 1].set_title("Synthetic Data")
    axes[0, 1].set_xlabel(xl)
    axes[0, 1].set_ylabel(yl)

    # Overlay
    axes[1, 0].scatter(X_r[:, 0], X_r[:, 1], c=[rc], edgecolors=[rc_edge],
                       s=35, linewidths=0.4, alpha=0.7, label="Real", zorder=3)
    axes[1, 0].scatter(X_s[:, 0], X_s[:, 1], c=[sc], edgecolors=[sc_edge],
                       s=18, linewidths=0.3, alpha=0.35, label="Synthetic",
                       zorder=2)
    axes[1, 0].set_title("Overlay")
    axes[1, 0].set_xlabel(xl)
    axes[1, 0].set_ylabel(yl)
    axes[1, 0].legend()

    # Scree plot
    n_comp = len(ev)
    x_comp = np.arange(1, n_comp + 1)
    axes[1, 1].bar(x_comp, ev, color=_accent_color(2, 0.7),
                   edgecolor=_accent_color(2, 1.0), linewidth=0.6)
    cumulative = np.cumsum(ev)
    axes[1, 1].plot(x_comp, cumulative, color=_accent_color(3),
                    marker="o", markersize=5, linewidth=1.5,
                    label="Cumulative")
    axes[1, 1].axhline(y=90, color="grey", linestyle=":", linewidth=0.8,
                       label="90% threshold")
    axes[1, 1].set_xlabel("Principal Component")
    axes[1, 1].set_ylabel("Explained Variance (%)")
    axes[1, 1].set_title("Scree Plot")
    axes[1, 1].set_xticks(x_comp)
    axes[1, 1].legend()

    fig.suptitle("PCA Projection: Real vs Synthetic SPIHF Data",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════
#  5.  t-SNE ANALYSIS
# ════════════════════════════════════════════════════════════════════
def perform_tsne_analysis(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    perplexity: float = 30.0,
) -> matplotlib.figure.Figure:
    """Project both datasets into 2-D t-SNE space.

    t-SNE is fitted on the *combined* (standardised) dataset.  Three
    panels show the real data, synthetic data, and overlay.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.
    perplexity : float
        t-SNE perplexity.  Default 30.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.

    Engineering note
    ----------------
    t-SNE preserves local neighbourhood structure better than PCA.
    If the synthetic points form tight clusters that do not overlap
    with the real data clusters, the interpolation step is producing
    points in "gaps" of the real manifold — a sign of poor fidelity.
    """
    cols, _, _ = _prepare_numeric(real, synth)

    r_clean = real[cols].dropna()
    s_clean = synth[cols].dropna()
    combined = pd.concat([r_clean, s_clean], axis=0, ignore_index=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined.values)
    n_r = len(r_clean)

    # Adjust perplexity if dataset is small
    n_total = X_scaled.shape[0]
    perp = min(perplexity, max(5.0, n_total / 4.0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                     max_iter=1000, learning_rate="auto", init="pca")
        X_tsne = tsne.fit_transform(X_scaled)

    X_r = X_tsne[:n_r]
    X_s = X_tsne[n_r:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    rc = _real_color(0.6)
    sc = _synth_color(0.6)
    rc_edge = _real_color(1.0)
    sc_edge = _synth_color(1.0)

    axes[0].scatter(X_r[:, 0], X_r[:, 1], c=[rc], edgecolors=[rc_edge],
                    s=30, linewidths=0.4, alpha=0.7)
    axes[0].set_title("Real Data")

    axes[1].scatter(X_s[:, 0], X_s[:, 1], c=[sc], edgecolors=[sc_edge],
                    s=20, linewidths=0.3, alpha=0.5)
    axes[1].set_title("Synthetic Data")

    axes[2].scatter(X_r[:, 0], X_r[:, 1], c=[rc], edgecolors=[rc_edge],
                    s=35, linewidths=0.4, alpha=0.7, label="Real", zorder=3)
    axes[2].scatter(X_s[:, 0], X_s[:, 1], c=[sc], edgecolors=[sc_edge],
                    s=18, linewidths=0.3, alpha=0.35, label="Synthetic", zorder=2)
    axes[2].set_title("Overlay")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")

    fig.suptitle(f"t-SNE Projection (perplexity={perp:.0f}): "
                 f"Real vs Synthetic SPIHF Data",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════
#  5b. UMAP ANALYSIS (optional)
# ════════════════════════════════════════════════════════════════════
def perform_umap_analysis(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> Optional[matplotlib.figure.Figure]:
    """Project both datasets into 2-D UMAP space (if umap-learn is installed).

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.
    n_neighbors : int
        UMAP neighbour count.
    min_dist : float
        UMAP minimum distance.

    Returns
    -------
    matplotlib.figure.Figure or None
        The completed figure, or None if umap-learn is not available.

    Engineering note
    ----------------
    UMAP better preserves global structure than t-SNE and is faster on
    larger datasets.  It is especially useful for checking whether the
    synthetic data fills the same manifold "holes" as the real data.
    """
    if not HAS_UMAP:
        print("[perform_umap_analysis] umap-learn not installed -- skipping.")
        return None

    cols, _, _ = _prepare_numeric(real, synth)
    r_clean = real[cols].dropna()
    s_clean = synth[cols].dropna()
    combined = pd.concat([r_clean, s_clean], axis=0, ignore_index=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined.values)
    n_r = len(r_clean)

    reducer = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors,
                            min_dist=min_dist, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    X_r = X_umap[:n_r]
    X_s = X_umap[n_r:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    rc = _real_color(0.6)
    sc = _synth_color(0.6)
    rc_edge = _real_color(1.0)
    sc_edge = _synth_color(1.0)

    axes[0].scatter(X_r[:, 0], X_r[:, 1], c=[rc], edgecolors=[rc_edge],
                    s=30, linewidths=0.4, alpha=0.7)
    axes[0].set_title("Real Data")

    axes[1].scatter(X_s[:, 0], X_s[:, 1], c=[sc], edgecolors=[sc_edge],
                    s=20, linewidths=0.3, alpha=0.5)
    axes[1].set_title("Synthetic Data")

    axes[2].scatter(X_r[:, 0], X_r[:, 1], c=[rc], edgecolors=[rc_edge],
                    s=35, linewidths=0.4, alpha=0.7, label="Real", zorder=3)
    axes[2].scatter(X_s[:, 0], X_s[:, 1], c=[sc], edgecolors=[sc_edge],
                    s=18, linewidths=0.3, alpha=0.35, label="Synthetic", zorder=2)
    axes[2].set_title("Overlay")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("UMAP Dim 1")
        ax.set_ylabel("UMAP Dim 2")

    fig.suptitle("UMAP Projection: Real vs Synthetic SPIHF Data",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════
#  6.  ENGINEERING RELATIONSHIP SCATTER PLOTS
# ════════════════════════════════════════════════════════════════════
def plot_engineering_relationships(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Plot six domain-specific engineering scatter relationships.

    Each panel shows real vs synthetic data side-by-side for one of
    the key SPIHF process-response relationships:

      1. HER vs Flange Height — forming severity indicator.
      2. Step Depth vs Roughness — surface quality driver.
      3. Stages vs HER — multi-pass effect.
      4. Thickness vs Minimum Thickness — thinning law.
      5. Elongation vs HER — material formability.
      6. R-value vs HER — anisotropy effect.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.

    Engineering note
    ----------------
    These plots encode the most fundamental physics of incremental hole
    flanging.  If the synthetic data distorts these relationships (e.g.
    inverts the Thickness vs Min-Thickness trend), the data is unsuitable
    for training surrogate models or feeding into FEA calibration.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    rc = _real_color(0.65)
    sc = _synth_color(0.45)
    rc_edge = _real_color(1.0)
    sc_edge = _synth_color(1.0)

    for idx, (xcol, ycol, xlabel, ylabel) in enumerate(ENGINEERING_PAIRS):
        ax = axes_flat[idx]

        # Real data
        if xcol in real.columns and ycol in real.columns:
            rr = real[[xcol, ycol]].dropna()
            ax.scatter(rr[xcol], rr[ycol], c=[rc], edgecolors=[rc_edge],
                       s=40, linewidths=0.5, alpha=0.8,
                       label="Real", zorder=3)
            # Add trend line for real data
            if len(rr) >= 3:
                try:
                    z = np.polyfit(rr[xcol].values, rr[ycol].values, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(rr[xcol].min(), rr[xcol].max(), 50)
                    ax.plot(x_trend, p(x_trend), color=rc_edge,
                            linewidth=1.8, linestyle="-", alpha=0.8,
                            zorder=4)
                except Exception:
                    pass

        # Synthetic data
        if xcol in synth.columns and ycol in synth.columns:
            ss = synth[[xcol, ycol]].dropna()
            ax.scatter(ss[xcol], ss[ycol], c=[sc], edgecolors=[sc_edge],
                       s=20, linewidths=0.3, alpha=0.45,
                       label="Synthetic", zorder=2)
            # Trend line for synthetic data
            if len(ss) >= 3:
                try:
                    z2 = np.polyfit(ss[xcol].values, ss[ycol].values, 1)
                    p2 = np.poly1d(z2)
                    x_trend2 = np.linspace(ss[xcol].min(), ss[xcol].max(), 50)
                    ax.plot(x_trend2, p2(x_trend2), color=sc_edge,
                            linewidth=1.8, linestyle="--", alpha=0.8,
                            zorder=4)
                except Exception:
                    pass

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{_short(xcol)} vs {_short(ycol)}",
                     fontweight="bold")
        ax.legend(loc="best", frameon=True)

    fig.suptitle("Engineering Relationships: Real vs Synthetic SPIHF Data",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════
#  7.  SAVE ALL FIGURES
# ════════════════════════════════════════════════════════════════════
def save_all_figures(
    figures: Dict[str, matplotlib.figure.Figure],
    fmt: str = "png",
) -> None:
    """Write every figure to disk and close them to free memory.

    Parameters
    ----------
    figures : Dict[str, matplotlib.figure.Figure]
        Mapping from output filename (without extension) to figure.
    fmt : str
        Image format.  Default ``"png"``.

    Returns
    -------
    None
    """
    for name, fig in figures.items():
        if fig is None:
            continue
        path = f"{name}.{fmt}"
        fig.savefig(path)
        plt.close(fig)
        print(f"[save_all_figures] Saved '{path}'.")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run the complete visualisation pipeline.

    Workflow
    --------
    1. Load and harmonise both datasets.
    2. Generate all figure objects.
    3. Save to disk.
    """
    np.random.seed(42)
    _apply_rc()

    # ── Load data ──────────────────────────────────────────────────
    print("Loading datasets...")
    real_raw = pd.read_csv("SPIHF_Data.csv")
    synth_raw = pd.read_csv("synthetic_SPIHF.csv")

    real = _harmonise_columns(real_raw)
    synth = _harmonise_columns(synth_raw)
    print(f"  Real  : {real.shape[0]} rows x {real.shape[1]} cols")
    print(f"  Synth : {synth.shape[0]} rows x {synth.shape[1]} cols")
    print()

    figures: Dict[str, matplotlib.figure.Figure] = {}

    # ── 1. Distribution comparison ─────────────────────────────────
    print("[1/6] Plotting distribution histograms + KDE...")
    figures["distribution_comparison"] = plot_distributions(real, synth)

    # ── 2. Boxplots + Violin plots ─────────────────────────────────
    print("[2/6] Plotting box + violin plots...")
    fig_box = plot_boxplots(real, synth)
    # Save separately (not in the required list but useful)
    fig_box.savefig("boxviolin_comparison.png")
    plt.close(fig_box)
    print("       -> Saved 'boxviolin_comparison.png'.")

    # ── 3. Correlation heatmaps ────────────────────────────────────
    print("[3/6] Plotting correlation heatmaps...")
    figures["correlation_heatmap"] = plot_correlation_heatmaps(real, synth)

    # ── 4. PCA ─────────────────────────────────────────────────────
    print("[4/6] Performing PCA analysis...")
    figures["pca_comparison"] = perform_pca_analysis(real, synth)

    # ── 5. t-SNE ───────────────────────────────────────────────────
    print("[5/6] Performing t-SNE analysis...")
    figures["tsne_comparison"] = perform_tsne_analysis(real, synth)

    # ── 5b. UMAP (optional) ───────────────────────────────────────
    umap_fig = perform_umap_analysis(real, synth)
    if umap_fig is not None:
        figures["umap_comparison"] = umap_fig

    # ── 6. Engineering relationships ───────────────────────────────
    print("[6/6] Plotting engineering relationships...")
    figures["engineering_relationships"] = plot_engineering_relationships(
        real, synth
    )

    # ── Save all ───────────────────────────────────────────────────
    save_all_figures(figures)

    print("\n[OK] All visualisations generated.")


if __name__ == "__main__":
    main()
