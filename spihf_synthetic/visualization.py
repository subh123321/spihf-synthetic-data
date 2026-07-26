"""
visualization.py
================
Publication-quality comparative visualisation of Original vs Synthetic
SPIHF datasets using matplotlib only (no seaborn styles).

Output figures:
  distribution_comparison.png
  boxviolin_comparison.png
  correlation_heatmap.png
  pca_comparison.png
  tsne_comparison.png
  umap_comparison.png  (optional)
  engineering_relationships.png

Functions
---------
plot_distributions              Histograms + KDE overlays.
plot_boxplots                   Box + violin side-by-side.
plot_correlation_heatmaps       3-panel correlation heatmap.
perform_pca_analysis            PCA 2-D projection.
perform_tsne_analysis           t-SNE 2-D projection.
perform_umap_analysis           UMAP 2-D projection (optional).
plot_engineering_relationships  6-panel scatter plots.
save_all_figures                Write figures to disk.
run_visualization               Full visualisation pipeline (convenience).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from spihf_synthetic.config import (
    ENGINEERING_PAIRS,
    NUMERIC_FEATURES_VALIDATION,
    RANDOM_SEED,
)
from spihf_synthetic.utils import harmonise_columns, short_label

# ──────────────────────────── Optional UMAP ──────────────────────────
try:
    import umap as umap_lib
    HAS_UMAP: bool = True
except ImportError:
    HAS_UMAP = False

# ──────────────────────────── Colour Palette ─────────────────────────
_PROP_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _real_color(alpha: float = 1.0) -> Tuple[float, ...]:
    """RGBA for the 'Original' dataset (position 0 in cycle)."""
    return matplotlib.colors.to_rgba(_PROP_CYCLE[0], alpha)


def _synth_color(alpha: float = 1.0) -> Tuple[float, ...]:
    """RGBA for the 'Synthetic' dataset (position 1 in cycle)."""
    return matplotlib.colors.to_rgba(_PROP_CYCLE[1], alpha)


def _accent_color(idx: int = 2, alpha: float = 1.0) -> Tuple[float, ...]:
    """Accent colour from position *idx* of the prop cycle."""
    return matplotlib.colors.to_rgba(
        _PROP_CYCLE[idx % len(_PROP_CYCLE)], alpha
    )


# ──────────────────────────── Global rcParams ────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

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
    """Return plottable numeric columns (≥5 non-NaN in both datasets)."""
    avail = [
        c for c in NUMERIC_FEATURES_VALIDATION
        if c in real.columns and c in synth.columns
    ]
    keep = [
        c for c in avail
        if real[c].dropna().shape[0] >= 5
        and synth[c].dropna().shape[0] >= 5
    ]
    return keep, real, synth


# ═══════════════════════════════════════════════════════════════════════
#  1.  DISTRIBUTION PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_distributions(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    n_bins: int = 35,
) -> matplotlib.figure.Figure:
    """Plot overlaid histograms and KDE curves for every numeric feature.

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
    """
    cols, real, synth = _prepare_numeric(real, synth)
    n = len(cols)
    ncols_grid = 3
    nrows_grid = int(np.ceil(n / ncols_grid))

    fig, axes = plt.subplots(
        nrows_grid, ncols_grid,
        figsize=(5.5 * ncols_grid, 4.0 * nrows_grid),
    )
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
                edgecolor=rc_line, linewidth=0.4, label="Original", zorder=2)
        ax.hist(sv, bins=bins, density=True, color=sc,
                edgecolor=sc_line, linewidth=0.4, label="Synthetic", zorder=2)

        ax.plot(x_grid, _safe_kde(rv, x_grid),
                color=rc_line, linewidth=1.6, zorder=3)
        ax.plot(x_grid, _safe_kde(sv, x_grid),
                color=sc_line, linewidth=1.6, linestyle="--", zorder=3)

        ax.set_title(short_label(col))
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", frameon=True)

    for j in range(n, nrows_grid * ncols_grid):
        row_idx, col_idx = divmod(j, ncols_grid)
        axes[row_idx, col_idx].set_visible(False)

    fig.suptitle("Distribution Comparison: Original vs Synthetic SPIHF Data",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  2.  BOXPLOTS & VIOLIN PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_boxplots(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Side-by-side boxplots with violin overlays.

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
    """
    cols, real, synth = _prepare_numeric(real, synth)
    n = len(cols)
    ncols_grid = 3
    nrows_grid = int(np.ceil(n / ncols_grid))

    fig, axes = plt.subplots(
        nrows_grid, ncols_grid,
        figsize=(5.5 * ncols_grid, 4.5 * nrows_grid),
    )
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

        bp = ax.boxplot(
            [rv, sv], positions=[1, 2], widths=0.35,
            patch_artist=True, showfliers=True,
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
            medianprops=dict(color="black", linewidth=1.5),
        )
        bp["boxes"][0].set_facecolor(rc_face)
        bp["boxes"][0].set_edgecolor(rc_edge)
        bp["boxes"][1].set_facecolor(sc_face)
        bp["boxes"][1].set_edgecolor(sc_edge)

        if len(rv) >= 5 and np.std(rv) > 1e-12:
            vp_r = ax.violinplot(
                [rv], positions=[1], widths=0.55,
                showmeans=False, showmedians=False, showextrema=False,
            )
            for body in vp_r["bodies"]:
                body.set_facecolor(rc_face)
                body.set_edgecolor(rc_edge)
                body.set_alpha(0.25)

        if len(sv) >= 5 and np.std(sv) > 1e-12:
            vp_s = ax.violinplot(
                [sv], positions=[2], widths=0.55,
                showmeans=False, showmedians=False, showextrema=False,
            )
            for body in vp_s["bodies"]:
                body.set_facecolor(sc_face)
                body.set_edgecolor(sc_edge)
                body.set_alpha(0.25)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Original", "Synthetic"])
        ax.set_title(short_label(col))

    for j in range(n, nrows_grid * ncols_grid):
        row_idx, col_idx = divmod(j, ncols_grid)
        axes[row_idx, col_idx].set_visible(False)

    fig.suptitle(
        "Box + Violin Comparison: Original vs Synthetic SPIHF Data",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  3.  CORRELATION HEATMAPS
# ═══════════════════════════════════════════════════════════════════════

def plot_correlation_heatmaps(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Side-by-side correlation heatmaps plus a difference map.

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
    """
    cols, _, _ = _prepare_numeric(real, synth)
    short_labels = [short_label(c) for c in cols]

    r_corr = real[cols].corr().values
    s_corr = synth[cols].corr().values
    diff = r_corr - s_corr

    cmap_main = plt.cm.RdBu_r
    cmap_diff = plt.cm.PiYG

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))

    def _draw_heatmap(
        ax: plt.Axes,
        data: np.ndarray,
        title: str,
        cmap: Any,
        vmin: float,
        vmax: float,
    ) -> None:
        im = ax.imshow(
            data, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="equal", interpolation="nearest",
        )
        ax.set_xticks(range(len(short_labels)))
        ax.set_yticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=55, ha="right", fontsize=7)
        ax.set_yticklabels(short_labels, fontsize=7)
        ax.set_title(title, fontweight="bold")
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

    _draw_heatmap(ax1, r_corr, "Original Data Correlations", cmap_main, -1, 1)
    _draw_heatmap(ax2, s_corr, "Synthetic Data Correlations", cmap_main, -1, 1)

    max_diff = max(np.nanmax(np.abs(diff)), 0.1)
    _draw_heatmap(ax3, diff, "Difference (Original − Synthetic)",
                  cmap_diff, -max_diff, max_diff)

    fig.suptitle("Correlation Matrix Comparison", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  4.  PCA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def perform_pca_analysis(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Project both datasets into 2-D PCA space.

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
    """
    cols, _, _ = _prepare_numeric(real, synth)

    r_clean = real[cols].dropna()
    s_clean = synth[cols].dropna()
    combined = pd.concat([r_clean, s_clean], axis=0, ignore_index=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined.values)
    n_r = len(r_clean)

    pca = PCA(n_components=min(10, X_scaled.shape[1]), random_state=RANDOM_SEED)
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

    axes[0, 0].scatter(X_r[:, 0], X_r[:, 1], c=[rc], edgecolors=[rc_edge],
                       s=30, linewidths=0.4, alpha=0.7, zorder=2)
    axes[0, 0].set_title("Original Data")
    axes[0, 0].set_xlabel(xl)
    axes[0, 0].set_ylabel(yl)

    axes[0, 1].scatter(X_s[:, 0], X_s[:, 1], c=[sc], edgecolors=[sc_edge],
                       s=20, linewidths=0.3, alpha=0.5, zorder=2)
    axes[0, 1].set_title("Synthetic Data")
    axes[0, 1].set_xlabel(xl)
    axes[0, 1].set_ylabel(yl)

    axes[1, 0].scatter(X_r[:, 0], X_r[:, 1], c=[rc], edgecolors=[rc_edge],
                       s=35, linewidths=0.4, alpha=0.7, label="Original",
                       zorder=3)
    axes[1, 0].scatter(X_s[:, 0], X_s[:, 1], c=[sc], edgecolors=[sc_edge],
                       s=18, linewidths=0.3, alpha=0.35, label="Synthetic",
                       zorder=2)
    axes[1, 0].set_title("Overlay")
    axes[1, 0].set_xlabel(xl)
    axes[1, 0].set_ylabel(yl)
    axes[1, 0].legend()

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

    fig.suptitle("PCA Projection: Original vs Synthetic SPIHF Data",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  5.  t-SNE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def perform_tsne_analysis(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    perplexity: float = 30.0,
) -> matplotlib.figure.Figure:
    """Project both datasets into 2-D t-SNE space.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.
    perplexity : float
        t-SNE perplexity.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.
    """
    cols, _, _ = _prepare_numeric(real, synth)

    r_clean = real[cols].dropna()
    s_clean = synth[cols].dropna()
    combined = pd.concat([r_clean, s_clean], axis=0, ignore_index=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined.values)
    n_r = len(r_clean)

    n_total = X_scaled.shape[0]
    perp = min(perplexity, max(5.0, n_total / 4.0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(
            n_components=2, perplexity=perp, random_state=RANDOM_SEED,
            max_iter=1000, learning_rate="auto", init="pca",
        )
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
    axes[0].set_title("Original Data")

    axes[1].scatter(X_s[:, 0], X_s[:, 1], c=[sc], edgecolors=[sc_edge],
                    s=20, linewidths=0.3, alpha=0.5)
    axes[1].set_title("Synthetic Data")

    axes[2].scatter(X_r[:, 0], X_r[:, 1], c=[rc], edgecolors=[rc_edge],
                    s=35, linewidths=0.4, alpha=0.7, label="Original",
                    zorder=3)
    axes[2].scatter(X_s[:, 0], X_s[:, 1], c=[sc], edgecolors=[sc_edge],
                    s=18, linewidths=0.3, alpha=0.35, label="Synthetic",
                    zorder=2)
    axes[2].set_title("Overlay")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")

    fig.suptitle(
        f"t-SNE Projection (perplexity={perp:.0f}): "
        f"Original vs Synthetic SPIHF Data",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  5b. UMAP ANALYSIS (optional)
# ═══════════════════════════════════════════════════════════════════════

def perform_umap_analysis(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> Optional[matplotlib.figure.Figure]:
    """Project both datasets into 2-D UMAP space (if umap-learn installed).

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
        Figure, or ``None`` if umap-learn is unavailable.
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

    reducer = umap_lib.UMAP(
        n_components=2, n_neighbors=n_neighbors,
        min_dist=min_dist, random_state=RANDOM_SEED,
    )
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
    axes[0].set_title("Original Data")

    axes[1].scatter(X_s[:, 0], X_s[:, 1], c=[sc], edgecolors=[sc_edge],
                    s=20, linewidths=0.3, alpha=0.5)
    axes[1].set_title("Synthetic Data")

    axes[2].scatter(X_r[:, 0], X_r[:, 1], c=[rc], edgecolors=[rc_edge],
                    s=35, linewidths=0.4, alpha=0.7, label="Original",
                    zorder=3)
    axes[2].scatter(X_s[:, 0], X_s[:, 1], c=[sc], edgecolors=[sc_edge],
                    s=18, linewidths=0.3, alpha=0.35, label="Synthetic",
                    zorder=2)
    axes[2].set_title("Overlay")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("UMAP Dim 1")
        ax.set_ylabel("UMAP Dim 2")

    fig.suptitle("UMAP Projection: Original vs Synthetic SPIHF Data",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  6.  ENGINEERING RELATIONSHIP SCATTER PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_engineering_relationships(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Plot six domain-specific engineering scatter relationships.

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
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    rc = _real_color(0.65)
    sc = _synth_color(0.45)
    rc_edge = _real_color(1.0)
    sc_edge = _synth_color(1.0)

    for idx, (xcol, ycol, xlabel, ylabel) in enumerate(ENGINEERING_PAIRS):
        ax = axes_flat[idx]

        if xcol in real.columns and ycol in real.columns:
            rr = real[[xcol, ycol]].dropna()
            ax.scatter(rr[xcol], rr[ycol], c=[rc], edgecolors=[rc_edge],
                       s=40, linewidths=0.5, alpha=0.8,
                       label="Original", zorder=3)
            if len(rr) >= 3:
                try:
                    z = np.polyfit(rr[xcol].values, rr[ycol].values, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(rr[xcol].min(), rr[xcol].max(), 50)
                    ax.plot(x_trend, p(x_trend), color=rc_edge,
                            linewidth=1.8, linestyle="-", alpha=0.8, zorder=4)
                except Exception:
                    pass

        if xcol in synth.columns and ycol in synth.columns:
            ss = synth[[xcol, ycol]].dropna()
            ax.scatter(ss[xcol], ss[ycol], c=[sc], edgecolors=[sc_edge],
                       s=20, linewidths=0.3, alpha=0.45,
                       label="Synthetic", zorder=2)
            if len(ss) >= 3:
                try:
                    z2 = np.polyfit(ss[xcol].values, ss[ycol].values, 1)
                    p2 = np.poly1d(z2)
                    x_trend2 = np.linspace(ss[xcol].min(), ss[xcol].max(), 50)
                    ax.plot(x_trend2, p2(x_trend2), color=sc_edge,
                            linewidth=1.8, linestyle="--", alpha=0.8, zorder=4)
                except Exception:
                    pass

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{short_label(xcol)} vs {short_label(ycol)}",
                     fontweight="bold")
        ax.legend(loc="best", frameon=True)

    fig.suptitle(
        "Engineering Relationships: Original vs Synthetic SPIHF Data",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  7.  SAVE ALL FIGURES
# ═══════════════════════════════════════════════════════════════════════

def save_all_figures(
    figures: Dict[str, matplotlib.figure.Figure],
    output_dir: str = "outputs/figures",
    fmt: str = "png",
) -> None:
    """Write every figure to disk and close to free memory.

    Parameters
    ----------
    figures : Dict[str, matplotlib.figure.Figure]
        Mapping from output filename (without extension) to figure.
    output_dir : str
        Directory to save figures into.
    fmt : str
        Image format.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for name, fig in figures.items():
        if fig is None:
            continue
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path)
        plt.close(fig)
        print(f"[save_all_figures] Saved '{path}'.")


# ═══════════════════════════════════════════════════════════════════════
#  CONVENIENCE: FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_visualization(
    real_path: str = "SPIHF_Data.csv",
    synth_path: str = "outputs/synthetic_SPIHF.csv",
    output_dir: str = "outputs/figures",
) -> None:
    """Run the complete visualisation pipeline.

    Parameters
    ----------
    real_path : str
        Path to original dataset CSV.
    synth_path : str
        Path to synthetic dataset CSV.
    output_dir : str
        Directory for output figures.
    """
    np.random.seed(RANDOM_SEED)
    _apply_rc()

    print("Loading datasets...")
    real = harmonise_columns(pd.read_csv(real_path))
    synth = harmonise_columns(pd.read_csv(synth_path))
    print(f"  Original : {real.shape[0]} rows x {real.shape[1]} cols")
    print(f"  Synthetic: {synth.shape[0]} rows x {synth.shape[1]} cols\n")

    figures: Dict[str, matplotlib.figure.Figure] = {}

    print("[1/7] Plotting distribution histograms + KDE...")
    figures["distribution_comparison"] = plot_distributions(real, synth)

    print("[2/7] Plotting box + violin plots...")
    figures["boxviolin_comparison"] = plot_boxplots(real, synth)

    print("[3/7] Plotting correlation heatmaps...")
    figures["correlation_heatmap"] = plot_correlation_heatmaps(real, synth)

    print("[4/7] Performing PCA analysis...")
    figures["pca_comparison"] = perform_pca_analysis(real, synth)

    print("[5/7] Performing t-SNE analysis...")
    figures["tsne_comparison"] = perform_tsne_analysis(real, synth)

    print("[6/7] Performing UMAP analysis (if available)...")
    umap_fig = perform_umap_analysis(real, synth)
    if umap_fig is not None:
        figures["umap_comparison"] = umap_fig

    print("[7/7] Plotting engineering relationships...")
    figures["engineering_relationships"] = plot_engineering_relationships(
        real, synth
    )

    save_all_figures(figures, output_dir=output_dir)
    print("\n[OK] All visualisations generated.")
