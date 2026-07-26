"""
reporting.py
============
Automated generation of publication-quality Markdown reports for the
SPIHF synthetic-data study.

Output files:
  1. engineering_report.md   -- Engineering validity analysis.
  2. validation_report.md    -- Statistical validation results.
  3. methodology_report.md   -- Synthetic generation methodology.

Functions
---------
generate_dataset_summary       Sample counts, material distribution, stats.
generate_methodology_section   Augmentation pipeline description.
generate_validation_section    KS, Wasserstein, JSD, correlations, importance.
generate_engineering_section   Physics-based validity analysis.
generate_limitations_section   Research caveats and risk factors.
compile_reports                Assemble all sections into three .md files.
run_reporting                  Full report pipeline (convenience).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from spihf_synthetic.config import (
    NUMERIC_FEATURES_VALIDATION,
    RANDOM_SEED,
)
from spihf_synthetic.utils import (
    detect_outliers_iqr,
    fmt_float,
    fmt_pct,
    harmonise_columns,
    short_label,
)


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _grade_emoji(grade: str) -> str:
    """Return an ASCII indicator for a letter grade."""
    mapping = {
        "A": "[Excellent]", "B": "[Good]", "C": "[Moderate]",
        "D": "[Weak]", "F": "[Fail]",
    }
    return mapping.get(grade, f"[{grade}]")


def _verdict_indicator(v: str) -> str:
    """Return a readable indicator for PASS/MARGINAL/FAIL."""
    mapping = {
        "PASS": "[PASS]", "MARGINAL": "[MARGINAL]", "FAIL": "[FAIL]",
    }
    return mapping.get(v, f"[{v}]")


def _ks_verdict(data: Dict[str, Any]) -> str:
    """Derive KS test verdict from JSON data."""
    if "verdict" in data:
        return data["verdict"]
    if "same_distribution" in data:
        return "PASS" if data["same_distribution"] else "FAIL"
    return "?"


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1 : Dataset Summary
# ═══════════════════════════════════════════════════════════════════════

def generate_dataset_summary(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> str:
    """Generate the *Dataset Summary* section as Markdown text.

    Includes sample counts, material distribution, descriptive statistics,
    missing-value audit, and outlier census (Tukey IQR method).

    Parameters
    ----------
    real : pd.DataFrame
        Original (experimental) dataset, already harmonised.
    synth : pd.DataFrame
        Synthetic (augmented) dataset, already harmonised.

    Returns
    -------
    str
        Markdown text.
    """
    lines: List[str] = []

    # 1.1 Sample counts
    lines.append("## 1. Dataset Summary\n")
    lines.append("### 1.1 Sample Counts\n")
    lines.append("| Dataset | Samples | Features |")
    lines.append("|---------|--------:|---------:|")
    lines.append(f"| Original (experimental) | {real.shape[0]} | {real.shape[1]} |")
    lines.append(f"| Synthetic (augmented) | {synth.shape[0]} | {synth.shape[1]} |")
    lines.append(f"| **Combined** | **{real.shape[0] + synth.shape[0]}** "
                 f"| **{max(real.shape[1], synth.shape[1])}** |")
    lines.append("")
    lines.append(
        f"Augmentation factor: **{synth.shape[0] / max(real.shape[0], 1):.1f}x**, "
        f"combined dataset: **{real.shape[0] + synth.shape[0]}** observations.\n"
    )

    # 1.2 Material distribution
    lines.append("### 1.2 Material Distribution\n")
    if "Material" in real.columns:
        r_mat = real["Material"].value_counts().sort_index()
        s_mat = synth["Material"].value_counts().sort_index()
        all_mats = sorted(set(r_mat.index) | set(s_mat.index))
        lines.append("| Material | Original (n) | Original (%) | Synthetic (n) | Synthetic (%) |")
        lines.append("|----------|--------:|--------:|--------------:|--------------:|")
        for mat in all_mats:
            rn = int(r_mat.get(mat, 0))
            sn = int(s_mat.get(mat, 0))
            rp = 100.0 * rn / real.shape[0] if real.shape[0] > 0 else 0
            sp = 100.0 * sn / synth.shape[0] if synth.shape[0] > 0 else 0
            lines.append(f"| {mat} | {rn} | {rp:.1f}% | {sn} | {sp:.1f}% |")
        lines.append("")

    # 1.3 Descriptive statistics
    lines.append("### 1.3 Numerical Feature Statistics\n")
    avail = [c for c in NUMERIC_FEATURES_VALIDATION
             if c in real.columns and c in synth.columns]
    lines.append("| Feature | Orig. Mean | Orig. Std | Orig. Median "
                 "| Synth. Mean | Synth. Std | Synth. Median |")
    lines.append("|---------|----------:|---------:|------------:"
                 "|-----------:|----------:|-------------:|")
    for col in avail:
        rm, rs, rmed = real[col].mean(), real[col].std(), real[col].median()
        sm, ss, smed = synth[col].mean(), synth[col].std(), synth[col].median()
        lines.append(
            f"| {short_label(col)} | {fmt_float(rm, 2)} | {fmt_float(rs, 2)} "
            f"| {fmt_float(rmed, 2)} | {fmt_float(sm, 2)} | {fmt_float(ss, 2)} "
            f"| {fmt_float(smed, 2)} |"
        )
    lines.append("")

    # 1.4 Missing values
    lines.append("### 1.4 Missing Value Audit\n")
    lines.append("| Feature | Orig. Missing | Orig. Missing (%) "
                 "| Synth. Missing | Synth. Missing (%) |")
    lines.append("|---------|------------:|----------------:"
                 "|--------------:|------------------:|")
    for col in avail:
        rmiss = int(real[col].isna().sum())
        smiss = int(synth[col].isna().sum())
        if rmiss > 0 or smiss > 0:
            rp = 100.0 * rmiss / real.shape[0]
            sp = 100.0 * smiss / synth.shape[0]
            lines.append(f"| {short_label(col)} | {rmiss} | {rp:.1f}% "
                         f"| {smiss} | {sp:.1f}% |")
    lines.append("")

    # 1.5 Outlier census
    lines.append("### 1.5 Outlier Census (Tukey IQR Method)\n")
    lines.append("| Feature | Orig. Outliers | Synth. Outliers |")
    lines.append("|---------|-------------:|--------------:|")
    for col in avail:
        ro = detect_outliers_iqr(real[col])
        so = detect_outliers_iqr(synth[col])
        lines.append(f"| {short_label(col)} | {ro} | {so} |")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2 : Methodology
# ═══════════════════════════════════════════════════════════════════════

def generate_methodology_section(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> str:
    """Generate the *Synthetic Generation Methodology* section.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.

    Returns
    -------
    str
        Markdown text.
    """
    n_real = real.shape[0]
    n_synth = synth.shape[0]
    n_mat = real["Material"].nunique() if "Material" in real.columns else "unknown"

    lines: List[str] = []
    lines.append("## 2. Synthetic Data Generation Methodology\n")

    lines.append("### 2.1 Overview\n")
    lines.append(
        f"The SPIHF dataset comprises **{n_real}** observations spanning "
        f"**{n_mat}** alloys.  A physics-informed pipeline generated "
        f"**{n_synth}** synthetic samples.\n"
    )

    lines.append("### 2.2 Material-Wise Stratified Generation\n")
    lines.append(
        "Synthetic samples are generated **independently within each material "
        "group** to prevent cross-material interpolation artefacts.\n"
    )

    lines.append("### 2.3 SMOTE-Inspired Interpolation\n")
    lines.append("```")
    lines.append("x_new = α · x_i + (1 − α) · x_j")
    lines.append("α ~ Uniform(0.2, 0.8)")
    lines.append("```\n")

    lines.append("### 2.4 Gaussian Perturbation\n")
    lines.append("```")
    lines.append("x_perturbed = x_interpolated + ε")
    lines.append("ε ~ N(0, σ_material · η),  η ∈ [0.01, 0.03]")
    lines.append("```\n")

    lines.append("### 2.5 Physics-Informed Rejection Sampling\n")
    lines.append("| # | Constraint | Rationale |")
    lines.append("|:-:|-----------|----------|")
    lines.append("| 1 | UTS ≥ YS | Engineering stress–strain curve definition. |")
    lines.append("| 2 | Thickness > 0 | Strictly positive. |")
    lines.append("| 3 | HER > 0 | Positive geometric ratio. |")
    lines.append("| 4 | Min Thickness ≤ Thickness | Thinning constraint. |")
    lines.append("| 5 | 0 < n < 1 | Hollomon exponent bounds. |")
    lines.append("| 6 | R ≥ 0 | Lankford coefficient non-negative. |")
    lines.append("| 7 | Step Depth > 0 | Positive tool step-down. |")
    lines.append("| 8 | Stages ≥ 1 | At least one pass. |")
    lines.append("| 9 | Angle ∈ [0, 180] | Wall angle bounds. |")
    lines.append("")

    lines.append("### 2.6 Confidence Scoring\n")
    lines.append(
        "Each sample receives a confidence score in [0, 1] combining "
        "range proximity, nearest-neighbour distance, and correlation "
        "agreement.\n"
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3 : Validation Results
# ═══════════════════════════════════════════════════════════════════════

def generate_validation_section(
    metrics: Dict[str, Any],
) -> str:
    """Generate the *Validation Results* section from pre-computed metrics.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Contents of ``validation_metrics.json``.

    Returns
    -------
    str
        Markdown text.
    """
    summary = metrics["summary"]
    ks_tests = metrics.get("ks_tests", {})
    wass_jsd = metrics.get("wasserstein_and_jsd", {})
    corr = metrics.get("correlation_analysis", {})
    fimp = metrics.get("feature_importance", {})
    dist_stats = metrics.get("distribution_statistics", {})

    lines: List[str] = []
    lines.append("## 3. Validation Results\n")

    # 3.1 Overall
    lines.append("### 3.1 Overall Quality Summary\n")
    grade = summary.get("grade", "?")
    score = summary.get("composite_score", 0)
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| Composite Score | **{fmt_float(score, 2)}** / 100 |")
    lines.append(f"| Letter Grade | **{grade}** {_grade_emoji(grade)} |")
    lines.append(f"| KS Pass Rate | {fmt_pct(summary.get('ks_pass_rate', 0) * 100)} |")
    lines.append(f"| Mean Wasserstein (norm.) | "
                 f"{fmt_float(summary.get('mean_wasserstein_normalised', 0), 4)} |")
    lines.append(f"| Mahalanobis Distance | "
                 f"{fmt_float(summary.get('mahalanobis_distance', 0), 2)} |")
    lines.append("")

    # 3.2 KS tests
    lines.append("### 3.2 Kolmogorov–Smirnov Test Results\n")
    lines.append("| Feature | KS Statistic | p-value | Verdict |")
    lines.append("|---------|------------:|--------:|--------:|")
    for feat, data in ks_tests.items():
        stat = data.get("ks_statistic", 0)
        pval = data.get("p_value", 0)
        verdict = _ks_verdict(data)
        lines.append(f"| {short_label(feat)} | {fmt_float(stat, 4)} "
                     f"| {fmt_float(pval, 4)} | {_verdict_indicator(verdict)} |")
    lines.append("")

    # 3.3 Wasserstein / JSD
    lines.append("### 3.3 Wasserstein Distance and Jensen–Shannon Divergence\n")
    lines.append("| Feature | Wasserstein | W (norm.) | JSD |")
    lines.append("|---------|------------:|----------:|----:|")
    for feat, data in wass_jsd.items():
        wd = data.get("wasserstein_distance", 0)
        wn = data.get("wasserstein_normalised", 0)
        jsd = data.get("jensen_shannon_divergence", 0)
        lines.append(f"| {short_label(feat)} | {fmt_float(wd, 4)} "
                     f"| {fmt_float(wn, 4)} | {fmt_float(jsd, 6)} |")
    lines.append("")

    # 3.4 Correlations
    lines.append("### 3.4 Correlation Preservation\n")
    lines.append(f"- **Frobenius norm:** {fmt_float(corr.get('frobenius_norm', 0), 4)}")
    lines.append(f"- **Mean |Δρ|:** {fmt_float(corr.get('mean_abs_corr_diff', 0), 4)}\n")

    phys = corr.get("physics_pairs", {})
    if phys:
        lines.append("| Pair | Orig. ρ | Synth. ρ | |Δ| | Sign OK |")
        lines.append("|------|--------:|---------:|----:|:-------:|")
        for pair_name, data in phys.items():
            rr = data.get("real_correlation", 0)
            sr = data.get("synth_correlation", 0)
            ad = data.get("abs_difference", 0)
            sp = "Yes" if data.get("sign_preserved", False) else "No"
            lines.append(f"| {pair_name} | {fmt_float(rr, 4)} "
                         f"| {fmt_float(sr, 4)} | {fmt_float(ad, 4)} | {sp} |")
        lines.append("")

    # 3.5 Feature importance
    lines.append("### 3.5 Feature Importance Similarity\n")
    spear_rf = fimp.get("spearman_rf", {})
    spear_mi = fimp.get("spearman_mi", {})
    lines.append("| Method | Spearman ρ | p-value | Verdict |")
    lines.append("|--------|----------:|--------:|--------:|")
    lines.append(f"| Random Forest | {fmt_float(spear_rf.get('rho', 0), 4)} "
                 f"| {fmt_float(spear_rf.get('p_value', 0), 6)} "
                 f"| {_verdict_indicator(fimp.get('verdict_rf', '?'))} |")
    lines.append(f"| Mutual Info | {fmt_float(spear_mi.get('rho', 0), 4)} "
                 f"| {fmt_float(spear_mi.get('p_value', 0), 6)} "
                 f"| {_verdict_indicator(fimp.get('verdict_mi', '?'))} |")
    lines.append("")

    # 3.6 Descriptive stats
    lines.append("### 3.6 Descriptive Statistics Comparison\n")
    lines.append("| Feature | Orig. Mean | Synth. Mean | % Diff Mean "
                 "| Orig. Std | Synth. Std | % Diff Std |")
    lines.append("|---------|----------:|-----------:|:-----------:"
                 "|---------:|----------:|:----------:|")
    for feat, data in dist_stats.items():
        if "frequencies" in feat or not isinstance(data, dict):
            continue
        rm = data.get("real_mean")
        if rm is None:
            continue
        sm = data.get("synth_mean", 0)
        pm = data.get("pct_diff_mean", 0)
        rs = data.get("real_std", 0)
        ss = data.get("synth_std", 0)
        ps = data.get("pct_diff_std", 0)
        lines.append(f"| {short_label(feat)} | {fmt_float(rm, 2)} "
                     f"| {fmt_float(sm, 2)} | {fmt_float(pm, 2)}% "
                     f"| {fmt_float(rs, 2)} | {fmt_float(ss, 2)} "
                     f"| {fmt_float(ps, 2)}% |")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 4 : Engineering Validity
# ═══════════════════════════════════════════════════════════════════════

def generate_engineering_section(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    metrics: Dict[str, Any],
) -> str:
    """Generate the *Engineering Validity* discussion section.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.
    metrics : Dict[str, Any]
        Contents of ``validation_metrics.json``.

    Returns
    -------
    str
        Markdown text.
    """
    lines: List[str] = []
    lines.append("## 4. Engineering Validity\n")

    # 4.1 UTS >= YS
    lines.append("### 4.1 UTS ≥ YS Constraint\n")
    if "UTS (MPa)" in synth.columns and "YS (MPa)" in synth.columns:
        violations = synth[synth["UTS (MPa)"] < synth["YS (MPa)"]].shape[0]
        total = synth.dropna(subset=["UTS (MPa)", "YS (MPa)"]).shape[0]
        lines.append(
            f"**{violations} out of {total}** samples violate UTS < YS "
            f"({fmt_pct(100.0 * violations / max(total, 1))}).\n"
        )

    # 4.2 HER
    lines.append("### 4.2 Hole Expansion Ratio Behaviour\n")
    if "HER" in synth.columns:
        r_her = real["HER"].dropna()
        s_her = synth["HER"].dropna()
        lines.append(
            f"Original HER: [{fmt_float(r_her.min(), 2)}, "
            f"{fmt_float(r_her.max(), 2)}], mean = {fmt_float(r_her.mean(), 2)}.  "
            f"Synthetic HER: [{fmt_float(s_her.min(), 2)}, "
            f"{fmt_float(s_her.max(), 2)}], mean = {fmt_float(s_her.mean(), 2)}.\n"
        )

    # 4.3 Thickness
    lines.append("### 4.3 Thickness Evolution\n")
    if ("Thickness (mm)" in synth.columns
            and "Minimum thickness (after final stage, mm)" in synth.columns):
        violations = int(
            (synth["Minimum thickness (after final stage, mm)"]
             > synth["Thickness (mm)"]).sum()
        )
        lines.append(
            f"**{violations}** synthetic samples violate Min Thickness ≤ "
            f"Initial Thickness.\n"
        )

    # 4.4 Surface roughness
    lines.append("### 4.4 Surface Roughness Trends\n")
    phys = metrics.get("correlation_analysis", {}).get("physics_pairs", {})
    rough_pair = phys.get("Step depth (mm) <-> Roughness (um)", {})
    if rough_pair:
        lines.append(
            f"Step Depth vs Roughness ρ: original = "
            f"{fmt_float(rough_pair.get('real_correlation', 0), 4)}, "
            f"synthetic = {fmt_float(rough_pair.get('synth_correlation', 0), 4)}.\n"
        )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 5 : Limitations
# ═══════════════════════════════════════════════════════════════════════

def generate_limitations_section(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    metrics: Dict[str, Any],
) -> str:
    """Generate the *Research Limitations* discussion section.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.
    metrics : Dict[str, Any]
        Contents of ``validation_metrics.json``.

    Returns
    -------
    str
        Markdown text.
    """
    n_real = real.shape[0]
    n_synth = synth.shape[0]
    n_mat = real["Material"].nunique() if "Material" in real.columns else "?"
    grade = metrics.get("summary", {}).get("grade", "?")

    lines: List[str] = []
    lines.append("## 5. Research Limitations\n")

    lines.append("### 5.1 Small Original Dataset\n")
    lines.append(
        f"The original dataset contains only **{n_real}** observations "
        f"across **{n_mat}** materials.  Consequences include sampling "
        f"noise amplification and unreliable higher-order statistics.\n"
    )

    lines.append("### 5.2 Synthetic Data Bias\n")
    lines.append(
        "No extrapolation beyond tested configurations. "
        "Correlation attenuation from independent Gaussian noise. "
        "Mode-collapse risk for bimodal features.\n"
    )

    lines.append("### 5.3 Overfitting Dangers\n")
    lines.append(
        f"Augmented dataset is **{n_synth / max(n_real, 1):.1f}×** "
        f"larger.  Always hold out the original dataset for final "
        f"evaluation.\n"
    )

    lines.append("### 5.4 Physical Assumptions\n")
    lines.append(
        "Simplified constraints (UTS ≥ YS, sine-law thinning) are "
        "necessary but not sufficient.  Strain-path dependence, "
        "anisotropy coupling, and temperature effects are not captured.\n"
    )

    lines.append("### 5.5 Generalisation Limitations\n")
    lines.append(
        f"Overall grade **{grade}**.  Valid only for the specific "
        f"alloy systems and process configurations in the dataset.\n"
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  REPORT COMPILATION
# ═══════════════════════════════════════════════════════════════════════

def compile_reports(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    metrics: Dict[str, Any],
) -> Dict[str, str]:
    """Compile all three output reports from data and metrics.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset (harmonised).
    synth : pd.DataFrame
        Synthetic dataset (harmonised).
    metrics : Dict[str, Any]
        Contents of ``validation_metrics.json``.

    Returns
    -------
    Dict[str, str]
        ``{filename: markdown_content}``.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"> **Auto-generated** on {timestamp} by `spihf_synthetic.reporting`\n"
        f"> Original samples: {real.shape[0]} | "
        f"Synthetic samples: {synth.shape[0]}\n\n---\n\n"
    )

    sec_summary = generate_dataset_summary(real, synth)
    sec_method = generate_methodology_section(real, synth)
    sec_valid = generate_validation_section(metrics)
    sec_eng = generate_engineering_section(real, synth, metrics)
    sec_lim = generate_limitations_section(real, synth, metrics)

    eng_report = (
        "# Engineering Report: SPIHF Synthetic Data Analysis\n\n"
        + header + sec_summary + "\n\n" + sec_eng + "\n\n" + sec_lim + "\n"
    )
    val_report = (
        "# Validation Report: Statistical Fidelity of Synthetic SPIHF Data\n\n"
        + header + sec_valid + "\n\n" + sec_lim + "\n"
    )
    meth_report = (
        "# Methodology Report: Synthetic Data Generation for SPIHF\n\n"
        + header + sec_method + "\n\n" + sec_summary + "\n\n" + sec_valid + "\n"
    )

    return {
        "engineering_report.md": eng_report,
        "validation_report.md": val_report,
        "methodology_report.md": meth_report,
    }


# ═══════════════════════════════════════════════════════════════════════
#  CONVENIENCE: FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_reporting(
    real_path: str = "SPIHF_Data.csv",
    synth_path: str = "outputs/synthetic_SPIHF.csv",
    metrics_path: str = "outputs/reports/validation_metrics.json",
    output_dir: str = "outputs/reports",
) -> None:
    """Load data, generate all three reports, write to disk.

    Parameters
    ----------
    real_path : str
        Path to original dataset CSV.
    synth_path : str
        Path to synthetic dataset CSV.
    metrics_path : str
        Path to validation_metrics.json.
    output_dir : str
        Directory for output reports.
    """
    import os
    np.random.seed(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets...")
    real = harmonise_columns(pd.read_csv(real_path))
    synth = harmonise_columns(pd.read_csv(synth_path))
    print(f"  Original : {real.shape[0]} rows x {real.shape[1]} cols")
    print(f"  Synthetic: {synth.shape[0]} rows x {synth.shape[1]} cols")

    print("Loading validation metrics...")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print("Compiling reports...")
    reports = compile_reports(real, synth, metrics)

    for filename, content in reports.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        n_lines = content.count("\n") + 1
        print(f"  -> Saved '{path}' ({n_lines} lines, {len(content)} bytes)")

    print("\n[OK] All three reports generated.")
