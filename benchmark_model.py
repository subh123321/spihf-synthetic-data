"""
benchmark_model.py
==================
Downstream ML benchmark: does synthetic data actually improve predictive
performance for SPIHF Hole Expansion Ratio (HER) prediction?

This script trains three model families under three data regimes:

Data Regimes
------------
1. **Real-only** — Train on original 304 experimental samples.
2. **Augmented** — Train on real + synthetic combined data.
3. **Synthetic-only** — Train on synthetic data alone (sanity check).

Models
------
- Random Forest Regressor
- Gradient Boosting Regressor
- Ridge Regression (linear baseline)

Evaluation
----------
- Test set is **always real data only** (honest evaluation).
- Metrics: R², MAE, RMSE.
- 5-fold cross-validation on each regime for robust estimates.

Usage::

    python benchmark_model.py
    python benchmark_model.py --real SPIHF_Data.csv --synth synthetic_SPIHF.csv
    python benchmark_model.py --target "HER" --test-frac 0.25
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Ensure the package is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spihf_synthetic.config import NUMERIC_FEATURES, RANDOM_SEED
from spihf_synthetic.utils import build_material_map, harmonise_columns


# ═══════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════

def load_and_prepare(
    path: str,
    target: str = "HER",
    encode_material: bool = True,
) -> pd.DataFrame:
    """Load a CSV, harmonise columns, encode categoricals, drop NaN targets.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    target : str
        Target column name.
    encode_material : bool
        If True, label-encode Material to a numeric column.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for modelling.
    """
    df = harmonise_columns(pd.read_csv(path))

    # Canonicalise material names
    if "Material" in df.columns:
        mat_map = build_material_map()
        df["Material"] = (
            df["Material"]
            .astype(str)
            .str.strip()
            .map(lambda m: mat_map.get(m, m))
        )

    # Drop rows missing the target
    df = df.dropna(subset=[target])

    # Label-encode Material
    if encode_material and "Material" in df.columns:
        le = LabelEncoder()
        df["Material_encoded"] = le.fit_transform(df["Material"].astype(str))

    return df


def get_features_and_target(
    df: pd.DataFrame,
    target: str = "HER",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y.

    Uses all numeric features plus the encoded Material column.
    Drops rows with any NaN in X.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared DataFrame.
    target : str
        Target column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) pair with no NaN values.
    """
    # Build feature list
    feature_cols = [
        c for c in NUMERIC_FEATURES
        if c != target and c in df.columns
    ]
    # Use the validation-safe roughness column name if present
    for alt in ["Roughness (um)", "Roughness (µm)"]:
        if alt in df.columns and alt not in feature_cols and alt != target:
            if "Roughness (um)" not in feature_cols and "Roughness (µm)" not in feature_cols:
                feature_cols.append(alt)

    if "Material_encoded" in df.columns:
        feature_cols.append("Material_encoded")

    available = [c for c in feature_cols if c in df.columns]
    subset = df[available + [target]].dropna()

    X = subset[available]
    y = subset[target]
    return X, y


# ═══════════════════════════════════════════════════════════════════════
#  MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def get_models() -> Dict[str, Any]:
    """Return a dictionary of model instances to benchmark."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
        ),
        "Ridge Regression": Ridge(
            alpha=1.0,
            random_state=RANDOM_SEED,
        ),
    }


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Train a model and return test-set metrics.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model.
    X_train, y_train : np.ndarray
        Training data.
    X_test, y_test : np.ndarray
        Test data (always real-only).

    Returns
    -------
    Dict[str, float]
        R², MAE, RMSE on the test set.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return {
        "R²": round(float(r2_score(y_test, y_pred)), 4),
        "MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
    }


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> Dict[str, float]:
    """Run k-fold cross-validation and return mean ± std R².

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model.
    X, y : np.ndarray
        Full dataset for CV.
    n_folds : int
        Number of folds.

    Returns
    -------
    Dict[str, float]
        Mean and std of R² across folds.
    """
    n_folds = min(n_folds, len(y))
    if n_folds < 2:
        return {"cv_r2_mean": float("nan"), "cv_r2_std": float("nan")}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    return {
        "cv_r2_mean": round(float(scores.mean()), 4),
        "cv_r2_std": round(float(scores.std()), 4),
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN BENCHMARK
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark(
    real_path: str = "SPIHF_Data.csv",
    synth_path: str = "synthetic_SPIHF.csv",
    target: str = "HER",
    test_frac: float = 0.25,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Run the full benchmark comparing Real-only vs Augmented training.

    Parameters
    ----------
    real_path : str
        Path to the original experimental CSV.
    synth_path : str
        Path to the synthetic CSV.
    target : str
        Target variable for prediction.
    test_frac : float
        Fraction of real data to hold out for testing.
    output_dir : str
        Directory to save the benchmark report.

    Returns
    -------
    Dict[str, Any]
        Nested results dictionary.
    """
    np.random.seed(RANDOM_SEED)

    w = 76
    print("=" * w)
    print("  SPIHF DOWNSTREAM ML BENCHMARK")
    print("=" * w)
    print(f"  Target variable : {target}")
    print(f"  Test fraction   : {test_frac}")
    print(f"  Random seed     : {RANDOM_SEED}")
    print("=" * w + "\n")

    # ── Load data ──────────────────────────────────────────────────
    print("[1/4] Loading datasets ...")
    real_df = load_and_prepare(real_path, target)
    synth_df = load_and_prepare(synth_path, target)

    X_real, y_real = get_features_and_target(real_df, target)
    X_synth, y_synth = get_features_and_target(synth_df, target)

    print(f"  Real  : {len(X_real)} samples × {X_real.shape[1]} features")
    print(f"  Synth : {len(X_synth)} samples × {X_synth.shape[1]} features")

    # Align feature columns
    common_cols = sorted(set(X_real.columns) & set(X_synth.columns))
    X_real = X_real[common_cols]
    X_synth = X_synth[common_cols]
    print(f"  Common features : {len(common_cols)}")

    # ── Train/test split (stratified by index, test = real only) ──
    print("\n[2/4] Splitting data ...")
    n_test = max(int(len(X_real) * test_frac), 10)
    indices = np.random.permutation(len(X_real))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_test = X_real.iloc[test_idx].values
    y_test = y_real.iloc[test_idx].values

    X_train_real = X_real.iloc[train_idx].values
    y_train_real = y_real.iloc[train_idx].values

    X_train_augmented = np.vstack([X_train_real, X_synth.values])
    y_train_augmented = np.concatenate([y_train_real, y_synth.values])

    X_train_synth_only = X_synth.values
    y_train_synth_only = y_synth.values

    print(f"  Test set (real only)     : {len(X_test)} samples")
    print(f"  Train — Real-only       : {len(X_train_real)} samples")
    print(f"  Train — Augmented       : {len(X_train_augmented)} samples")
    print(f"  Train — Synthetic-only  : {len(X_train_synth_only)} samples")

    # ── Scale features ──────────────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train_real)  # fit on real data only to avoid data leakage

    X_test_s = scaler.transform(X_test)
    X_train_real_s = scaler.transform(X_train_real)
    X_train_aug_s = scaler.transform(X_train_augmented)
    X_train_syn_s = scaler.transform(X_train_synth_only)

    # ── Benchmark ──────────────────────────────────────────────────
    print("\n[3/4] Training and evaluating models ...\n")
    regimes = {
        "Real-only": (X_train_real_s, y_train_real),
        "Augmented (Real + Synthetic)": (X_train_aug_s, y_train_augmented),
        "Synthetic-only": (X_train_syn_s, y_train_synth_only),
    }

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for regime_name, (X_tr, y_tr) in regimes.items():
        print(f"  -- {regime_name} ({len(X_tr)} samples) --")
        regime_results: Dict[str, Dict[str, float]] = {}

        for model_name, model in get_models().items():
            metrics = evaluate_model(model, X_tr, y_tr, X_test_s, y_test)

            # Also do CV on the training set
            from sklearn.base import clone
            cv_model = clone(model)
            cv_metrics = cross_validate_model(cv_model, X_tr, y_tr)
            metrics.update(cv_metrics)

            regime_results[model_name] = metrics
            summary_rows.append({
                "Regime": regime_name,
                "Model": model_name,
                **metrics,
            })
            print(f"    {model_name:25s}  R²={metrics['R²']:+.4f}  "
                  f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
                  f"CV-R²={metrics['cv_r2_mean']:.4f}±{metrics['cv_r2_std']:.4f}")

        all_results[regime_name] = regime_results
        print()

    # ── Generate report ────────────────────────────────────────────
    print("[4/4] Generating benchmark report ...\n")
    report = _generate_report(
        all_results, summary_rows, len(X_test),
        len(X_train_real), len(X_train_augmented),
        len(X_train_synth_only), target, common_cols,
    )

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * w)
    print("  BENCHMARK COMPLETE")
    print("=" * w)

    # Find best regime for Random Forest (the primary model)
    rf_results = {
        regime: results.get("Random Forest", {}).get("R²", -999)
        for regime, results in all_results.items()
    }
    best_regime = max(rf_results, key=rf_results.get)
    improvement = (
        rf_results.get("Augmented (Real + Synthetic)", 0)
        - rf_results.get("Real-only", 0)
    )
    print(f"  Best regime (RF R²) : {best_regime}")
    print(f"  Augmentation Delta R2 : {improvement:+.4f}")
    if improvement > 0:
        print("  Verdict               : Synthetic data IMPROVES prediction [OK]")
    elif improvement == 0:
        print("  Verdict             : No change (neutral)")
    else:
        print("  Verdict             : Synthetic data did not improve prediction")
        print("                        (may need hyperparameter tuning or "
              "augmentation refinement)")
    print("=" * w)

    return all_results


def _generate_report(
    results: Dict[str, Dict[str, Dict[str, float]]],
    rows: List[Dict[str, Any]],
    n_test: int,
    n_train_real: int,
    n_train_aug: int,
    n_train_syn: int,
    target: str,
    features: List[str],
) -> str:
    """Generate a Markdown benchmark report.

    Parameters
    ----------
    results : Dict
        Nested results from benchmark.
    rows : List[Dict]
        Flat summary rows for the comparison table.
    n_test, n_train_real, n_train_aug, n_train_syn : int
        Sample counts per regime.
    target : str
        Target variable.
    features : List[str]
        Feature names used.

    Returns
    -------
    str
        Markdown report text.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []

    lines.append("# SPIHF Downstream ML Benchmark Report\n")
    lines.append(f"> **Auto-generated** on {timestamp} by `benchmark_model.py`\n")
    lines.append("---\n")

    # Experiment setup
    lines.append("## 1. Experiment Setup\n")
    lines.append(f"- **Target variable:** {target}")
    lines.append(f"- **Features used:** {len(features)}")
    lines.append(f"- **Random seed:** {RANDOM_SEED}")
    lines.append(f"- **Test set (real only):** {n_test} samples")
    lines.append(f"- **Train — Real-only:** {n_train_real} samples")
    lines.append(f"- **Train — Augmented:** {n_train_aug} samples")
    lines.append(f"- **Train — Synthetic-only:** {n_train_syn} samples\n")

    lines.append("### Feature List\n")
    for i, f in enumerate(sorted(features), 1):
        lines.append(f"{i}. {f}")
    lines.append("")

    # Results table
    lines.append("## 2. Results Summary\n")
    lines.append("| Regime | Model | R² | MAE | RMSE | CV R² (mean±std) |")
    lines.append("|--------|-------|----|-----|------|-----------------|")
    for row in rows:
        cv_str = f"{row.get('cv_r2_mean', 0):.4f}±{row.get('cv_r2_std', 0):.4f}"
        lines.append(
            f"| {row['Regime']} | {row['Model']} | "
            f"{row['R²']:+.4f} | {row['MAE']:.4f} | {row['RMSE']:.4f} | "
            f"{cv_str} |"
        )
    lines.append("")

    # Analysis
    lines.append("## 3. Analysis\n")

    rf_real = results.get("Real-only", {}).get("Random Forest", {})
    rf_aug = results.get("Augmented (Real + Synthetic)", {}).get("Random Forest", {})
    rf_syn = results.get("Synthetic-only", {}).get("Random Forest", {})

    r2_real = rf_real.get("R²", 0)
    r2_aug = rf_aug.get("R²", 0)
    r2_syn = rf_syn.get("R²", 0)
    delta = r2_aug - r2_real

    lines.append("### 3.1 Augmentation Impact (Random Forest)\n")
    lines.append(f"- **Real-only R²:** {r2_real:+.4f}")
    lines.append(f"- **Augmented R²:** {r2_aug:+.4f}")
    lines.append(f"- **Δ R²:** {delta:+.4f}\n")

    if delta > 0.02:
        lines.append(
            "**Conclusion:** Synthetic data provides a **meaningful improvement** "
            "in predictive performance. The augmented training set helps the model "
            "generalise better to unseen real data.\n"
        )
    elif delta > -0.02:
        lines.append(
            "**Conclusion:** Augmentation has a **neutral effect** on performance. "
            "The synthetic data neither helps nor hurts, suggesting it faithfully "
            "reproduces the statistical structure without adding new information.\n"
        )
    else:
        lines.append(
            "**Conclusion:** Augmentation **decreased** performance. This may "
            "indicate that the synthetic data introduced noise or distribution "
            "shift. Consider tightening confidence thresholds or noise levels.\n"
        )

    lines.append("### 3.2 Synthetic-Only Sanity Check\n")
    lines.append(
        f"Training on synthetic data alone yields R² = {r2_syn:+.4f}. "
    )
    if r2_syn > 0:
        lines.append(
            "The synthetic data captures enough signal to produce "
            "positive predictions on real test data, confirming that the "
            "augmentation pipeline preserves meaningful structure.\n"
        )
    else:
        lines.append(
            "Poor synthetic-only performance suggests the model cannot "
            "generalise from synthetic to real data without anchoring on "
            "real observations.\n"
        )

    # Recommendations
    lines.append("## 4. Recommendations\n")
    lines.append("1. **Always use real data in the training set** — synthetic "
                 "data is a supplement, not a replacement.")
    lines.append("2. **Hold out real data for testing** — never test on "
                 "synthetic data when evaluating real-world performance.")
    lines.append("3. **Monitor for overfitting** — cross-validation R² should "
                 "be close to test R²; large gaps indicate memorisation.")
    lines.append("4. **Iterate on augmentation parameters** — if augmented "
                 "performance is lower, try reducing noise level or "
                 "increasing confidence threshold.\n")

    lines.append("---\n")
    lines.append(f"*Report generated by `benchmark_model.py` on {timestamp}*\n")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SPIHF Downstream ML Benchmark: "
                    "does synthetic data improve HER prediction?",
    )
    parser.add_argument(
        "--real", default="SPIHF_Data.csv",
        help="Path to the original experimental CSV (default: SPIHF_Data.csv)",
    )
    parser.add_argument(
        "--synth", default="synthetic_SPIHF.csv",
        help="Path to the synthetic CSV (default: synthetic_SPIHF.csv)",
    )
    parser.add_argument(
        "--target", default="HER",
        help="Target variable for prediction (default: HER)",
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.25,
        help="Fraction of real data to hold out for testing (default: 0.25)",
    )
    parser.add_argument(
        "--output", default="outputs",
        help="Output directory for benchmark report (default: outputs)",
    )
    args = parser.parse_args()

    run_benchmark(
        real_path=args.real,
        synth_path=args.synth,
        target=args.target,
        test_frac=args.test_frac,
        output_dir=args.output,
    )
