# SPIHF Synthetic Data Augmentation Pipeline

**Physics-Informed Synthetic Data Augmentation for Single Point Incremental Hole Flanging (SPIHF)**

A modular Python package that generates physically plausible synthetic manufacturing data by combining SMOTE-inspired interpolation with metallurgical and mechanical engineering constraints.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python -m spihf_synthetic.main

# 3. Run with custom parameters
python -m spihf_synthetic.main --data SPIHF_Data.csv --output outputs --samples 1000 --seed 42

# 4. Run the downstream ML benchmark
python benchmark_model.py
```

---

## Project Structure

```
forming 2/
├── SPIHF_Data.csv                  # Original experimental dataset (304 samples)
├── requirements.txt                # Python dependencies
├── benchmark_model.py              # Downstream ML benchmark (real-only vs augmented)
│
├── spihf_synthetic/                # Core Python package
│   ├── __init__.py
│   ├── config.py                   # Central configuration & physics limits
│   ├── main.py                     # Pipeline orchestrator (7-stage)
│   ├── augmentation.py             # SMOTE interpolation + Gaussian perturbation
│   ├── constraints.py              # Physics-informed constraint repair
│   ├── confidence.py               # Multi-component confidence scoring
│   ├── validation.py               # Statistical validation (KS, Wasserstein, JSD)
│   ├── visualization.py            # Publication-quality matplotlib figures
│   ├── reporting.py                # Auto-generated Markdown reports
│   └── utils.py                    # Shared helpers
│
├── tests/                          # Unit tests (pytest)
│   ├── conftest.py                 # Shared fixtures
│   ├── test_constraints.py
│   ├── test_augmentation.py
│   ├── test_confidence.py
│   └── test_utils.py
│
└── outputs/                        # Generated outputs (after running pipeline)
    ├── synthetic_SPIHF.csv
    ├── reports/
    │   ├── engineering_report.md
    │   ├── validation_report.md
    │   ├── methodology_report.md
    │   └── validation_metrics.json
    └── figures/
        ├── distribution_comparison.png
        ├── correlation_heatmap.png
        ├── pca_comparison.png
        └── ...
```

---

## Pipeline Overview

The augmentation pipeline operates in **7 stages**:

| Stage | Description |
|:-----:|-------------|
| 1 | **Load & Preprocess** — Ingest raw CSV, harmonise column names, canonicalise 68 material aliases into 24 groups |
| 2 | **Feature Statistics** — Compute per-material descriptive statistics (mean, std, min, max) |
| 3 | **Synthetic Generation** — SMOTE-inspired convex interpolation within material groups + calibrated Gaussian noise (1–3% of σ) |
| 4 | **Save CSV** — Write synthetic dataset with confidence scores |
| 5 | **Statistical Validation** — KS tests, Wasserstein distance, Jensen-Shannon divergence, correlation preservation |
| 6 | **Visualisation** — Distribution histograms, box/violin plots, correlation heatmaps, PCA, t-SNE |
| 7 | **Reports** — Auto-generated engineering, validation, and methodology Markdown reports |

### Physics-Informed Constraints

Every synthetic sample is passed through a deterministic repair layer:

- **Strength:** UTS ≥ YS, Hollomon consistency (k ≥ UTS), 0.01 ≤ n ≤ 1.0
- **Geometry:** Thickness > 0, sine-law thinning bound, min thickness ≤ initial thickness
- **Process:** Feed rate ≥ 1 mm/min, stages ∈ ℤ⁺, lubricant ∈ {0, 1}
- **Formability:** R-value ∈ [0.1, 5.0], elongation ∈ [0.5%, 99%]

---

## Validation

The synthetic dataset is validated using:

- **Kolmogorov-Smirnov tests** (distributional equivalence)
- **Wasserstein / Earth Mover's Distance** (distribution similarity)
- **Jensen-Shannon Divergence** (information-theoretic comparison)
- **Correlation matrix Frobenius norm** (multivariate structure)
- **Physics correlation pairs** (domain-specific sign preservation)
- **Feature importance ranking** (Random Forest + Mutual Information Spearman ρ)

Results are aggregated into a composite score (0–100) with letter grades (A–F).

---

## Downstream ML Benchmark

`benchmark_model.py` trains Random Forest, Gradient Boosting, and Ridge Regression models to predict HER (Hole Expansion Ratio) under three regimes:

1. **Real-only** — Train on original 304 samples
2. **Augmented** — Train on real + synthetic data
3. **Synthetic-only** — Train on synthetic data alone

The test set is **always real data only**, ensuring honest evaluation. Results include R², MAE, RMSE, and cross-validated scores.

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=spihf_synthetic
```

---

## Dependencies

- Python ≥ 3.9
- NumPy, Pandas, SciPy, scikit-learn, Matplotlib
- Optional: `umap-learn` (for UMAP visualisation), `xgboost` (for benchmark)

See `requirements.txt` for pinned versions.

---

## Dataset

The original SPIHF dataset (`SPIHF_Data.csv`) contains **304 experimental observations** compiled from peer-reviewed journal articles spanning:

- **Materials:** Aluminium alloys (1050, 5052, 6061, 7075), steels (DC01, DC04, DC05, DDQ), titanium, copper, stainless steel
- **Features:** 20 columns covering material properties, process parameters, and forming outcomes
- **Target variable:** HER (Hole Expansion Ratio)

---

## Citation

If you use this pipeline or dataset in your research, please cite the original SPIHF studies from which the experimental data was compiled.

---

## License

This project is developed for academic research purposes.
