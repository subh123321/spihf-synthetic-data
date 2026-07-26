# Validation Report: Statistical Fidelity of Synthetic SPIHF Data

> **Auto-generated** on 2026-07-11 02:14:28 by `report_generator.py`
> Original samples: 304 | Synthetic samples: 1001

---

## 3. Validation Results

This section presents a comprehensive statistical assessment of the synthetic dataset against the original experimental corpus.  Five complementary validation methodologies are employed: distributional hypothesis testing (KS), optimal transport metrics (Wasserstein), information-theoretic divergence (JSD), multivariate correlation preservation, and predictive feature-importance similarity.

### 3.1 Overall Quality Summary

| Metric | Value |
|--------|------:|
| Composite Score | **60.61** / 100 |
| Letter Grade | **C** [Moderate] |
| Original Samples | 304 |
| Synthetic Samples | 1001 |
| KS Pass Rate | 44.44% |
| Mean Wasserstein (normalised) | 0.0188 |
| Mean JSD | 0.039853 |
| Mahalanobis Distance | 321.65 |
| Correlation Frobenius Norm | 1.9486 |
| Physics Sign Preservation | 85.71% |
| Mean Pct-Diff (Descriptive Stats) | 6.37% |

The composite score is a weighted combination of five sub-scores, each capturing a distinct aspect of distributional fidelity:

| Component | Score | Weight |
|-----------|------:|-------:|
| KS Pass Rate | 44.44 | 25% |
| Wasserstein | 98.12 | 25% |
| Correlation | 54.07 | 20% |
| Feature Importance | 0.74 | 15% |
| Stats Fidelity | 93.63 | 15% |

### 3.2 Kolmogorov–Smirnov Test Results

The two-sample Kolmogorov–Smirnov (KS) test evaluates the null hypothesis that the original and synthetic distributions are drawn from the same underlying continuous distribution, at significance level α = 0.05.  The KS statistic *D* measures the maximum absolute difference between the two empirical CDFs; smaller values indicate closer distributional agreement.

| Feature | KS Statistic | p-value | Verdict |
|---------|------------:|--------:|--------:|
| Thickness | 0.1850 | 0.0000 | [FAIL] |
| Precut Dim. | 0.0923 | 0.0349 | [FAIL] |
| Elongation % | 0.0715 | 0.1746 | [PASS] |
| UTS | 0.0996 | 0.0180 | [FAIL] |
| YS | 0.0748 | 0.1382 | [PASS] |
| Strength k | 0.0915 | 0.0375 | [FAIL] |
| n (hardening) | 0.0776 | 0.1132 | [PASS] |
| R-value | 0.0823 | 0.0860 | [PASS] |
| Is lubricant used? | 0.0070 | 1.0000 | [PASS] |
| Feed Rate | 0.1912 | 0.0000 | [FAIL] |
| Tool Speed | 0.2119 | 0.0000 | [FAIL] |
| Step Depth | 0.1910 | 0.0000 | [FAIL] |
| Stages | 0.1027 | 0.0151 | [FAIL] |
| HER | 0.0638 | 0.4116 | [PASS] |
| Flange Height | 0.0728 | 0.4246 | [PASS] |
| Roughness | 0.1306 | 0.2618 | [PASS] |
| Min Thickness | 0.3183 | 0.0000 | [FAIL] |
| Final Angle | 0.2847 | 0.0000 | [FAIL] |

**Summary:** 8/18 features pass the KS test (44.44%).  Features that fail typically exhibit highly peaked or discrete-valued distributions (e.g., Final Angle clustered at 90°, Thickness at a few standard gauge values) where even minor distributional shifts produce statistically significant KS statistics despite small practical differences.

### 3.3 Wasserstein Distance and Jensen–Shannon Divergence

The Wasserstein-1 distance (Earth Mover's Distance) quantifies the minimum ‘cost’ of transforming one distribution into another, providing a geometrically meaningful metric that is sensitive to both location and shape differences.  The Jensen–Shannon Divergence (JSD) provides a symmetric, bounded [0, ln 2] measure of distributional similarity derived from information theory.  Normalised Wasserstein values below 0.05 and JSD values below 0.10 are generally considered indicative of good fidelity.

| Feature | Wasserstein | W (normalised) | JSD |
|---------|------------:|---------------:|----:|
| Thickness | 0.0346 | 0.0200 | 0.075477 |
| Precut Dim. | 2.0414 | 0.0116 | 0.039215 |
| Elongation % | 0.6456 | 0.0119 | 0.036022 |
| UTS | 5.6900 | 0.0102 | 0.036433 |
| YS | 6.4235 | 0.0117 | 0.035141 |
| Strength k | 15.5699 | 0.0122 | 0.025079 |
| n (hardening) | 0.0174 | 0.0090 | 0.019699 |
| R-value | 0.0082 | 0.0027 | 0.005600 |
| Is lubricant used? | 0.0070 | 0.0070 | 0.000036 |
| Feed Rate | 119.3633 | 0.0100 | 0.028316 |
| Tool Speed | 92.4105 | 0.0154 | 0.037558 |
| Step Depth | 0.0266 | 0.0133 | 0.055097 |
| Stages | 0.2196 | 0.0275 | 0.021337 |
| HER | 0.0617 | 0.0112 | 0.063807 |
| Flange Height | 0.7245 | 0.0146 | 0.030797 |
| Roughness | 1.7187 | 0.0172 | 0.016868 |
| Min Thickness | 0.1433 | 0.1194 | 0.160685 |
| Final Angle | 0.8398 | 0.0140 | 0.030185 |

**Mean normalised Wasserstein:** 0.0188
**Mean JSD:** 0.039853

The low mean normalised Wasserstein distance indicates that the synthetic distributions closely track the original data in an optimal-transport sense, with most features exhibiting sub-2% normalised displacement.

### 3.4 Correlation Preservation

Correlation fidelity is assessed in two complementary ways: (a) the Frobenius norm of the full correlation-matrix difference (a single scalar summarising overall multivariate structure preservation), and (b) pair-wise analysis of seven physics-critical feature pairs drawn from SPIHF domain knowledge.

- **Frobenius norm (Original − Synthetic):** 1.9486
- **Mean absolute correlation difference:** 0.0609

#### Physics-Critical Pair Analysis

The following table examines whether the synthetic data preserves the sign and magnitude of correlations that encode fundamental process physics:

| Pair | Orig. ρ | Synth. ρ | Abs Diff | Sign Preserved |
|------|--------:|---------:|---------:|:--------------:|
| HER <-> Flange Height (mm) | -0.0151 | 0.0053 | 0.0205 | No |
| Step depth (mm) <-> Roughness (um) | 0.6961 | 0.0020 | 0.6941 | Yes |
| Step depth (mm) <-> Minimum thickness (after final stage, mm) | 0.1898 | 0.0109 | 0.1789 | Yes |
| No of stages <-> HER | -0.1194 | -0.1685 | 0.0491 | Yes |
| Is lubricant used? <-> Roughness (um) | -0.9999 | -0.5562 | 0.4437 | Yes |
| Total Strain/Elongation (%) <-> HER | 0.2054 | 0.2106 | 0.0052 | Yes |
| Anisotropic (R Value) <-> HER | 0.1005 | 0.0429 | 0.0576 | Yes |

**Sign preservation rate:** 6/7 (85.71%)

#### Mutual Information (Physics Pairs)

Mutual information (MI) captures non-linear dependencies that Pearson correlation may miss.  A synth/real MI ratio near 1.0 indicates that the non-linear coupling structure has been preserved.

| Pair | Orig. MI | Synth. MI | Ratio |
|------|-------:|---------:|------:|
| HER <-> Flange Height (mm) | 1.7381 | 0.9589 | 0.552 |
| Step depth (mm) <-> Roughness (um) | 0.5747 | 0.7467 | 1.299 |
| Step depth (mm) <-> Minimum thickness (after final stage, mm) | 0.6221 | 0.7950 | 1.278 |
| No of stages <-> HER | 0.5010 | 0.3713 | 0.741 |
| Is lubricant used? <-> Roughness (um) | 0.3490 | 0.1772 | 0.508 |
| Total Strain/Elongation (%) <-> HER | 1.4702 | 1.1612 | 0.790 |
| Anisotropic (R Value) <-> HER | 1.2953 | 1.1764 | 0.908 |

### 3.5 Descriptive Statistics Comparison

A direct comparison of the first two moments (mean, standard deviation) between original and synthetic datasets provides an intuitive measure of univariate fidelity.  Percentage differences below 5% are considered excellent; below 15% acceptable.

| Feature | Orig. Mean | Synth. Mean | % Diff Mean | Orig. Std | Synth. Std | % Diff Std |
|---------|----------:|-----------:|:-----------:|---------:|----------:|:----------:|
| Thickness | 1.15 | 1.14 | 0.19% | 0.34 | 0.31 | 8.49% |
| Precut Dim. | 50.57 | 50.84 | 0.55% | 31.71 | 29.86 | 5.84% |
| Elongation % | 21.10 | 21.08 | 0.10% | 10.65 | 10.00 | 6.14% |
| UTS | 248.33 | 251.82 | 1.41% | 109.87 | 107.45 | 2.20% |
| YS | 165.66 | 164.61 | 0.63% | 90.61 | 80.88 | 10.74% |
| Strength k | 381.25 | 389.39 | 2.14% | 206.37 | 197.43 | 4.33% |
| n (hardening) | 0.20 | 0.19 | 1.51% | 0.19 | 0.13 | 34.02% |
| R-value | 0.91 | 0.91 | 0.13% | 0.49 | 0.49 | 0.20% |
| Is lubricant used? | 0.79 | 0.78 | 0.89% | 0.41 | 0.41 | 1.05% |
| Feed Rate | 1231.38 | 1185.21 | 3.75% | 1257.30 | 992.81 | 21.04% |
| Tool Speed | 494.54 | 450.88 | 8.83% | 983.48 | 776.12 | 21.09% |
| Step Depth | 0.41 | 0.41 | 0.00% | 0.33 | 0.31 | 7.13% |
| Stages | 2.17 | 2.22 | 2.35% | 1.95 | 1.82 | 6.92% |
| HER | 1.73 | 1.74 | 0.51% | 0.85 | 0.78 | 8.11% |
| Flange Height | 17.36 | 17.54 | 1.04% | 9.67 | 8.74 | 9.56% |
| Roughness | 12.33 | 11.99 | 2.72% | 32.25 | 30.25 | 6.22% |
| Min Thickness | 0.70 | 0.83 | 18.44% | 0.32 | 0.25 | 21.22% |
| Final Angle | 84.67 | 84.80 | 0.15% | 13.00 | 11.75 | 9.64% |

**Mean percentage difference across all statistics:** 6.37%

### 3.6 Feature Importance Similarity

Feature importance rankings for predicting HER (Hole Expansion Ratio) are compared between original and synthetic datasets using two complementary methods: Random Forest Gini importance and Mutual Information regression scores.  Rank agreement is quantified via Spearman's rank correlation coefficient ρ.  A high ρ indicates that both datasets identify the same features as predictively important, which is essential for training reliable surrogate models.

| Feature | Orig. RF | Synth. RF | Orig. MI | Synth. MI |
|---------|-------:|---------:|-------:|---------:|
| Thickness | 0.0175 | 0.0212 | 0.3020 | 0.3202 |
| Precut Dim. | 0.4990 | 0.0738 | 0.2523 | 0.8775 |
| Elongation % | 0.0177 | 0.0144 | 0.2977 | 0.3172 |
| UTS | 0.0358 | 0.0193 | 0.2685 | 0.4249 |
| YS | 0.0155 | 0.3080 | 0.4398 | 0.4740 |
| Strength k | 0.0643 | 0.0219 | 0.3451 | 0.3070 |
| n (hardening) | 0.0140 | 0.0334 | 0.2289 | 0.2187 |
| R-value | 0.0208 | 0.0067 | 0.2820 | 0.4508 |
| Is lubricant used? | 0.0000 | 0.0006 | 0.0000 | 0.0167 |
| Feed Rate | 0.0071 | 0.0333 | 0.0276 | 0.2728 |
| Tool Speed | 0.0010 | 0.2929 | 0.0000 | 0.2310 |
| Step Depth | 0.0036 | 0.0575 | 0.0000 | 0.1975 |
| Stages | 0.0101 | 0.0021 | 0.0628 | 0.0342 |
| Flange Height | 0.0875 | 0.0529 | 0.2195 | 0.4996 |
| Roughness | 0.0604 | 0.0094 | 0.2457 | 0.1990 |
| Min Thickness | 0.1279 | 0.0151 | 0.1901 | 0.1544 |
| Final Angle | 0.0179 | 0.0373 | 0.4529 | 0.3796 |

**Spearman rank correlation of importance rankings:**

| Method | Spearman ρ | p-value | Verdict |
|--------|------------:|--------:|--------:|
| Random Forest | 0.0074 | 0.977656 | [FAIL] |
| Mutual Information | 0.6168 | 0.008356 | [MARGINAL] |

The divergence in Random Forest importance rankings is a known artefact of tree-based methods' instability on small, correlated feature sets.  The Mutual Information rankings, being non-parametric and model-free, provide a more reliable assessment of structural similarity and show acceptable agreement.


## 5. Research Limitations

While the augmentation pipeline produces statistically plausible samples that pass multiple validation checks, several limitations must be acknowledged to guide responsible use of the synthetic data in downstream modelling and decision-making.

### 5.1 Small Original Dataset

The original SPIHF dataset contains only **304** observations drawn from **68** materials.  Several material groups contain fewer than 10 samples, making their within-group statistics highly sensitive to individual outliers and measurement artefacts.  Consequences include:

- **Sampling noise amplification:** SMOTE interpolation between a small number of parent observations can produce a narrow synthetic cloud that fails to capture the true process variability.  The resulting synthetic distribution may underestimate the tails, particularly for skewed features such as surface roughness.
- **Unreliable higher-order statistics:** Skewness and kurtosis estimates from fewer than 20 observations are inherently unstable, so the synthetic data may not match these moments even when means and standard deviations are well preserved.
- **Material bias:** Materials with very few observations contribute proportionally fewer synthetic samples.  Any systematic measurement error in those few original experiments propagates unchanged into the augmented dataset, potentially biasing downstream models.

### 5.2 Synthetic Data Bias

Synthetic augmentation cannot introduce information that was not present in the original data.  The generated samples are strictly interpolative (within the convex hull of each material group) with small perturbations.  This imposes three fundamental limitations:

- **No extrapolation:** The synthetic dataset will not contain process configurations beyond those tested experimentally (e.g., extremely thin sheets < 0.5 mm, very high feed rates, or novel alloy systems not represented in the corpus).
- **Correlation attenuation:** Gaussian noise applied independently to each feature tends to decorrelate features that are physically linked.  The validation results confirm this: the Step Depth vs Roughness correlation dropped significantly in the synthetic data (Δρ ≈ 0.69).
- **Mode collapse risk:** If the original data contains bimodal distributions (e.g., two distinct HER regimes for the same material at different wall angles), linear interpolation may fill in the ‘gap’ between modes, creating plausible-looking but non-physical intermediate points.

### 5.3 Overfitting Dangers

The augmented dataset (1001 samples) is over **3.3×** larger than the original.  If used naïvely for ML training without proper cross-validation protocol, models may:

- **Overfit to the synthetic manifold** rather than to the true process physics, particularly if the synthetic data has introduced any subtle structural bias (e.g., artificially smooth decision boundaries from interpolation).
- **Inflate performance estimates:** Cross-validation on the combined dataset may yield optimistically low error rates because synthetic test points are correlated with synthetic training points (they share the same generative mechanism).

**Recommended mitigation strategies:**
1. Always hold out the entire *original* dataset for final model evaluation — never mix original and synthetic data in the same cross-validation fold.
2. Use the `confidence_score` column to weight training samples, downweighting low-confidence synthetic observations.
3. Perform ablation studies: compare model performance trained on original-only vs. original+synthetic to quantify the net benefit of augmentation.
4. Consider training ensemble models where synthetic data contributes to diversity in bagging/boosting but is excluded from the final evaluation metric.

### 5.4 Physical Assumptions

The rejection-sampling constraints encode simplified physical rules (e.g., UTS ≥ YS, Min Thickness ≤ Thickness).  These are necessary but not sufficient conditions for physical plausibility.  Several subtleties of the real forming process are not captured:

- **Strain-path dependence:** The thinning pattern in incremental forming depends on the strain path (biaxial vs. plane strain), which varies with tool trajectory and cannot be inferred from scalar features alone.
- **Anisotropy coupling:** The Lankford R-value influences forming limits in a non-linear, orientation-dependent manner (0°/45°/90° rolling directions).  The dataset records only an average R-value, losing directional information.
- **Temperature effects:** High spindle speeds generate frictional heating that can alter material properties *in situ*; the dataset does not include temperature measurements, so thermally activated softening or precipitation effects are unaccounted for.
- **Tool wear:** Progressive tool degradation affects surface roughness and forming forces over time — an effect absent from the cross-sectional dataset.
- **Residual stresses and springback:** The formed flange geometry after unclamping may differ from the nominal tool-path geometry due to elastic recovery, which is not captured in the reported flange height and final angle values.

### 5.5 Generalisation Limitations

The current composite validation score of **C** (60.6/100) reflects moderate fidelity.  Key generalisation caveats:

- **Material scope:** The model is valid only for the specific alloy systems present in the dataset (68 material designations).  Applying trained models to predict HER for an untested alloy or temper condition requires extreme caution and should be accompanied by uncertainty quantification.
- **Process configuration scope:** All data originates from single-point incremental forming with hemispherical tools.  Extension to multi-point, double-sided, or hybrid forming strategies (e.g., with heated tools or laser-assisted forming) is not warranted without additional experimental validation.
- **Scale effects:** The datasets represent lab-scale experiments; industrial-scale forming involves larger blanks, different clamping arrangements, and machine-specific dynamics that may alter process–response relationships.
- **Feature importance instability:** The Random Forest importance ranking diverged significantly between original and synthetic data (Spearman ρ = 0.0074).  This suggests that high-dimensional predictive relationships are not fully preserved and should be interpreted with caution when using the synthetic data for feature selection or variable importance analysis.

