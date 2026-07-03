# Methodology Report: Synthetic Data Generation for SPIHF

> **Auto-generated** on 2026-07-03 13:02:13 by `report_generator.py`
> Original samples: 304 | Synthetic samples: 1001

---

## 2. Synthetic Data Generation Methodology

### 2.1 Overview

The SPIHF experimental dataset comprises **304** observations collected from published literature spanning **68** distinct sheet-metal alloys.  To enable robust machine-learning modelling while preserving the physics of the incremental hole-flanging process, a synthetic augmentation pipeline was designed to generate **1001** scientifically plausible samples.

The pipeline employs a five-stage strategy: (i) material-wise stratification, (ii) SMOTE-inspired interpolation, (iii) Gaussian perturbation with feature-aware noise, (iv) physics-informed rejection sampling, and (v) confidence scoring.  Each stage is described below.

### 2.2 Material-Wise Stratified Generation

Synthetic samples are generated **independently within each material group**.  This is critical because material properties (UTS, YS, strain-hardening exponent *n*, Lankford R-value) are intrinsically coupled, and interpolating across dissimilar alloys (e.g., between AA1050 aluminium and DP590 dual-phase steel) would produce non-physical property combinations.  The number of synthetic samples generated per material is proportional to the material's representation in the original dataset, ensuring that minority materials are not under-represented in the augmented corpus.

### 2.3 SMOTE-Inspired Interpolation

For each material group, pairs of real observations *(x_i, x_j)* are sampled and linearly interpolated:

```
x_new = alpha * x_i + (1 - alpha) * x_j
alpha ~ Uniform(0.2, 0.8)
```

Restricting alpha to [0.2, 0.8] prevents the synthetic point from collapsing onto either parent observation (near-duplicate generation), while ensuring that it remains within the convex hull of the real data manifold.  This is an adaptation of the Synthetic Minority Over-sampling Technique (SMOTE) by Chawla et al. (2002), applied in a regression context rather than the original classification setting.

### 2.4 Gaussian Perturbation

After interpolation, each numeric feature is perturbed by additive Gaussian noise scaled to the within-material standard deviation:

```
x_perturbed = x_interpolated + epsilon
epsilon ~ N(0, sigma_material * noise_fraction)
noise_fraction in {0.03, 0.05, 0.08}  (feature-dependent)
```

The noise fraction is deliberately small (3-8% of within-group standard deviation) to introduce stochastic variation without distorting the underlying physical distributions.  Features with inherently tight tolerances (e.g., sheet thickness, step depth) receive lower noise fractions than response variables (e.g., surface roughness, flange height).

### 2.5 Physics-Informed Rejection Sampling

Every candidate synthetic sample is screened against a set of domain-derived constraints before acceptance.  Samples that violate any constraint are discarded and regenerated.  The constraint set includes:

| # | Constraint | Physical Rationale |
|:-:|-----------|-------------------|
| 1 | UTS >= YS | Ultimate tensile strength cannot be lower than yield strength by definition. |
| 2 | Thickness > 0 | Sheet thickness must be strictly positive. |
| 3 | HER > 0 | The hole expansion ratio is a positive geometric quantity. |
| 4 | Min Thickness <= Thickness | Thinning during forming means the minimum post-forming thickness cannot exceed the initial blank thickness. |
| 5 | 0 < n < 1 | The strain-hardening exponent is bounded between 0 (perfectly plastic) and 1 (linear hardening). |
| 6 | R-value >= 0 | Lankford's anisotropy coefficient is non-negative. |
| 7 | Step Depth > 0 | Tool step-down must be positive. |
| 8 | No. of Stages >= 1 | At least one forming pass is required. |
| 9 | Final Angle in [0, 90] | The wall angle cannot exceed 90 degrees in single-point incremental forming. |

Rejection sampling ensures that the synthetic dataset remains physically realisable, preventing any downstream model from learning from thermodynamically or mechanically impossible observations.

### 2.6 Physics Correction Layer

In addition to hard rejection constraints, a soft correction layer adjusts continuous features to improve physical plausibility.  For example, if a perturbed sample has UTS only marginally above YS, the layer widens the gap to a material-realistic minimum.  Similarly, the minimum thickness is clamped to a physically meaningful fraction of the initial thickness based on the number of forming stages.  These corrections reduce the rate of rejection while preserving the distributional shape of the features.

### 2.7 Confidence Scoring

Each accepted synthetic sample is assigned a confidence score in [0, 1] that quantifies its proximity to the real data manifold.  The score is computed as a weighted average of:

1. **Mahalanobis proximity** -- inverse of the normalised Mahalanobis distance to the centroid of the material group.
2. **Constraint margin** -- how far the sample is from the nearest rejection boundary (farther = higher confidence).
3. **Interpolation balance** -- samples with alpha closer to 0.5 (equidistant from both parents) receive a slight bonus.

The confidence score is included as a column (`confidence_score`) in the output CSV, allowing downstream consumers to weight observations or filter by quality threshold.


## 1. Dataset Summary

### 1.1 Sample Counts

| Dataset | Samples | Features |
|---------|--------:|---------:|
| Original (experimental) | 304 | 20 |
| Synthetic (augmented) | 1001 | 21 |
| **Combined** | **1305** | **21** |

### 1.2 Material Distribution

| Material | Real (n) | Real (%) | Synthetic (n) | Synthetic (%) |
|----------|--------:|--------:|--------------:|--------------:|
|  AA1060 | 2 | 0.7% | 0 | 0.0% |
| 1060 aluminum sheet | 2 | 0.7% | 0 | 0.0% |
| 2205 Dual Phase Steel | 0 | 0.0% | 13 | 1.3% |
| 2205 DualPhase Steel | 2 | 0.7% | 0 | 0.0% |
| 304 stainless steel | 2 | 0.7% | 0 | 0.0% |
| 7075-O aluminium alloy | 16 | 5.3% | 0 | 0.0% |
| 7075-O aluminum alloy | 2 | 0.7% | 0 | 0.0% |
| AA1050 | 6 | 2.0% | 79 | 7.9% |
| AA1050 Aluminium | 1 | 0.3% | 0 | 0.0% |
| AA1050-H111 | 6 | 2.0% | 82 | 8.2% |
| AA1050-H111 Aluminium | 1 | 0.3% | 0 | 0.0% |
| AA1060 | 7 | 2.3% | 86 | 8.6% |
| AA1060 aluminium alloy | 11 | 3.6% | 0 | 0.0% |
| AA5052 | 12 | 3.9% | 69 | 6.9% |
| AA5052 aluminium alloy | 4 | 1.3% | 0 | 0.0% |
| AA5052 sheet | 4 | 1.3% | 0 | 0.0% |
| AA5052-H32 | 12 | 3.9% | 79 | 7.9% |
| AA5052-H32 Aluminium | 1 | 0.3% | 0 | 0.0% |
| AA5052-H32 aluminium | 10 | 3.3% | 0 | 0.0% |
| AA6061- T6 aluminium | 10 | 3.3% | 0 | 0.0% |
| AA6061-T6 | 13 | 4.3% | 95 | 9.5% |
| AA6061-T6 (CHF) | 1 | 0.3% | 0 | 0.0% |
| AA6061-T6 (IHF) | 1 | 0.3% | 0 | 0.0% |
| AA6061-T6 aluminum alloy | 1 | 0.3% | 0 | 0.0% |
| AA6061-T7 | 2 | 0.7% | 7 | 0.7% |
| AA6061-T8 | 2 | 0.7% | 7 | 0.7% |
| AA7075-0 | 10 | 3.3% | 0 | 0.0% |
| AA7075-0 Aluminium alloy | 4 | 1.3% | 0 | 0.0% |
| AA7075-1 | 2 | 0.7% | 7 | 0.7% |
| AA7075-2 | 1 | 0.3% | 5 | 0.5% |
| AA7075-O | 11 | 3.6% | 178 | 17.8% |
| AA7075-O (SPIF Single-Stage) | 6 | 2.0% | 0 | 0.0% |
| AA7075-O Aluminium alloy | 3 | 1.0% | 0 | 0.0% |
| AA7075-O aluminium | 2 | 0.7% | 0 | 0.0% |
| AISI 304 | 1 | 0.3% | 0 | 0.0% |
| Al 1050 | 1 | 0.3% | 0 | 0.0% |
| Al 1050A | 4 | 1.3% | 0 | 0.0% |
| Al 1051 | 1 | 0.3% | 1 | 0.1% |
| Al 6061 | 3 | 1.0% | 0 | 0.0% |
| Al1060 | 2 | 0.7% | 0 | 0.0% |
| Aluminium 1000 series | 1 | 0.3% | 0 | 0.0% |
| Aluminium 1050 | 12 | 3.9% | 0 | 0.0% |
| Aluminium 5052 sheets | 1 | 0.3% | 0 | 0.0% |
| Aluminium AA1050-H111 | 16 | 5.3% | 0 | 0.0% |
| Aluminum (unspecified) | 9 | 3.0% | 36 | 3.6% |
| Aluminum Alloy | 1 | 0.3% | 0 | 0.0% |
| Copper | 4 | 1.3% | 13 | 1.3% |
| DC01 | 4 | 1.3% | 39 | 3.9% |
| DC01	 | 4 | 1.3% | 0 | 0.0% |
| DC01  | 2 | 0.7% | 0 | 0.0% |
| DC01 Steel | 2 | 0.7% | 0 | 0.0% |
| DC04 | 6 | 2.0% | 20 | 2.0% |
| DC05 | 0 | 0.0% | 13 | 1.3% |
| DC05 steel | 4 | 1.3% | 0 | 0.0% |
| DD Steel | 1 | 0.3% | 0 | 0.0% |
| DD Steel (Failure Case) | 1 | 0.3% | 0 | 0.0% |
| DDQ Steel | 4 | 1.3% | 59 | 5.9% |
| DDQ steel | 12 | 3.9% | 0 | 0.0% |
| Dual-Phase Steel | 1 | 0.3% | 0 | 0.0% |
| EN AW-6181-T1 | 16 | 5.3% | 59 | 5.9% |
| EN AW-6181-T1 Aluminium alloy | 2 | 0.7% | 0 | 0.0% |
| SUS 304 | 2 | 0.7% | 16 | 1.6% |
| Sheet Steel | 2 | 0.7% | 13 | 1.3% |
| Ti-6Al-4V | 1 | 0.3% | 5 | 0.5% |
| Titanium (grade 2) | 6 | 2.0% | 0 | 0.0% |
| Titanium Grade 2 | 0 | 0.0% | 20 | 2.0% |
| aluminium 5052-H32 sheet | 1 | 0.3% | 0 | 0.0% |
| aluminum AA1050-H111 | 2 | 0.7% | 0 | 0.0% |
| aluminum alloy (AA1060) | 2 | 0.7% | 0 | 0.0% |
| dual phase steel | 1 | 0.3% | 0 | 0.0% |
| steel sheet | 2 | 0.7% | 0 | 0.0% |

### 1.3 Numerical Feature Statistics

| Feature | Real Mean | Real Std | Real Median | Synth Mean | Synth Std | Synth Median |
|---------|----------:|---------:|------------:|-----------:|----------:|-------------:|
| Thickness | 1.15 | 0.34 | 1.00 | 1.14 | 0.31 | 1.00 |
| Precut Dim. | 50.57 | 31.71 | 45.00 | 50.84 | 29.86 | 44.00 |
| Elongation % | 21.10 | 10.65 | 17.00 | 21.08 | 10.00 | 17.11 |
| UTS | 248.33 | 109.87 | 228.00 | 251.82 | 107.45 | 243.65 |
| YS | 165.66 | 90.61 | 165.00 | 164.61 | 80.88 | 165.00 |
| Strength k | 381.25 | 206.37 | 400.00 | 389.39 | 197.43 | 399.43 |
| n (hardening) | 0.20 | 0.19 | 0.17 | 0.19 | 0.13 | 0.17 |
| R-value | 0.91 | 0.49 | 0.77 | 0.91 | 0.49 | 0.77 |
| Feed Rate | 1231.38 | 1257.30 | 1000.00 | 1185.21 | 992.81 | 1000.00 |
| Tool Speed | 494.54 | 983.48 | 170.00 | 450.88 | 776.12 | 248.68 |
| Step Depth | 0.41 | 0.33 | 0.25 | 0.41 | 0.31 | 0.35 |
| Stages | 2.17 | 1.95 | 1.00 | 2.22 | 1.82 | 1.00 |
| HER | 1.73 | 0.85 | 1.50 | 1.74 | 0.78 | 1.53 |
| Flange Height | 17.36 | 9.67 | 15.20 | 17.54 | 8.74 | 15.01 |
| Roughness | 12.33 | 32.25 | 0.65 | 11.99 | 30.25 | 0.34 |
| Min Thickness | 0.70 | 0.32 | 0.61 | 0.83 | 0.25 | 0.74 |
| Final Angle | 84.67 | 13.00 | 90.00 | 84.80 | 11.75 | 89.96 |

### 1.4 Missing Value Audit

| Feature | Real Missing | Real Missing (%) | Synth Missing | Synth Missing (%) |
|---------|------------:|----------------:|--------------:|------------------:|
| R-value | 10 | 3.3% | 1 | 0.1% |
| Feed Rate | 54 | 17.8% | 91 | 9.1% |
| Tool Speed | 86 | 28.3% | 173 | 17.3% |
| Step Depth | 39 | 12.8% | 35 | 3.5% |
| Stages | 9 | 3.0% | 6 | 0.6% |
| HER | 69 | 22.7% | 90 | 9.0% |
| Flange Height | 131 | 43.1% | 268 | 26.8% |
| Roughness | 236 | 77.6% | 660 | 65.9% |
| Min Thickness | 141 | 46.4% | 291 | 29.1% |
| Final Angle | 66 | 21.7% | 101 | 10.1% |
| **Total** | **841** | | **1716** | |

### 1.5 Outlier Census (Tukey IQR Method)

Outliers are defined as observations beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR.

| Feature | Real Outliers | Synth Outliers |
|---------|-------------:|--------------:|
| Thickness | 0 | 0 |
| Precut Dim. | 32 | 91 |
| Elongation % | 5 | 30 |
| UTS | 2 | 13 |
| YS | 6 | 10 |
| Strength k | 6 | 28 |
| n (hardening) | 15 | 60 |
| R-value | 51 | 169 |
| Feed Rate | 95 | 213 |
| Tool Speed | 16 | 50 |
| Step Depth | 27 | 69 |
| Stages | 6 | 20 |
| HER | 40 | 106 |
| Flange Height | 4 | 4 |
| Roughness | 8 | 53 |
| Min Thickness | 0 | 0 |
| Final Angle | 35 | 117 |


## 3. Validation Results

### 3.1 Overall Quality Summary

| Metric | Value |
|--------|------:|
| Composite Score | **60.61** / 100 |
| Letter Grade | **C** [Moderate] |
| Real Samples | 304 |
| Synthetic Samples | 1001 |
| KS Pass Rate | 44.44% |
| Mean Wasserstein (norm.) | 0.0188 |
| Mean JSD | 0.039853 |
| Mahalanobis Distance | 321.65 |
| Correlation Frobenius Norm | 1.9486 |
| Physics Sign Preservation | 85.71% |
| Mean Pct-Diff (Descriptive Stats) | 6.37% |

**Sub-scores (weighted contribution to composite):**

| Component | Score | Weight |
|-----------|------:|-------:|
| KS Pass Rate | 44.44 | 25% |
| Wasserstein | 98.12 | 25% |
| Correlation | 54.07 | 20% |
| Feature Importance | 0.74 | 15% |
| Stats Fidelity | 93.63 | 15% |

### 3.2 Kolmogorov-Smirnov Test Results

The two-sample KS test assesses whether the real and synthetic distributions are drawn from the same underlying distribution (null hypothesis) at significance level alpha = 0.05.

| Feature | KS Statistic | p-value | Verdict |
|---------|------------:|--------:|--------:|
| Thickness | 0.1850 | 0.0000 | [?] |
| Precut Dim. | 0.0923 | 0.0349 | [?] |
| Elongation % | 0.0715 | 0.1746 | [?] |
| UTS | 0.0996 | 0.0180 | [?] |
| YS | 0.0748 | 0.1382 | [?] |
| Strength k | 0.0915 | 0.0375 | [?] |
| n (hardening) | 0.0776 | 0.1132 | [?] |
| R-value | 0.0823 | 0.0860 | [?] |
| Is lubricant used? | 0.0070 | 1.0000 | [?] |
| Feed Rate | 0.1912 | 0.0000 | [?] |
| Tool Speed | 0.2119 | 0.0000 | [?] |
| Step Depth | 0.1910 | 0.0000 | [?] |
| Stages | 0.1027 | 0.0151 | [?] |
| HER | 0.0638 | 0.4116 | [?] |
| Flange Height | 0.0728 | 0.4246 | [?] |
| Roughness | 0.1306 | 0.2618 | [?] |
| Min Thickness | 0.3183 | 0.0000 | [?] |
| Final Angle | 0.2847 | 0.0000 | [?] |

**Summary:** 0/18 features pass the KS test (0.00%).

### 3.3 Wasserstein Distance and Jensen-Shannon Divergence

The Wasserstein-1 distance (Earth Mover's Distance) measures the minimum 'cost' of transforming one distribution into another.  The Jensen-Shannon Divergence (JSD) provides a symmetric, bounded [0, ln(2)] measure of distributional similarity.

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

### 3.4 Correlation Preservation

Correlation fidelity is assessed in two ways: (a) full-matrix Frobenius norm of the difference, and (b) pair-wise analysis of seven physics-critical feature pairs.

- **Frobenius norm (Real - Synth):** 1.9486
- **Mean absolute correlation difference:** 0.0609

#### Physics-Critical Pair Analysis

| Pair | Real rho | Synth rho | Abs Diff | Sign Preserved |
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

| Pair | Real MI | Synth MI | Ratio |
|------|-------:|---------:|------:|
| HER <-> Flange Height (mm) | 1.7381 | 0.9589 | 0.552 |
| Step depth (mm) <-> Roughness (um) | 0.5747 | 0.7467 | 1.299 |
| Step depth (mm) <-> Minimum thickness (after final stage, mm) | 0.6221 | 0.7950 | 1.278 |
| No of stages <-> HER | 0.5010 | 0.3713 | 0.741 |
| Is lubricant used? <-> Roughness (um) | 0.3490 | 0.1772 | 0.508 |
| Total Strain/Elongation (%) <-> HER | 1.4702 | 1.1612 | 0.790 |
| Anisotropic (R Value) <-> HER | 1.2953 | 1.1764 | 0.908 |

### 3.5 Descriptive Statistics Comparison

| Feature | Real Mean | Synth Mean | % Diff Mean | Real Std | Synth Std | % Diff Std |
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
| Material (frequencies) | 0.00 | 0.00 | 0.00% | 0.00 | 0.00 | 0.00% |
| Precut Shape (circle/s | 0.00 | 0.00 | 0.00% | 0.00 | 0.00 | 0.00% |

**Mean percentage difference across all statistics:** 6.37%

### 3.6 Feature Importance Similarity

Feature importance rankings for predicting HER (Hole Expansion Ratio) are compared between real and synthetic datasets using two methods: Random Forest Gini importance and Mutual Information.

| Feature | Real RF | Synth RF | Real MI | Synth MI |
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

| Method | Spearman rho | p-value | Verdict |
|--------|------------:|--------:|--------:|
| Random Forest | 0.0074 | 0.977656 | [FAIL] |
| Mutual Information | 0.6168 | 0.008356 | [MARGINAL] |

