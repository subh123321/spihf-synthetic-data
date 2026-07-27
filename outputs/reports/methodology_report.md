# Methodology Report: Synthetic Data Generation for SPIHF

> **Auto-generated** on 2026-07-26 12:34:01 by `spihf_synthetic.reporting`
> Original samples: 304 | Synthetic samples: 1001

---

## 2. Synthetic Data Generation Methodology

### 2.1 Overview

The SPIHF dataset comprises **304** observations spanning **65** alloys.  A physics-informed pipeline generated **1001** synthetic samples.

### 2.2 Material-Wise Stratified Generation

Synthetic samples are generated **independently within each material group** to prevent cross-material interpolation artefacts.

### 2.3 SMOTE-Inspired Interpolation

```
x_new = α · x_i + (1 − α) · x_j
α ~ Uniform(0.2, 0.8)
```

### 2.4 Gaussian Perturbation

```
x_perturbed = x_interpolated + ε
ε ~ N(0, σ_material · η),  η ∈ [0.01, 0.03]
```

### 2.5 Physics-Informed Rejection Sampling

| # | Constraint | Rationale |
|:-:|-----------|----------|
| 1 | UTS ≥ YS | Engineering stress–strain curve definition. |
| 2 | Thickness > 0 | Strictly positive. |
| 3 | HER > 0 | Positive geometric ratio. |
| 4 | Min Thickness ≤ Thickness | Thinning constraint. |
| 5 | 0 < n < 1 | Hollomon exponent bounds. |
| 6 | R ≥ 0 | Lankford coefficient non-negative. |
| 7 | Step Depth > 0 | Positive tool step-down. |
| 8 | Stages ≥ 1 | At least one pass. |
| 9 | Angle ∈ [0, 180] | Wall angle bounds. |

### 2.6 Confidence Scoring

Each sample receives a confidence score in [0, 1] combining range proximity, nearest-neighbour distance, and correlation agreement.


## 1. Dataset Summary

### 1.1 Sample Counts

| Dataset | Samples | Features |
|---------|--------:|---------:|
| Original (experimental) | 304 | 20 |
| Synthetic (augmented) | 1001 | 21 |
| **Combined** | **1305** | **21** |

Augmentation factor: **3.3x**, combined dataset: **1305** observations.

### 1.2 Material Distribution

| Material | Original (n) | Original (%) | Synthetic (n) | Synthetic (%) |
|----------|--------:|--------:|--------------:|--------------:|
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
| AA1060 | 9 | 3.0% | 86 | 8.6% |
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
| DC01 | 10 | 3.3% | 39 | 3.9% |
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

| Feature | Orig. Mean | Orig. Std | Orig. Median | Synth. Mean | Synth. Std | Synth. Median |
|---------|----------:|---------:|------------:|-----------:|----------:|-------------:|
| Thickness | 1.15 | 0.34 | 1.00 | 1.14 | 0.31 | 1.00 |
| Precut dim. | 50.57 | 31.71 | 45.00 | 50.75 | 29.57 | 43.93 |
| Elongation % | 21.10 | 10.65 | 17.00 | 21.09 | 9.96 | 17.06 |
| UTS | 248.33 | 109.87 | 228.00 | 251.58 | 106.70 | 243.72 |
| YS | 165.66 | 90.61 | 165.00 | 164.43 | 80.30 | 165.00 |
| Strength k | 381.25 | 206.37 | 400.00 | 389.06 | 196.98 | 399.72 |
| n (hardening) | 0.20 | 0.19 | 0.17 | 0.20 | 0.13 | 0.17 |
| R-value | 0.91 | 0.49 | 0.77 | 0.91 | 0.49 | 0.77 |
| Lubricant | 0.79 | 0.41 | 1.00 | 0.78 | 0.41 | 1.00 |
| Feed rate | 1231.38 | 1257.30 | 1000.00 | 1191.15 | 1000.63 | 1000.00 |
| Tool speed | 494.54 | 983.48 | 170.00 | 451.41 | 770.33 | 249.37 |
| Step depth | 0.41 | 0.33 | 0.25 | 0.40 | 0.30 | 0.36 |
| Stages | 2.17 | 1.95 | 1.00 | 2.21 | 1.80 | 1.00 |
| HER | 1.73 | 0.85 | 1.50 | 1.73 | 0.78 | 1.53 |
| Flange Height | 17.36 | 9.67 | 15.20 | 17.52 | 8.69 | 15.01 |
| Roughness | 11.35 | 31.08 | 0.49 | 12.08 | 30.28 | 0.34 |
| Min thickness | 0.70 | 0.32 | 0.61 | 0.83 | 0.25 | 0.76 |
| Final angle | 84.67 | 13.00 | 90.00 | 84.81 | 11.70 | 89.98 |

### 1.4 Missing Value Audit

| Feature | Orig. Missing | Orig. Missing (%) | Synth. Missing | Synth. Missing (%) |
|---------|------------:|----------------:|--------------:|------------------:|
| R-value | 10 | 3.3% | 1 | 0.1% |
| Feed rate | 54 | 17.8% | 91 | 9.1% |
| Tool speed | 86 | 28.3% | 173 | 17.3% |
| Step depth | 39 | 12.8% | 35 | 3.5% |
| Stages | 9 | 3.0% | 6 | 0.6% |
| HER | 69 | 22.7% | 90 | 9.0% |
| Flange Height | 131 | 43.1% | 268 | 26.8% |
| Roughness | 230 | 75.7% | 660 | 65.9% |
| Min thickness | 141 | 46.4% | 291 | 29.1% |
| Final angle | 66 | 21.7% | 101 | 10.1% |

### 1.5 Outlier Census (Tukey IQR Method)

| Feature | Orig. Outliers | Synth. Outliers |
|---------|-------------:|--------------:|
| Thickness | 0 | 0 |
| Precut dim. | 32 | 90 |
| Elongation % | 5 | 11 |
| UTS | 2 | 15 |
| YS | 6 | 6 |
| Strength k | 6 | 32 |
| n (hardening) | 15 | 64 |
| R-value | 51 | 169 |
| Lubricant | 65 | 221 |
| Feed rate | 95 | 210 |
| Tool speed | 16 | 49 |
| Step depth | 27 | 70 |
| Stages | 6 | 20 |
| HER | 40 | 109 |
| Flange Height | 4 | 4 |
| Roughness | 8 | 51 |
| Min thickness | 0 | 0 |
| Final angle | 35 | 112 |


## 3. Validation Results

### 3.1 Overall Quality Summary

| Metric | Value |
|--------|------:|
| Composite Score | **60.85** / 100 |
| Letter Grade | **C** [Moderate] |
| KS Pass Rate | 44.44% |
| Mean Wasserstein (norm.) | 0.0194 |
| Mahalanobis Distance | 329.43 |

### 3.2 Kolmogorov–Smirnov Test Results

| Feature | KS Statistic | p-value | Verdict |
|---------|------------:|--------:|--------:|
| Thickness | 0.1889 | 0.0000 | [FAIL] |
| Precut dim. | 0.0923 | 0.0349 | [FAIL] |
| Elongation % | 0.0735 | 0.1523 | [PASS] |
| UTS | 0.1026 | 0.0135 | [FAIL] |
| YS | 0.0673 | 0.2285 | [PASS] |
| Strength k | 0.0955 | 0.0264 | [FAIL] |
| n (hardening) | 0.0795 | 0.0982 | [PASS] |
| R-value | 0.0873 | 0.0583 | [PASS] |
| Lubricant | 0.0070 | 1.0000 | [PASS] |
| Feed rate | 0.1978 | 0.0000 | [FAIL] |
| Tool speed | 0.2119 | 0.0000 | [FAIL] |
| Step depth | 0.1921 | 0.0000 | [FAIL] |
| Stages | 0.1047 | 0.0125 | [FAIL] |
| HER | 0.0638 | 0.4116 | [PASS] |
| Flange Height | 0.0709 | 0.4578 | [PASS] |
| Roughness | 0.0686 | 0.9150 | [PASS] |
| Min thickness | 0.3090 | 0.0000 | [FAIL] |
| Final angle | 0.2813 | 0.0000 | [FAIL] |

### 3.3 Wasserstein Distance and Jensen–Shannon Divergence

| Feature | Wasserstein | W (norm.) | JSD |
|---------|------------:|----------:|----:|
| Thickness | 0.0366 | 0.0211 | 0.092802 |
| Precut dim. | 2.2008 | 0.0125 | 0.044996 |
| Elongation % | 0.6642 | 0.0122 | 0.038862 |
| UTS | 6.2529 | 0.0112 | 0.042241 |
| YS | 6.9071 | 0.0125 | 0.038817 |
| Strength k | 16.5164 | 0.0130 | 0.028290 |
| n (hardening) | 0.0174 | 0.0090 | 0.018307 |
| R-value | 0.0088 | 0.0029 | 0.006031 |
| Lubricant | 0.0070 | 0.0070 | 0.000036 |
| Feed rate | 118.5470 | 0.0099 | 0.035046 |
| Tool speed | 97.9039 | 0.0163 | 0.036849 |
| Step depth | 0.0284 | 0.0142 | 0.054012 |
| Stages | 0.2437 | 0.0305 | 0.025652 |
| HER | 0.0651 | 0.0119 | 0.063863 |
| Flange Height | 0.7479 | 0.0151 | 0.025316 |
| Roughness | 1.7245 | 0.0173 | 0.014512 |
| Min thickness | 0.1410 | 0.1175 | 0.155630 |
| Final angle | 0.8916 | 0.0149 | 0.030006 |

### 3.4 Correlation Preservation

- **Frobenius norm:** 1.9303
- **Mean |Δρ|:** 0.0608

| Pair | Orig. ρ | Synth. ρ | |Δ| | Sign OK |
|------|--------:|---------:|----:|:-------:|
| HER <-> Flange Height (mm) | -0.0151 | 0.0065 | 0.0216 | No |
| Step depth (mm) <-> Roughness (um) | 0.7168 | 0.0035 | 0.7133 | Yes |
| Step depth (mm) <-> Minimum thickness (after final stage, mm) | 0.1898 | 0.0188 | 0.1710 | Yes |
| No of stages <-> HER | -0.1194 | -0.1699 | 0.0505 | Yes |
| Is lubricant used? <-> Roughness (um) | -0.9999 | -0.5501 | 0.4498 | Yes |
| Total Strain/Elongation (%) <-> HER | 0.2054 | 0.2128 | 0.0074 | Yes |
| Anisotropic (R Value) <-> HER | 0.1005 | 0.0441 | 0.0564 | Yes |

### 3.5 Feature Importance Similarity

| Method | Spearman ρ | p-value | Verdict |
|--------|----------:|--------:|--------:|
| Random Forest | 0.0196 | 0.940458 | [FAIL] |
| Mutual Info | 0.5910 | 0.012469 | [MARGINAL] |

### 3.6 Descriptive Statistics Comparison

| Feature | Orig. Mean | Synth. Mean | % Diff Mean | Orig. Std | Synth. Std | % Diff Std |
|---------|----------:|-----------:|:-----------:|---------:|----------:|:----------:|
| Thickness | 1.15 | 1.14 | 0.18% | 0.34 | 0.31 | 9.14% |
| Precut dim. | 50.57 | 50.75 | 0.36% | 31.71 | 29.57 | 6.73% |
| Elongation % | 21.10 | 21.09 | 0.05% | 10.65 | 9.96 | 6.50% |
| UTS | 248.33 | 251.58 | 1.31% | 109.87 | 106.70 | 2.89% |
| YS | 165.66 | 164.43 | 0.74% | 90.61 | 80.30 | 11.37% |
| Strength k | 381.25 | 389.06 | 2.05% | 206.37 | 196.98 | 4.55% |
| n (hardening) | 0.20 | 0.20 | 0.88% | 0.19 | 0.13 | 32.53% |
| R-value | 0.91 | 0.91 | 0.14% | 0.49 | 0.49 | 0.18% |
| Lubricant | 0.79 | 0.78 | 0.89% | 0.41 | 0.41 | 1.05% |
| Feed rate | 1231.38 | 1191.15 | 3.27% | 1257.30 | 1000.63 | 20.41% |
| Tool speed | 494.54 | 451.41 | 8.72% | 983.48 | 770.33 | 21.67% |
| Step depth | 0.41 | 0.40 | 0.26% | 0.33 | 0.30 | 8.73% |
| Stages | 2.17 | 2.21 | 2.17% | 1.95 | 1.80 | 7.64% |
| HER | 1.73 | 1.73 | 0.41% | 0.85 | 0.78 | 8.72% |
| Flange Height | 17.36 | 17.52 | 0.91% | 9.67 | 8.69 | 10.16% |
| Roughness | 11.35 | 12.08 | 6.43% | 31.08 | 30.28 | 2.56% |
| Min thickness | 0.70 | 0.83 | 18.06% | 0.32 | 0.25 | 21.10% |
| Final angle | 84.67 | 84.81 | 0.16% | 13.00 | 11.70 | 10.01% |

