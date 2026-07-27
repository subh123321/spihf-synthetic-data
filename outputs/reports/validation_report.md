# Validation Report: Statistical Fidelity of Synthetic SPIHF Data

> **Auto-generated** on 2026-07-26 12:34:01 by `spihf_synthetic.reporting`
> Original samples: 304 | Synthetic samples: 1001

---

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


## 5. Research Limitations

### 5.1 Small Original Dataset

The original dataset contains only **304** observations across **65** materials.  Consequences include sampling noise amplification and unreliable higher-order statistics.

### 5.2 Synthetic Data Bias

No extrapolation beyond tested configurations. Correlation attenuation from independent Gaussian noise. Mode-collapse risk for bimodal features.

### 5.3 Overfitting Dangers

Augmented dataset is **3.3×** larger.  Always hold out the original dataset for final evaluation.

### 5.4 Physical Assumptions

Simplified constraints (UTS ≥ YS, sine-law thinning) are necessary but not sufficient.  Strain-path dependence, anisotropy coupling, and temperature effects are not captured.

### 5.5 Generalisation Limitations

Overall grade **C**.  Valid only for the specific alloy systems and process configurations in the dataset.

