# Engineering Report: SPIHF Synthetic Data Analysis

> **Auto-generated** on 2026-07-26 12:34:01 by `spihf_synthetic.reporting`
> Original samples: 304 | Synthetic samples: 1001

---

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


## 4. Engineering Validity

### 4.1 UTS ≥ YS Constraint

**0 out of 1001** samples violate UTS < YS (0.00%).

### 4.2 Hole Expansion Ratio Behaviour

Original HER: [0.21, 5.70], mean = 1.73.  Synthetic HER: [1.00, 5.70], mean = 1.73.

### 4.3 Thickness Evolution

**0** synthetic samples violate Min Thickness ≤ Initial Thickness.

### 4.4 Surface Roughness Trends

Step Depth vs Roughness ρ: original = 0.7168, synthetic = 0.0035.


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

