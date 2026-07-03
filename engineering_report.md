# Engineering Report: SPIHF Synthetic Data Analysis

> **Auto-generated** on 2026-07-03 13:02:13 by `report_generator.py`
> Original samples: 304 | Synthetic samples: 1001

---

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


## 4. Engineering Validity

This section evaluates whether the synthetic dataset respects the fundamental physical laws and empirical trends that govern Single Point Incremental Hole Flanging (SPIHF).  Each sub-section examines a specific engineering relationship.

### 4.1 UTS >= YS Constraint

By definition, the Ultimate Tensile Strength (UTS) of any metallic alloy must equal or exceed its Yield Strength (YS).  In the synthetic dataset, **0 out of 1001** samples violate this constraint (0.00% violation rate).
  The physics-informed rejection layer has successfully enforced this fundamental metallurgical inequality across all generated samples.

### 4.2 Hole Expansion Ratio (HER) Behaviour

The HER quantifies the formability limit during hole flanging.  In the original dataset, HER ranges from **0.21** to **5.70** (mean = 1.73, std = 0.85).  The synthetic dataset preserves a comparable range: **1.00** to **5.70** (mean = 1.74, std = 0.78).

As expected from materials science, elongation is positively correlated with HER (real rho = 0.2054, synth rho = 0.2106).  Higher ductility enables greater hole expansion before edge fracture.

### 4.3 Effect of Number of Forming Stages

Multi-stage incremental forming redistributes strain across passes, typically allowing higher total HER values while reducing the risk of localised necking.  
The Pearson correlation between Stages and HER is **-0.1194** (real) and **-0.1685** (synthetic).  
The stage-wise mean HER in the real data is:

- 1 stage(s): mean HER = 1.757
- 2 stage(s): mean HER = 1.912
- 3 stage(s): mean HER = 1.647
- 4 stage(s): mean HER = 1.766
- 5 stage(s): mean HER = 1.917
- 6 stage(s): mean HER = 1.115
- 9 stage(s): mean HER = 1.090

The negative or weak correlation may appear counter-intuitive but reflects the fact that multi-stage strategies are preferentially applied to difficult-to-form materials with inherently lower HER, creating a confounding effect in the observational data.

### 4.4 Lubrication Effects

Lubrication reduces tool-sheet friction, which in turn lowers surface roughness on the formed flange.  
The correlation between lubrication and roughness is strongly negative in both datasets (real rho = **-0.9999**, synth rho = **-0.5562**), confirming that the synthetic data captures the friction-mitigation effect of lubricant application.

- **Real** mean roughness: lubricated = 0.64 um, unlubricated = 100.00 um
- **Synthetic** mean roughness: lubricated = 3.84 um, unlubricated = 46.61 um

### 4.5 Thickness Evolution

During incremental hole flanging, the sheet undergoes progressive thinning.  The thinning ratio (min thickness / initial thickness) averages **0.587** in the real dataset and **0.708** in the synthetic dataset.  
**0** synthetic samples violate the constraint Min Thickness <= Initial Thickness, indicating effective rejection sampling.

Step depth is expected to influence thinning: larger incremental steps produce more severe localised deformation.  The correlation is rho_real = 0.1898, rho_synth = 0.0109.

### 4.6 Flange Height Relationships

Flange height is the primary dimensional output of the SPIHF process.  The real dataset records a range of **0.42** to **50.00 mm** (mean = 17.36 mm).  The synthetic dataset spans **0.34** to **49.79 mm** (mean = 17.54 mm).

The correlation between HER and Flange Height is near zero in both datasets (real rho = -0.0151, synth rho = 0.0053), which is consistent with the fact that flange height is primarily determined by precut hole diameter and tool path geometry, not the expansion ratio per se.

### 4.7 Surface Roughness Trends

Surface roughness in SPIF-type processes is predominantly controlled by step depth (tool step-down per pass), tool diameter, feed rate, and lubrication.  Larger step depths produce more pronounced scalloping on the inner surface, increasing Ra values.  
The Step Depth vs Roughness correlation is **0.6961** (real) and **0.0020** (synthetic).  
The absolute difference of **0.6941** is notable and suggests that the synthetic data has attenuated this correlation -- likely because Gaussian noise added to both step depth and roughness independently reduces the marginal signal.  This is an area for future improvement in the augmentation pipeline (e.g., correlated noise injection).


## 5. Research Limitations

While the augmentation pipeline produces statistically plausible samples, several limitations must be acknowledged to guide responsible use of the synthetic data.

### 5.1 Small Original Dataset

The original SPIHF dataset contains only **304** observations drawn from **68** materials.  Several material groups contain fewer than 10 samples, making their within-group statistics highly sensitive to individual outliers.  Consequences include:

- **Sampling noise amplification:** SMOTE interpolation between a small number of parents can produce a narrow synthetic cloud that fails to capture the true process variability.
- **Unreliable higher-order statistics:** Skewness and kurtosis estimates from fewer than 20 points are unstable, so the synthetic data may not match these moments even if means and standard deviations are well preserved.
- **Material bias:** Materials with very few observations contribute proportionally fewer synthetic samples; any systematic measurement error in those few experiments propagates unchanged into the augmented dataset.

### 5.2 Synthetic Data Bias

Synthetic augmentation cannot introduce information that was not present in the original data.  The generated samples are strictly interpolative (within the convex hull of each material group) with small perturbations.  This means:

- **No extrapolation:** The synthetic dataset will not contain process configurations beyond those tested experimentally (e.g., extremely thin sheets, very high feed rates, or novel alloys).
- **Correlation attenuation:** Gaussian noise applied independently to each feature tends to decorrelate features that are physically linked.  The validation results confirm this: the Step Depth vs Roughness correlation dropped significantly in the synthetic data.
- **Mode collapse risk:** If the original data contains bimodal distributions (e.g., two distinct HER regimes for the same material), linear interpolation may fill in the 'gap' between modes, creating plausible-looking but non-physical intermediate points.

### 5.3 Overfitting Dangers

The augmented dataset (1001 samples) is over 3.3x larger than the original.  If used naively for ML training, models may:

- **Overfit to the synthetic manifold** rather than to the true process physics, particularly if the synthetic data has introduced any subtle structural bias.
- **Inflate performance estimates:** Cross-validation on the combined dataset may yield optimistically low error rates because synthetic test points are correlated with synthetic training points (they share the same generative mechanism).

**Mitigation strategies:**
1. Always hold out the entire *real* dataset for final model evaluation -- never mix real and synthetic in the same fold.
2. Use the `confidence_score` column to weight training samples, downweighting low-confidence synthetic observations.
3. Perform ablation studies: compare model performance trained on real-only vs. real+synthetic to quantify the net benefit of augmentation.

### 5.4 Physical Assumptions

The rejection-sampling constraints encode simplified physical rules (e.g., UTS >= YS, Min Thickness <= Thickness).  These are necessary but not sufficient conditions for physical plausibility.  Several subtleties are not captured:

- **Strain-path dependence:** The thinning pattern in incremental forming depends on the strain path (biaxial vs. plane strain), which varies with tool trajectory and cannot be inferred from scalar features alone.
- **Anisotropy coupling:** The R-value influences forming limits in a non-linear, orientation-dependent manner (0/45/90 degree rolling directions).  The dataset records only an average R-value, losing directional information.
- **Temperature effects:** High spindle speeds generate frictional heating that can alter material properties in situ; the dataset does not include temperature measurements.
- **Tool wear:** Progressive tool degradation affects surface roughness and forming forces over time -- an effect absent from the cross-sectional dataset.

### 5.5 Generalisation Limitations

The current composite validation score of **C** (60.6/100) reflects moderate fidelity.  Key generalisation caveats:

- **Material scope:** The model is valid only for the specific alloys present in the dataset (68 materials).  Applying trained models to predict HER for an untested alloy (e.g., titanium Ti-6Al-4V) requires caution.
- **Process configuration scope:** All data originates from single-point incremental forming with hemispherical tools.  Extension to multi-point, double-sided, or hybrid forming strategies is not warranted.
- **Scale effects:** The datasets represent lab-scale experiments; industrial-scale forming involves larger blanks, different clamping arrangements, and machine-specific dynamics that may alter process-response relationships.
- **Feature importance instability:** The Random Forest importance ranking diverged significantly between real and synthetic data (Spearman rho = 0.0074).  This suggests that high-dimensional predictive relationships are not fully preserved and should be interpreted with caution.

