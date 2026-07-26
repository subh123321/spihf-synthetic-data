# Physics-Informed Synthetic Data Augmentation for Single Point Incremental Hole Flanging (SPIHF)

## 1. Executive Summary

This project addresses a critical bottleneck in the application of Machine Learning (ML) to advanced manufacturing processes: **Data Scarcity**. Single Point Incremental Hole Flanging (SPIHF) is a highly non-linear sheet-metal forming process governed by complex elasto-plastic deformation mechanics. Generating empirical data for SPIHF is time-consuming, material-intensive, and expensive, resulting in a sparse experimental corpus (304 real samples). 

To enable the training of robust, generalizable predictive models (e.g., Deep Neural Networks or advanced ensemble methods), we engineered a **Physics-Informed Data Augmentation Pipeline**. This pipeline autonomously scales the dataset to 1,305 samples by leveraging SMOTE-inspired interpolations bound by strict metallurgical and mechanical constraints. The result is a mathematically rigorous and physically plausible synthetic dataset.

---

## 2. Background and Motivation

Data-driven modelling of SPIHF requires mapping complex input parameters (e.g., sheet thickness, tool speed, feed rate, precut geometry, and intrinsic material properties like the strain-hardening coefficient, *n*) to formability outcomes (e.g., maximum flanging ratio, failure modes). 

Traditional ML augmentation techniques (like standard SMOTE, GANs, or VAEs) often fail in engineering domains because they are purely statistical. They might "hallucinate" impossible physical states, such as:
- Interpolating between Aluminum and Titanium to create an alloy that doesn't exist.
- Generating negative tool speeds or non-integer forming stages.
- Violating the principle of volume constancy or forming limit curves (FLC).

**Our Approach:** We fused Artificial Intelligence with Mechanical Engineering. We built an augmentation pipeline that generates data statistically, but filters, repairs, and validates that data physically.

---

## 3. System Architecture and Pipeline Methodology

The project was refactored into a highly modular, enterprise-grade Python package (`spihf_synthetic`) to ensure reproducibility, scalability, and clean separation of concerns.

### 3.1. Data Preprocessing & Harmonization
The pipeline begins by ingesting a heterogeneous compilation of experimental data from diverse academic sources.
- **Harmonization:** Standardizes column nomenclature, strips embedded string units from numeric arrays, and canonicalizes over 68 distinct raw material strings into 24 distinct metallurgical categories (e.g., `AA1050-H111 Aluminium` and `Al 1050` are cleanly mapped to a single unified class).
- **Stratification:** Feature statistics (mean, variance, boundaries) are computed *strictly on a per-material basis* to prevent cross-contamination of physical properties.

### 3.2. The Augmentation Engine: SMOTE-Inspired Convex Interpolation
To generate a new synthetic observation for a specific material class (e.g., *DC01 Steel*), the algorithm samples two empirical points $X_i$ and $X_j$ from that same class. It then applies a convex combination:

$$ X_{synthetic} = \alpha X_i + (1 - \alpha) X_j $$

where $\alpha \sim \mathcal{U}(\alpha_{low}, \alpha_{high})$. This ensures the new sample lies on the multi-dimensional manifold connecting real experimental runs.

### 3.3. Calibrated Gaussian Perturbation
To prevent the synthetic distribution from collapsing into a finite set of linear segments, a controlled isotropic Gaussian noise is introduced.
- **Micro-perturbation:** Noise amplitude is dynamically scaled to exactly **1% to 3%** of the standard deviation of each specific feature. 
- This introduces necessary variance to improve ML model generalization without drifting outside the domain of validity.

---

## 4. Physics-Informed Constraint Layer

This is the mechanical engineering core of the project. A purely statistical ML algorithm does not "understand" physics. We introduced a deterministic repair layer that enforces physical realities onto the generated tensors:

1. **Kinematic Bounds:** Tool rotational speed ($N \ge 0$ RPM) and Feed rate ($f \ge 1.0$ mm/min) are clamped to strictly non-negative or strictly positive operational minimums.
2. **Discrete Operations:** The number of forming stages is rounded to strict integers ($N_{stages} \in \mathbb{Z}^+$), and boolean flags (like *Lubricant Used*) are mapped to $\{0, 1\}$.
3. **Metallurgical Reality:** Material intrinsic properties (Yield Strength, Ultimate Tensile Strength, Hollomon strength coefficient $K$) are bound strictly by the real min/max bounds observed for that specific material in the experimental data.

---

## 5. Statistical Validation & Outlier Rejection

Generating data is trivial; proving its validity is computationally rigorous. We implemented a multi-stage Quality Assurance protocol:

### 5.1. Confidence Scoring and Rejection Sampling
Every generated sample is graded. The confidence module evaluates the Mahalanobis distance / Isolation Forest anomaly score of the sample relative to the real data distribution. If a generated sample falls into a low-probability density region (an outlier), it is systematically **rejected** and a new sample is generated.

### 5.2. Near-Duplicate Pruning
Using $L_2$ Euclidean distance in normalized hyperspace, the pipeline detects and aggressively prunes near-duplicate samples to prevent ML model overfitting (memorization).

### 5.3. Global Statistical Validation
The pipeline utilizes non-parametric statistical tests to compare the probability density functions of the real vs. synthetic datasets:
- **Kolmogorov-Smirnov (KS) Tests:** Verifies that the cumulative distribution functions (CDFs) of real and synthetic features are statistically indistinguishable.
- **Wasserstein Distance (Earth Mover's Distance):** Measures the minimal "work" required to transform the synthetic distribution into the real distribution.
- **Jensen-Shannon Divergence:** Measures the similarity between the two probability distributions.
- **Covariance Preservation:** Generates correlation heatmaps to prove that multivariate relationships (e.g., the inverse relationship between elongation and yield strength) are perfectly preserved.

---

## 6. Conclusion

By merging **Machine Learning generative techniques (SMOTE, Gaussian noise)** with **Mechanical Engineering domain expertise (physics constraints, metallurgical bounding)**, we successfully and safely scaled the SPIHF dataset from 304 to 1,305 observations. 

The resulting pipeline is not just a statistical data-multiplier, but a **scientifically rigorous digital twin** of the SPIHF experimental design space. This mathematically validated dataset is now primed for the deployment of high-capacity deep learning models to predict forming limits, optimize toolpaths, and prevent sheet tearing.
