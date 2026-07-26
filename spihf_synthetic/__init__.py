"""
spihf_synthetic
===============
Physics-Informed Synthetic Data Augmentation for Single Point Incremental
Hole Flanging (SPIHF) Datasets.

Modules
-------
config          Global constants and reproducibility settings.
augmentation    SMOTE-inspired interpolation, Gaussian perturbation, pipeline.
constraints     Engineering constraint checks and sample repair.
confidence      Multi-component confidence scoring.
validation      Statistical validation (KS, Wasserstein, JSD, correlation).
visualization   Publication-quality matplotlib figures.
reporting       Markdown report generation.
utils           Shared helpers (column harmonisation, formatting, I/O).
"""

from __future__ import annotations

__version__: str = "1.0.0"
__author__: str = "SPIHF Research Group"

from spihf_synthetic.config import RANDOM_SEED
