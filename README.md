# Does Gravity See Slow Mantle Regions as Heavy or Light?

A frozen-intermediate publication package for a global degree-2 comparison of eight full-mantle model fields with the observed static-gravity field.

## Central result

Seven of the eight full-mantle model fields favored the globally reversed sign in the primary reference-compatible comparison. Among the seven velocity-based fields, six favored the mapping in which slow seismic anomalies contribute as relatively lighter rather than denser mass anomalies. The remaining reversed result came from a native density-related field.

All eight model fields retained their deterministic sign classification in 10,000 Monte Carlo realizations under the documented formal gravity-coefficient covariance: seven remained reversed and one remained adopted.

This is a conditional coefficient-space result. It does not directly determine intrinsic LLSVP density, composition, rheology, buoyancy, mechanical stability, or behavior during mantle displacement.

## What this repository contains

```text
paper/
    Buchanan_When Gravity Favors a Lighter Slow Mantle.pdf

data/
    degree2_gravity_product_summary.csv
    mantle_model_degree2_predictions.csv
    compatible_residual_coefficients.csv
    compatible_residual_definitions.csv
    compatible_reference_state_registry.csv

validation/
    signed_beta_significance.csv
    statistical_robustness_summary.csv
    monte_carlo_sign_confidence.csv
    constrained_sign_hypothesis_comparison.csv
    delta_chi_square_by_model.csv

code/
    preferred_axis_sign_convention_review_final.py

requirements.txt
README.md
```

## Reproducibility boundary

This repository is designed to support **frozen-intermediate reproducibility** of the published sign test.

Reproduction begins with:

- the observed degree-2 gravity-residual coefficients;
- the archived degree-2 prediction vector for each of the eight model fields;
- the documented formal gravity-coefficient covariance;
- the archived validation outputs;
- the fitting, sign-decision, constrained-residual, and Monte Carlo procedures implemented in the Python code.

The repository does not reconstruct those degree-2 vectors from the original three-dimensional tomography models. Private working notebooks, raw tomography volumes, broader cleaned intermediate datasets, upstream helper libraries, and preprocessing pipelines are intentionally excluded.

A complete end-to-end reconstruction would additionally require the original tomography grids, gravity products, shallow-correction inputs, model parsers, preprocessing code, software environment, and full upstream provenance.


## Method summary

For each mantle model, the observed global-scale gravity pattern is represented by five degree-2 coefficients:

```math
\mathbf{d}
=
\begin{bmatrix}
C_{20} \\
C_{21} \\
S_{21} \\
C_{22} \\
S_{22}
\end{bmatrix}.
```

The corresponding gravity pattern predicted by model $m$ is represented by $\mathbf{g}_m$.

The analysis estimates the signed amplitude $\hat{\beta}_m$ that best scales the model prediction to match the observed gravity residual:

```math
\hat{\beta}_m
=
\frac{
\mathbf{g}_m^{\mathsf{T}}
\mathbf{C}_d^{-1}
\mathbf{d}
}{
\mathbf{g}_m^{\mathsf{T}}
\mathbf{C}_d^{-1}
\mathbf{g}_m
}.
```

Here, $\mathbf{C}_d$ is the documented covariance matrix for the gravity coefficients.

The sign of $\hat{\beta}_m$ determines which mapping is favored:

- $\hat{\beta}_m > 0$: the observed gravity field favors the adopted model sign.
- $\hat{\beta}_m < 0$: the observed gravity field favors the globally reversed sign.
- If the uncertainty interval crosses zero, the sign is treated as indeterminate.


## Important uncertainty limits

The Monte Carlo intervals represent the documented formal gravity-coefficient uncertainty only. They do not include the complete uncertainty associated with:

- shallow-load corrections;
- gravity-reference closure;
- tomography-model structure;
- unavailable coefficient correlations;
- alternative density conversions;
- thermodynamic or dynamical behavior of the LLSVPs.

The leave-one-coefficient-out test also shows substantial \(S_{22}\) leverage. Removing \(S_{22}\) changes the model count from seven reversed and one adopted to four reversed and four adopted. That sensitivity is part of the reported result and should not be omitted when interpreting the central model count.

## Installation

Create and activate a Python environment, then install the listed dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## File integrity and validation

The `validation/` directory contains the archived outputs against which the standalone reconstruction was checked.

## Citation

Buchanan, Stephanie. 2026. *Does Gravity See Slow Mantle Regions as Heavy or Light?  
A Global-Scale Comparison of Mantle Models with Observed Gravity.*  
Independent geophysical modeling case study.

## Author

Stephanie Buchanan, M.S.
_Independent Researcher in Geophysical Modeling_

