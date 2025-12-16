# Uncertainty Quantification Metrics for Probabilistic Regression

This repository provides Python implementations of **deterministic** and **stochastic uncertainty quantification (UQ) metrics** used for evaluating surrogate and machine learning models in safety-critical engineering systems.

The code accompanies the manuscript published in **IEEE Transactions on Industrial Informatics (2026)**.

---

## Associated Publication

**Nazia Raza**, Farrah Moazeni, PhD, Javad Khazaei, PhD  
*Uncertainty-Aware Surrogate Modeling for Safety-Critical Power Systems*  
IEEE Transactions on Industrial Informatics, 2026.

Affiliation:  
Lehigh University, Bethlehem, PA, USA

Contact:  
- nar522@lehigh.edu  

---

## Repository Contents

- `compute_deterministic_scores.py`  
  Computes deterministic regression metrics:
  - R²
  - Adjusted R²
  - Normalized RMSE
  - Weighted MAPE

- `compute_stochastic_scores.py`  
  Computes probabilistic uncertainty metrics assuming Gaussian predictive distributions:
  - Prediction Interval Coverage Probability (PICP)
  - Prediction Interval Width (PIW)
  - Normalized PIW (PINAW)
  - Predictive Entropy
  - Continuous Ranked Probability Score (CRPS)
  - Normalized CRPS

---

## Intended Use

These scripts are intended for:
- Evaluation of surrogate models
- Probabilistic regression benchmarking
- Uncertainty-aware modeling in safety-critical systems
- Reproducibility of reported metrics in the associated paper

The code supports **multi-output regression** and is framework-agnostic (compatible with PyTorch, TensorFlow, and scikit-learn outputs).

---

## Dependencies

The scripts require the following Python packages:

```text
numpy
scipy
pandas
