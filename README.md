# Uncertainty-Aware Surrogate Modeling using Monte Carlo Dropout and Heteroscedastic Neural Network 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17955864.svg)](https://doi.org/10.5281/zenodo.17955864)

This repository provides Python implementations of a **Monte Carlo Dropout Neural Network** trained with a **Modified Heteroscedastic Loss Function** for uncertainty-aware surrogate modeling. 

The script folder contains a complete uncertainty decomposition framework, along with **deterministic** and **stochastic uncertainty quantification (UQ) metrics** for safety-critical engineering systems.

The code accompanies the research article *"Uncertainty Decomposition in Learning Nonlinear Stochastic Dynamics:
A Dropout-Based Surrogate Modeling Framework"* published in **IEEE Transactions on Industrial Informatics (2026)**.

---

## Associated Publication

**Nazia Raza**, Farrah Moazeni, PhD, Javad Khazaei, PhD  
*Uncertainty Decomposition in Learning Nonlinear Stochastic Dynamics:
A Dropout-Based Surrogate Modeling Framework*  
IEEE Transactions on Industrial Informatics, 2026.

Affiliation:  
Lehigh University, Bethlehem, PA, USA

Contact:  
- nar522@lehigh.edu  

---
## Repository Contents

- `loss_funcs.py`
  - Original heteroscedastic loss function
  - Modified heteroscedastic loss function
  - Beta-NLL loss function

- `Evaluation Scripts`  

  - `model_inference.py`  
    Computes MCD-HNN predictions:
    - Mean Prediction
    - Applies dropout at inference
    - Epistemic and aleatory uncertainty computation
      
  - `compute_deterministic_scores.py`  
    Computes deterministic regression performance metrics:
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
- Evaluation and benchmarking of surrogate models
- Probabilistic regression analysis
- Uncertainty-aware modeling in safety-critical systems
- Explainable AI 

The code supports **multi-output regression** and is framework-agnostic (compatible with PyTorch, TensorFlow, and scikit-learn outputs).

---
##   Simulation Dataset on Zenodo

A Monte Carlo simulation dataset of a naval shipboard microgrid is uploaded to **Zenodo** for benchmarking and reuse. 
> **Zenodo DOI:** 10.5281/zenodo.17955864

---
##  Citation

If you use this code or the dataset (available on Zenodo), please cite the associated IEEE Transactions on Industrial Informatics paper.
