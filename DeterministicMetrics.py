####################################################################################################################
# Code for uncertainty quantification (UQ) metrics
#
# Associated with:
# "Uncertainty Disentanglement: A Surrogate Modeling Framework
# Bridging Monte Carlo Dropout and Heteroscedastic Neural Network"
# IEEE Transactions on Industrial Informatics, 2026
#
# Authors:
# Nazia Raza (Lehigh University)        | nar522@lehigh.edu
# Farrah Moazeni, PhD (Lehigh University)
# Javad Khazaei, PhD (Lehigh University)
#
# Description:
# This script evaluates deterministic predictive performance of regression models.
#
# The metrics include:
#   - R²: Coefficient of determination
#   - Adjusted R²: R² adjusted for model complexity
#   - Normalized RMSE: Root mean squared error normalized
#     by the mean absolute value of the target
#   - Weighted MAPE: Weighted mean absolute percentage error
#
# License:
# Released under CC BY 4.0 for research and academic use.
####################################################################################################################

import pandas as pd
import numpy as np

def compute_deterministic_scores(y_true, y_pred, n_inputs):
    """
    Compute deterministic regression performance metrics.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples, n_outputs)
        Ground-truth target values.

    y_pred : np.ndarray, shape (n_samples, n_outputs)
        Deterministic model predictions (e.g., predictive mean).

    n_inputs : int
        Number of input features used by the regression model.
        Required for computing the adjusted R² metric.

    Returns
    -------
    det_metrics : pandas.DataFrame
        DataFrame containing deterministic performance metrics
        computed independently for each output variable:
            - R²: Coefficient of determination
            - Adjusted R²: R² adjusted for model complexity
            - Normalized RMSE: Root mean squared error normalized
              by the mean absolute value of the target
            - Weighted MAPE: Weighted mean absolute percentage error
    """

    n_samples = y_true.shape[0]
    r2              = 1 - ((y_true - y_pred)**2).sum(axis=0)/ ((y_true - y_true.mean(axis=0))**2).sum(axis=0) 
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_inputs - 1)
    nrmse           = np.sqrt(((y_true - y_pred)**2).mean(axis=0)) / np.abs(y_true).mean(axis=0)
    weighted_mape   = (np.abs(y_true - y_pred).sum(axis=0) / np.abs(y_true).sum(axis=0))
   
    det_metrics = pd.DataFrame([r2, adjusted_r2, nrmse, weighted_mape]).T
    det_metrics.columns = ['R²', 'Adjusted R²', 'Normalized RMSE', 'Weighted MAPE'] 
    
    return det_metrics
