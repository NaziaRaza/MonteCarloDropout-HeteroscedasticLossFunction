####################################################################################################################
# Code for uncertainty quantification (UQ) metrics
#
# Associated with:
# "Uncertainty Disentanglement: A Surrogate Modeling Framework
#  Bridging Monte Carlo Dropout and Heteroscedastic Neural Network"
# IEEE Transactions on Industrial Informatics, 2026
#
# Authors:
# Nazia Raza (Lehigh University)        | nar522@lehigh.edu
# Farrah Moazeni, PhD (Lehigh University)
# Javad Khazaei, PhD (Lehigh University)
#
# Description:
# This script computes standard stochastic uncertainty quantification metrics
# for probabilistic regression models assuming Gaussian predictive distributions.
#
# The metrics include:
#   - Prediction Interval Coverage Probability (PICP)
#   - Prediction Interval Width (PIW)
#   - Prediction Interval Normalized Average Width (PINAW)
#   - Predictive Entropy
#   - Continuous Ranked Probability Score (CRPS)
#   - Normalized CRPS
#
# License:
# Released under CC BY 4.0 for research and academic use.
####################################################################################################################


import numpy as np
from scipy.stats import norm

def compute_stochastic_scores(y_true, mean_preds, std_preds, confidence_levels):
    """
    Compute probabilistic uncertainty quantification metrics for Gaussian predictions.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples, n_outputs)
        Ground-truth target values for each output variable.

    mean_preds : np.ndarray, shape (n_samples, n_outputs)
        Predictive mean values output by the surrogate or probabilistic model.

    std_preds : np.ndarray, shape (n_samples, n_outputs)
        Predictive standard deviations (aleatory, epistemic, or total).
        Assumes a Gaussian predictive distribution.

    confidence_levels : list or np.ndarray
        Confidence levels for prediction intervals (e.g., [0.90, 0.95, 0.99]).

    Returns
    -------
    picp : list of lists
        Prediction Interval Coverage Probability for each output and confidence level.
        Shape: [n_outputs][len(confidence_levels)]

    piw : list of lists
        Average Prediction Interval Width for each output and confidence level.

    pinaw : list of lists
        Normalized Prediction Interval Width (normalized by output range).

    entropy : list
        Mean differential entropy of the Gaussian predictive distribution
        for each output variable.

    crps : list
        Mean Continuous Ranked Probability Score (CRPS) per output.

    crps_norm : list
        CRPS normalized by the range of the true output variable.

    Notes
    -----
    - All metrics assume independent Gaussian predictive distributions.
    - Normalization is performed using the empirical range of y_true per output.
    - This function supports multi-output regression.
    """

    n_samples, n_outputs   =  mean_preds.shape
    assert std_preds.shape == (n_samples, n_outputs),     "Shape mismatch for std_preds"
    assert y_true.shape    == (n_samples, n_outputs),     "Shape mismatch for y_true"

    picp    = []
    piw     = []
    pinaw   = []
    entropy = []
    crps    = []
    crps_norm = []
    
    # For each output variable
    for output_idx in range(n_outputs):
        picp_scores    = []
        piw_scores     = []
        pinaw_scores   = []

        #  For each level of confidence
        for conf in confidence_levels:
            z_score = norm.ppf(0.5 + conf/2)

            lower_bounds = mean_preds[:, output_idx] - z_score * std_preds[:, output_idx]
            upper_bounds = mean_preds[:, output_idx] + z_score * std_preds[:, output_idx]

            # perdiction interval coverage probability (PICP)
            covered = np.logical_and(y_true[:, output_idx] >= lower_bounds, y_true[:, output_idx] <= upper_bounds)
            observed_coverage = np.mean(covered)
            picp_scores.append(observed_coverage)
            
            # prediction interval width (PIW)
            range_y             = y_true[:, output_idx].max() - y_true[:, output_idx].min() 
            piw_scores_observed = (upper_bounds-lower_bounds).sum()/(y_true.shape[0])
            pinaw_scores_observed = piw_scores_observed / range_y 
            piw_scores.append(piw_scores_observed)
            pinaw_scores.append(pinaw_scores_observed)

        # Append for all outputs 
        picp.append(picp_scores)
        piw.append(piw_scores)
        pinaw.append(pinaw_scores)

        # Entropy for gaussian distribution
        entropy_scores_observed = 0.5* np.log(2*np.pi*np.exp(1)*std_preds[:, output_idx]**2).mean(axis=0)    

        # Continuous ranked probability score (CRPS) for gaussian distribuiton
        w = (y_true[:, output_idx] - mean_preds[:, output_idx])/std_preds[:, output_idx]
        crps_scores_observed = std_preds[:, output_idx] * (w * (2 * norm.cdf(w) -1) + 2 * norm.pdf(w) - np.pi**(-0.5)) 
        crps_scores_norm     = std_preds[:, output_idx] * (w * (2 * norm.cdf(w) -1) + 2 * norm.pdf(w) - np.pi**(-0.5)) / range_y

        entropy.append(entropy_scores_observed)
        crps.append(crps_scores_observed.mean(axis=0))
        crps_norm.append(crps_scores_norm.mean(axis=0))
        
    return picp, piw, pinaw, entropy, crps, crps_norm