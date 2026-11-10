from Scripts.evaluation.probabilistic_metrics import compute_stochastic_scores 

def compute_stochastic_scores_all(results, confidence_levels):
    """
    Compute stochastic metrics for all realizations already stored in `results`.

    Parameters:
        - results: list of dictionaries, each with keys:
              'file', 'mean_prediction', 'lower_bound', 'upper_bound', 
              'total_std', 'epistemic_std', 'aleatoric_std'
        - confidence_levels: list of floats (e.g., [0.9, 0.95])

    Assumes each result dict has:
        - 'true': torch.Tensor of shape [N, D]
        - 'mean_prediction': torch.Tensor of shape [N, D]
        - 'total_std': torch.Tensor of shape [N, D]

    Returns:
        - metrics_dict: dictionary with keys = metric names, values = list over realizations
    """
    metrics_dict = {
        'PICP': [], 'PIW': [], 'PINAW': [], 'Entropy': [], 'CRPS': [], 'CRPS_Norm': []
    }

    for res in results:
        y_true     = res['true'].cpu().numpy()
        mean_preds = res['mean_prediction']
        std_preds  = res['total_std']

        picp, piw, pinaw, entropy, crps, crps_norm = compute_stochastic_scores(
            y_true=y_true,
            mean_preds=mean_preds,
            std_preds=std_preds,
            confidence_levels=confidence_levels
        )

        metrics_dict['PICP'].append(picp)
        metrics_dict['PIW'].append(piw)
        metrics_dict['PINAW'].append(pinaw)
        metrics_dict['Entropy'].append(entropy)
        metrics_dict['CRPS'].append(crps)
        metrics_dict['CRPS_Norm'].append(crps_norm)

    return metrics_dict
