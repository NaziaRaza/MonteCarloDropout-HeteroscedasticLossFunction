import pandas as pd

from Scripts.evaluation.deterministic_metrics import compute_deterministic_scores 

def compute_deterministic_scores_all(results, n_inputs):
    """
    Compute deterministic metrics for all realizations in the results list.

    Returns:
        - List of DataFrames (one per realization)
        - Combined DataFrame (realization x metric x output)
    """
    all_det_metrics = []

    for res in results:
        y_true = res['true'].cpu().numpy()
        y_pred = res['mean_prediction']

        df_metrics = compute_deterministic_scores(y_true, y_pred, n_inputs=n_inputs)
        all_det_metrics.append(df_metrics)

    # Stack into one long DataFrame with realization/output index
    combined_df = pd.concat(all_det_metrics, keys=range(len(all_det_metrics)), names=['Realization', 'Output'])

    return all_det_metrics, combined_df