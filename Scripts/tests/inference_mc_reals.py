import torch
import os
import pandas as pd
from Scripts.evaluation.model_inference import *

def test_mcmlp_on_folder(folder_path, model, scaler,scaler_y, mc_samples=50, ci=0.90, device='cpu'):
    """
    Test MCMLP model across all Excel files in a folder.

    Parameters:
    - folder_path: Path to the folder containing Excel files.
    - model: Trained MCMLP model.
    - scaler: Scaler for input features.
    - mc_samples: Number of MC dropout forward passes.
    - ci: Confidence interval for prediction bounds.
    - device: 'cpu' or 'cuda'.

    Returns:
    - results: List of dictionaries with predictions and statistics.
    """
    results = []
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.xlsx')])

    for file in file_list:
        file_path = os.path.join(folder_path, file)
        data = pd.read_excel(file_path)

        # Prepare inputs and outputs
        # states at time t (drop last row to align with next-step targets)
        x_t = data.iloc[:-1, :10].reset_index(drop=True)

        # inputs at time t
        u_t = data.iloc[:-1][['CPL_Prop', 'PPL']].reset_index(drop=True)

        # constant Vref at time t
        vref = pd.Series(6000.0, index=x_t.index, name='Vref')

        # final inputs: [x_t, CPL_Prop_t, PPL_t, Vref]
        input_data = pd.concat([x_t, u_t, vref], axis=1)

        # targets are next-step states x_{t+1}
        output_data = data.iloc[1:, :10].reset_index(drop=True)

        input_scaled = torch.from_numpy(scaler.transform(input_data.values)).to(dtype=torch.float32).to(device)

        # Get MC Dropout predictions
        mean_pred, lb, ub, std_pred, epi_std, ale_std = get_mc_dropout_pred(
            model, scaler_y, input_scaled, mc_samples=mc_samples, ci=ci
        )

        results.append({
            'file': file,
            'mean_prediction': mean_pred,
            'true output': output_data,   
            'lower_bound': lb,
            'upper_bound': ub,
            'total_std': std_pred,
            'epistemic_std': epi_std,
            'aleatoric_std': ale_std
        })

    return results
