import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import norm
import random

def set_seed(seed):
    'Sets the seed'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_mc_dropout_pred(model, scaler_y, x_input, mc_samples=50, ci=0.90):
    set_seed(0)
    model.eval()
    mean_list, log_var_list = [], []

    with torch.no_grad():
        for _ in range(mc_samples):
            output = model(x_input)
            if isinstance(output, tuple):
                mean, log_var = output
                mean_list.append(mean)
                log_var_list.append(log_var)     # log variance for heteroscedastic case, aleatoric uncertainty
            else:
                mean_list.append(output)

    mean_stack    = torch.stack(mean_list)           # [mc_samples, batch, output]
    mean_pred     = mean_stack.mean(dim=0)           # [batch, output]
    epistemic_var = mean_stack.var(dim=0, unbiased=False)            # [batch, output]

    if log_var_list:  # heteroscedastic case
        log_var_stack = torch.stack(log_var_list)             # [mc_samples, batch, output]
        aleatoric_var = torch.exp(log_var_stack).mean(dim=0)  # [batch, output]
        total_std     = torch.sqrt(epistemic_var + aleatoric_var)
    else:
        total_std     = torch.sqrt(epistemic_var)

    # Confidence Interval
    z     = norm.ppf(0.5 + ci / 2)
    lower = mean_pred - z * total_std
    upper = mean_pred + z * total_std

    # Inverse transform
    mean_np     = scaler_y.inverse_transform(mean_pred.cpu().numpy())
    lower_np    = scaler_y.inverse_transform(lower.cpu().numpy())
    upper_np    = scaler_y.inverse_transform(upper.cpu().numpy())

    # scale_range =1.0 / scaler_y.scale_
    scale_range = (scaler_y.data_max_ - scaler_y.data_min_)*0.5      
    total_std   = total_std.cpu().numpy() * scale_range
    epi_std     = torch.sqrt(epistemic_var).cpu().numpy() * scale_range
    ale_std     = torch.sqrt(aleatoric_var).cpu().numpy() * scale_range if log_var_list else None

    return mean_np, lower_np, upper_np, total_std, epi_std, ale_std

def get_confidence_bounds(ci, mean_pred, std):
    # Confidence Interval
    z     = norm.ppf(0.5 + ci / 2)
    lower = mean_pred - z * std
    upper = mean_pred + z * std
    return lower, upper
