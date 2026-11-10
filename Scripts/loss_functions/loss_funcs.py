import torch

def heteroscedastic_loss(model, x, y_true):
    mu , log_var = model(x)
    
    precision = torch.exp(-log_var)

    loss = (0.5 * precision * (mu-y_true)**2) + 0.5*log_var

    loss = loss.mean()
    
    return loss

def beta_nll_loss(model, x, y_true, beta=0.5):
    """Compute beta-NLL loss
    
    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative 
        weighting between data points, where `0` corresponds to 
        high weight on low error points and `1` to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    mu , log_var = model(x)
    
    variance = torch.exp(log_var)
     
    loss = 0.5 * ((y_true - mu) ** 2 / variance + log_var)

    if beta > 0:
        loss = loss * (variance.detach() ** beta)
        loss = loss.sum(dim=-1)  # sum over output dimensions to get per-sample loss
    
    return loss.mean()

def mod_heteroscedastic_loss(model, x, y_true, mse_weights=None):
    mu , log_var = model(x)
     
    precision = torch.exp(-log_var)

    loss = 0.5 * precision * (mu-y_true)**2 + 0.5*log_var  + mse_weights*(mu-y_true)**2

    loss = loss.mean()
    
    return loss

def beta_nll_mod_loss(model, x, y_true, beta=0.5, mse_weights=None):
    """Compute beta-NLL loss
    
    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative 
        weighting between data points, where `0` corresponds to 
        high weight on low error points and `1` to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    mu , log_var = model(x)
    
    variance = torch.exp(log_var)
     
    loss = 0.5 * ((y_true - mu) ** 2 / variance + log_var)

    if beta > 0:
        loss = loss * (variance.detach() ** beta) * mse_weights
        # loss = loss.sum(dim=-1)  # sum over output dimensions to get per-sample loss
    
    return loss.mean()


