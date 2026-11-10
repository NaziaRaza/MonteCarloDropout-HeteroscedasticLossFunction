import torch
import torch.nn as nn
import torch.nn.functional as F

class MC_DropoutMLP(nn.Module):
    
    def __init__(self, input_features, output_features, layer_sizes, dropout_rate=0.1, loss_type="mse"):
        super(MC_DropoutMLP, self).__init__()

        self.layers = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.loss_type = loss_type
        self.output_features = output_features

        # Input layer
        self.layers.append(nn.Linear(input_features, layer_sizes[0]))

        # Hidden layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        # Output layer
        if self.loss_type == "heteroscedastic":
            self.output_layer = nn.Linear(layer_sizes[-1], 2 * output_features)  # mean + log_var
        else:
            self.output_layer = nn.Linear(layer_sizes[-1], output_features)

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
            x = F.dropout(x, p=self.dropout_rate, training=True)   # Apply dropout during training and inference

        x = self.output_layer(x)

        if self.loss_type == "heteroscedastic":
            mu, log_var = torch.chunk(x, 2, dim=-1)
            mu = torch.tanh(mu)  # Output is in the range [-1, 1]  
            return mu, log_var
        else:
            return torch.tanh(x)  # Output is in the range [-1, 1]; standard 'MLP' case  