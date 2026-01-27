# Alexander Ye 

import torch
import numpy as np
import torch.nn as nn

class CVE(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        int_dim = int(np.sqrt(hidden_size))
        self.W1 = nn.Parameter(torch.empty(1, int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, hidden_size), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    def forward(self, x): # x is (batch_size, max_len)
        x = torch.unsqueeze(x, -1) # (batch_size, max_len, 1)
        x = x @ self.W1 + self.b1[None, None, :]
        x = self.activation(x)
        x = x @ self.W2
        return x

# class Transformer(nn.Module)

class STraTS(nn.Module):
    def __init__(self, num_features, d_model, hidden_size):
        super().__init__()
        # Time embed
        # Feature embed
        # Value embed
        # Transformer

        # Dropout