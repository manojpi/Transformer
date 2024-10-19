import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):

    def __init__(self, batch_size, d_model, dropout=0.2):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(batch_size, d_model)
        position = torch.arange(0, batch_size).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2)/d_model)
        
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        self.register_buffer('pe', pe)
        print(pe.shape)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)