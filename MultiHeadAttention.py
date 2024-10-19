import torch
import torch.nn as nn
import torch.nn.functional as F
import model_dimensions
from Head import Head 


embedding_dim = model_dimensions.embedding_dim
n_heads = model_dimensions.n_heads
head_dim = model_dimensions.head_dim
dropout= model_dimensions.dropout


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_dim) for _ in range(n_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, embeddings):

        output = torch.concat([head(embeddings) for head in self.heads], dim=-1) # concat along last dimension b/c the original embedding_dim is divided into n_heads times, each of size head_dim
        output = self.dropout(self.proj(output))

        return output