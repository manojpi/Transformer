import torch
import torch.nn as nn
import torch.nn.functional as F
import model_dimensions


context_length = model_dimensions.context_length
embedding_dim = model_dimensions.embedding_dim


class Head(nn.Module):

    def __init__(self, head_dim):
        super().__init__()

        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)
        self.register('tril', torch.tril(torch.ones(context_length, context_length)))
    
    def forward(self, embeddings):

        B, T, C = embeddings.shape

        key = self.key(embeddings) # (B, T, C)
        query = self.query(embeddings) # (B, T, C)

        # compute the weigts or scores
        wei = query @ key.transpose(-2, -1) * C**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1)

        value = self.value(embeddings) # (B, T, C)
        output = wei @ value # (B, T, T) * (B, T, C) -> (B, T, C)

        return output