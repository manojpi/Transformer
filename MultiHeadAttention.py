import torch
import torch.nn as nn
import torch.nn.functional as F
import model_dimensions


context_length = model_dimensions.context_length
embedding_dim = model_dimensions.embedding_dim
num_heads = model_dimensions.num_heads
head_dim = model_dimensions.head_dim
dropout= model_dimensions.dropout


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


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_dim):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, embeddings):

        output = torch.concat([head(embeddings) for head in self.heads], dim=-1) # concat along last dimension b/c the original embedding_dim is divided into n_heads times, each of size head_dim
        output = self.dropout(self.proj(output))

        return output