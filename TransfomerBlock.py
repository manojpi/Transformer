import torch
import torch.nn as nn
import torch.nn.functional as F
import model_dimensions
from MultiHeadAttention import MultiHeadAttention


context_length = model_dimensions.context_length
embedding_dim = model_dimensions.embedding_dim
n_heads = model_dimensions.n_heads
head_dim = model_dimensions.head_dim
dropout= model_dimensions.dropout


class FeedForward(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, embeddings):
        return self.ffn(embeddings)


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, head_dim)
        self.ffwd = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
    

    def forward(self, embeddings):
        output = embeddings + self.ln1(self.mha(embeddings))
        output = output + self.ln2(self.ffwd(output))

        return output