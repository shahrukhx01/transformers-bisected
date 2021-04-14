import torch 
import torch.nn as nn
from multihead_attention.self_attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, mask):
        attention = self.attention(queries, keys, values, mask)
        x = self.dropout(self.norm1(attention + queries))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out