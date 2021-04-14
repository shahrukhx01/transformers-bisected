import torch 
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dimension = embed_size // heads

        assert (self.head_dimension * self.heads  == self.embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.keys = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.queries = nn.Linear(self.head_dimension, self.head_dimension, bias=False)

        self.fc = nn.Linear(self.head_dimension * self.heads, self.embed_size)
    

    def forward(self, queries, keys, values, mask):
        N = query.shape[0]
        key_len, query_len, value_len = keys.shape[1], queries.shape[1], values.shape[1]

        ## split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dimension) 
        keys = keys.reshape(N, key_len, self.heads, self.head_dimension) 
        query = query.reshape(N, query_len, self.heads, self.head_dimension) 

       ## get attention based keys, queries and values
       keys = self.keys(keys)
       values = self.values(values)
       queries = self.queries(queries)

      ## now perform Q.K step from the paper
      ## queries shape: (N, query_len, heads, head_dimension)
      ## keys shape: (N, key_len, heads, head_dimension)
      ## energy shape: (N, heads, query_len, key_len)

      energy = torch.einsum("nqhd, nkhq-> nhqk")

      if mask is not None:
          energy = energy.masked_fill(mask==0, float("-1e20"))

     ## attention = softmax(QK/sqrt(dk))
     attention = torch.softmax(energy / (self.embed_size**(1/2)))
    
     ## get final attention
     ## attention shape: (N, heads, query_len, key_len)
     ## values shape: (N, value_len, heads, head_dimension)
     
    
     
