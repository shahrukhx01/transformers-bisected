import torch 
import torch.nn as nn
from transformer.transformer import TransformerBlock

class TransformerClassifier(nn.Module):
    def __init__(self, 
                vocab_size, 
                embed_size, 
                num_layers,
                heads,
                device,
                forward_expansion,
                dropout,
                max_length,
                out_size
                ):
        super(TransformerClassifier, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)
        self.max_length = max_length

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
         for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(max_length * embed_size, out_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))


        for layer in self.layers:
            out = layer(out, out, out)

        print(out.shape)
        out = out.reshape(N, self.max_length * embed_size) 
        out = self.fc_out(out)
        
        print(out.shape)
        out = self.sigmoid(out)

        return out
        

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 10
    embed_size = 256
    num_layers = 6
    forward_expansion = 4
    heads = 8
    dropout = 0.5
    max_length = 10
    out_size = 1
    transformer_classifier = TransformerClassifier(
                                                vocab_size, 
                                                embed_size, 
                                                num_layers, 
                                                heads, 
                                                device, 
                                                forward_expansion, 
                                                dropout, 
                                                max_length, 
                                                out_size )

    x = torch.tensor([[1, 2, 5, 2, 1, 2, 6, 8, 9, 0], [1, 2, 5, 2, 1, 2, 6, 7, 4, 1]])
    print(transformer_classifier(x))
    
