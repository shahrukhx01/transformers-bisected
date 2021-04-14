import torch 
import torch.nn as nn
from transformer.transformer import TransformerBlock
from dataset.data import *
import torch.optim as optim

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

        #print(out.shape)
        out = out.reshape(N, self.max_length * embed_size) 
        out = self.fc_out(out)
        
        #print(out.shape)
        out = self.sigmoid(out)

        return out
        

if __name__ == "__main__":
    DATA_PATH = '~/Desktop/transformers-bisected/dataset/'
    TRAIN_FILE_NAME = 'train.tsv'
    max_length = 100
    batch_size = 32
    dataset = ReviewsDataset(DATA_PATH, TRAIN_FILE_NAME, max_length, batch_size)
    dataset.load_data() ## load data to memory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(dataset.TEXT.vocab.stoi)
    embed_size = 256
    num_layers = 6
    forward_expansion = 4
    heads = 8
    dropout = 0.5
    
    out_size = 1
    


    

    train_dl = BatchWrapper(dataset.train_iterator, "Phrase", "Sentiment")
    
    """for X, y in train_dl:
        print(X.shape)
        break"""
    model = TransformerClassifier(
                                                vocab_size, 
                                                embed_size, 
                                                num_layers, 
                                                heads, 
                                                device, 
                                                forward_expansion, 
                                                dropout, 
                                                max_length, 
                                                out_size )
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dl, 0):
            # get the inputs; data is a list of [inputs, labels]
            
            inputs, labels = data
            inputs = inputs.T
            labels = labels.type(torch.FloatTensor)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
           
            print(outputs.type(), labels.type())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        #x = torch.tensor([[1, 2, 5, 2, 1, 2, 6, 8, 9, 0], [1, 2, 5, 2, 1, 2, 6, 7, 4, 1]])
        #print(transformer_classifier(x))
        
