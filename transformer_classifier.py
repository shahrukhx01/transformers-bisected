import torch 
import torch.nn as nn
from transformer import TransformerBlock
from data import *
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

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
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))


        for layer in self.layers:
            out = layer(out, out, out)

        out = out.reshape(N, self.max_length * embed_size) 
        out = self.fc_out(out)
        
        out = self.softmax(out)

        return out
        

if __name__ == "__main__":
    DATA_PATH = './'
    TRAIN_FILE_NAME = 'train.tsv'
    max_length = 10
    batch_size = 128
    dataset = ReviewsDataset(DATA_PATH, TRAIN_FILE_NAME, max_length, batch_size)
    dataset.load_data() ## load data to memory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(dataset.TEXT.vocab.stoi)
    embed_size = 256
    num_layers = 4
    forward_expansion = 4
    heads = 8
    dropout = 0.5
    out_size = 6    

    train_dl = BatchWrapper(dataset.train_iterator, "Phrase", "Sentiment")
    
    model = TransformerClassifier(
                                                vocab_size, 
                                                embed_size, 
                                                num_layers, 
                                                heads, 
                                                device, 
                                                forward_expansion, 
                                                dropout, 
                                                max_length, 
                                                out_size ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = []
        y_true = list()
        y_pred = list()
        for data in tqdm(train_dl):
            # get the inputs; data is a list of [inputs, labels]
            
            inputs, labels = data
            inputs = inputs.T.to(device)
            labels = labels#.type(torch.FloatTensor)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            y_true += list(labels.data.int().detach().cpu().numpy()) ## accumulate targets from batch
            #print(torch.max(outputs, 1)[1])
            y_pred += list(torch.max(outputs, 1)[1].data.int().detach().cpu().numpy()) ## accumulate preds from batch 

            # print statistics
            running_loss.append(loss.item())
        acc = accuracy_score(y_true, y_pred) ## computing accuracy using sklearn's function
        print("Train loss: {} - acc: {}".format(torch.mean(torch.tensor(running_loss)), acc))
        
