import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, LabelField
import spacy
import pandas as pd
import tqdm

class ReviewsDataset():
    def __init__(self, data_path, train_path, max_length, batch_size):

        self.max_length = max_length
        self.batch_size = batch_size

        ## write the tokenizer
        tokenize = lambda review : review.split()
        ## define your fields for ID filed you can use RAWField class
        self.TEXT  = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True, fix_length=max_length)
        self.LABEL  = LabelField()
        

        
        self.fields = [("PhraseId", None), # we won't be needing the id, so we pass in None as the field
                 ("SentenceId", None), ("Phrase", self.TEXT),
                 ("Sentiment", self.LABEL)] #{ 'Phrase': ('r', self.review), 'Sentiment': ('s', self.sentiment) }
        ## set paths
        self.data_path = data_path
        self.train_path = train_path

    def load_data(self):
        self.train_data = TabularDataset.splits(
            path='{}'.format(self.data_path),
            train='{}'.format(self.train_path),
            format='tsv',
            fields=self.fields)[0]
        
        self.TEXT.build_vocab(self.train_data, max_size=10000, min_freq=1)
        self.LABEL.build_vocab(self.train_data)
        self.train_iterator, _ = BucketIterator.splits((self.train_data, None), 
                                                    batch_sizes=(self.batch_size, self.batch_size), 
                                                    sort_within_batch=True,
                                                    sort_key=lambda x: len(x.Phrase))


    def __str__(self):
        return 'review: {} \n sentiment: {}'.format(self.train_data[0][0].__dict__['r'], self.train_data[0][0].__dict__['s'])


class BatchWrapper:
      def __init__(self, dl, x_var, y_var):
            self.dl, self.x_var, self.y_var = dl, x_var, y_var # we pass in the list of attributes for x 

      def __iter__(self):
            for batch in self.dl:
                  x = getattr(batch, self.x_var) # we assume only one input in this wrapper and one output
                  y =  getattr(batch, self.y_var)
                  yield (x, y)

      def __len__(self):
            return len(self.dl)

    