import torch
import nltk
import pandas
import math
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.test.utils import common_texts
import gensim 
from torch import nn
#nltk.download('punkt')  ## uncomment if running for the first time

class PositionalEmbedding:
    def __init__(self, embedding_dimensions, data_path):
        self.embedding_dimensions = embedding_dimensions
        self.data = self.read_data(data_path)
        self.word_embeddings = self.create_embeddings()
    
    def read_data(self, data_path):
        f = open(data_path, "r").read() 

        data = [] 
        # sentence parsing 
        for i in sent_tokenize(f): 
            temp = [] 
            # tokenize the sentence into words 
            for j in word_tokenize(i): 
                temp.append(j.lower()) 
            data.append(temp)
        return data
    
    def create_embeddings(self):
        # create Skip Gram model 
        """
        model = gensim.models.Word2Vec(sentences=self.data, min_count= 1, vector_size=self.embedding_dimensions, window = 5, sg = 1)  
        model.train(self.data, total_examples=3, epochs=10000)      
        model.save("word2vec.model")
        """
        model = gensim.models.Word2Vec.load("word2vec.model")
        return model

    def get_vanilla_word_emb(self, word):
        return torch.tensor(self.word_embeddings.wv[word], dtype=torch.float)
    
    def get_vanilla_pos_emb(self, position):
        ## init an empty array equivalent to embedding dimension 
        positional_embedding = torch.arange(0, self.embedding_dimensions, dtype=torch.float)

        ## iterate over all dimensions of the embedding vector
        for i in range(0, self.embedding_dimensions,2):
                ## for even dimension in embedding vector use sine transform
                positional_embedding[i] = math.sin(position / (10000 ** ((2 * i)/self.embedding_dimensions)))

                ## for odd dimension in embedding vector use cosine transform
                positional_embedding[i+1] = math.cos(position / (10000 ** ((2 * i)/self.embedding_dimensions)))
        
        return positional_embedding
    
    def get_positional_word_embedding(self, word_emb, positional_emb):
        word_emb *= math.sqrt(self.embedding_dimensions)
        return word_emb + positional_emb

    @staticmethod
    def compute_similarity(embedding1, embedding2):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        return cos(embedding1, embedding2)

if __name__ == "__main__":
    positional_embed = PositionalEmbedding(embedding_dimensions=100, data_path="positional_embedding/text.txt")
    ## lets choose the following sequence, I have annotated the position of words for latter reference
    ## "The(1) black(2) cat(3) sat(4) on(5) the(6) couch(7) and(8) the(9) brown(10) dog(11) slept(12) on(13) the(14) rug.(15)" from the corpus 'text.txt'
    
    ## and get word embeddings of "brown" and "black"
    black_word_emb = positional_embed.get_vanilla_word_emb("brown")
    brown_word_emb = positional_embed.get_vanilla_word_emb("black")

    print('similarity based on word embedding')
    print(PositionalEmbedding.compute_similarity(black_word_emb, brown_word_emb))

    ## and get word embeddings of "brown" and "black"
    black_pos_emb = positional_embed.get_vanilla_pos_emb(2)
    brown_pos_emb = positional_embed.get_vanilla_pos_emb(10)
    
    print('similarity based on position')
    print(PositionalEmbedding.compute_similarity(black_pos_emb, brown_pos_emb))

    ## and get positional word embeddings of "brown" and "black"
    black_pos_word_emb = positional_embed.get_positional_word_embedding(black_word_emb, black_pos_emb)
    brown_pos_word_emb = positional_embed.get_positional_word_embedding(brown_word_emb, brown_pos_emb)

    print('similarity based on positional word emb')
    print(PositionalEmbedding.compute_similarity(black_pos_word_emb, brown_pos_word_emb))