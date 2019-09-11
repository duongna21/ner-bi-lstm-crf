import torch
import torch.nn as nn
import torchtext
from gensim.models.fasttext import FastText


class Vocab:
    def __init__(self, path):
        self.model = FastText.load(path)
        self.index2word = self.model.wv.index2word
        self.word2index = {word: i for i, word in enumerate(self.index2word)}
        self.num_words = len(self.index2word)

        self.unk_index = self.num_words
        self.padding_index = self.num_words + 1
        self.embedding_dim = self.model.vector_size
        self.embeddings = nn.Embedding.from_pretrained(
            torch.cat([torch.tensor(self.model.wv.vectors), torch.zeros(2, self.embedding_dim)]),
            padding_idx=self.padding_index, freeze=True)
        self.vocab_size = self.num_words + 2

    def get_embedding(self):
        return self.embeddings

    def word_to_index(self, word):
        return self.word2index.get(word, self.unk_index)

    def sentence2indexes(self, sentence):
        return torch.tensor([self.word_to_index(word) for word in sentence])
