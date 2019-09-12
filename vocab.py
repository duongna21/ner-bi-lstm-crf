import torch
import torch.nn as nn
import torchtext
from fasttext import FastText


class Vocab:
    def __init__(self, path):
        self.model = FastText.load_model(path)

        self.embedding_dim = self.model.get_dimension()
        self.words = self.model.words
        self.word2index = {word: i for i, word in enumerate(self.words)}
        self.unk_index = len(self.words)
        self.padding_index = len(self.words) + 1
        self.embeddings = nn.Embedding.from_pretrained(torch.cat([torch.tensor(self.model.get_input_matrix()),
                                                                  torch.zeros(2, self.embedding_dim,
                                                                              dtype=torch.float)]),
                                                       padding_idx=self.padding_index,
                                                       freeze=True)

    def get_embedding(self):
        return self.embeddings

    def word_to_index(self, word):
        return self.word2index.get(word, self.unk_index)

