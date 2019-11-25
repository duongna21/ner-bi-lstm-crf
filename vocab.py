import torch
import torch.nn as nn
from fasttext import FastText
import math


class Vocab:
    def __init__(self, path, freeze=False):
        self.model = FastText.load_model(path)

        self.embedding_dim = self.model.get_dimension()
        self.words = self.model.words
        self.word2index = {word: i for i, word in enumerate(self.words)}
        self.unk_index = len(self.words)
        self.padding_index = len(self.words) + 1
        self.embeddings = nn.Embedding.from_pretrained(torch.cat([torch.tensor(self.model.get_input_matrix()),
                                                                  torch.rand(1, self.embedding_dim,
                                                                             dtype=torch.float).uniform_(
                                                                      - math.sqrt(3 / self.embedding_dim),
                                                                      math.sqrt(3 / self.embedding_dim)),
                                                                  torch.zeros(1, self.embedding_dim, dtype=torch.float)]),
                                                       padding_idx=self.padding_index,
                                                       freeze=freeze)

    def get_embedding(self):
        return self.embeddings

    def word_to_index(self, word):
        return self.word2index.get(word, self.unk_index)
