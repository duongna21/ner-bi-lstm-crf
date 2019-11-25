import torch
import torch.nn as nn
from fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
import math


class Vocab:
    def __init__(self, path, model_type='fasttext_cc', freeze=False):
        self.model = None
        self.embedding_dim = None
        self.words = None
        self.word2index = None
        self.unk_index = None
        self.padding_index = None
        self.embeddings = None

        assert model_type in ['fasttext_cc', 'gensim_fasttext']

        if model_type == 'fasttext_cc':
            self.load_pretrained_embedding_from_fasttext_cc_model(path, freeze)
        elif model_type == 'gensim_fasttext':
            self.load_pretrained_embedding_from_gensim_fasttext_model(path, freeze)

    def load_pretrained_embedding_from_fasttext_cc_model(self, path, freeze):
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
                                                                  torch.zeros(1, self.embedding_dim,
                                                                              dtype=torch.float)]),
                                                       padding_idx=self.padding_index,
                                                       freeze=freeze)

    def load_pretrained_embedding_from_gensim_fasttext_model(self, path, freeze):
        self.model = KeyedVectors.load_word2vec_format(path, binary=True)
        self.embedding_dim = self.model.vector_size
        self.words = self.model.index2word
        self.word2index = {word: i for i, word in enumerate(self.words)}
        self.unk_index = len(self.words)
        self.padding_index = len(self.words) + 1
        self.embeddings = nn.Embedding.from_pretrained(torch.cat([torch.tensor(self.model.vectors),
                                                                  torch.rand(1, self.embedding_dim,
                                                                             dtype=torch.float).uniform_(
                                                                      - math.sqrt(3 / self.embedding_dim),
                                                                      math.sqrt(3 / self.embedding_dim)),
                                                                  torch.zeros(1, self.embedding_dim,
                                                                              dtype=torch.float)]),
                                                       padding_idx=self.padding_index,
                                                       freeze=freeze)

    def get_embedding(self):
        return self.embeddings

    def word_to_index(self, word):
        return self.word2index.get(word, self.unk_index)
