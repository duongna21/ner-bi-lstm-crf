import torch

import const
import utils


class Sentence:
    def __init__(self, sentence, word_vocab, character_vocab, pos2index, chunk2index):
        self.length = len(sentence)
        self.word_indexes = []
        self.character_indexes = []
        self.pos_indexes = []
        self.chunk_indexes = []
        self.word_lengths = []

        self.get_value(sentence, word_vocab, character_vocab, pos2index, chunk2index)

        self.padded_character_indexes_tensor = torch.tensor(utils.zero_padding(self.character_indexes,
                                                                               fill_value=character_vocab.padding_index),
                                                            dtype=torch.long,
                                                            device=const.DEVICE)
        self.word_lengths_tensor = torch.tensor(self.word_lengths,
                                                dtype=torch.long,
                                                device=const.DEVICE)

    def get_value(self, sentence, word_vocab, character_vocab, pos2index, chunk2index):
        for word, pos, chunk, tag in sentence:
            self.word_indexes.append(word_vocab.word2index(word))
            self.character_indexes.append(character_vocab.word2indexes(word))
            self.pos_indexes.append(pos2index[pos])
            self.chunk_indexes.append(chunk2index[chunk])
            self.word_lengths.append(len(word))
