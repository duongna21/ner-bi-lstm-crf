import torch

import const
import utils


class Sentence:
    def __init__(self, sentence, word_vocab):
        self.length = len(sentence)
        self.word_indexes = []
        self.character_indexes = []
        self.pos_indexes = []
        self.chunk_indexes = []
        self.tag_indexes = []
        self.word_lengths = []

        self.get_value(sentence, word_vocab)

        self.padded_character_indexes_tensor = torch.tensor(utils.zero_padding(self.character_indexes,
                                                                               fill_value=const.CHARACTER2INDEX[
                                                                                   '<PAD>']),
                                                            dtype=torch.long,
                                                            device=const.DEVICE)
        self.word_lengths_tensor = torch.tensor(self.word_lengths,
                                                dtype=torch.long,
                                                device=const.DEVICE)

    def get_value(self, sentence, word_vocab):
        for word, pos, chunk, tag in sentence:
            self.word_indexes.append(word_vocab.word_to_index(word))
            self.character_indexes.append(self.word2indexes(word))
            self.pos_indexes.append(const.POS2INDEX[pos])
            self.chunk_indexes.append(const.CHUNK2INDEX[chunk])
            self.tag_indexes.append(const.TAG2INDEX[tag])
            self.word_lengths.append(len(word))

    @staticmethod
    def word2indexes(word):
        return [const.CHARACTER2INDEX.get(character, const.CHARACTER2INDEX['<UNK>']) for character in word]
