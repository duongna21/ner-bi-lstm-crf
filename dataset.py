import torch
from torch.utils.data import Dataset as TorchDataset
import itertools

import utils
import const


class TempDataset:
    def __init__(self,
                 sentences,
                 word_padding_idx,
                 pos_padding_idx,
                 chunk_padding_idx,
                 character_padding_idx,
                 tag_padding_idx):
        self.sentences = sentences
        self.word_padding_idx = word_padding_idx
        self.pos_padding_idx = pos_padding_idx
        self.chunk_padding_idx = chunk_padding_idx
        self.character_padding_idx = character_padding_idx
        self.tag_padding_idx = tag_padding_idx

    def __len__(self):
        return len(self.sentences)


class Dataset(TorchDataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.num_batches = (len(dataset) - 1) // batch_size + 1

        self.batches = []

        for i in range(self.num_batches):
            sentences = self.dataset.sentences[i * batch_size: (i + 1) * batch_size]
            self.batches.append(self.get_batch(sentences,
                                               self.dataset.word_padding_idx,
                                               self.dataset.pos_padding_idx,
                                               self.dataset.chunk_padding_idx,
                                               self.dataset.character_padding_idx,
                                               self.dataset.tag_padding_idx))

    @staticmethod
    def get_batch(sentences,
                  word_padding_idx,
                  pos_padding_idx,
                  chunk_padding_idx,
                  character_padding_idx,
                  tag_padding_idx):
        batch_sentence_word_indexes = utils.zero_padding([sentence.word_indexes for sentence in sentences],
                                                         fill_value=word_padding_idx)
        batch_sentence_pos_indexes = utils.zero_padding([sentence.pos_indexes for sentence in sentences],
                                                        fill_value=pos_padding_idx)
        batch_sentence_chunk_indexes = utils.zero_padding([sentence.chunk_indexes for sentence in sentences],
                                                          fill_value=chunk_padding_idx)
        batch_sentence_tag_indexes = utils.zero_padding([sentence.tag_indexes for sentence in sentences],
                                                        fill_value=tag_padding_idx)
        batch_lengths = [sentence.length for sentence in sentences]
        batch_sentence_word_character_indexes = utils.zero_padding(itertools.chain.from_iterable(
            [sentence.character_indexes for sentence in sentences]
        ), fill_value=character_padding_idx)
        batch_word_lengths = [len(characters) for sentence in sentences for characters in sentence.character_indexes]

        return ((torch.tensor(batch_sentence_word_indexes, dtype=torch.long, device=const.DEVICE),
                torch.tensor(batch_sentence_pos_indexes, dtype=torch.long, device=const.DEVICE),
                torch.tensor(batch_sentence_chunk_indexes, dtype=torch.long, device=const.DEVICE),
                torch.tensor(batch_sentence_word_character_indexes, dtype=torch.long, device=const.DEVICE)
                 ),
                torch.tensor(batch_sentence_tag_indexes, dtype=torch.long, device=const.DEVICE),
                torch.tensor(batch_lengths, dtype=torch.long, device=const.DEVICE),
                torch.tensor(batch_word_lengths, dtype=torch.long, device=const.DEVICE))

    def __iter__(self):
        batches = iter(self.batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        return self.batches[index]
