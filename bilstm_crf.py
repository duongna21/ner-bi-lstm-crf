import torch
import torch.nn as nn
from torchcrf import CRF

from character_lstm import CharacterLSTM
import utils
import const


class BiLSTMCrf(nn.Module):
    def __init__(self,
                 vocab,
                 character_list,
                 character_embedding_dim,
                 character_hidden_dim,
                 context_hidden_dim,
                 tag_list):
        super(BiLSTMCrf, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.vocab_size
        self.embedding_dim = vocab.embedding_dim
        self.num_tags = len(tag_list)
        self.tag2index = {tag: i for i, tag in enumerate(tag_list)}
        self.index2tag = tag_list
        self.padding_tag_idx = 0
        self.context_hidden_dim = context_hidden_dim

        self.embeddings = vocab.embeddings
        self.character_lstm = CharacterLSTM(character_list, character_embedding_dim,
                                            character_hidden_dim)
        self.context_lstm = nn.LSTM(self.embedding_dim + character_hidden_dim, context_hidden_dim // 2,
                                    bidirectional=True)
        self.hidden2tag = nn.Linear(self.context_hidden_dim, self.num_tags)
        self.dropout = nn.Dropout(0.5)
        self.crf = CRF(num_tags=self.num_tags)

    def forward(self, batch_text_seq, batch_tags):
        batch_seq = [[self.vocab.word_to_index(word=word) for word in sentence] for sentence in batch_text_seq]
        target = [[self.tag2index[tag] for tag in tags] for tags in batch_tags]
        padded_seq = torch.tensor(utils.zero_padding(batch_seq, fill_value=self.vocab.padding_index),
                                  device=const.DEVICE)
        padded_tags = torch.tensor(utils.zero_padding(target, fill_value=self.padding_tag_idx), device=const.DEVICE)
        mask = (padded_seq != self.vocab.padding_index).type(torch.uint8)

        lengths = torch.tensor([len(sentence) for sentence in batch_text_seq], dtype=torch.long, device=const.DEVICE)
        embs = self.embeddings(padded_seq)

        embs = self.dropout(embs)
        padded_batch_character_seq = self.character_lstm(batch_text_seq)
        padded_batch_character_seq = self.dropout(padded_batch_character_seq)

        combined = torch.cat([embs, padded_batch_character_seq], dim=2)

        packed = nn.utils.rnn.pack_padded_sequence(combined, lengths=lengths, enforce_sorted=False)

        context_outputs, _ = self.context_lstm(packed)
        context_outputs, _ = nn.utils.rnn.pad_packed_sequence(context_outputs)
        # context_outputs = context_outputs[:, :, :self.context_hidden_dim] + context_outputs[:, :,
        #                                                                     self.context_hidden_dim:]

        # Dropout
        context_outputs = self.dropout(context_outputs)

        out = self.hidden2tag(context_outputs)

        return self.crf(out, padded_tags, mask, reduction='sum')

    def decode(self, batch_text_seq):
        batch_seq = [[self.vocab.word_to_index(word=word) for word in sentence] for sentence in batch_text_seq]
        padded_seq = torch.tensor(utils.zero_padding(batch_seq, fill_value=self.vocab.padding_index),
                                  dtype=torch.long,
                                  device=const.DEVICE)
        mask = (padded_seq != self.vocab.padding_index).type(torch.uint8)
        lengths = torch.tensor([len(sentence) for sentence in batch_text_seq], dtype=torch.long, device=const.DEVICE)

        embs = self.embeddings(padded_seq)
        padded_batch_character_seq = self.character_lstm(batch_text_seq)
        combined = torch.cat([embs, padded_batch_character_seq], dim=2)
        packed = nn.utils.rnn.pack_padded_sequence(combined, lengths=lengths, enforce_sorted=False)

        context_outputs, _ = self.context_lstm(packed)
        context_outputs, _ = nn.utils.rnn.pad_packed_sequence(context_outputs)
        # context_outputs = context_outputs[:, :, :self.context_hidden_dim] + context_outputs[:, :, self.context_hidden_dim:]

        context_outputs = self.hidden2tag(context_outputs)
        batch_tag_indexes = self.crf.decode(context_outputs, mask)
        return [[self.index2tag[index] for index in tag_indexes] for tag_indexes in batch_tag_indexes]
