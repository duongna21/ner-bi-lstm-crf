import torch
import torch.nn as nn

import utils
import const


class CharacterLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(CharacterLSTM, self).__init__()
        self.num_characters = len(const.CHARACTER_LIST)
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_dim
        self.unk_idx = const.CHARACTER2INDEX['<UNK>']
        self.padding_idx = const.CHARACTER2INDEX['<PAD>']

        self.embeddings = nn.Embedding(num_embeddings=len(const.CHARACTER2INDEX), embedding_dim=embedding_dim,
                                       padding_idx=self.padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_size=hidden_dim // 2, num_layers=1, bidirectional=True)

    def forward_seq(self, seq, lengths, padded_length):
        seq_length = seq.size(1)

        # Padded embeddings
        embs = self.embeddings(seq)

        # Packed padded embeddings
        packed = nn.utils.rnn.pack_padded_sequence(embs, lengths=lengths, enforce_sorted=False)

        outputs, hidden = self.lstm(packed)

        # Padded packed outputs
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(outputs)

        outputs = torch.cat([outputs[0, :, :self.hidden_size // 2],
                             outputs[lengths - 1, range(seq_length), self.hidden_size // 2:]], dim=1)
        outputs = torch.cat([outputs,
                             torch.zeros(padded_length - seq_length, self.hidden_size, device=const.DEVICE)],
                            dim=0).view(padded_length, 1, self.hidden_size)
        return outputs

    def forward(self, batch_character_indexes_seq, batch_word_lengths, batch_sentence_lengths, padded_length):
        # outputs = torch.cat([self.forward_seq(seq, lengths, padded_length) for seq, lengths in batch_seq], dim=1)
        # return outputs
        num_words = batch_character_indexes_seq.size(1)

        # Padded embeddings
        embs = self.embeddings(batch_character_indexes_seq)

        # packed padded embeddings
        packed = nn.utils.rnn.pack_padded_sequence(embs, lengths=batch_word_lengths, enforce_sorted=False)

        # forward
        outputs, hidden = self.lstm(packed)

        # Padded packed outputs
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(outputs)

        outputs = torch.cat([outputs[0, :, :self.hidden_size // 2],
                             outputs[lengths - 1, range(num_words), self.hidden_size // 2:]], dim=1)
        outputs_sentences = torch.split(outputs, batch_sentence_lengths.tolist(), dim=0)

        outputs = torch.cat([torch.cat([outputs_sentence,
                                       torch.zeros(padded_length - outputs_sentence.size(0), self.hidden_size,
                                                   device=const.DEVICE)]).view(padded_length, 1, self.hidden_size)
                             for outputs_sentence in outputs_sentences], dim=1)
        return outputs
