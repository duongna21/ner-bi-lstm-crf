import torch
import torch.nn as nn

import utils
import const


class CharacterLSTM(nn.Module):
    def __init__(self, character_list, embedding_dim, hidden_dim):
        super(CharacterLSTM, self).__init__()
        self.num_characters = len(character_list)
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_dim
        self.unk_inx = self.num_characters
        self.padding_idx = self.num_characters + 1

        self.character2index = {character: i for i, character in enumerate(character_list)}

        self.embeddings = nn.Embedding(num_embeddings=self.num_characters + 2, embedding_dim=embedding_dim,
                                       padding_idx=self.padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_size=hidden_dim // 2, num_layers=1, bidirectional=True)

    def forward_seq(self, input_text_seq, padded_length):
        input_seq_length = len(input_text_seq)
        batch_word_indexes = [[self.character2index.get(character, self.unk_inx) for character in word]
                              for word in input_text_seq]
        padded_batch_word_indexes = torch.tensor(utils.zero_padding(batch_word_indexes, fill_value=self.padding_idx),
                                                 dtype=torch.long, device=const.DEVICE)
        lengths = torch.tensor([len(word) for word in input_text_seq], dtype=torch.long, device=const.DEVICE)
        embs = self.embeddings(padded_batch_word_indexes)
        packed = nn.utils.rnn.pack_padded_sequence(embs, lengths=lengths, enforce_sorted=False)
        outputs, hidden = self.lstm(packed)
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = torch.cat([outputs[0, :, :self.hidden_size // 2],
                             outputs[lengths - 1, range(input_seq_length), self.hidden_size // 2:]], dim=1)
        outputs = torch.cat([outputs,
                             torch.zeros(padded_length - input_seq_length, self.hidden_size, device=const.DEVICE)],
                            dim=0).view(padded_length, 1, self.hidden_size)
        return outputs

    def forward(self, batch_sentences):
        lengths = [len(sentence) for sentence in batch_sentences]
        max_length = max(lengths)

        outputs = torch.cat([self.forward_seq(sentence, max_length) for sentence in batch_sentences], dim=1)
        return outputs
