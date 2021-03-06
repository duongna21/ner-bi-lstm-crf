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

        self.embeddings = self.embedding(num_embeddings=len(const.CHARACTER2INDEX), embedding_dim=embedding_dim,
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

    @staticmethod
    def trunc_normal_(x: torch.Tensor, mean: float = 0., std: float = 1.) -> torch.Tensor:
        """Truncated normal initialization."""
        # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
        return x.normal_().fmod_(2).mul_(std).add_(mean)

    def embedding(self, num_embeddings: int, embedding_dim: int, padding_idx=0) -> nn.Module:
        """Create an embedding layer."""
        emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        # See https://arxiv.org/abs/1711.09160
        with torch.no_grad(): self.trunc_normal_(emb.weight, std=0.01)
        return emb

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

        # lengths = ((lengths - 1) >= 0).type(torch.long) * lengths
        outputs = torch.cat([outputs[0, :, :self.hidden_size // 2],
                             outputs[lengths - 1, range(num_words), self.hidden_size // 2:]], dim=1)
        outputs = outputs.view(-1, padded_length, self.hidden_size).transpose(0, 1)
        return outputs
