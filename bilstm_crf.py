import torch
import torch.nn as nn
from torchcrf import CRF
import numpy as np

from character_lstm import CharacterLSTM
import utils
import const


class BiLSTMCrf(nn.Module):
    def __init__(self,
                 vocab,
                 character_embedding_dim,
                 character_hidden_dim,
                 context_hidden_dim,
                 using_pos_chunk=True,
                 dropout=0.35,
                 crf_loss_reduction='sum'):
        super(BiLSTMCrf, self).__init__()
        self.crf_loss_reduction = crf_loss_reduction
        self.vocab = vocab
        self.embedding_dim = vocab.embedding_dim
        self.num_tags = len(const.TAG_LIST) - 1
        self.context_hidden_dim = context_hidden_dim
        self.using_pos_chunk = using_pos_chunk

        self.embeddings = vocab.embeddings
        if using_pos_chunk:
            self.pos_embeddings = self.init_pos_embedding()
            self.chunk_embeddings = self.init_chunk_embedding()
        self.character_lstm = CharacterLSTM(character_embedding_dim, character_hidden_dim)
        self.context_lstm = nn.LSTM(
            self.embedding_dim + character_hidden_dim + (
                (const.NUM_POS_TAGS + const.NUM_CHUNK_TAGS) if using_pos_chunk else 0
            ),
            context_hidden_dim // 2,
            bidirectional=True)
        # self.hidden2tag = nn.Linear(self.context_hidden_dim, self.num_tags)
        self.hidden2tag = self.get_linear_layer(self.context_hidden_dim, self.num_tags)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(num_tags=self.num_tags)

    def forward(self, batch_padded_seq,
                batch_padded_pos_seq,
                batch_padded_chunk_seq,
                batch_character_indexes_seq,
                batch_sentence_lengths,
                batch_word_lengths,
                batch_padded_tags=None):
        padded_sentence_length = batch_sentence_lengths.max().item()
        # Mask padding position
        mask = (batch_padded_seq != self.vocab.padding_index).type(torch.uint8)

        # Word embeddings
        word_embs = self.embeddings(batch_padded_seq)

        # Character level presentation
        padded_batch_character_seq = self.character_lstm(batch_character_indexes_seq,
                                                         batch_word_lengths,
                                                         batch_sentence_lengths,
                                                         padded_sentence_length)

        if self.using_pos_chunk:
            # Pos embeddings
            pos_embs = self.pos_embeddings(batch_padded_pos_seq)

            # Chunk embeddings
            chunk_embs = self.chunk_embeddings(batch_padded_chunk_seq)

            # Combine word embs, pos embs, chunk embs, character level
            combined = torch.cat([word_embs, padded_batch_character_seq, pos_embs, chunk_embs], dim=2)
        else:
            # Combine word embs, character level
            combined = torch.cat([word_embs, padded_batch_character_seq], dim=2)
        combined = self.dropout(combined)

        packed = nn.utils.rnn.pack_padded_sequence(combined, lengths=batch_sentence_lengths, enforce_sorted=False)

        # Context outputs
        context_outputs, _ = self.context_lstm(packed)
        # self.dropout(context_outputs[0])
        context_outputs, _ = nn.utils.rnn.pad_packed_sequence(context_outputs)

        # Dropout
        context_outputs = self.dropout(context_outputs)

        out = self.hidden2tag(context_outputs)

        if batch_padded_tags is not None:
            return self.crf(out, batch_padded_tags, mask, reduction=self.crf_loss_reduction)
        else:
            return self.crf.decode(out, mask)

    def init_pos_embedding(self):
        weight = torch.cat([*[self.init_one_hot_tensor(const.NUM_POS_TAGS, i) for i in range(const.NUM_POS_TAGS)],
                            torch.zeros(1, const.NUM_POS_TAGS, dtype=torch.float, device=const.DEVICE)],
                           dim=0)
        return nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=const.POS_PADDING_IDX)

    def init_chunk_embedding(self):
        weight = torch.cat([*[self.init_one_hot_tensor(const.NUM_CHUNK_TAGS, i) for i in range(const.NUM_CHUNK_TAGS)],
                            torch.zeros(1, const.NUM_CHUNK_TAGS, dtype=torch.float, device=const.DEVICE)],
                           dim=0)
        return nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=const.CHUNK_PADDING_IDX)

    @staticmethod
    def init_one_hot_tensor(size, position):
        tensor = torch.zeros(1, size, dtype=torch.float, device=const.DEVICE)
        tensor[0][position] = 1
        return tensor

    @staticmethod
    def get_linear_layer(input_dim, output_dim):
        linear_layer = nn.Linear(input_dim, output_dim, bias=True)
        mean = 0.0  # std_dev = np.sqrt(variance)
        std_dev = np.sqrt(2 / (output_dim + input_dim))  # np.sqrt(1 / m) # np.sqrt(1 / n)
        weight = np.random.normal(mean, std_dev, size=(output_dim, input_dim)).astype(np.float32)
        std_dev = np.sqrt(1 / output_dim)  # np.sqrt(2 / (m + 1))
        bt = np.random.normal(mean, std_dev, size=output_dim).astype(np.float32)

        linear_layer.weight.data = torch.tensor(weight, requires_grad=True)
        linear_layer.bias.data = torch.tensor(bt, requires_grad=True)
        return linear_layer
