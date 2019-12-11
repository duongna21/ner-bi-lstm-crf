import torch
import torch.optim as optim
import os

import utils
import vocab
import const
from bilstm_crf import BiLSTMCrf
import dataset
import evaluator
from sentence import Sentence

args = {
    'pretrained_path': 'data/pretrained_embedding/fasttext_pretrained_embeddings_300.bin',
    'checkpoint_fn': 'data/model/200_300_token_mean/best_model.tar',
    'word_embedding_dim': 300,
    'character_embedding_dim': 100,
    'context_hidden_dim': 150 * 2,
    'character_hidden_dim': 100 * 2,
    'crf_loss_reduction': 'token_mean',
    'dropout': 0.35,
    'using_pos_chunk': False
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'gpu')


def load_model(
        model_fn,
        voc,
        character_embedding_dim,
        character_hidden_dim,
        context_hidden_dim,
        dropout,
        crf_loss_reduction,
        using_pos_chunk):
    model = BiLSTMCrf(vocab=voc,
                      character_embedding_dim=character_embedding_dim,
                      character_hidden_dim=character_hidden_dim,
                      context_hidden_dim=context_hidden_dim,
                      using_pos_chunk=using_pos_chunk,
                      dropout=dropout,
                      crf_loss_reduction=crf_loss_reduction)

    if model_fn is not None:
        checkpoint = torch.load(model_fn)
        model_sd = checkpoint['model']

        model.load_state_dict(model_sd)

    return model


def fill(text, width):
    text_len = len(text)
    head_space = (width - text_len) // 2
    tail_space = width - text_len - head_space
    return ''.join([' ' * head_space, text, ' ' * tail_space])


def print_result(sentences, tags):
    max_words_per_line = 15
    for sentence, tag in zip(sentences, tags):
        formated_sentence = []
        formated_tag = []
        for (token, _, _, _), tag_ in zip(sentence, tag):
            max_len = max(len(token), len(tag_)) + 4
            formated_sentence.append(fill(token, max_len))
            formated_tag.append(fill(tag_, max_len))

        no_lines = len(formated_sentence) // max_words_per_line + (
            0 if len(formated_sentence) % max_words_per_line == 0 else 1)

        for i in range(no_lines):
            print(' '.join(formated_sentence[max_words_per_line * i: max_words_per_line * (i + 1)]))
            print(' '.join(formated_tag[max_words_per_line * i: max_words_per_line * (i + 1)]))
            print('\n')


if __name__ == '__main__':
    voc = vocab.Vocab(args['pretrained_path'])
    model = load_model(
        model_fn=args['checkpoint_fn'],
        voc=voc,
        character_embedding_dim=args['character_embedding_dim'],
        character_hidden_dim=args['character_hidden_dim'],
        context_hidden_dim=args['context_hidden_dim'],
        dropout=args['dropout'],
        crf_loss_reduction=args['crf_loss_reduction'],
        using_pos_chunk=args['using_pos_chunk']
    )
    model = model.to(device)
    model.eval()

    is_stop = False
    while not is_stop:
        paragraph = input('Enter a paragraph: ')
        if paragraph == 'n' or paragraph == 'N':
            is_stop = True
        else:
            text_sentences = utils.get_sentences(paragraph)
            sentences = [Sentence(s, word_vocab=voc) for s in text_sentences]

            ds = dataset.Dataset(sentences, word_padding_idx=voc.padding_index,
                                 pos_padding_idx=const.POS_PADDING_IDX,
                                 chunk_padding_idx=const.CHUNK_PADDING_IDX,
                                 character_padding_idx=const.CHARACTER2INDEX['<PAD>'],
                                 tag_padding_idx=const.CHUNK_PADDING_IDX)
            dl = dataset.DataLoader(ds, batch_size=len(sentences))

            for ((batch_sentence_word_indexes,
                  batch_sentence_pos_indexes,
                  batch_sentence_chunk_indexes,
                  batch_sentence_word_character_indexes),
                 batch_sentence_tag_indexes,
                 batch_sentence_lengths,
                 batch_word_lengths) in dl:
                pred_seqs = model(batch_sentence_word_indexes,
                                  batch_sentence_pos_indexes,
                                  batch_sentence_chunk_indexes,
                                  batch_sentence_word_character_indexes,
                                  batch_sentence_lengths,
                                  batch_word_lengths,
                                  None)

                # for i, (pred_seq, length) in enumerate(zip(pred_seqs, batch_sentence_lengths.tolist())):
                #     label_true.extend(batch_sentence_tag_indexes[:length, i].tolist())
                #     label_pred.extend(pred_seq)
                tags = [
                    [const.TAG_LIST[i] for i in tags_sent] for tags_sent in pred_seqs
                ]

                print_result(text_sentences, tags)
