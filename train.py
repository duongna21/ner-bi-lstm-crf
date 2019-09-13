import torch
import torch.optim as optim
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

import utils
import vocab
import const
from bilstm_crf import BiLSTMCrf
from conlleval import evaluate
import dataset
import evaluator
from sentence import Sentence


def adjust_learning_rate(optimizer, learning_rate, epoch, learning_rate_decay):
    lr = learning_rate / (1 + learning_rate_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, optimizer, train_dl, dev_dl, test_dl, epochs, start_epoch, all_losses, eval_losses, save_every,
          save_dir):
    for epoch in range(start_epoch, epochs + 1):
        adjust_learning_rate(optimizer, learning_rate, epoch, learning_rate_decay)
        model.train()
        current_loss = 0
        for ((batch_sentence_word_indexes,
              batch_sentence_pos_indexes,
              batch_sentence_chunk_indexes,
              batch_sentence_word_character_indexes),
             batch_sentence_tag_indexes,
             batch_sentence_lengths,
             batch_word_lengths) in tqdm(train_dl):
            optimizer.zero_grad()
            loss = - model(batch_sentence_word_indexes,
                           batch_sentence_pos_indexes,
                           batch_sentence_chunk_indexes,
                           batch_sentence_word_character_indexes,
                           batch_sentence_lengths,
                           batch_word_lengths,
                           batch_sentence_tag_indexes)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

        epoch_loss = current_loss / len(train_dl)
        all_losses.append(epoch_loss)
        eval_losses.append(evaluator.evaluate_loss(model, dev_dl))
        test_precision, test_recall, test_f1_score = evaluator.evaluate_test(model, test_dl)
        print(f"Epoch {epoch}: \tloss = {epoch_loss}"
              f"\neval_loss = {eval_losses[-1]}"
              f"\ntest_precision = {test_precision}"
              f"\ntest_recall = {test_recall}"
              f"\ntest_f1_score = {test_f1_score}")
        print('-----------------------------------------------')

        if epoch % save_every == 0:
            directory = os.path.join(save_dir, f'{character_hidden_dim}_{context_hidden_dim}_{crf_loss_reduction}')
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'all_losses': all_losses,
                'eval_losses': eval_losses
            }, os.path.join(directory, f"{epoch}_checkpoint.tar"))

        if is_stopping(eval_losses):
            break


def is_stopping(eval_losses):
    best_eval_losses = min(eval_losses)
    return min(eval_losses[-10:]) > best_eval_losses


def load_model(model_fn, voc, character_embedding_dim, character_hidden_dim, context_hidden_dim, dropout,
               crf_loss_reduction):
    print('Building model and optimizer...')
    epoch = 0
    all_losses = []
    eval_losses = []

    model = BiLSTMCrf(vocab=voc,
                      character_embedding_dim=character_embedding_dim,
                      character_hidden_dim=character_hidden_dim,
                      context_hidden_dim=context_hidden_dim,
                      dropout=dropout,
                      crf_loss_reduction=crf_loss_reduction)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if model_fn is not None:
        checkpoint = torch.load(model_fn)
        epoch = checkpoint['epoch']
        model_sd = checkpoint['model']
        optimizer_sd = checkpoint['optimizer']
        all_losses = checkpoint['all_losses']
        eval_losses = checkpoint['eval_losses']

        model.load_state_dict(model_sd)
        optimizer.load_state_dict(optimizer_sd)

    model = model.to(const.DEVICE)
    for state in optimizer.state.values():
        for k, v in state.items():
            if type(v) is torch.Tensor:
                state[k] = v.to(const.DEVICE)

    return model, optimizer, epoch + 1, all_losses, eval_losses


if __name__ == '__main__':
    print('Loading vocab...')
    voc = vocab.Vocab('data/pretrained_embedding/fasttext_pretrained_embedding_dim300.bin')
    batch_size = 8

    print('Loading data ...')
    train_sentences = [Sentence(sentence, voc) for sentence in utils.read_data('data/data/train.txt')]
    dev_sentences = [Sentence(sentence, voc) for sentence in utils.read_data('data/data/dev.txt')]
    test_sentences = [Sentence(sentence, voc) for sentence in utils.read_data('data/data/test.txt')]

    train_ds = dataset.Dataset(train_sentences, word_padding_idx=voc.padding_index,
                               pos_padding_idx=const.POS_PADDING_IDX,
                               chunk_padding_idx=const.CHUNK_PADDING_IDX,
                               character_padding_idx=const.CHARACTER2INDEX['<PAD>'],
                               tag_padding_idx=const.CHUNK_PADDING_IDX)
    dev_ds = dataset.Dataset(dev_sentences, word_padding_idx=voc.padding_index,
                             pos_padding_idx=const.POS_PADDING_IDX,
                             chunk_padding_idx=const.CHUNK_PADDING_IDX,
                             character_padding_idx=const.CHARACTER2INDEX['<PAD>'],
                             tag_padding_idx=const.CHUNK_PADDING_IDX)
    test_ds = dataset.Dataset(test_sentences, word_padding_idx=voc.padding_index,
                              pos_padding_idx=const.POS_PADDING_IDX,
                              chunk_padding_idx=const.CHUNK_PADDING_IDX,
                              character_padding_idx=const.CHARACTER2INDEX['<PAD>'],
                              tag_padding_idx=const.CHUNK_PADDING_IDX)

    train_dl = dataset.DataLoader(train_ds, batch_size=batch_size)
    dev_dl = dataset.DataLoader(dev_ds, batch_size=batch_size)
    test_dl = dataset.DataLoader(test_ds, batch_size=batch_size)

    word_embedding_dim = 300
    character_embedding_dim = 100
    character_hidden_dim = 100
    context_hidden_dim = 150
    learning_rate = 0.0035
    learning_rate_decay = 0.05
    momentum = 0.9
    dropout = 0.35
    epochs = 30
    crf_loss_reduction = 'sum'

    # model_fn = 'data/model/100_150_sum/5_checkpoint.tar'
    model_fn = None

    model, optimizer, epoch, all_losses, eval_losses = load_model(model_fn,
                                                                  voc,
                                                                  character_embedding_dim,
                                                                  character_hidden_dim,
                                                                  context_hidden_dim,
                                                                  dropout,
                                                                  crf_loss_reduction)

    print('Training...')
    train(model, optimizer, train_dl, dev_dl, test_dl, epochs, epoch, all_losses, eval_losses, save_every=1,
          save_dir='data/model')
