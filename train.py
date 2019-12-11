import torch
import torch.optim as optim
import os
from tqdm import tqdm

import utils
import vocab
import const
from bilstm_crf import BiLSTMCrf
import dataset
import evaluator
from sentence import Sentence
import argparse


def adjust_learning_rate(optimizer, learning_rate, epoch, learning_rate_decay):
    lr = learning_rate / (1 + learning_rate_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

def parse():
    parser = argparse.ArgumentParser(description='BiLSTM-CRF model')
    # parser.add_argument('-m', '--mode', help='Training mode or testing mode: value either train or test', required=True)
    parser.add_argument('-tr', '--trainpath', help='Training file location', default='data/data/train.txt')
    parser.add_argument('-te', '--testpath', help='Test file location', default='data/data/test.txt')
    parser.add_argument('-de', '--devpath', help='Development file location', default='data/data/dev.txt')
    parser.add_argument('-pr', '--pretrained_path', default='data/pretrained_embedding/fasttext_pretrained_embeddings_300.bin', help='Pretrained word embedding used in training')
    parser.add_argument('-wb', '--word_embedding_dim', type=int, default=300, help='Number of word embedding dimensions')
    parser.add_argument('-cb', '--character_embedding_dim', type=int, default=100, help='Number of character embedding dimensions')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('-wi', '--context_hidden_dim', type=int, default=150 * 2, help='Context hidden size')
    parser.add_argument('-ci', '--character_hidden_dim', type=int, default=100 * 2, help='Character hidden size')
    # parser.add_argument('-ls', '--patience', type=int, default=10, help='Patience used in early stopping')
    parser.add_argument('-lr', '--lr', type=float, default=0.0035, help='Learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.35, help='Dropout probability for lstm and dropout layers')
    parser.add_argument('-cp', '--checkpoint_dir', default='data/model', help='Checkpoint directory')
    parser.add_argument('-ep', '--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('-dc', '--learning_rate_decay', default=0.06, type=float, help='Decay rate in each epoch')
    parser.add_argument('-upc', '--using_pos_chunk', action='store_true', help='Using pos chunk option')

    args = parser.parse_args()
    return args


def train(
        model,
        optimizer,
        train_dl,
        dev_dl,
        test_dl,
        epochs,
        start_epoch,
        all_losses,
        eval_losses,
        test_scores,
        best_test_score,
        save_every,
        save_dir
):
    for epoch in range(start_epoch, epochs + 1):
        adjust_learning_rate(optimizer, args.lr, epoch, args.learning_rate_decay)
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
        test_scores.append(test_f1_score)
        print(f"Epoch {epoch}: \tloss = {epoch_loss}"
              f"\neval_loss = {eval_losses[-1]}"
              f"\ntest_precision = {test_precision}"
              f"\ntest_recall = {test_recall}"
              f"\ntest_f1_score = {test_f1_score}")
        print('-----------------------------------------------')

        if test_f1_score > best_test_score:
            best_test_score = test_f1_score
            directory = os.path.join(save_dir, f'{args.character_hidden_dim}_{args.context_hidden_dim}_{crf_loss_reduction}')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'all_losses': all_losses,
                'eval_losses': eval_losses,
                'test_scores': test_scores,
            }, os.path.join(directory, "best_model.tar"))

        if epoch % save_every == 0:
            directory = os.path.join(save_dir, f'{args.character_hidden_dim}_{args.context_hidden_dim}_{crf_loss_reduction}')
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'all_losses': all_losses,
                'eval_losses': eval_losses,
                'test_scores': test_scores,
            }, os.path.join(directory, f"{epoch}_checkpoint.tar"))

        if is_stopping(eval_losses):
            break


def is_stopping(eval_losses):
    best_eval_losses = min(eval_losses)
    return min(eval_losses[-10:]) > best_eval_losses


def load_model(model_fn, voc, character_embedding_dim, character_hidden_dim, context_hidden_dim, dropout,
               crf_loss_reduction, lr, using_pos_chunk):
    print('Building model and optimizer...')
    epoch = 0
    all_losses = []
    eval_losses = []
    test_scores = []
    best_test_score = 0

    model = BiLSTMCrf(vocab=voc,
                      character_embedding_dim=character_embedding_dim,
                      character_hidden_dim=character_hidden_dim,
                      context_hidden_dim=context_hidden_dim,
                      using_pos_chunk=using_pos_chunk,
                      dropout=dropout,
                      crf_loss_reduction=crf_loss_reduction)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if model_fn is not None:
        checkpoint = torch.load(model_fn)
        epoch = checkpoint['epoch']
        model_sd = checkpoint['model']
        optimizer_sd = checkpoint['optimizer']
        all_losses = checkpoint['all_losses']
        eval_losses = checkpoint['eval_losses']
        test_scores = checkpoint['test_scores']
        best_test_score = max(test_scores)

        model.load_state_dict(model_sd)
        optimizer.load_state_dict(optimizer_sd)

    model = model.to(const.DEVICE)
    for state in optimizer.state.values():
        for k, v in state.items():
            if type(v) is torch.Tensor:
                state[k] = v.to(const.DEVICE)

    return model, optimizer, epoch + 1, all_losses, eval_losses, test_scores, best_test_score


if __name__ == '__main__':
    args = parse()
    print(args)
    print('Loading vocab...')
    voc = vocab.Vocab(args.pretrained_path, freeze=True)

    print('Loading data ...')
    train_sentences = [Sentence(sentence, voc) for sentence in utils.read_data(args.trainpath)]
    dev_sentences = [Sentence(sentence, voc) for sentence in utils.read_data(args.devpath)]
    test_sentences = [Sentence(sentence, voc) for sentence in utils.read_data(args.testpath)]

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

    train_dl = dataset.DataLoader(train_ds, batch_size=args.batch_size)
    dev_dl = dataset.DataLoader(dev_ds, batch_size=args.batch_size)
    test_dl = dataset.DataLoader(test_ds, batch_size=args.batch_size)

    # word_embedding_dim = 300
    # momentum = 0.9
    crf_loss_reduction = 'token_mean'

    model_fn = 'data/model/200_300_token_mean/best_model.tar'
    # model_fn = None

    model, optimizer, epoch, all_losses, eval_losses, test_scores, best_test_score = load_model(model_fn,
                                                                                                voc,
                                                                                                args.character_embedding_dim,
                                                                                                args.character_hidden_dim,
                                                                                                args.context_hidden_dim,
                                                                                                args.dropout,
                                                                                                crf_loss_reduction,
                                                                                                args.lr,
                                                                                                args.using_pos_chunk)

    # print('Training...')
    # train(
    #     model,
    #     optimizer,
    #     train_dl,
    #     dev_dl,
    #     test_dl,
    #     args.epochs,
    #     epoch,
    #     all_losses,
    #     eval_losses,
    #     test_scores,
    #     best_test_score,
    #     save_every=5,
    #     save_dir=args.checkpoint_dir
    # )

    evaluator.evaluate_test(model, test_dl, 'output/output_no_pos_chunk.txt')
