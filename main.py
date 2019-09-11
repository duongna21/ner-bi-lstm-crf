import torch
import torch.optim as optim
import os

import utils
import vocab
import const
from bilstm_crf import BiLSTMCrf
from conlleval import evaluate


def train_iters(model, optimizer, epochs, current_epoch, batch_size, print_every_batch, save_every_epoch, save_dir):
    all_losses = []

    for epoch in range(current_epoch, epochs + 1):
        model.train()
        num_batches = (len(train_sentences) - 1) // batch_size + 1

        current_loss = 0

        for batch_i in range(num_batches):
            optimizer.zero_grad()
            start_idx = batch_i * batch_size
            end_idx = (batch_i + 1) * batch_size

            sentences, tags = train_sentences[start_idx: end_idx], train_tags[start_idx: end_idx]
            loss = - model(sentences, tags)
            loss.backward()
            optimizer.step()

            loss_value = loss.item() / sum([len(sentence) for sentence in sentences])
            # loss_value = loss.item()
            all_losses.append(loss_value)
            current_loss += loss_value

            if (batch_i + 1) % print_every_batch == 0:
                print(f"Epoch {epoch}, batch {batch_i}: loss = {current_loss / print_every_batch}")
                current_loss = 0

        if epoch % save_every_epoch == 0:
            directory = os.path.join(save_dir, f'{character_hidden_dim}_{context_hidden_dim}')
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(directory, f"{epoch}_checkpoint.tar"))

        print('Evaluating...')
        eval_result = eval2(model)
        print(f"Epoch {epoch}: {eval_result}")

    utils.plot_losses(all_losses)


def evaluate1(model):
    with torch.no_grad():
        model.eval()
        with open('output.txt', mode='w', encoding='utf8', newline='\n') as f:
            for sentence, tags in zip(test_sentences, test_tags):
                tags_pred = model.decode([sentence])
                for word, true_tag, pred_tag in zip(sentence, tags, tags_pred[0]):
                    f.write(f"{word}\t{true_tag}\t{pred_tag}\n")
                f.write('\n')
            f.close()


def eval2(model):
    y_true = []
    y_pred = []

    for sentence, tags in zip(test_sentences, test_tags):
        tags_pred = model.decode([sentence])
        for word, true_tag, pred_tag in zip(sentence, tags, tags_pred[0]):
            y_true.append(true_tag)
            y_pred.append(pred_tag)

    precision, recall, f1_score = evaluate(y_true, y_pred, verbose=False)
    return "Precision: %.2f%%\tRecall: %.2f%%\tF1_score: %.2f%%".format(precision, recall, f1_score)


def load_model(model_fn):
    print('Building model and optimizer...')
    epoch = 0
    voc = vocab.Vocab('data/pretrained_embedding/pretrained_embedding_5M.vec')

    model = BiLSTMCrf(voc, const.CHARACTER_LIST, character_embedding_dim, character_hidden_dim, context_hidden_dim,
                      tag_list)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    if model_fn is not None:
        checkpoint = torch.load(model_fn)
        epoch = checkpoint['epoch']
        model_sd = checkpoint['model']
        optimizer_sd = checkpoint['optimizer']

        model.load_state_dict(model_sd)
        optimizer.load_state_dict(optimizer_sd)

    model = model.to(const.DEVICE)
    for state in optimizer.state.values():
        for k, v in state.items():
            if type(v) is torch.Tensor:
                state[k] = v.to(const.DEVICE)

    return model, optimizer, epoch + 1


if __name__ == '__main__':
    train_sentences, train_tags = utils.read_sentences_tags('data/data/train.txt')
    test_sentences, test_tags = utils.read_sentences_tags('data/data/test.txt')
    # tag_list = list(set([tag for tags in train_tags for tag in tags]))
    tag_list = ['B-MISC', 'I-MISC', 'B-PER', 'O', 'B-LOC', 'I-PER', 'B-ORG', 'I-ORG', 'I-LOC']

    batch_size = 32
    character_embedding_dim = 25
    character_hidden_dim = 50
    context_hidden_dim = 256
    learning_rate = 0.0001
    epochs = 50

    model_fn = None
    # model_fn = 'data/model/50_150/8_checkpoint.tar'
    # model_fn = 'data/model/colab/50_checkpoint.tar'

    model, optimizer, current_epoch = load_model(model_fn)

    train_iters(model, optimizer, epochs, current_epoch, batch_size, print_every_batch=50, save_every_epoch=2,
                save_dir='data/model')
    # evaluate1(model)
    # print(eval2(model))

# best 40, dropout 0.2
