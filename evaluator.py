import torch
from conlleval import evaluate
import const


def evaluate_loss(model, dev_dl):
    with torch.no_grad():
        model.eval()
        current_loss = 0

        for ((batch_sentence_word_indexes,
             batch_sentence_pos_indexes,
             batch_sentence_chunk_indexes,
             batch_sentence_word_character_indexes),
             batch_sentence_tag_indexes,
             lengths) in dev_dl:
            loss = - model(batch_sentence_word_indexes,
                           batch_sentence_pos_indexes,
                           batch_sentence_chunk_indexes,
                           batch_sentence_word_character_indexes,
                           lengths,
                           batch_sentence_tag_indexes)
            current_loss += loss.item()

        return current_loss / len(dev_dl)


def evaluate_test(model, test_dl):
    label_pred = []
    label_true = []
    with torch.no_grad():
        model.eval()

        for ((batch_sentence_word_indexes,
             batch_sentence_pos_indexes,
             batch_sentence_chunk_indexes,
             batch_sentence_word_character_indexes),
             batch_sentence_tag_indexes,
             lengths) in test_dl:
            pred_seqs = model(batch_sentence_word_indexes,
                              batch_sentence_pos_indexes,
                              batch_sentence_chunk_indexes,
                              batch_sentence_word_character_indexes,
                              lengths,
                              None)

            for i, (pred_seq, length) in enumerate(zip(pred_seqs, lengths.tolist())):
                label_true.extend(batch_sentence_tag_indexes[:length, i].tolist())
                label_pred.extend(pred_seq)

    label_pred = [const.TAG_LIST[i] for i in label_pred]
    label_true = [const.TAG_LIST[i] for i in label_true]
    precision, recall, f1_score = evaluate(label_true, label_pred, verbose=False)
    return precision, recall, f1_score
