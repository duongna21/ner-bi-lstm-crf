from gensim.models.fasttext import FastText
import json
from pyvi.ViTokenizer import ViTokenizer
import re
import pickle


def split_into_sentences(fn):
    with open('data/sentences.txt', mode='w', encoding='utf8') as fw:
        with open(fn, mode='r', encoding='utf8') as fr:
            for line in fr:
                obj = json.loads(line)
                documents = [obj['title'],
                             obj['description'],
                             *obj['content'],
                             *obj.get('comments', [])]

                documents = [document for document in documents if document]

                for document in documents:
                    sentences = tokenize(document)

                    for sentence in sentences:
                        if len(sentence) > 0:
                            fw.write(' '.join(sentence))
                            fw.write('\n')

            fr.close()
        fw.close()


def tokenize(s):
    tokenized = ViTokenizer.tokenize(s)
    sentences = re.split(r'\s+[.?!]+\s+', tokenized)
    return [re.findall(r'[^\W\d]+', sentence) for sentence in sentences]


def load_sentences(fn, max_sentences=None):
    sentences = []
    with open(fn, mode='r', encoding='utf8') as f:
        for line in f:
            if max_sentences is not None and len(sentences) == max_sentences:
                break
            sentence = line.strip().lower().split(' ')
            if len(sentence) > 5:
                sentences.append(sentence)
        f.close()
    return sentences


def train_embedding(fn):
    sentences = load_sentences(fn, 5000000)
    model = FastText(size=100)
    model.build_vocab(sentences=sentences)
    model.train(sentences=sentences, total_examples=len(sentences), epochs=10)
    model.save('data/pretrained_embedding/pretrained_embedding_5M.vec')
    with open('data/pretrained_embedding/pretrained_embedding_fasttext_5M.pkl', mode='wb') as f:
        pickle.dump(model, f)
        f.close()


def load_model(fn):
    model = FastText.load('data/pretrained_embedding/pretrained_embedding.vec')
    return model


if __name__ == '__main__':
    # split_into_sentences('data/vnexpress_450k.jsonl')
    train_embedding('data/sentences/sentences.txt')

    # sentences = load_sentences('data/sentences.txt', 1000000)
    # with open('sentences_1M.pkl', mode='wb') as f:
    #     pickle.dump(sentences, f)
    #     f.close()
