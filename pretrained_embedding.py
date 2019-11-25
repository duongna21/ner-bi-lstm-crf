from fasttext import FastText
import json
from pyvi.ViTokenizer import ViTokenizer
import re


def read_documents(fn):
    with open('data/content_documents.txt', mode='w', encoding='utf8') as fw:
        with open(fn, mode='r', encoding='utf8') as fr:
            for line in fr:
                obj = json.loads(line)
                documents = obj['content']

                documents = [tokenize(document) for document in documents if document]

                for document in documents:
                    fw.write(document)
                    fw.write('\n')

            fr.close()
        fw.close()


def tokenize(s):
    s = re.sub(r'\d+([.,]\d+)?', '__NUM__', s.lower())
    tokenized = ViTokenizer.tokenize(s)
    return tokenized


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
    model = FastText.train_unsupervised(fn, model='skipgram', dim=300, maxn=0)
    model.save_model('data/pretrained_embedding/fasttext_pretrained_embeddings_300.bin')


if __name__ == '__main__':
    # read_documents('data/vnexpress_450k.jsonl')
    train_embedding('data/fixed_documents.txt')
