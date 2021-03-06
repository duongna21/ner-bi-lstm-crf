import itertools
import matplotlib.pyplot as plt
import random
from pyvi.ViTokenizer import ViTokenizer
from underthesea import word_tokenize
import re

random.seed(1)


def read_data(fn):
    sentences = []
    sentence = []
    with open(fn, mode='r', encoding='utf8') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                tokens = line.split()
                if len(tokens) == 4:
                    sentence.append(tuple(tokens))
            else:
                sentences.append(sentence)
                sentence = []
        if len(sentence) > 0:
            sentences.append(sentence)
        f.close()
    return sentences


def read_sentences_tags(fn):
    data = read_data(fn)
    random.shuffle(data)
    sentences = [[word for word, _, _, _ in sentence] for sentence in data]
    tags = [[tag for _, _, _, tag in sentence] for sentence in data]
    return sentences, tags


def zero_padding(l, fill_value=0):
    return list(itertools.zip_longest(*l, fillvalue=fill_value))


def get_sentences(paragraph):
    tokenized_sentences = [word_tokenize(sentence, format='text') for sentence in
                           re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s+', paragraph)]
    sentences = [[
        (token, '<PAD_POS>', '<PAD_CHUNK>', '<PAD_TAG>')
        for token in re.sub(r'(?<=\d\s[/-])\s|(?=\s[/-]\s\d)\s', '', sentence).split()
    ] for sentence in tokenized_sentences]
    return sentences


def plot_losses(losses):
    plt.figure()
    plt.plot(losses)
    plt.show()
    plt.savefig('losses.png')
