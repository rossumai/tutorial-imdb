#!/usr/bin/env python3

from keras.layers import Activation, Dense, Input
from keras.models import Model
import numpy as np
import os
import re


# Vocabulary: All words used, starting by the most frequent
with open('aclImdb/imdb.vocab') as f:
    vocab = [word.rstrip() for word in f]
    # Keep only most frequent 5000 words rather than all 90000
    # Just saving memory - the long tail occurs too few times
    # for the model to learn anything anyway
    vocab = vocab[:5000]
    print('%d words in vocabulary' % (len(vocab),))


def text_tokens(text):
    text = text.lower()
    text = re.sub("\\s", " ", text)
    text = re.sub("[^a-zA-Z' ]", "", text)
    tokens = text.split(' ')
    return tokens


def review_bow_vector(tokens):
    vector = [0] * len(vocab)
    for t in tokens:
        try:
            vector[vocab.index(t)] = 1
        except:
            pass  # ignore missing words
    return vector


def load_dataset(dirname):
    X, y = [], []
    # Review files: neg/0_3.txt neg/10000_4.txt neg/10001_4.txt ...
    for y_val, y_label in enumerate(['neg', 'pos']):
        y_dir = os.path.join(dirname, y_label)
        for fname in os.listdir(y_dir):
            fpath = os.path.join(y_dir, fname)
            print('\r' + fpath + '   ', end='')
            with open(fpath) as f:
                tokens = text_tokens(f.read())
            bow = review_bow_vector(tokens)
            X.append(bow)
            y.append(y_val)  # 0 for 'neg', 1 for 'pos'
    print()
    return np.array(X), np.array(y)


class SentimentModel(object):
    def __init__(self):
        bow = Input(shape=(len(vocab),), name='bow_input')
        # weights of all inputs
        sentiment = Dense(1)(bow)
        # normalize to [0, 1] range
        sentiment = Activation('sigmoid')(sentiment)

        self.model = Model(input=bow, output=sentiment)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X, y, X_val, y_val):
        self.model.fit(X, y, validation_data=(X_val, y_val), epochs=25, verbose=1)

    def predict(self, X):
        return self.model.predict(np.array(X))


if __name__ == "__main__":
    X_train, y_train = load_dataset('aclImdb/train/')
    X_val, y_val = load_dataset('aclImdb/test/')
    sentiment = SentimentModel()
    sentiment.train(X_train, y_train, X_val, y_val)

    test_text = 'Good story about a backwoods community in the Ozarks around the turn of the century. Moonshine is the leading industry, fighting and funning the major form of entertainment. One day a stranger enters the community and causes a shake-up among the locals. Beautiful scenery adds much to the story.'
    test_bow = review_bow_vector(text_tokens(test_text))
    print(test_text, sentiment.predict([test_bow])[0])
