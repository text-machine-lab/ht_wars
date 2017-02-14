import os
import sys
import itertools
from random import random
import pickle

import numpy as np
from sklearn.metrics import accuracy_score

from nltk.tokenize import TweetTokenizer

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import merge
from keras.layers import LSTM
from keras.regularizers import l2, activity_l2
from keras.layers.normalization import BatchNormalization

from char_based_cnn import load_vocab, load_hashtags, get_is_oov, create_pairwise_data, create_data_matrices


def create_model(input_dim, max_len):
    embedding_size = 50
    embedding_dropout = 0.2

    rnn_hidden_size = 256

    hidden_size = 256
    hidden_dropout = 0.2
    regularization = 0.0005

    input1 = Input(shape=(max_len,))
    input2 = Input(shape=(max_len,))

    embedding = Embedding(input_dim, embedding_size, input_length=max_len, dropout=embedding_dropout, W_regularizer=l2(
        regularization))
    embedded1 = embedding(input1)
    embedded2 = embedding(input2)

    lstm = LSTM(rnn_hidden_size)
    last_state1 = lstm(embedded1)
    last_state2 = lstm(embedded1)

    dense = Dense(hidden_size, activation='relu', W_regularizer=l2(regularization))
    # dense_dropout = Dropout(hidden_dropout)(dense)
    # dense_dropout_activation = Activation('relu')(dense_dropout)

    dense1 = dense(last_state1)
    dense2 = dense(last_state2)

    merged = merge([dense1, dense2], mode='concat', concat_axis=-1)

    output = Dense(256, activation='relu', W_regularizer=l2(regularization))
    output = Dense(128, activation='relu', W_regularizer=l2(regularization))
    output = Dense(1, activation='sigmoid', W_regularizer=l2(regularization))
    model_output = output(merged)

    model = Model(input=[input1, input2], output=model_output)

    return model


def main():
    vocab = load_vocab()
    print('Vocabulary:', len(vocab))

    data_dir = '../../data/cleaned_tweets/'
    data = load_hashtags(data_dir)

    hashtags = sorted(data.keys())
    data = [data[h] for h in hashtags]
    nb_hashtags = len(hashtags)
    print('Hashtags:', nb_hashtags)

    mode = 'all'
    nb_epoch = 1

    acc_sum = 0
    total_pairs = 0
    accs = []

    for i in range(nb_hashtags):  # nb_hashtags
        print('Starting hashtag {}/{} {}'.format(i + 1, nb_hashtags, hashtags[i]))
        data_test = data[i]
        data_train = data[:i] + data[i + 1:]

        is_oov_test = get_is_oov(data_test, vocab)
        print('OOV stats:', sum(is_oov_test.values()), len(is_oov_test))

        character_set = sorted(set([c for d in data_train for t in d for c in t[0]]))
        char2idx = {c: i + 1 for i, c in enumerate(character_set)}
        print('Characters:', len(char2idx))

        samples_train = [t for d in data_train for t in create_pairwise_data(d)]
        samples_test = create_pairwise_data(data_test, mode=mode, is_ovv=is_oov_test)

        max_len = max(max([max(len(d[0]), len(d[1])) for d in samples_train]),
                      max([max(len(d[0]), len(d[1])) for d in samples_test]))
        input_dim = len(character_set) + 1

        print('Max len', max_len)
        print('Input dim', input_dim)

        X1_train, X2_train, y_train = create_data_matrices(samples_train, char2idx, max_len)
        X1_test, X2_test, y_test = create_data_matrices(samples_test, char2idx, max_len)

        print('Train:', X1_train.shape, X2_train.shape, y_train.shape)
        print('Test:', X1_test.shape, X2_test.shape, y_test.shape)

        model = create_model(input_dim, max_len)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit([X1_train, X2_train], y_train, nb_epoch=nb_epoch, verbose=1, batch_size=256)

        y_pred_prob = model.predict([X1_test, X2_test], verbose=1)
        y_pred = (y_pred_prob > 0.5).astype('int32')
        acc = accuracy_score(y_test, y_pred)

        accs.append(acc)

        nb_test = len(y_test)
        print('Accuracy:', acc, 'Samples:', nb_test)

        acc_sum += acc * nb_test
        total_pairs += nb_test

    weighted_acc = acc_sum / total_pairs
    gt_05 = len([a for a in accs if a >= 0.5])
    min_acc = min(accs)
    max_acc = max(accs)
    print('Mode: {}, Weighted Accuracy: {}, % > 0.5: {}, Min Accuracy: {}, Max Accuracy: {}'.format(
        mode, weighted_acc, 100 * gt_05 / nb_hashtags, min_acc, max_acc))


if __name__ == '__main__':
    main()
