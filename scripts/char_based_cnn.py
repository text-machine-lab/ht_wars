import os
import sys
import itertools
from random import random

import numpy as np
from sklearn.metrics import accuracy_score

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import merge
from keras.regularizers import l2, activity_l2
from keras.layers.normalization import BatchNormalization


def load_file(filename):
    label_map = {'0': 0, '9': 1, '1': 2}

    data = []
    with open(filename, 'r') as f:
        for line in f:
            tweet_data = line.strip().split('\t')

            tweet = tweet_data[1]
            if len(tweet_data) == 3:
                label = label_map[tweet_data[2]]
            else:
                label = label_map['0']

            data.append((tweet, label))

    return data


def load_hashtags(directory):
    hashtags = os.listdir(directory)
    filenames = [os.path.join(directory, f) for f in hashtags]

    data = {h: load_file(f) for h, f in zip(hashtags, filenames)}

    return data


def create_pairwise_data(data):
    """Creates pairwise examples"""

    winner = [d[0] for d in data if d[1] == 2]
    top10 = [d[0] for d in data if d[1] == 1]
    rest = [d[0] for d in data if d[1] == 0]

    pairs = [(winner, top10), (winner, rest), (top10, rest), ]

    training_data = []
    for pair in pairs:
        funny_tweets = pair[0]
        not_so_funny_tweets = pair[1]

        for funny_tweet, not_so_funny_tweet in itertools.product(funny_tweets, not_so_funny_tweets):
            if random() > 0.5:
                sample = (funny_tweet, not_so_funny_tweet, 1)
            else:
                sample = (not_so_funny_tweet, funny_tweet, 0)

            training_data.append(sample)

    return training_data


def create_model(input_dim, max_len):
    embedding_size = 50
    embedding_dropout = 0.2

    nb_filter_l1 = 100
    filter_length_l1 = 5
    pool_length_l1 = 2

    nb_filter_l2 = 100
    filter_length_l2 = 3
    pool_length_l2 = 2

    hidden_size = 256
    hidden_dropout = 0.2
    regularization = 0.0005

    input1 = Input(shape=(max_len,))
    input2 = Input(shape=(max_len,))

    embedding = Embedding(input_dim, embedding_size, input_length=max_len, dropout=embedding_dropout, W_regularizer=l2(
        regularization))
    embedded1 = embedding(input1)
    embedded2 = embedding(input2)

    conv_l1 = Convolution1D(nb_filter=nb_filter_l1, filter_length=filter_length_l1,
                         border_mode='valid',
                         # activation='relu',
                         # subsample_length=1,
                         # W_regularizer = l2(regularization),
                        )
    pool_l1 = MaxPooling1D(pool_length=pool_length_l1, stride=None, border_mode='valid')
    batch_norm_l1 = BatchNormalization()
    activation_l1 = Activation('relu')

    conv1 = conv_l1(embedded1)
    conv2 = conv_l1(embedded2)
    normed1 = batch_norm_l1(conv1)
    normed2 = batch_norm_l1(conv2)
    activated1 = activation_l1(normed1)
    activated2 = activation_l1(normed2)
    pool1 = pool_l1(activated1)
    pool2 = pool_l1(activated2)


    conv_l2 = Convolution1D(nb_filter=nb_filter_l2, filter_length=filter_length_l2,
                         border_mode='valid',
                         # activation='relu',
                         # subsample_length=1,
                         # W_regularizer = l2(regularization),
                        )
    pool_l2 = MaxPooling1D(pool_length=pool_length_l2, stride=None, border_mode='valid')
    batch_norm_l2 = BatchNormalization()
    activation_l2 = Activation('relu')


    conv3 = conv_l2(pool1)
    conv4 = conv_l2(pool2)
    normed3 = batch_norm_l2(conv3)
    normed4 = batch_norm_l2(conv4)
    activated3 = activation_l2(normed3)
    activated4 = activation_l2(normed4)
    pool3 = pool_l2(activated3)
    pool4 = pool_l2(activated4)

    flatten = Flatten()
    flatten1 = flatten(pool3)
    flatten2 = flatten(pool4)

    dense = Dense(hidden_size, activation='relu', W_regularizer=l2(regularization))
    # dense_dropout = Dropout(hidden_dropout)(dense)
    # dense_dropout_activation = Activation('relu')(dense_dropout)

    dense1 = dense(flatten1)
    dense2 = dense(flatten2)

    merged = merge([dense1, dense2], mode='concat', concat_axis=-1)

    output = Dense(256, activation='relu', W_regularizer=l2(regularization))
    output = Dense(128, activation='relu', W_regularizer=l2(regularization))
    output = Dense(1, activation='sigmoid', W_regularizer=l2(regularization))
    model_output = output(merged)

    model = Model(input=[input1, input2], output=model_output)

    return model


def create_data_matrices(samples, char2idx, max_len):
    X1 = [[char2idx[c] if c in char2idx else 0 for c in d[0]] for d in samples]
    X2 = [[char2idx[c] if c in char2idx else 0 for c in d[1]] for d in samples]

    X1 = sequence.pad_sequences(X1, maxlen=max_len)
    X2 = sequence.pad_sequences(X2, maxlen=max_len)

    y = np.array([d[2] for d in samples])

    return X1, X2, y

def main():
    directory = '../../data/cleaned_tweets/'

    data = load_hashtags(directory)

    hashtags = sorted(data.keys())
    data = [data[h] for h in hashtags]
    nb_hashtags = len(hashtags)
    print('Hashtags:', nb_hashtags)

    nb_epoch = 1

    acc_sum = 0
    total_pairs = 0
    accs = []

    for i in range(nb_hashtags): # nb_hashtags
        print('Starting hashtag {}/{} {}'.format(i + 1, nb_hashtags, hashtags[i]))
        data_test = data[i]
        data_train = data[:i] + data[i+1:]

        character_set = sorted(set([c for d in data_train for t in d for c in t[0]]))
        char2idx = {c: i+1 for i, c in enumerate(character_set)}
        print('Characters:', len(char2idx))

        samples_train = [t for d in data_train for t in create_pairwise_data(d)]
        samples_test = create_pairwise_data(data_test)

        max_len = max(max([max(len(d[0]),len(d[1])) for d in samples_train]), max([max(len(d[0]),len(d[1])) for d in samples_test]))
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
    print('Weighted Accuracy: {}, % > 0.5: {}, Min Accuracy: {}, Max Accuracy: {}'.format(weighted_acc, 100 * gt_05 / nb_hashtags, min_acc, max_acc))


if __name__ == '__main__':
    main()
