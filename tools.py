"""David Donahue 2016. This is a place to keep functions that more than one script could potentially use.
This makes a function easily accessible and independent of the application. Functions
here are dependent on data stored in the data/ folder."""
from os import walk
import numpy as np
import cPickle as pickle


def convert_words_to_indices(words, char_to_index, max_word_size=20):
    m = len(words)
    # Convert all words to indices using char_to_index dictionary.
    np_word_indices = np.zeros([m, max_word_size], dtype=float)
    for word_index in range(m):
        word = words[word_index]
        for char_index in range(len(word)):
            num_non_characters = 0
            if char_index - num_non_characters < max_word_size:
                char = word[char_index]
                if char.isalpha():
                    if char in char_to_index:
                        np_word_indices[word_index, char_index - num_non_characters] = char_to_index[char]
                else:
                    num_non_characters += 1

    return np_word_indices


def load_hashtag_data_and_vocabulary(tweet_pairs_path, char_to_index_path):
    '''Load in tweet pairs per hashtag. Create a list of [hashtag_name, pairs, labels] entries.
    Return tweet pairs, tweet labels, char_to_index.cpkl and vocabulary size.'''
    hashtag_datas = []
    for (dirpath, dirnames, filenames) in walk(tweet_pairs_path):
        for filename in filenames:
            if '_pairs.npy' in filename:
                hashtag_name = filename.replace('_pairs.npy','')
                tweet_pairs = np.load(tweet_pairs_path + filename)
                tweet_labels = np.load(tweet_pairs_path + hashtag_name + '_labels.npy')
                hashtag_datas.append([hashtag_name, tweet_pairs, tweet_labels])
    char_to_index = pickle.load(open(char_to_index_path, 'rb'))
    vocab_size = len(char_to_index)
    return hashtag_datas, char_to_index, vocab_size


def invert_dictionary(dictionary):
    inv_dictionary = {v: k for k, v in dictionary.iteritems()}
    return inv_dictionary


