"""David Donahue 2016. This is a place to keep functions that more than one script could potentially use.
This makes a function easily accessible and independent of the application. Functions
here are dependent on data stored in the data/ folder."""
from os import walk
import os
import numpy as np
import cPickle as pickle
import random
import csv


HUMOR_MAX_WORDS_IN_TWEET = 30


def extract_tweet_pairs_from_file(hashtag_file):
    '''This script extracts tweet pairs from the file hashtag_file.
    It stores them in an array of tweet pairs, each tweet pair
    being a list of the form [tweet_1_text, tweet_2_text, first_tweet_funnier].
    first_tweet_funnier is 1 if the first tweet is funnier and 0 if the second
    tweet is funnier.'''
    pairs = []
    non_winners = []
    top_ten = []
    winner = []
    # Find winner, top-ten, and non-winning tweets.
    with open(hashtag_file) as tsv:
        for line in csv.reader(tsv, dialect='excel-tab'):
            tweet_rank = int(line[2])
            tweet_text = line[1]
            if tweet_rank == 0:
                non_winners.append(tweet_text)
            if tweet_rank == 1:
                top_ten.append(tweet_text)
            if tweet_rank == 2:
                winner.append(tweet_text)
    # Create pairs from non-winning and top-ten tweets.
    for non_winning_tweet in non_winners:
        for top_ten_tweet in winner + top_ten:
            #Create pair
            funnier_tweet_first = bool(random.getrandbits(1))
            if funnier_tweet_first:
                pairs.append([top_ten_tweet, non_winning_tweet, 0])
            else:
                pairs.append([non_winning_tweet, top_ten_tweet, 1])
    # Create pairs from top-ten and winning tweet.
    for top_ten_tweet in top_ten:
        for winning_tweet in winner:
            #Create pair
            funnier_tweet_first = bool(random.getrandbits(1))
            if funnier_tweet_first:
                pairs.append([winning_tweet, top_ten_tweet, 0])
            else:
                pairs.append([top_ten_tweet, winning_tweet, 1])
    return pairs


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


def get_hashtag_file_names(tweet_pairs_dir):
    '''Returns .tsv file name for each hashtag in the dataset (extension omitted).'''
    f = []
    for (dirpath, dirnames, filenames) in walk(tweet_pairs_dir):
        f.extend(filenames)
        break
    g = [os.path.splitext(hashtag)[0] for hashtag in f]
    return g

