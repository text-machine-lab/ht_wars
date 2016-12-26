"""David Donahue 2016. This is a place to keep functions that more than one script could potentially use.
This makes a function easily accessible and independent of the application. Functions
here are dependent on data stored in the data/ folder."""
from os import walk
import os
import numpy as np
import cPickle as pickle
import random
import csv


HUMOR_MAX_WORDS_IN_TWEET = 20  # All winning tweets are under 30 words long
GLOVE_SIZE = 200
PHONETIC_EMB_SIZE = 200


def extract_tweet_pairs_from_file(hashtag_file):
    '''This script extracts tweet pairs from the file hashtag_file.
    It stores them in an array of tweet pairs, each tweet pair
    being a list of the form [tweet_1_text, tweet_2_text, first_tweet_funnier].
    first_tweet_funnier is 1 if the first tweet is funnier and 0 if the second
    tweet is funnier.'''
    tweets = []
    tweet_ranks = []

    # Get all tweets in file along with their ranks.
    with open(hashtag_file) as tsv:
        for line in csv.reader(tsv, dialect='excel-tab'):
            tweets.append(line[1])
            tweet_ranks.append(int(line[2]))
    return extract_tweet_pairs(tweets, tweet_ranks)


def extract_tweet_pairs(tweets, tweet_ranks, tweet_ids):
    """Creates pairs of the form [first_tweet, first_tweet_id, second_tweet, second_tweet_id, first_tweet_is_funnier]
    and """
    pairs = []
    non_winners = []
    top_ten = []
    winner = []
    id_pairs = []
    non_winner_ids = []
    top_ten_ids = []
    winner_ids = []
    # Find winner, top-ten, and non-winning tweets.
    for i in range(len(tweets)):
        tweet_rank = tweet_ranks[i]
        tweet_text = tweets[i]
        tweet_id = tweet_ids[i]
        if tweet_rank == 0:
            non_winners.append(tweet_text)
            non_winner_ids.append(tweet_id)
        elif tweet_rank == 1:
            top_ten.append(tweet_text)
            top_ten_ids.append(tweet_id)
        elif tweet_rank == 2:
            winner.append(tweet_text)
            winner_ids.append(tweet_id)
        else:
            print 'Error: Invalid tweet rank'
    # Create pairs from non-winning and top-ten tweets.
    for non_winning_tweet, non_winning_id in zip(non_winners, non_winner_ids):
        for top_ten_tweet, top_ten_id in zip(winner + top_ten, winner_ids + top_ten_ids):
            # Create pair
            funnier_tweet_first = bool(random.getrandbits(1))
            if funnier_tweet_first:
                pairs.append([top_ten_tweet, top_ten_id, non_winning_tweet, non_winning_id, 1])
            else:
                pairs.append([non_winning_tweet, non_winning_id, top_ten_tweet, top_ten_id, 0])
    # Create pairs from top-ten and winning tweet.
    for top_ten_tweet, top_ten_id in zip(top_ten, top_ten_ids):
        for winning_tweet, winning_id in zip(winner, winner_ids):
            # Create pair
            funnier_tweet_first = bool(random.getrandbits(1))
            if funnier_tweet_first:
                pairs.append([winning_tweet, winning_id, top_ten_tweet, top_ten_id, 1])
            else:
                pairs.append([top_ten_tweet, top_ten_id, winning_tweet, winning_id, 0])
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
    random.shuffle(g)
    return g

