"""David Donahue 2016. This is a place to keep functions that more than one script could potentially use.
This makes a function easily accessible and independent of the application. Functions
here are dependent on data stored in the data/ folder."""
from os import walk
import os
import numpy as np
import cPickle as pickle
import random
import csv
import nltk


HUMOR_MAX_WORDS_IN_TWEET = 20  # All winning tweets are under 30 words long
HUMOR_MAX_WORDS_IN_HASHTAG = 8
GLOVE_SIZE = 200
PHONETIC_EMB_SIZE = 200
TWEET_SIZE = 140


def extract_tweet_pair_from_hashtag_datas(hashtag_datas, hashtag_name, tweet_size=TWEET_SIZE):
    for hashtag_data in hashtag_datas:
        current_hashtag_name = hashtag_data[0]
        if current_hashtag_name == hashtag_name:
            np_tweet_pairs = hashtag_data[1]
            np_first_tweets = np_tweet_pairs[:, :TWEET_SIZE]
            np_second_tweets = np_tweet_pairs[:, TWEET_SIZE:]
            return np_first_tweets, np_second_tweets
    return None


def extract_tweet_pairs_from_file(hashtag_file):
    '''This script extracts tweet pairs from the file hashtag_file.
    It stores them in an array of tweet pairs, each tweet pair
    being a list of the form [tweet_1_text, tweet_2_text, first_tweet_funnier].
    first_tweet_funnier is 1 if the first tweet is funnier and 0 if the second
    tweet is funnier.'''
    tweets = []
    tweet_ranks = []
    tweet_ids = []

    # Get all tweets in file along with their ranks.
    with open(hashtag_file) as tsv:
        for line in csv.reader(tsv, dialect='excel-tab'):
            tweet_ids.append(int(line[0]))
            tweets.append(line[1])
            tweet_ranks.append(int(line[2]))
    return extract_tweet_pairs(tweets, tweet_ranks, tweet_ids)


def remove_hashtag_from_tweets(tweets):
    tweets_without_hashtags = []
    for tweet in tweets:
        tweet_without_hashtags = ''
        outside_hashtag = True
        for char in tweet:
            if char == '#':
                outside_hashtag = False
            if outside_hashtag:
                tweet_without_hashtags += char
            if not outside_hashtag and char == ' ':
                outside_hashtag = True
        tweets_without_hashtags.append(tweet_without_hashtags)
    return tweets_without_hashtags


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


def format_text_with_hashtag(text, hashtag_replace=None):
    """Split up existing hashtags. If hashtag_replace=None, then hashtags
    existing in tweet will be broken up and placed at the beginning. If
    hashtag_replace='', no hashtag will be added to the beginning. If
    hashtag_replace is a series of words, they will be placed at the beginning
    of the tweet. A # token will be placed after hashtags placed at the beginning
    of the tweet."""
    # If you hit a hashtag symbol, you are inside a hashtag.
    # While inside hashtag, don't add characters to output.
    # You are no longer inside a hashtag if you encounter a space (don't add space to output).
    formatted_text = ''
    formatted_hashtag = ''
    inside_hashtag = False
    for i in range(len(text)):
        if text[i] == '#':
            inside_hashtag = True
            continue
        if inside_hashtag:
            if hashtag_replace is None:
                if text[i].isupper():
                    formatted_hashtag += ' '
                if not text[i].isalpha() and text[i-1].isalpha():
                    formatted_hashtag += ' '
                formatted_hashtag += text[i]
        else:
            if text[i].isalpha() or text[i] == ' ' or text[i] == '@':
                formatted_text += text[i]
            # if text[i] == '_' or text[i] == '-':
            #     formatted_text += ' '
        if text[i] == ' ':
            inside_hashtag = False
    if hashtag_replace is not None:
        formatted_hashtag += hashtag_replace
    if hashtag_replace != '':
        formatted_hashtag += ' # '
    raw_output = (formatted_hashtag + formatted_text).lower()
    return ' '.join(raw_output.split())


def load_tweets_from_hashtag(filename, explicit_hashtag=None):
    """Open hashtag file, and read each line. For each line,
    read a tweet, its corresponding tweet id, and a label that indicates
    if the tweet was a winner (2), top-ten (1) or non-winner (0) tweet.
    Format the tweet and return [tweets, labels, tweet_ids]."""
    tweet_ids = []
    tweets = []
    labels = []
    # Open hashtag file line for line. File is tsv.
    # Tweet is second variable, tweet win/top10/lose status is third variable
    # Replace any Twitter hashtag with a '$'
    with open(filename, 'rb') as f:
        tsvread = csv.reader(f, delimiter='\t')
        for line in tsvread:
            id = line[0]
            tweet = line[1]
            formatted_tweet = format_text_with_hashtag(tweet, hashtag_replace=explicit_hashtag)
            tweet_tokens = nltk.word_tokenize(formatted_tweet)
            tweet_ids.append(int(id))
            tweets.append(' '.join(tweet_tokens).lower())
            labels.append(int(line[2]))

    return tweets, labels, tweet_ids


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
    if char_to_index_path is not None:
        char_to_index = pickle.load(open(char_to_index_path, 'rb'))
        vocab_size = len(char_to_index)
    else:
        char_to_index = None
        vocab_size = 0
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


def load_hashtag_data(directory, hashtag_name):
    """Load first tweet, second tweet, and
    tweet pair winner label for the hashtag file.
    Example hashtag: America_In_4_Words"""
    #print 'Loading hashtag data for %s' % hashtag_name
    np_first_tweets = np.load(open(directory + hashtag_name + '_first_tweet_glove.npy', 'rb'))
    first_tweet_ids = pickle.load(open(directory + hashtag_name + '_first_tweet_ids.cpkl', 'rb'))
    np_second_tweets = np.load(open(directory + hashtag_name + '_second_tweet_glove.npy', 'rb'))
    second_tweet_ids = pickle.load(open(directory + hashtag_name + '_second_tweet_ids.cpkl', 'rb'))
    np_labels = np.load(open(directory + hashtag_name + '_label.npy', 'rb'))
    np_hashtag = np.load(open(directory + hashtag_name + '_hashtag.npy', 'rb'))
    return np_first_tweets, np_second_tweets, np_labels, first_tweet_ids, second_tweet_ids, np_hashtag


def expected_value(np_prob):
    weighted_sum = 0.0
    for i in range(np_prob.size):
        weighted_sum += i * np_prob[i]
    return weighted_sum / np.sum(np_prob)


def find_indices_larger_than_threshold(np_x, n):
    """Returns indices into array np_x, where the value
    at each index is larger than the threshold n. This
    does not include elements where the value is equal
    to the threshold n."""
    largest_value_indices = []

    for index in range(0, np_x.size):
        if np_x[index] > n:
            largest_value_indices.append(index)
    return largest_value_indices

