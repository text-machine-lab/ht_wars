"""David Donahue 2016. This is a place to keep functions that more than one script could potentially use.
This makes a function easily accessible and independent of the application. Functions
here are dependent on data stored in the data/ folder."""
import cPickle as pickle
import csv
import os
import random
from os import walk

import nltk
import numpy as np

from config import TWEET_SIZE, TWEET_PAIR_LABEL_RANDOM_SEED
from config import HUMOR_MAX_WORDS_IN_HASHTAG, HUMOR_MAX_WORDS_IN_TWEET
from config import GLOVE_EMB_SIZE, PHONETIC_EMB_SIZE
from config import SEMEVAL_HUMOR_TRAIN_DIR, HUMOR_TRAIN_TWEET_PAIR_CHAR_DIR


def output_tweet_statistics(hashtags, directory=SEMEVAL_HUMOR_TRAIN_DIR):
    """This function analyzes the dataset and prints statistics for it.
    These statistics have to do with the number of tweets, the largest and average
    length of tweets - for all tweets, top-ten tweets, and winning tweets."""
    largest_tweet_length = 0
    largest_winning_tweet_length = 0
    number_of_tweets = 0
    number_of_top_ten_tweets = 0
    number_of_winning_tweets = 0
    tweet_length_sum = 0
    winning_tweet_length_sum = 0

    # Find tweet length statistics (max, average, std dev) and number of tweets.
    for hashtag in hashtags:
        with open(directory + hashtag + '.tsv') as tsv:
            for line in csv.reader(tsv, dialect='excel-tab'):
                # Count number of tweets, find longest tweet, find average tweet length
                # for all tweets, top ten, and winning.
                tweet_length = len(line[1])
                tweet_rank = int(line[2])
                number_of_tweets += 1
                tweet_length_sum += tweet_length
                if tweet_length > largest_tweet_length:
                    largest_tweet_length = tweet_length
                if tweet_rank == 2:
                    if tweet_length > largest_winning_tweet_length:
                        largest_winning_tweet_length = tweet_length
                    winning_tweet_length_sum += tweet_length
                    number_of_winning_tweets += 1
                if tweet_rank == 1:
                    number_of_top_ten_tweets += 1
    average_tweet_length = (float(tweet_length_sum) / number_of_tweets)
    average_winning_tweet_length = (float(winning_tweet_length_sum) / number_of_winning_tweets)

    # Find standard deviation.
    tweet_std_dev_sum = 0
    winning_tweet_std_dev_sum = 0
    for hashtag in hashtags:
        with open(directory + hashtag + '.tsv') as tsv:
            for line in csv.reader(tsv, dialect='excel-tab'):
                tweet_length = len(line[1])
                tweet_rank = int(line[2])
                tweet_std_dev_sum += abs(tweet_length - average_tweet_length)
                if tweet_rank == 2:
                    winning_tweet_std_dev_sum += abs(tweet_length - average_winning_tweet_length)
    tweet_std_dev = float(tweet_std_dev_sum) / number_of_tweets
    winning_tweet_std_dev = float(winning_tweet_std_dev_sum) / number_of_winning_tweets

    # Print statistics found above.
    print 'The largest tweet length is %s characters' % largest_tweet_length
    print 'The largest winning tweet length is %s characters' % largest_winning_tweet_length
    print 'Number of tweets: %s' % number_of_tweets
    print 'Number of top-ten tweets: %s' % number_of_top_ten_tweets
    print 'Number of winning tweets: %s' % number_of_winning_tweets
    print 'Average tweet length: %s' % average_tweet_length
    print 'Average winning tweet length: %s' % average_winning_tweet_length
    print 'Tweet length standard deviation: %s' % tweet_std_dev
    print 'Winning tweet length standard deviation: %s' % winning_tweet_std_dev


def build_character_vocabulary(hashtags, directory=SEMEVAL_HUMOR_TRAIN_DIR):
    """Find all characters special or alphabetical that appear in the dataset.
    Construct a vocabulary that assigns a unique index to each character and
    return that vocabulary. Vocabulary does not include anything with a backslash."""
    characters = ['']
    #Create list of all characters that appear in dataset.
    for hashtag in hashtags:
        with open(directory + hashtag + '.tsv') as tsv:
            for line in csv.reader(tsv, dialect='excel-tab'):
                for char in line[1]:
                    # If character hasn't been seen before, add it to the vocabulary.
                    if char not in characters:
                        characters.append(char)
    # Create dictionary from list to map from characters to their indices.
    vocabulary = {}
    for i in range(len(characters)):
        vocabulary[characters[i]] = i
    return vocabulary


def save_hashtag_data(np_tweet_pairs, np_tweet_pair_labels, hashtag, directory=HUMOR_TRAIN_TWEET_PAIR_CHAR_DIR):
    print 'Saving data for hashtag %s' % hashtag
    # Create directories if they don't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save hashtag tweet pair data into training or testing folders depending on training_hashtag
    np.save(directory + hashtag + '_pairs.npy', np_tweet_pairs)
    np.save(directory + hashtag + '_labels.npy', np_tweet_pair_labels)


def process_hashtag_data(hashtag_dir, char_to_index_path, tweet_pair_path):
    hashtags = get_hashtag_file_names(hashtag_dir)
    char_to_index = build_character_vocabulary(hashtags, directory=hashtag_dir)
    print('Size of character vocabulary: %s' % len(char_to_index))
    output_tweet_statistics(hashtags, directory=hashtag_dir)
    print 'Extracting tweet pairs...'
    for i in range(len(hashtags)):
        hashtag = hashtags[i]
        random.seed(TWEET_PAIR_LABEL_RANDOM_SEED + hashtag)
        data = extract_tweet_pairs_from_file(hashtag_dir + hashtag + '.tsv')
        np_tweet_pairs, np_tweet_pair_labels = format_tweet_pairs(data, char_to_index)
        save_hashtag_data(np_tweet_pairs, np_tweet_pair_labels, hashtag, directory=tweet_pair_path)
    print 'Saving char_to_index.cpkl containing character vocabulary'
    if char_to_index_path is not None:
        pickle.dump(char_to_index, open(char_to_index_path, 'wb'))
    print "Done!"


def format_tweet_pairs(data, char_to_index, max_tweet_size=140):
    """This script converts every character in all tweets into an index.
    It stores each tweet side by side, each tweet constrained to 150 characters long.
    The total matrix is m x 300, for m tweet pairs, two 150 word tweets per row."""
    labels_exist = (len(data[0]) > 4)
    # Create numpy matrices to hold tweet pairs and their labels.
    np_tweet_pairs = np.zeros(shape=[len(data), max_tweet_size * 2], dtype=int)
    if labels_exist:
        np_tweet_pair_labels = np.zeros(shape=[len(data)], dtype=int)
    else:
        np_tweet_pair_labels = None
    for pair_index in range(len(data)):
        first_tweet = data[pair_index][0]
        second_tweet = data[pair_index][2]
        # Insert label for tweet pair into numpy array.
        if labels_exist:
            np_tweet_pair_labels[pair_index] = data[pair_index][4]
        # Insert first tweet of pair into numpy array.
        for i in range(len(first_tweet)):
            if i < max_tweet_size:
                character = first_tweet[i]
                if character in char_to_index:
                    np_tweet_pairs[pair_index][i] = char_to_index[character]
        # Insert second tweet of pair into numpy array.
        for i in range(len(second_tweet)):
            if i < max_tweet_size:
                character = second_tweet[i]
                if character in char_to_index:
                    np_tweet_pairs[pair_index][i + max_tweet_size] = char_to_index[character]

    return np_tweet_pairs, np_tweet_pair_labels


def convert_tweet_to_embeddings(tweets, word_to_glove, word_to_phonetic, max_number_of_words, glove_size, phonetic_emb_size, lm):
    """Pack GloVe vectors and phonetic embeddings side by side for each word in each tweet as a numpy array. New: Add
    GloVe expectation feature!

    tweets - list of tweet strings
    word_to_glove - dictionary mapping from words to their glove vectors
    word_to_phonetic - dictionary mapping from words to their phonetic embeddings
    max_number_of_words - leave padding to fit all tweets in same space
    glove_size - size of glove vectors used
    phonetic_emb_size - size of phonetic embeddings used
    lm - trained language model"""
    word_embedding_size = glove_size * 2 + phonetic_emb_size  # Make room for GloVe expectation!
    np_tweet_embs = np.zeros([len(tweets), max_number_of_words * word_embedding_size])
    # Calculate GloVe expectation, then calculate actual GloVe and phonetic embeddings.
    # Insert embeddings into feature vector.
    for i in range(len(tweets)):
        np_glove_expectation = compute_glove_expectation(tweets[i], lm, word_to_glove, GLOVE_EMB_SIZE)

        tokens = tweets[i].split()
        for j in range(len(tokens)):
            if j < max_number_of_words:
                if tokens[j] in word_to_glove:
                    np_token_glove = np.array(word_to_glove[tokens[j]])
                    for k in range(glove_size):
                        np_tweet_embs[i, j*word_embedding_size + k] = np_token_glove[k]

                if tokens[j] in word_to_phonetic:
                    np_token_phonetic = np.array(word_to_phonetic[tokens[j]])
                    for k in range(phonetic_emb_size):
                        np_tweet_embs[i, j*word_embedding_size + glove_size + k] = np_token_phonetic[k]

                np_tweet_embs[i, j * word_embedding_size + glove_size + phonetic_emb_size:
                j * word_embedding_size + glove_size * 2 + phonetic_emb_size] = np_glove_expectation[j]

    return np_tweet_embs


def convert_hashtag_to_embedding_tweet_pairs(tweet_input_dir, hashtag_name, word_to_glove, word_to_phonetic, lm):
    """Load a tweets from a hashtag by its directory and name. Convert tweets to tweet pairs and return.

    tweet_input_dir - location of tweet .tsv file
    hashtag_name - name of hashtag file without .tsv extension
    word_to_glove - dictionary mapping from words to glove vectors
    word_to_phonetic - dictionary mapping from words to phonetic embeddings
    lm - language model used to create GloVe expectation
    Returns:
    np_tweet1_gloves - numpy array of glove/phonetic vectors for all first tweets
    np_tweet2_gloves - numpy array of glove/phonetic vectors for all second tweets
    tweet1_id - tweet id of all first tweets in np_tweet1_gloves
    tweet2_id - tweet id of all second tweets in np_tweet2_gloves
    np_label - numpy array of funnier tweet labels; None if hashtag does not contain labels
    np_hashtag_gloves - numpy array of glove/phonetic vectors for hashtag name"""
    formatted_hashtag_name = ' '.join(hashtag_name.split('_')).lower()
    tweets, labels, tweet_ids = load_tweets_from_hashtag(tweet_input_dir + hashtag_name + '.tsv',
                                                         explicit_hashtag=formatted_hashtag_name)  # formatted_hashtag_name)
    random.seed(TWEET_PAIR_LABEL_RANDOM_SEED + hashtag_name)
    if len(labels) > 0:
        tweet_pairs = extract_tweet_pairs_by_rank(tweets, labels, tweet_ids)
    else:
        tweet_pairs = extract_tweet_pairs_by_combination(tweets, tweet_ids)
    tweet1 = [tweet_pair[0] for tweet_pair in tweet_pairs]
    tweet1_id = [tweet_pair[1] for tweet_pair in tweet_pairs]
    tweet2 = [tweet_pair[2] for tweet_pair in tweet_pairs]
    tweet2_id = [tweet_pair[3] for tweet_pair in tweet_pairs]
    np_label = None
    if len(labels) > 0:
        labels = [tweet_pair[4] for tweet_pair in tweet_pairs]
        np_label = np.array(labels)
    np_hashtag_gloves_col = convert_tweet_to_embeddings([formatted_hashtag_name], word_to_glove, word_to_phonetic,
                                                        HUMOR_MAX_WORDS_IN_HASHTAG, GLOVE_EMB_SIZE, PHONETIC_EMB_SIZE, lm)
    np_hashtag_gloves = np.repeat(np_hashtag_gloves_col, len(tweet1), axis=0)
    np_tweet1_gloves = convert_tweet_to_embeddings(tweet1, word_to_glove, word_to_phonetic, HUMOR_MAX_WORDS_IN_TWEET,
                                                   GLOVE_EMB_SIZE, PHONETIC_EMB_SIZE, lm)
    np_tweet2_gloves = convert_tweet_to_embeddings(tweet2, word_to_glove, word_to_phonetic, HUMOR_MAX_WORDS_IN_TWEET,
                                                   GLOVE_EMB_SIZE, PHONETIC_EMB_SIZE, lm)
    return np_tweet1_gloves, np_tweet2_gloves, tweet1_id, tweet2_id, np_label, np_hashtag_gloves


def compute_glove_expectation(tweet, lm, word_to_glove, glove_emb_size):
    glove_expectations = []
    tweet_tokens = tweet.split()
    for i in range(len(tweet_tokens)):
        next_word_dict = lm.calculate_expected_next_word(' '.join(tweet_tokens[:i]))
        glove_expectation = np.zeros(glove_emb_size)
        num_gloves_summed = 0
        if next_word_dict is not None:
            for word in next_word_dict:
                if word in word_to_glove:
                    num_gloves_summed += next_word_dict[word]
                    glove_expectation += np.multiply(np.array(word_to_glove[word]), next_word_dict[word])
        if num_gloves_summed == 0:
            glove_expectations.append(glove_expectation)
        elif num_gloves_summed > 0:
            glove_expectations.append(np.divide(glove_expectation, num_gloves_summed))
    return glove_expectations


def extract_tweet_pair_from_hashtag_datas(hashtag_datas, hashtag_name, tweet_size=TWEET_SIZE):
    for hashtag_data in hashtag_datas:
        current_hashtag_name = hashtag_data[0]
        if current_hashtag_name == hashtag_name:
            np_tweet_pairs = hashtag_data[1]
            np_first_tweets = np_tweet_pairs[:, :tweet_size]
            np_second_tweets = np_tweet_pairs[:, tweet_size:]
            return np_first_tweets, np_second_tweets
    return None


def extract_tweet_pairs_from_file(hashtag_file):
    """This script extracts tweet pairs from the file hashtag_file.
    It stores them in an array of tweet pairs, each tweet pair
    being a list of the form [tweet_1_text, tweet_2_text, first_tweet_funnier].
    first_tweet_funnier is 1 if the first tweet is funnier and 0 if the second
    tweet is funnier."""
    tweets = []
    tweet_ranks = []
    tweet_ids = []

    # Get all tweets in file along with their ranks.
    with open(hashtag_file) as tsv:
        for line in csv.reader(tsv, dialect='excel-tab'):
            tweet_ids.append(int(line[0]))
            tweets.append(line[1])
            if len(line) > 2:
                tweet_ranks.append(int(line[2]))
    # If there are labels, create pairs from rank
    # If there are no labels, create pairs from all combinations of tweets
    if len(tweet_ranks) == 0:
        return extract_tweet_pairs_by_combination(tweets, tweet_ids)
    elif len(tweet_ranks) < len(tweets):
        print 'Warning: Problem reading %s' % hashtag_file
    else:
        return extract_tweet_pairs_by_rank(tweets, tweet_ranks, tweet_ids)


def remove_hashtag_from_tweets(tweets):
    """Takes a list of tweets. For each tweet, if it contains
    a hashtag, that hashtag is removed. Returns the tweets
    without hashtags. Copies input tweets."""
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


def extract_tweet_pairs_by_rank(tweets, tweet_ranks, tweet_ids):
    """Creates pairs of the form [first_tweet, first_tweet_id, second_tweet, second_tweet_id, first_tweet_is_funnier]"""
    winner, winner_ids, top_ten, top_ten_ids, non_winners, non_winner_ids = \
        divide_tweets_by_rank(tweets, tweet_ids, tweet_ranks)

    # Create pairs from non-winning and top-ten tweets.
    pairs = []
    id_pairs = []
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


def extract_tweet_pairs_by_combination(tweets, tweet_ids):
    """Creates tweet pairs out of every combination of tweets. Each
    pair takes on the form [tweet1, tweet1_id, tweet2, tweet2_id].

    tweets - tweets to create pairs from
    tweet_ids - each id is that of tweet with same index in tweets"""
    pairs = []
    for first_index in range(len(tweets)):
        for second_index in range(first_index + 1, len(tweets)):
            pairs.append([tweets[first_index], tweet_ids[first_index],
                         tweets[second_index], tweet_ids[second_index]])
    return pairs


def divide_tweets_by_rank(tweets, tweet_ids, tweet_ranks):
    """Tweets are labelled as 'winner', 'top-ten' or 'non-winner'.
    Divide tweets by their rank and return lists of tweets from
    each rank along with their corresponding ids. Returns six
    lists: winner tweets, winner tweet ids, top ten tweets,
    top ten tweet ids, non winner tweets, non winner tweet ids."""
    non_winners = []
    top_ten = []
    winner = []
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
    return winner, winner_ids, top_ten, top_ten_ids, non_winners, non_winner_ids


def format_text_for_embedding_model(text, hashtag_replace=None):
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
    line_does_not_contain_label = False
    with open(filename, 'rb') as f:
        tsvread = csv.reader(f, delimiter='\t')
        for line in tsvread:
            tweet_id = line[0]
            tweet = line[1]
            formatted_tweet = format_text_for_embedding_model(tweet, hashtag_replace=explicit_hashtag)
            tweet_tokens = nltk.word_tokenize(formatted_tweet)
            tweet_ids.append(int(tweet_id))
            tweets.append(' '.join(tweet_tokens).lower())
            if len(line) > 2:
                if line_does_not_contain_label:
                    print 'Warning: Hashtag labels not formatted correctly.'
                labels.append(int(line[2]))
            else:
                line_does_not_contain_label = True

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
    """Load in tweet pairs per hashtag. Create a list of [hashtag_name, pairs, labels] entries.
    Return tweet pairs, tweet labels, char_to_index.cpkl and vocabulary size."""
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
    """Returns .tsv file name for each hashtag in the dataset (extension omitted)."""
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

