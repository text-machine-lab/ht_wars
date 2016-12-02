'''David Donahue 2016. This script is intended to process the #HashtagWars dataset into an appropriate format
for training a model. This script extracts tweets for all hashtags in the training data located in the current
directory (training_dir located in ./) and generates tweet pairs from them. Tweets are categorized as 2 (winning tweet),
1 (one of top-ten tweets) or 0 (did not make it into top ten). Pairs are generated as follows:

    1.) Matching winning tweet with each other tweet in the top ten
    2.) Matching each tweet in the top ten with each non-winning tweet
    
Output of this script is saved to output_dir, in the form of hashtag files. Each file is a numpy array (.npy) that holds
all the tweet pairs for that hashtag. The array is of the dimension tweet_pairs by (2 * max tweet length). Each row is
then a tweet pair, with the first max_tweet_length elements being the first tweet, and the second max_tweet_length elements
being the second tweet. Each element is an index to a character that appears in the dataset. The conversion from a character
to its corresponding index is dictionary that can be found in char_to_index.cpkl, a file found in the ./ directory.
'''
from os import walk
import csv
import cPickle as pickle
import os
import random
import numpy as np
import sys

sys.path.append('../')
from config import SEMEVAL_HUMOR_DIR
from config import HUMOR_TWEET_PAIR_DIR
from config import CHAR_TO_INDEX_FILE_PATH


def main():
    # Find hashtags, create character vocabulary, print dataset statistics, extract/format tweet pairs and save everything.
    print "Processing #HashtagWars data..."
    hashtags = get_hashtag_file_names()
    char_to_index = build_character_vocabulary(hashtags)
    print('Size of character vocabulary: %s' % len(char_to_index))
    output_tweet_statistics(hashtags)
    print 'Extracting tweet pairs...'
    for i in range(len(hashtags)):
        hashtag = hashtags[i]
        data = extract_tweet_pairs_from_file(SEMEVAL_HUMOR_DIR + hashtag + '.tsv')
        np_tweet_pairs, np_tweet_pair_labels = format_tweet_pairs(data, char_to_index)
        save_hashtag_data(np_tweet_pairs, np_tweet_pair_labels, hashtag)
    print 'Saving char_to_index.cpkl containing character vocabulary'
    pickle.dump(char_to_index, open(CHAR_TO_INDEX_FILE_PATH, 'wb'))
    print "Done!"
     
# def main():
#     test_reconstruct_tweets_from_file()


def format_tweet_pairs(data, char_to_index, max_tweet_size=140):
    '''This script converts every character in all tweets into an index.
    It stores each tweet side by side, each tweet constrained to 150 characters long.
    The total matrix is m x 300, for m tweet pairs, two 150 word tweets per row.'''
    # Create numpy matrices to hold tweet pairs and their labels.
    np_tweet_pairs = np.zeros(shape=[len(data), max_tweet_size * 2], dtype=int)
    np_tweet_pair_labels = np.zeros(shape=[len(data)], dtype=int)
    for pair_index in range(len(data)):
        first_tweet = data[pair_index][0]
        second_tweet = data[pair_index][1]
        # Insert label for tweet pair into numpy array.
        np_tweet_pair_labels[pair_index] = data[pair_index][2]
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


def output_tweet_statistics(hashtags):
    '''This function analyzes the dataset and prints statistics for it.
    These statistics have to do with the number of tweets, the largest and average
    length of tweets - for all tweets, top-ten tweets, and winning tweets.'''
    largest_tweet_length = 0
    largest_winning_tweet_length = 0
    number_of_tweets = 0
    number_of_top_ten_tweets = 0
    number_of_winning_tweets = 0
    tweet_length_sum = 0
    winning_tweet_length_sum = 0
    
    # Find tweet length statistics (max, average, std dev) and number of tweets.
    for hashtag in hashtags:
        with open(SEMEVAL_HUMOR_DIR + hashtag + '.tsv') as tsv:
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
        with open(SEMEVAL_HUMOR_DIR + hashtag + '.tsv') as tsv:
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


def build_character_vocabulary(hashtags):
    '''Find all characters special or alphabetical that appear in the dataset.
    Construct a vocabulary that assigns a unique index to each character and
    return that vocabulary. Vocabulary does not include anything with a backslash.'''
    characters = []
    characters.append('')
    #Create list of all characters that appear in dataset.
    for hashtag in hashtags:
        with open(SEMEVAL_HUMOR_DIR + hashtag + '.tsv') as tsv:
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


def get_hashtag_file_names():
    '''Returns .tsv file name for each hashtag in the dataset (extension omitted).'''
    f = []
    for (dirpath, dirnames, filenames) in walk(SEMEVAL_HUMOR_DIR):
        f.extend(filenames)
        break
    g = [os.path.splitext(hashtag)[0] for hashtag in f]
    return g


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


def save_hashtag_data(np_tweet_pairs, np_tweet_pair_labels, hashtag):
    print 'Saving data for hashtag %s' % hashtag
    # Create directories if they don't exist
    if not os.path.exists(HUMOR_TWEET_PAIR_DIR):
        os.makedirs(HUMOR_TWEET_PAIR_DIR)
    # Save hashtag tweet pair data into training or testing folders depending on training_hashtag
    np.save(HUMOR_TWEET_PAIR_DIR + hashtag + '_pairs.npy', np_tweet_pairs)
    np.save(HUMOR_TWEET_PAIR_DIR + hashtag + '_labels.npy', np_tweet_pair_labels)


### Unit Tests ###
def test_get_hashtag_file_names():
    file_names = get_hashtag_file_names()
    for name in file_names:
        assert '.tsv' not in name
    print len(file_names)
    assert len(file_names) == 101 #Number of hashtags in dataset


def test_reconstruct_tweets_from_file():
    max_tweet_size = 140
    hashtags = get_hashtag_file_names()
    char_to_index = pickle.load(open('char_to_index.cpkl', 'rb'))
    index_to_char = {v: k for k, v in char_to_index.items()}
    for (dirpath, dirnames, filenames) in walk('.'):
        for filename in filenames:
            if '_pairs.npy' in filename:
                tweets = []
                np_tweet_pairs = np.load(os.path.join(dirpath, filename))
                for i in range(np_tweet_pairs.shape[0]):
                    tweet_1_indices = np_tweet_pairs[i][:max_tweet_size]
                    tweet_2_indices = np_tweet_pairs[i][max_tweet_size:]
                    tweet1 = ''.join([index_to_char[tweet_1_indices[i]] for i in range(tweet_1_indices.size)])
                    tweet2 = ''.join([index_to_char[tweet_2_indices[i]] for i in range(tweet_2_indices.size)])
                    tweets.append(tweet1)
                    tweets.append(tweet2)
                tweets = list(set(tweets))
                with open(SEMEVAL_HUMOR_DIR + filename.replace('_pairs.npy', '.tsv')) as tsv:
                    for line in csv.reader(tsv, dialect='excel-tab'):
                        tweet = line[1]
                        if tweet <= max_tweet_size:
                            assert tweet in tweets


def test_training_and_testing_sets_are_disjoint():
    for (dirpath, dirnames, filenames) in walk(HUMOR_TWEET_PAIR_DIR):
        for (dirpath2, dirnames2, filenames2) in walk(test_output_dir):
            for filename in filenames:
                for filename2 in filenames2:
                    assert filename != filename2
    
    
    
    
    
if __name__ == '__main__':
    main()