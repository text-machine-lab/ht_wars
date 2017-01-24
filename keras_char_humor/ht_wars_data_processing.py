"""David Donahue 2016. This script is intended to process the #HashtagWars dataset into an appropriate format
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
"""
from os import walk
import csv
import cPickle as pickle
import os
import random
import numpy as np
import sys

sys.path.append('../')
from config import SEMEVAL_HUMOR_TRAIN_DIR
from config import SEMEVAL_HUMOR_TRIAL_DIR
from config import HUMOR_TRAIN_TWEET_PAIR_CHAR_DIR
from config import HUMOR_TRIAL_TWEET_PAIR_CHAR_DIR
from config import HUMOR_CHAR_TO_INDEX_FILE_PATH
from tools import get_hashtag_file_names
from tools import process_hashtag_data

def main():
    # Find hashtags, create character vocabulary, print dataset statistics, extract/format tweet pairs and save everything.
    # Repeat this for both training and trial sets.
    print "Processing #HashtagWars training data..."
    process_hashtag_data(SEMEVAL_HUMOR_TRAIN_DIR, HUMOR_CHAR_TO_INDEX_FILE_PATH, HUMOR_TRAIN_TWEET_PAIR_CHAR_DIR)
    print "Processing #HashtagWars trial data..."
    process_hashtag_data(SEMEVAL_HUMOR_TRIAL_DIR, None, HUMOR_TRIAL_TWEET_PAIR_CHAR_DIR)


def test_reconstruct_tweets_from_file():
    max_tweet_size = 140
    hashtags = get_hashtag_file_names(SEMEVAL_HUMOR_TRAIN_DIR)
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
                with open(SEMEVAL_HUMOR_TRAIN_DIR + filename.replace('_pairs.npy', '.tsv')) as tsv:
                    for line in csv.reader(tsv, dialect='excel-tab'):
                        tweet = line[1]
                        if tweet <= max_tweet_size:
                            assert tweet in tweets

    
if __name__ == '__main__':
    main()