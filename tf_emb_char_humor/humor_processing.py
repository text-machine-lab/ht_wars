"""David Donahue 2016. This script preprocesses the data for the humor model.
At the very least, the script is designed to prepare all tweets in the hashtag wars dataset.
It will load all tweet pairs for each hashtag. It will look up all glove embeddings for all tweet
pairs in the hashtag. It will generate pronunciation embeddings for all tweet pairs in the hashtag.
For each hashtag, it will generate and save a numpy array for each tweet. The numpy array will
be shaped m x n x e where m is the number of tweet pairs in the hashtag, n is the max tweet size and
e is the concatenated size of both the phonetic and glove embeddings."""
from tools import get_hashtag_file_names
from tools import convert_hashtag_to_embedding_tweet_pairs
from config import SEMEVAL_HUMOR_TRAIN_DIR, HUMOR_MAX_WORDS_IN_TWEET, HUMOR_MAX_WORDS_IN_HASHTAG, GLOVE_EMB_SIZE, \
    PHONETIC_EMB_SIZE
from config import SEMEVAL_HUMOR_TRIAL_DIR
from config import WORD_VECTORS_FILE_PATH
from config import HUMOR_INDEX_TO_WORD_FILE_PATH
from config import HUMOR_WORD_TO_GLOVE_FILE_PATH
from config import HUMOR_WORD_TO_PHONETIC_FILE_PATH
from tools import extract_tweet_pairs_by_rank
from config import HUMOR_MAX_WORDS_IN_TWEET
from config import HUMOR_MAX_WORDS_IN_HASHTAG
from config import GLOVE_EMB_SIZE
from config import PHONETIC_EMB_SIZE
from tools import load_tweets_from_hashtag
from config import HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR
from config import HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR
from config import CMU_CHAR_TO_INDEX_FILE_PATH
from config import CMU_PHONE_TO_INDEX_FILE_PATH
from tf_tools import generate_phonetic_embs_from_words
import os
import sys
import cPickle as pickle
import numpy as np
import random


def main():
    print 'Starting program'
    if not len(sys.argv) > 1:
        print 'Please specify argument: vocabulary|tweet_pairs'

    else:
        if sys.argv[1] == 'vocabulary' or sys.argv[1] == 'all':
            create_vocabulary_and_glove_phonetic_mappings()
        if sys.argv[1] == 'tweet_pairs' or sys.argv[1] == 'all':
            """Load vocabulary created from vocabulary step. Load tweets
            and create tweet pairs using winner/top10/loser labels. Split tweet pairs into
            first tweet, second tweet, and first_tweet_funnier label. Convert all left tweets
            and all right tweets into glove embeddings per word. Save numpy array for left tweets,
            right tweets, and labels. Do this for both training and trial datasets, and save both."""
            print 'Creating tweet pairs (may take a while)'
            convert_all_tweets_to_tweet_pairs_represented_by_glove_and_phonetic_embeddings()


def convert_all_tweets_to_tweet_pairs_represented_by_glove_and_phonetic_embeddings():
    vocabulary = pickle.load(open(HUMOR_INDEX_TO_WORD_FILE_PATH, 'rb'))
    word_to_glove = pickle.load(open(HUMOR_WORD_TO_GLOVE_FILE_PATH, 'rb'))
    word_to_phonetic = pickle.load(open(HUMOR_WORD_TO_PHONETIC_FILE_PATH, 'rb'))
    convert_tweets_to_embedding_tweet_pairs(word_to_glove,
                                            word_to_phonetic,
                                            SEMEVAL_HUMOR_TRAIN_DIR,
                                            HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR)
    convert_tweets_to_embedding_tweet_pairs(word_to_glove,
                                            word_to_phonetic,
                                            SEMEVAL_HUMOR_TRIAL_DIR,
                                            HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR)


def create_vocabulary_and_glove_phonetic_mappings():
    """For all hashtags, for the first and second tweet in all tweet pairs separately,
    for all words in the tweet, look up a glove embedding and generate a phonetic embedding.
    Save everything."""
    print 'Creating vocabulary, phonetic embeddings and GloVe mappings (may take a while)'
    train_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRAIN_DIR)
    vocabulary = []
    for hashtag_name in train_hashtag_names:
        tweets, labels, tweet_ids = load_tweets_from_hashtag(SEMEVAL_HUMOR_TRAIN_DIR + hashtag_name + '.tsv')
        vocabulary = build_vocabulary(tweets, vocabulary=vocabulary)

    test_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRIAL_DIR)
    for hashtag_name in test_hashtag_names:
        tweets, labels, tweet_ids = load_tweets_from_hashtag(SEMEVAL_HUMOR_TRIAL_DIR + hashtag_name + '.tsv')
        vocabulary = build_vocabulary(tweets, vocabulary=vocabulary)

    word_to_glove = look_up_glove_embeddings(vocabulary)
    index_to_phonetic = generate_phonetic_embs_from_words(vocabulary, CMU_CHAR_TO_INDEX_FILE_PATH,
                                                          CMU_PHONE_TO_INDEX_FILE_PATH)
    word_to_phonetic = create_dictionary_mapping(vocabulary, index_to_phonetic)
    print 'Size of vocabulary: %s' % len(vocabulary)
    print 'Number of GloVe vectors found: %s' % len(word_to_glove)
    print 'Size of a GloVe vector: %s' % len(word_to_glove['the'])
    print 'Size of a phonetic embedding: %s' % len(word_to_phonetic['the'])
    print 'Saving %s' % HUMOR_INDEX_TO_WORD_FILE_PATH
    pickle.dump(vocabulary, open(HUMOR_INDEX_TO_WORD_FILE_PATH, 'wb'))
    print 'Saving %s' % HUMOR_WORD_TO_GLOVE_FILE_PATH
    pickle.dump(word_to_glove, open(HUMOR_WORD_TO_GLOVE_FILE_PATH, 'wb'))
    print 'Saving %s' % HUMOR_WORD_TO_PHONETIC_FILE_PATH
    pickle.dump(word_to_phonetic, open(HUMOR_WORD_TO_PHONETIC_FILE_PATH, 'wb'))


def create_dictionary_mapping(first_list, second_list):
    """Both lists must be the same length. First list is key.
    Second list is value in returned dictionary.
    first_list - list to be interpretted as keys
    second_list - list to be interpretted as values"""
    assert len(first_list) == len(second_list)
    mapping_dict = {}
    for i in range(len(first_list)):
        mapping_dict[first_list[i]] = second_list[i]
    return mapping_dict


def convert_tweets_to_embedding_tweet_pairs(word_to_glove, word_to_phonetic, tweet_input_dir, tweet_pair_output_dir, hashtag_names=None):
    """Create output directory if it doesn't exist. For all hashtags, generate tweet pairs from all tweets by comparing
    winner with top-ten, top-ten with loser, and winner with loser tweets. Convert each word in each tweet of a tweet pair into
    both a GloVe embedding and phonetic embedding. For each hashtag, produce a numpy array for the first tweet in each pair, produce
    a numpy array for the second tweet in each pair, and produce a label array to tell which one is funnier (1 indicates first tweet is funnier).
    For each row of each tweet vector, insert the GloVe and phonetic embeddings for each word, and insert padding up up to the max words in tweet.
    Can specify hashtag names to convert to embedding tweet pairs instead of all hashtags in tweet_input_dir.

    word_to_glove - a dictionary mapping from words to glove vectors
    word_to_phonetic - a dictionary mapping from words to phonetic embedding vectors
    tweet_input_dir - directory of hashtags to convert into embedding tweet pairs
    tweet_pair_output_dir - output directory, where to store hashtag tweet pair numpy arrays
    hashtag_names - default None; can be used to set specific hashtags to convert"""
    if not os.path.exists(tweet_pair_output_dir):
        os.makedirs(tweet_pair_output_dir)
    if hashtag_names is None:
        hashtag_names = get_hashtag_file_names(tweet_input_dir)

    print hashtag_names

    for hashtag_name in hashtag_names:
        print 'Loading hashtag: %s' % hashtag_name
        np_tweet1_gloves, np_tweet2_gloves, tweet1_id, tweet2_id, np_label, np_hashtag_gloves = \
            convert_hashtag_to_embedding_tweet_pairs(tweet_input_dir, hashtag_name, word_to_glove, word_to_phonetic)
        # Save
        print 'Saving embedding vector tweet pairs with labels'
        np.save(open(tweet_pair_output_dir + hashtag_name + '_label.npy', 'wb'), np_label)
        np.save(open(tweet_pair_output_dir + hashtag_name + '_hashtag.npy', 'wb'), np_hashtag_gloves)
        np.save(open(tweet_pair_output_dir + hashtag_name + '_first_tweet_glove.npy', 'wb'), np_tweet1_gloves)
        pickle.dump(tweet1_id, open(tweet_pair_output_dir + hashtag_name + '_first_tweet_ids.cpkl', 'wb'))
        np.save(open(tweet_pair_output_dir + hashtag_name + '_second_tweet_glove.npy', 'wb'),
                np_tweet2_gloves)
        pickle.dump(tweet2_id, open(tweet_pair_output_dir + hashtag_name + '_second_tweet_ids.cpkl', 'wb'))


def look_up_glove_embeddings(index_to_word):
    """Find a GloVe embedding for each word in
    index_to_word, if it exists. Create a dictionary
    mapping from words to GloVe vectors and return it."""
    word_to_glove = {}
    with open(WORD_VECTORS_FILE_PATH, 'rb') as f:
        for line in f:
            line_tokens = line.split()
            glove_word = line_tokens[0].lower()
            if glove_word in index_to_word:
                glove_emb = [float(line_token) for line_token in line_tokens[1:]]
                word_to_glove[glove_word] = glove_emb

    return word_to_glove


def format_hashtag(hashtag_name):
    """Remove # and _ symbols and convert to lowercase."""
    name_without_hashtag = hashtag_name.replace('#', '')
    hashtag_with_spaces = name_without_hashtag.replace('_', ' ')
    return hashtag_with_spaces.lower()


def build_vocabulary(lines, vocabulary=None, max_word_size=15):
    if vocabulary is None:
        vocabulary = []
    for line in lines:
        tokens = line.split()
        for word in tokens:
            if len(word) < max_word_size and word not in vocabulary:
                vocabulary.append(word)
    return vocabulary


if __name__ == '__main__':
    main()