"""David Donahue 2016. This script preprocesses the data for the humor model.
At the very least, the script is designed to prepare all tweets in the hashtag wars dataset.
It will load all tweet pairs for each hashtag. It will look up all glove embeddings for all tweet
pairs in the hashtag. It will generate pronunciation embeddings for all tweet pairs in the hashtag.
For each hashtag, it will generate and save a numpy array for each tweet. The numpy array will
be shaped m x n x e where m is the number of tweet pairs in the hashtag, n is the max tweet size and
e is the concatenated size of both the phonetic and glove embeddings."""
import tools
import tools_tf
import config
import os
import sys
import cPickle as pickle
import numpy as np
import random
import nltk
import humor_formatter




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
    word_to_phonetic = pickle.load(open(config.HUMOR_WORD_TO_PHONETIC_FILE_PATH, 'rb'))
    print 'Length of word_to_phonetic: %s' % len(word_to_phonetic.keys())
    print 'Length of phonetic embedding: %s' % len(word_to_phonetic['the'])
    word_to_glove = pickle.load(open(config.HUMOR_WORD_TO_GLOVE_FILE_PATH, 'rb'))
    convert_tweets_to_embedding_tweet_pairs(word_to_glove,
                                            word_to_phonetic,
                                            config.SEMEVAL_HUMOR_TRAIN_DIR,
                                            config.HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR)
    convert_tweets_to_embedding_tweet_pairs(word_to_glove,
                                            word_to_phonetic,
                                            config.SEMEVAL_HUMOR_TRIAL_DIR,
                                            config.HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR)


def create_vocabulary_and_glove_phonetic_mappings():
    hf = humor_formatter.HumorFormatter()

    """For all hashtags, for the first and second tweet in all tweet pairs separately,
    for all words in the tweet, look up a glove embedding and generate a phonetic embedding.
    Save everything."""
    print 'Creating vocabulary, phonetic embeddings and GloVe mappings (may take a while)'
    train_hashtag_names = tools.get_hashtag_file_names(config.SEMEVAL_HUMOR_TRAIN_DIR)
    hf = humor_formatter.HumorFormatter()
    for hashtag_name in train_hashtag_names:
        tweets, labels, tweet_ids = tools.load_tweets_from_hashtag(config.SEMEVAL_HUMOR_TRAIN_DIR + hashtag_name + '.tsv')
        f_tweets = format_tweet_text(tweets)
        hf.update_vocab(f_tweets)

    test_hashtag_names = tools.get_hashtag_file_names(config.SEMEVAL_HUMOR_TRIAL_DIR)
    for hashtag_name in test_hashtag_names:
        tweets, labels, tweet_ids = tools.load_tweets_from_hashtag(config.SEMEVAL_HUMOR_TRIAL_DIR + hashtag_name + '.tsv')
        f_tweets = format_tweet_text(tweets)
        hf.update_vocab(f_tweets)

    word_to_glove = look_up_glove_embeddings(hf.vocabulary)
    index_to_phonetic = tools_tf.generate_phonetic_embs_from_words(hf.vocabulary, config.CMU_CHAR_TO_INDEX_FILE_PATH,
                                                                   config.CMU_PHONE_TO_INDEX_FILE_PATH)
    word_to_phonetic = create_dictionary_mapping(hf.vocabulary, index_to_phonetic)
    print 'Size of vocabulary: %s' % len(hf.vocabulary)
    print 'Number of GloVe vectors found: %s' % len(word_to_glove)
    print 'Size of a GloVe vector: %s' % len(word_to_glove['the'])
    # print 'Size of a phonetic embedding: %s' % len(word_to_phonetic['the'])
    print 'Saving %s' % config.HUMOR_WORD_TO_GLOVE_FILE_PATH
    pickle.dump(word_to_glove, open(config.HUMOR_WORD_TO_GLOVE_FILE_PATH, 'wb'))
    print 'Saving %s' % config.HUMOR_WORD_TO_PHONETIC_FILE_PATH
    pickle.dump(word_to_phonetic, open(config.HUMOR_WORD_TO_PHONETIC_FILE_PATH, 'wb'))
    print 'Saving %s' % config.HUMOR_INDEX_TO_WORD_FILE_PATH
    pickle.dump(hf.vocabulary, open(config.HUMOR_INDEX_TO_WORD_FILE_PATH, 'wb'))


def format_tweet_text(tweets, explicit_hashtag=None):
    formatted_tweets = []
    for each_tweet in tweets:
        each_formatted_tweet = tools.format_text_for_embedding_model(each_tweet, hashtag_replace=explicit_hashtag)
        tweet_tokens = nltk.word_tokenize(each_formatted_tweet)
        formatted_tweets.append(' '.join(tweet_tokens).lower())
    return formatted_tweets


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
        hashtag_names = tools.get_hashtag_file_names(tweet_input_dir)

    hf = humor_formatter.HumorFormatter(word_to_glove=word_to_glove, word_to_phonetic=word_to_phonetic)

    for hashtag_name in hashtag_names:
        print 'Loading hashtag: %s' % hashtag_name
        random.seed(config.TWEET_PAIR_LABEL_RANDOM_SEED + hashtag_name)
        formatted_hashtag_name = ' '.join(hashtag_name.split('_')).lower()
        tweets, labels, tweet_ids = tools.load_tweets_from_hashtag(tweet_input_dir + hashtag_name + '.tsv')  # formatted_hashtag_name)
        tweets = format_tweet_text(tweets, explicit_hashtag=formatted_hashtag_name)
        np_tweet1_gloves, np_tweet2_gloves, tweet1_id, tweet2_id, np_labels = \
            hf.format(tweets, labels, tweet_ids)
        # Save
        print 'Saving embedding vector tweet pairs with labels'
        np.save(open(tweet_pair_output_dir + hashtag_name + '_label.npy', 'wb'), np_labels)
        np.save(open(tweet_pair_output_dir + hashtag_name + '_first_tweet_glove.npy', 'wb'),
                np_tweet1_gloves)
        pickle.dump(tweet1_id, open(tweet_pair_output_dir + hashtag_name + '_first_tweet_ids.cpkl', 'wb'))
        np.save(open(tweet_pair_output_dir + hashtag_name + '_second_tweet_glove.npy', 'wb'),
                np_tweet2_gloves)
        pickle.dump(tweet2_id, open(tweet_pair_output_dir + hashtag_name + '_second_tweet_ids.cpkl', 'wb'))


def look_up_glove_embeddings(index_to_word):
    """Find a GloVe embedding for each word in
    index_to_word, if it exists. Create a dictionary
    mapping from words to GloVe vectors and return it."""
    word_to_glove = {}
    with open(config.WORD_VECTORS_FILE_PATH, 'rb') as f:
        for line in f:
            line_tokens = line.split()
            glove_word = line_tokens[0].lower()
            if glove_word in index_to_word:
                glove_emb = [float(line_token) for line_token in line_tokens[1:]]
                word_to_glove[glove_word] = glove_emb

    return word_to_glove


def build_vocabulary(self, lines, vocabulary=None, max_word_size=15):
    if vocabulary is None:
        vocabulary = []
    for line in lines:
        tokens = line.split()
        for word in tokens:
            if len(word) < max_word_size and word not in vocabulary:
                vocabulary.append(word)
    return vocabulary


def format_hashtag(hashtag_name):
    """Remove # and _ symbols and convert to lowercase."""
    name_without_hashtag = hashtag_name.replace('#', '')
    hashtag_with_spaces = name_without_hashtag.replace('_', ' ')
    return hashtag_with_spaces.lower()


def convert_hashtag_to_embedding_tweet_pairs(tweet_input_dir, hashtag_name, word_to_glove, word_to_phonetic):
    """Load a tweets from a hashtag by its directory and name. Convert tweets to tweet pairs and return.
    tweet_input_dir - location of tweet .tsv file
    hashtag_name - name of hashtag file without .tsv extension
    word_to_glove - dictionary mapping from words to glove vectors
    word_to_phonetic - dictionary mapping from words to phonetic embeddings
    Returns:
    np_tweet1_gloves - numpy array of glove/phonetic vectors for all first tweets
    np_tweet2_gloves - numpy array of glove/phonetic vectors for all second tweets
    tweet1_id - tweet id of all first tweets in np_tweet1_gloves
    tweet2_id - tweet id of all second tweets in np_tweet2_gloves
    np_label - numpy array of funnier tweet labels; None if hashtag does not contain labels
    np_hashtag_gloves - numpy array of glove/phonetic vectors for hashtag name"""
    formatted_hashtag_name = ' '.join(hashtag_name.split('_')).lower()
    tweets, labels, tweet_ids = tools.load_tweets_from_hashtag(tweet_input_dir + hashtag_name + '.tsv')  # formatted_hashtag_name)
    tweets = format_tweet_text(tweets, explicit_hashtag=formatted_hashtag_name)
    random.seed(config.TWEET_PAIR_LABEL_RANDOM_SEED + hashtag_name)
    if len(labels) > 0:
        tweet_pairs = tools.extract_tweet_pairs_by_rank(tweets, labels, tweet_ids)
    else:
        tweet_pairs = tools.extract_tweet_pairs_by_combination(tweets, tweet_ids)
    tweet1 = [tweet_pair[0] for tweet_pair in tweet_pairs]
    tweet1_id = [tweet_pair[1] for tweet_pair in tweet_pairs]
    tweet2 = [tweet_pair[2] for tweet_pair in tweet_pairs]
    tweet2_id = [tweet_pair[3] for tweet_pair in tweet_pairs]
    np_label = None
    if len(labels) > 0:
        labels = [tweet_pair[4] for tweet_pair in tweet_pairs]
        np_label = np.array(labels)
    np_hashtag_gloves_col = tools.convert_tweet_to_embeddings([formatted_hashtag_name], word_to_glove, word_to_phonetic,
                                                        config.HUMOR_MAX_WORDS_IN_HASHTAG, config.GLOVE_EMB_SIZE, config.PHONETIC_EMB_SIZE)
    np_hashtag_gloves = np.repeat(np_hashtag_gloves_col, len(tweet1), axis=0)
    np_tweet1_gloves = tools.convert_tweet_to_embeddings(tweet1, word_to_glove, word_to_phonetic, config.HUMOR_MAX_WORDS_IN_TWEET,
                                                   config.GLOVE_EMB_SIZE, config.PHONETIC_EMB_SIZE)
    np_tweet2_gloves = tools.convert_tweet_to_embeddings(tweet2, word_to_glove, word_to_phonetic, config.HUMOR_MAX_WORDS_IN_TWEET,
                                                   config.GLOVE_EMB_SIZE, config.PHONETIC_EMB_SIZE)
    return np_tweet1_gloves, np_tweet2_gloves, tweet1_id, tweet2_id, np_label, np_hashtag_gloves



if __name__ == '__main__':
    main()