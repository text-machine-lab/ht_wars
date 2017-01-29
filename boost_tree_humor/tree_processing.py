"""David Donahue 2016. Create feature bucket dataset for the boosted decision tree model."""
import numpy as np
import random
import os
import cPickle as pickle
from collections import Counter

import scipy.spatial.distance
from nltk import pos_tag

from tools import get_hashtag_file_names
from tools import load_tweets_from_hashtag
from tools import extract_tweet_pairs_by_rank
from tools import remove_hashtag_from_tweets
from config import SEMEVAL_HUMOR_TRAIN_DIR, HUMOR_WORD_TO_GLOVE_FILE_PATH
from config import TWEET_PAIR_LABEL_RANDOM_SEED
from config import BOOST_TREE_TWEET_PAIR_TRAINING_DIR
from config import BOOST_TREE_TWEET_PAIR_TESTING_DIR
from twitter_hawk import TwitterHawk
from twitter_hawk import TWITTERHAWK_ADDRESS
from config import SEMEVAL_HUMOR_TRIAL_DIR


# XGBoost Feature Bucket:
# Length of tweet in tokens
# Sentiment tweet
# Sentiment of hashtag
# Distance of average GloVe of tweet with hashtag
# Average distance between token GloVe in tweet to all other tweets
# Number of tokens in hashtag
# Number of verbs, other POS
# Number of out-of-vocabulary words for GloVe embedding
# Number of capital letters
# Is hashtag in the beginning or the end


def main():
    if not os.path.exists(BOOST_TREE_TWEET_PAIR_TRAINING_DIR):
        os.makedirs(BOOST_TREE_TWEET_PAIR_TRAINING_DIR)
    if not os.path.exists(BOOST_TREE_TWEET_PAIR_TESTING_DIR):
        os.makedirs(BOOST_TREE_TWEET_PAIR_TESTING_DIR)

    print 'Starting program'

    generate_tree_model_input_data_from_dir(SEMEVAL_HUMOR_TRAIN_DIR, BOOST_TREE_TWEET_PAIR_TRAINING_DIR)
    generate_tree_model_input_data_from_dir(SEMEVAL_HUMOR_TRIAL_DIR, BOOST_TREE_TWEET_PAIR_TESTING_DIR)


def generate_tree_model_input_data_from_dir(directory, output_dir):
    """Construct feature bucket for the decision tree model. Calculate sentiment for both
    tweets in each tweet pair. Finally, save the feature bucket and labels."""

    # load glove vectors
    word_to_glove = pickle.load(open(HUMOR_WORD_TO_GLOVE_FILE_PATH, 'rb'))

    hashtag_names = get_hashtag_file_names(directory)
    for hashtag_number, hashtag_name in enumerate(hashtag_names):
        print 'Processing hashtag %s [%s/%s]' % (hashtag_name, hashtag_number + 1, len(hashtag_names))

        formatted_hashtag = ' '.join(hashtag_name.split('_')).lower()
        print 'Formatted hashtag: %s' % formatted_hashtag

        tweets, tweet_labels, tweet_ids = load_tweets_from_hashtag(directory + hashtag_name + '.tsv')
        tweets = remove_hashtag_from_tweets(tweets)
        print 'Creating tweet pairs'
        random.seed(TWEET_PAIR_LABEL_RANDOM_SEED)
        tweet_pairs = extract_tweet_pairs_by_rank(tweets, tweet_labels, tweet_ids)
        tweet1 = [tweet_pair[0] for tweet_pair in tweet_pairs]
        tweet1_id = [tweet_pair[1] for tweet_pair in tweet_pairs]
        tweet2 = [tweet_pair[2] for tweet_pair in tweet_pairs]
        tweet2_id = [tweet_pair[3] for tweet_pair in tweet_pairs]
        labels = [tweet_pair[4] for tweet_pair in tweet_pairs]

        # Numpy arrays of features can be added to this list to be automatically inserted into model input.
        list_of_features = []

        print 'Calculating tweet pair sentiment'
        list_of_features.append(calculate_tweet_pair_sentiment(tweets, tweet1, tweet2))

        print 'Calculating hashtag sentiment'
        list_of_features.append(calculate_hashtag_sentiment(len(tweet1), formatted_hashtag))

        print 'Calculating tweet pair lengths'
        list_of_features.append(calculate_tweet_lengths_per_pair(tweet1, tweet2))

        print 'Calculating distance to centroid'
        list_of_features.append(
            calculate_tweet_pair_distance_to_centroid_word_embeddings(tweets, tweet1, tweet2, word_to_glove))

        print 'Calculating number of OOV tokens'
        list_of_features.append(calculate_tweet_pair_oov(tweets, tweet1, tweet2, word_to_glove))

        print 'Calculating average, max and min distance to the hashtag'
        list_of_features.append(
            calculate_tweet_pair_hashtag_distance(tweets, tweet1, tweet2, formatted_hashtag, word_to_glove))

        print 'Calculating POS features'
        list_of_features.append(calculate_tweet_pair_pos(tweets, tweet1, tweet2))

        print 'Features:'
        for i, feature in enumerate(list_of_features):
            print i, feature.shape

        np_data = np.concatenate(list_of_features, axis=1)
        np_labels = np.array(labels)
        print 'Data:', np_data.shape, 'Labels:', np_labels.shape

        labels_filename = output_dir + hashtag_name + '_labels.npy'
        np.save(open(labels_filename, 'wb'), np_labels)
        print 'Labels saved', labels_filename

        data_filename = output_dir + hashtag_name + '_data.npy'
        np.save(open(data_filename, 'wb'), np_data)
        print 'Data saved', data_filename


def calculate_hashtag_sentiment(number_of_examples, hashtag):
    hashtag_sentiment = calculate_sentiment_value_of_lines([hashtag])[0]

    np_hashtag_sentiment = np.tile(hashtag_sentiment, (number_of_examples, 1))
    return np_hashtag_sentiment


def calculate_tweet_lengths_per_pair(tweet1, tweet2):
    assert len(tweet1) == len(tweet2)

    tweet_pair_lengths = []
    for i in range(len(tweet1)):
        tweet1_token_len = len(tweet1[i].split(' '))
        tweet1_char_len = len(tweet1[i])
        tweet2_token_len = len(tweet2[i].split(' '))
        tweet2_char_len = len(tweet2[i])

        tweet_pair_lengths.append([tweet1_token_len, tweet1_char_len, tweet2_token_len, tweet2_char_len])

    np_data = np.array(tweet_pair_lengths)
    return np_data


def give_model_the_label(labels):
    """Will be used to test that model can learn to give good accuracy."""
    return np.reshape(labels, [len(labels), 1])


def calculate_sentiment_value_of_lines(tweets):
    th = TwitterHawk(TWITTERHAWK_ADDRESS)
    tweet_sentiment_input = [{'id': 1234, 'text': tweet} for tweet in tweets]
    tweet_sentiments = th.analyze(tweet_sentiment_input)

    result = [
        (ts['negative'], ts['positive'], ts['neutral'])
        for ts in tweet_sentiments
        ]
    return result


def calculate_tweet_pair_sentiment(tweets, tweet1, tweet2):
    """Use TwitterHawk URL to generate sentiment per tweet, for each tweet in a tweet pair. First generates
    a tweet to sentiment mapping, then generates tweet pairs using a function in tools.py. Looks up a sentiment
    for each tweet in tweet pair and produces a numpy array of two sentiments per tweet pair for all pairs."""
    tweet_sentiments = calculate_sentiment_value_of_lines(tweets)
    tweet_to_sentiment = {}
    for i in range(len(tweets)):
        tweet_to_sentiment[tweets[i]] = tweet_sentiments[i]

    tweet_pair_sentiments = []
    for i in range(len(tweet1)):
        tweet1_sentiment = tweet_to_sentiment[tweet1[i]]
        tweet2_sentiment = tweet_to_sentiment[tweet2[i]]
        tweet_pair_sentiments.append(tweet1_sentiment + tweet2_sentiment)
    np_data = np.array(tweet_pair_sentiments)
    return np_data


def calculate_tweet_pair_distance_to_centroid_word_embeddings(tweets, tweet1, tweet2, word_to_glove):
    tweet_centroids = {}
    for i, tw in enumerate(tweets):
        centroid = np.mean([word_to_glove[token] for token in tw.split(' ') if token in word_to_glove], axis=0)
        tweet_centroids[tw] = centroid

    all_tweets_centroid = np.mean([c for c in tweet_centroids.values()], axis=0)

    tweet_pair_distances = []
    for i in range(len(tweet1)):
        tweet1_centroid = tweet_centroids[tweet1[i]]
        tweet2_centroid = tweet_centroids[tweet2[i]]

        tweet1_cosine = scipy.spatial.distance.cosine(tweet1_centroid, all_tweets_centroid)
        tweet1_euclidean = scipy.spatial.distance.euclidean(tweet1_centroid, all_tweets_centroid)
        tweet2_cosine = scipy.spatial.distance.cosine(tweet2_centroid, all_tweets_centroid)
        tweet2_euclidean = scipy.spatial.distance.euclidean(tweet2_centroid, all_tweets_centroid)

        tweet_pair_distances.append([tweet1_cosine, tweet1_euclidean, tweet2_cosine, tweet2_euclidean, ])

    np_data = np.array(tweet_pair_distances)
    return np_data


def calculate_tweet_pair_oov(tweets, tweet1, tweet2, word_to_glove):
    tweet_oov = {}
    for i, tw in enumerate(tweets):
        oov = np.sum([1 if token not in word_to_glove else 0 for token in tw.split(' ')], axis=0)
        tweet_oov[tw] = oov

    tweet_pair_oovs = []
    for i in range(len(tweet1)):
        tweet1_oov = tweet_oov[tweet1[i]]
        tweet2_oov = tweet_oov[tweet2[i]]

        tweet_pair_oovs.append([tweet1_oov, tweet2_oov])

    np_data = np.array(tweet_pair_oovs)
    return np_data


def calculate_tweet_pair_hashtag_distance(tweets, tweet1, tweet2, formatted_hashtag, word_to_glove):
    hashtag_embeddings = [word_to_glove[t] for t in formatted_hashtag.split(' ') if t in word_to_glove]
    if len(hashtag_embeddings) != 0:
        hashtag_embedding = np.mean(hashtag_embeddings, axis=0)
    else:
        hashtag_embedding = np.zeros_like(word_to_glove[list(word_to_glove.keys())[0]]) + 0.0001

    tweet_embeddings = {}
    for i, tw in enumerate(tweets):
        embeddings = [word_to_glove[token] for token in tw.split(' ') if token in word_to_glove]
        tweet_embeddings[tw] = embeddings

    tweet_pair_distances = []
    for i in range(len(tweet1)):
        tweet1_distances_cosine = [
            scipy.spatial.distance.cosine(te, hashtag_embedding)
            for te in tweet_embeddings[tweet1[i]]
            ]
        tweet1_distances_euclidean = [
            scipy.spatial.distance.euclidean(te, hashtag_embedding)
            for te in tweet_embeddings[tweet1[i]]
            ]
        tweet2_distances_cosine = [
            scipy.spatial.distance.cosine(te, hashtag_embedding)
            for te in tweet_embeddings[tweet2[i]]
            ]
        tweet2_distances_euclidean = [
            scipy.spatial.distance.euclidean(te, hashtag_embedding)
            for te in tweet_embeddings[tweet2[i]]
            ]

        tweet_pair_distances.append([
            max(tweet1_distances_cosine), min(tweet1_distances_cosine), np.mean(tweet1_distances_cosine),
            max(tweet1_distances_euclidean), min(tweet1_distances_euclidean), np.mean(tweet1_distances_euclidean),
            max(tweet2_distances_cosine), min(tweet2_distances_cosine), np.mean(tweet2_distances_cosine),
            max(tweet2_distances_euclidean), min(tweet2_distances_euclidean), np.mean(tweet2_distances_euclidean),
        ])

    np_data = np.array(tweet_pair_distances)
    return np_data


def calculate_tweet_pair_pos(tweets, tweet1, tweet2):
    # possible_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
    #                  'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
    #                  'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '.']
    possible_tags = ['NN', 'NNS', 'JJ', 'IN', 'DT', 'VBP', 'NNP', 'VB', 'VBD', 'RB'] # top10 tags on training data
    tag2id = {t: i for i, t in enumerate(possible_tags)}


    tweet_pos = {}
    for i, tw in enumerate(tweets):
        tags = [tt[1] for tt in pos_tag(tw.split(' '))]  # pos_tag returns a list of (token, tag) pairs

        tags_counts = Counter(tags)

        np_tags = np.zeros(len(possible_tags))
        for tag, counts in tags_counts.most_common():
            if tag in tag2id:
                np_tags[tag2id[tag]] = counts

        tweet_pos[tw] = np_tags

    tweet_pair_tags = []
    for i in range(len(tweet1)):
        tweet1_pos = tweet_pos[tweet1[i]]
        tweet2_pos = tweet_pos[tweet2[i]]

        tweet_pair_tags.append(np.hstack([tweet1_pos, tweet2_pos]))

    np_data = np.array(tweet_pair_tags)
    return np_data


if __name__ == '__main__':
    main()
