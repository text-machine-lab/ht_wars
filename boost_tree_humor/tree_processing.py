"""David Donahue 2016. Create feature bucket dataset for the boosted decision tree model."""
import numpy as np
import random
import os
from tools import get_hashtag_file_names
from tools import load_tweets_from_hashtag
from tools import extract_tweet_pairs_by_rank
from tools import remove_hashtag_from_tweets
from config import SEMEVAL_HUMOR_TRAIN_DIR
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
    hashtag_names = get_hashtag_file_names(directory)
    for hashtag_name in hashtag_names:
        print 'Processing hashtag %s' % hashtag_name
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
        formatted_hashtag = ' '.join(hashtag_name.split('_')).lower()
        print 'Formatted hashtag: %s' % formatted_hashtag
        list_of_features.append(calculate_hashtag_sentiment(len(tweet1), formatted_hashtag))

        print 'Calculating tweet pair lengths'
        list_of_features.append(calculate_tweet_lengths_per_pair(tweet1, tweet2))

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
    m = len(tweet1)
    np_tweet_lengths = np.zeros([m, 2])
    for index in range(m):
        np_tweet_lengths[index, 0] = len(tweet1[index].split(' '))
        np_tweet_lengths[index, 1] = len(tweet2[index].split(' '))
    return np_tweet_lengths


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


if __name__ == '__main__':
    main()
