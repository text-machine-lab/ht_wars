"""Copyright 2017 David Donahue. Generates pop culture features using the PopCultureFeatureGenerator.
Saves them into directory of data/hm."""
import numpy as np
import cPickle as pkl
import os
import pcfg
import pop_culture
import tools
import config
import random
import sys


def generate_pop_culture_features_directory(pcf_gen, hashtag_files_dir, feature_output_dir):
    """Given a directory of hashtag files, generates pop culture features for tweet pairs and saves
    them in an output directory.

    pcf_gen - a pop culture feature generator loaded with titles to do the heavy lifting
    hashtag_files_dir - input directory of hashtag files, each containing labelled tweets
    feature_output_dir - directory to output pop culture features, labels and ids

    Returns: float representing fraction of tweets in all tweet pairs that have a title. Average
    over fraction of tweets with title per hashtag."""

    if not os.path.exists(feature_output_dir):
        os.makedirs(feature_output_dir)
    train_hashtag_names = tools.get_hashtag_file_names(hashtag_files_dir)
    sum_fraction_titles_detected = 0.0
    for each_hashtag_name in train_hashtag_names:
        tweets, tweet_ranks, tweet_ids = tools.load_tweets_from_hashtag(os.path.join(hashtag_files_dir, each_hashtag_name + '.tsv'))
        random.seed(config.TWEET_PAIR_LABEL_RANDOM_SEED)
        np_hashtag_features, np_labels, tweet_pair_ids = pcf_gen.generate(tweets, tweet_ranks, tweet_ids)
        sum_fraction_titles_detected += np.mean(np_hashtag_features)
        np.save(open(os.path.join(feature_output_dir, each_hashtag_name + '_features.npy'), 'wb'), np_hashtag_features)
        np.save(open(os.path.join(feature_output_dir, each_hashtag_name + '_labels.npy'), 'wb'), np_labels)
        pkl.dump(tweet_pair_ids, open(os.path.join(feature_output_dir, each_hashtag_name + '_tweet_ids.npy'), 'wb'))
    return sum_fraction_titles_detected / len(train_hashtag_names) * 4.0  # Can only be one of four genres


def main():
    print 'Starting program'
    movie_titles, song_titles, tv_show_titles, book_titles = pop_culture.load_all_genre_title_names()
    pcf_gen = pcfg.PopCultureFeatureGenerator(song_titles, book_titles, tv_show_titles, movie_titles)
    train_fraction_titles_detected = generate_pop_culture_features_directory(pcf_gen, config.SEMEVAL_HUMOR_TRAIN_DIR,
                                                                             config.HUMOR_TRAIN_POP_CULTURE_FEATURE_DIR)
    print 'Training set fraction of titles detected: %s' % train_fraction_titles_detected
    trial_fraction_titles_detected = generate_pop_culture_features_directory(pcf_gen, config.SEMEVAL_HUMOR_TRIAL_DIR,
                                                                             config.HUMOR_TRIAL_POP_CULTURE_FEATURE_DIR)
    print 'Trial set fraction of titles detected: %s' % trial_fraction_titles_detected

if __name__ == '__main__':
    main()