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


def main():
    print 'Starting program'
    print sys.argv
    movie_entries = pop_culture.load_movie_titles()
    movie_titles = [movie_entry[0] for movie_entry in movie_entries]
    song_entries = pop_culture.load_song_titles()
    song_titles = [song_entry[0] for song_entry in song_entries]
    tv_show_entries = pop_culture.load_tv_show_titles()
    tv_show_titles = [tv_show_entry[0] for tv_show_entry in tv_show_entries]
    book_entries = pop_culture.load_book_titles()
    book_titles = [book_entry[0] for book_entry in book_entries]
    pcf_gen = pcfg.PopCultureFeatureGenerator(song_titles, book_titles, tv_show_titles, movie_titles)

    if not os.path.exists(config.HUMOR_TRAIN_POP_CULTURE_FEATURE_DIR):
        os.makedirs(config.HUMOR_TRAIN_POP_CULTURE_FEATURE_DIR)

    train_hashtag_names = tools.get_hashtag_file_names(config.SEMEVAL_HUMOR_TRAIN_DIR)
    print train_hashtag_names
    for each_hashtag_name in train_hashtag_names:
        tweets, tweet_ranks, tweet_ids = tools.load_tweets_from_hashtag(os.path.join(config.SEMEVAL_HUMOR_TRAIN_DIR, each_hashtag_name + '.tsv'))
        random.seed(config.TWEET_PAIR_LABEL_RANDOM_SEED)
        np_hashtag_features, np_labels, tweet_pair_ids = pcf_gen.generate(tweets, tweet_ranks, tweet_ids)
        np.save(open(os.path.join(config.HUMOR_TRAIN_POP_CULTURE_FEATURE_DIR, each_hashtag_name + '_features.npy'), 'wb'), np_hashtag_features)
        np.save(open(os.path.join(config.HUMOR_TRAIN_POP_CULTURE_FEATURE_DIR, each_hashtag_name + '_labels.npy'), 'wb'), np_labels)
        pkl.dump(tweet_pair_ids, open(os.path.join(config.HUMOR_TRAIN_POP_CULTURE_FEATURE_DIR, each_hashtag_name + '_tweet_ids.npy'), 'wb'))


if __name__ == '__main__':
    main()