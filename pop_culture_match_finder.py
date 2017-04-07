"""David Donahue 2016. Finds the number of matches of titles in the HashtagWars corpus using the pop culture database."""
import config
import os
import tools
import pop_culture
import numpy as np
from nltk.corpus import stopwords


def find_movie_matches():
    tweets, labels, tweet_ids = tools.load_tweets_from_hashtag(os.path.join(config.SEMEVAL_HUMOR_TRAIN_DIR, 'Elderly_Movies.tsv'))
    print 'Number of tweets: %s' % len(tweets)
    movies = pop_culture.load_tsv_as_2d_list(os.path.join(config.POP_CULTURE_CORPUS_DIR, config.MOVIE_TITLES_LARGE_FILE))
    movie_names = [movie[0] for movie in movies]
    print_titles_that_match_tweets(movie_names, tweets, labels)


def find_book_matches():
    tweets, labels, tweet_ids = tools.load_tweets_from_hashtag(os.path.join(config.SEMEVAL_HUMOR_TRAIN_DIR, 'Crapper_Books.tsv'))
    print 'Number of tweets: %s' % len(tweets)
    movies = pop_culture.load_book_titles()
    movie_names = [movie[0] for movie in movies]
    print_titles_that_match_tweets(movie_names, tweets, labels)


def find_song_matches():
    tweets, labels, tweet_ids = tools.load_tweets_from_hashtag(os.path.join(config.SEMEVAL_HUMOR_TRAIN_DIR, 'Dad_Songs.tsv'))
    print 'Number of tweets: %s' % len(tweets)
    movies = pop_culture.load_song_titles()
    movie_names = [movie[0] for movie in movies]
    print_titles_that_match_tweets(movie_names, tweets, labels)


def find_tv_show_matches():
    tweets, labels, tweet_ids = tools.load_tweets_from_hashtag(os.path.join(config.SEMEVAL_HUMOR_TRAIN_DIR, 'Historical_TV_Shows.tsv'))
    print 'Number of tweets: %s' % len(tweets)
    movies = pop_culture.load_tv_show_titles()
    movie_names = [movie[0] for movie in movies]
    print_titles_that_match_tweets(movie_names, tweets, labels)


def print_titles_that_match_tweets(titles, tweets, labels, threshold=0.6):
    num_found = 0
    num_found_top_ten = 0
    for i, each_tweet in enumerate(tweets):
        titles_found = pop_culture.find_titles_in_tweet(each_tweet, titles, min_frac=0)
        scores_found = [title_found[1] for title_found in titles_found]
        index_largest_score = 0
        if len(titles_found) > 0:
            index_largest_score = np.argmax(scores_found)
            if scores_found[index_largest_score] > threshold:
                num_found += 1
                if labels[i] != 0:
                    num_found_top_ten += 1
            print each_tweet + ", " + str(labels[i]) + ": " + titles_found[index_largest_score][0] + ", " + str(scores_found[index_largest_score])
        else:
            print 'Titles not found: %s' % each_tweet


    print 'Number of tweets with titles found: %s' % num_found
    print 'Found Fraction: %s' % (num_found * 1.0 / len(tweets))
    print 'Found Top Ten Fraction: %s' % (num_found_top_ten * 1.0 / 10)


def main():
    find_movie_matches()
    print
    print
    find_book_matches()
    print
    print
    find_song_matches()
    print
    print
    find_tv_show_matches()
    print
    print
    print stopwords.words('english')


if __name__ == '__main__':
    main()