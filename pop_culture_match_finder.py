"""David Donahue 2016. Finds the number of matches of titles in the HashtagWars corpus using the pop culture database."""
import config
import os
import tools
import pop_culture
from nltk.corpus import stopwords


def find_movie_matches():
    tweets, labels, tweet_ids = tools.load_tweets_from_hashtag(os.path.join(config.SEMEVAL_HUMOR_TRAIN_DIR, 'Elderly_Movies.tsv'))
    print 'Number of tweets: %s' % len(tweets)
    movies = pop_culture.load_movie_titles()
    movie_names = [movie[0] for movie in movies]
    num_found = 0
    for i, each_tweet in enumerate(tweets):
        titles_found = pop_culture.find_titles_in_tweet(each_tweet, movie_names)
        if len(titles_found) > 0:
            num_found += 1
        for each_title in titles_found:
            print each_tweet + ": " + str(each_title) + " => " + str(labels[i])
    print 'Number of tweets with titles found: %s' % num_found
    print 'Found Fraction: %s' % (num_found * 1.0 / len(tweets))


def find_book_matches():
    tweets, labels, tweet_ids = tools.load_tweets_from_hashtag(os.path.join(config.SEMEVAL_HUMOR_TRAIN_DIR, 'Crapper_Books.tsv'))
    print 'Number of tweets: %s' % len(tweets)
    movies = pop_culture.load_book_titles()
    movie_names = [movie[0] for movie in movies]
    num_found = 0
    for i, each_tweet in enumerate(tweets):
        titles_found = pop_culture.find_titles_in_tweet(each_tweet, movie_names)
        if len(titles_found) > 0:
            num_found += 1
        for each_title in titles_found:
            print each_tweet + ": " + str(each_title) + " => " + str(labels[i])
    print 'Number of tweets with titles found: %s' % num_found
    print 'Found Fraction: %s' % (num_found * 1.0 / len(tweets))


def find_song_matches():
    tweets, labels, tweet_ids = tools.load_tweets_from_hashtag(os.path.join(config.SEMEVAL_HUMOR_TRAIN_DIR, 'Dad_Songs.tsv'))
    print 'Number of tweets: %s' % len(tweets)
    movies = pop_culture.load_song_titles()
    movie_names = [movie[0] for movie in movies]
    num_found = 0
    for i, each_tweet in enumerate(tweets):
        titles_found = pop_culture.find_titles_in_tweet(each_tweet, movie_names)
        if len(titles_found) > 0:
            num_found += 1
        for each_title in titles_found:
            print each_tweet + ": " + str(each_title) + " => " + str(labels[i])
    print 'Number of tweets with titles found: %s' % num_found
    print 'Found Fraction: %s' % (num_found * 1.0 / len(tweets))


def find_tv_show_matches():
    tweets, labels, tweet_ids = tools.load_tweets_from_hashtag(os.path.join(config.SEMEVAL_HUMOR_TRAIN_DIR, 'Historical_TV_Shows.tsv'))
    print 'Number of tweets: %s' % len(tweets)
    movies = pop_culture.load_tv_show_titles()
    movie_names = [movie[0] for movie in movies]
    num_found = 0
    for i, each_tweet in enumerate(tweets):
        titles_found = pop_culture.find_titles_in_tweet(each_tweet, movie_names)
        if len(titles_found) > 0:
            num_found += 1
        for each_title in titles_found:
            print each_tweet + ": " + str(each_title) + " => " + str(labels[i])
    print 'Number of tweets with titles found: %s' % num_found
    print 'Found Fraction: %s' % (num_found * 1.0 / len(tweets))


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