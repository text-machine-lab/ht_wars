"""David Donahue 2017. Pop Culture Feature Generator. Uses the pop culture corpus to generate features for
the humor model. Feature generator takes in tweets, and detects movie, song, book and tv show titles.
Labels tweets containing movie titles and sorts them into pairs."""
import unittest2
import pop_culture
import numpy as np


class PopCultureFeatureGenerator:
    def __init__(self, song_list, book_list, tv_show_list, movie_list):
        """Constructs the pop culture feature generator.

        song_list - list of song names
        book_list - list of book names
        tv_show_list - list of tv show names
        movie_list - list of movie names
        """
        self.song_list = song_list
        self.book_list = book_list
        self.tv_show_list = tv_show_list
        self.movie_list = movie_list

    def generate(self, tweets):
        """Detect titles in tweets based on hashtag. Divide tweets into tweet pairs.
        Return tweet pair features as numpy array, labelling each tweet in pair as belonging
        to a specific pop culture genre.

        Returns: m by n numpy array of m examples. Each example/row are features for each tweet
        in the pair concatentated together. Features classify each tweet as a song, movie, tv show
        or book."""
        num_features_per_tweet = 4
        if len(tweets) == 0:
            return np.zeros([0, num_features_per_tweet * 2])




class PopCultureFeatureGeneratorTest(unittest2.TestCase):
    def setUp(self):
        movie_list = pop_culture.load_movie_titles()
        song_list = pop_culture.load_song_titles()
        tv_show_list = pop_culture.load_tv_show_titles()
        book_list = pop_culture.load_book_titles()
        self.pcfg = PopCultureFeatureGenerator(song_list, book_list, tv_show_list, movie_list)

    def test_construction(self):
        pcfg_demo = PopCultureFeatureGenerator(1, 2, 3, 4)
        assert pcfg_demo.song_list == 1
        assert pcfg_demo.book_list == 2
        assert pcfg_demo.tv_show_list == 3
        assert pcfg_demo.movie_list == 4

    def test_generate_no_tweets(self):

        np_empty_features = self.pcfg.generate([])
        print np_empty_features.size
        assert np_empty_features.size == 0

    def test_generate_tweet(self):
        tweets = [""]