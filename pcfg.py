"""David Donahue 2017. Pop Culture Feature Generator. Uses the pop culture corpus to generate features for
the humor model. Feature generator takes in tweets, and detects movie, song, book and tv show titles.
Labels tweets containing movie titles and sorts them into pairs."""
import pop_culture
import numpy as np
import tools
import unittest2


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

    def classify_tweet_by_genre(self, tweet, min_match_score=0.6):
        """Classifies a tweet as containing a movie, tv show, song, or book title.

        tweet - tweet to examine
        min_match_score - [0-1] minimum match score for a title in the corpus to be
        considered present in the tweet (default: 0.6)

        Returns: array of the format [movie, tv show, song, book]."""
        detected_movie_titles = pop_culture.find_titles_in_tweet(tweet, self.movie_list, min_frac=min_match_score)
        detected_tv_show_titles = pop_culture.find_titles_in_tweet(tweet, self.tv_show_list, min_frac=min_match_score)
        detected_song_titles = pop_culture.find_titles_in_tweet(tweet, self.song_list, min_frac=min_match_score)
        detected_book_titles = pop_culture.find_titles_in_tweet(tweet, self.book_list, min_frac=min_match_score)
        tweet_contains_movie = (len(detected_movie_titles) > 0)
        tweet_contains_tv_show = (len(detected_tv_show_titles) > 0)
        tweet_contains_song = (len(detected_song_titles) > 0)
        tweet_contains_book = (len(detected_book_titles) > 0)
        result = [tweet_contains_movie, tweet_contains_tv_show,
                  tweet_contains_song, tweet_contains_book]
        return result

    def generate(self, tweets, ranks, tweet_ids, funnier_first=False, min_match_score=0.6):
        """Detect titles in tweets based on hashtag. Divide tweets into tweet pairs.
        Return tweet pair features as numpy array, labelling each tweet in pair as belonging
        to a specific pop culture genre.

        tweets - a list of tweets to create pair features from
        ranks - list of labels indicating tweet rank 0-2
        tweet_ids - list of Twitter ids for each tweet
        funnier_first - place funnier tweet first in pair (debugging)

        Returns: m by n numpy array of m examples. Each example/row are features for each tweet
        in the pair concatentated together. Features classify each tweet as a song, movie, tv show
        or book. So each row has eight features, four for each tweet in pair. Each four features
        are of format [movie, tv show, song, book]."""
        assert len(tweets) == len(ranks) and len(ranks) == len(tweet_ids)
        tweets = tools.remove_hashtag_from_tweets(tweets)
        results_per_tweet = []
        # Detect titles in each tweet
        for each_tweet in tweets:
            results_per_tweet.append(self.classify_tweet_by_genre(each_tweet, min_match_score=min_match_score))

        # Construct pairs from tweets and build tweet pair features of pop culture genre label
        tweet_pairs = tools.extract_tweet_pairs_by_rank(tweets, ranks, tweet_ids,
                                                        funnier_tweet_always_first=funnier_first)
        number_of_features_per_tweet = 4
        np_features = np.zeros([len(tweet_pairs), number_of_features_per_tweet * 2])
        np_labels = np.array([tweet_pair[4] for tweet_pair in tweet_pairs])
        tweet_pair_ids = []
        for i, each_tweet_pair in enumerate(tweet_pairs):
            first_tweet, first_tweet_id, second_tweet, second_tweet_id, first_tweet_is_funnier = each_tweet_pair
            first_tweet_index = tweet_ids.index(first_tweet_id)
            np_features[i, :4] = np.array(results_per_tweet[first_tweet_index])
            second_tweet_index = tweet_ids.index(second_tweet_id)
            np_features[i, 4:] = np.array(results_per_tweet[second_tweet_index])
            tweet_pair_ids.append([first_tweet_id, second_tweet_id])

        return np_features, np_labels, tweet_pair_ids


class PopCultureFeatureGeneratorTest(unittest2.TestCase):
    def setUp(self):
        movie_entries = pop_culture.load_movie_titles()
        movie_titles = [movie_entry[0] for movie_entry in movie_entries]
        song_entries = pop_culture.load_song_titles()
        song_titles = [song_entry[0] for song_entry in song_entries]
        tv_show_entries = pop_culture.load_tv_show_titles()
        tv_show_titles = [tv_show_entry[0] for tv_show_entry in tv_show_entries]
        book_entries = pop_culture.load_book_titles()
        book_titles = [book_entry[0] for book_entry in book_entries]
        self.pcfg = PopCultureFeatureGenerator(song_titles, book_titles, tv_show_titles, movie_titles)

    def test_construction(self):
        pcfg_demo = PopCultureFeatureGenerator(1, 2, 3, 4)
        assert pcfg_demo.song_list == 1
        assert pcfg_demo.book_list == 2
        assert pcfg_demo.tv_show_list == 3
        assert pcfg_demo.movie_list == 4

    def test_generate_no_tweets(self):
        np_empty_features, np_labels, tweet_pair_ids = self.pcfg.generate([], [], [])
        print np_empty_features.size
        assert np_empty_features.size == 0

    def test_generate_two_tweets(self):
        """Test that given two tweets containing a movie and song title, the pcfg can generate
        a single tweet pair where one tweet is labelled movie and the other is labelled song."""
        tweets = ["Harry Potter and the Sorcerer's Stone #Hashtag", "Love Yourself #Hashtag"]
        np_tweet_pair, np_labels, tweet_pair_ids = self.pcfg.generate(tweets, [2, 1], [1234, 5678], funnier_first=True)
        assert np_tweet_pair.shape[1] == 8
        assert np_tweet_pair[0, 3] == 1
        assert np_tweet_pair[0, 4 + 2] == 1

    def test_classify_tweet_by_genre_on_four_tweets(self):
        """Test that the PCFG can classify four tweets as belonging to four different genres."""
        tweets = ["Harry Potter and the Sorcerer's Stone #Hashtag", "Love Yourself #Hashtag", "#Hashtag Touch of Evil", "Band of bros #Hashtag"]
        book_tweet_result = self.pcfg.classify_tweet_by_genre(tweets[0])
        song_tweet_result = self.pcfg.classify_tweet_by_genre(tweets[1])
        movie_tweet_result = self.pcfg.classify_tweet_by_genre(tweets[2])
        tv_show_tweet_result = self.pcfg.classify_tweet_by_genre(tweets[3])
        assert book_tweet_result == [0, 0, 0, 1]
        assert song_tweet_result == [0, 0, 1, 0]
        assert movie_tweet_result == [1, 0, 0, 0]
        assert tv_show_tweet_result == [0, 1, 0, 0]

    def test_generate_four_tweets(self):
        """Test that PCFG can create five pairs from four tweets, each tweet from a different genre."""
        tweets = ["Harry Potter and the Sorcerer's Stone #Hashtag", "Love Yourself #Hashtag", "#Hashtag Touch of Evil",
                  "Band of bros #Hashtag"]
        np_features, np_labels, tweet_pair_ids = self.pcfg.generate(tweets, [2, 1, 1, 0], [0, 1, 2, 3], funnier_first=True)
        assert np_features.shape == (5, 8)
        # Array printed and confirmed to be proper graph of tweet pairings (below)
        assert np.isclose(np_features, np.array([[ 0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.],
                                                 [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
                                                 [ 1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
                                                 [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
                                                 [ 0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.]])).all()

    def test_min_frac_1point1(self):
        """Test that a minimum fraction of 1.1 captures no titles, and the features are equivalent to numpy zeros."""
        tweets = ["Harry Potter and the Sorcerer's Stone #Hashtag", "Love Yourself #Hashtag", "#Hashtag Touch of Evil",
                  "Band of bros #Hashtag"]
        np_features, np_labels, tweet_pair_ids = self.pcfg.generate(tweets, [2, 1, 1, 0], [0, 1, 2, 3],
                                                                    funnier_first=True, min_match_score=1.1)
        assert np_features.shape == (5, 8)
        assert np.isclose(np_features, np.zeros([5,8])).all()

    def test_labels_and_ids(self):
        """Generate tweet pairs, then confirm labels are correct."""
        tweets = ["Harry Potter and the Sorcerer's Stone #Hashtag", "Love Yourself #Hashtag", "#Hashtag Touch of Evil",
                  "Band of bros #Hashtag"]
        np_features, np_labels, tweet_pair_ids = self.pcfg.generate(tweets, [2, 1, 1, 0], [0, 1, 2, 3],
                                                                    funnier_first=True)
        assert np.array_equal(np_labels, [1, 1, 1, 1, 1])
        assert tweet_pair_ids == [[0, 3], [1, 3], [2, 3], [0, 1], [0, 2]]