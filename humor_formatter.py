"""David Donahue 2016. HumorFormatter class used to process input features going into the humor model."""
import config
import tools
import numpy as np
import cPickle as pickle
import unittest2


class HumorFormatter:
    """The humor predictor is meant to simplify the process of converting #HashtagWars tweets
    into ready-to-use tweet pair features that can be fed directly into a humor model. This class will
    take tweets, labels, and tweet ids to produce model input features."""
    def __init__(self, word_to_glove=None, word_to_phonetic=None):
        self.word_to_glove = word_to_glove
        self.word_to_phonetic = word_to_phonetic
        self.vocabulary = []

    def update_vocab(self, lines, max_word_size=15):
        for line in lines:
            tokens = line.lower().split()
            for word in tokens:
                if len(word) <= max_word_size and word not in self.vocabulary:
                    self.vocabulary.append(word)
        return self.vocabulary

    def format(self, tweets, labels, tweet_ids):
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
        if labels is not None and len(labels) > 0:
            assert len(tweets) == len(labels)
            assert len(labels) == len(tweet_ids)
            tweet_pairs = tools.extract_tweet_pairs_by_rank(tweets, labels, tweet_ids)
        else:
            tweet_pairs = tools.extract_tweet_pairs_by_combination(tweets, tweet_ids)

        tweet1 = [tweet_pair[0] for tweet_pair in tweet_pairs]
        tweet1_id = [tweet_pair[1] for tweet_pair in tweet_pairs]
        tweet2 = [tweet_pair[2] for tweet_pair in tweet_pairs]
        tweet2_id = [tweet_pair[3] for tweet_pair in tweet_pairs]
        np_label = None
        if labels is not None:
            labels = [tweet_pair[4] for tweet_pair in tweet_pairs]
            np_label = np.array(labels)
        np_tweet1_gloves = tools.convert_tweet_to_embeddings(tweet1, self.word_to_glove, self.word_to_phonetic,
                                                       config.HUMOR_MAX_WORDS_IN_TWEET,
                                                       config.GLOVE_EMB_SIZE, config.PHONETIC_EMB_SIZE)
        np_tweet2_gloves = tools.convert_tweet_to_embeddings(tweet2, self.word_to_glove, self.word_to_phonetic,
                                                       config.HUMOR_MAX_WORDS_IN_TWEET,
                                                       config.GLOVE_EMB_SIZE, config.PHONETIC_EMB_SIZE)
        return np_tweet1_gloves, np_tweet2_gloves, tweet1_id, tweet2_id, np_label


class HumorFormatterTest(unittest2.TestCase):
    def setUp(self):
        self.word_to_phonetic = pickle.load(open(config.HUMOR_WORD_TO_PHONETIC_FILE_PATH, 'rb'))
        self.word_to_glove = pickle.load(open(config.HUMOR_WORD_TO_GLOVE_FILE_PATH, 'rb'))

    def test_init(self):
        """Test that members are initialized properly."""
        hf = HumorFormatter()
        assert hf.word_to_glove is None
        assert hf.word_to_phonetic is None
        assert len(hf.vocabulary) == 0

    def test_update_vocab(self):
        """Test that update_vocab adds only new words,
        only words that are smaller in size than max_word_size,
        and only words that it hasn't encountered before."""
        hf = HumorFormatter()
        hf.update_vocab([])
        assert len(hf.vocabulary) == 0
        hf.update_vocab(['This is a test'])
        assert len(hf.vocabulary) == 4
        hf2 = HumorFormatter()
        hf2.update_vocab(['This is another test'], max_word_size=2)
        assert len(hf2.vocabulary) == 1
        hf.update_vocab(['this is a another test'])
        assert len(hf.vocabulary) == 5

    def test_format_empty_input(self):
        hf = HumorFormatter(word_to_glove=self.word_to_glove, word_to_phonetic=self.word_to_phonetic)
        assert hf.word_to_glove is not None
        assert hf.word_to_phonetic is not None
        np_tweet1_embs, np_tweet2_embs, tweet1_id, tweet2_id, np_label = hf.format([], [], [])
        assert np_tweet1_embs.size == 0
        assert np_tweet2_embs.size == 0
        assert len(tweet1_id) == 0
        assert len(tweet2_id) == 0
        assert np_label.size == 0

    def test_format_simple_input(self):
        hf = HumorFormatter(word_to_glove=self.word_to_glove, word_to_phonetic=self.word_to_phonetic)
        tweets = ['This is the first tweet', 'This is the second tweet yay']
        tweet_ids = [123, 456]
        np_tweet1_embs, np_tweet2_embs, tweet1_id, tweet2_id, np_label = hf.format(tweets, None, tweet_ids)
        assert np_tweet1_embs.shape == np_tweet2_embs.shape
        assert len(tweet1_id) == len(tweet2_id)
        assert np_label is None
        assert np_tweet1_embs.shape[0] == 1
        assert np.isclose(np_tweet1_embs[0, :1200], np_tweet2_embs[0, :1200]).all()
