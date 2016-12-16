"""David Donahue 2016. Tests functions and data output from humor_processing.py script."""

import numpy as np
import cPickle as pickle
from humor_processing import build_vocabulary
from humor_processing import format_text_with_hashtag
from humor_processing import load_tweets_from_hashtag
from humor_processing import look_up_glove_embeddings
from humor_processing import convert_tweet_to_gloves
from tools import HUMOR_MAX_WORDS_IN_TWEET
from config import HUMOR_TWEET_PAIR_EMBEDDING_DIR
from config import DATA_DIR


def main():
    test_build_vocabulary()
    test_format_text_with_hashtag()
    test_load_tweets_from_hashtag()
    test_look_up_glove_embeddings_and_convert_tweets_to_gloves()
    test_saved_files()

    print 'Tests successful'


def test_saved_files():
    """Loads index_to_word and word_to_glove, and
    first tweet, second tweet, and winner label for
    one example hashtag. Checks first and second tweet
    have same dimensions. Checks that the winner label has
    same first dimension. Checks that index to word is
    larger size than word_to_glove, and all words in
    word_to_glove are occur once and exist in index_to_word."""
    example_hashtag = '420_Celebs'
    index_to_word = pickle.load(open(DATA_DIR + 'humor_index_to_word.cpkl', 'rb'))
    word_to_glove = pickle.load(open(DATA_DIR + 'humor_word_to_glove.cpkl', 'rb'))
    np_first_tweets = np.load(open(HUMOR_TWEET_PAIR_EMBEDDING_DIR + example_hashtag + '_first_tweet_glove.npy'))
    np_second_tweets = np.load(open(HUMOR_TWEET_PAIR_EMBEDDING_DIR + example_hashtag + '_second_tweet_glove.npy'))
    np_winner_labels = np.load(open(HUMOR_TWEET_PAIR_EMBEDDING_DIR + example_hashtag + '_label.npy'))
    assert np_first_tweets.shape == np_second_tweets.shape
    assert np_winner_labels.shape[0] == np_first_tweets.shape[0]
    assert len(index_to_word) >= len(word_to_glove)
    assert len(word_to_glove) > 0
    for key in word_to_glove:
        assert key in index_to_word



def test_look_up_glove_embeddings_and_convert_tweets_to_gloves():
    index_to_word = ['banana', 'apple', 'car']
    word_to_glove = look_up_glove_embeddings(index_to_word)
    np_banana_glove = np.array(word_to_glove['banana'])
    np_apple_glove = np.array(word_to_glove['apple'])
    np_car_glove = np.array(word_to_glove['car'])
    assert np.dot(np_apple_glove, np_banana_glove) > np.dot(np_apple_glove, np_car_glove)
    assert np.dot(np_apple_glove, np_banana_glove) > np.dot(np_banana_glove, np_car_glove)

    np_converted_glove = convert_tweet_to_gloves(['apple banana'], word_to_glove, HUMOR_MAX_WORDS_IN_TWEET, 200)
    assert np.array_equal(np_converted_glove[0, :200], np_apple_glove)
    assert np.array_equal(np_converted_glove[0, 200:400], np_banana_glove)
    assert np.array_equal(np_converted_glove[0, 400:600], np.zeros([200]))


def test_load_tweets_from_hashtag():
    tweets, labels = load_tweets_from_hashtag('./test_hashtag_file.txt')
    assert tweets == ['this is a text file containing a single hashtag # waste hard disk 2016 #']
    assert labels == [1]


def test_build_vocabulary():
    text1 = 'Hello hello world man'
    text2 = 'Bird man hello bird'
    vocabulary = build_vocabulary([text1, text2])
    words = ['hello', 'world', 'man', 'bird']
    for word in words:
        num_of_word = 0
        for entry in vocabulary:
            if word == entry:
                num_of_word += 1
        assert num_of_word == 1


def test_format_text_with_hashtag():
    text = 'The methlamine must not stop flowing. #TheBreakingBad show'
    formatted_text = format_text_with_hashtag(text)
    assert formatted_text == 'The methlamine must not stop flowing. # The Breaking Bad # show'


if __name__ == '__main__':
    main()