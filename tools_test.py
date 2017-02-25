"""David Donahue 2016. Script to test tools.py and tf_tools.py functionality."""
import tools
from tools import expected_value
from tools import find_indices_larger_than_threshold
from tools import format_text_for_embedding_model
import numpy as np


def main():
    test_convert_tweet_to_embeddings()
    test_expected_value()
    test_find_indices_of_largest_n_values()
    test_format_text_with_hashtag()


def test_convert_tweet_to_embeddings():
    print 'TEST: convert_tweet_to_embeddings'
    #convert_to_to_embeddings()
    #tweets, word_to_glove, word_to_phonetic, max_words, glove_size, phone_size
    tweets = ['i went to the park',
              'to the zoo we go',
              'i went to the zoo',
              'hop skip jump']
    word_to_glove = {'i': [.3, .1, 4.5, 2.3],
                     'went': [.2, .2, .2, .2],
                     'to': [.5, .5, .5, .5],
                     'the': [.6, .6, .6, .6],
                     'park': [.7, .7, .7, .7],
                     'zoo': [.8, .8, .8, .8],
                     'hop': [.9, .9, .9, .9],
                     'skip': [1.0, 1.0, 1.0, 1.0]}
    word_to_phonetic = {'i': [.21, .21, .21, .21],
                        'went': [.32, .12, 4.52, 2.32],
                        'to': [.54, .54, .54, .54],
                        'the': [.68, .68, .68, .68],
                        'park': [.73, .75, .73, .75],
                        'zoo': [.84, .87, .88, .89],
                        'hop': [.92, .93, .94, .95],
                        'skip': [1.04, 1.02, 1.08, 1.06]}
    max_words = 6
    glove_size = 4
    phone_size = 4
    np_embeddings = tools.convert_tweet_to_embeddings(tweets, word_to_glove, word_to_phonetic, max_words, glove_size, phone_size)
    print np_embeddings
    for tweet_index in range(len(tweets)):
        tweet = tweets[tweet_index].split()
        for word_index in range(len(tweet)):
            word = tweet[word_index]
            if word in word_to_glove:
                assert np.array_equal(np_embeddings[tweet_index, word_index * (glove_size+phone_size):
                       word_index * (glove_size+phone_size) + glove_size], np.array(word_to_glove[word]))
            else:
                assert np.array_equal(np_embeddings[tweet_index, word_index * (glove_size + phone_size):
                word_index * (glove_size + phone_size) + glove_size], np.zeros(glove_size))
            if word in word_to_phonetic:
                assert np.array_equal(np_embeddings[tweet_index, word_index * (glove_size+phone_size) + glove_size:
                       word_index * (glove_size+phone_size) + glove_size + phone_size], np.array(word_to_phonetic[word]))
            else:
                assert np.array_equal(np_embeddings[tweet_index, word_index * (glove_size + phone_size) + glove_size:
                word_index * (glove_size + phone_size) + glove_size + phone_size], np.zeros(phone_size))


def test_format_text_with_hashtag():
    tweet = 'This is an example hashtag #Hashtag'
    tweet_proc1 = format_text_for_embedding_model(tweet)
    tweet_proc2 = format_text_for_embedding_model(tweet, hashtag_replace='tweet')
    tweet_proc3 = format_text_for_embedding_model(tweet, hashtag_replace='')
    assert tweet_proc1 == 'hashtag # this is an example hashtag'
    assert tweet_proc2 == 'tweet # this is an example hashtag'
    assert tweet_proc3 == 'this is an example hashtag'


def test_find_indices_of_largest_n_values():
    my_array = np.array([4, 2, 7, 1, 9, 0, 5, 14, 22, -4])
    indices = find_indices_larger_than_threshold(my_array, 5)
    assert indices == [2, 4, 7, 8]


def test_expected_value():
    np_1 = np.array([0, 1])
    np_2 = np.array([1, 0])
    np_3 = np.array([.5, .5])
    np_4 = np.array([123, 239])
    assert expected_value(np_1) == 1
    assert expected_value(np_2) == 0
    assert expected_value(np_3) == .5
    assert expected_value(np_4) <= 1
    assert expected_value(np_4) > .5


if __name__ == '__main__':
    main()