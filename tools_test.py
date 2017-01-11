"""David Donahue 2016. Script to test tools.py and tf_tools.py functionality."""
from tools import expected_value
from tools import find_indices_larger_than_threshold
from tools import format_text_with_hashtag
import numpy as np


def main():
    test_expected_value()
    test_find_indices_of_largest_n_values()
    test_format_text_with_hashtag()


def test_format_text_with_hashtag():
    tweet = 'This is an example hashtag #Hashtag'
    tweet_proc1 = format_text_with_hashtag(tweet)
    tweet_proc2 = format_text_with_hashtag(tweet, hashtag_replace='tweet')
    tweet_proc3 = format_text_with_hashtag(tweet, hashtag_replace='')
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