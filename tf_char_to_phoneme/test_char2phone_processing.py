"""Test script for char2phone_processing.py which uses pytest."""

import numpy as np
import cPickle as pickle
from char2phone_processing import CMU_DICTIONARY_FILE_PATH
from char2phone_processing import CMU_SYMBOLS_FILE_PATH
from char2phone_processing import CMU_NP_WORDS_FILE_PATH
from char2phone_processing import CMU_NP_PRONUNCIATIONS_FILE_PATH
from char2phone_processing import get_number_of_word_pronunciation_pairs
from char2phone_processing import MAX_WORD_SIZE
from char2phone_processing import MAX_PRONUNCIATION_SIZE
from char2phone_processing import CMU_CHAR_TO_INDEX_FILE_PATH
from char2phone_processing import CMU_PHONE_TO_INDEX_FILE_PATH


def main():
    test_word_and_pronunciation_pairs_are_correct_size()
    test_word_and_pronunciation_pairs_contain_valid_indices()
    print 'Tests finished successfully'


def test_word_and_pronunciation_pairs_are_correct_size():
    np_words = np.load(CMU_NP_WORDS_FILE_PATH)
    np_pronunciations = np.load(CMU_NP_PRONUNCIATIONS_FILE_PATH)
    num_pairs = get_number_of_word_pronunciation_pairs()
    assert np_words.shape[0] == np_pronunciations.shape[0] == num_pairs
    assert np_words.shape[1] == MAX_WORD_SIZE
    assert np_pronunciations.shape[1] == MAX_PRONUNCIATION_SIZE


def test_word_and_pronunciation_pairs_contain_valid_indices():
    np_words = np.load(CMU_NP_WORDS_FILE_PATH)
    np_pronunciations = np.load(CMU_NP_PRONUNCIATIONS_FILE_PATH)
    char_to_index = pickle.load(open(CMU_CHAR_TO_INDEX_FILE_PATH, 'rb'))
    phone_to_index = pickle.load(open(CMU_PHONE_TO_INDEX_FILE_PATH, 'rb'))
    
    # Check upper and lower bounds of words and pronunciations.
    assert np.min(np_words) == 0
    assert np.min(np_pronunciations) == 0
    assert np.max(np_words) <= max(char_to_index)
    assert np.max(np_pronunciations) <= max(phone_to_index)


if __name__ == '__main__':
    main()
