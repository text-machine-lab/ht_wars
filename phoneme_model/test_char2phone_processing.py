'''Test script for char2phone_processing.py which uses pytest.'''

import numpy as np
import cPickle as pickle
from char2phone_processing import CMU_DICTIONARY_FILE_PATH
from char2phone_processing import CMU_SYMBOLS_FILE_PATH
from char2phone_processing import word_output
from char2phone_processing import pronunciation_output
from char2phone_processing import get_number_of_word_pronunciation_pairs
from char2phone_processing import max_word_size
from char2phone_processing import max_pronunciation_size
from char2phone_processing import char_to_index_output
from char2phone_processing import phone_to_index_output

def test_word_and_pronunciation_pairs_are_correct_size():
    np_words = np.load(word_output)
    np_pronunciations = np.load(pronunciation_output)
    num_pairs = get_number_of_word_pronunciation_pairs()
    assert np_words.shape[0] == np_pronunciations.shape[0] == num_pairs
    assert np_words.shape[1] == max_word_size
    assert np_pronunciations.shape[1] == max_pronunciation_size
    
def test_word_and_pronunciation_pairs_contain_valid_indices():
    np_words = np.load(word_output)
    np_pronunciations = np.load(pronunciation_output)
    char_to_index = pickle.load(open(char_to_index_output, 'rb'))
    phone_to_index = pickle.load(open(phone_to_index_output, 'rb'))
    
    # Check upper and lower bounds of words and pronunciations.
    assert np.min(np_words) == 0
    assert np.min(np_pronunciations) == 0
    assert np.max(np_words) <= max(char_to_index)
    assert np.max(np_pronunciations) <= max(phone_to_index)
