'''David Donahue 18 October 2016. Unit tests for ht_wars_char_model.py'''

import ht_wars_char_model
from ht_wars_char_model import HashtagWarsCharacterModel
from ht_wars_char_model import load_hashtag_data_and_vocabulary
from ht_wars_char_model import extract_hashtag_data_for_leave_one_out
from ht_wars_char_model import TWEET_SIZE
from ht_wars_char_model import tweet_pairs_dir
import numpy as np
from os import walk

class TestHTWarsCharModel():
    @classmethod
    def setup_class(self):
        print 'Setting up  char model testing'
        self.hashtag_datas, self.char_to_index, self.vocab_size = load_hashtag_data_and_vocabulary(tweet_pairs_dir)
        
    
    def teardown_class(self):
        print 'Tearing down char model testing'
        
    def test_load_hashtag_data_and_vocabulary(self):
         # Invert char_to_index.
        index_to_char = {v: k for k, v in self.char_to_index.iteritems()}
        # Vocab size is accurate.
        assert self.vocab_size == len(self.char_to_index)
        # No character index in dataset is greater than the vocab size.
        # All numpy arrays are m x 280.
        # All hashtags found in tweet_pairs_dir.
        for hashtag_data in self.hashtag_datas:
            np_tweet_pairs = hashtag_data[1]
            hashtag_name = hashtag_data[0]
            assert np.max(np_tweet_pairs) < self.vocab_size
            assert np_tweet_pairs.shape[1] == TWEET_SIZE * 2
            hashtag_found_in_filenames = False
            for (dirpath, dirnames, filenames) in walk(tweet_pairs_dir):
                for filename in filenames:
                    if hashtag_name in filename:
                        hashtag_found_in_filenames = True
            assert hashtag_found_in_filenames
        # Test all tweet pairs are distinct.
    #     for hashtag_data in hashtag_datas:
    #         np_tweet_pairs = hashtag_data[1]
    #         for i in range(np_tweet_pairs.shape[0]):
    #             for j in range(np_tweet_pairs.shape[0]):
    #                 if i != j:
    #                     assert not np.array_equal(np_tweet_pairs[i,:], np_tweet_pairs[j,:])

    def test_hashtag_data_doesnt_contain_nan_values(self):
        
        for i in range(len(self.hashtag_datas)):
            hashtag_name, np_hashtag_tweet1, np_hashtag_tweet2, np_hashtag_tweet_labels, np_other_tweet1, np_other_tweet2, np_other_tweet_labels = extract_hashtag_data_for_leave_one_out(self.hashtag_datas, i)
            
            assert not np.isnan(np.sum(np_hashtag_tweet1))
            assert not np.isnan(np.sum(np_hashtag_tweet2))
            assert not np.isnan(np.sum(np_hashtag_tweet_labels))
            assert not np.isnan(np.sum(np_other_tweet1))
            assert not np.isnan(np.sum(np_other_tweet1))
            assert not np.isnan(np.sum(np_other_tweet_labels))

                