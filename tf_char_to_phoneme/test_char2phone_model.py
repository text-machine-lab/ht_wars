"""David Donahue 2016. This script tests the implementation of the character-to-phoneme model."""

from char2phone_processing import CMU_NP_WORDS_FILE_PATH
from char2phone_processing import CMU_NP_PRONUNCIATIONS_FILE_PATH
from char2phone_model import import_words_and_pronunciations_from_files
from tf_tools import GPU_OPTIONS
from char2phone_model import build_chars_to_phonemes_model
from tf_tools import PHONE_CHAR_EMB_DIM

import tensorflow as tf
import numpy as np

class TestModel:
    def setup_class(self):
        self.np_words, self.np_pronunciations, self.char_to_index, self.phone_to_index = import_words_and_pronunciations_from_files(CMU_NP_WORDS_FILE_PATH, CMU_NP_PRONUNCIATIONS_FILE_PATH)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=GPU_OPTIONS))
    
    def teardown_class(self):
        print 'Tearing down char model testing'
        
    def test_inputs_loaded_correctly(self):
        [tf_words, tf_pronunciations], [tf_output, tf_word_char_embs] = build_chars_to_phonemes_model()
        self.sess.run(tf.initialize_all_variables())
        num_samples = 100
        np_word_char_embs = self.sess.run(tf_word_char_embs, feed_dict={tf_words:self.np_words[:num_samples,:], 
                                                                                        tf_pronunciations:self.np_pronunciations[:num_samples]})
        assert np_word_char_embs.shape == (num_samples, self.np_words.shape[1], PHONE_CHAR_EMB_DIM)
        
        