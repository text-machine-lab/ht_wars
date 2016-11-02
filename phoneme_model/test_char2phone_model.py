'''David Donahue 2016. This script tests the implementation of the character-to-phoneme model.'''

from char2phone_processing import word_output
from char2phone_processing import pronunciation_output
from char2phone_model import import_words_and_pronunciations_from_files
from char2phone_model import gpu_options
from char2phone_model import build_model
from char2phone_model import char_emb_dim

import tensorflow as tf
import numpy as np

class TestModel:
    def setup_class(self):
        self.np_words, self.np_pronunciations, self.char_to_index, self.phone_to_index = import_words_and_pronunciations_from_files(word_output, pronunciation_output)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    
    def teardown_class(self):
        print 'Tearing down char model testing'
        
    def test_inputs_loaded_correctly(self):
        [tf_words, tf_pronunciations], [tf_output, tf_word_char_embs] = build_model()
        self.sess.run(tf.initialize_all_variables())
        num_samples = 100
        np_word_char_embs = self.sess.run(tf_word_char_embs, feed_dict={tf_words:self.np_words[:num_samples,:], 
                                                                                        tf_pronunciations:self.np_pronunciations[:num_samples]})
        assert np_word_char_embs.shape == (num_samples, self.np_words.shape[1], char_emb_dim)
        
        