'''David Donahue 2016. The data already exists to train this model. This model will be trained in Tensorflow
on sequences of characters. It will attempt to pronounce these sequences by producing, for each one, a
sequence of phonemes. The model is trained on the CMU dataset.'''

import tensorflow as tf
import numpy as np
import cPickle as pickle
from char2phone_processing import max_word_size
from char2phone_processing import max_pronunciation_size
from char2phone_processing import word_output
from char2phone_processing import pronunciation_output
from char2phone_processing import char_to_index_output
from char2phone_processing import phone_to_index_output

os.environ['GLOG_minloglevel'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    print 'Starting program'
    np_words, np_pronunciations, char_to_index, phone_to_index = import_words_and_pronunciations_from_files(word_output, pronunciation_output)
    model_outputs = build_model()
    training_outputs = build_trainer()
    
def import_words_and_pronunciations_from_files(word_file, pronunciation_file):
    np_words = np.load(word_output)
    np_pronunciations = np.load(pronunciation_output)
    char_to_index = pickle.load(open(char_to_index_output, 'rb'))
    phone_to_index = pickle.load(open(phone_to_index_output, 'rb'))
    return np_words, np_pronunciations, char_to_index, phone_to_index

def build_model():
    print 'Building model'
    

def build_trainer():
    print 'Building trainer component'






















if __name__ == '__main__':
    main()