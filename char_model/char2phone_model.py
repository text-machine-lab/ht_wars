'''David Donahue 2016. This model is designed to convert a sequence of characters into a sequence of phonemes.
The model in essence learns how to pronounce each word. This model is trained on the CMU Pronouncing Dictionary.
An LSTM will take in characters (encoding) and another LSTM will produce phonemes (decoding). The model will
train in batches on the 100k word pronunciations in the dictionary. The dataset will be ordered randomly for
this purpose.'''

import tensorflow as tf
import numpy as np
import sys

CMU_datafile = 'cmudict-0.7b.txt'
CMU_symbols = 'cmudict-0.7b.symbols.txt'

max_word_size = 20

def main():
    print 'Starting program'
    change_program_parameters_with_command_line_arguments()
    run_command_specified_from_command_line()
    print 'Done!'

def run_command_specified_from_command_line():
    # Run specified command (sys.argv[1])
    command = sys.argv[1]
    if command == 'extract_dataset':
        np_words, np_pronunciations = extract_CMU_dataset(max_word_size=max_word_size)
        save_numpy_array(np_words, 'cmu_words.npy')
        save_numpy_array(np_pronunciations, 'cmu_pronunciations.npy')
    elif command == 'train':
        train_model()
    elif command == 'help':
        print_help_info()
    
def change_program_parameters_with_command_line_arguments():
    # Make sure user entered a command
    if len(sys.argv) < 2:
        print 'Must enter command'
        print 'Usage: python char2phone_model.py [command] {args}'
        print 'Type "python char2phone_model.py help" for more info'
        exit()
    # Analyze arguments to modify program behavior (sys.argv[>1])
    for i in range(2, len(sys.argv)):
        arg = sys.argv[i]
        if arg.startswith('-'):
            if arg == '-max_word_size':
                data = sys.argv[i+1]
                print 'Got here'
                global max_word_size 
                max_word_size = int(data)

def save_numpy_array(np_array, filename):
    print 'Saving numpy array as %s' % filename

def print_help_info():
    print 'No help info for you'
    
def extract_CMU_dataset(max_word_size=20):
    print 'Extracting dataset from %s and %s' % (CMU_datafile, CMU_symbols)
    char_to_index = build_character_vocabulary_from_CMU()
    phoneme_to_index = build_phoneme_vocabulary_from_CMU()
    
    return [1, 2, 3], [4, 5, 6]
    
#     with open(CMU_datafile) as f:
#         for line in f:
#             print line,
def build_character_vocabulary_from_CMU():
    print 'Building character vocabulary'
    
def build_phoneme_vocabulary_from_CMU():
    print 'Building phoneme vocabulary'

def train_model():
    print 'Training model'

if __name__ == '__main__':
    main()