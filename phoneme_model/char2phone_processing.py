'''David Donahue 2016. This model is designed to convert a sequence of characters into a sequence of phonemes.
The model in essence learns how to pronounce each word. This model is trained on the CMU Pronouncing Dictionary.
An LSTM will take in characters (encoding) and another LSTM will produce phonemes (decoding). The model will
train in batches on the 100k word pronunciations in the dictionary. The dataset will be ordered randomly for
this purpose.'''

import tensorflow as tf
import numpy as np
import sys
import cPickle as pickle
from numpy import int64

CMU_datafile = 'cmudict-0.7b.txt'
CMU_symbols = 'cmudict-0.7b.symbols.txt'

word_output = 'cmu_words.npy'
pronunciation_output = 'cmu_pronunciations.npy'
char_to_index_output = 'cmu_char_to_index.cpkl'
phone_to_index_output = 'cmu_phone_to_index.cpkl'

max_word_size = 20
max_pronunciation_size = 20


def main():
    print 'Starting program'
    change_program_parameters_with_command_line_arguments()
    run_command_specified_from_command_line()
    print_word_pronunciation_pairs_from_file()
    print 'Done!'


def run_command_specified_from_command_line():
    # Run specified command (sys.argv[1])
    command = sys.argv[1]
    if command == 'extract_dataset':
        np_words, np_pronunciations, char_to_index, phone_to_index = extract_CMU_dataset(max_word_size=max_word_size)
        save_numpy_array(np_words, word_output)
        save_numpy_array(np_pronunciations, pronunciation_output)
        save_pickle_file(char_to_index, char_to_index_output)
        save_pickle_file(phone_to_index, phone_to_index_output)
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
                global max_word_size 
                max_word_size = int(data)
            if arg == '-max_pronunciation_size':
                data = sys.argv[i+1]
                global max_pronunciation_size
                max_pronunciation_size = int(data)


def save_numpy_array(np_array, filename):
    print 'Saving numpy array as %s' % filename
    np.save(filename, np_array)


def save_pickle_file(object, filename):
    print 'Saving object as %s' % filename
    pickle.dump(object, open(filename, 'wb'))


def print_help_info():
    print 'No help info for you'


def extract_CMU_dataset(max_word_size=20):
    print 'Extracting dataset from %s and %s' % (CMU_datafile, CMU_symbols)
    char_to_index = build_character_vocabulary_from_cmu()
    print 'Size of character vocabulary: %s' % len(char_to_index)
    print char_to_index
    
    phone_to_index = build_phoneme_vocabulary_from_cmu()
    print 'Size of phoneme vocabulary: %s' % len(phone_to_index)
    print phone_to_index
    
    num_pairs = get_number_of_word_pronunciation_pairs()
    print 'Number of word-pronunciation pairs: %s' % num_pairs
    
    np_words, np_pronunciations = extract_CMU_words_and_pronunciations_in_index_format(char_to_index, phone_to_index, num_pairs)
    print np_words
    print np_pronunciations
    
    return np_words, np_pronunciations, char_to_index, phone_to_index


def build_character_vocabulary_from_cmu():
    '''Runs through all lines and builds a vocabulary over
    all characters present in the dataset. Maps each character
    to a unique index. All lines that start with a ; are ignored; they are comments.'''
    print 'Building character vocabulary'
    characters = []
    characters.append('')
    with open(CMU_datafile) as f:
        for line in f:
            if line[0].isalpha(): # All lines with ; are comments, ignore them
                word, pronunciation = extract_word_and_pronunciation_from_line(line)
                for char in word:
                    if char != '\n':
                        if char not in characters:
                            characters.append(char)
    char_to_index = {}
    for i in range(len(characters)):
        char_to_index[characters[i]] = i
    return char_to_index


def get_number_of_word_pronunciation_pairs():
    '''Counts the number of word pronunciation pairs.'''
    num_pairs = 0
    with open(CMU_datafile) as f:
        for line in f:
            if line[0].isalpha(): # All lines with ; are comments
                # Valid pair. Count this one.
                num_pairs += 1
    return num_pairs


def build_phoneme_vocabulary_from_cmu():
    '''Runs through all lines in the symbols list
    and associates each phoneme with an index.'''
    print 'Building phoneme vocabulary'
    phonemes = []
    phonemes.append('')
    with open(CMU_symbols) as f:
        for line in f:
            phonemes.append(line[:-1])
    phone_to_index = {}
    for i in range(len(phonemes)):
        phone_to_index[phonemes[i]] = i
    return phone_to_index


def extract_CMU_words_and_pronunciations_in_index_format(char_to_index, phone_to_index, num_pairs):
    '''Returns a numpy array of words and a numpy array of pronunciations. Each word
    is a numpy row of indices, each index mapped to a character using char_to_index. Each pronunciation is
    a numpy row of indices, each index mapped to a phoneme using phone_to_index.'''
    print 'Extracting words and their pronunciations as separate numpy arrays'
    np_words = np.zeros([num_pairs, max_word_size], dtype=int64)
    np_pronunciations = np.zeros([num_pairs, max_pronunciation_size], dtype=int64)
    with open(CMU_datafile) as f:
        counter = 0
        for line in f:
            if line[0].isalpha(): # All lines with ; are comments, ignore them.
                word, pronunciation = extract_word_and_pronunciation_from_line(line)
                for i in range(len(word)):
                    if i < max_word_size:
                        np_words[counter, i] = char_to_index[word[i]] # Convert character to index and store in array.
                for i in range(len(pronunciation)):
                    if i < max_pronunciation_size:
                        np_pronunciations[counter, i] = phone_to_index[pronunciation[i]]
                counter += 1
    # Shuffle.
    np_word_phone_pairs = np.concatenate([np_words, np_pronunciations], axis=1)
    np.random.shuffle(np_word_phone_pairs)
    np_words_shuffled = np_word_phone_pairs[:,:max_word_size]
    np_pronunciations_shuffled = np_word_phone_pairs[:,max_word_size:max_word_size+max_pronunciation_size]
    
    return np_words_shuffled, np_pronunciations_shuffled


def extract_word_and_pronunciation_from_line(line):
    '''Extracts a word and a pronunciation from each line. Each word is a string
    and each pronunciation is a list of phonemes. All instances of (*) are removed 
    to include multiple spellings for the same word. In the dataset, all symbols
    include a spelling afterwards. These spellings are ignored. Returns the word
    and the pronunciation. Example "hello", ["H","AE1","L","OH"] '''
    first_char = line[0]
    line_split = line[:-1].split(' ')
    word = line_split[0]
    pronunciation = line_split[2:]
    if not first_char.isalpha() and first_char != "'": # If not alpha, then it is a special character
        word = first_char
    paren_index = word.find('(') # Remove (*) ending of words with more than one pronunciation
    if paren_index > 0:
        word = word[:paren_index]
    return word, pronunciation


def print_word_pronunciation_pairs_from_file():
    '''This program opens the file, runs through each valid
    line and prints word-pronunciation pairs.'''
    print 'Printing saved word-pronunciation pairs'
    np_words = np.load(word_output)
    np_pronunciations = np.load(pronunciation_output)
    char_to_index = pickle.load(open(char_to_index_output, 'rb'))
    phone_to_index = pickle.load(open(phone_to_index_output, 'rb'))
    index_to_char = {v: k for k, v in char_to_index.iteritems()}
    index_to_phone = {v: k for k, v in phone_to_index.iteritems()}
    num_pairs_to_print = 100
    for i in range(-num_pairs_to_print, num_pairs_to_print):
        np_word = np_words[i,:]
        np_pronunciation = np_pronunciations[i,:]
        word = ''.join([index_to_char[np_word[j]] for j in range(max_word_size)])
        pronunciation = ' '.join([index_to_phone[np_pronunciation[j]] for j in range(max_pronunciation_size)])
        
        print word, pronunciation
#     with open(CMU_datafile) as f:
#         counter = 0
#         for line in f:
#             if line[0] != ';': # All lines with ; are comments, ignore them.
#                 word, pronunciation = extract_word_and_pronunciation_from_line(line)


def train_model():
    print 'Training model'




























if __name__ == '__main__':
    main()