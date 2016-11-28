'''David Donahue 2016. This script is used to create a dataset for the sound-to-meaning model.
It produces a matrix of phonetic embeddings for a sample of words in the GloVe corpus. It also
produces a matrix of GloVe embeddings for those words. The same index on both matrices indicates
a pronunciation-meaning pair, where the phonetic embedding is used to pronounce the word and the
GloVe embedding infers the word's meaning. The dataset produced by this script will be used to
train a sound-to-meaning model, capable of approximating the meaning of a word by the way it sounds.'''

import tensorflow as tf
import numpy as np
import cPickle as pickle
import imp
import sys

sys.path.append('../')
from config import WORD_VECTORS_FILE_PATH

char2phone_processing_path = '../phoneme_model/char2phone_processing.py'
char2phone_processing = imp.load_source('char2phone_processing', char2phone_processing_path)

from char2phone_processing import word_output
from char2phone_processing import pronunciation_output

char2phone_model_path = '../phoneme_model/char2phone_model.py'
char2phone_model = imp.load_source('char2phone_model', char2phone_model_path)

from char2phone_model import model_path

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

char2phone_dir = '../phoneme_model/'

num_glove_words_to_extract = 100000

def main():
    words, glove_embs = extract_words_and_embs_from_glove_corpus(sample_size=num_glove_words_to_extract)
    phonetic_embs = generate_phonetic_embs_from_words(words)
    save_dataset(words, phonetic_embs, glove_embs)


def extract_words_and_embs_from_glove_corpus(start_reading_at_line=0, sample_size=100000):
    print 'Extracting words and GloVe embeddings from corpus'
    '''Reads through each line in the GloVe datafile. Extracts sample_size
    words and their corresponding GloVe vectors. Does save words containing
    symbols.'''
    glove_words = []
    glove_embeddings = []

    # Read through all lines. Example line in GloVe file:
    # crocodiles 0.11604 0.5333 0.45436 -0.09288 -0.25571 1.0824 -0.13401...
    # So each line is a word and 200 numbers representing a GloVe embedding.
    with open(WORD_VECTORS_FILE_PATH) as f:
        number_of_glove_words_extracted = 0
        for line_index in range(start_reading_at_line):
            f.next()
        for line_of_text in f:
            line_tokens = line_of_text.split(' ')
            glove_word = line_tokens[0]
            glove_embedding = [float(i) for i in line_tokens[1:]]
            np_glove_embedding = np.array(glove_embedding, dtype=float)
            if glove_word[0].isalpha():
                if sample_size is not None:
                    if number_of_glove_words_extracted >= sample_size:
                        break
                glove_words.append(line_tokens[0])
                glove_embeddings.append(np_glove_embedding)
                number_of_glove_words_extracted += 1

    np_glove_embeddings = np.vstack(glove_embeddings)
    print np_glove_embeddings
    print 'Number of GloVe words/embeddings extracted: %s' % len(glove_words)
    print 'Glove embeddings dimensionality: %s' % str(np_glove_embeddings.shape)
    print 'Example GloVe words extracted: %s' % str(glove_words[200:210])
    return [glove_words, np_glove_embeddings]


def generate_phonetic_embs_from_words(words):
    '''Generates a phonetic embedding for each word using the pretrained char2phone model.'''
    print 'Generating phonetic embeddings for GloVe words'
    _, _, char_to_index, phone_to_index = \
        char2phone_model.import_words_and_pronunciations_from_files(dir_path=char2phone_dir)
    character_vocab_size = len(char_to_index)
    phoneme_vocab_size = len(phone_to_index)
    model_inputs, model_outputs = char2phone_model.build_model(character_vocab_size, phoneme_vocab_size)
    [tf_words, tf_batch_size] = model_inputs
    [tf_phonemes, lstm_hidden_state] = model_outputs
    tf_phonetic_emb = tf.concat(1, lstm_hidden_state)

    np_word_indices = convert_words_to_indices(words, char_to_index)
    print np_word_indices
    # Prove words converted to indices correctly by reversing the process and printing.
    index_to_char = char2phone_model.invert_dictionary(char_to_index)
    print 'Example GloVe words recreated from indices:'
    for i in range(130, 140):
        np_word = np_word_indices[i, :]
        char_list = []
        for j in np_word:
            if j in index_to_char:
                char_list.append(index_to_char[j])
        word = ''.join(char_list)
        print word,
    print

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver(max_to_keep=10)
    # Restore model from previous save.
    ckpt = tf.train.get_checkpoint_state(char2phone_dir + model_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No checkpoint found!")
        return -1

    np_phonetic_emb = sess.run(tf_phonetic_emb, feed_dict={tf_words: np_word_indices,
                                                           tf_batch_size: len(words),})

    print np_phonetic_emb.shape
    print np.mean(np.abs(np_phonetic_emb))

    return np_phonetic_emb


def convert_words_to_indices(words, char_to_index, max_word_size=20):
    m = len(words)
    # Convert all words to indices using char_to_index dictionary.
    np_word_indices = np.zeros([m, max_word_size], dtype=float)
    for word_index in range(m):
        word = words[word_index]
        for char_index in range(len(word)):
            num_non_characters = 0
            if char_index - num_non_characters < max_word_size:
                char = word[char_index]
                if char.isalpha():
                    if char in char_to_index:
                        np_word_indices[word_index, char_index - num_non_characters] = char_to_index[char]
                else:
                    num_non_characters += 1

    return np_word_indices


def save_dataset(words, phonetic_embs, glove_embs):
    print "Saving glove_words.cpkl, glove_embs.npy, and phoneme_embs.npy!"
    '''Save dataset as pickle files and numpy array files. Words will be saved
    as glove_words.cpkl, the phonetic embeddings will be saved as phonetic_embs.npy,
    and the glove embeddings will be saved as glove_embs.npy'''
    np.save('phonetic_embs.npy', phonetic_embs)
    np.save('glove_embs.npy', glove_embs)
    pickle.dump(words, open('glove_words.cpkl', 'wb'))










if __name__ == '__main__':
    print 'Starting program'
    main()