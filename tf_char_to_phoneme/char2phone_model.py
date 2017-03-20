"""David Donahue 2016. The data already exists to train this model. This model will be trained in Tensorflow
on sequences of characters. It will attempt to pronounce these sequences by producing, for each one, a
sequence of phonemes. The model is trained on the CMU dataset."""

import tensorflow as tf
import numpy as np
import cPickle as pickle
import os
import math
from config import CMU_NP_WORDS_FILE_PATH
from config import CMU_NP_PRONUNCIATIONS_FILE_PATH
from char2phone_processing import CMU_CHAR_TO_INDEX_FILE_PATH
from char2phone_processing import CMU_PHONE_TO_INDEX_FILE_PATH

from config import CHAR_2_PHONE_MODEL_DIR
from tf_tools import MAX_PRONUNCIATION_SIZE
from tf_tools import MAX_WORD_SIZE
from tf_tools import GPU_OPTIONS
from tf_tools import create_tensorboard_visualization


from tf_tools import build_chars_to_phonemes_model
from tools import invert_dictionary

# GPU configuration.
os.environ['GLOG_minloglevel'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Model parameters.
batch_size = 100
training_fraction = .6
learning_rate = 0.0005
n_epochs = 15


def main():
    print 'Starting program'
    '''Here we load the phoneme dataset from file, train a pronunciation model and use it to pronounce new words.'''
    # Load dataset from file and split training and testing.
    np_words, np_pronunciations, char_to_index, phone_to_index = \
        import_words_and_pronunciations_from_files()
    index_to_char = invert_dictionary(char_to_index)
    index_to_phone = invert_dictionary(phone_to_index)

    np_training_words = np_words[:int(np_words.shape[0] * training_fraction)]
    np_training_pronunciations = np_pronunciations[:int(np_pronunciations.shape[0] * training_fraction)]
    np_testing_words = np_words[int(np_words.shape[0] * training_fraction):]
    np_testing_pronunciations = np_pronunciations[int(np_pronunciations.shape[0] * training_fraction):]
    print 'Training set size: %s' % np_training_words.shape[0]
    
    model_inputs, model_outputs = build_chars_to_phonemes_model(len(char_to_index), len(phone_to_index))
    training_inputs, training_outputs = build_trainer(model_inputs, model_outputs)
    create_tensorboard_visualization('c2p_model')
    sess = train_model(model_inputs,
                       model_outputs,
                       training_inputs,
                       training_outputs,
                       np_training_words,
                       np_training_pronunciations)
    np_testing_predictions, accuracy = evaluate_model_performance_on_test_set(model_inputs,
                                                                              model_outputs,
                                                                              np_testing_words,
                                                                              np_testing_pronunciations,
                                                                              sess=sess)
    print 'Model test accuracy: %s' % accuracy
    num_examples_to_print = 100
    print_phoneme_label_prediction_pairs(np_testing_words[:num_examples_to_print],
                                         np_testing_predictions[:num_examples_to_print],
                                         np_testing_pronunciations[:num_examples_to_print],
                                         index_to_char,
                                         index_to_phone)


def build_trainer(model_inputs, model_outputs):
    print 'Building trainer component'
    tf_batch_size = model_inputs[1]
    tf_labels = tf.placeholder(tf.int32, [None, MAX_PRONUNCIATION_SIZE], 'pronunciations')
    tf_phonemes = model_outputs[0]
    tf_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_phonemes, labels=tf_labels, name='loss')
    tf_loss = tf.reduce_sum(tf_cross_entropy) / tf.cast(tf_batch_size, tf.float32)
    return [tf_labels], [tf_loss]


def train_model(model_inputs, model_outputs, training_inputs, training_outputs, np_words, np_pronunciations):
    print 'Training model'
    if not os.path.exists(CHAR_2_PHONE_MODEL_DIR):
        os.makedirs(CHAR_2_PHONE_MODEL_DIR)
    # Extract tf variables.
    tf_words = model_inputs[0]
    tf_batch_size = model_inputs[1]
    tf_loss = training_outputs[0]
    tf_phonemes = model_outputs[0]
    tf_labels = training_inputs[0]

    with tf.name_scope("SAVER"):
        saver = tf.train.Saver(max_to_keep=10)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=GPU_OPTIONS))
    sess.run(tf.global_variables_initializer())
    current_batch_size = batch_size
    for epoch in range(n_epochs):
        print 'Epoch %s' % epoch
        # Use these parameters to keep track of batch size and start location.
        starting_training_example = 0
        num_batches = np_words.shape[0] / batch_size
        remaining_batch_size = np_words.shape[0] % batch_size
        batch_accuracies = []
        for i in range(num_batches+1):
            # If we are on the last batch, we are running leftover examples. Otherwise, we stick to global batch_size.
            if i == num_batches:
                current_batch_size = remaining_batch_size
            else:
                current_batch_size = batch_size
            # Extract a batch of words of size current_batch_size.
            np_word_batch = np_words[starting_training_example:starting_training_example+current_batch_size]
            # Extract a batch of pronunciations of size current_batch_size.
            np_pronunciation_batch = np_pronunciations[starting_training_example:starting_training_example+current_batch_size]
            # Calculate the predicted phonemes for the word batch using the model.
            _, loss, np_batch_phonemes = sess.run([train_op, tf_loss, tf_phonemes],
                                                  feed_dict={tf_words: np_word_batch,
                                                             tf_batch_size: current_batch_size,
                                                             tf_labels: np_pronunciation_batch})
            # Model outputs a probability distribution over all phonemes.
            # Collapse this distribution to get the predicted phoneme.
            np_batch_phoneme_predictions = np.argmax(np_batch_phonemes,axis=2)
            accuracy = calculate_accuracy(np_batch_phoneme_predictions, np_pronunciation_batch)
            if not math.isnan(accuracy):
                batch_accuracies.append(accuracy)
            else:
                print 'Skipping accuracy'
            starting_training_example += current_batch_size
        average_epoch_training_accuracy = sum(batch_accuracies) / len(batch_accuracies)
        print 'Epoch accuracy: %s' % average_epoch_training_accuracy
        print 'Saving model %s' % epoch
        print
        saver.save(sess, os.path.join(CHAR_2_PHONE_MODEL_DIR, 'c2p-model'), global_step=epoch)
    return sess


def evaluate_model_performance_on_test_set(model_inputs, model_outputs, np_words, np_pronunciations, sess=None):
    if sess is None:
        # Start a session to run model in gpu.
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=GPU_OPTIONS))
        sess.run(tf.initialize_all_variables())
    '''Evaluate model on test examples.'''
    # Unroll model inputs.
    tf_words = model_inputs[0]
    tf_batch_size = model_inputs[1]
    tf_phonemes = model_outputs[0]
    # Initialize model parameters.
    np_all_phoneme_batches = []
    current_batch_size = batch_size
    # Use these parameters to keep track of batch size and start location.
    starting_training_example = 0
    num_batches = np_words.shape[0] / batch_size
    remaining_batch_size = np_words.shape[0] % batch_size
    for i in range(num_batches+1):
        # If we are on the last batch, we are running leftover examples. Otherwise, we stick to global batch_size.
        if i == num_batches:
            current_batch_size = remaining_batch_size
        else:
            current_batch_size = batch_size
        # Extract a batch of words of size current_batch_size.
        np_word_batch = np_words[starting_training_example:starting_training_example+current_batch_size]
        # Extract a batch of pronunciations of size current_batch_size.
        np_pronunciation_batch = np_pronunciations[starting_training_example:starting_training_example + current_batch_size]
        # Calculate the predicted phonemes for the word batch using the model.
        np_batch_phonemes = sess.run(tf_phonemes, feed_dict={tf_words: np_word_batch, tf_batch_size: current_batch_size})
        # Model outputs a probability distribution over all phonemes.
        # Collapse this distribution to get the predicted phoneme.
        np_batch_phoneme_predictions = np.argmax(np_batch_phonemes,axis=2)
        # Append batch pronunciation predictions to list of all pronunciations predicted in this session.
        np_all_phoneme_batches.append(np_batch_phoneme_predictions)
        # Update starting training example for next batch.
        starting_training_example += current_batch_size
    # We have made predictions for whole dataset. Concatenate them into a numpy array.
    np_phoneme_predictions = np.concatenate(np_all_phoneme_batches)
    # Confirm predictions and labels have same dimension.
    print np_phoneme_predictions.shape
    print np_pronunciations.shape
    # Calculate acurracy as the average fraction of phonemes model got correct.
    accuracy = calculate_accuracy(np_phoneme_predictions, np_pronunciations)
    return np_phoneme_predictions, accuracy


def import_words_and_pronunciations_from_files(dir_path=''):
    np_words = np.load(dir_path + CMU_NP_WORDS_FILE_PATH)
    np_pronunciations = np.load(dir_path + CMU_NP_PRONUNCIATIONS_FILE_PATH)
    char_to_index = pickle.load(open(dir_path + CMU_CHAR_TO_INDEX_FILE_PATH, 'rb'))
    phone_to_index = pickle.load(open(dir_path + CMU_PHONE_TO_INDEX_FILE_PATH, 'rb'))
    return np_words, np_pronunciations, char_to_index, phone_to_index


def calculate_accuracy(np_predictions, np_labels):
    """This function calculates accuracy between a set of predictions
    and a set of labels. This function does not include matches where
    the prediction is zero!"""
    # Get an array representing all matches between predictions and labels.
    np_matches = (np_predictions == np_labels)
    np_non_zeros = (np_predictions != 0)
    np_non_zero_matches = np_matches * np_non_zeros
    accuracy = np.sum(np_non_zero_matches.astype(float)) / np.sum(np_non_zeros.astype(float))
    return accuracy


def print_phoneme_label_prediction_pairs(np_words, np_predictions, np_labels, index_to_char, index_to_phone):
    """This function prints a series of word, prediction, true pronunciation pairs to test the model's performance."""
    print 'Printing pairs...'
    for i in range(np_predictions.shape[0]):
        np_word = np_words[i,:]
        np_prediction = np_predictions[i,:]
        np_label = np_labels[i,:]
        word = ''.join([index_to_char[np_word[j]] for j in range(MAX_WORD_SIZE)])
        prediction = ' '.join([index_to_phone[np_prediction[j]] for j in range(MAX_PRONUNCIATION_SIZE)])
        label = ' '.join([index_to_phone[np_label[j]] for j in range(MAX_PRONUNCIATION_SIZE)])
        print word, ' | ', prediction, ' | ', label

















if __name__ == '__main__':
    main()