'''David Donahue 2016. The data already exists to train this model. This model will be trained in Tensorflow
on sequences of characters. It will attempt to pronounce these sequences by producing, for each one, a
sequence of phonemes. The model is trained on the CMU dataset.'''

import tensorflow as tf
import numpy as np
import cPickle as pickle
import os
from char2phone_processing import max_word_size
from char2phone_processing import max_pronunciation_size
from char2phone_processing import word_output
from char2phone_processing import pronunciation_output
from char2phone_processing import char_to_index_output
from char2phone_processing import phone_to_index_output

# GPU configuration.
os.environ['GLOG_minloglevel'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

# Model parameters.
char_emb_dim = 30
lstm_dim = 400
batch_size = 100
training_fraction = .6
learning_rate = 0.0005
n_epochs = 30


def main():
    print 'Starting program'
    '''Here we load the phoneme dataset from file, train a pronunciation model and use it to pronounce new words.'''
    # Load dataset from file and split training and testing.
    np_words, np_pronunciations, char_to_index, phone_to_index = \
        import_words_and_pronunciations_from_files(word_output, pronunciation_output)
    index_to_char = invert_dictionary(char_to_index)
    index_to_phone = invert_dictionary(phone_to_index)

    np_training_words = np_words[:np_words.shape[0] * training_fraction]
    np_training_pronunciations = np_pronunciations[:np_pronunciations.shape[0] * training_fraction]
    np_testing_words = np_words[np_words.shape[0] * training_fraction:]
    np_testing_pronunciations = np_pronunciations[np_pronunciations.shape[0] * training_fraction:]
    print 'Training set size: %s' % np_training_words.shape[0]

    # Code for testing that training and testing sets are disjoint. Takes too long to run.
    # for i in range(np_training_pronunciations.shape[0]):
    #     for j in range(np_testing_pronunciations.shape[0]):
    #         np_training_pronunciation = np_training_pronunciations[i,:]
    #         np_testing_pronunciation = np_testing_pronunciations[j,:]
    #         same_phonemes = (np_training_pronunciation == np_testing_pronunciation)
    #         same_pronunciation = np.mean(same_phonemes)
    #         np_training_word = np_training_words[i,:]
    #         np_testing_word = np_testing_words[j,:]
    #         same_characters = (np_training_word == np_testing_word)
    #         same_word = np.mean(same_characters)
    #         if same_pronunciation == 1 and same_word == 1:
    #             print 'Same word found in training and testing!'
    #             print np_training_pronunciation
    #             print np_testing_pronunciation
    #             print np_training_word
    #             print np_testing_word
    #             print same_pronunciation
    #             print same_phonemes
    #             print same_characters
    #             print same_word
    
    model_inputs, model_outputs = build_model(len(char_to_index), len(phone_to_index))
    training_inputs, training_outputs = build_trainer(model_inputs, model_outputs)
    create_tensorboard_visualization()
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

def invert_dictionary(dictionary):
    inv_dictionary = {v: k for k, v in dictionary.iteritems()}
    return inv_dictionary


def create_tensorboard_visualization():
    print 'Creating Tensorboard visualization'
    writer = tf.train.SummaryWriter("/tmp/c2p_model/")
    writer.add_graph(tf.get_default_graph())


def build_model(char_vocab_size, phone_vocab_size):
    '''Here we build a model that takes in a series of characters and outputs a series of phonemes.
    The model, once trained, can pronounce words.'''
    print 'Building model'
    with tf.name_scope('CHAR_TO_PHONE_MODEL'):
        # PLACEHOLDERS. Model takes in a sequence of characters contained in tf_words.
        # The model also needs to know the batch size.
        tf_batch_size = tf.placeholder(tf.int32, name='batch_size')
        tf_words = tf.placeholder(tf.int32, [None, max_word_size], 'words')
        # Lookup up embeddings for all characters in each word.
        tf_char_emb = tf.Variable(tf.random_normal([char_vocab_size, char_emb_dim]), name='character_emb')
        tf_word_char_embs = tf.nn.embedding_lookup(tf_char_emb, tf_words)
        # Insert each character one by one into an LSTM.
        lstm = tf.nn.rnn_cell.LSTMCell(num_units = lstm_dim, state_is_tuple=True)
        lstm_hidden_state = lstm.zero_state(tf_batch_size, tf.float32)
        for i in range(max_word_size):
            tf_char_embedding = tf.nn.embedding_lookup(tf_char_emb, tf_words[:, i])
            
            with tf.variable_scope('LSTM_ENCODER') as lstm_scope:
                if i > 0:
                    lstm_scope.reuse_variables()
                lstm_output, lstm_hidden_state = lstm(tf_char_embedding, lstm_hidden_state)
        # Run encoder output through dense layer to process output
        tf_encoder_output_W = tf.Variable(tf.random_normal([lstm_dim, lstm_dim]), name='encoder_output_emb')
        tf_encoder_output_b = tf.Variable(tf.random_normal([lstm_dim]), name='encoder_output_b')
        lstm_first_hidden_state = tf.matmul(lstm_hidden_state[0], tf_encoder_output_W) + tf_encoder_output_b
        lstm_second_hidden_state = tf.matmul(lstm_hidden_state[1], tf_encoder_output_W) + tf_encoder_output_b
        lstm_hidden_state = (lstm_first_hidden_state, lstm_second_hidden_state)
        # Use hidden state of character encoding stage (this is the phoneme embedding) to predict phonemes.
        phonemes = []
        tf_phone_pred_W = tf.Variable(tf.random_normal([lstm.output_size, phone_vocab_size]), name='phoneme_prediction_emb')
        tf_phone_pred_b = tf.Variable(tf.random_normal([phone_vocab_size]), name='phoneme_prediction_bias')
        for j in range(max_pronunciation_size):
            with tf.variable_scope('LSTM_DECODER') as lstm_scope:
                if j > 0:
                    lstm_scope.reuse_variables()
                output, lstm_hidden_state = lstm(tf.zeros([tf_batch_size, char_emb_dim]), lstm_hidden_state)
                phoneme = tf.matmul(output, tf_phone_pred_W) + tf_phone_pred_b
                phonemes.append(phoneme)
        tf_phonemes = tf.pack(phonemes, axis=1)
    # Print model variables.
    model_variables = tf.trainable_variables()
    print 'Model variables:'
    for model_variable in model_variables:
        print ' - ',model_variable.name
    
    return [tf_words, tf_batch_size], [tf_phonemes]
    

def build_trainer(model_inputs, model_outputs):
    print 'Building trainer component'
    tf_batch_size = model_inputs[1]
    tf_labels = tf.placeholder(tf.int32, [None, max_pronunciation_size], 'pronunciations')
    tf_phonemes = model_outputs[0]
    tf_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(tf_phonemes, tf_labels, name='loss')
    tf_loss = tf.reduce_sum(tf_cross_entropy) / tf.cast(tf_batch_size, tf.float32)
    return [tf_labels], [tf_loss]


def train_model(model_inputs, model_outputs, training_inputs, training_outputs, np_words, np_pronunciations):
    print 'Training model'
    tf_words = model_inputs[0]
    tf_batch_size = model_inputs[1]
    tf_loss = training_outputs[0]
    tf_phonemes = model_outputs[0]
    print tf_loss.get_shape()
    tf_labels = training_inputs[0]
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.initialize_all_variables())
    current_batch_size = batch_size
    for epoch in range(n_epochs):
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
            np_pronunciation_batch = np_pronunciations[starting_training_example:starting_training_example+current_batch_size]
            # Calculate the predicted phonemes for the word batch using the model.
            _, loss, np_batch_phonemes = sess.run([train_op, tf_loss, tf_phonemes], feed_dict={tf_words:np_word_batch,tf_batch_size:current_batch_size, tf_labels:np_pronunciation_batch})
            # Model outputs a probability distribution over all phonemes. Collapse this distribution to get the predicted phoneme.
            np_batch_phoneme_predictions = np.argmax(np_batch_phonemes,axis=2)
            accuracy = calculate_accuracy(np_batch_phoneme_predictions, np_pronunciation_batch)
            if i%10 == 0:
                print 'Loss: %s, Accuracy: %s' % (loss, accuracy)
            starting_training_example += current_batch_size
    return sess


def evaluate_model_performance_on_test_set(model_inputs, model_outputs, np_words, np_pronunciations, sess=None):
    if sess == None:
        # Start a session to run model in gpu.
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
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
        np_pronunciation_batch = np_pronunciations[starting_training_example:starting_training_example+current_batch_size]
        # Calculate the predicted phonemes for the word batch using the model.
        np_batch_phonemes = sess.run(tf_phonemes, feed_dict = {tf_words:np_word_batch,tf_batch_size:current_batch_size})
        # Model outputs a probability distribution over all phonemes. Collapse this distribution to get the predicted phoneme.
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


def import_words_and_pronunciations_from_files(word_file, pronunciation_file):
    np_words = np.load(word_output)
    np_pronunciations = np.load(pronunciation_output)
    char_to_index = pickle.load(open(char_to_index_output, 'rb'))
    phone_to_index = pickle.load(open(phone_to_index_output, 'rb'))
    return np_words, np_pronunciations, char_to_index, phone_to_index


def calculate_accuracy(np_predictions, np_labels):
    """This function calculates accuracy between a set of predictions
    and a set of labels. This function does not include matches where
    the prediction is zero!"""
    #Get an array representing all matches between predictions and labels.
    np_matches = (np_predictions == np_labels)
    np_non_zeros = (np_predictions != 0)
    np_non_zero_matches = np_matches * np_non_zeros
    accuracy = np.sum(np_non_zero_matches, dtype=float) / np.sum(np_non_zeros, dtype=float)
    return accuracy


def print_phoneme_label_prediction_pairs(np_words, np_predictions, np_labels, index_to_char, index_to_phone):
    """This function prints a series of word, prediction, true pronunciation pairs to test the model's performance."""
    print 'Printing pairs...'
    for i in range(np_predictions.shape[0]):
        np_word = np_words[i,:]
        np_prediction = np_predictions[i,:]
        np_label = np_labels[i,:]
        word = ''.join([index_to_char[np_word[j]] for j in range(max_word_size)])
        prediction = ' '.join([index_to_phone[np_prediction[j]] for j in range(max_pronunciation_size)])
        label = ' '.join([index_to_phone[np_label[j]] for j in range(max_pronunciation_size)])
        print word, ' | ', prediction, ' | ', label

















if __name__ == '__main__':
    main()