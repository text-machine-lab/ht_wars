"""David Donahue 2016. This script deals primarily with tensorflow build operations. This script
separates functions that do import tensorflow from those that don't."""
import cPickle as pickle

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import config
import tools

GPU_OPTIONS = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)


def build_lstm(lstm_hidden_dim, tf_batch_size, inputs, input_time_step_size, num_time_steps, lstm_scope=None,
               reuse=False, time_step_inputs=None):
    """Runs an LSTM over input data and returns LSTM output and hidden state. Arguments:
    lstm_hidden_dim - Size of hidden state of LSTM
    tf_batch_size - Tensor value representing size of current batch. Required for LSTM package
    inputs - Full input into LSTM. List of tensors as input. Per tensor: First dimension of m examples, with second dimension holding concatenated input for all timesteps
    input_time_step_size - Size of input from tf_input that will go into LSTM in a single timestep
    num_time_steps - Number of time steps to run LSTM
    lstm_scope - Can be a string or a scope object. Used to disambiguate variable scopes of different LSTM objects
    time_step_inputs - Inputs that are per time step. The same tensor is inserted into the model at each time step"""
    if time_step_inputs is None:
        time_step_inputs = []
    lstm = tf.contrib.rnn.LSTMCell(num_units=lstm_hidden_dim, state_is_tuple=True)
    tf_hidden_state = lstm.zero_state(tf_batch_size, tf.float32)
    for i in range(num_time_steps):
        # Grab time step input for each input tensor
        current_time_step_inputs = []
        for tf_input in inputs:
            current_time_step_inputs.append(
                tf.slice(tf_input, [0, i * input_time_step_size], [-1, input_time_step_size]))

        tf_input_time_step = tf.concat(current_time_step_inputs + time_step_inputs, 1)

        with tf.variable_scope(lstm_scope) as scope:
            if i > 0 or reuse:
                scope.reuse_variables()
            tf_lstm_output, tf_hidden_state = lstm(tf_input_time_step, tf_hidden_state)
    return tf_lstm_output, tf_hidden_state


def build_chars_to_phonemes_model(char_vocab_size, phone_vocab_size):
    """Here we build a model that takes in a series of characters and outputs a series of phonemes.
    The model, once trained, can pronounce words."""
    print 'Building model'
    with tf.name_scope('CHAR_TO_PHONE_MODEL'):
        # PLACEHOLDERS. Model takes in a sequence of characters contained in tf_words.
        # The model also needs to know the batch size.
        tf_batch_size = tf.placeholder(tf.int32, name='batch_size')
        tf_words = tf.placeholder(tf.int32, [None, config.MAX_WORD_SIZE], 'words')
        # Lookup up embeddings for all characters in each word.
        tf_char_emb = tf.Variable(tf.random_normal([char_vocab_size, config.PHONE_CHAR_EMB_DIM]), name='character_emb')
        # Insert each character one by one into an LSTM.
        lstm = tf.contrib.rnn.LSTMCell(num_units=config.PHONE_ENCODER_LSTM_EMB_DIM, state_is_tuple=True)
        encoder_hidden_state = lstm.zero_state(tf_batch_size, tf.float32)
        for i in range(config.MAX_WORD_SIZE):
            tf_char_embedding = tf.nn.embedding_lookup(tf_char_emb, tf_words[:, i])

            with tf.variable_scope('LSTM_ENCODER') as lstm_scope:
                if i > 0:
                    lstm_scope.reuse_variables()
                encoder_output, encoder_hidden_state = lstm(tf_char_embedding, encoder_hidden_state)
        # Run encoder output through dense layer to process output
        tf_encoder_output_w = tf.Variable(tf.random_normal([config.PHONE_ENCODER_LSTM_EMB_DIM, config.PHONE_ENCODER_LSTM_EMB_DIM]), name='encoder_output_emb')
        tf_encoder_output_b = tf.Variable(tf.random_normal([config.PHONE_ENCODER_LSTM_EMB_DIM]), name='encoder_output_bias')
        encoder_output_emb = tf.matmul(encoder_output, tf_encoder_output_w) + tf_encoder_output_b

        decoder_hidden_state = lstm.zero_state(tf_batch_size, tf.float32)

        # Use hidden state of character encoding stage (this is the phoneme embedding) to predict phonemes.
        phonemes = []
        tf_phone_pred_w = tf.Variable(tf.random_normal([lstm.output_size, phone_vocab_size]),
                                      name='phoneme_prediction_emb')
        tf_phone_pred_b = tf.Variable(tf.random_normal([phone_vocab_size]), name='phoneme_prediction_bias')
        for j in range(config.MAX_PRONUNCIATION_SIZE):
            with tf.variable_scope('LSTM_DECODER') as lstm_scope:
                if j == 0:
                    decoder_output, decoder_hidden_state = lstm(encoder_output_emb, decoder_hidden_state)
                else:
                    lstm_scope.reuse_variables()
                    # decoder_output, decoder_hidden_state = lstm(tf.zeros([tf_batch_size, LSTM_EMB_DIM]), decoder_hidden_state)
                    decoder_output, decoder_hidden_state = lstm(encoder_output_emb, decoder_hidden_state)
                phoneme = tf.matmul(decoder_output, tf_phone_pred_w) + tf_phone_pred_b
                phonemes.append(phoneme)
        tf_phonemes = tf.stack(phonemes, axis=1)
    # Print model variables.
    model_variables = tf.trainable_variables()
    print 'Model variables:'
    # for model_variable in model_variables:
    #     print ' - ', model_variable.name

    return [tf_words, tf_batch_size], [tf_phonemes, encoder_output_emb]


def create_dense_layer(input_layer, input_size, output_size, activation=None, include_bias=True, reg_const=.0005, name=None):
    with tf.name_scope(name):
        tf_w = tf.Variable(tf.random_normal([input_size, output_size], stddev=.1))
        tf_b = tf.Variable(tf.random_normal([output_size]))
        output_layer = tf.matmul(input_layer, tf_w)
        if include_bias:
            output_layer = output_layer + tf_b
        if activation == 'relu':
            output_layer = tf.nn.relu(output_layer)
        elif activation == 'sigmoid':
            output_layer = tf.nn.sigmoid(output_layer)
        elif activation is None:
            pass
        else:
            print 'Error: Did not specify layer activation'

    regularizer = slim.l2_regularizer(reg_const)
    regularizer_loss = regularizer(tf_w) + regularizer(tf_b)
    slim.losses.add_loss(regularizer_loss)

    return output_layer, tf_w, tf_b


def generate_phonetic_embs_from_words(words, char_to_index_path, phone_to_index_path):
    """Generates a phonetic embedding for each word using the pretrained char2phone model."""
    print 'Generating phonetic embeddings for GloVe words'
    char_to_index = pickle.load(open(char_to_index_path, 'rb'))
    phone_to_index = pickle.load(open(phone_to_index_path, 'rb'))
    character_vocab_size = len(char_to_index)
    phoneme_vocab_size = len(phone_to_index)
    model_inputs, model_outputs = build_chars_to_phonemes_model(character_vocab_size, phoneme_vocab_size)
    [tf_words, tf_batch_size] = model_inputs
    [tf_phonemes, lstm_hidden_state] = model_outputs
    tf_phonetic_emb = lstm_hidden_state

    np_word_indices = tools.convert_words_to_indices(words, char_to_index)
    print np_word_indices
    # Prove words converted to indices correctly by reversing the process and printing.
    index_to_char = tools.invert_dictionary(char_to_index)
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

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=GPU_OPTIONS))
    saver = tf.train.Saver(max_to_keep=10)
    # Restore model from previous save.
    ckpt = tf.train.get_checkpoint_state(config.CHAR_2_PHONE_MODEL_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No checkpoint found!")
        return -1

    np_phonetic_emb = sess.run(tf_phonetic_emb, feed_dict={tf_words: np_word_indices,
                                                           tf_batch_size: len(words)})

    print np_phonetic_emb.shape
    print np.mean(np.abs(np_phonetic_emb))

    return np_phonetic_emb


def create_tensorboard_visualization(model_name):
    """Saves the Tensorflow graph of your model, so you can view it in a TensorBoard console."""
    print 'Creating Tensorboard visualization'
    writer = tf.summary.FileWriter("/tmp/" + model_name + "/")
    writer.add_graph(tf.get_default_graph())


def predict_on_hashtag(sess, model_vars, hashtag_name, hashtag_dir, hashtag_datas, error_analysis_stats=None):
    """Predicts on a hashtag. Returns the accuracy of predictions on all tweet pairs and returns
    a list. The list contains the predictions on all tweet pairs, and tweet ids for the first and second tweets in
    each pair. If error analysis stats are provided, the function will print the tweet pairs the model performed the worst
    on. error_analysis_stats should be a string and a number. The string is the location where Semeval hashtag .tsv files are kept(training or testing).
    The number is the number of worst tweet pairs to print."""
    print 'Predicting on hashtag %s' % hashtag_name
    np_first_tweets_char, np_second_tweets_char = tools.extract_tweet_pair_from_hashtag_datas(hashtag_datas, hashtag_name)

    [tf_first_input_tweets, tf_second_input_tweets, tf_predictions, tf_tweet_humor_rating, tf_batch_size, tf_hashtag, tf_output_prob, tf_dropout_rate,
     tf_tweet1, tf_tweet2] = model_vars

    np_first_tweets, np_second_tweets, np_labels, first_tweet_ids, second_tweet_ids, np_hashtag = tools.load_hashtag_data(hashtag_dir, hashtag_name)
    np_predictions, np_output_prob = sess.run([tf_predictions, tf_output_prob],
                                              feed_dict={tf_first_input_tweets: np_first_tweets,
                                                         tf_second_input_tweets: np_second_tweets,
                                                         tf_batch_size: np_first_tweets.shape[0],
                                                         tf_hashtag: np_hashtag,
                                                         tf_dropout_rate: 1.0,
                                                         tf_tweet1: np_first_tweets_char,
                                                         tf_tweet2: np_second_tweets_char})

    accuracy = None
    if np_labels is not None:
        accuracy = np.mean(np_predictions == np_labels)
    return accuracy, [np_predictions, first_tweet_ids, second_tweet_ids]
