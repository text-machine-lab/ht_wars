"""David Donahue 2016. This model will be trained and used to predict winning tweets
in the Semeval 2017 Humor Task. The goal of the model is to read tweet pairs and determine
which of the two tweets in a pair is funnier. For each word in a tweet, it will receive a
phonetic embedding and a GloVe embedding to describe how to pronounce the word and what the
word means, respectively. These will serve as features. Other features may be added over time.
This model will be built in Tensorflow."""
import tensorflow as tf
import numpy as np
import cPickle as pickle
import random
import os
import sklearn
from config import HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR
from config import HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR
from config import EMBEDDING_HUMOR_MODEL_DIR
from config import SEMEVAL_HUMOR_TRIAL_DIR
from tools import HUMOR_MAX_WORDS_IN_TWEET
from config import SEMEVAL_HUMOR_TRAIN_DIR
from config import SEMEVAL_HUMOR_TRIAL_DIR
from tools import GLOVE_SIZE
from tools import PHONETIC_EMB_SIZE
from tools import get_hashtag_file_names
from tools import load_hashtag_data
from tf_tools import GPU_OPTIONS
from tf_tools import create_dense_layer
from tf_tools import predict_on_hashtag


EMBEDDING_HUMOR_MODEL_LEARNING_RATE = .00001
N_TRAIN_EPOCHS = 1
DROPOUT = 1  # Off


def main():
    """Game plan: Use train_dir hashtags to train the model. Then use trial_dir hashtags
    to evaluate its performance. This creates a faster development environment than
    leave-one-out."""
    model_vars = build_embedding_humor_model()
    trainer_vars = build_embedding_humor_model_trainer(model_vars)
    training_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRAIN_DIR)
    testing_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRIAL_DIR)
    accuracies = []

    sess, training_accuracy = train_on_all_other_hashtags(model_vars, trainer_vars, training_hashtag_names, '', n_epochs=N_TRAIN_EPOCHS)  # Blank hashtag '' means train on all training hashtags
    print 'Mean training accuracy: %s' % training_accuracy
    print
    for hashtag_name in testing_hashtag_names:
        accuracy, _ = predict_on_hashtag(sess,
                                         model_vars,
                                         hashtag_name,
                                         HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR,
                                         error_analysis_stats=[SEMEVAL_HUMOR_TRIAL_DIR, 10])
        print 'Hashtag %s accuracy: %s' % (hashtag_name, accuracy)
        accuracies.append(accuracy)

    print 'Final test accuracy: %s' % np.mean(accuracies)


def build_embedding_humor_model():
    print 'Building embedding humor model'
    tf_batch_size = tf.placeholder(tf.int32, name='batch_size')
    word_embedding_size = GLOVE_SIZE + PHONETIC_EMB_SIZE
    # Output of two, allowing the model to choose which is funnier, the left or the right tweet.
    tf_first_input_tweets = tf.placeholder(dtype=tf.float32, shape=[None, HUMOR_MAX_WORDS_IN_TWEET * word_embedding_size], name='first_tweets')
    tf_second_input_tweets = tf.placeholder(dtype=tf.float32, shape=[None, HUMOR_MAX_WORDS_IN_TWEET * word_embedding_size], name='second_tweets')

    lstm = tf.nn.rnn_cell.LSTMCell(num_units=word_embedding_size * 2, state_is_tuple=True)  # LSTM size is 4 times emb size
    tf_first_tweet_hidden_state = lstm.zero_state(tf_batch_size, tf.float32)
    for i in range(HUMOR_MAX_WORDS_IN_TWEET):
        tf_first_tweet_current_word = tf.slice(tf_first_input_tweets, [0, i * GLOVE_SIZE], [-1, GLOVE_SIZE])
        with tf.variable_scope('TWEET_ENCODER') as lstm_scope:
            if i > 0:
                lstm_scope.reuse_variables()
            tf_first_tweet_encoder_output, tf_first_tweet_hidden_state = lstm(tf_first_tweet_current_word, tf_first_tweet_hidden_state)

    tf_second_tweet_hidden_state = lstm.zero_state(tf_batch_size, tf.float32)
    for i in range(HUMOR_MAX_WORDS_IN_TWEET):
        tf_second_tweet_current_word = tf.slice(tf_second_input_tweets, [0, i * GLOVE_SIZE], [-1, GLOVE_SIZE])
        with tf.variable_scope('TWEET_ENCODER', reuse=True):
            tf_second_tweet_encoder_output, tf_second_tweet_hidden_state = lstm(tf_second_tweet_current_word, tf_second_tweet_hidden_state)

    tweet_output_size = int(tf_first_tweet_encoder_output.get_shape()[1])

    tf_first_tweet_humor_ratings = tf_first_tweet_encoder_output
    tf_second_tweet_humor_ratings = tf_second_tweet_encoder_output

    tf_tweet_pair_emb = tf.concat(1, [tf_first_tweet_humor_ratings, tf_second_tweet_humor_ratings])

    tweet_pair_emb_size = int(tf_tweet_pair_emb.get_shape()[1])
    # # END MODEL HIDDEN LAYERS #

    tf_tweet_dense_layer1, _, _ = create_dense_layer(tf_tweet_pair_emb, tweet_pair_emb_size, tweet_pair_emb_size * 3 / 4, activation='relu')
    tf_tweet_dense_layer1_dropout = tf.nn.dropout(tf_tweet_dense_layer1, keep_prob=DROPOUT)
    tf_tweet_dense_layer2, _, _ = create_dense_layer(tf_tweet_dense_layer1_dropout, tweet_pair_emb_size * 3 / 4, tweet_pair_emb_size / 2, activation='relu')
    tf_tweet_humor_rating, _, _ = create_dense_layer(tf_tweet_dense_layer2, tweet_pair_emb_size / 2, 1)

    # Label conversions
    output_logits = tf.reshape(tf_tweet_humor_rating, [-1])
    output_prob = tf.nn.sigmoid(output_logits)
    output = tf.select(tf.greater_equal(output_prob, 0.5), tf.ones_like(output_prob, dtype=tf.int32), tf.zeros_like(output_prob, dtype=tf.int32))

    # tf_predictions = tf.nn.sigmoid(tf_tweet_humor_rating)  # tf.argmax(tf_tweet_humor_ratings, 1, name='prediction')

    return [tf_first_input_tweets, tf_second_input_tweets, output, tf_tweet_humor_rating, tf_batch_size]


def build_embedding_humor_model_trainer(model_vars):
    print 'Building embedding humor model trainer'
    tf_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')
    [_, _, _, tf_tweet_humor_rating, tf_batch_size] = model_vars
    print tf_labels.get_shape()
    print tf_tweet_humor_rating.get_shape()
    tf_reshaped_labels = tf.cast(tf.reshape(tf_labels, [-1, 1]), tf.float32)
    print 'tf_reshaped_labels: %s' % str(tf_reshaped_labels.get_shape())
    tf_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(tf_tweet_humor_rating, tf_reshaped_labels) # tf.nn.sparse_softmax_cross_entropy_with_logits(tf_tweet_humor_ratings, tf_labels)

    print 'tf_cross_entropy: %s' % str(tf_cross_entropy.get_shape())
    tf_loss = tf.reduce_sum(tf_cross_entropy) / tf.cast(tf_batch_size, tf.float32)
    print 'tf_loss: %s' % str(tf_loss.get_shape())

    return [tf_loss, tf_labels]


def train_on_all_other_hashtags(model_vars, trainer_vars, hashtag_names, hashtag_name, n_epochs=1, batch_size=50):
    """Trains on all hashtags in the SEMEVAL_HUMOR_TRAIN_DIR directory. Extracts inputs and labels from
    each hashtag, and trains in batches. Inserts input into model and evaluates output using model_vars.
    Minimizes loss defined in trainer_vars. Repeats for n_epoch epochs. For zero epochs, the model is not trained
    and an accuracy of -1 is returned."""
    if not os.path.exists(EMBEDDING_HUMOR_MODEL_DIR):
        os.makedirs(EMBEDDING_HUMOR_MODEL_DIR)
    [tf_first_input_tweets, tf_second_input_tweets, tf_predictions, tf_tweet_humor_ratings, tf_batch_size] = model_vars
    [tf_loss, tf_labels] = trainer_vars
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=GPU_OPTIONS))
    train_op = tf.train.AdamOptimizer(EMBEDDING_HUMOR_MODEL_LEARNING_RATE).minimize(tf_loss)
    init = tf.initialize_all_variables()
    with tf.name_scope("SAVER"):
        saver = tf.train.Saver(max_to_keep=10)
    sess.run(init)
    accuracies = []
    for epoch in range(n_epochs):
        if n_epochs > 1:
            print 'Epoch %s' % epoch
        for trainer_hashtag_name in hashtag_names:
            if trainer_hashtag_name != hashtag_name:
                # Train on this hashtag.
                np_first_tweets, np_second_tweets, np_labels, first_tweet_ids, second_tweet_ids = \
                    load_hashtag_data(HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR, trainer_hashtag_name)
                current_batch_size = batch_size
                # Use these parameters to keep track of batch size and start location.
                starting_training_example = 0
                num_batches = np_first_tweets.shape[0] / batch_size
                remaining_batch_size = np_first_tweets.shape[0] % batch_size
                batch_accuracies = []
                batch_losses = []
                # print 'Training on hashtag %s' % trainer_hashtag_name
                for i in range(num_batches + 1):
                    # If we are on the last batch, we are running leftover examples.
                    # Otherwise, we stick to global batch_size.
                    if i == num_batches:
                        current_batch_size = remaining_batch_size
                    else:
                        current_batch_size = batch_size
                    np_batch_first_tweets = np_first_tweets[starting_training_example:starting_training_example+current_batch_size]
                    np_batch_second_tweets = np_second_tweets[starting_training_example:starting_training_example+current_batch_size]
                    np_batch_labels = np_labels[starting_training_example:starting_training_example+current_batch_size]
                    # Run train step here.
                    [np_batch_predictions, batch_loss, _] = sess.run([tf_predictions, tf_loss, train_op], feed_dict={tf_first_input_tweets:np_batch_first_tweets,
                                                                                                tf_second_input_tweets:np_batch_second_tweets,
                                                                                                tf_labels: np_batch_labels,
                                                                                                tf_batch_size: current_batch_size})

                    # print np_batch_predictions
                    # print np.mean(np_batch_predictions)
                    if current_batch_size > 0:
                        # np_batch_predictions_reshape = np.reshape(np_batch_predictions, [-1])
                        # batch_accuracy = np.mean(np.round(np_batch_predictions_reshape) == np_batch_labels)
                        batch_accuracy = sklearn.metrics.accuracy_score(np_batch_labels, np_batch_predictions)

                        batch_accuracies.append(batch_accuracy)
                        batch_losses.append(batch_loss)
                        print 'Batch accuracy: %s' % batch_accuracy
                        print 'Batch loss: %s' % batch_loss

                    starting_training_example += batch_size

                hashtag_accuracy = np.mean(batch_accuracies)
                hashtag_loss = np.mean(batch_losses)
                print 'Hashtag %s accuracy: %s' % (trainer_hashtag_name, hashtag_accuracy)
                print 'Hashtag loss: %s' % hashtag_loss
                accuracies.append(hashtag_accuracy)

            else:
                print 'Do not train on current hashtag: %s' % trainer_hashtag_name
        print 'Saving..'
        saver.save(sess, os.path.join(EMBEDDING_HUMOR_MODEL_DIR, 'emb_humor_model'), global_step=epoch)  # Save model after every epoch
    if len(accuracies) > 0:
        training_accuracy = np.mean(accuracies)
    else:
        training_accuracy = -1

    # Save trained model.

    return sess, training_accuracy


def calculate_accuracy_on_batches(batch_predictions, np_labels):
    """batch_predictions is a list of numpy arrays. Each numpy
    array represents the predictions for a single batch. Evaluates
    performance of each batch using np_labels. There must be
    at least one batch containing at least one example"""
    accuracy_sum = 0
    total_examples = 0
    for np_batch_predictions in batch_predictions:
        batch_accuracy = np.mean(np_batch_predictions == np_labels)
        batch_size = np_batch_predictions.shape[0]
        if batch_size > 0:
            accuracy_sum += batch_accuracy * batch_size
            total_examples += batch_size
    return accuracy_sum / total_examples


if __name__ == '__main__':
    main()