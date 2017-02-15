"""David Donahue 2016. This model will be trained and used to predict winning tweets
in the Semeval 2017 Humor Task. The goal of the model is to read tweet pairs and determine
which of the two tweets in a pair is funnier. For each word in a tweet, it will receive a
phonetic embedding and a GloVe embedding to describe how to pronounce the word and what the
word means, respectively. These will serve as features. Other features may be added over time.
This model will be built in Tensorflow."""

import os
import random
import sys

import numpy as np
import sklearn
import tensorflow as tf

from sacred import Experiment
from sacred.observers import MongoObserver

from config import CHAR_HUMOR_MODEL_DIR
from config import EMB_CHAR_HUMOR_MODEL_DIR
from config import EMB_HUMOR_MODEL_DIR
from config import GLOVE_EMB_SIZE
from config import HUMOR_CHAR_TO_INDEX_FILE_PATH
from config import HUMOR_MAX_WORDS_IN_HASHTAG
from config import HUMOR_MAX_WORDS_IN_TWEET
from config import HUMOR_TRAIN_TWEET_PAIR_CHAR_DIR
from config import HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR
from config import HUMOR_TRIAL_TWEET_PAIR_CHAR_DIR
from config import HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR
from config import PHONETIC_EMB_SIZE
from config import SEMEVAL_HUMOR_TRAIN_DIR
from config import SEMEVAL_HUMOR_TRIAL_DIR
from config import TWEET_SIZE
from tf_tools import GPU_OPTIONS
from tf_tools import HUMOR_DROPOUT
from tf_tools import create_dense_layer
from tf_tools import create_tensorboard_visualization
from tf_tools import predict_on_hashtag
from tf_tools import build_humor_model
from tools import extract_tweet_pair_from_hashtag_datas
from tools import get_hashtag_file_names
from tools import load_hashtag_data
from tools import load_hashtag_data_and_vocabulary

ex = Experiment('humor_model')
ex.observers.append(MongoObserver.create(db_name='humor_runs'))

EMBEDDING_HUMOR_MODEL_LEARNING_RATE = .00005
N_TRAIN_EPOCHS = 2


@ex.config
def my_config():
    learning_rate = .00001  # np.random.uniform(.00005, .0000005)
    num_epochs = 1  # int(np.random.uniform(1.0, 4.0))
    dropout = 1  # np.random.uniform(.5, 1.0)
    hidden_dim_size = 800  # int(np.random.uniform(200, 3200))
    use_emb_model = True
    use_char_model = True
    model_save_dir = EMB_CHAR_HUMOR_MODEL_DIR
    if '-emb-only' in sys.argv:
        use_char_model = False
        model_save_dir = EMB_HUMOR_MODEL_DIR
    elif '-char-only' in sys.argv:
        use_emb_model = False
        model_save_dir = CHAR_HUMOR_MODEL_DIR


def build_embedding_humor_model_trainer(model_vars):
    """Take fractional predictions from embedding humor model. Each prediction is a number representing which
    tweet in a pair is funnier(number > 0 => first tweet funnier, number < 0 => second tweet funnier). Runs
    prediction through sigmoid to produce value between 0 and 1. Compares fractional prediction with
    actual label using tf.sigmoid_cross_entropy_with_logits() (sigmoid is done internally)."""
    print 'Building embedding humor model trainer'
    tf_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')
    [tf_first_input_tweets, tf_second_input_tweets, output, tf_tweet_humor_rating, tf_batch_size, tf_hashtag,
     output_prob, tf_dropout_rate, tf_tweet1, tf_tweet2] = model_vars
    tf_reshaped_labels = tf.cast(tf.reshape(tf_labels, [-1, 1]), tf.float32)

    tf_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(tf_tweet_humor_rating, tf_reshaped_labels)

    tf_loss = tf.reduce_sum(tf_cross_entropy) / tf.cast(tf_batch_size, tf.float32)

    return [tf_loss, tf_labels]


def train_on_all_other_hashtags(model_vars, trainer_vars, hashtag_names, hashtag_datas, n_epochs=1,
                                batch_size=50, learning_rate=EMBEDDING_HUMOR_MODEL_LEARNING_RATE,
                                dropout=HUMOR_DROPOUT, model_save_dir=EMB_CHAR_HUMOR_MODEL_DIR, leave_out_hashtags=None):
    """Trains on all hashtags in the SEMEVAL_HUMOR_TRAIN_DIR directory. Extracts inputs and labels from
    each hashtag, and trains in batches. Inserts input into model and evaluates output using model_vars.
    Minimizes loss defined in trainer_vars. Repeats for n_epoch epochs. For zero epochs, the model is not trained
    and an accuracy of -1 is returned."""
    if leave_out_hashtags is None:
        leave_out_hashtags = []
    if isinstance(leave_out_hashtags, str):
        leave_out_hashtags = [leave_out_hashtags]
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    [tf_first_input_tweets, tf_second_input_tweets, tf_predictions, tf_tweet_humor_rating, tf_batch_size, tf_hashtag,
     output_prob, tf_dropout_rate,
     tf_tweet1, tf_tweet2] = model_vars
    [tf_loss, tf_labels] = trainer_vars

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=GPU_OPTIONS))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    init = tf.initialize_all_variables()
    with tf.name_scope("SAVER"):
        saver = tf.train.Saver(max_to_keep=10)
    sess.run(init)

    accuracies = []
    for epoch in range(n_epochs):
        if n_epochs > 1:
            print 'Epoch %s' % epoch
        print hashtag_names
        for trainer_hashtag_name in hashtag_names:
            print trainer_hashtag_name
            if trainer_hashtag_name not in leave_out_hashtags:
                # Train on this hashtag.
                print 'Got here first.'
                np_first_tweets, np_second_tweets, np_labels, first_tweet_ids, second_tweet_ids, np_hashtag = \
                    load_hashtag_data(HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR, trainer_hashtag_name)

                np_first_tweets_char, np_second_tweets_char = \
                    extract_tweet_pair_from_hashtag_datas(hashtag_datas, trainer_hashtag_name)
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
                    if current_batch_size > 0:
                        np_batch_first_tweets = np_first_tweets[
                                                starting_training_example:starting_training_example + current_batch_size,
                                                :]
                        np_batch_second_tweets = np_second_tweets[
                                                 starting_training_example:starting_training_example + current_batch_size,
                                                 :]
                        np_batch_labels = np_labels[
                                          starting_training_example:starting_training_example + current_batch_size]
                        np_batch_hashtag = np_hashtag[
                                           starting_training_example:starting_training_example + current_batch_size, :]
                        np_batch_first_tweets_char = np_first_tweets_char[
                                                     starting_training_example:starting_training_example + current_batch_size,
                                                     :]
                        np_batch_second_tweets_char = np_second_tweets_char[
                                                      starting_training_example:starting_training_example + current_batch_size,
                                                      :]
                        # Run train step here.
                        [np_batch_predictions, batch_loss, _] = sess.run([tf_predictions, tf_loss, train_op],
                                                                         feed_dict={
                                                                             tf_first_input_tweets: np_batch_first_tweets,
                                                                             tf_second_input_tweets: np_batch_second_tweets,
                                                                             tf_labels: np_batch_labels,
                                                                             tf_batch_size: current_batch_size,
                                                                             tf_hashtag: np_batch_hashtag,
                                                                             tf_dropout_rate: dropout,
                                                                             tf_tweet1: np_batch_first_tweets_char,
                                                                             tf_tweet2: np_batch_second_tweets_char})

                        batch_accuracy = sklearn.metrics.accuracy_score(np_batch_labels, np_batch_predictions)

                        batch_accuracies.append(batch_accuracy)
                        batch_losses.append(batch_loss)

                    starting_training_example += current_batch_size

                hashtag_accuracy = np.mean(batch_accuracies)
                hashtag_loss = np.mean(batch_losses)
                print 'Hashtag %s accuracy: %s' % (trainer_hashtag_name, hashtag_accuracy)
                print 'Hashtag loss: %s' % hashtag_loss
                accuracies.append(hashtag_accuracy)

            else:
                print 'Do not train on current hashtag: %s' % trainer_hashtag_name
        print 'Saving..'
        saver.save(sess, os.path.join(model_save_dir, 'emb_humor_model'),
                   global_step=epoch)  # Save model after every epoch
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


def load_build_train_and_predict(learning_rate, num_epochs, dropout, use_emb_model,
                                 use_char_model, model_save_dir, hidden_dim_size, leave_out_hashtags=[]):
    """Builds and trains a humor model on the semeval task training set. Evaluates on the semeval task trial set,
    prints accuracy. Saves model after each epoch of training.

    learning_rate - rate at which model learns dataset
    num_epochs - number of times model trains on entire training dataset
    dropout - specifies the keep rate of dropout in the model. 1 means no dropout
    use_emb_model - use embedding features in model prediction
    use_char_model - use characters of tweet as features in model prediction
    model_save_dir - where to save model parameters after each epoch
    hidden_dim_size - size of lstm embedding encoder
    leave_out_hashtags - hashtag names to omit from training step (could use for ensemble model training)"""
    print 'Learning rate: %s' % learning_rate
    print 'Number of epochs: %s' % num_epochs
    print 'Dropout keep rate: %s' % dropout
    print 'Use embedding model: %s' % use_emb_model
    print 'Use character model: %s' % use_char_model
    print 'Model save directory: %s' % model_save_dir
    print 'Tweet encoder state size: %s' % hidden_dim_size

    random.seed('hello world')
    hashtag_datas, char_to_index, vocab_size = load_hashtag_data_and_vocabulary(HUMOR_TRAIN_TWEET_PAIR_CHAR_DIR,
                                                                                HUMOR_CHAR_TO_INDEX_FILE_PATH)
    trial_hashtag_datas, _, trial_vocab_size = \
        load_hashtag_data_and_vocabulary(HUMOR_TRIAL_TWEET_PAIR_CHAR_DIR, None)

    g = tf.Graph()
    with g.as_default():
        model_vars = build_humor_model(vocab_size, use_embedding_model=use_emb_model,
                                       use_character_model=use_char_model,
                                       hidden_dim_size=hidden_dim_size)
        trainer_vars = build_embedding_humor_model_trainer(model_vars)
        create_tensorboard_visualization('emb_humor_model')
        training_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRAIN_DIR)
        testing_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRIAL_DIR)
        accuracies = []

        sess, training_accuracy = train_on_all_other_hashtags(model_vars, trainer_vars, training_hashtag_names,
                                                              hashtag_datas, n_epochs=num_epochs,
                                                              learning_rate=learning_rate,
                                                              dropout=dropout,
                                                              model_save_dir=model_save_dir,
                                                              leave_out_hashtags=leave_out_hashtags)
        print 'Mean training accuracy: %s' % training_accuracy
        print
        for hashtag_name in testing_hashtag_names:
            accuracy, _ = predict_on_hashtag(sess,
                                             model_vars,
                                             hashtag_name,
                                             HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR,
                                             trial_hashtag_datas,
                                             error_analysis_stats=[SEMEVAL_HUMOR_TRIAL_DIR, 10])
            print 'Hashtag %s accuracy: %s' % (hashtag_name, accuracy)
            accuracies.append(accuracy)

        test_accuracy = np.mean(accuracies)
        print 'Final test accuracy: %s' % test_accuracy

    return {'train_accuracy': training_accuracy,
            'test_accuracy': test_accuracy}


@ex.main
def main(learning_rate, num_epochs, dropout, use_emb_model,
         use_char_model, model_save_dir, hidden_dim_size, leave_out_hashtags=[]):
    load_build_train_and_predict(learning_rate, num_epochs, dropout, use_emb_model,
                                 use_char_model, model_save_dir, hidden_dim_size, leave_out_hashtags=[])


if __name__ == '__main__':
    num_experiments_run = 1
    for index in range(num_experiments_run):
        print 'Experiment: %s' % index
        r = ex.run()
