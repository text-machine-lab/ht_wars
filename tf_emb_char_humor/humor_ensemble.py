"""David Donahue 2016. Trying out training and predicting using the humor model."""
from keras import backend as K
import tensorflow as tf
import random
import numpy as np
import humor_predictor
from humor_model import load_build_train_and_predict
from config import EMB_CHAR_HUMOR_MODEL_DIR
from config import HUMOR_TRIAL_TWEET_PAIR_CHAR_DIR, HUMOR_TRAIN_TWEET_PAIR_CHAR_DIR
from config import HUMOR_CHAR_TO_INDEX_FILE_PATH, SEMEVAL_HUMOR_TRAIN_DIR
from config import SEMEVAL_HUMOR_TRIAL_DIR
from tf_tools import GPU_OPTIONS
from tools import load_hashtag_data_and_vocabulary, get_hashtag_file_names


learning_rate = .00005
num_epochs = 0
dropout = 1
use_emb_model = True
use_char_model = True
model_save_dir = EMB_CHAR_HUMOR_MODEL_DIR
hidden_dim_size = 800


def main():

    # Load the training set
    training_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRAIN_DIR)
    num_groups = 5

    for hashtag_group_index in range(num_groups):
        # Train on all hashtags not in group
        hashtags_in_group, trainable_vars = train_on_hashtags_in_group(training_hashtag_names, hashtag_group_index, num_groups)
        tf.reset_default_graph()

        # Predict on hashtags in group
        hp = humor_predictor.HumorPredictor(EMB_CHAR_HUMOR_MODEL_DIR)
        K.set_session(tf.get_default_session())
        accuracies = []
        for hashtag_name in hashtags_in_group:
            print hashtag_name
            np_predictions, np_output_prob, np_labels = hp(SEMEVAL_HUMOR_TRAIN_DIR, hashtag_name)
            accuracy = np.mean(np_predictions == np_labels)
            print np_output_prob
            print np_predictions
            print 'Hashtag accuracy: %s' % accuracy
            accuracies.append(accuracy)
        print 'Trial accuracy: %s' % np.mean(accuracies)


def train_on_hashtags_in_group(training_hashtag_names, hashtag_group_index, num_groups):
    num_hashtags = len(training_hashtag_names)
    num_hashtags_in_group = num_hashtags / num_groups
    starting_hashtag_index = num_hashtags_in_group * hashtag_group_index
    if hashtag_group_index == num_groups - 1:
        num_hashtags_in_group = num_hashtags - starting_hashtag_index
    hashtags_in_group = training_hashtag_names[starting_hashtag_index:starting_hashtag_index + num_hashtags_in_group]
    print hashtags_in_group
    print len(hashtags_in_group)
    load_build_train_and_predict(learning_rate, num_epochs, dropout, use_emb_model,
                                 use_char_model, model_save_dir, hidden_dim_size,
                                 leave_out_hashtags=hashtags_in_group)
    trainable_vars = tf.trainable_variables()
    return hashtags_in_group, trainable_vars


if __name__ == '__main__':
    main()