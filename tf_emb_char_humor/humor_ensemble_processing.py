"""David Donahue 2017. Trying out training and predicting using the humor model."""
from keras import backend as K
import tensorflow as tf
import random
import numpy as np
import humor_predictor
import cPickle as pickle
from humor_model import load_build_train_and_predict
from config import EMB_CHAR_HUMOR_MODEL_DIR, CHAR_HUMOR_MODEL_DIR, EMB_HUMOR_MODEL_DIR
from config import HUMOR_TRIAL_TWEET_PAIR_CHAR_DIR, HUMOR_TRAIN_TWEET_PAIR_CHAR_DIR
from config import HUMOR_CHAR_TO_INDEX_FILE_PATH, SEMEVAL_HUMOR_TRAIN_DIR
from config import HUMOR_TRAIN_TWEET_PAIR_PREDICTIONS, HUMOR_TRAIN_PREDICTION_HASHTAGS
from config import HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR, HUMOR_TRAIN_PREDICTION_LABELS
from config import SEMEVAL_HUMOR_TRIAL_DIR
from tf_tools import GPU_OPTIONS
from tools import load_hashtag_data_and_vocabulary, get_hashtag_file_names
from tools import load_hashtag_data


learning_rate = .00005
num_epochs = 1
dropout = 1
hidden_dim_size = 800
num_groups = 5


def main():
    """Use embedding model, character model, and joint model to make predictions on all
    hashtags in the training directory. These predictions will be used as features to the
    ensemble model."""
    sync_seed = 'hello world'
    emb_char_predictions, hashtag_names, emb_char_accuracies = \
        train_and_make_predictions_on_all_hashtags(num_groups,
                                                   model_save_dir=EMB_CHAR_HUMOR_MODEL_DIR,
                                                   use_emb_model=True,
                                                   use_char_model=True,
                                                   seed=sync_seed)

    emb_predictions, hashtag_names2, emb_accuracies = \
        train_and_make_predictions_on_all_hashtags(num_groups,
                                                   model_save_dir=EMB_HUMOR_MODEL_DIR,
                                                   use_emb_model=True,
                                                   use_char_model=False,
                                                   seed=sync_seed)

    char_predictions, hashtag_names3, char_accuracies = \
        train_and_make_predictions_on_all_hashtags(num_groups,
                                                   model_save_dir=CHAR_HUMOR_MODEL_DIR,
                                                   use_emb_model=False,
                                                   use_char_model=True,
                                                   seed=sync_seed)

    assert len(emb_char_predictions) == len(emb_predictions)
    assert len(emb_predictions) == len(char_predictions)
    assert len(char_predictions) == len(hashtag_names)
    print str(char_predictions[0].shape)
    assert hashtag_names == hashtag_names2
    assert hashtag_names2 == hashtag_names3
    random.seed(sync_seed)
    hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRAIN_DIR)
    # Get labels
    hashtag_labels = []
    for hashtag_name in hashtag_names:
        print 'Loading label for hashtag %s' % hashtag_name
        np_first_tweets, np_second_tweets, np_labels, first_tweet_ids, second_tweet_ids, np_hashtag = \
            load_hashtag_data(HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR, hashtag_name)
        hashtag_labels.append(np_labels)

    all_predictions = []
    for i in range(len(hashtag_names)):
        hashtag_all_predictions = np.concatenate(
            [np.reshape(emb_char_predictions[i], [-1, 1]), np.reshape(emb_predictions[i], [-1, 1]), np.reshape(char_predictions[i], [-1, 1])], axis=1)
        all_predictions.append(hashtag_all_predictions)

    # Save
    pickle.dump(all_predictions, open(HUMOR_TRAIN_TWEET_PAIR_PREDICTIONS, 'wb'))
    pickle.dump(hashtag_names, open(HUMOR_TRAIN_PREDICTION_HASHTAGS, 'wb'))
    pickle.dump(hashtag_labels, open(HUMOR_TRAIN_PREDICTION_LABELS, 'wb'))


def train_and_make_predictions_on_all_hashtags(num_groups, model_save_dir=EMB_CHAR_HUMOR_MODEL_DIR, use_emb_model=True, use_char_model=True, seed=None):
    print 'use_emb_model: %s' % use_emb_model
    print 'use_char_model: %s' % use_char_model
    """Makes predictions on all hashtags in training directory, by dividing them into num_groups groups. For each group, trains on all
    hashtags not in the group and predicts on group. Saves trained model in model_save_dir.

    use_emb_model - flag indicating whether to train using word embeddings as features
    use_char_model - flag indicating whether to train using character indices as features"""
    if seed is not None:
        random.seed(seed)
    hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRAIN_DIR)
    print len(hashtag_names)
    hashtag_predictions = []
    hashtag_accuracies = []

    # for i in range(len(hashtag_names)):
    #     hashtag_name = hashtag_names[i]
    #     num_lines = sum(1 for line in open(SEMEVAL_HUMOR_TRAIN_DIR + hashtag_name + '.tsv'))
    for hashtag_group_index in range(num_groups):

        num_hashtags = len(hashtag_names)
        num_hashtags_in_group = num_hashtags / num_groups + 1
        starting_hashtag_index = num_hashtags_in_group * hashtag_group_index
        hashtags_in_group = hashtag_names[starting_hashtag_index:starting_hashtag_index + num_hashtags_in_group]

        print hashtag_group_index, starting_hashtag_index, starting_hashtag_index + num_hashtags_in_group, num_hashtags_in_group
        print hashtags_in_group
        #
        # Train on all hashtags not in group
        K.clear_session()
        K.set_session(tf.get_default_session())
        trainable_vars = train_on_hashtags_in_group(hashtag_names, hashtags_in_group,
                                                                       model_save_dir, use_emb_model,
                                                                       use_char_model)

        # Predict on hashtags in group
        K.clear_session()
        K.set_session(tf.get_default_session())
        hp = humor_predictor.HumorPredictor(model_save_dir, use_char_model=use_char_model, use_emb_model=use_emb_model)
        accuracies = []
        counter = 0
        for hashtag_name in hashtags_in_group:
            assert hashtag_names[hashtag_group_index * num_hashtags_in_group + counter] == hashtag_name
            print hashtag_name
            np_predictions, np_output_prob, np_labels, first_tweet_ids, second_tweet_ids = hp(SEMEVAL_HUMOR_TRAIN_DIR, hashtag_name)
            # np_first_tweets2, np_second_tweets2, np_labels2, first_tweet_ids2, second_tweet_ids2, np_hashtag2 = \
            #     load_hashtag_data(HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR, hashtag_name)
            # assert np_labels == np_labels2
            accuracy = np.mean(np_predictions == np_labels)
            print 'Hashtag accuracy: %s' % accuracy
            accuracies.append(accuracy)
            hashtag_predictions.append(np_output_prob)
            hashtag_accuracies.append(accuracy)

            print hashtag_name
            print np_predictions.shape
            counter += 1
        print 'Trial accuracy: %s' % np.mean(accuracies)
    return hashtag_predictions, hashtag_names, hashtag_accuracies


def train_on_hashtags_in_group(hashtag_names, hashtags_in_group,
                               model_save_dir, use_emb_model, use_char_model):
    """Given a list of hashtag names, divide into num_groups and train on all hashtags
    but those in that group. Save trained model in model_save_dir.

    use_emb_model - flag indicating whether to train using word embeddings as features
    use_char_model - flag indicating whether to train using character indices as features
    model_save_dir - location to save model, should correspond with model configuration"""

    load_build_train_and_predict(learning_rate, num_epochs, dropout, use_emb_model,
                                 use_char_model, model_save_dir, hidden_dim_size,
                                 leave_out_hashtags=hashtags_in_group)
    trainable_vars = tf.trainable_variables()
    return trainable_vars


if __name__ == '__main__':
    main()