"""David Donahue 2017. This script creates predictions for the trial and evaluation datasets.
Saves labels for the trial dataset."""
import cPickle as pickle

import numpy as np
import tensorflow as tf
from keras import backend as K

import humor_predictor
import config
from tools import get_hashtag_file_names
from tools import load_hashtag_data


def main():
    # Predict on trial dataset
    trial_hashtag_names = get_hashtag_file_names(config.SEMEVAL_HUMOR_TRIAL_DIR)

    trial_all_predictions, trial_hashtag_labels, \
    trial_per_hashtag_first_tweet_ids, trial_per_hashtag_second_tweet_ids = \
        predict_with_three_models_on_hashtags(config.SEMEVAL_HUMOR_TRIAL_DIR, trial_hashtag_names)

    print len(trial_hashtag_labels)
    print trial_hashtag_labels[0].shape
    print trial_hashtag_labels[0]

    pickle.dump(trial_all_predictions, open(config.HUMOR_TRIAL_TWEET_PAIR_PREDICTIONS, 'wb'))
    pickle.dump(trial_hashtag_names, open(config.HUMOR_TRIAL_PREDICTION_HASHTAGS, 'wb'))
    pickle.dump(trial_hashtag_labels, open(config.HUMOR_TRIAL_PREDICTION_LABELS, 'wb'))
    pickle.dump(trial_per_hashtag_first_tweet_ids, open(config.HUMOR_TRIAL_PREDICTION_FIRST_TWEET_IDS, 'wb'))
    pickle.dump(trial_per_hashtag_second_tweet_ids, open(config.HUMOR_TRIAL_PREDICTION_SECOND_TWEET_IDS, 'wb'))

    # Predict on evaluation dataset (no labels)
    eval_hashtag_names = get_hashtag_file_names(config.SEMEVAL_HUMOR_GOLD_DIR)

    eval_all_predictions, eval_hashtag_labels, \
    eval_per_hashtag_first_tweet_ids, eval_per_hashtag_second_tweet_ids = \
        predict_with_three_models_on_hashtags(config.SEMEVAL_HUMOR_GOLD_DIR, eval_hashtag_names)

    print len(eval_hashtag_labels)
    print eval_hashtag_labels[0].shape
    print eval_hashtag_labels[0]

    pickle.dump(eval_all_predictions, open(config.HUMOR_EVAL_TWEET_PAIR_PREDICTIONS, 'wb'))
    pickle.dump(eval_hashtag_names, open(config.HUMOR_EVAL_PREDICTION_HASHTAGS, 'wb'))
    pickle.dump(eval_hashtag_labels, open(config.HUMOR_EVAL_PREDICTION_LABELS, 'wb'))
    pickle.dump(eval_per_hashtag_first_tweet_ids, open(config.HUMOR_EVAL_PREDICTION_FIRST_TWEET_IDS, 'wb'))
    pickle.dump(eval_per_hashtag_second_tweet_ids, open(config.HUMOR_EVAL_PREDICTION_SECOND_TWEET_IDS, 'wb'))


def predict_with_three_models_on_hashtags(hashtag_dir, hashtag_names, labels_exist=True):
    # eval_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_EVAL_DIR)
    emb_char_predictions = []
    emb_predictions = []
    char_predictions = []
    per_hashtag_first_tweet_ids = []
    per_hashtag_second_tweet_ids = []
    hashtag_labels = []
    K.clear_session()
    K.set_session(tf.get_default_session())
    hp1 = humor_predictor.HumorPredictor(config.EMB_CHAR_HUMOR_MODEL_DIR, use_emb_model=True, use_char_model=True)
    # Generate predictions for emb-char model
    for trial_hashtag_name in hashtag_names:
        np_predictions, np_output_prob, np_labels, first_tweet_ids, second_tweet_ids = hp1(hashtag_dir,
                                                                                          trial_hashtag_name)
        emb_char_predictions.append(np_output_prob)
        per_hashtag_first_tweet_ids.append(first_tweet_ids)
        per_hashtag_second_tweet_ids.append(second_tweet_ids)
        if labels_exist:
            hashtag_labels.append(np_labels)

    K.clear_session()
    K.set_session(tf.get_default_session())
    hp2 = humor_predictor.HumorPredictor(config.EMB_HUMOR_MODEL_DIR, use_emb_model=True, use_char_model=False)
    # Generate predictions for emb model
    for trial_hashtag_name in hashtag_names:
        np_predictions, np_output_prob, np_labels, first_tweet_ids, second_tweet_ids = hp2(hashtag_dir,
                                                                                          trial_hashtag_name)
        emb_predictions.append(np_output_prob)

    K.clear_session()
    K.set_session(tf.get_default_session())
    hp3 = humor_predictor.HumorPredictor(config.CHAR_HUMOR_MODEL_DIR, use_emb_model=False, use_char_model=True)
    # Generate predictions for char model
    for trial_hashtag_name in hashtag_names:
        np_predictions, np_output_prob, np_labels, first_tweet_ids, second_tweet_ids = hp3(hashtag_dir,
                                                                                       trial_hashtag_name)
        char_predictions.append(np_output_prob)

    all_predictions = []
    for i in range(len(hashtag_names)):
        hashtag_all_predictions = np.concatenate(
            [np.reshape(emb_char_predictions[i], [-1, 1]), np.reshape(emb_predictions[i], [-1, 1]), np.reshape(char_predictions[i], [-1, 1])], axis=1)
        all_predictions.append(hashtag_all_predictions)

    # hashtag_labels = None
    # if labels_exist:
    #     hashtag_labels = []
    #     for hashtag_name in hashtag_names:
    #         print 'Loading label for hashtag %s' % hashtag_name
    #         np_first_tweets, np_second_tweets, np_labels, first_tweet_ids, second_tweet_ids, np_hashtag = \
    #             load_hashtag_data(hashtag_emb_dir, hashtag_name)
    #         hashtag_labels.append(np_labels)

    return all_predictions, hashtag_labels, per_hashtag_first_tweet_ids, per_hashtag_second_tweet_ids


if __name__ == '__main__':
    main()