"""David Donahue 2017. This script creates predictions for the trial and evaluation datasets.
Saves labels for the trial dataset."""
import cPickle as pickle

import numpy as np
import tensorflow as tf
from keras import backend as K

import humor_predictor
from config import EMB_CHAR_HUMOR_MODEL_DIR, EMB_HUMOR_MODEL_DIR, CHAR_HUMOR_MODEL_DIR
from config import HUMOR_EVAL_PREDICTION_FIRST_TWEET_IDS, HUMOR_EVAL_PREDICTION_SECOND_TWEET_IDS
from config import HUMOR_EVAL_TWEET_PAIR_PREDICTIONS, HUMOR_EVAL_PREDICTION_HASHTAGS
from config import HUMOR_TRIAL_PREDICTION_FIRST_TWEET_IDS, HUMOR_TRIAL_PREDICTION_SECOND_TWEET_IDS
from config import HUMOR_TRIAL_PREDICTION_LABELS
from config import HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR
from config import HUMOR_TRIAL_TWEET_PAIR_PREDICTIONS, HUMOR_TRIAL_PREDICTION_HASHTAGS
from config import SEMEVAL_HUMOR_EVAL_DIR
from config import SEMEVAL_HUMOR_TRIAL_DIR
from tools import get_hashtag_file_names
from tools import load_hashtag_data


def main():
    # Predict on trial dataset
    trial_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRIAL_DIR)

    trial_all_predictions, trial_hashtag_labels, \
    trial_per_hashtag_first_tweet_ids, trial_per_hashtag_second_tweet_ids = \
        predict_with_three_models_on_hashtags(SEMEVAL_HUMOR_TRIAL_DIR,
                                              HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR, trial_hashtag_names)

    pickle.dump(trial_all_predictions, open(HUMOR_TRIAL_TWEET_PAIR_PREDICTIONS, 'wb'))
    pickle.dump(trial_hashtag_names, open(HUMOR_TRIAL_PREDICTION_HASHTAGS, 'wb'))
    pickle.dump(trial_hashtag_labels, open(HUMOR_TRIAL_PREDICTION_LABELS, 'wb'))
    pickle.dump(trial_per_hashtag_first_tweet_ids, open(HUMOR_TRIAL_PREDICTION_FIRST_TWEET_IDS, 'wb'))
    pickle.dump(trial_per_hashtag_second_tweet_ids, open(HUMOR_TRIAL_PREDICTION_SECOND_TWEET_IDS, 'wb'))

    # Predict on evaluation dataset (no labels)
    eval_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_EVAL_DIR)

    eval_all_predictions, eval_hashtag_labels, \
    eval_per_hashtag_first_tweet_ids, eval_per_hashtag_second_tweet_ids = \
        predict_with_three_models_on_hashtags(SEMEVAL_HUMOR_EVAL_DIR, None, eval_hashtag_names, labels_exist=False)

    pickle.dump(eval_all_predictions, open(HUMOR_EVAL_TWEET_PAIR_PREDICTIONS, 'wb'))
    pickle.dump(eval_hashtag_names, open(HUMOR_EVAL_PREDICTION_HASHTAGS, 'wb'))
    pickle.dump(eval_per_hashtag_first_tweet_ids, open(HUMOR_EVAL_PREDICTION_FIRST_TWEET_IDS, 'wb'))
    pickle.dump(eval_per_hashtag_second_tweet_ids, open(HUMOR_EVAL_PREDICTION_SECOND_TWEET_IDS, 'wb'))


def predict_with_three_models_on_hashtags(hashtag_dir, hashtag_emb_dir, trial_hashtag_names, labels_exist=True):
    # eval_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_EVAL_DIR)
    emb_char_predictions = []
    emb_predictions = []
    char_predictions = []
    per_hashtag_first_tweet_ids = []
    per_hashtag_second_tweet_ids = []
    K.clear_session()
    K.set_session(tf.get_default_session())
    hp1 = humor_predictor.HumorPredictor(EMB_CHAR_HUMOR_MODEL_DIR, use_emb_model=True, use_char_model=True)
    for trial_hashtag_name in trial_hashtag_names:
        np_predictions, np_output_prob, np_labels, first_tweet_ids, second_tweet_ids = hp1(hashtag_dir,
                                                                                          trial_hashtag_name)
        emb_char_predictions.append(np_output_prob)
        per_hashtag_first_tweet_ids.append(first_tweet_ids)
        per_hashtag_second_tweet_ids.append(second_tweet_ids)

    K.clear_session()
    K.set_session(tf.get_default_session())
    hp2 = humor_predictor.HumorPredictor(EMB_HUMOR_MODEL_DIR, use_emb_model=True, use_char_model=False)
    for trial_hashtag_name in trial_hashtag_names:
        np_predictions, np_output_prob, np_labels, first_tweet_ids, second_tweet_ids = hp2(hashtag_dir,
                                                                                          trial_hashtag_name)
        emb_predictions.append(np_output_prob)

    K.clear_session()
    K.set_session(tf.get_default_session())
    hp3 = humor_predictor.HumorPredictor(CHAR_HUMOR_MODEL_DIR, use_emb_model=False, use_char_model=True)

    for trial_hashtag_name in trial_hashtag_names:
        np_predictions, np_output_prob, np_labels, first_tweet_ids, second_tweet_ids = hp3(hashtag_dir,
                                                                                       trial_hashtag_name)
        char_predictions.append(np_output_prob)

    all_predictions = []
    for i in range(len(trial_hashtag_names)):
        hashtag_all_predictions = np.concatenate(
            [np.reshape(emb_char_predictions[i], [-1, 1]), np.reshape(emb_predictions[i], [-1, 1]), np.reshape(char_predictions[i], [-1, 1])], axis=1)
        all_predictions.append(hashtag_all_predictions)

    hashtag_labels = None
    if labels_exist:
        hashtag_labels = []
        for hashtag_name in trial_hashtag_names:
            print 'Loading label for hashtag %s' % hashtag_name
            np_first_tweets, np_second_tweets, np_labels, first_tweet_ids, second_tweet_ids, np_hashtag = \
                load_hashtag_data(hashtag_emb_dir, hashtag_name)
            hashtag_labels.append(np_labels)

    return all_predictions, hashtag_labels, per_hashtag_first_tweet_ids, per_hashtag_second_tweet_ids


if __name__ == '__main__':
    main()