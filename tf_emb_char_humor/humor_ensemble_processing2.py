"""David Donahue 2017. This script creates predictions for the trial and evaluation datasets.
Saves labels for the trial dataset."""
import humor_predictor
import tensorflow as tf
import numpy as np
import cPickle as pickle
from keras import backend as K
from config import SEMEVAL_HUMOR_TRIAL_DIR
from config import SEMEVAL_HUMOR_EVAL_DIR
from config import EMB_CHAR_HUMOR_MODEL_DIR, EMB_HUMOR_MODEL_DIR, CHAR_HUMOR_MODEL_DIR
from config import HUMOR_TRIAL_TWEET_PAIR_PREDICTIONS, HUMOR_TRIAL_PREDICTION_HASHTAGS
from config import HUMOR_TRIAL_PREDICTION_LABELS
from tools import get_hashtag_file_names
from tools import load_hashtag_data
from config import HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR
from config import HUMOR_TRIAL_PREDICTION_FIRST_TWEET_IDS
from config import HUMOR_TRIAL_PREDICTION_SECOND_TWEET_IDS


def main():
    trial_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRIAL_DIR)
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
        np_predictions, np_output_prob, np_labels, first_tweet_ids, second_tweet_ids = hp1(SEMEVAL_HUMOR_TRIAL_DIR,
                                                                                          trial_hashtag_name)
        emb_char_predictions.append(np_output_prob)
        per_hashtag_first_tweet_ids.append(first_tweet_ids)
        per_hashtag_second_tweet_ids.append(second_tweet_ids)

    K.clear_session()
    K.set_session(tf.get_default_session())
    hp2 = humor_predictor.HumorPredictor(EMB_HUMOR_MODEL_DIR, use_emb_model=True, use_char_model=False)
    for trial_hashtag_name in trial_hashtag_names:
        np_predictions, np_output_prob, np_labels, first_tweet_ids, second_tweet_ids = hp2(SEMEVAL_HUMOR_TRIAL_DIR,
                                                                                          trial_hashtag_name)
        emb_predictions.append(np_output_prob)

    K.clear_session()
    K.set_session(tf.get_default_session())
    hp3 = humor_predictor.HumorPredictor(CHAR_HUMOR_MODEL_DIR, use_emb_model=False, use_char_model=True)

    for trial_hashtag_name in trial_hashtag_names:
        np_predictions, np_output_prob, np_labels, first_tweet_ids, second_tweet_ids = hp3(SEMEVAL_HUMOR_TRIAL_DIR,
                                                                                       trial_hashtag_name)
        char_predictions.append(np_output_prob)

    all_predictions = []
    for i in range(len(trial_hashtag_names)):
        hashtag_all_predictions = np.concatenate(
            [np.reshape(emb_char_predictions[i], [-1, 1]), np.reshape(emb_predictions[i], [-1, 1]), np.reshape(char_predictions[i], [-1, 1])], axis=1)
        all_predictions.append(hashtag_all_predictions)

    hashtag_labels = []
    for hashtag_name in trial_hashtag_names:
        print 'Loading label for hashtag %s' % hashtag_name
        np_first_tweets, np_second_tweets, np_labels, first_tweet_ids, second_tweet_ids, np_hashtag = \
            load_hashtag_data(HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR, hashtag_name)
        hashtag_labels.append(np_labels)

    pickle.dump(all_predictions, open(HUMOR_TRIAL_TWEET_PAIR_PREDICTIONS, 'wb'))
    pickle.dump(trial_hashtag_names, open(HUMOR_TRIAL_PREDICTION_HASHTAGS, 'wb'))
    pickle.dump(hashtag_labels, open(HUMOR_TRIAL_PREDICTION_LABELS, 'wb'))
    pickle.dump(per_hashtag_first_tweet_ids, open(HUMOR_TRIAL_PREDICTION_FIRST_TWEET_IDS, 'wb'))
    pickle.dump(per_hashtag_second_tweet_ids, open(HUMOR_TRIAL_PREDICTION_SECOND_TWEET_IDS, 'wb'))

if __name__ == '__main__':
    main()