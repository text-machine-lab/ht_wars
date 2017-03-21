"""David Donahue 2016. For a given hashtag Dog_Jobs.tsv, it produces a file Dog_Jobs_PREDICT.tsv."""
import os

import humor_predictor
from config import EMB_CHAR_HUMOR_MODEL_DIR
from config import SEMEVAL_EVAL_PREDICTIONS
from config import SEMEVAL_HUMOR_EVAL_DIR
from tools import get_hashtag_file_names


def main():
    SEMEVAL_TRIAL_PREDICTIONS = '../data/trial_dir/trial_predict/'
    SEMEVAL_HUMOR_TRIAL_DIR_NO_LABELS = '../data/trial_dir/trial_data_eval_format/'
    """Creates a humor predictor that uses the embedding/character joint model.
    Predicts on evaluation dataset. Converts to submission format."""
    if not os.path.exists(SEMEVAL_EVAL_PREDICTIONS):
        os.makedirs(SEMEVAL_EVAL_PREDICTIONS)
    hp = humor_predictor.HumorPredictor(EMB_CHAR_HUMOR_MODEL_DIR)
    eval_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_EVAL_DIR)
    for hashtag_name in eval_hashtag_names:
        np_predictions, np_output_prob, np_labels, first_tweet_ids, second_tweet_ids= hp(SEMEVAL_HUMOR_EVAL_DIR, hashtag_name)
        write_predictions_to_file(os.path.join(SEMEVAL_EVAL_PREDICTIONS, hashtag_name + '_PREDICT.tsv'),
                                  np_predictions, first_tweet_ids, second_tweet_ids)


def write_predictions_to_file(filename, np_predictions, first_tweet_ids, second_tweet_ids):
    """Written for evaluation script, this function writes predictions of a humor
    model on tweet pairs to file. Each line corresponds to a tweet pair, and takes
    on the form: <tweet1_id>\t<tweet2_id>\t<first_tweet_is_funnier>\n"""
    f = open(filename, 'wb')
    for index in range(len(first_tweet_ids)):
        first_tweet_is_funnier = np_predictions[index]
        first_tweet_id = first_tweet_ids[index]
        second_tweet_id = second_tweet_ids[index]
        f.write(str(first_tweet_id))
        f.write('\t')
        f.write(str(second_tweet_id))
        f.write('\t')
        f.write(str(first_tweet_is_funnier))
        f.write('\n')

if __name__ == '__main__':
    main()