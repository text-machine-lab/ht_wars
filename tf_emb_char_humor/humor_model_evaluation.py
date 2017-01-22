"""David Donahue 2016. For a given hashtag Dog_Jobs.tsv, it produces a file Dog_Jobs_PREDICT.tsv for all hashtags in the current
directory."""
import sys
import os
import humor_model
import tensorflow as tf
from tf_tools import GPU_OPTIONS
from config import EMB_CHAR_HUMOR_MODEL_DIR
from config import HUMOR_CHAR_TO_INDEX_FILE_PATH
from tf_tools import predict_on_hashtag
from tools import get_hashtag_file_names
from tools import load_hashtag_data_and_vocabulary
from config import HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR


def main():
    """Creates a humor model, loads variables and predicts on all hashtags
    in directory of first argument. Saves a prediction file for each hashtag
    in a directory specified by the second command line argument.
    Only works for combined embedding/character model, not for each model separately."""
    if len(sys.argv) < 4:
        print 'Usage: humor_model_evaluation [tweet_dir] [tweet_emb_pair_dir] [ tweet_char_pair_dir] [output_dir]'
    else:
        tweet_dir = sys.argv[1]
        tweet_emb_pair_dir = sys.argv[2]
        tweet_char_pair_dir = sys.argv[3]
        output_dir = sys.argv[4]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        hashtag_datas, char_to_index, vocab_size = load_hashtag_data_and_vocabulary(tweet_char_pair_dir,
                                                                                    HUMOR_CHAR_TO_INDEX_FILE_PATH)

        model_vars = humor_model.build_embedding_humor_model(vocab_size)

        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=GPU_OPTIONS))
        saver = tf.train.Saver(max_to_keep=10)
        # Restore model from previous save.
        ckpt = tf.train.get_checkpoint_state(EMB_CHAR_HUMOR_MODEL_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint found!")
            return -1
        hashtag_names = get_hashtag_file_names(tweet_dir)
        for hashtag_name in hashtag_names:
            accuracy, [np_predictions, first_tweet_ids, second_tweet_ids] = \
                predict_on_hashtag(sess, model_vars, hashtag_name, tweet_emb_pair_dir, hashtag_datas)
            write_predictions_to_file(output_dir + hashtag_name + '_PREDICT.tsv',
                                      np_predictions,
                                      first_tweet_ids,
                                      second_tweet_ids)


def write_predictions_to_file(filename, np_predictions, first_tweet_ids, second_tweet_ids):
    """Written for evaluation script, this function writes predictions of a humor
    model on tweet pairs to file. Each line corresponds to a tweet pair, and takes
    on the form: <tweet1_id>\t<tweet2_id>\t<first_tweet_is_funnier>\n"""
    file = open(filename, 'wb')
    for index in range(len(first_tweet_ids)):
        first_tweet_is_funnier = np_predictions[index]
        first_tweet_id = first_tweet_ids[index]
        second_tweet_id = second_tweet_ids[index]
        file.write(str(first_tweet_id))
        file.write('\t')
        file.write(str(second_tweet_id))
        file.write('\t')
        file.write(str(first_tweet_is_funnier))
        file.write('\n')

if __name__ == '__main__':
    main()