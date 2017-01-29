"""David Donahue 2016. Class to make predictions on a hashtag from file. Can make predictions
with embedding model, character model, or both."""
import cPickle as pickle
import random
import tensorflow as tf
import numpy as np
from config import HUMOR_INDEX_TO_WORD_FILE_PATH
from config import HUMOR_WORD_TO_GLOVE_FILE_PATH
from config import HUMOR_WORD_TO_PHONETIC_FILE_PATH
from config import EMB_CHAR_HUMOR_MODEL_DIR, TWEET_SIZE
from config import EMB_HUMOR_MODEL_DIR, CHAR_HUMOR_MODEL_DIR
from config import TWEET_PAIR_LABEL_RANDOM_SEED
from config import HUMOR_CHAR_TO_INDEX_FILE_PATH
from config import SEMEVAL_HUMOR_TRIAL_DIR, SEMEVAL_HUMOR_EVAL_DIR
from tools import convert_hashtag_to_embedding_tweet_pairs
from tools import extract_tweet_pairs_from_file
from tools import format_tweet_pairs, save_hashtag_data, get_hashtag_file_names

from tf_tools import build_humor_model, predict_on_hashtag, GPU_OPTIONS


class HumorPredictor:
    """Makes predictions on individual hashtags from pre-trained model. To run
    multiple humor predictors, use a different graph for each one.

    model_var_dir - location of model variables corresponding to current model build
    use_emb_model - true if model will use embeddings to make predictions
    use_char_model - true if model will use individual chars to make predictions"""
    def __init__(self, model_var_dir, use_emb_model=True, use_char_model=True, scope=None, v=True, sess=None):
        print use_emb_model
        print use_char_model
        self.model_var_dir = model_var_dir
        if v:
            print self.model_var_dir
        self.use_emb_model = use_emb_model
        if v:
            print 'self.use_emb_model: %s' % self.use_emb_model
        self.use_char_model = use_char_model
        if v:
            print 'self.use_char_model: %s' % self.use_char_model
        self.vocabulary = pickle.load(open(HUMOR_INDEX_TO_WORD_FILE_PATH, 'rb'))
        if v:
            print 'len vocabulary: %s' % len(self.vocabulary)
        self.word_to_glove = pickle.load(open(HUMOR_WORD_TO_GLOVE_FILE_PATH, 'rb'))
        if v:
            print 'len word_to_glove: %s' % len(self.word_to_glove)
        self.word_to_phonetic = pickle.load(open(HUMOR_WORD_TO_PHONETIC_FILE_PATH, 'rb'))
        if v:
            print 'len word_to_phonetic: %s' % len(self.word_to_phonetic)
        self.char_to_index = pickle.load(open(HUMOR_CHAR_TO_INDEX_FILE_PATH, 'rb'))
        if v:
            print 'len char_to_index: %s' % len(self.char_to_index)

        [self.tf_first_input_tweets, self.tf_second_input_tweets, self.tf_output, tf_tweet_humor_rating,
         self.tf_batch_size, tf_hashtag, self.tf_output_prob, self.tf_dropout_rate, self.tf_tweet1, self.tf_tweet2] \
            = build_humor_model(len(self.char_to_index), use_embedding_model=self.use_emb_model,
                                use_character_model=self.use_char_model, hidden_dim_size=None)
        self.sess = restore_model_from_save(model_var_dir, sess=sess)

    def __call__(self, tweet_input_dir, hashtag_name):
        """Makes prediction on a single hashtag.

        tweet_input_dir - location of hashtag .tsv file
        hashtag_name - name of hashtag file without .tsv extension"""
        np_first_tweets, np_second_tweets, first_tweet_ids, second_tweet_ids, np_labels, np_hashtag_gloves = \
            convert_hashtag_to_embedding_tweet_pairs(tweet_input_dir, hashtag_name,
                                                     self.word_to_glove, self.word_to_phonetic)
        random.seed(TWEET_PAIR_LABEL_RANDOM_SEED + hashtag_name)
        data = extract_tweet_pairs_from_file(tweet_input_dir + hashtag_name + '.tsv')
        np_tweet_pairs, np_tweet_pair_labels = format_tweet_pairs(data, self.char_to_index)
        np_first_tweets_char = np_tweet_pairs[:, :TWEET_SIZE]
        np_second_tweets_char = np_tweet_pairs[:, TWEET_SIZE:]
        hashtag_datas = [[hashtag_name, np_tweet_pairs]]
        np_predictions, np_output_prob = self.sess.run([self.tf_output, self.tf_output_prob],
                                                  feed_dict={self.tf_first_input_tweets: np_first_tweets,
                                                             self.tf_second_input_tweets: np_second_tweets,
                                                             self.tf_batch_size: np_first_tweets.shape[0],
                                                             self.tf_dropout_rate: 1.0,
                                                             self.tf_tweet1: np_first_tweets_char,
                                                             self.tf_tweet2: np_second_tweets_char})
        return np_predictions, np_output_prob, np_labels


def restore_model_from_save(model_var_dir, sess=None):
    """Restores all model variables from the specified directory."""
    if sess is None:
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=GPU_OPTIONS))

    saver = tf.train.Saver(max_to_keep=10)
    # Restore model from previous save.
    ckpt = tf.train.get_checkpoint_state(model_var_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No checkpoint found!")
        return -1

    return sess


def main():
    """Test for HumorMultiPredictor."""
    hp = HumorPredictor(EMB_CHAR_HUMOR_MODEL_DIR)
    hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRIAL_DIR)
    eval_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_EVAL_DIR)
    accuracies = []
    for hashtag_name in hashtag_names:
        print hashtag_name
        np_predictions, np_output_prob, np_labels = hp(SEMEVAL_HUMOR_TRIAL_DIR, hashtag_name)
        accuracy = np.mean(np_predictions == np_labels)
        print np_output_prob
        print np_predictions
        print 'Hashtag accuracy: %s' % accuracy
        accuracies.append(accuracy)
    print 'Trial accuracy: %s' % np.mean(accuracies)
    for hashtag_name in eval_hashtag_names:
        print hashtag_name
        np_predictions2, np_output_prob, np_labels = hp(SEMEVAL_HUMOR_EVAL_DIR, hashtag_name)
        print np_output_prob
        print np_predictions2
        print np_labels


if __name__ == '__main__':
    main()
