import os

DATA_DIR = '../data/'

try:
    from config_local import *
except ImportError:
    pass

# Random seed to synchronize tweet pair creation between models
TWEET_PAIR_LABEL_RANDOM_SEED = 'hello world'

# GloVe embedding dataset path
WORD_VECTORS_FILE_PATH = os.path.join(DATA_DIR, 'glove.twitter.27B/glove.twitter.27B.200d.txt')

# Main #HashtagWars dataset paths
SEMEVAL_HUMOR_TRAIN_DIR = os.path.join(DATA_DIR, 'train_dir/train_data/')
SEMEVAL_HUMOR_TRIAL_DIR = os.path.join(DATA_DIR, 'trial_dir/trial_data/')

# Character-to-phoneme model paths
CMU_SYMBOLS_FILE_PATH = os.path.join(DATA_DIR, 'cmudict-0.7b.symbols.txt')
CMU_DICTIONARY_FILE_PATH = os.path.join(DATA_DIR, 'cmudict-0.7b.txt')

CMU_CHAR_TO_INDEX_FILE_PATH = os.path.join(DATA_DIR, 'cmu_char_to_index.cpkl')
CMU_PHONE_TO_INDEX_FILE_PATH = os.path.join(DATA_DIR, 'cmu_phone_to_index.cpkl')
CMU_NP_WORDS_FILE_PATH = os.path.join(DATA_DIR, 'cmu_words.npy')
CMU_NP_PRONUNCIATIONS_FILE_PATH = os.path.join(DATA_DIR, 'cmu_pronunciations.npy')

CHAR_2_PHONE_MODEL_DIR = os.path.join(DATA_DIR, 'char_2_phone_models/')

# Embedding humor model paths
EMBEDDING_HUMOR_MODEL_DIR = os.path.join(DATA_DIR, 'embedding_humor_models/')

HUMOR_TRAIN_TWEET_PAIR_CHAR_DIR = os.path.join(DATA_DIR, 'train_numpy_tweet_pairs/')
HUMOR_TRIAL_TWEET_PAIR_CHAR_DIR = os.path.join(DATA_DIR, 'trial_numpy_tweet_pairs/')
HUMOR_CHAR_TO_INDEX_FILE_PATH = os.path.join(DATA_DIR, 'humor_char_to_index.cpkl')
HUMOR_INDEX_TO_WORD_FILE_PATH = os.path.join(DATA_DIR, 'humor_index_to_word.cpkl')
HUMOR_WORD_TO_GLOVE_FILE_PATH = os.path.join(DATA_DIR, 'humor_word_to_glove.cpkl')
HUMOR_WORD_TO_PHONETIC_FILE_PATH = os.path.join(DATA_DIR, 'humor_word_to_phonetic.cpkl')

HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR = os.path.join(DATA_DIR, 'training_tweet_pair_embeddings/')
HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR = os.path.join(DATA_DIR, 'trial_tweet_pair_embeddings/')

# Boost tree humor model paths
BOOST_TREE_TWEET_PAIR_TRAINING_DIR = os.path.join(DATA_DIR, 'training_tweet_pair_tree_data/')
BOOST_TREE_TWEET_PAIR_TESTING_DIR = os.path.join(DATA_DIR, 'testing_tweet_pair_tree_data/')
HUMOR_MAX_WORDS_IN_TWEET = 20  # All winning tweets are under 30 words long
HUMOR_MAX_WORDS_IN_HASHTAG = 8
GLOVE_EMB_SIZE = 200
TWEET_SIZE = 140
PHONETIC_EMB_SIZE = 200

# Mongo for Sacred's observer
MONGO_ADDRESS = '127.0.0.1:27018'
