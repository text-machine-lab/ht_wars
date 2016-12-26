import os

DATA_DIR = '../data/'

try:
    from config_local import *
except ImportError:
    pass


WORD_VECTORS_FILE_PATH = os.path.join(DATA_DIR, 'glove.twitter.27B/glove.twitter.27B.200d.txt')

SEMEVAL_HUMOR_TRAINING_DIR = os.path.join(DATA_DIR, 'train_dir/train_data/')
SEMEVAL_HUMOR_TRIAL_DIR = os.path.join(DATA_DIR, 'trial_dir/trial_data/')

CMU_SYMBOLS_FILE_PATH = os.path.join(DATA_DIR, 'cmudict-0.7b.symbols.txt')
CMU_DICTIONARY_FILE_PATH = os.path.join(DATA_DIR, 'cmudict-0.7b.txt')

CMU_CHAR_TO_INDEX_FILE_PATH = os.path.join(DATA_DIR, 'cmu_char_to_index.cpkl')
CMU_PHONE_TO_INDEX_FILE_PATH = os.path.join(DATA_DIR, 'cmu_phone_to_index.cpkl')
CMU_NP_WORDS_FILE_PATH = os.path.join(DATA_DIR, 'cmu_words.npy')
CMU_NP_PRONUNCIATIONS_FILE_PATH = os.path.join(DATA_DIR, 'cmu_pronunciations.npy')

CHAR_2_PHONE_MODEL_DIR = os.path.join(DATA_DIR, 'char_2_phone_models/')
EMBEDDING_HUMOR_MODEL_DIR = os.path.join(DATA_DIR, 'embedding_humor_models/')

HUMOR_TWEET_PAIR_DIR = os.path.join(DATA_DIR, 'numpy_tweet_pairs/')
HUMOR_CHAR_TO_INDEX_FILE_PATH = os.path.join(DATA_DIR, 'humor_char_to_index.cpkl')
HUMOR_INDEX_TO_WORD_FILE_PATH = os.path.join(DATA_DIR, 'humor_index_to_word.cpkl')
HUMOR_WORD_TO_GLOVE_FILE_PATH = os.path.join(DATA_DIR, 'humor_word_to_glove.cpkl')
HUMOR_WORD_TO_PHONETIC_FILE_PATH = os.path.join(DATA_DIR, 'humor_word_to_phonetic.cpkl')  # phonetic pronunciation embedding

HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR = os.path.join(DATA_DIR, 'training_tweet_pair_embeddings/')
HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR = os.path.join(DATA_DIR, 'trial_tweet_pair_embeddings/')

