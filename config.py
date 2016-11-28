import os

DATA_DIR = '../data/'

try:
    from config_local import *
except ImportError:
    pass


WORD_VECTORS_FILE_PATH = os.path.join(DATA_DIR, 'glove.twitter.27B/glove.twitter.27B.200d.txt')
SEMEVAL_HUMOR_DIR = os.path.join(DATA_DIR, 'train_dir/train_data/')
CMU_SYMBOLS_FILE_PATH = os.path.join(DATA_DIR, 'cmudict-0.7b.symbols.txt')
CMU_DICTIONARY_FILE_PATH = os.path.join(DATA_DIR, 'cmudict-0.7b.txt')

