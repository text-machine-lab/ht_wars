import os

DATA_DIR = './data/'

try:
    from config_local import *
except ImportError:
    pass


WORD_VECTORS_FILE_PATH = os.path.join(DATA_DIR, 'glove.twitter.27B/glove.twitter.27B.200d.txt')
