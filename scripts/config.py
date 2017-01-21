TWITTER_CONSUMER_KEY = ''
TWITTER_CONSUMER_SECRET = ''
TWITTER_ACCESS_TOKEN = ''
TWITTER_ACCESS_TOKEN_SECRET = ''

DATA_DIR = '.'
WORD_VECTORS_FILENAME = ''

try:
    from config_local import *
except ImportError:
    pass
