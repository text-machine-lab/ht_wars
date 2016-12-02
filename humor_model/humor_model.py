"""David Donahue 2016. This model will be trained and used to predict winning tweets
in the Semeval 2017 Humor Task. The goal of the model is to read tweet pairs and determine
which of the two tweets in a pair is funnier. For each word in a tweet, it will receive a
phonetic embedding and a GloVe embedding to describe how to pronounce the word and what the
word means, respectively. These will serve as features. Other features may be added over time.
This model will be built in Tensorflow."""
from config import HUMOR_TWEET_PAIR_DIR
from config import CHAR_TO_INDEX_FILE_PATH
from tools import load_hashtag_data_and_vocabulary


def main():
    dataset = load_hashtag_data_and_vocabulary(HUMOR_TWEET_PAIR_DIR, CHAR_TO_INDEX_FILE_PATH)
    model_params = build_humor_model()


def build_humor_model():
    print 'Building humor model'

    return [1]


if __name__ == '__main__':
    main()