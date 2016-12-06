"""David Donahue 2016. This script preprocesses the data for the humor model.
At the very least, the script is designed to prepare all tweets in the hashtag wars dataset.
It will load all tweet pairs for each hashtag. It will look up all glove embeddings for all tweet
pairs in the hashtag. It will generate pronunciation embeddings for all tweet pairs in the hashtag.
For each hashtag, it will generate and save a numpy array for each tweet. The numpy array will
be shaped m x n x e where m is the number of tweet pairs in the hashtag, n is the max tweet size and
e is the concatenated size of both the phonetic and glove embeddings."""
from tools import get_hashtag_file_names
from config import SEMEVAL_HUMOR_DIR


def main():
    print 'Starting program'
    """For all hashtags, for the first and second tweet in all tweet pairs separately,
    for all words in the tweet, look up a glove embedding and generate a phonetic embedding."""
    hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_DIR)
    for hashtag_name in hashtag_names:
        first_tweet_in_pair, second_tweet_in_pair = load_tweet_pairs_from_hashtag(hashtag_name)
        np_first_tweet_gloves = look_up_gloves_for_tweet(first_tweet_in_pair)
        np_second_tweet_gloves = look_up_gloves_for_tweet(second_tweet_in_pair)


def load_tweet_pairs_from_hashtag(hashtag_name):
    print 'Loading tweet pairs for hashtag %s' % hashtag_name
    return [0, 1]


def look_up_gloves_for_tweet(tweet):
    print 'Looking up gloves for tweet'
    return [2]


if __name__ == '__main__':
    main()