"""David Donahue 2016. This script preprocesses the data for the humor model.
At the very least, the script is designed to prepare all tweets in the hashtag wars dataset.
It will load all tweet pairs for each hashtag. It will look up all glove embeddings for all tweet
pairs in the hashtag. It will generate pronunciation embeddings for all tweet pairs in the hashtag.
For each hashtag, it will generate and save a numpy array for each tweet. The numpy array will
be shaped m x n x e where m is the number of tweet pairs in the hashtag, n is the max tweet size and
e is the concatenated size of both the phonetic and glove embeddings."""
from tools import get_hashtag_file_names
from config import SEMEVAL_HUMOR_DIR
from config import WORD_VECTORS_FILE_PATH
from config import HUMOR_INDEX_TO_WORD_FILE_PATH
from config import HUMOR_WORD_TO_GLOVE_FILE_PATH
from tools import extract_tweet_pairs_from_file
from tools import HUMOR_MAX_WORDS_IN_TWEET
import sys
import csv
import nltk
import cPickle as pickle
import numpy as np





def main():
    print 'Starting program'
    if not len(sys.argv) > 1:
        print 'Please specify argument: vocabulary|tweet_pairs'

    elif sys.argv[1] == 'vocabulary':
        """For all hashtags, for the first and second tweet in all tweet pairs separately,
        for all words in the tweet, look up a glove embedding and generate a phonetic embedding."""
        hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_DIR)
        vocabulary = []
        for hashtag_name in hashtag_names:
            tweets, labels = load_tweets_from_hashtag(hashtag_name)
            vocabulary = build_vocabulary(tweets, vocabulary=vocabulary)
        word_to_glove = look_up_glove_embeddings(vocabulary)
        print 'Size of vocabulary: %s' % len(vocabulary)
        print 'Number of GloVe vectors found: %s' % len(word_to_glove)
        print 'Size of a GloVe vector: %s' % len(word_to_glove['#'])
        print 'Saving %s' % HUMOR_INDEX_TO_WORD_FILE_PATH
        pickle.dump(vocabulary, open(HUMOR_INDEX_TO_WORD_FILE_PATH, 'wb'))
        print 'Saving %s' % HUMOR_WORD_TO_GLOVE_FILE_PATH
        pickle.dump(word_to_glove, open(HUMOR_WORD_TO_GLOVE_FILE_PATH, 'wb'))

    elif sys.argv[1] == 'tweet_pairs':
        """Load vocabulary created from vocabulary step. Load tweets
        and create tweet pairs using winner/top10/loser labels. Split tweet pairs into
        left tweet, right tweet, and left-right-funnier label. Convert all left tweets
        and all right tweets into glove embeddings per word. Save numpy array for left tweets,
        right tweets, and labels."""
        vocabulary = pickle.load(open(HUMOR_INDEX_TO_WORD_FILE_PATH, 'rb'))
        word_to_glove = pickle.load(open(HUMOR_WORD_TO_GLOVE_FILE_PATH, 'rb'))
        hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_DIR)
        all_tweets = []
        all_labels = []
        for hashtag_name in hashtag_names:
            tweets, labels = load_tweets_from_hashtag(hashtag_name)
            all_tweets = all_tweets + tweets
            all_labels = all_labels + labels
            tweet_pairs = extract_tweet_pairs_from_file(SEMEVAL_HUMOR_DIR + hashtag_name + '.tsv')
            tweet1 = [tweet_pair[0] for tweet_pair in tweet_pairs]
            tweet2 = [tweet_pair[1] for tweet_pair in tweet_pairs]
            labels = [tweet_pair[2] for tweet_pair in tweet_pairs]
            np_label = np.array(labels)
            np_tweet1_gloves = convert_tweet_to_gloves(tweet1, word_to_glove, HUMOR_MAX_WORDS_IN_TWEET, 200)


def compute_most_words_in_winning_tweets(tweets, labels):
    winning_tweet_number_of_words = []
    for i in range(len(tweets)):
        if labels[i] == 2:  # Winning tweet
            tokens = tweets[i].split(' ')
            winning_tweet_number_of_words.append(len(tokens))
    return np.max(winning_tweet_number_of_words), np.mean(winning_tweet_number_of_words)


def convert_tweet_to_gloves(tweet, word_to_glove, max_number_of_words, glove_size):
    np_tweet_gloves = np.zeros([len(tweet), max_number_of_words * glove_size])
    for i in range(len(tweet)):
        tokens = tweet[i].split()
        for j in range(len(tokens)):
            if tokens[j] in word_to_glove:
                np_token_glove = np.array(word_to_glove[tokens[j]])
                for k in range(glove_size):
                    pass
                    # np_tweet_gloves[i, j*glove_size + k] = np_token_glove[k]

    return [0]


def look_up_glove_embeddings(index_to_word):
    print 'Looking up GloVe embeddings for words'
    word_to_glove = {}
    with open(WORD_VECTORS_FILE_PATH, 'rb') as f:
        for line in f:
            line_tokens = line.split()
            glove_word = line_tokens[0]
            if glove_word in index_to_word:
                glove_emb = [float(line_token) for line_token in line_tokens[1:]]
                word_to_glove[glove_word] = glove_emb
    return word_to_glove


def format_hashtag(hashtag_name):
    name_without_hashtag = hashtag_name.replace('#', '')
    hashtag_with_spaces = name_without_hashtag.replace('_', ' ')
    return hashtag_with_spaces.lower()


def load_tweets_from_hashtag(hashtag_name):
    print 'Loading tweet pairs for hashtag %s' % hashtag_name
    tweets = []
    labels = []
    # Open hashtag file line for line. File is tsv.
    # Tweet is second variable, tweet win/top10/lose status is third variable
    # Replace any Twitter hashtag with a '$'
    with open(SEMEVAL_HUMOR_DIR + hashtag_name + '.tsv', 'rb') as f:
        tsvread = csv.reader(f, delimiter='\t')
        for line in tsvread:
            tweet = line[1]
            formatted_tweet = format_text_with_hashtag(tweet)
            tweet_tokens = nltk.word_tokenize(formatted_tweet)
            tweets.append(' '.join(tweet_tokens).lower())
            labels.append(int(line[2]))

    return tweets, labels


def format_text_with_hashtag(text):
    """Can only handle one hashtag in text."""
    formatted_text = ''
    inside_hashtag = False
    for i in range(len(text)):
        if text[i] == '#':
            formatted_text = formatted_text + text[i]
            inside_hashtag = True
        elif inside_hashtag:
            if text[i] == ' ':
                inside_hashtag = False
                formatted_text = formatted_text + '#' + text[i]
            elif text[i].isupper() or not text[i].isalpha():
                formatted_text = formatted_text + ' ' + text[i]
            else:
                formatted_text = formatted_text + text[i]
        else:
            formatted_text = formatted_text + text[i]
    return formatted_text


def build_vocabulary(lines, vocabulary=[], max_word_size=15):
    print 'Building vocabulary'
    for line in lines:
        tokens = line.split()
        for word in tokens:
            if len(word) < max_word_size and word not in vocabulary:
                vocabulary.append(word)
    return vocabulary


if __name__ == '__main__':
    main()