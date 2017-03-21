"""David Donahue 2016. Tests functions and data output from humor_processing.py script."""

import cPickle as pickle

import numpy as np
import random

from config import DATA_DIR, HUMOR_MAX_WORDS_IN_TWEET, HUMOR_MAX_WORDS_IN_HASHTAG, GLOVE_EMB_SIZE, PHONETIC_EMB_SIZE
from config import HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR, HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR
from config import HUMOR_TRAIN_TWEET_PAIR_CHAR_DIR, HUMOR_TRIAL_TWEET_PAIR_CHAR_DIR
from config import SEMEVAL_HUMOR_TRAIN_DIR
from config import TWEET_PAIR_LABEL_RANDOM_SEED
from humor_processing import build_vocabulary
from humor_processing import create_dictionary_mapping
from humor_processing import look_up_glove_embeddings
from config import GLOVE_EMB_SIZE
from config import HUMOR_MAX_WORDS_IN_HASHTAG
from config import HUMOR_MAX_WORDS_IN_TWEET
from config import PHONETIC_EMB_SIZE
from tools import format_text_for_embedding_model
from tools import load_tweets_from_hashtag
from tools import extract_tweet_pairs_by_rank


def main():
    test_embedding_character_labels_match()
    test_print_words_without_gloves()
    test_build_vocabulary()
    test_format_text_with_hashtag()
    test_load_tweets_from_hashtag()
    test_saved_files()
    test_create_dictionary_mapping()
    test_extract_tweet_pairs()

    print 'Tests successful'


def test_embedding_character_labels_match():
    print 'TEST: test_embedding_character_labels_match'
    example_hashtag = 'Cat_History'
    np_emb_labels = np.load(open(HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR + example_hashtag + '_label.npy'))
    np_char_labels = np.load(open(HUMOR_TRAIN_TWEET_PAIR_CHAR_DIR + example_hashtag + '_labels.npy'))
    assert np.array_equal(np_emb_labels, np_char_labels)

    example_hashtag_trial = 'Gentler_Songs'
    np_emb_labels_trial = np.load(open(HUMOR_TRIAL_TWEET_PAIR_EMBEDDING_DIR + example_hashtag_trial + '_label.npy'))
    np_char_labels_trial = np.load(open(HUMOR_TRIAL_TWEET_PAIR_CHAR_DIR + example_hashtag_trial + '_labels.npy'))
    assert np.array_equal(np_emb_labels_trial, np_char_labels_trial)


def test_extract_tweet_pairs():
    print 'TEST: test_extract_tweet_pairs'
    """Check if extract_tweet_pairs function returns same pairs
    if given the same seed twice."""
    hashtag_name = 'America_In_4_Words'
    tweets, labels, tweet_ids = load_tweets_from_hashtag(SEMEVAL_HUMOR_TRAIN_DIR + hashtag_name + '.tsv',
                                                         explicit_hashtag='')
    random.seed(TWEET_PAIR_LABEL_RANDOM_SEED)
    tweet_pairs = extract_tweet_pairs_by_rank(tweets, labels, tweet_ids)
    tweet1 = [tweet_pair[0] for tweet_pair in tweet_pairs]
    tweet1_id = [tweet_pair[1] for tweet_pair in tweet_pairs]
    tweet2 = [tweet_pair[2] for tweet_pair in tweet_pairs]
    tweet2_id = [tweet_pair[3] for tweet_pair in tweet_pairs]
    labels = [tweet_pair[4] for tweet_pair in tweet_pairs]

    random.seed(TWEET_PAIR_LABEL_RANDOM_SEED)
    copy_tweet_pairs = extract_tweet_pairs_by_rank(tweets, labels, tweet_ids)
    copy_tweet1 = [tweet_pair[0] for tweet_pair in tweet_pairs]
    copy_tweet1_id = [tweet_pair[1] for tweet_pair in tweet_pairs]
    copy_tweet2 = [tweet_pair[2] for tweet_pair in tweet_pairs]
    copy_tweet2_id = [tweet_pair[3] for tweet_pair in tweet_pairs]
    copy_labels = [tweet_pair[4] for tweet_pair in tweet_pairs]

    assert tweet1 == copy_tweet1
    assert tweet1_id == copy_tweet1_id
    assert tweet2 == copy_tweet2
    assert tweet2_id == copy_tweet2_id
    assert labels == copy_labels




def test_print_words_without_gloves():
    print 'TEST: test_print_words_without_gloves'
    index_to_word = pickle.load(open(DATA_DIR + 'humor_index_to_word.cpkl', 'rb'))
    word_to_glove = pickle.load(open(DATA_DIR + 'humor_word_to_glove.cpkl', 'rb'))
    print len(index_to_word)
    print len(word_to_glove)
    counter = 0
    for word in index_to_word:
        if word not in word_to_glove:
            print word,
            counter += 1
        if counter > 10:
            print
            counter = 0


def test_create_dictionary_mapping():
    print 'TEST: test_create_dictionary_mapping'
    list1 = ['apple', 'banana', 'orange']
    list2 = ['one', 'two', 'three']
    mapping = create_dictionary_mapping(list1, list2)
    assert mapping['apple'] == 'one'
    assert mapping['banana'] == 'two'
    assert mapping['orange'] == 'three'


def test_saved_files():
    print 'TEST: test_saved_files'
    """Loads index_to_word and word_to_glove, and
    first tweet, second tweet, and winner label for
    one example hashtag. Checks first and second tweet
    have same dimensions. Checks that the winner label has
    same first dimension. Checks that index to word is
    larger size than word_to_glove, and all words in
    word_to_glove are occur once and exist in index_to_word. Test
    that hashtag"""
    example_hashtag = 'Bad_Monster_Movies'
    index_to_word = pickle.load(open(DATA_DIR + 'humor_index_to_word.cpkl', 'rb'))
    word_to_glove = pickle.load(open(DATA_DIR + 'humor_word_to_glove.cpkl', 'rb'))
    word_to_phone = pickle.load(open(DATA_DIR + 'humor_word_to_phonetic.cpkl', 'rb'))
    np_first_tweets = np.load(open(HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR + example_hashtag + '_first_tweet_glove.npy'))
    np_second_tweets = np.load(open(HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR + example_hashtag + '_second_tweet_glove.npy'))
    np_winner_labels = np.load(open(HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR + example_hashtag + '_label.npy'))
    np_hashtag = np.load(open(HUMOR_TRAIN_TWEET_PAIR_EMBEDDING_DIR + example_hashtag + '_hashtag.npy'))

    assert '420' in index_to_word
    assert 'celebs' in word_to_phone

    print np.sum(np_hashtag)
    first_word_glove = list(np_hashtag[400:600])
    first_word_phonetic = list(np_hashtag[600:800])
    print first_word_glove
    print len(first_word_glove)
    print len(first_word_phonetic)
    print first_word_phonetic

    print list(np_hashtag.shape)
    assert list(np_hashtag.shape) == [np_first_tweets.shape[0], (GLOVE_EMB_SIZE + PHONETIC_EMB_SIZE) * HUMOR_MAX_WORDS_IN_HASHTAG]
    assert np_first_tweets.shape == np_second_tweets.shape
    assert np_winner_labels.shape[0] == np_first_tweets.shape[0]
    assert len(index_to_word) >= len(word_to_glove)
    assert len(word_to_glove) > 0
    for key in word_to_glove:
        assert key in index_to_word

    word_emb_size = GLOVE_EMB_SIZE + PHONETIC_EMB_SIZE

    # Print out hashtag from embeddings
    glove_reconstructed_hashtag, phone_reconstructed_hashtag = reconstruct_text_from_gloves(np_hashtag, HUMOR_MAX_WORDS_IN_HASHTAG,
                                                         word_emb_size, word_to_glove, word_to_phone)
    num_tweets_to_reconstruct = 0
    for i in range(num_tweets_to_reconstruct):
        glove_reconstructed_first_tweet, phone_reconstructed_first_tweet = \
            reconstruct_text_from_gloves(np_first_tweets, HUMOR_MAX_WORDS_IN_TWEET,
                                         word_emb_size, word_to_glove,
                                         word_to_phone, tweet_index=i)
        print glove_reconstructed_first_tweet + ' | ' + phone_reconstructed_first_tweet

    print glove_reconstructed_hashtag
    assert glove_reconstructed_hashtag == 'bad monster movies _ _ _ _ _'


def reconstruct_text_from_gloves(np_text, max_len_text, word_emb_size, word_to_glove, word_to_phone, tweet_index=0):
    reconstructed_tokens_glove = []
    reconstructed_tokens_phone = []
    for i in range(max_len_text):
        np_glove_emb = np_text[tweet_index, i*word_emb_size:i*word_emb_size + GLOVE_EMB_SIZE]
        np_phone_emb = np_text[tweet_index, i*word_emb_size + GLOVE_EMB_SIZE:i * word_emb_size + GLOVE_EMB_SIZE + PHONETIC_EMB_SIZE]
        glove_embedding_exists = False
        phone_embedding_exists = False
        for word in word_to_glove:
            if np.array_equal(np_glove_emb, np.array(word_to_glove[word])):
                reconstructed_tokens_glove.append(word)
                glove_embedding_exists = True
                break
        for word in word_to_phone:
            if np.array_equal(np_phone_emb, np.array(word_to_phone[word])):
                reconstructed_tokens_phone.append(word)
                phone_embedding_exists = True
                break
        if not glove_embedding_exists:
            reconstructed_tokens_glove.append('_')
        if not phone_embedding_exists:
            reconstructed_tokens_phone.append('_')
    glove_reconstructed_text = ' '.join(reconstructed_tokens_glove)
    phone_reconstructed_text = ' '.join(reconstructed_tokens_phone)
    return glove_reconstructed_text, phone_reconstructed_text


def test_load_tweets_from_hashtag():
    print 'TEST: test_load_tweets_from_hashtag'
    tweets, labels, tweet_ids = load_tweets_from_hashtag('./test_hashtag_file.txt')
    print tweets[0]
    print tweets[1]
    assert tweets[0] == 'waste hard disk 2016 # this is a text file containing a single hashtag'
    assert labels[0] == 1
    assert tweets[1] == 'ldsfjlkajdf hasldjasfdlk aldksjfaldksfj # so many hashtags'
    assert labels[1] == 0
    assert tweets[2] == "yo lo # this is an apostrophe test of davids"
    assert labels[2] == 1


def test_build_vocabulary():
    print 'TEST: test_build_vocabulary'
    text1 = 'Hello hello world man'
    text2 = 'Bird man hello bird'
    vocabulary = build_vocabulary([text1, text2])
    words = ['hello', 'world', 'man', 'bird']
    for word in words:
        num_of_word = 0
        for entry in vocabulary:
            if word == entry:
                num_of_word += 1
        assert num_of_word == 1


def test_format_text_with_hashtag():
    print 'TEST: test_format_text_with_hashtag'
    text = 'The methlamine must not stop flowing. #2014TheBreakingBad show'
    formatted_text = format_text_for_embedding_model(text)
    print formatted_text
    print formatted_text.split()
    assert formatted_text == '2014 the breaking bad # the methlamine must not stop flowing show'


if __name__ == '__main__':
    main()