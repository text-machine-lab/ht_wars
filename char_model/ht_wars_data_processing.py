'''David Donahue 2016. This script is intended to process the #HashtagWars dataset into an appropriate format
for training a model. This script extracts tweets for all hashtags in the training data located in the current
directory (training_dir located in ./) and generates tweet pairs from them. Tweets are categorized as 2 (winning tweet),
1 (one of top-ten tweets) or 0 (did not make it into top ten). Pairs are generated as follows:

    1.) Matching winning tweet with each other tweet in the top ten
    2.) Matching each tweet in the top ten with each non-winning tweet
    
Each hashtag is saved as a list of tweet pairs. Each tweet pair is saved as two tweet texts and 0 or 1 for which is funnier.
Each tweet text is saved as a numpy vector of numbers.
'''
from os import walk
import csv
import cPickle as pickle
import os
import random
import numpy as np

output_dir = './tweet_pairs_per_hashtag/'
dataset_path = './train_dir/train_data/'

def main():
    # Find hashtags, create character vocabulary, extract tweet pairs and save everything.
    print "Processing #HashtagWars data..."
    hashtags = get_hashtag_file_names()
    char_to_index = build_character_vocabulary(hashtags)
    print('Size of character vocabulary: %s' % len(char_to_index))
    output_tweet_statistics(hashtags)
    print 'Extracting tweet pairs...'
    for hashtag in hashtags:
        data = extract_tweet_pairs_from_file(dataset_path + hashtag + '.tsv')
        np_tweet_pairs, np_tweet_pair_labels = format_tweet_pairs(data, char_to_index)
        save_hashtag_data(np_tweet_pairs, np_tweet_pair_labels, hashtag)
    print 'Saving char_to_index.cpkl containing character vocabulary'
    pickle.dump(char_to_index, open('char_to_index.cpkl', 'wb'))
    print "Done!"
    
def format_tweet_pairs(data, char_to_index, max_tweet_size=150):
    '''This script converts every character in all tweets into an index.
    It stores each tweet side by side, each tweet constrained to 150 characters long.
    The total matrix is m x 300, for m tweet pairs, two 150 word tweets per row.'''
    np_tweet_pairs = np.zeros(shape=[len(data), max_tweet_size * 2], dtype=int)
    np_tweet_pair_labels = np.zeros(shape=[len(data)], dtype=int)
    for pair_index in range(len(data)):
        first_tweet = data[pair_index][0]
        second_tweet = data[pair_index][1]
        np_tweet_pair_labels[pair_index] = data[pair_index][2] #First tweet more funny
        for i in range(len(first_tweet)):
            if i < max_tweet_size:
                character = first_tweet[i]
                if character in char_to_index:
                    np_tweet_pairs[pair_index][i] = char_to_index[character]
        for i in range(len(second_tweet)):
            if i < max_tweet_size:
                character = second_tweet[i]
                if character in char_to_index:
                    np_tweet_pairs[pair_index][i + max_tweet_size] = char_to_index[character]
            
    return np_tweet_pairs, np_tweet_pair_labels
    
def output_tweet_statistics(hashtags):
    '''This function analyzes the dataset and prints statistics for it.
    These statistics have to do with the number of tweets, the largest and average
    length of tweets for all tweets, top-ten tweets, and winning tweets.'''
    largest_tweet_length = 0
    largest_winning_tweet_length = 0
    number_of_tweets = 0
    number_of_top_ten_tweets = 0
    number_of_winning_tweets = 0
    tweet_length_sum = 0
    winning_tweet_length_sum = 0
    for hashtag in hashtags:
        with open(dataset_path + hashtag + '.tsv') as tsv:
            for line in csv.reader(tsv, dialect='excel-tab'):
                # Count number of tweets, find longest tweet, find average tweet length
                # for all tweets, top ten, and winning.
                tweet_length = len(line[1])
                tweet_rank = int(line[2])
                number_of_tweets += 1
                tweet_length_sum += tweet_length
                if tweet_length > largest_tweet_length:
                    largest_tweet_length = tweet_length
                if tweet_rank == 2:
                    if tweet_length > largest_winning_tweet_length:
                        largest_winning_tweet_length = tweet_length
                    winning_tweet_length_sum += tweet_length
                    number_of_winning_tweets += 1
                if tweet_rank == 1:
                    number_of_top_ten_tweets += 1
    average_tweet_length = (float(tweet_length_sum) / number_of_tweets)
    average_winning_tweet_length = (float(winning_tweet_length_sum) / number_of_winning_tweets)
    
    # Find standard deviation.
    tweet_std_dev_sum = 0
    winning_tweet_std_dev_sum = 0
    for hashtag in hashtags:
        with open(dataset_path + hashtag + '.tsv') as tsv:
            for line in csv.reader(tsv, dialect='excel-tab'):
                tweet_length = len(line[1])
                tweet_rank = int(line[2])
                tweet_std_dev_sum += abs(tweet_length - average_tweet_length)
                if tweet_rank == 2:
                    winning_tweet_std_dev_sum += abs(tweet_length - average_winning_tweet_length)
    tweet_std_dev = float(tweet_std_dev_sum) / number_of_tweets
    winning_tweet_std_dev = float(winning_tweet_std_dev_sum) / number_of_winning_tweets
    print 'The largest tweet length is %s characters' % largest_tweet_length
    print 'The largest winning tweet length is %s characters' % largest_winning_tweet_length
    print 'Number of tweets: %s' % number_of_tweets
    print 'Number of top-ten tweets: %s' % number_of_top_ten_tweets
    print 'Number of winning tweets: %s' % number_of_winning_tweets
    print 'Average tweet length: %s' % average_tweet_length
    print 'Average winning tweet length: %s' % average_winning_tweet_length
    print 'Tweet length standard deviation: %s' % tweet_std_dev
    print 'Winning tweet length standard deviation: %s' % winning_tweet_std_dev
    
def build_character_vocabulary(hashtags):
    '''Find all characters special or alphabetical that appear in the dataset.
    Construct a vocabulary that assigns a unique index to each character and
    return that vocabulary. Vocabulary does not include anything with a backslash.'''
    characters = []
    for hashtag in hashtags:
        with open(dataset_path + hashtag + '.tsv') as tsv:
            for line in csv.reader(tsv, dialect='excel-tab'):
                for char in line[1]:
                    # If character hasn't been seen before, add it to the vocabulary.
                    if char not in characters:
                        characters.append(char)
    # Create dictionary from list.
    vocabulary = {}
    for i in range(len(characters)):
        vocabulary[characters[i]] = i
    return vocabulary
    
def get_hashtag_file_names():
    '''Returns .tsv file name for each hashtag in the dataset.'''
    f = []
    for (dirpath, dirnames, filenames) in walk(dataset_path):
        f.extend(filenames)
        break
    g = [os.path.splitext(hashtag)[0] for hashtag in f]
    return g

def convert_tweet_to_indices(tweet, char_to_index):
    return []
    
def extract_tweet_pairs_from_file(hashtag_file):
    '''This script extracts tweet pairs from the file hashtag_file.
    It stores them in an array of tweet pairs, each tweet pair
    being a list of the form [tweet_1_text, tweet_2_text, first_tweet_funnier].
    first_tweet_funnier is 1 if the first tweet is funnier and 0 if the second
    tweet is funnier.
    '''
    pairs = []
    non_winners = []
    top_ten = []
    winner = []
    with open(hashtag_file) as tsv:
        for line in csv.reader(tsv, dialect='excel-tab'):
            tweet_rank = int(line[2])
            tweet_text = line[1]
            if tweet_rank == 0:
                non_winners.append(tweet_text)
            if tweet_rank == 1:
                top_ten.append(tweet_text)
            if tweet_rank == 2:
                winner.append(tweet_text)
    for non_winning_tweet in non_winners:
        for top_ten_tweet in winner + top_ten:
            #Create pair
            funnier_tweet_first = bool(random.getrandbits(1))
            if funnier_tweet_first:
                pairs.append([top_ten_tweet, non_winning_tweet, 0])
            else:
                pairs.append([non_winning_tweet, top_ten_tweet, 1])
    for top_ten_tweet in top_ten:
        for winning_tweet in winner:
            #Create pair
            funnier_tweet_first = bool(random.getrandbits(1))
            if funnier_tweet_first:
                pairs.append([winning_tweet, top_ten_tweet, 0])
            else:
                pairs.append([top_ten_tweet, winning_tweet, 1])
#     for i in range(1):
#         print pairs[i][0]
#         print pairs[i][1]
#         print pairs[i][2]
#         print
    return pairs

def save_hashtag_data(np_tweet_pairs, np_tweet_pair_labels, hashtag):
    print 'Saving data for hashtag %s' % hashtag
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_dir + hashtag + '.npy', np_tweet_pairs)
    np.save(output_dir + hashtag + '.npy', np_tweet_pair_labels)
    
### Unit Tests ###
def test_get_hashtag_file_names():
    file_names = get_hashtag_file_names()
    for name in file_names:
        assert '.tsv' not in name
    print len(file_names)
    assert len(file_names) == 101 #Number of hashtags in dataset
    
if __name__ == '__main__':
    main()