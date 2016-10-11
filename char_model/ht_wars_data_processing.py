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

output_dir = './processed_ht_wars_data/'
dataset_path = './train_dir/train_data/'

def main():
    print "Processing #HashtagWars data..."
    hashtags = get_hashtag_file_names()
    char_to_index = build_character_vocabulary(hashtags)
    for hashtag in hashtags:
        data = extract_hashtag_features_from_dataset(hashtag)
        save_hashtag_data(data, hashtag)
    print "Done!"
    
def build_character_vocabulary(hashtags):
    '''Find all characters special or alphabetical that appear in the dataset.
    Construct a vocabulary that assigns a unique index to each character and
    return that vocabulary.'''
    vocabulary = []
    for hashtag in hashtags:
        with open(dataset_path + hashtag + '.tsv') as tsv:
            for line in csv.reader(tsv, dialect='excel-tab'):
                for character in line[1]:
                    # If character hasn't been seen before, add it to the vocabulary.
                    if character not in vocabulary:
                        vocabulary.append(character)
    print vocabulary
    return vocabulary
    
def get_hashtag_file_names():
    '''Returns .tsv file name for each hashtag in the dataset.'''
    f = []
    for (dirpath, dirnames, filenames) in walk(dataset_path):
        f.extend(filenames)
        break
    g = [os.path.splitext(hashtag)[0] for hashtag in f]
    return g
    
def extract_hashtag_features_from_dataset(hashtag):
    '''For this hashtag, a numpy array of tweets vs characters will be constructed.
    The characters will be restricted to 28 possible symbols. These symbols are
    letters numbers 0 - 9 (1 - 10), A - Z (10 - 35), $ (36) and @ (37).
    '''
    pairs = []
    non_winners = []
    top_ten = []
    winner = []
    with open(dataset_path + hashtag + '.tsv') as tsv:
        for line in csv.reader(tsv, dialect='excel-tab'):
            tweet_rank = int(line[2])
            if tweet_rank == 0:
                non_winners.append(line[1])
            if tweet_rank == 1:
                top_ten.append(line[1])
            if tweet_rank == 2:
                winner.append(line[1])
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
    return pairs

def save_hashtag_data(data, hashtag):
    print 'Saving data for hashtag %s' % hashtag
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pickle.dump(data, open(output_dir + hashtag + '.cpkl', 'wb'))
    
### Unit Tests ###
def test_get_hashtag_file_names():
    file_names = get_hashtag_file_names()
    for name in file_names:
        assert '.tsv' not in name
    print len(file_names)
    assert len(file_names) == 101 #Number of hashtags in dataset
    
if __name__ == '__main__':
    main()