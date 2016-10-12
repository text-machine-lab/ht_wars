'''David Donahue 2016. This script is intended to run a model for the Semeval-2017 Task 6. It is
trained on the #HashtagWars dataset and is designed to predict which of a pair of tweets is funnier.
It is intended to reconstruct Alexey Romanov's model, which uses a character-by-character processing approach.'''

import numpy as np
import tensorflow as tf

tweet_pairs_path = './tweet_pairs_per_hashtag/'

def main():
    print 'Running #HashtagWars model...'
    tweet_pairs, tweet_labels = load_all_tweet_pairs_and_labels(tweet_pairs_path)
    model = HashtagWarsCharacterModel()
    tweet_label_predictions = model.predict(tweet_pairs)
    print_model_performance_statistics(tweet_labels, tweet_label_predictions)
    print 'Done!'   
    
def print_model_performance_statistics(tweet_labels, tweet_label_predictions):
    correct_predictions = np.equal(tweet_labels, tweet_label_predictions)
    accuracy = np.mean(correct_predictions)
    print('Model test accuracy: %s' % accuracy)
    
def load_all_tweet_pairs_and_labels(tweet_pairs_path):
    m = 10000 # Pretend number of tweet pairs.
    return np.random.choice(138, size=[m, 300]), np.random.choice(2, size=[m])
    
class HashtagWarsCharacterModel:
    def __init__(self):
        print 'Initializing #HashtagWars character model'
    def predict(self, tweet_pairs):
        return np.random.choice(1, size=[len(tweet_pairs)])
    
if __name__ == '__main__':
    main()