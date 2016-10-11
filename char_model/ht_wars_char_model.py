'''David Donahue 2016. This script is intended to run a model for the Semeval-2017 Task 6. It is
trained on the #HashtagWars dataset and is designed to predict which of a pair of tweets is funnier.
It is intended to reconstruct Alexey Romanov's model, which uses a character-by-character processing approach.'''

import numpy as np

tweet_pairs_path = './tweet_pairs_per_hashtag/'

def main():
    print 'Running #HashtagWars model...'
    tweet_pairs = load_all_tweet_pairs(tweet_pairs_path)
    funnier_tweets = 
    print 'Done!'   
    
def load_all_tweet_pairs(tweet_pairs_path):
    return np.random.choice(138, size=[145])
    
class HashtagWarsCharacterModel:
    def __init__(self):
        print 'Initializing #HashtagWars character model'
    def predict(self, tweet_pairs):
        return np.random.choice(1, size=[len(tweet_pairs)])
    
if __name__ == '__main__':
    main()