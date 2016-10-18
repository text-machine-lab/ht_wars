'''David Donahue 2016. This script is intended to run a model for the Semeval-2017 Task 6. It is
trained on the #HashtagWars dataset and is designed to predict which of a pair of tweets is funnier.
It is intended to reconstruct Alexey Romanov's model, which uses a character-by-character processing approach.'''

# Many unused imports.
import numpy as np
import tensorflow as tf
import sys
from os import walk
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge, Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
import cPickle as pickle

tf.logging.set_verbosity(tf.logging.ERROR)

# Training and testing data pulled from these directories.
train_pairs_dir = './training_tweet_pairs/'
test_pairs_dir = './testing_tweet_pairs/'

tweet_size = 140

def main():
    # User must enter 'train' or 'test' for the program to execute successfully.
    if len(sys.argv) == 1:
        print 'No arguments provided'
        print 'Usage: python ht_wars_char_model.py [train/test]'
    elif sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
    else:
        print 'Invalid arguments provided'
        print 'Usage: python ht_wars_char_model.py [train/test]'

def train():
    # Load training tweet pairs/labels, and train on them.
    tweet_pairs, tweet_labels = load_tweet_pairs_and_labels(train_pairs_dir)
    vocab_size = np.max(tweet_pairs) + 1
    tweet1 = tweet_pairs[:,:tweet_size]
    tweet2 = tweet_pairs[:,tweet_size:]
    ht_wars_model = HashtagWarsCharacterModel(tweet_size=tweet_size, vocab_size=vocab_size)
    ht_wars_model.train(tweet1, tweet2, tweet_labels)
    
    # Load testing tweet pairs/labels, and test model accuracy.
    test_tweet_pairs, test_tweet_labels = load_tweet_pairs_and_labels(test_pairs_dir)
    test_tweet1 = test_tweet_pairs[:,:tweet_size]
    test_tweet2 = test_tweet_pairs[:,tweet_size:]
    ht_wars_model.predict(test_tweet1, test_tweet2, test_tweet_labels)
    print 'Done!'  
    
def test():
    print 'Done!'

# Created Python class for model. This was not necessary because the Keras.io
# interace is already organized this way.
class HashtagWarsCharacterModel:
    def __init__(self, tweet_size, vocab_size):
        self.tweet_size = tweet_size
        self.vocab_size = vocab_size
        self.model = 0 # Uninitialized
        print 'Initializing #HashtagWars character model'
        
    def create_model(self):
        '''Load two tweets, analyze them with convolution and predict which is funnier.'''
        print 'Building model'
        #Model parameters I can mess with:
        num_filters_1 = 32
        num_filters_2 = 64
        filter_size_1 = 3
        filter_size_2 = 5
        dropout = 0.7
        fc1_dim = 200
        fc2_dim = 50
        tweet_emb_dim = 50
        pool_length = 5
        
        print 'Vocabulary size: %s' % self.vocab_size
        # Two tweets as input. Run them through an embedding layer
        tweet1 = Input(shape=[self.tweet_size])
        tweet2 = Input(shape=[self.tweet_size])
        
        tweet_input_emb_lookup = Embedding(self.vocab_size, tweet_emb_dim, input_length=self.tweet_size)
        tweet1_emb = tweet_input_emb_lookup(tweet1)
        tweet2_emb = tweet_input_emb_lookup(tweet2)
        
        # Run both tweets separately through convolution layers, a max pool layer, 
        # and then flatten them for dense layer.
        convolution_layer_1 = Convolution1D(num_filters_1, filter_size_1, input_shape=[self.tweet_size, self.vocab_size])
        convolution_layer_2 = Convolution1D(num_filters_2, filter_size_2)
        max_pool_layer = MaxPooling1D(stride=4)
        flatten = Flatten()
        tweet_conv_emb = Dense(fc1_dim, activation='relu')
        
        tweet_1_conv1 = max_pool_layer(convolution_layer_1(tweet1_emb))
        tweet_2_conv1 = max_pool_layer(convolution_layer_1(tweet2_emb))
        tweet1_conv2 = flatten(max_pool_layer(convolution_layer_2(tweet_1_conv1)))
        tweet2_conv2 = flatten(max_pool_layer(convolution_layer_2(tweet_2_conv1)))
        tweet1_conv_emb = tweet_conv_emb(tweet1_conv2)
        tweet2_conv_emb = tweet_conv_emb(tweet2_conv2)
        
        # Combine embeddings for each tweet as inputs to two dense layers.
        tweet_pair_emb = merge([tweet1_conv_emb, tweet2_conv_emb], mode='concat')
        tweet_pair_emb = Dropout(dropout)(tweet_pair_emb)
        dense_layer1 = Dense(fc2_dim, activation='relu')(tweet_pair_emb)
        output = Dense(1, activation='sigmoid')(dense_layer1)
        model = Model(input=[tweet1, tweet2], output=[output])
        return model
        
    def train(self, tweet1, tweet2, labels):
        '''Construct humor model, then train it on batches of tweet pairs.'''
        model = self.create_model()
        # Hold model for predictions later
        self.model = model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        batch_size=100
        
        print 'Training #HashtagWars model...'
        num_batches = (len(tweet1)-batch_size) / batch_size + 1
        print 'Number of training examples: %s' % (num_batches * batch_size)
        for i in range(num_batches):
            if i % 100 == 0:
                print('Trained on %s examples' % (i * batch_size))
            # Extract a batch of tweet pairs from total, convert to one-hot, and train.
            tweet1_batch = tweet1[i*batch_size:i*batch_size+batch_size]
            
            tweet2_batch = tweet2[i*batch_size:i*batch_size+batch_size]
            
            tweet_labels_batch = labels[i*batch_size:i*batch_size+batch_size]

            model.train_on_batch([tweet1_batch, tweet2_batch], [tweet_labels_batch])
        print 'Finished training model'
    
    def predict(self, tweet1, tweet2, labels):
        ''''''
        model = self.model
        batch_size=1000
        accuracies = []
        num_batches = (len(tweet1)-batch_size) / batch_size + 1
        for i in range(num_batches):
            # Extract a batch of tweet pairs from total, convert to one-hot, and train.
            tweet1_batch = tweet1[i*batch_size:i*batch_size+batch_size]
            
            tweet2_batch = tweet2[i*batch_size:i*batch_size+batch_size]
            
            tweet_label_batch = labels[i*batch_size:i*batch_size+batch_size]
            loss, accuracy = model.evaluate([tweet1_batch, tweet2_batch], tweet_label_batch, batch_size=25)
            accuracies.append(accuracy)
            
        print 'Accuracy: %s' % np.mean(accuracies)
        
        return np.random.choice(1, size=[len(tweet1)])

def convert_tweets_to_one_hot(tweets, vocab_size):
    '''Converts all tweets (2d vector) into one-hot (3d vector).'''
    # Just converts to one-hot in a cheating way.
    tweets_one_hot = (np.arange(vocab_size) == tweets[:,:,None]-1).astype(int)
    return tweets_one_hot
    
def print_model_performance_statistics(tweet_labels, tweet_label_predictions):
    correct_predictions = np.equal(tweet_labels, tweet_label_predictions)
    accuracy = np.mean(correct_predictions)
    print('Model test accuracy: %s' % accuracy)
    
def load_tweet_pairs_and_labels(tweet_pairs_path):
    all_tweet_pairs = []
    all_tweet_labels = []
    for (dirpath, dirnames, filenames) in walk(tweet_pairs_path):
        for filename in filenames:
            if '_pairs.npy' in filename:
                tweet_pairs = np.load(tweet_pairs_path + filename)
                all_tweet_pairs.append(tweet_pairs)
            elif '_labels.npy' in filename:
                tweet_labels = np.load(tweet_pairs_path + filename)
                all_tweet_labels.append(tweet_labels)
    np_all_tweet_pairs = np.concatenate(all_tweet_pairs, axis=0)
    np_all_tweet_labels = np.concatenate(all_tweet_labels, axis=0)
    print 'Number of tweet pairs: %s' % np_all_tweet_pairs.shape[0]
    print 'Size of each tweet pair: %s' % np_all_tweet_pairs.shape[1]
    return np_all_tweet_pairs, np_all_tweet_labels
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()