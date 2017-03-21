"""David Donahue 2016. This script is intended to run a model for the Semeval-2017 Task 6. It is
trained on the #HashtagWars dataset and is designed to predict which of a pair of tweets is funnier.
It is intended to modify Alexey Romanov's model, which uses a character-by-character processing approach
with CNN. This model will implement an LSTM instead of a CNN."""

# Many unused imports.
import numpy as np
import tensorflow as tf
import sys
from os import walk
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge, Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras import backend as K
import cPickle as pickle
from config import HUMOR_TRAIN_TWEET_PAIR_CHAR_DIR
from config import HUMOR_CHAR_TO_INDEX_FILE_PATH
import gc
from tools import load_hashtag_data_and_vocabulary
import resource

tf.logging.set_verbosity(tf.logging.ERROR)


TWEET_SIZE = 140


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
    """Load all tweet pairs per all hashtags. Per hashtag, train on all other hashtags, test on current hashtag.
    Print out micro-accuracy each iteration and print out overall accuracy after."""
    # Load training tweet pairs/labels, and train on them.
    print print_memory_usage()
    hashtag_datas, char_to_index, vocab_size = load_hashtag_data_and_vocabulary(HUMOR_TRAIN_TWEET_PAIR_CHAR_DIR, HUMOR_CHAR_TO_INDEX_FILE_PATH)
    print print_memory_usage()
#     all_tweet_pairs = np.concatenate([hashtag_datas[i][1] for i in range(len(hashtag_datas))])
#     all_tweet_labels = np.concatenate([hashtag_datas[i][0] for i in range(len(hashtag_datas))])
    accuracies = []
    print
    for i in range(len(hashtag_datas)):
        # Train on all hashtags but one, test on one
        print 'Epoch beginning ',
        print_memory_usage()
        ht_model = HashtagWarsCharacterModel(TWEET_SIZE, vocab_size)
        print 'Model build ',
        print_memory_usage()
        hashtag_name, np_hashtag_tweet1, np_hashtag_tweet2, np_hashtag_tweet_labels, np_other_tweet1, np_other_tweet2, np_other_tweet_labels = extract_hashtag_data_for_leave_one_out(hashtag_datas, i)
        print 'Data splice ',
        print_memory_usage()

        print('Training model and testing on hashtag: %s' % hashtag_name)
        print('Shape of training tweet1 input: %s' % str(np_other_tweet1.shape))
        print('Shape of training tweet2 input: %s' % str(np_other_tweet2.shape))
        print('Shape of testing hashtag tweet1 input: %s' % str(np_hashtag_tweet1.shape))
        print('Shape of testing hashtag tweet2 input: %s' % str(np_hashtag_tweet2.shape))
        
        ht_model.train(np_other_tweet1, np_other_tweet2, np_other_tweet_labels)
        print 'Training ',
        print_memory_usage()
        accuracy = ht_model.predict(np_hashtag_tweet1, np_hashtag_tweet2, np_hashtag_tweet_labels)
        print 'Testing ',
        print_memory_usage()
        accuracies.append(accuracy)
        gc.collect()
    print 'Total Model Accuracy: %s' % np.mean(accuracies)
    print 'Done!'


def print_memory_usage():
    print 'Memory Usage: %s' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def test():
    print 'Done!'


# Created Python class for model. This was not necessary because the Keras.io
# interace is already organized this way.
class HashtagWarsCharacterModel:
    def __init__(self, tweet_size, vocab_size):
        self.tweet_size = tweet_size
        self.vocab_size = vocab_size
        self.model = 0 # Uninitialized
#         print 'Initializing #HashtagWars character model'
        
    def create_model(self):
        """Load two tweets, analyze them with convolution and predict which is funnier."""
        print 'Building model'
        # MODEL PARAMETERS
        num_filters_1 = 32
        num_filters_2 = 64
        filter_size_1 = 3
        filter_size_2 = 5
        dropout = 0.7
        fc1_dim = 75
        fc2_dim = 50
        lstm_out_dim = 100
        tweet_emb_dim = 50
        pool_length = 5
        
        print 'Vocabulary size: %s' % self.vocab_size
        # CHARACTER INPUT EMBEDDING.Two tweets as input. Run them through an embedding layer
        tweet1 = Input(shape=[self.tweet_size])
        tweet2 = Input(shape=[self.tweet_size])
        
        tweet_input_emb_lookup = Embedding(self.vocab_size, tweet_emb_dim, input_length=self.tweet_size)
        tweet1_emb = tweet_input_emb_lookup(tweet1)
        tweet2_emb = tweet_input_emb_lookup(tweet2) # (np_samples, tweet_size, emb_size)
        
        # PROCESS EMBEDDINGS WITH LSTM AND DENSE LAYER. Use LSTM here instead of CNN. 
        lstm = LSTM(lstm_out_dim, input_dim=[self.tweet_size, tweet_emb_dim], input_length=self.tweet_size, go_backwards=True)
        tweet1_lstm = lstm(tweet1_emb)
        tweet2_lstm = lstm(tweet2_emb)
        
        # Run both tweets separately through convolution layers, a max pool layer, 
        # and then flatten them for dense layer.
#         convolution_layer_1 = Convolution1D(num_filters_1, filter_size_1, input_shape=[self.tweet_size, self.vocab_size])
#         convolution_layer_2 = Convolution1D(num_filters_2, filter_size_2)
#         max_pool_layer = MaxPooling1D(stride=4)
#         flatten = Flatten()
#         tweet_conv_emb = Dense(fc1_dim, activation='relu')
#         
#         tweet_1_conv1 = max_pool_layer(convolution_layer_1(tweet1_emb))
#         tweet_2_conv1 = max_pool_layer(convolution_layer_1(tweet2_emb))
#         tweet1_conv2 = flatten(max_pool_layer(convolution_layer_2(tweet_1_conv1)))
#         tweet2_conv2 = flatten(max_pool_layer(convolution_layer_2(tweet_2_conv1)))

        tweet_lstm_emb = Dense(fc1_dim, activation='relu')
        tweet1_lstm_emb = tweet_lstm_emb(tweet1_lstm)
        tweet2_lstm_emb = tweet_lstm_emb(tweet2_lstm)
        
        # PROCESS TWEET PAIR TOGEHTER WITH DENSE LAYERS. Combine embeddings for each tweet as inputs to two dense layers.
        tweet_pair_emb = merge([tweet1_lstm_emb, tweet2_lstm_emb], mode='concat')
        tweet_pair_emb = Dropout(dropout)(tweet_pair_emb)
        dense_layer1 = Dense(fc2_dim, activation='relu')(tweet_pair_emb)
        output = Dense(1, activation='sigmoid')(dense_layer1)
        model = Model(input=[tweet1, tweet2], output=[output])
        return model

    def train(self, tweet1, tweet2, labels):
        """Construct humor model, then train it on batches of tweet pairs."""
        model = self.create_model()
        # Hold model for predictions later
        self.model = model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        batch_size=1000
        m = tweet1.shape[0]
        print 'Training #HashtagWars model...'
        num_batches = m / batch_size
        remaining_examples = m % batch_size
        print 'Number of training examples: %s' % (num_batches * batch_size)

        for i in range(num_batches):
            print('Trained on %s examples' % (i * batch_size))
            self.train_batch(model, tweet1, tweet2, labels, i * num_batches, i * num_batches + num_batches)
            if i % 10 == 0:
                print 'Batch training',
                print_memory_usage()
        if remaining_examples > 0:
            self.train_batch(model, tweet1, tweet2, labels, num_batches * batch_size, num_batches * batch_size + remaining_examples)

        print 'Finished training model'

    def predict(self, tweet1, tweet2, labels):
        """This function uses the pretrained model in this object to predict the funnier of tweet pairs
        tweet1 and tweet2, and calculates accuracy using the ground truth labels for each pair. The mean
        accuracy is returned."""
        m = tweet1.shape[0]
        model = self.model
        batch_size=5000
        accuracies = []
        num_batches = m / batch_size
        remaining_examples = m % batch_size
        # Predict for all batches, then predict for remaining examples that don't fit in a batch.
        for i in range(num_batches):
            accuracy = self.predict_batch(model, tweet1, tweet2, labels, batch_size * i, batch_size * i + batch_size)
            if np.isnan(accuracy):
                print 'Model produces NAN accuracy!'
            accuracies.append(accuracy)
        if remaining_examples > 0:
            accuracy = self.predict_batch(model, tweet1, tweet2, labels, num_batches * batch_size, num_batches * batch_size + remaining_examples)
            accuracies.append(accuracy)
        
        mean_accuracy = np.mean(accuracies)
        if np.isnan(mean_accuracy):
            print 'Model produces NAN mean accuracy!'
        print 'Accuracy: %s' % mean_accuracy
        print
        return mean_accuracy

    def predict_batch(self, model, tweet1, tweet2, labels, start_index, end_index):
        """Predict which tweet is funnier for all tweet1 and tweet2 from start_index to end_index. Returns accuracy of prediction.
        Warning: end_index represents the index after the batch has ended. This is the default access format for numpy arrays."""
        # Extract a batch of tweet pairs from total, convert to one-hot, and train.
        tweet1_batch = tweet1[start_index:end_index,:]
        
        tweet2_batch = tweet2[start_index:end_index,:]
        
        tweet_label_batch = labels[start_index:end_index]
        loss, accuracy = model.evaluate([tweet1_batch, tweet2_batch], tweet_label_batch, batch_size=tweet_label_batch.shape[0])
        return accuracy

    def train_batch(self, model, tweet1, tweet2, labels, start_index, end_index):
        tweet1_batch = tweet1[start_index:end_index,:]
        
        tweet2_batch = tweet2[start_index:end_index,:]
        
        tweet_labels_batch = labels[start_index:end_index]

        model.train_on_batch([tweet1_batch, tweet2_batch], [tweet_labels_batch])


def extract_hashtag_data_for_leave_one_out(hashtag_datas, i):
    """This function takes an index i representing a particular hashtag.
    The hashtag name is returned, along with tweet pair/label data for both that hashtag and all other
    hashtags combined. This corresponds with the leave-one-out methodology."""
    np_hashtag_tweet_pairs = hashtag_datas[i][1]
    np_hashtag_tweet_labels = hashtag_datas[i][2]
    hashtag_name = hashtag_datas[i][0]
    other_tweet_pairs = [hashtag_datas[j][1] for j in range(i)] + [hashtag_datas[k][1] for k in range(i+1,len(hashtag_datas))]
    other_tweet_labels = [hashtag_datas[j][2] for j in range(i)] + [hashtag_datas[k][2] for k in range(i+1,len(hashtag_datas))]
    np_other_tweet_pairs = np.concatenate(other_tweet_pairs)
    np_other_tweet_labels = np.concatenate(other_tweet_labels)
    
    np_hashtag_tweet1 = np_hashtag_tweet_pairs[:, :TWEET_SIZE]
    np_hashtag_tweet2 = np_hashtag_tweet_pairs[:, TWEET_SIZE:]
    np_other_tweet1 = np_other_tweet_pairs[:,:TWEET_SIZE]
    np_other_tweet2 = np_other_tweet_pairs[:, TWEET_SIZE:]
    
    return hashtag_name, np_hashtag_tweet1, np_hashtag_tweet2, np_hashtag_tweet_labels, np_other_tweet1, np_other_tweet2, np_other_tweet_labels
    

def convert_tweets_to_one_hot(tweets, vocab_size):
    """Converts all tweets (2d vector) into one-hot (3d vector)."""
    # Just converts to one-hot in a cheating way.
    tweets_one_hot = (np.arange(vocab_size) == tweets[:,:,None]-1).astype(int)
    return tweets_one_hot


def print_model_performance_statistics(tweet_labels, tweet_label_predictions):
    correct_predictions = np.equal(tweet_labels, tweet_label_predictions)
    accuracy = np.mean(correct_predictions)
    print('Model test accuracy: %s' % accuracy)

    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()