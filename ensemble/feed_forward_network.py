import math
import tempfile
import os

import numpy as np

from keras.layers.core import Dense
from keras.models import Sequential
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from _ctypes import ArgumentError


class NotFittedError(Exception):
    def __init__(self):
        super(NotFittedError, self).__init__('Call `fit` method first')


class FeedForwardNetwork():

    def __init__(self, input_dim, layer_size=50, num_layers=2, regularization=1e-3, checkpoint_filename=None):
        self.input_dim = input_dim

        self.layer_size = layer_size
        self.num_layers = num_layers
        self.regularization = regularization

        self.checkpoint_filename = checkpoint_filename

        self.model = None

    def __create_network(self):
        self.model = Sequential()

        # add the first layer with input dim
        self.model.add(Dense(input_dim=self.input_dim, output_dim=self.layer_size, activation='relu',
                             W_regularizer=l2(self.regularization), b_regularizer=l2(self.regularization)))

        # add hidden layers
        for i in range(self.num_layers):
            self.model.add(Dense(output_dim=self.layer_size, activation='relu',
                                 W_regularizer=l2(self.regularization), b_regularizer=l2(self.regularization)))

        self.model.add(Dense(1, activation='sigmoid',
                             W_regularizer=l2(self.regularization), b_regularizer=l2(self.regularization)))

    def fit(self, X, y, nb_epoch=5, batch_size=512, validation_data=None, use_the_best_model=False, verbose=1):
        """
        Train the NN

        Args:
            X: the set of sample with shape (nb_samples, nb_features)
            y: the vector of similarity score with shape (nb_samples,)
            nb_epoch: number of epoch to train
            batch_size: The size of the bacthes
            regularization: The strength of regularization
            validation_data: Tuple (X, y) to be used as held-out validation data
            use_the_best_model: If True, test the model on the validation data during the training and save the best model.
                                Load the best model after training will finished
                                You must provide validation data to use this feature
            verbose: The level of verbosity
        """

        # check validation data and use_the_best_model - see the description of use_the_best_model argument

        if use_the_best_model:
            if not validation_data:
                raise ArgumentError('You must provide validation data with `use_the_best_model=True`')

            if not self.checkpoint_filename:
                raise ArgumentError('You must provide the path for saving the model')

        # check that the dir for weights exists
        if self.checkpoint_filename is not None:
            checkpoint_dir = os.path.dirname(self.checkpoint_filename)
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)


        # create the model
        if not self.model:
            self.__create_network()

        self.model.compile(loss='binary_crossentropy', optimizer='adam')

        # create callback to save the best model during the training
        callbacks = []
        if self.checkpoint_filename is not None:
            checkpointer = ModelCheckpoint(filepath=self.checkpoint_filename, verbose=0, save_best_only=use_the_best_model)
            callbacks = [checkpointer]

        # fit the model
        self.model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose,
                       validation_data=validation_data, callbacks=callbacks)

        # load the best weights
        if use_the_best_model:
            self.model.load_weights(self.checkpoint_filename)


    def predict(self, X, batch_size=512, verbose=1):
        """
        Predict the score

        Args:
            X: the set of sample with shape (nb_samples, nb_features)

        Returs:
            Similarity score with shape (nb_samples,)
        """

        if not self.model:
            raise NotFittedError()

        y_predict = self.model.predict_classes(X, batch_size=batch_size, verbose=verbose)

        y_predict = np.reshape(y_predict, (-1))

        return y_predict

    def restore(self):
        # create the model
        if not self.model:
            self.__create_network()

        self.model.load_weights(self.checkpoint_filename)

