from __future__ import print_function

import os
import sys
import itertools

import numpy as np

class UnsupervisedExpRunner(object):
    def __init__(self, compare_labels, measure=None, *args, **kwargs):
        super(UnsupervisedExpRunner, self).__init__(*args, **kwargs)

        self.compare_labels = compare_labels

        if measure is None:
            measure = lambda a, b : np.linalg.norm(a) > np.linalg.norm(b)

        self.measure = measure

        self.results = {}

    def group_samples(self, X, y):
        labels = np.unique(y)

        # convert X into a 2-dim array (nb_samples, nb_features)
        # (nb_samples,) -> (nb_samples, 1)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        X_grouped = {l:[] for i, l in enumerate(labels)}
        nb_samples = X.shape[0]
        for i in range(nb_samples):
            X_grouped[y[i]].append(X[i, :])

        return X_grouped

    def run_subexp(self, data, y1, y2, funnier=None):
        X1 = data[y1]
        X2 = data[y2]

        if funnier is None:
            funnier = y1

        correct = 0
        incorrect = 0
        for pair in itertools.product(X1, X2):
            if self.measure(pair[0], pair[1]) and funnier == y1:
                correct += 1
            else:
                incorrect += 1

        return correct, correct + incorrect

    def run_exp(self, hashtag, X, y):
        data_grouped = self.group_samples(X, y)

        total_correct = 0
        total_all = 0

        print('Hashtag:', hashtag)
        for lc in self.compare_labels:
            correct, all = self.run_subexp(data_grouped, lc[0], lc[1])
            total_correct += correct
            total_all += all

            print('Labels:', lc, 'Accuracy:', round(float(correct) / all, 2))
        print()

        self.results[hashtag] = round(float(total_correct) / total_all, 2)

