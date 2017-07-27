# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:54:20 2017

@author: Sajid
"""

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.ensemble.weight_boosting import _samme_proba


class AdaBoost:
    def __init__(self, M, depth):
        self.M = M
        self.depth = depth
        self.undersampler = RandomUnderSampler(return_indices=True)
        # self.undersampler = EditedNearestNeighbours(return_indices=True)

    def fit(self, X, Y):
        self.models = []
        self.alphas = []

        N, _ = X.shape
        W = np.ones(N) / N

        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=self.depth, splitter='best')

            X_undersampled, y_undersampled, chosen_indices = self.undersampler.fit_sample(X, Y)

            # this part for SMOTE
            # ==============================================================================
            #       len_before_sampling = len(y_undersampled)
            #
            #       X_undersampled , y_undersampled = self.oversampler.fit_sample(X,Y)
            #
            #       len_after_sampling = len(y_undersampled)
            #
            #       data_increased = len_after_sampling - len_before_sampling
            #
            #       new_weights = np.ones(data_increased) / data_increased
            #
            #       weights = np.concatenate((W[chosen_indices],new_weights),axis=0)
            # ==============================================================================

            tree.fit(X_undersampled, y_undersampled, sample_weight=W[chosen_indices])

            P = tree.predict(X)

            err = np.sum(W[P != Y])

            if err > 0.5:
                m = m - 1
            if err <= 0:
                err = 0.0000001
            else:

                try:
                    if (np.log(1 - err) - np.log(err)) == 0:
                        alpha = 0
                    else:
                        alpha = 0.5 * (np.log(1 - err) - np.log(err))
                    W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1
                except:
                    alpha = 0
                    # W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1

                self.models.append(tree)
                self.alphas.append(alpha)


    def predict(self, X):
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            FX += alpha * tree.predict(X)
        return np.sign(FX), FX

    def predict_proba(self, X):
        # if self.alphas == 'SAMME'
        proba = sum(tree.predict_proba(X) * alpha for tree , alpha in zip(self.models,self.alphas) )


        proba = np.array(proba)


        proba = proba / sum(self.alphas)

        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        # proba =  np.linspace(proba)
        # proba = np.array(proba).astype(float)
        proba = proba /  normalizer

        # print(proba)
        return proba
    def predict_proba2(self, X):
        # if self.alphas == 'SAMME.R'
        proba = sum(_samme_proba(est , 2 ,X) for est in self.models )

        proba = np.array(proba)

        proba = proba / sum(self.alphas)

        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        # proba =  np.linspace(proba)
        # proba = np.array(proba).astype(float)
        proba = proba / normalizer

        # print('proba = ',proba)
        return proba.astype(float)
