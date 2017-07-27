# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:54:20 2017

@author: Sajid
"""

import numpy as np

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour, NeighbourhoodCleaningRule
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.weight_boosting import _samme_proba
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.kd_tree import KDTree
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.tree import ExtraTreeClassifier


class AdaBoost:
    def __init__(self, n_estimators, depth, split, neighbours):
        self.M = n_estimators
        self.depth = depth
        self.split = split
        # self.undersampler = RandomUnderSampler(return_indices=True,replacement=False,ratio=.09)
        # self.undersampler = CondensedNearestNeighbour(return_indices=True,n_neighbors=neighbours)
        # self.undersampler = RepeatedEditedNearestNeighbours(return_indices=True,n_neighbors=neighbours)
        # self.undersampler = NeighbourhoodCleaningRule(return_indices=True,n_neighbors=neighbours)
        # self.undersampler = self.customSampler()
        # self.undersampler = EditedNearestNeighbours(return_indices=True,n_neighbors=neighbours)
        # self.undersampler = ClusterCentroids(ratio='auto' , n_jobs= 4 )
        # self.undersampler = AllKNN(return_indices=True,n_neighbors=neighbours,n_jobs=4)

    def customSampler(self, X, Y):
        neighbours = 3
        sampler = RandomUnderSampler(return_indices=True, replacement=False, ratio=.9)
        # sampler = CondensedNearestNeighbour(return_indices=True,n_neighbors=neighbours)
        # sampler = RepeatedEditedNearestNeighbours(return_indices=True,n_neighbors=neighbours)
        # sampler = NeighbourhoodCleaningRule(return_indices=True,n_neighbors=2)
        # sampler = EditedNearestNeighbours(return_indices=True,n_neighbors=neighbours)
        # sampler = SMOTE()
        # sampler = ClusterCentroids(ratio='auto' , n_jobs= 4 )
        # sampler = AllKNN(return_indices=True,n_neighbors=neighbours,n_jobs=4)

        index = []
        for i in range(len(X)):
            index.append(i)

        # print("before x " , len(X)  , ' y ', len(Y) )
        a, b, c = sampler.fit_sample(X, Y)

        # print("after x " , len(a)  , ' y ', len(b) , ' c ' , len(c))
        # print(c)

        # return a,b,c
        return X, Y, index
    #
    def fit(self, X, Y):
        self.models = []
        self.alphas = []

        best_alpha = []
        best_tree = []

        top_score = 0

        N, _ = X.shape
        W = np.ones(N) / N

        for m in range(self.M):
            d = int(np.random.uniform(low=2, high=12))
            s = int(np.random.uniform(low=2, high=18))
            e = int(np.random.uniform(low=5, high=20))

            if m %2 == 0:
            #     tree = DecisionTreeClassifier(max_depth=self.depth, min_samples_split=self.split)
            # tree = DecisionTreeClassifier(max_depth=self.depth, min_samples_split=self.split)
                tree = DecisionTreeClassifier(max_depth=self.depth, min_samples_split=self.split)
            # tree = RandomForestClassifier(n_estimators=e,max_depth=d,min_samples_split=s)
            # tree = KNeighborsClassifier(n_neighbors=3)
            else:
            #     tree = ExtraTreeClassifier(max_depth=self.depth, min_samples_split=self.split)
                tree = ExtraTreeClassifier(max_depth=self.depth, min_samples_split=self.split)
            # tree = ExtraTreeClassifier(max_depth=self.depth, min_samples_split=self.split)
            #     tree = RandomForestClassifier(n_estimators=e,max_depth=d,min_samples_split=s)
            # tree = AdaBoostClassifier(
            # DecisionTreeClassifier(max_depth=d, min_samples_split=s),n_estimators=e,learning_rate=1, algorithm='SAMME')
            # tree = RandomForestClassifier(n_estimators=50, max_depth=d, min_samples_split=s)

            X_undersampled, y_undersampled, chosen_indices = self.customSampler(X, Y)

            # chosen_indices = []

            tree.fit(X_undersampled, y_undersampled, sample_weight=W[chosen_indices])

            P = tree.predict(X)

            err = np.sum(W[P != Y])

            if err > 0.5:
                m = m - 1
            if err <= 0:
                err = 0.00000001
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
                    W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1

                self.models.append(tree)
                self.alphas.append(alpha)

                FX = np.zeros(N)
                for alpha, tree in zip(self.alphas, self.models):
                    FX += alpha * tree.predict(X)
                FX = np.sign(FX)
                score = roc_auc_score(Y, FX)
                # print("for m = " , m , " roc " , score)
                if top_score < score:
                    # print("for estimators train score = ", m, " roc ", score)
                    top_score = score
                    best_alpha = self.alphas
                    best_tree = self.models

        # print("assigning best models")

        self.alphas = best_alpha
        self.models = best_tree


    def predict(self, X):
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            FX +=  alpha* tree.predict(X)
        # print('FX = ' , FX)
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
