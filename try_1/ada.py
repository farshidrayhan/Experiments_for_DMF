# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:36:52 2017

@author: Farshid
"""
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from imblearn.over_sampling.random_over_sampler import RandomOverSampler
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, precision_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

df = pd.read_csv("/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/ecoli-0-3-4_vs_5.txt" , header=None)
df['label'] = df[df.shape[1] - 1]
df.drop([df.shape[1] - 2], axis=1, inplace=True)
labelencoder = LabelEncoder()
df['label'] = labelencoder.fit_transform(df['label'])
X = np.array(df.drop(['label'], axis=1))
print("Number of feature  = ", len(X[0]))
print("Number of Instances  = ", len(X))

y = np.array(df['label'])

normalization_object = Normalizer()
# X = labelencoder.fit_transform(X)
X = normalization_object.fit_transform(X)

number_of_estiator = 10

train, test = train_test_split(df, test_size=0.2)
X_train, Y_train = train.ix[:, :-1], train.ix[:, -1]
X_test, Y_test = test.ix[:, :-1], test.ix[:, -1]

w = np.ones(len(X_train)) / len(X_train)


for m in range(1,number_of_estiator):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, Y_train, sample_weight=w)
    pred_train_i = clf.predict(X_train)
    miss = [int(x) for x in (pred_train_i != Y_train)]
    # Equivalent with 1/-1 to update weights
    miss2 = [x if x == 1 else -1 for x in miss]
    # Error
    err_m = np.dot(w, miss) / sum(w)
    # Alpha
    alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
    # New weights
    w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))