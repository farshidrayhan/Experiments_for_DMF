# -*- coding: utf-8 -*-
"""
Created on Thu May 18 21:46:18 2017

@author: Farshid
"""
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.externals import joblib
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from multiprocessing import Pool
import math
from math import factorial

from sklearn.metrics import roc_auc_score,average_precision_score

from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler


datasets = [
    'new-thyroid1.txt',
    'segment0.txt',
    'pima.txt',
    'yeast4.txt',
    'yeast5.txt',
    'yeast6.txt',
    'glass5.txt',
    'glass6.txt',
    'newthyroid2.txt',
    'yeast-2_vs_4.txt',
    'glass-0-1-2-3_vs_4-5-6.txt',
    'page-blocks-1-3_vs_4.txt',
    # 'ecoli-0-1-3-7_vs_2-6.txt',
    # 'led7digit-0-2-4-5-6-7-8-9_vs_1.txt',
    # 'shuttle-c2-vs-c4.txt',
    # 'yeast-1-2-8-9_vs_7.txt',
    # 'yeast-1-4-5-8_vs_7.txt',
    # 'yeast-2_vs_8.txt'





]
def clf(dataset):
    #

    print("dataset : ", dataset)
    df = pd.read_csv('/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/' + dataset, header=None)

    print('reading', dataset)
    df['label'] = df[df.shape[1] - 1]
    #
    df.drop([df.shape[1] - 2], axis=1, inplace=True)
    labelencoder = LabelEncoder()
    df['label'] = labelencoder.fit_transform(df['label'])
    #
    X = np.array(df.drop(['label'], axis=1))
    y = np.array(df['label'])

    # number_of_clusters = 23
    # sampler = RandomUnderSampler()
    normalization_object = Normalizer()
    X = normalization_object.fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    # n_classes = 2

    X_train = []
    X_test = []
    y_train = []
    y_test = []


    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        break
    print('training', dataset)
    top_roc = 0

    depth_for_rus = 0
    split_for_rus = 0

    for c in range(1, 200,5):
        for split in np.arange(.1, 2, .5):
            classifier = AdaBoostClassifier(
                svm.SVC(C=c ,gamma=split,probability=True),
                n_estimators=10,
                learning_rate=1, algorithm='SAMME')

            # X_train, y_train = sampler.fit_sample(X_train, y_train)

            classifier.fit(X_train, y_train)

            predictions = classifier.predict_proba(X_test)

            score = roc_auc_score(y_test, predictions[:, 1])
            if score > top_roc:
                top_roc = score

    print("dataset " , dataset , " roc " , top_roc)


if __name__ == '__main__':
    pool = Pool(8)
    results = pool.map(clf, datasets)