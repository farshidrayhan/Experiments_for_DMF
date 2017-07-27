"""
Created on Wed May 17 00:43:51 2017

@author: Farshid
"""

from multiprocessing import Pool
import math
from math import factorial

import os
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

datasets = ['nr_dataset_1477.txt', 'nr_dataset_1404.txt', 'nr_dataset_1294.txt', 'nr_dataset_1282.txt',
            'gpcr_dataset_1477.txt', 'gpcr_dataset_1404.txt', 'gpcr_dataset_1294.txt', 'gpcr_dataset_1282.txt',
            'enzyme_dataset_1477.txt', 'enzyme_dataset_1404.txt', 'enzyme_dataset_1294.txt', 'enzyme_dataset_1282.txt']

# datasets = ['nr_dataset_1477.txt', 'nr_dataset_1404.txt', 'nr_dataset_1294.txt', 'nr_dataset_1282.txt']


def create_model(dataset):
    print("dataset : ", dataset)
    df = pd.read_csv('/home/farshid/Desktop/' + dataset, header=None)

    df['label'] = df[df.shape[1] - 1]

    df.drop([df.shape[1] - 2], axis=1, inplace=True)

    labelencoder = LabelEncoder()
    df['label'] = labelencoder.fit_transform(df['label'])

    X = np.array(df.drop(['label'], axis=1))
    y = np.array(df['label'])

    normalization_object = Normalizer()
    X = normalization_object.fit_transform(X)

    # This part is for stratified cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    # This part is for Random Undersampling
    sampler = RandomUnderSampler()

    top_roc = 0
    for depth in range(2, 20):
        for split in range(2, 9):

            all_auc = []

            classifier = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=depth, min_samples_split=split),
                n_estimators=100,
                learning_rate=1, algorithm='SAMME')

            for train_index, test_index in skf.split(X, y):
                X_train = X[train_index]
                X_test = X[test_index]

                y_train = y[train_index]
                y_test = y[test_index]

                X_train, y_train = sampler.fit_sample(X_train, y_train)

                classifier.fit(X_train, y_train)

                predictions = classifier.predict_proba(X_test)

                all_auc.append(roc_auc_score(y_test, predictions[:, 1]))

            average_auc = sum(all_auc) / len(all_auc)
            # print("for depth", depth, " and split ", split, "roc = ", average_auc)
            if average_auc > top_roc:
                print(dataset, " for depth", depth, " and split ", split, "roc = ", average_auc, end=' ')
                joblib.dump(classifier, '/home/farshid/Desktop/classification_models/' + dataset + '.pkl')
                top_roc = average_auc
                print("stored !!!!")


def func(x):
    #    print('process id:', os.getpid() )

    print("for ", x, "fact started")
    # math.factorial(x)
    create_model(x)
    print("for ", x, "fact done")  # ,


if __name__ == '__main__':

    pool = Pool(12)
    results = pool.map(func, datasets)
