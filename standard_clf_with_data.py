"""
Created on Wed May 17 00:43:51 2017

@author: Farshid
"""

from multiprocessing import Pool
import math
from math import factorial

import os

from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

# datasets = ['nr_dataset_1477.txt', 'nr_dataset_1404.txt', 'nr_dataset_1294.txt', 'nr_dataset_1282.txt',
#             'gpcr_dataset_1477.txt', 'gpcr_dataset_1404.txt', 'gpcr_dataset_1294.txt', 'gpcr_dataset_1282.txt',
#             'enzyme_dataset_1477.txt', 'enzyme_dataset_1404.txt', 'enzyme_dataset_1294.txt', 'enzyme_dataset_1282.txt']
#
# datasets = ['nr_dataset_1477.txt', 'nr_dataset_1404.txt', 'nr_dataset_1294.txt', 'nr_dataset_1282.txt']
datasets = [
    # 'segment0.txt',
    # 'pima.txt',
    # 'abalone9-18.txt',
    # 'glass-0-1-2-3_vs_4-5-6.txt',
    # 'shuttle.txt',
    # 'yeast5.txt',
    # 'ecoli-0-3-4_vs_5.txt',
    # 'led7digit-0-2-4-5-6-7-8-9_vs_1.txt',
    # 'ecoli-0-1-4-7_vs_5-6.txt'

]

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
    for i in range(len(y_hat)):
        if y_hat[i]==1 and y_actual!=y_hat[i]:
           FP += 1
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==0:
           TN += 1
    for i in range(len(y_hat)):
        if y_hat[i]==0 and y_actual!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN


def create_model(dataset):
    print("dataset : ", dataset)
    df = pd.read_csv("/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/"+ dataset, header=None)

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
            all_aup = []

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
                all_aup.append(average_precision_score(y_test, predictions[:, 1]))

                predictions = classifier.predict(X_test)

                report  = classification_report(y_test,predictions)

            average_auc = sum(all_auc) / len(all_auc)
            average_aup = sum(all_aup) / len(all_aup)

            # print("for depth", depth, " and split ", split, "roc = ", average_auc)
            if average_auc > top_roc:
                print(dataset, "Ada for depth", depth, " and split ", split, " roc = ", average_auc," aupr = ", average_aup ,'report = '  )
                print(report)
                # joblib.dump(classifier, '/home/farshid/Desktop/classification_models/' + dataset + '.pkl')
                top_roc = average_auc
                # print("stored !!!!")
    top_roc =  0
    for depth in range(2, 20):
        for split in range(2, 9):

            all_auc = []
            all_aup = []

            classifier = RandomForestClassifier(n_estimators=50,max_depth=depth,min_samples_split=split)

            for train_index, test_index in skf.split(X, y):
                X_train = X[train_index]
                X_test = X[test_index]

                y_train = y[train_index]
                y_test = y[test_index]

                X_train, y_train = sampler.fit_sample(X_train, y_train)

                classifier.fit(X_train, y_train)

                predictions = classifier.predict_proba(X_test)

                all_auc.append(roc_auc_score(y_test, predictions[:, 1]))
                all_aup.append(average_precision_score(y_test, predictions[:, 1]))

                predictions = classifier.predict(X_test)

                report = classification_report(y_test, predictions)

            average_auc = sum(all_auc) / len(all_auc)
            average_aup = sum(all_aup) / len(all_aup)
            # print("for depth", depth, " and split ", split, "roc = ", average_auc)
            if average_auc > top_roc:
                print(dataset, "RF  for depth", depth, " and split ", split, " roc = ", average_auc, " aupr = ", average_aup, 'report = ')

                print(report)
                # joblib.dump(classifier, '/home/farshid/Desktop/classification_models/' + dataset + '.pkl')
                top_roc = average_auc
                # print("stored !!!!")


def func(x):
    #    print('process id:', os.getpid() )

    print("for ", x, "fact started")
    # math.factorial(x)
    create_model(x)
    print("for ", x, "fact done")  # ,


if __name__ == '__main__':

    pool = Pool(1)
    results = pool.map(func, datasets)
