# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:36:52 2017

@author: Sajid
"""

import pandas as pd
import numpy as np
import sklearn
from imblearn.under_sampling import RandomUnderSampler

from sklearn.cluster import KMeans

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from imblearn.over_sampling.random_over_sampler import RandomOverSampler
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# df = pd.read_csv('F:\\CMR sir thesis\\ecoli\\ecoli.dat',skiprows=12,header=None)
#
# df = pd.read_csv('F:\\CMR sir thesis\\pima\\pima.dat',skiprows=13,header=None)
#
# df = pd.read_csv('F:\\CMR sir thesis\\banana\\banana.dat',skiprows=7,header=None)

# just set the path to the txt file here
dataset = '/home/farshid/Desktop/nr_dataset_1294.txt'

print("Dataset ", dataset)

df = pd.read_csv(dataset, header=None)

df['label'] = df[df.shape[1] - 1]

df.drop([df.shape[1] - 2], axis=1, inplace=True)

labelencoder = LabelEncoder()

df['label'] = labelencoder.fit_transform(df['label'])

X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])

normalization_object = Normalizer()
X = normalization_object.fit_transform(X)

number_of_folds = 7

best_clustered_trees_average_auc = 0
best_one_tree_average_auc = 0
best_cluster = 0

skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True)

sampler = RandomUnderSampler()

fold_counTer = 0
number_of_clusters = 23     # this is a hyper parameter
trees = {}
all_auc_with_clustered_trees = []
all_auc_with_one_tree = []
X_train_major = np.zeros((0, 1294))
y_train_major = []

avg_roc = 0
avg_aupr = 0

for train_index, test_index in skf.split(X, y):
    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y[train_index]
    y_test = y[test_index]

    major_class = max(sampler.fit(X_train, y_train).stats_c_, key=sampler.fit(X_train, y_train).stats_c_.get)

    major_class_X_train = []
    major_class_y_train = []
    minor_class_X_train = []
    minor_class_y_train = []

    for index in range(len(X_train)):
        if y_train[index] == major_class:
            major_class_X_train.append(X_train[index])
            major_class_y_train.append(y_train[index])
        else:
            minor_class_X_train.append(X_train[index])
            minor_class_y_train.append(y_train[index])

    # optimize for number of clusters here
    kmeans = KMeans(max_iter=200, n_jobs=4, n_clusters=number_of_clusters)
    kmeans.fit(major_class_X_train)

    # get the centroids of each of the clusters
    cluster_centroids = kmeans.cluster_centers_

    # get the points under each cluster
    points_under_each_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

    for i in range(number_of_clusters):
        size = len(points_under_each_cluster[i])
        random_indexes = np.random.randint(low=0, high=size, size=int(size / 2))        # this 2 means we are taking 50% of the total
        temp = points_under_each_cluster[i]
        feature_indexes = temp[random_indexes]
        X_train_major = np.concatenate((X_train_major, X_train[feature_indexes]), axis=0)
        y_train_major = np.concatenate((y_train_major, y_train[feature_indexes]), axis=0)

    final_train_x = np.concatenate((X_train_major, minor_class_X_train), axis=0)
    final_train_y = np.concatenate((y_train_major, minor_class_y_train), axis=0)

    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=150))
    # classifier = sklearn.svm.SVC(C=50 , gamma= .0008 , kernel='rbf', probability=True)
    # classifier = sklearn.svm.SVC(C=100, gamma=.006, kernel='rbf', probability=True)

    classifier.fit(final_train_x, final_train_y)

    predicted = classifier.predict_proba(X_test)

    print("# Roc auc score                :- ", end='')
    # print(sklearn.metrics.classification_report(y_test, predicted))
    print(sklearn.metrics.roc_auc_score(y_test, predicted[:, 1]), end='')

    print("# Average precision score      :- ", end='')
    print(average_precision_score(y_test, predicted[:, 1]))

    avg_roc += sklearn.metrics.roc_auc_score(y_test, predicted[:, 1])
    avg_aupr += average_precision_score(y_test, predicted[:, 1])

    fold_counTer += 1
    if fold_counTer == 5:
        tpr_c = dict()
        fpr_c = dict()
        roc_c = dict()
        for i in range(2):
            fpr_c[i], tpr_c[i], _ = roc_curve(y_test,
                                              predicted[:, i])
            roc_c[i] = roc_auc_score(y_test, predicted[:, i])

        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(2):
            precision[i], recall[i], _ = precision_recall_curve(y_test,
                                                                predicted[:, i])
            average_precision[i] = average_precision_score(y_test, predicted[:, i])

print("avg roc = ", avg_roc / number_of_folds)
print("avg aupr = ", avg_aupr / number_of_folds)



top_roc = 0

#
# for train_index, test_index in skf.split(X, y):
#     X_train = X[train_index]
#     X_test = X[test_index]
#
#     y_train = y[train_index]
#     y_test = y[test_index]
#
#     break
#
# for depth in range(3, 20, 20):
#     for split in range(3, 9, 20):
#
#         classifier = AdaBoostClassifier(
#             DecisionTreeClassifier(max_depth=depth, min_samples_split=split),
#             n_estimators=100,
#             learning_rate=1, algorithm='SAMME')
#
#         X_train, y_train = sampler.fit_sample(X_train, y_train)
#
#         classifier.fit(X_train, y_train)
#
#         predictions = classifier.predict_proba(X_test)
#
#         score = roc_auc_score(y_test, predictions[:, 1])
#
#         if top_roc < score:
#             top_roc = score
#
#             tpr = dict()
#             fpr = dict()
#             roc = dict()
#             for i in range(2):
#                 fpr[i], tpr[i], _ = roc_curve(y_test,
#                                               predictions[:, i])
#                 roc[i] = roc_auc_score(y_test, predictions[:, i])

############## aupr

plt.clf()
plt.plot(recall[1], precision[1], lw=2, color='red',
         label='Precision-Recall Clustered sampling')

# plt.plot(recall_c[1], precision_c[1], lw=2, color='navy',
#          label='Precision-Recall random under sampling')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall'.format(average_precision))
plt.legend(loc="upper right")
plt.savefig('/home/farshid/Desktop/aupr/nr_dataset_1282.png')
plt.show()

########################################### roc


plt.plot(fpr_c[1], tpr_c[1], lw=2, color='red', label='Roc curve: Clustered sampling')
plt.plot(fpr[1], tpr[1], lw=2, color='navy',
         label='Roc curve: random under sampling')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Area under ROC curve')
plt.legend(loc="lower right")
plt.savefig('/home/farshid/Desktop/roc/nr_dataset_1282.png')
plt.show()

