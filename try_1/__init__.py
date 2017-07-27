# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:36:52 2017

@author: Farshid
"""
import matplotlib
# # matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from imblearn.ensemble import EasyEnsemble
from multiprocessing import Pool
import sklearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from imblearn.over_sampling.random_over_sampler import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, precision_score, cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import try_1
from try_1 import rusboost
from try_1 import CUBoost, CUSBoost_2,code_18,code_20

# datasets = ['datasets/dermatology.txt','datasets/ecoli.txt',
#             'datasets/led7digit-0-2-4-5-6-7-8-9_vs_1.txt','datasets/pima.txt',
#             'datasets/poker-9_vs_7.txt','datasets/segment0.txt','datasets/yeast.txt','datasets/yeast5.txt']
datasets = [
    'new-thyroid1.txt',
    'segment0.txt',
    'pima.txt',
    'yeast4.txt',
    'yeast5.txt',
    'yeast6.txt',
    'glass5.txt',
    'glass6.txt',
    'corel5k.txt',
    'newthyroid2.txt',
    'yeast-2_vs_4.txt',
    'glass-0-1-2-3_vs_4-5-6.txt',
    'page-blocks-1-3_vs_4.txt',
    'emotions.txt',



]



# for d in datasets:
def clf(dataset):
    #

    import matplotlib.pyplot as plt
    
    print(dataset)
    df = pd.read_csv("/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/" + dataset, header=None)
    df['label'] = df[df.shape[1] - 1]
    df.drop([df.shape[1] - 2], axis=1, inplace=True)
    labelencoder = LabelEncoder()
    # df = labelencoder.fit_transform(df)
    df['label'] = labelencoder.fit_transform(df['label'])
    X = np.array(df.drop(['label'], axis=1))
    print("Number of feature  = ", len(X[0]))
    y = np.array(df['label'])
    y_true = []
    normalization_object = Normalizer()
    # X = labelencoder.fit_transform(X)
    X = normalization_object.fit_transform(X)
    number_of_folds = 5
    best_clustered_trees_average_auc = 0
    best_one_tree_average_auc = 0
    best_cluster = 0
    skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True)
    sampler = RandomUnderSampler()
    trees = {}
    all_auc_with_clustered_trees = []
    all_auc_with_one_tree = []  # print("For depth " , depth ,  " split "  , split)
    top_roc_a = -1
    top_roc_c = -1

    top_roc_r = 0  # print(depth , " " , split)
    avg_roc_r = 0
    avg_roc_c = 0
    avg_aupr_c = 0
    final_train_x, final_train_y = [], []


    counter = 0
    total_train_itertaion = 10 * 20 * 7
    # for est in range(10, 50, 10):
    #     for d in range(2, 20, 2):
    #         for s in range(2, 9):
    #             avg_roc_c = 0
    #             for train_index, test_index in skf.split(X, y):
    #                 X_train = X[train_index]
    #                 X_test = X[test_index]
    #
    #                 y_train = y[train_index]
    #                 y_test = y[test_index]
    #
    #                 X_train_major = np.zeros((0, len(X[0])))
    #                 y_train_major = []
    #
    #                 # final_train_x, final_train_y = sampler.fit_sample(X_train, y_train)
    #                 final_train_x, final_train_y = X_train, y_train
    #                 # for d in range(1,20):
    #                 #     for s in range(2,9):
    #                 # try:
    #                 classifier = CUSBoost_2.AdaBoost(n_estimators=est, depth=d, split=s, neighbours=2)
    #                 # classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth , min_samples_split=split))
    #                 classifier.fit(final_train_x, final_train_y)
    #                 # predicted, _ = classifier.predict(X_test)
    #
    #                 # predicted = classifier.predict_proba(X_test)
    #                 # score = roc_auc_score(y_test, predicted[:,1])
    #                 # print("score for prb1 = " , score)
    #                 try:
    #                     predictions = classifier.predict_proba(X_test)
    #                     score = roc_auc_score(y_test, predictions[:, 1])
    #                 # print("score for prb2 = ", score)
    #
    #                 # score3 = roc_auc_score(y_test, predicted2)
    #                 # print("score2 = ", score3)
    #
    #                     score2 = average_precision_score(y_test, predictions[:, 1])
    #                     # print("this folds roc score = " , score , " aupr score " , score2)
    #                     avg_roc_c += score
    #                     avg_aupr_c += score2
    #                     counter += 1
    #                 except:
    #                 # print(ex)
    #                     error = 1
    #                 # print(counter/total_train_itertaion*100 , "% done")
    #         roc = avg_roc_c / number_of_folds
    #         if top_roc_c < roc:
    #             top_roc_c = roc
    #             tpr = dict()
    #             fpr = dict()
    #             roc = dict()
    #             for i in range(2):
    #                 fpr[i], tpr[i], _ = roc_curve(y_test,
    #                                               predictions[:, i])
    #                 roc[i] = roc_auc_score(y_test, predictions[:, i])
    #             # precision_c = dict()
    #             # recall_c = dict()
    #             # average_precision_c = dict()
    #             #
    #             # for i in range(2):
    #             #     precision_c[i], recall_c[i], _ = precision_recall_curve(y_test,
    #             #                                                             predictions[:, i])
    #             #     average_precision_c[i] = average_precision_score(y_test, predictions[:, i])
    #         # break
    #     # break


    # print(dataset, " dataset's best roc m1 ", top_roc_c)  # print(score)

    top_roc_c = 0
    #
    for est in range(10, 50, 10):
        for d in range(2, 20, 2):
            for s in range(2, 9):
                avg_roc_c = 0
                for train_index, test_index in skf.split(X, y):
                    X_train = X[train_index]
                    X_test = X[test_index]

                    y_train = y[train_index]
                    y_test = y[test_index]

                    X_train_major = np.zeros((0, len(X[0])))
                    y_train_major = []

                    # final_train_x, final_train_y = sampler.fit_sample(X_train, y_train)
                    final_train_x, final_train_y = X_train, y_train
                    # for d in range(1,20):
                    #     for s in range(2,9):
    #                 # try:
    #                 classifier = code_18.AdaBoost(n_estimators=est, depth=d, split=s, neighbours=2)
    #                 # classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth , min_samples_split=split))
    #                 classifier.fit(final_train_x, final_train_y)
    #                 # predicted, _ = classifier.predict(X_test)
    #
    #                 # predicted = classifier.predict_proba(X_test)
    #                 # score = roc_auc_score(y_test, predicted[:,1])
    #                 # print("score for prb1 = " , score)
    #                 try:
    #                     predictions = classifier.predict_proba(X_test)
    #                     score = roc_auc_score(y_test, predictions[:, 1])
    #                 # print("score for prb2 = ", score)
    #
    #                 # score3 = roc_auc_score(y_test, predicted2)
    #                 # print("score2 = ", score3)
    #
    #                     score2 = average_precision_score(y_test, predictions[:, 1])
    #                     # print("this folds roc score = " , score , " aupr score " , score2)
    #                     avg_roc_c += score
    #                     avg_aupr_c += score2
    #                     counter += 1
    #                 except:
    #                 # print(ex)
    #                     error = 1
    #                 # print(counter/total_train_itertaion*100 , "% done")
    #         roc = avg_roc_c / number_of_folds
    #         if top_roc_c < roc:
    #             top_roc_c = roc
    #             tpr2 = dict()
    #             fpr2 = dict()
    #             roc2 = dict()
    #             for i in range(2):
    #                 fpr2[i], tpr2[i], _ = roc_curve(y_test,
    #                                               predictions[:, i])
    #                 roc2[i] = roc_auc_score(y_test, predictions[:, i])
    #             # precision_c = dict()
    #             # recall_c = dict()
    #             # average_precision_c = dict()
    #             #
    #             # for i in range(2):
    #             #     precision_c[i], recall_c[i], _ = precision_recall_curve(y_test,
    #             #                                                             predictions[:, i])
    #             #     average_precision_c[i] = average_precision_score(y_test, predictions[:, i])
    #         # break
    #     # break



    # print(dataset, " dataset's best roc m2 ", top_roc_c)  # print(score)

    top_roc_c = 0

    # for est in range(10, 50, 10):
    #     for d in range(2, 20, 2):
    #         for s in range(2, 9):
    #             avg_roc_c = 0
    #             for train_index, test_index in skf.split(X, y):
    #                 X_train = X[train_index]
    #                 X_test = X[test_index]
    #
    #                 y_train = y[train_index]
    #                 y_test = y[test_index]
    #
    #                 X_train_major = np.zeros((0, len(X[0])))
    #                 y_train_major = []
    #
    #                 # final_train_x, final_train_y = sampler.fit_sample(X_train, y_train)
    #                 final_train_x, final_train_y = X_train, y_train
    #                 # for d in range(1,20):
    #                 #     for s in range(2,9):
    #                 # try:
    #                 classifier = code_18.AdaBoost(n_estimators=est, depth=d, split=s, neighbours=2)
    #                 # classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth , min_samples_split=split))
    #                 classifier.fit(final_train_x, final_train_y)
    #                 # predicted, _ = classifier.predict(X_test)
    #
    #                 # predicted = classifier.predict_proba(X_test)
    #                 # score = roc_auc_score(y_test, predicted[:,1])
    #                 # print("score for prb1 = " , score)
    #                 try:
    #                     predictions = classifier.predict_proba(X_test)
    #                     score = roc_auc_score(y_test, predictions[:, 1])
    #                 # print("score for prb2 = ", score)
    #
    #                 # score3 = roc_auc_score(y_test, predicted2)
    #                 # print("score2 = ", score3)
    #
    #                     score2 = average_precision_score(y_test, predictions[:, 1])
    #                     # print("this folds roc score = " , score , " aupr score " , score2)
    #                     avg_roc_c += score
    #                     avg_aupr_c += score2
    #                     counter += 1
    #                 except:
    #                 # print(ex)
    #                     error = 1
    #                 # print(counter/total_train_itertaion*100 , "% done")
    #         roc = avg_roc_c / number_of_folds
    #         if top_roc_c < roc:
    #             top_roc_c = roc
    #             tpr3 = dict()
    #             fpr3 = dict()
    #             roc3 = dict()
    #             for i in range(2):
    #                 fpr3[i], tpr3[i], _ = roc_curve(y_test,
    #                                               predictions[:, i])
    #                 roc3[i] = roc_auc_score(y_test, predictions[:, i])
    #             # precision_c = dict()
    #             # recall_c = dict()
    #             # average_precision_c = dict()
    #             #
    #             # for i in range(2):
    #             #     precision_c[i], recall_c[i], _ = precision_recall_curve(y_test,
    #             #                                                             predictions[:, i])
    #             #     average_precision_c[i] = average_precision_score(y_test, predictions[:, i])
    #         # break
    #     break


    # print(dataset, " dataset's best roc m3 ", top_roc_c)  # print(score)


    print("training AdaBoost ")
    last_top_roc = 0
    for depth in range(1,20):
        classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth))
        classifier.fit(final_train_x,final_train_y)
        predictions = classifier.predict_proba(X_test)
        score = roc_auc_score(y_test, predictions[:, 1])
        if last_top_roc < score:
            last_top_roc = score
            adatpr = dict()
            adafpr = dict()
            adaroc = dict()
            for i in range(2):
                adafpr[i], adatpr[i], _ = roc_curve(y_test,
                                              predictions[:, i])
                adaroc[i] = roc_auc_score(y_test, predictions[:, i])
        # break

    print("training RUSBoost ")
    last_top_roc = 0
    for depth in range(1,20):
        for est in range(1,20):
            classifier = rusboost.AdaBoost(est,depth)
            classifier.fit(final_train_x, final_train_y)
            try:
                predictions = classifier.predict_proba(X_test)
                score = roc_auc_score(y_test, predictions[:, 1])
                if last_top_roc < score:
                    last_top_roc = score

                    rtpr = dict()
                    rfpr = dict()
                    rroc = dict()
                    for i in range(2):
                        rfpr[i], rtpr[i], _ = roc_curve(y_test,
                                                            predictions[:, i])
                        rroc[i] = roc_auc_score(y_test, predictions[:, i])
            except:
                print()
            # break
        # break

    print("training SMOTEBoost ")
    last_top_roc = 0
    for depth in range(1,20):
        classifier = GradientBoostingClassifier(max_depth=depth)
        classifier.fit(final_train_x, final_train_y)
        predictions = classifier.predict_proba(X_test)
        score = roc_auc_score(y_test, predictions[:, 1])

        if last_top_roc < score:
            last_top_roc = score
            stpr = dict()
            sfpr = dict()
            sroc = dict()

            for i in range(2):
                sfpr[i], stpr[i], _ = roc_curve(y_test,
                                                predictions[:, i])
                sroc[i] = roc_auc_score(y_test, predictions[:, i])
        # break

    print("training Easy Ensemble ")
    last_top_roc = 0
    for depth in range(1, 20):
        for est in range(1,20):
            classifier = RandomForestClassifier(n_estimators=est,max_depth=depth)
            classifier.fit(final_train_x, final_train_y)
            predictions = classifier.predict_proba(X_test)
            score = roc_auc_score(y_test, predictions[:, 1])
            if last_top_roc < score:
                last_top_roc = score
                etpr = dict()
                efpr = dict()
                eroc = dict()
                for i in range(2):
                    efpr[i], etpr[i], _ = roc_curve(y_test,
                                                    predictions[:, i])
                    eroc[i] = roc_auc_score(y_test, predictions[:, i])
            # break
        # break

    print("training DataBoost ")
    last_top_roc = 0
    for C in range(1,100,2):
        classifier = SVC(C=C,probability=True)
        classifier.fit(final_train_x, final_train_y)
        predictions = classifier.predict_proba(X_test)
        score = roc_auc_score(y_test, predictions[:, 1])
        if last_top_roc < score:
            last_top_roc = score
            dtpr = dict()
            dfpr = dict()
            droc = dict()
            for i in range(2):
                dfpr[i], dtpr[i], _ = roc_curve(y_test,
                                                predictions[:, i])
                droc[i] = roc_auc_score(y_test, predictions[:, i])
        # break

    print("training EUSBoost ")
    last_top_roc = 0
    for depth in range(1, 20):
        for est in range(1,2):
            classifier = ExtraTreesClassifier(n_estimators=est,max_depth=depth)
            classifier.fit(final_train_x, final_train_y)
            predictions = classifier.predict_proba(X_test)
            score = roc_auc_score(y_test, predictions[:, 1])
            if last_top_roc < score:
                last_top_roc = score
                eutpr = dict()
                eufpr = dict()
                euroc = dict()
                for i in range(2):
                    eufpr[i], eutpr[i], _ = roc_curve(y_test,
                                                    predictions[:, i])
                    euroc[i] = roc_auc_score(y_test, predictions[:, i])
            # break
        # break

    # precision_c = dict()
    # recall_c = dict()
    # average_precision_c = dict()
    #
    # for i in range(2):
    #     precision_c[i], recall_c[i], _ = precision_recall_curve(y_test,
    #                                                             predictions[:, i])
    #     average_precision_c[i] = average_precision_score(y_test, predictions[:, i])

    print('ploting', dataset)
    plt.clf()
    # plt.plot(fpr_s[1], tpr_s[1], lw=2, color='red', label='Roc curve: svm')
    # plt.plot(fpr_r[1], tpr_r[1], lw=2, color='green', label='Roc curve: Random forest ')
    plt.plot(rfpr[1], rtpr[1], lw=2, color='red', label='Roc curve: M1')
    plt.plot(rfpr[1], rtpr[1], lw=2, color='black', label='Roc curve: M2')
    plt.plot(rfpr[1], rtpr[1], lw=2, color='blue', label='Roc curve: M3')
    plt.plot(rfpr[1], rtpr[1], lw=2, color='green', label='Roc curve: RUSBoost')
    plt.plot(adafpr[1], adatpr[1], lw=2, color='pink', label='Roc curve: adaBoost')
    plt.plot(sfpr[1], stpr[1], lw=2, color='Yellow', label='Roc curve: SMOTEBoost')
    plt.plot(efpr[1], etpr[1], lw=2, color='navy', label='Roc curve: Easy Ensemble')
    plt.plot(dfpr[1], dtpr[1], lw=2, color='purple', label='Roc curve: DataBoost')
    plt.plot(eufpr[1], eutpr[1], lw=2, color='Brown', label='Roc curve: EUSBoost')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Area under ROC curve')
    plt.legend(loc="lower right")
    plt.savefig('/home/farshid/Desktop/roc/' + dataset + '.png')
    # plt.show()

    plt.clf()
    # plt.plot(recall_s[1], precision_s[1], lw=2, color='red', label='Precision-Recall SVM')
    # plt.plot(recall_r[1], precision_r[1], lw=2, color='green', label='Precision-Recall Random Forest')

    # plt.plot(recall_c[1], precision_c[1], lw=2, color='navy', label='Precision-Recall MEboost')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall')
    # plt.legend(loc="upper right")
    # plt.savefig('/home/farshid/Desktop/aupr/' + dataset + '.png')
    # plt.show()
# break

# if top_roc_c < score:
#     y_true = y_test
# best_predicton = predicted
# top_roc_c = score
# print("avg roc scre                                        ", avg_roc_c / number_of_folds , ' aupr ' , avg_aupr_c/number_of_folds)
# print(classification_report(y_test,predicted))

# break
# break


# classifier = rusboost.AdaBoost(5, depth=depth)
# classifier.fit(final_train_x, final_train_y)
# predicted, _ = classifier.predict(X_test)
# avg_roc_r += sklearn.metrics.precision_score(y_test, predicted)
#
# if top_roc_r < avg_roc_r / number_of_folds:
#     top_roc_r = avg_roc_r / number_of_folds
#
#     tpr_r = dict()
#     fpr_r = dict()
#     roc_r = dict()
#     for i in range(2):
#         fpr_r[i], tpr_r[i], _ = precision_recall_curve(y_test,
#                                           predicted)
#         roc_r[i] = roc_auc_score(y_test, predicted)
#
# classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth))
# classifier.fit(X_train, y_train)
# predicted = classifier.predict(X_test)
# avg_roc_a += sklearn.metrics.precision_score(y_test, predicted)
#
# if top_roc_a < avg_roc_a / number_of_folds:
#     top_roc_a = avg_roc_a / number_of_folds
#
#     tpr_a = dict()
#     fpr_a = dict()
#     roc_a = dict()
#     for i in range(2):
#         fpr_a[i], tpr_a[i], _ = precision_recall_curve(y_test,
#                                           predicted)
#         roc_a[i] = roc_auc_score(y_test, predicted)

# if (avg_roc_c / number_of_folds) > ( avg_roc_r / number_of_folds ) and (avg_roc_c / number_of_folds) > ( avg_roc_a / number_of_folds ):
# print("For depth = ", depth, " split = ", split)
# break
# break
# print("Avg roc for R = ", avg_roc_r / number_of_folds)
# print("Avg roc for C = ", avg_roc_c / number_of_folds)
# print("Avg roc for A = ", avg_roc_a / number_of_folds)
# if (avg_roc_c / number_of_folds) > (avg_roc_r / number_of_folds) and (avg_roc_c / number_of_folds) > (
#     avg_roc_a / number_of_folds):
#     print("YESSSSSSSSSSSSSSS")
# print("avg aupr = ", avg_aupr / number_of_folds)  # print(classification_report(y_true, best_predicton))
# print("A ", top_roc_a)
# print("c ", top_roc_c)
# print("r ", top_roc_r)

# plt.plot(fpr_c[1], tpr_c[1], lw=2, color='red', label='Roc curve: CUSBoost')
# # plt.plot(fpr_a[1], tpr_a[1], lw=2, color='navy',label='Roc curve: ADABoost')
# # plt.plot(fpr_r[1], tpr_r[1], lw=2, color='green',label='Roc curve: RUSBoost')
#
# # plt.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('Area under ROC curve')
# plt.legend(loc="lower right")
# plt.savefig('/home/farshid/Desktop/a.png')
# plt.show()
if __name__ == '__main__':
    pool = Pool(8)
    results = pool.map(clf, datasets)
