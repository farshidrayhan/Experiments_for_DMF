import csv

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFdr, SelectFpr, SelectKBest, SelectFwe
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import Lasso
import math
import matplotlib.pyplot as plt
import matplotlib


def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5

def catagorical_to_numeric(df):
    df['label'] = df[df.shape[1] - 1]
    # df['index'] = df[df.shape[1] - 1]
    df.drop([df.shape[1] - 2], axis=1, inplace=True)

    df['label'] = LabelEncoder().fit_transform(df['label'])
    # print(X)
    X = df.drop(['label'], axis=1)
    #
    X = df.drop([0], axis=1)
    X = X.drop([1], axis=1)

    new_x = X.select_dtypes(exclude=['number'])
    new_X = pd.DataFrame(
        np.matrix.transpose(np.array([LabelEncoder().fit_transform(X[i].astype('str')) for i in new_x])))
    new_k = X.select_dtypes(include=['number'])
    X = pd.concat([new_X, new_k], axis=1)
    X = np.array(X)


    # print(len(X[0]),len(X))
    y = np.array(df['label'])

    return X,y
def catagorical_to_onehot(df):

    df['label'] = df[df.shape[1] - 1]
    # df['index'] = df[df.shape[1] - 1]
    df.drop([df.shape[1] - 2], axis=1, inplace=True)

    # df['label'] = LabelEncoder().fit_transform(df['label'])
    # print(X)
    X = df.drop(['label'], axis=1)
    #
    X = df.drop([0], axis=1)
    X = X.drop([1], axis=1)

    X = np.array(X.select_dtypes(['number']).join(pd.get_dummies(X.select_dtypes(exclude=['number']))))

    # print(len(X[0]), len(X))
    y = np.array(df['label'])

    return X, y

def feature_selection():
    # print('ploting', dataset)
    df = pd.read_csv('/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/org_train.csv',
                     header=None)
    # dff = pd.read_csv('/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/second_test.csv',
    #                   header=None)

    # X_testdata = np.array(dff)
    X , y = catagorical_to_numeric(df)
    # X , y = onehot(df)

    # X = LabelEncoder().fit(X[1])
    # X = np.array(df.select_dtypes(['number']).join(pd.get_dummies(df.select_dtypes(exclude=['number']))))
    # X = LabelEncoder.fit()

    values = []
    for i in range(0,len(X[0])):

        values.append(np.corrcoef(X[:,i], y , )[0,1])
        print(i,np.corrcoef(X[:,i], y , )[0,1])
    #
    # print(np.argsort(values)[-20:])
    # print(np.argsort(values)[-20,-6:])
    # print(np.sort(values))

    X_col1 = X[:,np.argsort(values)[-10:]]
    X_col2 = X[:,np.argsort(values)[:50]]

    new_x = np.concatenate((X_col1,X_col2),axis=1)

    return X_col2,y


    # plot2(X_col,y)

def plot2(X,y):

    for i in range(0,len(X[0])):
        plt.clf()

        plt.scatter(X[:,i], y, lw=2, color='red', label='Correlation')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Feature number '+ str(i))
        plt.ylabel('Class')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        plt.title('Area under ROC curve')
        plt.legend(loc="lower right")
        # plt.show()

        plt.savefig('/home/farshid/Desktop/kaggle_feature_co_rel/' + str(i) + '.png')


def main2():

    df = pd.read_csv('/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/second_train.csv',
                     header=None)
    # dff = pd.read_csv('/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/second_test.csv',
    #                   header=None)

    # X_testdata = np.array(dff)

    df['label'] = df[df.shape[1] - 1]
    df.drop([df.shape[1] - 2], axis=1, inplace=True)

    # df['label'] = labelencoder.fit_transform(df['label'])

    # X = df.select_dtypes(['number']).join(pd.get_dummies(df.select_dtypes(exclude=['number'])))
    # X_testdata = dff.select_dtypes(['number']).join(pd.get_dummies(dff.select_dtypes(exclude=['number'])))
    # X_testdata = np.array(dff)

    X = np.array(df.drop(['label'], axis=1))
    y = np.array(df['label'])

    # print(len(X_testdata[0]) , len(X[0]))


    obj1 = VarianceThreshold(threshold=(.01))
    obj2 = SelectFdr(alpha=.8)
    obj3 = SelectFpr()
    obj4 = SelectKBest(k=30)
    obj5 = SelectFwe()
    feature_selection_list = [obj1, obj2, obj3, obj4, obj5]

    reg1 = AdaBoostRegressor
    reg2 = ExtraTreesRegressor
    reg3 = RandomForestRegressor
    reg4 = GradientBoostingRegressor
    reg_list = [reg1, reg3]

    last_error = 9000

    est1 = DecisionTreeRegressor
    est2 = ExtraTreeRegressor
    est_list = [est1, est2]

    number = range(5, 100, 10)

    number_of_split = 5
    skf = StratifiedKFold(n_splits=number_of_split, shuffle=True)
    i = 0
    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        # selector = f_regression(X_train,y_train)
        X_train = obj1.fit_transform(X_train, y_train)
        X_test = obj1.transform(X_test)
        # if i == 0 :
        # X_testdat = obj1.transform(X_testdata)

        # reg = regg(max_depth=depth, min_samples_split=split)
        reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=142, min_samples_split=9), n_estimators=25, learning_rate=1)

        reg.fit(X_train, y_train)

        predictions = reg.predict(X_test)

        # error = np.sqrt(np.mean((predictions - y_test)**2))
        error = rmsle(predictions, y_test)
        # result = reg.predict(X_testdat)

        # if last_error > error:
        # last_error = error

        print(' error = ', error)

        # with open('newfile' + str(i) + '.csv', "w") as f:
        #     writer = csv.writer(f)
        #     for row in zip(result):
        #         writer.writerow(row)

        i += 1
    # break
if __name__ ==  '__main__':
    feature_selection()