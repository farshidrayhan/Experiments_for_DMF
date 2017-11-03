import csv

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFdr, SelectFpr, SelectKBest, SelectFwe
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import Lasso
import math


def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5


labelencoder = LabelEncoder()

df = pd.read_csv('/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/second_train.csv',
                 header=None)
dff = pd.read_csv('/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/second_test.csv',
                  header=None)

# X_testdata = np.array(dff)

df['label'] = df[df.shape[1] - 1]
df.drop([df.shape[1] - 2], axis=1, inplace=True)

# df['label'] = labelencoder.fit_transform(df['label'])

# X = df.select_dtypes(['number']).join(pd.get_dummies(df.select_dtypes(exclude=['number'])))
# X_testdata = dff.select_dtypes(['number']).join(pd.get_dummies(dff.select_dtypes(exclude=['number'])))
X_testdata = np.array(dff)

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
    X_testdat = obj1.transform(X_testdata)

    # reg = regg(max_depth=depth, min_samples_split=split)
    reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=142, min_samples_split=9), n_estimators=25, learning_rate=1)

    reg.fit(X_train, y_train)

    predictions = reg.predict(X_test)

    # error = np.sqrt(np.mean((predictions - y_test)**2))
    error = rmsle(predictions, y_test)
    result = reg.predict(X_testdat)

    # if last_error > error:
    # last_error = error

    print(' error = ', error)

    with open('newfile' + str(i) + '.csv', "w") as f:
        writer = csv.writer(f)
        for row in zip(result):
            writer.writerow(row)

    i += 1
# break
