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
import final_version


def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5


labelencoder = LabelEncoder()

# df = pd.read_csv('/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/second_train.csv', header=None)
# dff = pd.read_csv('/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/test.csv', header=None)
#
#
#
# df['label'] = df[df.shape[1] - 1]
# df.drop([df.shape[1] - 2], axis=1, inplace=True)
#
# # df['label'] = labelencoder.fit_transform(df['label'])
#
# # X = df.select_dtypes(['number']).join(pd.get_dummies(df.select_dtypes(exclude=['number'])))
# X = np.array(df.drop(['label'], axis=1))
# y = np.array(df['label'])

# print(len(X[0]))

# selected_feats = np.array(dff[0])
# nor = Normalizer()
# X = nor.fit_transform(X)
# X = X[:,selected_feats]
# print(selected_feats)

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
est_list = [est1,est2]

number = range(5,100,10)

X, y = final_version.feature_selection()
print(len(X[0]))

number_of_split = 5
skf = StratifiedKFold(n_splits=number_of_split, shuffle=True)
for regg in reg_list:
    for estimator in est_list:
        for n in number:
            for depth in range(2, 200, 10):
                for split in range(2, 15, 1):
                    for selector in feature_selection_list:
                        for train_index, test_index in skf.split(X, y):
                            X_train = X[train_index]
                            X_test = X[test_index]

                            y_train = y[train_index]
                            y_test = y[test_index]

                            # selector = f_regression(X_train,y_train)
                            # X_train = selector.fit_transform(X_train, y_train)
                            # X_test = selector.transform(X_test)

                            # reg = regg(max_depth=depth, min_samples_split=split)
                            reg = regg(estimator(max_depth=depth, min_samples_split=split),n_estimators=n,learning_rate=1)

                            reg.fit(X_train, y_train)

                            predictions = reg.predict(X_test)

                            # error = np.sqrt(np.mean((predictions - y_test)**2))
                            error = rmsle(predictions, y_test)

                            if last_error > error:
                                last_error = error
                                print('reg ', str(reg), ' using', str(selector), ' for depth ', depth, ' and split ', split,
                                      ' error = ', error)

                            break
