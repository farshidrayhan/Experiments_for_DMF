from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier(max_depth=5)

classifier = AdaBoostClassifier(
    clf,
    n_estimators=10,
    learning_rate=1,algorithm='SAMME')



# This part is for stratified cross validation
skf = StratifiedKFold(n_splits=5)

# This part is for Random Undersampling
sampler = RandomUnderSampler()



all_auc = []

for train_index, test_index in skf.split(X, y):
    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y[train_index]
    y_test = y[test_index]
    

    X_train , y_train = sampler.fit_sample(X_train,y_train)
    
    classifier.fit(X_train, y_train)
    
    
    predictions  = classifier.predict(X_test)
    
    all_auc.append(roc_auc_score(y_test, predictions))
    
    print('1 fold done')
    
average_auc = sum(all_auc)/len(all_auc)