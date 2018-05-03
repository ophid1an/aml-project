# Machine Learning
# Bozas Aristeidis
# AM: 740
# Dermentzoglou Ioannis
# AM: 743

import time
from itertools import product

import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

RANDOM_STATE = 0

# Read csv as dataframe
cc_df = pd.read_csv('./Datasets/creditcard.csv')

# Define y
y = cc_df['Class']

# Define X after dropping Time and Class features
X = cc_df.drop(['Time', 'Class'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, stratify=y)

# Standardize X
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


def fit_clf(*args):
    pipeline = make_pipeline(*args)
    start = time.time()
    pipeline.fit(X_train, y_train)
    stop = time.time()
    print("Time:%.4f" % (stop - start))
    return pipeline


def print_results(clf, label='DEFAULT'):
    print('*** ' + label + " ***\n\n")
    y_pred = clf.predict(X_test)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred)))
    print(classification_report_imbalanced(y_test, y_pred, digits=3))


clfs = [
    {'name': 'RandomForestClassifier', 'obj': RandomForestClassifier(random_state=RANDOM_STATE)},
    {'name': 'LinearSVC', 'obj': LinearSVC(random_state=RANDOM_STATE)},
    {'name': 'GaussianNB', 'obj': GaussianNB()},
]

methods = [
    {'name': 'NONE'},
    {'name': 'NearMiss-2', 'obj': NearMiss(random_state=RANDOM_STATE, version=2)},
    {'name': 'SMOTE', 'obj': SMOTE(random_state=RANDOM_STATE)},
    {'name': 'BalancedBaggingClassifier'},
]

for m, c in product(methods, clfs):
    if m['name'] == 'NONE':
        print_results(fit_clf(c['obj']), c['name'])
    elif m['name'] == 'BalancedBaggingClassifier':
        bbc = BalancedBaggingClassifier(random_state=RANDOM_STATE, base_estimator=c['obj'])
        print_results(fit_clf(bbc), m['name'] + ' - ' + c['name'])
    else:
        print_results(fit_clf(m['obj'], c['obj']), m['name'] + ' - ' + c['name'])
