import time

import pandas as pa
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 0
READ_RESULTS = True
WRITE_RESULTS = False
RESULTS_VERSION = 2

iris_df = pa.read_csv("./Datasets/iris.csv", header=None)  # load Iris Dataset
wine_df = pa.read_csv("./Datasets/wine.csv", header=None)  # load Wine Dataset
bcancer_df = pa.read_csv("./Datasets/wdbc.csv", header=None)  # load Breast Cancer Dataset
balance_df = pa.read_csv("./Datasets/balance-scale.csv", header=None)  # load Balance Scale Dataset
hayesroth_df = pa.read_csv("./Datasets/hayesroth.csv", header=None)  # load Hayes-Roth Dataset
haberman_df = pa.read_csv("./Datasets/haberman.csv", header=None)  # load Haberman's Survival Dataset
liver_df = pa.read_csv("./Datasets/liverdisorder.csv", header=None)  # load Liver Disorders Dataset
bank_df = pa.read_csv("./Datasets/data_banknote_authentication.csv",
                      header=None)  # load Banknote Authentication Dataset
ionosphere_df = pa.read_csv("./Datasets/ionosphere.csv", header=None)  # load Ionosphere Dataset
cmc_df = pa.read_csv("./Datasets/cmc.csv", header=None)  # load Contraceptive Method Choice Dataset


def to_dict_data(n, x, y):
    return {'name': n, 'X': x, 'y': y}


data = []

# Iris data and target
data.append(to_dict_data(
    'Iris',
    iris_df.iloc[:, :-1],
    iris_df.iloc[:, -1]))

# Wine data and target
data.append(to_dict_data(
    'Wine',
    wine_df.iloc[:, 1:],
    wine_df.iloc[:, 0]))

# Breast cancer data and target
data.append(to_dict_data(
    'Breast Cancer',
    bcancer_df.iloc[:, 2:],
    bcancer_df.iloc[:, 1]))

# Balance-scale data and target
data.append(to_dict_data(
    'Balance Scale',
    balance_df.iloc[:, 1:],
    balance_df.iloc[:, 0]))

# hayes-roth data and target
data.append(to_dict_data(
    'Hayes-Roth',
    hayesroth_df.iloc[:, 1:-1],
    hayesroth_df.iloc[:, -1]))

# Haberman survival data and target
data.append(to_dict_data(
    'Haberman\'s Survival',
    haberman_df.iloc[:, :-1],
    haberman_df.iloc[:, -1]))

# Liver Disorder  data and target
data.append(to_dict_data(
    'Liver Disorders',
    liver_df.iloc[:, :-2],
    liver_df.iloc[:, -2].map(lambda x: 0 if x < 3 else 1)))

# Banknote Authentication data and target
data.append(to_dict_data(
    'Banknote Authentication',
    bank_df.iloc[:, :-1],
    bank_df.iloc[:, -1]))

# Ionosphere data and target
data.append(to_dict_data(
    'Ionosphere',
    ionosphere_df.iloc[:, :-1],
    ionosphere_df.iloc[:, -1]))

# cmc data and target
data.append(to_dict_data(
    'Contraceptive Method Choice',
    cmc_df.iloc[:, :-1],
    cmc_df.iloc[:, -1]))

classifiers = []

# Tree Classifier
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
classifiers.append([dt, "tree"])

# -----Manipulating the training examples-------
# Bagging
bagged_dt = BaggingClassifier(dt, n_estimators=100, random_state=RANDOM_STATE)
classifiers.append([bagged_dt, "bagged tree"])
# Boosting
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=100, random_state=RANDOM_STATE)
classifiers.append([ada, "AdaBoost-ed tree"])

gbm = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
classifiers.append([gbm, "gradient boosting tree"])
# -----Manipulating the target variable------
# One vs One
ovo = OneVsOneClassifier(dt)
classifiers.append([ovo, "one-vs-one tre"])
# One vs Rest
ovr = OneVsRestClassifier(dt)
classifiers.append([ovr, "one-vs-rest tree"])
# -----Injecting randomness------
# Random forest
rf = RandomForestClassifier(random_state=RANDOM_STATE)
classifiers.append([rf, "Random forest"])
# -----Manipulating Features------
# RandomPatches
bagged_dt_mf = BaggingClassifier(dt, n_estimators=100, max_samples=0.7, max_features=0.7, random_state=RANDOM_STATE)
classifiers.append([bagged_dt_mf, "bagged tree random patches"])

results = []

for d in data:
    print("------ %20s ------ " % (d['name']))
    row = []
    for classifier, label in classifiers:
        start = time.time()
        pipeln = make_pipeline(preprocessing.StandardScaler(), classifier)
        scores = cross_val_score(pipeln, d['X'], d['y'], cv=10)
        stop = time.time()
        print("%20s accuracy: %0.3f (+/- %0.3f), time:%.3f" % (label, scores.mean(), scores.std() * 2, stop - start))
        row.append(round(scores.mean(), 3))
    results.append(row)

if READ_RESULTS or WRITE_RESULTS:
    filename = 'erwthma1-results.csv'
    filename_diff = filename.split('.')[0] + '-diff.' + filename.split('.')[1]
    results_df = pa.DataFrame(data=results, columns=list(map(lambda x: x[1], classifiers)),
                              index=list(map(lambda x: x['name'], data)))

    if READ_RESULTS:
        print(results_df)
        try:
            prev_results_df = pa.read_csv(filename, index_col=0)
            results_diff_df = results_df.subtract(prev_results_df).round(decimals=3)
            print(results_diff_df)
        except Exception:
            pass

    if WRITE_RESULTS:
        try:
            prev_results_df = pa.read_csv(filename, index_col=0)
            results_diff_df = results_df.subtract(prev_results_df).round(decimals=3)
            results_diff_df.to_csv(filename_diff)
        except Exception:
            pass
        results_df.to_csv(
            './stuff/results/' + filename.split('.')[0] + str(RESULTS_VERSION).rjust(3, '0') + '.' +
            filename.split('.')[1])
        results_df.to_csv(filename)
