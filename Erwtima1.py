import time
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as nu
import pandas as pa
import scipy

RANDOM_STATE = 0

iris_df = pa.read_csv("./Datasets/iris.csv")  # load Iris Dataset
wine_df = pa.read_csv("./Datasets/wine.csv")  # load Wine Dataset
bcancer_df = pa.read_csv("./Datasets/wdbc.csv")  # load Breast Cancer Dataset
balance_df = pa.read_csv("./Datasets/balance-scale.csv")  # load Balance Scale Dataset
hayesroth_df = pa.read_csv("./Datasets/hayesroth.csv")  # load Hayes-Roth Dataset
haberman_df = pa.read_csv("./Datasets/haberman.csv")  # load Haberman's Survival Dataset
liver_df = pa.read_csv("./Datasets/liverdisorder.csv")  # load Liver Disorders Dataset
bank_df = pa.read_csv("./Datasets/data_banknote_authentication.csv")  # load Banknote Authentication Dataset
ionosphere_df = pa.read_csv("./Datasets/ionosphere.csv")  # load Ionosphere Dataset
cmc_df = pa.read_csv("./Datasets/cmc.csv")  # load Contraceptive Method Choice Dataset"

Data = []
target = []
# Iris data and target
iris_data = iris_df.iloc[:, 0:3]
iris_target = iris_df.iloc[:, 4]
# Wine data and target
wine_data = wine_df.iloc[:, 1:12]
wine_target = wine_df.iloc[:, 0]
# Breast cancer data and target
bcancer_data = bcancer_df.iloc[:, 2:31]
bcancer_target = bcancer_df.iloc[:, 1]
# Balance-scale data and target
balance_data = balance_df.iloc[:, 1:4]
balance_target = balance_df.iloc[:, 0]
# hayes-roth data and target
hayesroth_data = hayesroth_df.iloc[:, 0:4]
hayesroth_target = hayesroth_df.iloc[:, 5]
# Haberman survival data and target
haberman_data = haberman_df.iloc[:, 0:2]
haberman_target = haberman_df.iloc[:, 3]
# Liver Disorder  data and target
liver_data = liver_df.iloc[:, 0:5]
liver_target = liver_df.iloc[:, 6]
# Chess(Rook vs King) data and target
bank_data = bank_df.iloc[:, 0:3]
bank_target = bank_df.iloc[:, 4]
# Ionosphere data and target
ionosphere_data = ionosphere_df.iloc[:, 0:33]
ionosphere_target = ionosphere_df.iloc[:, 34]
# cmc data and target
cmc_data = cmc_df.iloc[:, 0:8]
cmc_target = cmc_df.iloc[:, 9]
# Data is the table with all dataset data
Data.append(iris_data)
Data.append(wine_data)
Data.append(bcancer_data)
Data.append(balance_data)
Data.append(hayesroth_data)
Data.append(haberman_data)
Data.append(liver_data)
Data.append(bank_data)
Data.append(ionosphere_data)
Data.append(cmc_data)
# Target is the table with all dataset targets
target.append(iris_target)
target.append(wine_target)
target.append(bcancer_target)
target.append(balance_target)
target.append(hayesroth_target)
target.append(haberman_target)
target.append(liver_target)
target.append(bank_target)
target.append(ionosphere_target)
target.append(cmc_target)
# Table with the labels for each dataset
Data_label = ["Iris Dataset", "Wine Dataset", "Breast Cancer Dataset", "Balance Scale Dataset", "Hayes-Roth Dataset",
              "Haberman's Survival Dataset", "Liver Disorders Dataset", "Banknote Authentication Dataset",
              "Ionosphere Dataset",
              "Contraceptive Method Choice Dataset"]

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

for i in range(len(Data)):
    print("------ %20s ------ " % (Data_label[i]))
    for classifier, label in classifiers:
        start = time.time()
        scores = cross_val_score(classifier, Data[i], target[i], cv=10)
        stop = time.time()
        print("%20s accuracy: %0.3f (+/- %0.3f), time:%.3f" % (label, scores.mean(), scores.std() * 2, stop - start))
