import numpy as np
import pandas as pa
from costcla.metrics import savings_score, cost_loss
from costcla.models import BayesMinimumRiskClassifier, ThresholdingOptimization, CostSensitiveRandomPatchesClassifier
from costcla.sampling import cost_sampling, undersampling
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn import preprocessing

RANDOM_STATE = 0

heart_df = pa.read_csv("./Datasets/heart.txt", delimiter=" ", header=None)  # load StatLog(heart) dataset

# Iris data and target
X = heart_df.iloc[:, :-1].values
y = heart_df.iloc[:, -1]

# set target to binary 0, 1 instead 1,2
y = preprocessing.LabelEncoder().fit_transform(y)

cost = []
# for n, i in enumerate(y):
#     if i == 1:
#         y[n] = 0
#         cost.append([1, 5, 0, 0])
#     elif i == 2:
#         y[n] = 1
#         cost.append([1, 5, 0, 0])
for i in range(len(y)):
    cost.append([1, 5, 0, 0])
cost_mat = np.array(cost)

# Table with the labels for each dataset
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = \
    train_test_split(X, y, cost_mat, random_state=RANDOM_STATE)
# Cost sampling
X_cps_o, y_cps_o, cost_mat_cps_o = cost_sampling(X_train, y_train, cost_mat_train, method='OverSampling')
X_cps_r, y_cps_r, cost_mat_cps_r = cost_sampling(X_train, y_train, cost_mat_train, method='RejectionSampling')
X_u, y_u, cost_mat_u = undersampling(X_train, y_train, cost_mat_train)

# Calibrated models
classifiers = [
    {"name": "RF", "f": CalibratedClassifierCV(RandomForestClassifier(random_state=RANDOM_STATE), cv=10,
                                               method='sigmoid')},
    {"name": "LSVM", "f": CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE), cv=10, method='sigmoid')},
    {"name": "GNB", "f": CalibratedClassifierCV(GaussianNB(), cv=10, method='sigmoid')},
]

for clf in classifiers:
    # Fit
    clf["f"].fit(X_train, y_train)
    # Predict
    clf["c"] = clf["f"].predict(X_test)
    clf["p"] = clf["f"].predict_proba(X_test)
    clf["ptrain"] = clf["f"].predict_proba(X_train)
    # Sampling
    clf["c_o"] = clf["f"].fit(X_cps_o, y_cps_o).predict(
        X_test)  # Probabilities cost Oversampling
    clf["c_o_p"] = clf["f"].fit(X_cps_o, y_cps_o).predict_proba(
        X_test)  # Probabilities cost Oversampling
    clf["c_r"] = clf["f"].fit(X_cps_r, y_cps_r).predict(
        X_test)  # Probabilities cost RejectionSampling
    clf["c_u"] = clf["f"].fit(X_u, y_u).predict(X_test)  # Probabilities undersampling

measures = {"f1": f1_score, "pre": precision_score,
            "rec": recall_score, "acc": accuracy_score}

my_measures = [
    {"name": "acc", "f": accuracy_score},{"name": "pre", "f": precision_score},
    {"name": "rec", "f": recall_score},{"name": "f1", "f": f1_score},
]

results = pa.DataFrame(columns=measures.keys())

# Evaluate each model in classifiers
for clf in classifiers:
    results.loc[clf['name']] = [measures[measure](y_test, clf["c"]) for measure in measures.keys()]
    results.loc[clf['name'] + "-O"] = [measures[measure](y_test, clf["c_o"]) for measure in measures.keys()]
    results.loc[clf['name'] + "-R"] = [measures[measure](y_test, clf["c_r"]) for measure in measures.keys()]
    results.loc[clf['name'] + "-U"] = [measures[measure](y_test, clf["c_u"]) for measure in measures.keys()]

results["sav_score"] = np.zeros(results.shape[0])
results["cost_loss"] = np.zeros(results.shape[0])

for clf in classifiers:
    results["sav_score"].loc[clf['name']] = savings_score(y_test, clf["c"], cost_mat_test)
    results["cost_loss"].loc[clf['name']] = cost_loss(y_test, clf["c"], cost_mat_test)
    # Evaluate cost Oversampling
    results["sav_score"].loc[clf['name'] + "-O"] = savings_score(y_test, clf["c_o"], cost_mat_test)
    results["cost_loss"].loc[clf['name'] + "-O"] = cost_loss(y_test, clf["c_o"], cost_mat_test)
    # Evaluate cost RejectionSampling
    results["sav_score"].loc[clf['name'] + "-R"] = savings_score(y_test, clf["c_r"], cost_mat_test)
    results["cost_loss"].loc[clf['name'] + "-R"] = cost_loss(y_test, clf["c_r"], cost_mat_test)
    # Evaluate undersampling
    results["sav_score"].loc[clf['name'] + "-U"] = savings_score(y_test, clf["c_u"], cost_mat_test)
    results["cost_loss"].loc[clf['name'] + "-U"] = cost_loss(y_test, clf["c_u"], cost_mat_test)

print(results)

# ci_models = classifiers.keys()
#
# for model in list(ci_models):
#     # ---------------Cost Minimazation-----------------------
#     # Model BMR
#     classifiers[model + "-BMR"] = {"f": BayesMinimumRiskClassifier()}  # BMR
#     # Fit
#     classifiers[model + "-BMR"]["f"].fit(y_test, classifiers[model]["p"])
#     # Predict
#     classifiers[model + "-BMR"]["c"] = classifiers[model + "-BMR"]["f"].predict(classifiers[model]["p"], cost_mat_test)
#     # Evaluate
#     results.loc[model + "-BMR"] = 0
#     results.loc[model + "-BMR", measures.keys()] = \
#         [measures[measure](y_test, classifiers[model + "-BMR"]["c"]) for measure in measures.keys()]
#     results["sav_score"].loc[model + "-BMR"] = savings_score(y_test, classifiers[model + "-BMR"]["c"], cost_mat_test)
#     results["cost_loss"].loc[model + "-BMR"] = cost_loss(y_test, classifiers[model + "-BMR"]["c"], cost_mat_test)
#     # Model TO
#     classifiers[model + "-TO"] = {"f": ThresholdingOptimization()}  #
#     # Fit
#     classifiers[model + "-TO"]["f"].fit(classifiers[model]["ptrain"], cost_mat_train, y_train)  # Classifier TO
#     # Predict
#     classifiers[model + "-TO"]["c"] = classifiers[model + "-TO"]["f"].predict(classifiers[model]["p"])  # Classifier TO
#     # Evaluate
#     results.loc[model + "-TO"] = 0
#     results.loc[model + "-TO", measures.keys()] = \
#         [measures[measure](y_test, classifiers[model + "-TO"]["c"]) for measure in measures.keys()]
#     results["sav_score"].loc[model + "-TO"] = savings_score(y_test, classifiers[model + "-TO"]["c"], cost_mat_test)
#     results["cost_loss"].loc[model + "-TO"] = cost_loss(y_test, classifiers[model + "-TO"]["c"], cost_mat_test)
#
# classifiers["CSRP"] = {"f": CostSensitiveRandomPatchesClassifier(combination='weighted_voting')}
# # Fit
# classifiers["CSRP"]["f"].fit(X_train, y_train, cost_mat_train)
# # Predict
# classifiers["CSRP"]["c"] = classifiers["CSRP"]["f"].predict(X_test)
# # Evaluate
# results.loc["CSRP"] = 0
# results.loc["CSRP", measures.keys()] = \
#     [measures[measure](y_test, classifiers["CSRP"]["c"]) for measure in measures.keys()]
# results["sav_score"].loc["CSRP"] = savings_score(y_test, classifiers["CSRP"]["c"], cost_mat_test)
# results["cost_loss"].loc["CSRP"] = cost_loss(y_test, classifiers["CSRP"]["c"], cost_mat_test)
#
# CostSensitiveClassifiers = {"RFC": {"f": CostSensitiveRandomPatchesClassifier()}}
#
# for model in CostSensitiveClassifiers.keys():
#     CostSensitiveClassifiers[model]["f"].fit(X_train, y_train, cost_mat_train)
#     # Predict
#     CostSensitiveClassifiers[model]["f"] = CostSensitiveClassifiers[model]["f"].predict(X_test)
#     print(savings_score(y_test, CostSensitiveClassifiers[model]["f"], cost_mat_test))
#     print(cost_loss(y_test, CostSensitiveClassifiers[model]["f"], cost_mat_test))
#
# print(results)
