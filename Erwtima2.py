# Machine Learning
# Bozas Aristeidis
# AM: 740
# Dermentzoglou Ioannis
# AM: 743

import matplotlib.pyplot as plt
import numpy as nu
import pandas as p
import seaborn as sns
from IPython.core.pylabtools import figsize
from costcla.metrics import savings_score, cost_loss, binary_classification_metrics
from costcla.models import BayesMinimumRiskClassifier, ThresholdingOptimization, CostSensitiveRandomForestClassifier
from costcla.sampling import cost_sampling, undersampling
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

Data = []
target = []
data = "./Datasets/heart.txt"
heart_data = p.read_csv(data, delimiter=" ")  # load StatLog(heart) dataset

# Iris data and target
array_heart = heart_data.values
heart_data = array_heart[:, 0:12]
heart_target = array_heart[:, 13]
# set target to binary 0, 1 instead 1,2
cost = []
for n, i in enumerate(heart_target):
    if i == 1:
        heart_target[n] = 0
        cost.append([1, 5, 0, 0])
    elif i == 2:
        heart_target[n] = 1
        cost.append([1, 5, 0, 0])
# for i in range(0,269):
# cost.append([1,5,0,0])
cost_mat = nu.array(cost)
# Data is the table with all dataset data
Data.append(heart_data)
# Target is the table with all dataset targets
target.append(heart_target)

# Table with the labels for each dataset
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = \
    train_test_split(heart_data, heart_target, cost_mat, test_size=0.33, random_state=0)

# Cost sampling
X_cps_o, y_cps_o, cost_mat_cps_o = cost_sampling(X_train, y_train, cost_mat_train, method='OverSampling')
X_cps_r, y_cps_r, cost_mat_cps_r = cost_sampling(X_train, y_train, cost_mat_train, method='RejectionSampling')
X_u, y_u, cost_mat_u = undersampling(X_train, y_train, cost_mat_train)

# Calibrated models
classifiers = {"RF": {"f": CalibratedClassifierCV(RandomForestClassifier(random_state=0), cv=10, method='sigmoid')},
               "LSVM": {"f": CalibratedClassifierCV(LinearSVC(random_state=0), cv=10, method='sigmoid')},
               "GNB": {"f": CalibratedClassifierCV(GaussianNB(), cv=10, method='sigmoid')},
               }

for model in classifiers.keys():
    # Fit
    classifiers[model]["f"].fit(X_train, y_train)
    # Predict
    classifiers[model]["c"] = classifiers[model]["f"].predict(X_test)
    classifiers[model]["p"] = classifiers[model]["f"].predict_proba(X_test)
    classifiers[model]["ptrain"] = classifiers[model]["f"].predict_proba(X_train)
    # Sampling

    classifiers[model]["c_o"] = classifiers[model]["f"].fit(X_cps_o, y_cps_o).predict(
        X_test)  # Probabilities cost Oversampling
    classifiers[model]["c_o_p"] = classifiers[model]["f"].fit(X_cps_o, y_cps_o).predict_proba(
        X_test)  # Probabilities cost Oversampling

    classifiers[model]["c_r"] = classifiers[model]["f"].fit(X_cps_r, y_cps_r).predict(
        X_test)  # Probabilities cost RejectionSampling
    classifiers[model]["c_r_p"] = classifiers[model]["f"].fit(X_cps_r, y_cps_r).predict_proba(
        X_test)  # Probabilities cost RejectionSampling

    classifiers[model]["c_u"] = classifiers[model]["f"].fit(X_u, y_u).predict(X_test)  # Probabilities undersampling
    classifiers[model]["c_u_p"] = classifiers[model]["f"].fit(X_u, y_u).predict_proba(
        X_test)  # Probabilities cost undersampling

# a = binary_classification_metrics(y_test, classifiers[model]["c_r"], classifiers[model]["c_r_p"][:, 1])
# print(a)

measures = {"f1": f1_score, "pre": precision_score,
            "rec": recall_score, "acc": accuracy_score}
results = p.DataFrame(columns=measures.keys())

# Evaluate each model in classifiers
for model in classifiers.keys():
    results.loc[model] = [measures[measure](y_test, classifiers[model]["c"]) for measure in measures.keys()]
    results.loc[model + "-O"] = [measures[measure](y_test, classifiers[model]["c_o"]) for measure in measures.keys()]
    results.loc[model + "-R"] = [measures[measure](y_test, classifiers[model]["c_r"]) for measure in measures.keys()]
    results.loc[model + "-U"] = [measures[measure](y_test, classifiers[model]["c_u"]) for measure in measures.keys()]

results["sav_score"] = nu.zeros(results.shape[0])
results["cost_loss"] = nu.zeros(results.shape[0])
results["brier_loss"] = nu.zeros(results.shape[0])
results["auc"] = nu.zeros(results.shape[0])
for model in classifiers.keys():
    # Evaluate RF,LSVM,GNB
    results["sav_score"].loc[model] = savings_score(y_test, classifiers[model]["c"], cost_mat_test)
    results["cost_loss"].loc[model] = cost_loss(y_test, classifiers[model]["c"], cost_mat_test)
    tempO = binary_classification_metrics(y_test, classifiers[model]["c"],
                                          classifiers[model]["p"][:, 1])  # various statistics
    results["brier_loss"].loc[model] = tempO["brier_loss"]
    results["auc"].loc[model] = tempO["auc"]
    # Evaluate cost Oversampling
    results["sav_score"].loc[model + "-O"] = savings_score(y_test, classifiers[model]["c_o"], cost_mat_test)
    results["cost_loss"].loc[model + "-O"] = cost_loss(y_test, classifiers[model]["c_o"], cost_mat_test)
    tempO = binary_classification_metrics(y_test, classifiers[model]["c_o"],
                                          classifiers[model]["c_o_p"][:, 1])  # various statistics
    results["brier_loss"].loc[model + "-O"] = tempO["brier_loss"]
    results["auc"].loc[model + "-O"] = tempO["auc"]
    # Evaluate cost RejectionSampling
    results["sav_score"].loc[model + "-R"] = savings_score(y_test, classifiers[model]["c_r"], cost_mat_test)
    results["cost_loss"].loc[model + "-R"] = cost_loss(y_test, classifiers[model]["c_r"], cost_mat_test)
    tempR = binary_classification_metrics(y_test, classifiers[model]["c_r"],
                                          classifiers[model]["c_r_p"][:, 1])  # various statistics
    results["brier_loss"].loc[model + "-R"] = tempR["brier_loss"]
    results["auc"].loc[model + "-R"] = tempO["auc"]
    # Evaluate undersampling
    results["sav_score"].loc[model + "-U"] = savings_score(y_test, classifiers[model]["c_u"], cost_mat_test)
    results["cost_loss"].loc[model + "-U"] = cost_loss(y_test, classifiers[model]["c_u"], cost_mat_test)
    tempU = binary_classification_metrics(y_test, classifiers[model]["c_u"],
                                          classifiers[model]["c_u_p"][:, 1])  # various statistics
    results["brier_loss"].loc[model + "-U"] = tempU["brier_loss"]
    results["auc"].loc[model + "-U"] = tempO["auc"]

ci_models = classifiers.keys()

for model in list(ci_models):
    # ---------------Cost Minimazation-----------------------
    # Model BMR:BayesMinimumRiskClassifier
    classifiers[model + "-BMR"] = {"f": BayesMinimumRiskClassifier()}  # BMR
    # Fit
    classifiers[model + "-BMR"]["f"].fit(y_test, classifiers[model]["p"])
    # Predict
    classifiers[model + "-BMR"]["c"] = classifiers[model + "-BMR"]["f"].predict(classifiers[model]["p"], cost_mat_test)
    # Evaluate
    results.loc[model + "-BMR"] = 0
    results.loc[model + "-BMR", measures.keys()] = \
        [measures[measure](y_test, classifiers[model + "-BMR"]["c"]) for measure in measures.keys()]
    results["sav_score"].loc[model + "-BMR"] = savings_score(y_test, classifiers[model + "-BMR"]["c"], cost_mat_test)
    results["cost_loss"].loc[model + "-BMR"] = cost_loss(y_test, classifiers[model + "-BMR"]["c"], cost_mat_test)
    tempBMR = binary_classification_metrics(y_test, classifiers[model + "-BMR"]["c"],
                                            classifiers[model]["p"][:, 0])  # various statistics
    results["brier_loss"].loc[model + "-BMR"] = tempBMR["brier_loss"]
    results["auc"].loc[model + "-BMR"] = tempBMR["auc"]
    # Model TO:ThresholdingOptimization
    classifiers[model + "-TO"] = {"f": ThresholdingOptimization()}  #
    # Fit
    classifiers[model + "-TO"]["f"].fit(classifiers[model]["ptrain"], cost_mat_train, y_train)  # Classifier TO
    # Predict
    classifiers[model + "-TO"]["c"] = classifiers[model + "-TO"]["f"].predict(classifiers[model]["p"])  # Classifier TO
    # Evaluate
    results.loc[model + "-TO"] = 0
    results.loc[model + "-TO", measures.keys()] = \
        [measures[measure](y_test, classifiers[model + "-TO"]["c"]) for measure in measures.keys()]
    results["sav_score"].loc[model + "-TO"] = savings_score(y_test, classifiers[model + "-TO"]["c"], cost_mat_test)
    results["cost_loss"].loc[model + "-TO"] = cost_loss(y_test, classifiers[model + "-TO"]["c"], cost_mat_test)
    tempTO = binary_classification_metrics(y_test, classifiers[model + "-TO"]["c"],
                                           classifiers[model]["p"][:, 0])  # various statistics
    results["brier_loss"].loc[model + "-TO"] = tempTO["brier_loss"]
    results["auc"].loc[model + "-TO"] = tempTO["auc"]
    print(tempTO)

CostSensitiveClassifiers = {"RFC": {"f": CostSensitiveRandomForestClassifier()}}
label = ["RFC"]

for model in CostSensitiveClassifiers.keys():
    CostSensitiveClassifiers[model]["f"].fit(X_train, y_train, cost_mat_train)
    # Predict
    CostSensitiveClassifiers[model]["c"] = CostSensitiveClassifiers[model]["f"].predict(X_test)
    CostSensitiveClassifiers[model]["c_p"] = CostSensitiveClassifiers[model]["f"].predict_proba(X_test)
    results.loc[model, measures.keys()] = \
        [measures[measure](y_test, CostSensitiveClassifiers[model]["c"]) for measure in measures.keys()]
    results["sav_score"].loc[model] = savings_score(y_test, CostSensitiveClassifiers[model]["c"], cost_mat_test)
    results["cost_loss"].loc[model] = cost_loss(y_test, CostSensitiveClassifiers[model]["c"], cost_mat_test)
    temp = binary_classification_metrics(y_test, CostSensitiveClassifiers[model]["c"],
                                         CostSensitiveClassifiers[model]["c_p"][:, 0])  # various statistics
    results["brier_loss"].loc[model] = temp["brier_loss"]
    results["auc"].loc[model] = temp["auc"]

print(results)
results = results.sort_values(['cost_loss'])
colors = sns.color_palette()
ind = nu.arange(results.shape[0])
figsize(10, 5)
ax = plt.subplot(111)
b = ax.bar(ind - 0.3, results['cost_loss'], 0.6, label='cost_loss', color=colors[0])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim([-0.5, ind[-1] + .5])
ax.set_xticks(ind)
ax.set_xticklabels(results.index)
plt.show()

colors = sns.color_palette()
ind = nu.arange(results.shape[0])
figsize(10, 5)
ax = plt.subplot(111)
b = ax.bar(ind - 0.3, results['brier_loss'], 0.6, label='brier_loss', color=colors[0])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim([-0.5, ind[-1] + .5])
ax.set_xticks(ind)
ax.set_xticklabels(results.index)
plt.show()

colors = sns.color_palette()
ind = nu.arange(results.shape[0])
figsize(10, 5)
ax = plt.subplot(111)
b = ax.bar(ind - 0.3, results['auc'], 0.6, label='auc', color=colors[0])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim([-0.5, ind[-1] + .5])
ax.set_xticks(ind)
ax.set_xticklabels(results.index)
plt.show()

colors = sns.color_palette()
ind = nu.arange(results.shape[0])

figsize(10, 5)
ax = plt.subplot(111)
l = ax.plot(ind, results["f1"], "-o", label='F1Score', color=colors[2])
C = ax.plot(ind, results["acc"], "-o", label='Accuracy', color=colors[1])
b = ax.bar(ind - 0.3, results['sav_score'], 0.6, label='Savings', color=colors[0])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim([-0.5, ind[-1] + .5])
ax.set_xticks(ind)
ax.set_xticklabels(results.index)
plt.show()
