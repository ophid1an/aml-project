import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as nu
import pandas as p
from costcla.models import BayesMinimumRiskClassifier
from costcla.models import ThresholdingOptimization
from costcla.metrics import savings_score
from sklearn.cross_validation import train_test_split
from costcla.datasets import load_creditscoring1
from costcla.metrics import savings_score, cost_loss, brier_score_loss,binary_classification_metrics

from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import seaborn as sns


data1 = load_creditscoring1()
Data=[]
target=[]
data="./Datasets/heart.txt"
heart_data= p.read_csv(data,delimiter=" ") #load StatLog(heart) dataset

#Iris data and target
array_heart = heart_data.values
heart_data = array_heart [:,0:12]
heart_target=array_heart[:,13]
#set target to binary 0, 1 instead 1,2
cost=[]
for n, i in enumerate(heart_target):
    if i == 1:
        heart_target[n] = 0
        cost.append([5,5,0,0])
    elif i== 2:
        heart_target[n] = 1
        cost.append([5,5,0,0])

    cost_mat = nu.array(cost)
#Data is the table with all dataset data
Data.append(heart_data)
#Target is the table with all dataset targets
target.append(heart_target)

#Table with the labels for each dataset
Data_label=["Statlog(heart) dataset"]


#for i in range(0,269):
    #cost.append([1,5,0,0])


X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = \
train_test_split(heart_data, heart_target, cost_mat)
#X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = \
#train_test_split(data1.data, data1.target, data1.cost_mat)

#
classifiers = {"RF": {"f": RandomForestClassifier()},
               "DT": {"f": DecisionTreeClassifier()},
               "LR": {"f": LogisticRegression()}}

for model in classifiers.keys():
    # Fit
    classifiers[model]["f"].fit(X_train, y_train)
    # Predict
    classifiers[model]["c"] = classifiers[model]["f"].predict(X_test)
    classifiers[model]["p"] = classifiers[model]["f"].predict_proba(X_test)
    classifiers[model]["p_train"] = classifiers[model]["f"].predict_proba(X_train)

measures = {"f1": f1_score, "pre": precision_score,
            "rec": recall_score, "acc": accuracy_score}
results = p.DataFrame(columns=measures.keys())

# Evaluate each model in classifiers
for model in classifiers.keys():
    results.loc[model] = [measures[measure](y_test, classifiers[model]["c"]) for measure in measures.keys()]

print (results)



results["sav"] = nu.zeros(results.shape[0])
for model in classifiers.keys():
    results["sav"].loc[model] = savings_score(y_test, classifiers[model]["c"], cost_mat_test)

# TODO: plot results
print (results)




ci_models = classifiers.keys()

for model in list(ci_models):
    classifiers[model+"-BMR"] = {"f": BayesMinimumRiskClassifier()}
    # Fit
    classifiers[model+"-BMR"]["f"].fit(y_test, classifiers[model]["p"])
    # Calibration must be made in a validation set
    # Predict
    classifiers[model+"-BMR"]["c"] = classifiers[model+"-BMR"]["f"].predict(classifiers[model]["p"], cost_mat_test)
    # Evaluate
    results.loc[model+"-BMR"] = 0
    results.loc[model+"-BMR", measures.keys()] = \
    [measures[measure](y_test, classifiers[model+"-BMR"]["c"]) for measure in measures.keys()]
    results["sav"].loc[model+"-BMR"] = savings_score(y_test, classifiers[model+"-BMR"]["c"], cost_mat_test)

print (results)
colors = sns.color_palette()

ind = nu.arange(results.shape[0])
figsize(10, 5)
ax = plt.subplot(111)
l = ax.plot(ind, results["f1"], "-o", label='F1Score', color=colors[2])
b = ax.bar(ind-0.3, results['sav'], 0.6, label='Savings', color=colors[0])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim([-0.5, ind[-1]+.5])
ax.set_xticks(ind)
ax.set_xticklabels(results.index)
plt.show()

f = RandomForestClassifier(random_state=0).fit(X_train, y_train)
y_prob_train = f.predict_proba(X_train)
y_prob_test = f.predict_proba(X_test)
y_pred_test_rf = f.predict(X_test)


f_t = ThresholdingOptimization(calibration="true").fit(y_prob_train, cost_mat_train, y_train)

y_pred_test_rf_t = f_t.predict(y_prob_test)
y_pred_train_rf_t = f_t.predict(y_prob_train)
print(y_test)
print(y_pred_test_rf)
print(y_prob_test)
print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
print(savings_score(y_test, y_pred_test_rf_t, cost_mat_test))
