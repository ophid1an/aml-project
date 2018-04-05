import time
from sklearn import datasets
from sklearn import tree
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

data1="./Datasets/iris.csv"
data2="./Datasets/wine.csv"
data3="./Datasets/wdbc.csv"
data4="./Datasets/balance-scale.csv"
data5="./Datasets/hayesroth.csv"
data6="./Datasets/haberman.csv"
data7="./Datasets/liverdisorder.csv"
data8="./Datasets/data_banknote_authentication.csv"
data9="./Datasets/ionosphere.csv"
data10="./Datasets/cmc.csv"

iris_data= pa.read_csv(data1) #load iris dataset
wine_data= pa.read_csv(data2) #load wine dataset
bcancer_data=pa.read_csv(data3)#load breast cancer dataset
balance_data=pa.read_csv(data4)#load balance-scale dataset
hayesroth_data=pa.read_csv(data5)#load hayes-roth dataset
haberman_data=pa.read_csv(data6)#load haberman survival dataset
liver_data=pa.read_csv(data7)#load Liver disorders  dataset
bank_data=pa.read_csv(data8)#load banknote authentication dataset
ionosphere_data=pa.read_csv(data9)#load ionosphere dataset
cmc_data=pa.read_csv(data10)#load Contraceptive Method Choice dataset

Data=[]
target=[]
#Iris data and target
array_iris = iris_data.values
iris_data = array_iris [:,0:3]
iris_target = array_iris [:,4]
#Wine data and target
array_wine = wine_data.values
wine_data = array_wine [:,1:12]
wine_target = array_wine [:,0]
#Breast cancer data and target
array_bcancer=bcancer_data.values
bcancer_data = array_bcancer [:,2:31]
bcancer_target = array_bcancer [:,1]
#Balance-scale data and target
array_balance=balance_data.values
balance_data= array_balance [:,1:4]
balance_target= array_balance [:,0]
#hayes-roth data and target
array_hayesroth=hayesroth_data.values
hayesroth_data= array_hayesroth [:,0:4]
hayesroth_target= array_hayesroth [:,5]
#Haberman survival data and target
array_haberman=haberman_data.values
haberman_data= array_haberman [:,0:2]
haberman_target= array_haberman [:,3]
#Liver Disorder  data and target
array_liver=liver_data.values
liver_data= array_liver [:,0:5]
liver_target= array_liver [:,6]
#Chess(Rook vs King) data and target
array_bank=bank_data.values
bank_data= array_bank[:,0:3]
bank_target= array_bank [:,4]
#Ionosphere data and target
array_ionosphere=ionosphere_data.values
ionosphere_data= array_ionosphere[:,0:33]
ionosphere_target= array_ionosphere[:,34]
#cmc data and target
array_cmc=cmc_data.values
cmc_data= array_cmc[:,0:8]
cmc_target= array_cmc[:,9]
#Data is the table with all dataset data
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
#Target is the table with all dataset targets
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
#Table with the labels for each dataset
Data_label=["Iris dataset","Wine Dataset","Breast Cancer Dataset","Balance-Scale Dataset","Hayes-Roth Dataset",
            "Haberman Survival Dataset","Liver Disorder Dataset","Bank-Note Dataset","Ionosphere Dataset","Contraceptive Method Choice Dataset"]

classifiers = []
#Tree Classifier
dt = tree.DecisionTreeClassifier()
classifiers.append([dt, "tree"])

#-----Manipulating the training examples-------
#Bagging
bagged_dt = BaggingClassifier(tree.DecisionTreeClassifier(),n_estimators=100)
classifiers.append([bagged_dt, "bagged tree"])
#Boosting
ada = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),n_estimators=100)
classifiers.append([ada, "AdaBoost-ed tree"])

gbm = GradientBoostingClassifier(n_estimators=100)
classifiers.append([gbm, "gradient boosting tree"])
#-----Manipulating the target variable------
#One vs One
ovo = OneVsOneClassifier(tree.DecisionTreeClassifier())
classifiers.append([ovo, "one-vs-one tre"])
#One vs Rest
ovr = OneVsRestClassifier(tree.DecisionTreeClassifier())
classifiers.append([ovr, "one-vs-rest tree"])
#-----Injecting randomness------
#Random forest
rf = RandomForestClassifier()
classifiers.append([rf,"Random forest"])
#-----Manipulating Features------
#RandomPatches
bagged_dt_mf = BaggingClassifier(tree.DecisionTreeClassifier(),n_estimators=100,max_samples=0.7, max_features=0.7)
classifiers.append([bagged_dt_mf, "bagged tree random patches"])


for i in range(len(Data)):
    print("------ %20s ------ " % (Data_label[i]))
    for classifier, label in classifiers:
        start = time.time()
        scores = cross_val_score(classifier, Data[i], target[i], cv=10)
        stop = time.time()
        print("%20s accuracy: %0.2f (+/- %0.2f), time:%.4f" % (label, scores.mean(), scores.std() * 2, stop - start))
