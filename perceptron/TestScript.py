import sys
import numpy as np
import random
sys.path.append("/home/elkin/university/gradSchool/Fall202/CS472/CS472")
from tools import arff, splitData, generatePerceptronData
import perceptron


## Part 1 ##

# arff_path = r"training/data_banknote_authentication.arff"
arff_path = r"training/linsep2nonorigin.arff"
dataRaw = arff.Arff(arff=arff_path, label_count=1)
data = dataRaw.data[:,0:-1]
labels = dataRaw.data[:,-1].reshape(-1,1)
trainingSet,trainingLabels,testSet,testLabels= splitData.getSplitData(data,labels,0.7)


## Define the initial Parameters ##
LR = 0.1
DET = 10
SHUFFLE = False
# IW = [0,0,0,0,0]
IW = [0,0,0]



PClass = perceptron.PerceptronClassifier(lr=LR,shuffle=SHUFFLE, deterministic=DET,initial_weights=IW)
PClass.fit(data,labels)
Accuracy = PClass.score(data,labels)
print("Accuray = [{:.2f}]".format(Accuracy))
print("Final Weights =",PClass.get_weights())


##Part 2

## Uncomment both to generate the linear and not linear datasets
# generatePerceptronData.generateDataSet("linearSeparableDataSet",True,(8,),[-1,1])
# generatePerceptronData.generateDataSet("nonLinSeparableDataSet",False,(8,),[-1,1])

## Part 3 ##

ls_path_part3 = r"training/linearSeparableDataSet.arff"
not_ls_path_part3 = r"training/notLinearSeparableDataSet.arff"

dataRaw_ls = arff.Arff(arff=ls_path_part3, label_count=1)
dataRaw_nls = arff.Arff(arff=not_ls_path_part3, label_count=1)

data_ls = dataRaw_ls.data[:,0:-1]
labels_ls = dataRaw_ls.data[:,-1].reshape(-1,1)
data_nls = dataRaw_nls.data[:,0:-1]
labels_nls = dataRaw_nls.data[:,-1].reshape(-1,1)


## Define the initial Parameters ##
DET3 = 10
SHUFFLE3 = False
IW_ls = [0,0,0]
IW_nls = [0,0,0,0]
LR_array = np.arange(1,100,1).tolist()
# print(LR_array)
# LRs = 0.1

print("Linear data set")
for LRs in LR_array:
    LRs= round(LRs,1)
    PClass_ls = perceptron.PerceptronClassifier(lr=LRs,shuffle=SHUFFLE3, deterministic=DET3,initial_weights=IW_ls)
    PClass_ls.fit(data_ls,labels_ls)
    Accuracy_ls = PClass.score(data_ls,labels_ls)
    print("LR = " , LRs)
    print("Accuray = [{:.2f}]".format(Accuracy_ls))
    print("Final Weights =",PClass_ls.get_weights())

print("Non linear data set")
for LRs in LR_array:
    LRs= round(LRs,1)
    PClass_nls = perceptron.PerceptronClassifier(lr=LRs,shuffle=SHUFFLE3, deterministic=DET3,initial_weights=IW_nls)
    PClass_nls.fit(data_nls,labels_nls)
    Accuracy_nls = PClass_nls.score(data_nls,labels_nls)
    print("LR = " , LRs)
    print("Accuray = [{:.2f}]".format(Accuracy_nls))
    print("Final Weights =",PClass_nls.get_weights())
