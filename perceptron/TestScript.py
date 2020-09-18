import sys
import numpy as np
sys.path.append("/home/elkin/university/gradSchool/Fall202/CS472/CS472")
from tools import arff, splitData
import perceptron


## load data in the terminal ##

# arff_path = r"training/data_banknote_authentication.arff"
arff_path = r"training/linsep2nonorigin.arff"
dataRaw = arff.Arff(arff=arff_path, label_count=1)
data = dataRaw.data[:,0:-1]
labels = dataRaw.data[:,-1].reshape(-1,1)
trainingSet,trainingLabels,testSet,testLabels= splitData.getSplitData(data,labels,0.7)

# print(data,labels)
splitData.printDataNicely(trainingSet,trainingLabels,testSet,testLabels)


## Define the initial Parameters ##
LR = 0.1
DET = 10
SHUFFLE = False
IW = [0,0,0]


PClass = perceptron.PerceptronClassifier(lr=LR,shuffle=SHUFFLE, deterministic=DET,initial_weights=IW)
# PClass = perceptron.PerceptronClassifier(lr=LR,shuffle=SHUFFLE, deterministic=DET)
# PClass.fit(trainingSet,trainingLabels)
# Accuracy = PClass.score(testSet,testLabels)
PClass.fit(data,labels)
Accuracy = PClass.score(data,labels)
print("Accuray = [{:.2f}]".format(Accuracy))
print("Final Weights =",PClass.get_weights())
