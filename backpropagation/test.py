import sys
import numpy as np
import random
## change Path ##
sys.path.append("/home/elkin/university/gradSchool/Fall202/CS472/CS472")
from tools import arff, splitData, generatePerceptronData, graph_tools,list2csv
import itertools
import mlp


arff_path = r"training/linsep2nonorigin.arff"
# arff_path = r"training/iris.arff"
dataRaw = arff.Arff(arff=arff_path, label_count=1)
data = dataRaw.data[:,0:-1]
labels = dataRaw.data[:,-1].reshape(-1,1)
# print(labels)

## Define the initial Parameters ##
LR = 0.1
DET = 10
SHUFFLE = False
MOMENTUM = 0.5
VALIDATION_SIZE = 0.0



# mat = Arff("../datasets/linsep2nonorigin.arff")
# data = mat.data[:,0:-1]
# labels = mat.data[:,-1].reshape(-1,1)
# BClass = mlp.MLPClassifier(lr = LR,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, hidden_layer_widths = [2])
BClass = mlp.MLPClassifier(lr = LR,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, validationSize = VALIDATION_SIZE)
# testArray = np.array([[0,0],[0,1]])
# testArrayY = np.array([[1],[0]])
# BClass.fit(testArray,testArrayY)
BClass.fit(data,labels)
print(BClass.get_weights())
list2csv.write_to_csv(BClass.get_weights())
