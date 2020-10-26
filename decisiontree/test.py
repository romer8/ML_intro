import sys
import numpy as np
import random
import matplotlib.pyplot as plt
## change Path ##
sys.path.append("/home/elkin/university/gradSchool/Fall2020/CS472/CS472")
from tools import arff
import itertools
from decisiontree import DTClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


print("*******************PART 1************************************")

arff_path_train = r"training/lenses.arff"
arff_path_test = r"training/all_lenses.arff"
mat = arff.Arff(arff_path_train)

counts = [] ## this is so you know how many types for each column

for i in range(mat.data.shape[1]):
       counts += [mat.unique_value_count(i)]
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
# print(data)
# print(labels)
DTClass = DTClassifier(counts)
DTClass.fit(data,labels)
mat2 = arff.Arff(arff_path_test)
data2 = mat2.data[:,0:-1]
labels2 = mat2.data[:,-1].reshape(-1,1)
# print(data2)
# print(labels2)
pred = DTClass.predict(data2)
Acc = DTClass.score(data2,labels2)

# np.savetxt("pred_lenses.csv",pred,delimiter=",")
print("Accuracy = [{:.2f}]".format(Acc))

print("*******************EVALUATION************************************")
arff_path_train = r"training/zoo.arff"
arff_path_test = r"training/all_zoo.arff"
mat = arff.Arff(arff_path_train)

counts = [] ## this is so you know how many types for each column

for i in range(mat.data.shape[1]):
       counts += [mat.unique_value_count(i)]
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
# print(data)
# print(labels)
DTClass = DTClassifier(counts)
DTClass.fit(data,labels)
mat2 = arff.Arff(arff_path_test)
data2 = mat2.data[:,0:-1]
labels2 = mat2.data[:,-1].reshape(-1,1)
# print(data2)
# print(labels2)
pred = DTClass.predict(data2)
Acc = DTClass.score(data2,labels2)

# np.savetxt("pred_zoo.csv",pred,delimiter=",")
print("Accuracy = [{:.2f}]".format(Acc))

print("*******************PART 2************************************")

arff_path= r"training/cars.arff"
mat = arff.Arff(arff_path)

# counts = [] ## this is so you know how many types for each column
# for i in range(mat.data.shape[1]):
#        counts += [mat.unique_value_count(i)]
data2 = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
data = np.concatenate((data2, labels), axis = 1)
# print(data)
""" now K-folding """
# prepare cross validation
kfold = KFold(n_splits=10, random_state=None, shuffle=True)
scores = []

for train, test in kfold.split(data):
    print("DATA SIZE ", len(data[train]))
    print("TEST SIZE ", len(data[test]))
	# print('train: %s, test: %s' % (data2[train], data2[test]))
    columns = len(data[train][0])
    data_transposed = data[train].T
    counts = []
    for column in range(0, columns):
        unique = np.unique(data_transposed[column])
        counts.append(len(unique))
    print("COUNTS TRAIN",counts)
    DTClass = DTClassifier(counts)
    splitXindxRange = data[train].shape[1] - 1
    labelsK = data[train][:, np.r_[splitXindxRange]]
    dataK = data[train][:, np.r_[0:splitXindxRange]]
    labelsTestK = data[test][:, np.r_[splitXindxRange]]
    dataTestK = data[test][:, np.r_[0:splitXindxRange]]
    DTClass.fit(dataK,labelsK)

    columnsTest = len(data[test][0])
    dataTest_transposed = data[test].T
    countsTest = []
    for column in range(0, columnsTest):
        unique = np.unique(dataTest_transposed[column])
        countsTest.append(len(unique))
    print("COUNTS TEST",countsTest)

    scores.append(DTClass.score(dataTestK,labelsTestK))

    # break
print("Scores ", scores)
