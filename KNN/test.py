import sys
import numpy as np
import random
import matplotlib.pyplot as plt
## change Path ##
sys.path.append("/home/elkin/university/gradSchool/Fall2020/CS472/CS472")
from tools import arff
import itertools
from sklearn.model_selection import train_test_split
from KNN import KNNClassifier


print("*******************PART 1************************************")

arff_path_train = r"training/seismic-bumps_train.arff"
arff_path_test = r"training/seismic-bumps_test.arff"
mat = arff.Arff(arff_path_train)
mat2 = arff.Arff(arff_path_test)
arrayTypes = mat.attr_types
del arrayTypes[-1]
# print(mat.data)
train_data = mat.data[:,0:-1]
train_labels = mat.data[:,-1].reshape(-1,1)


test_data = mat2.data[:,0:-1]
test_labels = mat2.data[:,-1].reshape(-1,1)


KNN = KNNClassifier(columntype= arrayTypes, labelType ='classification' ,weight_type='inverse_distance',k=3)
KNN.fit(train_data,train_labels)
pred = KNN.predict(test_data)
score = KNN.score(test_data,test_labels)
np.savetxt("seismic-bump-prediction.csv",pred,delimiter=',',fmt="%i")
print(score)
