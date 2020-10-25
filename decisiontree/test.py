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


# data = np.array([[0,0,0],
#         [1,1,0],
#         [1,2,1],
#         [0,2,1],
#         [0,1,0],
#         [0,1,1],
#         [1,0,1],
#         [0,1,0],
#         [1,0,0]])
# labels = np.array([[0],
#           [1],
#           [2],
#           [0],
#           [2],
#           [0],
#           [2],
#           [2],
#           [1]])
# counts = [2,3,2,3] ## this is so you know how many types for each column

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
# pred = DTClass.predict(data2)
Acc = DTClass.score(data2,labels2)
# np.savetxt("pred_lenses.csv",pred,delimiter=",")
print("Accuracy = [{:.2f}]".format(Acc))



# DTClass = DTClassifier(counts)
# DTClass.fit(data,labels)
# score = DTClass.score(data,labels)
# print("Score",score)




# data = np.array([[1,2,1,4],
#         [1,1,1,4],
#         [1,2,1,3]])
# print(DTClass.noMoreAttributes(data))
#
# splitXindxRange = data.shape[1]-1
# X_transposed = data[:, np.r_[splitXindxRange]]
# print(X_transposed)
