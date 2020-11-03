import sys
import numpy as np
import random
import matplotlib.pyplot as plt
## change Path ##
sys.path.append("/home/elkin/university/gradSchool/Fall2020/CS472/CS472")
from tools import arff, normalization
import itertools
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from KNN import KNNClassifier


# print("*******************PART 1************************************")
# print("*******************DEBUG************************************")
#
# arff_path_train = r"training/seismic-bumps_train.arff"
# arff_path_test = r"training/seismic-bumps_test.arff"
# mat = arff.Arff(arff_path_train)
# mat2 = arff.Arff(arff_path_test)
# arrayTypes = mat.attr_types
# del arrayTypes[-1]
# # print(mat.data)
# train_data = mat.data[:,0:-1]
# train_labels = mat.data[:,-1].reshape(-1,1)
#
#
# test_data = mat2.data[:,0:-1]
# test_labels = mat2.data[:,-1].reshape(-1,1)
#
#
# KNN = KNNClassifier(columntype= arrayTypes, labelType ='classification' ,weight_type='inverse_distance',k=3)
# KNN.fit(train_data,train_labels)
# pred = KNN.predict(test_data)
# score = KNN.score(test_data,test_labels)
# np.savetxt("seismic-bump-prediction.csv",pred,delimiter=',',fmt="%i")
# print(score)
#
# print("*******************EVALUATION************************************")
# arff_path_train = r"training/diabetes.arff"
# arff_path_test = r"training/diabetes_test.arff"
# mat = arff.Arff(arff_path_train)
# mat2 = arff.Arff(arff_path_test)
# arrayTypes = mat.attr_types
# del arrayTypes[-1]
# # print(mat.data)
# train_data = mat.data[:,0:-1]
# train_labels = mat.data[:,-1].reshape(-1,1)
#
#
# test_data = mat2.data[:,0:-1]
# test_labels = mat2.data[:,-1].reshape(-1,1)
#
#
# KNN = KNNClassifier(columntype= arrayTypes, labelType ='classification' ,weight_type='inverse_distance',k=3)
# KNN.fit(train_data,train_labels)
# pred = KNN.predict(test_data)
# score = KNN.score(test_data,test_labels)
# np.savetxt("diabetes-prediction.csv",pred,delimiter=',',fmt="%i")
# print(score)

print("*******************PART 2************************************")

arff_path_train = r"training/magic_training.arff"
arff_path_test = r"training/magic_test.arff"
mat = arff.Arff(arff_path_train)
mat2 = arff.Arff(arff_path_test)
arrayTypes = mat.attr_types
del arrayTypes[-1]
# print(mat.data)
train_data = mat.data[:,0:-1]
# train_data_norm = preprocessing.normalize(train_data, axis = 0, norm ='l1')
train_data_norm = normalization.normalizeData(train_data)
train_labels = mat.data[:,-1].reshape(-1,1)


test_data = mat2.data[:,0:-1]
# test_data_norm = preprocessing.normalize(test_data, axis=0,norm ='l1' )
test_data_norm = normalization.normalizeData(test_data)
test_labels = mat2.data[:,-1].reshape(-1,1)

# print(normalization.giveMaxValues(test_data_norm))
# print(normalization.giveMinValues(test_data_norm))

KNN = KNNClassifier(columntype= arrayTypes, labelType ='classification' ,weight_type='no_weight',k=3)
KNN.fit(train_data,train_labels)
# pred = KNN.predict(test_data)
score = KNN.score(test_data,test_labels)
print("*******************NOT NORMALIZED************************************")

print(score)


KNN = KNNClassifier(columntype= arrayTypes, labelType ='classification' ,weight_type='no_weight',k=3)
KNN.fit(train_data_norm,train_labels)
# pred = KNN.predict(test_data_norm)
score = KNN.score(test_data_norm,test_labels)
print("*******************NORMALIZED************************************")
print(score)

arrayScores = []
arrayK = []
for a in range(1, 17, 2):
    KNN = KNNClassifier(columntype= arrayTypes, labelType ='classification' ,weight_type='no_weight',k=a)
    KNN.fit(train_data_norm,train_labels)
    # pred = KNN.predict(test_data_norm)
    score = KNN.score(test_data_norm,test_labels)
    arrayScores.append(score)
    arrayK.append(a)
    print("*******************NORMALIZED************************************")
    print("K = ",a,score)

x = np.arange(len(arrayK))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
ax.bar(x - width/2, arrayScores, width, label='k')
ax.set_ylim([0.7,0.85])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('TS Accuracy')
ax.set_title('Accuracy for Different Values of "K"')
ax.set_xticks(x)
ax.set_xticklabels(arrayK)
ax.set_xlabel('k')
ax.legend()
save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/KNN/plots/kvsac2"
fig.savefig(save_path)
