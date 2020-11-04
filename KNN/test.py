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
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors,KNeighborsClassifier
from distython import HEOM
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
# # np.savetxt("seismic-bump-prediction.csv",pred,delimiter=',',fmt="%i")
# print("Score", score)
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
# # np.savetxt("diabetes-prediction.csv",pred,delimiter=',',fmt="%i")
# print("Score",score)
#
# print("*******************PART 2************************************")
#
# arff_path_train = r"training/magic_training.arff"
# arff_path_test = r"training/magic_test.arff"
# mat = arff.Arff(arff_path_train)
# mat2 = arff.Arff(arff_path_test)
# arrayTypes = mat.attr_types
# del arrayTypes[-1]
# # print(mat.data)
# train_data = mat.data[:,0:-1]
# # train_data_norm = preprocessing.normalize(train_data, axis = 0, norm ='l1')
# train_data_norm = normalization.normalizeData(train_data)
# train_labels = mat.data[:,-1].reshape(-1,1)
#
#
# test_data = mat2.data[:,0:-1]
# # test_data_norm = preprocessing.normalize(test_data, axis=0,norm ='l1' )
# test_data_norm = normalization.normalizeData(test_data)
# test_labels = mat2.data[:,-1].reshape(-1,1)
#
# # print(normalization.giveMaxValues(test_data_norm))
# # print(normalization.giveMinValues(test_data_norm))
#
# KNN = KNNClassifier(columntype= arrayTypes, labelType ='classification' ,weight_type='no_weight',k=3)
# KNN.fit(train_data,train_labels)
# # pred = KNN.predict(test_data)
# score = KNN.score(test_data,test_labels)
# print("*******************NOT NORMALIZED************************************")
#
# print("Score ",score)
#
#
# KNN = KNNClassifier(columntype= arrayTypes, labelType ='classification' ,weight_type='no_weight',k=3)
# KNN.fit(train_data_norm,train_labels)
# # pred = KNN.predict(test_data_norm)
# score = KNN.score(test_data_norm,test_labels)
# print("*******************NORMALIZED************************************")
# print("Score", score)
#
# arrayScores = []
# arrayK = []
# for a in range(1, 17, 2):
#     KNN = KNNClassifier(columntype= arrayTypes, labelType ='classification' ,weight_type='no_weight',k=a)
#     KNN.fit(train_data_norm,train_labels)
#     # pred = KNN.predict(test_data_norm)
#     score = KNN.score(test_data_norm,test_labels)
#     arrayScores.append(score)
#     arrayK.append(a)
#     print("K = ",a,score)
#
# x = np.arange(len(arrayK))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# ax.bar(x - width/2, arrayScores, width, label='k')
# ax.set_ylim([0.7,0.85])
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('TS Accuracy')
# ax.set_title('Accuracy for Different Values of "K"')
# ax.set_xticks(x)
# ax.set_xticklabels(arrayK)
# ax.set_xlabel('k')
# ax.legend()
# save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/KNN/plots/kvsac2"
# fig.savefig(save_path)


# print("*******************PART 3************************************")
#
# arff_path_train = r"training/housing_training.arff"
# arff_path_test = r"training/housing_test.arff"
# mat = arff.Arff(arff_path_train)
# mat2 = arff.Arff(arff_path_test)
# arrayTypes = mat.attr_types
# del arrayTypes[-1]
# scaler = preprocessing.MinMaxScaler()
#
# train_data = mat.data[:,0:-1]
# train_data_norm = normalization.normalizeData(train_data)
# # train_data_norm = scaler.fit_transform(train_data)
# train_labels = mat.data[:,-1].reshape(-1,1)
#
#
# test_data = mat2.data[:,0:-1]
# test_data_norm = normalization.normalizeData(test_data)
# # test_data_norm = scaler.fit_transform(test_data)
#
# test_labels = mat2.data[:,-1].reshape(-1,1)
#
# KNN = KNNClassifier(columntype= arrayTypes, labelType ='regression' ,weight_type='no_weight',k=3)
# KNN.fit(train_data_norm,train_labels)
# pred = KNN.predict(test_data_norm)
# score = KNN.score(test_data_norm,test_labels)
# print("Score",score)
# # neigh = KNeighborsRegressor(n_neighbors=3)
# # neigh.fit(train_data_norm, train_labels)
# # skscore = neigh.predict(test_data_norm)
# # print(skscore.reshape(skscore.shape[0],))
# # print(pred)
# arrayScores = []
# arrayK = []
# for a in range(1, 17, 2):
#     KNN = KNNClassifier(columntype= arrayTypes, labelType ='regression' ,weight_type='no_weight',k=a)
#     KNN.fit(train_data_norm,train_labels)
#     # pred = KNN.predict(test_data_norm)
#     score = KNN.score(test_data_norm,test_labels)
#     arrayScores.append(score)
#     arrayK.append(a)
#     print("K = ",a,score)
#
# x = np.arange(len(arrayK))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# ax.bar(x - width/2, arrayScores, width, label='k')
# # ax.set_ylim([0.7,0.85])
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('TS MSE')
# ax.set_title('MSE for Different Values of "K"')
# ax.set_xticks(x)
# ax.set_xticklabels(arrayK)
# ax.set_xlabel('k')
# ax.legend()
# save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/KNN/plots/kvsac3"
# fig.savefig(save_path)

# print("*******************PART 4************************************")
# print(" MAGIC DATA")
# arff_path_train = r"training/magic_training.arff"
# arff_path_test = r"training/magic_test.arff"
# mat = arff.Arff(arff_path_train)
# mat2 = arff.Arff(arff_path_test)
# arrayTypes = mat.attr_types
# del arrayTypes[-1]
# # print(mat.data)
# train_data = mat.data[:,0:-1]
# # train_data_norm = preprocessing.normalize(train_data, axis = 0, norm ='l1')
# train_data_norm = normalization.normalizeData(train_data)
# train_labels = mat.data[:,-1].reshape(-1,1)
#
#
# test_data = mat2.data[:,0:-1]
# # test_data_norm = preprocessing.normalize(test_data, axis=0,norm ='l1' )
# test_data_norm = normalization.normalizeData(test_data)
# test_labels = mat2.data[:,-1].reshape(-1,1)
# KNN = KNNClassifier(columntype= arrayTypes, labelType ='classification' ,weight_type='inverse_distance',k=3)
# KNN.fit(train_data_norm,train_labels)
# # pred = KNN.predict(test_data_norm)
# score = KNN.score(test_data_norm,test_labels)
# print("Score",score)
#
# arrayScores = []
# arrayK = []
# for a in range(1, 17, 2):
#     KNN = KNNClassifier(columntype= arrayTypes, labelType ='classification' ,weight_type='inverse_distance',k=a)
#     KNN.fit(train_data_norm,train_labels)
#     # pred = KNN.predict(test_data_norm)
#     score = KNN.score(test_data_norm,test_labels)
#     arrayScores.append(score)
#     arrayK.append(a)
#     print("K = ",a,score)
#
# x = np.arange(len(arrayK))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# ax.bar(x - width/2, arrayScores, width, label='k')
# ax.set_ylim([0.7,0.85])
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('TS Accuracy')
# ax.set_title('Accuracy for Different Values of "K"')
# ax.set_xticks(x)
# ax.set_xticklabels(arrayK)
# ax.set_xlabel('k')
# ax.legend()
# save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/KNN/plots/kvsac41"
# fig.savefig(save_path)
# print("HOSUING DATA")
#
# arff_path_train = r"training/housing_training.arff"
# arff_path_test = r"training/housing_test.arff"
# mat = arff.Arff(arff_path_train)
# mat2 = arff.Arff(arff_path_test)
# arrayTypes = mat.attr_types
# del arrayTypes[-1]
#
# train_data = mat.data[:,0:-1]
# train_data_norm = normalization.normalizeData(train_data)
# train_labels = mat.data[:,-1].reshape(-1,1)
#
#
# test_data = mat2.data[:,0:-1]
# test_data_norm = normalization.normalizeData(test_data)
# test_labels = mat2.data[:,-1].reshape(-1,1)
#
# KNN = KNNClassifier(columntype= arrayTypes, labelType ='regression' ,weight_type='inverse_distance',k=3)
# KNN.fit(train_data_norm,train_labels)
# # pred = KNN.predict(test_data_norm)
# score = KNN.score(test_data_norm,test_labels)
# print("Score ",score)
#
# arrayScores = []
# arrayK = []
# for a in range(1, 17, 2):
#     KNN = KNNClassifier(columntype= arrayTypes, labelType ='regression' ,weight_type='inverse_distance',k=a)
#     KNN.fit(train_data_norm,train_labels)
#     # pred = KNN.predict(test_data_norm)
#     score = KNN.score(test_data_norm,test_labels)
#     arrayScores.append(score)
#     arrayK.append(a)
#     print("K = ",a,score)
#
# x = np.arange(len(arrayK))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# ax.bar(x - width/2, arrayScores, width, label='k')
# # ax.set_ylim([0.7,0.85])
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('TS MSE')
# ax.set_title('MSE for Different Values of "K"')
# ax.set_xticks(x)
# ax.set_xticklabels(arrayK)
# ax.set_xlabel('k')
# ax.legend()
# save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/KNN/plots/kvsac42"
# fig.savefig(save_path)

# print("*******************PART 5************************************")
# nan_eqv = 12345
# arff_path_train = r"training/credit.arff"
# mat = arff.Arff(arff_path_train)
# arrayTypes = mat.attr_types
# del arrayTypes[-1]
# # print(arrayTypes)
# data = mat.data[:,0:-1]
# labels = mat.data[:,-1].reshape(-1,1)
# data_norm = normalization.normalizeData(data)
# # data_norm = data
# where_are_NaNs = np.isnan(data_norm)
# data_norm[where_are_NaNs] = nan_eqv
# X_train, X_test, y_train, y_test = train_test_split(data_norm, labels, test_size=0.10)
# print(len(data))
# print(len(X_test))
# categorical_ix = []
# for arrat in range(0,len(arrayTypes)):
#     if arrayTypes[arrat] == 'nominal':
#         categorical_ix.append(arrat)
#
#
# heom_metric = HEOM(X_train, categorical_ix, nan_equivalents = [nan_eqv])
# # Declare NearestNeighbor and link the metric
# neighbor = NearestNeighbors(metric = heom_metric.heom)
#
# # Fit the model which uses the custom distance metric
# neighbor.fit(X_train)
#
# """Function defining the VDM """
# def testOne (X_test,y_train,y_test,k):
#     preds = []
#     for dataX in X_test:
#         distances,indices = neighbor.kneighbors(dataX.reshape(1, -1), n_neighbors = k)
#         distances = distances[0]
#         indices = indices[0]
#         # print(distances)
#         # print(indices)
#         neighbors = []
#         uniquevals = np.unique(y_train)
#         # print(uniquevals)
#         sums =[]
#         for unq in uniquevals:
#             sum = 0
#             for indx, dist in zip(indices,distances):
#                 if y_train[indx][0] == unq:
#                     sum = sum + dist
#                 # neighbors.append(distances[indx],self.y[indx][0])
#             neighbors.append(sum)
#         max = np.max(neighbors)
#         result = np.where(neighbors == max)[0]
#         prediction = uniquevals[result][0]
#         preds.append(prediction)
#
#     y_reshaped = y_test.reshape(y_test.shape[0],)
#     matches = []
#     for indx in range(0,len(preds)):
#         if y_reshaped[indx] == preds[indx]:
#             matches.append(indx)
#     # print(matches)
#     accuracy = len(matches)/len(y_reshaped)
#     return accuracy
# arrayScores = []
# arrayK = []
# for a in range(1, 17, 2):
#     score = testOne(X_test,y_train,y_test,a)
#     arrayScores.append(score)
#     arrayK.append(a)
#     print("K = ",a,score)
#
# x = np.arange(len(arrayK))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# ax.bar(x - width/2, arrayScores, width, label='k')
# ax.set_ylim([0.7,0.9])
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('TS Accuracy')
# ax.set_title('Accuracy for Different Values of "K"')
# ax.set_xticks(x)
# ax.set_xticklabels(arrayK)
# ax.set_xlabel('k')
# ax.legend()
# save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/KNN/plots/kvsac5"
# fig.savefig(save_path)

print("*******************PART 6************************************")
print(" MAGIC DATA")
arff_path_train = r"training/magic_training.arff"
arff_path_test = r"training/magic_test.arff"
mat = arff.Arff(arff_path_train)
mat2 = arff.Arff(arff_path_test)
arrayTypes = mat.attr_types
del arrayTypes[-1]
arrayScores = []

# print(mat.data)
train_data = mat.data[:,0:-1]
# train_data_norm = preprocessing.normalize(train_data, axis = 0, norm ='l1')
train_data_norm = normalization.normalizeData(train_data)
train_labels = mat.data[:,-1].reshape(-1,1)


test_data = mat2.data[:,0:-1]
# test_data_norm = preprocessing.normalize(test_data, axis=0,norm ='l1' )
test_data_norm = normalization.normalizeData(test_data)
test_labels = mat2.data[:,-1].reshape(-1,1)
KNN = KNNClassifier(columntype= arrayTypes, labelType ='classification' ,weight_type='inverse_distance',k=3)
KNN.fit(train_data_norm,train_labels)
# pred = KNN.predict(test_data_norm)
score = KNN.score(test_data_norm,test_labels)
arrayScores.append(score)
print("Score",score)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_data_norm, train_labels)
skscore = neigh.score(test_data_norm,test_labels)
arrayScores.append(skscore)
arrayK = ["Mine", "sklearn"]
x = np.arange(len(arrayK))  # the label locations

width = 0.35  # the width of the bars

fig, ax = plt.subplots()
ax.bar(x - width/2, arrayScores, width, label='k')
ax.set_ylim([0.7,0.9])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('TS Accuracy')
ax.set_title('Accuracy for Different Values of "K"')
ax.set_xticks(x)
ax.set_xticklabels(arrayK)
ax.set_xlabel('k')
ax.legend()
save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/KNN/plots/kvsac61"
fig.savefig(save_path)

print("HOUSING DATA")
arff_path_train = r"training/housing_training.arff"
arff_path_test = r"training/housing_test.arff"
mat = arff.Arff(arff_path_train)
mat2 = arff.Arff(arff_path_test)
arrayTypes = mat.attr_types
del arrayTypes[-1]
arrayScores = []
arrayK = ["Mine","sklearn"]
scaler = preprocessing.MinMaxScaler()

train_data = mat.data[:,0:-1]
train_data_norm = normalization.normalizeData(train_data)
# train_data_norm = scaler.fit_transform(train_data)
train_labels = mat.data[:,-1].reshape(-1,1)
test_data = mat2.data[:,0:-1]
test_data_norm = normalization.normalizeData(test_data)
# test_data_norm = scaler.fit_transform(test_data)

test_labels = mat2.data[:,-1].reshape(-1,1)

KNN = KNNClassifier(columntype= arrayTypes, labelType ='regression' ,weight_type='no_weight',k=3)
KNN.fit(train_data_norm,train_labels)
score = KNN.score(test_data_norm,test_labels)
arrayScores.append(score)
print("Score",score)
neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(train_data_norm, train_labels)
preds = neigh.predict(test_data_norm)

mse = 0
arraymse =[]
y_reshaped = test_labels.reshape(test_labels.shape[0],)
for indx in range(0,len(preds)):
    mse = mse + (y_reshaped[indx] - preds[indx])**2
    # print((y_reshaped[indx] - preds[indx])**2)
    arraymse.append((y_reshaped[indx] - preds[indx])**2)
accuracy = mse/len(y_reshaped)
arrayScores.append(accuracy)
x = np.arange(len(arrayK))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
ax.bar(x - width/2, arrayScores, width, label='k')
ax.set_ylim([10,15])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('TS MSE')
ax.set_title('Accuracy for Different Values of "K"')
ax.set_xticks(x)
ax.set_xticklabels(arrayK)
ax.set_xlabel('k')
ax.legend()
save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/KNN/plots/kvsac62"
fig.savefig(save_path)
