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
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn import tree
import graphviz
import pprint

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
pred = DTClass.predict(data2)
Acc = DTClass.score(data2,labels2)

# np.savetxt("pred_lenses.csv",pred,delimiter=",")
print("Accuracy = [{:.2f}]".format(Acc))

# print("*******************EVALUATION************************************")
# arff_path_train = r"training/zoo.arff"
# arff_path_test = r"training/all_zoo.arff"
# mat = arff.Arff(arff_path_train)
#
# counts = [] ## this is so you know how many types for each column
#
# for i in range(mat.data.shape[1]):
#        counts += [mat.unique_value_count(i)]
# data = mat.data[:,0:-1]
# labels = mat.data[:,-1].reshape(-1,1)
#
# DTClass = DTClassifier(counts)
# DTClass.fit(data,labels)
# mat2 = arff.Arff(arff_path_test)
# data2 = mat2.data[:,0:-1]
# labels2 = mat2.data[:,-1].reshape(-1,1)
# pred = DTClass.predict(data2)
#
# Acc = DTClass.score(data2,labels2)
#
# np.savetxt("pred_zoo.csv",pred,delimiter=",")
# print("Accuracy = [{:.2f}]".format(Acc))
#
# print("*******************PART 2************************************")
# print("*******CARS DATASET")
# arff_path= r"training/cars.arff"
# mat = arff.Arff(arff_path)
#
#
# data2 = mat.data[:,0:-1]
# labels = mat.data[:,-1].reshape(-1,1)
# data = np.concatenate((data2, labels), axis = 1)
# """ now K-folding """
# # prepare cross validation
# kfold = KFold(n_splits=10, random_state=None, shuffle=True)
# scores = []
#
# for train, test in kfold.split(data):
#     columns = len(data[train][0])
#     data_transposed = data[train].T
#     counts = []
#     for column in range(0, columns):
#         unique = np.unique(data_transposed[column])
#         counts.append(len(unique))
#     DTClass = DTClassifier(counts)
#     splitXindxRange = data[train].shape[1] - 1
#     labelsK = data[train][:, np.r_[splitXindxRange]]
#     dataK = data[train][:, np.r_[0:splitXindxRange]]
#     labelsTestK = data[test][:, np.r_[splitXindxRange]]
#     dataTestK = data[test][:, np.r_[0:splitXindxRange]]
#     DTClass.fit(dataK,labelsK)
#
#     columnsTest = len(data[test][0])
#     dataTest_transposed = data[test].T
#     countsTest = []
#     for column in range(0, columnsTest):
#         unique = np.unique(dataTest_transposed[column])
#         countsTest.append(len(unique))
#     scores.append(DTClass.score(dataTestK,labelsTestK))
#
#     # break
# print("Scores Cars Dataset", scores)
# print("*******VOTING DATASET")
#
# arff_path= r"training/voting.arff"
# mat = arff.Arff(arff_path)
#
# data2 = mat.data[:,0:-1]
# labels = mat.data[:,-1].reshape(-1,1)
# data = np.concatenate((data2, labels), axis = 1)
#
#
# """ now K-folding """
# # prepare cross validation
# kfold = KFold(n_splits=10, random_state=None, shuffle=True)
# scores = []
#
# for train, test in kfold.split(data):
#     columns = len(data[train][0])
#     data_transposed = data[train].T
#     counts = []
#     for column in range(0, columns):
#         unique = np.unique(data_transposed[column])
#         counts.append(len(unique))
#     DTClass = DTClassifier(counts)
#     splitXindxRange = data[train].shape[1] - 1
#     labelsK = data[train][:, np.r_[splitXindxRange]]
#     dataK = data[train][:, np.r_[0:splitXindxRange]]
#     labelsTestK = data[test][:, np.r_[splitXindxRange]]
#     dataTestK = data[test][:, np.r_[0:splitXindxRange]]
#     DTClass.fit(dataK,labelsK)
#
#     columnsTest = len(data[test][0])
#     dataTest_transposed = data[test].T
#     countsTest = []
#     for column in range(0, columnsTest):
#         unique = np.unique(dataTest_transposed[column])
#         countsTest.append(len(unique))
#
#     scores.append(DTClass.score(dataTestK,labelsTestK))
#
#     # break
# print("Scores Voting Dataset", scores)
#
# print("*******************PART 5************************************")
# print("*******CARS DATASET*******************************************")
# arff_path= r"training/cars.arff"
# mat = arff.Arff(arff_path)
#
# data2 = mat.data[:,0:-1]
# labels = mat.data[:,-1].reshape(-1,1)
# X_train, X_test, y_train, y_test = train_test_split(data2, labels, test_size= 0.10)
# clf = DecisionTreeClassifier()
# path = clf.cost_complexity_pruning_path(X_train, y_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities
# clfs = []
# for ccp_alpha in ccp_alphas:
#     clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
#     clf.fit(X_train, y_train)
#     clfs.append(clf)
# clfs = clfs[:-1]
# ccp_alphas = ccp_alphas[:-1]
# node_counts = [clf.tree_.node_count for clf in clfs]
# depth = [clf.tree_.max_depth for clf in clfs]
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
# ax[0].set_xlabel("alpha")
# ax[0].set_ylabel("number of nodes")
# ax[0].set_title("Number of nodes vs alpha")
# ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
# ax[1].set_xlabel("alpha")
# ax[1].set_ylabel("depth of tree")
# ax[1].set_title("Depth vs alpha")
# fig.tight_layout()
# save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/decisiontree/plots/carsdepth"
# fig.savefig(save_path)
# train_scores = [clf.score(X_train, y_train) for clf in clfs]
# test_scores = [clf.score(X_test, y_test) for clf in clfs]
# print("Cars Scores: ", test_scores)
# fig, ax = plt.subplots()
# ax.set_xlabel("alpha")
# ax.set_ylabel("accuracy")
# ax.set_title("Accuracy vs alpha for training and testing sets")
# ax.plot(ccp_alphas, train_scores, marker='o', label="train",
#         drawstyle="steps-post")
# ax.plot(ccp_alphas, test_scores, marker='o', label="test",
#         drawstyle="steps-post")
# ax.legend()
# # plt.show()
# save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/decisiontree/plots/carsaccuracy"
# fig.savefig(save_path)
#
#
# print("******* VOTING DATASET*******************************************")
# arff_path= r"training/voting.arff"
# mat = arff.Arff(arff_path)
#
# data = mat.data[:,0:-1]
# labels = mat.data[:,-1].reshape(-1,1)
# inds = np.where(np.isnan(data))
# col_mean = np.nanmean(data, axis=0)
# col_new_val = []
# for col in col_mean:
#     col_new_val.append(-1)
#
# #Place column means in the indices. Align the arrays using take
# data[inds] = np.take(np.array(col_new_val), inds[1])
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size= 0.10)
# clf = DecisionTreeClassifier()
# path = clf.cost_complexity_pruning_path(X_train, y_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities
#
# clfs = []
# for ccp_alpha in ccp_alphas:
#     clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
#     clf.fit(X_train, y_train)
#     clfs.append(clf)
# clfs = clfs[:-1]
# ccp_alphas = ccp_alphas[:-1]
# node_counts = [clf.tree_.node_count for clf in clfs]
# depth = [clf.tree_.max_depth for clf in clfs]
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
# ax[0].set_xlabel("alpha")
# ax[0].set_ylabel("number of nodes")
# ax[0].set_title("Number of nodes vs alpha")
# ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
# ax[1].set_xlabel("alpha")
# ax[1].set_ylabel("depth of tree")
# ax[1].set_title("Depth vs alpha")
# fig.tight_layout()
# save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/decisiontree/plots/votingdepth"
# fig.savefig(save_path)
#
#
# train_scores = [clf.score(X_train, y_train) for clf in clfs]
# test_scores = [clf.score(X_test, y_test) for clf in clfs]
# print("Voting Scores: ", test_scores)
# fig, ax = plt.subplots()
# ax.set_xlabel("alpha")
# ax.set_ylabel("accuracy")
# ax.set_title("Accuracy vs alpha for training and testing sets")
# ax.plot(ccp_alphas, train_scores, marker='o', label="train",
#         drawstyle="steps-post")
# ax.plot(ccp_alphas, test_scores, marker='o', label="test",
#         drawstyle="steps-post")
# ax.legend()
# # plt.show()
# save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/decisiontree/plots/votingaccuracy"
# fig.savefig(save_path)
#
#
# print("******* CANCER DATASET*******************************************")
# X, y = load_breast_cancer(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# clf = DecisionTreeClassifier(criterion = "entropy")
# clf = clf.fit(X_train, y_train)
# scores = clf.score(X_test,y_test)
# print(scores)
# dot_data = tree.export_graphviz(clf, out_file=None, special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render("cancerBreast")
