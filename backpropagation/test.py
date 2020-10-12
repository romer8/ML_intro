import sys
import numpy as np
import random
import matplotlib.pyplot as plt
## change Path ##
sys.path.append("/home/elkin/university/gradSchool/Fall2020/CS472/CS472")
from tools import arff, splitData, generatePerceptronData, graph_tools,list2csv
import itertools
import mlp
from sklearn.neural_network import MLPClassifier

print("PART 1")
print("DATA SET 1")

arff_path = r"training/linsep2nonorigin.arff"
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


BClass = mlp.MLPClassifier(lr = LR,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, validationSize = VALIDATION_SIZE, allWeightsValue = 0)

BClass.fit(data,labels)
scores = BClass.score(data,labels)
print("Score ", scores)
# print("Weights")
# print(BClass.get_weights())
list2csv.write_to_csv(BClass.get_weights(),"weightsP1D1.csv")
# clf = MLPClassifier(hidden_layer_sizes=(4,), activation = 'logistic',solver = 'sgd',alpha = MOMENTUM,learning_rate_init = LR, max_iter=2,shuffle = SHUFFLE).fit(data, labels)

# print("DATA SET 2")
# arff_path = r"training/data_banknote_authentication.arff"
# dataRaw = arff.Arff(arff=arff_path, label_count=1)
# data = dataRaw.data[:,0:-1]
# labels = dataRaw.data[:,-1].reshape(-1,1)
#
#
# BClass = mlp.MLPClassifier(lr = LR,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, validationSize = VALIDATION_SIZE, allWeightsValue = 0)
#
# BClass.fit(data,labels)
# scores = BClass.score(data,labels)
# print("Score ", scores)
# # print("Weights")
# # print(BClass.get_weights())
# list2csv.write_to_csv(BClass.get_weights(),"weightsP1D2.csv")
#
# print("")
# print("PART 2 IRIS DATA SET")
#
# arff_path = r"training/iris.arff"
# dataRaw = arff.Arff(arff=arff_path, label_count=1)
# data = dataRaw.data[:,0:-1]
# labels = dataRaw.data[:,-1].reshape(-1,1)
# save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/backpropagation/plots/MSEvsAccuracy"
# ## Define the initial Parameters ##
# LR = 0.1
# DET = 10
# SHUFFLE = True
# MOMENTUM = 0.5
# VALIDATION_SIZE = 0.25
#
# BClass = mlp.MLPClassifier(lr = LR,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, validationSize = VALIDATION_SIZE)
#
# BClass.fit(data,labels)
# print("Number of Epochs to run ", BClass.get_numberEpochs())
# scores = BClass.score(data,labels)
# print("Score ", scores)
# # print("Weights")
# # print(BClass.get_weights())
# mse_epochs  = BClass.get_mse_epochs()
# accuracy_epochs  = BClass.get_accuracy_epochs()
# number_epochs = a_list = list(range(1, BClass.get_numberEpochs()+1))
#
# print("MSE epochs")
# print(len(mse_epochs))
# print("Accuracy Epochs")
# print(len(accuracy_epochs))
#
#
# fig, ax1 = plt.subplots()
#
# color = 'tab:red'
# ax1.set_xlabel('Number of Epochs')
# ax1.set_ylabel('Accuracy % Val Set', color=color)
# ax1.plot(number_epochs, accuracy_epochs, color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:blue'
# ax2.set_ylabel('MSE Val Set', color=color)  # we already handled the x-label with ax1
# ax2.plot(number_epochs, mse_epochs, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# # plt.show()
# fig.savefig(save_path)
# print("")
#
# print("PART 3 VOWEL DATASET")
#
# arff_path = r"training/vowel.arff"
# dataRaw = arff.Arff(arff=arff_path, label_count=1)
# data = dataRaw.data[:,0:-1]
# labels = dataRaw.data[:,-1].reshape(-1,1)
# save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/backpropagation/plots/MSEvsAccuracy"
# ## Define the initial Parameters ##
# LR = 0.1
# DET = 10
# SHUFFLE = True
# MOMENTUM = 0.5
# VALIDATION_SIZE = 0.25
#
# BClass = mlp.MLPClassifier(lr = LR,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, validationSize = VALIDATION_SIZE)
#
# BClass.fit(data,labels)
# print("Number of Epochs to run ", BClass.get_numberEpochs())
# scores = BClass.score(data,labels)
# print("Score ", scores)
# # print("Weights")
# # print(BClass.get_weights())
