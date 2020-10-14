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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

print("PART 1")
print("DATA SET 1")

arff_path = r"training/linsep2nonorigin.arff"
dataRaw = arff.Arff(arff=arff_path, label_count=1)
data = dataRaw.data[:,0:-1]
labels = dataRaw.data[:,-1].reshape(-1,1)


## Define the initial Parameters ##
LR = 0.1
DET = 10
SHUFFLE = False
MOMENTUM = 0.5
VALIDATION_SIZE = 0.0
HOTENCODING = True
# data = [[0,0],[0,1]]
# BClass = mlp.MLPClassifier(lr = LR,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, hidden_layer_widths =[2,2,2] ,validationSize = VALIDATION_SIZE, allWeightsValue = 0)
BClass = mlp.MLPClassifier(lr = LR,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, validationSize = VALIDATION_SIZE, allWeightsValue = 0,isHotEncoding= HOTENCODING)

BClass.fit(data,labels)
scores = BClass.score(data,labels)
print("Score ", scores)
# print("Weights")
# print(BClass.get_weights())
list2csv.write_to_csv(BClass.get_weights(),"weightsP1D1.csv")
# clf = MLPClassifier(hidden_layer_sizes=(4,), activation = 'logistic',solver = 'sgd',alpha = MOMENTUM,learning_rate_init = LR, max_iter=2,shuffle = SHUFFLE).fit(data, labels)

print("DATA SET 2")
arff_path = r"training/data_banknote_authentication.arff"
dataRaw = arff.Arff(arff=arff_path, label_count=1)
data = dataRaw.data[:,0:-1]
labels = dataRaw.data[:,-1].reshape(-1,1)


BClass = mlp.MLPClassifier(lr = LR,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, validationSize = VALIDATION_SIZE, allWeightsValue = 0)

BClass.fit(data,labels)
scores = BClass.score(data,labels)
print("Score ", scores)
# print("Weights")
# print(BClass.get_weights())
list2csv.write_to_csv(BClass.get_weights(),"evaluation.csv")

print("")
print("PART 2 IRIS DATA SET")

arff_path = r"training/iris.arff"
dataRaw = arff.Arff(arff=arff_path, label_count=1)
data = dataRaw.data[:,0:-1]
labels = dataRaw.data[:,-1].reshape(-1,1)
save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/backpropagation/plots/MSEvsAccuracyIRIS"
save_path3="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/backpropagation/plots/MSE_ValidationTrain"
data_train,data_test , labels_train, labels_test = train_test_split(data, labels, test_size=0.25)

## Define the initial Parameters ##
LR = 0.1
DET = 10
SHUFFLE = True
MOMENTUM = 0.5
VALIDATION_SIZE = 0.25
HOTENCODING = True

BClass = mlp.MLPClassifier(lr = LR,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, validationSize = VALIDATION_SIZE, isHotEncoding= HOTENCODING)

BClass.fit(data_train,labels_train)
print("Number of Epochs to run ", BClass.get_numberEpochs())
scores = BClass.score(data,labels)
print("Score ", scores)
# print("Weights")
# # print(BClass.get_weights())
mse_epochs_val, mse_epochs_training  = BClass.get_mse_epochs()
accuracy_epochs  = BClass.get_accuracy_epochs()
number_epochs = a_list = list(range(1, BClass.get_numberEpochs()+1))

""" GRAPHS MSE VALIDATION"""
plot_list = []
fig, ax1 = plt.subplots()
ax1.title.set_text('MSE and Accuracy % in Validation and Training Set  ')
color = 'tab:red'
ax1.set_xlabel('Number of Epochs')
ax1.set_ylabel('Accuracy % Val Set', color=color)

plot_list.append(ax1.plot(number_epochs, accuracy_epochs, color=color, label = "% Accuracy validation set")[0])
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('MSE sets')  # we already handled the x-label with ax1
# ax2.plot(number_epochs, mse_epochs_val, color=color,label = "MSE validation set")
plot_list.append(ax2.plot(number_epochs, mse_epochs_val, color=color,label = "MSE validation set")[0])
# ax2.tick_params(axis='y', labelcolor=color)

ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:green'
# ax3.plot(number_epochs, mse_epochs_training, color=color, label ="MSE training set")
plot_list.append(ax3.plot(number_epochs, mse_epochs_training, color=color, label ="MSE training set")[0])
# ax3.tick_params(axis='y', labelcolor=color)

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=5, handles=plot_list,
              facecolor = 'white', edgecolor = 'black')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
fig.savefig(save_path)


""" GRAPHS MSE BOTH"""
fig, ax1 = plt.subplots()
ax1.title.set_text('MSE Validation and Training Set')

color = 'tab:red'
ax1.set_xlabel('Number of Epochs')
ax1.set_ylabel('MSE Validation Set', color=color)
ax1.plot(number_epochs, mse_epochs_val, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('MSE Training Set', color=color)  # we already handled the x-label with ax1
ax2.plot(number_epochs, mse_epochs_training, color=color)
ax2.tick_params(axis='y', labelcolor=color)


fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
fig.savefig(save_path3)


print("")
print("PART 3 VOWEL DATASET")

arff_path = r"training/vowel.arff"
dataRaw = arff.Arff(arff=arff_path, label_count=1)
data = dataRaw.data[:,0:-1]
labels = dataRaw.data[:,-1].reshape(-1,1)
data = data[:, [0,1,3,4,5,6,7,8,9,10,11,12]]
save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/backpropagation/plots/vowelMSE"
save_path2="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/backpropagation/plots/vowelEpochs"
## Define the initial Parameters ##
LR = 0.1
DET = 10
SHUFFLE = True
MOMENTUM = 0.5
VALIDATION_SIZE = 0.1
HOTENCODING = True
# BClass = mlp.MLPClassifier(lr = LR,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, validationSize = VALIDATION_SIZE,isHotEncoding= HOTENCODING)

# BClass.fit(data_train,labels_train)
# print("Number of Epochs to run ", BClass.get_numberEpochs())
# scores = BClass.score(data_test,labels_test)
# print("Score ", scores)

LRs = [0.005, 0.01, 0.02,0.05,0.1,0.2,0.5,0.75,0.9,2]
MSEs_Training = []
MSEs_Validation = []
MSEs_Test = []
numberEpochsToStop = []
enc = OneHotEncoder()

for lrx in LRs:
    data_train,data_test , labels_train, labels_test = train_test_split(data, labels, test_size=0.25)
    BClass = mlp.MLPClassifier(lr = lrx,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, validationSize = VALIDATION_SIZE,isHotEncoding= HOTENCODING)
    BClass.fit(data_train,labels_train)
    numberEpochsToStop.append(BClass.get_numberEpochs())
    enc.fit(labels_train)
    enc.fit(labels_test)
    labels_train_hot = enc.transform(labels_train).toarray()
    labels_test_hot = enc.transform(labels_test).toarray()
    mse_train = BClass._get_mse_valSet(data_train,labels_train_hot)
    mse_val = BClass.getMSEvalSet()
    mse_test = BClass._get_mse_valSet(data_test, labels_test_hot)
    MSEs_Test.append(mse_test)
    MSEs_Training.append(mse_train)
    MSEs_Validation.append(mse_val)

x = np.arange(len(LRs))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
barVal = ax.bar(x - width, MSEs_Validation, width, label='Validation Set')
barTrain = ax.bar(x , MSEs_Training, width, label='Training Set')
barTest = ax.bar(x + width, MSEs_Test, width, label='Test Set')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MSE')
ax.set_title('Vowel MSE vs. Learning Rate')
ax.set_xticks(x)
ax.set_xticklabels(LRs)
ax.set_xlabel('Learning Rate')
ax.legend()
fig.savefig(save_path)

"""BAR GRAPHS # epochs"""
x = np.arange(len(LRs))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
numberEpochsStop = ax.bar(x - width/2, numberEpochsToStop, width, label='Epochs')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('# of Epochs to Stop')
ax.set_title('Vowel # Epochs per Learning Rate')
ax.set_xticks(x)
ax.set_xticklabels(LRs)
ax.set_xlabel('Learning Rate')
ax.legend()
fig.savefig(save_path2)

print("")
print("PART 4 VOWEL DATASET")

arff_path = r"training/vowel.arff"
dataRaw = arff.Arff(arff=arff_path, label_count=1)
data = dataRaw.data[:,0:-1]
labels = dataRaw.data[:,-1].reshape(-1,1)
data = data[:, [0,1,3,4,5,6,7,8,9,10,11,12]]
save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/backpropagation/plots/vowelMSENoded"
## Define the initial Parameters ##
LR = 0.1
DET = 10
SHUFFLE = True
MOMENTUM = 0.5
VALIDATION_SIZE = 0.1
HOTENCODING = True
numberNodes = np.array([2])
ns = 0
numberNodesArray=[]
MSEs_Training = []
MSEs_Validation = []
MSEs_Test = []
numberOfEpochWithNoImprovement= []
bestAccuracy = 0
enc = OneHotEncoder()

while len(numberOfEpochWithNoImprovement) < 1:
    numberNodes2 = np.power(numberNodes,ns)
    numberNodesList = list(numberNodes2)
    print("number Nodes",numberNodesList)
    numberNodesArray.append(numberNodes2[0])
    data_train,data_test , labels_train, labels_test = train_test_split(data, labels, test_size=0.25)
    BClass = mlp.MLPClassifier(lr = LR,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, hidden_layer_widths= numberNodesList, validationSize = VALIDATION_SIZE,isHotEncoding= HOTENCODING)
    BClass.fit(data_train,labels_train)
    enc.fit(labels_train)
    enc.fit(labels_test)
    labels_train_hot = enc.transform(labels_train).toarray()
    labels_test_hot = enc.transform(labels_test).toarray()
    mse_train = BClass._get_mse_valSet(data_train,labels_train_hot)
    mse_val = BClass.getMSEvalSet()
    mse_test = BClass._get_mse_valSet(data_test, labels_test_hot)
    MSEs_Test.append(mse_test)
    MSEs_Training.append(mse_train)
    MSEs_Validation.append(mse_val)
    score_run = BClass.getAccuracyValSet()
    if score_run > 0.9999:
        break
    if abs(bestAccuracy - score_run) < bestAccuracy * 0.00001:
        numberOfEpochWithNoImprovement.append("1")

    else:
        if bestAccuracy < score_run :
            bestAccuracy= score_run
            numberOfEpochWithNoImprovement.clear()
        else:
            numberOfEpochWithNoImprovement.append("1")
    ns = ns + 1
    print(numberNodes)
    print(numberOfEpochWithNoImprovement)
print("number nodes ", numberNodesArray)
x = np.arange(len(numberNodesArray))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
barVal = ax.bar(x - width, MSEs_Validation, width, label='Validation Set')
barTrain = ax.bar(x , MSEs_Training, width, label='Training Set')
barTest = ax.bar(x + width, MSEs_Test, width, label='Test Set')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MSE')
ax.set_title('Vowel MSE vs. # Nodes in Hidden Layer')
ax.set_xticks(x)
ax.set_xticklabels(numberNodesArray)
ax.set_xlabel('Hidden Layer Nodes')
ax.legend()
fig.savefig(save_path)

print("")
print("PART 5 VOWEL DATASET")
arff_path = r"training/vowel.arff"
dataRaw = arff.Arff(arff=arff_path, label_count=1)
data = dataRaw.data[:,0:-1]
labels = dataRaw.data[:,-1].reshape(-1,1)
data = data[:, [0,1,3,4,5,6,7,8,9,10,11,12]]
save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/backpropagation/plots/vowelEpochsMomentum"
## Define the initial Parameters ##
LR = 0.1
DET = 10
SHUFFLE = True
MOMENTUM = 0.0
VALIDATION_SIZE = 0.1
HOTENCODING = True
ns = 0
MSEs_Training = []
MSEs_Validation = []
MSEs_Test = []
numberOfEpochWithNoImprovement= []
bestAccuracy = 0
momentumList=[]
numberEpochsToStop=[]
enc = OneHotEncoder()
MOMENTUMS = [0.0,0.1,0.2,0.4,0.5,0.6,0.8]

# while len(numberOfEpochWithNoImprovement) < 2:
for MOMENTUM in MOMENTUMS:
    momentumList.append(MOMENTUM)
    data_train,data_test , labels_train, labels_test = train_test_split(data, labels, test_size=0.25)
    BClass = mlp.MLPClassifier(lr = LR,momentum = MOMENTUM, shuffle = SHUFFLE, deterministic = DET, hidden_layer_widths= [16], validationSize = VALIDATION_SIZE,isHotEncoding= HOTENCODING)
    BClass.fit(data_train,labels_train)
    numberEpochsToStop.append(BClass.get_numberEpochs())
    print(momentumList, numberEpochsToStop)


""" GRAPHS MOMENTUM VS EPOCHS """
fig, ax1 = plt.subplots()
ax1.title.set_text('Vowel Epochs vs. Momentum')

color = 'tab:red'
ax1.set_xlabel('Momentum')
ax1.set_ylabel('Epochs', color=color)
ax1.plot(momentumList,numberEpochsToStop, color=color)
ax1.tick_params(axis='y', labelcolor=color)


fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
fig.savefig(save_path)
