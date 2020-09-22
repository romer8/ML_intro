import sys
import numpy as np
import random
sys.path.append("/home/elkin/university/gradSchool/Fall202/CS472/CS472")
from tools import arff, splitData, generatePerceptronData, graph_tools
import perceptron
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

## Part 1 ##

arff_path = r"training/data_banknote_authentication.arff"
# arff_path = r"training/linsep2nonorigin.arff"
dataRaw = arff.Arff(arff=arff_path, label_count=1)
data = dataRaw.data[:,0:-1]
labels = dataRaw.data[:,-1].reshape(-1,1)
# trainingSet,trainingLabels,testSet,testLabels= splitData.getSplitData(data,labels,0.7)


## Define the initial Parameters ##
LR = 0.1
DET = 10
SHUFFLE = False
IW = [0,0,0,0,0]
# IW = [0,0,0]



PClass = perceptron.PerceptronClassifier(lr=LR,shuffle=SHUFFLE, deterministic=DET,initial_weights=IW)
PClass.fit(data,labels)
Accuracy = PClass.score(data,labels)
print("Accuray = [{:.2f}]".format(Accuracy))
print("Final Weights =",PClass.get_weights())


# x1, x2 = zip(*data.tolist())
# x1 = list(x1)
# x2 = list(x2)
# labels_ = labels.reshape(len(labels),)
#
# graph_tools.graph(x1, x2, labels_,None, "Test", "X", "Y", True, 'fivethirtyeight',
#           None, None, True)


##Part 2

## Uncomment both to generate the linear and not linear datasets
# generatePerceptronData.generateDataSet("linearSeparableDataSet",True,(8,),[-1,1])
# generatePerceptronData.generateDataSet("nonLinSeparableDataSet",False,(8,),[-1,1])


## Part 3 ##

ls_path_part3 = r"training/linearSeparableDataSet.arff"
not_ls_path_part3 = r"training/notLinearSeparableDataSet.arff"

dataRaw_ls = arff.Arff(arff=ls_path_part3, label_count=1)
dataRaw_nls = arff.Arff(arff=not_ls_path_part3, label_count=1)

data_ls = dataRaw_ls.data[:,0:-1]
labels_ls = dataRaw_ls.data[:,-1].reshape(-1,1)
x1_ls, x2_ls = zip(*data_ls.tolist())
x1_ls = list(x1_ls)
x2_ls = list(x2_ls)
labels_ls_array = labels_ls.reshape(len(labels_ls),)

graph_tools.graph(x1_ls, x2_ls, labels_ls_array,None, title="Linear Separable", xlabel="X", ylabel="Y", points=True, style='fivethirtyeight',
          xlim=None, ylim=None, legend=True, save_path="/home/elkin/university/gradSchool/Fall202/CS472/CS472/perceptron/plots/linear_separable")




data_nls = dataRaw_nls.data[:,0:-1]
labels_nls = dataRaw_nls.data[:,-1].reshape(-1,1)
x1_nls, x2_nls = zip(*data_nls.tolist())
x1_nls = list(x1_nls)
x2_nls = list(x2_nls)
labels_nls_array = labels_nls.reshape(len(labels_nls),)

graph_tools.graph(x1_nls, x2_nls, labels_nls_array, None, "Non Linear Separable", "X", "Y", True, 'fivethirtyeight',
          None, None, True, save_path="/home/elkin/university/gradSchool/Fall202/CS472/CS472/perceptron/plots/nonlinear_separable")

## Define the initial Parameters ##
DET3 = 10
SHUFFLE3 = False
IW = [0,0,0]
LR_array = np.arange(0.1,1.1,0.1).tolist()

print("Linear data set")
for LRs in LR_array:
    LRs= round(LRs,1)
    PClass_ls = perceptron.PerceptronClassifier(lr=LRs,shuffle=SHUFFLE3, deterministic=DET3,initial_weights=IW)
    PClass_ls.fit(data_ls,labels_ls)
    Accuracy_ls = PClass_ls.score(data_ls,labels_ls)
    print("LR = " , LRs)
    print("Accuray = [{:.2f}]".format(Accuracy_ls))
    print("Final Weights =",PClass_ls.get_weights())

print("Non linear data set")
for LRs in LR_array:
    LRs= round(LRs,1)
    PClass_nls = perceptron.PerceptronClassifier(lr=LRs,shuffle=SHUFFLE3, deterministic=DET3,initial_weights=IW)
    PClass_nls.fit(data_nls,labels_nls)
    Accuracy_nls = PClass_nls.score(data_nls,labels_nls)
    print("LR = " , LRs)
    print("Accuray = [{:.2f}]".format(Accuracy_nls))
    print("Final Weights =",PClass_nls.get_weights())

## PART 4
print("PART 4")
print("Linear data set")

PClass_ls = perceptron.PerceptronClassifier(lr=0.1,shuffle=SHUFFLE3, deterministic=DET3,initial_weights=IW)
PClass_ls.fit(data_ls,labels_ls)
Accuracy_ls = PClass_ls.score(data_ls,labels_ls)
print("LR = " , 0.1)
print("Accuray = [{:.2f}]".format(Accuracy_ls))
print("Final Weights =",PClass_ls.get_weights())

graph_tools.graph(x1_ls, x2_ls, labels_ls_array,PClass_ls.get_weights(), title="Linear Separable", xlabel="X", ylabel="Y", points=True, style='fivethirtyeight',
          xlim=None, ylim=None, legend=True, save_path="/home/elkin/university/gradSchool/Fall202/CS472/CS472/perceptron/plots/boundary_linear_separable")

print("Non linear data set")

PClass_nls = perceptron.PerceptronClassifier(lr=0.1,shuffle=SHUFFLE3, deterministic=DET3,initial_weights=IW)
PClass_nls.fit(data_nls,labels_nls)
Accuracy_nls = PClass_nls.score(data_nls,labels_nls)
print("LR = " , 0.1)
print("Accuray = [{:.2f}]".format(Accuracy_nls))
print("Final Weights =",PClass_nls.get_weights())

weights_nls = PClass_nls.get_weights()
graph_tools.graph(x1_nls, x2_nls, labels_nls_array,weights_nls , "Non Linear Separable", "X", "Y", True, 'fivethirtyeight',
          None, None, True, save_path="/home/elkin/university/gradSchool/Fall202/CS472/CS472/perceptron/plots/boundary_nonlinear_separable")

## PART 5

print("PART 5")

vote_data_path = r"training/vote_data.arff"
# arff_path = r"training/linsep2nonorigin.arff"
dataRaw_vote = arff.Arff(arff=vote_data_path, label_count=1)
data_vote = dataRaw_vote.data[:,0:-1]
labels_vote = dataRaw_vote.data[:,-1].reshape(-1,1)
## Define the initial Parameters ##
LR = 0.1
DET = 10
SHUFFLE = False
IW = np.zeros((data_vote[0]).shape[0]+1,).tolist()
# IW = [0,0,0]
misclassification_All_dict = {}
for indx in range(5):

    trainingSet,trainingLabels,testSet,testLabels= splitData.getSplitData(data_vote,labels_vote,0.7)
    PClass_splits = perceptron.PerceptronClassifier(lr=LR,shuffle=SHUFFLE, deterministic=DET,initial_weights=IW,misclassification= True)
    PClass_splits.fit(trainingSet,trainingLabels)
    Accuracy_training = PClass_splits.score(trainingSet,trainingLabels)
    Accuracy_test = PClass_splits.score(testSet,testLabels)
    misclassification_All_dict[len(PClass_splits.get_missclassification())] =PClass_splits.get_missclassification()

    trainingLabels_converted = trainingLabels.reshape(trainingLabels.shape[0],)
    testLabels_converted = testLabels.reshape(testLabels.shape[0],)
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(trainingSet, trainingLabels_converted)
    skitLearnWeights = clf.get_params()
    Perceptron()
    cfl_score_training = clf.score(trainingSet,trainingLabels_converted)
    cfl_score_test = clf.score(testSet, testLabels_converted)

    print("# Epochs = ", len(PClass_splits.get_missclassification()))
    print("Accuracy Training = [{:.2f}]".format(Accuracy_training))
    print("Accuracy Test = [{:.2f}]".format(Accuracy_test))
    print("Final Weights = ",PClass_splits.get_weights())
    print("SkitLearn Accuracy Training = [{:.2f}] ".format(cfl_score_training))
    print("SkitLearn Accuracy Training = [{:.2f}] ".format(cfl_score_test))
    # print("Final Weights = ",skitLearnWeights)

epoch_runs = sorted(misclassification_All_dict)
print(epoch_runs)

## create averages for missclassification ##
misclassification_ordered = []
array_ind = 0
for indx in sorted (misclassification_All_dict):
    misclassification_ordered.append(misclassification_All_dict[indx])

T = list(itertools.zip_longest(*misclassification_ordered, fillvalue=np.nan))
mean_rate = [np.nanmean(i) for i in T]

epochs_list = np.arange(epoch_runs[-1]).tolist()


graph_tools.graph(epochs_list, mean_rate,labels=None, weights=None, title="Misclassification Rate vs # Epoch", xlabel= "# Epochs", ylabel="Misclassification rate", points=False, style='fivethirtyeight',
          xlim=None, ylim=None, legend=False, save_path="/home/elkin/university/gradSchool/Fall202/CS472/CS472/perceptron/plots/misclassification_rate_epoch")

## PART 6
print("Part 6")
arff_path_Iris = r"training/iris.arff"
# arff_path = r"training/linsep2nonorigin.arff"
dataRawIris = arff.Arff(arff=arff_path_Iris, label_count=1)
dataIris = dataRawIris.data[:,0:-1]
labelsIris = dataRawIris.data[:,-1].reshape(-1,1)
labelsIris_reshaped = labelsIris.reshape(labelsIris.shape[0],)
perceptron_iris = Perceptron(tol=1e-3, random_state=0, shuffle = True)
perceptron_iris.fit(dataIris, labelsIris_reshaped)
Perceptron()
perceptron_iris_score = perceptron_iris.score(dataIris, labelsIris_reshaped)
weights_skici = perceptron_iris.sparsify()
print("SkitLearn Accuracy Training = [{:.2f}] ".format(perceptron_iris_score))
print("Weights = ", weights_skici)
