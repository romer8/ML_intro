import sys
import numpy as np
import random
## change Path ##
sys.path.append("/home/elkin/university/gradSchool/Fall202/CS472/CS472")
import mlp


# mat = Arff("../datasets/linsep2nonorigin.arff")
# data = mat.data[:,0:-1]
# labels = mat.data[:,-1].reshape(-1,1)
BClass = mlp.MLPClassifier(lr=0.1,momentum=0.5,shuffle=False,deterministic=10,hidden_layer_widths=[3,4,5])
testArray = np.array([[0,1,3],[1,2,3],[7,5,5]])
BClass.initialize_weights(testArray)
# BClass.fit(data,labels)
