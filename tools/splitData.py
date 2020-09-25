import numpy as np
# from prettytable import PrettyTable

def getSplitData(X,y,split):
    indices = np.random.permutation(X.shape[0])
    # indices = np.indices((X.shape[0],))[0]
    rounded_idx_start = round((X.shape[0]*split))
    training_idx, test_idx = indices[:rounded_idx_start], indices[rounded_idx_start:]
    trainingX, testX = X[training_idx,:], X[test_idx,:]
    training_y, test_y = y[training_idx,:], y[test_idx,:]
    return [trainingX,training_y,testX,test_y]

# def printDataNicely(trainingSet,trainingLabels,testSet,testLabels):
#
#     t = PrettyTable(['X','labels'])
#     t2 = PrettyTable(['X','labels'])
#     for trs, trl in zip(trainingSet,trainingLabels):
#         t.add_row([trs,trl])
#     print("LENGTH TRAINING SET",len(trainingSet))
#     print(t)
#     t2 = PrettyTable(['X','labels'])
#     for ts, tl in zip(testSet,testLabels):
#         t2.add_row([ts,tl])
#     print("LENGTH TEST SET ",len(testSet))
#     print(t2)
