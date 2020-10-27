import numpy as np
import math
import random
from sklearn.base import BaseEstimator, ClassifierMixin
import sys
sys.setrecursionlimit(10000)
### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score

class DTClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,counts=None):
        """ Initialize class with chosen hyperparameters.
        Args:
        Optional Args (Args we think will make your life easier):
            counts: A list of Ints that tell you how many types of each feature there are
        Example:
            DT  = DTClassifier()
            or
            DT = DTClassifier(count = [2,3,2,2])
            Dataset =
            [[0,1,0,0],
            [1,2,1,1],
            [0,1,1,0],
            [1,2,0,1],
            [0,0,1,1]]

        """
        self.counts = counts
        self.three = None
        self.check = 0

    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        inds = np.where(np.isnan(X))
        col_mean = np.nanmean(X, axis=0)
        col_new_val = []
        for col in col_mean:
            col_new_val.append(-1)

        #Place column means in the indices. Align the arrays using take
        X[inds] = np.take(np.array(col_new_val), inds[1])
        self.MakeThree(X, y)
        return self

    def MakeThree(self, X,y):
        newXy = np.concatenate((X, y), axis = 1)
        X_transposed = np.transpose(newXy)
        # print("**********ROOT*************")
        self.three = self.getSplit(newXy,-1)
        # print("**********RECURSION*************")
        self.splitData(self.three)
        return

    def splitData(self,node):
        splitGroups = []
        splitGroupsValues = []
        for nd in node['groups']:
            splitGroups.append(nd)
        for nd in node['groupsValues']:
            splitGroupsValues.append(nd)
        del(node['groups'])
        del(node['groupsValues'])
        # print("splitGroups")
        # print(splitGroups)
        for indxNode in range(0,len(splitGroups)):
            node["nodeSplit"+str(indxNode)]  = None
            if self.isPure(splitGroups[indxNode]):
                node["nodeSplit"+str(indxNode)] = {'label': self.getOutputs(splitGroups[indxNode],node['mostCommonValue']),
                                                   'value':splitGroupsValues[indxNode]}
                # print(splitGroups[indxNode],"-----Pure--->>>>>")
            elif self.noMoreAttributes(splitGroups[indxNode]):
                node["nodeSplit"+str(indxNode)] = {'label': self.getOutputs(splitGroups[indxNode],node['mostCommonValue']),
                                                   'value':splitGroupsValues[indxNode]}
                # print(splitGroups[indxNode],"-----No more attributes--->>>>>")
            # return
            else:
                # print(self.check)
                # print(splitGroups[indxNode])
                node["nodeSplit"+str(indxNode)] = self.getSplit(splitGroups[indxNode],splitGroupsValues[indxNode])
                # print("THREE")
                # print(self.three)
                # self.check = self.check +  1
                # if self.check > 3:
                #     return
                self.splitData(node["nodeSplit"+str(indxNode)])
        return
    def getOutputs(self,group,mode):
        # print(group)
        splitXindxRange = group.shape[1] - 1
        labels = group[:, np.r_[splitXindxRange]]
        isSame = np.all(labels == labels[0])
        # print((labels))
        indice = np.random.choice(np.arange(labels.size), replace=False,
                           size= 1)
        if isSame:
            return labels[0][0]
        else:
            # return labels[indice][0]
            return mode

    def isPure(self,group):
        splitXindxRange = group.shape[1] - 1
        labels = group[:, np.r_[splitXindxRange]]
        # print(labels)
        isSame = np.all(labels == labels[0])
        # print(isSame)
        return isSame

    def noMoreAttributes(self,group):
        splitXindxRange = group.shape[1] - 1
        X = group[:, np.r_[0:splitXindxRange]]
        X_transposed = np.transpose(X)
        noMore = []
        for column in range (0, len(X_transposed)):
        # for x in X_transposed:
            if len(self.getUniqueClassTable(X_transposed, column)) > 1:
                noMore.append(False)
            else:
                noMore.append(True)
        isSame = np.all(noMore == True)

        return isSame


    """ Generate the Information gain"""
    def gainRatio(self,subinfos):
        minElement = np.amin(np.array(subinfos))
        indxs = np.where(subinfos == np.amin(subinfos))[0]
        random_indx = indxs[0]
        return random_indx


    """ Generate the Info(S)"""
    def generalInfo(self, y):
        totalInfo = 0
        (unique, counts) = np.unique(y, return_counts=True)
        tableClasses = np.asarray((unique, counts)).T
        for labelClass in tableClasses:
            p = labelClass[1] / len(y)
            totalInfo += -p * math.log2(p)
        return totalInfo

    def getUniqueClassTable(self,X_transposed, columnIndx):
        (unique, counts) = np.unique(X_transposed[columnIndx], return_counts=True)
        tableClasses = np.asarray((unique, counts)).T
        return tableClasses

    def noMoreColumnAttributes(self,group,indx):
        isAttributeDone = True

        if  np.all(group[indx] == group[indx][0]):
            isAttributeDone = True
        else:
            isAttributeDone = False
        return isAttributeDone

    def getSplit(self,newXy,valueNodeSplit):
        # print("SPLIT")
        # print("XY")
        # print(newXy)
        X_transposed = np.transpose(newXy)
        robject = {}
        listInformations = []
        for column in range (0, len(X_transposed)-1):
            # print("check ", self.noMoreColumnAttributes(X_transposed,column))
            if self.noMoreColumnAttributes(X_transposed,column) == False:
                tableClasses = self.getUniqueClassTable(X_transposed,column)
                # print("tableClass 1")
                # print(tableClasses)
                PClassesFeatureInfo = []
                for columnClass in range(0, len(tableClasses)):
                    filter = np.asarray([tableClasses[columnClass][0]])
                    # print("filter",filter)
                    mask = np.in1d(newXy[:, column], filter)
                    allFromOneClass = newXy[mask]
                    # print("complete table ", newXy)
                    # print("allFromOneClass ",allFromOneClass)
                    partialP = len(allFromOneClass)/len(newXy)
                    # print("partial_P ", partialP)
                    labelP = 0
                    tableLabelOneClasses = self.getUniqueClassTable(np.transpose(allFromOneClass),-1)
                    # print("tableLabelOneClasses")
                    # print(tableLabelOneClasses)
                    for oneFeatureClass in tableLabelOneClasses:
                        p = oneFeatureClass[1]/len(allFromOneClass)
                        labelP += -p*math.log2(p)
                    # print("labelP ", labelP)
                    oneClassFeatureInfo = partialP * labelP
                    # print("Total p ", oneClassFeatureInfo)
                    PClassesFeatureInfo.append(oneClassFeatureInfo)
                total_PClassesFeatureInfo = np.sum(np.array(PClassesFeatureInfo))
                listInformations.append(total_PClassesFeatureInfo)
            else:
                listInformations.append(10000)
        branchMostInfoIndex = self.gainRatio(listInformations)
        # print("listInformations ", listInformations,branchMostInfoIndex)
        robject['value'] = valueNodeSplit
        robject['attribute_split'] =  branchMostInfoIndex
        robject['mostCommonValue'] = self.mostCommonValue(newXy)
        nodes = []
        groups = []
        ## this is the part ot append the groups##
        tableClasses = self.getUniqueClassTable(X_transposed, branchMostInfoIndex)
        # print("tableClass ", tableClasses )
        # print(branchMostInfoIndex,len(tableClasses),self.counts[branchMostInfoIndex])
        # print(tableClasses)
        for columnClass in range(0, len(tableClasses)):
            filter = np.asarray([tableClasses[columnClass][0]])
            # print("filter",filter)
            # print("column ",branchMostInfo)
            mask = np.in1d(newXy[:,branchMostInfoIndex], filter)
            newXy2 = newXy[mask]
            nodes.append(newXy2)
            valueNode = tableClasses[columnClass][0]
            groups.append(valueNode)
            # print("newXy2 ", newXy2)
            # X_transposed2 = np.transpose(newXy2)
        robject['groups'] = nodes
        robject['groupsValues'] = groups
        # print(robject)
        return robject

    def mostCommonValue(self,group):
        splitXindxRange = group.shape[1] - 1
        labels = group[:, np.r_[splitXindxRange]]
        labels = labels.reshape(labels.shape[0],)
        u, indices = np.unique(labels, return_inverse=True)
        mode_labels = u[np.argmax(np.bincount(indices))]
        return mode_labels

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        inds = np.where(np.isnan(X))
        col_mean = np.nanmean(X, axis=0)
        col_new_val = []
        for col in col_mean:
            col_new_val.append(-1)

        #Place column means in the indices. Align the arrays using take
        X[inds] = np.take(np.array(col_new_val), inds[1])
        labelsPred = []
        for row in X:
            pred_label = self.predictOneRow(row,self.three)
            # print("ROW ",row, "ROW_PRED", pred_label)
            labelsPred.append(pred_label)
        # print("*****************PREDICTIONS*************")
        # print(labelsPred)
        return labelsPred

    def predictOneRow(self,row, node):
        # print("***************************************")
        indx = node['attribute_split']
        nodeValue = 10000
        # print(list(node.keys()))
        filter = ['value','attribute_split','mostCommonValue']
        nodesArray = []
        for attri in list(node.keys()):
            if attri not in filter:
                nodesArray.append(attri)

        # for attributeClass in range(0,self.counts[indx]):
        for attributeClass in range(0,len(nodesArray)):
            # print(node)
            # print('nodeSplit'+str(attributeClass),self.counts[indx])
            # print(self.counts)
            # print(indx)
            nodo = node['nodeSplit'+str(attributeClass)]

            # print("NODO", node)
            # print("")
            # print(nodo)
            # print("")
            # if isinstance(nodo,dict):
            if nodo['value'] == row[indx]:
                if "label" in nodo:
                    return nodo['label']
                else:
                    nodeValue = self.predictOneRow(row,nodo)
            # else:
            #     return nodo
        return nodeValue

    def score(self, X, y):
        """ Return accuracy(Classification Acc) of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array of the targets
        """
        predicted_labels = np.array(self.predict(X))
        y_reshaped = y.reshape(y.shape[0],)
        matches = []
        for indx in range(0,len(predicted_labels)):
            if y_reshaped[indx] == predicted_labels[indx]:
                matches.append(indx)
        accuracy = len(matches)/len(y)

        return accuracy
