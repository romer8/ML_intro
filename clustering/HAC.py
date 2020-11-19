import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial import distance_matrix
class HACClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,link_type='single'): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k
        self.clusterDict = {}
        self.output = []
    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        # print(X)
        distancesMatrix = self._getDistancesMatrix(X,2)
        distanceLowerTriangle = np.tril(distancesMatrix, -1)
        initialLevel = 0
        self.clusterDict = {initialLevel:{}}
        for i in range(0, len(X)):
            self.clusterDict[initialLevel][i] = [i]
        print(self.clusterDict)
        # for i in range(0, self.k):
        numClusters = len(list(self.clusterDict[initialLevel].keys()))
        i= 0
        while numClusters > self.k:
            if len(distanceLowerTriangle) > 2:
                print("ITERATION",i, numClusters)
                # print(distanceLowerTriangle)
                distanceLowerTriangle[distanceLowerTriangle <= 0 ] = np.inf
                minVal,rowIndex,colIndex = self.typeLinkDistance(distanceLowerTriangle)
                distanceLowerTriangle[distanceLowerTriangle == np.inf ] = 0
                print("MIN VALUE")
                large_width = 400
                np.set_printoptions(linewidth=large_width,precision=4)
                print(minVal,rowIndex,colIndex) ## DELETE
                print(np.around(distanceLowerTriangle,decimals = 4))## DELETE

                valMin = min(rowIndex,colIndex)
                valMax = max(rowIndex,colIndex)
                initialLevel = initialLevel + 1
                self.clusterDict[initialLevel] = self._Hlevel(initialLevel,valMax,valMin)
                newClusterValues = self._getNewClusterValues(distanceLowerTriangle, rowIndex, colIndex)
                distanceLowerTriangle = self._updateDistanceMatrix(distanceLowerTriangle, newClusterValues,rowIndex,colIndex)

                numClusters = len(list(self.clusterDict[initialLevel].keys()))
                # print(self._helperClusterDictPrinter(initialLevel))

                # return
                # print(newClusterValues)#### DELETE
                print(self.clusterDict[initialLevel]) #### DELETE


            i= i +1
        # self._helperClusterDictPrinter()
        print("*********************************************")
        print(self.clusterDict[initialLevel])
        self.output = self._createOutputClusters(initialLevel,X)
        self._printOutput()
        return self

    def typeLinkDistance(self,distanceLowerTriangle):
        minVal = np.min(distanceLowerTriangle)
        minValIndexes = np.argwhere(distanceLowerTriangle == minVal)[0]
        # print("just checking",np.argwhere(distanceLowerTriangle == minVal))
        return [minVal,minValIndexes[0],minValIndexes[1]]

    def _helperClusterDictPrinter(self,level):
        keysDict = list(self.clusterDict[level].keys())
        newdict={}
        for i in keysDict :
            newdict[i]= len(self.clusterDict[level][i])
        return print(newdict)


    def _Hlevel(self,level, maxIn, minIn):
        dicti = {}
        fatherDictKeys= list(self.clusterDict[level-1].keys())
        for i in fatherDictKeys:
            vals = self.clusterDict[level-1][i].copy()
            if i == minIn:
                newvals =vals + self.clusterDict[level-1][maxIn]
                dicti[i] = newvals
            elif i == maxIn or maxIn < i:
                # ke = i+1
                if i+1  in self.clusterDict[level-1]:
                    vals2 = self.clusterDict[level-1][i+1].copy()
                    dicti[i] = vals2
            else:
                dicti[i]=vals
        # del dicti[fatherDictKeys[-1]]
        return dicti

    def _updateDistanceMatrix(self,distanceLowerTriangle, newClusterValues,rowIndex, colIndex):
        # distanceLowerTriangle = np.delete(distanceLowerTriangle, [rowIndex, colIndex], axis=0)
        # distanceLowerTriangle = np.delete(distanceLowerTriangle, [rowIndex, colIndex], axis=1)
        # valToInser = min(rowIndex,colIndex)
        # newClusterValues= newClusterValues.reshape(newClusterValues.shape[0],1)
        # # print(newClusterValues.shape)
        # # print(distanceLowerTriangle.shape)
        # distanceLowerTriangle = np.hstack((distanceLowerTriangle[:,:valToInser], newClusterValues, distanceLowerTriangle[:,valToInser:]))
        # newClusterValues= newClusterValues.reshape(newClusterValues.shape[0],)
        # newClusterValues= np.insert(newClusterValues, valToInser,0)
        # distanceLowerTriangle = np.insert(distanceLowerTriangle,valToInser,newClusterValues, 0)
        # distanceLowerTriangle = np.tril(distanceLowerTriangle, -1)
        # newDistanceLowerTriangle = distanceLowerTriangle

        valToDelete = max(rowIndex,colIndex)
        valToInser = min(rowIndex,colIndex)

        distanceLowerTriangle2 = np.array(distanceLowerTriangle, copy= True)
        # print(distanceLowerTriangle2)
        distanceLowerTriangle2 = np.delete(distanceLowerTriangle2, [valToDelete], axis=0)
        distanceLowerTriangle2 = np.delete(distanceLowerTriangle2, [valToDelete], axis=1)
        # print(distanceLowerTriangle2)
        distanceLowerTriangle2[valToInser] = newClusterValues
        # print(distanceLowerTriangle2)
        distanceLowerTriangle2 = np.transpose(distanceLowerTriangle2)
        distanceLowerTriangle2[valToInser] = newClusterValues
        # print(distanceLowerTriangle2)
        distanceLowerTriangle2 = np.transpose(distanceLowerTriangle2)
        # print(distanceLowerTriangle2)
        distanceLowerTriangle2 = np.tril(distanceLowerTriangle2, -1)
        print(distanceLowerTriangle2)
        newDistanceLowerTriangle = distanceLowerTriangle2
        # print("####################################")



        return newDistanceLowerTriangle

    def _getNewClusterValues(self, distanceLowerTriangle, rowIndex, colIndex):
        rowList = np.array(distanceLowerTriangle[rowIndex,:],copy = True)
        colList = np.array(distanceLowerTriangle[:,colIndex],copy = True)

        colList1 = np.array(distanceLowerTriangle[colIndex,:],copy = True)
        colList2 = np.array(distanceLowerTriangle[:,colIndex],copy = True)

        rowList1 = np.array(distanceLowerTriangle[rowIndex,:],copy = True)
        rowList2 = np.array(distanceLowerTriangle[:,rowIndex],copy = True)
        # print(rowList1)
        # print(rowList2)
        valToHaveZero = min(rowIndex,colIndex)
        valToDelete = max(rowIndex,colIndex)
        print("FIRST HALF")
        rl = np.add(rowList1,rowList2)
        # rl = np.delete(rl,[rowIndex,colIndex])
        print(rl)
        print("SECOND HALF")
        rl2 = np.add(colList1,colList2)
        print(rl2)
        # rl2 = np.delete(rl2,[rowIndex,colIndex])

        # print(rl)
        # print(rl2)
        if self.link_type =='single':
            rl3 = np.minimum(rl, rl2)
            rl3[valToHaveZero] = 0
            rl3 = np.delete(rl3,[valToDelete])

            # print(rl3)
            # rl_final = rl3[rl3 != 0]
            # print(rl_final)
            # return rl_final
            print("NEW VALUES TO REPLACE")
            print(rl3)
            return rl3
        if self.link_type =='complete':
            rl3 = np.maximum(rl, rl2)
            # print(rl3)
            rl3[valToHaveZero] = 0
            # print(rl3)
            rl3 = np.delete(rl3,[valToDelete])
            # print(rl3)

            # print(rl3)
            # rl_final = rl3[rl3 != 0]
            # print(rl_final)
            # return rl_final
            return rl3

        # experi = rowList1[:rowIndex]
        # experi2 = rowList2[rowIndex+1:]
        #
        # experi3 = colList1[:colIndex]
        # experi4 = colList2[colIndex+1:]
        # expeFinal = np.concatenate((experi, experi2), axis=None)
        # expeFinal2 = np.concatenate((experi3, experi4), axis=None)
        #
        # print("EXPERI1", expeFinal)
        # print("EXPERI2", expeFinal2)
        # if self.link_type =='single':
        #     newClusterValues = np.minimum(expeFinal, expeFinal2)
        #     print("FIN_ESPERI", newClusterValues[1:])
        #     return newClusterValues[1:]
        # if self.link_type ==' complete':
        #     newClusterValues = np.maximun(expeFinal, expeFinal2)
        #     # print("FIN_ESPERI", newClusterValues[1:])
        #     # return newClusterValues[1:]
        #
        # for i in range(0,colIndex):
        #     # print(i)
        #     for j in range(0,len(colList)):
        #         if colList[j] == 0:
        #             colList[j] = distanceLowerTriangle[:,i][j]
        #
        # for i in range(len(distanceLowerTriangle)-1,rowIndex,-1):
        #     for j in range(0,len(rowList)):
        #         if rowList[j] == 0:
        #             rowList[j] = distanceLowerTriangle[i,:][j]
        #
        # print(np.around(rowList,decimals=4))
        # print(np.around(colList,decimals=4))
        # rowList = np.delete(rowList, colIndex)
        # colList = np.delete(colList, rowIndex)
        # colList = colList[colList != 0]
        # rowList = rowList[rowList != 0]
        #
        # if self.link_type =='single':
        #     newClusterValues = np.minimum(rowList, colList)
        #     print(newClusterValues)
        #     return newClusterValues
        # if self.link_type ==' complete':
        #     newClusterValues = np.maximum(rowList, colList)
        #     print(newClusterValues)
        #     return newClusterValues

    def _printOutput(self):
        for i in self.output:
            print(i)
        return
    def _createOutputClusters(self,level,X):
        arrayOut = []
        for clusterIndx in list(self.clusterDict[level].keys()):
            centroid_ = self._getClusterCentroid(X,self.clusterDict[level][clusterIndx])
            clusterSize = self._getSizeCluster(self.clusterDict[level][clusterIndx])
            SSE = self._getClusterSSE(X,self.clusterDict[level][clusterIndx],centroid_)
            arrayOut.append([np.around(centroid_,decimals=4),clusterSize,SSE])
        return arrayOut

    def _getClusterCentroid(self,X, clusterArray):
        listVals = []
        for clusterIndx in clusterArray:
            listVals.append(X[clusterIndx])

        centroid = np.mean(listVals, axis=0)
        return centroid

    def _getSizeCluster(self, clusterArray):
        return len(clusterArray)

    def _getClusterSSE(self, X, clusterArray, clusterCentroid):
        listVals = []
        for clusterIndx in clusterArray:
            listVals.append(X[clusterIndx])

        distances = np.linalg.norm(np.array(listVals) - clusterCentroid, axis=1)
        distances = np.square(distances)
        SSE =np.sum(distances)
        return SSE

    def _getDistancesMatrix(self,norm_data,distance_type):
        matrix_distance = distance_matrix(norm_data, norm_data, distance_type)
        return matrix_distance

    def save_clusters(self,filename):
        """
            f = open(filename,"w+")
            Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,separator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
            f.close()
        """
