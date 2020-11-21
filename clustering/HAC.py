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
        self.totalSSE = 0
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
        numClusters = len(list(self.clusterDict[initialLevel].keys()))
        i= 0
        while numClusters > self.k:
            if len(distanceLowerTriangle) > 2:
                large_width = 400
                np.set_printoptions(linewidth=large_width,precision=4)
                distanceLowerTriangle[distanceLowerTriangle <= 0 ] = np.inf
                minVal,rowIndex,colIndex = self.typeLinkDistance(distanceLowerTriangle)
                distanceLowerTriangle[distanceLowerTriangle == np.inf ] = 0
                valMin = min(rowIndex,colIndex)
                valMax = max(rowIndex,colIndex)
                initialLevel = initialLevel + 1
                self.clusterDict[initialLevel] = self._Hlevel(initialLevel,valMax,valMin)
                newClusterValues = self._getNewClusterValues(distanceLowerTriangle, rowIndex, colIndex)
                distanceLowerTriangle = self._updateDistanceMatrix(distanceLowerTriangle, newClusterValues,rowIndex,colIndex)
                numClusters = len(list(self.clusterDict[initialLevel].keys()))

            i= i +1

        self.output = self._createOutputClusters(initialLevel,X)
        return self

    def typeLinkDistance(self,distanceLowerTriangle):
        minVal = np.min(distanceLowerTriangle)
        minValIndexes = np.argwhere(distanceLowerTriangle == minVal)[0]
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
                if i+1  in self.clusterDict[level-1]:
                    vals2 = self.clusterDict[level-1][i+1].copy()
                    dicti[i] = vals2
            else:
                dicti[i]=vals
        return dicti

    def _updateDistanceMatrix(self,distanceLowerTriangle, newClusterValues,rowIndex, colIndex):
        valToDelete = max(rowIndex,colIndex)
        valToInser = min(rowIndex,colIndex)

        distanceLowerTriangle2 = np.array(distanceLowerTriangle, copy= True)
        distanceLowerTriangle2 = np.delete(distanceLowerTriangle2, [valToDelete], axis=0)
        distanceLowerTriangle2 = np.delete(distanceLowerTriangle2, [valToDelete], axis=1)
        distanceLowerTriangle2[valToInser] = newClusterValues
        distanceLowerTriangle2 = np.transpose(distanceLowerTriangle2)
        distanceLowerTriangle2[valToInser] = newClusterValues
        distanceLowerTriangle2 = np.transpose(distanceLowerTriangle2)
        distanceLowerTriangle2 = np.tril(distanceLowerTriangle2, -1)
        newDistanceLowerTriangle = distanceLowerTriangle2
        return newDistanceLowerTriangle

    def _getNewClusterValues(self, distanceLowerTriangle, rowIndex, colIndex):
        colList1 = np.array(distanceLowerTriangle[colIndex,:],copy = True)
        colList2 = np.array(distanceLowerTriangle[:,colIndex],copy = True)

        rowList1 = np.array(distanceLowerTriangle[rowIndex,:],copy = True)
        rowList2 = np.array(distanceLowerTriangle[:,rowIndex],copy = True)

        valToHaveZero = min(rowIndex,colIndex)
        valToDelete = max(rowIndex,colIndex)
        rl = np.add(rowList1,rowList2)
        rl2 = np.add(colList1,colList2)
        if self.link_type =='single':
            rl3 = np.minimum(rl, rl2)
            rl3[valToHaveZero] = 0
            rl3 = np.delete(rl3,[valToDelete])
            return rl3
        if self.link_type =='complete':
            rl3 = np.maximum(rl, rl2)
            rl3[valToHaveZero] = 0
            rl3 = np.delete(rl3,[valToDelete])
            return rl3

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
            self.totalSSE = self.totalSSE + SSE
            arrayOut.append([centroid_,clusterSize,SSE])
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
        f = open(filename,"w+")
        # Used for grading.
        f.write("{:d}\n".format(self.k))
        f.write("{:.4f}\n\n".format(self.totalSSE))
        # for each cluster and centroid:
        for cluster in self.output:
            f.write(np.array2string(cluster[0],precision=4,separator=","))
            f.write("\n")
            f.write("{:d}\n".format(cluster[1]))
            f.write("{:.4f}\n\n".format(cluster[2]))
        f.close()
