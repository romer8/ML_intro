import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class KMEANSClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,debug=False): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug
        self.clusterDict = {}
        self.clustersCentroids = []
    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self._initCentroidDictAndCentroids(X)
        totalSSEDifference = 10
        iterations = 0
        while totalSSEDifference > 0.001:
            print("***************ITERATION ", iterations, "************")
            totalSSE_before = 1000
            if iterations > 0:
                totalSSE_before = self._totalClusterSSE(X)

            for indx in range(0, len(X)):
                self._getKmeanFit(X,X[indx],indx)
            self._updateCentroid(X)
            totalSSE_after = self._totalClusterSSE(X)
            # print(totalSSE_after)
            totalSSEDifference = abs(totalSSE_after- totalSSE_before)
            # print(self.clusterDict)
            print(self._printClustersLength())

            print("TOTAL_SSE",totalSSE_before,totalSSE_after, totalSSEDifference)
            iterations += 1
        print(self._printClustersLength())
        return self

    def _initCentroidDictAndCentroids(self,X):
        print(np.random.choice(len(X), self.k))
        for i in range(0, self.k):
            # self.clusterDict[i] = [i]
            self.clusterDict[i] = []
        print(self.clusterDict)
        if self.debug is True:
            for clusterIndx in range(0, self.k):
                self.clustersCentroids.append(X[clusterIndx])
        else:
            indixes = np.random.choice(len(X), self.k)
            for clusterIndx in range(0, indixes):
                self.clustersCentroids.append(X[clusterIndx])

        print(self.clustersCentroids)
        return

    def _getClusterSSE(self, indexCluster,X):
        # print(np.array(self.clusterDict[indexCluster]))
        clusterVals = self._getClusters(X, indexCluster)
        # print(self.clustersCentroids[indexCluster])
        distances = np.linalg.norm(np.array(clusterVals) - self.clustersCentroids[indexCluster], axis=1)
        distances = np.square(distances)
        SSE =np.sum(distances)
        return SSE

    def _totalClusterSSE(self,X):
        totalSSE = 0
        for indx in range(self.k):
            totalSSE = totalSSE + self._getClusterSSE(indx,X)
        return totalSSE

    def _getKmeanFit(self,X,Xval,XIndex):
        indexMinDistance =  self._getClusterDistances(Xval)
        self._getAllClusterCentroidUpdate(X,XIndex, indexMinDistance)
        return

    def _getAllClusterCentroidUpdate(self,X,Xindex, indexCluster):
        for i in range(len(list(self.clusterDict.keys()))):
            if indexCluster == i:
                if Xindex not in self.clusterDict[indexCluster]:
                    self.clusterDict[indexCluster].append(Xindex)
            else:
                if Xindex in self.clusterDict[i]:
                    self.clusterDict[i].remove(Xindex)
        return
    def _updateCentroid(self,X):
        for i in range(0,len(self.clustersCentroids)):
            clusterVals = self._getClusters(X, i)
            self.clustersCentroids[i] = np.mean(clusterVals, axis=0)
        return
    def _getClusterDistances(self, Xval):

        # print("primer",np.array(clusterCentroids).shape, Xval.shape)
        # print(np.array(clusterCentroids))
        # print("my Xval")
        # print(Xval)
        distances = np.linalg.norm(np.array(self.clustersCentroids) - Xval, axis=1)
        minVal = np.min(distances)
        minValIndexes = np.argwhere(distances == minVal)[0]
        return minValIndexes[0]

    def _getClusters(self, X, indx):
        listVals = []
        for clusterIndx in self.clusterDict[indx]:
            listVals.append(X[clusterIndx])
        return listVals
    def _printClustersLength(self):
        listLenght = {}
        for i in range(len(list(self.clusterDict.keys()))):
            listLenght[i] = len(self.clusterDict[i])
        return listLenght
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
