import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin

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

    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        SINFO = self.generalInfo(y)
        newXy = np.concatenate((X, y), axis = 1)
        X_transposed = np.transpose(newXy)
        # uniqueLabels = numpy.unique(y)
        listInformations = []
        for column in range (0, len(self.counts)-1):
            (unique, counts) = np.unique(X_transposed[column], return_counts=True)
            tableClasses = np.asarray((unique, counts)).T
            PClassesFeatureInfo = []
            for columnClass in range(0, len(tableClasses)):
                filter = np.asarray([tableClasses[columnClass][0]])
                allFromOneClass = X[np.in1d(X[:, column], filter)]
                partialP = len(allFromOneClass)/len(newXy)
                labelP = 0
                (uniqueLabel, countsLabels) = np.unique(allFromOneClass[column], return_counts=True)
                tableOneClasses = np.asarray((uniqueLabel, countsLabels)).T
                for oneFeatureClass in tableOneClasses:
                    p = oneFeatureClass[1]/len(allFromOneClass)
                    labelP += -p*math.log2(p)

                oneClassFeatureInfo = partialP * labelP
                PClassesFeatureInfo.append(oneClassFeatureInfo)
            total_PClassesFeatureInfo = np.sum(np.array(PClassesFeatureInfo))
            listInformations.append(total_PClassesFeatureInfo)

        branchMostInfo = gainRatio(listInformations)
        print(branchMostInfo)
        return self

    """ Generate the Information gain"""
    def gainRatio(self,subinfos):
        minElement = np.amin(np.array(subinfos))
        indx = np.where(subinfos == np.amin(subinfos))
        return indx
    """ Generate the Info(S)"""

    def generalInfo(self, y):
        totalInfo = 0
        (unique, counts) = np.unique(y, return_counts=True)
        tableClasses = np.asarray((unique, counts)).T
        # print("tableClasses")
        # print(tableClasses)
        for labelClass in tableClasses:
            p = labelClass[1] / len(y)
            totalInfo += -p * math.log2(p)
        # print(totalInfo)
        return totalInfo


    def featureInfo(self,X,y):
        # (unique, counts) = numpy.unique(Xs, return_counts=True)
        # tableClasses = np.asarray((unique, counts)).T
        # for labelClass in tableClasses:
        #     if

        return 0
    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass


    def score(self, X, y):
        """ Return accuracy(Classification Acc) of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array of the targets
        """
        return 0
