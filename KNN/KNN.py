import numpy as np
import math
import operator
from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, ClassifierMixin

class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self,columntype=[],weight_type='inverse_distance'): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal[categoritcal].
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.columntype = columntype #Note This won't be needed until part 5
        self.weight_type = weight_type
        self.labelType = None
        self.X = None
        self.y = None


    def fit(self,data,labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.X = data
        self.y = labels

        return self
    def predict(self,data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """

        predictions = []
        if self.labelType == "classification":
            if 'nominal' in self.columntype:
                pass
            else:
                # newShapeLabel = self.y.reshape(self.y.shape[0],)
                uniqueLabels = np.unique(self.y)
                for xdata in data:
                    listSumVotes = []
                    distances = []
                    ## getting distances
                    for x in self.X:
                        summation = np.add(x,xdata)
                        summationSquare = np.square(summation)
                        totalSummation = np.sum(summationSquare)
                        totalSummationSquareRoot = sqrt(totalSummation)
                        if self.weight_type =='inverse_distance':
                            totalSummationSquareRoot = 1 / (totalSummationSquareRoot)**2
                        distances.append(totalSummationSquareRoot)
                    ## classificando las labels
                    for yunique in uniqueLabels:
                        listfeature = []
                        for index in range(0,len(self.y)):
                            if self.y[index][0] == yunique:
                                listfeature.append(distances[index])
                        total = np.sum(np.array(listfeature))
                        listSumVotes.append(total)

                    indx, value = max(enumerate(listSumVotes), key=operator.itemgetter(1))
                    predictions.append(uniqueLabels[indx])


        return predictions
    def hotEncoder(self, feature):
        enc = OneHotEncoder()
        enc.fit(feature)
        featureTrans = enc.transform(feature).toarray()
        return featureTrans

    def euclidianDistanceNominal(self,feature1, feature2):
        isTheSame = np.array_equal(feature1, feature2)
        if isTheSame:
            return 0
        else:
            return math.sqrt(2)

    #Returns the Mean score given input data and labels
    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
        		X (array-like): A 2D numpy array with data, excluding targets
        		y (array-like): A 2D numpy array with targets
        Returns:
        		score : float
        				Mean accuracy of self.predict(X) wrt. y.
        """
        preds = self.predict(X)
        y_reshaped = y.reshape(y.shape[0],)
        matches = []
        for indx in range(0,len(predicted_labels)):
            if y_reshaped[indx] == predicted_labels[indx]:
                matches.append(indx)
        accuracy = len(matches)/len(y)

        return accuracy
