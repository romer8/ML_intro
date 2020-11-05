import numpy as np
import math
import operator
from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, ClassifierMixin

class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self,columntype=[], labelType = 'classification', weight_type='inverse_distance', k = 3): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal[categoritcal].
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.columntype = columntype #Note This won't be needed until part 5
        self.weight_type = weight_type
        self.labelType = labelType
        self.k = k
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

        preds = []

        for dataX in data:
            prediction = self.predict_classification(dataX)
            preds.append(prediction)
        return preds

    def weighting_criteria(self,distances):
        if self.weight_type == 'inverse_distance':
            distancesw =np.power(distances,-2)
            indices = np.argsort(distancesw)[::-1][:self.k]
            return indices

        else:
            indices = np.argsort(distances)[:self.k]
            return indices

    def get_neighbors(self,test_row):
        distances = np.linalg.norm(self.X - test_row, axis=1)
        indices = self.weighting_criteria(distances)
        if self.labelType == "classification":

            neighbors = []
            uniquevals = np.unique(self.y)
            sums =[]
            for unq in uniquevals:
                sum = 0
                for indx in indices:
                    if self.y[indx][0] == unq:
                        sum = sum + distances[indx]
                neighbors.append(sum)
            max = np.max(neighbors)
            result = np.where(neighbors == max)[0]
            return uniquevals[result][0]

        else:
            if self.weight_type == 'no_weight':
                total = 0
                for indx in indices:
                    total = total + self.y[indx][0]
                return total/self.k
            else:
                total_num = 0
                total_den = 0
                for indx in indices:
                    total_num = total_num + (self.y[indx][0] / (distances[indx])**2)
                    total_den = total_den + (1/(distances[indx])**2)
                return total_num/total_den

    def predict_classification(self,test_row):
        neighbors = self.get_neighbors(test_row)
        return neighbors

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
        if self.labelType == 'classification':
            matches = []
            for indx in range(0,len(preds)):
                if y_reshaped[indx] == preds[indx]:
                    matches.append(indx)
            accuracy = len(matches)/len(y_reshaped)
            return accuracy

        else:
            mse = 0
            arraymse =[]
            for indx in range(0,len(preds)):
                mse = mse + (y_reshaped[indx] - preds[indx])**2
                arraymse.append((y_reshaped[indx] - preds[indx])**2)

            accuracy = mse/len(y_reshaped)
            return accuracy
