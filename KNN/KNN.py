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
        if self.labelType == "classification":
            if 'nominal' in self.columntype:
                pass
            else:
                newXY= np.concatenate((self.X, self.y), axis = 1).tolist()
                # print(newXY)
                for dataX in data:
                    prediction = self.predict_classification(newXY,dataX)
                    preds.append(prediction)
        return preds
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
    # calculate the Euclidean distance between two vectors
    def euclidean_distance(self,row1, row2):
        distance = 0.0
        for i in range(0,len(row1)):
            distance += (row1[i] - row2[i])**2
            # print(row1[i], row2[i],(row1[i] - row2[i])**2)
        return math.sqrt(distance)

    def weighting_criteria(self,distance):
        if self.weight_type == 'inverse_distance':
            distance_invert = 1/ distance**2
            return distance_invert
        else:
            return distance

    def get_neighbors(self,train, test_row):
        distances = list()
        ds = []
        for train_row in train:
            dist = self.euclidean_distance(test_row, train_row)
            dist2 = self.weighting_criteria(dist)
            distances.append((train_row, dist2))
            ds.append(dist2)
        # print(ds)
        distances.sort(key=lambda tup: tup[1], reverse=True)
        ds.sort()
        # print(distances)
        # print(ds)
        neighbors = list()
        for i in range(self.k):
            neighbors.append(distances[i][0])
        print(neighbors)
        return neighbors

    def predict_classification(self,train, test_row):
        neighbors = self.get_neighbors(train, test_row)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction
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
        print(preds)
        y_reshaped = y.reshape(y.shape[0],)
        print(y_reshaped)

        matches = []
        for indx in range(0,len(preds)):
            if y_reshaped[indx] == preds[indx]:
                matches.append(indx)
        # print(matches)
        accuracy = len(matches)/len(y_reshaped)

        return accuracy
