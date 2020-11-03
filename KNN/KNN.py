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
        total = len(self.X)
        if self.labelType == "classification":
            # if 'nominal' in self.columntype:
            #     pass
            # else:
            # newXY= np.concatenate((self.X, self.y), axis = 1).tolist()
            # newXY= self.X
            # print(newXY)
            for dataX in data:
                prediction = self.predict_classification2(dataX)
                # print(prediction)
                preds.append(prediction)
                # print(len(preds)/total)
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

    # def weighting_criteria(self,distance):
    #     if self.weight_type == 'inverse_distance':
    #         distance_invert = 1/ distance**2
    #         return distance_invert
    #     else:
    #         return distance
    def weighting_criteria(self,distances):
        if self.weight_type == 'inverse_distance':
            distancesw =np.power(distances,-2)
            indices = np.argsort(distancesw)[::-1][:self.k]
            return indices

        else:
            indices = np.argsort(distances)[:self.k]
            return indices

    def get_neighbors(self,train, test_row):
        distances = list()
        for train_row in train:
            dist = self.euclidean_distance(test_row, train_row)
            dist2 = self.weighting_criteria(dist)
            # distances.append((train_row, dist2))
            distances.append(dist)
        sum = np.sqrt(np.sum(np.array(distances)))
        # distances.sort(key=lambda tup: tup[1], reverse=True)
        # neighbors = list()
        # for i in range(self.k):
        #     neighbors.append(distances[i][0])
        # return neighbors
        return sum
    def get_neighbors2(self,test_row):
        # distances = list()
        distances = np.linalg.norm(self.X - test_row, axis=1)
        indices = self.weighting_criteria(distances)
        # distances = self.weighting_criteria(distances)
        # indices = np.argsort(distances)[::-1][:self.k]
        # distances[::-1].sort()
        neighbors = []
        uniquevals = np.unique(self.y)
        sums =[]
        for unq in uniquevals:
            sum = 0
            for indx in indices:
                if self.y[indx][0] == unq:
                    sum = sum + distances[indx]
                # neighbors.append(distances[indx],self.y[indx][0])
            neighbors.append(sum)
        max = np.max(neighbors)
        result = np.where(neighbors == max)[0]
        # print(type(distances))
        # distances.sort(key=lambda tup: tup[1], reverse=True)
        #
        # neighbors = list()
        # for i in range(self.k):
        #     neighbors.append(distances[i][0])
        # return neighbors
        return uniquevals[result]
        # return distances

    # def predict_classification(self,train, test_row):
    #     neighbors = self.get_neighbors2(train, test_row)
    #     # output_values = [row[-1] for row in neighbors]
    #     # prediction = max(set(output_values), key=output_values.count)
    #     prediction = max(set(neighbors), key=neighbors.count)
    #     return prediction
    #     # return neighbors
    def predict_classification2(self,test_row):
        neighbors = self.get_neighbors2(test_row)

        # prediction = max(set(neighbors), key=neighbors.count)
        return neighbors
        # return neighbors
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
        # print(y_reshaped)

        matches = []
        for indx in range(0,len(preds)):
            if y_reshaped[indx] == preds[indx]:
                matches.append(indx)
        # print(matches)
        accuracy = len(matches)/len(y_reshaped)

        return accuracy
