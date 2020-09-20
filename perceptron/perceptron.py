import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from prettytable import PrettyTable
### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True, deterministic=10,initial_weights=None,stopCriteria=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.deterministic = deterministic
        self.initial_weights = initial_weights
        self.stopCriteria = 0.015
    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        weightSize = X.shape[1]
        if self.initial_weights is None:
            self.initial_weights = self.initialize_weights(weightSize) if not initial_weights else initial_weights

        # self.initial_weights = self.initialize_weights(weightSize) if not initial_weights else initial_weights
        weightsCopy = self.initial_weights.copy()
        ## Add the bias to each one of the X patterns ##
        biasArray = np.full((X.shape[0],1),1)
        numberOfEpochWithNoImprovement = [];
        i = 1
        t = PrettyTable(['Epoch','Weights', 'first RMSE','after RSME','DeltaAccuracy < 0.01','number of Epochs without Improvement' ])
        ##Loop through each Epoch

        while len(numberOfEpochWithNoImprovement) < self.deterministic:
            X_shuffled,y_shuffled = self._shuffle_data(X,y)
            initialEpochAccuracy = self.getRSME(X_shuffled,y_shuffled)
            X_bias = np.concatenate((X_shuffled,biasArray),axis=1)

            for x_unit,label_unit in zip(X_bias, y_shuffled):
                netValue = x_unit.dot(weightsCopy)
                output = self._get_ouput(netValue)
                changeFactor = (label_unit[0] - output) * self.lr
                changeWeights = x_unit * changeFactor
                weightsCopy = weightsCopy + changeWeights

            self.initial_weights = weightsCopy
            # X_shuffled2, y_shuffled2 = self._shuffle_data(X_shuffled,y_shuffled)
            lastEpochAccuracy = self.getRSME(X_shuffled,y_shuffled)
            self._stopOrNot(initialEpochAccuracy,lastEpochAccuracy, numberOfEpochWithNoImprovement)
            t.add_row([i,self.initial_weights,initialEpochAccuracy,lastEpochAccuracy,abs(lastEpochAccuracy-initialEpochAccuracy),len(numberOfEpochWithNoImprovement)])

            i = i+1

        # print(t)
        print("Number of Epochs = ", i)
        return self

    def _stopOrNot(self,initialEpochAccuracy,lastEpochAccuracy,numberOfEpochWithNoImprovement):
        # print(initialEpochAccuracy)
        # print(lastEpochAccuracy)
        if (abs(initialEpochAccuracy - lastEpochAccuracy) < self.stopCriteria) or (initialEpochAccuracy > lastEpochAccuracy):
            numberOfEpochWithNoImprovement.append(abs(initialEpochAccuracy - lastEpochAccuracy))
        else:
            numberOfEpochWithNoImprovement.clear()
        return

    def _get_ouput(self,netValue):
        if netValue > 0:
            return 1
        else:
            return 0
    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        predicted_targets = []
        biasArray = np.full((X.shape[0],1),1)
        X_bias = np.concatenate((X,biasArray),axis=1)
        for x in X_bias:
            predicted_target= self._get_ouput(x.dot(self.initial_weights))
            predicted_targets.append(predicted_target)

        r_pred_targets = np.array(predicted_targets)
        # print(r_pred_targets)
        # print(r_pred_targets)
        return r_pred_targets

    def initialize_weights(self, Xmagnitude):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        initialWeight = np.random.uniform(low = -1, high=1, size=(Xmagnitude+1,))
        # initialWeight= np.zeros(Xmagnitude+1)

        ##add the bias
        print("Initial Weights..",initialWeight)
        return initialWeight

    def getRSME(self, X, y):
        target_pred = self.predict(X)
        y_reshaped = y.reshape(y.shape[0],)
        sse = np.sum((y_reshaped - target_pred)**2)
        mse = sse/y_reshaped.shape[0]
        rmse= mse**(1/2)

        # print("The RMSE is ",rmse)
        return rmse
    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        target_pred = self.predict(X)
        y_reshaped = y.reshape(y.shape[0],)
        correctPredictions = []
        correctPredictionsCount = 0
        for tp, y in zip(target_pred,y_reshaped):
            if tp ==y:
                correctPredictionsCount = correctPredictionsCount +1
                correctPredictions.append(correctPredictionsCount)

        accuracy_porcetange = correctPredictionsCount/y_reshaped.shape[0]

        # print("The RMSE is ",rmse)
        return accuracy_porcetange

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        if self.shuffle:
            X_unshuffled = X.copy()
            y_unshuffled = y.copy()
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            X_shuffled= X_unshuffled[idx]
            y_shuffled= y_unshuffled[idx]
            return [X_shuffled, y_shuffled]
        else:
            return[X,y]


    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        good_weights = []
        for weight in self.initial_weights:
            weight = round(weight,3)
            good_weights.append(weight)
        self.initial_weights = good_weights
        return self.initial_weights
