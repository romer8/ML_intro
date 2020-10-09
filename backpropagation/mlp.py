import numpy as np
import math
import random
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,lr=.1, momentum=0, shuffle=True,deterministic= 10,hidden_layer_widths=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle(boolean): Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
            momentum(float): The momentum coefficent
        Optional Args (Args we think will make your life easier):
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer if hidden layer is none do twice as many hidden nodes as input nodes.
        Example:
            mlp = MLPClassifier(lr=.2,momentum=.5,shuffle=False,hidden_layer_widths = [3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.weights = None
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.deterministic = deterministic

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Optional Args (Args we think will make your life easier):
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        if self.weights is None:
            self.weights = self.initialize_weights(X,y)

        #Initialize network ##
        networkObject = self._initialize_network()

        #
        # X_shuffled,y_shuffled = self._shuffle_data(X,y)
        # biasArray = np.full((X.shape[0],1),1)
        # X_bias = np.concatenate((X_shuffled,biasArray),axis=1)
        # ### Initialize the network and print###
        # print("network Initialization")
        # print(networkObject)
        #
        # print("forward")
        # self._forwardProp(X_bias,networkObject)
        # print(networkObject)
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass

    def initialize_weights(self,X,y):
        """ Initialize weights for perceptron. Don't forget the bias!
            Use one layer of hidden nodes with the number of hidden nodes being twice the number of inputs.

        Returns:

        """

        networkWeights = []
        sizeWidthInputs = len(X[0]) + 1
        sizeWidthOutputs = len(np.unique(y))
        # unique, counts = np.unique(x, return_counts=True)
        sizeWidthHiddenLayersDefault = len(X[0]) * 2 + 1
        randomValue = 0 # random.uniform(0 , 0.00001)
        if self.hidden_layer_widths is None:
            hidden_layer = [{'weights':[randomValue for i in range(sizeWidthInputs)]} for j in range(sizeWidthHiddenLayersDefault)]
            # print(hidden_layer)
            networkWeights.append(hidden_layer)

        else:
            sizeWidthHiddenLayersCustomized = self.hidden_layer_widths[0]
            for hiddenNodeIndx in range (0, len(self.hidden_layer_widths)):
                if hiddenNodeIndx < 1:
                    hidden_layer = [{'weights':[randomValue for i in range(sizeWidthInputs)]} for j in range(self.hidden_layer_widths[hiddenNodeIndx] + 1)]
                    networkWeights.append(hidden_layer)
                    # print(hidden_layer)
                else:
                    hidden_layer = [{'weights':[randomValue for i in range(self.hidden_layer_widths[hiddenNodeIndx-1] +1 )]} for j in range(self.hidden_layer_widths[hiddenNodeIndx] + 1)]
                    networkWeights.append(hidden_layer)


        ##Output Weights ##
        last_hidden_layer_width = len(networkWeights[-1])
        output_layer = [{'weights':[randomValue for i in range(last_hidden_layer_width)]} for j in range(sizeWidthOutputs)]
        networkWeights.append(output_layer)


        # print(len(networkWeights))
        # print(networkWeights)
        return networkWeights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        return 0

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

    ## Gives the validation adn train sets
    def _getValidationAndTrain(self,X):
        return 0

    """Initialize netweork by creating a dict"""
    def _initialize_network(self):
        network = []
        # print(self.weights)

        for layerIndx in range(0, len(self.weights)):
            layer = []
            if layerIndx < len(self.weights) - 1:
                for nodeIndx in range(0, len(self.weights[layerIndx])):
                    if nodeIndx < len(self.weights[layerIndx]) - 1:
                        node = {
                            "net":0,
                            "weights":self.weights[layerIndx][nodeIndx]['weights']
                        }
                        layer.append(node)
                    else:
                        node = {
                            "net":1,
                            "weights":self.weights[layerIndx][nodeIndx]['weights']
                        }
                        layer.append(node)
            else:
                for nodeIndx in range(0, len(self.weights[layerIndx])):
                    node = {
                        "net":0,
                            "weights":self.weights[layerIndx][nodeIndx]['weights']
                    }
                    layer.append(node)

            network.append(layer)
        print(network)
        print(len(network))
        return network


    """Get all the net values from a layer"""

    def _getNetValuesLayer(self,layer):
        nets = []
        for node in layer:
            nets.append(node['net'])
        return np.array(nets)

    """ Get the net value for a node
    X = 2d numpy array
    input = float
    return float
    """

    def _get_net_node(self,Input,w):
        X2 = w * Input
        return np.sum(X2)

    """ Activate the node function
    X = 2d numpy array
    net = float
    return float
    """
    def _activate_node(self,net):
        return (1/(1+exp(-net)))

    """ Derivative of Activation function
    net = float
    return float
    """
    def _activate_node_derivative(self, net):
        return _activate_node(net) * (1-_activate_node(net))

    """ Fordward Propagation of the Network"""
    def _forwardProp(self, inputs, network):
        inputForForward = inputs
        j = 1
        for layer in network:
            i = 1
            for node in layer:
                if i < len(layer) and j < len(network): ## safe guard for the bias node in the layer
                    n = self._get_net_node(node['weight'], inputForForward)
                    node['net'] = n
                    i = i + 1
            # print(layer)
            inputForForward = self._getNetValuesLayer(layer)
            j = j + 1

        return network

    ## General Nodes Equations
    def _change_weight_generalOutput_node(self,c,output,target):
        change_delta = (target - _output_node(net)) * _output_node_derivative(_output_node(net))
        return change_delta

    ## Hidden Nodes Equations ##
    def _change_weight_hidden_node(self,c,net,delta,weight):
        change_delta = c * _output_node(net) * delta * weight * _output_node_derivative(_output_node(net))
        return change_delta
    ## shared Equations ##



    def  _get_Change_hidden_node(self, X, W, isHidden):
        net = _get_net(X,W)
        output = _output_node(net)
        outputDerivative = _output_node_derivative(net)
        # if isHidden:
        #     for

        return 0
    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        pass
