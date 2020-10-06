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

    def __init__(self,lr=.1, momentum=0, shuffle=True,deterministic= 10,hidden_layer_widths=None, weights = None):
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
        self.weights = weights
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
        self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights
        # biasArray = np.full((X.shape[0],1),1)
        # X_bias = np.concatenate((X_shuffled,biasArray),axis=1)



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
        weightsTemp = []
        checkMean = []

        """ Hidden layer output initialization"""
        if self.hidden_layer_widths is not None:
            for hiddenLayer in self.hidden_layer_widths:
                layerWeight = []
                for node in range(0, hiddenLayer):
                    nodeWeight = random.uniform(0 , 0.001)
                    checkMean.append(nodeWeight)
                    layerWeight.append(nodeWeight)

                layerWeight.append(1)
                weightsTemp.append(layerWeight)
        else:
            layerWeight = []
            for node in range(0, len(X[0]) * 2):
                nodeWeight = random.uniform(0 , 0.01)
                checkMean.append(nodeWeight)
                layerWeight.append(nodeWeight)

            layerWeight.append(1)
            weightsTemp.append(layerWeight)

        """Output layer weigths added"""
        layerWeight = []
        for nodeOutput in range(0, len(y[0])):
            nodeWeight = random.uniform(0 , 0.01)
            checkMean.append(nodeWeight)
            layerWeight.append(nodeWeight)

        layerWeight.append(1)
        weightsTemp.append(layerWeight)


        self.weights = weightsTemp
        print(sum(checkMean)/len(checkMean))
        print(weightsTemp)
        return

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
    def _inittialize_network(self, X, y):
        network = []
        initialize_weights(X,y)
        for layerWeight in self.initial_weights:
            layer = []
            for weight in layerWeight:
                node = {
                    'net':0,
                    'weight':weight
                }
                layer.append(node)

            network.append(layer)

        return network


    """Get all the net values from a layer"""

    def _getNetValuesLayer(layer):
        nets = []
        for node in layer:
            nets.append(node.net)
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
        for layer in network:
            for node in layer:
                n = _get_net_node(node.weight, inputForForward)
                node['net'] = n
            inputForForward = _getNetValuesLayer(layer)

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
