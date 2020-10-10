import numpy as np
import math
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
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

    """makes a numpy hot plate array for the labels"""

    def _making_y_hot(self,y, singleY):
        y_hot = np.unique(y)
        for indx in range(0, len(y_hot)):
            if y_hot[indx] != singleY[0]:
                y_hot[indx] = 0
            else:
                y_hot[indx] = 1
        return y_hot

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
        ## INITIALIZATIONS

        if self.weights is None:
            self.weights = self.initialize_weights(X,y)
        print("INITIAL WEIGHTS")
        print(self.weights)
        #Initialize network ##
        networkObject = self._initialize_network()
        print("INITIALIZATION NETWORK")
        print(networkObject)

        lastDeltaWeight = []
        numberOfEpochWithNoImprovement = [];
        ##Make the validation and training sets ###
        X_train, X_val, y_train, y_val = self._getValidationAndTrain(X,y,0.20,True)
        biasArray = np.full((X_train.shape[0],1),1)
        # biasArray = np.full((X.shape[0],1),1)

        while len(numberOfEpochWithNoImprovement) < self.deterministic:

            X_shuffled,y_shuffled = self._shuffle_data(X_train,y_train)
            # X_shuffled,y_shuffled = self._shuffle_data(X,y)
            X_bias = np.concatenate((X_shuffled,biasArray),axis=1)
            for x_unit,label_unit in zip(X_bias, y_shuffled):
                ## MAKE Y HOT PLATE ##
                target = self._making_y_hot(y_shuffled,label_unit)

                ##FORWARD PROPAGATION ##
                self._forwardProp(x_unit,networkObject)
                # print("FORWARD PROPAGATION")
                # print(networkObject)

                #BACKWARD PROPAGATION WITH MOMENTUM ##
                weights_change_array = self._backwardProp(networkObject,target,x_unit)
                # print("BACKWARD PROPAGATION")
                # print(weights_change_array)
                self._updtate_weights(networkObject, weights_change_array,lastDeltaWeight)
                lastDeltaWeight = weights_change_array
                # print("NEW WEIGHTS")
                # print(networkObject)
            break
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
        # sizeWidthHiddenLayersDefault = len(X[0]) * 2 + 1
        sizeWidthHiddenLayersDefault = len(X[0]) * 2
        randomValue = round(1,5) # random.uniform(0 , 0.00001)
        if self.hidden_layer_widths is None:
            hidden_layer = [{'weights':[randomValue for i in range(sizeWidthInputs)]} for j in range(sizeWidthHiddenLayersDefault)]
            networkWeights.append(hidden_layer)

        else:
            sizeWidthHiddenLayersCustomized = self.hidden_layer_widths[0]
            for hiddenNodeIndx in range (0, len(self.hidden_layer_widths)):
                if hiddenNodeIndx < 1:
                    # hidden_layer = [{'weights':[randomValue for i in range(sizeWidthInputs)]} for j in range(self.hidden_layer_widths[hiddenNodeIndx] + 1)]
                    hidden_layer = [{'weights':[randomValue for i in range(sizeWidthInputs)]} for j in range(self.hidden_layer_widths[hiddenNodeIndx])]
                    networkWeights.append(hidden_layer)
                    # print(hidden_layer)
                else:
                    # hidden_layer = [{'weights':[randomValue for i in range(self.hidden_layer_widths[hiddenNodeIndx-1] +1 )]} for j in range(self.hidden_layer_widths[hiddenNodeIndx] + 1)]
                    hidden_layer = [{'weights':[randomValue for i in range(self.hidden_layer_widths[hiddenNodeIndx-1] + 1 )]} for j in range(self.hidden_layer_widths[hiddenNodeIndx])]
                    networkWeights.append(hidden_layer)


        ##Output Weights ##
        # last_hidden_layer_width = len(networkWeights[-1])
        last_hidden_layer_width = len(networkWeights[-1]) + 1
        output_layer = [{'weights':[randomValue for i in range(last_hidden_layer_width)]} for j in range(sizeWidthOutputs)]
        networkWeights.append(output_layer)


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

    """ Split data into validation and trainingSets"""
    def _getValidationAndTrain(self,X,y,valSize,isShuffle):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, shuffle = isShuffle) # 0.25 x 0.8 = 0.2
        return  X_train, X_val, y_train, y_val

    """ Split data into training and validation"""
    def _getTestTraining(self,X,y,testSize,isShuffle):
        ##put 0 to testSize for all the training

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, shuffle=isShuffle)

        return X_train, X_test, y_train, y_test

    """Initialize netweork by creating a dict"""
    def _initialize_network(self):
        network = []
        # print(self.weights)

        for layerIndx in range(0, len(self.weights)):
            layer = []
            # if layerIndx < len(self.weights) - 1:
            #     for nodeIndx in range(0, len(self.weights[layerIndx])):
            #         if nodeIndx < len(self.weights[layerIndx]) - 1:
            #             node = {
            #                 "net":0,
            #                 "weights":self.weights[layerIndx][nodeIndx]['weights']
            #             }
            #             layer.append(node)
            #         else:
            #             node = {
            #                 "net":1,
            #                 "weights":self.weights[layerIndx][nodeIndx]['weights']
            #             }
            #             layer.append(node)
            # else:
            for nodeIndx in range(0, len(self.weights[layerIndx])):
                node = {
                    "net":0,
                    "weights":self.weights[layerIndx][nodeIndx]['weights']
                }
                layer.append(node)

            network.append(layer)
        # print(len(network))
        return network
    """Get all the weights for a layer"""
    def _getWeightsLayer(self,layer,indx):
        weights = []
        for node in layer:
            weights.append(node['weights'][indx])

        return np.array(weights)

    """Get all the net values from a layer"""

    def _getNetValuesLayer(self,layer):
        nets = []
        for node in layer:
            nets.append(node['net'])
        return np.array(nets)

    """Get all the output values from a layer"""
    def _getOutputValuesLayer(self,layer, isOutput):
        outputs = []
        for node in layer:
            outputs.append(self._activate_node(node['net']))
        if isOutput == False:
            outputs.append(1)
            # outputs[-1] = 1
        return np.array(outputs)

    """ Get the net value for a node
    X = 2d numpy array
    input = float
    return float
    """
    def _get_net_node(self,Input,w):
        # print(np.dot(w,Input))
        return round(np.dot(w,Input),5)

    """ Activate the node function
    X = 2d numpy array
    net = float
    return float
    """
    def _activate_node(self,net):
        return round(1/(1 + math.exp(-net)),5)

    """ Derivative of Activation function
    net = float
    return float
    """
    def _activate_node_derivative(self, net):
        return round(self._activate_node(net) * (1 - self._activate_node(net)),5)

    """ Fordward Propagation of the Network"""
    def _forwardProp(self, inputs, network):
        inputForForward = inputs
        for indxLayer in range(0, len(network)):
            # if indxLayer < len(network) - 1:
            #     for indxNode in range(0, len(network[indxLayer])):
            #         if indxNode < len(network[indxLayer]) - 1:
            #             weight_np = np.array(network[indxLayer][indxNode]['weights'])
            #             # print(weight_np)
            #             # print(inputForForward)
            #             n = self._get_net_node(inputForForward, weight_np)
            #             network[indxLayer][indxNode]['net'] = n
            # else:
            for indxNode in range(0, len(network[indxLayer])):
                weight_np = np.array(network[indxLayer][indxNode]['weights'])
                # print(weight_np)
                # print(inputForForward)
                n = self._get_net_node(inputForForward, weight_np)
                network[indxLayer][indxNode]['net'] = n

            inputForForward = self._getOutputValuesLayer(network[indxLayer],False)

    """ Backward Propagation of the Network"""
    def _backwardProp(self, network, target, input):
        desv_network = []
        weight_change_network = []
        # print ("length Network ",len(network) )
        for indxLayer in range(len(network)-1 , -1, -1):
            desv_layer = []
            # print("Layer # ",indxLayer)
            layer_weight_change = []

            if indxLayer == len(network) -1:
                # print("OUTPUT LAYER")

                for indxNode in range(0, len(network[indxLayer])):
                    # print("Node # ",indxNode)

                    o = self._activate_node(network[indxLayer][indxNode]['net'])
                    # print("output ", o)
                    o_dev = self._activate_node_derivative(network[indxLayer][indxNode]['net'])
                    desv = round((target[indxNode] - o) * o_dev,5)
                    # print("desv value ", desv)
                    desv_layer.append(desv)
                    node_weight_change = []
                    outputsNextLayer = self._getOutputValuesLayer(network[indxLayer - 1], False)
                    for indxOutputs in outputsNextLayer:
                        changeW = round(self.lr * desv * indxOutputs,5)
                        node_weight_change.append(changeW)

                    layer_weight_change.append(node_weight_change)

                weight_change_network.append(layer_weight_change)
                # print("first layer weigh change")
                # print(layer_weight_change)
                desv_network.append(desv_layer)
            else:
                # print("HIDDEN LAYER")
                o = 1
                o_dev = 1
                desv_layer_temp = []
                layer_weight_change = []
                # print("this is the desv layer")
                # print(desv_network[-1])

                outputsNextLayer = np.array([])
                if indxLayer > 0:
                    outputsNextLayer = self._getOutputValuesLayer(network[indxLayer - 1], False)
                else:
                    outputsNextLayer = input
                # print("these are the outputs")
                # print(outputsNextLayer)
                for indxNode in range(0, len(network[indxLayer])):
                    # print("Node # ",indxNode)
                    o = self._activate_node(network[indxLayer][indxNode]['net'])
                    o_dev = self._activate_node_derivative(network[indxLayer][indxNode]['net'])
                    node_weight_change = []
                    weightsToUse = self._getWeightsLayer(network[indxLayer + 1],indxNode)
                    # print("this are the weights ", weightsToUse)

                    weight_desv = np.dot(np.array(desv_network[-1]), weightsToUse)
                    # print("this is the dot product ", weight_desv)
                    desv = round(weight_desv * o_dev,5)
                    # print("desv value ", desv)

                    desv_layer_temp.append(desv)
                    for indxOutputs in outputsNextLayer:
                        changeW = round(self.lr * desv * indxOutputs,5)
                        node_weight_change.append(changeW)

                    layer_weight_change.append(node_weight_change)
                # print("this is the layer weight change ")
                # print(layer_weight_change)
                weight_change_network.append(layer_weight_change)
                desv_layer = desv_layer_temp
                desv_network.append(desv_layer)

        ## reversed the list
        weight_change_network.reverse()
        return weight_change_network

    def _updtate_weights(self,network, Deltaweights,lastDeltaweights):

        for indxLayer in range(0, len(network)):
            for indxNode in range(0, len(network[indxLayer])):
                weights_original = np.array(network[indxLayer][indxNode]['weights'])
                weights_change = np.array(Deltaweights[indxLayer][indxNode])
                if len(lastDeltaweights) > 0:
                    print("fasfsaf")
                    last_weights_change = np.array(lastDeltaweights[indxLayer][indxNode]) * self.momentum
                    weights_change_momentum = np.add(weights_change, last_weights_change)
                    weights_updated = np.add(weights_original, weights_change_momentum)
                    network[indxLayer][indxNode]['weights'] = list(weights_updated)
                else:
                    weights_updated = np.add(weights_original, weights_change)
                    network[indxLayer][indxNode]['weights'] = list(weights_updated)

        return network

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        pass
