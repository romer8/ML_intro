import numpy as np
import math
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import scipy.stats as ss
### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,lr=.1, momentum=0, shuffle=True,deterministic= 10,hidden_layer_widths=None,weights = None,validationSize = None, allWeightsValue = None, isHotEncoding = True):
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
        self.validation_size = validationSize
        self.numberOfEpochs = 0
        self.allWeightsValue = allWeightsValue
        self.mse_val_epochs = []
        self.mse_training_epochs =[]
        self.accuracy_epochs_val = []
        self.isHotEncoding = isHotEncoding

    def fit(self, X, y, weights=None):
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
        # print(self.debugWeightsStructure())
        # print(self.weights)
        # self.randomizeWeights()
        # print(self.weights)
        #Initialize network ##
        networkObject = self._initialize_network()



        # print("INITIALIZATION NETWORK")
        # print(networkObject)

        lastDeltaWeight = self._getListWeights_from_network(networkObject)
        numberOfEpochWithNoImprovement = [];
        classesY = np.unique(y)
        print("classes" , classesY)
        ##Make the validation and training sets ###
        y_train = np.array([])
        y_train_copy = np.array([])
        y_val = np.array([])
        y_val_copy = np.array([])
        X_train = np.array([])
        X_val = np.array([])
        enc = OneHotEncoder()
        if self.validation_size > 0:
            X_train, X_val, y_train, y_val = self._getValidationAndTrain(X,y,self.validation_size,True)
            y_train_copy = y_train.copy()
            y_val_copy = y_val.copy()
            if self.isHotEncoding:
                enc.fit(y_train)
                y_train = enc.transform(y_train).toarray()
                enc.fit(y_val)
                y_val = enc.transform(y_val).toarray()
        else:
            X_train = X
            y_train = y
            y_train_copy = y_train.copy()
            y_val_copy = y_val.copy()
            # print(y_train)
            if self.isHotEncoding:
                enc.fit(y_train)
                y_train = enc.transform(y_train).toarray()
                print(y_train)


        biasArray = np.full((X_train.shape[0],1),1)
        # biasArray = np.full((X.shape[0],1),1)
        bestMSEsoFar = 1000.0
        bestScoresoFar = 0
        bestWeight = []
        # print(X_train)
        ## DO ONE HOT ENCODING##


        while len(numberOfEpochWithNoImprovement) < self.deterministic:
            self.numberOfEpochs = self.numberOfEpochs + 1
            print("***************************************EPOCH********************************************************************************************************")
            print("EPOCH NUMBER ",self.numberOfEpochs)
            X_shuffled,y_shuffled = self._shuffle_data(X_train,y_train)
            # X_shuffled,y_shuffled = self._shuffle_data(X,y)
            X_bias = np.concatenate((X_shuffled,biasArray),axis=1)
            mse_instances = np.array([])
            numberIns = 0
            weights_change_array = []
            for x_unit,label_unit in zip(X_bias, y_shuffled):
                numberIns = numberIns + 1

                # print("-------------------------INSTANCE ", numberIns ,"------------------------------------------------------------------------------------")
                ## MAKE Y HOT PLATE ##
                # target = self._making_y_hot(classesY,label_unit)
                # print(label_unit ,target)
                target = label_unit

                self._forwardProp(x_unit,networkObject)
                weights_change_array,error_array = self._backwardProp(networkObject,target,x_unit)
                last_instance_changeWeight_momentum = self._get_Delta_weights_mometum(lastDeltaWeight)
                finalDeltaWeights = self._get_Final_weights_change(weights_change_array,last_instance_changeWeight_momentum)
                lastDeltaWeight = finalDeltaWeights
                self._updtate_weights(networkObject, finalDeltaWeights)


            ##UPDATE THE WEIGHTS TO THE LAST MLP WEIGHTS ##
            # lastDeltaWeight = weights_change_array
            self.weights = self._getWeights_from_network(networkObject)
            # print(self.weights)
            ### CALCULATING THE MEAN MSE FOR THE VALIDATION SET
            if self.validation_size > 0:
                mean_mse = self._get_mse_valSet(X_val,y_val)
                mean_mse_training = self._get_mse_valSet(X_train,y_train)
                self.mse_val_epochs.append(mean_mse)
                self.mse_training_epochs.append(mean_mse_training)
                score_epoch = self.score(X_val,y_val_copy)
                print("score ", score_epoch)
                # print("BEST Score SO FAR = ", bestScoresoFar)

                self.accuracy_epochs_val.append(score_epoch)
                print("BEST MSE SO FAR = ", bestMSEsoFar)
                print("MEAN MSE = ", mean_mse)
                if mean_mse < 0.0001:
                    break
                # self._increaseNumberOfEpochWithNoImprovement(bestMSEsoFar,mean_mse,numberOfEpochWithNoImprovement,bestWeight)
                # print(numberOfEpochWithNoImprovement)
                # print("bssf ",bestMSEsoFar)
                # print("actual mse ", mean_mse)
                print(numberOfEpochWithNoImprovement)

                if abs(bestMSEsoFar - mean_mse) < bestMSEsoFar * 0.00001:
                # if abs(bestScoresoFar - score_epoch) < 0.0001:
                    numberOfEpochWithNoImprovement.append(bestMSEsoFar - mean_mse)

                    # numberOfEpochWithNoImprovement.append(bestScoresoFar - score_epoch)
                    if numberOfEpochWithNoImprovement == 10:
                        self.weights = bestWeight
                else:
                    if bestMSEsoFar > mean_mse:
                    # if bestScoresoFar < score_epoch:
                        bestMSEsoFar = mean_mse
                        # bestScoresoFar = score_epoch
                        bestWeight = self.weights
                        numberOfEpochWithNoImprovement.clear()
                    else:
                        numberOfEpochWithNoImprovement.append(-1)
                        # numberOfEpochWithNoImprovement.append(bestScoresoFar - score_epoch)

                        if numberOfEpochWithNoImprovement == 10:
                            self.weights = bestWeight
            else:
                numberOfEpochWithNoImprovement.append(1)

            # break
        return self

    """ Function to retrieve the mean instance MSE """

    def _get_mse_valSet(self,Xval,yVal):
        mse_instances = np.array([])

        networkObject = self._initialize_network()
        biasArray = np.full((Xval.shape[0],1),1)
        X_bias = np.concatenate((Xval,biasArray),axis=1)

        for x_unit,label_unit in zip(X_bias, yVal):
            target = label_unit
            self._forwardProp(x_unit,networkObject)
            outputs_outputLayer = self._getOutputValuesLayer(networkObject[-1], True)
            mse = self._getMSE(outputs_outputLayer,target)
            mse_instances = np.append(mse_instances,mse)

        ### CALCULATING THE MEAN MSE FOR THE EPOCH
        # print("MSE INSTANCES ", mse_instances)
        mean_mse = np.mean(mse_instances)

        return mean_mse

    """ Function to retrieve the weights from the network object"""
    def _getWeights_from_network(self, network):
        weights_network = []
        for layer in network:
            weights_layer = []
            for node in layer:
                w = {"weights": node['weights']}
                weights_layer.append(w)
            weights_network.append(weights_layer)

        return weights_network
    """ Function to retrieve the weights from the network object"""
    def _getListWeights_from_network(self, network):
        weights_network = []
        for layer in network:
            weights_layer = []
            for node in layer:
                w = node['weights']
                weights_layer.append(w)
            weights_network.append(weights_layer)

        return weights_network

    def _getMSE(self, outputs, targets):
        errors = np.subtract(targets,outputs)
        errors_squared = np.square(errors)
        mse = np.sum(errors_squared)
        return mse

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        outputs = np.array([])
        networkObject = self._initialize_network()
        biasArray = np.full((X.shape[0],1),1)
        X_bias = np.concatenate((X,biasArray),axis=1)
        m_size = 0
        for x_unit in X_bias:
            self._forwardProp(x_unit,networkObject)
            outputs_outputLayer = self._getOutputValuesLayer(networkObject[-1], True)
            m_size = outputs_outputLayer.shape[0]
            outputs = np.append(outputs, outputs_outputLayer)

        outputs = outputs.reshape(-1,m_size)
        outputs2 = (outputs == outputs.max(axis=1)[:,None]).astype(int)
        # for x in range(0,3):
        #     print(outputs[x], outputs2[x])
        # print("nuebo")
        # for x in range(3,-1,-1):
        #     print(outputs[x], outputs2[x])


        return outputs2

    def debugWeightsStructure(self):
        quickWeights = []
        netA = {
            'layers': len(self.weights)
        }
        quickWeights.append(netA)
        for layer in self.weights:
            layerA = {'nodos': len(layer)}
            NodesLength = []
            for node in layer:
                NodesLength.append(len(node['weights']))
            quickWeights.append(layerA)
            quickWeights.append(NodesLength)

        return quickWeights

    def initialize_weights(self,X,y):
        """ Initialize weights for perceptron. Don't forget the bias!
            Use one layer of hidden nodes with the number of hidden nodes being twice the number of inputs.

        Returns:

        """

        networkWeights = []
        sizeWidthInputs = len(X[0]) + 1
        sizeWidthOutputs = 1
        if self.isHotEncoding:
            sizeWidthOutputs = len(np.unique(y))
        # sizeWidthHiddenLayersDefault = len(X[0]) * 2 + 1
        sizeWidthHiddenLayersDefault = len(X[0]) * 2
        randomValue = self.allWeightsValue
        if self.allWeightsValue is None:
            randomValue = np.random.normal()
            # randomValue = 0

        if self.hidden_layer_widths is None:
            # hidden_layer = [{'weights':[randomValue for i in range(sizeWidthInputs)]} for j in range(sizeWidthHiddenLayersDefault)]
            # hidden_layer = [{'weights':[np.random.normal() for i in range(sizeWidthInputs)]} for j in range(sizeWidthHiddenLayersDefault)]
            hidden_layer = [{'weights':list(np.random.normal(size=(sizeWidthInputs))*0.01)} for j in range(sizeWidthHiddenLayersDefault)]
            # hidden_layer = [{'weights':list(np.random.randn(1,sizeWidthInputs)*0.01)} for j in range(sizeWidthHiddenLayersDefault)]
            networkWeights.append(hidden_layer)

        else:
            sizeWidthHiddenLayersCustomized = self.hidden_layer_widths[0]
            for hiddenNodeIndx in range (0, len(self.hidden_layer_widths)):
                if hiddenNodeIndx < 1:
                    # hidden_layer = [{'weights':[randomValue for i in range(sizeWidthInputs)]} for j in range(self.hidden_layer_widths[hiddenNodeIndx] + 1)]
                    # hidden_layer = [{'weights':[np.random.normal() for i in range(sizeWidthInputs)]} for j in range(self.hidden_layer_widths[hiddenNodeIndx] + 1)]
                    hidden_layer = [{'weights':list(np.random.normal(size=(sizeWidthInputs))*0.01)} for j in range(self.hidden_layer_widths[hiddenNodeIndx])]
                    # hidden_layer = [{'weights':list(np.random.randn(1,sizeWidthInputs)*0.01)} for j in range(self.hidden_layer_widths[hiddenNodeIndx])]

                    networkWeights.append(hidden_layer)
                    # print(hidden_layer)
                else:
                    # hidden_layer = [{'weights':[randomValue for i in range(self.hidden_layer_widths[hiddenNodeIndx-1] +1 )]} for j in range(self.hidden_layer_widths[hiddenNodeIndx] + 1)]
                    # hidden_layer = [{'weights':[np.random.normal() for i in range(self.hidden_layer_widths[hiddenNodeIndx-1] +1 )]} for j in range(self.hidden_layer_widths[hiddenNodeIndx] + 1)]
                    # hidden_layer = [{'weights':[randomValue for i in range(self.hidden_layer_widths[hiddenNodeIndx-1] + 1 )]} for j in range(self.hidden_layer_widths[hiddenNodeIndx])]
                    hidden_layer = [{'weights':list(np.random.normal(size=(self.hidden_layer_widths[hiddenNodeIndx-1] + 1))*0.01)} for j in range(self.hidden_layer_widths[hiddenNodeIndx])]
                    # hidden_layer = [{'weights':list(np.random.randn(1,self.hidden_layer_widths[hiddenNodeIndx-1] + 1)*0.01)} for j in range(self.hidden_layer_widths[hiddenNodeIndx])]

                    networkWeights.append(hidden_layer)


        ##Output Weights ##
        # last_hidden_layer_width = len(networkWeights[-1])
        last_hidden_layer_width = len(networkWeights[-1]) + 1
        # output_layer = [{'weights':[randomValue for i in range(last_hidden_layer_width)]} for j in range(sizeWidthOutputs)]
        # output_layer = [{'weights':[np.random.normal() for i in range(last_hidden_layer_width)]} for j in range(sizeWidthOutputs)]
        output_layer = [{'weights':list(np.random.normal(size=(last_hidden_layer_width))*0.01)} for j in range(sizeWidthOutputs)]
        # output_layer = [{'weights':list(np.random.randn(1,last_hidden_layer_width))} for j in range(sizeWidthOutputs)]

        networkWeights.append(output_layer)


        return networkWeights
    def randomizeWeights(self):
        total = 0
        for indxLayer in range (0,len(self.weights)):
            for indxnode in range(0,len(self.weights[indxLayer])):
                for w in self.weights[indxLayer][indxnode]['weights']:
                    total = total + 1

        a = np.random.normal(size=(total))
        indxRandomNormal = 0
        for indxLayer in range (0,len(self.weights)):
            for indxnode in range(0,len(self.weights[indxLayer])):
                newW = []
                for w in self.weights[indxLayer][indxnode]['weights']:
                    newW.append(a[indxRandomNormal])
                    indxRandomNormal = indxRandomNormal + 1

                self.weights[indxLayer][indxnode]['weights'] = newW

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        outputs = self.predict(X)
        # print("OUTPUTS PREDICTED")
        # print(outputs)
        # print("outputs ",outputs)
        # print("y ", y)
        enc2 = OneHotEncoder()
        enc2.fit(y)
        y_hot_encoding = enc2.transform(y).toarray()
        # for x in range(0,10):
        #     print(y[x], y_hot_encoding[x])
        # print("adsadas")
        # for x in range(3,-1,-1):
        #     print(y[x], y_hot_encoding[x])

        # print("this is  y hot encoding")
        # print(y_hot_encoding)
        # y_hot_encoding = np.array([])
        matches = []

        for y_hot_unit, output in zip(y_hot_encoding, outputs):
            # print(y_hot_unit.astype(int))
            # print(output)
            isTheSame = np.array_equal(y_hot_unit, output)
            # print("match? ", isTheSame)
            if isTheSame == True:
                matches.append(isTheSame)

        # print("Total Matches ",len(matches))
        # print("Total Rows ", len(y_hot_encoding) )
        accuracy_porcetange = len(matches)/len(y_hot_encoding)

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

    """ Split data into validation and trainingSets"""
    def _getValidationAndTrain(self,X,y,valSize,isShuffle):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=valSize, shuffle = isShuffle) # 0.25 x 0.8 = 0.2
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
        weights_ = []
        for node in layer:
            weights_.append(node['weights'][indx])

        return np.array(weights_)

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
        return np.dot(w,Input)

    """ Activate the node function
    X = 2d numpy array
    net = float
    return float
    """
    def _activate_node(self,net):
        return 1/(1 + math.exp(-net))

    """ Derivative of Activation function
    net = float
    return float
    """
    def _activate_node_derivative(self, net):
        return self._activate_node(net) * (1 - self._activate_node(net))

    """ Fordward Propagation of the Network"""
    def _forwardProp(self, inputs, network):
        inputForForward = inputs
        for indxLayer in range(0, len(network)):
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
                    # print("target value", target[indxNode])
                    targ = target[0]

                    if self.isHotEncoding:
                        targ = target[indxNode]
                    # desv = (target[indxNode] - o) * o_dev
                    desv = (targ - o) * o_dev
                    # print("desv value ", desv)
                    desv_layer.append(desv)
                    node_weight_change = []
                    outputsNextLayer = self._getOutputValuesLayer(network[indxLayer - 1], False)
                    # print("these are the outputs")
                    # print(outputsNextLayer)
                    for indxOutputs in outputsNextLayer:
                        changeW = self.lr * desv * indxOutputs
                        node_weight_change.append(changeW)

                    layer_weight_change.append(node_weight_change)

                weight_change_network.append(layer_weight_change)
                # print("LAYER WEIGHT CHANGE")
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
                    # print("this is the output", o)
                    o_dev = self._activate_node_derivative(network[indxLayer][indxNode]['net'])
                    node_weight_change = []
                    weightsToUse = self._getWeightsLayer(network[indxLayer + 1],indxNode)
                    # print("this are the weights ", weightsToUse)

                    weight_desv = np.dot(np.array(desv_network[-1]), weightsToUse)
                    # print("this is the dot product ", weight_desv)
                    desv = weight_desv * o_dev
                    # print("desv value ", desv)

                    desv_layer_temp.append(desv)
                    for indxOutputs in outputsNextLayer:
                        changeW = self.lr * desv * indxOutputs
                        node_weight_change.append(changeW)

                    layer_weight_change.append(node_weight_change)
                # print("this is the layer weight change ")
                # print(layer_weight_change)
                weight_change_network.append(layer_weight_change)
                desv_layer = desv_layer_temp
                desv_network.append(desv_layer)

        ## reversed the list
        weight_change_network.reverse()
        desv_network.reverse()
        return weight_change_network, desv_network

    # def _updtate_weights(self,network, Deltaweights,lastDeltaweights):
    #
    #     for indxLayer in range(0, len(network)):
    #         for indxNode in range(0, len(network[indxLayer])):
    #             weights_original = np.array(network[indxLayer][indxNode]['weights'])
    #             weights_change = np.array(Deltaweights[indxLayer][indxNode])
    #             if len(lastDeltaweights) > 0:
    #                 last_weights_change = np.array(lastDeltaweights[indxLayer][indxNode]) * self.momentum
    #                 # print("*MOMENTUM", self.momentum,last_weights_change,weights_change)
    #                 weights_change_momentum = np.add(weights_change, last_weights_change)
    #                 print("weight change  MOMENTUM")
    #                 print(weights_change_momentum)
    #                 weights_updated = np.add(weights_original, weights_change_momentum)
    #                 # print(weights_updated)
    #                 network[indxLayer][indxNode]['weights'] = list(weights_updated)
    #                 # print(network[indxLayer][indxNode]['weights'])
    #
    #             else:
    #                 weights_updated = np.add(weights_original, weights_change)
    #                 network[indxLayer][indxNode]['weights'] = list(weights_updated)
    #
    #     return network
    def _updtate_weights(self,network, Deltaweights):

        for indxLayer in range(0, len(network)):
            for indxNode in range(0, len(network[indxLayer])):
                weights_original = np.array(network[indxLayer][indxNode]['weights'])
                weights_change = np.array(Deltaweights[indxLayer][indxNode])
                weights_updated = np.add(weights_original, weights_change)
                network[indxLayer][indxNode]['weights'] = list(weights_updated)
        return network

    def _get_Delta_weights_mometum(self,lastDeltaweights):

        for indxLayer in range(0, len(lastDeltaweights)):
            for indxNode in range(0, len(lastDeltaweights[indxLayer])):
                lastDeltaweights[indxLayer][indxNode] = np.array(lastDeltaweights[indxLayer][indxNode]) * self.momentum
        return lastDeltaweights

    def _get_Final_weights_change(self,actualDeltaWeights, lastDeltaweights):
        actualDeltaWeights2 = actualDeltaWeights.copy()
        for indxLayer in range(0, len(actualDeltaWeights)):
            for indxNode in range(0, len(actualDeltaWeights[indxLayer])):
                weights_change_actual = np.array(actualDeltaWeights[indxLayer][indxNode])
                weights_change_last = np.array(lastDeltaweights[indxLayer][indxNode])
                final_weights_change = np.add(weights_change_actual, weights_change_last)
                actualDeltaWeights2[indxLayer][indxNode] = final_weights_change
        return actualDeltaWeights2


    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights

    def get_numberEpochs(self):
        return self.numberOfEpochs

    def get_mse_epochs(self):
        return [self.mse_val_epochs, self.mse_training_epochs]

    def get_accuracy_epochs(self):
        return self.accuracy_epochs_val
