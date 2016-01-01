'''
Created on Dec 11, 2015

@author: Israel
'''

from _collections import defaultdict

from DeepLearning.functions.Functions import Functions
from DeepLearning.neuralnetwork.neurons.NeuronNet import NeuronNet
from DeepLearning.util.PrintableData import PrintableData


class NeuralNetwork(PrintableData, object):
    
    def __init__(self, sizes, activation="sigmoid"):
        
        self.activations = defaultdict(self.setSigmoidActivation, {
            "sigmoid" : self.setSigmoidActivation,
            "tanh"    : self.setTanhActivation
        })
        
        self.activations[activation]()
        
        self._num_layers = len(sizes)
        self._sizes = sizes
        
        self._neuralNetwork = NeuronNet(sizes)
        
    @property
    def num_layers(self):
        """I'm the 'num_layers' property."""
        return self._num_layers
    
    @property
    def sizes(self):
        """I'm the 'sizes' property."""
        return self._sizes
    
    @property
    def neuralNetwork(self):
        """I'm the 'neurons' property."""
        return self._neuralNetwork
    
    def setSigmoidActivation(self):
        self.activation = "sigmoid"
        self.activation_method = Functions.sigmoid
        self.activation_prime = Functions.sigmoid_prime
        
    def setTanhActivation(self):
        self.activation = "tanh"
        self.activation_method = Functions.sigmoid
        self.activation_prime = Functions.sigmoid_prime
    
    def printData(self, precision=4):
        
        super(NeuralNetwork, self).printData()

        print("Neural Network:")
        print("Number of layers: " + str(self.num_layers))
        print("Layer sizes: " + str(self.sizes))

        self.neuralNetwork.printData()

        print()
    
    def feedforward(self, inputs):
        """ 
        `inputs` should be a regular list, for example: [1,0].
        This method returns the output in a list form.
        """
        
        # Getting the input layer, usually the first layer of the neural network.
        inputLayer = self.neuralNetwork.getFirstLayer()
        
        # If the input layer has a different length than the inputs provided, the `feedforward` method is stopped.
        if(len(inputs) != len(inputLayer.neurons)):
            print("Inputs must have the same length as the input layer in the neural network.")
            print("Inputs length: " + str(len(inputs)) + ", Input layer length: " + str(len(inputLayer.neurons)))
            return
        
        # Assigning the inputs to the input layer.
        for index, inputPart in enumerate(inputs):
            inputLayer.neurons[index].setInputNeuron(inputPart)
        
        # After we established the inputs, we feedforward the neural network to get out inputs.
        output = self.neuralNetwork.feedforward(self.activation_method, True)
        
        return output
     
    