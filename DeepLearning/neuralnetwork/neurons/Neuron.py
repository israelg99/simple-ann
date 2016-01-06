'''
Created on Dec 20, 2015

@author: Israel
'''

from DeepLearning.neuralnetwork.weights.Weights import Weights
from DeepLearning.util.PrintableData import PrintableData
import numpy as np


class Neuron(PrintableData, object):
    
    """ The neuron constructor, uses a randomly normally distributed float for the default bias """
    def __init__(self, bias = None, refNeuronLayer=None):
        if bias is None:
            bias = np.random.rand()
            
        self._bias = bias;

        self._weights = Weights(refNeuronLayer)
                
        self._value = 0;
        
    @property
    def bias(self):
        """I'm the 'bias' property."""
        return self._bias
    
    @property
    def weights(self):
        """I'm the 'bias' property."""
        return self._weights
    
    @property
    def value(self):
        """I'm the 'value' property."""
        return self._value
    
    def setInputNeuron(self, value):
        self._value = value
        
    def getOutputValue(self):
        return self.value
        
    def getWeight(self, index):
        return self.weights.getWeight(index)
    
    def getWeightValue(self, index):
        return self.getWeight(index).weight
    
    def createWeights(self, refNeuronLayer):
        self.weights.createWeights(refNeuronLayer)
    
    def updateWeightsNeuron(self, refNeuronLayer):
        self.weights.updateWeightsNeuron(refNeuronLayer)
        
    def feedforward(self, prevNeuronLayer, index, activation_method):
        
        sumOf = 0;
        for neuron in prevNeuronLayer.neurons:
            sumOf += neuron.value * neuron.getWeightValue(index)
            
        self._value = activation_method(sumOf + self.bias)
    
    def printData(self, precision=4):
        
        super(Neuron, self).printData()
        
        print("Bias: " + str(self.bias))

        self.weights.printData()
            
        print()
        
        