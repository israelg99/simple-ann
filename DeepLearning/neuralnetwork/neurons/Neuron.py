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
        self._errorFactor = 0;
        self._deltaValue = 0;
        
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
    
    @property
    def errorFactor(self):
        """I'm the 'errorFactor' property."""
        return self._errorFactor
    
    @property
    def delta(self):
        """I'm the 'delta' property."""
        return self._deltaValue
    
    def calculateOutputErrorFactor(self, desiredValue):
        self._errorFactor = desiredValue - self.value
        
    def calculateErrorFactor(self, nextLayerNeurons):
        # Looping the output layer to calculate the error factor and the delta value of the desired outputs and our actual outputs.
        errorFactor = 0;
        for index, neuron in enumerate(nextLayerNeurons.neurons):
            # Calculating the error factor.
            errorFactor += neuron.delta * self.getWeight(index).weight
        
        # Setting the error factor.
        self._errorFactor = errorFactor
        
    def calculateDeltaValue(self, errorFactor=None):
        if errorFactor is None:
            errorFactor = self.errorFactor
            
        self._deltaValue = self.value * (1 - self.value) * self.errorFactor
        
    def updateBias(self, learning_rate):
        self._bias = self.bias + learning_rate * 1 * self.delta
        
    def updateWeights(self, previousNeuronLayer, index, learning_rate):
        for neuron in previousNeuronLayer.neurons:
            neuron.weights.updateWeight(index, neuron.value, self.delta, learning_rate)
            
    def updateData(self, previousNeuronLayer, index, learning_rate):
        self.updateBias(learning_rate)
        self.updateWeights(previousNeuronLayer, index, learning_rate)
    
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
        # Getting the `net value`, which is the sum of all the associated input weights * by the input neurons value associated with those weights.
        sumOf = 0;
        for neuron in prevNeuronLayer.neurons:
            sumOf += neuron.value * neuron.getWeightValue(index)
        
        # Using the activation method provided on the net value we calculated above with the bias added, we got our neuron value!
        self._value = activation_method(sumOf + self.bias)
    
    def backPropagation(self, nextLayerNeurons, learning_rate):
        # Calculating the error factor.
        self.calculateErrorFactor(nextLayerNeurons)
                
        # Calculating the delta value.
        self.calculateDeltaValue()
    
    def printData(self, precision=4):
        
        super(Neuron, self).printData()
        
        print("Bias: " + str(self.bias))

        self.weights.printData()
            
        print()
        
        