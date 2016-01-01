'''
Created on Dec 25, 2015

@author: Israel
'''

from DeepLearning.neuralnetwork.weights.Weight import Weight
from DeepLearning.util.PrintableData import PrintableData
import numpy as np


class Weights(PrintableData, object):

    def __init__(self, refNeuronLayer=None):
        self.createWeights(refNeuronLayer)
        
    @property
    def weights(self):
        """I'm the 'bias' property."""
        return self._weights
    
    def getWeight(self, index):
        return self.weights[index]
    
    def createWeights(self, refNeuronLayer=None):
        self._weights = None if refNeuronLayer is None else np.array([Weight(refNeuron=refNeuron) for refNeuron in refNeuronLayer.neurons])
    
    def updateWeightsNeuron(self, refNeuronLayer):
        # 'refNeuronLayer' is expected to have the same length as the weights numpy array.
        for i, weight in enumerate(self.weights):
            weight.updateWeightNeuron(refNeuronLayer.neurons[i])
            
    def printData(self, precision=4):
        
        super(Weights, self).printData()
        if self.weights is not None:
            print("Weights:")
            for i, weight in enumerate(self.weights):
                print("Weight " + str(i+1) + ": " + str(weight.weight))
        else:
            print("Doesn't have weights, it's probably the last layer.")
        
        print()