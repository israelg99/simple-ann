'''
Created on Dec 25, 2015

@author: Israel
'''

from DeepLearning.neuralnetwork.neurons.Neuron import Neuron
from DeepLearning.util.PrintableData import PrintableData
import numpy as np


class NeuronLayer(PrintableData, object):

    def __init__(self, layerLength, nextLayerNeurons=None):
        self._neurons = np.array([Neuron(refNeuronLayer=nextLayerNeurons) for _ in range(layerLength)])
        
    @property
    def neurons(self):
        """I'm the 'neurons' property."""
        return self._neurons
    
    def createWeights(self, refNeuronLayer):
        for neuron in self.neurons:
            neuron.createWeights(refNeuronLayer)
    
    def updateWeightsNeuron(self, refNeuronLayer):
        for neuron in self.neurons:
            neuron.updateWeightsNeuron(refNeuronLayer)
    
    def feedforward(self, previousNeuronLayer, activation_method):     
        # Thanks to this great Object-Oriented design we are just iterating the neruons and calling feedforward on them.
        for index, neuron in enumerate(self.neurons):
            # We need to pass the index to the neuron because he will need it in order to know which weights to associate with himself.
            neuron.feedforward(previousNeuronLayer, index, activation_method)
    
    def printData(self, precision=4):
        
        super(NeuronLayer, self).printData()

        print("Neurons:")
        for i, neuron in enumerate(self.neurons):
            print("Neuron " + str(i+1) + ": ");
            neuron.printData()
            
        print()