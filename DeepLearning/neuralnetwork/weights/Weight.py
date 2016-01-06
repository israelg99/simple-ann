'''
Created on Dec 20, 2015

@author: Israel
'''

from DeepLearning.util.PrintableData import PrintableData
import numpy as np


class Weight(PrintableData, object):
    
    """ The weight constructor, uses a randomly normally distributed float for the default weight """
    def __init__(self, weight = None, refNeuron=None):
        if weight is None:
            weight = np.random.rand()
            
        self._weight = weight
        self.updateWeightNeuron(refNeuron)
        
    @property
    def weight(self):
        """I'm the 'weight' property."""
        return self._weight
    
    @property
    def refNeuron(self):
        """I'm the 'weight' property."""
        return self._refNeuron
    
    def updateWeight(self, inputNeuronValue, delta, learning_rate):
        self._weight += learning_rate * 1 * inputNeuronValue * delta
    
    def updateWeightNeuron(self, refNeuron):
        self._refNeuron = refNeuron
    
    def printData(self, precision=4):
        
        super(Weight, self).printData()

        print(self.weight)
            
        print()