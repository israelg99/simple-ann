'''
Created on Dec 25, 2015

@author: Israel
'''

from DeepLearning.neuralnetwork.neurons.NeuronLayer import NeuronLayer
from DeepLearning.util.PrintableData import PrintableData


class NeuronNet(PrintableData, object):
    
    def __init__(self, sizes):
        self.createNet(sizes)
        self.createWeights()
        
    @property
    def neuronLayers(self):
        """I'm the 'neurons' property."""
        return self._neuronLayers
    
    def getNeuronLayer(self, index):
        return self.neuronLayers[index]
    
    def getFirstLayer(self):
        return self.neuronLayers[0]
    
    def getLastLayer(self):
        return self.neuronLayers[-1]
    
    def createNet(self, sizes):
        self._neuronLayers = [NeuronLayer(layerLength)
                        for layerLength in sizes]
        
    def createWeights(self):
        for currentLayer, nextLayer in zip(self.neuronLayers[:-1], self.neuronLayers[1:]):
            currentLayer.createWeights(nextLayer)
    
    def updateWeightsNeuron(self):
        for currentLayer, nextLayer in zip(self.neuronLayers[:-1], self.neuronLayers[1:]):
            currentLayer.updateWeightsNeuron(nextLayer)
            
    def feedforward(self, activation_method, inputOffset=True):
        # Making sure the inputs are already established in the first layer.
        if not inputOffset:
            print("Feed-forward is only applied when inputs are established.")
            return
        
        # We start from layer 1(0-indexed), assuming the inputs are already established according to the check above.
        for index, neuronLayer in enumerate(self.neuronLayers[1:]):
            # The reason we are not using `[index-1]` is because we are iterating an already offset array/list by one.
            neuronLayer.feedforward(self.neuronLayers[index], activation_method)
        
        # Finally gathering the output from the last layer, which should be the output layer.
        output = [neuron.getOutputValue() for neuron in self.getLastLayer().neurons]
        
        return output;
            
    
    def printData(self, precision=4):
        
        super(NeuronNet, self).printData()

        print("Neuron Layers:")
        for i, neuronLayer in enumerate(self.neuronLayers):
            print("Neuron Layer " + str(i+1) + ": ")
            neuronLayer.printData()
            
        print()
            
        print()