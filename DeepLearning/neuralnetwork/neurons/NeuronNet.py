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
            print("Feed-forward is only applied when inputs are already established.")
            return
        
        # We start from layer 1(0-indexed), assuming the inputs are already established according to the check above.
        for index, neuronLayer in enumerate(self.neuronLayers[1:]):
            # The reason we are not using `[index-1]` is because we are iterating an already offset array/list by one.
            neuronLayer.feedforward(self.neuronLayers[index], activation_method)
        
        # Finally gathering the output from the last layer, which should be the output layer.
        output = [neuron.getOutputValue() for neuron in self.getLastLayer().neurons]
        
        return output;
    
    def backPropagation(self, trainingElement, learning_rate, outputOffset=True):
        # Making sure the outputs are already established in the first layer.
        if not outputOffset:
            print("Back-Propagation is only applied when outputs are already established.")
            return
        
        # Iterating through the layers and back-propagating them.
        # `([1:-1])[::-1]` means we exclude the last neuron layer and reversing the list.
        for index, neuronLayer in enumerate((self.neuronLayers[1:-1])[::-1]):
            # We back-propagate each neuron layer, and passing the next layer as a parameter.
            # `len(self.neuronLayers)-index` we do such an operation because the list is reversed!
            neuronLayer.backPropagation(self.neuronLayers[len(self.neuronLayers)-1-index], learning_rate)
        
        # Now we finally finished calculating the data with back-propagation.
        # Time to adjust and update the data, which includes the bias and the weights.
        for index, neuronLayer in enumerate(self.neuronLayers[1:]):
            neuronLayer.updateData(self.neuronLayers[index], learning_rate)
    
    def printData(self, precision=4):
        
        super(NeuronNet, self).printData()

        print("Neuron Layers:")
        for i, neuronLayer in enumerate(self.neuronLayers):
            print("Neuron Layer " + str(i+1) + ": ")
            neuronLayer.printData()
            
        print()
            
        print()