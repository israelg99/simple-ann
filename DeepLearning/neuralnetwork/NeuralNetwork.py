'''
Created on Dec 11, 2015

@author: Israel
'''

from _collections import defaultdict

from DeepLearning.functions.Functions import Functions
from DeepLearning.neuralnetwork.neurons.NeuronNet import NeuronNet
from DeepLearning.util.PrintableData import PrintableData


class NeuralNetwork(PrintableData, object):
    
    def __init__(self, sizes, activation="sigmoid", learning_rate=0.5):
        
        self.activations = defaultdict(self.setSigmoidActivation, {
            "sigmoid" : self.setSigmoidActivation,
            "tanh"    : self.setTanhActivation
        })
        
        self.activations[activation]()
        
        self._num_layers = len(sizes)
        self._sizes = sizes
        
        self._neuralNetwork = NeuronNet(sizes)
        
        self._learning_rate = learning_rate
        
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
    
    @property
    def learning_rate(self):
        """I'm the 'sizes' property."""
        return self._learning_rate
    
    def getInputLayer(self):
        return self.neuralNetwork.getFirstLayer()
    
    def getOutputLayer(self):
        return self.neuralNetwork.getLastLayer()
    
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
        
        # If the input layer has a different length than the inputs provided, the `feedforward` method is stopped.
        if(len(inputs) != len(self.getInputLayer().neurons)):
            print("Inputs must have the same length as the input layer in the neural network.")
            print("Inputs length: " + str(len(inputs)) + ", Input layer length: " + str(len(self.getInputLayer().neurons)))
            return
        
        # Assigning the inputs to the input layer.
        for index, inputPart in enumerate(inputs):
            self.getInputLayer().neurons[index].setInputNeuron(inputPart)
        
        # After we established the inputs, we feedforward the neural network to get out inputs.
        output = self.neuralNetwork.feedforward(self.activation_method, True)
        
        return output
    
    def backPropagation(self, trainingSet, learning_rate=None, epochs=10000):
        # If a learning rate was not provided, the default is set to the instance learning rate.
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        # To gradually decrease the learning rate, we need a good step.
        lr_step = (learning_rate - 0.1)/epochs if learning_rate > 0.1 else 0
        
        # Iterating the amount of epochs.
        for epoch in range(epochs):
            
            # Gradually decreasing the learning rate.
            learning_rate -= lr_step
            
            if epoch % 100 == 0:
                print("We are in Epoch: " + str(epoch))
            
            # Loop through the `TrainingSet` with `TrainingElement`s.
            for element in trainingSet.elements:
                
                # Making sure the training element inputs and outputs matches the neural network, otherwise, skip this training element.
                if((len(element.inputs) != self.getInputLayer().amountOfNeurons()) or 
                   (len(element.outputs) != self.getOutputLayer().amountOfNeurons())):
                    continue;
                
                # Feed-forwarding the neural network to adjust all the values for this specific training element inputs.
                self.feedforward(element.inputs)
                
                # Looping the output layer to calculate the error factor and the delta value of the desired outputs and our actual outputs.
                for index, neuron in enumerate(self.getOutputLayer().neurons):
                    # Calculating the error factor.
                    neuron.calculateOutputErrorFactor(element.outputs[index])
                    
                    # Calculating the delta value.
                    neuron.calculateDeltaValue()
                
                # Back-propagating the rest of the neural network.
                self.neuralNetwork.backPropagation(element, learning_rate, True)
    
    