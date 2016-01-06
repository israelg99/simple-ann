'''
Created on Dec 11, 2015

@author: Israel
'''

from DeepLearning.neuralnetwork.NeuralNetwork import NeuralNetwork
from DeepLearning.training.TrainingSet import TrainingSet
from DeepLearning.training.TrainingElement import TrainingElement


if __name__ == '__main__':
    
    trainingSet = TrainingSet(
            TrainingElement([1,0],[1]),
            TrainingElement([0,1],[1]),
            TrainingElement([1,1],[0]),
            TrainingElement([0,0],[0])
    )
    
    net = NeuralNetwork([2,2,1])
    net.printData()
    print(net.feedforward([1,0]))