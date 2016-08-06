'''
Created on Dec 11, 2015

@author: Israel
'''

import time

from DeepLearning.neuralnetwork.NeuralNetwork import NeuralNetwork
from DeepLearning.training.TrainingSet import TrainingSet
from DeepLearning.training.TrainingElement import TrainingElement


if __name__ == '__main__':
    
    trainingSet = TrainingSet(
            TrainingElement([0,0,0,0],[0]),
            TrainingElement([0,0,0,1],[1]),
            TrainingElement([0,0,1,0],[1]),
            TrainingElement([0,0,1,1],[0]),
            TrainingElement([0,1,0,0],[0]),
            TrainingElement([0,1,0,1],[1]),
            TrainingElement([0,1,1,0],[0]),
            TrainingElement([0,1,1,1],[0]),
            TrainingElement([1,0,0,0],[1]),
            TrainingElement([1,0,0,1],[1]),
            TrainingElement([1,0,1,0],[1]),
            TrainingElement([1,0,1,1],[0]),
            TrainingElement([1,1,0,0],[0]),
            TrainingElement([1,1,0,1],[1]),
            TrainingElement([1,1,1,0],[1]),
            TrainingElement([1,1,1,1],[0])
    )

    print("Neural Network Information:")
    neuralNetwork = NeuralNetwork([4,4,2,1])
    neuralNetwork.printData()

    print()

    input("Press Enter to see current evaluations...")

    print()

    print("Current evaluations:")
    trainingSet.printEvaluation(neuralNetwork)
    
    print()

    input("Press Enter to start training...")

    print()

    print("Training started:")
    neuralNetwork.backPropagation(trainingSet)

    print("Post Training Evaluations:")
    trainingSet.printEvaluation(neuralNetwork)