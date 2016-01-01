'''
Created on Dec 11, 2015

@author: Israel
'''

from DeepLearning.neuralnetwork.NeuralNetwork import NeuralNetwork


if __name__ == '__main__':
    net = NeuralNetwork([2,2,1])
    net.printData()
    print(net.feedforward([1,0]))