'''
Created on Dec 11, 2015

@author: Israel
'''

import numpy as np


class Functions(object):

    @staticmethod
    def sigmoid(x):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-x))
    
    @staticmethod
    def sigmoid_prime(x):
        """Derivative of the sigmoid function."""
        return Functions.sigmoid(x)*(1-Functions.sigmoid(x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_prime(x):
        return 1.0 - x**2
    
    @staticmethod
    def random(minimum, maximum):
        return (maximum - minimum) * np.random.rand() + minimum
    
    @staticmethod
    def cost_derivative(output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)