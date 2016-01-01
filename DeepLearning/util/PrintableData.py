'''
Created on Dec 25, 2015

@author: Israel
'''

from abc import ABCMeta, abstractmethod
import numpy as np


class PrintableData(metaclass=ABCMeta):
    
    @abstractmethod
    def printData(self, precision=4):
        np.set_printoptions(precision=precision, linewidth=100)