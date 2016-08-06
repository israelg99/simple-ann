'''
Created on Jan 6, 2016

@author: Israel
'''

class TrainingSet(object):

    def __init__(self, *training_elements):
        self._elements = list(training_elements)
        
    @property
    def elements(self):
        """I'm the 'elements' property."""
        return self._elements
    
    def getElement(self, index):
        return self.elements[index]
    
    def getElementByInput(self, inputs):
        for element in self.elements:
            if element.inputs == inputs:
                return element
            
        print("Training element not found!")
        return None
    
    def getElementByOutput(self, outputs):
        for element in self.elements:
            if element.outputs == outputs:
                return element
            
        print("Training element not found!")
        return None
    
    def addElements(self, *elements):
        self.elements.extend(list(elements))

    def printEvaluation(self, neuralnetwork):
        for element in self.elements:
            element.printEvaluation(neuralnetwork)