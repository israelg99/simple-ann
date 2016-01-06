'''
Created on Jan 6, 2016

@author: Israel
'''

class TrainingElement(object):

    def __init__(self, inputs, outputs):
        # `inputs` and `outputs` should be passed as lists.
        self._inputs = inputs
        self._outputs = outputs
        
    @property
    def inputs(self):
        """I'm the 'inputs' property."""
        return self._inputs
    
    @property
    def outputs(self):
        """I'm the 'outputs' property."""
        return self._outputs