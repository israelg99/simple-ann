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

    def printEvaluation(self, neuralnetwork):
        # We only check evaluation of the first output neuron in the neuralnetwork, implement all output neurons evaluation check.
        print("Input {inputs} is {evaluation}, was expected to be {outputs}".format(inputs=self.inputs, evaluation=neuralnetwork.feedforward(self.inputs), outputs=self.outputs))

    def __str__(self):
        return "inputs: {inputs} outputs: {outputs}".format(inputs=self.inputs, outputs=self.outputs);