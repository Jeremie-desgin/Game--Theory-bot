import numpy as np 

class NeuralNetwork():
    
    def __init__(self):
        
        np.random.seed(1)
        self.synaptic_weights = 2*np.random.random((3,1))-1

  