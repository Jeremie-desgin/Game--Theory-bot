import numpy as np 

class NeuralNetwork():
    
    def __init__(self):
        
        np.random.seed(1)
        self.synaptic_weights = 2*np.random.random((3,1))-1

    def sigmold(self,x):
          return 1/ (1* np.exp(-x))
  
      def sigmold_deritavtive(self, x):
          return x * (1-x)
  
     
     