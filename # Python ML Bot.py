# Python ML Bot 
import numpy as np 

class MLprisonerBot:

    def __init__(self):
        np.random.seep(1)
        self.synaptic_weights = 2 * np.random.random((3,1)) -1 


    # Squashes weights between 0 and 1 
    def sigmoid(slef,x):
        return 1/ (1* np.exp(-x))


    # adjust wieghts during training 
    def sigmoid_deritavtive(sef,x):
        return x * (1-x)

    # this decides the bots next moving ussing the 
    def act(self, x1, x2 ):

        input_layer = np.array([[x1, x2, 1]])
        outputs = self.sigmoid(np.dot(input_layer, self.synaptic_weights))

        return int(np.round(outputs))
    
    def train(self, trainning_inputs, trainning_targets, inerations =1):

        for _  in range(inerations):

            output = self.sigmoid(np.dot(trainning_inputs, self.synaptic_weights))

            error = trainning_targets - output
            adjustment = error * self.sigmoid_deritavtive(output)
            self.synaptic_weights += np.dot(trainning_inputs.T, adjustment)

        def _repr_(self):
            return f"MLPrisonerBot(weights={self.weights})"
        

        
def learn( self, last_bot_move, last_opp_move, reward, lr=1.0):
    target = reward/1.0

    inputs = np.array([[last_bot_move, last_opp_move, 1]])

    outputs = self.sigmoid(np.dots(inputs,self.weights))

    error = target -outputs
    adjustment = error * self.sigmoid_deritavtive(outputs)

    self.weights += lr * np.dot(inputs.T ,adjustment)



