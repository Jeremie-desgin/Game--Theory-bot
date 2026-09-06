import numpy as np 

class NeuralNetwork():
    
    def __init__(self):
        
        np.random.seed(1)
        self.synaptic_weights = 2*np.random.random((3,1))-1

    def sigmoid(self,x):
        return 1/ (1* np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1-x)

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def train(self, training_input, training_outputs, training_interations):

        for interation in range(training_interations):

            outputs = self.think(training_input)
            error = training_outputs - outputs
            adjustment = np.dot(training_input.T, error * self.sigmoid_derivative(outputs))
            self.synaptic_weights += adjustment


        
if __name__ == "__main__":

    neural_network = NeuralNetwork()
    print("Random weights")
    print(neural_network.synaptic_weights)

    training_input = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_outputs = np.array([[0,1,1,0]]).T

    neural_network.train(training_input,training_outputs,1000)

    A = float(input("input 1: "))
    B = float(input("input 2: "))
    C = float(input("input 3: "))

    print(neural_network.think(np.array([A,B,C])))


   