import numpy as np

def sigmold(x):
    return 1/ (1* np.exp(-x))

def sigmold_deritavtive(x):
    return x * (1-x)


training_input = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T



# weights of ML
np.random.seed(1)
rand_numbers = np.random.random((3,1))
scaled_up = 2 * rand_numbers
shifted = scaled_up -1
synaptic_weights = shifted

print ('Random starting synaptic weights')
print( synaptic_weights)


# Testing 
for interation in range(4):

    input_layer = training_input
    outputs = sigmold(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs
    adjustment = error * sigmold_deritavtive(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustment)

print("synaptic weights after training")
print(synaptic_weights)

print("Outputs afetr traning")
round_outputs = np.round(outputs)
print(round_outputs)