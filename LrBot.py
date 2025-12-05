# Python learning bot for prisoner's dilemma 
import numpy as np


class AdaptivePrisonerBot:
    def __init__(self, memory_size=3, lr=0.5):

        self.memory_size = memory_size
        self.lr = lr
        
        self.synaptic_weights = 2 * np.random.random((2 * memory_size + 1, 1)) - 1

        self.bot_history = [0] * memory_size
        self.opp_history = [0] * memory_size

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def act(self):

        input_layers = np.array(self.bot_history + self.opp_history + [1]).reshape(1, -1)
        output = self.sigmoid(np.dot(input_layers, self.synaptic_weights))
        action = int(np.round(output))
        return action
    
    def learn(self, bot_move, opp_move, reward):

        self.bot_history = self.bot_history[1:] + [bot_move]
        self.opp_history = self.opp_history[1:] + [opp_move]
        

        input_layers = np.array(self.bot_history + self.opp_history + [1]).reshape(1, -1)
        output = self.sigmoid(np.dot(input_layers, self.synaptic_weights))

        error = reward - output
        adjustment = error * self.sigmoid_derivative(output)
        self.synaptic_weights += self.lr * np.dot(input_layers.T, adjustment)

    def debug(self):
        return f"AdaptivePrisonerBot(weights={self.synaptic_weights.flatten()})"

