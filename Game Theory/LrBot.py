# Python learning bot for prisoner's dilemma 
# I WILL CLEAN THIS ALL UP 


import numpy as np

class AdaptivePrisonerBot:
   
    def __init__(self, memory_size=9, lr=0.3, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.memory_size = memory_size
        self.lr = lr

        self.synaptic_weights = (
            2 * np.random.random((2 * memory_size + 1, 1)) - 1
        )
        self._init_history()

   

    def _init_history(self):
        self.bot_history = [0] * self.memory_size
        self.opp_history = [0] * self.memory_size

    def _build_input(self):
        return np.array(
            self.bot_history + self.opp_history + [1]
        ).reshape(1, -1)

   

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.synaptic_weights = (
            2 * np.random.random((2 * self.memory_size + 1, 1)) - 1
        )
        self._init_history()

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def act(self):
        output = self.sigmoid(np.dot(self._build_input(), self.synaptic_weights))
        return int(np.round(float(output.flatten()[0])))

    def learn(self, bot_move, opp_move, reward, max_reward=5.0):
       
        self.bot_history = self.bot_history[1:] + [bot_move]
        self.opp_history = self.opp_history[1:] + [opp_move]

        norm_reward = reward / max_reward          
        x = self._build_input()
        output = self.sigmoid(np.dot(x, self.synaptic_weights))

        error = norm_reward - output
        adjustment = error * self.sigmoid_derivative(output)
        self.synaptic_weights += self.lr * np.dot(x.T, adjustment)

    def coop_rate_window(self, window=500):
        
        recent = self.bot_history[-window:]
        return recent.count(0) / len(recent) if recent else 0.0

    def debug(self):
        return (
            f"memory={self.memory_size}, lr={self.lr}, "
            f"weights_norm={np.linalg.norm(self.synaptic_weights):.3f})"
        )