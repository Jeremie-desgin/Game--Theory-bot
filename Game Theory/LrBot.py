import numpy as np 
 
class AdaptivePrisonerBot:
    
    def __init__(self, memory_size=9, hidden_size=6, lr=0.3, l2=0.001, seed=None):
       
        if seed is not None:
            np.random.seed(seed)
 
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.l2 = l2
 
        self._init_weights()
        self._init_history()
 
    def _init_weights(self):
        input_size = 2 * self.memory_size + 1  
        self.weights_input_hidden = 2 * np.random.random((input_size, self.hidden_size)) - 1
        self.weights_hidden_output = 2 * np.random.random((self.hidden_size + 1, 1)) - 1
 
    def _init_history(self):
        self.bot_history = [0] * self.memory_size
        self.opp_history = [0] * self.memory_size
 
    def _build_input(self):
        return np.array(self.bot_history + self.opp_history + [1]).reshape(1, -1)
 
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self._init_weights()
        self._init_history()
 
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
 
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)
 
    def _forward(self, x):
        hidden = self.sigmoid(np.dot(x, self.weights_input_hidden))
        hidden_with_bias = np.hstack([hidden, np.ones((1, 1))])
        output = self.sigmoid(np.dot(hidden_with_bias, self.weights_hidden_output))
        return hidden_with_bias, output
 
    def act(self):
        _, output = self._forward(self._build_input())
        return int(np.round(float(output.flatten()[0])))
 
    def learn(self, bot_move, opp_move, reward, max_reward=5.0):
        x = self._build_input()
        hidden_with_bias, output = self._forward(x)
 
        norm_reward = reward / max_reward
        error_output = norm_reward - output
        delta_output = error_output * self.sigmoid_derivative(output)
 
        error_hidden = np.dot(delta_output, self.weights_hidden_output[:-1].T)
        delta_hidden = error_hidden * self.sigmoid_derivative(hidden_with_bias[:, :-1])
 
        self.weights_hidden_output += self.lr * (np.dot(hidden_with_bias.T, delta_output) - self.l2 * self.weights_hidden_output)
        self.weights_input_hidden += self.lr * (np.dot(x.T, delta_hidden) - self.l2 * self.weights_input_hidden)
 
        self.bot_history = self.bot_history[1:] + [bot_move]
        self.opp_history = self.opp_history[1:] + [opp_move]
 
    def coop_rate_window(self, window=500):
        recent = self.bot_history[-window:]
        return recent.count(0) / len(recent) if recent else 0.0
 
    def debug(self):
        return (
            f"memory={self.memory_size}, hidden={self.hidden_size}, lr={self.lr}, l2={self.l2}, "
            f"w_ih_norm={np.linalg.norm(self.weights_input_hidden):.3f}, "
            f"w_ho_norm={np.linalg.norm(self.weights_hidden_output):.3f}"
        )