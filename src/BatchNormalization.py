import numpy as np


class BN:
    def __init__(self, count_neurons):
        self.gamma = [np.ones((neurons, 1)) for neurons in count_neurons]
        self.beta = [np.zeros((neurons, 1)) for neurons in count_neurons]
        self.epsilon = np.finfo(np.float32).eps
        self.running_mean = [np.zeros((neurons, 1)) for neurons in count_neurons]
        self.running_var = [np.ones((neurons, 1)) for neurons in count_neurons]

    def feed_bn(self, h_pre, layer):
        mean = np.mean(h_pre, axis=1, keepdims=True)
        var = np.var(h_pre, axis=1, keepdims=True)

        self.running_mean[layer - 1] = 0.9 * self.running_mean[layer - 1] + 0.1 * mean
        self.running_var[layer - 1] = 0.9 * self.running_var[layer - 1] + 0.1 * var

        h_pre = (h_pre - mean) / np.sqrt(var + self.epsilon)
        return self.gamma[layer - 1] * h_pre + self.beta[layer - 1]

    def back_bn(self, dh_pre, learn_rate, h_pre, layer):
        self.gamma[layer - 1] -= learn_rate * np.mean(dh_pre * h_pre, axis=1, keepdims=True)
        self.beta[layer - 1] -= learn_rate * np.mean(dh_pre, axis=1, keepdims=True)
