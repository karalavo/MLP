import numpy as np


def ELU(x, alpha=1.0):
    return x if x > 0 else alpha * (np.exp(x) - 1)


def ELU_grad(x, alpha=1.0):
    return 1 if x > 0 else alpha * np.exp(x)


def HardSigmoid(x):
    if x < -2.5:
        return 0
    elif -2.5 <= x <= 2.5:
        return 0.2 * x + 0.5
    else:
        return 1


def HardSigmoid_grad(x):
    return 0.2 if -2.5 <= x <= 2.5 else 0


def LeakyReLu(x, alpha=0.3):
    return x if x > 0 else alpha * x


def LeakyReLu_grad(x, alpha=0.3):
    return 1 if x > 0 else alpha


def ReLu(x):
    return x if x > 0 else 0


def ReLu_grad(x):
    return 1 if x > 0 else 0


def SELU(x, alpha=1.67326324, scale=1.05070098):
    return scale * x if x > 0 else scale * alpha * (np.exp(x) - 1)


def SELU_grad(x, alpha=1.67326324, scale=1.05070098):
    return scale if x > 0 else scale * alpha * x


def GELU(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def GELU_grad(x):
    s = x / np.sqrt(2)
    erf_prime = lambda x: (2 / np.sqrt(np.pi)) * np.exp(-(x ** 2))
    approx = np.tanh(np.sqrt(2 / np.pi) * (x + 0.0044715 * x ** 3))
    return 0.5 + 0.5 * approx + ((0.5 * x * erf_prime(s)) / np.sqrt(2))


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Sigmoid_grad(x):
    return Sigmoid(x) * (1 - Sigmoid(x)) * (1 - 2 * Sigmoid(x))


def SoftPlus(x):
    return np.log(1 + np.exp(x))


def SoftPlus_grad(x):
    return np.exp(x) / (1 + np.exp(x))


def Tanh(x):
    return np.tanh(x)


def Tanh_grad(x):
    return 1 - np.tanh(x) ** 2
