import numpy as np


def Dropout(x, dropout_rate):
    mask = np.random.rand(*x.shape) < (1 - dropout_rate)
    return x * mask / (1 - dropout_rate)


def AlphaDropout(x, dropout_rate, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):
    mask = np.random.rand(*x.shape) < (1 - dropout_rate)
    alpha_p = -alpha * scale
    p = 1. / (1. + np.exp(alpha_p))
    ret = x * mask / (1 - dropout_rate)
    ret[~mask] = p
    return ret


def DropConnect(x, dropout_rate):
    mask = np.random.rand(*x.shape) < (1 - dropout_rate)
    return x * mask


