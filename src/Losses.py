import numpy as np


def cross_entropy(y, y_pred):
    eps = np.finfo(float).eps
    return -np.sum(y * np.log(y_pred + eps))


def cross_entropy_grad(y, y_pred):
    return y_pred - y


# def squared_error(y, y_pred):
#     return 0.5 * np.linalg.norm(y_pred - y) ** 2
#
#
# def squared_error_grad(y, y_pred, act_fn_grad, x):
#     return (y_pred - y) * act_fn_grad(x)


