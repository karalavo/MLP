import numpy as np


def cross_entropy(y, y_pred):
    """
        Вычисляет кросс-энтропию между истинными метками и предсказанными значениями.

        Параметры:
        y : numpy.ndarray
            Истинные метки.
        y_pred : numpy.ndarray
            Предсказанные значения.
    """
    eps = np.finfo(float).eps
    return -np.sum(y * np.log(y_pred + eps))


def cross_entropy_grad(y, y_pred):
    """
        Вычисляет градиент функции кросс-энтропии по предсказанным значениям.

        Параметры:
        y : numpy.ndarray
            Истинные метки.
        y_pred : numpy.ndarray
            Предсказанные значения.

        Возвращает:
        numpy.ndarray
            Градиент функции кросс-энтропии по предсказанным значениям.
    """
    return y_pred - y
