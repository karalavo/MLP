import numpy as np


def Dropout(x, dropout_rate=0.2):
    """
    Применяет метод Dropout к данным.

    Параметры:
    x : numpy.ndarray
        Входные данные.
    dropout_rate : float, optional
        Коэффициент исключения нейронов (вероятность их исключения), по умолчанию 0.2.

    Возвращает:
    numpy.ndarray
        Массив данных после применения Dropout.
    """
    mask = np.random.rand(*x.shape) < (1 - dropout_rate)
    return x * mask / (1 - dropout_rate)


def AlphaDropout(x, dropout_rate=0.2, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):
    """
    Применяет метод Alpha Dropout к данным.

    Параметры:
    x : numpy.ndarray
        Входные данные.
    dropout_rate : float, optional
        Коэффициент исключения нейронов (вероятность их исключения), по умолчанию 0.2.
    alpha : float, optional
        Параметр alpha, по умолчанию 1.6732632423543772848170429916717.
    scale : float, optional
        Параметр масштабирования, по умолчанию 1.0507009873554804934193349852946.

    Возвращает:
    numpy.ndarray
        Массив данных после применения Alpha Dropout.
    """
    mask = np.random.rand(*x.shape) < (1 - dropout_rate)
    alpha_p = -alpha * scale
    p = 1. / (1. + np.exp(alpha_p))
    ret = x * mask / (1 - dropout_rate)
    ret[~mask] = p
    return ret


def DropConnect(x, dropout_rate=0.2):
    """
    Применяет метод DropConnect к данным.

    Параметры:
    x : numpy.ndarray
        Входные данные.
    dropout_rate : float, optional
        Коэффициент исключения связей (вероятность их исключения), по умолчанию 0.2.

    Возвращает:
    numpy.ndarray
        Массив данных после применения DropConnect.
    """
    mask = np.random.rand(*x.shape) < (1 - dropout_rate)
    return x * mask


