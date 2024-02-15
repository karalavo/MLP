import numpy as np


def ELU(x, alpha=1.0):
    """
    Экспоненциальный линейный элемент (ELU).

    Параметры:
    x : float или массив
        Входные данные.
    alpha : float, по умолчанию 1.0
        Параметр alpha.

    Возвращает:
    float или массив
        Результат применения ELU к входным данным.
    """
    return x if x > 0 else alpha * (np.exp(x) - 1)


def ELU_grad(x, alpha=1.0):
    """
    Вычисление градиента для экспоненциального линейного элемента (ELU).

    Параметры:
    x : float или массив
        Входные данные.
    alpha : float, по умолчанию 1.0
        Параметр alpha.

    Возвращает:
    float или массив
        Градиент для ELU.
    """
    return 1 if x > 0 else alpha * np.exp(x)


def HardSigmoid(x):
    """
    Жесткая сигмоидная функция.

    Параметры:
    x : float или массив
        Входные данные.

    Возвращает:
    float или массив
        Результат применения жесткой сигмоидной функции к входным данным.
    """
    if x < -2.5:
        return 0
    elif -2.5 <= x <= 2.5:
        return 0.2 * x + 0.5
    else:
        return 1


def HardSigmoid_grad(x):
    """
    Вычисление градиента для жесткой сигмоидной функции.

    Параметры:
    x : float или массив
        Входные данные.

    Возвращает:
    float или массив
        Градиент для жесткой сигмоидной функции.
    """
    return 0.2 if -2.5 <= x <= 2.5 else 0


def LeakyReLu(x, alpha=0.3):
    """
    Утечка линейного элемента ReLU (Leaky ReLU).

    Параметры:
    x : float или массив
        Входные данные.
    alpha : float, по умолчанию 0.3
        Параметр alpha.

    Возвращает:
    float или массив
        Результат применения Leaky ReLU к входным данным.
    """
    return x if x > 0 else alpha * x


def LeakyReLu_grad(x, alpha=0.3):
    """
    Вычисление градиента для утечки линейного элемента ReLU (Leaky ReLU).

    Параметры:
    x : float или массив
        Входные данные.
    alpha : float, по умолчанию 0.3
        Параметр alpha.

    Возвращает:
    float или массив
        Градиент для Leaky ReLU.
    """
    return 1 if x > 0 else alpha


def ReLu(x):
    """
    Линейный элемент ReLU.

    Параметры:
    x : float или массив
        Входные данные.

    Возвращает:
    float или массив
        Результат применения ReLU к входным данным.
    """
    return x if x > 0 else 0


def ReLu_grad(x):
    """
    Вычисление градиента для линейного элемента ReLU.

    Параметры:
    x : float или массив
        Входные данные.

    Возвращает:
    float или массив
        Градиент для ReLU.
    """
    return 1 if x > 0 else 0


def SELU(x, alpha=1.67326324, scale=1.05070098):
    """
    Элемент SELU (Scaled Exponential Linear Unit).

    Параметры:
    x : float или массив
        Входные данные.
    alpha : float, по умолчанию 1.67326324
        Параметр alpha.
    scale : float, по умолчанию 1.05070098
        Параметр масштабирования.

    Возвращает:
    float или массив
        Результат применения SELU к входным данным.
    """
    return scale * x if x > 0 else scale * alpha * (np.exp(x) - 1)


def SELU_grad(x, alpha=1.67326324, scale=1.05070098):
    """
    Вычисление градиента для элемента SELU (Scaled Exponential Linear Unit).

    Параметры:
    x : float или массив
        Входные данные.
    alpha : float, по умолчанию 1.67326324
        Параметр alpha.
    scale : float, по умолчанию 1.05070098
        Параметр масштабирования.

    Возвращает:
    float или массив
        Градиент для SELU.
    """
    return scale if x > 0 else scale * alpha * x


def GELU(x):
    """
    Гауссовский экспоненциальный линейный элемент (GELU).

    Параметры:
    x : float или массив
        Входные данные.

    Возвращает:
    float или массив
        Результат применения GELU к входным данным.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def GELU_grad(x):
    """
    Вычисление градиента для гауссовского экспоненциального линейного элемента (GELU).

    Параметры:
    x : float или массив
        Входные данные.

    Возвращает:
    float или массив
        Градиент для GELU.
    """
    s = x / np.sqrt(2)
    erf_prime = lambda x: (2 / np.sqrt(np.pi)) * np.exp(-(x ** 2))
    approx = np.tanh(np.sqrt(2 / np.pi) * (x + 0.0044715 * x ** 3))
    return 0.5 + 0.5 * approx + ((0.5 * x * erf_prime(s)) / np.sqrt(2))


def Sigmoid(x):
    """
    Сигмоидная функция.

    Параметры:
    x : float или массив
        Входные данные.

    Возвращает:
    float или массив
        Результат применения сигмоидной функции к входным данным.
    """
    return 1 / (1 + np.exp(-x))


def Sigmoid_grad(x):
    """
    Вычисление градиента для сигмоидной функции.

    Параметры:
    x : float или массив
        Входные данные.

    Возвращает:
    float или массив
        Градиент для сигмоидной функции.
    """
    return Sigmoid(x) * (1 - Sigmoid(x)) * (1 - 2 * Sigmoid(x))


def SoftPlus(x):
    """
    SoftPlus функция.

    Параметры:
    x : float или массив
        Входные данные.

    Возвращает:
    float или массив
        Результат применения SoftPlus к входным данным.
    """
    return np.log(1 + np.exp(x))


def SoftPlus_grad(x):
    """
    Вычисление градиента для SoftPlus функции.

    Параметры:
    x : float или массив
        Входные данные.

    Возвращает:
    float или массив
        Градиент для SoftPlus функции.
    """
    return np.exp(x) / (1 + np.exp(x))


def Tanh(x):
    """
    Гиперболический тангенс (Tanh).

    Параметры:
    x : float или массив
        Входные данные.

    Возвращает:
    float или массив
        Результат применения Tanh к входным данным.
    """
    return np.tanh(x)


def Tanh_grad(x):
    """
    Вычисление градиента для гиперболического тангенса (Tanh).

    Параметры:
    x : float или массив
        Входные данные.

    Возвращает:
    float или массив
        Градиент для Tanh.
    """
    return 1 - np.tanh(x) ** 2
