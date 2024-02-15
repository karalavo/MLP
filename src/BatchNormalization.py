import numpy as np


class BN:
    """
    Класс для реализации слоя нормализации пакетов (Batch Normalization).

    Параметры:
    count_neurons : list
        Список, содержащий количество нейронов в каждом слое сети.

    Атрибуты:
    gamma : list
        Список массивов параметров масштабирования для каждого слоя.
    beta : list
        Список массивов параметров сдвига для каждого слоя.
    epsilon : float
        Малое число, используемое для стабилизации вычислений при делении.
    running_mean : list
        Список массивов для хранения скользящего среднего средних значений для каждого слоя.
    running_var : list
        Список массивов для хранения скользящего среднего дисперсий для каждого слоя.

    Методы:
    feed_bn(h_pre, layer):
        Производит нормализацию входных данных и применяет параметры масштабирования и сдвига.
        Возвращает нормализованные данные для передачи в следующий слой.

    back_bn(dh_pre, learn_rate, h_pre, layer):
        Обратное распространение градиента для параметров gamma и beta.
    """

    def __init__(self, count_neurons):
        self.gamma = [np.ones((neurons, 1)) for neurons in count_neurons]
        self.beta = [np.zeros((neurons, 1)) for neurons in count_neurons]
        self.epsilon = np.finfo(np.float32).eps
        self.running_mean = [np.zeros((neurons, 1)) for neurons in count_neurons]
        self.running_var = [np.ones((neurons, 1)) for neurons in count_neurons]

    def feed_bn(self, h_pre, layer):
        """
        Производит нормализацию входных данных и применяет параметры масштабирования и сдвига.

        Параметры:
        h_pre : numpy.ndarray
            Входные данные.
        layer : int
            Номер слоя.

        Возвращает:
        numpy.ndarray
            Нормализованные данные для передачи в следующий слой.
        """
        mean = np.mean(h_pre, axis=1, keepdims=True)
        var = np.var(h_pre, axis=1, keepdims=True)

        self.running_mean[layer - 1] = 0.9 * self.running_mean[layer - 1] + 0.1 * mean
        self.running_var[layer - 1] = 0.9 * self.running_var[layer - 1] + 0.1 * var

        h_pre = (h_pre - mean) / np.sqrt(var + self.epsilon)
        return self.gamma[layer - 1] * h_pre + self.beta[layer - 1]

    def back_bn(self, dh_pre, learn_rate, h_pre, layer):
        """
        Обратное распространение градиента для параметров gamma и beta.

        Параметры:
        dh_pre : numpy.ndarray
            Градиент на выходе слоя.
        learn_rate : float
            Скорость обучения.
        h_pre : numpy.ndarray
            Входные данные слоя.
        layer : int
            Номер слоя.
        """
        self.gamma[layer - 1] -= learn_rate * np.mean(dh_pre * h_pre, axis=1, keepdims=True)
        self.beta[layer - 1] -= learn_rate * np.mean(dh_pre, axis=1, keepdims=True)
