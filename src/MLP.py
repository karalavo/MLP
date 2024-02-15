import BatchNormalization
import Activations
import DropOut
import Losses
import pickle
import sys
import time

import numpy as np


class MLP:
    """
        Класс для реализации многослойного перцептрона (MLP).

        Атрибуты:
            activation (str): Тип активации для скрытых слоев.
            count_layers (int): Количество слоев нейронов (включая выходной).
            count_neurons (list): Список, содержащий количество нейронов в каждом скрытом слое.
            dropout (bool): Флаг использования dropout.
            bn (bool): Флаг использования нормализации по мини-батчам (batch normalization).
            w (list): Список матриц весов для каждого слоя.
            b (list): Список векторов смещений для каждого слоя.
            act_fn (function): Функция активации.
            act_fn_grad (function): Производная функции активации.
            dropout (function): Функция dropout.
            bn (BatchNormalization.BN): Объект нормализации по мини-батчам.
            true_labels (list): Список истинных меток.
            predicted_labels (list): Список предсказанных меток.

        Методы:
            fit(train, epochs=3, learn_rate=0.1, decay_rate=0.01): Обучает модель на тренировочных данных.
            predict(test): Предсказывает метки для тестовых данных.
    """

    def __init__(self, activation, count_layers, count_neurons, dropout=False, bn=False):
        """
            Инициализирует объект MLP.

            Аргументы:
                activation (str): Тип активации для скрытых слоев.
                count_layers (int): Количество слоев нейронов (включая выходной).
                count_neurons (list): Список, содержащий количество нейронов в каждом скрытом слое.
                dropout (bool): Флаг использования dropout.
                bn (bool): Флаг использования нормализации по мини-батчам (batch normalization).
        """
        self.w = self._initialisation_of_w(count_neurons)
        self.b = self._initialisation_of_b(count_neurons)
        self.count_layers = count_layers + 1
        self.act_fn, self.act_fn_grad = self._import_activator(activation)
        self.dropout = self._import_dropout(dropout) if dropout else False
        self.bn = BatchNormalization.BN(count_neurons) if bn else False
        self.true_labels = []
        self.predicted_labels = []

    @staticmethod
    def _import_activator(activator):
        """
            Импортирует функцию активации и ее производную из модуля Activations.

            Аргументы:
                activator (str): Название функции активации.

            Возвращает:
                tuple: Кортеж, содержащий функцию активации и ее производную.
        """
        act_fn = getattr(sys.modules['Activations'], activator)
        act_fn_grad = getattr(sys.modules['Activations'], activator + '_grad')
        return act_fn, act_fn_grad

    @staticmethod
    def _import_dropout(dropout):
        """
            Импортирует функцию dropout из модуля DropOut.

            Аргументы:
               dropout (str): Тип dropout.

            Возвращает:
               function: Функция dropout.
        """
        return getattr(sys.modules['DropOut'], dropout)

    @staticmethod
    def _separate_date(data):
        """
            Разделяет данные на входные и выходные.

            Аргументы:
                data (array): Массив данных.

            Возвращает:
                tuple: Кортеж с входными и выходными данными.
        """
        data = np.array(data)
        np.random.shuffle(data)
        data = data.T
        y = data[0]
        X = data[1:] / 255
        return y, X

    @staticmethod
    def _one_hot(y):
        """
            Преобразует метки в формат one-hot encoding.

            Аргументы:
                y (array): Массив меток.

            Возвращает:
                array: Массив в формате one-hot encoding.
        """
        one_hot_y = np.zeros((y.size, y.max() + 1))
        one_hot_y[np.arange(y.size), y] = 1
        return one_hot_y.T

    @staticmethod
    def _log_softmax(z):
        """
            Применяет логарифм функции softmax к вектору z.

            Аргументы:
                z (array): Входной вектор.

            Возвращает:
                array: Результат применения логарифма softmax.
        """
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
        softmax_output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return np.log(softmax_output)

    def _initialisation_of_w(self, count_neurons):
        """
            Инициализирует матрицы весов для каждого слоя.

            Аргументы:
                count_neurons (list): Список с количеством нейронов в каждом слое.

            Возвращает:
                list: Список матриц весов.
        """
        count_neurons_ = count_neurons + [26, 784]
        w = []
        for i in range(self.count_layers):
            w.append(np.random.uniform(-0.5, 0.5, (count_neurons_[i], count_neurons_[i - 1])))
        return w

    def _initialisation_of_b(self, count_neurons):
        """
            Инициализирует векторы смещений для каждого слоя.

            Аргументы:
               count_neurons (list): Список с количеством нейронов в каждом слое.

            Возвращает:
                list: Список векторов смещений.
        """
        count_neurons_ = count_neurons + [26]
        b = []
        for i in range(self.count_layers):
            b.append(np.zeros((count_neurons_[i], 1)))
        return b

    def _forward(self, img):
        """
            Производит прямой проход по сети.

            Аргументы:
                img (array): Входное изображение.

            Возвращает:
                tuple: Кортеж с выходами каждого слоя и линейными комбинациями для обратного прохода.
        """
        h = [img]
        h_pre = []
        for layer in range(self.count_layers):
            h_pre.append(self.b[layer] + self.w[layer] @ h[-1])
            h_pre[-1] = h_pre[-1] if layer == 0 and layer == self.count_layers and self.bn \
                else self.bn.feed_bn(h_pre[-1], layer)
            h.append(self.act_fn(h_pre[-1]))
            if self.dropout and layer < self.count_layers - 1:
                h[-1] = self.dropout(h[-1])
        h[-1] = self._log_softmax(h_pre.pop())
        return h, h_pre

    def _backward(self, h, h_pre, y, learn_rate):
        """
            Производит обратное распространение ошибки.

            Аргументы:
                h (list): Выходы каждого слоя.
                h_pre (list): Линейные комбинации для обратного прохода.
                y (array): Истинные метки.
                learn_rate (float): Скорость обучения.
        """
        dh = 0
        dh_pre = 0
        y = self._one_hot(y)
        for layer in range(self.count_layers):
            if layer == 0:
                dh = Losses.cross_entropy_grad(y, h.pop())
                self.w[self.count_layers - layer] -= learn_rate * dh @ h.pop()
                self.b[self.count_layers - layer] -= learn_rate * dh
            else:
                dh = self.w[-1].T @ dh if layer == 1 else self.w[-layer] @ dh_pre
                h_pre_ = h_pre.pop()
                dh_pre = dh * self.act_fn_grad(h_pre_)
                self.w[self.count_layers - layer] -= learn_rate * dh_pre @ h.pop()
                self.b[self.count_layers - layer] -= learn_rate * dh_pre
                if self.bn and layer != self.count_layers - 1:
                    self.bn.back_bn(dh_pre, learn_rate, h_pre_, layer)

    @staticmethod
    def _exponential_lr_decay(initial_lr, epoch, decay_rate):
        """
            Выполняет экспоненциальное затухание скорости обучения.

            Аргументы:
                initial_lr (float): Начальная скорость обучения.
                epoch (int): Номер эпохи.
                decay_rate (float): Скорость затухания.

            Возвращает:
                float: Скорость обучения после затухания.
        """
        return initial_lr * np.exp(-decay_rate * epoch)

    def fit(self, train, epochs=3, learn_rate=0.1, decay_rate=0.01):
        """
            Обучает модель на тренировочных данных.

            Аргументы:
                train (array): Тренировочные данные.
                epochs (int): Количество эпох обучения.
                learn_rate (float): Начальная скорость обучения.
                decay_rate (float): Скорость затухания.
        """
        X, y = self._separate_date(train)
        for epoch in range(epochs):
            start_time = time.time()
            learn_rate = self._exponential_lr_decay(learn_rate, epoch, decay_rate)
            for img, l in zip(X, y):
                h, h_pre = self._forward(img)
                self.true_labels.append(np.argmax(l))
                self.predicted_labels.append(np.argmax(h[-1], axis=0))
                self._backward(h, h_pre, y, learn_rate)
            precision, recall, accuracy, f_measure, loss = self._compute_metrics(
                np.array(self.true_labels), np.array(self.predicted_labels)
            )

            end_time = time.time()
            duration = end_time - start_time
            with open("training_log.txt", 'wb') as tl:
                tl.write(f"Epoch {epoch + 1}: Recall={recall}, F-measure={f_measure}, Accuracy={accuracy}, "
                         f"Precision={precision}, Duration={duration}, CossEntropy={loss} seconds\n")
            self.true_labels = []
            self.predicted_labels = []
        self._save_wb()

    @staticmethod
    def _compute_metrics(true, predict):
        """
            Вычисляет метрики качества модели.

            Аргументы:
                true (array): Истинные метки.
                predict (array): Предсказанные метки.

            Возвращает:
                tuple: Кортеж с метриками: precision, recall, accuracy, f_measure, loss.
        """
        tp = np.sum(np.logical_and(true == 1, predict == 1))
        fp = np.sum(np.logical_and(true == 0, predict == 1))
        fn = np.sum(np.logical_and(true == 1, predict == 0))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        accuracy = np.mean(true == predict)
        f_measure = 2 * precision * recall / (precision + recall + 1e-9)
        loss = Losses.cross_entropy(true, predict)
        return precision, recall, accuracy, f_measure, loss

    def predict(self, test):
        """
            Предсказывает метки для тестовых данных.

            Аргументы:
                test (array): Тестовые данные.

            Возвращает:
                list: Список предсказанных меток.
        """
        result = []
        X, y = self._separate_date(test)
        self._open_wb()
        for img, l in zip(X, y):
            h, _ = self._forward(img)
            result.append(np.argmax(Losses.cross_entropy(y, h[-1])))
        return result

    def _save_wb(self):
        """Сохраняет веса и смещения модели."""
        with open("model_wb.pkl", 'wb') as f:
            pickle.dump({'w': self.w, 'b': self.b, 'count_layers': self.count_layers - 1}, f)

    def _open_wb(self):
        """Загружает веса и смещения модели из файла."""
        with open('model_wb.pkl', 'rb') as f:
            saved_wb = pickle.load(f)
            self.w = saved_wb['w']
            self.b = saved_wb['b']
            self.count_layers = saved_wb['count_layers']
