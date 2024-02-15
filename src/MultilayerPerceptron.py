import pickle
import sys

import numpy as np
import pandas as pd
import Activations
import DropOut
import Losses
import BatchNormalization


# test = '../datasets/emnist-letters/emnist-letters-test.csv'
# train = '../datasets/emnist-letters/emnist-letters-train.csv'


# lr decay
# how often
# test dataset, train dataset, количество скрытых слоев, сколько нейронов в каждом слое,
# лр, decay, количество эпох, выбор активатора

# логирование обучения (на выходе тебе нужно посчитать Recall, Fmeasure, Accuracy and Precision)
# необязательно(для каждой эпохи - cross_entropy, сколько он угадал, время) в текстовик
# сохранение модели в текстовик - сколько слоев, какого размера слои, все веса

# по рофлу: оптимизаторы, l1/l2/dropout,
# KingScheduler
class MLP:

    def __init__(self, test, train, learn_rate, decay, count_layers=2, count_neurons=[198, 49],  epochs=3, activator="ReLu", dropout='Dropout',
                 dropout_rate=0.2):
        self.count_layers = count_layers
        self.Y_train, self.Y_test, self.X_train, self.X_test = self.dataset_processing(pd.read_csv(test),
                                                                                       pd.read_csv(train))
        self.w = self.__initialisation_of_w(count_neurons)
        self.b = self.__initialisation_of_b(count_neurons)
        self.act_fn, self.act_fn_grad = self.__import_activator(activator)
        self.dropout = self.__import_dropout(dropout)

        self.bn = BatchNormalization.BN(count_neurons)

        self.learn_rate = learn_rate
        self.decay = decay
        self.epochs = epochs
        self.dropout_rate = dropout_rate

    @staticmethod
    def dataset_processing(train, test):
        m_train, n_train  = train.shape
        m_test, n_test = test.shape
        train_ = np.array(train).T
        test_ = np.array(test).T
        y_train = np.zeros((train_[0].size, train_[0].max()))
        y_train[np.arange(train_[0].size), train_[0]] = 1
        y_test = np.zeros((test_[0].size, test_[0].max()))
        y_test[np.arange(test_[0].size), test_[0]] = 1
        return y_train.T, y_test.T, train_[1:] / 255, test_[1:] / 255, m_train, n_train, m_test, n_test

    @staticmethod
    def __import_dropout(dropout):
        return getattr(sys.modules['DropOut'], dropout)

    def __initialisation_of_w(self, count_neurons):
        count_neurons += [26, 784]
        w = []
        for i in range(self.count_layers + 1):
            w.append(np.random.uniform(-0.5, 0.5, (count_neurons[i], count_neurons[i - 1])))
        return w

    def __initialisation_of_b(self, count_neurons):
        b = []
        for i in range(self.count_layers + 1):
            b.append(np.zeros((count_neurons[i], 1)))
        return b

    @staticmethod
    def __import_activator(activator):
        act_fn = getattr(sys.modules['Activations'], activator)
        act_fn_grad = getattr(sys.modules['Activations'], activator + '_grad')
        return act_fn, act_fn_grad

    @staticmethod
    def __log_softmax(z):
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
        softmax_output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return np.log(softmax_output)

    def __feedforward(self, img):
        h = [img]
        h_pre = []
        for layer in range(self.count_layers + 1):
            h_pre.append(self.b[layer] + self.w[layer] @ h[-1])
            h_pre[-1] = h_pre[-1] if layer == 0 and layer == self.count_layers else self.bn.feed_bn(h_pre[-1], layer)
            h.append(self.act_fn(h_pre[-1]))
            if layer < self.count_layers:
                h[-1] = self.dropout(h[-1], self.dropout_rate)
        h[-1] = self.__softmax(h_pre.pop())
        return h, h_pre

    def losses(self, Y, Y_pre):
        return Losses.cross_entropy(Y, Y_pre)

    def __backpropagation(self, h, h_pre, Y):
        dh = 0
        dh_pre = 0
        for layer in range(self.count_layers + 1):
            if layer == 0:
                dh = Losses.cross_entropy_grad(Y, h.pop())
                self.w[self.count_layers - layer] += -self.learn_rate * dh @ h.pop()
                self.b[self.count_layers - layer] += -self.learn_rate * dh
            else:
                dh = self.w[-1].T @ dh if layer == 1 else self.w[-layer] @ dh_pre
                h_pre_ = h_pre.pop()
                dh_pre = dh * self.act_fn_grad(h_pre_)
                self.w[self.count_layers - layer] += -self.learn_rate * dh_pre @ h.pop()
                self.b[self.count_layers - layer] += -self.learn_rate * dh_pre
                if layer != self.count_layers:
                    self.bn.back_bn(dh_pre, self.learn_rate, h_pre_, layer)

    def fit(self):
        for epoch in range(self.epochs):
            for _ in range()
            h, h_pre = self.__feedforward(self.X_train)
            self.losses(self.Y_train, h[-1])
            self.__backpropagation(h, h_pre, self.Y_train)
        self.save_wb()

    def predict(self):
        self.open_wb()
        h, _ = self.__feedforward(self.X_test)
        self.losses(self.Y_test, h[-1])

    def save_wb(self):
        with open("model_wb.pkl", 'wb') as f:
            pickle.dump({'w': self.w, 'b': self.b}, f)

    def open_wb(self):
        with open('model_wb.pkl', 'rb') as f:
            saved_wb = pickle.load(f)
            self.w = saved_wb['w']
            self.b = saved_wb['b']
